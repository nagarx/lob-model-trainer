"""
LSTM-based models for LOB price prediction.

LSTM is a standard sequence model baseline for financial time series,
providing a strong benchmark for comparing more complex architectures.

Architecture:
    Input: [batch, seq_len, n_features]
    -> LSTM layers with dropout
    -> Linear head for classification
    Output: [batch, n_classes]

Design principles (RULE.md):
- Configuration-driven hyperparameters
- Explicit dropout and regularization
- Support for bidirectional and multi-layer
- Deterministic given same seed

Reference:
    Hochreiter & Schmidhuber (1997). Long Short-Term Memory.
    Zhang et al. (2019). DeepLOB: Deep Convolutional Neural Networks for 
                         Limit Order Books.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class LSTMConfig:
    """
    Configuration for LSTM model.
    
    All hyperparameters are explicit and configurable per RULE.md.
    """
    
    input_size: int = 98
    """Number of input features. Must match feature extractor output."""
    
    hidden_size: int = 64
    """Hidden dimension for LSTM layers. Range: [32, 512]."""
    
    num_layers: int = 2
    """Number of stacked LSTM layers. Range: [1, 4]."""
    
    num_classes: int = 3
    """Number of output classes (Down=0, Stable=1, Up=2)."""
    
    dropout: float = 0.2
    """Dropout probability between LSTM layers. Range: [0, 0.5]."""
    
    bidirectional: bool = False
    """Use bidirectional LSTM (doubles hidden dimension for classifier)."""
    
    attention: bool = False
    """Use attention mechanism over sequence (experimental)."""
    
    head_dropout: float = 0.2
    """Dropout before classification head. Range: [0, 0.5]."""
    
    def __post_init__(self):
        """Validate configuration."""
        if self.input_size < 1:
            raise ValueError(f"input_size must be >= 1, got {self.input_size}")
        if self.hidden_size < 1:
            raise ValueError(f"hidden_size must be >= 1, got {self.hidden_size}")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")
        if not 0 <= self.dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if not 0 <= self.head_dropout < 1:
            raise ValueError(f"head_dropout must be in [0, 1), got {self.head_dropout}")


# =============================================================================
# LSTM Model
# =============================================================================


class LSTMClassifier(nn.Module):
    """
    LSTM classifier for sequence-to-label prediction.
    
    Takes a sequence of LOB feature vectors and outputs class probabilities.
    Uses the final hidden state for classification.
    
    Args:
        config: LSTMConfig with hyperparameters
    
    Input:
        x: Tensor of shape [batch, seq_len, input_size]
    
    Output:
        logits: Tensor of shape [batch, num_classes]
    
    Example:
        >>> config = LSTMConfig(input_size=98, hidden_size=64, num_layers=2)
        >>> model = LSTMClassifier(config)
        >>> x = torch.randn(32, 100, 98)  # batch, seq, features
        >>> logits = model(x)  # [32, 3]
    """
    
    def __init__(self, config: Optional[LSTMConfig] = None):
        super().__init__()
        
        self.config = config or LSTMConfig()
        cfg = self.config
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0,
            bidirectional=cfg.bidirectional,
        )
        
        # Calculate classifier input dimension
        lstm_output_size = cfg.hidden_size * (2 if cfg.bidirectional else 1)
        
        # Optional attention
        self.use_attention = cfg.attention
        if cfg.attention:
            self.attention = SelfAttention(lstm_output_size)
        
        # Classification head
        self.head_dropout = nn.Dropout(cfg.head_dropout)
        self.classifier = nn.Linear(lstm_output_size, cfg.num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights using Xavier/Kaiming initialization."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1 (helps with gradient flow)
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1.0)
        
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch, seq_len, input_size]
            lengths: Optional sequence lengths for packed sequences
        
        Returns:
            Logits of shape [batch, num_classes]
        """
        batch_size = x.size(0)
        
        # LSTM forward
        # output: [batch, seq_len, hidden * num_directions]
        # h_n: [num_layers * num_directions, batch, hidden]
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        if self.use_attention:
            # Use attention over all timesteps
            context = self.attention(lstm_out)  # [batch, hidden * num_directions]
        else:
            # Use final hidden state
            if self.config.bidirectional:
                # Concatenate forward and backward final hidden states
                h_forward = h_n[-2, :, :]  # Last layer forward
                h_backward = h_n[-1, :, :]  # Last layer backward
                context = torch.cat([h_forward, h_backward], dim=1)
            else:
                context = h_n[-1, :, :]  # [batch, hidden]
        
        # Classification
        context = self.head_dropout(context)
        logits = self.classifier(context)
        
        return logits
    
    @property
    def name(self) -> str:
        """Model name for logging."""
        cfg = self.config
        name = f"LSTM-{cfg.num_layers}L-{cfg.hidden_size}H"
        if cfg.bidirectional:
            name += "-Bi"
        if cfg.attention:
            name += "-Attn"
        return name


# =============================================================================
# Attention Module
# =============================================================================


class SelfAttention(nn.Module):
    """
    Self-attention over sequence for aggregation.
    
    Learns attention weights over timesteps and returns weighted sum.
    This is a simplified attention (not multi-head) suitable for
    aggregating LSTM outputs.
    
    Args:
        hidden_size: Size of input features
    
    Input:
        x: Tensor of shape [batch, seq_len, hidden_size]
    
    Output:
        context: Tensor of shape [batch, hidden_size]
    """
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute attention-weighted sum.
        
        Args:
            x: Input [batch, seq_len, hidden]
        
        Returns:
            Context [batch, hidden]
        """
        # Compute attention scores
        scores = self.attention(x)  # [batch, seq_len, 1]
        weights = F.softmax(scores, dim=1)  # [batch, seq_len, 1]
        
        # Weighted sum
        context = torch.sum(weights * x, dim=1)  # [batch, hidden]
        
        return context


# =============================================================================
# GRU Variant
# =============================================================================


class GRUClassifier(nn.Module):
    """
    GRU classifier for sequence-to-label prediction.
    
    GRU is a simpler alternative to LSTM with fewer parameters.
    Uses the same configuration as LSTMClassifier for consistency.
    
    Args:
        config: LSTMConfig with hyperparameters
    
    Reference:
        Cho et al. (2014). Learning Phrase Representations using RNN
                          Encoder-Decoder for Statistical Machine Translation.
    """
    
    def __init__(self, config: Optional[LSTMConfig] = None):
        super().__init__()
        
        self.config = config or LSTMConfig()
        cfg = self.config
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0,
            bidirectional=cfg.bidirectional,
        )
        
        # Calculate classifier input dimension
        gru_output_size = cfg.hidden_size * (2 if cfg.bidirectional else 1)
        
        # Classification head
        self.head_dropout = nn.Dropout(cfg.head_dropout)
        self.classifier = nn.Linear(gru_output_size, cfg.num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights."""
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # GRU forward
        gru_out, h_n = self.gru(x)
        
        # Use final hidden state
        if self.config.bidirectional:
            h_forward = h_n[-2, :, :]
            h_backward = h_n[-1, :, :]
            context = torch.cat([h_forward, h_backward], dim=1)
        else:
            context = h_n[-1, :, :]
        
        # Classification
        context = self.head_dropout(context)
        logits = self.classifier(context)
        
        return logits
    
    @property
    def name(self) -> str:
        cfg = self.config
        name = f"GRU-{cfg.num_layers}L-{cfg.hidden_size}H"
        if cfg.bidirectional:
            name += "-Bi"
        return name


# =============================================================================
# Factory Functions
# =============================================================================


def create_lstm(
    input_size: int = 98,
    hidden_size: int = 64,
    num_layers: int = 2,
    num_classes: int = 3,
    dropout: float = 0.2,
    bidirectional: bool = False,
    attention: bool = False,
) -> LSTMClassifier:
    """
    Factory function to create LSTM model.
    
    Args:
        input_size: Number of input features
        hidden_size: Hidden dimension
        num_layers: Number of LSTM layers
        num_classes: Number of output classes
        dropout: Dropout probability
        bidirectional: Use bidirectional LSTM
        attention: Use attention mechanism
    
    Returns:
        LSTMClassifier instance
    """
    config = LSTMConfig(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        bidirectional=bidirectional,
        attention=attention,
    )
    return LSTMClassifier(config)


def create_gru(
    input_size: int = 98,
    hidden_size: int = 64,
    num_layers: int = 2,
    num_classes: int = 3,
    dropout: float = 0.2,
    bidirectional: bool = False,
) -> GRUClassifier:
    """
    Factory function to create GRU model.
    
    Args:
        input_size: Number of input features
        hidden_size: Hidden dimension
        num_layers: Number of GRU layers
        num_classes: Number of output classes
        dropout: Dropout probability
        bidirectional: Use bidirectional GRU
    
    Returns:
        GRUClassifier instance
    """
    config = LSTMConfig(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        bidirectional=bidirectional,
    )
    return GRUClassifier(config)

