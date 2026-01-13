"""
Model implementations for LOB price prediction.

Available models:
- Baselines: NaivePreviousLabel, NaiveClassPrior, LogisticBaseline
- Sequence: LSTMClassifier, GRUClassifier, TransformerClassifier (TODO)
- Deep: DeepLOB (from lobmodels package)

Design principles (RULE.md):
- Each model class has a consistent interface
- Configuration-driven hyperparameters
- Factory function for creating models from config
- Deterministic predictions given same input

Usage:
    >>> from lobtrainer.models import create_model
    >>> from lobtrainer.config import ModelConfig, ModelType
    >>> 
    >>> config = ModelConfig(model_type=ModelType.LSTM, hidden_size=128)
    >>> model = create_model(config)
    >>> 
    >>> # DeepLOB model
    >>> config = ModelConfig(model_type=ModelType.DEEPLOB)
    >>> model = create_model(config)
"""

import logging
from typing import Union

import torch.nn as nn

from lobtrainer.models.baselines import (
    BaseModel,
    NaivePreviousLabel,
    NaiveClassPrior,
    LogisticBaseline,
    LogisticBaselineConfig,
)
from lobtrainer.models.lstm import (
    LSTMClassifier,
    GRUClassifier,
    LSTMConfig,
    create_lstm,
    create_gru,
)

# Conditional import of lobmodels (external dependency)
try:
    from lobmodels import (
        DeepLOB,
        DeepLOBConfig,
        FeatureLayout,
        # TLOB imports (Berti & Kasneci 2025)
        TLOB,
        TLOBConfig,
    )
    LOBMODELS_AVAILABLE = True
except ImportError:
    LOBMODELS_AVAILABLE = False
    DeepLOB = None
    DeepLOBConfig = None
    FeatureLayout = None
    TLOB = None
    TLOBConfig = None

logger = logging.getLogger(__name__)


__all__ = [
    # Base
    "BaseModel",
    # Baselines
    "NaivePreviousLabel",
    "NaiveClassPrior",
    "LogisticBaseline",
    "LogisticBaselineConfig",
    # Sequence models
    "LSTMClassifier",
    "GRUClassifier",
    "LSTMConfig",
    # Factory functions
    "create_model",
    "create_lstm",
    "create_gru",
    # Availability flags
    "LOBMODELS_AVAILABLE",
]


# =============================================================================
# Model Factory
# =============================================================================


def create_model(config) -> nn.Module:
    """
    Create a model from configuration.
    
    Args:
        config: ModelConfig or dict with model settings
    
    Returns:
        PyTorch model instance
    
    Raises:
        ValueError: If model type is not recognized
        ImportError: If DeepLOB is requested but lobmodels is not installed
    
    Example:
        >>> from lobtrainer.config import ModelConfig, ModelType
        >>> config = ModelConfig(model_type=ModelType.LSTM, hidden_size=64)
        >>> model = create_model(config)
        >>> 
        >>> # DeepLOB model
        >>> config = ModelConfig(model_type=ModelType.DEEPLOB)
        >>> model = create_model(config)
    """
    from lobtrainer.config import ModelConfig, ModelType, DeepLOBMode
    
    # Handle dict input
    if isinstance(config, dict):
        config = ModelConfig(**config)
    
    model_type = config.model_type
    
    if model_type == ModelType.LSTM:
        # Get attention parameter, default to False for backward compatibility
        attention = getattr(config, 'lstm_attention', False)
        lstm_config = LSTMConfig(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_classes=config.num_classes,
            dropout=config.dropout,
            bidirectional=config.lstm_bidirectional,
            attention=attention,
        )
        model = LSTMClassifier(lstm_config)
        logger.info(f"Created {model.name}")
        return model
    
    elif model_type == ModelType.GRU:
        lstm_config = LSTMConfig(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_classes=config.num_classes,
            dropout=config.dropout,
            bidirectional=config.lstm_bidirectional,
        )
        model = GRUClassifier(lstm_config)
        logger.info(f"Created {model.name}")
        return model
    
    elif model_type == ModelType.LOGISTIC:
        raise ValueError(
            "LogisticBaseline uses sklearn interface, not PyTorch. "
            "Use LogisticBaseline class directly."
        )
    
    elif model_type == ModelType.XGBOOST:
        raise NotImplementedError(
            "XGBoost model not yet implemented in PyTorch trainer. "
            "Use LogisticBaseline or LSTM for now."
        )
    
    elif model_type == ModelType.TRANSFORMER:
        raise NotImplementedError(
            "TransformerClassifier not yet implemented. "
            "Use LSTM or GRU for now."
        )
    
    elif model_type == ModelType.DEEPLOB:
        if not LOBMODELS_AVAILABLE:
            raise ImportError(
                "lobmodels package is required for DeepLOB. "
                "Install it with: pip install -e ../lob-models"
            )
        
        # Map lobtrainer DeepLOBMode to lobmodels mode string
        mode_str = config.deeplob_mode.value  # "benchmark" or "extended"
        
        # Create DeepLOBConfig from lobmodels
        deeplob_config = DeepLOBConfig(
            mode=mode_str,
            feature_layout=FeatureLayout.GROUPED,  # Our data uses grouped layout
            num_levels=config.deeplob_num_levels,
            sequence_length=100,  # Standard DeepLOB input length
            num_classes=config.num_classes,
            conv_filters=config.deeplob_conv_filters,
            inception_filters=config.deeplob_inception_filters,
            lstm_hidden=config.deeplob_lstm_hidden,
            lstm_layers=1,  # Paper uses single layer
            dropout=config.dropout,
        )
        
        model = DeepLOB(deeplob_config)
        logger.info(f"Created {model.name}")
        return model
    
    elif model_type == ModelType.TLOB:
        if not LOBMODELS_AVAILABLE:
            raise ImportError(
                "lobmodels package is required for TLOB. "
                "Install it with: pip install -e ../lob-models"
            )
        
        # Create TLOBConfig from lobmodels
        # Reference: Berti & Kasneci (2025), "TLOB: A Novel Transformer Model..."
        tlob_config = TLOBConfig(
            num_features=config.input_size,
            sequence_length=100,  # Standard window size from Rust export
            num_classes=config.num_classes,
            hidden_dim=config.tlob_hidden_dim,
            num_layers=config.tlob_num_layers,
            num_heads=config.tlob_num_heads,
            mlp_expansion=config.tlob_mlp_expansion,
            use_sinusoidal_pe=config.tlob_use_sinusoidal_pe,
            use_bin=config.tlob_use_bin,
            dropout=config.dropout,
            dataset_type=config.tlob_dataset_type,
        )
        
        model = TLOB(tlob_config)
        logger.info(
            f"Created {model.name} with {model.num_parameters:,} parameters "
            f"(hidden_dim={config.tlob_hidden_dim}, layers={config.tlob_num_layers})"
        )
        return model
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_model_info(model: nn.Module) -> dict:
    """
    Get information about a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dict with model info:
        - name: Model name
        - num_params: Total parameters
        - trainable_params: Trainable parameters
    """
    num_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    name = getattr(model, 'name', model.__class__.__name__)
    
    return {
        'name': name,
        'num_params': num_params,
        'trainable_params': trainable,
    }
