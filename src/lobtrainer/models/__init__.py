"""
Model implementations for LOB price prediction.

Available models:
- Baselines: NaivePreviousLabel, NaiveClassPrior, LogisticBaseline
- Sequence: LSTMClassifier, GRUClassifier, TransformerClassifier (TODO)
- Deep: DeepLOB (TODO)

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
    
    Example:
        >>> from lobtrainer.config import ModelConfig, ModelType
        >>> config = ModelConfig(model_type=ModelType.LSTM, hidden_size=64)
        >>> model = create_model(config)
    """
    from lobtrainer.config import ModelConfig, ModelType
    
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
    
    elif model_type == ModelType.TRANSFORMER:
        raise NotImplementedError(
            "TransformerClassifier not yet implemented. "
            "Use LSTM or GRU for now."
        )
    
    elif model_type == ModelType.DEEPLOB:
        raise NotImplementedError(
            "DeepLOB not yet implemented. "
            "Use LSTM or GRU for now."
        )
    
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
