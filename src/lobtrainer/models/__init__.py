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

# lobmodels is a declared dependency (lob-models in pyproject.toml)
from lobmodels import (
    DeepLOB,
    DeepLOBConfig,
    FeatureLayout,
    # TLOB imports (Berti & Kasneci 2025)
    TLOB,
    TLOBConfig,
    # LogisticLOB baseline
    LogisticLOB,
    LogisticLOBConfig,
    SequencePooling,
    # HMHP imports (Hierarchical Multi-Horizon Predictor)
    HierarchicalMultiHorizonPredictor,
    HMHPConfig,
    CascadeMode,
    StateFusion,
    create_hmhp,
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


def create_model(config, *, sequence_length: int = 100) -> nn.Module:
    """
    Create a model from configuration using the ModelRegistry.

    The model's architecture-specific parameters come from config.params,
    which is auto-populated from legacy flat fields in ModelConfig.__post_init__
    for backward compatibility with existing YAML configs.

    Injected parameters (not in params dict, added automatically):
        - sequence_length: from data.sequence.window_size
        - task_type: from train.task_type (for classification vs regression head)

    Args:
        config: ModelConfig with name/model_type and params dict.
        sequence_length: Temporal dimension of input sequences. Default: 100.

    Returns:
        PyTorch model instance.

    Raises:
        ValueError: If model name is not registered.
        ImportError: If lobmodels package is not installed.
    """
    from lobtrainer.config import ModelConfig, ModelType
    from lobmodels.registry import ModelRegistry

    if isinstance(config, dict):
        config = ModelConfig(**config)

    model_name = config.name

    # HMHP models use their own factory functions (they need sequence_length
    # in the factory call, not in the config). Delegate to lobmodels factories.
    if model_name in ("hmhp", "hmhp_regressor"):
        return _create_hmhp_model(config, sequence_length)

    # All other models: use ModelRegistry
    try:
        entry = ModelRegistry.get(model_name)
    except KeyError:
        available = ModelRegistry.list_models()
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available models: {available}"
        )

    # Build params: start from config.params, inject sequence_length if accepted
    import inspect
    params = {**config.params}
    config_sig = inspect.signature(entry.config_class.__init__)
    if "sequence_length" in config_sig.parameters and "sequence_length" not in params:
        params["sequence_length"] = sequence_length

    # LogisticLOB needs SequencePooling enum conversion
    if model_name == "logistic_lob" and "pooling" in params:
        if isinstance(params["pooling"], str):
            params["pooling"] = SequencePooling(params["pooling"])

    # Construct model via registry
    model_config = entry.config_class(**params)
    model = entry.model_class(model_config)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Created {model_name} via registry ({num_params:,} params)")
    return model


def _create_hmhp_model(config, sequence_length: int) -> nn.Module:
    """Create HMHP or HMHP-R model via lobmodels factory functions.

    HMHP factories accept individual keyword arguments rather than a config
    object, so we unpack config.params into the factory call.
    """
    params = {**config.params}
    params["sequence_length"] = sequence_length

    if config.name == "hmhp":
        # create_hmhp() accepts num_classes as explicit kwarg
        params["num_classes"] = config.num_classes
        model = create_hmhp(**params)
        logger.info(
            f"Created HMHP via factory ({model.num_parameters:,} params, "
            f"horizons={config.hmhp_horizons})"
        )
    else:
        from lobmodels.models.hmhp_regressor import create_hmhp_regressor
        model = create_hmhp_regressor(**params)
        logger.info(
            f"Created HMHP-R via factory ({model.num_parameters:,} params, "
            f"horizons={config.hmhp_horizons})"
        )

    return model


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
