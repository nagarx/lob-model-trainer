"""
Configuration module for LOB Model Trainer.

Provides type-safe, serializable configuration dataclasses for:
- Data loading and preprocessing
- Model architecture
- Training hyperparameters
"""

from lobtrainer.config.schema import (
    # Configuration classes
    DataConfig,
    SequenceConfig,
    NormalizationConfig,
    ModelConfig,
    TrainConfig,
    ExperimentConfig,
    # Enums
    ModelType,
    NormalizationStrategy,
    LabelEncoding,
    # Functions
    load_config,
    save_config,
)

__all__ = [
    # Configuration classes
    "DataConfig",
    "SequenceConfig",
    "NormalizationConfig",
    "ModelConfig",
    "TrainConfig",
    "ExperimentConfig",
    # Enums
    "ModelType",
    "NormalizationStrategy",
    "LabelEncoding",
    # Functions
    "load_config",
    "save_config",
]

