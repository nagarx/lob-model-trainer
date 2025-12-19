"""
Configuration module for LOB Model Trainer.

Provides type-safe, serializable configuration dataclasses for:
- Data loading and preprocessing
- Model architecture
- Training hyperparameters
"""

from lobtrainer.config.schema import (
    DataConfig,
    SequenceConfig,
    NormalizationConfig,
    ModelConfig,
    TrainConfig,
    ExperimentConfig,
    load_config,
    save_config,
)

__all__ = [
    "DataConfig",
    "SequenceConfig",
    "NormalizationConfig",
    "ModelConfig",
    "TrainConfig",
    "ExperimentConfig",
    "load_config",
    "save_config",
]

