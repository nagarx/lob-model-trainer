"""
LOB Model Trainer: ML experimentation for limit order book price prediction.

This package provides tools for training and evaluating machine learning models
on limit order book data exported from the feature-extractor-MBO-LOB pipeline.

Data Contract:
    - Features: (N, 98) float64 arrays from Rust pipeline
    - Labels: (N_seq,) or (N_seq, n_horizons) int8 arrays (0=Down, 1=Stable, 2=Up)
    - Sequences: (N_seq, 100, 98) for sequence models

Feature Layout (98 total):
    - 0-39: Raw LOB features (10 levels × 4 values)
    - 40-47: Derived features (spread, microprice, etc.)
    - 48-83: MBO features (order flow, queue stats, etc.)
    - 84-97: Trading signals (OFI, asymmetry, regime, etc.)

Quick Start:
    >>> from lobtrainer import create_trainer
    >>> from lobtrainer.config import load_config
    >>> 
    >>> config = load_config("configs/baseline_lstm.yaml")
    >>> trainer = create_trainer(config)
    >>> trainer.train()
    >>> metrics = trainer.evaluate("test")
    >>> print(metrics.summary())

See: plan/03-FEATURE-INDEX-MAP-v2.md for complete index mapping.
"""

__version__ = "0.2.0"
__author__ = "Knight"

# Constants
from lobtrainer.constants import FeatureIndex, SignalIndex, FEATURE_COUNT

# Configuration
from lobtrainer.config import (
    DataConfig,
    ModelConfig,
    TrainConfig,
    ExperimentConfig,
    load_config,
    save_config,
)

# Data loading
from lobtrainer.data import (
    DayData,
    LOBFlatDataset,
    LOBSequenceDataset,
    load_split_data,
    create_dataloaders,
)

# Models
from lobtrainer.models import (
    create_model,
    LSTMClassifier,
    GRUClassifier,
    LogisticBaseline,
)

# Training
from lobtrainer.training import (
    Trainer,
    create_trainer,
    EarlyStopping,
    ModelCheckpoint,
    ClassificationMetrics,
)

# Utilities
from lobtrainer.utils import set_seed

__all__ = [
    # Version
    "__version__",
    # Constants
    "FeatureIndex",
    "SignalIndex",
    "FEATURE_COUNT",
    # Configuration
    "DataConfig",
    "ModelConfig",
    "TrainConfig",
    "ExperimentConfig",
    "load_config",
    "save_config",
    # Data
    "DayData",
    "LOBFlatDataset",
    "LOBSequenceDataset",
    "load_split_data",
    "create_dataloaders",
    # Models
    "create_model",
    "LSTMClassifier",
    "GRUClassifier",
    "LogisticBaseline",
    # Training
    "Trainer",
    "create_trainer",
    "EarlyStopping",
    "ModelCheckpoint",
    "ClassificationMetrics",
    # Utilities
    "set_seed",
]
