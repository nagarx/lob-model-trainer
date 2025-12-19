"""
LOB Model Trainer: ML experimentation for limit order book price prediction.

This package provides tools for training and evaluating machine learning models
on limit order book data exported from the feature-extractor-MBO-LOB pipeline.

Data Contract:
    - Features: (N, 98) float64 arrays from Rust pipeline
    - Labels: (N_seq,) int8 arrays (0=Down, 1=Stable, 2=Up)
    - Sequences: Built from features with configurable window_size and stride

Feature Layout (98 total):
    - 0-39: Raw LOB features (10 levels × 4 values)
    - 40-47: Derived features (spread, microprice, etc.)
    - 48-83: MBO features (order flow, queue stats, etc.)
    - 84-97: Trading signals (OFI, asymmetry, regime, etc.)

See: plan/03-FEATURE-INDEX-MAP-v2.md for complete index mapping.
"""

__version__ = "0.1.0"
__author__ = "Knight"

from lobtrainer.constants import FeatureIndex, SignalIndex
from lobtrainer.config import DataConfig, ModelConfig, TrainConfig

__all__ = [
    "FeatureIndex",
    "SignalIndex",
    "DataConfig",
    "ModelConfig",
    "TrainConfig",
    "__version__",
]

