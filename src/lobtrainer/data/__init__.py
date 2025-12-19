"""
Data module for LOB Model Trainer.

Provides PyTorch Dataset and DataLoader implementations for loading
LOB features exported from the Rust pipeline.

Key classes:
- LOBDataset: PyTorch Dataset for sequence loading
- LOBDataModule: High-level data loading interface
- Transforms: Feature normalization and preprocessing
"""

from lobtrainer.data.dataset import (
    LOBDataset,
    LOBSequenceDataset,
    load_day_data,
    load_split_data,
)
from lobtrainer.data.transforms import (
    Normalizer,
    ZScoreNormalizer,
    compute_statistics,
)

__all__ = [
    "LOBDataset",
    "LOBSequenceDataset",
    "load_day_data",
    "load_split_data",
    "Normalizer",
    "ZScoreNormalizer",
    "compute_statistics",
]

