"""
Data module for LOB Model Trainer.

Provides PyTorch Dataset and DataLoader implementations for loading
LOB features exported from the Rust pipeline.

Supports both export formats:
- NEW aligned: *_sequences.npy [N_seq, 100, 98] - 3D, 1:1 with labels
- LEGACY flat: *_features.npy [N_samples, 98] - 2D, requires alignment

Key classes:
- LOBFlatDataset: For flat features (Logistic, XGBoost, MLP)
- LOBSequenceDataset: For 3D sequences (LSTM, Transformer)
- load_numpy_data: For sklearn-based models
"""

from lobtrainer.data.dataset import (
    DayData,
    LOBFlatDataset,
    LOBSequenceDataset,
    load_day_data,
    load_split_data,
    load_numpy_data,
    create_dataloaders,
    get_dataset_info,
)
from lobtrainer.data.transforms import (
    Normalizer,
    ZScoreNormalizer,
    compute_statistics,
    BinaryLabelTransform,
    ComposeTransform,
)

__all__ = [
    # Data structures
    "DayData",
    # PyTorch datasets
    "LOBFlatDataset",
    "LOBSequenceDataset",
    # Data loading
    "load_day_data",
    "load_split_data",
    "load_numpy_data",
    "create_dataloaders",
    "get_dataset_info",
    # Transforms
    "Normalizer",
    "ZScoreNormalizer",
    "compute_statistics",
    "BinaryLabelTransform",
    "ComposeTransform",
]
