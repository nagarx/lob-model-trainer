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

Normalization (matching official TLOB repo):
- GlobalZScoreNormalizer: Global Z-score with training stats for all prices/sizes
- GlobalNormalizationStats: Container for normalization statistics
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
    compute_statistics,
    BinaryLabelTransform,
    ComposeTransform,
)
from lobtrainer.data.normalization import (
    GlobalZScoreNormalizer,
    GlobalNormalizationStats,
    HybridNormalizer,
    HybridNormalizationStats,
    compute_and_save_normalization_stats,
    compute_hybrid_stats_streaming,
    load_or_compute_hybrid_stats,
)
from lobtrainer.data.feature_selector import (
    FeatureSelector,
    create_feature_selector,
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
    # Transforms (per-sample)
    "compute_statistics",
    "BinaryLabelTransform",
    "ComposeTransform",
    # Global normalization (TLOB repo matching)
    "GlobalZScoreNormalizer",
    "GlobalNormalizationStats",
    "compute_and_save_normalization_stats",
    # Hybrid normalization (98-feature datasets)
    "HybridNormalizer",
    "HybridNormalizationStats",
    "compute_hybrid_stats_streaming",
    "load_or_compute_hybrid_stats",
    # Feature selection
    "FeatureSelector",
    "create_feature_selector",
]
