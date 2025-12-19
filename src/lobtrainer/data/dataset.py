"""
PyTorch Dataset implementations for LOB feature data.

Data Contract (from Rust pipeline):
    - Features: (N, 98) float64 arrays - flat per-sample features
    - Labels: (N_seq,) int8 arrays - sequence labels (0=Down, 1=Stable, 2=Up)
    - Relationship: N_seq = (N - window_size) // stride + 1

Files are organized as:
    data_dir/
    ├── train/
    │   ├── 20250203_features.npy
    │   ├── 20250203_labels.npy
    │   ├── 20250203_metadata.json
    │   └── ...
    ├── val/
    └── test/

Design principles:
- Memory-efficient: Option to stream from disk or cache in memory
- Deterministic: Same sequence of data for reproducibility
- Validated: Check data integrity on load
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json

import numpy as np
import torch
from torch.utils.data import Dataset

from lobtrainer.constants import FEATURE_COUNT, FeatureIndex


# =============================================================================
# Data Loading Functions
# =============================================================================


@dataclass
class DayData:
    """
    Data from a single trading day.
    
    Attributes:
        date: Trading date as string (YYYYMMDD)
        features: Feature array of shape (N, 98)
        labels: Label array of shape (N_seq,)
        metadata: Optional metadata dict
    """
    date: str
    features: np.ndarray
    labels: np.ndarray
    metadata: Optional[Dict] = None
    
    @property
    def num_samples(self) -> int:
        """Number of flat feature samples."""
        return self.features.shape[0]
    
    @property
    def num_sequences(self) -> int:
        """Number of labeled sequences."""
        return len(self.labels)
    
    def validate(self, feature_count: int = FEATURE_COUNT) -> None:
        """
        Validate data integrity.
        
        Raises:
            ValueError: If data is invalid
        """
        if self.features.ndim != 2:
            raise ValueError(f"Expected 2D features, got shape {self.features.shape}")
        if self.features.shape[1] != feature_count:
            raise ValueError(
                f"Expected {feature_count} features, got {self.features.shape[1]}"
            )
        if self.labels.ndim != 1:
            raise ValueError(f"Expected 1D labels, got shape {self.labels.shape}")
        
        # Check for NaN/Inf in features (excluding book_valid=0 samples)
        book_valid = self.features[:, FeatureIndex.BOOK_VALID]
        valid_mask = book_valid > 0.5
        if valid_mask.any():
            valid_features = self.features[valid_mask]
            if not np.isfinite(valid_features).all():
                nan_count = np.isnan(valid_features).sum()
                inf_count = np.isinf(valid_features).sum()
                raise ValueError(
                    f"Found {nan_count} NaN and {inf_count} Inf values in valid samples"
                )


def load_day_data(
    features_path: Union[str, Path],
    labels_path: Union[str, Path],
    metadata_path: Optional[Union[str, Path]] = None,
    validate: bool = True,
) -> DayData:
    """
    Load data for a single trading day.
    
    Args:
        features_path: Path to features .npy file
        labels_path: Path to labels .npy file
        metadata_path: Optional path to metadata .json file
        validate: Whether to validate data integrity
    
    Returns:
        DayData instance
    
    Raises:
        FileNotFoundError: If files don't exist
        ValueError: If data validation fails
    """
    features_path = Path(features_path)
    labels_path = Path(labels_path)
    
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    # Extract date from filename (e.g., "20250203_features.npy" -> "20250203")
    date = features_path.stem.split("_")[0]
    
    # Load arrays with explicit dtype
    features = np.load(features_path).astype(np.float64)
    labels = np.load(labels_path).astype(np.int64)
    
    # Load metadata if provided
    metadata = None
    if metadata_path is not None:
        metadata_path = Path(metadata_path)
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
    
    day_data = DayData(date=date, features=features, labels=labels, metadata=metadata)
    
    if validate:
        day_data.validate()
    
    return day_data


def load_split_data(
    data_dir: Union[str, Path],
    split: str = "train",
    validate: bool = True,
) -> List[DayData]:
    """
    Load all data for a split (train/val/test).
    
    Args:
        data_dir: Root data directory
        split: Split name ("train", "val", or "test")
        validate: Whether to validate data integrity
    
    Returns:
        List of DayData instances, sorted by date
    
    Raises:
        FileNotFoundError: If split directory doesn't exist
    """
    data_dir = Path(data_dir)
    split_dir = data_dir / split
    
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    
    # Find all feature files
    feature_files = sorted(split_dir.glob("*_features.npy"))
    
    if not feature_files:
        raise FileNotFoundError(f"No feature files found in {split_dir}")
    
    days = []
    for feature_path in feature_files:
        date = feature_path.stem.split("_")[0]
        label_path = split_dir / f"{date}_labels.npy"
        metadata_path = split_dir / f"{date}_metadata.json"
        
        if not label_path.exists():
            raise FileNotFoundError(f"Missing labels for {date}: {label_path}")
        
        day_data = load_day_data(
            feature_path,
            label_path,
            metadata_path if metadata_path.exists() else None,
            validate=validate,
        )
        days.append(day_data)
    
    return days


# =============================================================================
# PyTorch Datasets
# =============================================================================


class LOBDataset(Dataset):
    """
    PyTorch Dataset for flat LOB features (no sequence windowing).
    
    Use this for models that don't need temporal context (e.g., XGBoost, MLP).
    
    Args:
        days: List of DayData instances
        transform: Optional transform to apply to features
    
    Example:
        >>> days = load_split_data("data/exports/nvda_98feat", "train")
        >>> dataset = LOBDataset(days)
        >>> features, label = dataset[0]
    """
    
    def __init__(
        self,
        days: List[DayData],
        transform: Optional[callable] = None,
    ):
        self.days = days
        self.transform = transform
        
        # Build index mapping: global_idx -> (day_idx, local_idx)
        self._index_map: List[Tuple[int, int]] = []
        for day_idx, day in enumerate(days):
            for local_idx in range(day.num_sequences):
                self._index_map.append((day_idx, local_idx))
        
        # Precompute cumulative sample counts for faster indexing
        self._cumulative_samples = np.cumsum([0] + [d.num_samples for d in days])
    
    def __len__(self) -> int:
        return len(self._index_map)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        day_idx, local_idx = self._index_map[idx]
        day = self.days[day_idx]
        
        # For flat dataset, we use the label's corresponding feature sample
        # Labels are aligned to sequence ends, so we need to compute the sample index
        # This is approximate for flat loading - use LOBSequenceDataset for precise alignment
        features = day.features[local_idx].copy()
        label = day.labels[local_idx]
        
        if self.transform is not None:
            features = self.transform(features)
        
        return (
            torch.from_numpy(features).float(),
            torch.tensor(label, dtype=torch.long),
        )
    
    @property
    def num_features(self) -> int:
        return self.days[0].features.shape[1] if self.days else 0
    
    @property
    def num_classes(self) -> int:
        return 3  # Down, Stable, Up


class LOBSequenceDataset(Dataset):
    """
    PyTorch Dataset for LOB feature sequences with temporal windowing.
    
    Builds sequences of length `window_size` from flat features with configurable
    stride. Labels are aligned to the END of each sequence.
    
    Args:
        days: List of DayData instances
        window_size: Sequence length
        stride: Step between consecutive sequences
        transform: Optional transform to apply to features
    
    Data alignment:
        - Sequence i: features[i*stride : i*stride + window_size]
        - Label i: aligned to sample at position i*stride + window_size - 1
    
    Example:
        >>> days = load_split_data("data/exports/nvda_98feat", "train")
        >>> dataset = LOBSequenceDataset(days, window_size=100, stride=10)
        >>> sequence, label = dataset[0]
        >>> sequence.shape  # (100, 98)
    """
    
    def __init__(
        self,
        days: List[DayData],
        window_size: int = 100,
        stride: int = 10,
        transform: Optional[callable] = None,
    ):
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        
        self.days = days
        self.window_size = window_size
        self.stride = stride
        self.transform = transform
        
        # Build index mapping: global_idx -> (day_idx, start_idx)
        self._index_map: List[Tuple[int, int]] = []
        self._label_offset: List[int] = []  # Offset into labels array for each day
        
        label_offset = 0
        for day_idx, day in enumerate(days):
            num_samples = day.num_samples
            # Number of complete sequences in this day
            num_sequences = (num_samples - window_size) // stride + 1
            
            # Validate that we have enough labels
            if num_sequences > day.num_sequences:
                raise ValueError(
                    f"Day {day.date}: Expected {num_sequences} sequences but only "
                    f"{day.num_sequences} labels available. Check window_size and stride."
                )
            
            for seq_idx in range(num_sequences):
                start_idx = seq_idx * stride
                self._index_map.append((day_idx, start_idx))
            
            self._label_offset.append(label_offset)
            label_offset += day.num_sequences
    
    def __len__(self) -> int:
        return len(self._index_map)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        day_idx, start_idx = self._index_map[idx]
        day = self.days[day_idx]
        
        # Extract sequence
        end_idx = start_idx + self.window_size
        sequence = day.features[start_idx:end_idx].copy()
        
        # Get corresponding label
        # Label index = sequence index within day
        seq_idx_in_day = start_idx // self.stride
        label = day.labels[seq_idx_in_day]
        
        if self.transform is not None:
            sequence = self.transform(sequence)
        
        return (
            torch.from_numpy(sequence).float(),
            torch.tensor(label, dtype=torch.long),
        )
    
    @property
    def num_features(self) -> int:
        return self.days[0].features.shape[1] if self.days else 0
    
    @property
    def num_classes(self) -> int:
        return 3  # Down, Stable, Up
    
    @property
    def sequence_shape(self) -> Tuple[int, int]:
        """Shape of each sequence: (window_size, num_features)."""
        return (self.window_size, self.num_features)


# =============================================================================
# Dataset Factory
# =============================================================================


def create_dataloaders(
    data_dir: Union[str, Path],
    window_size: int = 100,
    stride: int = 10,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    transform: Optional[callable] = None,
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create DataLoaders for train/val/test splits.
    
    Args:
        data_dir: Root data directory
        window_size: Sequence length
        stride: Step between sequences
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        transform: Optional feature transform
    
    Returns:
        Dict with 'train', 'val', 'test' DataLoaders
    """
    from torch.utils.data import DataLoader
    
    loaders = {}
    
    for split in ["train", "val", "test"]:
        try:
            days = load_split_data(data_dir, split, validate=True)
        except FileNotFoundError:
            continue
        
        dataset = LOBSequenceDataset(
            days,
            window_size=window_size,
            stride=stride,
            transform=transform,
        )
        
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(split == "train"),
        )
    
    return loaders

