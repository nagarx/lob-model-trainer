"""
PyTorch Dataset implementations for LOB feature data.

Data Contract (Rust pipeline Schema v2.1):
    - NEW aligned format (*_sequences.npy): [N_seq, 100, 98] - 3D sequences, 1:1 with labels
    - LEGACY flat format (*_features.npy): [N_samples, 98] - requires manual alignment
    - Labels:
        - Single-horizon: [N_seq] int8 arrays - (-1=Down, 0=Stable, 1=Up)
        - Multi-horizon: [N_seq, n_horizons] int8 arrays - one label per horizon

Files are organized as:
    data_dir/
    ├── train/
    │   ├── 20250203_sequences.npy  # NEW: [N_seq, 100, 98]
    │   ├── 20250203_labels.npy     # [N_seq] or [N_seq, n_horizons]
    │   ├── 20250203_metadata.json
    │   └── ...
    ├── val/
    └── test/

Design principles (RULE.md):
- Memory-efficient: Option to stream from disk or cache in memory
- Deterministic: Same sequence of data for reproducibility
- Validated: Check data integrity on load
- Format-agnostic: Auto-detect and handle both export formats
- Horizon-flexible: Support single and multi-horizon labels
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import logging

import numpy as np
import torch
from torch.utils.data import Dataset

from lobtrainer.constants import FEATURE_COUNT, FeatureIndex

logger = logging.getLogger(__name__)


# =============================================================================
# Data Loading Functions
# =============================================================================


def _detect_export_format(split_dir: Path) -> str:
    """
    Detect whether directory contains new aligned format or legacy format.
    
    Returns:
        'aligned' for *_sequences.npy (3D), 'legacy' for *_features.npy (2D)
    """
    seq_files = list(split_dir.glob('*_sequences.npy'))
    feat_files = list(split_dir.glob('*_features.npy'))
    
    if seq_files:
        return 'aligned'
    elif feat_files:
        return 'legacy'
    else:
        raise FileNotFoundError(f"No data files found in {split_dir}")


@dataclass
class DayData:
    """
    Data from a single trading day.
    
    For aligned format:
        - features: [N_seq, 98] - last timestep of each sequence (for flat models)
        - sequences: [N_seq, 100, 98] - full 3D sequences (for sequence models)
        - labels: [N_seq] or [N_seq, n_horizons] - 1:1 aligned with features/sequences
    
    For legacy format:
        - features: [N_samples, 98] - flat samples (NOT aligned with labels)
        - sequences: None
        - labels: [N_labels] - requires manual alignment
    
    Multi-horizon labels:
        - When labels.ndim == 2, each column is a different prediction horizon
        - horizons metadata indicates the actual horizon values
        - Use get_labels(horizon_idx) to select a specific horizon
    """
    date: str
    features: np.ndarray  # [N, 98] - flat features
    labels: np.ndarray    # [N_seq], [N_labels], or [N_seq, n_horizons]
    sequences: Optional[np.ndarray] = None  # [N_seq, 100, 98] - only for aligned format
    metadata: Optional[Dict] = None
    is_aligned: bool = False  # True if features are 1:1 with labels
    
    @property
    def num_samples(self) -> int:
        """Number of flat feature samples."""
        return self.features.shape[0]
    
    @property
    def num_sequences(self) -> int:
        """Number of labeled sequences."""
        return self.labels.shape[0]
    
    @property
    def window_size(self) -> int:
        """Window size (100 for aligned, 0 for legacy)."""
        if self.sequences is not None:
            return self.sequences.shape[1]
        return 0
    
    @property
    def is_multi_horizon(self) -> bool:
        """Check if labels have multiple horizons."""
        return self.labels.ndim == 2
    
    @property
    def num_horizons(self) -> int:
        """Number of prediction horizons (1 for single-horizon, n for multi-horizon)."""
        if self.is_multi_horizon:
            return self.labels.shape[1]
        return 1
    
    @property
    def horizons(self) -> Optional[List[int]]:
        """Get horizon values from metadata (if available).
        
        Checks both top-level 'horizons' (new format) and 
        'label_config.horizons' (legacy format) for compatibility.
        """
        if self.metadata:
            # New format: horizons at top level
            if 'horizons' in self.metadata:
                return self.metadata['horizons']
            # Legacy format: nested in label_config
            if 'label_config' in self.metadata:
                label_config = self.metadata['label_config']
                if 'horizons' in label_config:
                    return label_config['horizons']
        return None
    
    def get_labels(self, horizon_idx: Optional[int] = None) -> np.ndarray:
        """
        Get labels for a specific horizon.
        
        Args:
            horizon_idx: Index of horizon to select (0-based).
                         None returns all labels (1D for single-horizon, 2D for multi).
        
        Returns:
            Label array of shape [N_seq] for single horizon or selected horizon,
            or [N_seq, n_horizons] for all multi-horizon labels.
        """
        if horizon_idx is None:
            return self.labels
        
        if not self.is_multi_horizon:
            if horizon_idx != 0:
                raise ValueError(
                    f"Single-horizon data only has horizon_idx=0, got {horizon_idx}"
                )
            return self.labels
        
        if horizon_idx < 0 or horizon_idx >= self.num_horizons:
            raise ValueError(
                f"horizon_idx {horizon_idx} out of range [0, {self.num_horizons})"
            )
        
        return self.labels[:, horizon_idx]
    
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
        # Labels can be 1D [N_seq] or 2D [N_seq, n_horizons]
        if self.labels.ndim not in (1, 2):
            raise ValueError(f"Expected 1D or 2D labels, got shape {self.labels.shape}")
        
        # For aligned format, check 1:1 alignment
        if self.is_aligned and len(self.features) != self.labels.shape[0]:
            raise ValueError(
                f"Aligned format requires features[0] == labels[0], got "
                f"{len(self.features)} features and {self.labels.shape[0]} labels"
            )
        
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
    data_file: Union[str, Path],
    labels_path: Union[str, Path],
    metadata_path: Optional[Union[str, Path]] = None,
    validate: bool = True,
) -> DayData:
    """
    Load data for a single trading day.
    
    Auto-detects format:
    - *_sequences.npy: 3D aligned format [N_seq, 100, 98]
    - *_features.npy: 2D legacy format [N_samples, 98]
    
    Args:
        data_file: Path to data .npy file (sequences or features)
        labels_path: Path to labels .npy file
        metadata_path: Optional path to metadata .json file
        validate: Whether to validate data integrity
    
    Returns:
        DayData instance
    
    Raises:
        FileNotFoundError: If files don't exist
        ValueError: If data validation fails
    """
    data_file = Path(data_file)
    labels_path = Path(labels_path)
    
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    # Detect format from filename
    is_aligned = '_sequences' in data_file.stem
    
    # Extract date from filename
    date = data_file.stem.replace('_sequences', '').replace('_features', '')
    
    # Load data
    raw_data = np.load(data_file)
    labels = np.load(labels_path).astype(np.int64)
    
    # Handle 3D sequences vs 2D flat
    if is_aligned and len(raw_data.shape) == 3:
        # NEW format: [N_seq, 100, 98]
        sequences = raw_data.astype(np.float64)
        features = sequences[:, -1, :]  # Extract last timestep for flat models
    else:
        # LEGACY format: [N_samples, 98]
        sequences = None
        features = raw_data.astype(np.float64)
    
    # Load metadata if provided
    metadata = None
    if metadata_path is not None:
        metadata_path = Path(metadata_path)
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
    
    day_data = DayData(
        date=date,
        features=features,
        labels=labels,
        sequences=sequences,
        metadata=metadata,
        is_aligned=is_aligned,
    )
    
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
    
    Auto-detects format (aligned 3D or legacy 2D).
    
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
    
    # Detect format
    export_format = _detect_export_format(split_dir)
    
    if export_format == 'aligned':
        data_files = sorted(split_dir.glob("*_sequences.npy"))
        suffix = '_sequences'
    else:
        data_files = sorted(split_dir.glob("*_features.npy"))
        suffix = '_features'
    
    if not data_files:
        raise FileNotFoundError(f"No data files found in {split_dir}")
    
    days = []
    for data_file in data_files:
        date = data_file.stem.replace(suffix, '')
        label_path = split_dir / f"{date}_labels.npy"
        metadata_path = split_dir / f"{date}_metadata.json"
        
        if not label_path.exists():
            raise FileNotFoundError(f"Missing labels for {date}: {label_path}")
        
        day_data = load_day_data(
            data_file,
            label_path,
            metadata_path if metadata_path.exists() else None,
            validate=validate,
        )
        days.append(day_data)
    
    logger.info(f"Loaded {len(days)} days from {split} ({export_format} format)")
    return days


# =============================================================================
# PyTorch Datasets
# =============================================================================


class LOBFlatDataset(Dataset):
    """
    PyTorch Dataset for flat LOB features (no sequence dimension).
    
    Use this for models that don't need temporal context (e.g., Logistic, XGBoost, MLP).
    
    For aligned format: Uses last timestep of each sequence.
    For legacy format: Uses raw flat features.
    
    Supports both single-horizon and multi-horizon labels:
        - Single-horizon: Returns scalar label per sample
        - Multi-horizon: Specify horizon_idx to select one horizon, or get all
    
    Args:
        days: List of DayData instances
        transform: Optional transform to apply to features
        feature_indices: Optional list of feature indices to select
        horizon_idx: For multi-horizon labels, which horizon to use (0-based).
                     None returns all horizons (for multi-output models).
    
    Example:
        >>> days = load_split_data("data/exports/nvda_98feat_full", "train")
        >>> dataset = LOBFlatDataset(days)
        >>> features, label = dataset[0]
        >>> features.shape  # (98,) or (n_selected,)
        
        >>> # Multi-horizon: select horizon index 2
        >>> dataset = LOBFlatDataset(days, horizon_idx=2)
        >>> features, label = dataset[0]
        >>> label.shape  # () - scalar
        
        >>> # Multi-horizon: get all horizons
        >>> dataset = LOBFlatDataset(days, horizon_idx=None)
        >>> features, label = dataset[0]
        >>> label.shape  # (n_horizons,) if multi-horizon
    """
    
    def __init__(
        self,
        days: List[DayData],
        transform: Optional[callable] = None,
        feature_indices: Optional[List[int]] = None,
        horizon_idx: Optional[int] = 0,  # Default: first horizon (backward compatible)
    ):
        self.days = days
        self.transform = transform
        self.feature_indices = feature_indices
        self.horizon_idx = horizon_idx
        
        # Build index mapping: global_idx -> (day_idx, local_idx)
        self._index_map: List[Tuple[int, int]] = []
        for day_idx, day in enumerate(days):
            for local_idx in range(day.num_sequences):
                self._index_map.append((day_idx, local_idx))
        
        # Verify we have aligned data
        self._is_aligned = all(d.is_aligned for d in days)
        if not self._is_aligned:
            logger.warning(
                "Dataset contains legacy (non-aligned) data. "
                "Features may not be correctly aligned with labels."
            )
        
        # Detect multi-horizon
        self._is_multi_horizon = any(d.is_multi_horizon for d in days) if days else False
        if self._is_multi_horizon:
            self._num_horizons = days[0].num_horizons
            logger.info(f"Multi-horizon labels detected: {self._num_horizons} horizons")
            if horizon_idx is not None:
                logger.info(f"Using horizon index {horizon_idx}")
        else:
            self._num_horizons = 1
    
    def __len__(self) -> int:
        return len(self._index_map)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        day_idx, local_idx = self._index_map[idx]
        day = self.days[day_idx]
        
        # Get features (already extracted from sequences for aligned format)
        features = day.features[local_idx].copy()
        
        # Get labels - handle multi-horizon
        if self._is_multi_horizon:
            if self.horizon_idx is not None:
                label = day.get_labels(self.horizon_idx)[local_idx]
            else:
                label = day.labels[local_idx]  # All horizons: [n_horizons]
        else:
            label = day.labels[local_idx]
        
        # Select specific features if requested
        if self.feature_indices is not None:
            features = features[self.feature_indices]
        
        if self.transform is not None:
            features = self.transform(features)
        
        # Handle label tensor dtype
        # IMPORTANT: Labels are stored as {-1, 0, 1} but PyTorch CrossEntropyLoss
        # expects {0, 1, ..., num_classes-1}, so we shift by +1 to get {0, 1, 2}
        if isinstance(label, np.ndarray) and label.ndim > 0:
            # Multi-horizon: [n_horizons]
            label_shifted = label.astype(np.int64) + 1  # {-1,0,1} -> {0,1,2}
            label_tensor = torch.from_numpy(label_shifted)
        else:
            # Single-horizon: scalar
            label_tensor = torch.tensor(int(label) + 1, dtype=torch.long)
        
        return (
            torch.from_numpy(features).float(),
            label_tensor,
        )
    
    @property
    def num_features(self) -> int:
        if self.feature_indices is not None:
            return len(self.feature_indices)
        return self.days[0].features.shape[1] if self.days else 0
    
    @property
    def num_classes(self) -> int:
        return 3  # Down, Stable, Up
    
    @property
    def is_aligned(self) -> bool:
        return self._is_aligned
    
    @property
    def is_multi_horizon(self) -> bool:
        """Check if dataset has multi-horizon labels."""
        return self._is_multi_horizon
    
    @property
    def num_horizons(self) -> int:
        """Number of prediction horizons."""
        return self._num_horizons
    
    @property
    def horizons(self) -> Optional[List[int]]:
        """Get horizon values from metadata (if available)."""
        if self.days and self.days[0].horizons:
            return self.days[0].horizons
        return None


class LOBSequenceDataset(Dataset):
    """
    PyTorch Dataset for LOB feature sequences with temporal dimension.
    
    Use this for sequence models (LSTM, Transformer, etc.).
    
    REQUIRES aligned format (*_sequences.npy). For legacy format,
    use LOBFlatDataset or manually window the data.
    
    Supports both single-horizon and multi-horizon labels:
        - Single-horizon: Returns scalar label per sample
        - Multi-horizon: Specify horizon_idx to select one horizon, or get all
    
    Args:
        days: List of DayData instances (must be aligned format)
        transform: Optional transform to apply to sequences
        feature_indices: Optional list of feature indices to select
        horizon_idx: For multi-horizon labels, which horizon to use (0-based).
                     None returns all horizons (for multi-output models).
    
    Example:
        >>> days = load_split_data("data/exports/nvda_98feat_full", "train")
        >>> dataset = LOBSequenceDataset(days)
        >>> sequence, label = dataset[0]
        >>> sequence.shape  # (100, 98) or (100, n_selected)
        
        >>> # Multi-horizon: select horizon index 2
        >>> dataset = LOBSequenceDataset(days, horizon_idx=2)
        >>> sequence, label = dataset[0]
        >>> label.shape  # () - scalar
    """
    
    def __init__(
        self,
        days: List[DayData],
        transform: Optional[callable] = None,
        feature_indices: Optional[List[int]] = None,
        horizon_idx: Optional[int] = 0,  # Default: first horizon (backward compatible)
    ):
        # Verify all days have sequences
        for day in days:
            if day.sequences is None:
                raise ValueError(
                    f"Day {day.date} has no sequences. "
                    "LOBSequenceDataset requires aligned format (*_sequences.npy)."
                )
        
        self.days = days
        self.transform = transform
        self.feature_indices = feature_indices
        self.horizon_idx = horizon_idx
        
        # Build index mapping: global_idx -> (day_idx, local_idx)
        self._index_map: List[Tuple[int, int]] = []
        for day_idx, day in enumerate(days):
            for local_idx in range(day.num_sequences):
                self._index_map.append((day_idx, local_idx))
        
        # Detect multi-horizon
        self._is_multi_horizon = any(d.is_multi_horizon for d in days) if days else False
        if self._is_multi_horizon:
            self._num_horizons = days[0].num_horizons
            logger.info(f"Multi-horizon labels detected: {self._num_horizons} horizons")
            if horizon_idx is not None:
                logger.info(f"Using horizon index {horizon_idx}")
        else:
            self._num_horizons = 1
    
    def __len__(self) -> int:
        return len(self._index_map)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        day_idx, local_idx = self._index_map[idx]
        day = self.days[day_idx]
        
        # Get full sequence
        sequence = day.sequences[local_idx].copy()
        
        # Get labels - handle multi-horizon
        if self._is_multi_horizon:
            if self.horizon_idx is not None:
                label = day.get_labels(self.horizon_idx)[local_idx]
            else:
                label = day.labels[local_idx]  # All horizons: [n_horizons]
        else:
            label = day.labels[local_idx]
        
        # Select specific features if requested
        if self.feature_indices is not None:
            sequence = sequence[:, self.feature_indices]
        
        if self.transform is not None:
            sequence = self.transform(sequence)
        
        # Handle label tensor dtype
        # IMPORTANT: Labels are stored as {-1, 0, 1} but PyTorch CrossEntropyLoss
        # expects {0, 1, ..., num_classes-1}, so we shift by +1 to get {0, 1, 2}
        if isinstance(label, np.ndarray) and label.ndim > 0:
            # Multi-horizon: [n_horizons]
            label_shifted = label.astype(np.int64) + 1  # {-1,0,1} -> {0,1,2}
            label_tensor = torch.from_numpy(label_shifted)
        else:
            # Single-horizon: scalar
            label_tensor = torch.tensor(int(label) + 1, dtype=torch.long)
        
        return (
            torch.from_numpy(sequence).float(),
            label_tensor,
        )
    
    @property
    def num_features(self) -> int:
        if self.feature_indices is not None:
            return len(self.feature_indices)
        return self.days[0].sequences.shape[2] if self.days else 0
    
    @property
    def num_classes(self) -> int:
        return 3  # Down, Stable, Up
    
    @property
    def sequence_length(self) -> int:
        return self.days[0].sequences.shape[1] if self.days else 0
    
    @property
    def sequence_shape(self) -> Tuple[int, int]:
        """Shape of each sequence: (sequence_length, num_features)."""
        return (self.sequence_length, self.num_features)
    
    @property
    def is_multi_horizon(self) -> bool:
        """Check if dataset has multi-horizon labels."""
        return self._is_multi_horizon
    
    @property
    def num_horizons(self) -> int:
        """Number of prediction horizons."""
        return self._num_horizons
    
    @property
    def horizons(self) -> Optional[List[int]]:
        """Get horizon values from metadata (if available)."""
        if self.days and self.days[0].horizons:
            return self.days[0].horizons
        return None


# =============================================================================
# Dataset Factory
# =============================================================================


def create_dataloaders(
    data_dir: Union[str, Path],
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    transform: Optional[callable] = None,
    feature_indices: Optional[List[int]] = None,
    use_sequences: bool = True,
    horizon_idx: Optional[int] = 0,
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create DataLoaders for train/val/test splits.
    
    Args:
        data_dir: Root data directory
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        transform: Optional feature transform
        feature_indices: Optional feature selection
        use_sequences: If True, return LOBSequenceDataset; if False, return LOBFlatDataset
        horizon_idx: For multi-horizon labels, which horizon to use (0-based).
                     None returns all horizons (for multi-output models).
    
    Returns:
        Dict with 'train', 'val', 'test' DataLoaders
    """
    from torch.utils.data import DataLoader
    
    loaders = {}
    
    for split in ["train", "val", "test"]:
        try:
            days = load_split_data(data_dir, split, validate=True)
        except FileNotFoundError:
            logger.info(f"Split '{split}' not found, skipping")
            continue
        
        # Choose dataset type
        if use_sequences:
            try:
                dataset = LOBSequenceDataset(
                    days,
                    transform=transform,
                    feature_indices=feature_indices,
                    horizon_idx=horizon_idx,
                )
            except ValueError:
                logger.warning(
                    f"Split '{split}' has no sequences, falling back to flat dataset"
                )
                dataset = LOBFlatDataset(
                    days,
                    transform=transform,
                    feature_indices=feature_indices,
                    horizon_idx=horizon_idx,
                )
        else:
            dataset = LOBFlatDataset(
                days,
                transform=transform,
                feature_indices=feature_indices,
                horizon_idx=horizon_idx,
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


# =============================================================================
# NumPy-based data loading (for sklearn models)
# =============================================================================


def load_numpy_data(
    data_dir: Union[str, Path],
    split: str = "train",
    feature_indices: Optional[List[int]] = None,
    flat: bool = True,
    horizon_idx: Optional[int] = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load data as NumPy arrays (for sklearn/XGBoost).
    
    Args:
        data_dir: Root data directory
        split: Split name
        feature_indices: Optional feature selection
        flat: If True, return 2D features; if False, return 3D sequences
        horizon_idx: For multi-horizon labels, which horizon to use (0-based).
                     None returns all horizons as 2D array [N, n_horizons].
    
    Returns:
        Tuple of (features, labels) arrays
        - features: [N, 98] if flat, [N, 100, 98] if sequences
        - labels: [N] if single horizon or horizon_idx specified, [N, n_horizons] otherwise
    """
    days = load_split_data(data_dir, split, validate=True)
    
    features_list = []
    labels_list = []
    
    for day in days:
        if flat:
            features_list.append(day.features)
        else:
            if day.sequences is None:
                raise ValueError(f"Day {day.date} has no sequences")
            features_list.append(day.sequences)
        
        # Handle horizon selection
        if horizon_idx is not None:
            labels_list.append(day.get_labels(horizon_idx))
        else:
            labels_list.append(day.labels)
    
    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    
    # Select features
    if feature_indices is not None:
        if flat:
            features = features[:, feature_indices]
        else:
            features = features[:, :, feature_indices]
    
    return features, labels


def get_dataset_info(data_dir: Union[str, Path]) -> Dict:
    """
    Get information about a dataset without loading all data.
    
    Returns:
        Dict with keys:
        - num_days: Number of trading days
        - is_multi_horizon: Whether labels have multiple horizons
        - num_horizons: Number of prediction horizons
        - horizons: List of horizon values (if available in metadata)
        - feature_count: Number of features
        - export_format: 'aligned' or 'legacy'
    """
    data_dir = Path(data_dir)
    
    # Try to find any split
    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        if split_dir.exists():
            break
    else:
        raise FileNotFoundError(f"No splits found in {data_dir}")
    
    export_format = _detect_export_format(split_dir)
    
    # Load first day to get info
    if export_format == 'aligned':
        data_files = sorted(split_dir.glob("*_sequences.npy"))
        suffix = '_sequences'
    else:
        data_files = sorted(split_dir.glob("*_features.npy"))
        suffix = '_features'
    
    if not data_files:
        raise FileNotFoundError(f"No data files in {split_dir}")
    
    first_file = data_files[0]
    date = first_file.stem.replace(suffix, '')
    label_path = split_dir / f"{date}_labels.npy"
    metadata_path = split_dir / f"{date}_metadata.json"
    
    # Load labels to check shape
    labels = np.load(label_path)
    
    # Load metadata if available
    metadata = None
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    
    # Count total days
    total_days = 0
    for s in ["train", "val", "test"]:
        s_dir = data_dir / s
        if s_dir.exists():
            if export_format == 'aligned':
                total_days += len(list(s_dir.glob("*_sequences.npy")))
            else:
                total_days += len(list(s_dir.glob("*_features.npy")))
    
    # Get feature count
    if export_format == 'aligned':
        data = np.load(first_file)
        feature_count = data.shape[2] if len(data.shape) == 3 else data.shape[1]
    else:
        data = np.load(first_file)
        feature_count = data.shape[1]
    
    # Multi-horizon info
    is_multi_horizon = labels.ndim == 2
    num_horizons = labels.shape[1] if is_multi_horizon else 1
    
    horizons = None
    if metadata:
        # New format: horizons at top level
        if 'horizons' in metadata:
            horizons = metadata['horizons']
        # Legacy format: nested in label_config
        elif 'label_config' in metadata:
            horizons = metadata['label_config'].get('horizons')
    
    return {
        'num_days': total_days,
        'is_multi_horizon': is_multi_horizon,
        'num_horizons': num_horizons,
        'horizons': horizons,
        'feature_count': feature_count,
        'export_format': export_format,
    }
