"""
Data loading utilities for Phase 2A analysis.

Handles:
- Loading train/val/test splits from both formats:
  - NEW (aligned): *_sequences.npy [N_seq, 100, 98] + *_labels.npy [N_seq]
  - LEGACY: *_features.npy [N_samples, 98] + *_labels.npy [N_labels]

For the NEW aligned format, sequences and labels are already 1:1 aligned.
For analysis, we extract the LAST timestep (sequence endpoint) from 3D sequences.

See: docs/nvda_alignment_validation.json for quantified alignment drift.
"""

import warnings
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

from ..constants import FEATURE_COUNT

# Export configuration (must match Rust pipeline)
WINDOW_SIZE = 100  # Samples per sequence
STRIDE = 10        # Samples between sequence starts


def _detect_export_format(split_dir: Path) -> str:
    """Detect whether directory contains new aligned format or legacy format."""
    seq_files = list(split_dir.glob('*_sequences.npy'))
    feat_files = list(split_dir.glob('*_features.npy'))
    
    if seq_files:
        return 'aligned'
    elif feat_files:
        return 'legacy'
    else:
        raise ValueError(f"No data files found in {split_dir}")


def load_split(data_dir: Path, split_name: str, horizon_idx: Optional[int] = 0) -> Dict:
    """
    Load all data for a single split (train/val/test).
    
    Automatically detects and handles both export formats:
    - NEW aligned: *_sequences.npy [N_seq, 100, 98] - extracts last timestep
    - LEGACY: *_features.npy [N_samples, 98]
    
    Supports multi-horizon labels:
    - Single-horizon: labels shape is (N,)
    - Multi-horizon: labels shape is (N, n_horizons), use horizon_idx to select one
    
    Args:
        data_dir: Path to dataset root (e.g., data/exports/nvda_98feat_full)
        split_name: One of 'train', 'val', 'test'
        horizon_idx: Which horizon to use for multi-horizon labels (0-based).
                     None returns all horizons. Default: 0 (first horizon).
    
    Returns:
        dict with:
            - features: (N, 98) array - aligned with labels
            - labels: (N,) array or (N, n_horizons) if horizon_idx=None
            - n_days: number of trading days
            - dates: list of date strings
            - format: 'aligned' or 'legacy'
            - is_multi_horizon: bool
            - num_horizons: int
    """
    split_dir = data_dir / split_name
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    
    export_format = _detect_export_format(split_dir)
    
    features_list = []
    labels_list = []
    dates = []
    is_multi_horizon = False
    num_horizons = 1
    
    if export_format == 'aligned':
        # NEW format: *_sequences.npy [N_seq, window_size, n_features]
        for seq_file in sorted(split_dir.glob('*_sequences.npy')):
            date = seq_file.stem.replace('_sequences', '')
            label_file = seq_file.parent / f"{date}_labels.npy"
            
            if not label_file.exists():
                raise FileNotFoundError(f"Label file not found: {label_file}")
            
            # Load 3D sequences and extract LAST timestep (sequence endpoint)
            sequences = np.load(seq_file)  # [N_seq, 100, 98]
            if len(sequences.shape) == 3:
                # Extract last timestep for 2D analysis compatibility
                features = sequences[:, -1, :]  # [N_seq, 98]
            else:
                features = sequences  # Already 2D
            
            labels = np.load(label_file)
            
            # Detect multi-horizon
            if labels.ndim == 2:
                is_multi_horizon = True
                num_horizons = labels.shape[1]
                if horizon_idx is not None:
                    labels = labels[:, horizon_idx]
            
            # Validate 1:1 alignment (new format guarantees this)
            if len(features) != labels.shape[0]:
                warnings.warn(
                    f"Day {date}: Feature/label mismatch - {len(features)} features vs {labels.shape[0]} labels"
                )
            
            features_list.append(features)
            labels_list.append(labels)
            dates.append(date)
    else:
        # LEGACY format: *_features.npy [N_samples, n_features]
        for feat_file in sorted(split_dir.glob('*_features.npy')):
            date = feat_file.stem.replace('_features', '')
            label_file = feat_file.parent / f"{date}_labels.npy"
            
            if not label_file.exists():
                raise FileNotFoundError(f"Label file not found: {label_file}")
            
            features_list.append(np.load(feat_file))
            labels = np.load(label_file)
            
            # Detect multi-horizon
            if labels.ndim == 2:
                is_multi_horizon = True
                num_horizons = labels.shape[1]
                if horizon_idx is not None:
                    labels = labels[:, horizon_idx]
            
            labels_list.append(labels)
            dates.append(date)
    
    if not features_list:
        raise ValueError(f"No data files found in {split_dir}")
    
    return {
        'features': np.vstack(features_list),
        'labels': np.concatenate(labels_list, axis=0),
        'n_days': len(features_list),
        'dates': dates,
        'format': export_format,
        'is_multi_horizon': is_multi_horizon,
        'num_horizons': num_horizons,
    }


def load_all_splits(data_dir: Path) -> Dict[str, Dict]:
    """
    Load all available splits (raw, unaligned).
    
    WARNING: For signal-label correlation analysis, use load_split_aligned() instead.
    
    Returns:
        dict mapping split_name -> split_data
    """
    data_dir = Path(data_dir)
    splits = {}
    
    for split_name in ['train', 'val', 'test']:
        split_path = data_dir / split_name
        if split_path.exists():
            splits[split_name] = load_split(data_dir, split_name)
    
    return splits


def load_split_aligned(
    data_dir: Path,
    split_name: str,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
    horizon_idx: Optional[int] = 0,
) -> Dict:
    """
    Load split with CORRECT day-boundary-aware alignment.
    
    Automatically handles both export formats:
    - NEW aligned format (*_sequences.npy): Already 1:1 aligned, just extract last timestep
    - LEGACY format (*_features.npy): Apply per-day alignment formula
    
    Supports multi-horizon labels:
    - Single-horizon: labels shape is (N,)
    - Multi-horizon: labels shape is (N, n_horizons), use horizon_idx to select one
    
    Args:
        data_dir: Path to dataset root (e.g., data/exports/nvda_98feat_full)
        split_name: One of 'train', 'val', 'test'
        window_size: Samples per sequence window (default: 100)
        stride: Samples between sequence starts (default: 10)
        horizon_idx: Which horizon to use for multi-horizon labels (0-based).
                     None returns all horizons. Default: 0 (first horizon).
    
    Returns:
        dict with:
            - features: (N, 98) aligned features - 1:1 with labels
            - labels: (N,) label array or (N, n_horizons) if horizon_idx=None
            - n_days: number of trading days
            - dates: list of date strings
            - day_boundaries: list of (start_idx, end_idx) for each day
            - format: 'aligned' or 'legacy'
            - is_multi_horizon: bool
            - num_horizons: int
    
    Example:
        >>> data = load_split_aligned(Path("data/exports/nvda_98feat_full"), "train")
        >>> features = data['features']  # Shape: (N, 98) - aligned
        >>> labels = data['labels']      # Shape: (N,) - same length!
        >>> assert len(features) == len(labels)  # Always true
    """
    data_dir = Path(data_dir)
    split_dir = data_dir / split_name
    
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    
    export_format = _detect_export_format(split_dir)
    
    aligned_features_list = []
    labels_list = []
    dates = []
    day_boundaries = []
    current_idx = 0
    is_multi_horizon = False
    num_horizons = 1
    
    if export_format == 'aligned':
        # NEW format: Already aligned! Just extract last timestep
        for seq_file in sorted(split_dir.glob('*_sequences.npy')):
            date = seq_file.stem.replace('_sequences', '')
            label_file = seq_file.parent / f"{date}_labels.npy"
            
            if not label_file.exists():
                raise FileNotFoundError(f"Label file not found: {label_file}")
            
            # Load 3D sequences and extract LAST timestep
            sequences = np.load(seq_file)  # [N_seq, 100, 98]
            if len(sequences.shape) == 3:
                day_features = sequences[:, -1, :]  # [N_seq, 98]
            else:
                day_features = sequences
            
            day_labels = np.load(label_file)
            
            # Detect multi-horizon
            if day_labels.ndim == 2:
                is_multi_horizon = True
                num_horizons = day_labels.shape[1]
                n_labels = day_labels.shape[0]
                if horizon_idx is not None:
                    day_labels = day_labels[:, horizon_idx]
            else:
                n_labels = len(day_labels)
            
            # Track day boundaries
            start_idx = current_idx
            end_idx = current_idx + n_labels
            day_boundaries.append((start_idx, end_idx))
            current_idx = end_idx
            
            aligned_features_list.append(day_features)
            labels_list.append(day_labels)
            dates.append(date)
    else:
        # LEGACY format: Need to apply alignment per day
        for feat_file in sorted(split_dir.glob('*_features.npy')):
            date = feat_file.stem.replace('_features', '')
            label_file = feat_file.parent / f"{date}_labels.npy"
            
            if not label_file.exists():
                raise FileNotFoundError(f"Label file not found: {label_file}")
            
            # Load day data
            day_features = np.load(feat_file)
            day_labels = np.load(label_file)
            
            # Detect multi-horizon
            if day_labels.ndim == 2:
                is_multi_horizon = True
                num_horizons = day_labels.shape[1]
                n_labels = day_labels.shape[0]
                if horizon_idx is not None:
                    day_labels = day_labels[:, horizon_idx]
            else:
                n_labels = len(day_labels)
            
            # Validate label count against expected formula
            expected_labels = (day_features.shape[0] - window_size) // stride + 1
            if n_labels != expected_labels:
                warnings.warn(
                    f"Day {date}: Label count {n_labels} != expected {expected_labels} "
                    f"(features={day_features.shape[0]}, window={window_size}, stride={stride}). "
                    f"Using actual label count."
                )
            
            # Align PER DAY (critical for legacy format)
            day_aligned = align_features_with_labels(
                day_features, n_labels, window_size, stride
            )
            
            # Track day boundaries
            start_idx = current_idx
            end_idx = current_idx + n_labels
            day_boundaries.append((start_idx, end_idx))
            current_idx = end_idx
            
            aligned_features_list.append(day_aligned)
            labels_list.append(day_labels)
            dates.append(date)
    
    if not aligned_features_list:
        raise ValueError(f"No data files found in {split_dir}")
    
    return {
        'features': np.vstack(aligned_features_list),
        'labels': np.concatenate(labels_list, axis=0),
        'n_days': len(aligned_features_list),
        'dates': dates,
        'day_boundaries': day_boundaries,
        'format': export_format,
        'is_multi_horizon': is_multi_horizon,
        'num_horizons': num_horizons,
    }


def align_features_with_labels(
    features: np.ndarray, 
    n_labels: int,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
) -> np.ndarray:
    """
    Align features (sample-level) with labels (sequence-level) for a SINGLE day.
    
    WARNING: This function only works correctly for SINGLE-DAY data!
    For multi-day datasets, use load_split_aligned() which applies this 
    function per-day before concatenating.
    
    Each label corresponds to the END of a sequence window.
    We extract the feature vector at the end of each window for alignment.
    
    Args:
        features: (N_samples, N_features) array from a SINGLE day
        n_labels: Number of labels for this day
        window_size: Samples per sequence window
        stride: Samples between sequence starts
    
    Returns:
        aligned_features: (n_labels, N_features) array
    
    Formula:
        For label[i], the corresponding feature is at:
        feat_idx = i * stride + window_size - 1
        
        This is the LAST feature in the sequence window:
        [i * stride, i * stride + window_size)
    
    Example (single day with 1000 samples, window=100, stride=10):
        - label[0] → feature[99]   (end of window [0, 100))
        - label[1] → feature[109]  (end of window [10, 110))
        - label[2] → feature[119]  (end of window [20, 120))
        - ...
        - label[90] → feature[999] (end of window [900, 1000))
    """
    n_features = features.shape[1]
    aligned = np.zeros((n_labels, n_features), dtype=features.dtype)
    
    for i in range(n_labels):
        # Feature index at end of sequence window
        feat_idx = i * stride + window_size - 1
        
        # Boundary check with warning
        if feat_idx >= features.shape[0]:
            # This should only happen if n_labels doesn't match the formula
            feat_idx = features.shape[0] - 1
        
        aligned[i] = features[feat_idx]
    
    return aligned


def get_signal_info() -> Dict[int, Dict]:
    """
    Return metadata about each signal (indices 84-97).
    
    Returns:
        dict mapping signal_index -> {name, description, type, expected_sign}
    """
    return {
        84: {
            'name': 'true_ofi',
            'description': 'Cont et al. Order Flow Imbalance',
            'type': 'continuous',
            'expected_sign': '+',  # Positive OFI → expect Up
        },
        85: {
            'name': 'depth_norm_ofi',
            'description': 'OFI normalized by average depth',
            'type': 'continuous',
            'expected_sign': '+',
        },
        86: {
            'name': 'executed_pressure',
            'description': 'Net executed trade imbalance',
            'type': 'continuous',
            'expected_sign': '+',
        },
        87: {
            'name': 'signed_mp_delta_bps',
            'description': 'Microprice deviation from mid (bps)',
            'type': 'continuous',
            'expected_sign': '+',
        },
        88: {
            'name': 'trade_asymmetry',
            'description': 'Trade count imbalance ratio',
            'type': 'continuous',
            'expected_sign': '+',
        },
        89: {
            'name': 'cancel_asymmetry',
            'description': 'Cancel imbalance ratio',
            'type': 'continuous',
            'expected_sign': '+',
        },
        90: {
            'name': 'fragility_score',
            'description': 'Book concentration / ln(depth)',
            'type': 'continuous',
            'expected_sign': '?',
        },
        91: {
            'name': 'depth_asymmetry',
            'description': 'Depth imbalance ratio',
            'type': 'continuous',
            'expected_sign': '+',
        },
        92: {
            'name': 'book_valid',
            'description': 'Book validity flag',
            'type': 'binary',
            'expected_sign': 'N/A',
        },
        93: {
            'name': 'time_regime',
            'description': 'Market session encoding',
            'type': 'categorical',
            'expected_sign': 'N/A',
        },
        94: {
            'name': 'mbo_ready',
            'description': 'MBO warmup complete flag',
            'type': 'binary',
            'expected_sign': 'N/A',
        },
        95: {
            'name': 'dt_seconds',
            'description': 'Time since last sample',
            'type': 'continuous',
            'expected_sign': '?',
        },
        96: {
            'name': 'invalidity_delta',
            'description': 'Count of feed problems',
            'type': 'count',
            'expected_sign': '-',
        },
        97: {
            'name': 'schema_version',
            'description': 'Schema version constant',
            'type': 'constant',
            'expected_sign': 'N/A',
        },
    }


# Core signals for analysis (exclude categorical/binary/constant)
CORE_SIGNAL_INDICES = [84, 85, 86, 87, 88, 89, 90, 91]

