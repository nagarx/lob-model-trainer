"""
Data loading utilities for Phase 2A analysis.

Handles:
- Loading train/val/test splits
- Aligning features (sample-level) with labels (sequence-level)

CRITICAL: For multi-day datasets, always use load_split_aligned() to get
correct feature-label alignment. The align_features_with_labels() function
only works correctly for SINGLE-DAY data.

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


def load_split(data_dir: Path, split_name: str) -> Dict:
    """
    Load all data for a single split (train/val/test).
    
    Args:
        data_dir: Path to dataset root (e.g., data/exports/nvda_98feat)
        split_name: One of 'train', 'val', 'test'
    
    Returns:
        dict with:
            - features: (N_samples, 98) array
            - labels: (N_labels,) array
            - n_days: number of trading days
            - dates: list of date strings
    """
    split_dir = data_dir / split_name
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    
    features_list = []
    labels_list = []
    dates = []
    
    for feat_file in sorted(split_dir.glob('*_features.npy')):
        date = feat_file.stem.replace('_features', '')
        label_file = feat_file.parent / f"{date}_labels.npy"
        
        if not label_file.exists():
            raise FileNotFoundError(f"Label file not found: {label_file}")
        
        features_list.append(np.load(feat_file))
        labels_list.append(np.load(label_file))
        dates.append(date)
    
    if not features_list:
        raise ValueError(f"No data files found in {split_dir}")
    
    return {
        'features': np.vstack(features_list),
        'labels': np.concatenate(labels_list),
        'n_days': len(features_list),
        'dates': dates,
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
) -> Dict:
    """
    Load split with CORRECT day-boundary-aware alignment.
    
    This function aligns features with labels PER DAY before concatenating,
    ensuring correct 1:1 correspondence between aligned features and labels.
    
    CRITICAL: This is the ONLY correct way to load multi-day data for 
    signal-label correlation analysis.
    
    The Bug This Fixes:
        When days have different lengths, using a global alignment formula
        causes cumulative drift. For 165 days of NVDA data, this drift 
        reaches 28,200 samples (2,820 labels) - causing ~10x underestimation
        of signal-label correlations.
    
    Args:
        data_dir: Path to dataset root (e.g., data/exports/nvda_98feat)
        split_name: One of 'train', 'val', 'test'
        window_size: Samples per sequence window (default: 100)
        stride: Samples between sequence starts (default: 10)
    
    Returns:
        dict with:
            - features: (N_labels, 98) aligned features - 1:1 with labels
            - labels: (N_labels,) label array
            - n_days: number of trading days
            - dates: list of date strings
            - day_boundaries: list of (start_idx, end_idx) for each day
    
    Formula (per day):
        For label[i] within a day, the corresponding feature is at:
        feat_idx = i * stride + window_size - 1
        
        This is the LAST feature in the sequence window for that label.
    
    Example:
        >>> data = load_split_aligned(Path("data/exports/nvda_98feat"), "train")
        >>> features = data['features']  # Shape: (N, 98) - aligned
        >>> labels = data['labels']      # Shape: (N,) - same length!
        >>> assert len(features) == len(labels)  # Always true
    """
    data_dir = Path(data_dir)
    split_dir = data_dir / split_name
    
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    
    aligned_features_list = []
    labels_list = []
    dates = []
    day_boundaries = []
    current_idx = 0
    
    feature_files = sorted(split_dir.glob('*_features.npy'))
    
    for feat_file in feature_files:
        date = feat_file.stem.replace('_features', '')
        label_file = feat_file.parent / f"{date}_labels.npy"
        
        if not label_file.exists():
            raise FileNotFoundError(f"Label file not found: {label_file}")
        
        # Load day data
        day_features = np.load(feat_file)
        day_labels = np.load(label_file)
        n_labels = len(day_labels)
        
        # Validate label count against expected formula
        expected_labels = (day_features.shape[0] - window_size) // stride + 1
        if n_labels != expected_labels:
            warnings.warn(
                f"Day {date}: Label count {n_labels} != expected {expected_labels} "
                f"(features={day_features.shape[0]}, window={window_size}, stride={stride}). "
                f"Using actual label count."
            )
        
        # Align PER DAY (critical fix)
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
        'labels': np.concatenate(labels_list),
        'n_days': len(aligned_features_list),
        'dates': dates,
        'day_boundaries': day_boundaries,
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

