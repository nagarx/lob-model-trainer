"""
Data loading utilities for Phase 2A analysis.

Handles:
- Loading train/val/test splits
- Aligning features (sample-level) with labels (sequence-level)
"""

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
    Load all available splits.
    
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


def align_features_with_labels(
    features: np.ndarray, 
    n_labels: int,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
) -> np.ndarray:
    """
    Align features (sample-level) with labels (sequence-level).
    
    Each label corresponds to the END of a sequence window.
    We extract the feature vector at the end of each window for alignment.
    
    Args:
        features: (N_samples, N_features) array
        n_labels: Number of labels to align with
        window_size: Samples per sequence window
        stride: Samples between sequence starts
    
    Returns:
        aligned_features: (n_labels, N_features) array
    
    Formula:
        For label[i], the corresponding feature is at:
        feat_idx = i * stride + window_size - 1
        
    This is the last feature in the sequence window [i*stride, i*stride + window_size)
    """
    n_features = features.shape[1]
    aligned = np.zeros((n_labels, n_features), dtype=features.dtype)
    
    for i in range(n_labels):
        # Feature index at end of sequence window
        feat_idx = min(i * stride + window_size - 1, features.shape[0] - 1)
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

