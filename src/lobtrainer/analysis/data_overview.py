"""
Data overview and validation utilities.

Provides comprehensive data quality checks and validation functions
that work with any symbol/dataset exported from the Rust pipeline.

Functions:
- validate_file_structure: Check file existence and naming
- compute_shape_validation: Verify feature/label dimensions
- compute_data_quality: Check NaN, Inf, finite values
- compute_categorical_validation: Validate categorical features
- compute_signal_statistics: Basic statistics for signals
- generate_data_summary: Aggregate summary for a dataset
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
from dataclasses import dataclass, asdict

from ..constants import (
    FEATURE_COUNT, 
    LOB_FEATURE_COUNT, 
    DERIVED_FEATURE_COUNT,
    MBO_FEATURE_COUNT, 
    SIGNAL_FEATURE_COUNT,
    SCHEMA_VERSION,
    FeatureIndex,
    LABEL_DOWN, LABEL_STABLE, LABEL_UP, LABEL_NAMES,
    SAFETY_GATES,  # Categorical/safety features
)


@dataclass
class FileInventory:
    """Inventory of data files for a split."""
    split: str
    feature_files: List[Path]
    label_files: List[Path]
    metadata_files: List[Path]
    dates: List[str]
    
    @property
    def n_days(self) -> int:
        return len(self.feature_files)


@dataclass
class ShapeValidation:
    """Validation results for a single day's data."""
    split: str
    date: str
    n_samples: int
    n_features: int
    n_labels: int
    feature_dtype: str
    label_dtype: str
    feature_dim_ok: bool
    sample_label_ratio: float


@dataclass
class DataQuality:
    """Data quality metrics for a feature array."""
    total_values: int
    finite_count: int
    nan_count: int
    inf_count: int
    pct_finite: float
    pct_nan: float
    pct_inf: float
    columns_with_nan: List[int]
    columns_with_inf: List[int]
    
    @property
    def is_clean(self) -> bool:
        return self.nan_count == 0 and self.inf_count == 0


@dataclass
class LabelDistribution:
    """Label distribution statistics."""
    total: int
    down_count: int
    stable_count: int
    up_count: int
    down_pct: float
    stable_pct: float
    up_pct: float
    imbalance_ratio: float
    
    @property
    def is_balanced(self) -> bool:
        return self.imbalance_ratio < 1.5


@dataclass
class CategoricalValidation:
    """Validation results for a categorical feature."""
    name: str
    index: int
    unique_values: List[float]
    expected_values: Optional[List[float]]
    value_counts: Dict[float, int]
    is_valid: bool
    message: str


@dataclass  
class SignalStatistics:
    """Statistics for a single signal feature."""
    index: int
    name: str
    signal_type: str
    mean: float
    std: float
    min_val: float
    max_val: float
    median: float
    q25: float
    q75: float
    n_unique: int
    pct_outliers: float  # |z| > 4


def discover_files(data_dir: Path, split: str) -> FileInventory:
    """
    Discover all data files for a split.
    
    Args:
        data_dir: Root directory (e.g., data/exports/nvda_98feat)
        split: One of 'train', 'val', 'test'
    
    Returns:
        FileInventory with lists of discovered files
    """
    split_dir = data_dir / split
    
    # Detect format: try new format first
    seq_files = sorted(split_dir.glob('*_sequences.npy'))
    if seq_files:
        # NEW format: *_sequences.npy
        feature_files = seq_files
        dates = [f.stem.replace('_sequences', '') for f in seq_files]
    else:
        # LEGACY format: *_features.npy
        feature_files = sorted(split_dir.glob('*_features.npy'))
        dates = [f.stem.replace('_features', '') for f in feature_files]
    
    label_files = sorted(split_dir.glob('*_labels.npy'))
    metadata_files = sorted(split_dir.glob('*_metadata.json'))
    
    return FileInventory(
        split=split,
        feature_files=feature_files,
        label_files=label_files,
        metadata_files=metadata_files,
        dates=dates,
    )


def validate_file_structure(data_dir: Path) -> Dict[str, FileInventory]:
    """
    Validate file structure for all splits.
    
    Args:
        data_dir: Root directory
    
    Returns:
        dict mapping split_name -> FileInventory
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    inventories = {}
    for split in ['train', 'val', 'test']:
        split_dir = data_dir / split
        if split_dir.exists():
            inventories[split] = discover_files(data_dir, split)
    
    if not inventories:
        raise ValueError(f"No split directories found in {data_dir}")
    
    return inventories


def compute_shape_validation(
    features: np.ndarray, 
    labels: np.ndarray,
    split: str,
    date: str,
    expected_features: int = FEATURE_COUNT,
) -> ShapeValidation:
    """
    Validate shapes and dtypes for a single day's data.
    
    Args:
        features: Feature array
        labels: Label array
        split: Split name
        date: Date string
        expected_features: Expected feature count (default: 98)
    
    Returns:
        ShapeValidation with validation results
    """
    n_samples, n_features = features.shape
    n_labels = len(labels)
    
    return ShapeValidation(
        split=split,
        date=date,
        n_samples=n_samples,
        n_features=n_features,
        n_labels=n_labels,
        feature_dtype=str(features.dtype),
        label_dtype=str(labels.dtype),
        feature_dim_ok=(n_features == expected_features),
        sample_label_ratio=(n_samples / n_labels) if n_labels > 0 else 0.0,
    )


def compute_data_quality(features: np.ndarray) -> DataQuality:
    """
    Compute data quality metrics for a feature array.
    
    Checks for NaN, Inf, and identifies problematic columns.
    
    Args:
        features: (N, D) feature array
    
    Returns:
        DataQuality with quality metrics
    """
    total_values = features.size
    nan_mask = np.isnan(features)
    inf_mask = np.isinf(features)
    
    nan_count = nan_mask.sum()
    inf_count = inf_mask.sum()
    finite_count = np.isfinite(features).sum()
    
    # Find columns with issues
    columns_with_nan = np.where(nan_mask.any(axis=0))[0].tolist()
    columns_with_inf = np.where(inf_mask.any(axis=0))[0].tolist()
    
    return DataQuality(
        total_values=int(total_values),
        finite_count=int(finite_count),
        nan_count=int(nan_count),
        inf_count=int(inf_count),
        pct_finite=100 * finite_count / total_values,
        pct_nan=100 * nan_count / total_values,
        pct_inf=100 * inf_count / total_values,
        columns_with_nan=columns_with_nan,
        columns_with_inf=columns_with_inf,
    )


def compute_label_distribution(labels: np.ndarray) -> LabelDistribution:
    """
    Compute label distribution statistics.
    
    Args:
        labels: Label array
    
    Returns:
        LabelDistribution with distribution metrics
    """
    total = len(labels)
    
    down_count = int((labels == LABEL_DOWN).sum())
    stable_count = int((labels == LABEL_STABLE).sum())
    up_count = int((labels == LABEL_UP).sum())
    
    down_pct = 100 * down_count / total
    stable_pct = 100 * stable_count / total
    up_pct = 100 * up_count / total
    
    counts = [down_count, stable_count, up_count]
    max_count = max(counts)
    min_count = max(min(counts), 1)  # Avoid division by zero
    imbalance_ratio = max_count / min_count
    
    return LabelDistribution(
        total=total,
        down_count=down_count,
        stable_count=stable_count,
        up_count=up_count,
        down_pct=down_pct,
        stable_pct=stable_pct,
        up_pct=up_pct,
        imbalance_ratio=imbalance_ratio,
    )


def validate_categorical_feature(
    features: np.ndarray,
    name: str,
    index: int,
    expected_values: Optional[List[float]] = None,
) -> CategoricalValidation:
    """
    Validate a single categorical feature.
    
    Args:
        features: Full feature array
        name: Feature name
        index: Feature column index
        expected_values: Optional list of expected unique values
    
    Returns:
        CategoricalValidation with validation results
    """
    col = features[:, index]
    unique_vals = sorted(np.unique(col).tolist())
    
    value_counts = {}
    for val in unique_vals:
        value_counts[float(val)] = int((col == val).sum())
    
    if expected_values is not None:
        expected_set = set(expected_values)
        actual_set = set(unique_vals)
        
        if actual_set == expected_set:
            is_valid = True
            message = "Matches expected values"
        elif actual_set.issubset(expected_set):
            missing = expected_set - actual_set
            is_valid = True
            message = f"Subset of expected (missing: {sorted(missing)})"
        else:
            unexpected = actual_set - expected_set
            is_valid = False
            message = f"Unexpected values: {sorted(unexpected)}"
    else:
        is_valid = True
        message = "No expected values specified"
    
    return CategoricalValidation(
        name=name,
        index=index,
        unique_values=unique_vals,
        expected_values=expected_values,
        value_counts=value_counts,
        is_valid=is_valid,
        message=message,
    )


def compute_all_categorical_validations(features: np.ndarray) -> List[CategoricalValidation]:
    """
    Validate all categorical features.
    
    Args:
        features: Full feature array
    
    Returns:
        List of CategoricalValidation for each categorical feature
    """
    # Define expected values for each categorical feature
    categorical_config = {
        'book_valid': {
            'index': FeatureIndex.BOOK_VALID,
            'expected': [0.0, 1.0],
        },
        'time_regime': {
            'index': FeatureIndex.TIME_REGIME,
            'expected': [0.0, 1.0, 2.0, 3.0, 4.0],
        },
        'mbo_ready': {
            'index': FeatureIndex.MBO_READY,
            'expected': [0.0, 1.0],
        },
        'invalidity_delta': {
            'index': FeatureIndex.INVALIDITY_DELTA,
            'expected': None,  # Count, any non-negative
        },
        'schema_version': {
            'index': FeatureIndex.SCHEMA_VERSION_FEATURE,
            'expected': [float(SCHEMA_VERSION)],
        },
    }
    
    validations = []
    for name, config in categorical_config.items():
        validation = validate_categorical_feature(
            features,
            name=name,
            index=config['index'],
            expected_values=config['expected'],
        )
        validations.append(validation)
    
    return validations


def compute_signal_statistics(features: np.ndarray) -> List[SignalStatistics]:
    """
    Compute statistics for all 14 signal features (indices 84-97).
    
    Args:
        features: Full feature array
    
    Returns:
        List of SignalStatistics for each signal
    """
    signal_info = {
        84: ('true_ofi', 'continuous'),
        85: ('depth_norm_ofi', 'continuous'),
        86: ('executed_pressure', 'continuous'),
        87: ('signed_mp_delta_bps', 'continuous'),
        88: ('trade_asymmetry', 'continuous'),
        89: ('cancel_asymmetry', 'continuous'),
        90: ('fragility_score', 'continuous'),
        91: ('depth_asymmetry', 'continuous'),
        92: ('book_valid', 'binary'),
        93: ('time_regime', 'categorical'),
        94: ('mbo_ready', 'binary'),
        95: ('dt_seconds', 'continuous'),
        96: ('invalidity_delta', 'count'),
        97: ('schema_version', 'constant'),
    }
    
    stats_list = []
    for idx, (name, signal_type) in signal_info.items():
        col = features[:, idx]
        
        mean = float(col.mean())
        std = float(col.std())
        min_val = float(col.min())
        max_val = float(col.max())
        median = float(np.median(col))
        q25 = float(np.percentile(col, 25))
        q75 = float(np.percentile(col, 75))
        n_unique = int(len(np.unique(col)))
        
        # Compute outlier percentage for continuous signals
        if signal_type == 'continuous' and std > 0:
            z_scores = np.abs((col - mean) / std)
            pct_outliers = float(100 * (z_scores > 4).mean())
        else:
            pct_outliers = 0.0
        
        stats_list.append(SignalStatistics(
            index=idx,
            name=name,
            signal_type=signal_type,
            mean=mean,
            std=std,
            min_val=min_val,
            max_val=max_val,
            median=median,
            q25=q25,
            q75=q75,
            n_unique=n_unique,
            pct_outliers=pct_outliers,
        ))
    
    return stats_list


@dataclass
class DatasetSummary:
    """Comprehensive summary for a dataset."""
    symbol: str
    data_dir: str
    date_range: Tuple[str, str]
    total_days: int
    train_days: int
    val_days: int
    test_days: int
    total_samples: int
    total_labels: int
    feature_count: int
    data_quality: DataQuality
    label_distribution: LabelDistribution
    categorical_validations: List[CategoricalValidation]
    signal_stats: List[SignalStatistics]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            'symbol': self.symbol,
            'data_dir': self.data_dir,
            'date_range': self.date_range,
            'total_days': self.total_days,
            'train_days': self.train_days,
            'val_days': self.val_days,
            'test_days': self.test_days,
            'total_samples': self.total_samples,
            'total_labels': self.total_labels,
            'feature_count': self.feature_count,
            'data_quality': asdict(self.data_quality),
            'label_distribution': asdict(self.label_distribution),
            'categorical_validations': [asdict(v) for v in self.categorical_validations],
            'signal_stats': [asdict(s) for s in self.signal_stats],
        }


def generate_dataset_summary(
    data_dir: Path,
    symbol: str = "UNKNOWN",
) -> DatasetSummary:
    """
    Generate comprehensive summary for a dataset.
    
    Args:
        data_dir: Path to dataset root
        symbol: Symbol name (e.g., "NVDA")
    
    Returns:
        DatasetSummary with all validation and statistics
    """
    from .data_loading import load_split
    
    data_dir = Path(data_dir)
    
    # Discover files
    inventories = validate_file_structure(data_dir)
    
    # Load all data
    all_features = []
    all_labels = []
    all_dates = []
    
    split_counts = {'train': 0, 'val': 0, 'test': 0}
    
    for split_name, inv in inventories.items():
        split_data = load_split(data_dir, split_name)
        all_features.append(split_data['features'])
        all_labels.append(split_data['labels'])
        all_dates.extend(split_data['dates'])
        split_counts[split_name] = split_data['n_days']
    
    features = np.vstack(all_features)
    labels = np.concatenate(all_labels)
    
    # Date range
    sorted_dates = sorted(all_dates)
    date_range = (sorted_dates[0], sorted_dates[-1])
    
    # Compute all validations and statistics
    data_quality = compute_data_quality(features)
    label_distribution = compute_label_distribution(labels)
    categorical_validations = compute_all_categorical_validations(features)
    signal_stats = compute_signal_statistics(features)
    
    return DatasetSummary(
        symbol=symbol,
        data_dir=str(data_dir),
        date_range=date_range,
        total_days=sum(split_counts.values()),
        train_days=split_counts['train'],
        val_days=split_counts['val'],
        test_days=split_counts['test'],
        total_samples=features.shape[0],
        total_labels=len(labels),
        feature_count=features.shape[1],
        data_quality=data_quality,
        label_distribution=label_distribution,
        categorical_validations=categorical_validations,
        signal_stats=signal_stats,
    )


def print_data_overview(summary: DatasetSummary) -> None:
    """
    Print formatted data overview to console.
    
    Args:
        summary: DatasetSummary to print
    """
    print("=" * 70)
    print("DATASET PROFILE SUMMARY")
    print("=" * 70)
    
    print(f"\nSymbol: {summary.symbol}")
    print(f"Data directory: {summary.data_dir}")
    print(f"Date range: {summary.date_range[0]} to {summary.date_range[1]}")
    print(f"Total trading days: {summary.total_days}")
    
    print(f"\n--- Sample Counts ---")
    print(f"Total feature samples: {summary.total_samples:,}")
    print(f"Total labels: {summary.total_labels:,}")
    print(f"Feature count: {summary.feature_count}")
    
    print(f"\n--- Split Distribution ---")
    print(f"Train: {summary.train_days} days")
    print(f"Val:   {summary.val_days} days")
    print(f"Test:  {summary.test_days} days")
    
    print(f"\n--- Label Distribution ---")
    ld = summary.label_distribution
    print(f"Down:   {ld.down_count:6,} ({ld.down_pct:5.2f}%)")
    print(f"Stable: {ld.stable_count:6,} ({ld.stable_pct:5.2f}%)")
    print(f"Up:     {ld.up_count:6,} ({ld.up_pct:5.2f}%)")
    print(f"Imbalance ratio: {ld.imbalance_ratio:.2f}")
    status = "✅ Balanced" if ld.is_balanced else "⚠️ Imbalanced"
    print(f"Status: {status}")
    
    print(f"\n--- Data Quality ---")
    dq = summary.data_quality
    print(f"Total values: {dq.total_values:,}")
    print(f"NaN values:   {dq.nan_count} ({dq.pct_nan:.6f}%)")
    print(f"Inf values:   {dq.inf_count} ({dq.pct_inf:.6f}%)")
    status = "✅ All values finite" if dq.is_clean else "❌ Non-finite values detected"
    print(f"Status: {status}")
    
    if dq.columns_with_nan:
        print(f"Columns with NaN: {dq.columns_with_nan}")
    if dq.columns_with_inf:
        print(f"Columns with Inf: {dq.columns_with_inf}")
    
    print(f"\n--- Categorical Feature Validation ---")
    for cv in summary.categorical_validations:
        status = "✅" if cv.is_valid else "❌"
        print(f"{status} {cv.name} (idx {cv.index}): {cv.message}")
        print(f"   Values: {cv.unique_values}")
    
    print(f"\n--- Signal Statistics ---")
    print(f"{'Index':<6} {'Name':<22} {'Type':<12} {'Mean':>10} {'Std':>10} {'Outliers':>10}")
    print("-" * 72)
    for s in summary.signal_stats:
        print(f"{s.index:<6} {s.name:<22} {s.signal_type:<12} {s.mean:>10.4f} {s.std:>10.4f} {s.pct_outliers:>9.2f}%")
    
    print("\n" + "=" * 70)
    
    # Overall status
    all_valid = all(cv.is_valid for cv in summary.categorical_validations)
    if dq.is_clean and all_valid:
        print("✅ DATA OVERVIEW COMPLETE - Ready for analysis")
    else:
        print("⚠️ DATA OVERVIEW COMPLETE - Issues detected, review above")
    
    print("=" * 70)

