"""
Memory-efficient streaming data utilities.

This module provides memory-efficient alternatives to bulk data loading:
- Stream files one at a time (never load all into memory)
- Use memory mapping for read-only access
- Compute statistics incrementally (online algorithms)
- Support float32 for 50% memory reduction

Design Principles:
1. Process data in chunks/days, never all at once
2. Use generators/iterators instead of lists
3. Compute statistics incrementally (Welford's algorithm, etc.)
4. Explicit memory management with gc.collect()

Memory Budget:
- Target: < 4GB for any dataset size
- One day of data: ~100K samples × 98 features × 4 bytes = ~40MB (float32)
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterator, Generator, Any, Callable
from dataclasses import dataclass, field
import gc
import warnings


# ============================================================================
# STREAMING DATA ITERATORS
# ============================================================================

@dataclass
class DayData:
    """
    Container for a single day's data (minimal memory footprint).
    
    Supports both single-horizon and multi-horizon labels:
    - Single-horizon: labels shape is (M,)
    - Multi-horizon: labels shape is (M, n_horizons)
    """
    date: str
    features: np.ndarray  # (N, 98) - loaded on demand
    labels: np.ndarray    # (M,) or (M, n_horizons) - loaded on demand
    n_samples: int
    n_labels: int
    is_multi_horizon: bool = False  # True if labels.ndim == 2
    num_horizons: int = 1           # Number of prediction horizons
    
    @property
    def memory_bytes(self) -> int:
        """Approximate memory usage in bytes."""
        return self.features.nbytes + self.labels.nbytes
    
    def get_labels(self, horizon_idx: Optional[int] = 0) -> np.ndarray:
        """
        Get labels for a specific horizon.
        
        Args:
            horizon_idx: Which horizon to return (0-based). 
                         None returns all labels (full array).
                         Default: 0 (first/only horizon)
        
        Returns:
            Label array: (M,) for single horizon, (M, n_horizons) if horizon_idx=None
        """
        if horizon_idx is None:
            return self.labels
        
        if not self.is_multi_horizon:
            if horizon_idx != 0:
                raise ValueError(f"Single-horizon data only has horizon_idx=0, got {horizon_idx}")
            return self.labels
        
        if horizon_idx < 0 or horizon_idx >= self.num_horizons:
            raise ValueError(f"horizon_idx {horizon_idx} out of range [0, {self.num_horizons})")
        
        return self.labels[:, horizon_idx]


def _detect_format(split_dir: Path) -> str:
    """Detect export format: 'aligned' (*_sequences.npy) or 'legacy' (*_features.npy)."""
    if list(split_dir.glob('*_sequences.npy')):
        return 'aligned'
    elif list(split_dir.glob('*_features.npy')):
        return 'legacy'
    else:
        raise ValueError(f"No data files found in {split_dir}")


def iter_days(
    data_dir: Path,
    split: str,
    dtype: np.dtype = np.float32,
    mmap_mode: Optional[str] = None,
) -> Generator[DayData, None, None]:
    """
    Iterate over days in a split, yielding one day at a time.
    
    Automatically handles both export formats:
    - NEW aligned: *_sequences.npy [N_seq, 100, 98] - extracts last timestep
    - LEGACY: *_features.npy [N_samples, 98]
    
    This is the primary memory-efficient data access pattern.
    Each day is loaded, processed, then freed before the next.
    
    Args:
        data_dir: Path to dataset root
        split: One of 'train', 'val', 'test'
        dtype: Data type for features (default: float32 for memory efficiency)
        mmap_mode: If 'r', use memory-mapped files (read-only, even more efficient)
    
    Yields:
        DayData for each day in chronological order
    
    Example:
        for day in iter_days(data_dir, 'train'):
            # Use get_labels(0) for first horizon (works with both single/multi-horizon)
            process(day.features, day.get_labels(0))
            # Memory freed after each iteration
    """
    split_dir = Path(data_dir) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    
    export_format = _detect_format(split_dir)
    
    if export_format == 'aligned':
        # NEW format: *_sequences.npy
        data_files = sorted(split_dir.glob('*_sequences.npy'))
        suffix = '_sequences'
    else:
        # LEGACY format: *_features.npy
        data_files = sorted(split_dir.glob('*_features.npy'))
        suffix = '_features'
    
    for data_file in data_files:
        date = data_file.stem.replace(suffix, '')
        label_file = data_file.parent / f"{date}_labels.npy"
        
        if not label_file.exists():
            warnings.warn(f"Label file not found for {date}, skipping")
            continue
        
        # Load with specified dtype and mmap mode
        if mmap_mode:
            raw_features = np.load(data_file, mmap_mode=mmap_mode)
            labels = np.load(label_file, mmap_mode=mmap_mode)
        else:
            raw_features = np.load(data_file)
            labels = np.load(label_file)
        
        # Handle 3D sequences: extract last timestep
        if len(raw_features.shape) == 3:
            # [N_seq, window_size, n_features] -> [N_seq, n_features]
            features = raw_features[:, -1, :].astype(dtype, copy=False)
        else:
            features = raw_features.astype(dtype, copy=False)
        
        # Detect multi-horizon labels
        is_multi_horizon = labels.ndim == 2
        num_horizons = labels.shape[1] if is_multi_horizon else 1
        
        yield DayData(
            date=date,
            features=features,
            labels=labels,
            n_samples=features.shape[0],
            n_labels=labels.shape[0],
            is_multi_horizon=is_multi_horizon,
            num_horizons=num_horizons,
        )
        
        # Explicit cleanup (important for memory)
        if not mmap_mode:
            del features, labels, raw_features
            gc.collect()


def count_days(data_dir: Path, split: str) -> int:
    """Count days in a split without loading data."""
    split_dir = Path(data_dir) / split
    if not split_dir.exists():
        return 0
    # Try new format first, then legacy
    seq_files = list(split_dir.glob('*_sequences.npy'))
    if seq_files:
        return len(seq_files)
    return len(list(split_dir.glob('*_features.npy')))


def get_dates(data_dir: Path, split: str) -> List[str]:
    """Get sorted list of dates in a split without loading data."""
    split_dir = Path(data_dir) / split
    if not split_dir.exists():
        return []
    # Try new format first
    seq_files = sorted(split_dir.glob('*_sequences.npy'))
    if seq_files:
        return [f.stem.replace('_sequences', '') for f in seq_files]
    # Fall back to legacy
    feat_files = sorted(split_dir.glob('*_features.npy'))
    return [f.stem.replace('_features', '') for f in feat_files]


# ============================================================================
# ALIGNED DATA ITERATORS
# ============================================================================

# Import alignment constants (avoid circular import by importing at module level)
WINDOW_SIZE = 100  # Must match data_loading.py
STRIDE = 10        # Must match data_loading.py


@dataclass
class AlignedDayData:
    """
    Container for a single day's ALIGNED data.
    
    Features are already aligned with labels (1:1 correspondence).
    This is the correct data structure for signal-label correlation analysis.
    
    Supports multi-horizon labels:
    - Single-horizon: labels shape is (N_labels,)
    - Multi-horizon: labels shape is (N_labels, n_horizons)
    """
    date: str
    features: np.ndarray  # (N_labels, 98) - aligned with labels
    labels: np.ndarray    # (N_labels,) or (N_labels, n_horizons)
    n_pairs: int          # Number of aligned feature-label pairs
    is_multi_horizon: bool = False  # True if labels.ndim == 2
    num_horizons: int = 1           # Number of prediction horizons
    
    @property
    def memory_bytes(self) -> int:
        """Approximate memory usage in bytes."""
        return self.features.nbytes + self.labels.nbytes
    
    def get_labels(self, horizon_idx: Optional[int] = 0) -> np.ndarray:
        """
        Get labels for a specific horizon.
        
        Args:
            horizon_idx: Which horizon to return (0-based). 
                         None returns all labels (full array).
                         Default: 0 (first/only horizon)
        
        Returns:
            Label array: (N,) for single horizon, (N, n_horizons) if horizon_idx=None
        """
        if horizon_idx is None:
            return self.labels
        
        if not self.is_multi_horizon:
            if horizon_idx != 0:
                raise ValueError(f"Single-horizon data only has horizon_idx=0, got {horizon_idx}")
            return self.labels
        
        if horizon_idx < 0 or horizon_idx >= self.num_horizons:
            raise ValueError(f"horizon_idx {horizon_idx} out of range [0, {self.num_horizons})")
        
        return self.labels[:, horizon_idx]


def align_features_for_day(
    features: np.ndarray,
    n_labels: int,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
) -> np.ndarray:
    """
    Align features with labels for a SINGLE day.
    
    This is the streaming-module equivalent of data_loading.align_features_with_labels(),
    kept here to avoid circular imports and for self-containment.
    
    Formula:
        For label[i], the corresponding feature is at:
        feat_idx = i * stride + window_size - 1
        
        This is the LAST feature in the sequence window for that label.
    
    Args:
        features: (N_samples, N_features) array from a single day
        n_labels: Number of labels for this day
        window_size: Samples per sequence window
        stride: Samples between sequence starts
    
    Returns:
        aligned_features: (n_labels, N_features) array
    """
    n_features = features.shape[1]
    aligned = np.zeros((n_labels, n_features), dtype=features.dtype)
    
    for i in range(n_labels):
        feat_idx = i * stride + window_size - 1
        if feat_idx >= features.shape[0]:
            feat_idx = features.shape[0] - 1
        aligned[i] = features[feat_idx]
    
    return aligned


def iter_days_aligned(
    data_dir: Path,
    split: str,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
    dtype: np.dtype = np.float32,
) -> Generator[AlignedDayData, None, None]:
    """
    Iterate over days, yielding ALIGNED feature-label pairs.
    
    This is the memory-efficient equivalent of load_split_aligned() from data_loading.
    Use this for any analysis that needs signal-label correlation.
    
    CRITICAL: This function performs correct per-day alignment, avoiding the
    day-boundary drift that occurs with global alignment on concatenated data.
    
    Args:
        data_dir: Path to dataset root
        split: One of 'train', 'val', 'test'
        window_size: Samples per sequence window (default: 100)
        stride: Samples between sequence starts (default: 10)
        dtype: Data type for features (default: float32)
    
    Yields:
        AlignedDayData for each day in chronological order
    
    Example:
        for day in iter_days_aligned(data_dir, 'train'):
            # day.features[i] corresponds to day.labels[i] (single-horizon)
            # or day.labels[i, :] for all horizons (multi-horizon)
            labels = day.labels if day.labels.ndim == 1 else day.labels[:, 0]
            corr = np.corrcoef(day.features[:, 84], labels)[0, 1]
            
    Memory Usage:
        ~40MB per day (float32) - same as iter_days() but with aligned features
    """
    split_dir = Path(data_dir) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    
    export_format = _detect_format(split_dir)
    
    if export_format == 'aligned':
        # NEW format: Already aligned! Just extract last timestep
        data_files = sorted(split_dir.glob('*_sequences.npy'))
        suffix = '_sequences'
    else:
        # LEGACY format: Need to apply alignment
        data_files = sorted(split_dir.glob('*_features.npy'))
        suffix = '_features'
    
    for data_file in data_files:
        date = data_file.stem.replace(suffix, '')
        label_file = data_file.parent / f"{date}_labels.npy"
        
        if not label_file.exists():
            warnings.warn(f"Label file not found for {date}, skipping")
            continue
        
        # Load raw data
        raw_data = np.load(data_file)
        labels = np.load(label_file)
        n_labels = labels.shape[0]
        
        # Detect multi-horizon labels
        is_multi_horizon = labels.ndim == 2
        num_horizons = labels.shape[1] if is_multi_horizon else 1
        
        if export_format == 'aligned':
            # NEW format: 3D sequences [N_seq, 100, 98] - extract last timestep
            if len(raw_data.shape) == 3:
                aligned_features = raw_data[:, -1, :].astype(dtype, copy=False)
            else:
                aligned_features = raw_data.astype(dtype, copy=False)
        else:
            # LEGACY format: Need to align
            features = raw_data.astype(dtype, copy=False)
            aligned_features = align_features_for_day(
                features, n_labels, window_size, stride
            )
            del features
        
        yield AlignedDayData(
            date=date,
            features=aligned_features,
            labels=labels,
            n_pairs=n_labels,
            is_multi_horizon=is_multi_horizon,
            num_horizons=num_horizons,
        )
        
        # Explicit cleanup
        del raw_data, aligned_features, labels
        gc.collect()


# ============================================================================
# INCREMENTAL STATISTICS (Online Algorithms)
# ============================================================================

@dataclass
class RunningStats:
    """
    Welford's online algorithm for computing mean and variance.
    
    Numerically stable, single-pass, constant memory.
    
    Reference: Welford, B. P. (1962). "Note on a method for calculating 
               corrected sums of squares and products"
    """
    n: int = 0
    mean: float = 0.0
    M2: float = 0.0  # Sum of squared deviations
    min_val: float = float('inf')
    max_val: float = float('-inf')
    
    def update(self, x: float) -> None:
        """Update with a single value."""
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
        self.min_val = min(self.min_val, x)
        self.max_val = max(self.max_val, x)
    
    def update_batch(self, values: np.ndarray) -> None:
        """Update with a batch of values (more efficient)."""
        for x in values.flat:
            self.update(x)
    
    @property
    def variance(self) -> float:
        """Population variance."""
        return self.M2 / self.n if self.n > 0 else 0.0
    
    @property
    def std(self) -> float:
        """Population standard deviation."""
        return np.sqrt(self.variance)
    
    @classmethod
    def merge(cls, a: 'RunningStats', b: 'RunningStats') -> 'RunningStats':
        """Merge two RunningStats (for parallel computation)."""
        if a.n == 0:
            return b
        if b.n == 0:
            return a
        
        combined = cls()
        combined.n = a.n + b.n
        delta = b.mean - a.mean
        combined.mean = (a.n * a.mean + b.n * b.mean) / combined.n
        combined.M2 = a.M2 + b.M2 + delta * delta * a.n * b.n / combined.n
        combined.min_val = min(a.min_val, b.min_val)
        combined.max_val = max(a.max_val, b.max_val)
        return combined


@dataclass
class StreamingColumnStats:
    """
    Streaming statistics for multiple columns.
    
    Memory: O(n_columns) - constant regardless of data size.
    """
    n_columns: int
    stats: List[RunningStats] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.stats:
            self.stats = [RunningStats() for _ in range(self.n_columns)]
    
    def update(self, row: np.ndarray) -> None:
        """Update with a single row."""
        for i, val in enumerate(row):
            if np.isfinite(val):
                self.stats[i].update(float(val))
    
    def update_batch(self, data: np.ndarray) -> None:
        """Update with a batch of rows (2D array)."""
        for col_idx in range(min(data.shape[1], self.n_columns)):
            col = data[:, col_idx]
            finite_mask = np.isfinite(col)
            if finite_mask.any():
                for val in col[finite_mask]:
                    self.stats[col_idx].update(float(val))
    
    def get_summary(self) -> Dict[int, Dict[str, float]]:
        """Get summary for all columns."""
        return {
            i: {
                'n': s.n,
                'mean': s.mean,
                'std': s.std,
                'min': s.min_val,
                'max': s.max_val,
            }
            for i, s in enumerate(self.stats)
        }


@dataclass
class StreamingLabelCounter:
    """
    Streaming label distribution counter.
    
    Memory: O(1) - constant regardless of data size.
    """
    down_count: int = 0
    stable_count: int = 0
    up_count: int = 0
    total: int = 0
    
    def update(self, labels: np.ndarray) -> None:
        """Update with a batch of labels."""
        self.down_count += int((labels == -1).sum())
        self.stable_count += int((labels == 0).sum())
        self.up_count += int((labels == 1).sum())
        self.total += len(labels)
    
    @property
    def down_pct(self) -> float:
        return 100 * self.down_count / self.total if self.total > 0 else 0
    
    @property
    def stable_pct(self) -> float:
        return 100 * self.stable_count / self.total if self.total > 0 else 0
    
    @property
    def up_pct(self) -> float:
        return 100 * self.up_count / self.total if self.total > 0 else 0


@dataclass
class StreamingDataQuality:
    """
    Streaming data quality checker.
    
    Memory: O(n_columns) - for tracking which columns have issues.
    """
    total_values: int = 0
    nan_count: int = 0
    inf_count: int = 0
    columns_with_nan: set = field(default_factory=set)
    columns_with_inf: set = field(default_factory=set)
    
    def update(self, features: np.ndarray) -> None:
        """Update with a batch of features."""
        self.total_values += features.size
        
        nan_mask = np.isnan(features)
        inf_mask = np.isinf(features)
        
        self.nan_count += int(nan_mask.sum())
        self.inf_count += int(inf_mask.sum())
        
        # Track columns with issues
        nan_cols = np.where(nan_mask.any(axis=0))[0]
        inf_cols = np.where(inf_mask.any(axis=0))[0]
        
        self.columns_with_nan.update(nan_cols.tolist())
        self.columns_with_inf.update(inf_cols.tolist())
    
    @property
    def is_clean(self) -> bool:
        return self.nan_count == 0 and self.inf_count == 0
    
    @property
    def finite_count(self) -> int:
        return self.total_values - self.nan_count - self.inf_count


# ============================================================================
# STREAMING ANALYSIS FUNCTIONS
# ============================================================================

def compute_streaming_overview(
    data_dir: Path,
    symbol: str = "UNKNOWN",
    dtype: np.dtype = np.float32,
) -> Dict[str, Any]:
    """
    Compute dataset overview with streaming (memory-efficient).
    
    Memory usage: O(n_features) - constant regardless of dataset size.
    
    Args:
        data_dir: Path to dataset root
        symbol: Symbol name
        dtype: Data type for loading
    
    Returns:
        Dict with overview statistics
    """
    data_dir = Path(data_dir)
    
    # Initialize streaming counters
    n_features = 98  # Known from schema
    column_stats = StreamingColumnStats(n_columns=n_features)
    label_counter = StreamingLabelCounter()
    data_quality = StreamingDataQuality()
    
    all_dates = []
    split_counts = {'train': 0, 'val': 0, 'test': 0}
    total_samples = 0
    
    # Process each split
    for split in ['train', 'val', 'test']:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        
        day_count = 0
        for day in iter_days(data_dir, split, dtype=dtype):
            # Update streaming statistics
            column_stats.update_batch(day.features)
            # Use first horizon for multi-horizon data (backward compatible)
            label_counter.update(day.get_labels(0))
            data_quality.update(day.features)
            
            all_dates.append(day.date)
            total_samples += day.n_samples
            day_count += 1
            
            # Memory freed automatically after each day
        
        split_counts[split] = day_count
    
    # Build result
    sorted_dates = sorted(all_dates)
    
    return {
        'symbol': symbol,
        'data_dir': str(data_dir),
        'date_range': (sorted_dates[0], sorted_dates[-1]) if sorted_dates else (None, None),
        'total_days': len(all_dates),
        'train_days': split_counts['train'],
        'val_days': split_counts['val'],
        'test_days': split_counts['test'],
        'total_samples': total_samples,
        'total_labels': label_counter.total,
        'feature_count': n_features,
        'data_quality': {
            'total_values': data_quality.total_values,
            'finite_count': data_quality.finite_count,
            'nan_count': data_quality.nan_count,
            'inf_count': data_quality.inf_count,
            'is_clean': data_quality.is_clean,
            'columns_with_nan': sorted(data_quality.columns_with_nan),
            'columns_with_inf': sorted(data_quality.columns_with_inf),
        },
        'label_distribution': {
            'total': label_counter.total,
            'down_count': label_counter.down_count,
            'stable_count': label_counter.stable_count,
            'up_count': label_counter.up_count,
            'down_pct': label_counter.down_pct,
            'stable_pct': label_counter.stable_pct,
            'up_pct': label_counter.up_pct,
        },
        'signal_stats': column_stats.get_summary(),
    }


def compute_streaming_label_analysis(
    data_dir: Path,
    split: str = 'train',
    dtype: np.dtype = np.float32,
    max_samples_for_acf: int = 100000,
) -> Dict[str, Any]:
    """
    Compute label analysis with streaming.
    
    Some analyses (like autocorrelation) require all labels in memory,
    but labels are small (1 byte per sample typically).
    
    Args:
        data_dir: Path to dataset root
        split: Which split to analyze
        dtype: Data type for loading
        max_samples_for_acf: Maximum samples to use for ACF computation
    
    Returns:
        Dict with label analysis results
    """
    data_dir = Path(data_dir)
    
    # Labels are small enough to collect
    all_labels = []
    dates = []
    
    # Streaming counters for per-day stats
    day_stats = []
    transition_counts = np.zeros((3, 3), dtype=np.int64)
    label_map = {-1: 0, 0: 1, 1: 2}
    
    for day in iter_days(data_dir, split, dtype=dtype):
        # Use first horizon for multi-horizon data (backward compatible)
        day_labels = day.get_labels(0)
        all_labels.append(day_labels)
        dates.append(day.date)
        
        # Per-day distribution
        day_stats.append({
            'date': day.date,
            'n_labels': len(day_labels),
            'up_pct': float(100 * (day_labels == 1).mean()),
            'down_pct': float(100 * (day_labels == -1).mean()),
            'stable_pct': float(100 * (day_labels == 0).mean()),
        })
        
        # Update transition counts
        for i in range(len(day_labels) - 1):
            from_idx = label_map.get(day_labels[i], 1)
            to_idx = label_map.get(day_labels[i + 1], 1)
            transition_counts[from_idx, to_idx] += 1
    
    # Concatenate labels (small memory footprint)
    labels = np.concatenate(all_labels)
    del all_labels
    gc.collect()
    
    # Compute distribution
    total = len(labels)
    distribution = {
        'total': total,
        'down_count': int((labels == -1).sum()),
        'stable_count': int((labels == 0).sum()),
        'up_count': int((labels == 1).sum()),
    }
    distribution['down_pct'] = 100 * distribution['down_count'] / total
    distribution['stable_pct'] = 100 * distribution['stable_count'] / total
    distribution['up_pct'] = 100 * distribution['up_count'] / total
    
    # Autocorrelation (subsample if needed)
    if len(labels) > max_samples_for_acf:
        step = len(labels) // max_samples_for_acf
        labels_for_acf = labels[::step][:max_samples_for_acf]
    else:
        labels_for_acf = labels
    
    acf = _compute_acf(labels_for_acf, max_lag=100)
    
    # Transition probabilities
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    transition_probs = np.divide(
        transition_counts, row_sums, 
        where=row_sums > 0, 
        out=np.zeros_like(transition_counts, dtype=float)
    )
    
    return {
        'split': split,
        'date_range': (dates[0], dates[-1]) if dates else (None, None),
        'n_days': len(dates),
        'distribution': distribution,
        'autocorrelation': {
            'lag_1': float(acf[1]) if len(acf) > 1 else 0.0,
            'lag_5': float(acf[5]) if len(acf) > 5 else 0.0,
            'lag_10': float(acf[10]) if len(acf) > 10 else 0.0,
            'acf_values': acf[:20].tolist(),  # First 20 lags
        },
        'transition_matrix': {
            'labels': [-1, 0, 1],
            'probabilities': transition_probs.tolist(),
        },
        'day_stats': day_stats,
    }


def _compute_acf(labels: np.ndarray, max_lag: int = 100) -> np.ndarray:
    """Compute autocorrelation function efficiently."""
    n = len(labels)
    labels_float = labels.astype(np.float32)
    mean = labels_float.mean()
    var = labels_float.var()
    
    if var == 0:
        return np.ones(min(max_lag + 1, n))
    
    acf = np.zeros(min(max_lag + 1, n))
    acf[0] = 1.0
    
    # Use numpy vectorization for efficiency
    for lag in range(1, len(acf)):
        cov = np.mean((labels_float[:-lag] - mean) * (labels_float[lag:] - mean))
        acf[lag] = cov / var
    
    return acf


def compute_streaming_signal_stats(
    data_dir: Path,
    split: str = 'train',
    signal_indices: Optional[List[int]] = None,
    dtype: np.dtype = np.float32,
) -> Dict[str, Any]:
    """
    Compute signal statistics with streaming.
    
    Memory: O(n_signals) - constant regardless of dataset size.
    
    Args:
        data_dir: Path to dataset root
        split: Which split to analyze
        signal_indices: Which signals to analyze (default: 84-91)
        dtype: Data type for loading
    
    Returns:
        Dict with signal statistics
    """
    if signal_indices is None:
        signal_indices = list(range(84, 92))  # Core signals
    
    signal_names = {
        84: 'true_ofi',
        85: 'depth_norm_ofi',
        86: 'executed_pressure',
        87: 'signed_mp_delta_bps',
        88: 'trade_asymmetry',
        89: 'cancel_asymmetry',
        90: 'fragility_score',
        91: 'depth_asymmetry',
    }
    
    # Initialize streaming stats for each signal
    signal_stats = {idx: RunningStats() for idx in signal_indices}
    
    # Process each day
    for day in iter_days(data_dir, split, dtype=dtype):
        for idx in signal_indices:
            col = day.features[:, idx]
            finite_mask = np.isfinite(col)
            if finite_mask.any():
                for val in col[finite_mask]:
                    signal_stats[idx].update(float(val))
    
    # Build results
    results = {}
    for idx, stats in signal_stats.items():
        name = signal_names.get(idx, f'signal_{idx}')
        results[name] = {
            'index': idx,
            'n': stats.n,
            'mean': stats.mean,
            'std': stats.std,
            'min': stats.min_val if stats.n > 0 else None,
            'max': stats.max_val if stats.n > 0 else None,
        }
    
    return results


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def estimate_memory_usage(
    data_dir: Path,
    dtype: np.dtype = np.float32,
) -> Dict[str, Any]:
    """
    Estimate memory usage without loading data.
    
    Handles both export formats:
    - NEW aligned: *_sequences.npy [N_seq, 100, 98]
    - LEGACY: *_features.npy [N_samples, 98]
    
    Args:
        data_dir: Path to dataset root
        dtype: Data type that would be used
    
    Returns:
        Dict with memory estimates
    """
    bytes_per_element = np.dtype(dtype).itemsize
    n_features = 98
    
    estimates = {}
    total_samples = 0
    
    for split in ['train', 'val', 'test']:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        
        # Count samples from file sizes - try new format first
        split_samples = 0
        seq_files = list(split_dir.glob('*_sequences.npy'))
        
        if seq_files:
            # NEW format: shape is [N_seq, window_size, n_features]
            for seq_file in seq_files:
                with open(seq_file, 'rb') as f:
                    np.lib.format.read_magic(f)
                    shape, _, _ = np.lib.format.read_array_header_1_0(f)
                    split_samples += shape[0]  # N_seq (number of sequences)
        else:
            # LEGACY format: shape is [N_samples, n_features]
            for feat_file in split_dir.glob('*_features.npy'):
                with open(feat_file, 'rb') as f:
                    np.lib.format.read_magic(f)
                    shape, _, _ = np.lib.format.read_array_header_1_0(f)
                    split_samples += shape[0]
        
        bytes_needed = split_samples * n_features * bytes_per_element
        estimates[split] = {
            'samples': split_samples,
            'bytes': bytes_needed,
            'mb': bytes_needed / (1024 * 1024),
            'gb': bytes_needed / (1024 * 1024 * 1024),
        }
        total_samples += split_samples
    
    total_bytes = total_samples * n_features * bytes_per_element
    estimates['total'] = {
        'samples': total_samples,
        'bytes': total_bytes,
        'mb': total_bytes / (1024 * 1024),
        'gb': total_bytes / (1024 * 1024 * 1024),
    }
    
    return estimates


def get_memory_efficient_config() -> Dict[str, Any]:
    """
    Get recommended configuration for memory-efficient processing.
    
    Returns:
        Dict with configuration recommendations
    """
    return {
        'dtype': 'float32',  # vs float64
        'mmap_mode': 'r',    # Memory-mapped read
        'max_days_in_memory': 1,  # Process one day at a time
        'gc_after_each_day': True,
        'subsample_for_expensive_ops': True,
        'max_samples_for_acf': 100000,
        'max_samples_for_correlation': 500000,
    }

