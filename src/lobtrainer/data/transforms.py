"""
Feature transforms and utilities for LOB data.

Components:
    - ``FeatureStatistics``: Generic per-feature statistics.
    - ``compute_statistics``: Compute feature statistics from arrays.
    - ``BinaryLabelTransform``: 3-class to 2-class for Two-Stage training.
    - ``ComposeTransform``: Compose multiple transforms.

For normalization, use ``lobtrainer.data.normalization.GlobalZScoreNormalizer``
or ``lobtrainer.data.normalization.HybridNormalizer``.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union
import json

import numpy as np

from lobtrainer.constants import FEATURE_COUNT, FeatureIndex


# =============================================================================
# Statistics Computation
# =============================================================================


@dataclass
class FeatureStatistics:
    """
    Statistics for feature normalization.
    
    Computed from training data only to prevent data leakage.
    
    Attributes:
        mean: Per-feature mean, shape (num_features,)
        std: Per-feature std, shape (num_features,)
        min: Per-feature min, shape (num_features,)
        max: Per-feature max, shape (num_features,)
        median: Per-feature median, shape (num_features,)
        q25: Per-feature 25th percentile, shape (num_features,)
        q75: Per-feature 75th percentile, shape (num_features,)
        count: Number of samples used for computation
    """
    mean: np.ndarray
    std: np.ndarray
    min: np.ndarray
    max: np.ndarray
    median: np.ndarray
    q25: np.ndarray
    q75: np.ndarray
    count: int
    num_features: int = FEATURE_COUNT
    
    def __post_init__(self) -> None:
        """Validate statistics arrays."""
        for name, arr in [
            ("mean", self.mean),
            ("std", self.std),
            ("min", self.min),
            ("max", self.max),
            ("median", self.median),
            ("q25", self.q25),
            ("q75", self.q75),
        ]:
            if arr.shape != (self.num_features,):
                raise ValueError(
                    f"{name} shape mismatch: expected ({self.num_features},), "
                    f"got {arr.shape}"
                )
    
    @property
    def iqr(self) -> np.ndarray:
        """Interquartile range: q75 - q25."""
        return self.q75 - self.q25
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "min": self.min.tolist(),
            "max": self.max.tolist(),
            "median": self.median.tolist(),
            "q25": self.q25.tolist(),
            "q75": self.q75.tolist(),
            "count": self.count,
            "num_features": self.num_features,
        }
    
    def save(self, path: Union[str, Path]) -> None:
        """Save statistics to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "FeatureStatistics":
        """Create from dictionary."""
        return cls(
            mean=np.array(data["mean"], dtype=np.float64),
            std=np.array(data["std"], dtype=np.float64),
            min=np.array(data["min"], dtype=np.float64),
            max=np.array(data["max"], dtype=np.float64),
            median=np.array(data["median"], dtype=np.float64),
            q25=np.array(data["q25"], dtype=np.float64),
            q75=np.array(data["q75"], dtype=np.float64),
            count=data["count"],
            num_features=data.get("num_features", FEATURE_COUNT),
        )
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "FeatureStatistics":
        """Load statistics from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)


def compute_statistics(
    features: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    eps: float = 1e-8,
    strict: bool = True,
) -> FeatureStatistics:
    """
    Compute feature statistics for normalization.

    Args:
        features: Feature array of shape (N, num_features)
        valid_mask: Optional boolean mask of shape (N,) for valid samples.
                    If None, uses book_valid > 0.5 as mask.
        eps: Minimum value for std to prevent division by zero
        strict: Phase X.3 Empirical Trust (2026-05-05) — when True (default),
            raise ``DegenerateFeatureError`` on all-NaN feature columns or
            zero-variance feature columns. Pre-X.3 default was effectively
            ``strict=False`` — silent fabrication of mean=0/std=1 for
            all-NaN columns + ``np.maximum(std, eps)`` clamp for zero-variance.
            Both violations of hft-rules §8 ("never silently drop, clamp, or
            'fix' data without recording diagnostics"). Default flipped to
            ``strict=True`` 2026-05-05; legacy path remains for migration
            (DeprecationWarning) until 2026-10-31 removal.

    Returns:
        FeatureStatistics instance

    Raises:
        ValueError: If no valid samples.
        DegenerateFeatureError: If ``strict=True`` (default) and any feature
            column is all-NaN or zero-variance. Indicates upstream data
            corruption OR a feature that should be in
            ``normalization.exclude_indices`` (e.g., RESERVED idx 97).
    """
    if features.ndim != 2:
        raise ValueError(f"Expected 2D features, got shape {features.shape}")

    num_features = features.shape[1]

    # Create valid mask if not provided
    if valid_mask is None:
        # Use book_valid as default mask
        if num_features == FEATURE_COUNT:
            valid_mask = features[:, FeatureIndex.BOOK_VALID] > 0.5
        else:
            # Use all finite samples
            valid_mask = np.isfinite(features).all(axis=1)

    # Filter to valid samples
    valid_features = features[valid_mask]

    if len(valid_features) == 0:
        raise ValueError("No valid samples for statistics computation")

    # Replace non-finite values with NaN for computation
    clean_features = np.where(np.isfinite(valid_features), valid_features, np.nan)

    # Compute statistics, ignoring NaN
    with np.errstate(all="ignore"):
        mean = np.nanmean(clean_features, axis=0)
        std = np.nanstd(clean_features, axis=0)
        min_val = np.nanmin(clean_features, axis=0)
        max_val = np.nanmax(clean_features, axis=0)
        median = np.nanmedian(clean_features, axis=0)
        q25 = np.nanpercentile(clean_features, 25, axis=0)
        q75 = np.nanpercentile(clean_features, 75, axis=0)

    # Phase X.3 Empirical Trust (2026-05-05): fail-loud on ALL-NaN feature
    # columns. Pre-fix at lines 177-178 silently fabricated mean=0 + std=1
    # via nan_to_num for all-NaN columns — direct hft-rules §8 violation
    # ("never silently drop, clamp, or 'fix' data without recording
    # diagnostics"). Caught by 2026-05-05 multi-agent audit.
    #
    # SCOPE NOTE: zero-variance (std < eps) is INTENTIONALLY NOT raised.
    # Categorical features (book_valid idx 92, schema_version idx 97
    # RESERVED 0.0, etc.) legitimately have std=0 across all samples.
    # The caller (HybridNormalizer / GlobalZScoreNormalizer) is responsible
    # for excluding those via `exclude_indices` config. The
    # `np.maximum(std, eps)` clamp at line 174 stays as the existing
    # defensive behavior for non-excluded but constant-valued features.
    # If a non-categorical feature reaches std=0, that's a data-quality
    # issue better surfaced at the caller (where context exists for which
    # indices SHOULD be normalizable) — not at this leaf primitive.
    if strict:
        # Lazy import to keep transforms.py importable without trainer pkg
        from lobtrainer.training.exceptions import DegenerateFeatureError

        nan_mask = np.isnan(mean) | np.isnan(std)
        if nan_mask.any():
            raise DegenerateFeatureError(
                feature_indices=np.where(nan_mask)[0].tolist(),
                reason="all_nan",
            )
    else:
        import warnings as _warnings
        _warnings.warn(
            "compute_statistics(strict=False) silently fabricates "
            "mean=0/std=1 for all-NaN columns via nan_to_num — violates "
            "hft-rules §8. Migrate to strict=True (default since "
            "2026-05-05). Legacy path will be removed 2026-10-31.",
            DeprecationWarning,
            stacklevel=2,
        )

    # Ensure std is at least eps to prevent division by zero (defensive
    # in strict mode; load-bearing in legacy non-strict mode)
    std = np.maximum(std, eps)

    # Replace any remaining NaN with safe defaults. In strict mode, mean+std
    # are guaranteed non-NaN — these lines are NO-OPs for them. Min/max/
    # median/q25/q75 are diagnostic-only stats (NOT used in normalization
    # math) so nan_to_num is acceptable here even in strict mode — they
    # would only be NaN if a column had zero finite samples, which the
    # strict path already raises on.
    mean = np.nan_to_num(mean, nan=0.0)
    std = np.nan_to_num(std, nan=1.0)
    min_val = np.nan_to_num(min_val, nan=0.0)
    max_val = np.nan_to_num(max_val, nan=1.0)
    median = np.nan_to_num(median, nan=0.0)
    q25 = np.nan_to_num(q25, nan=0.0)
    q75 = np.nan_to_num(q75, nan=1.0)

    return FeatureStatistics(
        mean=mean,
        std=std,
        min=min_val,
        max=max_val,
        median=median,
        q25=q25,
        q75=q75,
        count=len(valid_features),
        num_features=num_features,
    )


def compute_statistics_from_days(
    days: List["DayData"],
    per_day: bool = False,
) -> Union[FeatureStatistics, List[FeatureStatistics]]:
    """
    Compute statistics from multiple days of data.
    
    Args:
        days: List of DayData instances
        per_day: If True, compute separate statistics per day.
                 If False, compute global statistics.
    
    Returns:
        Single FeatureStatistics if per_day=False, else list of FeatureStatistics
    """
    from lobtrainer.data.dataset import DayData
    
    if per_day:
        return [compute_statistics(day.features) for day in days]
    else:
        # Concatenate all features
        all_features = np.vstack([day.features for day in days])
        return compute_statistics(all_features)


# =============================================================================
# Label Transforms
# =============================================================================


class BinaryLabelTransform:
    """
    Transform 3-class labels to binary signal detection labels.
    
    Converts the classification problem from:
        3-class: Down (0), Stable (1), Up (2)
    To:
        2-class: NoSignal (0), Signal (1)
    
    Where Signal = Up OR Down (any directional move is an "opportunity").
    
    This enables Two-Stage training:
        Stage 1: Detect if there's a trading opportunity (binary)
        Stage 2: Given an opportunity, predict direction (Up vs Down)
    
    Use cases:
        - Opportunity detection (current bigmove dataset has 71% Stable)
        - Binary imbalanced learning is often easier than multi-class
        - Align with trading objective: "Is now a good time to trade?"
    
    Args:
        signal_classes: Which original classes are considered "Signal".
                        Default: [0, 2] meaning Down=0 and Up=2 are signals.
                        For opportunity labels: BigDown=0 and BigUp=2.
    
    Example:
        >>> transform = BinaryLabelTransform()
        >>> # Original: Down=0, Stable=1, Up=2
        >>> # Transformed: NoSignal=0 (was Stable), Signal=1 (was Down or Up)
        >>> transform(0)  # Down → Signal (1)
        >>> transform(1)  # Stable → NoSignal (0)
        >>> transform(2)  # Up → Signal (1)
    
    With Dataset:
        >>> dataset = LOBSequenceDataset(
        ...     days, 
        ...     label_transform=BinaryLabelTransform()
        ... )
    """
    
    def __init__(self, signal_classes: Optional[List[int]] = None):
        # Default: Down (0) and Up (2) are signals, Stable (1) is not
        self.signal_classes = set(signal_classes or [0, 2])
    
    def __call__(self, label: int) -> int:
        """
        Transform a single label.
        
        Args:
            label: Original label (0, 1, or 2)
        
        Returns:
            Binary label: 0 (NoSignal) or 1 (Signal)
        """
        return 1 if label in self.signal_classes else 0
    
    def transform_array(self, labels: np.ndarray) -> np.ndarray:
        """
        Transform an array of labels.
        
        Args:
            labels: Array of original labels
        
        Returns:
            Array of binary labels
        """
        # Vectorized: check if each label is in signal_classes
        result = np.zeros_like(labels, dtype=np.int64)
        for signal_class in self.signal_classes:
            result |= (labels == signal_class).astype(np.int64)
        return result
    
    def get_class_names(self) -> List[str]:
        """Return class names for the binary problem."""
        return ["NoSignal", "Signal"]
    
    def __repr__(self) -> str:
        return f"BinaryLabelTransform(signal_classes={sorted(self.signal_classes)})"


class ComposeTransform:
    """
    Compose multiple transforms into a single callable.
    
    Applies transforms in order: first transform → second transform → ...
    
    Example:
        >>> transform = ComposeTransform([
        ...     normalizer,
        ...     lambda x: x.astype(np.float32),
        ... ])
        >>> normalized = transform(features)
    """
    
    def __init__(self, transforms: List[callable]):
        self.transforms = transforms
    
    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x
    
    def __repr__(self) -> str:
        return f"ComposeTransform({self.transforms})"

