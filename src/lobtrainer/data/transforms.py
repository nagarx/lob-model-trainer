"""
Feature transforms and normalization for LOB data.

Design principles (per RULE.md):
- All transforms are computed from TRAINING data only (no leakage)
- Explicit handling of NaN/Inf values
- Categorical features (e.g., time_regime) excluded from normalization
- Statistics are serializable for reproducibility

Normalization strategies:
- Z-score: (x - mean) / std
- Per-day Z-score: Normalize within each trading day
- Min-max: (x - min) / (max - min)
- Robust: (x - median) / IQR

Label transforms:
- BinaryLabelTransform: 3-class (Down/Stable/Up) → 2-class (Signal/NoSignal)
- Supports Two-Stage training: Stage 1 detects signals, Stage 2 predicts direction

WARNING: Always check np.isfinite() before comparisons per RULE.md.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
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
) -> FeatureStatistics:
    """
    Compute feature statistics for normalization.
    
    Args:
        features: Feature array of shape (N, num_features)
        valid_mask: Optional boolean mask of shape (N,) for valid samples.
                    If None, uses book_valid > 0.5 as mask.
        eps: Minimum value for std to prevent division by zero
    
    Returns:
        FeatureStatistics instance
    
    Raises:
        ValueError: If no valid samples
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
    
    # Ensure std is at least eps to prevent division by zero
    std = np.maximum(std, eps)
    
    # Replace any remaining NaN with safe defaults
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
# Normalizer Base Class
# =============================================================================


class Normalizer(ABC):
    """
    Abstract base class for feature normalizers.
    
    Normalizers are fitted on training data and applied to all splits.
    They handle NaN/Inf values and support feature exclusion.
    """
    
    @abstractmethod
    def fit(self, features: np.ndarray) -> "Normalizer":
        """
        Fit normalizer on training data.
        
        Args:
            features: Training features of shape (N, num_features)
        
        Returns:
            self for method chaining
        """
        pass
    
    @abstractmethod
    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted statistics.
        
        Args:
            features: Features to transform, shape (N, num_features) or (T, num_features)
        
        Returns:
            Transformed features with same shape
        """
        pass
    
    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(features).transform(features)
    
    def __call__(self, features: np.ndarray) -> np.ndarray:
        """Apply transform (for use as Dataset transform)."""
        return self.transform(features)
    
    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """Save normalizer state to file."""
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path: Union[str, Path]) -> "Normalizer":
        """Load normalizer from file."""
        pass


# =============================================================================
# Z-Score Normalizer
# =============================================================================


class ZScoreNormalizer(Normalizer):
    """
    Z-score normalizer: (x - mean) / std.
    
    Features are normalized to have zero mean and unit variance.
    Handles NaN/Inf by preserving them (model should handle separately).
    
    Args:
        eps: Minimum std to prevent division by zero
        clip_value: Clip normalized values to [-clip_value, clip_value]
        exclude_features: Feature indices to exclude from normalization
    
    Example:
        >>> normalizer = ZScoreNormalizer(clip_value=10.0)
        >>> normalizer.fit(train_features)
        >>> normalized = normalizer.transform(test_features)
    """
    
    def __init__(
        self,
        eps: float = 1e-8,
        clip_value: Optional[float] = 10.0,
        exclude_features: Optional[List[int]] = None,
    ):
        self.eps = eps
        self.clip_value = clip_value
        self.exclude_features = set(exclude_features or [FeatureIndex.TIME_REGIME])
        
        self._stats: Optional[FeatureStatistics] = None
        self._is_fitted = False
    
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
    
    @property
    def stats(self) -> FeatureStatistics:
        if not self._is_fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        return self._stats
    
    def fit(self, features: np.ndarray) -> "ZScoreNormalizer":
        """
        Fit normalizer on training data.
        
        Args:
            features: Training features of shape (N, num_features)
        
        Returns:
            self for method chaining
        """
        self._stats = compute_statistics(features, eps=self.eps)
        self._is_fitted = True
        return self
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Apply Z-score normalization.
        
        Args:
            features: Features to transform, shape (..., num_features)
        
        Returns:
            Normalized features with same shape
        """
        if not self._is_fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        
        # Work with copy to avoid modifying input
        result = features.astype(np.float64, copy=True)
        original_shape = result.shape
        
        # Flatten to 2D for processing
        if result.ndim == 1:
            result = result.reshape(1, -1)
        elif result.ndim > 2:
            result = result.reshape(-1, result.shape[-1])
        
        num_features = result.shape[1]
        
        # Apply normalization to each feature
        for i in range(num_features):
            if i in self.exclude_features:
                continue
            
            # Z-score: (x - mean) / std
            result[:, i] = (result[:, i] - self._stats.mean[i]) / self._stats.std[i]
        
        # Clip if specified
        if self.clip_value is not None:
            # Only clip normalized features (not excluded ones)
            for i in range(num_features):
                if i not in self.exclude_features:
                    result[:, i] = np.clip(result[:, i], -self.clip_value, self.clip_value)
        
        # Restore original shape
        return result.reshape(original_shape)
    
    def inverse_transform(self, features: np.ndarray) -> np.ndarray:
        """
        Inverse Z-score normalization: x = normalized * std + mean.
        
        Args:
            features: Normalized features, shape (..., num_features)
        
        Returns:
            Original-scale features
        """
        if not self._is_fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        
        result = features.astype(np.float64, copy=True)
        original_shape = result.shape
        
        if result.ndim == 1:
            result = result.reshape(1, -1)
        elif result.ndim > 2:
            result = result.reshape(-1, result.shape[-1])
        
        num_features = result.shape[1]
        
        for i in range(num_features):
            if i in self.exclude_features:
                continue
            result[:, i] = result[:, i] * self._stats.std[i] + self._stats.mean[i]
        
        return result.reshape(original_shape)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save normalizer state to JSON file."""
        data = {
            "type": "zscore",
            "eps": self.eps,
            "clip_value": self.clip_value,
            "exclude_features": list(self.exclude_features),
            "stats": self._stats.to_dict() if self._stats else None,
            "is_fitted": self._is_fitted,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "ZScoreNormalizer":
        """Load normalizer from JSON file."""
        with open(path) as f:
            data = json.load(f)
        
        normalizer = cls(
            eps=data["eps"],
            clip_value=data["clip_value"],
            exclude_features=data["exclude_features"],
        )
        
        if data["stats"] is not None:
            normalizer._stats = FeatureStatistics.from_dict(data["stats"])
            normalizer._is_fitted = data["is_fitted"]
        
        return normalizer


# =============================================================================
# Per-Day Normalizer
# =============================================================================


class PerDayNormalizer(Normalizer):
    """
    Normalizer that computes statistics per trading day.
    
    Each day is normalized independently using its own statistics.
    Useful for handling intraday patterns and day-to-day variations.
    
    Args:
        base_normalizer: Normalizer class to use for each day
        kwargs: Arguments to pass to base normalizer
    """
    
    def __init__(
        self,
        eps: float = 1e-8,
        clip_value: Optional[float] = 10.0,
        exclude_features: Optional[List[int]] = None,
    ):
        self.eps = eps
        self.clip_value = clip_value
        self.exclude_features = set(exclude_features or [FeatureIndex.TIME_REGIME])
        
        # Per-day statistics are computed at transform time
        self._global_stats: Optional[FeatureStatistics] = None
        self._is_fitted = False
    
    def fit(self, features: np.ndarray) -> "PerDayNormalizer":
        """
        Fit global fallback statistics (used if day has insufficient data).
        
        Args:
            features: Training features (all days combined)
        
        Returns:
            self
        """
        self._global_stats = compute_statistics(features, eps=self.eps)
        self._is_fitted = True
        return self
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Transform using per-sample statistics (for single day).
        
        For per-day normalization, call transform_day() with day boundaries.
        This method normalizes using global statistics as fallback.
        """
        if not self._is_fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        
        # Use Z-score with global stats
        result = features.astype(np.float64, copy=True)
        original_shape = result.shape
        
        if result.ndim == 1:
            result = result.reshape(1, -1)
        elif result.ndim > 2:
            result = result.reshape(-1, result.shape[-1])
        
        num_features = result.shape[1]
        
        for i in range(num_features):
            if i in self.exclude_features:
                continue
            result[:, i] = (result[:, i] - self._global_stats.mean[i]) / self._global_stats.std[i]
        
        if self.clip_value is not None:
            for i in range(num_features):
                if i not in self.exclude_features:
                    result[:, i] = np.clip(result[:, i], -self.clip_value, self.clip_value)
        
        return result.reshape(original_shape)
    
    def transform_day(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize a single day's features using that day's statistics.
        
        Args:
            features: Features for a single day, shape (N, num_features)
        
        Returns:
            Normalized features
        """
        # Compute day-specific statistics
        day_stats = compute_statistics(features, eps=self.eps)
        
        result = features.astype(np.float64, copy=True)
        num_features = result.shape[1]
        
        for i in range(num_features):
            if i in self.exclude_features:
                continue
            result[:, i] = (result[:, i] - day_stats.mean[i]) / day_stats.std[i]
        
        if self.clip_value is not None:
            for i in range(num_features):
                if i not in self.exclude_features:
                    result[:, i] = np.clip(result[:, i], -self.clip_value, self.clip_value)
        
        return result
    
    def save(self, path: Union[str, Path]) -> None:
        """Save normalizer state."""
        data = {
            "type": "per_day",
            "eps": self.eps,
            "clip_value": self.clip_value,
            "exclude_features": list(self.exclude_features),
            "global_stats": self._global_stats.to_dict() if self._global_stats else None,
            "is_fitted": self._is_fitted,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "PerDayNormalizer":
        """Load normalizer from file."""
        with open(path) as f:
            data = json.load(f)
        
        normalizer = cls(
            eps=data["eps"],
            clip_value=data["clip_value"],
            exclude_features=data["exclude_features"],
        )
        
        if data["global_stats"] is not None:
            normalizer._global_stats = FeatureStatistics.from_dict(data["global_stats"])
            normalizer._is_fitted = data["is_fitted"]
        
        return normalizer


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
        ...     ZScoreNormalizer().fit(train_data),
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

