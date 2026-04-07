"""
Global normalization for LOB data — Layout-Aware Implementation.

This module implements global Z-score normalization matching the official TLOB
repository methodology, but with support for different feature layouts.

The official TLOB repo (TLOB/preprocessing/lobster.py::_normalize_dataframes)
computes GLOBAL mean/std from ALL training data and applies to train/val/test:
    - mean_prices = ALL_TRAIN_PRICES.stack().mean()  (one value for all 20 price cols)
    - std_prices  = ALL_TRAIN_PRICES.stack().std()   (one value for all 20 price cols)
    - mean_sizes  = ALL_TRAIN_SIZES.stack().mean()   (one value for all 20 size cols)
    - std_sizes   = ALL_TRAIN_SIZES.stack().std()    (one value for all 20 size cols)

Feature Layouts Supported:
    - GROUPED: Our Rust pipeline format [ask_prices, ask_sizes, bid_prices, bid_sizes]
    - LOBSTER/FI2010: Interleaved format [ask_p_L0, ask_s_L0, bid_p_L0, bid_s_L0, ...]

The layout determines which columns are prices vs sizes. This is detected
automatically from export metadata, or can be specified explicitly.

Usage:
    from lobtrainer.data.normalization import GlobalZScoreNormalizer
    
    # Auto-detect layout from training data metadata (RECOMMENDED)
    normalizer = GlobalZScoreNormalizer.from_train_data(train_days)
    
    # Or specify layout explicitly
    normalizer = GlobalZScoreNormalizer.from_train_data(train_days, layout="grouped")
    
    # Apply to all splits
    train_dataset = LOBSequenceDataset(train_days, transform=normalizer)
    val_dataset = LOBSequenceDataset(val_days, transform=normalizer)
    test_dataset = LOBSequenceDataset(test_days, transform=normalizer)

Data Contract (Schema v2.2):
    - Price/size indices are determined by feature_index.py (single source of truth)
    - Layout is detected from metadata 'normalization.feature_layout' field
    - Backward compatible: defaults to 'grouped' (our pipeline) if not specified
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import logging
import numpy as np

from lobtrainer.constants.feature_index import (
    GROUPED_PRICE_INDICES,
    GROUPED_SIZE_INDICES,
    LOBSTER_PRICE_INDICES,
    LOBSTER_SIZE_INDICES,
    get_price_size_indices,
    detect_layout_from_metadata,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Welford/Chan Parallel Merge — Numerically Stable Streaming Statistics
# =============================================================================
#
# Replaces the naive Var = E[X^2] - E[X]^2 formula which suffers from
# catastrophic cancellation when the mean is large relative to the standard
# deviation (e.g., LOB prices ~130 USD with std ~0.01).
#
# Reference: Chan, Golub, LeVeque (1979) "Updating Formulae and a Pairwise
# Algorithm for Computing Sample Variances"


def _welford_init_scalar() -> Dict:
    """Initialize scalar Welford accumulators (for pooled price/size stats)."""
    return {"n": np.int64(0), "mean": np.float64(0.0), "m2": np.float64(0.0)}


def _welford_update_batch(state: Dict, batch: np.ndarray) -> None:
    """Update Welford accumulators with a pre-filtered numpy batch.

    The batch must already have NaN/Inf values removed (caller responsibility,
    matching the existing pipeline convention). Uses Chan's parallel merge
    formula: exact per-batch numpy statistics, numerically stable cross-batch
    merge.

    Args:
        state: Dict with 'n', 'mean', 'm2' accumulators (modified in-place).
        batch: 1D numpy array of finite values.
    """
    batch_n = np.int64(len(batch))
    if batch_n == 0:
        return  # No-op for empty batches

    batch_mean = np.float64(np.mean(batch, dtype=np.float64))
    batch_m2 = np.float64(np.sum((batch.astype(np.float64) - batch_mean) ** 2))

    # Chan's parallel merge formula
    delta = batch_mean - state["mean"]
    total_n = state["n"] + batch_n

    state["mean"] = state["mean"] + delta * np.float64(batch_n) / np.float64(total_n)
    state["m2"] = (
        state["m2"]
        + batch_m2
        + delta ** 2 * np.float64(state["n"]) * np.float64(batch_n) / np.float64(total_n)
    )
    state["n"] = total_n


def _welford_finalize(state: Dict, min_std: Optional[float] = None) -> Tuple[float, float]:
    """Finalize Welford accumulators to (mean, std).

    Uses population variance (M2/N), NOT sample variance (M2/(N-1)),
    matching the existing normalizer behavior and numpy.std default.

    Args:
        state: Dict with 'n', 'mean', 'm2' accumulators.
        min_std: Floor for std. None = no floor (GlobalZScoreNormalizer).
                 1.0 = floor to 1.0 when std < eps (HybridNormalizer).

    Returns:
        Tuple of (mean, std) as Python floats.
    """
    n = state["n"]
    if n == 0:
        mean = 0.0
        std = min_std if min_std is not None else 0.0
        return mean, std

    mean = float(state["mean"])

    if n > 1:
        var = float(state["m2"] / np.float64(n))
        var = max(var, 0.0)  # Guard against floating-point noise
        std = float(np.sqrt(var))
    else:
        std = 0.0

    if min_std is not None and std < 1e-8:
        std = min_std

    return mean, std


# =============================================================================
# Normalization Statistics
# =============================================================================


@dataclass
class GlobalNormalizationStats:
    """
    Global normalization statistics for LOB data.
    
    Stores single mean/std values for all prices and all sizes, matching
    the TLOB repository approach of computing global statistics.
    
    Attributes:
        mean_prices: Single mean for all price columns (20 columns in 40-feature LOB)
        std_prices: Single std for all price columns
        mean_sizes: Single mean for all size columns (20 columns in 40-feature LOB)
        std_sizes: Single std for all size columns
        layout: Feature layout used ("grouped" or "lobster")
        num_features: Number of features (typically 40)
    """
    mean_prices: float
    std_prices: float
    mean_sizes: float
    std_sizes: float
    layout: str = "grouped"
    num_features: int = 40
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "mean_prices": self.mean_prices,
            "std_prices": self.std_prices,
            "mean_sizes": self.mean_sizes,
            "std_sizes": self.std_sizes,
            "layout": self.layout,
            "num_features": self.num_features,
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> "GlobalNormalizationStats":
        """Create from dictionary (backward compatible)."""
        return cls(
            mean_prices=d["mean_prices"],
            std_prices=d["std_prices"],
            mean_sizes=d["mean_sizes"],
            std_sizes=d["std_sizes"],
            layout=d.get("layout", "grouped"),  # Default for backward compat
            num_features=d.get("num_features", 40),
        )
    
    def save(self, path: Union[str, Path]) -> None:
        """Save stats to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved normalization stats to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "GlobalNormalizationStats":
        """Load stats from JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


# =============================================================================
# Global Z-Score Normalizer
# =============================================================================


class GlobalZScoreNormalizer:
    """
    Layout-aware global Z-score normalizer for LOB data.
    
    Computes ONE mean/std for ALL prices and ONE mean/std for ALL sizes
    from the training set, then applies to all data. This matches the
    TLOB repository methodology.
    
    Key Innovation: Layout-Aware
    ----------------------------
    Different data formats arrange price/size columns differently:
    
    GROUPED (our Rust pipeline):
        [ask_prices(10), ask_sizes(10), bid_prices(10), bid_sizes(10)]
        Price indices: 0-9, 20-29 (20 total)
        Size indices: 10-19, 30-39 (20 total)
    
    LOBSTER/FI2010 (interleaved):
        [ask_p_L0, ask_s_L0, bid_p_L0, bid_s_L0, ask_p_L1, ...]
        Price indices: 0, 2, 4, ..., 38 (even indices)
        Size indices: 1, 3, 5, ..., 39 (odd indices)
    
    The normalizer auto-detects the layout from export metadata, ensuring
    correct normalization regardless of data source.
    
    Usage:
        # Auto-detect layout (RECOMMENDED)
        normalizer = GlobalZScoreNormalizer.from_train_data(train_days)
        
        # Explicit layout
        normalizer = GlobalZScoreNormalizer.from_train_data(train_days, layout="grouped")
        
        # Apply as transform
        normalized_data = normalizer(raw_data)
    
    Data Contract:
        - Price/size indices from lobtrainer.constants.feature_index
        - Layout stored in stats for reproducibility
        - Thread-safe for inference (read-only after construction)
    """
    
    def __init__(
        self,
        stats: GlobalNormalizationStats,
        layout: Optional[str] = None,
        eps: float = 1e-8,
    ):
        """
        Initialize normalizer with pre-computed stats.
        
        Args:
            stats: Global normalization statistics (includes layout info)
            layout: Override layout from stats (optional, for testing)
            eps: Small constant for numerical stability
        """
        self.stats = stats
        self.eps = eps
        
        # Determine layout (prefer explicit, fall back to stats)
        effective_layout = layout or stats.layout
        self.layout = effective_layout
        
        # Get column indices based on layout
        self._price_cols, self._size_cols = get_price_size_indices(effective_layout)
        
        # Limit indices to actual feature count
        num_features = stats.num_features
        self._price_cols = tuple(i for i in self._price_cols if i < num_features)
        self._size_cols = tuple(i for i in self._size_cols if i < num_features)
        
        logger.debug(
            f"Initialized normalizer: layout={self.layout}, "
            f"num_features={num_features}, "
            f"price_cols={len(self._price_cols)}, size_cols={len(self._size_cols)}"
        )
    
    @classmethod
    def from_train_data(
        cls,
        train_days: List,  # List[DayData]
        layout: Optional[str] = None,
        num_features: Optional[int] = None,
        eps: float = 1e-8,
    ) -> "GlobalZScoreNormalizer":
        """
        Compute global stats from all training data.
        
        This exactly matches TLOB/preprocessing/lobster.py::_normalize_dataframes(),
        but is layout-aware to support different feature arrangements.
        
        Args:
            train_days: List of DayData from training split
            layout: Feature layout ("grouped", "lobster"). If None, auto-detects
                    from metadata of first day.
            num_features: Override feature count (default: from data)
            eps: Small constant for numerical stability
        
        Returns:
            GlobalZScoreNormalizer with computed stats
        
        Raises:
            ValueError: If layout cannot be determined
        """
        if not train_days:
            raise ValueError("train_days cannot be empty")
        
        # Auto-detect layout from metadata if not specified
        if layout is None:
            if train_days[0].metadata:
                try:
                    layout = detect_layout_from_metadata(train_days[0].metadata)
                    logger.info(f"Auto-detected feature layout: {layout}")
                except ValueError:
                    layout = "grouped"  # Default to our pipeline format
                    logger.warning(
                        f"Could not detect layout from metadata, defaulting to '{layout}'"
                    )
            else:
                layout = "grouped"
                logger.info(f"No metadata available, using default layout: {layout}")
        
        # Infer num_features from data
        if num_features is None:
            if train_days[0].sequences is not None:
                num_features = train_days[0].sequences.shape[-1]
            else:
                num_features = train_days[0].features.shape[-1]
        
        # Get column indices for this layout
        price_cols, size_cols = get_price_size_indices(layout)
        
        # Limit to actual feature count
        price_cols = tuple(i for i in price_cols if i < num_features)
        size_cols = tuple(i for i in size_cols if i < num_features)
        
        logger.info(
            f"Computing global normalization stats from {len(train_days)} training days "
            f"(layout={layout}, features={num_features})"
        )
        logger.info(f"  Price columns: {len(price_cols)} indices: {price_cols[:5]}...")
        logger.info(f"  Size columns: {len(size_cols)} indices: {size_cols[:5]}...")
        
        # =====================================================================
        # STREAMING STATISTICS using Chan's parallel merge (vectorized Welford)
        # Memory: O(1) instead of O(total_samples * 20)
        #
        # Each day: compute exact mean + M2 with numpy (fast, vectorized)
        # Across days: merge using parallel Welford formula (numerically stable)
        # Reference: Chan, Golub, LeVeque (1979)
        # =====================================================================
        price_state = _welford_init_scalar()
        size_state = _welford_init_scalar()

        total_days = len(train_days)
        log_interval = max(1, total_days // 10)
        price_cols_list = list(price_cols)
        size_cols_list = list(size_cols)

        for day_idx, day in enumerate(train_days):
            if day_idx % log_interval == 0 or day_idx == total_days - 1:
                logger.info(f"  Processing day {day_idx + 1}/{total_days}...")

            # Use sequences if available, otherwise features
            if day.sequences is not None:
                data = day.sequences.reshape(-1, day.sequences.shape[-1])
            else:
                data = day.features

            # Extract and accumulate prices (vectorized)
            prices = data[:, price_cols_list].flatten()
            finite_mask = np.isfinite(prices)
            if not finite_mask.all():
                prices = prices[finite_mask]
            _welford_update_batch(price_state, prices)

            # Extract and accumulate sizes (vectorized)
            sizes = data[:, size_cols_list].flatten()
            finite_mask = np.isfinite(sizes)
            if not finite_mask.all():
                sizes = sizes[finite_mask]
            _welford_update_batch(size_state, sizes)

            del data  # Explicit cleanup

        # Finalize: population variance (M2/N). No min_std for GlobalZScore.
        mean_prices, std_prices = _welford_finalize(price_state)
        mean_sizes, std_sizes = _welford_finalize(size_state)

        stats = GlobalNormalizationStats(
            mean_prices=mean_prices,
            std_prices=std_prices,
            mean_sizes=mean_sizes,
            std_sizes=std_sizes,
            layout=layout,
            num_features=num_features,
        )

        logger.info(f"Global normalization stats (layout={layout}):")
        logger.info(f"  Prices: mean={stats.mean_prices:.6f}, std={stats.std_prices:.6f}")
        logger.info(f"  Sizes:  mean={stats.mean_sizes:.2f}, std={stats.std_sizes:.2f}")
        logger.info(
            f"  (Computed from {int(price_state['n']):,} price values and "
            f"{int(size_state['n']):,} size values)"
        )
        
        return cls(stats, layout=layout, eps=eps)
    
    @classmethod
    def from_data_dir(
        cls,
        data_dir: Union[str, Path],
        layout: Optional[str] = None,
        num_features: Optional[int] = None,
        eps: float = 1e-8,
        cache: bool = True,
    ) -> "GlobalZScoreNormalizer":
        """
        Compute or load cached global stats from data directory.
        
        If cache=True and stats file exists, loads from cache.
        Otherwise computes from training data and optionally saves.
        
        Args:
            data_dir: Root data directory containing train/val/test splits
            layout: Feature layout (auto-detected if None)
            num_features: Override feature count (auto-detected if None)
            eps: Small constant for numerical stability
            cache: If True, cache/load stats to/from data_dir/normalization_stats.json
        
        Returns:
            GlobalZScoreNormalizer with computed or loaded stats
        """
        from lobtrainer.data.dataset import load_split_data
        
        data_dir = Path(data_dir)
        stats_path = data_dir / "normalization_stats.json"
        
        # Try to load cached stats
        if cache and stats_path.exists():
            logger.info(f"Loading cached normalization stats from {stats_path}")
            stats = GlobalNormalizationStats.load(stats_path)
            return cls(stats, layout=layout, eps=eps)
        
        # Compute from training data
        train_days = load_split_data(data_dir, "train", validate=True)
        normalizer = cls.from_train_data(
            train_days,
            layout=layout,
            num_features=num_features,
            eps=eps,
        )
        
        # Cache stats
        if cache:
            normalizer.stats.save(stats_path)
        
        return normalizer
    
    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Apply global Z-score normalization.
        
        Args:
            data: Input array of shape [T, F] (sequence) or [F] (flat)
        
        Returns:
            Normalized array of same shape
        """
        # Handle both sequence [T, F] and flat [F] inputs
        if data.ndim == 1:
            return self._normalize_1d(data)
        elif data.ndim == 2:
            return self._normalize_2d(data)
        else:
            raise ValueError(f"Expected 1D or 2D array, got shape {data.shape}")
    
    def _normalize_1d(self, data: np.ndarray) -> np.ndarray:
        """Normalize flat [F] array."""
        result = data.copy()
        
        # Normalize prices
        for col in self._price_cols:
            if col < len(result):
                result[col] = (data[col] - self.stats.mean_prices) / (self.stats.std_prices + self.eps)
        
        # Normalize sizes
        for col in self._size_cols:
            if col < len(result):
                result[col] = (data[col] - self.stats.mean_sizes) / (self.stats.std_sizes + self.eps)
        
        return result
    
    def _normalize_2d(self, data: np.ndarray) -> np.ndarray:
        """Normalize sequence [T, F] array."""
        result = data.copy()
        
        # Normalize all price columns (vectorized)
        price_cols = [c for c in self._price_cols if c < data.shape[1]]
        if price_cols:
            result[:, price_cols] = (
                data[:, price_cols] - self.stats.mean_prices
            ) / (self.stats.std_prices + self.eps)
        
        # Normalize all size columns (vectorized)
        size_cols = [c for c in self._size_cols if c < data.shape[1]]
        if size_cols:
            result[:, size_cols] = (
                data[:, size_cols] - self.stats.mean_sizes
            ) / (self.stats.std_sizes + self.eps)
        
        return result
    
    def normalize_tensor(self, tensor) -> "torch.Tensor":
        """
        Apply normalization to PyTorch tensor.
        
        Args:
            tensor: Input tensor of shape [B, T, F] or [B, F]
        
        Returns:
            Normalized tensor of same shape
        """
        import torch
        
        price_cols = list(self._price_cols)
        size_cols = list(self._size_cols)
        
        if tensor.dim() == 2:
            # [B, F]
            result = tensor.clone()
            if price_cols:
                result[:, price_cols] = (
                    tensor[:, price_cols] - self.stats.mean_prices
                ) / (self.stats.std_prices + self.eps)
            if size_cols:
                result[:, size_cols] = (
                    tensor[:, size_cols] - self.stats.mean_sizes
                ) / (self.stats.std_sizes + self.eps)
        elif tensor.dim() == 3:
            # [B, T, F]
            result = tensor.clone()
            if price_cols:
                result[:, :, price_cols] = (
                    tensor[:, :, price_cols] - self.stats.mean_prices
                ) / (self.stats.std_prices + self.eps)
            if size_cols:
                result[:, :, size_cols] = (
                    tensor[:, :, size_cols] - self.stats.mean_sizes
                ) / (self.stats.std_sizes + self.eps)
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got shape {tensor.shape}")
        
        return result
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"GlobalZScoreNormalizer("
            f"layout='{self.layout}', "
            f"mean_prices={self.stats.mean_prices:.4f}, "
            f"std_prices={self.stats.std_prices:.4f}, "
            f"mean_sizes={self.stats.mean_sizes:.2f}, "
            f"std_sizes={self.stats.std_sizes:.2f})"
        )


# =============================================================================
# Hybrid Normalizer (Global LOB Z-score + Per-Feature Z-score)
# =============================================================================


@dataclass
class HybridNormalizationStats:
    """
    Normalization statistics for hybrid 98-feature normalization.
    
    Combines:
    - Global Z-score for raw LOB (indices 0-39): one mean/std for prices, one for sizes
    - Per-feature Z-score for derived/MBO/signals (indices 40-97)
    
    Attributes:
        lob_stats: GlobalNormalizationStats for raw LOB features
        per_feature_mean: Per-feature mean for indices 40-97
        per_feature_std: Per-feature std for indices 40-97
        exclude_indices: Feature indices to skip (categorical/binary)
        num_features: Total feature count
    """
    lob_stats: GlobalNormalizationStats
    per_feature_mean: np.ndarray  # shape (num_features,)
    per_feature_std: np.ndarray   # shape (num_features,)
    exclude_indices: Tuple[int, ...]
    num_features: int = 98
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "lob_stats": self.lob_stats.to_dict(),
            "per_feature_mean": self.per_feature_mean.tolist(),
            "per_feature_std": self.per_feature_std.tolist(),
            "exclude_indices": list(self.exclude_indices),
            "num_features": self.num_features,
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> "HybridNormalizationStats":
        """Create from dictionary."""
        return cls(
            lob_stats=GlobalNormalizationStats.from_dict(d["lob_stats"]),
            per_feature_mean=np.array(d["per_feature_mean"]),
            per_feature_std=np.array(d["per_feature_std"]),
            exclude_indices=tuple(d["exclude_indices"]),
            num_features=d.get("num_features", 98),
        )
    
    def save(self, path: Union[str, Path]) -> None:
        """Save stats to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved hybrid normalization stats to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "HybridNormalizationStats":
        """Load stats from JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


class HybridNormalizer:
    """
    Hybrid normalizer for 98-feature datasets.
    
    Applies different normalization strategies to different feature groups:
    
    | Range   | Features       | Strategy                              |
    |---------|----------------|---------------------------------------|
    | 0-39    | Raw LOB        | Global Z-score (prices pooled, sizes pooled) |
    | 40-91   | Derived/MBO/Signals | Per-feature Z-score             |
    | 92      | BOOK_VALID     | EXCLUDE (binary 0/1)                  |
    | 93      | TIME_REGIME    | EXCLUDE (categorical 0-4)             |
    | 94      | MBO_READY      | EXCLUDE (binary 0/1)                  |
    | 95      | DT_SECONDS     | Per-feature Z-score                   |
    | 96      | INVALIDITY_DELTA | EXCLUDE (counter)                   |
    | 97      | SCHEMA_VERSION | EXCLUDE (constant)                    |
    
    This matches the research design where:
    - Raw LOB prices/sizes have shared distributions across levels
    - Derived features have heterogeneous scales (bps, ratios, counts)
    - Categorical/binary features should not be normalized
    
    Usage:
        normalizer = HybridNormalizer.from_train_data(train_days)
        normalized = normalizer(raw_features)
    
    Reference: RULE.md §9 (ML Data Pipeline Integrity)
    """
    
    # Indices to exclude from normalization. Sourced from the pipeline contract
    # (pipeline_contract.toml [normalization].non_normalizable_indices) via
    # hft_contracts.NON_NORMALIZABLE_INDICES. Includes categorical, binary,
    # counter, and constant features.
    try:
        from hft_contracts import NON_NORMALIZABLE_INDICES as _CONTRACT_NON_NORM
        DEFAULT_EXCLUDE_INDICES: Tuple[int, ...] = tuple(sorted(_CONTRACT_NON_NORM))
    except ImportError:
        DEFAULT_EXCLUDE_INDICES: Tuple[int, ...] = (92, 93, 94, 96, 97, 115)
    
    # Raw LOB feature range
    LOB_FEATURE_END = 40
    
    def __init__(
        self,
        stats: HybridNormalizationStats,
        eps: float = 1e-8,
        clip_value: Optional[float] = 10.0,
    ):
        """
        Initialize with pre-computed stats.
        
        Args:
            stats: HybridNormalizationStats computed from training data
            eps: Small constant for numerical stability
            clip_value: Clip normalized values to [-clip_value, clip_value]
        """
        self.stats = stats
        self.eps = eps
        self.clip_value = clip_value
        
        # Get LOB column indices from the stats
        self._price_cols, self._size_cols = get_price_size_indices(
            stats.lob_stats.layout
        )
        # Limit to actual LOB range
        self._price_cols = tuple(i for i in self._price_cols if i < self.LOB_FEATURE_END)
        self._size_cols = tuple(i for i in self._size_cols if i < self.LOB_FEATURE_END)
        
        logger.debug(
            f"HybridNormalizer initialized: "
            f"LOB={self.LOB_FEATURE_END} features (prices={len(self._price_cols)}, sizes={len(self._size_cols)}), "
            f"per-feature={stats.num_features - self.LOB_FEATURE_END - len(stats.exclude_indices)} features, "
            f"excluded={len(stats.exclude_indices)} features"
        )
    
    @classmethod
    def from_train_data(
        cls,
        train_days: List,  # List[DayData]
        layout: Optional[str] = None,
        num_features: Optional[int] = None,
        exclude_indices: Optional[Tuple[int, ...]] = None,
        eps: float = 1e-8,
        clip_value: Optional[float] = 10.0,
    ) -> "HybridNormalizer":
        """
        Compute hybrid normalization stats from training data using streaming algorithm.

        Uses Chan's parallel merge formula (vectorized Welford) for numerically
        stable, memory-efficient computation. Processes one day at a time using
        numpy-vectorized batch statistics, then merges across days with the
        numerically stable parallel update formula.
        
        Memory: O(num_features) instead of O(num_days * samples_per_day * num_features)
        
        Args:
            train_days: List of DayData from training split
            layout: Feature layout for LOB ("grouped", "lobster"). Auto-detected if None.
            num_features: Total feature count. Auto-detected if None.
            exclude_indices: Feature indices to exclude. Uses DEFAULT_EXCLUDE_INDICES if None.
            eps: Small constant for numerical stability
            clip_value: Clip value for normalized features
        
        Returns:
            HybridNormalizer with computed stats
        """
        if not train_days:
            raise ValueError("train_days cannot be empty")
        
        # Auto-detect layout
        if layout is None:
            if train_days[0].metadata:
                try:
                    layout = detect_layout_from_metadata(train_days[0].metadata)
                    logger.info(f"Auto-detected feature layout: {layout}")
                except ValueError:
                    layout = "grouped"
                    logger.warning(f"Could not detect layout, defaulting to '{layout}'")
            else:
                layout = "grouped"
        
        # Infer num_features
        if num_features is None:
            if train_days[0].sequences is not None:
                num_features = train_days[0].sequences.shape[-1]
            else:
                num_features = train_days[0].features.shape[-1]
        
        # Use default exclude indices if not specified
        if exclude_indices is None:
            # Only use defaults that are within range
            exclude_indices = tuple(i for i in cls.DEFAULT_EXCLUDE_INDICES if i < num_features)
        
        logger.info(
            f"Computing hybrid normalization stats from {len(train_days)} training days "
            f"(layout={layout}, features={num_features}, exclude={exclude_indices})"
        )
        
        # Get LOB column indices
        price_cols, size_cols = get_price_size_indices(layout)
        price_cols = tuple(i for i in price_cols if i < cls.LOB_FEATURE_END)
        size_cols = tuple(i for i in size_cols if i < cls.LOB_FEATURE_END)
        price_cols_list = list(price_cols)
        size_cols_list = list(size_cols)
        
        # =====================================================================
        # STREAMING STATISTICS using Chan's parallel merge (vectorized Welford)
        # Memory: O(num_features) instead of O(total_samples * num_features)
        #
        # Each day: compute exact mean + M2 with numpy (fast, vectorized)
        # Across days: merge using parallel Welford formula (numerically stable)
        # Reference: Chan, Golub, LeVeque (1979)
        # =====================================================================

        # Welford accumulators for LOB prices and sizes (scalar, pooled)
        price_state = _welford_init_scalar()
        size_state = _welford_init_scalar()

        # Per-feature Welford accumulators (only for cols >= LOB_FEATURE_END
        # that are not excluded — LOB cols use the pooled accumulators above)
        feature_states = {}
        for col in range(cls.LOB_FEATURE_END, num_features):
            if col not in exclude_indices:
                feature_states[col] = _welford_init_scalar()

        total_days = len(train_days)
        log_interval = max(1, total_days // 10)

        for day_idx, day in enumerate(train_days):
            if day_idx % log_interval == 0 or day_idx == total_days - 1:
                logger.info(f"  Processing day {day_idx + 1}/{total_days}...")

            if day.sequences is not None:
                data = day.sequences.reshape(-1, day.sequences.shape[-1])
            else:
                data = day.features

            # Update LOB price stats (vectorized batch)
            prices = data[:, price_cols_list].flatten()
            finite_mask = np.isfinite(prices)
            if not finite_mask.all():
                prices = prices[finite_mask]
            _welford_update_batch(price_state, prices)

            # Update LOB size stats (vectorized batch)
            sizes = data[:, size_cols_list].flatten()
            finite_mask = np.isfinite(sizes)
            if not finite_mask.all():
                sizes = sizes[finite_mask]
            _welford_update_batch(size_state, sizes)

            # Update per-feature stats (non-LOB, non-excluded)
            for col, col_state in feature_states.items():
                col_data = data[:, col]
                finite_mask = np.isfinite(col_data)
                if not finite_mask.all():
                    col_data = col_data[finite_mask]
                _welford_update_batch(col_state, col_data)

            del data

        # Finalize LOB stats (min_std=1.0 to prevent division by zero)
        price_mean, price_std = _welford_finalize(price_state, min_std=1.0)
        size_mean, size_std = _welford_finalize(size_state, min_std=1.0)

        lob_stats = GlobalNormalizationStats(
            mean_prices=price_mean,
            std_prices=price_std,
            mean_sizes=size_mean,
            std_sizes=size_std,
            layout=layout,
            num_features=cls.LOB_FEATURE_END,
        )

        logger.info(f"LOB global stats (layout={layout}):")
        logger.info(f"  Prices: mean={lob_stats.mean_prices:.6f}, std={lob_stats.std_prices:.6f}")
        logger.info(f"  Sizes:  mean={lob_stats.mean_sizes:.2f}, std={lob_stats.std_sizes:.2f}")

        # Finalize per-feature stats
        per_feature_mean = np.zeros(num_features, dtype=np.float64)
        per_feature_std = np.ones(num_features, dtype=np.float64)

        for col in range(num_features):
            if col in exclude_indices:
                per_feature_mean[col] = 0.0
                per_feature_std[col] = 1.0
            elif col in feature_states:
                col_mean, col_std = _welford_finalize(feature_states[col], min_std=1.0)
                per_feature_mean[col] = col_mean
                per_feature_std[col] = col_std
            # else: LOB cols 0-39 keep default (mean=0, std=1) — they use lob_stats

        # Log per-feature stats for non-LOB features
        logger.info(f"Per-feature stats for indices {cls.LOB_FEATURE_END}-{num_features-1}:")
        for i in range(cls.LOB_FEATURE_END, min(num_features, cls.LOB_FEATURE_END + 5)):
            if i not in exclude_indices:
                logger.info(f"  [{i}]: mean={per_feature_mean[i]:.6f}, std={per_feature_std[i]:.6f}")
        if num_features > cls.LOB_FEATURE_END + 5:
            logger.info(f"  ... ({num_features - cls.LOB_FEATURE_END - 5} more)")

        logger.info(
            f"Normalization stats computed from {int(price_state['n']):,} price values, "
            f"{int(size_state['n']):,} size values"
        )
        
        stats = HybridNormalizationStats(
            lob_stats=lob_stats,
            per_feature_mean=per_feature_mean,
            per_feature_std=per_feature_std,
            exclude_indices=exclude_indices,
            num_features=num_features,
        )
        
        return cls(stats, eps=eps, clip_value=clip_value)
    
    @classmethod
    def from_cached_or_compute(
        cls,
        data_dir: Union[str, Path],
        eps: float = 1e-8,
        clip_value: Optional[float] = 10.0,
        force_recompute: bool = False,
        **kwargs,
    ) -> "HybridNormalizer":
        """
        Load HybridNormalizer from cached stats, or compute if not found.
        
        RECOMMENDED ENTRY POINT for training pipelines. This method:
        1. Checks for cached stats at data_dir/hybrid_normalization_stats.json
        2. If found, loads instantly (no data loading required)
        3. If not found, computes stats using streaming algorithm (memory-efficient)
        4. Caches results for future runs
        
        Args:
            data_dir: Root data directory containing train split
            eps: Small constant for numerical stability
            clip_value: Clip value for normalized features
            force_recompute: If True, recompute even if cache exists
            **kwargs: Passed to compute_hybrid_stats_streaming if computing
        
        Returns:
            HybridNormalizer with loaded or computed stats
        
        Example:
            # Fast loading (uses cache if available)
            normalizer = HybridNormalizer.from_cached_or_compute("data/exports/nvda")
            
            # Force recompute (e.g., after re-exporting data)
            normalizer = HybridNormalizer.from_cached_or_compute(
                "data/exports/nvda", force_recompute=True
            )
        """
        stats = load_or_compute_hybrid_stats(
            data_dir, 
            force_recompute=force_recompute,
            eps=eps,
            clip_value=clip_value,
            **kwargs
        )
        return cls(stats, eps=eps, clip_value=clip_value)
    
    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Apply hybrid normalization.
        
        Args:
            data: Input array of shape [T, F] (sequence) or [F] (flat)
        
        Returns:
            Normalized array of same shape
        """
        if data.ndim == 1:
            return self._normalize_1d(data)
        elif data.ndim == 2:
            return self._normalize_2d(data)
        else:
            raise ValueError(f"Expected 1D or 2D array, got shape {data.shape}")
    
    def _normalize_1d(self, data: np.ndarray) -> np.ndarray:
        """Normalize flat [F] array."""
        result = data.astype(np.float64, copy=True)
        num_features = min(len(result), self.stats.num_features)
        
        # Apply global Z-score to LOB prices
        for col in self._price_cols:
            if col < num_features:
                result[col] = (
                    data[col] - self.stats.lob_stats.mean_prices
                ) / (self.stats.lob_stats.std_prices + self.eps)
        
        # Apply global Z-score to LOB sizes
        for col in self._size_cols:
            if col < num_features:
                result[col] = (
                    data[col] - self.stats.lob_stats.mean_sizes
                ) / (self.stats.lob_stats.std_sizes + self.eps)
        
        # Apply per-feature Z-score to non-LOB features
        for col in range(self.LOB_FEATURE_END, num_features):
            if col not in self.stats.exclude_indices:
                result[col] = (
                    data[col] - self.stats.per_feature_mean[col]
                ) / (self.stats.per_feature_std[col] + self.eps)
        
        # Clip if specified
        if self.clip_value is not None:
            for col in range(num_features):
                if col not in self.stats.exclude_indices:
                    result[col] = np.clip(result[col], -self.clip_value, self.clip_value)
        
        return result
    
    def _normalize_2d(self, data: np.ndarray) -> np.ndarray:
        """Normalize sequence [T, F] array."""
        result = data.astype(np.float64, copy=True)
        num_features = min(data.shape[1], self.stats.num_features)
        
        # Apply global Z-score to LOB prices (vectorized)
        price_cols = [c for c in self._price_cols if c < num_features]
        if price_cols:
            result[:, price_cols] = (
                data[:, price_cols] - self.stats.lob_stats.mean_prices
            ) / (self.stats.lob_stats.std_prices + self.eps)
        
        # Apply global Z-score to LOB sizes (vectorized)
        size_cols = [c for c in self._size_cols if c < num_features]
        if size_cols:
            result[:, size_cols] = (
                data[:, size_cols] - self.stats.lob_stats.mean_sizes
            ) / (self.stats.lob_stats.std_sizes + self.eps)
        
        # Apply per-feature Z-score to non-LOB features (vectorized where possible)
        non_lob_cols = [
            c for c in range(self.LOB_FEATURE_END, num_features)
            if c not in self.stats.exclude_indices
        ]
        for col in non_lob_cols:
            result[:, col] = (
                data[:, col] - self.stats.per_feature_mean[col]
            ) / (self.stats.per_feature_std[col] + self.eps)
        
        # Clip if specified
        if self.clip_value is not None:
            all_normalize_cols = price_cols + size_cols + non_lob_cols
            for col in all_normalize_cols:
                result[:, col] = np.clip(result[:, col], -self.clip_value, self.clip_value)
        
        return result
    
    def normalize_tensor(self, tensor) -> "torch.Tensor":
        """
        Apply normalization to PyTorch tensor.
        
        Args:
            tensor: Input tensor of shape [B, T, F] or [B, F]
        
        Returns:
            Normalized tensor of same shape
        """
        import torch
        
        num_features = min(tensor.shape[-1], self.stats.num_features)
        price_cols = [c for c in self._price_cols if c < num_features]
        size_cols = [c for c in self._size_cols if c < num_features]
        non_lob_cols = [
            c for c in range(self.LOB_FEATURE_END, num_features)
            if c not in self.stats.exclude_indices
        ]
        
        result = tensor.clone().float()
        
        if tensor.dim() == 2:
            # [B, F]
            if price_cols:
                result[:, price_cols] = (
                    tensor[:, price_cols] - self.stats.lob_stats.mean_prices
                ) / (self.stats.lob_stats.std_prices + self.eps)
            if size_cols:
                result[:, size_cols] = (
                    tensor[:, size_cols] - self.stats.lob_stats.mean_sizes
                ) / (self.stats.lob_stats.std_sizes + self.eps)
            for col in non_lob_cols:
                result[:, col] = (
                    tensor[:, col] - self.stats.per_feature_mean[col]
                ) / (self.stats.per_feature_std[col] + self.eps)
                
        elif tensor.dim() == 3:
            # [B, T, F]
            if price_cols:
                result[:, :, price_cols] = (
                    tensor[:, :, price_cols] - self.stats.lob_stats.mean_prices
                ) / (self.stats.lob_stats.std_prices + self.eps)
            if size_cols:
                result[:, :, size_cols] = (
                    tensor[:, :, size_cols] - self.stats.lob_stats.mean_sizes
                ) / (self.stats.lob_stats.std_sizes + self.eps)
            for col in non_lob_cols:
                result[:, :, col] = (
                    tensor[:, :, col] - self.stats.per_feature_mean[col]
                ) / (self.stats.per_feature_std[col] + self.eps)
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got shape {tensor.shape}")
        
        # Clip only normalized columns (matching numpy _normalize_1d/_normalize_2d)
        if self.clip_value is not None:
            all_normalize_cols = price_cols + size_cols + non_lob_cols
            if all_normalize_cols:
                idx = torch.tensor(all_normalize_cols, dtype=torch.long, device=tensor.device)
                if tensor.dim() == 2:
                    result[:, idx] = torch.clamp(
                        result[:, idx], -self.clip_value, self.clip_value
                    )
                elif tensor.dim() == 3:
                    result[:, :, idx] = torch.clamp(
                        result[:, :, idx], -self.clip_value, self.clip_value
                    )

        return result
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"HybridNormalizer("
            f"lob_features={self.LOB_FEATURE_END}, "
            f"total_features={self.stats.num_features}, "
            f"excluded={self.stats.exclude_indices})"
        )


# =============================================================================
# Utility Functions
# =============================================================================


def compute_and_save_normalization_stats(
    data_dir: Union[str, Path],
    layout: Optional[str] = None,
    num_features: Optional[int] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> GlobalNormalizationStats:
    """
    Utility function to compute and save normalization stats.
    
    This can be run once before training to pre-compute stats.
    
    Args:
        data_dir: Root data directory containing train split
        layout: Feature layout (auto-detected if None)
        num_features: Number of features (auto-detected if None)
        output_path: Path to save stats (default: data_dir/normalization_stats.json)
    
    Returns:
        Computed normalization stats
    """
    from lobtrainer.data.dataset import load_split_data
    
    data_dir = Path(data_dir)
    output_path = output_path or data_dir / "normalization_stats.json"
    
    # Load training data with lazy loading for memory efficiency
    train_days = load_split_data(data_dir, "train", validate=False, lazy=True)
    
    # Compute stats
    normalizer = GlobalZScoreNormalizer.from_train_data(
        train_days,
        layout=layout,
        num_features=num_features,
    )
    
    # Save
    normalizer.stats.save(output_path)
    
    return normalizer.stats


def compute_hybrid_stats_streaming(
    data_dir: Union[str, Path],
    layout: Optional[str] = None,
    num_features: Optional[int] = None,
    exclude_indices: Optional[Tuple[int, ...]] = None,
    eps: float = 1e-8,
    clip_value: Optional[float] = 10.0,
    output_path: Optional[Union[str, Path]] = None,
) -> HybridNormalizationStats:
    """
    Compute hybrid normalization stats using streaming algorithm.
    
    MEMORY EFFICIENT: Loads one day at a time, computes partial stats,
    then discards. Never holds more than one day's data in memory.
    
    This should be called BEFORE training to pre-compute stats, which are
    then cached to a JSON file and loaded instantly on subsequent runs.
    
    Args:
        data_dir: Root data directory containing train split
        layout: Feature layout ("grouped" or "lobster"). Auto-detected if None.
        num_features: Total feature count. Auto-detected if None.
        exclude_indices: Feature indices to exclude from normalization.
        eps: Small constant for numerical stability.
        clip_value: Clip value for normalized features.
        output_path: Path to save stats. Default: data_dir/hybrid_normalization_stats.json
    
    Returns:
        HybridNormalizationStats
    
    Example:
        # Pre-compute stats (run once after export)
        stats = compute_hybrid_stats_streaming("data/exports/nvda_98feat")
        
        # Training will then load cached stats instantly
        normalizer = HybridNormalizer.from_cached_or_compute(data_dir)
    """
    import gc
    
    data_dir = Path(data_dir)
    split_dir = data_dir / "train"
    output_path = output_path or data_dir / "hybrid_normalization_stats.json"
    
    if not split_dir.exists():
        raise FileNotFoundError(f"Train split not found: {split_dir}")
    
    # Find all sequence files
    seq_files = sorted(split_dir.glob("*_sequences.npy"))
    if not seq_files:
        seq_files = sorted(split_dir.glob("*_features.npy"))
    
    if not seq_files:
        raise FileNotFoundError(f"No data files found in {split_dir}")
    
    # Auto-detect layout from first day's metadata
    first_date = seq_files[0].stem.replace('_sequences', '').replace('_features', '')
    meta_path = split_dir / f"{first_date}_metadata.json"
    
    if layout is None and meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)
        try:
            layout = detect_layout_from_metadata(metadata)
            logger.info(f"Auto-detected feature layout: {layout}")
        except ValueError:
            layout = "grouped"
    elif layout is None:
        layout = "grouped"
    
    # Auto-detect num_features from first file (header only, no data load)
    with open(seq_files[0], 'rb') as f:
        version = np.lib.format.read_magic(f)
        if version[0] == 1:
            shape, _, _ = np.lib.format.read_array_header_1_0(f)
        else:
            shape, _, _ = np.lib.format.read_array_header_2_0(f)
    
    if num_features is None:
        num_features = shape[-1]
    
    # Use default exclude indices
    if exclude_indices is None:
        exclude_indices = tuple(
            i for i in HybridNormalizer.DEFAULT_EXCLUDE_INDICES 
            if i < num_features
        )
    
    logger.info(
        f"Computing hybrid normalization stats (STREAMING) from {len(seq_files)} days "
        f"(layout={layout}, features={num_features})"
    )
    
    # Get LOB column indices
    price_cols, size_cols = get_price_size_indices(layout)
    price_cols = [i for i in price_cols if i < HybridNormalizer.LOB_FEATURE_END]
    size_cols = [i for i in size_cols if i < HybridNormalizer.LOB_FEATURE_END]
    
    # =========================================================================
    # STREAMING ACCUMULATION using Chan's parallel merge (vectorized Welford)
    # One file at a time, mmap for memory efficiency
    # Reference: Chan, Golub, LeVeque (1979)
    # =========================================================================

    price_state = _welford_init_scalar()
    size_state = _welford_init_scalar()

    feature_states = {}
    for col in range(HybridNormalizer.LOB_FEATURE_END, num_features):
        if col not in exclude_indices:
            feature_states[col] = _welford_init_scalar()

    total_files = len(seq_files)
    log_interval = max(1, total_files // 10)

    for file_idx, seq_file in enumerate(seq_files):
        if file_idx % log_interval == 0 or file_idx == total_files - 1:
            logger.info(f"  Processing file {file_idx + 1}/{total_files}...")

        raw = np.load(seq_file, mmap_mode='r')
        data = raw.reshape(-1, raw.shape[-1]) if raw.ndim == 3 else raw

        # LOB prices
        prices = data[:, price_cols].flatten()
        finite_mask = np.isfinite(prices)
        if not finite_mask.all():
            prices = prices[finite_mask]
        _welford_update_batch(price_state, prices)

        # LOB sizes
        sizes = data[:, size_cols].flatten()
        finite_mask = np.isfinite(sizes)
        if not finite_mask.all():
            sizes = sizes[finite_mask]
        _welford_update_batch(size_state, sizes)

        # Per-feature (non-LOB, non-excluded)
        for col, col_state in feature_states.items():
            col_data = data[:, col]
            finite_mask = np.isfinite(col_data)
            if not finite_mask.all():
                col_data = col_data[finite_mask]
            _welford_update_batch(col_state, col_data)

        del raw, data
        gc.collect()

    # =========================================================================
    # FINALIZE STATISTICS
    # =========================================================================

    price_mean, price_std = _welford_finalize(price_state, min_std=1.0)
    size_mean, size_std = _welford_finalize(size_state, min_std=1.0)

    lob_stats = GlobalNormalizationStats(
        mean_prices=price_mean,
        std_prices=price_std,
        mean_sizes=size_mean,
        std_sizes=size_std,
        layout=layout,
        num_features=HybridNormalizer.LOB_FEATURE_END,
    )

    per_feature_mean = np.zeros(num_features, dtype=np.float64)
    per_feature_std = np.ones(num_features, dtype=np.float64)

    for col in range(num_features):
        if col in exclude_indices:
            per_feature_mean[col] = 0.0
            per_feature_std[col] = 1.0
        elif col in feature_states:
            col_mean, col_std = _welford_finalize(feature_states[col], min_std=1.0)
            per_feature_mean[col] = col_mean
            per_feature_std[col] = col_std

    logger.info(f"LOB global stats:")
    logger.info(f"  Prices: mean={lob_stats.mean_prices:.6f}, std={lob_stats.std_prices:.6f}")
    logger.info(f"  Sizes:  mean={lob_stats.mean_sizes:.2f}, std={lob_stats.std_sizes:.2f}")
    logger.info(
        f"Computed from {int(price_state['n']):,} price values, "
        f"{int(size_state['n']):,} size values"
    )
    
    stats = HybridNormalizationStats(
        lob_stats=lob_stats,
        per_feature_mean=per_feature_mean,
        per_feature_std=per_feature_std,
        exclude_indices=exclude_indices,
        num_features=num_features,
    )
    
    # Save to cache
    stats.save(output_path)
    logger.info(f"Saved hybrid normalization stats to {output_path}")
    
    return stats


def load_or_compute_hybrid_stats(
    data_dir: Union[str, Path],
    force_recompute: bool = False,
    **kwargs,
) -> HybridNormalizationStats:
    """
    Load cached hybrid normalization stats, or compute if not found.
    
    This is the recommended entry point for training pipelines.
    
    Args:
        data_dir: Root data directory
        force_recompute: If True, recompute even if cache exists
        **kwargs: Passed to compute_hybrid_stats_streaming if computing
    
    Returns:
        HybridNormalizationStats (loaded or computed)
    """
    data_dir = Path(data_dir)
    cache_path = data_dir / "hybrid_normalization_stats.json"
    
    if cache_path.exists() and not force_recompute:
        logger.info(f"Loading cached hybrid normalization stats from {cache_path}")
        return HybridNormalizationStats.load(cache_path)
    
    logger.info("Computing hybrid normalization stats (no cache found)...")
    return compute_hybrid_stats_streaming(data_dir, **kwargs)
