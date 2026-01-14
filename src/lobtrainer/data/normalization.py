"""
Global normalization for LOB data matching the official TLOB repository.

The official TLOB repo (TLOB/preprocessing/lobster.py::_normalize_dataframes)
computes GLOBAL mean/std from ALL training data and applies to train/val/test:
    - mean_prices = ALL_TRAIN_PRICES.stack().mean()  (one value for all 20 price cols)
    - std_prices  = ALL_TRAIN_PRICES.stack().std()   (one value for all 20 price cols)
    - mean_sizes  = ALL_TRAIN_SIZES.stack().mean()   (one value for all 20 size cols)
    - std_sizes   = ALL_TRAIN_SIZES.stack().std()    (one value for all 20 size cols)

This module implements the same normalization for our PyTorch datasets.

Usage:
    from lobtrainer.data.normalization import GlobalZScoreNormalizer
    
    # Compute stats from training data
    normalizer = GlobalZScoreNormalizer.from_train_data(train_days)
    
    # Apply to all splits
    train_dataset = LOBSequenceDataset(train_days, transform=normalizer)
    val_dataset = LOBSequenceDataset(val_days, transform=normalizer)
    test_dataset = LOBSequenceDataset(test_days, transform=normalizer)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import logging
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GlobalNormalizationStats:
    """
    Global normalization statistics matching TLOB repo.
    
    Attributes:
        mean_prices: Single mean for all 20 price columns
        std_prices: Single std for all 20 price columns
        mean_sizes: Single mean for all 20 size columns  
        std_sizes: Single std for all 20 size columns
    """
    mean_prices: float
    std_prices: float
    mean_sizes: float
    std_sizes: float
    
    # Column indices for 40-feature LOB format (TLOB repo layout)
    # Price columns: 0, 2, 4, ..., 38 (ask_p1, bid_p1, ask_p2, bid_p2, ...)
    # Size columns: 1, 3, 5, ..., 39 (ask_s1, bid_s1, ask_s2, bid_s2, ...)
    PRICE_COLS = list(range(0, 40, 2))  # [0, 2, 4, ..., 38]
    SIZE_COLS = list(range(1, 40, 2))   # [1, 3, 5, ..., 39]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "mean_prices": self.mean_prices,
            "std_prices": self.std_prices,
            "mean_sizes": self.mean_sizes,
            "std_sizes": self.std_sizes,
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> "GlobalNormalizationStats":
        """Create from dictionary."""
        return cls(
            mean_prices=d["mean_prices"],
            std_prices=d["std_prices"],
            mean_sizes=d["mean_sizes"],
            std_sizes=d["std_sizes"],
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


class GlobalZScoreNormalizer:
    """
    Global Z-score normalizer matching official TLOB repository.
    
    Computes ONE mean/std for ALL prices and ONE mean/std for ALL sizes
    from the training set, then applies to all data.
    
    This matches TLOB/utils/utils_data.py::z_score_orderbook():
        mean_prices = data.iloc[:, 0::2].stack().mean()
        std_prices = data.iloc[:, 0::2].stack().std()
        mean_size = data.iloc[:, 1::2].stack().mean()
        std_size = data.iloc[:, 1::2].stack().std()
    
    Note: Assumes 40-feature LOB format where:
        - Columns 0, 2, 4, ..., 38 are prices (ask_p1, bid_p1, ask_p2, bid_p2, ...)
        - Columns 1, 3, 5, ..., 39 are sizes (ask_s1, bid_s1, ask_s2, bid_s2, ...)
    """
    
    def __init__(
        self,
        stats: GlobalNormalizationStats,
        num_features: int = 40,
        eps: float = 1e-8,
    ):
        """
        Initialize normalizer with pre-computed stats.
        
        Args:
            stats: Global normalization statistics
            num_features: Number of features (default 40 for standard LOB)
            eps: Small constant for numerical stability
        """
        self.stats = stats
        self.num_features = num_features
        self.eps = eps
        
        # Precompute column masks for efficiency
        self._price_cols = GlobalNormalizationStats.PRICE_COLS[:num_features // 2]
        self._size_cols = GlobalNormalizationStats.SIZE_COLS[:num_features // 2]
    
    @classmethod
    def from_train_data(
        cls,
        train_days: List,  # List[DayData]
        num_features: int = 40,
        eps: float = 1e-8,
    ) -> "GlobalZScoreNormalizer":
        """
        Compute global stats from all training data.
        
        This exactly matches TLOB/preprocessing/lobster.py::_normalize_dataframes().
        
        Args:
            train_days: List of DayData from training split
            num_features: Number of features per sample
            eps: Small constant for numerical stability
        
        Returns:
            GlobalZScoreNormalizer with computed stats
        """
        price_cols = GlobalNormalizationStats.PRICE_COLS[:num_features // 2]
        size_cols = GlobalNormalizationStats.SIZE_COLS[:num_features // 2]
        
        all_prices = []
        all_sizes = []
        
        logger.info(f"Computing global normalization stats from {len(train_days)} training days...")
        
        for day in train_days:
            # Use sequences if available, otherwise features
            if day.sequences is not None:
                # Flatten sequences: [N_seq, T, F] -> [N_seq * T, F]
                data = day.sequences.reshape(-1, day.sequences.shape[-1])
            else:
                data = day.features
            
            # Extract prices and sizes (stack all together like TLOB repo)
            all_prices.append(data[:, price_cols].flatten())
            all_sizes.append(data[:, size_cols].flatten())
        
        # Concatenate all and compute global stats
        all_prices = np.concatenate(all_prices)
        all_sizes = np.concatenate(all_sizes)
        
        stats = GlobalNormalizationStats(
            mean_prices=float(np.mean(all_prices)),
            std_prices=float(np.std(all_prices)),
            mean_sizes=float(np.mean(all_sizes)),
            std_sizes=float(np.std(all_sizes)),
        )
        
        logger.info(f"Global normalization stats:")
        logger.info(f"  Prices: mean={stats.mean_prices:.4f}, std={stats.std_prices:.4f}")
        logger.info(f"  Sizes:  mean={stats.mean_sizes:.4f}, std={stats.std_sizes:.4f}")
        logger.info(f"  (Computed from {len(all_prices):,} price values and {len(all_sizes):,} size values)")
        
        return cls(stats, num_features, eps)
    
    @classmethod
    def from_data_dir(
        cls,
        data_dir: Union[str, Path],
        num_features: int = 40,
        eps: float = 1e-8,
        cache: bool = True,
    ) -> "GlobalZScoreNormalizer":
        """
        Compute or load cached global stats from data directory.
        
        If cache=True and stats file exists, loads from cache.
        Otherwise computes from training data and optionally saves.
        
        Args:
            data_dir: Root data directory containing train/val/test splits
            num_features: Number of features per sample
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
            return cls(stats, num_features, eps)
        
        # Compute from training data
        train_days = load_split_data(data_dir, "train", validate=True)
        normalizer = cls.from_train_data(train_days, num_features, eps)
        
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
            result[col] = (data[col] - self.stats.mean_prices) / (self.stats.std_prices + self.eps)
        
        # Normalize sizes
        for col in self._size_cols:
            result[col] = (data[col] - self.stats.mean_sizes) / (self.stats.std_sizes + self.eps)
        
        return result
    
    def _normalize_2d(self, data: np.ndarray) -> np.ndarray:
        """Normalize sequence [T, F] array."""
        result = data.copy()
        
        # Normalize all price columns
        for col in self._price_cols:
            result[:, col] = (data[:, col] - self.stats.mean_prices) / (self.stats.std_prices + self.eps)
        
        # Normalize all size columns
        for col in self._size_cols:
            result[:, col] = (data[:, col] - self.stats.mean_sizes) / (self.stats.std_sizes + self.eps)
        
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
        
        if tensor.dim() == 2:
            # [B, F]
            result = tensor.clone()
            for col in self._price_cols:
                result[:, col] = (tensor[:, col] - self.stats.mean_prices) / (self.stats.std_prices + self.eps)
            for col in self._size_cols:
                result[:, col] = (tensor[:, col] - self.stats.mean_sizes) / (self.stats.std_sizes + self.eps)
        elif tensor.dim() == 3:
            # [B, T, F]
            result = tensor.clone()
            for col in self._price_cols:
                result[:, :, col] = (tensor[:, :, col] - self.stats.mean_prices) / (self.stats.std_prices + self.eps)
            for col in self._size_cols:
                result[:, :, col] = (tensor[:, :, col] - self.stats.mean_sizes) / (self.stats.std_sizes + self.eps)
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got shape {tensor.shape}")
        
        return result


def compute_and_save_normalization_stats(
    data_dir: Union[str, Path],
    num_features: int = 40,
    output_path: Optional[Union[str, Path]] = None,
) -> GlobalNormalizationStats:
    """
    Utility function to compute and save normalization stats.
    
    This can be run once before training to pre-compute stats.
    
    Args:
        data_dir: Root data directory containing train split
        num_features: Number of features
        output_path: Path to save stats (default: data_dir/normalization_stats.json)
    
    Returns:
        Computed normalization stats
    """
    from lobtrainer.data.dataset import load_split_data
    
    data_dir = Path(data_dir)
    output_path = output_path or data_dir / "normalization_stats.json"
    
    # Load training data
    train_days = load_split_data(data_dir, "train", validate=True)
    
    # Compute stats
    normalizer = GlobalZScoreNormalizer.from_train_data(train_days, num_features)
    
    # Save
    normalizer.stats.save(output_path)
    
    return normalizer.stats
