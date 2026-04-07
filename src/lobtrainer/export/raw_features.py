"""
Raw feature extraction from disk for signal export.

The Trainer's DataLoader yields normalized, feature-subsetted tensors.
Raw spread_bps and mid_price are NOT accessible through it — they may be
excluded by feature selection and are always normalized. This module
extracts these raw values directly from the .npy export files on disk.

Uses the same sorted(glob()) ordering as load_split_data() (dataset.py
line 594) to guarantee alignment with the DataLoader iteration order.

Feature indices sourced exclusively from hft_contracts — no hardcoded
fallbacks. hft-contracts is a declared dependency (pyproject.toml).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import logging
import numpy as np

from hft_contracts import SIGNAL_SPREAD_FEATURE_INDEX, SIGNAL_PRICE_FEATURE_INDEX

logger = logging.getLogger(__name__)


@dataclass
class RawFeatures:
    """Raw (pre-normalization) ancillary features for signal export."""
    spreads: np.ndarray    # [N] float64 — raw spread_bps
    prices: np.ndarray     # [N] float64 — raw mid_price
    n_samples: int


class RawFeatureExtractor:
    """Extract raw spread and price from .npy export files on disk.

    Reads the last timestep of each sequence ([:, -1, index]) to get the
    raw feature values that correspond to each prediction sample.

    Args:
        data_dir: Root data directory containing train/val/test splits.
        split: Data split to extract from ("val" or "test").
    """

    def __init__(self, data_dir: Path, split: str):
        self._split_dir = Path(data_dir) / split
        if not self._split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self._split_dir}")

    def extract(self) -> RawFeatures:
        """Extract raw spread_bps and mid_price from all days in the split.

        Uses memory-mapped reading (mmap_mode='r') for efficiency.
        Files are sorted alphabetically to match DataLoader iteration order.

        Returns:
            RawFeatures with concatenated spreads, prices, and total count.

        Raises:
            FileNotFoundError: If no sequence files found.
        """
        seq_files = sorted(self._split_dir.glob("*_sequences.npy"))
        if not seq_files:
            raise FileNotFoundError(
                f"No *_sequences.npy files found in {self._split_dir}"
            )

        all_spreads: List[np.ndarray] = []
        all_prices: List[np.ndarray] = []
        total = 0

        for seq_file in seq_files:
            raw = np.load(seq_file, mmap_mode="r")

            # Extract last timestep for each sequence
            if raw.ndim == 3:
                # [N, T, F] — aligned sequences
                last_step = raw[:, -1, :]
            elif raw.ndim == 2:
                # [N, F] — flat features (legacy)
                last_step = raw
            else:
                raise ValueError(
                    f"Unexpected array shape {raw.shape} in {seq_file}"
                )

            n = last_step.shape[0]

            spreads = last_step[:, SIGNAL_SPREAD_FEATURE_INDEX].astype(np.float64)
            prices = last_step[:, SIGNAL_PRICE_FEATURE_INDEX].astype(np.float64)

            all_spreads.append(np.array(spreads))  # copy from mmap
            all_prices.append(np.array(prices))

            total += n
            del raw

        spreads_concat = np.concatenate(all_spreads)
        prices_concat = np.concatenate(all_prices)

        logger.info(
            f"Extracted raw features from {len(seq_files)} files: "
            f"{total:,} samples, spread mean={spreads_concat.mean():.2f} bps, "
            f"price mean={prices_concat.mean():.2f} USD"
        )

        return RawFeatures(
            spreads=spreads_concat,
            prices=prices_concat,
            n_samples=total,
        )
