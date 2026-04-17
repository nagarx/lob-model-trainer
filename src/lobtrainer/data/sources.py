"""Multi-source data abstractions (T12).

Defines the building blocks for loading and fusing features from multiple
data sources (e.g., MBO order book + BASIC off-exchange trades).

Each source produces its own sequences and features with potentially
different feature counts, window sizes, and sequence counts per day.
The DayBundle (in bundle.py) aligns and fuses these into a standard
DayData for the training pipeline.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass(frozen=True)
class DataSource:
    """Configuration for one data source in a multi-source pipeline.

    At least one source must have role='primary'. The primary source
    provides labels, forward_prices, and the label-computation contract.
    Auxiliary sources contribute features only.

    Args:
        name: Unique identifier (e.g., "mbo", "basic", "arcx_mbo", "opra").
        data_dir: Path to the export directory containing train/val/test splits.
        role: "primary" (labels + features) or "auxiliary" (features only).
    """

    name: str
    data_dir: str
    role: str = "primary"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("DataSource.name must be non-empty")
        if not self.data_dir:
            raise ValueError("DataSource.data_dir must be non-empty")
        if self.role not in ("primary", "auxiliary"):
            raise ValueError(
                f"DataSource.role must be 'primary' or 'auxiliary', "
                f"got {self.role!r}"
            )


@dataclass
class SourceDay:
    """One source's data for a single trading day.

    This is the per-source building block. A DayBundle holds multiple
    SourceDay objects (one per source) for the same calendar date.

    Fields:
        name: Source identifier matching DataSource.name.
        date: Normalized date string (YYYYMMDD format).
        sequences: [N, T, F] float64 — full 3D sequences.
        features: [N, F] float64 — last timestep of each sequence.
        metadata: Parsed per-day metadata JSON.
        n_features: Feature count F (from metadata or array shape).
        window_size: Temporal window T (from metadata or array shape).
    """

    name: str
    date: str
    sequences: np.ndarray
    features: np.ndarray
    metadata: Optional[Dict] = None
    n_features: int = 0
    window_size: int = 0

    def __post_init__(self) -> None:
        if self.sequences is not None and self.sequences.ndim == 3:
            if self.n_features == 0:
                self.n_features = self.sequences.shape[2]
            if self.window_size == 0:
                self.window_size = self.sequences.shape[1]

    @property
    def n_sequences(self) -> int:
        """Number of sequences in this source for this day."""
        return self.sequences.shape[0] if self.sequences is not None else 0


def normalize_date(date_str: str) -> str:
    """Normalize date string to compact YYYYMMDD format.

    Handles both YYYY-MM-DD (BASIC exports) and YYYYMMDD (MBO exports).
    """
    return date_str.replace("-", "")
