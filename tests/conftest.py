"""
Shared test fixtures for lob-model-trainer tests.

Provides deterministic synthetic data factories matching the Rust feature
extractor's output format. All fixtures use np.random.default_rng(42) for
reproducibility (independent of legacy np.random.seed used in existing tests).

Fixture Hierarchy:
    rng                     -- deterministic Generator
    synthetic_lob_98        -- [T, 98] realistic LOB data
    day_data_factory        -- callable(date, ...) -> DayData
    train_days_98           -- 3 days of 98-feature training data
    synthetic_export_dir    -- full disk export with train/val splits
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pytest

from lobtrainer.data.dataset import DayData

# ---------------------------------------------------------------------------
# Constants matching the pipeline contract (Schema v2.2)
# ---------------------------------------------------------------------------

NUM_SEQS_DEFAULT = 50
SEQ_LEN_DEFAULT = 100
NUM_HORIZONS = 3
HORIZONS = [10, 60, 300]

# Grouped layout column ranges (pipeline default)
GROUPED_ASK_PRICE_COLS = list(range(0, 10))    # indices 0-9
GROUPED_ASK_SIZE_COLS = list(range(10, 20))     # indices 10-19
GROUPED_BID_PRICE_COLS = list(range(20, 30))    # indices 20-29
GROUPED_BID_SIZE_COLS = list(range(30, 40))     # indices 30-39
LOB_FEATURE_END = 40

# Non-normalizable indices (from hft_contracts)
EXCLUDE_INDICES = (92, 93, 94, 96, 97)
# 92=BOOK_VALID, 93=TIME_REGIME, 94=MBO_READY, 96=INVALIDITY_DELTA, 97=SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Core fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic numpy Generator for reproducible tests."""
    return np.random.default_rng(42)


def _make_synthetic_lob_data(
    rng: np.random.Generator,
    num_rows: int,
    num_features: int = 98,
    base_price: float = 130.0,
    spread_bps: float = 5.0,
) -> np.ndarray:
    """Create synthetic LOB data with realistic structure.

    Prices centered at base_price with small spread. Sizes drawn from
    lognormal. Derived features as standard normal. Categorical/gate
    features set to appropriate values.

    Args:
        rng: Numpy random Generator for reproducibility.
        num_rows: Number of sample rows (e.g., num_seqs * seq_len for 3D).
        num_features: Total feature count (40, 98, 116, or 148).
        base_price: Center price for LOB levels (default: 130.0 for NVDA).
        spread_bps: Spread in basis points between bid/ask (default: 5.0).
    """
    data = np.zeros((num_rows, num_features), dtype=np.float64)
    half_spread = base_price * spread_bps / 10000 / 2

    # LOB prices: ask prices slightly above base, bid slightly below
    if num_features >= 40:
        for level in range(10):
            tick_offset = level * 0.01  # 1 cent per level
            # Ask prices: base + half_spread + level offset + noise
            data[:, level] = base_price + half_spread + tick_offset + rng.normal(0, 0.005, num_rows)
            # Bid prices: base - half_spread - level offset + noise
            data[:, 20 + level] = base_price - half_spread - tick_offset + rng.normal(0, 0.005, num_rows)
            # Ask sizes: lognormal (positive, right-skewed)
            data[:, 10 + level] = rng.lognormal(mean=5.0, sigma=1.0, size=num_rows)
            # Bid sizes: lognormal
            data[:, 30 + level] = rng.lognormal(mean=5.0, sigma=1.0, size=num_rows)

    # Derived features (40-91): standard normal, various scales
    if num_features > LOB_FEATURE_END:
        derived_end = min(num_features, 92)
        n_derived = derived_end - LOB_FEATURE_END
        if n_derived > 0:
            data[:, LOB_FEATURE_END:derived_end] = rng.standard_normal((num_rows, n_derived))

    # Categorical / gate features
    if num_features > 92:
        data[:, 92] = 1.0      # BOOK_VALID: always valid
    if num_features > 93:
        data[:, 93] = rng.choice([0, 1, 2, 3, 4], size=num_rows)  # TIME_REGIME
    if num_features > 94:
        data[:, 94] = 1.0      # MBO_READY: always ready
    if num_features > 95:
        data[:, 95] = rng.uniform(0.01, 1.0, num_rows)  # DT_SECONDS
    if num_features > 96:
        data[:, 96] = 0.0      # INVALIDITY_DELTA: always valid
    if num_features > 97:
        data[:, 97] = 2.2      # SCHEMA_VERSION

    return data


@pytest.fixture
def synthetic_lob_98(rng) -> np.ndarray:
    """Synthetic 98-feature LOB data [500, 98] with realistic structure.

    Prices ~130 USD, sizes lognormal, derived features normal,
    categoricals set to appropriate values. 500 rows = 5 sequences x 100 timesteps.
    """
    return _make_synthetic_lob_data(rng, num_rows=500, num_features=98)


# ---------------------------------------------------------------------------
# DayData factory
# ---------------------------------------------------------------------------


def _make_metadata(
    date: str,
    num_seqs: int,
    num_features: int,
    seq_len: int = SEQ_LEN_DEFAULT,
    label_strategy: str = "tlob",
    feature_layout: str = "grouped",
) -> Dict:
    """Create realistic metadata matching Rust exporter output."""
    return {
        "day": date,
        "n_sequences": num_seqs,
        "window_size": seq_len,
        "n_features": num_features,
        "schema_version": "2.2",
        "contract_version": "2.2",
        "label_strategy": label_strategy,
        "label_dtype": "int8",
        "tensor_format": None,
        "labeling": {
            "label_mode": "classification",
            "horizons": HORIZONS,
            "num_horizons": NUM_HORIZONS,
            "strategy": label_strategy,
            "label_encoding": {
                "format": "signed",
                "dtype": "int8",
                "note": "class labels -1=Down, 0=Stable, 1=Up",
            },
        },
        "export_timestamp": "2026-04-01T12:00:00Z",
        "normalization": {
            "strategy": "none",
            "applied": False,
            "levels": 10,
            "sample_count": num_seqs * seq_len,
            "feature_layout": feature_layout,
            "params_file": f"{date}_normalization.json",
        },
        "provenance": {
            "extractor_version": "0.1.0",
            "git_commit": "test_fixture",
            "git_dirty": False,
            "config_hash": "test",
            "contract_version": "2.2",
            "export_timestamp_utc": "2026-04-01T12:00:00Z",
        },
        "validation": {
            "sequences_labels_match": True,
            "label_range_valid": True,
            "no_nan_inf": True,
        },
        "processing": {
            "messages_processed": 10000,
            "features_extracted": 5000,
            "sequences_generated": num_seqs + 5,
            "sequences_aligned": num_seqs,
            "sequences_dropped": 5,
        },
    }


@pytest.fixture
def day_data_factory(rng):
    """Factory fixture that creates DayData with controllable parameters.

    Usage:
        day = day_data_factory("2025-03-01")
        day = day_data_factory("2025-03-01", num_seqs=100, num_features=40)
    """
    def _create(
        date: str = "2025-03-01",
        num_seqs: int = NUM_SEQS_DEFAULT,
        num_features: int = 98,
        seq_len: int = SEQ_LEN_DEFAULT,
        label_strategy: str = "tlob",
        feature_layout: str = "grouped",
        include_regression_labels: bool = False,
        metadata_overrides: Optional[Dict] = None,
    ) -> DayData:
        # Create reproducible data per date
        date_rng = np.random.default_rng(hash(date) % 2**31)

        # Sequences [N, T, F]
        flat_data = _make_synthetic_lob_data(date_rng, num_seqs * seq_len, num_features)
        sequences = flat_data.reshape(num_seqs, seq_len, num_features).astype(np.float32)

        # Features [N, F] — last timestep of each sequence
        features = sequences[:, -1, :].copy()

        # Classification labels {-1, 0, 1}
        labels = date_rng.choice([-1, 0, 1], size=num_seqs).astype(np.int8)

        # Optional regression labels [N, H] in bps
        regression_labels = None
        if include_regression_labels:
            regression_labels = date_rng.standard_normal((num_seqs, NUM_HORIZONS)).astype(np.float64) * 5.0

        # Metadata
        metadata = _make_metadata(date, num_seqs, num_features, seq_len, label_strategy, feature_layout)
        if metadata_overrides:
            metadata.update(metadata_overrides)

        return DayData(
            date=date,
            features=features,
            labels=labels,
            sequences=sequences,
            regression_labels=regression_labels,
            metadata=metadata,
            is_aligned=True,
        )

    return _create


@pytest.fixture
def train_days_98(day_data_factory) -> List[DayData]:
    """3 days of 98-feature training data for normalization testing."""
    return [
        day_data_factory("2025-03-01"),
        day_data_factory("2025-03-02"),
        day_data_factory("2025-03-03"),
    ]


# ---------------------------------------------------------------------------
# Disk-based export fixtures
# ---------------------------------------------------------------------------


def _write_export_to_disk(
    base_dir: Path,
    day_data_factory,
    split_dates: Dict[str, List[str]],
    num_features: int = 98,
    label_strategy: str = "tlob",
    include_regression_labels: bool = False,
):
    """Write synthetic export files matching Rust exporter output format."""
    for split, dates in split_dates.items():
        split_dir = base_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        for date in dates:
            day = day_data_factory(
                date,
                num_features=num_features,
                label_strategy=label_strategy,
                include_regression_labels=include_regression_labels,
            )
            np.save(split_dir / f"{date}_sequences.npy", day.sequences)
            np.save(split_dir / f"{date}_labels.npy", day.labels)
            if day.regression_labels is not None:
                np.save(split_dir / f"{date}_regression_labels.npy", day.regression_labels)
            with open(split_dir / f"{date}_metadata.json", "w") as f:
                json.dump(day.metadata, f)


@pytest.fixture
def synthetic_export_dir(tmp_path, day_data_factory) -> Path:
    """Full export directory with train/val splits on disk.

    Structure:
        tmp_path/
            train/
                2025-03-01_sequences.npy, _labels.npy, _metadata.json
                2025-03-02_sequences.npy, _labels.npy, _metadata.json
            val/
                2025-03-03_sequences.npy, _labels.npy, _metadata.json
    """
    _write_export_to_disk(
        base_dir=tmp_path,
        day_data_factory=day_data_factory,
        split_dates={
            "train": ["2025-03-01", "2025-03-02"],
            "val": ["2025-03-03"],
        },
    )
    return tmp_path
