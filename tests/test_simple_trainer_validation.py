"""C-1 regression tests for simple_trainer._load_split contract validation.

Phase O Cycle 1 consumer-side hardening (2026-05-04). Before these fixes,
TemporalRidge / TemporalGradBoost runs went through `_load_split`, which
called `np.load` directly with ZERO contract validation. Pre-Phase-O
exports (`schema_version="2.2"`) coexisted silently with v3.0 baseline
exports if both directories were ever mixed.

After C-1, `_load_split` invokes `_validate_day_metadata` per-day BEFORE
any numpy load, so schema mismatches raise ContractError up to the caller
(fail-fast at the boundary per hft-rules §8).

C-2 hardens missing-metadata and missing-schema-version branches —
those tests live in tests/test_dataset_strict_validation.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hft_contracts.validation import ContractError
from lobtrainer.training.simple_trainer import _load_split


NUM_SEQS = 8
SEQ_LEN = 20
NUM_FEATURES = 98
NUM_HORIZONS = 3


def _write_day(
    split_dir: Path,
    day: str,
    schema_version: str = "3.0",
    n: int = NUM_SEQS,
    rng: np.random.Generator | None = None,
) -> None:
    """Write a single synthetic day's worth of files into ``split_dir``."""
    if rng is None:
        rng = np.random.default_rng(42)

    sequences = rng.standard_normal((n, SEQ_LEN, NUM_FEATURES)).astype(np.float32)
    reg_labels = rng.standard_normal((n, NUM_HORIZONS)).astype(np.float64)

    np.save(split_dir / f"{day}_sequences.npy", sequences)
    np.save(split_dir / f"{day}_regression_labels.npy", reg_labels)

    metadata = {
        "day": day,
        "n_sequences": n,
        "n_features": NUM_FEATURES,
        "schema_version": schema_version,
    }
    with open(split_dir / f"{day}_metadata.json", "w") as f:
        json.dump(metadata, f)


@pytest.fixture
def split_dir_v3p0(tmp_path: Path) -> Path:
    """Two-day synthetic split with v3.0 metadata (clean baseline)."""
    train = tmp_path / "train"
    train.mkdir()
    _write_day(train, "20250203", schema_version="3.0")
    _write_day(train, "20250204", schema_version="3.0")
    return tmp_path


@pytest.fixture
def split_dir_legacy_v22(tmp_path: Path) -> Path:
    """Two-day synthetic split with pre-Phase-O v2.2 metadata."""
    train = tmp_path / "train"
    train.mkdir()
    _write_day(train, "20250203", schema_version="2.2")
    _write_day(train, "20250204", schema_version="2.2")
    return tmp_path


@pytest.fixture
def split_dir_mixed(tmp_path: Path) -> Path:
    """Three-day split: day 1 v3.0, day 2 v2.2 (corrupt), day 3 v3.0.

    Models the silent-mixing failure mode that motivated C-1: a day with
    stale schema slipping into an otherwise-clean directory.
    """
    train = tmp_path / "train"
    train.mkdir()
    _write_day(train, "20250203", schema_version="3.0")
    _write_day(train, "20250204", schema_version="2.2")
    _write_day(train, "20250205", schema_version="3.0")
    return tmp_path


class TestLoadSplitContractValidation:
    """C-1: _load_split must validate each day's metadata before np.load."""

    def test_v3p0_split_loads_cleanly(self, split_dir_v3p0: Path) -> None:
        """Baseline: well-formed v3.0 split passes validation and returns data."""
        seqs, labels, spreads, prices = _load_split(split_dir_v3p0, "train")
        assert seqs.shape[0] == 2 * NUM_SEQS
        assert labels.shape[0] == 2 * NUM_SEQS
        assert spreads.shape[0] == 2 * NUM_SEQS
        assert prices.shape[0] == 2 * NUM_SEQS

    def test_legacy_v22_split_raises_contract_error(
        self, split_dir_legacy_v22: Path
    ) -> None:
        """Pre-Phase-O v2.2 export must fail-loud against the current v3.0 contract."""
        with pytest.raises(ContractError, match=r"schema version"):
            _load_split(split_dir_legacy_v22, "train")

    def test_mixed_split_raises_on_first_bad_day(
        self, split_dir_mixed: Path
    ) -> None:
        """A single corrupt day in an otherwise-clean dir blocks the load."""
        with pytest.raises(ContractError, match=r"schema version"):
            _load_split(split_dir_mixed, "train")

    def test_validation_runs_before_numpy_load(
        self, split_dir_legacy_v22: Path
    ) -> None:
        """Sanity: deleting the sequences file does NOT change the failure mode.

        If validation runs BEFORE np.load, the schema-mismatch ContractError
        wins (not FileNotFoundError). This locks the fail-fast ordering.
        """
        train = split_dir_legacy_v22 / "train"
        for f in train.glob("*_sequences.npy"):
            f.unlink()
        with pytest.raises(ContractError, match=r"schema version"):
            _load_split(split_dir_legacy_v22, "train")
