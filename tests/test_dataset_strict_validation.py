"""C-2 / C-3 regression tests for fail-loud day-metadata validation.

Phase O Cycle 1 consumer-side hardening (2026-05-04). Before C-2,
``_validate_day_metadata`` silently returned on two paths:
  * ``metadata is None`` — partial export / NFS lag / legacy day with
    no metadata.json file.
  * ``"schema_version" not in metadata`` — pre-Phase-O legacy days that
    pre-date schema-version emission.

After C-2 both branches raise ``ContractError`` per hft-rules §8
("never silently drop, clamp, or fix data without recording diagnostics").

After C-3 the per-split validation in ``load_split_data`` runs for every
day rather than just the first day with metadata, so a corrupted day at
position 50 in a 233-day corpus is caught at load time.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hft_contracts.validation import ContractError
from lobtrainer.data.dataset import (
    _validate_day_metadata,
    load_split_data,
)


NUM_SEQS = 8
SEQ_LEN = 20
NUM_FEATURES = 98


def _write_day(
    split_dir: Path,
    day: str,
    *,
    schema_version: str | None = "3.0",
    write_metadata: bool = True,
    n: int = NUM_SEQS,
    rng: np.random.Generator | None = None,
) -> None:
    """Write a synthetic day to ``split_dir``.

    If ``schema_version`` is ``None`` the metadata is written without that
    key (used to test C-2's missing-key branch). If ``write_metadata`` is
    False the metadata.json file is omitted entirely (used to test C-2's
    None branch via the ``load_split_data`` chain).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    sequences = rng.standard_normal((n, SEQ_LEN, NUM_FEATURES)).astype(np.float32)
    reg_labels = rng.standard_normal((n, 3)).astype(np.float64)
    np.save(split_dir / f"{day}_sequences.npy", sequences)
    np.save(split_dir / f"{day}_regression_labels.npy", reg_labels)

    if write_metadata:
        meta: dict = {"day": day, "n_sequences": n, "n_features": NUM_FEATURES}
        if schema_version is not None:
            meta["schema_version"] = schema_version
        with open(split_dir / f"{day}_metadata.json", "w") as f:
            json.dump(meta, f)


# ---------------------------------------------------------------------------
# C-2 unit tests on _validate_day_metadata directly
# ---------------------------------------------------------------------------


class TestValidateDayMetadataFailLoud:
    """C-2: silent returns are now ContractError raises."""

    def test_none_metadata_raises(self) -> None:
        with pytest.raises(ContractError, match=r"missing or could not be loaded"):
            _validate_day_metadata(None, "20250203")

    def test_missing_schema_version_raises(self) -> None:
        meta = {"day": "20250203", "n_features": NUM_FEATURES, "n_sequences": NUM_SEQS}
        with pytest.raises(ContractError, match=r"no 'schema_version' field"):
            _validate_day_metadata(meta, "20250203")

    def test_v3p0_schema_passes(self) -> None:
        meta = {
            "day": "20250203",
            "n_features": NUM_FEATURES,
            "n_sequences": NUM_SEQS,
            "schema_version": "3.0",
        }
        _validate_day_metadata(meta, "20250203")

    def test_mismatched_schema_raises(self) -> None:
        meta = {
            "day": "20250203",
            "n_features": NUM_FEATURES,
            "n_sequences": NUM_SEQS,
            "schema_version": "2.2",
        }
        with pytest.raises(ContractError, match=r"schema version"):
            _validate_day_metadata(meta, "20250203")


# ---------------------------------------------------------------------------
# C-3 integration tests through load_split_data
# ---------------------------------------------------------------------------


@pytest.fixture
def split_dir_first_day_clean_others_bad(tmp_path: Path) -> Path:
    """5-day split where day 1 is v3.0 but day 3 has a bad schema."""
    train = tmp_path / "train"
    train.mkdir()
    _write_day(train, "20250203", schema_version="3.0")
    _write_day(train, "20250204", schema_version="3.0")
    _write_day(train, "20250205", schema_version="2.2")  # corrupt
    _write_day(train, "20250206", schema_version="3.0")
    _write_day(train, "20250207", schema_version="3.0")
    return tmp_path


@pytest.fixture
def split_dir_missing_metadata_at_day3(tmp_path: Path) -> Path:
    """5-day split where day 3 has NO metadata.json."""
    train = tmp_path / "train"
    train.mkdir()
    _write_day(train, "20250203", schema_version="3.0")
    _write_day(train, "20250204", schema_version="3.0")
    _write_day(train, "20250205", write_metadata=False)
    _write_day(train, "20250206", schema_version="3.0")
    _write_day(train, "20250207", schema_version="3.0")
    return tmp_path


class TestLoadSplitDataValidatesEveryDay:
    """C-3: every day's metadata is validated, not just day 1."""

    def test_corrupt_day3_in_clean_corpus_raises(
        self, split_dir_first_day_clean_others_bad: Path
    ) -> None:
        """Day-1-only validation would pass this corpus (day 1 is clean).

        After C-3 this raises with the corrupt day's date in the message.
        """
        with pytest.raises(ContractError) as excinfo:
            load_split_data(
                data_dir=split_dir_first_day_clean_others_bad,
                split="train",
                validate=False,
                lazy=False,
            )
        assert "20250205" in str(excinfo.value), (
            f"ContractError must name the corrupt day, got: {excinfo.value!r}"
        )

    def test_missing_metadata_day3_raises(
        self, split_dir_missing_metadata_at_day3: Path
    ) -> None:
        """Missing metadata.json on a non-first day fails-loud after C-2 + C-3."""
        with pytest.raises(ContractError) as excinfo:
            load_split_data(
                data_dir=split_dir_missing_metadata_at_day3,
                split="train",
                validate=False,
                lazy=False,
            )
        assert "20250205" in str(excinfo.value), (
            f"ContractError must name the missing-metadata day, got: {excinfo.value!r}"
        )
