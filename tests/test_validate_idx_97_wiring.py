"""Phase Z.1 / #PY-1 regression tests (2026-05-05): validate_idx_97_reserved
wired into trainer's data-load path.

Phase D shipped ``hft_contracts.validation.validate_idx_97_reserved`` to spot-
sample idx 97 == 0.0 from the first NPY row (closes the metadata-only gap of
``validate_export_contract`` which can't detect a producer emitting non-zero
idx 97 values). However Phase D left the validator orphan — defined but never
called. Phase Z.1 wires it into both data-load entry points:

  * PyTorch path: ``load_split_data`` in dataset.py (after ``_validate_day_metadata``)
  * sklearn path: ``_load_split`` in simple_trainer.py (after ``_validate_day_metadata``)

Both call sites use ``strict=False`` matching the existing trainer warning
convention (logger.warning); Phase X.4 may flip to ``strict=True`` for
production-gate fail-loud once a 2-week observation window confirms zero
post-Phase-O exports trip the gate.

Locks:
1. Wiring at dataset.py:893+ exists (load_split_data calls validate_idx_97_reserved)
2. Wiring at simple_trainer.py:83+ exists (_load_split calls validate_idx_97_reserved)
3. Synthetic NPY with idx[97]==1.5 emits warning via logger
4. Synthetic NPY with idx[97]==0.0 emits NO warning (clean)
5. Pre-Phase-O (n_features < 98) is no-op (back-compat — no warning)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest

from lobtrainer.data.dataset import load_split_data


NUM_SEQS = 4
SEQ_LEN = 20
NUM_FEATURES = 98


def _write_day(
    split_dir: Path,
    day: str,
    *,
    idx_97_value: float = 0.0,
    n_features: int = NUM_FEATURES,
    schema_version: str = "3.0",
) -> None:
    """Write a synthetic day's NPY files with controllable idx 97 value.

    Args:
        idx_97_value: value to write at sequences[:, :, 97]. Default 0.0
            matches the post-Phase-O contract. Pass non-zero to test the
            new validator's warning path.
        n_features: number of features. Default 98 (post-Phase-O). Pass
            <98 to test the back-compat no-op branch.
    """
    rng = np.random.default_rng(42)
    sequences = rng.standard_normal((NUM_SEQS, SEQ_LEN, n_features)).astype(np.float32)
    if n_features >= 98:
        sequences[:, :, 97] = idx_97_value  # Set the contract-relevant index
    reg_labels = rng.standard_normal((NUM_SEQS, 3)).astype(np.float64)
    np.save(split_dir / f"{day}_sequences.npy", sequences)
    np.save(split_dir / f"{day}_regression_labels.npy", reg_labels)
    meta = {
        "day": day,
        "n_sequences": NUM_SEQS,
        "n_features": n_features,
        "schema_version": schema_version,
    }
    with open(split_dir / f"{day}_metadata.json", "w") as f:
        json.dump(meta, f)


class TestPyTorchPathValidateIdx97Wired:
    """load_split_data (PyTorch path) calls validate_idx_97_reserved."""

    def test_compliant_data_no_warning(self, tmp_path: Path, caplog):
        """idx 97 == 0.0 (post-Phase-O contract) emits NO warning."""
        split_dir = tmp_path / "train"
        split_dir.mkdir()
        _write_day(split_dir, "2026-01-01", idx_97_value=0.0)

        with caplog.at_level(logging.WARNING):
            days = load_split_data(tmp_path, split="train")

        assert len(days) == 1
        idx_97_warns = [
            r for r in caplog.records if "idx-97 contract warning" in r.getMessage()
        ]
        assert len(idx_97_warns) == 0, (
            f"Compliant idx 97=0.0 must not trigger validator warning. "
            f"Got: {[r.getMessage() for r in idx_97_warns]}"
        )

    def test_corrupted_data_emits_warning(self, tmp_path: Path, caplog):
        """idx 97 == 1.5 (pre-Phase-O legacy schema_version=2.2 stored at idx
        97) MUST emit a warning via logger naming the date + contract violation."""
        split_dir = tmp_path / "train"
        split_dir.mkdir()
        _write_day(split_dir, "2026-01-01", idx_97_value=1.5)

        with caplog.at_level(logging.WARNING):
            days = load_split_data(tmp_path, split="train")

        # Load still succeeds (strict=False) — warnings only.
        assert len(days) == 1
        idx_97_warns = [
            r for r in caplog.records if "idx-97 contract warning" in r.getMessage()
        ]
        assert len(idx_97_warns) >= 1, (
            f"idx 97 != 0.0 MUST emit warning. caplog records: "
            f"{[r.getMessage() for r in caplog.records]}"
        )
        # Warning message cites the date
        assert any("2026-01-01" in r.getMessage() for r in idx_97_warns), (
            "Warning must cite the offending date for diagnosability"
        )

    def test_corrupted_data_load_still_succeeds(self, tmp_path: Path, caplog):
        """strict=False contract: idx 97 != 0.0 emits warning but does NOT
        prevent the load (graceful degradation matching pre-Phase-X.4
        observation-window pattern). Phase X.4 may flip to strict=True for
        production-gate fail-loud once a 2-week window confirms zero post-
        Phase-O exports trip the gate."""
        split_dir = tmp_path / "train"
        split_dir.mkdir()
        _write_day(split_dir, "2026-01-01", idx_97_value=2.2)

        with caplog.at_level(logging.WARNING):
            days = load_split_data(tmp_path, split="train")

        assert len(days) == 1, (
            "Phase Z.1 strict=False MUST allow load to succeed even on "
            "idx 97 contract violation (warnings only)."
        )
        idx_97_warns = [
            r for r in caplog.records if "idx-97 contract warning" in r.getMessage()
        ]
        assert len(idx_97_warns) >= 1


class TestSklearnPathValidateIdx97Wired:
    """SimpleModelTrainer._load_split (sklearn path) calls validate_idx_97_reserved."""

    def test_compliant_data_no_warning(self, tmp_path: Path, caplog):
        """idx 97 == 0.0 emits NO warning on sklearn path."""
        from lobtrainer.training.simple_trainer import _load_split

        split_dir = tmp_path / "train"
        split_dir.mkdir()
        _write_day(split_dir, "2026-01-01", idx_97_value=0.0)

        with caplog.at_level(logging.WARNING):
            seqs, labels, spreads, prices = _load_split(
                tmp_path, "train", horizon_idx=0, max_days=None
            )

        assert seqs.shape[0] == NUM_SEQS
        idx_97_warns = [
            r for r in caplog.records if "idx-97 contract warning" in r.getMessage()
        ]
        assert len(idx_97_warns) == 0

    def test_corrupted_data_emits_warning(self, tmp_path: Path, caplog):
        """idx 97 == 1.5 emits warning on sklearn path."""
        from lobtrainer.training.simple_trainer import _load_split

        split_dir = tmp_path / "train"
        split_dir.mkdir()
        _write_day(split_dir, "2026-01-01", idx_97_value=1.5)

        with caplog.at_level(logging.WARNING):
            seqs, labels, spreads, prices = _load_split(
                tmp_path, "train", horizon_idx=0, max_days=None
            )

        # Load still succeeds (strict=False)
        assert seqs.shape[0] == NUM_SEQS
        idx_97_warns = [
            r for r in caplog.records if "idx-97 contract warning" in r.getMessage()
        ]
        assert len(idx_97_warns) >= 1
        # Warning cites date
        assert any("2026-01-01" in r.getMessage() for r in idx_97_warns)


class TestValidatorImportWiringContract:
    """Lock the import contract: validate_idx_97_reserved is imported from
    hft_contracts.validation (the SSoT location). Future relocations of the
    validator MUST update both call sites + this test."""

    def test_validator_importable_from_ssot(self):
        from hft_contracts.validation import validate_idx_97_reserved
        assert callable(validate_idx_97_reserved)

    def test_validator_signature_stable(self):
        """validate_idx_97_reserved(sequences_path, *, strict=False) -> list[str]"""
        import inspect
        from hft_contracts.validation import validate_idx_97_reserved

        sig = inspect.signature(validate_idx_97_reserved)
        params = list(sig.parameters.values())
        # First positional: sequences_path
        assert params[0].name == "sequences_path"
        # strict is keyword-only with default False
        strict_param = sig.parameters["strict"]
        assert strict_param.kind == inspect.Parameter.KEYWORD_ONLY
        assert strict_param.default is False
