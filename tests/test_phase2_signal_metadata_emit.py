"""Phase II end-to-end: trainer emits CompatibilityContract → SignalManifest parses it.

Locks the producer→consumer round-trip for the CompatibilityContract block emitted
by ``build_signal_metadata`` and consumed by ``hft_contracts.SignalManifest``.

Covered invariants (Phase II, 2026-04-20):
    - ``compatibility`` + ``compatibility_fingerprint`` emitted when contract is provided.
    - ``calibration_method`` emitted at top-level for manifest-authoritative gate.
    - ``data_source`` emitted at top-level.
    - Absent when ``compatibility`` is None (pre-Phase-II callers unaffected).
    - Fingerprint recomputation on the parsed manifest matches producer fingerprint.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("hft_contracts")
from hft_contracts.compatibility import CompatibilityContract
from hft_contracts.signal_manifest import SignalManifest

from lobtrainer.export.metadata import build_signal_metadata


def _fixture_contract(**overrides) -> CompatibilityContract:
    defaults = dict(
        contract_version="2.2",
        schema_version="2.2",
        feature_count=98,
        window_size=100,
        feature_layout="default",
        data_source="mbo_lob",
        label_strategy_hash="a" * 64,
        calibration_method=None,
        primary_horizon_idx=0,
        horizons=(10, 60, 300),
        normalization_strategy="none",
    )
    defaults.update(overrides)
    return CompatibilityContract(**defaults)


class TestEmitCompatibilityBlock:
    def test_compatibility_block_present_when_contract_provided(self):
        contract = _fixture_contract()
        meta = build_signal_metadata(
            model_type="hmhp",
            model_name="hmhp",
            parameters=171000,
            signal_type="classification",
            split="test",
            total_samples=1000,
            checkpoint="/tmp/best.pt",
            compatibility=contract,
            data_source=contract.data_source,
            calibration_method=contract.calibration_method,
        )
        assert "compatibility" in meta
        assert "compatibility_fingerprint" in meta
        assert meta["compatibility_fingerprint"] == contract.fingerprint()
        assert meta["data_source"] == "mbo_lob"
        # calibration_method=None is NOT emitted (consistent with other optional fields)
        assert "calibration_method" not in meta

    def test_calibration_method_emitted_when_set(self):
        contract = _fixture_contract(calibration_method="variance_match")
        meta = build_signal_metadata(
            model_type="hmhp_regressor",
            model_name="hmhp_regressor",
            parameters=100000,
            signal_type="regression",
            split="test",
            total_samples=500,
            checkpoint="/tmp/best.pt",
            compatibility=contract,
            data_source=contract.data_source,
            calibration_method=contract.calibration_method,
        )
        assert meta["calibration_method"] == "variance_match"
        assert meta["compatibility"]["calibration_method"] == "variance_match"

    def test_compatibility_absent_when_not_provided(self):
        """Pre-Phase-II callers (no compatibility arg) get no new fields."""
        meta = build_signal_metadata(
            model_type="tlob",
            model_name="tlob",
            parameters=100000,
            signal_type="classification",
            split="test",
            total_samples=500,
            checkpoint="/tmp/best.pt",
        )
        assert "compatibility" not in meta
        assert "compatibility_fingerprint" not in meta
        assert "calibration_method" not in meta
        assert "data_source" not in meta

    def test_horizons_serialize_as_list_not_tuple(self):
        """JSON-native type for horizons — tuple → list for canonical form parity."""
        contract = _fixture_contract(horizons=(5, 30, 120))
        meta = build_signal_metadata(
            model_type="hmhp",
            model_name="hmhp",
            parameters=1,
            signal_type="classification",
            split="test",
            total_samples=1,
            checkpoint="/tmp/x",
            compatibility=contract,
        )
        assert meta["compatibility"]["horizons"] == [5, 30, 120]
        assert isinstance(meta["compatibility"]["horizons"], list)


class TestProducerConsumerRoundTrip:
    """Emit via trainer → parse via SignalManifest → validate succeeds."""

    def test_round_trip_fingerprint_matches(self, tmp_path: Path):
        contract = _fixture_contract()
        meta = build_signal_metadata(
            model_type="hmhp",
            model_name="hmhp",
            parameters=1,
            signal_type="classification",
            split="test",
            total_samples=10,
            checkpoint="/tmp/x",
            horizons=list(contract.horizons),
            compatibility=contract,
            data_source=contract.data_source,
            calibration_method=contract.calibration_method,
        )
        # Serialize as the trainer does.
        (tmp_path / "signal_metadata.json").write_text(json.dumps(meta, sort_keys=True))

        # Write minimum NPYs so SignalManifest.from_signal_dir doesn't skip.
        import numpy as np

        np.save(tmp_path / "prices.npy", np.abs(np.random.RandomState(0).randn(10)) + 100.0)
        np.save(tmp_path / "predictions.npy", np.random.RandomState(1).randint(0, 3, 10).astype(np.int32))
        np.save(tmp_path / "labels.npy", np.random.RandomState(2).randint(0, 3, 10).astype(np.int32))

        manifest = SignalManifest.from_signal_dir(tmp_path)
        # Producer fingerprint roundtrip: manifest parses block + fingerprint, recomputation matches.
        assert manifest.compatibility is not None
        assert manifest.compatibility.fingerprint() == contract.fingerprint()
        assert manifest.compatibility_fingerprint == contract.fingerprint()

        # validate() runs the tamper detection + calibration precedence — should pass.
        warnings = manifest.validate(tmp_path)
        # No legacy-manifest warning since we emitted the block.
        assert not any("Legacy signal manifest" in w for w in warnings)

    def test_round_trip_expected_contract_match(self, tmp_path: Path):
        """Consumer can validate against its own expected contract."""
        producer = _fixture_contract()
        consumer = _fixture_contract()  # identical
        meta = build_signal_metadata(
            model_type="hmhp",
            model_name="hmhp",
            parameters=1,
            signal_type="classification",
            split="test",
            total_samples=10,
            checkpoint="/tmp/x",
            horizons=list(producer.horizons),
            compatibility=producer,
            data_source=producer.data_source,
        )
        (tmp_path / "signal_metadata.json").write_text(json.dumps(meta, sort_keys=True))
        import numpy as np
        np.save(tmp_path / "prices.npy", np.abs(np.random.RandomState(0).randn(10)) + 100.0)
        np.save(tmp_path / "predictions.npy", np.random.RandomState(1).randint(0, 3, 10).astype(np.int32))
        np.save(tmp_path / "labels.npy", np.random.RandomState(2).randint(0, 3, 10).astype(np.int32))

        manifest = SignalManifest.from_signal_dir(tmp_path)
        manifest.validate(tmp_path, expected_contract=consumer)  # no raise

    def test_round_trip_expected_contract_mismatch_raises(self, tmp_path: Path):
        """Consumer expects different contract → ContractError with diff."""
        from hft_contracts.validation import ContractError

        producer = _fixture_contract(feature_count=98)
        consumer = _fixture_contract(feature_count=148)
        meta = build_signal_metadata(
            model_type="hmhp",
            model_name="hmhp",
            parameters=1,
            signal_type="classification",
            split="test",
            total_samples=10,
            checkpoint="/tmp/x",
            compatibility=producer,
            data_source=producer.data_source,
        )
        (tmp_path / "signal_metadata.json").write_text(json.dumps(meta, sort_keys=True))
        import numpy as np
        np.save(tmp_path / "prices.npy", np.abs(np.random.RandomState(0).randn(10)) + 100.0)
        np.save(tmp_path / "predictions.npy", np.random.RandomState(1).randint(0, 3, 10).astype(np.int32))
        np.save(tmp_path / "labels.npy", np.random.RandomState(2).randint(0, 3, 10).astype(np.int32))

        manifest = SignalManifest.from_signal_dir(tmp_path)
        with pytest.raises(ContractError, match=r"feature_count"):
            manifest.validate(tmp_path, expected_contract=consumer)
