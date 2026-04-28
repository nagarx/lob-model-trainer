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
        contract_version="3.0",
        schema_version="3.0",
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


class TestCompatibilityContractConstruction:
    """Phase A (2026-04-23): producer-path regression tests for ``_build_compatibility_contract``.

    Historically (Phase II shipped 2026-04-20 through Phase A shipped 2026-04-23),
    this file tested ``build_signal_metadata`` and ``SignalManifest`` with
    hand-constructed fixture ``CompatibilityContract`` objects — bypassing the
    producer path entirely. Phase A's 4-agent adversarial validation identified
    this as the reason P0-1 (producer silently returns None on every
    ``ExperimentConfig``) shipped undetected.

    This class closes that gap by exercising ``_build_compatibility_contract()``
    directly with a real ``ExperimentConfig``. Complements
    ``tests/test_signal_exporter_producer_roundtrip.py`` — the integration-level
    counterpart lives there; this class covers unit-level invariants.
    """

    def test_direct_call_from_real_experiment_config_returns_valid_contract(self) -> None:
        """Smoke: ``_build_compatibility_contract(ExperimentConfig(), ...)`` works.

        Pre-C1, this returned None silently because ``config.labels`` raised
        AttributeError inside the broad-catch. Post-C1, the resolve_labels_config
        helper sources from the canonical ``config.data.labels`` path.
        """
        from lobtrainer.config.schema import ExperimentConfig
        from lobtrainer.export.exporter import _build_compatibility_contract

        contract = _build_compatibility_contract(
            config=ExperimentConfig(),
            feature_set_ref=None,
            calibration_method=None,
        )
        assert contract is not None, (
            "Producer silently returned None — path-drift bug class regressed"
        )
        # Sanity: fingerprint is a 64-char lowercase hex SHA-256
        import re as _re
        assert _re.fullmatch(r"[0-9a-f]{64}", contract.fingerprint())

    def test_window_size_sourced_from_sequence_config(self) -> None:
        """C1 adjacent-bug fix: ``config.data.sequence.window_size`` is canonical.

        Pre-C1 path ``config.data.window_size`` didn't exist on DataConfig
        (window_size lives on the nested ``sequence`` subconfig). getattr
        fallback chain returned 0 — masked by the broad-catch silent-None.
        Post-C1, the correct path is probed first.
        """
        from lobtrainer.config.schema import (
            DataConfig,
            ExperimentConfig,
            LabelsConfig,
            SequenceConfig,
        )
        from lobtrainer.export.exporter import _build_compatibility_contract

        config = ExperimentConfig(
            data=DataConfig(
                feature_count=98,
                sequence=SequenceConfig(window_size=50),  # non-default
                labels=LabelsConfig(
                    horizons=[10], primary_horizon_idx=0, task="classification"
                ),
            ),
        )
        contract = _build_compatibility_contract(
            config=config, feature_set_ref=None, calibration_method=None
        )
        assert contract is not None
        assert contract.window_size == 50, (
            f"Expected window_size=50 from config.data.sequence.window_size; "
            f"got {contract.window_size}. Suggests the getattr chain fell through "
            f"to the (non-existent) DataConfig.window_size and returned 0."
        )

    def test_horizons_sourced_from_labels_config_not_model_hmhp(self) -> None:
        """Horizons come from ``config.data.labels.horizons`` (preferred).

        Pre-C1, the fallback read ``config.labels.horizons`` (AttributeError)
        → broad catch → silent None → hmhp_horizons fallback. Post-C1, the
        helper routes through the canonical path.
        """
        from lobtrainer.config.schema import (
            DataConfig,
            ExperimentConfig,
            LabelsConfig,
        )
        from lobtrainer.export.exporter import _build_compatibility_contract

        config = ExperimentConfig(
            data=DataConfig(
                feature_count=98,
                labels=LabelsConfig(
                    horizons=[10, 60, 300],
                    primary_horizon_idx=2,
                    task="classification",
                ),
            ),
        )
        contract = _build_compatibility_contract(
            config=config, feature_set_ref=None, calibration_method=None
        )
        assert contract is not None
        assert contract.horizons == (10, 60, 300)
        assert contract.primary_horizon_idx == 2  # not hardcoded 0 / legacy field

    def test_raises_attribute_error_on_broken_config_surface(self) -> None:
        """The helper raises loudly when the canonical path is missing.

        Pre-C1, the inner ``try / except Exception`` swallowed this as
        silent None. Post-C1, the caller must fix their config or receive
        a diagnostic error.
        """
        from unittest.mock import MagicMock

        from lobtrainer.export.exporter import _build_compatibility_contract

        broken = MagicMock(spec=[])  # no attributes at all
        with pytest.raises(AttributeError, match="resolve_labels_config"):
            _build_compatibility_contract(
                config=broken, feature_set_ref=None, calibration_method=None
            )

    def test_returns_none_when_hft_contracts_import_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Outer ImportError guard preserved — pre-Phase-II venv degrades gracefully.

        The C1 rewrite kept the outer ``try: from hft_contracts import ... except``
        block as a legitimate soft-dep guard. Verify that path still returns None.
        """
        import builtins

        from lobtrainer.config.schema import ExperimentConfig
        from lobtrainer.export.exporter import _build_compatibility_contract

        real_import = builtins.__import__

        def _raise_on_hft_contracts(name, *args, **kwargs):
            if name.startswith("hft_contracts"):
                raise ImportError(
                    f"Simulated pre-Phase-II venv (no hft_contracts): {name}"
                )
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _raise_on_hft_contracts)
        result = _build_compatibility_contract(
            config=ExperimentConfig(),
            feature_set_ref=None,
            calibration_method=None,
        )
        assert result is None, (
            "ImportError soft-dep guard regressed — pre-Phase-II venvs will break."
        )

    def test_exporter_never_directly_accesses_config_labels(self) -> None:
        """Regression-proof: lock the canonical-path-drift bug class out.

        An AST-level static check over ``exporter.py`` — fails if any future
        edit reintroduces ``config.labels.*`` (direct, non-helper access).
        All LabelsConfig reads MUST route through ``resolve_labels_config``.
        Accepts ``config.data.labels.*`` (canonical) and
        ``config.model.*`` / ``config.train.*`` (unrelated).
        """
        import ast
        from pathlib import Path

        exporter_src = (
            Path(__file__).resolve().parent.parent
            / "src"
            / "lobtrainer"
            / "export"
            / "exporter.py"
        )
        tree = ast.parse(exporter_src.read_text())
        violations: list[str] = []
        for node in ast.walk(tree):
            # Pattern: Attribute(value=Attribute(value=Name("config"), attr="labels"), attr=X)
            # i.e., `config.labels.X`
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "labels"
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "config"
            ):
                violations.append(
                    f"exporter.py:{node.lineno} accesses `config.labels.{node.attr}` "
                    f"directly — route through `resolve_labels_config(config)` "
                    f"instead (retires the Phase II producer-path bug class)."
                )
            # Also flag the `compute_label_strategy_hash(config.labels)` pattern.
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "labels"
                and isinstance(node.value, ast.Name)
                and node.value.id == "config"
            ):
                # This is bare `config.labels` (possibly as argument to a call).
                # Legit case: none — canonical is `config.data.labels`.
                violations.append(
                    f"exporter.py:{node.lineno} references bare `config.labels` — "
                    f"route through `resolve_labels_config(config)` instead."
                )
        assert not violations, (
            "exporter.py reintroduced canonical-path-drift risk:\n  - "
            + "\n  - ".join(violations)
        )
