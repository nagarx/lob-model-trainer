"""Phase Q.6.5.A SSoT tests for ``SimpleModelTrainer.export_signals``
emitting Phase II ``compatibility`` block + ``compatibility_fingerprint``
+ Phase 4c.4 ``feature_set_ref`` + Phase II top-level ``data_source``.

Closes F-18 (sklearn signal_metadata.json missing compatibility block —
HARD prereq for Phase Y ``experiment_provenance_hash`` composition for
sklearn experiments).

Producer-side architectural pattern: extends Phase X.1.A's lifted
``build_compatibility_contract`` SSoT to the sklearn signal_metadata
producer. The PyTorch path's ``SignalExporter._build_metadata`` already
produces the same surfaces (exporter.py:638-700) — these tests lock the
sklearn path at parity with the PyTorch wire format.

Locks:
- compatibility block + fingerprint emission via ``from_config`` path
- top-level ``data_source`` field (legacy consumers without nested-block
  parsing read this directly per metadata.py:166-167)
- back-compat for legacy flat-keyword construction (``self.config = None``)
- 11-field ``CompatibilityContract`` canonical structure
- 64-hex SHA-256 ``compatibility_fingerprint`` format
- ``feature_set_ref`` propagation when ``_feature_set_ref_resolved`` cache
  is populated by the trainer's resolver
- determinism — same config → same fingerprint across runs
- top-level + nested ``contract_version`` parity (Phase Q.9 invariant)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from lobtrainer.training.simple_trainer import SimpleModelTrainer


# =============================================================================
# Constants — keep in sync with test_simple_trainer.py for fixture parity
# =============================================================================

NUM_SEQS = 20
SEQ_LEN = 100
NUM_FEATURES = 98
NUM_HORIZONS = 3
HORIZONS = [10, 60, 300]

# 64-hex SHA-256 fingerprint regex (mirrors hft_contracts.signal_manifest.CONTENT_HASH_RE)
SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def synthetic_data_dir(tmp_path):
    """Create synthetic .npy data + metadata.json matching the v3p0 contract.

    Mirrors the pattern from test_simple_trainer.py::simple_data_dir but
    with schema_version=3.0 + horizons=[10,60,300] so that
    ``_validate_day_metadata`` (now Phase X.2.A.2 SSoT shim → hft_contracts)
    accepts the metadata.
    """
    rng = np.random.default_rng(42)

    for split in ["train", "val", "test"]:
        split_dir = tmp_path / split
        split_dir.mkdir()
        for i, day in enumerate(["2025-01-01", "2025-01-02"]):
            n = NUM_SEQS + i * 5
            sequences = rng.standard_normal((n, SEQ_LEN, NUM_FEATURES)).astype(np.float32)
            sequences[:, -1, 40] = 130.0 + rng.standard_normal(n).astype(np.float32) * 0.5
            sequences[:, -1, 42] = 2.5 + rng.standard_normal(n).astype(np.float32) * 0.1
            reg_labels = rng.standard_normal((n, NUM_HORIZONS)).astype(np.float64)
            np.save(split_dir / f"{day}_sequences.npy", sequences)
            np.save(split_dir / f"{day}_regression_labels.npy", reg_labels)
            metadata = {
                "day": day,
                "n_sequences": n,
                "n_features": NUM_FEATURES,
                "schema_version": "3.0",  # Phase G G.6.A bump
            }
            with open(split_dir / f"{day}_metadata.json", "w") as f:
                json.dump(metadata, f)
    return tmp_path


def _build_synthetic_config(synthetic_data_dir, tmp_path, *, params=None):
    """Construct an ExperimentConfig pointing at the synthetic data dir.

    Mirrors test_simple_trainer.py::TestFromConfig._build_synthetic_config
    but as a module-level helper so it can be reused across this file's
    test classes.
    """
    from lobtrainer.config.schema import (
        DataConfig,
        ExperimentConfig,
        LabelsConfig,
        LossType,
        ModelConfig,
        ModelType,
        NormalizationConfig,
        SequenceConfig,
        TaskType,
        TrainConfig,
    )
    if params is None:
        params = {"alpha": 1.0}
    data = DataConfig(
        data_dir=str(synthetic_data_dir),
        feature_count=NUM_FEATURES,
        sequence=SequenceConfig(window_size=SEQ_LEN, stride=1),
        normalization=NormalizationConfig(strategy="none"),
        labels=LabelsConfig(
            primary_horizon_idx=0,
            horizons=HORIZONS,
            source="forward_prices",
            task="regression",
        ),
    )
    model = ModelConfig(
        model_type=ModelType.TEMPORAL_RIDGE,
        input_size=NUM_FEATURES,
        params=params,
    )
    train = TrainConfig(task_type=TaskType.REGRESSION, loss_type=LossType.HUBER)
    return ExperimentConfig(
        name="phase_q65a_test",
        data=data,
        model=model,
        train=train,
        output_dir=str(tmp_path / "output"),
    )


def _train_export_via_from_config(synthetic_data_dir, tmp_path):
    """Helper: full setup → train → export_signals via canonical from_config path.

    Returns ``(trainer, signal_dir, signal_metadata_dict)``.
    """
    config = _build_synthetic_config(synthetic_data_dir, tmp_path)
    trainer = SimpleModelTrainer.from_config(config)
    trainer.setup()
    trainer.train()
    trainer.evaluate("test")  # populates self.test_metrics for metrics= field
    signal_dir = trainer.export_signals("test")
    with open(signal_dir / "signal_metadata.json") as f:
        meta = json.load(f)
    return trainer, signal_dir, meta


# =============================================================================
# Phase Q.6.5.A — F-18 closure: sklearn signal_metadata carries
# Phase II compatibility block + compatibility_fingerprint
# =============================================================================


class TestSklearnSignalMetadataCompatibilityBlock:
    """F-18 closure — the sklearn ``SimpleModelTrainer.export_signals``
    must emit the Phase II ``compatibility`` 11-key block +
    ``compatibility_fingerprint`` (64-hex SHA-256) when constructed via
    the canonical ``from_config`` path.
    """

    def test_compatibility_block_present(self, synthetic_data_dir, tmp_path):
        _, _, meta = _train_export_via_from_config(synthetic_data_dir, tmp_path)
        assert "compatibility" in meta, (
            "F-18 NOT closed — sklearn signal_metadata.json must carry "
            "Phase II 'compatibility' block when self.config is set."
        )
        assert isinstance(meta["compatibility"], dict)

    def test_compatibility_fingerprint_present_and_64hex(self, synthetic_data_dir, tmp_path):
        _, _, meta = _train_export_via_from_config(synthetic_data_dir, tmp_path)
        fp = meta.get("compatibility_fingerprint")
        assert fp is not None, (
            "F-18 NOT closed — sklearn signal_metadata.json must carry "
            "compatibility_fingerprint when compatibility block is emitted."
        )
        assert SHA256_HEX_RE.match(fp), (
            f"compatibility_fingerprint must be 64-hex SHA-256, got: {fp!r}"
        )

    def test_compatibility_block_has_11_canonical_fields(self, synthetic_data_dir, tmp_path):
        """The canonical CompatibilityContract has 11 fields per
        hft_contracts.compatibility — verify the wire format includes them
        (modulo None values for optional fields). Locks the producer
        contract surface."""
        _, _, meta = _train_export_via_from_config(synthetic_data_dir, tmp_path)
        compat = meta["compatibility"]
        required_fields = {
            "contract_version",
            "schema_version",
            "feature_count",
            "window_size",
            "feature_layout",
            "data_source",
            "label_strategy_hash",
            "calibration_method",  # may be None
            "primary_horizon_idx",
            "horizons",
            "normalization_strategy",
        }
        missing = required_fields - set(compat.keys())
        assert not missing, (
            f"compatibility block missing canonical fields: {missing}. "
            f"Got keys: {sorted(compat.keys())}"
        )

    def test_top_level_data_source_present(self, synthetic_data_dir, tmp_path):
        """Phase II convenience: ``data_source`` is a top-level field
        (read by legacy consumers without nested-block parsing)."""
        _, _, meta = _train_export_via_from_config(synthetic_data_dir, tmp_path)
        assert "data_source" in meta, (
            "data_source top-level field missing — Phase II convenience for "
            "legacy consumers (metadata.py:166-167) requires it."
        )
        # Synthetic data_dir does not start with 'basic_', so derive_data_source returns 'mbo_lob'
        assert meta["data_source"] == "mbo_lob"

    def test_top_level_and_nested_contract_version_match(self, synthetic_data_dir, tmp_path):
        """Phase Q.9 invariant: top-level + nested contract_version + schema_version
        must agree when both are emitted. Locks against future producer-paths
        that mutate one but not the other."""
        _, _, meta = _train_export_via_from_config(synthetic_data_dir, tmp_path)
        assert meta["contract_version"] == meta["compatibility"]["contract_version"], (
            "Top-level vs nested contract_version DRIFT — Phase Q.9 invariant violated. "
            f"top-level={meta['contract_version']!r}, "
            f"nested={meta['compatibility']['contract_version']!r}"
        )
        assert meta["schema_version"] == meta["compatibility"]["schema_version"], (
            "Top-level vs nested schema_version DRIFT — Phase Q.9 invariant violated."
        )

    def test_legacy_flat_keyword_construction_omits_compat(self, synthetic_data_dir, tmp_path):
        """Back-compat: when ``self.config`` is None (legacy ad-hoc Python
        construction, NOT through ``from_config``), no compatibility block
        is emitted. This preserves pre-Q.6.5.A behavior for callers that
        use the flat-keyword constructor directly.
        """
        # Construct via flat-keyword (legacy path); self.config remains None
        trainer = SimpleModelTrainer(
            data_dir=str(synthetic_data_dir),
            model_type="temporal_ridge",
            model_config={"alpha": 1.0},
            output_dir=str(tmp_path / "output_legacy"),
        )
        assert trainer.config is None, "Legacy flat-keyword path must leave self.config = None"
        trainer.setup()
        trainer.train()
        trainer.evaluate("test")
        signal_dir = trainer.export_signals("test")
        with open(signal_dir / "signal_metadata.json") as f:
            meta = json.load(f)
        # Top-level schema_version + contract_version are ALWAYS emitted by
        # build_signal_metadata (H-1 contract pin), so verify those are still present.
        assert meta["schema_version"] == "3.0"
        assert meta["contract_version"] == "3.0"
        # But the Phase II compatibility block + fingerprint MUST be absent
        # (graceful back-compat for the legacy construction path).
        assert "compatibility" not in meta, (
            "Legacy flat-keyword path must NOT emit compatibility block "
            "(back-compat — self.config is None)."
        )
        assert "compatibility_fingerprint" not in meta
        assert "data_source" not in meta


class TestSklearnFingerprintDeterminism:
    """Determinism — same config → same fingerprint across runs.

    Locks the byte-stability invariant of CompatibilityContract.fingerprint()
    (uses canonical_json_blob + sha256_hex SSoT) for the sklearn path.
    """

    def test_two_identical_configs_produce_identical_fingerprints(
        self, synthetic_data_dir, tmp_path
    ):
        """Two independent ``from_config`` calls with the same YAML inputs
        must produce identical fingerprints. Validates that
        ``build_compatibility_contract`` is deterministic w.r.t. the
        ExperimentConfig dict shape."""
        config1 = _build_synthetic_config(synthetic_data_dir, tmp_path / "run1")
        trainer1 = SimpleModelTrainer.from_config(config1)
        trainer1.setup()
        trainer1.train()
        trainer1.evaluate("test")
        sig_dir1 = trainer1.export_signals("test")
        with open(sig_dir1 / "signal_metadata.json") as f:
            meta1 = json.load(f)

        config2 = _build_synthetic_config(synthetic_data_dir, tmp_path / "run2")
        trainer2 = SimpleModelTrainer.from_config(config2)
        trainer2.setup()
        trainer2.train()
        trainer2.evaluate("test")
        sig_dir2 = trainer2.export_signals("test")
        with open(sig_dir2 / "signal_metadata.json") as f:
            meta2 = json.load(f)

        assert meta1["compatibility_fingerprint"] == meta2["compatibility_fingerprint"], (
            "Same config produced different fingerprints — determinism violated. "
            f"run1={meta1['compatibility_fingerprint'][:16]}..., "
            f"run2={meta2['compatibility_fingerprint'][:16]}..."
        )


class TestSklearnFeatureSetRefPropagation:
    """Phase 4 4c.4 — when the trainer's resolver populates
    ``DataConfig._feature_set_ref_resolved`` (Tuple[str, str] of
    ``(name, content_hash)`` per schema.py:1109), the export_signals
    method must propagate it to ``signal_metadata.feature_set_ref`` as
    ``{"name": ..., "content_hash": ...}`` dict.

    This test sets the cache MANUALLY (bypasses the resolver) so the
    propagation logic is exercised in isolation. Real resolver behavior
    is tested in test_feature_set_resolver.py.
    """

    def test_feature_set_ref_propagated_when_cache_populated(
        self, synthetic_data_dir, tmp_path
    ):
        config = _build_synthetic_config(synthetic_data_dir, tmp_path)
        # Manually populate the private cache that trainer.py:424 normally
        # writes during _create_dataloaders. Use sample 64-hex content_hash.
        synthetic_hash = "a" * 64
        # PrivateAttr in Pydantic v2 SafeBaseModel — direct attribute assignment
        # is the documented pattern (matches trainer.py:424 setter).
        config.data._feature_set_ref_resolved = ("synthetic_v1", synthetic_hash)

        trainer = SimpleModelTrainer.from_config(config)
        trainer.setup()
        trainer.train()
        trainer.evaluate("test")
        signal_dir = trainer.export_signals("test")
        with open(signal_dir / "signal_metadata.json") as f:
            meta = json.load(f)

        assert "feature_set_ref" in meta, (
            "Phase 4 4c.4 — feature_set_ref must be propagated when "
            "_feature_set_ref_resolved cache is populated."
        )
        assert meta["feature_set_ref"] == {
            "name": "synthetic_v1",
            "content_hash": synthetic_hash,
        }
        # Compatibility block's feature_layout should also reflect the FeatureSet
        # content_hash (per build_compatibility_contract:172-176).
        assert meta["compatibility"]["feature_layout"] == synthetic_hash, (
            "When feature_set_ref is provided, CompatibilityContract.feature_layout "
            "is set to the content_hash (compatibility.py:172-176)."
        )

    def test_feature_set_ref_absent_when_no_cache(self, synthetic_data_dir, tmp_path):
        """When the resolver did NOT populate the cache (ad-hoc / preset
        selection paths), ``feature_set_ref`` is omitted from
        signal_metadata.json. ``feature_layout`` falls back to ``"default"``."""
        config = _build_synthetic_config(synthetic_data_dir, tmp_path)
        # Cache is None by default (PrivateAttr default per schema.py:1109).
        assert config.data._feature_set_ref_resolved is None

        trainer = SimpleModelTrainer.from_config(config)
        trainer.setup()
        trainer.train()
        trainer.evaluate("test")
        signal_dir = trainer.export_signals("test")
        with open(signal_dir / "signal_metadata.json") as f:
            meta = json.load(f)

        assert "feature_set_ref" not in meta, (
            "When no FeatureSet was resolved, feature_set_ref must be omitted."
        )
        # feature_layout falls back to "default" per compatibility.py:172-176.
        assert meta["compatibility"]["feature_layout"] == "default"

    @pytest.mark.parametrize(
        "malformed_value,description",
        [
            (("name_only",), "1-tuple"),
            (("", "valid_hash" + "0" * 54), "empty name"),
            (("valid_name", ""), "empty content_hash"),
            (("", ""), "both empty"),
            ("not_a_tuple", "string instead of tuple"),
            (("name", "hash", "extra"), "3-tuple"),
        ],
    )
    def test_malformed_cache_rejected_gracefully(
        self, synthetic_data_dir, tmp_path, malformed_value, description
    ):
        """Q.6.5.A post-audit hardening (Gap-3 + HIGH-2 regression lock):
        defensive tuple-unpack must reject malformed cache values without
        crashing the export. The ``feature_set_ref`` key is omitted from
        signal_metadata.json; downstream consumers' CONTENT_HASH_RE
        validation is not exercised on a poisoned dict.

        Mirrors the empty-string guard at importance/callback.py:595 +
        the try/except at exporter.py:42-64 / simple_trainer.py:393-413.
        """
        config = _build_synthetic_config(synthetic_data_dir, tmp_path)
        # Inject malformed value (Pydantic v2 PrivateAttr accepts any type)
        config.data._feature_set_ref_resolved = malformed_value

        trainer = SimpleModelTrainer.from_config(config)
        trainer.setup()
        trainer.train()
        trainer.evaluate("test")
        # Should NOT raise — defensive branch returns None, export proceeds.
        signal_dir = trainer.export_signals("test")
        with open(signal_dir / "signal_metadata.json") as f:
            meta = json.load(f)

        assert "feature_set_ref" not in meta, (
            f"Malformed cache ({description}: {malformed_value!r}) silently "
            f"emitted feature_set_ref — defensive guard at simple_trainer.py "
            f"failed. Expected: omitted (graceful rejection)."
        )
        # The compatibility block should still be emitted (compat builder
        # tolerates feature_set_ref=None — feature_layout defaults to "default").
        assert "compatibility" in meta
        assert meta["compatibility"]["feature_layout"] == "default"


class TestSklearnExportSignalsSignatureParity:
    """Phase Q.6.5.B Part 1 — ``SimpleModelTrainer.export_signals`` signature
    extended with ``output_dir`` + ``calibration`` keyword-only kwargs for
    parity with ``Trainer.export_signals``. Locks the new signature
    against accidental drift.
    """

    def test_output_dir_override_honored(self, synthetic_data_dir, tmp_path):
        """When ``output_dir`` is explicitly provided, signals are written
        there instead of ``<self.output_dir>/signals/<split>/``."""
        config = _build_synthetic_config(synthetic_data_dir, tmp_path)
        trainer = SimpleModelTrainer.from_config(config)
        trainer.setup()
        trainer.train()
        trainer.evaluate("test")

        custom_dir = tmp_path / "custom_signal_location"
        result = trainer.export_signals("test", output_dir=custom_dir)
        assert result == custom_dir
        assert custom_dir.exists()
        assert (custom_dir / "signal_metadata.json").exists()
        assert (custom_dir / "predicted_returns.npy").exists()
        # Default location MUST NOT be created when override is provided.
        default_signals = trainer.output_dir / "signals" / "test"
        assert not default_signals.exists(), (
            "When output_dir=<override>, default location must NOT be created."
        )

    def test_calibration_none_default_works(self, synthetic_data_dir, tmp_path):
        """Default ``calibration="none"`` is the documented sklearn behavior.
        Equivalent to omitting the kwarg entirely."""
        config = _build_synthetic_config(synthetic_data_dir, tmp_path)
        trainer = SimpleModelTrainer.from_config(config)
        trainer.setup()
        trainer.train()
        trainer.evaluate("test")
        # Both equivalent — no error, returns Path.
        path1 = trainer.export_signals("test", calibration="none")
        assert path1.exists()
        path2 = trainer.export_signals("test")  # default kwarg
        assert path2 == path1

    @pytest.mark.parametrize(
        "calibration",
        ["variance_match", "isotonic", "quantile", "unknown"],
    )
    def test_calibration_non_none_raises(
        self, synthetic_data_dir, tmp_path, calibration
    ):
        """Phase Q.6.5.B fail-fast (hft-rules §5): sklearn rejects any
        calibration strategy other than ``"none"`` until Phase X.6 wires
        the calibration pipeline. Silent-degrade was the anti-pattern."""
        config = _build_synthetic_config(synthetic_data_dir, tmp_path)
        trainer = SimpleModelTrainer.from_config(config)
        trainer.setup()
        trainer.train()
        with pytest.raises(ValueError, match=r"does not yet support calibration"):
            trainer.export_signals("test", calibration=calibration)

    def test_split_val_raises_with_actionable_message(
        self, synthetic_data_dir, tmp_path
    ):
        """Sklearn currently restricts to ``"test"`` only — error message
        must point operators to the simple_trainer.py:218 root cause
        (val arrays loaded but spread/price columns discarded)."""
        config = _build_synthetic_config(synthetic_data_dir, tmp_path)
        trainer = SimpleModelTrainer.from_config(config)
        trainer.setup()
        trainer.train()
        with pytest.raises(ValueError, match=r"Only 'test' split"):
            trainer.export_signals("val")

    def test_output_dir_and_calibration_are_keyword_only(
        self, synthetic_data_dir, tmp_path
    ):
        """``output_dir`` and ``calibration`` MUST be keyword-only (after
        ``*`` in the signature) so positional misuse is rejected at call
        time. Locks against accidental positional drift."""
        config = _build_synthetic_config(synthetic_data_dir, tmp_path)
        trainer = SimpleModelTrainer.from_config(config)
        trainer.setup()
        trainer.train()
        trainer.evaluate("test")
        # Positional 2nd arg must NOT be interpreted as output_dir.
        with pytest.raises(TypeError, match=r"positional argument"):
            trainer.export_signals("test", tmp_path / "positional_misuse")  # type: ignore[misc]


class TestSklearnSidecarLegacyCallerDeprecation:
    """Phase Q.6.5.F — N-7 closure: sidecar legacy-caller silent-degrade
    promoted to ``DeprecationWarning`` per hft-rules §5 fail-fast.

    Locks emission of the warning when ``SimpleModelTrainer.save_checkpoint``
    is called with ``self.config is None`` (legacy flat-keyword construction
    path). Pre-Q.6.5.F this was a silent ``logger.warning`` that test runners
    could miss; post-Q.6.5.F it surfaces via Python's ``warnings`` machinery
    so CI / IDEs / pytest catch it.
    """

    def test_legacy_flat_keyword_save_emits_deprecation_warning(
        self, synthetic_data_dir, tmp_path
    ):
        """When ``SimpleModelTrainer`` is constructed via the legacy
        flat-keyword path (``self.config = None``), ``save_checkpoint``
        must emit a ``DeprecationWarning`` pointing operators to
        ``from_config``. Removal target: 2027-04-01."""
        # Legacy flat-keyword construction (NOT through from_config).
        trainer = SimpleModelTrainer(
            data_dir=str(synthetic_data_dir),
            model_type="temporal_ridge",
            model_config={"alpha": 1.0},
            output_dir=str(tmp_path / "legacy_output"),
        )
        assert trainer.config is None, (
            "Legacy flat-keyword path must leave self.config = None"
        )
        trainer.setup()
        trainer.train()

        # save_checkpoint MUST emit DeprecationWarning per Q.6.5.F.
        with pytest.warns(
            DeprecationWarning,
            match=r"legacy flat-keyword construction",
        ) as warnings_record:
            trainer.save_checkpoint()

        # At least one DeprecationWarning was captured (pytest.warns asserts >=1
        # but we explicitly verify the actionable message points to the migration
        # path + removal calendar).
        msg_texts = [str(w.message) for w in warnings_record.list]
        relevant = [m for m in msg_texts if "legacy flat-keyword" in m]
        assert len(relevant) >= 1, (
            f"Q.6.5.F DeprecationWarning not emitted. Captured: {msg_texts}"
        )
        msg = relevant[0]
        assert "from_config" in msg, (
            "Migration path (SimpleModelTrainer.from_config) must be cited "
            f"in the DeprecationWarning. Got: {msg}"
        )
        assert "2027-04-01" in msg, (
            f"Removal target date (2027-04-01) must be cited. Got: {msg}"
        )

    def test_from_config_save_does_not_emit_deprecation_warning(
        self, synthetic_data_dir, tmp_path
    ):
        """Canonical ``from_config`` path is the supported construction
        idiom — ``save_checkpoint`` MUST NOT emit DeprecationWarning when
        invoked through it. Locks against false-positive emissions."""
        config = _build_synthetic_config(synthetic_data_dir, tmp_path)
        trainer = SimpleModelTrainer.from_config(config)
        assert trainer.config is config, (
            "from_config must populate self.config"
        )
        trainer.setup()
        trainer.train()

        # No DeprecationWarning expected on this path.
        import warnings as _w
        with _w.catch_warnings(record=True) as captured:
            _w.simplefilter("always")
            trainer.save_checkpoint()
            deprecations = [
                w for w in captured
                if issubclass(w.category, DeprecationWarning)
                and "legacy flat-keyword" in str(w.message)
            ]
            assert not deprecations, (
                f"from_config path falsely emitted Q.6.5.F DeprecationWarning. "
                f"Captured: {[str(w.message) for w in deprecations]}"
            )


class TestSklearnDataSourceTagging:
    """Phase II ``data_source`` derivation — the ``derive_data_source`` helper
    inspects the data_dir basename to classify mbo_lob vs off_exchange. This
    test exercises the heuristic on the sklearn path."""

    def test_mbo_lob_data_dir_yields_mbo_lob_tag(self, synthetic_data_dir, tmp_path):
        """Default synthetic_data_dir is at tmp_path (not prefixed 'basic_'),
        so derive_data_source returns 'mbo_lob'."""
        _, _, meta = _train_export_via_from_config(synthetic_data_dir, tmp_path)
        assert meta["compatibility"]["data_source"] == "mbo_lob"
        assert meta["data_source"] == "mbo_lob"

    def test_basic_prefix_data_dir_yields_off_exchange_tag(self, tmp_path):
        """Off-exchange convention: directory basename starts with 'basic_'.
        derive_data_source(...) returns 'off_exchange'."""
        # Build a fixture with basic_-prefixed parent dir
        rng = np.random.default_rng(42)
        basic_root = tmp_path / "basic_synthetic_60s"
        basic_root.mkdir()
        for split in ["train", "val", "test"]:
            split_dir = basic_root / split
            split_dir.mkdir()
            for i, day in enumerate(["2025-01-01", "2025-01-02"]):
                n = NUM_SEQS + i * 5
                sequences = rng.standard_normal((n, SEQ_LEN, NUM_FEATURES)).astype(np.float32)
                sequences[:, -1, 40] = 130.0 + rng.standard_normal(n).astype(np.float32) * 0.5
                sequences[:, -1, 42] = 2.5 + rng.standard_normal(n).astype(np.float32) * 0.1
                reg_labels = rng.standard_normal((n, NUM_HORIZONS)).astype(np.float64)
                np.save(split_dir / f"{day}_sequences.npy", sequences)
                np.save(split_dir / f"{day}_regression_labels.npy", reg_labels)
                metadata = {
                    "day": day,
                    "n_sequences": n,
                    "n_features": NUM_FEATURES,
                    "schema_version": "3.0",
                }
                with open(split_dir / f"{day}_metadata.json", "w") as f:
                    json.dump(metadata, f)

        config = _build_synthetic_config(basic_root, tmp_path)
        trainer = SimpleModelTrainer.from_config(config)
        trainer.setup()
        trainer.train()
        trainer.evaluate("test")
        signal_dir = trainer.export_signals("test")
        with open(signal_dir / "signal_metadata.json") as f:
            meta = json.load(f)
        assert meta["compatibility"]["data_source"] == "off_exchange", (
            "data_dir prefixed 'basic_' must yield data_source='off_exchange' "
            "per derive_data_source heuristic (compatibility.py:117-131)."
        )
        assert meta["data_source"] == "off_exchange"
