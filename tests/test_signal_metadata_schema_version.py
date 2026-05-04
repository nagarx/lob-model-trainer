"""H-1 regression tests for signal_metadata.json schema_version + contract_version.

Phase O Cycle 1 consumer-side hardening (2026-05-04). Pre-H-1 the trainer's
two signal_metadata producers (`SimpleModelTrainer.export_signals` and
`build_signal_metadata`) both OMITTED `schema_version` and `contract_version`.
The backtester loader's truthiness check (`metadata.get("schema_version")`)
silently skipped validation, so version-skew between trainer and backtester
went undetected.

After H-1 both producers stamp `schema_version` + `contract_version` from
`hft_contracts.SCHEMA_VERSION` so the backtester (post-C-4 presence check
in `lob-backtester/.../loader.py`) can fail-loud on mismatch.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hft_contracts import SCHEMA_VERSION
from lobtrainer.export.metadata import build_signal_metadata
from lobtrainer.training.simple_trainer import SimpleModelTrainer


class TestBuildSignalMetadataSchemaPin:
    """build_signal_metadata always emits schema_version + contract_version."""

    def _build(self, **overrides):
        kwargs = dict(
            model_type="tlob",
            model_name="test",
            parameters=100,
            signal_type="regression",
            split="test",
            total_samples=1000,
            checkpoint="/tmp/ckpt",
        )
        kwargs.update(overrides)
        return build_signal_metadata(**kwargs)

    def test_schema_version_present_with_contract_value(self):
        meta = self._build()
        assert meta["schema_version"] == str(SCHEMA_VERSION), (
            f"signal_metadata.json schema_version must equal contract "
            f"SCHEMA_VERSION ({SCHEMA_VERSION!r}), got {meta.get('schema_version')!r}"
        )

    def test_contract_version_present_with_contract_value(self):
        meta = self._build()
        assert meta["contract_version"] == str(SCHEMA_VERSION), (
            f"signal_metadata.json contract_version must equal contract "
            f"SCHEMA_VERSION ({SCHEMA_VERSION!r}), got {meta.get('contract_version')!r}"
        )

    def test_schema_fields_are_first_for_consumer_robustness(self):
        """Schema-pin keys appear at the top of the dict so they are read
        first by streaming JSON consumers."""
        meta = self._build()
        first_keys = list(meta.keys())[:2]
        assert first_keys == ["schema_version", "contract_version"], (
            f"First two keys must be schema-pin, got {first_keys}"
        )


class TestContractVersionParity:
    """Phase Q.9 (2026-05-04): when ``compatibility`` block is emitted,
    top-level ``contract_version`` / ``schema_version`` MUST equal the
    nested values. Closes the audit MEDIUM (no invariant test pinning
    the two paths) and locks the producer-side equality so any future
    bug touching one path but not the other is caught immediately.
    """

    def test_top_level_matches_nested_contract_version(self):
        from hft_contracts.compatibility import CompatibilityContract
        compat = CompatibilityContract(
            contract_version="3.0",
            schema_version="3.0",
            feature_count=98,
            window_size=20,
            feature_layout="ask_prices_10_ask_sizes_10_bid_prices_10_bid_sizes_10",
            data_source="MBO",
            label_strategy_hash="a" * 64,
            calibration_method=None,
            primary_horizon_idx=0,
            horizons=(10, 60, 300),
            normalization_strategy="hybrid",
        )
        meta = build_signal_metadata(
            model_type="tlob", model_name="test", parameters=100,
            signal_type="regression", split="test", total_samples=1000,
            checkpoint="/tmp/ckpt", compatibility=compat,
        )
        assert meta["contract_version"] == meta["compatibility"]["contract_version"], (
            f"top-level contract_version={meta['contract_version']!r} must "
            f"equal compatibility.contract_version="
            f"{meta['compatibility']['contract_version']!r}"
        )
        assert meta["schema_version"] == meta["compatibility"]["schema_version"], (
            f"top-level schema_version={meta['schema_version']!r} must "
            f"equal compatibility.schema_version="
            f"{meta['compatibility']['schema_version']!r}"
        )

    def test_no_compatibility_block_does_not_raise_invariant(self):
        """Without a `compatibility` arg, only top-level fields are
        present; no parity assertion needed."""
        meta = build_signal_metadata(
            model_type="tlob", model_name="test", parameters=100,
            signal_type="regression", split="test", total_samples=1000,
            checkpoint="/tmp/ckpt",
        )
        assert "compatibility" not in meta
        assert meta["contract_version"] == str(SCHEMA_VERSION)


class TestSimpleTrainerSignalMetadataSchemaPin:
    """SimpleModelTrainer.export_signals also stamps schema_version."""

    @pytest.fixture
    def synthetic_data_dir(self, tmp_path: Path) -> Path:
        rng = np.random.default_rng(42)
        for split in ["train", "val", "test"]:
            split_dir = tmp_path / split
            split_dir.mkdir()
            day = "20250203"
            n = 12
            seqs = rng.standard_normal((n, 20, 98)).astype(np.float32)
            reg = rng.standard_normal((n, 3)).astype(np.float64)
            np.save(split_dir / f"{day}_sequences.npy", seqs)
            np.save(split_dir / f"{day}_regression_labels.npy", reg)
            with open(split_dir / f"{day}_metadata.json", "w") as f:
                json.dump(
                    {
                        "day": day,
                        "n_sequences": n,
                        "n_features": 98,
                        "schema_version": str(SCHEMA_VERSION),
                    },
                    f,
                )
        return tmp_path

    def test_simple_trainer_emits_schema_pin(
        self, synthetic_data_dir: Path, tmp_path: Path
    ) -> None:
        trainer = SimpleModelTrainer(
            data_dir=str(synthetic_data_dir),
            model_type="temporal_ridge",
            model_config={"alpha": 1.0},
            output_dir=str(tmp_path / "output"),
        )
        trainer.setup()
        trainer.train()
        trainer.evaluate()
        signal_dir = trainer.export_signals("test")

        meta_path = signal_dir / "signal_metadata.json"
        assert meta_path.exists()
        with open(meta_path) as f:
            meta = json.load(f)

        assert meta.get("schema_version") == str(SCHEMA_VERSION)
        assert meta.get("contract_version") == str(SCHEMA_VERSION)


class TestSimpleTrainerCanonicalFieldSet:
    """Phase Q.7 (2026-05-04): SimpleModelTrainer.export_signals now uses
    the canonical ``build_signal_metadata`` SSoT. This locks that the
    sklearn signal_metadata.json carries the rich field set previously
    only present on the PyTorch path.
    """

    @pytest.fixture
    def synthetic_data_dir(self, tmp_path: Path) -> Path:
        rng = np.random.default_rng(42)
        for split in ["train", "val", "test"]:
            split_dir = tmp_path / split
            split_dir.mkdir()
            day = "20250203"
            n = 12
            seqs = rng.standard_normal((n, 20, 98)).astype(np.float32)
            reg = rng.standard_normal((n, 3)).astype(np.float64)
            np.save(split_dir / f"{day}_sequences.npy", seqs)
            np.save(split_dir / f"{day}_regression_labels.npy", reg)
            with open(split_dir / f"{day}_metadata.json", "w") as f:
                json.dump(
                    {"day": day, "n_sequences": n, "n_features": 98,
                     "schema_version": str(SCHEMA_VERSION)}, f,
                )
        return tmp_path

    def test_sklearn_signal_metadata_has_canonical_fields(
        self, synthetic_data_dir: Path, tmp_path: Path,
    ):
        """Pre-Q.7 the sklearn path emitted only 9 fields; post-Q.7 the
        canonical builder emits 12+ for a regression run (incl. the
        sklearn-only `feature_config`, plus `signal_type`, `checkpoint`,
        `exported_at`)."""
        trainer = SimpleModelTrainer(
            data_dir=str(synthetic_data_dir),
            model_type="temporal_ridge",
            model_config={"alpha": 1.0},
            output_dir=str(tmp_path / "output"),
        )
        trainer.setup()
        trainer.train()
        trainer.evaluate()
        signal_dir = trainer.export_signals("test")
        with open(signal_dir / "signal_metadata.json") as f:
            meta = json.load(f)

        # Schema-pin (H-1)
        assert meta["schema_version"] == str(SCHEMA_VERSION)
        assert meta["contract_version"] == str(SCHEMA_VERSION)
        # Canonical-builder always-emitted fields
        assert meta["model_type"] == "temporal_ridge"
        assert meta["signal_type"] == "regression"  # NEW after Q.7
        assert meta["split"] == "test"
        assert "checkpoint" in meta and "best.pkl" in meta["checkpoint"]
        assert "exported_at" in meta  # NEW after Q.7
        # Sklearn-only optional field
        assert "feature_config" in meta
        assert "signal_indices" in meta["feature_config"]


class TestAtomicWriteForSignalMetadata:
    """Phase Q.8 (2026-05-04): both producers (PyTorch exporter + sklearn
    SimpleModelTrainer) write signal_metadata.json through
    ``atomic_write_json`` — tmp + fsync + os.replace + cleanup. Locks
    that no producer regresses to a naive ``open(... 'w'); json.dump``
    pattern that could leave partial files on SIGKILL/ENOSPC.
    """

    def test_simple_trainer_uses_atomic_write(self, tmp_path):
        """SimpleModelTrainer.export_signals invokes atomic_write_json
        for signal_metadata.json (verified by monkeypatch)."""
        from unittest.mock import patch
        from lobtrainer.training.simple_trainer import SimpleModelTrainer

        # Build minimal synthetic data dir.
        rng = np.random.default_rng(0)
        for split in ("train", "val", "test"):
            d = tmp_path / "data" / split
            d.mkdir(parents=True)
            day = "20250203"
            np.save(d / f"{day}_sequences.npy",
                    rng.standard_normal((6, 20, 98)).astype(np.float32))
            np.save(d / f"{day}_regression_labels.npy",
                    rng.standard_normal((6, 3)).astype(np.float64))
            with open(d / f"{day}_metadata.json", "w") as f:
                json.dump({"day": day, "n_sequences": 6, "n_features": 98,
                           "schema_version": str(SCHEMA_VERSION)}, f)

        trainer = SimpleModelTrainer(
            data_dir=str(tmp_path / "data"),
            model_type="temporal_ridge",
            model_config={"alpha": 1.0},
            output_dir=str(tmp_path / "out"),
        )
        trainer.setup()
        trainer.train()
        trainer.evaluate()

        with patch("hft_contracts.atomic_io.atomic_write_json") as mock_write:
            mock_write.return_value = None
            # Invoke the real method, which should call our mocked
            # atomic_write_json. We can't import the same symbol the
            # production module imports lazily without patching its
            # binding — the lazy import inside export_signals fetches
            # the function from hft_contracts.atomic_io at call time,
            # so the module-level patch is effective.
            trainer.export_signals("test")

        assert mock_write.called, (
            "SimpleModelTrainer.export_signals must call "
            "atomic_write_json for signal_metadata.json"
        )
        # Path argument is signal_metadata.json under signals/test/.
        call_args = mock_write.call_args
        target_path = Path(call_args[0][0])
        assert target_path.name == "signal_metadata.json"
        assert "signals/test" in str(target_path)

    def test_pytorch_exporter_uses_atomic_write(self):
        """The PyTorch SignalExporter at exporter.py:268-275 also uses
        atomic_write_json. Verified by source-code inspection — a
        naive ``with open(...,"w"); json.dump`` regression must NOT
        slip back in.
        """
        import inspect
        from lobtrainer.export import exporter
        source = inspect.getsource(exporter)
        # The exporter source must reference atomic_write_json AND must
        # NOT contain a naive `json.dump(metadata, f` write to
        # signal_metadata.json.
        assert "atomic_write_json" in source, (
            "exporter.py must use atomic_write_json for signal_metadata.json"
        )
        # Specifically check that the legacy pattern is gone for the
        # signal_metadata.json site.
        bad_pattern = 'json.dump(metadata, f'
        # The pattern may still appear in helper code, but the
        # signal_metadata.json write site at meta_path must not.
        # Heuristic: line near `meta_path =` should not be followed
        # by a `with open` block.
        lines = source.split("\n")
        for i, line in enumerate(lines):
            if "meta_path =" in line and "signal_metadata.json" in line:
                # Inspect next 5 lines for the bad pattern.
                window = "\n".join(lines[i:i + 5])
                assert "atomic_write_json" in window, (
                    f"Line {i}: meta_path block does not call "
                    f"atomic_write_json. Window:\n{window}"
                )
                assert bad_pattern not in window, (
                    f"Line {i}: meta_path block still uses naive "
                    f"json.dump. Window:\n{window}"
                )
