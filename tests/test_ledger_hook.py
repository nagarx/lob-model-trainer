"""#PY-223 MINIMAL Step 2 (2026-05-14): direct-trainer-invocation ledger hook.

Test coverage for ``lobtrainer.ledger_hook.write_minimal_ledger_record``.
The helper closes the R-17a-class ~26% experiment-invisibility gap by
writing a minimal ``ExperimentRecord`` JSON when ``scripts/train.py`` is
invoked with ``--record-to-ledger`` AND the harvest-from-signal-metadata
path can be located.

Test classes:
- ``TestResolveLedgerDir`` — directory resolution priority order
  (explicit -> pipeline_root -> climb).
- ``TestHarvestSignalMetadata`` — JSON-load + trust-column extraction +
  graceful-failure on malformed files.
- ``TestWriteMinimalLedgerRecord`` — end-to-end happy path + opt-out
  paths + observation-tier error swallow.
- ``TestPY223PartialRecordContract`` — locks the record shape so future
  cycles can't accidentally drift the schema.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import pytest

from lobtrainer.ledger_hook import (
    _find_signal_metadata_path,
    _resolve_ledger_dir,
    write_minimal_ledger_record,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config_stub(name: str = "test_exp", tags=None) -> SimpleNamespace:
    """Minimal config stub supporting ``model_dump`` semantics."""
    stub = SimpleNamespace(
        name=name,
        tags=tags or ["test", "py223"],
        contract_version="3.0",
    )
    # Inject a model_dump method to mimic Pydantic v2 BaseModel.
    stub.model_dump = lambda mode=None: {
        "name": name,
        "tags": list(stub.tags),
        "contract_version": "3.0",
        "_some_setting": 42,
    }
    return stub


def _write_signal_metadata(
    path: Path,
    *,
    compatibility_fingerprint: Optional[str] = None,
    model_config_hash: Optional[str] = None,
    feature_set_ref: Optional[Dict[str, str]] = None,
) -> None:
    """Write a minimal signal_metadata.json with the requested trust columns.

    Phase 8D / #PY-223 SSoT migration (2026-05-14): match the REAL producer
    schema at ``lobtrainer/export/metadata.py:189`` which emits the
    compatibility fingerprint at TOP-LEVEL ``metadata["compatibility_fingerprint"]``
    (the computed 64-hex SHA), distinct from the nested
    ``metadata["compatibility"]`` block (the 11 INPUT fields used to compute
    the fingerprint). The parallel-session helper wrote the fingerprint
    value at ``metadata["compatibility"]["fingerprint"]`` which is WRONG
    schema; SSoT-migration corrects this so harvest matches real producer.
    """
    metadata: Dict[str, Any] = {"name": "test"}
    if compatibility_fingerprint is not None:
        # Top-level fingerprint VALUE (matches real producer schema).
        metadata["compatibility_fingerprint"] = compatibility_fingerprint
    if model_config_hash is not None:
        metadata["model_config_hash"] = model_config_hash
    if feature_set_ref is not None:
        metadata["feature_set_ref"] = feature_set_ref
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# _resolve_ledger_dir
# ---------------------------------------------------------------------------


class TestResolveLedgerDir:
    def test_explicit_dir_wins_when_exists(self, tmp_path: Path):
        explicit = tmp_path / "my_ledger"
        explicit.mkdir()
        resolved = _resolve_ledger_dir(
            explicit_dir=explicit,
            output_dir=tmp_path / "out",
        )
        assert resolved == explicit.resolve()

    def test_explicit_dir_missing_falls_through(self, tmp_path: Path, caplog):
        explicit = tmp_path / "nonexistent"
        # No fallback path will work either; expect None + WARN log.
        with caplog.at_level(logging.WARNING):
            resolved = _resolve_ledger_dir(
                explicit_dir=explicit,
                output_dir=tmp_path / "out",
            )
        assert resolved is None
        assert "does not exist" in caplog.text

    def test_pipeline_root_fallback(self, tmp_path: Path):
        # Build <pipeline_root>/hft-ops/ledger/records/
        pipeline_root = tmp_path / "pipeline"
        (pipeline_root / "hft-ops" / "ledger" / "records").mkdir(parents=True)
        resolved = _resolve_ledger_dir(
            explicit_dir=None,
            output_dir=tmp_path / "out",
            pipeline_root=pipeline_root,
        )
        assert resolved is not None
        assert resolved == (
            pipeline_root / "hft-ops" / "ledger" / "records"
        ).resolve()

    def test_climb_from_output_dir(self, tmp_path: Path):
        # tmp_path/hft-ops/ledger/records exists; output_dir is deep.
        records = tmp_path / "hft-ops" / "ledger" / "records"
        records.mkdir(parents=True)
        deep_out = tmp_path / "lob-model-trainer" / "outputs" / "experiments" / "foo"
        deep_out.mkdir(parents=True)
        resolved = _resolve_ledger_dir(
            explicit_dir=None,
            output_dir=deep_out,
        )
        assert resolved is not None
        assert resolved == records.resolve()

    def test_no_resolution_returns_none(self, tmp_path: Path, caplog):
        with caplog.at_level(logging.WARNING):
            resolved = _resolve_ledger_dir(
                explicit_dir=None,
                output_dir=tmp_path / "out",
            )
        assert resolved is None
        assert "could not locate" in caplog.text


# ---------------------------------------------------------------------------
# _harvest_signal_metadata
# ---------------------------------------------------------------------------


class TestFindSignalMetadataPath:
    """Phase 8D / #PY-223 (2026-05-14) SSoT migration: harvest semantics
    moved to ``hft_contracts.experiment_recorder.harvest_trust_columns_from_signal_metadata``
    (40 tests at ``hft-contracts/tests/test_experiment_recorder.py``).
    The trainer-local helper now ONLY returns the resolved Path to the
    first existing candidate — probe order is the only trainer-specific
    concern. Coverage shifted from 7 inline tests to 3 probe tests +
    SSoT inheritance.
    """

    def test_signal_metadata_at_output_root(self, tmp_path: Path):
        path = tmp_path / "signal_metadata.json"
        _write_signal_metadata(
            path, compatibility_fingerprint="a" * 64,
        )
        result = _find_signal_metadata_path(tmp_path)
        assert result is not None
        assert result.resolve() == path.resolve()

    def test_signal_metadata_under_signals_test(self, tmp_path: Path):
        path = tmp_path / "signals" / "test" / "signal_metadata.json"
        _write_signal_metadata(path, compatibility_fingerprint="d" * 64)
        result = _find_signal_metadata_path(tmp_path)
        assert result is not None
        assert result.resolve() == path.resolve()

    def test_missing_signal_metadata_returns_none(self, tmp_path: Path):
        assert _find_signal_metadata_path(tmp_path) is None

    def test_first_hit_wins(self, tmp_path: Path):
        """Root-level signal_metadata.json wins over signals/test/ per
        ``_SIGNAL_METADATA_CANDIDATES`` priority order."""
        root_path = tmp_path / "signal_metadata.json"
        _write_signal_metadata(root_path, compatibility_fingerprint="r" * 64)
        _write_signal_metadata(
            tmp_path / "signals" / "test" / "signal_metadata.json",
            compatibility_fingerprint="t" * 64,
        )
        result = _find_signal_metadata_path(tmp_path)
        assert result is not None
        assert result.resolve() == root_path.resolve()


# ---------------------------------------------------------------------------
# write_minimal_ledger_record
# ---------------------------------------------------------------------------


class TestWriteMinimalLedgerRecord:
    def test_happy_path_with_signal_metadata(self, tmp_path: Path):
        ledger = tmp_path / "ledger_records"
        ledger.mkdir()
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        _write_signal_metadata(
            output_dir / "signal_metadata.json",
            compatibility_fingerprint="f" * 64,
            model_config_hash="9" * 64,
        )
        record_path = write_minimal_ledger_record(
            config=_make_config_stub(name="happy_test"),
            output_dir=output_dir,
            train_result={
                "total_epochs": 10,
                "best_epoch": 7,
                "best_val_metric": 0.5,
            },
            test_metrics={"test_ic": 0.3, "test_da": 0.62},
            ledger_dir=ledger,
            duration_seconds=42.5,
        )
        assert record_path is not None
        assert record_path.exists()
        assert record_path.parent == ledger.resolve()
        # Read back + verify shape.
        with open(record_path, "r") as f:
            doc = json.load(f)
        assert doc["name"] == "happy_test"
        assert doc["compatibility_fingerprint"] == "f" * 64
        assert doc["training_config"]["model_config_hash"] == "9" * 64
        assert doc["status"] == "completed"

    def test_no_ledger_dir_returns_none(self, tmp_path: Path, caplog):
        """When no ledger dir resolves, return None + log WARN."""
        with caplog.at_level(logging.WARNING):
            record_path = write_minimal_ledger_record(
                config=_make_config_stub(),
                output_dir=tmp_path / "out",
            )
        assert record_path is None

    def test_partial_record_when_signal_metadata_missing(self, tmp_path: Path):
        """No signal_metadata.json present — record still written with the
        3 trust columns as None."""
        ledger = tmp_path / "ledger"
        ledger.mkdir()
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        record_path = write_minimal_ledger_record(
            config=_make_config_stub(name="partial_test"),
            output_dir=output_dir,
            ledger_dir=ledger,
        )
        assert record_path is not None
        with open(record_path, "r") as f:
            doc = json.load(f)
        assert doc["compatibility_fingerprint"] is None
        assert doc["feature_set_ref"] is None
        # training_config should NOT have model_config_hash injected when
        # signal_metadata is absent.
        assert "model_config_hash" not in doc.get("training_config", {})

    def test_explicit_timestamp_silently_ignored_post_ssot_migration(
        self, tmp_path: Path,
    ):
        """Phase 8D / #PY-223 (2026-05-14): the ``timestamp`` kwarg is
        currently a no-op post-SSoT migration. ``record_from_artifacts``
        uses its own ``datetime.now(timezone.utc)`` for experiment_id
        construction. The kwarg is preserved on the
        ``write_minimal_ledger_record`` signature for API stability +
        future re-wiring but does NOT affect the experiment_id. Operators
        who need a deterministic timestamp can pre-stub ``datetime.now``.

        This test locks the current behavior (timestamp ignored, record
        still written) and documents the gap for a future cycle.
        """
        ledger = tmp_path / "ledger"
        ledger.mkdir()
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        ts = datetime(2026, 5, 14, 12, 30, 45, tzinfo=timezone.utc)
        record_path = write_minimal_ledger_record(
            config=_make_config_stub(name="ts_test"),
            output_dir=output_dir,
            ledger_dir=ledger,
            timestamp=ts,
        )
        assert record_path is not None
        # timestamp arg is NOT threaded through SSoT post-migration;
        # experiment_id uses datetime.now(utc) inside record_from_artifacts.
        # Just verify the record was written and starts with the config name.
        assert record_path.name.startswith("ts_test_")

    def test_construction_failure_does_not_propagate(
        self, tmp_path: Path, caplog, monkeypatch,
    ):
        """Per hft-rules §8: never propagate observation-tier errors.

        Phase 8D / #PY-223 SSoT migration (2026-05-14): ``ExperimentRecord``
        is imported at module-load time in ``experiment_recorder`` so
        patching ``hft_contracts.experiment_record.ExperimentRecord``
        wouldn't affect SSoT's bound reference. Patch
        ``record_from_artifacts`` directly to inject a failure into the
        delegation path.
        """
        ledger = tmp_path / "ledger"
        ledger.mkdir()
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        # Force record_from_artifacts to fail (delegation-path failure).
        def _broken_record(*args, **kwargs):
            raise RuntimeError("synthetic failure for test")

        monkeypatch.setattr(
            "hft_contracts.experiment_recorder.record_from_artifacts",
            _broken_record,
        )

        with caplog.at_level(logging.WARNING):
            record_path = write_minimal_ledger_record(
                config=_make_config_stub(),
                output_dir=output_dir,
                ledger_dir=ledger,
            )
        # Helper swallowed the error + returned None + WARN-logged.
        assert record_path is None
        assert "synthetic failure" in caplog.text

    def test_training_metrics_flattening(self, tmp_path: Path):
        """val_metrics + test_metrics + train_result keys land in the record's
        training_metrics flat-scalar dict."""
        ledger = tmp_path / "ledger"
        ledger.mkdir()
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        record_path = write_minimal_ledger_record(
            config=_make_config_stub(),
            output_dir=output_dir,
            train_result={
                "total_epochs": 25,
                "best_epoch": 18,
                "best_val_metric": 0.42,
            },
            val_metrics={"loss": 0.5, "accuracy": 0.6},
            test_metrics={"loss": 0.55, "ic": 0.31},
            ledger_dir=ledger,
        )
        assert record_path is not None
        with open(record_path, "r") as f:
            doc = json.load(f)
        tm = doc["training_metrics"]
        assert tm["val_loss"] == 0.5
        assert tm["test_loss"] == 0.55
        assert tm["test_ic"] == 0.31
        assert tm["total_epochs"] == 25.0
        assert tm["best_epoch"] == 18.0
        assert tm["best_val_metric"] == 0.42


# ---------------------------------------------------------------------------
# Record contract regression
# ---------------------------------------------------------------------------


class TestPY223PartialRecordContract:
    """Lock the minimal-record shape against drift. Future cycles
    refactoring write_minimal_ledger_record must preserve these fields.
    """

    def test_record_contains_required_top_level_keys(self, tmp_path: Path):
        ledger = tmp_path / "ledger"
        ledger.mkdir()
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        record_path = write_minimal_ledger_record(
            config=_make_config_stub(name="contract_test"),
            output_dir=output_dir,
            ledger_dir=ledger,
        )
        assert record_path is not None
        with open(record_path, "r") as f:
            doc = json.load(f)
        # These keys MUST be present (queried by hft-ops ledger list).
        for required_key in (
            "experiment_id",
            "name",
            "fingerprint",
            "training_config",
            "training_metrics",
            "tags",
            "compatibility_fingerprint",  # may be None — but key must exist
            "feature_set_ref",
            "created_at",
            "status",
            "stages_completed",
        ):
            assert required_key in doc, (
                f"Required key {required_key!r} missing from partial record"
            )

    def test_record_status_is_completed(self, tmp_path: Path):
        """Direct trainer invocations that reach this helper have already
        completed training — record.status should reflect that."""
        ledger = tmp_path / "ledger"
        ledger.mkdir()
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        record_path = write_minimal_ledger_record(
            config=_make_config_stub(),
            output_dir=output_dir,
            ledger_dir=ledger,
        )
        assert record_path is not None
        with open(record_path, "r") as f:
            doc = json.load(f)
        assert doc["status"] == "completed"
        assert "training" in doc["stages_completed"]

    def test_atomic_write_no_partial_file(self, tmp_path: Path):
        """Atomic write — the record file either exists complete OR doesn't
        exist. No partial files in the records dir."""
        ledger = tmp_path / "ledger"
        ledger.mkdir()
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        record_path = write_minimal_ledger_record(
            config=_make_config_stub(name="atomic_test"),
            output_dir=output_dir,
            ledger_dir=ledger,
        )
        assert record_path is not None
        # No .tmp.* files in ledger.
        tmp_files = list(ledger.glob("*.tmp.*"))
        assert tmp_files == [], f"Found partial files: {tmp_files}"
