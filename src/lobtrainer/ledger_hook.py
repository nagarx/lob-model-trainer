"""Direct-trainer-invocation ledger-write helper (#PY-223 MINIMAL Step 2).

Closes the R-17a-class ~26% experiment-invisibility gap surfaced by Wave 1
Agent E + Wave 2 W2-2 (2026-05-14). Per Wave 3 W3-1 empirical reframing:
the ValidationStage rejects-1-D-labels premise is FALSE; bypassing the
orchestrator for classification cycles is a DEPRECATION CHOICE
(per CLAUDE.md "supported entry point because it records gate_report into
ledger fingerprint"), not a forced workaround. Until the deprecation
runway closes, direct trainer invocations need a way to write a minimal
``ExperimentRecord`` so that ``hft-ops ledger list`` queries cover the
class — without re-implementing the full ``cli._record_experiment``
orchestrator surface (manifest + Phase Y composer + Provenance compose +
trust-column harvester = ~250 LOC).

This module is the MINIMAL Step 2 ship (per user authorization 2026-05-14
post FIND-070 closure cycle): harvest-from-signal-metadata when available,
partial-record otherwise. Default-OFF (opt-in via ``--record-to-ledger``)
to avoid surprising operators.

Architectural design:

- ``write_minimal_ledger_record`` is a single pure function — no class
  surface, no hidden state.
- Reuses ``hft_contracts.atomic_io.atomic_write_json`` SSoT for atomic
  writes (no in-module re-implementation).
- Reuses ``hft_contracts.experiment_record.ExperimentRecord`` dataclass
  (no record-shape duplication).
- Reuses ``hft_contracts.canonical_hash.{canonical_json_blob, sha256_hex}``
  SSoT for fingerprint computation.
- Falls back gracefully when signal_metadata.json is missing (trainer ran
  but signals haven't been exported yet) — the record will be PARTIAL
  but valid for ledger queries.
- Returns ``None`` when ledger_dir can't be located OR write fails —
  caller decides how to react (typically log + continue).

Per hft-rules §5 fail-fast: when the caller explicitly opts into
ledger-write (``write=True``) but the ledger directory cannot be located,
the helper logs a WARN and returns None — never silently degrades the
training run.

Per hft-rules §8: harvest errors (malformed signal_metadata.json,
non-existent path) are surfaced via ``logger.warning`` with explicit
diagnostic; never swallowed.

Per hft-rules §0 reuse-first: ZERO new SSoT primitives created. Mirrors
``hft-ops/src/hft_ops/cli.py::_record_experiment`` semantics for the
narrow direct-invocation subset.
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


# Common locations where ``scripts/export_signals.py`` writes
# ``signal_metadata.json``. The helper tries them in order; first hit wins.
# These paths are RELATIVE to ``output_dir``.
_SIGNAL_METADATA_CANDIDATES = (
    "signal_metadata.json",
    "signals/test/signal_metadata.json",
    "signals/val/signal_metadata.json",
    "signals/train/signal_metadata.json",
)


def _resolve_ledger_dir(
    explicit_dir: Optional[Path],
    output_dir: Path,
    pipeline_root: Optional[Path] = None,
) -> Optional[Path]:
    """Locate the hft-ops ledger records directory.

    Priority order:
      1. ``explicit_dir`` (operator-provided ``--ledger-dir`` flag)
      2. ``<pipeline_root>/hft-ops/ledger/records`` (explicit pipeline_root)
      3. Climb from ``output_dir`` upward looking for ``hft-ops/ledger/records``

    Returns ``None`` when none of the above resolve. Caller MUST treat
    ``None`` as opt-out (skip the write) per the helper's graceful-failure
    contract.
    """
    if explicit_dir is not None:
        resolved = Path(explicit_dir).expanduser().resolve()
        if resolved.exists():
            return resolved
        logger.warning(
            "ledger_hook: --ledger-dir=%s does not exist; trying fallback",
            resolved,
        )

    if pipeline_root is not None:
        candidate = Path(pipeline_root).expanduser().resolve() / "hft-ops" / "ledger" / "records"
        if candidate.exists():
            return candidate
        logger.warning(
            "ledger_hook: pipeline_root=%s has no hft-ops/ledger/records; trying climb",
            pipeline_root,
        )

    # Climb from output_dir looking for hft-ops/ledger/records sibling.
    current = Path(output_dir).expanduser().resolve()
    for _ in range(8):  # bounded climb to avoid runaway upward traversal
        candidate = current / "hft-ops" / "ledger" / "records"
        if candidate.exists():
            return candidate
        if current.parent == current:  # filesystem root
            break
        current = current.parent

    logger.warning(
        "ledger_hook: could not locate hft-ops/ledger/records (tried explicit + "
        "pipeline_root + climb from %s up to 8 levels)",
        output_dir,
    )
    return None


def _find_signal_metadata_path(output_dir: Path) -> Optional[Path]:
    """Return the first existing ``signal_metadata.json`` under ``output_dir``.

    Probes ``_SIGNAL_METADATA_CANDIDATES`` in priority order. Returns the
    resolved absolute Path on first hit, ``None`` otherwise.

    Phase 8D / #PY-223 SSoT migration (2026-05-14): trust-column harvest
    itself now delegates to
    :func:`hft_contracts.experiment_recorder.harvest_trust_columns_from_signal_metadata`
    inside :func:`write_minimal_ledger_record` — this helper retains the
    trainer-local probe semantics ONLY (probe order + multi-split fallback;
    trainer is the only consumer that needs this).
    """
    output_dir_resolved = Path(output_dir).expanduser().resolve()
    for relpath in _SIGNAL_METADATA_CANDIDATES:
        candidate = output_dir_resolved / relpath
        if candidate.exists():
            return candidate
    return None


def _config_to_dict(config: Any) -> Dict[str, Any]:
    """Render an ``ExperimentConfig`` to a plain dict for ledger storage.

    Supports both Pydantic v2 BaseModel (``model_dump``) and dataclass
    paths. Falls back to an empty dict if neither works — the record will
    have an empty ``training_config`` but stays valid.
    """
    # Pydantic v2 path (Phase A.5.3i — ExperimentConfig is SafeBaseModel)
    if hasattr(config, "model_dump") and callable(config.model_dump):
        try:
            return config.model_dump(mode="json")
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning(
                "ledger_hook: config.model_dump failed (%s); falling back to "
                "empty config dict",
                exc,
            )
            return {}

    # Dataclass path (legacy)
    try:
        from dataclasses import asdict, is_dataclass
        if is_dataclass(config):
            return asdict(config)
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning(
            "ledger_hook: dataclasses.asdict failed (%s); falling back to "
            "empty config dict",
            exc,
        )

    return {}


def _flatten_metric_dict(
    metrics: Optional[Any],
    prefix: str,
) -> Dict[str, float]:
    """Flatten a trainer eval result into ``{<prefix><key>: float}``.

    Mirrors the convention in ``scripts/train.py::_dump_test_metrics`` so
    the keys land in ``ExperimentRecord.index_entry()`` whitelist.
    Non-scalar values (numpy arrays, dicts, strings) are dropped per the
    flat-key contract documented at experiment_record.py:619-641.
    """
    if metrics is None:
        return {}

    # ClassificationMetrics has ``.to_dict()``; regression returns a plain dict.
    if hasattr(metrics, "to_dict") and callable(metrics.to_dict):
        flat = metrics.to_dict()
    elif isinstance(metrics, dict):
        flat = metrics
    else:
        return {}

    out: Dict[str, float] = {}
    for key, value in flat.items():
        if isinstance(value, bool) or value is None:
            continue
        try:
            out[f"{prefix}{key}"] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def write_minimal_ledger_record(
    *,
    config: Any,
    output_dir: Path,
    train_result: Optional[Dict[str, Any]] = None,
    val_metrics: Optional[Any] = None,
    test_metrics: Optional[Any] = None,
    ledger_dir: Optional[Path] = None,
    pipeline_root: Optional[Path] = None,
    duration_seconds: Optional[float] = None,
    timestamp: Optional[datetime] = None,
) -> Optional[Path]:
    """Write a minimal ``ExperimentRecord`` JSON file to the hft-ops ledger.

    Closes #PY-223 R-17a-class invisibility gap (~26% of recent experiment
    directories lack orchestrator-side ledger records). Per Wave 2 NEW-3,
    this helper COEXISTS with ``save_config(line 371 in scripts/train.py)``
    — it does NOT replace the config sidecar; rather, it ADDS an
    ``ExperimentRecord`` entry that the operator can query via
    ``hft-ops ledger list --provenance-hash <hex>`` and similar.

    Per hft-rules §8: this helper NEVER silently degrades the training
    run. If the ledger directory cannot be located OR write fails, it
    logs a WARN diagnostic and returns ``None`` (caller continues).

    Args:
        config: ``ExperimentConfig`` instance (Pydantic v2 BaseModel or
            legacy dataclass).
        output_dir: trainer's output directory (where final.pt was saved).
        train_result: optional dict returned by ``trainer.train()``
            (``total_epochs`` / ``best_epoch`` / ``best_val_metric``).
            None when called from eval-only path.
        val_metrics: optional final-eval val metrics
            (``ClassificationMetrics`` OR regression dict).
        test_metrics: optional final-eval test metrics.
        ledger_dir: explicit override for the records directory. When
            ``None``, the helper tries ``<pipeline_root>/hft-ops/ledger/records``
            then climbs from ``output_dir``.
        pipeline_root: optional explicit pipeline root for ledger
            location fallback.
        duration_seconds: optional total training duration.
        timestamp: optional UTC timestamp; defaults to ``datetime.now(utc)``.

    Returns:
        ``Path`` to the written record file, or ``None`` if the write
        was skipped (no ledger dir found) OR failed (caught + logged).
    """
    try:
        # Resolve ledger directory (trainer-local: handles operator-provided
        # --ledger-dir + climb-from-output_dir fallback).
        records_dir = _resolve_ledger_dir(
            explicit_dir=ledger_dir,
            output_dir=output_dir,
            pipeline_root=pipeline_root,
        )
        if records_dir is None:
            return None
        records_dir.mkdir(parents=True, exist_ok=True)

        # Lazy imports to keep module-load surface small (matches
        # SimpleModelTrainer pattern at simple_trainer.py:773).
        from hft_contracts.canonical_hash import canonical_json_blob, sha256_hex
        from hft_contracts.experiment_recorder import record_from_artifacts

        # Compute a stable fingerprint over the config dict. This is NOT
        # the orchestrator's structural fingerprint (which hashes a
        # resolved manifest); it's a content-derived identity so the
        # ledger record dedupes deterministically on repeat invocations
        # of the same config. Not used for cross-cycle dedup (records
        # written by this helper don't compete with orchestrator-written
        # records).
        config_dict = _config_to_dict(config)
        fingerprint = sha256_hex(canonical_json_blob(config_dict))

        # Probe for signal_metadata.json across the 4 candidate paths
        # (Phase 8D / #PY-223 retains trainer-local probe semantics; the
        # actual HARVEST is delegated to the SSoT below).
        signal_metadata_path = _find_signal_metadata_path(output_dir)

        # Flatten training_metrics (val_* + test_* + best-val-metric).
        # Mirrors scripts/train.py::_dump_test_metrics flat-scalar convention.
        training_metrics: Dict[str, Any] = {}
        training_metrics.update(_flatten_metric_dict(val_metrics, prefix="val_"))
        training_metrics.update(_flatten_metric_dict(test_metrics, prefix="test_"))
        if train_result is not None:
            for key in ("total_epochs", "best_epoch"):
                value = train_result.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    training_metrics[key] = float(value)
            best_val = train_result.get("best_val_metric")
            if isinstance(best_val, (int, float)) and not isinstance(best_val, bool):
                training_metrics["best_val_metric"] = float(best_val)

        # Tags from config (Pydantic v2 SafeBaseModel OR legacy attr).
        tags = list(getattr(config, "tags", []) or [])

        # Phase 8D / #PY-223 (2026-05-14): delegate ExperimentRecord
        # construction + trust-column harvest + Phase Y composer to the
        # hft-contracts SSoT (v2.8.0+; commit d773ac4).
        #
        # The SSoT handles:
        #   - harvest_trust_columns_from_signal_metadata (when path provided)
        #     — populates feature_set_ref + compatibility_fingerprint +
        #     model_config_hash from signal_metadata.json (graceful None when
        #     file missing OR pre-Phase-Y schema OR malformed)
        #   - build_provenance (with empty git_info if not in repo; full
        #     git capture when pipeline_root is a git working tree). The
        #     trainer-side path computes data_dir_hash when data_dir
        #     resolved — empty when not.
        #   - ExperimentRecord construction with all 22+ fields
        #   - Phase Y compute_experiment_provenance_hash composition AND
        #     graceful-None when sources missing + WARN diagnostic
        #   - model_config_hash injection into training_config (nested
        #     location read by Phase Y composer)
        #
        # ledger_path: ABSOLUTE path that record_from_artifacts uses as the
        # parent of "records/" subdir. _resolve_ledger_dir returns the
        # records/ dir directly; pass its parent.
        contract_version = getattr(config, "contract_version", "3.0")
        now = timestamp or datetime.now(timezone.utc)

        # The trainer-side data_dir is config.data.data_dir (when accessible).
        # build_provenance needs Path-like. Graceful None: SSoT handles.
        data_dir = None
        cfg_data = getattr(config, "data", None)
        if cfg_data is not None:
            raw_data_dir = getattr(cfg_data, "data_dir", None)
            if raw_data_dir:
                data_dir = Path(raw_data_dir).expanduser().resolve()

        # Hypothesis / description / pipeline_root with graceful fallbacks.
        # pipeline_root resolution: explicit > climb-from-records_dir
        # (records_dir.parent.parent is typically <pipeline_root>/hft-ops/ledger).
        if pipeline_root is None:
            pipeline_root = records_dir.parent.parent.parent  # records/ → ledger/ → hft-ops/ → root

        # ledger_path=None: SSoT constructs the record + composes Phase Y
        # composer but does NOT write. We save manually below to preserve
        # the trainer-local `records_dir` semantic where _resolve_ledger_dir
        # returns the FILE-PARENT directory (not a parent-of-records-dir).
        # This avoids SSoT's `<ledger_path>/records/<id>.json` layout
        # convention that would land the file one directory deeper than
        # the trainer-local helper expects.
        record = record_from_artifacts(
            name=getattr(config, "name", "unnamed"),
            pipeline_root=Path(pipeline_root),
            contract_version=contract_version,
            fingerprint=fingerprint,
            signal_metadata_path=signal_metadata_path,
            training_metrics=training_metrics,
            training_config=config_dict,
            data_dir=data_dir,
            trainer_config_dict=config_dict,
            stages_completed=["training"],
            status="completed",
            duration_seconds=duration_seconds if duration_seconds is not None else 0.0,
            tags=tags,
            ledger_path=None,  # save manually below to preserve trainer path semantic
        )

        # Save manually to records_dir per trainer-local path contract.
        # Uses ExperimentRecord.save() → hft_contracts.atomic_io.atomic_write_json
        # SSoT (tmp + fsync + os.replace; overwrite-safe per
        # ``atomic_write_json`` documented contract).
        record_path = records_dir / f"{record.experiment_id}.json"
        record.save(record_path)
        logger.info(
            "ledger_hook: wrote minimal ExperimentRecord to %s "
            "(compat_fp=%s, epH=%s)",
            record_path,
            "✓" if record.compatibility_fingerprint else "absent",
            "✓" if record.experiment_provenance_hash else "absent",
        )
        return record_path

    except Exception as exc:  # pragma: no cover — observation-tier wrap
        # Per hft-rules §8: never silently swallow. Log with traceback +
        # return None so the training run continues unaffected.
        logger.warning(
            "ledger_hook: write_minimal_ledger_record failed (%s: %s); "
            "training run not affected. Traceback:\n%s",
            type(exc).__name__,
            exc,
            "".join(traceback.format_exception(*sys.exc_info())),
        )
        return None
