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


def _harvest_signal_metadata(
    output_dir: Path,
) -> Dict[str, Optional[str]]:
    """Best-effort harvest of trust columns from ``signal_metadata.json``.

    Returns a dict with the 3 trust-column keys (``compatibility_fingerprint``,
    ``model_config_hash``, ``feature_set_ref``) — values are ``None`` when
    the file is missing OR malformed OR the key is absent.

    Per hft-rules §8: malformed JSON / OS errors are surfaced via WARN log
    + the affected key set to ``None`` (never silently swallow).
    """
    out: Dict[str, Optional[Any]] = {
        "compatibility_fingerprint": None,
        "model_config_hash": None,
        "feature_set_ref": None,
        "_metadata_source": None,
    }

    output_dir_resolved = Path(output_dir).expanduser().resolve()
    for relpath in _SIGNAL_METADATA_CANDIDATES:
        candidate = output_dir_resolved / relpath
        if not candidate.exists():
            continue
        try:
            with open(candidate, "r") as f:
                metadata = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "ledger_hook: failed to load %s: %s — skipping harvest",
                candidate, exc,
            )
            continue

        if not isinstance(metadata, dict):
            logger.warning(
                "ledger_hook: %s root is not a dict (got %s); skipping harvest",
                candidate, type(metadata).__name__,
            )
            continue

        # Phase II compatibility block (lives under "compatibility" sub-key
        # post Phase Q.6.5). Harvest both nested and top-level for legacy
        # pre-Phase-II signal_metadata.json (data_source != "mbo_lob" path).
        compat_block = metadata.get("compatibility")
        if isinstance(compat_block, dict):
            fp = compat_block.get("fingerprint")
            if isinstance(fp, str):
                out["compatibility_fingerprint"] = fp

        # model_config_hash also lives at signal_metadata top level
        # (per Phase Y deployment 2026-05-05).
        mch = metadata.get("model_config_hash")
        if isinstance(mch, str):
            out["model_config_hash"] = mch

        # feature_set_ref top-level dict (per Phase 4 4c.4 propagation).
        fs_ref = metadata.get("feature_set_ref")
        if isinstance(fs_ref, dict):
            out["feature_set_ref"] = fs_ref

        out["_metadata_source"] = str(candidate)
        # First hit wins; do not look further.
        return out

    return out


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
        # Resolve ledger directory.
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
        from hft_contracts.atomic_io import atomic_write_json
        from hft_contracts.canonical_hash import canonical_json_blob, sha256_hex
        from hft_contracts.experiment_record import ExperimentRecord
        from hft_contracts.provenance import Provenance, GitInfo

        # Timestamp + name.
        now = timestamp or datetime.now(timezone.utc)
        timestamp_str = now.strftime("%Y%m%dT%H%M%S")
        name = getattr(config, "name", "unnamed")

        # Compute a stable fingerprint over the config dict. This is NOT
        # the orchestrator's structural fingerprint (which hashes a
        # resolved manifest); it's a content-derived identity so the
        # ledger record dedupes deterministically on repeat invocations
        # of the same config. Not used for cross-cycle dedup (records
        # written by this helper don't compete with orchestrator-written
        # records).
        config_dict = _config_to_dict(config)
        fingerprint = sha256_hex(canonical_json_blob(config_dict))
        experiment_id = f"{name}_{timestamp_str}_{fingerprint[:8]}"

        # Harvest trust columns from signal_metadata.json if present.
        harvested = _harvest_signal_metadata(output_dir)
        compatibility_fingerprint = harvested["compatibility_fingerprint"]
        model_config_hash = harvested["model_config_hash"]
        feature_set_ref = harvested["feature_set_ref"]
        signal_metadata_source = harvested["_metadata_source"]

        # Build training_config payload. We embed the resolved config
        # dict + optionally the model_config_hash trust column (mirroring
        # the orchestrator's pattern at cli.py:747).
        training_config: Dict[str, Any] = dict(config_dict)
        if model_config_hash is not None:
            training_config["model_config_hash"] = model_config_hash

        # Flatten training_metrics (val_* + test_* + best-val-metric).
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

        # Minimal Provenance — no git context, no data_dir hash (those
        # would require subprocess calls the orchestrator handles via
        # build_provenance). The orchestrator-side ``hft-ops ledger
        # rebuild-index`` path can re-derive provenance from a fuller
        # context if needed later.
        contract_version = getattr(config, "contract_version", "3.0")
        try:
            provenance = Provenance(
                git=GitInfo(commit_hash="", branch="", dirty=False, short_hash=""),
                config_hashes={"trainer": fingerprint},
                data_dir_hash=None,
                contract_version=contract_version,
                timestamp_utc=now.isoformat(),
                retroactive=False,
                schema_version="1.0",
            )
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning(
                "ledger_hook: Provenance construction failed (%s); "
                "skipping ledger write",
                exc,
            )
            return None

        # Tags — pull from config if available.
        tags = list(getattr(config, "tags", []) or [])

        # Construct ExperimentRecord. Only fields documented as required by
        # ``hft_contracts.experiment_record.ExperimentRecord`` are populated;
        # the rest fall through to defaults.
        record = ExperimentRecord(
            experiment_id=experiment_id,
            name=name,
            manifest_path="",  # direct invocation — no manifest
            fingerprint=fingerprint,
            feature_set_ref=feature_set_ref,
            compatibility_fingerprint=compatibility_fingerprint,
            signal_export_output_dir=signal_metadata_source,
            provenance=provenance,
            contract_version=contract_version,
            training_config=training_config,
            training_metrics=training_metrics,
            tags=tags,
            created_at=now.isoformat(),
            duration_seconds=duration_seconds,
            status="completed",
            stages_completed=["training"],
        )

        # Atomic write — fails loud on OSError per atomic_io.py contract.
        #
        # Timestamp-collision semantic: same name + same timestamp + same
        # config fingerprint produces the same ``experiment_id`` → atomic
        # write OVERWRITES the prior record. Acceptable per the
        # ``atomic_write_json`` contract (it is overwrite-safe); operators
        # re-running the same config within a single second get a single
        # canonical record.
        record_path = records_dir / f"{experiment_id}.json"
        record_dict = record.to_dict()
        atomic_write_json(record_path, record_dict)
        logger.info(
            "ledger_hook: wrote minimal ExperimentRecord to %s "
            "(compat_fp=%s, mch=%s)",
            record_path,
            "✓" if compatibility_fingerprint else "absent",
            "✓" if model_config_hash else "absent",
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
