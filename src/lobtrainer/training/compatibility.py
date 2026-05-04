"""Shared producer of CompatibilityContract + model_config_hash for the
checkpoint + signal-metadata artifacts.

Phase X.1 v2 (2026-05-04): extracts ``_build_compatibility_contract`` from
``lobtrainer.export.exporter`` (where it lived since Phase II 2026-04-20)
to a shared module. Adds ``compute_model_config_hash`` for the checkpoint
plane. Both reuse ``hft_contracts.compatibility.CompatibilityContract`` and
``hft_contracts.canonical_hash`` SSoTs — NO new canonical-hash site.

Architectural pattern: extends Phase Q's "loader-validates-artifact"
discipline (already wired for ``signal_metadata.json``) to the checkpoint
plane via the same primitive (CompatibilityContract.fingerprint() +
diff()). The ``model_config_hash`` sub-field captures
architecturally-load-bearing model.params keys filtered by
``_LOSS_TUNING_KEYS`` denylist, so loss-tuning changes do NOT trip
checkpoint-resume warnings.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, FrozenSet, Optional, TYPE_CHECKING

from lobtrainer.config.paths import resolve_labels_config

if TYPE_CHECKING:
    from lobtrainer.config.schema import ExperimentConfig, ModelConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase X.1 v2 exception types — consumed by Trainer.load_checkpoint and
# SimpleModelTrainer.load_checkpoint to surface fingerprint mismatches.
# ---------------------------------------------------------------------------

class CheckpointConfigMismatchError(ValueError):
    """Raised when checkpoint compatibility/model_config_hash differs from
    active config and ``strict_config=True`` is passed to load_checkpoint.

    Phase X.1 v2 (2026-05-04). Default load_checkpoint behavior is warn-only
    (CheckpointConfigMismatchWarning); strict mode promotes to this error.
    Phase X.4 will flip the default to ``strict_config=True`` once all
    in-flight checkpoints have fingerprints (per §I.9 promotion gate).
    """


class CheckpointConfigMismatchWarning(UserWarning):
    """Emitted when checkpoint compatibility/model_config_hash differs from
    active config (warn mode — Phase X.1 v2 default).

    Carries an actionable ``CompatibilityContract.diff()``-formatted message
    naming the differing fields (feature_count, window_size, horizons, etc.)
    or the model_config_hash sub-field (hidden_dim, num_layers, num_heads,
    dropout, hmhp_pool_mode, attention, use_bin, etc.).
    """


class CheckpointMissingFingerprintWarning(UserWarning):
    """Emitted when a checkpoint lacks Phase X.1 v2 contract fields.

    Pre-X.1 artifacts (e.g., the 2026-03-15 HMHP-R checkpoint at
    ``outputs/experiments/nvda_hmhp_regression_h10_primary/checkpoints/best.pt``)
    have neither ``compatibility_fingerprint`` nor ``model_config_hash`` keys,
    so cross-config validation is impossible. Operators should re-train
    affected checkpoints (see PHASE_P_BACKLOG.md F-12 entry) or opt out via
    ``strict_config=False`` (the default — emits this warning, not raise).
    Same warning is also raised for partial-write / older-schema sidecars
    that parse cleanly but lack the contract keys.
    """


# ---------------------------------------------------------------------------
# Phase X.1 v2 _LOSS_TUNING_KEYS denylist — keys EXCLUDED from
# model_config_hash because they affect loss/training, not architecture.
#
# Curated list (verified post-Agent 4 Q7 sanity check 2026-05-04):
#   - Mutating any of these should NOT invalidate a checkpoint's fingerprint.
#   - num_classes is INTENTIONALLY NOT in the denylist (it determines output
#     head dimension — architectural).
#   - task_type IS in denylist because CompatibilityContract already captures
#     the same axis via compute_label_strategy_hash(LabelsConfig).
# ---------------------------------------------------------------------------

_LOSS_TUNING_KEYS: FrozenSet[str] = frozenset({
    # Loss-function hyperparameters per _build_params_from_legacy(schema.py:1610-1729)
    "gmadl_a", "gmadl_b",                            # GMADL (tlob/hmhp)
    "regression_loss_type", "regression_loss_delta", # regression loss type + delta
    "loss_type",                                     # hmhp_regression loss_type variant
    "loss_weights",                                  # per-horizon weights
    "huber_delta", "pinball_quantiles",              # other loss tuning
    # Auto-derived data axes — already captured by CompatibilityContract.feature_count
    # / window_size / horizons / label_strategy_hash. Excluded here to avoid
    # double-counting and to prevent the "internal refactor of auto-derivation"
    # silently rotating every fingerprint.
    "task_type",
    "num_features",
    "sequence_length",
    "input_size",
    # ---------------------------------------------------------------------
    # NOTE INTENTIONALLY NOT IN DENYLIST (must trip model_config_hash):
    #   - num_classes — output-head dimension; classification 2-class vs
    #     3-class IS architectural and is NOT in CompatibilityContract.
    #   - hidden_dim, num_layers, num_heads, dropout, hmhp_pool_mode,
    #     attention, use_bin, cascade_mode, state_fusion — all architectural.
    # ---------------------------------------------------------------------
})


# ---------------------------------------------------------------------------
# Phase X.1 v2 build_compatibility_contract (moved from exporter.py:46-119
# 2026-05-04). Body preserved verbatim modulo docstring updates.
# ---------------------------------------------------------------------------


def derive_data_source(data_dir: Any) -> str:
    """Infer the ``data_source`` tag for CompatibilityContract from the
    export directory path.

    Phase II (2026-04-20) heuristic: directory name prefixed with ``basic_``
    is the off-exchange (CMBP-1) pipeline; otherwise MBO LOB. Documented as
    the canonical "infer from convention" fallback until each DataConfig
    carries an explicit ``data_source: Literal[...]`` field. Future
    DataConfig schema bump should make this explicit — see root CLAUDE.md
    Change-Coordination Checklist (Phase II → Phase III).
    """
    name = Path(str(data_dir)).name
    if name.startswith("basic_"):
        return "off_exchange"
    return "mbo_lob"


def build_compatibility_contract(
    config: Any,
    feature_set_ref: Optional[Dict[str, str]] = None,
    calibration_method: Optional[str] = None,
) -> Optional[Any]:
    """Construct a ``hft_contracts.compatibility.CompatibilityContract`` from
    an ExperimentConfig.

    Returns ``None`` when ``hft_contracts`` import fails (pre-Phase-II
    environments) so callers gracefully omit the new metadata block.
    Production consumers see the full 11-key block; legacy consumers ignore
    the unrecognized key.

    Phase X.1 v2 (2026-05-04): moved from ``lobtrainer.export.exporter``
    (was ``_build_compatibility_contract``). Now consumed by both signal
    export (existing) and checkpoint save/load (new). Body preserved
    verbatim modulo docstring + name (dropped underscore prefix since the
    function is now a public reusable primitive).
    """
    try:
        from hft_contracts import SCHEMA_VERSION
        from hft_contracts.compatibility import (
            CompatibilityContract,
            compute_label_strategy_hash,
        )
    except Exception as exc:  # broad: package missing / import chain broken
        logger.warning(
            "CompatibilityContract not constructed — hft_contracts unavailable: %s", exc
        )
        return None

    # Phase A (2026-04-23): explicit defensive reads via ``resolve_labels_config``.
    # The inner ``try / except Exception`` was deleted — any path-drift (wrong
    # ``config.labels`` access, missing ``DataConfig.data`` attribute, etc.)
    # now raises ``AttributeError`` loudly at the helper boundary. Silent-None
    # was the anti-pattern that masked the Phase II producer-path bug cluster.
    labels_cfg = resolve_labels_config(config)

    feature_layout = (
        feature_set_ref["content_hash"]
        if feature_set_ref and "content_hash" in feature_set_ref
        else "default"
    )
    horizons_list = labels_cfg.horizons or getattr(
        config.model, "hmhp_horizons", None
    )
    horizons_tuple = tuple(horizons_list) if horizons_list else None

    # Label strategy hash captures the full LabelsConfig — strategy + horizons +
    # thresholds + smoothing — so parameter variations don't collide under a
    # flat strategy-name string.
    label_hash = compute_label_strategy_hash(labels_cfg)

    # Safely coerce feature_count — dataset configs use slightly different
    # attribute names across asset classes (``feature_count`` vs ``num_features``).
    feature_count = int(
        getattr(config.data, "feature_count", None)
        or getattr(config.data, "num_features", 0)
    )
    sequence_cfg = getattr(config.data, "sequence", None)
    window_size = int(
        getattr(sequence_cfg, "window_size", None)
        or getattr(config.data, "window_size", None)
        or getattr(config.data, "sequence_length", 0)
    )
    if feature_count == 0 or window_size == 0:
        logger.warning(
            "CompatibilityContract has zero feature_count (%d) or window_size (%d) — "
            "config surface may have drifted; check DataConfig attribute names.",
            feature_count, window_size,
        )

    return CompatibilityContract(
        contract_version=str(SCHEMA_VERSION),
        schema_version=str(SCHEMA_VERSION),
        feature_count=feature_count,
        window_size=window_size,
        feature_layout=str(feature_layout),
        data_source=derive_data_source(config.data.data_dir),
        label_strategy_hash=label_hash,
        calibration_method=calibration_method,
        primary_horizon_idx=labels_cfg.primary_horizon_idx,
        horizons=horizons_tuple,
        normalization_strategy=str(
            getattr(config.data.normalization, "strategy", "unknown")
        ),
    )


# ---------------------------------------------------------------------------
# Phase X.1 v2 compute_model_config_hash — SHA-256 hex over canonical-JSON
# of (model_type, params filtered by _LOSS_TUNING_KEYS denylist). Reuses
# hft_contracts.canonical_hash SSoT (canonical_json_blob + sha256_hex).
# ---------------------------------------------------------------------------


def compute_model_config_hash(model_config: Any) -> str:
    """SHA-256 hex over canonical-JSON of model.params (filtered for arch
    axis) + model_type.

    Mutating loss-tuning keys (``_LOSS_TUNING_KEYS`` denylist — gmadl_a/b,
    regression_loss_type, regression_loss_delta, loss_type, loss_weights,
    huber_delta, pinball_quantiles, task_type, num_features, sequence_length,
    input_size) does NOT change the hash. Only architectural keys
    (hidden_dim, num_layers, num_heads, dropout, hmhp_pool_mode, attention,
    use_bin, cascade_mode, num_classes, etc.) participate.

    Reuses ``hft_contracts.canonical_hash`` SSoT — NO new canonical-form site.
    Same canonical-JSON + sha256_hex primitives that produce
    ``CompatibilityContract.fingerprint()``,
    ``hft_contracts.feature_sets.hashing.compute_feature_set_hash``,
    ``hft_contracts.experiment_record.compatibility_fingerprint``, etc.

    Args:
        model_config: ``ModelConfig`` (Pydantic v2 SafeBaseModel) with
            ``model_type`` (Enum or str) and ``params: Dict[str, Any]``.

    Returns:
        64-character lowercase-hex SHA-256 digest. Stable across pickles
        (canonical-JSON sort_keys=True).
    """
    from hft_contracts.canonical_hash import canonical_json_blob, sha256_hex

    model_type_value = (
        model_config.model_type.value
        if hasattr(model_config.model_type, "value")
        else model_config.model_type
    )
    arch_params = {
        k: v for k, v in dict(model_config.params).items()
        if k not in _LOSS_TUNING_KEYS
    }
    return sha256_hex(canonical_json_blob({
        "model_type": str(model_type_value),
        "params": arch_params,
    }))
