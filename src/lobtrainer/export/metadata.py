"""
Signal metadata builder for signal export.

Produces a superset metadata JSON that includes everything all 3 old
export scripts produced, plus additional provenance fields.

Phase O Cycle 1 (H-1, 2026-05-04): emits ``schema_version`` and
``contract_version`` at the manifest root so the backtester's loader can
fail-loud on schema-version skew between the trainer and the consumer.
The previously-present Phase II ``compatibility`` block tracks tamper
detection but did not surface schema-version drift to non-Phase-II
backtester paths; these two top-level fields close that gap.
"""

from datetime import datetime, timezone
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from hft_contracts import SCHEMA_VERSION as _CONTRACT_SCHEMA_VERSION


def build_signal_metadata(
    *,
    # Core identification
    model_type: str,
    model_name: str,
    parameters: int,
    signal_type: str,
    split: str,
    total_samples: int,
    checkpoint: str,
    config_path: Optional[str] = None,
    horizons: Optional[List[int]] = None,
    primary_horizon: Optional[str] = None,
    horizon_idx: Optional[int] = None,
    data_dir: Optional[str] = None,
    # Feature configuration
    feature_preset: Optional[str] = None,
    feature_set_ref: Optional[Dict[str, str]] = None,
    normalization_strategy: Optional[str] = None,
    # Prediction statistics
    prediction_stats: Optional[Dict[str, float]] = None,
    # Raw feature statistics
    spread_stats: Optional[Dict[str, float]] = None,
    price_stats: Optional[Dict[str, float]] = None,
    # Classification-specific
    predictions_distribution: Optional[Dict[str, int]] = None,
    agreement_stats: Optional[Dict[str, Any]] = None,
    confirmation_percentiles: Optional[Dict[str, float]] = None,
    directional_rate: Optional[float] = None,
    # Regression-specific
    metrics: Optional[Dict[str, float]] = None,
    calibration: Optional[Dict[str, Any]] = None,
    # Phase Q.7 (2026-05-04): sklearn-path feature engineering config
    # (TemporalFeatureConfig.to_dict()). Emitted only when explicitly
    # provided; PyTorch path passes None.
    feature_config: Optional[Dict[str, Any]] = None,
    # Phase II (2026-04-20): cross-module CompatibilityContract.
    # When provided, emits the 11-key contract block + its SHA-256 fingerprint
    # into signal_metadata.json. Backtester validates by recomputing from the
    # block and comparing to the stored fingerprint (tamper detection) and
    # optionally comparing to a consumer-derived expected fingerprint
    # (contract version skew detection). See hft_contracts.compatibility.
    compatibility: Optional[Any] = None,  # CompatibilityContract (late-imported to avoid hard dep)
    # Phase II (2026-04-20): these two are EXTRACTED from compatibility for
    # quick inspection / back-compat; the manifest consumer reads them directly.
    # ``data_source`` is also a top-level CompatibilityContract field; emitting
    # it at the root of signal_metadata.json makes legacy consumers (pre-Phase-II)
    # see the data provenance without parsing the nested block.
    # ``calibration_method`` is the D10 fix — manifest-authoritative gate that
    # the backtester uses to choose between predicted_returns.npy vs
    # calibrated_returns.npy (replaces silent file-existence precedence).
    data_source: Optional[str] = None,
    calibration_method: Optional[str] = None,
    # Phase Y deployment (2026-05-05): emit ``model_config_hash`` at signal-
    # metadata root so hft-ops harvester can read it back via the existing
    # signal_metadata.json contract (mirrors compatibility_fingerprint pattern).
    # Pre-Phase-Y, ``model_config_hash`` was written ONLY to the checkpoint
    # sidecar (``<ckpt>.pt`` dict + ``<ckpt>.pkl.config.json``) which hft-ops
    # never reads — leaving the 4th of 4 ``experiment_provenance_hash`` source
    # fields silently None on every record. This emission closes the cross-
    # repo harvest gap. Producer: caller computes via
    # ``lobtrainer.training.compatibility.compute_model_config_hash(model_config)``.
    # Consumer: hft-ops ``_harvest_model_config_hash`` reads + injects into
    # ``record.training_config["model_config_hash"]`` for the Phase D
    # ``compute_experiment_provenance_hash`` composer. Per hft-rules §1
    # "single source of truth" + §0 reuse-first.
    model_config_hash: Optional[str] = None,
    # Phase Y / γ-1 LITE / #PY-88 (2026-05-10): top-level ``return_type``
    # human-visible field — ``LabelsConfig.return_type`` value (one of
    # ``smoothed_return | point_return | mean_return | peak_return`` per
    # ``LabelFactory._RETURN_FUNCTIONS``). The ``compatibility_fingerprint``
    # already encodes this via ``compute_label_strategy_hash``'s
    # ``model_dump()`` payload (opaque SHA-256), but a top-level string
    # field aids backtester / ledger / dashboard queries that filter by
    # return_type axis without parsing the nested ``compatibility`` block.
    # Additive; pre-#PY-88 consumers ignore. Validated as a known
    # return_type by the producer's ``LabelsConfig._VALID_RETURN_TYPES``
    # ClassVar.
    return_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Build comprehensive signal metadata JSON.

    Superset of all 3 old export scripts' metadata schemas, plus
    additional provenance fields for experiment tracking.

    All fields are keyword-only to prevent positional mistakes.
    """
    meta: Dict[str, Any] = {
        # Phase O Cycle 1 (H-1, 2026-05-04): contract pin. Required by
        # backtester loader.py post-C-4 (presence check; raises on absence).
        "schema_version": str(_CONTRACT_SCHEMA_VERSION),
        "contract_version": str(_CONTRACT_SCHEMA_VERSION),
        # Core (always present)
        "model_type": model_type,
        "model_name": model_name,
        "parameters": parameters,
        "signal_type": signal_type,
        "split": split,
        "total_samples": total_samples,
        "checkpoint": checkpoint,
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }

    # Optional core fields
    if config_path is not None:
        meta["config"] = config_path
    if horizons is not None:
        meta["horizons"] = horizons
    if primary_horizon is not None:
        meta["primary_horizon"] = primary_horizon
    if horizon_idx is not None:
        meta["horizon_idx"] = horizon_idx
    if data_dir is not None:
        meta["data_dir"] = data_dir

    # Feature configuration
    if feature_preset is not None:
        meta["feature_preset"] = feature_preset
    if feature_set_ref is not None:
        # Phase 4 Batch 4c.4: {name, content_hash} reference to FeatureSet
        # registry. Read-only propagation; backtester does NOT recompute
        # content_hash (integrity is the resolver's job at trainer load).
        meta["feature_set_ref"] = feature_set_ref
    if normalization_strategy is not None:
        meta["normalization_strategy"] = normalization_strategy
    if feature_config is not None:
        # Phase Q.7 (2026-05-04): the sklearn trainer's TemporalFeatureConfig
        # dict — emitted only by the sklearn (SimpleModelTrainer) path.
        # PyTorch trainers leave this None.
        meta["feature_config"] = feature_config

    # Statistics (always include when available)
    if prediction_stats is not None:
        meta["prediction_stats"] = prediction_stats
    if spread_stats is not None:
        meta["spread_stats"] = spread_stats
    if price_stats is not None:
        meta["price_stats"] = price_stats

    # Classification-specific
    if predictions_distribution is not None:
        meta["predictions_distribution"] = predictions_distribution
    if agreement_stats is not None:
        meta["agreement_stats"] = agreement_stats
    if confirmation_percentiles is not None:
        meta["confirmation_percentiles"] = confirmation_percentiles
    if directional_rate is not None:
        meta["directional_rate"] = directional_rate

    # Regression-specific
    if metrics is not None:
        meta["metrics"] = metrics
    if calibration is not None:
        meta["calibration"] = calibration

    # Phase II (2026-04-20): CompatibilityContract block + fingerprint.
    # Additive — pre-Phase-II consumers ignore unrecognized keys; post-Phase-II
    # consumers get the full 3-way validation surface.
    if compatibility is not None:
        # Serialize the frozen dataclass via asdict() — tuples become lists,
        # matching the canonical JSON shape (see compatibility.to_canonical_dict).
        compat_block = asdict(compatibility)
        # asdict leaves tuples as tuples in some Python versions; normalize to
        # lists for strict JSON parity with to_canonical_dict().
        if compat_block.get("horizons") is not None and not isinstance(compat_block["horizons"], list):
            compat_block["horizons"] = list(compat_block["horizons"])
        meta["compatibility"] = compat_block
        meta["compatibility_fingerprint"] = compatibility.fingerprint()

    # Phase II top-level conveniences (read by SignalManifest without parsing the nested block)
    if data_source is not None:
        meta["data_source"] = data_source
    if calibration_method is not None:
        meta["calibration_method"] = calibration_method

    # Phase Y deployment (2026-05-05): top-level ``model_config_hash`` —
    # parallel signal-side identity to ``compatibility_fingerprint`` (which
    # captures data-axis architecture: feature_count + window_size + horizons
    # + label_strategy_hash + ...). model_config_hash captures the model-axis
    # architecture (model_type + filtered model.params). Together they pin the
    # full identity of the producer that emitted these signals. hft-ops
    # ``_harvest_model_config_hash`` reads this key directly. Validated as
    # 64-lowercase-hex SHA-256 by the consumer's CONTENT_HASH_RE gate.
    if model_config_hash is not None:
        meta["model_config_hash"] = model_config_hash

    # Phase Y / γ-1 LITE / #PY-88 (2026-05-10): top-level human-visible
    # ``return_type`` field — see kwarg docstring above.
    if return_type is not None:
        meta["return_type"] = return_type

    return meta
