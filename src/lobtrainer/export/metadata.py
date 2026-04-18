"""
Signal metadata builder for signal export.

Produces a superset metadata JSON that includes everything all 3 old
export scripts produced, plus additional provenance fields.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


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
) -> Dict[str, Any]:
    """Build comprehensive signal metadata JSON.

    Superset of all 3 old export scripts' metadata schemas, plus
    additional provenance fields for experiment tracking.

    All fields are keyword-only to prevent positional mistakes.
    """
    meta: Dict[str, Any] = {
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

    return meta
