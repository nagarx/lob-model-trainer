"""Sample weight integration for the training pipeline (T10).

Thin wrapper that connects hft-metrics pure functions to the trainer's
DayData / LabelsConfig infrastructure. Resolves horizon event count
from metadata and LabelsConfig, then delegates to hft_metrics.sample_weights.

Reference: de Prado (2018) AFML Ch 4.5.1
"""

import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


def compute_sample_weights_for_day(
    n_samples: int,
    metadata: Optional[Dict],
    labels_config: "LabelsConfig",
    stride: int = 1,
) -> Optional[np.ndarray]:
    """Compute de Prado concurrent-overlap sample weights for one day.

    Resolves the actual horizon event count from:
        labels_config.primary_horizon_idx → metadata["horizons"][idx]

    Args:
        n_samples: Number of sequences in the day.
        metadata: Parsed per-day metadata JSON (needs 'horizons' field).
        labels_config: For primary_horizon_idx and sample_weights method.
        stride: Export stride in sample-space (from dataset_manifest.json).
            Default 1 (time-based exports). Event-based exports typically
            use 10 or higher.

    Returns:
        [n_samples] float64 with mean ≈ 1.0.
        None if sample_weights == "none" or horizon is unresolvable.
    """
    if labels_config.sample_weights == "none":
        return None

    if labels_config.sample_weights != "concurrent_overlap":
        logger.warning(
            "Unknown sample_weights method %r; returning None",
            labels_config.sample_weights,
        )
        return None

    if n_samples < 1:
        return None

    # Resolve horizon event count from metadata + primary_horizon_idx
    horizon = _resolve_horizon(metadata, labels_config)
    if horizon is None:
        logger.warning(
            "Cannot resolve horizon for sample weights "
            "(primary_horizon_idx=%r, metadata horizons=%r); skipping",
            labels_config.primary_horizon_idx,
            _get_horizons_from_metadata(metadata),
        )
        return None

    from hft_metrics.sample_weights import compute_sample_weights

    weights = compute_sample_weights(n_samples, horizon, stride)
    logger.debug(
        "Sample weights: n=%d, horizon=%d, stride=%d, "
        "mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
        n_samples, horizon, stride,
        weights.mean(), weights.std(), weights.min(), weights.max(),
    )
    return weights


def _resolve_horizon(
    metadata: Optional[Dict],
    labels_config: "LabelsConfig",
) -> Optional[int]:
    """Resolve the actual horizon event count for weight computation.

    Priority:
        1. labels_config.horizons (if non-empty) at primary_horizon_idx
        2. metadata horizons at primary_horizon_idx
        3. None (unresolvable)
    """
    idx = labels_config.primary_horizon_idx
    if idx is None:
        # HMHP mode: use max horizon (most conservative weighting)
        horizons = labels_config.horizons or _get_horizons_from_metadata(metadata)
        if horizons:
            return max(horizons)
        return None

    # Single-horizon mode: resolve event count at the selected index
    if labels_config.horizons:
        if idx < len(labels_config.horizons):
            return labels_config.horizons[idx]
        return None

    horizons = _get_horizons_from_metadata(metadata)
    if horizons and idx < len(horizons):
        return horizons[idx]
    return None


def _get_horizons_from_metadata(metadata: Optional[Dict]) -> Optional[list]:
    """Extract horizons list from metadata using the same fallback as DayData.horizons."""
    if metadata is None:
        return None
    h = metadata.get("horizons")
    if h is not None:
        return h
    labeling = metadata.get("labeling", {})
    if isinstance(labeling, dict) and "horizons" in labeling:
        return labeling["horizons"]
    label_config = metadata.get("label_config", {})
    if isinstance(label_config, dict) and "horizons" in label_config:
        return label_config["horizons"]
    return None
