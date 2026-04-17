"""Pre-training gates for experiment validation (T14).

Gates run BEFORE training starts. If a gate fails, training is skipped
and the failure is recorded. This prevents wasting compute on experiments
that are doomed to fail.

Reference: hft-rules.md §13 — mandatory signal quality gate.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    """Result of a pre-training gate check."""

    gate_name: str
    passed: bool
    details: Dict[str, float]
    message: str


def run_signal_quality_gate(
    train_days: list,
    horizon_idx: int = 0,
    min_ic: float = 0.05,
    min_features_passing: int = 1,
    max_features_to_check: int = 50,
) -> GateResult:
    """Check that features have predictive signal for the target label.

    Computes Spearman IC between each feature (last timestep of each
    sequence) and the regression label at ``horizon_idx`` across all
    training days. At least ``min_features_passing`` features must have
    |IC| > ``min_ic`` for the gate to pass.

    This is the MANDATORY pre-training gate per hft-rules.md §13:
    "If no feature has IC > 0.05 for the target label, no model will help."

    Args:
        train_days: List of DayData with regression_labels.
        horizon_idx: Which horizon column to check IC against.
        min_ic: Minimum absolute IC threshold.
        min_features_passing: How many features must pass.
        max_features_to_check: Cap on features to check (performance).

    Returns:
        GateResult with pass/fail, per-feature ICs, and message.
    """
    from hft_metrics.ic import spearman_ic

    # Collect all features and labels across days
    all_features = []
    all_labels = []

    for day in train_days:
        if day.features is None or day.regression_labels is None:
            continue
        feats = day.features  # [N, F]
        reg = day.regression_labels  # [N, H]
        if reg.ndim == 1:
            labels = reg
        elif horizon_idx < reg.shape[1]:
            labels = reg[:, horizon_idx]
        else:
            continue
        all_features.append(feats)
        all_labels.append(labels)

    if not all_features:
        return GateResult(
            gate_name="signal_quality",
            passed=False,
            details={},
            message="No valid training data with regression labels found.",
        )

    features = np.concatenate(all_features, axis=0)  # [N_total, F]
    labels = np.concatenate(all_labels, axis=0)  # [N_total]

    n_features = min(features.shape[1], max_features_to_check)
    feature_ics: Dict[str, float] = {}
    passing_features = 0

    for i in range(n_features):
        feat_col = features[:, i]
        # Skip constant features
        if np.std(feat_col) < 1e-10:
            feature_ics[f"feature_{i}"] = 0.0
            continue
        ic, _ = spearman_ic(feat_col, labels)
        feature_ics[f"feature_{i}"] = float(ic)
        if abs(ic) > min_ic:
            passing_features += 1

    passed = passing_features >= min_features_passing

    # Find top features for logging
    sorted_ics = sorted(
        feature_ics.items(), key=lambda x: abs(x[1]), reverse=True
    )
    top_5 = sorted_ics[:5]

    if passed:
        message = (
            f"Signal quality gate PASSED: {passing_features}/{n_features} "
            f"features with |IC| > {min_ic}. "
            f"Top: {', '.join(f'{k}={v:.3f}' for k, v in top_5)}"
        )
        logger.info(message)
    else:
        message = (
            f"Signal quality gate FAILED: only {passing_features}/{n_features} "
            f"features with |IC| > {min_ic} (need {min_features_passing}). "
            f"Top: {', '.join(f'{k}={v:.3f}' for k, v in top_5)}"
        )
        logger.warning(message)

    return GateResult(
        gate_name="signal_quality",
        passed=passed,
        details=feature_ics,
        message=message,
    )
