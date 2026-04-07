"""
Regression metrics for the training pipeline.

Thin adapter over hft-metrics (single source of truth per Rule 0).
Only ``compute_all_regression_metrics`` and two name-adapting wrappers
(``information_coefficient``, ``pearson_correlation``) live here.

All numerical implementations are in ``hft_metrics.regression`` and
``hft_metrics.ic``. This module preserves the function names and
signatures that the trainer codebase depends on.

Migration source: hft-metrics v0.1.0 (hft_metrics.regression, hft_metrics.ic)
"""

from typing import Dict

import numpy as np

# Direct re-exports — identical signatures and semantics
from hft_metrics.regression import (
    r_squared,
    mean_absolute_error,
    root_mean_squared_error,
    directional_accuracy,
    profitable_accuracy,
)


def information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank IC between predicted and actual returns.

    Wrapper over ``hft_metrics.ic.spearman_ic`` which returns ``(rho, p_value)``.
    This adapter returns rho only for backward compatibility.

    Reference: Grinold & Kahn (2000), "Active Portfolio Management", Ch 6.
    """
    from hft_metrics.ic import spearman_ic

    rho, _ = spearman_ic(y_true, y_pred)
    return rho


def pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson linear correlation between predicted and actual returns.

    Wrapper over ``hft_metrics.ic.pearson_r`` which returns ``(r, p_value)``.
    This adapter returns r only for backward compatibility.
    """
    from hft_metrics.ic import pearson_r

    r, _ = pearson_r(y_true, y_pred)
    return r


def compute_all_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    breakeven_bps: float = 5.0,
    prefix: str = "",
) -> Dict[str, float]:
    """Compute all regression metrics in one call.

    This is domain-specific glue that calls individual metric functions
    and assembles a dict with prefixed keys. The key names are a de facto
    contract consumed by strategies, evaluation, and experiment tracking.

    Args:
        y_true: Ground truth returns in bps [N].
        y_pred: Predicted returns in bps [N].
        breakeven_bps: Minimum move for profitable accuracy.
        prefix: Optional prefix for metric names (e.g., "val_h60_").

    Returns:
        Dict of metric_name -> value.
    """
    return {
        f"{prefix}r2": r_squared(y_true, y_pred),
        f"{prefix}ic": information_coefficient(y_true, y_pred),
        f"{prefix}mae": mean_absolute_error(y_true, y_pred),
        f"{prefix}rmse": root_mean_squared_error(y_true, y_pred),
        f"{prefix}pearson": pearson_correlation(y_true, y_pred),
        f"{prefix}directional_accuracy": directional_accuracy(y_true, y_pred),
        f"{prefix}profitable_accuracy": profitable_accuracy(
            y_true, y_pred, breakeven_bps
        ),
    }
