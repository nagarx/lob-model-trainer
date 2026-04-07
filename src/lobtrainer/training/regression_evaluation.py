"""
Regression evaluation utilities.

Provides regression-specific evaluation, baselines, and reporting
parallel to the classification evaluation in evaluation.py.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from lobtrainer.training.regression_metrics import compute_all_regression_metrics


@dataclass
class RegressionMetrics:
    """Structured regression metrics parallel to ClassificationMetrics."""

    r2: float = 0.0
    ic: float = 0.0
    pearson: float = 0.0
    mae: float = 0.0
    rmse: float = 0.0
    directional_accuracy: float = 0.0
    profitable_accuracy: float = 0.0
    extra: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "=== Regression Metrics ===",
            f"  R-squared:              {self.r2:.6f}",
            f"  Information Coefficient: {self.ic:.6f}",
            f"  Pearson Correlation:     {self.pearson:.6f}",
            f"  MAE (bps):              {self.mae:.2f}",
            f"  RMSE (bps):             {self.rmse:.2f}",
            f"  Directional Accuracy:   {self.directional_accuracy:.4f}",
            f"  Profitable Accuracy:    {self.profitable_accuracy:.4f}",
        ]
        for k, v in self.extra.items():
            lines.append(f"  {k}: {v:.6f}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, float]:
        d = {
            "r2": self.r2,
            "ic": self.ic,
            "pearson": self.pearson,
            "mae": self.mae,
            "rmse": self.rmse,
            "directional_accuracy": self.directional_accuracy,
            "profitable_accuracy": self.profitable_accuracy,
        }
        d.update(self.extra)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "RegressionMetrics":
        known = {
            "r2", "ic", "pearson", "mae", "rmse",
            "directional_accuracy", "profitable_accuracy",
        }
        extra = {k: v for k, v in d.items() if k not in known}
        return cls(
            r2=d.get("r2", 0.0),
            ic=d.get("ic", 0.0),
            pearson=d.get("pearson", 0.0),
            mae=d.get("mae", 0.0),
            rmse=d.get("rmse", 0.0),
            directional_accuracy=d.get("directional_accuracy", 0.0),
            profitable_accuracy=d.get("profitable_accuracy", 0.0),
            extra=extra,
        )

    @classmethod
    def from_arrays(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        breakeven_bps: float = 5.0,
    ) -> "RegressionMetrics":
        """Build from prediction/target arrays using compute_all_regression_metrics."""
        metrics = compute_all_regression_metrics(y_true, y_pred, breakeven_bps=breakeven_bps)
        return cls(
            r2=metrics.get("r2", 0.0),
            ic=metrics.get("ic", 0.0),
            pearson=metrics.get("pearson", 0.0),
            mae=metrics.get("mae", 0.0),
            rmse=metrics.get("rmse", 0.0),
            directional_accuracy=metrics.get("directional_accuracy", 0.0),
            profitable_accuracy=metrics.get("profitable_accuracy", 0.0),
        )
