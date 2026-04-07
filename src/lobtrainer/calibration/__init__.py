"""Prediction calibration for post-hoc magnitude correction.

Provides calibrators that adjust model prediction magnitudes to match
the target return distribution while preserving ranking (IC unchanged).

The primary use case: Huber-loss trained models produce conservative
predictions (std ratio ~0.27 vs actual returns). Variance-matching
calibration rescales predictions to match the target distribution,
enabling effective conviction-based filtering.

Modules:
    variance: Variance-matching calibration (linear rescaling)

Reference:
    E5 Comprehensive Statistical Report §8.1: "Model predictions are too
    conservative (std ratio = 0.27). Rescaling preserves ranking while
    correcting magnitude."
"""

from lobtrainer.calibration.variance import (
    VarianceCalibrationConfig,
    calibrate_variance,
    CalibrationResult,
)

__all__ = [
    "VarianceCalibrationConfig",
    "calibrate_variance",
    "CalibrationResult",
]
