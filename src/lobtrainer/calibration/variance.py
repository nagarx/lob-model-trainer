"""Variance-matching prediction calibration.

Rescales model predictions to match the target return distribution variance
while preserving ranking (Spearman IC unchanged). This corrects the
conservatism of Huber-loss trained models.

Formula:
    calibrated = (pred - pred_mean) * (target_std / pred_std) + target_mean

Properties:
    - IC(calibrated, target) == IC(raw, target) (linear transform preserves rank)
    - std(calibrated) == target_std (by construction)
    - mean(calibrated) == target_mean (by construction)

Reference:
    E5 Comprehensive Statistical Report §1.3:
        pred_std = 7.35 bps, target_std = 27.41 bps, scale_factor = 3.73
    E5 Comprehensive Statistical Report §7.1:
        Win rate at |smoothed| > 10 bps: 90.8% (23,179 samples)
        Win rate at |smoothed| > 20 bps: 94.5% (12,784 samples)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, TypedDict

import numpy as np


# =============================================================================
# Phase A.5.5 (2026-04-24) — CalibrationContext TypedDict
# =============================================================================
#
# Replaces the previous ``metadata: Optional[Dict[str, Any]]`` blob on
# ``CalibrationResult`` with a typed TypedDict. Closes plan v4 bug #4:
# pre-A.5.5 the blob was an open Dict so shallow ``dict(metadata)`` copies
# could share nested-dict references — silent-aliasing hazard if the
# calibration context ever grew nested dicts (Phase B: quantile_levels
# dict, conformal alphas per-horizon, etc.).
#
# Design per plan v4 Agent 2 refinement: TypedDict with ``total=False``
# (all fields optional). Zero runtime cost (no class creation at import
# time), trivial JSON serialization (dict-to-dict passthrough), and
# Phase B calibrators extend one-line-per-field (``quantile_levels:
# Optional[Tuple[float, ...]]``, ``conformal_alpha: Optional[float]``,
# etc.) — no schema migration per extension.
#
# Wire-format preservation: the JSON key in ``to_dict()`` output stays
# ``"metadata"`` (plan v4 round-2 refinement: "don't mix semantic rename
# with wire-format rename"). Downstream consumers that read
# ``signal_metadata.json["calibration"]["metadata"]`` continue to work
# unchanged; only the Python-side attribute name moves to ``context``.
#
# Ship-until-Phase-B: TypedDict inlined here (not extracted to its own
# module) until a second calibrator (quantile/isotonic/conformal)
# warrants a dedicated ``lobtrainer.calibration.context`` module. YAGNI
# until then — fights the "premature extraction" trap.
# =============================================================================

class CalibrationContext(TypedDict, total=False):
    """Calibration provenance / observability context.

    All fields optional (``total=False``) so Phase B calibrators add
    fields one-line-per-field as needed:

        class CalibrationContext(TypedDict, total=False):
            primary_horizon_idx: Optional[int]
            method_variant: Optional[str]
            # Phase B extension points:
            # quantile_levels: Optional[Tuple[float, ...]]
            # conformal_alpha: Optional[float]
            # isotonic_breakpoints: Optional[Tuple[float, ...]]

    Type-safety gain: operators constructing the context (in
    ``SignalExporter._apply_calibration`` + future calibrator call
    sites) get typos caught at static-analysis time via mypy /
    pyright / pyre. Zero runtime cost — a TypedDict IS a dict subclass
    at runtime.

    JSON serialization: passthrough via ``dict(context)`` — no
    model_dump / asdict needed. Wire-format key preserved as
    ``"metadata"`` in ``CalibrationResult.to_dict()`` output.
    """

    primary_horizon_idx: Optional[int]
    """Which horizon was sliced for 1-D calibrator input. Threaded from
    the caller's ``LabelsConfig.validate_primary_horizon_idx_for(...)``
    result (A.5.4 SSoT)."""

    method_variant: Optional[str]
    """Sub-variant label for the calibration method (e.g., "huber",
    "mse", "variance_match_v2"). Future-proofs the observability
    surface against same-method-different-settings runs."""


@dataclass
class VarianceCalibrationConfig:
    """Configuration for variance-matching calibration.

    Attributes:
        method: Calibration method. "variance_match" or "none".
        target_std_bps: Target standard deviation in basis points.
            Default: 27.41 (E5 test H10 return std).
        target_mean_bps: Target mean in basis points.
            Default: -0.167 (E5 test H10 return mean).
        compute_from_labels: If True, compute target_std/mean from labels
            array at calibration time (overrides static values).
    """
    method: str = "variance_match"
    target_std_bps: float = 27.41
    target_mean_bps: float = -0.167
    compute_from_labels: bool = True


@dataclass
class CalibrationResult:
    """Result of prediction calibration.

    Attributes:
        calibrated: Calibrated predictions array.
        scale_factor: Multiplicative scale factor applied (target_std / pred_std).
        pred_mean: Mean of raw predictions before calibration.
        pred_std: Std of raw predictions before calibration.
        target_mean: Target mean used for calibration.
        target_std: Target std used for calibration.
        n_samples: Number of samples calibrated.
        context: Optional observability / provenance context (Phase A.5.5,
            2026-04-24). TypedDict ``CalibrationContext`` — typed extension
            point for Phase B calibrators (quantile, isotonic, conformal)
            that need per-call observability fields. Field was named
            ``metadata`` pre-A.5.5 (Phase A 2026-04-23); renamed to
            ``context`` for semantic clarity. Wire-format JSON key
            (``to_dict()["metadata"]``) preserved for downstream consumer
            compatibility.
    """
    calibrated: np.ndarray
    scale_factor: float
    pred_mean: float
    pred_std: float
    target_mean: float
    target_std: float
    n_samples: int
    # Phase A.5.5 (2026-04-24): typed ``CalibrationContext`` replaces
    # the previous open ``Dict[str, Any]``. Closes plan v4 bug #4
    # (shallow-copy nested-dict aliasing hazard) — TypedDict fields
    # are all flat primitives, so the shallow ``dict(self.context)``
    # copy in ``to_dict()`` is safe by construction. Phase B adds
    # fields one-line-per-field on the TypedDict definition.
    context: Optional[CalibrationContext] = None

    def to_dict(self) -> dict:
        """Serialize calibration stats (no numpy arrays).

        Phase A.5.5 (2026-04-24): JSON wire-format key remains ``"metadata"``
        (plan v4 round-2 refinement — semantic Python rename only, not
        wire-format rename). Downstream consumers reading
        ``signal_metadata.json["calibration"]["metadata"]`` continue to
        work unchanged post-migration.
        """
        result: Dict[str, Any] = {
            "scale_factor": self.scale_factor,
            "pred_mean": self.pred_mean,
            "pred_std": self.pred_std,
            "target_mean": self.target_mean,
            "target_std": self.target_std,
            "n_samples": self.n_samples,
            "calibrated_mean": float(np.mean(self.calibrated)),
            "calibrated_std": float(np.std(self.calibrated)),
            "calibrated_min": float(np.min(self.calibrated)),
            "calibrated_max": float(np.max(self.calibrated)),
        }
        if self.context is not None:
            # Wire-format key stays "metadata" (preserved for
            # downstream consumer compat per Phase A.5.5 plan).
            # Shallow ``dict(...)`` copy is safe: CalibrationContext
            # is a TypedDict with flat primitive-typed fields — no
            # nested-dict aliasing hazard.
            result["metadata"] = dict(self.context)
        return result


def calibrate_variance(
    predictions: np.ndarray,
    labels: Optional[np.ndarray] = None,
    config: Optional[VarianceCalibrationConfig] = None,
    *,
    context: Optional[CalibrationContext] = None,
) -> CalibrationResult:
    """Calibrate prediction magnitudes via variance matching.

    Formula:
        calibrated = (pred - pred_mean) * (target_std / pred_std) + target_mean

    This is a linear transformation that:
        1. Centers predictions at zero
        2. Scales to match target variance
        3. Re-centers at target mean

    Because it is a monotone linear transform, Spearman IC is preserved exactly.

    Args:
        predictions: Raw model predictions in bps, shape ``(N,)``. Strict
            1-D contract — 2-D / higher-dim input raises :class:`ValueError`.
        labels: Regression labels in bps, shape ``(N,)``. Used to compute
            ``target_std / mean`` when ``config.compute_from_labels=True``.
            Strict 1-D contract — 2-D input raises :class:`ValueError`. Phase A
            (2026-04-23) tightened this from the previous silent ``labels[:, 0]``
            fallback, which masked caller-side bugs where the caller's
            ``primary_horizon_idx`` was non-zero.
        config: Calibration configuration. Defaults to
            :class:`VarianceCalibrationConfig`.
        context: Optional observability / provenance TypedDict propagated to
            :class:`CalibrationResult.context` (Phase A.5.5 — renamed from
            ``metadata=`` kwarg, typed via ``CalibrationContext`` TypedDict).
            Forward-compat hook for non-variance-match calibrators (quantile,
            isotonic, conformal) that need per-call provenance. Pure
            passthrough — never consumed by the calibration math itself.

    Returns:
        :class:`CalibrationResult` with calibrated predictions and stats.

    Raises:
        ValueError: If predictions or labels are not 1-D (shape contract).
        ValueError: If predictions are empty or have zero variance.
        ValueError: If ``compute_from_labels=True`` but ``labels`` is ``None``.
    """
    if config is None:
        config = VarianceCalibrationConfig()

    # Phase A (2026-04-23): strict 1-D contract. Multi-horizon slicing is the
    # CALLER's responsibility so the caller's primary_horizon_idx knowledge
    # is honored explicitly — the previous silent ``labels[:, 0]`` fallback
    # masked caller-side bugs (hft-rules §8: never silently drop data).
    if predictions.ndim != 1:
        raise ValueError(
            f"calibrate_variance expects 1-D predictions; got shape "
            f"{predictions.shape}. Callers must slice multi-horizon predictions "
            f"via their primary_horizon_idx at the call site."
        )
    if labels is not None and np.asarray(labels).ndim != 1:
        raise ValueError(
            f"calibrate_variance expects 1-D labels; got shape "
            f"{np.asarray(labels).shape}. Callers must slice multi-horizon "
            f"labels via their primary_horizon_idx at the call site."
        )

    if config.method == "none":
        return CalibrationResult(
            calibrated=predictions.copy(),
            scale_factor=1.0,
            pred_mean=float(np.mean(predictions)),
            pred_std=float(np.std(predictions)),
            target_mean=float(np.mean(predictions)),
            target_std=float(np.std(predictions)),
            n_samples=len(predictions),
            context=context,
        )

    predictions = np.asarray(predictions, dtype=np.float64)

    if len(predictions) == 0:
        raise ValueError("Cannot calibrate empty predictions array")

    pred_mean = float(np.mean(predictions))
    pred_std = float(np.std(predictions))

    if pred_std < 1e-10:
        raise ValueError(
            f"Predictions have near-zero variance (std={pred_std:.2e}). "
            "Cannot calibrate — model may have collapsed to constant prediction."
        )

    # Determine target statistics
    if config.compute_from_labels:
        if labels is None:
            raise ValueError(
                "compute_from_labels=True but labels is None. "
                "Provide labels or set compute_from_labels=False."
            )
        # Labels already enforced 1-D at the strict contract above.
        labels = np.asarray(labels, dtype=np.float64)
        target_std = float(np.std(labels))
        target_mean = float(np.mean(labels))
    else:
        target_std = config.target_std_bps
        target_mean = config.target_mean_bps

    if target_std < 1e-10:
        raise ValueError(
            f"Target has near-zero variance (std={target_std:.2e}). "
            "Cannot calibrate to a degenerate target distribution."
        )

    # Apply variance-matching calibration
    scale_factor = target_std / pred_std
    calibrated = (predictions - pred_mean) * scale_factor + target_mean

    return CalibrationResult(
        calibrated=calibrated,
        scale_factor=scale_factor,
        pred_mean=pred_mean,
        pred_std=pred_std,
        target_mean=target_mean,
        target_std=target_std,
        n_samples=len(predictions),
        context=context,
    )
