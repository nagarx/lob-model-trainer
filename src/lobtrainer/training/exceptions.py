"""
Training-pipeline exception classes.

Phase X.3 Empirical Trust cycle (2026-05-05): centralizes fail-loud error
types raised across the training surface. Replaces silent-zero / silent-NaN
fallback patterns identified in the 2026-05-05 multi-agent audit (Cluster I
silent-default + Cluster V NaN-propagation).

Per hft-rules §2 ("zero tolerance for precision errors") + §5 ("fail fast
with a precise error — never silently degrade") + §8 ("never silently
drop, clamp, or 'fix' data"). Raised at compute boundaries where degenerate
state would otherwise propagate as a benign-looking value (NaN loss → silent
parameter corruption; all-NaN feature column → fabricated normalization
stats; missing monitored metric → silent "no improvement" stall).

DUAL-GUARD CONTRACT — TrainingDivergedError raised from TWO sites:

1. **Direct guard** at `Trainer._train_epoch:902` (PRE-backward). Catches
   non-finite loss BEFORE `loss.backward()` propagates NaN/Inf into model
   parameters. Rich context (epoch, batch, global_step, loss_value).
   This is the PRIMARY defense — prevents downstream corruption.

2. **Post-hoc audit** at `TrainingDiagnostics.on_batch_end` (POST-step).
   Detects NaN loss AFTER backward+step (params already corrupted), but
   provides a defense-in-depth audit if the direct guard is bypassed (e.g.,
   user subclasses Trainer + overrides _train_epoch without re-applying the
   guard). Same exception type, same context — uniform error contract.

Both guards raise the SAME exception class so user code catching
`TrainingDivergedError` works regardless of which fired. Future readers
of this module: do NOT delete the post-hoc callback thinking it is
redundant — the timing differs, and the dual-protection ensures NaN
cannot silently propagate even on partial subclass overrides.
"""

from __future__ import annotations

from typing import List, Optional


class TrainingDivergedError(RuntimeError):
    """
    Raised when training loss is NaN or Inf.

    Per hft-rules §2 ("Always check ``np.isfinite()`` before comparisons —
    NaN comparisons fail silently"). Pre-Phase-X.3 the trainer's
    ``_train_epoch`` called ``loss.backward()`` without a finiteness guard,
    silently propagating NaN/Inf through gradients into model parameters.
    Stage 4 GMADL near-collapse (``pred_std=7.7e-5``) was caught only because
    IC was visibly near 0; future loss-explosions / sigmoid-saturation /
    log-of-zero / exp-overflow scenarios would have produced NaN parameters
    + later NaN val_loss + EarlyStopping silent-no-improvement loop.

    Recovery: Training MUST stop. Inspect the failing batch + loss components
    to diagnose. If the divergence is a known research-mode loss-collapse
    being studied, opt out via Trainer construction parameter or callback
    override (NOT by suppressing the exception).

    Attributes:
        epoch: Epoch index (0-based) when divergence detected.
        batch: Batch index within the epoch.
        loss_value: The non-finite loss value (NaN, +Inf, or -Inf as float).
        global_step: Cumulative step count across all epochs.
    """

    def __init__(
        self,
        epoch: int,
        batch: int,
        loss_value: float,
        global_step: int,
    ) -> None:
        self.epoch = epoch
        self.batch = batch
        self.loss_value = loss_value
        self.global_step = global_step
        super().__init__(
            f"Training diverged: non-finite loss={loss_value!r} at "
            f"epoch={epoch}, batch={batch}, global_step={global_step}. "
            f"Per hft-rules §2 + §8: fail-loud on numerical divergence "
            f"to prevent silent parameter corruption. Inspect loss "
            f"components + recent batch data; check for sigmoid "
            f"saturation, log-of-zero, divide-by-zero, or label NaN."
        )


class MonitorMetricUndefined(RuntimeError):
    """
    Raised when a monitored metric (e.g., val_loss) is non-finite.

    Pre-Phase-X.3, ``EarlyStopping._is_better(NaN, best)`` returned ``False``
    silently (since ``NaN < anything`` is ``False``); the patience counter
    incremented but never reset, and ``ModelCheckpoint`` never saved a
    "best" checkpoint. Training silently stalled with degraded weights
    until patience ran out. Per hft-rules §5: fail fast with precise error.

    Recovery: Training MUST stop. Non-finite val_loss almost always
    indicates the model has diverged (param NaN). Check ``TrainingDivergedError``
    is properly raised earlier; if it is, this exception fires only on
    val-time NaN that the train-time guard didn't catch (e.g., model
    overflows on val data but not train data).

    Attributes:
        metric: Name of the monitored metric (e.g., "val_loss").
        value: The non-finite value observed.
    """

    def __init__(self, metric: str, value: float) -> None:
        self.metric = metric
        self.value = value
        super().__init__(
            f"Monitored metric '{metric}' is non-finite (value={value!r}). "
            f"Per hft-rules §2: fail-loud on numerical divergence. "
            f"Pre-fix this would silently treat NaN as 'no improvement' → "
            f"patience counter increments without ever resetting → silent "
            f"training stall + best.pt never refreshed. Likely root cause: "
            f"model parameters are NaN (check TrainingDivergedError fired "
            f"during the prior train epoch)."
        )


class DegenerateFeatureError(ValueError):
    """
    Raised when normalization statistics cannot be computed.

    Two failure modes:

    1. **All-NaN feature column**: ``np.nanmean`` returns NaN when ALL
       samples are NaN. Pre-Phase-X.3, ``transforms.py:177-183`` silently
       fabricated ``mean=0.0`` + ``std=1.0`` via ``nan_to_num`` — corrupt
       training data passed downstream as ``(x - 0)/1 = x`` (no-op
       normalization, but stat metadata claims valid stats). Per hft-rules
       §8 ("never silently drop, clamp, or 'fix' data without recording
       diagnostics").

    2. **Zero-variance feature column** (``std < eps``): the
       ``np.maximum(std, eps)`` floor at ``transforms.py:174`` silently
       clamped truly-constant features to ``eps`` → divides by ``eps`` →
       produces near-zero normalized values that look like signal noise.
       Different from "feature is supposed to be constant" (e.g., RESERVED
       index 97); legitimate-constant features should be in the
       ``exclude_indices`` list, NOT silently clamped here. Per
       hft-rules §1 ("layout as contract").

    Recovery: Inspect upstream feature extraction. Either: (a) feature is
    legitimately all-NaN/constant — add to ``exclude_indices`` config; OR
    (b) data corruption — investigate the producer.

    Attributes:
        feature_indices: List of feature indices (0-based) that are degenerate.
        reason: One of "all_nan" or "zero_variance".
        eps: The eps threshold used for zero-variance detection (only set
            when reason == "zero_variance").
    """

    def __init__(
        self,
        feature_indices: List[int],
        reason: str,
        eps: Optional[float] = None,
    ) -> None:
        self.feature_indices = feature_indices
        self.reason = reason
        self.eps = eps

        if reason == "all_nan":
            detail = (
                f"all-NaN across all samples — cannot compute mean/std. "
                f"Investigate upstream extraction OR add to "
                f"`normalization.exclude_indices`."
            )
        elif reason == "zero_variance":
            detail = (
                f"zero variance (std < {eps!r}) across all samples — "
                f"feature is constant. Pre-fix the `np.maximum(std, eps)` "
                f"floor silently clamped to eps → divides by eps → "
                f"produces near-zero outputs that look like signal noise. "
                f"Add to `normalization.exclude_indices` if intentional."
            )
        else:
            detail = f"unknown reason='{reason}'"

        super().__init__(
            f"Degenerate feature(s) at indices {feature_indices}: {detail} "
            f"Per hft-rules §8 fail-loud on data integrity violation."
        )


__all__ = [
    "TrainingDivergedError",
    "MonitorMetricUndefined",
    "DegenerateFeatureError",
]
