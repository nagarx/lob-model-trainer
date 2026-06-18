"""Run-entry label nudges (Phase 3b — the E8 ``smoothed_return`` reminder).

Emits a single, run-entry ``UserWarning`` when a model is trained on the
``smoothed_return`` label — the mean of a k-event forward window, NOT the
tradeable point-to-point return. A model can score a high IC on the smoothed
label yet have directional accuracy below a coin on the executable point return
(the E8 trap; see FINDING-001 / the 2026-06-18 platform-hardening cycle). This
pairs with the post-training point-return-DA tripwire (Phase 3c) surfaced by the
hft-ops ``post_training_gate``.

Design note (a lesson learned + reverted in this cycle): this MUST live at the
run entry (each trainer's ``train()``), NOT in ``LabelsConfig`` validation. A
construction-time warning floods every nested default build (``DataConfig``
constructs a default ``LabelsConfig``) — ~150 spurious suite warnings and a
broken ``simplefilter("error")`` test. The run entry fires exactly once per run.

``return_type`` is part of the experiment compatibility fingerprint, so this is a
PURE warning: it changes no config value, default, or representation.
"""

from __future__ import annotations

import warnings

_SMOOTHED_RETURN_NUDGE_EMITTED = False

_E8_SMOOTHED_RETURN_MESSAGE = (
    "Training on the 'smoothed_return' label — the mean of a k-event forward "
    "window, NOT the tradeable point-to-point return. A model can score a high IC "
    "on this smoothed label yet have directional accuracy below a coin on the "
    "executable point return (the E8 trap; see FINDING-001). The post-training "
    "gate emits a point-return-DA tripwire (test_point_return_da) — review it "
    "before trusting this model for execution."
)


def warn_if_smoothed_return(return_type: str, *, force: bool = False) -> bool:
    """Emit the E8 run-entry nudge once per process when training on smoothed_return.

    Idempotent by design: CVTrainer spawns K fresh ``Trainer`` instances within
    ONE run, so the guard suppresses the K-1 duplicate warnings. A separate
    process (a fresh CLI invocation) starts with the guard reset, so each run
    warns once.

    Args:
        return_type: The resolved ``LabelsConfig.return_type`` for the run.
        force: Bypass the once-per-process guard (test hook).

    Returns:
        True if a warning was emitted, else False.
    """
    global _SMOOTHED_RETURN_NUDGE_EMITTED
    if return_type != "smoothed_return":
        return False
    if _SMOOTHED_RETURN_NUDGE_EMITTED and not force:
        return False
    _SMOOTHED_RETURN_NUDGE_EMITTED = True
    warnings.warn(_E8_SMOOTHED_RETURN_MESSAGE, UserWarning, stacklevel=2)
    return True


def _reset_smoothed_return_nudge() -> None:
    """Reset the once-per-process guard (test hook)."""
    global _SMOOTHED_RETURN_NUDGE_EMITTED
    _SMOOTHED_RETURN_NUDGE_EMITTED = False
