"""
hft-ops compatibility helper for trainer-side scripts.

Phase 1.4 of the training-pipeline-architecture migration: bypass entry
points (``train.py``, ``e4_baselines.py``, ``run_simple_training.py``, etc.)
import this module and call ``warn_if_not_orchestrated()`` at top of file.

If the script is invoked DIRECTLY (e.g., ``python scripts/train.py ...``)
without going through ``hft-ops run``, a ``UserWarning`` is emitted with
migration guidance. If invoked AS a subprocess by hft-ops (which sets
``HFT_OPS_ORCHESTRATED=1`` in the env), the warning is suppressed.

This is a SOFT deprecation — the script still works. Hard deprecation
(error instead of warning) is deferred to Phase 5 once the unified
manifest workflow is established.

Usage in a script:

    from _hft_ops_compat import warn_if_not_orchestrated
    warn_if_not_orchestrated(script_name="train.py")
"""

from __future__ import annotations

import os
import sys
import warnings


_HFT_OPS_ENV_VAR = "HFT_OPS_ORCHESTRATED"
_GUIDANCE_URL = (
    "https://github.com/nagarx/hft-pipeline-v2/blob/main/hft-ops/EXPERIMENT_GUIDE.md"
)


def is_orchestrated() -> bool:
    """Return True if the current process was launched by hft-ops.

    hft-ops stages set ``HFT_OPS_ORCHESTRATED=1`` in the subprocess env
    before invoking trainer/backtester/evaluator scripts.
    """
    return os.environ.get(_HFT_OPS_ENV_VAR) == "1"


def warn_if_not_orchestrated(
    script_name: str,
    suggestion: str = "Wrap this script's invocation in an hft-ops manifest.",
) -> None:
    """Emit a deprecation warning if the script was invoked directly.

    Args:
        script_name: The script's own name (e.g., "train.py"). Used to
            personalize the warning.
        suggestion: One-line guidance for the migration.
    """
    if is_orchestrated():
        return

    # Build banner explicitly to avoid adjacent-string-literal concatenation
    # interacting with ``*`` operator precedence (Python rule: adjacent string
    # literals concatenate at compile time, BEFORE ``*`` and ``+`` evaluate).
    rule = "=" * 72
    lines = [
        "",
        rule,
        f"  DEPRECATION: {script_name} was invoked directly.",
        rule,
        "  This script is being migrated to run via the hft-ops orchestrator.",
        "  All experiments should go through:",
        "      hft-ops run <manifest>",
        "",
        f"  {suggestion}",
        "",
        f"  Migration guide: {_GUIDANCE_URL}",
        "  Continuing in legacy mode (no error).",
        rule,
    ]
    msg = "\n".join(lines)

    # Two visibility channels:
    #   1. stderr banner — always visible (ignores warning filters)
    #   2. warnings.warn(UserWarning) — default-visible, test-detectable.
    # We use UserWarning (not DeprecationWarning) because Python's default
    # filter suppresses DeprecationWarning triggered from imported modules,
    # which would make Phase 1.4's deprecation invisible when the warning
    # is the only signal. The banner above catches that case; this call
    # makes the signal programmatically detectable.
    print(msg, file=sys.stderr, flush=True)
    warnings.warn(
        f"{script_name} invoked directly; use 'hft-ops run <manifest>' instead. "
        f"See {_GUIDANCE_URL}",
        UserWarning,
        stacklevel=2,
    )
