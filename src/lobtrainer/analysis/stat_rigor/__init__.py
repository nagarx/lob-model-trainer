"""Phase 2 STAT RIGOR FLOOR primitives (Cyclelet B, 2026-05-07).

Bootstrap CI on test metrics + (future) permutation importance + (future)
pairwise statistical comparison. Per hft-rules §0 reuse-first: thin
orchestration over ``hft_metrics`` primitives (``block_bootstrap_ci``,
``block_permutation``, ``pairwise_paired_bootstrap_compare``) — NO
re-derivation of statistical formulas.

Reference: Politis & Romano (1994) [block_length auto-derive]; Künsch
(1989) [moving-block bootstrap]; Plan v4 PHASE_2_STAT_RIGOR_PLAN.md §4.1.
"""

from lobtrainer.analysis.stat_rigor.ci import (
    TestMetricsCIConfig,
    compute_ci,
    from_signal_dir,
    compute_test_metrics_ci_for_experiment,
)

__all__ = [
    "TestMetricsCIConfig",
    "compute_ci",
    "from_signal_dir",
    "compute_test_metrics_ci_for_experiment",
]
