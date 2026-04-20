"""Phase 8C-α Stage C.1 — ImportanceConfig dataclass.

Per-experiment configuration for post-training feature-importance.
Frozen — locked at experiment-config parse time. Default
``enabled=False`` preserves pre-Phase-8C-α behavior (no importance
computed, no artifact emitted).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# Default constants (sensible MVP targets — compute-cost balanced
# against statistical reliability):
#
# - ``n_permutations = 100``: gives ~10% CI width on a metric with
#   std ~0.03 (typical val_ic std on NVDA regression). 500 would be
#   tighter but costs 5× compute.
# - ``n_seeds = 3``: minimum for across-seed std estimate; stability
#   metric requires ≥ 2. K=5 (plan default) shifts to 8C-β when we
#   have real empirical evidence on stability floor.
# - ``subsample = 5000``: ~10% of a typical 50K val split. Noise-
#   dominant but reproducible given the seed contract. Set to -1 to
#   use full eval split (≈14h compute per experiment — advanced use).
# - ``block_size_days = 1``: matches our intraday autocorrelation
#   scale (OFI ACF=0.266 @ 5m XNAS). Larger → less null-spread
#   destruction; smaller → approaches element-wise permutation.

IMPORTANCE_DEFAULT_METHOD: Literal["permutation"] = "permutation"
IMPORTANCE_DEFAULT_N_PERMUTATIONS: int = 100
IMPORTANCE_DEFAULT_N_SEEDS: int = 3
IMPORTANCE_DEFAULT_SUBSAMPLE: int = 5000  # -1 = full eval split
IMPORTANCE_DEFAULT_BLOCK_SIZE_DAYS: int = 1
IMPORTANCE_DEFAULT_SEED: int = 42
IMPORTANCE_DEFAULT_EVAL_SPLIT: Literal["test", "val"] = "test"


@dataclass(frozen=True)
class ImportanceConfig:
    """Per-experiment configuration for post-training feature importance.

    Frozen — locked at experiment-config parse time. Default
    ``enabled=False`` preserves pre-Phase-8C-α behavior (no artifact).

    Compute cost: O(n_features × n_permutations × n_seeds × eval_time).
    On a typical 98-feature sequence model with ``subsample=5000``:
      - defaults (100 × 3) → ~15-30 min per experiment
      - ``n_permutations=500, n_seeds=5`` → ~2-4 hours
      - ``subsample=-1, n_permutations=500, n_seeds=5`` → 10-20 hours

    Fields:
      enabled: Master switch. Default False = no-op (no artifact, no
        compute cost). Operators enable per-experiment for specific
        runs of interest (e.g., best model from a sweep).
      method: "permutation" for MVP. Future: "shap" / "integrated_gradients".
      n_permutations: Per-seed permutation replicate count. Higher =
        tighter CI, longer compute. Statistical minimum is ~50 for
        crude CI shape; 100+ for reliable CI.
      n_seeds: Number of random seeds aggregated. Minimum 2 for std.
        K=5 per plan; we default to 3 for compute-budget reasons.
        Feedback-merge (Phase 8C-β) requires K ≥ min_models_for_feedback
        (default 5) before reconciliation, so single-experiment K=3
        is NEVER used for a hard tier-flip — only as evidence in a
        cross-experiment ensemble.
      subsample: Limit eval-split size (random draw with the same
        seed per importance run). -1 = use full split. Default 5000
        to keep compute tractable.
      block_size_days: Block-permutation block size, in day units.
        Preserves intraday autocorrelation. Default 1.
      seed: Base RNG seed. Per-seed seeds derived as
        ``range(seed, seed + n_seeds)``. Determinism contract §7.
      eval_split: "test" (default) or "val". Test is preferred — val
        was implicitly used for early-stopping so the model is
        indirectly optimized on it.
      baseline_metric: Metric name used for the importance delta.
        "auto" resolves at compute time based on task_type:
          - regression → "val_ic"
          - classification → "val_macro_f1"
        Explicit values: "val_ic" / "val_r2" / "val_mae" /
        "val_directional_accuracy" / "val_macro_f1" / "val_accuracy".
    """

    enabled: bool = False
    method: Literal["permutation"] = IMPORTANCE_DEFAULT_METHOD
    n_permutations: int = IMPORTANCE_DEFAULT_N_PERMUTATIONS
    n_seeds: int = IMPORTANCE_DEFAULT_N_SEEDS
    subsample: int = IMPORTANCE_DEFAULT_SUBSAMPLE
    block_size_days: int = IMPORTANCE_DEFAULT_BLOCK_SIZE_DAYS
    seed: int = IMPORTANCE_DEFAULT_SEED
    eval_split: Literal["test", "val"] = IMPORTANCE_DEFAULT_EVAL_SPLIT
    baseline_metric: str = "auto"

    def __post_init__(self) -> None:
        """Validate config at construction time (fail-loud per §5)."""
        if self.n_permutations < 1:
            raise ValueError(
                f"n_permutations must be >= 1, got {self.n_permutations}"
            )
        if self.n_seeds < 1:
            raise ValueError(f"n_seeds must be >= 1, got {self.n_seeds}")
        if self.subsample == 0 or self.subsample < -1:
            raise ValueError(
                f"subsample must be -1 (use full) or positive integer, "
                f"got {self.subsample}"
            )
        if self.block_size_days < 1:
            raise ValueError(
                f"block_size_days must be >= 1, got {self.block_size_days}"
            )
        if self.method not in ("permutation",):
            raise ValueError(
                f"method must be 'permutation' (MVP); got {self.method!r}. "
                f"SHAP / IntegratedGradients are deferred to Phase 8D+."
            )
        if self.eval_split not in ("test", "val"):
            raise ValueError(
                f"eval_split must be 'test' or 'val'; got {self.eval_split!r}"
            )


def permutation_importance_enabled(config: ImportanceConfig) -> bool:
    """Gate helper — True iff the callback should actually run.

    Centralizes the enabled check so scripts + orchestrator + tests
    share one predicate. Future additions (e.g., per-model-type
    gating) extend here without touching call sites.
    """
    return config.enabled and config.method == "permutation"
