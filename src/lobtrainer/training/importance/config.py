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
IMPORTANCE_DEFAULT_BLOCK_LENGTH_SAMPLES: int = 1
# Post-audit (2026-04-20 Agent-D H1): renamed from ``block_size_days`` to
# ``block_length_samples`` to match the code's actual semantics. Callers
# that want DAY-preserving block permutation must pass
# ``block_length_samples = n_samples_per_day`` explicitly — the trainer
# has the per-split sample counts available via its data loader. The old
# name silently implied autocorrelation preservation that the code did
# not deliver (Politis & Romano 1994 semantics require block_length >
# autocorrelation lag; block_length=1 is element-wise permutation).
# Back-compat alias kept below for transition.
IMPORTANCE_DEFAULT_BLOCK_SIZE_DAYS: int = IMPORTANCE_DEFAULT_BLOCK_LENGTH_SAMPLES
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
      block_length_samples: Block-permutation block size, in SAMPLE units
        (NOT day units — post-audit 2026-04-20 rename for semantic
        accuracy). Default 1 = element-wise permutation (no
        autocorrelation preservation). To preserve intraday
        autocorrelation, caller must pass
        ``block_length_samples = round(n_eval / n_days_in_eval)`` — the
        trainer has this via its per-split sample counts. When n_eval
        is a pre-subsample count, operator must scale by subsample
        ratio.
      block_size_days: DEPRECATED alias for ``block_length_samples``
        (maintained for config back-compat until Phase 8C-β). Emits
        DeprecationWarning on use. Remove 2026-10-31.
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
    block_length_samples: int = IMPORTANCE_DEFAULT_BLOCK_LENGTH_SAMPLES
    seed: int = IMPORTANCE_DEFAULT_SEED
    eval_split: Literal["test", "val"] = IMPORTANCE_DEFAULT_EVAL_SPLIT
    baseline_metric: str = "auto"

    # Back-compat deprecated alias (post-audit 2026-04-20 Agent-D H1).
    # Use ``block_length_samples`` instead. If both are set to different
    # values, ``__post_init__`` raises to prevent silent preference bugs.
    # Removal target: 2026-10-31.
    block_size_days: int = -1  # sentinel for "not explicitly set"

    def __post_init__(self) -> None:
        """Validate config at construction time (fail-loud per §5)."""
        # -------- Back-compat alias resolution -------------------------
        if self.block_size_days != -1:
            # User passed the old name
            import warnings
            warnings.warn(
                "ImportanceConfig.block_size_days is deprecated (2026-04-20) — "
                "use block_length_samples instead. The old name was "
                "misleading: the code always treated it as a SAMPLE count, "
                "never a DAY count. See Phase 8C-α Agent-D H1. "
                "Removal target: 2026-10-31.",
                DeprecationWarning,
                stacklevel=3,
            )
            if (self.block_length_samples != IMPORTANCE_DEFAULT_BLOCK_LENGTH_SAMPLES
                    and self.block_length_samples != self.block_size_days):
                raise ValueError(
                    f"ImportanceConfig: block_size_days={self.block_size_days} "
                    f"and block_length_samples={self.block_length_samples} "
                    f"disagree. Use only block_length_samples (the new name)."
                )
            object.__setattr__(
                self, "block_length_samples", self.block_size_days,
            )

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
        if self.block_length_samples < 1:
            raise ValueError(
                f"block_length_samples must be >= 1, got "
                f"{self.block_length_samples}"
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
