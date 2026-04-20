"""Phase 8C-α Stage C.1 — compute_permutation_importance pure function.

Framework-agnostic post-training feature-importance computation via
block-permutation on a held-out eval split. Accepts callables
(predict_fn, metric_fn) so it works with PyTorch, XGBoost, ridge, or
any future model type — NO torch dependency at this boundary.

Consumes:
  - ``hft_metrics.block_permutation`` (Phase 8C-α Stage C.0) for the
    null-distribution sampling primitive.
  - ``hft_contracts.FeatureImportanceArtifact`` / ``FeatureImportance``
    (Phase 8C-α Stage C.2) for the output contract + ``compute_stability``.

Produces:
  - ``FeatureImportanceArtifact`` ready to ``.save()`` + be picked up
    by the hft-ops ledger-routing hook (Phase 8C-α Stage C.3).

-------------------------------------------------------------------------
Algorithm
-------------------------------------------------------------------------

1. OPTIONAL subsample: draw ``config.subsample`` random indices from
   the eval split (seeded by ``config.seed``). Reduces compute from
   O(full_eval) → O(subsample) at the cost of wider CI. Default
   subsample=5000 ≈ 10% of a typical 50K val split.

2. BASELINE: ``preds = predict_fn(X)``; ``baseline = metric_fn(preds, y)``.

3. PER-FEATURE LOOP (for feature_idx in feature_indices):
   a. For each seed s in range(n_seeds):
      - ``null_dist = block_permutation(score_fn, X[:, :, feature_idx_flat], y,
                                        n_permutations, block_length, seed=base_seed+s)``
        where ``score_fn(feat_perm, y)`` re-assembles X with that feature's
        values replaced by ``feat_perm`` and runs predict_fn + metric_fn.
   b. importance_mean = baseline - mean(all_null_values_across_seeds)
   c. importance_std = std(per_seed_importance_means)
   d. ci_lower_95 / ci_upper_95 = percentile(all_null_values, [2.5, 97.5])
      reversed (since importance = baseline - null, low-null-tail is
      high-importance-tail).
   e. stability = compute_stability(importance_mean, importance_std).

4. EMIT: FeatureImportanceArtifact with all per-feature records.

-------------------------------------------------------------------------
Sequence-model permutation semantics
-------------------------------------------------------------------------

For sequence inputs ``X[N, T, F]``, permuting "feature f" means replacing
the ENTIRE time-series column f for each sequence with a different
sequence's time-series column f. Concretely:

    X_perm[s, :, f] = X[perm[s], :, f]   for all s, t

This destroys the association between the feature's full trajectory and
the target label while preserving within-block (intraday) autocorrelation
if block_length matches the block-of-sequences that came from the same
day. The caller is responsible for choosing block_length consistent with
the sequences' temporal structure.

For flat inputs ``X[N, F]``, the same logic applies without the T axis.

-------------------------------------------------------------------------
Caveats
-------------------------------------------------------------------------

- Correlated features → split importance (Strobl et al. 2007). Recorded
  in ``method_caveats: ["correlation-split"]``.
- Permutation evaluates model behavior on OUT-OF-DISTRIBUTION inputs
  (the permuted feature paired with unrelated others). Interpret
  with care.
- ``feature_index`` in the output refers to the position within the
  model's input layout (typically 0..F-1 of the resolved FeatureSet),
  NOT the global FeatureIndex enum. Consumers that want the global
  index must map via ``resolved.feature_indices[local_idx]``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from hft_contracts import (
    FEATURE_IMPORTANCE_SCHEMA_VERSION,
    FeatureImportance,
    FeatureImportanceArtifact,
)
from hft_contracts.feature_importance_artifact import compute_stability

# Post-audit (2026-04-20 Agent-A M4 / Agent-D M5): we intentionally
# DO NOT consume ``hft_metrics.bootstrap.block_permutation`` as a
# dependency. The reason is a deliberate DRY trade-off: block_permutation
# wraps a statistic_fn per-replicate, but our use case must run a MODEL
# FORWARD PASS per replicate — far more expensive than the wrapping
# overhead savings. We re-implement the identical block-shuffle logic
# inline (~20 LOC) to avoid the statistic_fn indirection. If a future
# refactor makes the forward pass comparable-cost to the wrapping, we
# should route through block_permutation. This trade-off is documented
# here in code (not a silent duplication).

from lobtrainer.training.importance.config import ImportanceConfig

logger = logging.getLogger(__name__)


__all__ = [
    "compute_permutation_importance",
]


def compute_permutation_importance(
    *,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    feature_indices: Sequence[int],
    predict_fn: Callable[[np.ndarray], np.ndarray],
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    config: ImportanceConfig,
    feature_set_ref: Optional[dict] = None,
    experiment_id: str = "",
    fingerprint: str = "",
    model_type: str = "",
) -> FeatureImportanceArtifact:
    """Compute per-feature block-permutation importance on a held-out
    eval split.

    See module docstring for algorithm + caveats.

    Args:
      X: Input features. Shape ``(N,)`` / ``(N, F)`` / ``(N, T, F)`` —
        the last axis is always the feature axis. Not mutated.
      y: Targets. Shape ``(N,)`` or ``(N, H)``. Not mutated.
      feature_names: Per-feature semantic names. Length == F (the last
        axis of X).
      feature_indices: Per-feature GLOBAL indices (e.g., 0-147 in the
        148-feature layout). Same length as feature_names — enables
        downstream mapping from local-position → global index.
      predict_fn: ``predict_fn(X) -> preds``. Deterministic under the
        trainer's seed contract. Must accept the same shape as X and
        return predictions aligned with y.
      metric_fn: ``metric_fn(preds, y) -> float``. Higher is better for
        baseline_metric interpretation (e.g., IC / R² / accuracy). If
        the true metric is lower-is-better (MAE), caller should return
        -MAE so importance = baseline_minus_permuted stays positive-is-
        better.
      config: ``ImportanceConfig`` controlling n_permutations / n_seeds
        / subsample / block_size_days / seed / eval_split.
      feature_set_ref: Optional ``{name, content_hash}`` dict. Persisted
        on the artifact so downstream feedback-merge can match artifacts
        to evaluator profiles.
      experiment_id / fingerprint / model_type: Provenance fields.

    Returns:
      ``FeatureImportanceArtifact`` ready to ``.save()``.

    Raises:
      ValueError: if config is invalid (already validated at construction
        time, but re-checked here as a safety net), or if X/y/feature
        shapes disagree.
    """
    # ---- Shape + config validation -----------------------------------
    n_samples = len(X)
    if len(y) != n_samples:
        raise ValueError(
            f"X/y sample-count mismatch: X has {n_samples}, y has {len(y)}"
        )
    n_features = X.shape[-1] if X.ndim >= 2 else 1
    if len(feature_names) != n_features:
        raise ValueError(
            f"feature_names length ({len(feature_names)}) must match "
            f"X last-axis size ({n_features})"
        )
    if len(feature_indices) != n_features:
        raise ValueError(
            f"feature_indices length ({len(feature_indices)}) must "
            f"match X last-axis size ({n_features})"
        )

    # ---- Subsample for compute budget --------------------------------
    if config.subsample > 0 and config.subsample < n_samples:
        sub_rng = np.random.RandomState(config.seed)
        sub_idx = sub_rng.choice(n_samples, size=config.subsample, replace=False)
        # Sort to preserve temporal order (important for block_permutation)
        sub_idx = np.sort(sub_idx)
        X_eval = X[sub_idx]
        y_eval = y[sub_idx]
    else:
        X_eval = X
        y_eval = y
    n_eval = len(X_eval)

    logger.info(
        "Permutation importance: eval N=%d, features=%d, "
        "n_permutations=%d, n_seeds=%d, block_length_samples=%d",
        n_eval, n_features, config.n_permutations, config.n_seeds,
        config.block_length_samples,
    )

    # ---- Baseline metric ---------------------------------------------
    baseline_preds = predict_fn(X_eval)
    baseline_value = float(metric_fn(baseline_preds, y_eval))
    # Post-audit (2026-04-20 Agent-D M2): non-finite baseline poisons
    # every downstream importance calculation (baseline - null → NaN for
    # every feature, ci_lower/upper → NaN, stability → 0). Previously
    # we logged a warning and continued; fail-loud per hft-rules §8 is
    # the right disposition — the operator must fix the metric or data
    # before an importance artifact is worth trusting.
    if not math.isfinite(baseline_value):
        raise ValueError(
            f"Permutation importance: baseline metric is non-finite "
            f"({baseline_value}). Cannot compute importance deltas. "
            f"Verify predict_fn returns finite predictions on the eval "
            f"split + metric_fn handles NaN inputs correctly."
        )

    # ---- Per-feature loop --------------------------------------------
    # Block length in SAMPLE units (post-audit 2026-04-20 Agent-D H1
    # rename: was misleadingly named ``block_size_days``). Callers that
    # want DAY-preserving block permutation pass
    # ``block_length_samples = round(n_eval / n_days_in_eval)`` —
    # trainer has this via its per-split sample counts.
    block_length_samples = max(1, config.block_length_samples)

    # Post-audit round-2 (2026-04-20 arch-S3 + trainer-review): same
    # degenerate-null guard as ``hft_metrics.bootstrap.block_permutation``.
    # If ``block_length_samples >= n_eval`` the partition collapses to a
    # single block → every Fisher-Yates permutation is the identity →
    # null distribution has zero width → callers would silently report
    # p=0 / CI-width=0 against a nonexistent null. Fail loud per §8.
    # (Mirrors the guard at bootstrap.py line 187; inline here per the
    # documented DRY trade-off at top of this module. If either copy
    # evolves, the other must track.)
    _block_starts_precheck = np.arange(0, n_eval, block_length_samples)
    if len(_block_starts_precheck) < 2:
        raise ValueError(
            f"compute_permutation_importance degenerate: "
            f"block_length_samples={block_length_samples} ≥ n_eval={n_eval} "
            f"produces a single block. Every permutation is the identity → "
            f"zero-width null distribution → CI-width=0 and p=0 against "
            f"a non-existent null. Pass a smaller block_length_samples "
            f"(default 1 = element-wise permutation; for autocorrelation "
            f"preservation, pass ~round(n_eval / n_days_in_eval) so at "
            f"least 2 day-blocks exist)."
        )

    # Pre-allocate working tensor once; mutate in-place per feature, then
    # restore. Avoids allocating ~196MB × n_features × n_seeds copies.
    X_work = X_eval.copy()

    features_out: List[FeatureImportance] = []
    for local_idx, (fname, gidx) in enumerate(zip(feature_names, feature_indices)):
        # Collect per-seed null distributions
        all_null_values: List[float] = []
        per_seed_means: List[float] = []
        n_seeds_aggregated_actual = 0  # Post-audit Agent-D M3: track seeds
                                       # that produced >=1 finite permutation

        for seed_offset in range(config.n_seeds):
            # Post-audit (2026-04-20 Agent-D H2): per-(feature, seed) RNG
            # seed is offset by ``local_idx * config.n_seeds`` so each
            # (feature, seed) pair gets an INDEPENDENT permutation
            # sequence. Previously every feature shared the same
            # ``config.seed + seed_offset`` → rank-correlated
            # importance estimates that bias downstream A-vs-B
            # comparisons (especially problematic for feedback-merge
            # ensemble logic in Phase 8C-β).
            seed = config.seed + seed_offset + local_idx * config.n_seeds

            # Generate block-permuted indices directly (block-aware).
            # Inline rather than calling hft_metrics.block_permutation —
            # see the Phase 8C-α Agent-A M4 / Agent-D M5 comment at top
            # of this module for the DRY trade-off rationale.
            rng = np.random.RandomState(seed)

            # Partition into blocks at the sample level
            block_starts = np.arange(0, n_eval, block_length_samples)
            blocks = [
                np.arange(s, min(s + block_length_samples, n_eval))
                for s in block_starts
            ]
            n_blocks = len(blocks)

            seed_null_values: List[float] = []
            for _perm_idx in range(config.n_permutations):
                block_order = rng.permutation(n_blocks)
                perm = np.concatenate([blocks[b] for b in block_order])
                assert len(perm) == n_eval, (
                    f"Permutation length mismatch: expected {n_eval}, "
                    f"got {len(perm)}"
                )

                # Permute the feature column in X_work
                if X_eval.ndim == 3:
                    X_work[:, :, local_idx] = X_eval[perm, :, local_idx]
                elif X_eval.ndim == 2:
                    X_work[:, local_idx] = X_eval[perm, local_idx]
                else:
                    X_work[:] = X_eval[perm]

                preds_perm = predict_fn(X_work)
                metric_perm = float(metric_fn(preds_perm, y_eval))
                if math.isfinite(metric_perm):
                    seed_null_values.append(metric_perm)

                # Restore X_work (cheap — just the one column)
                if X_eval.ndim == 3:
                    X_work[:, :, local_idx] = X_eval[:, :, local_idx]
                elif X_eval.ndim == 2:
                    X_work[:, local_idx] = X_eval[:, local_idx]
                else:
                    X_work[:] = X_eval

            all_null_values.extend(seed_null_values)
            if seed_null_values:
                seed_mean_null = float(np.mean(seed_null_values))
                per_seed_means.append(baseline_value - seed_mean_null)
                n_seeds_aggregated_actual += 1
            # Post-audit (2026-04-20 Agent-D M3): DROP empty seeds
            # entirely. Appending 0.0 shifts the cross-seed mean + inflates
            # std (a failed seed is not evidence of "zero importance" — it
            # is evidence of a pipeline failure on that seed). Downstream
            # consumers read ``n_seeds_aggregated`` to judge reliability;
            # they are miscalibrated if we pad with 0.0.

        # ---- Aggregate per-seed importances ----
        if all_null_values:
            overall_mean_null = float(np.mean(all_null_values))
            importance_mean = baseline_value - overall_mean_null
            importance_std = (
                float(np.std(per_seed_means, ddof=1))
                if len(per_seed_means) > 1
                else 0.0
            )
            # CI on importance: percentile of (baseline - null)
            ci_lower_95 = float(baseline_value - np.percentile(all_null_values, 97.5))
            ci_upper_95 = float(baseline_value - np.percentile(all_null_values, 2.5))
        else:
            importance_mean = 0.0
            importance_std = 0.0
            ci_lower_95 = 0.0
            ci_upper_95 = 0.0

        stability = compute_stability(importance_mean, importance_std)
        features_out.append(FeatureImportance(
            feature_name=str(fname),
            feature_index=int(gidx),
            importance_mean=importance_mean,
            importance_std=importance_std,
            ci_lower_95=ci_lower_95,
            ci_upper_95=ci_upper_95,
            n_permutations=config.n_permutations,
            # Post-audit Agent-D M3: surface the ACTUAL seeds that
            # contributed — not the configured ceiling. Consumers
            # use this to down-weight features whose estimate rests
            # on fewer-than-configured seeds.
            n_seeds_aggregated=n_seeds_aggregated_actual,
            stability=stability,
        ))

    # ---- Resolve baseline_metric name --------------------------------
    baseline_metric_name = config.baseline_metric
    if baseline_metric_name == "auto":
        baseline_metric_name = "val_ic"  # MVP default; caller overrides
                                         # for classification

    artifact = FeatureImportanceArtifact(
        schema_version=FEATURE_IMPORTANCE_SCHEMA_VERSION,
        method=config.method,
        baseline_metric=baseline_metric_name,
        baseline_value=baseline_value,
        block_length_samples=config.block_length_samples,
        n_permutations=config.n_permutations,
        n_seeds=config.n_seeds,
        seed=config.seed,
        eval_split=config.eval_split,
        features=tuple(features_out),
        feature_set_ref=feature_set_ref,
        experiment_id=experiment_id,
        fingerprint=fingerprint,
        model_type=model_type,
        timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        method_caveats=("correlation-split",),
    )
    return artifact
