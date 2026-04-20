"""Phase 8C-α Stage C.1 — Post-training feature-importance computation.

Produces a ``hft_contracts.FeatureImportanceArtifact`` summarising each
feature's predictive contribution via BLOCK PERMUTATION on a held-out
eval split. Consumed downstream by the feature-evaluator feedback-merge
step (Phase 8C-β Stage C.5).

Public API:
  - ``ImportanceConfig`` — per-experiment configuration dataclass.
  - ``compute_permutation_importance`` — framework-agnostic pure
    function. Takes callables + raw tensors; returns
    ``FeatureImportanceArtifact``. Works with PyTorch, XGBoost, or any
    future model type.
  - ``permutation_importance_enabled(config)`` — gate helper.

Default ``enabled=False`` — operator opts in per-experiment via trainer
config:

```yaml
importance:
  enabled: true
  n_permutations: 100
  n_seeds: 3
  subsample: 5000
```

References:
- Breiman (2001), Random Forests. Permutation importance baseline.
- Politis & Romano (1994). Block permutation for time-series.
- Strobl et al. (2007). Correlation-split bias caveat.
"""

from lobtrainer.training.importance.config import (
    IMPORTANCE_DEFAULT_BLOCK_SIZE_DAYS,
    IMPORTANCE_DEFAULT_EVAL_SPLIT,
    IMPORTANCE_DEFAULT_METHOD,
    IMPORTANCE_DEFAULT_N_PERMUTATIONS,
    IMPORTANCE_DEFAULT_N_SEEDS,
    IMPORTANCE_DEFAULT_SEED,
    IMPORTANCE_DEFAULT_SUBSAMPLE,
    ImportanceConfig,
    permutation_importance_enabled,
)
from lobtrainer.training.importance.permutation import (
    compute_permutation_importance,
)

__all__ = [
    "IMPORTANCE_DEFAULT_BLOCK_SIZE_DAYS",
    "IMPORTANCE_DEFAULT_EVAL_SPLIT",
    "IMPORTANCE_DEFAULT_METHOD",
    "IMPORTANCE_DEFAULT_N_PERMUTATIONS",
    "IMPORTANCE_DEFAULT_N_SEEDS",
    "IMPORTANCE_DEFAULT_SEED",
    "IMPORTANCE_DEFAULT_SUBSAMPLE",
    "ImportanceConfig",
    "compute_permutation_importance",
    "permutation_importance_enabled",
]
