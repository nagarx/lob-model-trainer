# Base Configuration Hierarchy

This directory contains orthogonal base configs composed via multi-base inheritance (`_base: [list]`). Phase 3 of the training-pipeline-architecture refactor (2026-04-15) introduced this structure to replace the pre-existing monolith `e5_tlob_regression.yaml` (retired at end of Phase 3b).

## Directory Structure

```
bases/
├── models/                  # Model architecture bases
├── datasets/                # Dataset-level settings (data_dir, feature_count, normalization)
├── labels/                  # Label strategy + horizon + task_type
└── train/                   # Training hyperparameters (batch_size, epochs, scheduler, etc.)
```

## Composition Recipe

An experiment config composes four bases plus per-child overrides:

```yaml
# configs/experiments/my_experiment.yaml
_base:
  - "../bases/models/tlob_compact_regression.yaml"
  - "../bases/datasets/nvda_e5_60s.yaml"
  - "../bases/labels/regression_huber.yaml"
  - "../bases/train/regression_default.yaml"

name: my_experiment
description: "what this experiment tests"
tags: [e5, regression, my_variant]
output_dir: outputs/experiments/my_experiment

# Per-experiment overrides on top of the 4 bases:
model:
  tlob_use_cvml: true
  tlob_cvml_out_channels: 49
```

Merge semantics: bases merge **left-to-right** (each base overrides the previous); the child config overrides all accumulated bases.

## Field Ownership Rule (§3.4 of parent plan)

Each top-level dotted-key is owned by **exactly one axis**. This is enforced mechanically by `tests/test_base_axis_ownership.py` — CI fails if any field appears in more than one axis's bases.

| Axis | Owns | MUST NOT set |
|------|------|-------------|
| `models/` | `model.model_type`, `model.dropout`, `model.regression_loss_type`, `model.regression_loss_delta`, `model.task_type`, `model.tlob_*`, `model.hmhp_*`, `model.logistic_feature_indices` | `model.num_classes`, `model.input_size`, `train.task_type`, `train.loss_type` (moved to labels/ in Batch 2), `train.batch_size` |
| `datasets/` | `data.data_dir`, `data.feature_count`, `data.sources`, `data.normalization`, `data.feature_preset`, `data.feature_set`, `data.feature_sets_dir`, `data.sequence.window_size`, `data.sequence.stride`, `model.input_size` (locked to `feature_count` per T13 auto-derivation; when `data.feature_set` is set, T13 is DEFERRED to `_create_dataloaders` resolver-time per Phase 6 6A.1) | `data.labeling_strategy`, `data.horizon_idx`, `model.num_classes` |
| `labels/` | `data.labeling_strategy`, `data.horizon_idx`, `data.num_classes`, `data.labels` (T9 unified spec), `model.num_classes`, `train.task_type`, `train.loss_type` (task-coupled: regression→huber, tlob→weighted_ce, triple_barrier→focal — moved from models/ in Batch 2) | `model.*` (other than num_classes), `data.feature_count` |
| `train/` | `train.batch_size`, `train.epochs`, `train.optimizer`, `train.scheduler`, `train.learning_rate`, `train.weight_decay`, `train.seed`, `train.gradient_clip_norm`, `train.mixed_precision`, `train.early_stopping_patience`, `train.num_workers`, `train.pin_memory`, `train.use_class_weights`, `train.focal_gamma` (focal-loss hyperparameter, not model-family-specific) | `train.task_type`, `train.loss_type`, `model.*`, `data.*` |

**Per-child (NOT in any base)**: `name`, `description`, `tags`, `output_dir`, `log_level`.

## Partial Base Sentinel

Every base declares `_partial: true` at the top level. This is a marker meaning "this file is standalone-invalid — it only becomes a valid config when composed with peer bases via multi-base `_base: [...]`".

If a researcher accidentally runs `ExperimentConfig.from_yaml("bases/models/tlob_compact_regression.yaml")`, they get a descriptive error pointing at this README rather than a confusing dacite missing-field failure deep in the validator. See `src/lobtrainer/config/merge.py::is_partial_base` for the detection.

## Current Inventory (2026-04-15, post-Batch-3 + monolith retirement)

**21 axis-partitioned bases.** Monolith `e5_tlob_regression.yaml` RETIRED.

### models/ (5)
| File | Used by |
|---|---|
| `models/tlob_compact_bare.yaml` | E4 TLOB (1 config) — standalone without cvml defaults |
| `models/tlob_compact_regression.yaml` | E5 family (5), E6 — chains from `tlob_compact_bare.yaml`, adds cvml defaults |
| `models/tlob_paper_classification.yaml` | TLOB paper-spec classif (7 configs) — hidden=64, layers=4, heads=1 (distinct arch from compact) |
| `models/hmhp_cascade_bare.yaml` | HMHP classification (6), HMHP TB (3), HMHP hybrid (1) — 10 of 11 HMHP configs |
| `models/hmhp_cascade_regression.yaml` | HMHP regression (2) — chains from `hmhp_cascade_bare.yaml`, adds `model_type: hmhp_regression` + `hmhp_regression_loss_type: huber` |

### datasets/ (8)
| File | Used by |
|---|---|
| `datasets/nvda_e4_5s.yaml` | E4 TLOB (1) |
| `datasets/nvda_e5_30s.yaml` | E5 30s (2) |
| `datasets/nvda_e5_60s.yaml` | E5 60s (3), E6 |
| `datasets/nvda_xnas_128feat_full.yaml` | HMHP 128feat classif (3) |
| `datasets/nvda_40feat_short_term.yaml` | HMHP 40feat classif (3), HMHP regression (1) |
| `datasets/nvda_98feat_triple_barrier.yaml` | HMHP TB (3) |
| `datasets/nvda_98feat_zscore_per_day.yaml` | TLOB classif h10_v1, h100_v1 (2) — zscore_per_day normalization with exclude=[93] |
| `datasets/nvda_40feat_tlob_repo.yaml` | TLOB classif repo_match_h50, v2_h100 (2) — matches official TLOB repo preprocessing (global_zscore, clip=null) |

### labels/ (4)
| File | Used by |
|---|---|
| `labels/regression_huber.yaml` | All regression (E4, E5, E6, HMHP reg×2) — owns `train.loss_type: huber` (Batch 2 refinement) |
| `labels/tlob_smoothed.yaml` | HMHP TLOB classif (5) + TLOB paper classif (7) = 12 configs total |
| `labels/opportunity.yaml` | HMHP opportunity (1) |
| `labels/triple_barrier_volscaled.yaml` | HMHP TB (3) — intentionally OMITS `horizon_idx` |

### train/ (4)
| File | Used by |
|---|---|
| `train/regression_default.yaml` | All regression (E-family + HMHP reg) |
| `train/classification_default.yaml` | HMHP classification (6) |
| `train/classification_triple_barrier.yaml` | HMHP TB (3) — focal loss regime (focal_gamma, longer training, smaller batch) |
| `train/tlob_paper_classification_train.yaml` | TLOB paper classif (7) — distinct recipe (batch=64, lr=1e-4, epochs=50) vs HMHP classification |

### Chained-inheritance patterns

Two chained-inheritance patterns locked by tests in `test_base_axis_ownership.py::TestChainedInheritancePurity`:

**Pattern 1: TLOB compact bare → regression** — `tlob_compact_regression.yaml` uses `_base: "tlob_compact_bare.yaml"` to inherit 12 core TLOB arch fields, then adds `tlob_use_cvml: false` + `tlob_cvml_out_channels: 0` on top. E5/E6 use the full chain (need those cvml defaults to preserve byte-identity with their pre-migration golden fixtures). E4 uses `tlob_compact_bare.yaml` DIRECTLY.

**Pattern 2: HMHP cascade bare → regression** — `hmhp_cascade_regression.yaml` uses `_base: "hmhp_cascade_bare.yaml"` to inherit 10 base model fields (`model_type: hmhp` + `dropout` + 8 `hmhp_*` arch fields), then **overrides** `model_type: hmhp_regression` and **adds** `hmhp_regression_loss_type: huber` on top. HMHP classification + TB configs use bare DIRECTLY (those regression fields would corrupt their goldens). HMHP regression configs use the chain.

**Lock tests** (all in `test_base_axis_ownership.py`):
- `test_tlob_compact_bare_excludes_cvml_fields`
- `test_tlob_compact_regression_adds_cvml_fields_only`
- `test_hmhp_cascade_bare_model_type_is_hmhp`
- `test_hmhp_cascade_bare_excludes_regression_fields`
- `test_hmhp_cascade_regression_adds_regression_fields_only`
- `test_triple_barrier_label_omits_horizon_idx`

A future maintainer who mistakenly violates the split will fail CI immediately with a clear diagnostic pointing at the cause.

### Ownership refinement (Batch 2)

`train.loss_type` ownership moved from `models/` to `labels/` — it's TASK-coupled (regression→huber, tlob→weighted_ce, triple_barrier→focal), not model-coupled. HMHP uses the same cascade model across all 3 loss types, so model ownership would require 3 near-duplicate HMHP model bases. Label ownership aligns loss with labeling strategy cleanly. Batch 1 parity preserved (resolved dicts unchanged).

---

## Future bases (land with upcoming batches)

**Batch 2 (HMHP, 14 configs):**
- `models/hmhp_cascade_classification.yaml`, `models/hmhp_cascade_regression.yaml`, `models/hmhp_triple_barrier.yaml`
- `datasets/nvda_xnas_128feat.yaml`, `datasets/nvda_40feat_short_term.yaml`
- `labels/tlob_smoothed.yaml`, `labels/triple_barrier_volscaled.yaml`
- `train/classification_default.yaml`

**Batch 3 (TLOB classification, 7 configs):**
- `models/tlob_paper_classification.yaml` (hidden=64, layers=4, heads=1 — different arch from compact)
- `datasets/nvda_xnas_98feat.yaml` (if different from e5_60s variant)

**Batch 4 (Baselines, 7 configs):**
- `models/logistic_baseline.yaml`, `models/temporal_ridge.yaml`, `models/temporal_gradboost.yaml`

Migration progress (post-Batch-3 + monolith retirement): **25 of 42** in-scope configs migrated. **21 axis-partitioned bases** covering 3 model families (TLOB compact regression, TLOB paper classification, HMHP cascade). 17 configs stay standalone by design (baselines, XGBoost, archive, 2 niche HMHP). See `MERGE_MIGRATION_PLAN.md` for per-batch status and retirement record.

## Related Docs

- `/Users/knight/code_local/HFT-pipeline-v2/lob-model-trainer/MERGE_MIGRATION_PLAN.md` — migration ledger
- `/Users/knight/code_local/HFT-pipeline-v2/lob-model-trainer/src/lobtrainer/config/merge.py` — v2 inheritance implementation
- `/Users/knight/code_local/HFT-pipeline-v2/lob-model-trainer/src/lobtrainer/config/archive/merge-v1/ARCHIVE_README.md` — v1 archive
- `/Users/knight/.claude/plans/gentle-brewing-quail.md` §3.4 — ownership rule source
