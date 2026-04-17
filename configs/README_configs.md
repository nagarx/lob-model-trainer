# Configuration Reference for LOB Model Training

> **For LLM Coders**: This is the complete reference for all experiment configurations.
> Use this to select the right config for any experiment or to create new ones.

## Quick Start

```bash
# Run the primary TLOB experiment (H=10, short-term)
python scripts/train.py --config configs/experiments/nvda_tlob_h10_v1.yaml

# Run paper benchmark (H=100)
python scripts/train.py --config configs/experiments/nvda_tlob_h100_v1.yaml

# Run Triple Barrier experiment
python scripts/train.py --config configs/experiments/nvda_tlob_triple_barrier_11mo_v1.yaml
```

---

## Directory Structure

```
configs/
├── README_configs.md                      # This file (complete reference)
├── bases/                                 # 🏗️ 21 axis-partitioned BASE configs (Phase 3)
│   ├── README.md                          #   4-axis ownership rule + chained inheritance
│   ├── models/                            #   5 files — model architecture bases
│   │   ├── tlob_compact_bare.yaml
│   │   ├── tlob_compact_regression.yaml   #   chains from bare (cvml defaults added)
│   │   ├── tlob_paper_classification.yaml
│   │   ├── hmhp_cascade_bare.yaml
│   │   └── hmhp_cascade_regression.yaml   #   chains from bare (regression head added)
│   ├── datasets/                          #   8 files — per-export bases
│   │   ├── nvda_e4_5s.yaml
│   │   ├── nvda_e5_30s.yaml
│   │   ├── nvda_e5_60s.yaml
│   │   ├── nvda_xnas_128feat_full.yaml
│   │   ├── nvda_40feat_short_term.yaml
│   │   ├── nvda_98feat_triple_barrier.yaml
│   │   ├── nvda_98feat_zscore_per_day.yaml
│   │   └── nvda_40feat_tlob_repo.yaml
│   ├── labels/                            #   4 files — label strategy bases
│   │   ├── regression_huber.yaml
│   │   ├── tlob_smoothed.yaml
│   │   ├── opportunity.yaml
│   │   └── triple_barrier_volscaled.yaml
│   └── train/                             #   4 files — training hyperparam bases
│       ├── regression_default.yaml
│       ├── classification_default.yaml
│       ├── classification_triple_barrier.yaml
│       └── tlob_paper_classification_train.yaml
│   # Monolith bases/e5_tlob_regression.yaml was RETIRED 2026-04-15 at end of Batch 1.
│
├── experiments/                           # ✅ 42 in-scope ACTIVE configs
│   #   25 migrated to axis-composed _base: [...] form (E4×1, E5×5, E6×1, HMHP×11, TLOB classif×7)
│   #   17 standalone by design (baselines, XGBoost, archive-of-configs, niche HMHP, TLOB singletons)
│   #   See MERGE_MIGRATION_PLAN.md for the per-batch migration ledger.
│
└── archive/                               # 📦 REFERENCE configs (6) — legacy datasets, not in Phase 3 migration scope
    ├── baseline_lstm.yaml                 # Pure LSTM model
    ├── lstm_attn_bidir_h20.yaml          # LSTM + Attention + Bidirectional
    ├── deeplob_benchmark.yaml            # DeepLOB (Zhang et al. 2019)
    ├── nvda_bigmove_opportunity_v1.yaml  # Opportunity labeling
    ├── nvda_tlob_bigmove_v1.yaml         # TLOB + Opportunity
    └── nvda_tlob_binary_signal_v1.yaml   # Binary classification + Focal Loss
```

Related archive:
- `src/lobtrainer/config/archive/merge-v1/` — v1 `merge.py` (single-string `_base:` only) preserved for byte-identity parity testing. See `ARCHIVE_README.md` inside that directory. Mirrors `feature-extractor-MBO-LOB/archive/monolith-v1/` precedent.

---

## Config Inheritance (Phase 3 — `_base: str | list[str]`)

Configs compose via the `_base` key. A child config inherits all values from its bases and overrides only the fields that differ.

### v2 Multi-Base Composition (preferred — axis-composed)

```yaml
# configs/experiments/e5_60s_huber_cvml.yaml
_base:
  - "../bases/models/tlob_compact_regression.yaml"   # model architecture
  - "../bases/datasets/nvda_e5_60s.yaml"             # dataset + normalization
  - "../bases/labels/regression_huber.yaml"          # label strategy + loss
  - "../bases/train/regression_default.yaml"         # training hyperparams

name: E5_60s_Huber_CVML
description: "E5 60s-bin regression with CVML encoder"
tags: [e5, nvda, regression, h10, 60s, huber, cvml]
output_dir: outputs/experiments/e5_60s_huber_cvml

# Per-child overrides on top of the 4 composed bases:
model:
  tlob_use_cvml: true
  tlob_cvml_out_channels: 49
```

### v1 Single-Base Form (backward-compatible)

```yaml
# Still supported, still the right form for standalone experiments that
# don't fit the 4-axis decomposition (singletons, variants).
_base: "../bases/<some-base>.yaml"
```

### Semantics

- **List form**: bases merge **left-to-right** — each successive base overrides the previous; child config overrides all accumulated bases
- **Dicts**: recursively merged (base keys preserved unless overridden)
- **Lists**: REPLACED entirely (tags, horizons, exclude_features)
- **Scalars**: child value wins
- **`null`**: explicitly sets a value to None (use to clear `feature_preset` when switching to `feature_indices`)
- **Chained inheritance**: a base may itself have `_base` (max depth: 10, per-branch cycle detection)
- **`_base` paths**: resolved relative to the file containing the `_base` key

### Axis Ownership Rule (§3.4)

Each top-level dotted-key is owned by **exactly one axis**. Mechanically enforced by `tests/test_base_axis_ownership.py` — CI fails if any field appears in more than one axis's bases.

| Axis | Owns (high-level) | Must NOT set |
|------|-------------------|--------------|
| `models/` | `model.model_type`, `model.dropout`, `model.tlob_*`, `model.hmhp_*`, `model.regression_loss_type` | `model.num_classes`, `model.input_size`, `train.task_type`, `train.loss_type`, `train.batch_size` |
| `datasets/` | `data.data_dir`, `data.feature_count`, `data.normalization`, `data.sequence`, `model.input_size` (T13 auto-derivation) | `data.labeling_strategy`, `data.horizon_idx`, `model.num_classes` |
| `labels/` | `data.labeling_strategy`, `data.horizon_idx`, `data.num_classes`, `model.num_classes`, `train.task_type`, `train.loss_type` | `model.*` (other than num_classes), `data.feature_count` |
| `train/` | `train.batch_size`, `train.epochs`, `train.optimizer`, `train.scheduler`, `train.learning_rate`, `train.weight_decay`, `train.seed`, `train.gradient_clip_norm`, `train.use_class_weights`, `train.focal_gamma` | `train.task_type`, `train.loss_type`, `model.*`, `data.*` |
| **per-child (NOT in any base)** | `name`, `description`, `tags`, `output_dir`, `log_level` | (identity fields — unique per experiment) |

**Why `train.loss_type` lives in `labels/` (not `models/`)**: it is **task-coupled** (regression → `huber`, tlob → `weighted_ce`, triple_barrier → `focal`), not model-coupled. HMHP cascade shares one model base across three loss types; without this move, HMHP would require three near-duplicate HMHP model bases.

### Chained Inheritance Patterns

Two chained patterns are locked by `tests/test_base_axis_ownership.py::TestChainedInheritancePurity`:

**Pattern 1 — TLOB compact** (`tlob_compact_bare` → `tlob_compact_regression`):
- `bare` contains 12 shared arch fields
- `regression` chains from `bare` via `_base: "tlob_compact_bare.yaml"` and adds `tlob_use_cvml: false` + `tlob_cvml_out_channels: 0` on top
- **E4 TLOB** uses `bare` DIRECTLY (its pre-migration golden has no cvml fields — the chain would corrupt byte-identity)
- **E5 / E6** use the full chain (their pre-migration goldens DO have cvml fields)

**Pattern 2 — HMHP cascade** (`hmhp_cascade_bare` → `hmhp_cascade_regression`):
- `bare` contains 10 model fields (`model_type: hmhp` + `dropout` + 8 `hmhp_*` arch fields)
- `regression` chains from `bare` and adds `model_type: hmhp_regression` + `hmhp_regression_loss_type: huber` on top
- **HMHP classification / triple-barrier** use `bare` DIRECTLY (those regression fields would corrupt their goldens)
- **HMHP regression** uses the full chain

### Partial Bases (`_partial: true`)

Every axis-partitioned base declares `_partial: true` at top level. This sentinel marks the file as "standalone-invalid — only becomes a valid config when composed with peer bases via multi-base `_base: [...]`".

If a researcher accidentally runs `ExperimentConfig.from_yaml("bases/models/tlob_compact_regression.yaml")`, they get a descriptive error pointing at `configs/bases/README.md` rather than a confusing dacite missing-field failure. Detection: `src/lobtrainer/config/merge.py::is_partial_base`.

### Important Rules

1. When switching from `feature_preset` to `feature_indices`, set `feature_preset: null` in the child
2. When changing `feature_count`, also change `model.input_size` to match (datasets/ bases keep these locked together)
3. `_base` is only supported in YAML configs loaded via `from_yaml()`, not in `from_dict()`
4. Never run `ExperimentConfig.from_yaml()` on a partial base (`_partial: true`) — it will raise a descriptive error
5. Per-child overrides (`name`, `description`, `tags`, `output_dir`) MUST NOT appear in any base — every experiment owns its own identity fields

---

## Feature Selection (Phase 4 Batch 4c, 2026-04-15)

`DataConfig` exposes three mutually-exclusive feature-selection fields.
**At most one** may be set; setting two or more raises `ValueError("At most one of ...")`.

| Field | Form | Source | Lifetime |
|---|---|---|---|
| `data.feature_set` | str, e.g. `"momentum_hft_v1"` | FeatureSet registry `contracts/feature_sets/<name>.json` | **PREFERRED (Phase 4+)** |
| `data.feature_indices` | `list[int]` | Inline in the trainer YAML | Ad-hoc overrides, tests |
| `data.feature_preset` | str, e.g. `"short_term_40"` | `lobtrainer.constants.feature_presets` | **DEPRECATED** — ImportError on 2026-08-15 |

**Registry lookup** (`feature_set`): resolver loads `<name>.json`, verifies
SHA-256 content hash, then verifies `contract_version` and
`source_feature_count` match the runtime expectations. Populates a
runtime-only cache (`DataConfig._feature_indices_resolved`) that is
**stripped from `to_dict()`/`to_yaml()`/`to_json()`** at both dataclass
and dict branches — R3 invariant: the on-disk YAML round-trip
preserves the user's `feature_set: X`, never silently substituting
resolved `feature_indices: [...]`.

**Registry location**: by default `find_feature_sets_dir(data_dir)`
walks up to `contracts/feature_sets/`. Override with
`data.feature_sets_dir: "/custom/path"` for test isolation or
multi-registry workflows.

**Byte-parity guarantee**: trainer inlines `_compute_content_hash`
(~10 LOC, torch-free, cross-venv independent from hft-ops). Parity
against the producer's canonical form is locked by
`tests/test_feature_set_resolver_parity.py` via
`hft_contracts.canonical_hash` (the single source of truth).

**Producing a FeatureSet** (from `hft-feature-evaluator`):

```
hft-ops evaluate \
  --config evaluator.yaml \
  --criteria criteria.yaml \
  --save-feature-set momentum_hft_v1 \
  --applies-to-assets NVDA \
  --applies-to-horizons 60
```

---

## Datasets

### Current Datasets (Use These)

| Dataset | Days | Labels | Horizons | Status |
|---------|------|--------|----------|--------|
| `nvda_11month_complete` | **234** | TLOB (Down/Stable/Up) | [10, 20, 50, 100] | ✅ **PRIMARY** |
| `nvda_11month_triple_barrier` | **234** | Triple Barrier (StopLoss/Timeout/ProfitTarget) | [50, 100, 200] | ✅ **PRIMARY** |

### Legacy Datasets (Archive Reference Only)

| Dataset | Days | Labels | Used By |
|---------|------|--------|---------|
| `nvda_balanced` | 165 | TLOB | `baseline_lstm.yaml`, `deeplob_benchmark.yaml` |
| `nvda_bigmove` | 165 | Opportunity | `nvda_bigmove_opportunity_v1.yaml`, `nvda_tlob_bigmove_v1.yaml` |
| `nvda_triple_barrier` | 165 | Triple Barrier | Legacy experiments |

---

## Active Experiment Configs (42 in-scope — 25 migrated + 17 standalone by design)

42 experiment configs are ready to run on the current 233/234-day datasets. Phase 3 (2026-04-15) migrated 25 to the axis-composed `_base: [models/x, datasets/y, labels/z, train/w]` form across three batches:

- **Batch 1** (E-family): E4×1 + E5×5 + E6×1 = 7 configs
- **Batch 2** (HMHP): HMHP classification×6 + HMHP regression×2 + HMHP TB×3 = 11 configs
- **Batch 3** (TLOB paper-spec classification): H10/H50/H100 + raw + 98feat + repo_match + v2_h100 = 7 configs

**17 configs remain standalone by design** (out of Phase 3 migration scope):
- **Baselines** (7): logistic×4, temporal_ridge×2, temporal_gradboost×1 — planned for Batch 4 (cancelled as diminishing returns)
- **XGBoost** (2): `nvda_xgboost_baseline_h60.yaml`, `nvda_xgboost_baseline_arcx_h60.yaml` — different schema, bypasses `ExperimentConfig.from_yaml()`
- **Archive** (6): legacy datasets no longer supported
- **Niche HMHP** (2): `nvda_short_term_hmhp_v1.yaml`, `nvda_hmhp_multihorizon_v1.yaml` — too few peers to benefit from shared bases

Representative examples (not exhaustive — see `MERGE_MIGRATION_PLAN.md` for the full per-config ledger):

| Config | Model | Horizon | Labels | Purpose | When to Use |
|--------|-------|---------|--------|---------|-------------|
| `experiments/nvda_tlob_h10_v1.yaml` | TLOB | H=10 | TLOB | Short-term (~1s) | Fast iteration, high accuracy |
| `experiments/nvda_tlob_h100_v1.yaml` | TLOB | H=100 | TLOB | Paper benchmark (~10s) | Compare with DeepLOB paper |
| `experiments/nvda_tlob_triple_barrier_11mo_v1.yaml` | TLOB | H=50 | Triple Barrier | Risk-managed trading | Backtesting, win rate focus |
| `experiments/e5_60s_huber_cvml.yaml` | TLOB | H=10 | Regression | Canonical Phase 3 multi-base example | Reference for new axis-composed experiments |

---

## Archived Reference Configs (6)

These use legacy datasets but are **valuable references** for specific configurations.
To use them, update `data_dir` to a current dataset first.

| Config | Model | Unique Value | Learn From This For |
|--------|-------|--------------|---------------------|
| `archive/baseline_lstm.yaml` | **LSTM** | Pure LSTM model | How to configure LSTM without attention |
| `archive/lstm_attn_bidir_h20.yaml` | **LSTM+Attn+Bidir** | Enhanced LSTM | Attention mechanism, bidirectional processing |
| `archive/deeplob_benchmark.yaml` | **DeepLOB** | Paper architecture | DeepLOB model (Zhang et al. 2019) configuration |
| `archive/nvda_bigmove_opportunity_v1.yaml` | DeepLOB | **Opportunity labeling** | How to configure Opportunity labels |
| `archive/nvda_tlob_bigmove_v1.yaml` | TLOB | **TLOB + Opportunity** | Combining TLOB with Opportunity labels |
| `archive/nvda_tlob_binary_signal_v1.yaml` | TLOB | **Binary + Focal Loss** | 2-class task, focal loss configuration |

### Using Archived Configs

```yaml
# To use an archived config with current data, update:
data:
  data_dir: "../data/exports/nvda_11month_complete"  # or nvda_11month_triple_barrier
  labeling_strategy: tlob  # or triple_barrier (must match dataset)
```

---

## Horizon Index Mapping

### TLOB Labels (`nvda_11month_complete`)

| `horizon_idx` | Horizon | Time Ahead | Difficulty | Recommended For |
|---------------|---------|------------|------------|-----------------|
| 0 | H=10 | ~1 second | Easiest | Fast iteration, debugging |
| 1 | H=20 | ~2 seconds | Easy | Quick experiments |
| 2 | H=50 | ~5 seconds | Medium | Balanced accuracy/difficulty |
| 3 | H=100 | ~10 seconds | Hard | **Paper benchmark comparison** |

### Triple Barrier Labels (`nvda_11month_triple_barrier`)

| `horizon_idx` | Max Holding | Time Ahead | Recommended For |
|---------------|-------------|------------|-----------------|
| 0 | 50 ticks | ~5 seconds | Short-term trading |
| 1 | 100 ticks | ~10 seconds | Medium-term trading |
| 2 | 200 ticks | ~20 seconds | Longer holding periods |

---

## Model Types

| `model_type` | Architecture | Features Used | Reference |
|--------------|--------------|---------------|-----------|
| `tlob` | Transformer with dual attention | All 98-148 | Berti & Kasneci (2025) |
| `deeplob` | CNN + Inception + LSTM | First 40 (LOB only) or all | Zhang et al. (2019) |
| `lstm` | Stacked LSTM | All 98-148 | Hochreiter & Schmidhuber (1997) |
| `gru` | Stacked GRU | All 98-148 | Cho et al. (2014) |
| `hmhp` | Hierarchical Multi-Horizon | All 98-148, multi-horizon classification | Custom |
| `hmhp_regression` | HMHP Regressor | All 98-148, multi-horizon regression | Custom |
| `logistic` | Logistic Regression | Single snapshot | sklearn |
| `temporal_ridge` | Ridge + temporal features | 53 temporal features | sklearn |
| `temporal_gradboost` | GradientBoosting + temporal features | 53 temporal features | sklearn |

---

## Labeling Strategies

| `labeling_strategy` | Classes | Class Meanings | Best For |
|---------------------|---------|----------------|----------|
| `tlob` | 3 | 0=Down, 1=Stable, 2=Up | Trend prediction (classification) |
| `triple_barrier` | 3 | 0=StopLoss, 1=Timeout, 2=ProfitTarget | Trading decisions |
| `opportunity` | 3 | 0=BigDown, 1=NoOpportunity, 2=BigUp | Big move detection |
| `regression` | continuous | Forward returns in bps (float64) | Regression (point-return, smoothed) |

---

## Loss Functions

| `loss_type` | When to Use | Class Imbalance Handling |
|-------------|-------------|--------------------------|
| `cross_entropy` | Balanced classes | None |
| `weighted_ce` | Imbalanced classes | Inverse frequency weights |
| `focal` | Severely imbalanced | Down-weights easy examples |
| `mse` | Regression | N/A |
| `huber` | Regression (robust to outliers) | N/A (delta parameter) |
| `heteroscedastic` | Regression with uncertainty | N/A (learns variance) |

```yaml
# Focal loss example
train:
  loss_type: focal
  focal_gamma: 2.0      # Higher = more focus on hard examples
  focal_alpha: null     # Auto-compute from class distribution
  use_class_weights: false  # Focal handles imbalance internally
```

---

## Configuration Template

Use this as a starting point for new experiments:

```yaml
# =============================================================================
# Experiment: [NAME]
# =============================================================================
name: TLOB_NVDA_H{HORIZON}_v1
description: "TLOB model with H={HORIZON} on 11-month dataset"

tags:
  - nvda
  - tlob
  - horizon-{HORIZON}

# =============================================================================
# Data Configuration
# =============================================================================
data:
  # Dataset path (use current 234-day dataset)
  data_dir: "../data/exports/nvda_11month_complete"
  feature_count: 98
  
  # Labeling strategy (must match dataset)
  labeling_strategy: tlob  # Options: tlob, triple_barrier, opportunity, regression
  num_classes: 3
  
  # Horizon selection (see mapping above)
  horizon_idx: 0  # 0=H10, 1=H20, 2=H50, 3=H100
  
  # Sequence configuration (matches Rust export)
  sequence:
    window_size: 100
    stride: 10
  
  # Normalization
  normalization:
    strategy: zscore_per_day
    eps: 1.0e-8
    clip_value: 10.0
    exclude_features: [93]  # TIME_REGIME is categorical

# =============================================================================
# Model Configuration
# =============================================================================
model:
  model_type: tlob  # Options: tlob, deeplob, lstm, gru, hmhp, hmhp_regression, temporal_ridge, temporal_gradboost, logistic
  input_size: 98
  num_classes: 3
  dropout: 0.1
  
  # TLOB-specific (if model_type: tlob)
  tlob_hidden_dim: 64
  tlob_num_layers: 4
  tlob_num_heads: 1
  tlob_mlp_expansion: 4.0
  tlob_use_sinusoidal_pe: true
  tlob_use_bin: true
  tlob_dataset_type: nvda

# =============================================================================
# Training Configuration
# =============================================================================
train:
  batch_size: 64
  learning_rate: 1.0e-4
  weight_decay: 1.0e-5
  epochs: 50
  early_stopping_patience: 10
  gradient_clip_norm: 1.0
  scheduler: cosine
  num_workers: 4
  pin_memory: true
  seed: 42
  
  # Loss configuration
  loss_type: weighted_ce  # Options: cross_entropy, weighted_ce, focal, mse, huber, heteroscedastic
  use_class_weights: true
  task_type: multiclass   # Options: multiclass, binary_signal, regression

# =============================================================================
# Output
# =============================================================================
output_dir: outputs/experiments/{EXPERIMENT_NAME}
log_level: INFO
```

---

## Creating New Experiments

### Step 1: Copy from existing config

```bash
cp configs/experiments/nvda_tlob_h10_v1.yaml configs/experiments/nvda_tlob_h20_v1.yaml
```

### Step 2: Update key parameters

```yaml
# Required changes:
name: TLOB_NVDA_H20_v1              # Unique name
data:
  horizon_idx: 1                    # H=20
output_dir: outputs/experiments/nvda_tlob_h20_v1
```

### Step 3: Run experiment

```bash
python scripts/train.py --config configs/experiments/nvda_tlob_h20_v1.yaml
```

---

## Experiment Decision Guide

### Which config to use?

| Goal | Config | Why |
|------|--------|-----|
| Quick debugging | `nvda_tlob_h10_v1.yaml` | Fastest training, highest accuracy |
| Paper comparison | `nvda_tlob_h100_v1.yaml` | Matches DeepLOB paper's H=100 setting |
| Trading backtest | `nvda_tlob_triple_barrier_11mo_v1.yaml` | Win rate metrics, risk management |
| Try LSTM | Copy `archive/baseline_lstm.yaml` | Pure LSTM reference |
| Try DeepLOB | Copy `archive/deeplob_benchmark.yaml` | Paper architecture |
| Binary detection | Copy `archive/nvda_tlob_binary_signal_v1.yaml` | Signal vs NoSignal |

### Which loss to use?

| Class Distribution | Recommended Loss |
|--------------------|------------------|
| Balanced (~33% each) | `cross_entropy` |
| Imbalanced (e.g., 70% Stable) | `weighted_ce` or `focal` |
| Severely imbalanced | `focal` with `gamma=2.0` |

---

## Dataset Schema (v2.2)

All datasets follow this contract:
- **Features**: 98 per timestep
  - 0-39: Raw LOB (10 levels × 4 values)
  - 40-47: Derived (spread, microprice, etc.)
  - 48-83: MBO (order flow, queue stats)
  - 84-97: Trading signals (OFI, asymmetry, etc.)
- **Sequences**: `[N_seq, 100, 98]`
- **Labels**: `[N_seq]` or `[N_seq, n_horizons]`
- **Normalization**: Market-structure preserving z-score

See `plan/03-FEATURE-INDEX-MAP-v2.md` for complete feature index mapping.

---

## Related Documentation

- **`configs/bases/README.md`** — Full 4-axis ownership matrix, per-base inventory, chained-inheritance pattern docs.
- **`../MERGE_MIGRATION_PLAN.md`** — Phase 3 migration ledger: v1→v2 merge.py retirement, monolith decomposition, per-batch migration status (25/42 in-scope).
- **`../CHANGELOG.md`** — Per-release change log (current 0.4.0, 2026-04-15 — Phase 3 config composition).
- **`../src/lobtrainer/config/archive/merge-v1/ARCHIVE_README.md`** — Archived v1 `merge.py` (single-string `_base:` only). Loaded via `importlib` from parity tests only; mirrors `feature-extractor-MBO-LOB/archive/monolith-v1/` precedent.
- **`../CODEBASE.md` §5 (Configuration System)** — Technical reference for `ExperimentConfig` / `DataConfig` / `ModelConfig` / `TrainConfig` dataclasses + `_base:` resolution internals.
