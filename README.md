# LOB Model Trainer

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-orange.svg)](https://pytorch.org/)

Machine learning experimentation for limit order book price prediction — the training plane of a multi-repo HFT pipeline.

**Version**: 0.7.1 | **Schema**: 3.0 (Phase G G.6.A bump per CLAUDE.md root rule: any modification to stable features 0-97 = MAJOR) | **Tests**: 1432 (1367 passed + 65 skipped)

---

## Overview

This package trains neural-network and baseline models on LOB feature data exported from the `feature-extractor-MBO-LOB` Rust pipeline. It is a **training-focused library** — dataset analysis lives in `lob-dataset-analyzer`, backtesting in `lob-backtester`, orchestration in `hft-ops`.

### Key Features

- **Multiple architectures** — LSTM, GRU, DeepLOB, TLOB, HMHP, HMHP-R, LogisticLOB, XGBoostLOB, TemporalRidge, TemporalGradBoost (via `lob-models`)
- **Dual task support** — classification (multiclass, binary) + regression (continuous bps returns)
- **Strategy Pattern** — 4 concrete strategies (Classification, Regression, HMHPClassification, HMHPRegression); see `src/lobtrainer/training/strategies/`
- **Phase 3 multi-base config composition** — `_base:` YAML inheritance with axis-ownership enforcement
- **Phase 4 FeatureSet registry consumer** — `DataConfig.feature_set: nvda_short_term_40_src128_v1` resolves content-addressed feature subsets from `contracts/feature_sets/`
- **Phase 2b T10 sample weights** — concurrent-label-overlap weighting for regression (de Prado AFML 4.5.1)
- **Phase 2b T11 CV trainer** — purged k-fold with embargo days
- **Phase 2b T14 experiment gates** — mandatory pre-training IC gate (rule §13)
- **Phase 7 Stage 7.4 Round 4** — `test_metrics.json` persistence for PyTorch Trainer (feeds `hft-ops` PostTrainingGateRunner)
- **Advanced monitoring** — gradient tracking, learning rate monitoring, training diagnostics
- **Multiple losses** — CrossEntropy, Focal, WeightedCE, MSE, Huber, Heteroscedastic, GMADL

---

## Quick Start

**Preferred** — via `hft-ops` orchestrator (validates cross-module consistency, runs IC gate, records in ledger):

```bash
cd hft-ops
hft-ops run experiments/e5_60s_huber_cvml_unified.yaml
```

**Direct invocation** (development-only; emits deprecation warning unless `HFT_OPS_ORCHESTRATED=1`):

```bash
cd lob-model-trainer
uv venv && uv pip install -e ".[dev]"
# or: python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"

python scripts/train.py --config configs/experiments/e5_60s_huber_cvml_unified.yaml
python scripts/export_signals.py --checkpoint outputs/.../best.pt --split test
```

---

## Data Contract (Schema v3.0)

| Property | Value |
|----------|-------|
| Features (MBO full) | 128 per timestep (stable 0-97 + experimental 98-127) |
| Features (MBO 98-subset) | 98 per timestep (stable features only) |
| Features (BASIC off-exchange) | 34 per timestep |
| Sequence layout | `(N_seq, T, F)` (T=100 default MBO, T=20 BASIC) |
| Labels (TLOB, classification) | `{-1, 0, 1}` → `{0, 1, 2}` shifted for PyTorch |
| Labels (Triple Barrier, classification) | `{0, 1, 2}` = StopLoss / Timeout / ProfitTarget |
| Labels (Regression) | `(N_seq, H)` float64 basis points (via `LabelFactory`) |
| Multi-horizon | `(N_seq, H)` for HMHP |

### Feature Categories (MBO 128-feature layout)

| Range | Count | Category |
|-------|-------|----------|
| 0-39 | 40 | Raw LOB (10 levels × 4 values) |
| 40-47 | 8 | Derived (spread, microprice, volume imbalance, ...) |
| 48-83 | 36 | MBO (order flow, queue, size distribution, signals) |
| 84-97 | 14 | Trading Signals (OFI, asymmetry, fragility, regime) |
| 98-147 | 50 | Experimental (institutional v2, volatility, seasonality, MLOFI, Kolm OF) |

---

## Directory Structure

```
lob-model-trainer/
├── src/lobtrainer/
│   ├── __init__.py                  # Public API (v0.4.0)
│   ├── config/                      # Configuration schema + composition
│   │   ├── schema.py                # ExperimentConfig, DataConfig (3-field mutex:
│   │   │                            #   feature_set / feature_preset / feature_indices),
│   │   │                            #   ModelConfig, CVConfig, LabelsConfig
│   │   └── merge.py                 # Phase 3 multi-base `_base:` composition +
│   │                                # is_partial_base guard (6A.5 M6 dict-guard)
│   ├── constants/                   # Feature indices + presets (via hft-contracts SSoT)
│   │   ├── feature_index.py         # Re-exports FeatureIndex, SignalIndex from hft_contracts
│   │   └── feature_presets.py       # Named feature subsets; DEPRECATED 2026-04-15,
│   │                                # ImportError 2026-08-15 (migrate to FeatureSet registry)
│   ├── data/                        # Data loading + feature-set resolution
│   │   ├── dataset.py               # DayData, LOBSequenceDataset, LOBFlatDataset
│   │   ├── sources.py               # T12 multi-source data abstraction
│   │   ├── bundle.py                # T12 multi-source bundling (MBO + BASIC)
│   │   ├── feature_set_resolver.py  # Phase 4 4c.1 FeatureSet registry consumer;
│   │   │                            # Phase 6 6B.2 delegates canonical_hash to hft_contracts SSoT
│   │   ├── sample_weights.py        # T10 concurrent-label-overlap weights (de Prado AFML 4.5.1)
│   │   ├── normalization.py         # T15 Python-side normalization (hybrid, global z-score)
│   │   └── transforms.py            # FeatureStatistics, BinaryLabelTransform
│   ├── models/                      # Model implementations
│   │   └── baselines.py             # NaiveClassPrior, NaivePreviousLabel, LogisticBaseline
│   │                                # (DeepLOB, TLOB, HMHP, HMHP-R live in lob-models)
│   ├── training/                    # Training infrastructure
│   │   ├── trainer.py               # Trainer class (PyTorch epoch loop)
│   │   ├── simple_trainer.py        # SimpleModelTrainer (TemporalRidge, GradBoost)
│   │   ├── cv_trainer.py            # T11 CVTrainer (purged k-fold + embargo)
│   │   ├── strategy.py              # Strategy ABC + create_strategy dispatch
│   │   ├── strategies/              # 4 concrete strategies (Phase 2 refactor)
│   │   │   ├── classification.py
│   │   │   ├── regression.py
│   │   │   ├── hmhp_classification.py
│   │   │   └── hmhp_regression.py
│   │   ├── regression_metrics.py    # Thin adapter over hft_metrics.regression (SSoT)
│   │   ├── regression_evaluation.py # RegressionMetrics dataclass
│   │   ├── metrics.py               # MetricsCalculator, ClassificationMetrics
│   │   ├── callbacks.py             # EarlyStopping, ModelCheckpoint, MetricLogger, ProgressCallback
│   │   ├── loss.py                  # FocalLoss, BinaryFocalLoss, Huber, GMADL, Pinball
│   │   ├── evaluation.py            # BaselineReport, evaluate_model
│   │   └── monitoring.py            # GradientMonitor, LearningRateTracker, TrainingDiagnostics
│   ├── export/                      # Signal export for backtester
│   │   ├── exporter.py              # SignalExporter (predictions, calibration, feature_set_ref)
│   │   └── metadata.py              # signal_metadata.json writer (Phase 4 4c.4 feature_set_ref)
│   ├── experiments/                 # Experiment tracking (legacy; migrates to hft-ops ledger)
│   │   ├── experiment_spec.py       # T14 ExperimentSpec (config + gates)
│   │   ├── gates.py                 # T14 signal-quality gate library
│   │   ├── result.py                # ExperimentResult, ExperimentMetrics
│   │   └── registry.py              # ExperimentRegistry (Phase 7 Stage 7.3 retires to hft-ops)
│   ├── calibration/                 # Signal calibration (variance-match, etc.)
│   ├── cli.py                       # Legacy CLI (use hft-ops or scripts/ instead)
│   ├── _hft_ops_compat.py           # HFT_OPS_ORCHESTRATED=1 env marker check
│   └── utils/                       # Utilities
│       └── reproducibility.py       # set_seed, SeedManager
├── scripts/                         # PRODUCTION INFRA (hft-rules §4)
│   ├── train.py                     # Training CLI (writes test_metrics.json, Round 4)
│   ├── export_signals.py            # Unified signal export CLI
│   ├── precompute_norm_stats.py     # T15 normalization stats pre-computation
│   ├── _hft_ops_compat.py           # Deprecation banner if not orchestrated
│   └── archive/                     # Phase 6 6D fossil archive (NOT templates)
│       ├── README.md                # Migration map per fossil
│       ├── e4_baselines.py
│       ├── e5_baselines.py
│       ├── run_simple_model_ablation.py
│       ├── run_simple_training.py
│       └── run_experiment_spec.py
├── configs/
│   ├── bases/                       # 21 axis-partitioned base configs (Phase 3)
│   │   ├── README.md                # Axis-ownership rules (datasets / models / labels / train)
│   │   ├── models/*.yaml
│   │   ├── datasets/*.yaml
│   │   ├── labels/*.yaml
│   │   └── train/*.yaml
│   ├── experiments/                 # 40 experiment configs (25 multi-base + 15 standalone)
│   └── archive/                     # Legacy reference configs
├── tests/                           # 1149 tests collected (1084 passed + 65 skipped)
├── EXPERIMENT_INDEX.md              # Living experiment ledger
├── CODEBASE.md                      # Detailed module reference
└── pyproject.toml
```

---

## Training a Model

### Via hft-ops (preferred)

```bash
cd hft-ops
hft-ops run experiments/e5_60s_huber_cvml_unified.yaml
# → runs full chain: validation → training → post_training_gate → signal_export → backtesting
```

### Via Python API (programmatic)

```python
from lobtrainer import create_trainer, set_seed
from lobtrainer.training import EarlyStopping, ModelCheckpoint

set_seed(42)

trainer = create_trainer(
    "configs/experiments/e5_60s_huber_cvml_unified.yaml",
    callbacks=[
        EarlyStopping(patience=10),
        ModelCheckpoint(save_dir="outputs/e5/checkpoints/"),
    ],
)

result = trainer.train()
print(f"Best val loss: {result['best_val_metric']:.4f} at epoch {result['best_epoch']}")

metrics = trainer.evaluate("test")
print(metrics.summary() if hasattr(metrics, 'summary') else metrics)
```

### Cross-validation (Phase 2b T11)

```python
from lobtrainer.training import CVTrainer
from lobtrainer.config import CVConfig

cv_trainer = CVTrainer(
    base_config=config,
    cv_config=CVConfig(n_splits=5, embargo_days=1),
)
cv_results = cv_trainer.run()
```

---

## Configuration

Every experiment is a YAML config. Phase 3 introduced multi-base composition:

```yaml
# configs/experiments/my_experiment.yaml
_base:
  - models/tlob_compact_regression.yaml
  - datasets/nvda_e5_60s.yaml
  - labels/regression_huber.yaml
  - train/regression_default.yaml

name: my_experiment
description: |
  Override any field declared in the base configs below.

data:
  data_dir: "../data/exports/nvda_e5_60s"
  feature_set: nvda_short_term_40_src128_v1   # Phase 7.1 FeatureSet registry entry

train:
  epochs: 50
  learning_rate: 1.0e-4
```

See `configs/bases/README.md` for axis-ownership rules and `configs/README_configs.md` for the complete config reference.

### Feature Selection — Three Mutually-Exclusive Fields

```yaml
# Option 1 (RECOMMENDED, Phase 7.1) — FeatureSet registry
data:
  feature_set: nvda_short_term_40_src128_v1

# Option 2 (inline override)
data:
  feature_indices: [0, 5, 12, 40, 84, 85, 88]

# Option 3 (DEPRECATED 2026-04-15, ImportError 2026-08-15)
data:
  feature_preset: short_term_40
```

`DataConfig.__post_init__` raises `ValueError` if >1 is set.

---

## Key Modules

### `lobtrainer.constants`

```python
from lobtrainer.constants import (
    FeatureIndex, SignalIndex, FEATURE_COUNT, SCHEMA_VERSION,
    SHIFTED_LABEL_NAMES,
    get_feature_preset, list_presets,  # DEPRECATED — migrate to FeatureSet registry
)

assert FEATURE_COUNT == 98        # 98 stable features
assert SCHEMA_VERSION == "3.0"  # Phase G G.6.A bump 2.2 → 3.0 (string, not float — matches _generated.py)
assert FeatureIndex.TRUE_OFI == 84
```

### `lobtrainer.data.feature_set_resolver`

```python
from lobtrainer.data.feature_set_resolver import resolve_feature_set

resolved = resolve_feature_set(
    name="nvda_short_term_40_src128_v1",
    registry_dir="../contracts/feature_sets/",
    expected_contract_version="3.0",
    expected_source_feature_count=128,
)
# resolved.feature_indices → list of int
# resolved.content_hash → SHA-256 hex (verified via hft_contracts SSoT)
```

### `lobtrainer.training`

```python
from lobtrainer.training import (
    Trainer, create_trainer, CVTrainer,
    EarlyStopping, ModelCheckpoint, MetricLogger, ProgressCallback,
    MetricsCalculator, ClassificationMetrics,
    compute_classification_report, compute_trading_metrics,
    evaluate_model, create_baseline_report, BaselineReport,
    FocalLoss, BinaryFocalLoss, create_focal_loss,
    GradientMonitor, LearningRateTracker, TrainingDiagnostics,
    create_standard_monitoring,
)
```

### `lobtrainer.export`

```python
from lobtrainer.export import SignalExporter

exporter = SignalExporter(trainer=trainer, calibration=None)
out_dir = exporter.export(split="test", output_dir="outputs/.../signals/test/")
# Writes: predicted_returns.npy, regression_labels.npy, prices.npy, spreads.npy,
#         signal_metadata.json (includes feature_set_ref from Phase 4 4c.4)
```

---

## Critical Notes

### Label Encoding

Labels are shifted for PyTorch `CrossEntropyLoss`:

| Original | Shifted | Meaning |
|----------|---------|---------|
| -1 | 0 | Down |
| 0 | 1 | Stable |
| +1 | 2 | Up |

Shift happens in `LOBSequenceDataset.__getitem__()`.

### Sign Conventions (Schema v3.0)

All directional features:
- `> 0` = BULLISH (buy pressure)
- `< 0` = BEARISH (sell pressure)

Exception: `PRICE_IMPACT` (index 47) is unsigned.

### Safety Gates

Always check before using signals:

```python
from lobtrainer.constants import FeatureIndex

if features[i, FeatureIndex.BOOK_VALID] < 0.5:
    continue  # skip invalid sample
if features[i, FeatureIndex.MBO_READY] < 0.5:
    continue  # MBO features not ready (100-event warmup)
```

### `test_metrics.json` Persistence (Phase 7 Stage 7.4 Round 4)

`scripts/train.py` now writes `output_dir/test_metrics.json` after `trainer.evaluate("test")`. Flat `{test_<metric>: float}` dict consumed by `hft-ops` PostTrainingGateRunner for prior-best regression comparison.

---

## Running Tests

```bash
cd lob-model-trainer
pytest tests/ -v

# Expected: 1149 collected (1084 passed + 65 skipped)
# Skips: 15 real-data integration tests + 14 v1-archive parity tests (retired) + misc
```

---

## Scripts (Production Infra)

| Script | Purpose |
|--------|---------|
| `scripts/train.py` | Training CLI entry point (writes `test_metrics.json`) |
| `scripts/export_signals.py` | Unified signal export CLI |
| `scripts/precompute_norm_stats.py` | T15 normalization stats precomputation |

**Archived fossils** (Phase 6 6D, NOT templates — see `scripts/archive/README.md`):
`e4_baselines.py`, `e5_baselines.py`, `run_simple_training.py`, `run_simple_model_ablation.py`, `run_experiment_spec.py`. Replacements are hft-ops manifests + library modules per hft-rules §4.

---

## Related Libraries

| Library | Purpose |
|---------|---------|
| `hft-ops` | Experiment orchestrator (preferred entry point) |
| `hft-contracts` | Contract constants, LabelFactory, canonical_hash, validation |
| `hft-metrics` | Statistical primitives (IC, dCor, MI, bootstrap, ACF, sample_weights, purged_cv) |
| `hft-feature-evaluator` | 5-path feature evaluation → 4-tier classification |
| `lob-models` | Neural network architectures (DeepLOB, TLOB, HMHP, HMHP-R) |
| `lob-dataset-analyzer` | Dataset analysis and statistics (47 analyzers) |
| `lob-backtester` | Backtesting + P&L (IBKR-calibrated 0DTE costs) |
| `feature-extractor-MBO-LOB` | Rust pipeline for feature extraction (multi-crate workspace) |
| `basic-quote-processor` | Rust pipeline for off-exchange features (XNAS.BASIC) |

---

## Documentation

- `CODEBASE.md` — detailed module reference
- `EXPERIMENT_INDEX.md` — living experiment ledger
- `configs/bases/README.md` — Phase 3 axis-ownership rules
- `configs/README_configs.md` — config reference
- `scripts/archive/README.md` — fossil migration map
- Pipeline-wide ground truth (monorepo root): `CLAUDE.md`, `PIPELINE_ARCHITECTURE.md`, `DOCUMENTATION_INDEX.md`, `PHASE7_ROADMAP.md`

---

## Version History

| Version | Schema | Changes |
|---------|--------|---------|
| **0.7.1** | 3.0 | REV 3.1 Phase G G.6.A bump 2.2 → 3.0 (MAJOR per CLAUDE.md root rule: any modification to stable features 0-97 = BREAKING). Phase G.1 dropped in-NPY schema_version emission (RESERVED 0.0 at idx 97); JSON metadata is canonical SSoT. Phase G.6.D regenerated 3 production FeatureSet content_hashes + trainer golden hash rotation. Phase G.6.F + G.7 fixture cascade (analyzer 358 / hft-contracts 518 / trainer 1367 / hft-ops 633 / Rust 800 = 3,676 tests passing post-cascade). +2 xfailed for legacy-corpus xfail markers per Phase G+1 deferral. |
| **0.7.0** | 2.2 | Phase A.5 Scope D (Pydantic v2 migration). All 9 config classes migrated from dataclass+dacite to Pydantic v2 SafeBaseModel. 4 bug classes retired at TYPE layer. |
| **0.4.0** | 2.2 | Strategy Pattern refactoring (4 concrete strategies), Model Registry integration, Phase 3 multi-base `_base:` composition (21 axis-partitioned bases, monolith retired 2026-04-15), Phase 4 Batch 4c FeatureSet registry consumer (`DataConfig.feature_set` + `feature_set_resolver.py` + canonical_hash parity lock via hft_contracts SSoT), Phase 4 4c.4 `signal_metadata.json` feature_set_ref propagation, Phase 6 6B.2 trainer inline `_compute_content_hash` retired (delegates to hft_contracts), Phase 6 6D 5 experimental fossils archived, Phase 7 Stage 7.1 5 config migrations from `feature_preset:` → `feature_set:` (+ 14 parity tests), Phase 7 Stage 7.4 Round 4 `scripts/train.py::_dump_test_metrics` (`test_metrics.json` persistence for PyTorch Trainer), 1149 tests |
| 0.3.0 | 2.1 | Strategy-aware metrics, Focal Loss, TLOB support |
| 0.2.0 | 2.1 | Training infrastructure, multi-horizon support |
| 0.1.0 | 2.0 | Initial release |

---

*Last updated: 2026-04-27 (REV 3.1 Phase G G.6.A→G.6.F — SchemaVersion 2.2 → 3.0 MAJOR bump)*
