# LOB-Model-Trainer: Codebase Technical Reference

> **Version**: 0.4.0  
> **Schema**: 2.2 (via `hft-contracts` package)  
> **Tests**: 1183 collected (1118 passed + 65 skipped) — Phase 8C-α Integration Close-Out adds trainer wire-in: `PermutationImportanceCallback` (16 new callback unit tests) + `ImportanceConfig` field on `ExperimentConfig` via `_coerce_importance` (breaks schema.py ↔ training.importance.config circular) + `configs/bases/train/importance_default.yaml` partial base fragment  
> **Last Updated**: 2026-04-20 (Phase 8C-α Integration Close-Out — Q7 trainer wire-in COMPLETE: `PermutationImportanceCallback` at `training/importance/callback.py` auto-registered by `train_from_config` when `config.importance is not None`; `make_pytorch_predict_fn` / `make_metric_fn_for_task` / `_extract_eval_tensors` factories; graceful-failure (observation-tier errors log+swallow, do NOT kill training); preserves Stage C.1 + 2-round post-audit: `compute_permutation_importance` pure function + `ImportanceConfig` + `block_length_samples` rename + RNG decorrelation + NaN-baseline fail-loud + failed-seed drop + degenerate-block guard)  
> **Purpose**: Complete technical reference for LLMs and developers to understand, modify, and extend the codebase.
>
> **Scope**: This library focuses solely on **model training**. For dataset analysis, use `lob-dataset-analyzer`.
>
> **New in 0.4.0 (cumulative through Phase 7 Stage 7.4 Round 4)**:
> - Phase 2 Strategy Pattern refactoring — Trainer decomposed from 1,657L; 4 concrete strategies (Classification, Regression, HMHPClassification, HMHPRegression) under `src/lobtrainer/training/strategies/`. Model Registry integration via lob-models.
> - Phase 2b — `CVTrainer` (purged k-fold + embargo, T11), `sample_weights` (T10 de Prado AFML 4.5.1), data sources abstraction + bundle (T12 multi-source), experiment_spec + gates (T14 pre-training IC gate).
> - Phase 3 — multi-base config composition via `_base:` YAML inheritance (21 axis-partitioned bases, monolith retired 2026-04-15); 6A.5 M6 `yaml.safe_load` dict-guard; 6A.7 `data.feature_set` + `data.feature_sets_dir` axis ownership.
> - Phase 4 Batch 4c — FeatureSet registry consumer: `DataConfig.feature_set` field (3-field mutual exclusion with `feature_preset` + `feature_indices`), `feature_set_resolver.py` walks up to `contracts/feature_sets/`, verifies `content_hash` via hft_contracts canonical_hash SSoT. Batch 4c.4: `signal_metadata.json::feature_set_ref` propagation to backtester.
> - Phase 6 6B.2 — trainer inline `_compute_content_hash` retired; delegates to `hft_contracts.canonical_hash`. Golden-fixture drift detector at `tests/test_feature_set_resolver.py::TestCanonicalHashGolden`.
> - Phase 6 6D — 5 experimental fossils archived under `scripts/archive/` with fossil headers + migration map per hft-rules §4.
> - Phase 7 Stage 7.1 — 5 trainer config migrations from deprecated `feature_preset:` to Phase 4 `feature_set:` (+ 14 parity tests locking hand-curated JSON indices against trainer preset source). Closes 2026-08-15 `feature_preset` ImportError deadline.
> - Phase 7 Stage 7.4 Round 4 — `scripts/train.py::_dump_test_metrics` + `_safe_summary`: writes `output_dir/test_metrics.json` after `trainer.evaluate("test")`. Consumed by `hft-ops` PostTrainingGateRunner for prior-best regression comparison. Closes the silent-dead Round 1 `test_*` whitelist for every PyTorch TLOB/HMHP run.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Module Architecture](#2-module-architecture)
3. [Core Data Flow](#3-core-data-flow)
4. [Constants and Feature Indices](#4-constants-and-feature-indices)
5. [Configuration System](#5-configuration-system)
6. [Data Loading Pipeline](#6-data-loading-pipeline)
7. [PyTorch Dataset Classes](#7-pytorch-dataset-classes)
8. [Data Transforms](#8-data-transforms)
9. [Model Implementations](#9-model-implementations)
10. [Training Infrastructure](#10-training-infrastructure)
11. [Callback System](#11-callback-system)
12. [Metrics and Evaluation](#12-metrics-and-evaluation)
13. [Baseline Models](#13-baseline-models)
14. [Monitoring and Diagnostics](#14-monitoring-and-diagnostics)
15. [Experiment Tracking](#15-experiment-tracking)
16. [Reproducibility Utilities](#16-reproducibility-utilities)
17. [Scripts and CLI](#17-scripts-and-cli)
18. [Configuration Reference](#18-configuration-reference)
19. [Testing Patterns](#19-testing-patterns)
20. [Known Limitations](#20-known-limitations)

---

## 1. Project Overview

### Purpose

Python library for training and evaluating ML models on LOB (Limit Order Book) data for price movement prediction. Designed for HFT research with emphasis on:

- **Modularity**: Clean separation between data, models, and training
- **Reproducibility**: Explicit seed management and configuration-driven experiments
- **Flexibility**: Multi-horizon labels, multiple model architectures
- **Monitoring**: Gradient tracking, training diagnostics, experiment comparison

### Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| **Strategy Pattern** | ✅ Complete | 4 strategies: Classification, Regression, HMHPClassification, HMHPRegression |
| **Trainer Orchestrator** | ✅ Complete | 900L (was 1,657L), zero task-branching, delegates to strategy |
| **All Models via lob-models** | ✅ Complete | 10 registered models (TLOB, DeepLOB, MLPLOB, HMHP, HMHP-R, LSTM, GRU, LogisticLOB, Ridge, GradBoost) |
| **Strategy-Aware Metrics** | ✅ Complete | MetricsCalculator for TLOB/Triple Barrier/Opportunity |
| **Focal Loss** | ✅ Complete | For class imbalance handling |
| **Multi-Horizon Labels** | ✅ Complete | Support for multiple prediction horizons |
| **HMHP Regression** | ✅ Complete | Regression training with precomputed labels |
| **Experiment Tracking** | ✅ Complete | ExperimentRegistry, comparison tables |
| **Monitoring Callbacks** | ✅ Complete | Gradient, LR, diagnostics tracking (uses public properties) |
| **Tests** | ✅ Complete | 1149 collected (1084 passed + 65 skipped) — matches banner at line 5 |

### Core Dependencies

```toml
torch = ">=2.0"           # Deep learning framework
numpy = ">=1.24"          # Numerical operations
pandas = ">=2.0"          # Data manipulation
scikit-learn = ">=1.3"    # Classical ML, metrics
scipy = ">=1.10"          # Statistical tests
pyyaml = ">=6.0"          # Configuration files
dacite = ">=1.8"          # Dataclass from dict
tqdm = ">=4.65"           # Progress bars
```

---

## 2. Module Architecture

```
src/lobtrainer/
├── __init__.py                    # Public API exports (v0.4.0)
│
├── constants/
│   ├── __init__.py                # Module exports
│   ├── feature_index.py           # FeatureIndex, SignalIndex (98 features)
│   └── feature_presets.py         # Named feature subsets (8 presets)
│
├── config/
│   ├── __init__.py                # Module exports
│   ├── schema.py                  # ExperimentConfig, DataConfig, ModelConfig, TrainConfig
│   ├── merge.py                   # deep_merge(), resolve_inheritance(), is_partial_base()
│   │                              #   Supports _base: str | list[str] multi-base composition (v2, Phase 3).
│   │                              #   Preserves all v1 invariants byte-identically.
│   └── archive/merge-v1/          # Archived v1 merge.py (single-string _base: only, 127 LOC)
│       ├── merge.py               #   Loaded via importlib from tests/test_merge_v1_parity.py
│       └── ARCHIVE_README.md      #   Parity reference; mirrors feature-extractor archive/monolith-v1/
│
├── data/
│   ├── __init__.py                # Module exports
│   ├── dataset.py                 # DayData, LOBFlatDataset, LOBSequenceDataset
│   ├── feature_selector.py        # FeatureSelector (frozen dataclass, preset/custom indices)
│   ├── feature_set_resolver.py    # Phase 4 Batch 4c: ResolvedFeatureSet + resolve_feature_set()
│   │                              #   Loads <name>.json from contracts/feature_sets/, verifies content
│   │                              #   hash (integrity), contract_version + source_feature_count match.
│   │                              #   Error hierarchy: FeatureSetResolverError → FeatureSetNotFound/
│   │                              #   FeatureSetMalformed/FeatureSetIntegrityError/FeatureSetContractMismatch.
│   │                              #   _compute_content_hash inlined (torch-free; cross-venv independent
│   │                              #   from hft-ops); byte-parity LOCKED against hft_contracts.canonical_hash.
│   ├── transforms.py              # FeatureStatistics, BinaryLabelTransform, ComposeTransform
│   └── normalization.py           # HybridNormalizer, GlobalZScoreNormalizer (Welford/Chan streaming)
│
├── models/
│   ├── __init__.py                # create_model (45L, registry-based), LSTM/GRU re-exports from lobmodels
│   └── baselines.py               # NaiveClassPrior, NaivePreviousLabel, LogisticBaseline
│
├── training/
│   ├── __init__.py                # Module exports (40+ symbols)
│   ├── strategy.py                # TrainingStrategy ABC, BatchResult, create_strategy()
│   ├── strategies/                # Concrete training strategies (Phase 2)
│   │   ├── __init__.py            # Exports 4 strategies
│   │   ├── classification.py      # ClassificationStrategy (criterion + metrics)
│   │   ├── regression.py          # RegressionStrategy (model.compute_loss)
│   │   ├── hmhp_classification.py # HMHPClassificationStrategy (per-horizon)
│   │   └── hmhp_regression.py     # HMHPRegressionStrategy (per-horizon regression)
│   ├── trainer.py                 # Trainer orchestrator (900L), delegates to strategy
│   ├── callbacks.py               # EarlyStopping, ModelCheckpoint, MetricLogger
│   ├── metrics.py                 # MetricsCalculator, ClassificationMetrics
│   ├── regression_metrics.py     # Thin adapter over hft-metrics (R², IC, MAE, RMSE, DA, PA)
│   ├── regression_evaluation.py  # RegressionMetrics dataclass (from_arrays, summary, to_dict)
│   ├── simple_trainer.py          # SimpleModelTrainer for sklearn-style models (TemporalRidge, GradBoost)
│   ├── loss.py                    # FocalLoss, BinaryFocalLoss, create_focal_loss
│   ├── evaluation.py              # BaselineReport, evaluate_model, full_evaluation
│   └── monitoring.py              # GradientMonitor, TrainingDiagnostics, LRTracker
│
├── cli.py                         # CLI entry point: train_command, evaluate_command, apply_overrides
│
├── experiments/
│   ├── __init__.py                # Module exports
│   ├── result.py                  # ExperimentResult, ExperimentMetrics
│   └── registry.py                # ExperimentRegistry, create_comparison_table
│
├── calibration/
│   ├── __init__.py              # Prediction calibration package
│   └── variance.py              # VarianceCalibrator: post-hoc prediction rescaling (18 tests)
│
├── export/
│   ├── __init__.py              # Signal export public API
│   ├── exporter.py              # SignalExporter: unified signal export (replaces 3 scripts)
│   ├── raw_features.py          # RawFeatureExtractor: spread/price from disk via mmap
│   └── metadata.py              # build_signal_metadata: superset metadata builder
│
└── utils/
    ├── __init__.py                # Module exports
    └── reproducibility.py         # set_seed, SeedManager, worker_init_fn

scripts/
├── train.py                       # Training CLI
├── export_signals.py              # Unified signal export CLI (replaces 3 deprecated scripts)
├── run_simple_training.py         # SimpleModelTrainer CLI (TemporalRidge, GradBoost)
├── run_simple_model_ablation.py   # Ablation experiments for simple models
├── e4_baselines.py                # E4 time-based experiment baselines
├── e5_baselines.py                # E5 60s-bin experiment baselines
├── precompute_norm_stats.py       # Pre-compute normalization statistics cache
├── validate_export.py             # Dataset validation
└── analysis/                      # Analysis and evaluation scripts
    ├── evaluate_model.py          # Model evaluation from checkpoint
    └── run_baseline_evaluation.py # Baseline comparison

configs/
├── README_configs.md              # Complete config reference
├── bases/                         # 21 axis-partitioned base configs (Phase 3)
│   ├── README.md                  #   4-axis ownership rule + chained inheritance + _partial:
│   ├── models/                    #   5: tlob_compact_bare, tlob_compact_regression,
│   │                              #      tlob_paper_classification, hmhp_cascade_bare,
│   │                              #      hmhp_cascade_regression
│   ├── datasets/                  #   8: per-export normalisation/sampling bases
│   ├── labels/                    #   4: regression_huber, tlob_smoothed, opportunity,
│   │                              #      triple_barrier_volscaled
│   └── train/                     #   4: regression_default, classification_default,
│                                  #      classification_triple_barrier, tlob_paper_classification_train
├── experiments/                   # 42 in-scope YAML configs: 25 migrated to multi-base _base:,
│                                  #   17 standalone by design (baselines, XGBoost, archive,
│                                  #   niche HMHP, TLOB singletons). See MERGE_MIGRATION_PLAN.md.
└── archive/                       # Legacy reference configs (6) — not in Phase 3 migration scope

tests/                             # 43 test modules, 1052 tests pytest-collected (2026-04-15)
├── conftest.py                    # Shared fixtures (rng, day_data_factory, synthetic_export_dir)
├── test_baselines.py
├── test_calibration.py
├── test_config.py
├── test_create_dataloaders.py     # Feature selection cascade, num_workers override, strategy routing (11 tests)
├── test_deeplob_integration.py
├── test_evaluation.py
├── test_experiments.py
├── test_feature_index.py
├── test_feature_presets.py
├── test_feature_selector.py       # FeatureSelector: presets, validation, selection (25 tests)
├── test_hmhp_collate.py           # _hmhp_collate_fn: 2/3-tuple dict-label collation (7 tests)
├── test_integration.py
├── test_label_shift.py            # Label shift resolution: 4 paths (15 tests)
├── test_loss.py
├── test_monitoring.py
├── test_normalization.py          # GlobalZScore, HybridNormalizer, Welford, streaming (32 tests)
├── test_normalization_integration.py  # Normalizer as dataset transform (5 tests)
├── test_optimizer_scheduler.py    # Optimizer dispatch (adamw/adam/sgd), scheduler dispatch (8 tests)
├── test_regression_dataset.py
├── test_regression_evaluation.py  # RegressionMetrics: from_arrays, to_dict, summary (8 tests)
├── test_regression_metrics.py
├── test_regression_training.py
├── test_signal_export.py          # RawFeatureExtractor, metadata, file contracts (18 tests)
├── test_signal_export_inference.py # SignalExporter 4 inference paths: shapes, dtypes, dispatch (17 tests)
├── test_simple_trainer.py         # SimpleModelTrainer: _load_split, train, export, alignment (15 tests)
├── test_standard_regression_training.py
├── test_strategies.py             # All 4 strategies: process_batch, validate, evaluate, predict (35 tests)
├── test_strategy_metrics.py
├── test_tlob_integration.py
├── test_trainer.py
└── test_transforms.py
```

---

## 3. Core Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       TRAINING PIPELINE DATA FLOW                            │
└─────────────────────────────────────────────────────────────────────────────┘

    Raw NumPy Files           DayData Objects           PyTorch Tensors
         │                          │                          │
         ▼                          ▼                          ▼
┌─────────────────┐     ┌─────────────────────┐     ┌──────────────────┐
│ {date}_sequences│────▶│     DayData         │────▶│ LOBSequenceDataset│
│ {date}_labels   │     │                     │     │                  │
│ (per day)       │     │ .sequences [N,T,F]  │     │ .__getitem__()   │
│                 │     │ .labels [N] or [N,H]│     │  → (seq, label)  │
└─────────────────┘     │ .get_labels(h_idx)  │     └──────────────────┘
                        └─────────────────────┘              │
                                                             ▼
                                                    ┌──────────────────┐
                                                    │   DataLoader     │
                                                    │   (batches)      │
                                                    └──────────────────┘
                                                             │
                                                             ▼
                                                    ┌──────────────────┐
                                                    │     Trainer      │
                                                    │                  │
                                                    │ model(features)  │
                                                    │ → predictions    │
                                                    │                  │
                                                    │ loss.backward()  │
                                                    │ optimizer.step() │
                                                    └──────────────────┘
```

### Label Encoding Flow

```
Original Labels: {-1, 0, 1}    →    Dataset Shift: {0, 1, 2}    →    Metrics Display
    (Down, Stable, Up)                 (for CrossEntropyLoss)        (Down, Stable, Up)
                                  
IMPORTANT: Labels are shifted by +1 in __getitem__ for PyTorch compatibility.
```

---

## 4. Constants and Feature Indices

> **Important**: As of v0.2.0, all feature index constants are sourced from the
> `hft-contracts` package (single source of truth). The `lobtrainer/constants/`
> module is a thin shim that re-exports from `hft_contracts`, preserving backward
> compatibility for all existing `from lobtrainer.constants import ...` imports.
>
> Source of truth: `contracts/pipeline_contract.toml`  
> Regenerate: `python contracts/generate_python_contract.py`

### FeatureIndex (src/lobtrainer/constants/feature_index.py)

The feature vector has exactly **98 features**:

```python
class FeatureIndex:
    """Zero-based indices into the 98-feature vector."""
    
    # Raw LOB (40 features: indices 0-39)
    ASK_PRICE_0 = 0      # Best ask price
    ASK_PRICE_9 = 9      # Level 10 ask price
    ASK_SIZE_0 = 10      # Best ask size
    ASK_SIZE_9 = 19      # Level 10 ask size
    BID_PRICE_0 = 20     # Best bid price
    BID_PRICE_9 = 29     # Level 10 bid price
    BID_SIZE_0 = 30      # Best bid size
    BID_SIZE_9 = 39      # Level 10 bid size
    
    # Derived Features (8 features: indices 40-47)
    MID_PRICE = 40
    SPREAD = 41
    SPREAD_BPS = 42
    TOTAL_BID_VOLUME = 43
    TOTAL_ASK_VOLUME = 44
    VOLUME_IMBALANCE = 45
    WEIGHTED_MID_PRICE = 46
    PRICE_IMPACT = 47
    
    # MBO Features (36 features: indices 48-83)
    # ... order flow, size distribution, queue metrics
    
    # Trading Signals (14 features: indices 84-97)
    TRUE_OFI = 84
    DEPTH_NORM_OFI = 85
    EXECUTED_PRESSURE = 86
    SIGNED_MP_DELTA_BPS = 87
    TRADE_ASYMMETRY = 88
    CANCEL_ASYMMETRY = 89
    FRAGILITY_SCORE = 90
    DEPTH_ASYMMETRY = 91
    BOOK_VALID = 92          # Safety gate
    TIME_REGIME = 93         # Categorical {0-4}
    MBO_READY = 94           # Safety gate
    DT_SECONDS = 95
    INVALIDITY_DELTA = 96    # Safety gate
    SCHEMA_VERSION = 97
```

### Label Encoding

```python
# Original labels (from Rust pipeline)
LABEL_DOWN: Final[int] = -1
LABEL_STABLE: Final[int] = 0
LABEL_UP: Final[int] = 1
NUM_CLASSES: Final[int] = 3

# Shifted labels (for PyTorch CrossEntropyLoss)
SHIFTED_LABEL_DOWN: Final[int] = 0    # Was -1
SHIFTED_LABEL_STABLE: Final[int] = 1  # Was 0
SHIFTED_LABEL_UP: Final[int] = 2      # Was 1
```

### Feature Presets (src/lobtrainer/constants/feature_presets.py)

Named feature subsets for easy configuration:

```python
FEATURE_PRESETS = {
    "lob_only": list(range(0, 40)),         # 40 raw LOB features
    "lob_derived": list(range(0, 48)),      # LOB + derived (48)
    "full": list(range(0, 98)),             # All 98 features
    "signals_core": [84, 85, 86, 87, 88, 89, 90, 91],  # 8 core signals
    "signals_full": list(range(84, 98)),    # 14 signal features
    "lob_signals": list(range(0, 40)) + list(range(84, 92)),  # LOB + core signals
    "no_meta": list(range(0, 92)),          # Exclude meta (92-97)
    "deeplob_extended": list(range(0, 48)), # For extended DeepLOB mode
}

# Usage
from lobtrainer.constants import get_feature_preset, list_presets, describe_preset

indices = get_feature_preset("signals_core")  # [84, 85, 86, 87, 88, 89, 90, 91]
print(list_presets())  # ['lob_only', 'lob_derived', 'full', ...]
describe_preset("signals_core")  # Prints description and indices
```

---

## 5. Configuration System

### ExperimentConfig (src/lobtrainer/config/schema.py)

Root configuration object:

```python
@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    name: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    
    output_dir: str = "outputs"
    log_level: str = "INFO"
```

### DataConfig

```python
@dataclass
class DataConfig:
    data_dir: str = "../data/exports/nvda_11month_complete"
    feature_count: int = 98
    horizon_idx: Optional[int] = 0

    sequence: SequenceConfig = field(default_factory=SequenceConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)

    labeling_strategy: LabelingStrategy = LabelingStrategy.TLOB
    num_classes: int = 3
    cache_in_memory: bool = True

    # Phase 4 Batch 4c: FeatureSet-registry selection (mutually exclusive w/
    # feature_indices and feature_preset — at most one may be set).
    feature_set: Optional[str] = None            # e.g. "momentum_hft_v1" → registry lookup
    feature_indices: Optional[List[int]] = None  # explicit override (bypasses registry)
    feature_preset: Optional[str] = None         # DEPRECATED (hard-error 2026-08-15)
    feature_sets_dir: Optional[str] = None       # override registry root (test isolation)

    # Private runtime cache populated by resolver; NOT serialized to YAML.
    # R3 invariant: to_dict() strips all `_`-prefixed keys at both dataclass
    # and dict branches, preserving on-disk YAML round-trip.
    _feature_indices_resolved: Optional[List[int]] = field(init=False, repr=False, compare=False, default=None)
    _feature_set_ref_resolved: Optional[tuple[str, str]] = field(init=False, repr=False, compare=False, default=None)
```

### ModelConfig

```python
class ModelType(str, Enum):
    LOGISTIC = "logistic"
    XGBOOST = "xgboost"
    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"  # NOT IMPLEMENTED — reserved for future use
    DEEPLOB = "deeplob"
    TLOB = "tlob"
    HMHP = "hmhp"            # Multi-horizon classification (lob-models)
    HMHP_REGRESSION = "hmhp_regression"  # Multi-horizon regression (lob-models)
    TEMPORAL_RIDGE = "temporal_ridge"      # sklearn Ridge + temporal features
    TEMPORAL_GRADBOOST = "temporal_gradboost"  # sklearn GBR + temporal features

@dataclass
class ModelConfig:
    model_type: ModelType = ModelType.LSTM
    input_size: int = 98
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    num_classes: int = 3
    
    # LSTM-specific
    lstm_bidirectional: bool = False
    lstm_attention: bool = False
    
    # DeepLOB-specific
    deeplob_mode: DeepLOBMode = DeepLOBMode.BENCHMARK
    deeplob_conv_filters: int = 32
    deeplob_inception_filters: int = 64
    deeplob_lstm_hidden: int = 64
    deeplob_num_levels: int = 10
    
    # TLOB-specific
    tlob_hidden_dim: int = 64
    tlob_num_layers: int = 4
    tlob_num_heads: int = 1
    tlob_mlp_expansion: float = 4.0
    tlob_use_sinusoidal_pe: bool = True
    tlob_use_bin: bool = True
    tlob_dataset_type: str = "nvda"
```

### TrainConfig

```python
class LossType(str, Enum):
    CROSS_ENTROPY = "cross_entropy"
    FOCAL = "focal"
    WEIGHTED_CE = "weighted_ce"
    MSE = "mse"
    HUBER = "huber"
    HETEROSCEDASTIC = "heteroscedastic"

class TaskType(str, Enum):
    MULTICLASS = "multiclass"
    BINARY_SIGNAL = "binary_signal"
    REGRESSION = "regression"

@dataclass
class TrainConfig:
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 100
    early_stopping_patience: int = 10
    gradient_clip_norm: Optional[float] = 1.0
    
    optimizer: str = "adamw"       # Options: "adamw", "adam", "sgd"
    scheduler: str = "cosine"      # Options: "cosine", "step", "plateau", "none"
    num_workers: int = 4
    pin_memory: bool = True
    seed: int = 42
    mixed_precision: bool = False
    
    # Loss configuration
    loss_type: LossType = LossType.WEIGHTED_CE  # Also: CROSS_ENTROPY, FOCAL, MSE, HUBER, HETEROSCEDASTIC, GMADL
    use_class_weights: bool = True
    task_type: TaskType = TaskType.MULTICLASS
    
    # Focal loss parameters
    focal_gamma: float = 2.0
    focal_alpha: Optional[float] = None
```

### LabelingStrategy

```python
class LabelingStrategy(str, Enum):
    TLOB = "tlob"
    """Classes: 0=Down, 1=Stable, 2=Up (default)"""
    
    TRIPLE_BARRIER = "triple_barrier"
    """Classes: 0=StopLoss, 1=Timeout, 2=ProfitTarget"""
    
    OPPORTUNITY = "opportunity"
    """Classes: 0=BigDown, 1=NoOpportunity, 2=BigUp"""
    
    REGRESSION = "regression"
    """Continuous bps returns (float64), no discretization"""
```

### Config Inheritance (`_base`) — `src/lobtrainer/config/merge.py`

Phase 3 (2026-04-15) promoted `_base:` to a first-class multi-base composition mechanism. v1 (single-string only) is preserved in `archive/merge-v1/` for byte-identity parity testing.

`resolve_inheritance(data, config_path)` accepts:

| Form | Example | Semantics |
|------|---------|-----------|
| `_base: "path.yaml"` | monolith style (v1-compat) | Single base, backward compatible |
| `_base: ["a.yaml", "b.yaml"]` | axis-composed style (v2) | Bases merge **left-to-right** (each successive base overrides the previous); child config overrides all accumulated bases |
| `_base: null` or absent | — | No inheritance |
| `_base: []` / `_base: 42` / `_base: [""]` | — | Raises `ValueError` (exhaustive validation) |

**v1 invariants preserved byte-identically** (verified by 21 tests (pytest-collected) in `tests/test_config_inheritance.py` + golden-fixture parity in `tests/test_merge_v1_parity.py`):
- `deep_merge`: dicts recurse; lists REPLACE (not append); `None` explicitly overrides
- Entry-level cycle detection (`_seen` set tracks resolved absolute paths)
- Depth cap: `_MAX_INHERITANCE_DEPTH = 10` (per-branch budget — each base chain counted from the current depth)
- Relative `_base:` paths resolved against the child config's parent directory
- `_base` key stripped from the returned dict (pop-on-read mutation pattern)

**`is_partial_base()`**: detects the `_partial: true` sentinel. Bases declaring this at top level are standalone-invalid (only meaningful when composed with peer bases). `ExperimentConfig.from_yaml()` checks the raw YAML for this sentinel and raises a descriptive error if a partial base is loaded directly.

### Axis-Partitioned Bases (`configs/bases/`) — Phase 3

21 orthogonal base configs across 4 axes (see `configs/bases/README.md` for the complete ownership matrix and rule):

| Axis | Count | Owns (high-level) | Must NOT set |
|------|-------|---|---|
| `models/` | 5 | `model.model_type`, `model.dropout`, `model.tlob_*`, `model.hmhp_*`, `model.regression_loss_type` | `model.num_classes`, `model.input_size`, `train.task_type`, `train.loss_type`, `train.batch_size` |
| `datasets/` | 8 | `data.data_dir`, `data.feature_count`, `data.normalization`, `data.sequence`, `model.input_size` (T13 auto-derivation) | `data.labeling_strategy`, `data.horizon_idx`, `model.num_classes` |
| `labels/` | 4 | `data.labeling_strategy`, `data.horizon_idx`, `data.num_classes`, `model.num_classes`, `train.task_type`, `train.loss_type` (task-coupled) | `model.*` (other than num_classes), `data.feature_count` |
| `train/` | 4 | `train.batch_size`, `train.epochs`, `train.optimizer`, `train.scheduler`, `train.learning_rate`, `train.weight_decay`, `train.seed`, `train.gradient_clip_norm`, `train.use_class_weights`, `train.focal_gamma` | `train.task_type`, `train.loss_type`, `model.*`, `data.*` |

**Per-child (NOT in any base)**: `name`, `description`, `tags`, `output_dir`, `log_level`.

**Chained-inheritance patterns** (lock-tested by `tests/test_base_axis_ownership.py::TestChainedInheritancePurity`):

1. **TLOB compact**: `tlob_compact_bare` → `tlob_compact_regression` (regression adds `tlob_use_cvml: false` + `tlob_cvml_out_channels: 0` on top of 12 shared arch fields). E4 TLOB uses `bare` directly (keeps its pre-migration golden byte-identical); E5/E6 use the full chain (same byte-identity guarantee).

2. **HMHP cascade**: `hmhp_cascade_bare` → `hmhp_cascade_regression`. `bare` has 10 model fields (`model_type: hmhp` + `dropout` + 8 `hmhp_*` arch fields); `regression` chains from `bare`, overrides `model_type: hmhp_regression`, and adds `hmhp_regression_loss_type: huber`. HMHP classification/triple-barrier use `bare` directly (those regression fields would corrupt their goldens); HMHP regression uses the chain.

**Ownership refinement (Batch 2, 2026-04-15)**: `train.loss_type` moved from `models/` → `labels/` because it is **task-coupled** (regression → huber, tlob → weighted_ce, triple_barrier → focal), not model-coupled. Before this move, HMHP — which shares one cascade model across all three loss types — would have required three near-duplicate HMHP model bases. All migrated configs' resolved dicts are byte-identical before and after the refinement.

**Migration status (2026-04-15)**: 25 of 42 in-scope experiment configs migrated to axis-composed form across 3 batches (E4×1 + E5×5 + E6×1 + HMHP×11 + TLOB classif×7 = 25). 17 configs remain standalone by design (baselines×7, XGBoost×2, archive×6, niche HMHP×2). Monolith `bases/e5_tlob_regression.yaml` retired at end of Batch 1 after all 5 E5 consumers migrated with byte-identical resolved dicts.

See `MERGE_MIGRATION_PLAN.md` for the per-batch migration ledger.

---

## 6. Data Loading Pipeline

### Directory Structure Expected

```
data/exports/nvda_11month_complete/
├── train/
│   ├── 2025-02-03_sequences.npy    # [N_seq, 100, 98] float32
│   ├── 2025-02-03_labels.npy       # [N_seq, 4] int8 (multi-horizon)
│   ├── 2025-02-03_regression_labels.npy  # [N_seq, H] float64 bps (optional, for HMHP regression)
│   └── ...
├── val/
│   └── ...
├── test/
│   └── ...
└── dataset_manifest.json
```

### DayData Class

```python
@dataclass
class DayData:
    """Container for one day's data."""
    date: str                              # e.g., "2025-02-03"
    features: np.ndarray                   # [N, 98] - flat features
    labels: np.ndarray                     # [N_seq] or [N_seq, n_horizons]
    sequences: Optional[np.ndarray] = None # [N_seq, 100, 98]
    regression_labels: Optional[np.ndarray] = None  # [N_seq, n_horizons] float64 bps - from feature extractor
    metadata: Optional[Dict] = None
    is_aligned: bool = False
    
    @property
    def num_sequences(self) -> int:
        return self.labels.shape[0]
    
    @property
    def is_multi_horizon(self) -> bool:
        return self.labels.ndim == 2
    
    @property
    def num_horizons(self) -> int:
        return self.labels.shape[1] if self.is_multi_horizon else 1
    
    def get_labels(self, horizon_idx: Optional[int] = 0) -> np.ndarray:
        """Get labels for specific horizon or all horizons."""
        if horizon_idx is None:
            return self.labels
        if not self.is_multi_horizon:
            return self.labels
        return self.labels[:, horizon_idx]
```

### Loading Functions

```python
from lobtrainer.data import load_split_data, load_day_data, create_dataloaders

# Load all days in a split
train_days: List[DayData] = load_split_data(Path("data/exports/nvda_11month_complete"), "train")

# load_day_data accepts regression_labels_path for HMHP regression
day = load_day_data(
    split_dir / "2025-02-03_sequences.npy",
    split_dir / "2025-02-03_labels.npy",
    regression_labels_path=split_dir / "2025-02-03_regression_labels.npy",  # optional
)

# load_split_data auto-detects {day}_regression_labels.npy per day
# Create dataloaders
loaders = create_dataloaders(
    data_dir="data/exports/nvda_11month_complete",
    batch_size=64,
    horizon_idx=0,
)
```

### Contract Validation at Load Time

As of v0.2.0, `load_split_data()` enforces the pipeline contract at data loading
boundaries using `hft_contracts.validation`:

1. **Metadata mandatory** (aligned format): Raises `FileNotFoundError` if `{date}_metadata.json` is missing
2. **Contract validation** (first day only): Calls `validate_export_contract()` which checks:
   - Schema version matches `hft_contracts.SCHEMA_VERSION`
   - Feature count in `{FEATURE_COUNT, FULL_FEATURE_COUNT}`
   - Normalization not applied (safe for Python normalization)
   - Metadata completeness (warns on missing optional fields)
   - Label encoding matches contract
   - Provenance present (warns if missing)
3. **Expected feature count**: Passes `n_features` from metadata to `DayData.validate()`
4. **Label shift**: Uses `hft_contracts.get_contract(strategy).shift_for_crossentropy` instead of heuristic detection

---

## 7. PyTorch Dataset Classes

### LOBSequenceDataset

For sequence models (LSTM, Transformer, TLOB):

```python
class LOBSequenceDataset(Dataset):
    """
    Each item is (sequence, label) where:
    - sequence: [seq_len, n_features] tensor
    - label: scalar tensor (shifted to {0, 1, 2})
    
    For HMHP regression: use_precomputed_regression=True loads regression_labels.npy
    from the feature extractor instead of computing on-the-fly.
    """
    
    def __init__(
        self,
        days: List[DayData],
        horizon_idx: Optional[int] = 0,
        feature_indices: Optional[List[int]] = None,
        transform: Optional[Callable] = None,
        use_precomputed_regression: bool = False,  # Use feature extractor's regression_labels.npy
    ): ...
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get sequence and label
        sequence = day.sequences[local_idx].copy()
        label = day.get_labels(self.horizon_idx)[local_idx]
        
        # CRITICAL: Shift labels from {-1, 0, 1} to {0, 1, 2}
        label = label + 1
        
        return (
            torch.from_numpy(sequence).float(),
            torch.tensor(label, dtype=torch.long),
        )
```

### LOBFlatDataset

For non-sequence models (XGBoost, MLP, Logistic):

```python
class LOBFlatDataset(Dataset):
    """
    Each item is (features, label) where features is flattened.
    """
    
    def __init__(
        self,
        days: List[DayData],
        flatten_mode: str = "last",  # "last", "flatten", "mean"
        horizon_idx: Optional[int] = 0,
        ...
    ): ...
```

---

## 8. Data Transforms

### BinaryLabelTransform

```python
class BinaryLabelTransform:
    """Convert multi-class to binary (signal vs no-signal)."""
    
    def __init__(self, positive_classes: List[int] = [0, 2]):
        """Classes 0 (Down) and 2 (Up) become 1 (Signal)."""
        ...
```

### Normalization Module (`lobtrainer.data.normalization`)

The preferred normalization path. Key difference from `transforms.py`: exclusion
indices are sourced from `hft_contracts.NON_NORMALIZABLE_INDICES` (defined in
`pipeline_contract.toml`), not hardcoded.

```python
from lobtrainer.data.normalization import HybridNormalizer

normalizer = HybridNormalizer(exclude_indices=None)  # Uses contract defaults
# Default excludes: {92, 93, 94, 96, 97, 115} — categorical + counter features
```

`NON_NORMALIZABLE_INDICES` is a superset of `CATEGORICAL_INDICES`, also including
`invalidity_delta` (index 96, a counter with reset semantics) and `time_bucket`
(index 115, a discretized categorical).

---

## 9. Model Implementations

### LSTMClassifier / GRUClassifier (lobmodels.models.rnn)

LSTM and GRU models live in the `lob-models` package (`lobmodels.models.rnn`), registered in ModelRegistry as `"lstm"` and `"gru"`. They return `ModelOutput(logits=...)` and support optional attention and bidirectional modes. Re-exported by `lobtrainer.models` for convenience.

### Model Factory

```python
from lobtrainer.models import create_model
from lobtrainer.config import ModelConfig, ModelType

# LSTM
model = create_model(ModelConfig(model_type=ModelType.LSTM))

# DeepLOB (requires lob-models package)
model = create_model(ModelConfig(
    model_type=ModelType.DEEPLOB,
    deeplob_mode=DeepLOBMode.BENCHMARK,
))

# TLOB (requires lob-models package)
model = create_model(ModelConfig(
    model_type=ModelType.TLOB,
    tlob_hidden_dim=64,
    tlob_num_layers=4,
))
```

---

## 10. Training Infrastructure

### Trainer Class (src/lobtrainer/training/trainer.py)

```python
class Trainer:
    """Main training class for LOB models."""
    
    def __init__(
        self,
        config: ExperimentConfig,
        callbacks: Optional[List[Callback]] = None,
        device: Optional[torch.device] = None,
    ): ...
    
    def train(self) -> Dict[str, Any]:
        """
        Run full training loop.
        
        Returns:
            Dict with best_val_metric, best_epoch, total_epochs, history
        """
        ...
    
    def evaluate(self, split: str = "test") -> ClassificationMetrics:
        """Evaluate model on a data split."""
        ...
```

### Training Strategy Pattern (Phase 2)

The Trainer delegates all task-specific logic to a `TrainingStrategy` subclass, selected automatically from the config via `create_strategy()`:

| Strategy | Model Types | Key Behavior |
|----------|-------------|-------------|
| `ClassificationStrategy` | TLOB, DeepLOB, MLPLOB, LogisticLOB, LSTM, GRU | Owns criterion (CE/Focal with class weights). Uses `self._criterion(output.logits, labels)` for BOTH training and validation. Does NOT call `model.compute_loss()`. |
| `RegressionStrategy` | TLOB-R, DeepLOB-R | Uses `model.compute_loss(output, regression_targets=...)`. Computes R², IC, MAE, RMSE, DA. |
| `HMHPClassificationStrategy` | HMHP | Dict labels `{h: tensor}`. Per-horizon loss, accuracy, agreement, confirmation. |
| `HMHPRegressionStrategy` | HMHP-R | Dict regression targets. Per-horizon R², IC, MAE. Primary horizon surfaced for early stopping. |

Each strategy implements: `process_batch()`, `aggregate_epoch_metrics()`, `validate()`, `evaluate()`, `predict()`.

The Trainer (900L) handles the outer loop: epochs, callbacks, scheduling, checkpointing. Zero task-branching remains in the Trainer itself.

### ModelConfig (Phase 3)

`ModelConfig` has 7 core fields plus an opaque `params` dict:
- **Core**: `model_type`, `input_size`, `num_classes`, `params`, `hmhp_horizons`, `hmhp_use_regression`, `deeplob_mode`
- **`name` property**: Derives registry key from `model_type` (e.g., `ModelType.TLOB` → `"tlob"`)
- **`params` dict**: Passed through to the model's config class via `ModelRegistry.create()`
- **Legacy compat**: ~30 flat fields auto-migrate to `params` via `_build_params_from_legacy()` in `__post_init__`

`create_model()` (45L) uses `ModelRegistry.get(name)` for most models, lobmodels factory functions for HMHP.

### TrainingState

```python
@dataclass
class TrainingState:
    """Mutable training state."""
    current_epoch: int = 0
    global_step: int = 0
    best_val_metric: float = float('inf')
    best_epoch: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)
```

### Loss Functions (src/lobtrainer/training/loss.py)

```python
class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Reference: Lin et al. (2017), "Focal Loss for Dense Object Detection"
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ): ...

class BinaryFocalLoss(nn.Module):
    """Focal Loss for binary classification."""
    ...

def create_focal_loss(
    num_classes: int,
    gamma: float = 2.0,
    class_counts: Optional[torch.Tensor] = None,
) -> nn.Module:
    """Factory function for focal loss."""
    ...
```

---

## 11. Callback System

### Base Callback (src/lobtrainer/training/callbacks.py)

```python
class Callback:
    """Base class for training callbacks."""
    
    def on_train_start(self) -> None: ...
    def on_train_end(self) -> None: ...
    def on_epoch_start(self, epoch: int) -> None: ...
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None: ...
    def on_batch_start(self, batch_idx: int) -> None: ...
    def on_batch_end(self, batch_idx: int, metrics: Dict[str, float]) -> None: ...
    def on_validation_end(self, epoch: int, metrics: Dict[str, float]) -> None: ...
```

### Built-in Callbacks

```python
class EarlyStopping(Callback):
    """Stop training when metric stops improving."""
    def __init__(
        self,
        patience: int = 10,
        metric: str = "val_loss",
        mode: str = "min",
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
    ): ...

class ModelCheckpoint(Callback):
    """Save model checkpoints during training."""
    def __init__(
        self,
        save_dir: Union[str, Path],
        metric: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
        max_checkpoints: int = 5,
    ): ...

class MetricLogger(Callback):
    """Log metrics to file and console."""
    ...

class ProgressCallback(Callback):
    """Display training progress with tqdm."""
    ...
```

---

## 12. Metrics and Evaluation

### Regression Metrics (src/lobtrainer/training/regression_metrics.py)

For HMHP regression models. Wired into trainer validation and evaluation.

```python
# Individual metrics
r_squared(y_true, y_pred)           # R² coefficient of determination
information_coefficient(y_true, y_pred)  # Spearman rank correlation (IC)
pearson_correlation(y_true, y_pred)
mean_absolute_error(y_true, y_pred)
root_mean_squared_error(y_true, y_pred)
directional_accuracy(y_true, y_pred)     # Fraction where sign matches
profitable_accuracy(y_true, y_pred, breakeven_bps=5.0)  # Correct direction AND |actual| > threshold

# Wrapper
compute_all_regression_metrics(y_true, y_pred, breakeven_bps=5.0, prefix="") -> Dict[str, float]
```

### MetricsCalculator (src/lobtrainer/training/metrics.py)

Strategy-aware metrics calculator:

```python
class MetricsCalculator:
    """
    Computes metrics that match labeling strategy semantics.
    
    Example:
        >>> calc = MetricsCalculator("triple_barrier")
        >>> metrics = calc.compute(preds, labels)
        >>> print(metrics.strategy_metrics["predicted_trade_win_rate"])
    """
    
    def __init__(self, strategy: str = "tlob", num_classes: int = 3): ...
    
    def compute(
        self,
        predictions: Tensor,
        labels: Tensor,
        loss: Optional[float] = None,
    ) -> ClassificationMetrics: ...
```

### ClassificationMetrics

```python
@dataclass
class ClassificationMetrics:
    """Complete classification evaluation results."""
    accuracy: float
    loss: float
    
    per_class_precision: Dict[int, float]
    per_class_recall: Dict[int, float]
    per_class_f1: Dict[int, float]
    per_class_count: Dict[int, int]
    
    macro_precision: float
    macro_recall: float
    macro_f1: float
    
    confusion_matrix: np.ndarray
    
    # Strategy-specific metrics
    strategy_metrics: Dict[str, float]
    """
    Examples:
    - triple_barrier: predicted_trade_win_rate, decisive_prediction_rate
    - opportunity: directional_accuracy, opportunity_prediction_rate
    - tlob: directional_accuracy, signal_rate
    """
    
    class_names: List[str]
    
    def summary(self) -> str: ...
    def to_dict(self) -> Dict[str, float]: ...
```

### Strategy-Specific Metrics

| Strategy | Key Metrics | Description |
|----------|-------------|-------------|
| **Triple Barrier** | `predicted_trade_win_rate` | When predicting trade, actual win rate |
| | `decisive_prediction_rate` | How often we predict StopLoss or ProfitTarget |
| **Opportunity** | `directional_accuracy` | When predicting direction, correctness |
| | `opportunity_prediction_rate` | How often we predict BigUp or BigDown |
| **TLOB** | `directional_accuracy` | Accuracy on non-Stable predictions |
| | `signal_rate` | Fraction of Up/Down predictions |

### Evaluation Framework (src/lobtrainer/training/evaluation.py)

```python
@dataclass
class BaselineReport:
    """Comprehensive report comparing model against baselines."""
    
    model_name: str
    split: str
    n_samples: int
    
    model_metrics: ClassificationMetrics
    class_prior_metrics: ClassificationMetrics
    previous_label_metrics: ClassificationMetrics
    
    beats_class_prior: bool
    beats_previous_label: bool
    improvement_over_prior: float
    improvement_over_previous: float

def evaluate_model(model, X, y, name=None) -> ClassificationMetrics: ...
def evaluate_naive_baseline(y_true, split_name="test") -> Dict[str, ClassificationMetrics]: ...
def create_baseline_report(model, X, y, split="test") -> BaselineReport: ...
def full_evaluation(model, X, y, split="test") -> Dict: ...
```

---

## 13. Baseline Models

### Available Baselines (src/lobtrainer/models/baselines.py)

```python
class NaiveClassPrior(BaseModel):
    """Always predicts the most common class."""
    
    def fit(self, X, y) -> 'NaiveClassPrior': ...
    def predict(self, X) -> np.ndarray: ...

class NaivePreviousLabel(BaseModel):
    """
    Persistence baseline: predict the previous label.
    Exploits label autocorrelation.
    """
    
    def predict(self, X) -> np.ndarray: ...

class LogisticBaseline(BaseModel):
    """Logistic regression baseline using sklearn."""
    
    def __init__(
        self,
        config: LogisticBaselineConfig = None,
        flatten_mode: str = "last",
        feature_indices: Optional[List[int]] = None,
    ): ...
```

---

## 14. Monitoring and Diagnostics

### GradientMonitor (src/lobtrainer/training/monitoring.py)

```python
@dataclass
class GradientStats:
    total_norm: float
    max_norm: float
    min_norm: float
    mean_norm: float
    num_zero_grads: int
    num_nan_grads: int
    num_inf_grads: int
    layer_norms: Dict[str, float]

class GradientMonitor(Callback):
    """
    Monitor gradient statistics to detect training issues.
    
    Detects:
    - Vanishing gradients (norms < 1e-7)
    - Exploding gradients (norms > 1000)
    - NaN gradients
    """
    
    def __init__(self, log_every: int = 100): ...
    def get_history(self) -> List[GradientStats]: ...
```

### TrainingDiagnostics

```python
@dataclass
class HealthCheckResult:
    is_healthy: bool
    warnings: List[str]
    recommendations: List[str]

class TrainingDiagnostics(Callback):
    """
    Aggregate training diagnostics.
    
    Tracks:
    - Loss trends
    - Gradient health
    - Learning rate
    - Time per epoch
    """
    
    def get_summary(self) -> Dict: ...
    def check_health(self) -> HealthCheckResult: ...
```

### PerClassMetricsTracker

```python
class PerClassMetricsTracker(Callback):
    """
    Track per-class metrics (precision, recall, F1) per epoch.
    
    Useful for detecting:
    - Model ignoring minority class
    - Precision/recall trade-offs
    """
    ...
```

### Convenience Factory

```python
def create_standard_monitoring(
    log_every: int = 100,
    include_gradients: bool = True,
    include_lr: bool = True,
    include_diagnostics: bool = True,
    include_per_class: bool = False,
) -> List[Callback]:
    """Create standard monitoring callbacks."""
    ...
```

---

## 15. Experiment Tracking

### ExperimentResult (src/lobtrainer/experiments/result.py)

```python
@dataclass
class ExperimentMetrics:
    accuracy: float = 0.0
    loss: float = 0.0
    macro_f1: float = 0.0
    directional_accuracy: float = 0.0
    signal_rate: float = 0.0
    per_class_f1: Dict[str, float] = field(default_factory=dict)
    strategy_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class ExperimentResult:
    experiment_id: str
    name: str
    config: Dict[str, Any]
    
    train_metrics: Optional[ExperimentMetrics] = None
    val_metrics: Optional[ExperimentMetrics] = None
    test_metrics: Optional[ExperimentMetrics] = None
    
    timestamp: str = ""
    duration_seconds: float = 0.0
    checkpoint_path: Optional[str] = None
    
    def save(self, path: Path) -> None: ...
    
    @classmethod
    def load(cls, path: Path) -> 'ExperimentResult': ...
```

### ExperimentRegistry (src/lobtrainer/experiments/registry.py)

```python
class ExperimentRegistry:
    """Central tracker for all experiments."""
    
    def __init__(self, base_dir: Union[str, Path], create_if_missing: bool = True): ...
    
    def register(self, result: ExperimentResult) -> str: ...
    def get(self, experiment_id: str) -> Optional[ExperimentResult]: ...
    def list_all(self) -> List[ExperimentResult]: ...
    def filter(
        self,
        tags: Optional[List[str]] = None,
        model_type: Optional[str] = None,
        min_accuracy: Optional[float] = None,
    ) -> List[ExperimentResult]: ...

def create_comparison_table(
    registry: ExperimentRegistry,
    metric_keys: List[str] = None,
    sort_by: str = "macro_f1",
) -> str:
    """Create markdown comparison table of experiments."""
    ...
```

---

## 16. Reproducibility Utilities

### set_seed (src/lobtrainer/utils/reproducibility.py)

```python
def set_seed(seed: int = 42, deterministic_cudnn: bool = True) -> None:
    """
    Set random seeds for all RNGs.
    
    Sets:
    - Python random module
    - NumPy random
    - PyTorch CPU
    - PyTorch CUDA (all GPUs)
    - CuDNN determinism (if enabled)
    """
    ...

@dataclass
class SeedManager:
    """Context manager for reproducible code blocks."""
    seed: int
    restore_state: bool = False
    
    def __enter__(self) -> 'SeedManager': ...
    def __exit__(self, *args) -> None: ...

def create_worker_init_fn(base_seed: int):
    """Create worker_init_fn for DataLoader with deterministic seeding."""
    ...
```

---

## 17. Scripts and CLI

### train.py

```bash
# Basic training
python scripts/train.py --config configs/experiments/nvda_tlob_h10_v1.yaml

# With overrides
python scripts/train.py --config configs/experiments/nvda_tlob_h10_v1.yaml \
    --epochs 50 \
    --batch-size 128

# Resume from checkpoint
python scripts/train.py --config configs/experiments/nvda_tlob_h10_v1.yaml \
    --resume outputs/checkpoints/best.pt

# Evaluation only
python scripts/train.py --config configs/experiments/nvda_tlob_h10_v1.yaml \
    --eval-only --resume outputs/checkpoints/best.pt
```

### Other Scripts

| Script | Purpose |
|--------|---------|
| `scripts/export_signals.py` | Unified signal export CLI (replaces 3 deprecated scripts) |
| `scripts/run_simple_training.py` | SimpleModelTrainer CLI for TemporalRidge/GradBoost |
| `scripts/validate_export.py` | Validate exported dataset integrity |
| `scripts/analysis/evaluate_model.py` | Evaluate trained model checkpoint |
| `scripts/analysis/run_baseline_evaluation.py` | Compare against naive baselines |

---

## 18. Configuration Reference

See `configs/README_configs.md` for complete configuration reference including:

- Active experiment configs (42 in-scope: 25 migrated to axis-composed `_base: [...]` form + 17 standalone by design — see `MERGE_MIGRATION_PLAN.md`)
- Axis-partitioned bases (21 under `configs/bases/{models,datasets,labels,train}/` — see `configs/bases/README.md`)
- Archived reference configs (6 legacy, not in Phase 3 migration scope)
- Horizon index mapping
- Model type options
- Loss function selection
- Configuration template

### Current Datasets

| Dataset | Days | Labels | Horizons |
|---------|------|--------|----------|
| `nvda_11month_complete` | **234** | TLOB | [10, 20, 50, 100] |
| `nvda_11month_triple_barrier` | **234** | Triple Barrier | [50, 100, 200] |

---

## 19. Testing Patterns

### Test Structure

```python
# tests/test_trainer.py
class TestTrainingState:
    def test_initial_state(self):
        state = TrainingState()
        assert state.current_epoch == 0
        assert state.best_val_metric == float('inf')

class TestTrainer:
    @pytest.fixture
    def mock_config(self, tmp_path):
        return ExperimentConfig(...)
    
    def test_trainer_creation(self, mock_config):
        trainer = Trainer(mock_config)
        assert trainer.config.name == "test"
```

### Test Modules (43; 1052 tests pytest-collected on 2026-04-15)

| Test File | Coverage |
|-----------|----------|
| `test_baselines.py` | NaiveClassPrior, NaivePreviousLabel, LogisticBaseline |
| `test_regression_metrics.py` | r_squared, IC, MAE, RMSE, directional/profitable accuracy |
| `test_regression_dataset.py` | DayData.regression_labels, load_day_data, LOBSequenceDataset |
| `test_config.py` | Configuration loading and validation |
| `test_deeplob_integration.py` | DeepLOB model creation |
| `test_evaluation.py` | BaselineReport, evaluate_model |
| `test_experiments.py` | ExperimentResult, ExperimentRegistry |
| `test_feature_index.py` | Feature indices and constants |
| `test_feature_presets.py` | Feature preset functions |
| `test_integration.py` | End-to-end training |
| `test_loss.py` | FocalLoss, BinaryFocalLoss |
| `test_monitoring.py` | GradientMonitor, TrainingDiagnostics |
| `test_strategy_metrics.py` | MetricsCalculator |
| `test_tlob_integration.py` | TLOB model creation |
| `test_trainer.py` | Trainer class |
| `test_transforms.py` | FeatureStatistics, compute_statistics |
| `test_normalization.py` | GlobalZScore, HybridNormalizer, Welford, streaming |
| `test_normalization_integration.py` | Normalizer as dataset transform, end-to-end |
| `test_feature_selector.py` | FeatureSelector construction, validation, selection |
| `test_label_shift.py` | Label shift resolution (4 paths) |
| `test_regression_training.py` | HMHP regression training pipeline |
| `test_standard_regression_training.py` | DeepLOB regression training pipeline |
| `test_calibration.py` | Variance calibration metrics |
| `test_create_dataloaders.py` | Feature selection cascade, num_workers override, strategy routing |
| `test_hmhp_collate.py` | `_hmhp_collate_fn` 2/3-tuple dict-label collation |
| `test_optimizer_scheduler.py` | Optimizer + LR scheduler factories |
| `test_regression_evaluation.py` | Regression metrics + per-horizon reporting |
| `test_signal_export.py` | Signal export pipeline (classification + regression outputs) |
| `test_signal_export_inference.py` | Signal export inference loop |
| `test_simple_trainer.py` | SimpleModelTrainer (TemporalRidge, GradBoost) |
| `test_strategies.py` | TrainingStrategy ABC + per-model strategy implementations |
| `test_sources_and_bundle.py` | DayBundle multi-source fusion (T12) |
| `test_forward_prices_integration.py` | ForwardPriceContract / LabelFactory integration (T9–T10) |
| `test_integration_wrappers.py` | Wrapper-manifest integration path |
| `test_experiment_spec_and_gates.py` | `ExperimentSpec` orchestrator + decision gates (T14) |
| **Phase 3 (config composition)** — all 5 files added 2026-04-15: | |
| `test_config_inheritance.py` | **21 tests** (pytest-collected; static `grep "def test_"` = 22, one method filtered at collection) — v1 `_base:` single-string semantics (depth cap, cycle detection, list-REPLACE, None override, pop-on-read, relative paths, `_partial:` sentinel). All pre-Phase-3 invariants locked. |
| `test_multi_base_inheritance.py` | v2 `_base: list[str]` multi-base composition — exhaustive 2/3/4-base merge, explicit diamond case, left-to-right order |
| `test_merge_v1_parity.py` | Byte-identity parity between v1 (`importlib`-loaded from `archive/merge-v1/`) and v2 on every in-scope pre-migration golden JSON fixture |
| `test_base_axis_ownership.py` | Mechanical §3.4 ownership enforcement (each top-level dotted-key appears in exactly one axis directory); `TestChainedInheritancePurity` locks the TLOB compact and HMHP cascade chained patterns |
| `test_migrated_configs_e2e.py` | Auto-discovers migrated configs; verifies `ExperimentConfig.from_yaml` loads each without error; meta-tests cover every E-family / HMHP-family / TLOB-classif-family member |

---

## 20. Known Limitations

### Label Shift (+1)

Labels are shifted from `{-1, 0, 1}` to `{0, 1, 2}` in `__getitem__`:

```python
label = label + 1  # For PyTorch CrossEntropyLoss
```

### Single-Horizon Training (Classification)

While data supports multi-horizon labels, classification trainers use a single horizon (`horizon_idx: 0`). HMHP_REGRESSION trains on all horizons jointly.

### num_workers Default

DataLoader workers default to 4. Set to 0 if experiencing multiprocessing issues.

### External Model Dependency

DeepLOB, TLOB, HMHP, and HMHP_REGRESSION require the `lob-models` package:

```bash
pip install -e ../lob-models
```

Verify installation:

```python
from lobtrainer.models import create_model  # Will ImportError if lob-models missing
```

### TIME_REGIME Exclusion

Index 93 (TIME_REGIME) should be excluded from normalization - it's categorical `{0, 1, 2, 3, 4}`.

---

## Quick Reference

### Imports

```python
# Core
from lobtrainer import Trainer, create_trainer, set_seed
from lobtrainer.config import load_config, ExperimentConfig, ModelType

# Models
from lobtrainer.models import create_model, LSTMClassifier, LogisticBaseline

# Data
from lobtrainer.data import LOBSequenceDataset, create_dataloaders

# Metrics
from lobtrainer.training import MetricsCalculator, ClassificationMetrics

# Experiments
from lobtrainer.experiments import ExperimentResult, ExperimentRegistry

# Constants
from lobtrainer.constants import FeatureIndex, get_feature_preset
```

### Key Defaults

| Parameter | Default |
|-----------|---------|
| Sequence length | 100 |
| Stride | 10 |
| Batch size | 64 |
| Hidden size | 64 |
| Learning rate | 1e-4 |
| Epochs | 100 |
| Patience | 10 |
| Seed | 42 |

---

*Last updated: April 9, 2026*
*Version: 0.4.0*
