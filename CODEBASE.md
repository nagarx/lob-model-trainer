# LOB-Model-Trainer: Codebase Technical Reference

> **Pipeline scope (2026-06-02).** This module is part of an **intraday trading research pipeline** — an experiment-first platform for discovering and validating *any* profitable **intraday** trading edge (no overnight positions), across approach classes (microstructure/HFT, scalping, intraday momentum, intraday statistical arbitrage, …) and instruments (equities, futures, same-day options). The pipeline *originated* as a high-frequency NVDA MBO/LOB microstructure system — that origin explains the "HFT" / "LOB" / "MBO" naming here — and that microstructure-direction program is now one (largely-closed) track among many. **Names are historical; the mission is general.** This module's role: the Python training engine — data loaders, training loops, purged-CV, train-only normalization, callbacks, signal export, and ExperimentSpec orchestration; trains single-asset models at any intraday horizon (cross-sectional / panel training + ranking losses are a build per register §9). For the full mission + approach taxonomy + capability-readiness boundary, see root `CLAUDE.md` §Research Scope & Charter (+ `CROSS_ASSET_OFI_FINDINGS_AND_ISSUES_2026_06_01.md` §9).

> **Version**: 0.7.0 (Phase A.5 Scope D v2 Pydantic migration — 2026-04-25)  
> **Schema**: 3.0 (Phase G G.6.A bump 2.2 → 3.0 MAJOR per CLAUDE.md root rule: any modification to stable features 0-97 = BREAKING; via `hft-contracts` package, `hft-contracts>=2.8.0` runtime dep per `pyproject.toml`)  
> **Tests**: run `pytest --collect-only -q` for the live count (~2020 across 82 `tests/test_*.py` files, 0 errors — hand-typed counts are NOT maintained here per hft-rules §11) — Phase A.5 Scope D v2 adds 9 Pydantic-migration commits (A.5.3a-i) + 3 post-audit commits (A.5.7a-c): SafeBaseModel base class with `_canonical_form()` SSoT + `__pydantic_init_subclass__` auto-registry (`config/base.py`); parametric pickle/deepcopy/partial-base-rejection tests over auto-registry; full `Trainer.setup() + SignalExporter.export()` integration test in `tests/test_signal_exporter_integration.py` (`@pytest.mark.integration`); `LabelsConfig.validate_primary_horizon_idx_for()` bounds-validation method; `CalibrationContext` TypedDict.  
> **Last Updated**: 2026-07-07 (Phase-2 TRUTH doc-drift pass — module-tree `_compute_content_hash` note corrected to the delegating SSoT reality + stale stamps refreshed; content current through the 2026-07 ledger-freeze / T12-fusion-hazard notes. Prior major revision 2026-04-25, Phase A.5 Scope D v2 — Pydantic v2 migration COMPLETE: all 9 config classes inherit from `SafeBaseModel(BaseModel)` with `ConfigDict(frozen=True, extra="forbid", strict=True)` (exactly 3 flags — `validate_assignment` and `arbitrary_types_allowed` are deliberately NOT set, per `config/base.py`); `dacite>=1.8` DROPPED; `pydantic>=2.7,<3.0` pinned with explicit upper bound; `hatchling>=1.26` build-constraint. Four bug classes retired at TYPE layer: canonical-path-drift, silent mutation, extra-field acceptance, silent-None field access. 5-agent adversarial audit (A.5.7a-c) closed 4 ship-blockers (SB-1 canonical-form SSoT, SB-2 composite pickle/deepcopy, SB-3 full Trainer+export integration, SB-4 parametric partial-base rejection). See CHANGELOG.md v0.7.0 + `/contracts/pipeline_contract.toml` v2.26 + `/PIPELINE_ARCHITECTURE.md` §11 Configuration System Architecture.)  
> **Purpose**: Complete technical reference for LLMs and developers to understand, modify, and extend the codebase.
>
> **Scope**: This library focuses solely on **model training**. For dataset analysis, use `lob-dataset-analyzer`.
>
> **New in 0.7.0 (Phase A.5 Scope D v2, 2026-04-24→2026-04-25)**:
> - Pydantic v2 migration of all 9 config classes — `SafeBaseModel` base class at `src/lobtrainer/config/base.py` with `frozen=True, extra="forbid", strict=True`; dacite dropped; Pydantic v2 subsumes all coercion patterns. Four bug classes retired at the TYPE layer (canonical-path-drift, silent mutation, extra-field acceptance, silent-None field access).
> - `_canonical_form()` SSoT aligns `__eq__` + `__hash__` per Python invariant; `__pydantic_init_subclass__` auto-registry replaces hand-maintained `_PYDANTIC_CONFIG_CLASSES` list (5-agent adversarial audit closed 4 ship-blockers SB-1..SB-4 + 3 HP items).
> - New SSoT method `LabelsConfig.validate_primary_horizon_idx_for(n_horizons)` — bounds-validation co-located with data owner; fail-loud on negative (no silent last-column selection) or out-of-bounds; consumed by 4 exporter.py slicing sites + callback.py.
> - New `CalibrationContext(TypedDict, total=False)` inline in `variance.py` — typed calibration provenance context; extension point for Phase B calibrators.
> - Full `Trainer.setup() + SignalExporter.export()` integration test in `tests/test_signal_exporter_integration.py` (`@pytest.mark.integration`) — locks 7 invariants end-to-end including producer-consumer byte-identity fingerprint recompute.
> - CLI refactor: `ExperimentConfig.model_validate({**base.model_dump(), **args_overrides})` pattern replaces dataclass post-construction mutation; re-fires every validator on user-provided overrides.
>
> **New in 0.4.0 (cumulative through Phase 7 Stage 7.4 Round 4)**:
> - Phase 2 Strategy Pattern refactoring — Trainer decomposed from 1,657L; 4 concrete strategies (Classification, Regression, HMHPClassification, HMHPRegression) under `src/lobtrainer/training/strategies/`. Model Registry integration via lob-models.
> - Phase 2b — `CVTrainer` (purged k-fold + embargo, T11), `sample_weights` (T10 de Prado AFML 4.5.1), data sources abstraction + bundle (T12 multi-source), experiment_spec + gates (T14 pre-training IC gate).
> - Phase 3 — multi-base config composition via `_base:` YAML inheritance (24 axis-partitioned bases, monolith retired 2026-04-15); 6A.5 M6 `yaml.safe_load` dict-guard; 6A.7 `data.feature_set` + `data.feature_sets_dir` axis ownership.
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

Python library for training and evaluating ML models on LOB (Limit Order Book) data for price movement prediction. Built for intraday trading research (origin: HFT microstructure) with emphasis on:

- **Modularity**: Clean separation between data, models, and training
- **Reproducibility**: Explicit seed management and configuration-driven experiments
- **Flexibility**: Multi-horizon labels, multiple model architectures
- **Monitoring**: Gradient tracking, training diagnostics, experiment comparison

### Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| **Strategy Pattern** | ✅ Complete | 4 strategies: Classification, Regression, HMHPClassification, HMHPRegression |
| **Trainer Orchestrator** | ✅ Complete | ~1.9K L — run `wc -l`, zero task-branching, delegates to strategy |
| **All Models via lob-models** | ✅ Complete | 11 registered models (TLOB, DeepLOB, MLPLOB, HMHP, HMHP-R, LSTM, GRU, LogisticLOB, XGBoostLOB, Ridge, GradBoost) — per `../lob-models/src/lobmodels/registry/_snapshot.json` `model_count: 11`. NOTE: `xgboost` is registered but NOT dispatched by `create_strategy` — train it via `scripts/analysis/train_xgboost_baseline.py`, not the standard trainer path |
| **Strategy-Aware Metrics** | ✅ Complete | MetricsCalculator for TLOB/Triple Barrier/Opportunity |
| **Focal Loss** | ✅ Complete | For class imbalance handling |
| **Multi-Horizon Labels** | ✅ Complete | Support for multiple prediction horizons |
| **HMHP Regression** | ✅ Complete | Regression training with precomputed labels |
| **Experiment Tracking** | ✅ Complete | ExperimentRegistry, comparison tables (LEGACY local schema — canonical is `hft_contracts.ExperimentRecord`; see §15) |
| **Monitoring Callbacks** | ✅ Complete | Gradient, LR, diagnostics tracking (uses public properties) |
| **Tests** | ✅ Complete | run `pytest --collect-only -q` for the live count (~2020 across 82 `tests/test_*.py` files) — hand-typed counts not maintained (hft-rules §11) |

### Core Dependencies

```toml
torch = ">=2.0"           # Deep learning framework
numpy = ">=1.24"          # Numerical operations
pandas = ">=2.0"          # Data manipulation
scikit-learn = ">=1.3"    # Classical ML, metrics
scipy = ">=1.10"          # Statistical tests
pyyaml = ">=6.0"          # Configuration files
pydantic = ">=2.7,<3.0"   # Config schema (A.5.3i: replaces dacite; explicit v3 upper bound)
tqdm = ">=4.65"           # Progress bars
# dacite removed (Phase A.5.3i, 2026-04-24) — Pydantic v2 subsumes all coercion patterns
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
│   └── feature_presets.py         # Named feature subsets (13 presets — see §4)
│
├── config/
│   ├── __init__.py                # Module exports
│   ├── base.py                    # SafeBaseModel (Pydantic v2 base: frozen/extra=forbid/strict + auto-registry)
│   ├── schema.py                  # ExperimentConfig, DataConfig, ModelConfig, TrainConfig, LabelsConfig, ...
│   ├── merge.py                   # deep_merge(), resolve_inheritance(), is_partial_base()
│   │                              #   Supports _base: str | list[str] multi-base composition (v2, Phase 3).
│   │                              #   Preserves all v1 invariants byte-identically.
│   ├── experiment_spec.py         # ExperimentSpec (T14 — config + gate bundle)
│   ├── paths.py                   # resolve_labels_config() — canonical config.data.labels router
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
│   │                              #   _compute_content_hash DELEGATES to hft_contracts.canonical_hash
│   │                              #   SSoT (Phase 6 6B.2 — the inline copy was retired; drift detector:
│   │                              #   tests/test_feature_set_resolver.py::TestCanonicalHashGolden).
│   ├── transforms.py              # FeatureStatistics, BinaryLabelTransform, ComposeTransform
│   ├── normalization.py           # T15 Python-side normalization (HybridNormalizer, GlobalZScoreNormalizer; Welford/Chan streaming)
│   ├── sources.py                 # T12 DataSource multi-source abstraction
│   ├── bundle.py                  # T12 DayBundle multi-source fusion (MBO + BASIC)
│   ├── sample_weights.py          # T10 concurrent-label-overlap weights (de Prado AFML 4.5.1)
│   ├── horizons_resolver.py       # resolve_horizons_from_export() (Phase C.1)
│   └── preprocessing/
│       └── temporal_config.py     # TemporalFeatureConfig factory
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
│   ├── trainer.py                 # Trainer orchestrator (~1.9K L — run `wc -l`), delegates to strategy
│   ├── cv_trainer.py              # T11 CVTrainer (purged k-fold + embargo), FoldResult, CVResults
│   ├── base.py                    # TrainingState + shared training base surface
│   ├── compatibility.py          # CompatibilityContract / fingerprint surface
│   ├── gates.py                   # T14 run_signal_quality_gate + GateResult (pre-training IC gate, hft-rules §13)
│   ├── exceptions.py              # TrainingDivergedError, MonitorMetricUndefined, DegenerateFeatureError (fail-loud, Phase X.3)
│   ├── callbacks.py               # EarlyStopping, ModelCheckpoint, MetricLogger
│   ├── metrics.py                 # MetricsCalculator, ClassificationMetrics
│   ├── regression_metrics.py     # Thin adapter over hft-metrics (R², IC, MAE, RMSE, DA, PA)
│   ├── regression_evaluation.py  # RegressionMetrics dataclass (from_arrays, summary, to_dict)
│   ├── point_return_da.py         # compute_point_return_da_scalars (E8 point-return-DA tripwire producer)
│   ├── label_warnings.py          # warn_if_smoothed_return (E8 run-entry nudge, FINDING-001/008)
│   ├── simple_trainer.py          # SimpleModelTrainer for sklearn-style models (TemporalRidge, GradBoost)
│   ├── loss.py                    # FocalLoss, BinaryFocalLoss, create_focal_loss (classification only)
│   ├── importance/                # Post-training permutation feature-importance (Phase 8C-α)
│   │   ├── permutation.py         # compute_permutation_importance (framework-agnostic)
│   │   ├── callback.py            # PermutationImportanceCallback (emits FeatureImportanceArtifact to hft-ops ledger)
│   │   └── config.py              # ImportanceConfig
│   ├── evaluation.py              # BaselineReport, evaluate_model, full_evaluation
│   └── monitoring.py              # GradientMonitor, TrainingDiagnostics, LRTracker
│
├── analysis/
│   └── stat_rigor/               # Statistical-rigor package (Phase 2 P2.A/P2.C; NPY-safe np.load)
│       ├── ci.py                 # Block-bootstrap CI on 7 test metrics → hft_contracts.TestMetricsCIArtifact
│       └── pairwise.py           # K-way paired moving-block compare + BH-FDR → PairwiseCompareArtifact
│                                 #   CLI drivers: scripts/compute_test_metrics_ci.py + scripts/compare_experiments_pairwise.py
│
├── cli.py                         # CLI entry point: train_command, evaluate_command, apply_overrides
├── ledger_hook.py                 # write_minimal_ledger_record (#PY-223 — delegates to hft_contracts.experiment_recorder SSoT)
│
├── experiments/
│   ├── __init__.py                # Module exports
│   ├── result.py                  # ExperimentResult, ExperimentMetrics
│   └── registry.py                # ExperimentRegistry, create_comparison_table
│
├── calibration/
│   ├── __init__.py              # Prediction calibration package
│   └── variance.py              # calibrate_variance() + VarianceCalibrationConfig + CalibrationResult
│                                #   + CalibrationContext TypedDict (post-hoc variance-match rescaling)
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

scripts/                           # PRODUCTION INFRA (hft-rules §4)
├── train.py                       # Training CLI (writes test_metrics.json)
├── export_signals.py              # Unified signal export CLI (replaces 3 deprecated scripts)
├── precompute_norm_stats.py       # Pre-compute normalization statistics cache
├── validate_export.py             # Dataset validation
├── compute_test_metrics_ci.py     # Block-bootstrap CI driver (→ analysis/stat_rigor/ci.py)
├── compare_experiments_pairwise.py# K-way pairwise-compare driver (→ analysis/stat_rigor/pairwise.py)
├── check_experiment_index_completeness.py  # wiki_consultation soft validator (see CONTRIBUTING.md)
├── _hft_ops_compat.py             # HFT_OPS_ORCHESTRATED=1 deprecation-banner check
├── analysis/                      # Analysis and evaluation scripts (11 files)
│   ├── evaluate_model.py          # Model evaluation from checkpoint
│   ├── run_baseline_evaluation.py # Baseline comparison
│   ├── train_xgboost_baseline.py  # XGBoost training path (xgboost is NOT dispatched by the standard trainer)
│   └── analyze_*/diagnose_*/relabel_*/validate_training_setup.py  # ad-hoc analysis scripts
└── archive/                       # Phase 6 6D fossil archive (NOT templates — see archive/README.md)
    ├── e4_baselines.py, e5_baselines.py, run_simple_training.py
    └── run_simple_model_ablation.py, run_experiment_spec.py

configs/
├── README_configs.md              # Complete config reference
├── bases/                         # 24 axis-partitioned base configs (Phase 3)
│   ├── README.md                  #   4-axis ownership rule + chained inheritance + _partial:
│   ├── models/                    #   5: tlob_compact_bare, tlob_compact_regression,
│   │                              #      tlob_paper_classification, hmhp_cascade_bare,
│   │                              #      hmhp_cascade_regression
│   ├── datasets/                  #   10: per-export normalisation/sampling bases (incl. 2 v3p0
│   │                              #      regression fwd-prices bases: 128feat + 148feat)
│   ├── labels/                    #   4: regression_huber, tlob_smoothed, opportunity,
│   │                              #      triple_barrier_volscaled
│   └── train/                     #   5: regression_default, classification_default,
│                                  #      classification_triple_barrier, tlob_paper_classification_train,
│                                  #      importance_default
├── experiments/                   # 53 YAML configs (run `ls configs/experiments/*.yaml | wc -l`);
│                                  #   ~25 migrated to multi-base _base:, the rest standalone
│                                  #   (baselines, XGBoost, archive, niche HMHP, TLOB singletons).
│                                  #   See MERGE_MIGRATION_PLAN.md.
└── archive/                       # Legacy reference configs (6) — not in Phase 3 migration scope

tests/                             # run `pytest --collect-only -q` for the live count (~2020 across 82 files)
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
    TIME_REGIME = 93         # Categorical {0-6} (7-regime TimeRegime taxonomy, hft_statistics SSoT)
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
# 13 presets (values are Tuple[int, ...]; lengths noted):
FEATURE_PRESETS = {
    "lob_only": PRESET_LOB_ONLY,            # 40 raw LOB features
    "lob_derived": PRESET_LOB_DERIVED,      # LOB + derived (48)
    "full": PRESET_FULL,                    # All 98 features
    "full_98": PRESET_FULL,                 # Alias for "full" (98)
    "full_116": PRESET_FULL_116,            # 116 features
    "full_128": PRESET_FULL_128,            # 128 features
    "analysis_ready_128": PRESET_ANALYSIS_READY_128,  # 119 features
    "signals_core": PRESET_SIGNALS_CORE,    # 8 core signals [84..91]
    "signals_full": PRESET_SIGNALS_FULL,    # 14 signal features
    "lob_signals": PRESET_LOB_SIGNALS,      # LOB + core signals (54)
    "no_meta": PRESET_NO_META,              # Exclude meta 92-97 (92)
    "deeplob_extended": PRESET_DEEPLOB_EXTENDED,  # 52 = 40 raw LOB + 4 derived + 8 core signals
    "short_term_40": PRESET_SHORT_TERM_40,  # 40 (canonical short-term subset)
}

# Usage
from lobtrainer.constants import get_feature_preset, list_presets, describe_preset

indices = get_feature_preset("signals_core")  # [84, 85, 86, 87, 88, 89, 90, 91]
print(list_presets())  # ['lob_only', 'lob_derived', 'full', ...]
describe_preset("signals_core")  # Prints description and indices
```

---

## 5. Configuration System

### Pydantic v2 Architecture (Phase A.5 Scope D v2, 2026-04-24→2026-04-25)

> **All 9 config classes** — `ExperimentConfig`, `DataConfig`, `ModelConfig`, `TrainConfig`, `LabelsConfig`, `SequenceConfig`, `NormalizationConfig`, `SourceConfig`, `CVConfig` — **inherit from `SafeBaseModel`** (at `src/lobtrainer/config/base.py`) which is a Pydantic v2 `BaseModel` subclass with:
>
> ```python
> model_config = ConfigDict(
>     frozen=True,              # Post-construction assignment raises ValidationError
>     extra="forbid",           # Unknown fields raise ValidationError at model_validate
>     strict=True,              # No loose coercion (e.g., "123" → 123)
> )
> ```
>
> **Exactly 3 flags** (`config/base.py:169-173`). `validate_assignment` and `arbitrary_types_allowed` are **deliberately NOT set** — `frozen=True` already blocks assignment, and `PrivateAttr`-typed runtime caches do not require `arbitrary_types_allowed`. (Do not re-add them.)
>
> The class code samples below show the dataclass-era shape for historical reference. In the current codebase every `@dataclass` decorator shown is REPLACED by `class X(SafeBaseModel):` inheritance. Functional semantics match (same field names, defaults, types) with these additions:
>
> 1. **`_canonical_form() -> str` SSoT method** — returns `json.dumps(self.model_dump(mode="json"), sort_keys=True)`. Drives both `__eq__` and `__hash__` to preserve Python's `a == b ⇒ hash(a) == hash(b)` invariant (Phase A.5.7a — pre-A.5.7a `__hash__` used `repr()` which is dict-order-sensitive while `__eq__` compared via `__dict__ ==` which is dict-order-insensitive; silent invariant break for any dict-typed field).
> 2. **Auto-registry via `__pydantic_init_subclass__`** — `SafeBaseModel._registry: ClassVar[List[type]]` auto-populates on every non-`_`-prefixed subclass. `schema.py::_PYDANTIC_CONFIG_CLASSES` is a re-export shim `list(SafeBaseModel._registry)` — NEVER hand-maintain this list. Parametric tests (`TestSafeBaseModelRegistry`, pickle/deepcopy round-trip, partial-base rejection) iterate the auto-registry so future config classes are covered automatically.
> 3. **`ClassVar[...]` discipline on class-level constants** — every non-field class-level attribute (`_VALID_SOURCES`, `_VALID_RETURN_TYPES`, `SCIENTIFIC_NOTATION_PATTERN`, etc.) MUST carry explicit `ClassVar[frozenset | Tuple | re.Pattern | ...]` annotation. Without it, Pydantic v2 treats them as model fields → leaks into `model_dump()` → breaks `CompatibilityContract.fingerprint()` byte-identity (fingerprint drift on every post-migration record; silent ledger corruption).
> 4. **`PrivateAttr` for runtime caches** — `DataConfig._feature_indices_resolved` + `_feature_set_ref_resolved` use `PrivateAttr(default=None)` which permits mutation even under `frozen=True` AND is automatically stripped from `model_dump()` — preserving Phase 4 R3 invariant (YAML round-trip without private-cache leakage) at the type-system layer.
> 5. **In-validator self-mutation pattern** — `@model_validator(mode="after")` that needs to rewrite fields returns `self.model_copy(update={"field": new_value})`. `model_copy(update=...)` documentedly SKIPS validators, which is SAFE here because state is already-validated. **External CLI / user-data paths MUST use `ExperimentConfig.model_validate({**base.model_dump(), **overrides})`** — re-fires every validator on the merged dict. Round-3 adversarial audit caught CLI mutation `config.train.lr = args.lr` (crashes under frozen=True) + `model_copy(update={"train.lr": -1.0})` (silently accepts invalid override) as CRITICAL.
> 6. **Live-YAML corpus regression** — `tests/test_pydantic_migration.py` parametrizes `ExperimentConfig.from_yaml()` over every file under `configs/**/*.yaml` + `tests/fixtures/**/*.yaml`; fails-loud on any `ValidationError` (catches latent typos + unknown fields that pre-Pydantic dacite silently dropped).
> 7. **`_partial: true` strip** — `configs/bases/` partial YAMLs carrying the `_partial: true` sentinel MUST have it stripped by `config/merge.py::resolve_inheritance()` BEFORE `ExperimentConfig.from_dict()` — otherwise `extra="forbid"` rejects. `test_all_partial_bases_rejected_on_direct_load` parametrized over every partial base (all 24 base YAMLs carry `_partial: true`) locks the invariant.
>
> **Dependencies**: `pydantic>=2.7,<3.0` (explicit upper bound); `hatchling>=1.26` (build); `dacite>=1.8` DROPPED. See `/PIPELINE_ARCHITECTURE.md` §11 Configuration System Architecture for the full migration narrative + bug-class-retirement table + CLI override pattern.

### ExperimentConfig (src/lobtrainer/config/schema.py)

Root configuration object (now a `SafeBaseModel` subclass — the dataclass-era shape shown here matches field-for-field):

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
    cv: Optional[CVConfig] = None            # T11 purged k-fold (opt-in)
    importance: Optional[ImportanceConfig] = None  # permutation feature-importance (opt-in)

    output_dir: str = "outputs"
    log_level: str = "INFO"
```

### DataConfig

```python
@dataclass
class DataConfig:
    data_dir: str = "../data/exports/nvda_11month_complete"  # NOTE: this default export is
                                                             # superseded — see §18 "Current Datasets"
    feature_count: int = 98
    horizon_idx: Optional[int] = 0  # LEGACY — `labels` (LabelsConfig) takes precedence when set

    sequence: SequenceConfig = field(default_factory=SequenceConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)

    label_encoding: LabelEncoding = LabelEncoding.CATEGORICAL
    labeling_strategy: LabelingStrategy = LabelingStrategy.TLOB
    num_classes: int = 3
    cache_in_memory: bool = True

    # T9 unified label spec — the CANONICAL carrier of return_type + primary_horizon_idx.
    # None = derive from the legacy labeling_strategy/horizon_idx fields above.
    # Resolved everywhere via config/paths.py::resolve_labels_config(config) → config.data.labels.
    labels: Optional[LabelsConfig] = None
    sources: Optional[SourceConfig] = None       # T12 multi-source fusion (opt-in)

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
class ModelType(str, Enum):        # 12 members
    LOGISTIC = "logistic"
    XGBOOST = "xgboost"      # Registered in lob-models ModelRegistry as 'xgboost_lob', but
                            #   ModelType.XGBOOST → config.name 'xgboost' ≠ the registry key 'xgboost_lob',
                            #   so create_strategy's ModelRegistry.get('xgboost') KeyErrors → NOT reachable
                            #   via the standard create_strategy path — train via scripts/analysis/train_xgboost_baseline.py
    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"  # NOT IMPLEMENTED — reserved for future use
    DEEPLOB = "deeplob"
    TLOB = "tlob"
    MLPLOB = "mlplob"       # MLP-only LOB (Berti & Kasneci 2025); reachable via YAML since Phase X.1 v2
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
class LossType(str, Enum):        # 8 members
    CROSS_ENTROPY = "cross_entropy"
    FOCAL = "focal"
    WEIGHTED_CE = "weighted_ce"
    MSE = "mse"
    HUBER = "huber"
    HETEROSCEDASTIC = "heteroscedastic"
    GMADL = "gmadl"            # Generalized Mean Absolute Directional Loss (regression)
    PINBALL = "pinball"       # Quantile/distributional head (VARIANCE-DL); model must emit [B, Q];
                             #   impl: lobmodels.losses.pinball.PinballLoss

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
    mixed_precision: bool = False  # NOT IMPLEMENTED — mixed_precision=True raises ValueError (fail-fast, hft-rules §5)
    
    # Loss configuration
    loss_type: LossType = LossType.WEIGHTED_CE  # Also: CROSS_ENTROPY, FOCAL, MSE, HUBER, HETEROSCEDASTIC, GMADL, PINBALL
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

24 orthogonal base configs across 4 axes (see `configs/bases/README.md` for the complete ownership matrix and rule):

| Axis | Count | Owns (high-level) | Must NOT set |
|------|-------|---|---|
| `models/` | 5 | `model.model_type`, `model.dropout`, `model.tlob_*`, `model.hmhp_*`, `model.regression_loss_type` | `model.num_classes`, `model.input_size`, `train.task_type`, `train.loss_type`, `train.batch_size` |
| `datasets/` | 10 | `data.data_dir`, `data.feature_count`, `data.normalization`, `data.sequence`, `model.input_size` (T13 auto-derivation) | `data.labeling_strategy`, `data.horizon_idx`, `model.num_classes` |
| `labels/` | 4 | `data.labeling_strategy`, `data.horizon_idx`, `data.num_classes`, `model.num_classes`, `train.task_type`, `train.loss_type` (task-coupled) | `model.*` (other than num_classes), `data.feature_count` |
| `train/` | 5 | `train.batch_size`, `train.epochs`, `train.optimizer`, `train.scheduler`, `train.learning_rate`, `train.weight_decay`, `train.seed`, `train.gradient_clip_norm`, `train.use_class_weights`, `train.focal_gamma` | `train.task_type`, `train.loss_type`, `model.*`, `data.*` |

**Per-child (NOT in any base)**: `name`, `description`, `tags`, `output_dir`, `log_level`.

**Chained-inheritance patterns** (lock-tested by `tests/test_base_axis_ownership.py::TestChainedInheritancePurity`):

1. **TLOB compact**: `tlob_compact_bare` → `tlob_compact_regression` (regression adds `tlob_use_cvml: false` + `tlob_cvml_out_channels: 0` on top of 12 shared arch fields). E4 TLOB uses `bare` directly (keeps its pre-migration golden byte-identical); E5/E6 use the full chain (same byte-identity guarantee).

2. **HMHP cascade**: `hmhp_cascade_bare` → `hmhp_cascade_regression`. `bare` has 10 model fields (`model_type: hmhp` + `dropout` + 8 `hmhp_*` arch fields); `regression` chains from `bare`, overrides `model_type: hmhp_regression`, and adds `hmhp_regression_loss_type: huber`. HMHP classification/triple-barrier use `bare` directly (those regression fields would corrupt their goldens); HMHP regression uses the chain.

**Ownership refinement (Batch 2, 2026-04-15)**: `train.loss_type` moved from `models/` → `labels/` because it is **task-coupled** (regression → huber, tlob → weighted_ce, triple_barrier → focal), not model-coupled. Before this move, HMHP — which shares one cascade model across all three loss types — would have required three near-duplicate HMHP model bases. All migrated configs' resolved dicts are byte-identical before and after the refinement.

**Migration status (2026-04-15 snapshot)**: 25 of the 42 then-in-scope experiment configs migrated to axis-composed form across 3 batches (E4×1 + E5×5 + E6×1 + HMHP×11 + TLOB classif×7 = 25). 17 configs remained standalone by design (baselines×7, XGBoost×2, archive×6, niche HMHP×2). (`configs/experiments/` now holds **53** `*.yaml` total.) Monolith `bases/e5_tlob_regression.yaml` retired at end of Batch 1 after all 5 E5 consumers migrated with byte-identical resolved dicts.

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

### T12 Multi-Source Fusion (`data/sources.py`, `data/bundle.py`)

An alternate load path that fuses features from **multiple exports** (e.g. MBO order book + BASIC off-exchange) into one `DayData`. Activated by `data.sources` (a list of `SourceConfig`) in the config; when `data.sources is None` the standard single-source `load_split_data` path above is used. It is **trainer-wired** — `Trainer.setup()` branches to `load_split_bundles` when `data.sources` is set — but the sections above (`DayData`, single-source loading) cover the default path.

Contract:

- **`DataSource`** (`data/sources.py`): one export dir + a `role`. Exactly one source is `primary` (supplies labels, forward_prices, and the label-computation contract — T9/T10); the rest are `auxiliary` (features only). `DataConfig` validation over the `data.sources` list enforces exactly-one-primary + unique names (`SourceConfig` itself only validates the `role` enum + `feature_count >= 0`).
- **`DayBundle`** (`data/bundle.py`): holds one `SourceDay` per source for a **common date** (dates are intersected across all sources) plus the shared primary labels. `to_fused_day_data(source_order, feature_indices)` concatenates each source's sequences along the **feature axis** → a standard `DayData` `[N, T, F_total]` the Trainer/dataset consume unchanged. Fusion **guards matching window size** (T) and raises `ValueError` on a mismatch; per-source feature selection replaces (and is mutually exclusive with) the single-source `feature_preset` / `feature_indices`.
- **Alignment method** (`_align_sources`): every source is trimmed to the **first `min(N)` sequences** (positional first-N). This is only correct under the **load-bearing assumption that every source shares the same time grid / bin cadence** (60s bins anchored 09:30 ET, `stride=1`), so index *i* is the same wall-clock bin in every source and the N difference sits at the tail (differing label-truncation horizons). See §20 "Multi-source cadence alignment" for the trap. Cross-ref: hft-rules §14 (explicit, tested fusion contract).

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

### Trainer Dispatch: `create_trainer` + `BaseTrainer` Protocol (Phase Q)

Training is **not** a single class. `create_trainer(config)` (`training/trainer.py`) is a **framework-aware factory**: it resolves the model's registered `framework` field (`lobmodels.ModelRegistry.get(name).framework`) and returns one of two concrete trainers —

- `Trainer` (documented below) for `framework="pytorch"` — the strategy-pattern PyTorch path; also the default/fallback when the model name or registry is unresolvable.
- `SimpleModelTrainer` (`training/simple_trainer.py`) for `framework="sklearn"` — the live baseline path for `TemporalRidge` / `TemporalGradBoost`.

On the sklearn branch a `callbacks` kwarg is dropped with an INFO log (sklearn runs a one-shot `fit`; PyTorch callbacks like early-stopping / checkpoint don't apply).

Both concrete trainers satisfy the **`BaseTrainer` Protocol** (`training/base.py`) — a `@runtime_checkable` `typing.Protocol` (structural typing, no forced inheritance; future backends conform without touching `base.py`). Its contract is the lifecycle methods the entry points call polymorphically: `train()`, `evaluate(split)`, `save_checkpoint()`, `load_checkpoint()`, and `export_signals(split, *, output_dir, calibration)` (the last added Phase Q.6.5.B). `create_trainer` returns a `BaseTrainer`-satisfying trainer (per its documented `Returns:` contract), so `scripts/train.py` and the `hft-ops` orchestrator drive one polymorphic surface regardless of framework. (`runtime_checkable` verifies method-name presence only; signature equivalence is the static checker's job.)

`SimpleModelTrainer` is a **full parallel pipeline, not a thin wrapper**: `from_config(config)` bridges the canonical `ExperimentConfig` entry point to its constructor; `setup()` does its own per-day NPY split loading (`_load_split`) and temporal feature engineering (`hft_metrics.TemporalFeatureConfig` / `engineer_features`); it fits the sklearn estimator, evaluates, and emits the **same signal-export format** as `Trainer` so the backtester is unchanged. Checkpoints differ by framework — `Trainer` writes `best.pt` (torch), `SimpleModelTrainer` writes `best.pkl` (pickle + `.config.json` sidecar). It materializes only `temporal_ridge` / `temporal_gradboost`; any other `model_type` raises `ValueError`.

> **XGBoost note:** `XGBoostLOB` is also registered `framework="sklearn"`, so `create_trainer` would route it to `SimpleModelTrainer` — but `SimpleModelTrainer` does not construct it (unknown `model_type` → `ValueError`). The XGBoost baseline is trained via its own driver, `scripts/analysis/train_xgboost_baseline.py`, not through `create_trainer`.

The authoritative docstrings live in `training/base.py`, `training/simple_trainer.py`, and `create_trainer` in `training/trainer.py`.

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

The Trainer (~1.9K L — run `wc -l`) handles the outer loop: epochs, callbacks, scheduling, checkpointing. Zero task-branching remains in the Trainer itself.

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

> **LEGACY tracking schema — dual-schema debt (#PY-384).** `ExperimentResult` / `ExperimentMetrics` / `ExperimentRegistry` are this module's local JSON registry. The **pipeline-canonical** experiment record consumed downstream by `hft-ops` is `hft_contracts.experiment_record.ExperimentRecord` (written via `ledger_hook.write_minimal_ledger_record` → `hft_contracts.experiment_recorder` SSoT). This local registry is slated for retirement (root CLAUDE.md: "Phase 7 6B.1b will retire `lobtrainer.experiments.ExperimentRegistry`"). Prefer `ExperimentRecord` for cross-module work.

```python
@dataclass
class ExperimentMetrics:
    accuracy: float = 0.0
    loss: float = 0.0
    macro_f1: float = 0.0
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    per_class_precision: Dict[str, float] = field(default_factory=dict)
    per_class_recall: Dict[str, float] = field(default_factory=dict)
    per_class_f1: Dict[str, float] = field(default_factory=dict)
    directional_accuracy: float = 0.0
    signal_rate: float = 0.0
    predicted_trade_win_rate: float = 0.0
    decisive_prediction_rate: float = 0.0
    extra_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class ExperimentResult:
    experiment_id: str
    name: str
    description: str = ""
    config: Dict[str, Any] = field(default_factory=dict)

    train_metrics: Optional[ExperimentMetrics] = None
    val_metrics: Optional[ExperimentMetrics] = None
    test_metrics: Optional[ExperimentMetrics] = None

    training_time_seconds: float = 0.0   # (serialized key; NOT "duration_seconds")
    created_at: str = ""                 # ISO timestamp (serialized key; NOT "timestamp")
    checkpoint_path: Optional[str] = None
    # + best_epoch, total_epochs, output_dir, tags, model_type, model_params,
    #   labeling_strategy, num_{train,val,test}_samples, training_history

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
def set_seed(
    seed: int = 42,
    deterministic_cudnn: bool = True,
    *,
    strict_determinism: bool = False,  # DESIGN-1 (see block below)
) -> None:
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

### DESIGN-1 Architectural Patterns (2026-05-10)

Load-bearing reproducibility patterns established by Cycle DESIGN-1 (PyTorch Determinism Contract). See `CHANGELOG.md` under `[Unreleased]` → "DESIGN-1 PyTorch Determinism Contract (2026-05-10)" for the full ship narrative; the patterns below MUST be preserved by any future refactor of the reproducibility surface.

**1. `set_seed` extension contract** (`src/lobtrainer/utils/reproducibility.py::set_seed` around L161-260)

```python
def set_seed(
    seed: int,
    deterministic_cudnn: bool = True,
    *,
    strict_determinism: bool = False,  # DESIGN-1
) -> None:
    if seed >= 2**32:  # NEW-DET-2: numpy legacy RandomState upper bound
        raise ValueError(...)
    ...
    # NEW-C1: warn_only=True by default (loose); opt in via strict_determinism=True
    torch.use_deterministic_algorithms(True, warn_only=not strict_determinism)
```

**2. DataLoader determinism wiring pattern** (`trainer.py::Trainer._build_dataloaders` around L835-844)

```python
DataLoader(
    ...,
    worker_init_fn=create_worker_init_fn(self.config.train.seed),
    generator=torch.Generator().manual_seed(self.config.train.seed),
)
```

CRITICAL: `create_worker_init_fn` MUST use `functools.partial(worker_init_fn, base_seed=base_seed)` — closures are unpicklable under Python 3.14+ forkserver multiprocessing. Returning a closure here is a latent bug that breaks `num_workers > 0` paths on newer Python.

**3. `RngStatePolicy` enum** (`reproducibility.py:35-90,185-300`)

```python
class RngStatePolicy(str, Enum):
    STRICT = "strict"      # Raise RngStatePolicyError on missing keys
    GRACEFUL = "graceful"  # WARN + reseed from fallback_seed (DEFAULT)
    IGNORE = "ignore"      # No-op (cross-platform CI replays)

RNG_STATE_SCHEMA_VERSION = 1  # forward-compat versioning
```

`get_seed_state()` returns v1-versioned dict: `{schema_version, python, numpy, torch, [torch_cuda], [torch_mps]}`. CUDA + MPS captures are best-effort try/except (graceful degradation across backends).

**4. Option-C ordering pattern for `_pending_rng_state`** (`trainer.py:280-296,850-875`)

`load_checkpoint` does NOT immediately restore RNG state — it captures `_pending_rng_state` + `_rng_policy` attrs which `setup()` consumes AFTER calling `set_seed()`:

```python
def setup(self):
    set_seed(self.config.train.seed, strict_determinism=...)
    # DESIGN-1 Option-C: restore AFTER set_seed, BEFORE DataLoader workers spawn (lazy on iter)
    if getattr(self, '_pending_rng_state', None) is not None:
        set_seed_state(
            self._pending_rng_state,
            policy=self._rng_policy,
            fallback_seed=self.config.train.seed,
        )
        self._pending_rng_state = None  # Consumed-once invariant
```

This ordering guarantees workers observe the restored state, not the post-`set_seed` state. The `_pending_rng_state` attribute is consumed-once; nulling after use prevents accidental re-application.

**5. CRITICAL fingerprint invariant — `rng_state` MUST NOT enter `compatibility_fingerprint` or `model_config_hash`**

Construction-order guarantee: `compatibility_fingerprint` is computed in `trainer.py::Trainer._build_checkpoint_dict` (around L1338) BEFORE `rng_state` is added to the checkpoint dict (around L1353) (cannot include something not yet in scope). Defensively, `hft-ops/.../dedup.py::compute_fingerprint::exclude_keys` strip-set excludes `rng_state` + `callback_state`. Locked by 3 regression tests:

- `tests/test_training_compatibility.py::TestRngStateInvariance` — mutates `config.train.seed`, asserts fingerprint unchanged.
- `hft-ops/tests/test_dedup.py::test_rng_state_excluded_from_fingerprint` — injects `rng_state` at 3 nest levels in synthetic manifest.
- `lob-models/tests/integration/test_phase0_forward_pass.py:33-96` parametrized over `warn_only={True, False}`.

Future refactors that admit RNG state into fingerprint composition silently break experiment dedup + ledger-conflation (Phase-3-§3.3b class of bug).

**6. Sklearn-RNG-FREE invariant (Phase A.3 REDESIGN)**

Sklearn models (`TemporalRidge`, `TemporalGradBoost`) do NOT participate in rng_state load-bearing — verified via Agent X ground-truth: `Ridge` is closed-form (no `random_state` argument); `TemporalGradBoost` uses `random_state=42` hardcoded + hermetic local `np.random.RandomState(42)`. Capturing global RNG buys ZERO determinism on the predict path.

The sklearn sidecar `<path>.config.json` carries `rng_state_python` + `rng_state_numpy` keys for symmetry with the PyTorch path, but these fields are observational (not load-bearing). Documented in `simple_trainer.save_checkpoint` docstring. Lock test: `TestSklearnRngIndependence` (3 tests) perturbs global RNG between save and predict, asserts bit-identical output.

**7. INDEX_SCHEMA_VERSION NO BUMP for DESIGN-1**

Per V2 verdict over Wave 3 D3's claim: `rng_state` lives in the **checkpoint** (`.pt` binary), NOT in the ledger `ExperimentRecord` or its `index_entry()` projection. Schema additions are surfaced at the checkpoint-dict layer only; ledger filtering surfaces are unaffected. The hft-ops `dedup.py::exclude_keys` defensive strip ensures `rng_state` cannot leak into `compute_fingerprint` even if a future record were to harvest it. Do NOT bump `INDEX_SCHEMA_VERSION` for purely-checkpoint-scope additions.

### Phase Y Composer Healthy-State Preserve List (2026-05-13)

Cross-cycle validation finding from the comprehensive 2026-05-13 audit (16 parallel adversarial agents across Wave 1 + Wave 2). The following invariants are verified HEALTHY and MUST be preserved by any refactor of the Phase Y composer / atomic-write paths / canonical-hash SSoT consumers:

| # | Invariant | Evidence |
|---|---|---|
| 1 | Phase Y composer END-TO-END BIT-EXACT | Wave 1 Agent 2: 69/70 populated records reproduced via INDEPENDENT pure-stdlib reimplementation of `experiment_record.py:823-919`; 1 None==None legitimate (failed run) |
| 2 | Cross-cycle BIT-EXACT reproducibility | R-16a ridge×point arm 1 `experiment_provenance_hash=901c25dd...` matches cycle5 ridge×point post-#PY-88 baseline BIT-IDENTICAL across all 4 components (`data_export_fp`, `feature_set_content_hash`, `compatibility_fp`, `model_config_hash`) |
| 3 | `canonical_json_blob` frozen + 4 golden tests | `hft_contracts.canonical_hash.canonical_json_blob` SSoT is locked by 4 golden-hash tests; consumed by trainer `_compute_content_hash`, `compute_feature_set_hash`, `compute_label_strategy_hash`, `compute_model_config_hash` |
| 4 | `atomic_write_json` discipline universal | All experiment/signal/ledger JSON artifacts use `hft_contracts.atomic_io.atomic_write_json` SSoT (verified at 5 sampled migration sites incl. `simple_trainer.py:765-866`, `exporter.py:502-534`, hft-ops `extraction_cache.py:323,661,691,714`, lob-backtester `registry.py:123`, callbacks.py:683-694) |
| 5 | Class A/B SSoT primitives all in canonical-form usage | `canonical_hash`, `atomic_io.{atomic_write_json,binary,torch,npy,pickle,copy}`, `compatibility.CompatibilityContract`, `experiment_record.INDEX_SCHEMA_VERSION` |
| 6 | Schema versions synchronized 3.0 | `pipeline_contract.toml::schema_version="3.0"` ↔ `hft_contracts.SCHEMA_VERSION="3.0"` ↔ Rust `SchemaVersion::CURRENT={3,0}` |
| 7 | INDEX_SCHEMA_VERSION 1.6.0 | `experiment_record.py:68` matches root CLAUDE.md banner; `model_config_hash` top-level projection at `experiment_record.py:659-674` |
| 8 | CR6 LabelFactory parity locked rtol=1e-12 | `hft-contracts/tests/test_label_factory_parity.py:38` |
| 9 | sklearn arms RNG-FREE | `temporal_ridge.py:102` Ridge closed-form (no `random_state`); `temporal_gradboost.py:116` `random_state=42` hardcoded |
| 10 | DESIGN-1 bit-exact replay test alive (Phase G-4) | `tests/test_bit_exact_replay.py` uses `torch.testing.assert_close(rtol=0, atol=0)` walking model_state_dict |

Future refactors of the experiment-provenance surface (Phase Y composer, canonical-hash, atomic-write helpers, fingerprint exclude_keys, INDEX_SCHEMA_VERSION) MUST consult this list before deciding to modify any of the load-bearing invariants. Each invariant is locked by a regression test; breaking the test is the early-warning signal for cross-pipeline drift.

**See**: DESIGN-1 source spec preserved at `.archive/2026-05-15-doc-audit-phase1/CYCLE_DESIGN-1_AUTHORIZED_SPEC_2026_05_10.md` (full §6 anti-drift list + §3 finding-closure matrix + §7 file-by-file touch list); comprehensive validation findings at `COMPREHENSIVE_VALIDATION_2026_05_13.md` §7 (full 20-item PRESERVE list).

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
| `scripts/validate_export.py` | Validate exported dataset integrity |
| `scripts/compute_test_metrics_ci.py` | Block-bootstrap CI driver (→ `analysis/stat_rigor/ci.py`) |
| `scripts/compare_experiments_pairwise.py` | K-way pairwise compare + BH-FDR driver (→ `analysis/stat_rigor/pairwise.py`) |
| `scripts/check_experiment_index_completeness.py` | `wiki_consultation:` soft validator (CONTRIBUTING.md) |
| `scripts/analysis/evaluate_model.py` | Evaluate trained model checkpoint |
| `scripts/analysis/run_baseline_evaluation.py` | Compare against naive baselines |
| `scripts/analysis/train_xgboost_baseline.py` | XGBoost training path (not dispatched by the standard trainer) |
| `scripts/archive/run_simple_training.py` (fossil) | Archived — SimpleModelTrainer CLI; NOT a template (Phase 6 6D) |

---

## 18. Configuration Reference

See `configs/README_configs.md` for complete configuration reference including:

- Active experiment configs (**53** under `configs/experiments/*.yaml`; ~25 migrated to axis-composed `_base: [...]` form + the rest standalone — see `MERGE_MIGRATION_PLAN.md`)
- Axis-partitioned bases (**24** under `configs/bases/{models=5,datasets=10,labels=4,train=5}/` — see `configs/bases/README.md`)
- Archived reference configs (6 legacy, not in Phase 3 migration scope)
- Horizon index mapping
- Model type options
- Loss function selection
- Configuration template

### Current Datasets

> **⚠️ STALE (2026-07):** `nvda_11month_complete` (the `DataConfig.data_dir` default) is **no longer on disk** — it was superseded by the `e5_timebased_{5s,30s,60s}_v3p0` + `nvda_xnas_128feat_regression_fwd_prices_v3p0` baselines (present under `../data/exports/`; see root `CLAUDE.md` §Dataset Specification). The rows below are retained as the historical label/horizon reference.

| Dataset | Days | Labels | Horizons |
|---------|------|--------|----------|
| `nvda_11month_complete` (legacy) | **234** | TLOB | [10, 20, 50, 100] |
| `nvda_11month_triple_barrier` (legacy) | **234** | Triple Barrier | [50, 100, 200] |

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

### Test Modules (82 `tests/test_*.py` files; run `pytest --collect-only -q` for the live count, ~2020 tests — hft-rules §11)

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

Index 93 (TIME_REGIME) should be excluded from normalization - it's categorical `{0..6}` (7-regime TimeRegime taxonomy; `hft_statistics::time::regime` SSoT). (Excluded from normalization regardless of cardinality.)

### Multi-source cadence alignment (T12 — silent-misalignment hazard)

The T12 fusion path (§6) aligns sources by **positional first-N** (`_align_sources` trims to the first `min(N)` sequences). It **guards matching window size (T) but has no bin-cadence guard**: two sources at different bin cadences (e.g. a 60s export fused with a 30s one) that happen to share the same T concatenate **silently wrong** — index *i* no longer maps to the same wall-clock bin, and no error is raised. The alignment correctness rests entirely on the undocumented-in-data assumption that all sources use the same 60s/09:30-ET/`stride=1` grid. This path is trainer-wired but **not exercised in production** (root `CLAUDE.md`: A7c alignment audit open). Before activating fusion across heterogeneous exports, verify equal cadence out-of-band. Cross-ref: hft-rules §14 (a fusion contract must make its alignment method explicit and tested).

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

*Last updated: 2026-07-07 (Phase-2 TRUTH doc-drift pass; prior major revision: April 25, 2026 — Phase A.5 Scope D v2 Pydantic v2 migration)*
*Version: 0.7.0*
