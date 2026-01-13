# LOB Model Trainer

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-orange.svg)](https://pytorch.org/)

Machine learning experimentation for limit order book price prediction.

**Version**: 0.4.0 | **Schema**: 2.1

---

## Overview

This package provides tools for training and evaluating ML models on LOB feature data exported from the `feature-extractor-MBO-LOB` Rust pipeline. It is designed as a **training-focused library** - for dataset analysis, use `lob-dataset-analyzer`.

### Key Features

- **Multiple Model Architectures**: LSTM, GRU, DeepLOB, TLOB (via `lob-models`)
- **Strategy-Aware Metrics**: Metrics that understand TLOB, Triple Barrier, Opportunity labeling
- **Advanced Monitoring**: Gradient tracking, learning rate monitoring, training diagnostics
- **Experiment Tracking**: Structured comparison across experiments
- **Feature Presets**: Named feature subsets for easy configuration
- **Focal Loss**: Handle class imbalance effectively

---

## Quick Start

```bash
# Install using uv (recommended)
cd lob-model-trainer
uv venv && uv pip install -e ".[dev]"

# Or using pip
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Train a model
python scripts/train.py --config configs/experiments/nvda_tlob_h10_v1.yaml
```

---

## Data Contract (Schema v2.1)

| Property | Value |
|----------|-------|
| Features | 98 per timestep |
| Sequence Layout | `(N_seq, 100, 98)` - pre-aligned |
| Labels (original) | `{-1, 0, 1}` = Down, Stable, Up |
| Labels (PyTorch) | `{0, 1, 2}` = Down, Stable, Up (shifted +1) |
| Multi-Horizon | Optional `(N_seq, H)` for multiple horizons |

### Feature Categories

| Range | Count | Category |
|-------|-------|----------|
| 0-39 | 40 | Raw LOB (10 levels × 4 values) |
| 40-47 | 8 | Derived (spread, microprice, etc.) |
| 48-83 | 36 | MBO (order flow, queue stats) |
| 84-97 | 14 | Trading Signals (OFI, asymmetry, regime) |

---

## Directory Structure

```
lob-model-trainer/
├── src/lobtrainer/
│   ├── __init__.py              # Public API (v0.4.0)
│   ├── constants/               # Feature indices, label encoding, presets
│   │   ├── feature_index.py     # FeatureIndex, SignalIndex (98 features)
│   │   └── feature_presets.py   # Named feature subsets
│   ├── config/                  # Configuration schema
│   │   └── schema.py            # ExperimentConfig, DataConfig, ModelConfig
│   ├── data/                    # Dataset classes, transforms
│   │   ├── dataset.py           # DayData, LOBSequenceDataset, LOBFlatDataset
│   │   └── transforms.py        # ZScoreNormalizer, BinaryLabelTransform
│   ├── models/                  # Model implementations
│   │   ├── lstm.py              # LSTMClassifier, GRUClassifier
│   │   └── baselines.py         # NaiveClassPrior, NaivePreviousLabel, LogisticBaseline
│   ├── training/                # Training infrastructure
│   │   ├── trainer.py           # Trainer class, training loop
│   │   ├── callbacks.py         # EarlyStopping, ModelCheckpoint, MetricLogger
│   │   ├── metrics.py           # MetricsCalculator, ClassificationMetrics
│   │   ├── loss.py              # FocalLoss, BinaryFocalLoss
│   │   ├── evaluation.py        # BaselineReport, evaluate_model
│   │   └── monitoring.py        # GradientMonitor, TrainingDiagnostics
│   ├── experiments/             # Experiment tracking
│   │   ├── result.py            # ExperimentResult, ExperimentMetrics
│   │   └── registry.py          # ExperimentRegistry, create_comparison_table
│   └── utils/                   # Utilities
│       └── reproducibility.py   # set_seed, SeedManager
├── scripts/
│   ├── train.py                 # Training CLI
│   ├── evaluate_model.py        # Model evaluation CLI
│   ├── run_baseline_evaluation.py  # Baseline comparison
│   └── validate_export.py       # Dataset validation
├── configs/
│   ├── README_configs.md        # Complete config reference
│   ├── experiments/             # Active experiment configs (3)
│   └── archive/                 # Reference configs (6)
└── tests/                       # 14 test modules
```

---

## Training a Model

### 1. Load Data

```python
from lobtrainer.data import load_split_data, LOBSequenceDataset

# Load training data (aligned format: *_sequences.npy)
train_days = load_split_data("../data/exports/nvda_11month_complete", "train")
print(f"Loaded {len(train_days)} training days")
print(f"Total sequences: {sum(d.num_sequences for d in train_days)}")

# Multi-horizon support
if train_days[0].is_multi_horizon:
    print(f"Horizons: {train_days[0].horizons}")  # e.g., [10, 20, 50, 100]
```

### 2. Create DataLoader

```python
from lobtrainer.data import create_dataloaders

loaders = create_dataloaders(
    data_dir="../data/exports/nvda_11month_complete",
    batch_size=64,
    horizon_idx=0,  # H=10 (first horizon)
)

train_loader = loaders['train']
val_loader = loaders['val']
```

### 3. Train with Trainer

```python
from lobtrainer import create_trainer, set_seed
from lobtrainer.training import EarlyStopping, ModelCheckpoint

# Set seed for reproducibility
set_seed(42)

# Create trainer from config
trainer = create_trainer(
    "configs/experiments/nvda_tlob_h10_v1.yaml",
    callbacks=[
        EarlyStopping(patience=10),
        ModelCheckpoint(save_dir="checkpoints/"),
    ]
)

# Train
result = trainer.train()
print(f"Best val loss: {result['best_val_metric']:.4f} at epoch {result['best_epoch']}")

# Evaluate on test set
metrics = trainer.evaluate("test")
print(metrics.summary())
```

---

## Configuration

All experiments use YAML configuration. See `configs/README_configs.md` for complete reference.

### Active Experiments (3)

| Config | Model | Horizon | Labels | Purpose |
|--------|-------|---------|--------|---------|
| `nvda_tlob_h10_v1.yaml` | TLOB | H=10 | TLOB | Short-term (~1s) |
| `nvda_tlob_h100_v1.yaml` | TLOB | H=100 | TLOB | Paper benchmark (~10s) |
| `nvda_tlob_triple_barrier_11mo_v1.yaml` | TLOB | H=50 | Triple Barrier | Risk-managed trading |

### Archived Reference Configs (6)

| Config | Model | Unique Value |
|--------|-------|--------------|
| `baseline_lstm.yaml` | LSTM | Pure LSTM reference |
| `lstm_attn_bidir_h20.yaml` | LSTM+Attn | Attention + bidirectional |
| `deeplob_benchmark.yaml` | DeepLOB | Zhang et al. 2019 architecture |
| `nvda_bigmove_opportunity_v1.yaml` | DeepLOB | Opportunity labeling |
| `nvda_tlob_bigmove_v1.yaml` | TLOB | TLOB + Opportunity |
| `nvda_tlob_binary_signal_v1.yaml` | TLOB | Binary + Focal Loss |

---

## Key Modules

### `lobtrainer.constants`

Feature index mapping (Schema v2.1):

```python
from lobtrainer.constants import (
    FeatureIndex, SignalIndex, FEATURE_COUNT, SCHEMA_VERSION,
    SHIFTED_LABEL_NAMES,
    get_feature_preset, list_presets,
)

assert FEATURE_COUNT == 98
assert SCHEMA_VERSION == 2.1
assert FeatureIndex.TRUE_OFI == 84

# Feature presets
indices = get_feature_preset("signals_core")  # Core 8 signals
print(list_presets())  # ['lob_only', 'full', 'signals_core', ...]
```

### `lobtrainer.data`

Data loading and PyTorch datasets:

```python
from lobtrainer.data import (
    DayData,              # Container for one day's data
    LOBSequenceDataset,   # For LSTM/Transformer
    LOBFlatDataset,       # For MLP/XGBoost
    load_split_data,      # Load all days in a split
    create_dataloaders,   # Create train/val/test loaders
)
```

### `lobtrainer.models`

Model implementations:

```python
from lobtrainer.models import (
    LSTMClassifier,    # LSTM with optional attention/bidirectional
    GRUClassifier,     # GRU variant
    LogisticBaseline,  # Logistic regression baseline
    create_model,      # Factory function from config
    LOBMODELS_AVAILABLE,  # True if lob-models installed
)

# DeepLOB/TLOB via create_model (requires lob-models package)
from lobtrainer.config import ModelConfig, ModelType
model = create_model(ModelConfig(model_type=ModelType.TLOB))
```

### `lobtrainer.training`

Training infrastructure:

```python
from lobtrainer.training import (
    # Core
    Trainer, create_trainer,
    # Callbacks
    EarlyStopping, ModelCheckpoint, MetricLogger, ProgressCallback,
    # Metrics
    MetricsCalculator, ClassificationMetrics,
    compute_classification_report, compute_trading_metrics,
    # Evaluation
    evaluate_model, create_baseline_report, BaselineReport,
    # Loss
    FocalLoss, BinaryFocalLoss, create_focal_loss,
    # Monitoring
    GradientMonitor, LearningRateTracker, TrainingDiagnostics,
    create_standard_monitoring,
)
```

### `lobtrainer.experiments`

Experiment tracking:

```python
from lobtrainer.experiments import (
    ExperimentResult,
    ExperimentMetrics,
    ExperimentRegistry,
    create_comparison_table,
)

# Register experiment
registry = ExperimentRegistry("outputs/experiments")
registry.register(result)

# Compare experiments
table = create_comparison_table(registry, metric_keys=['macro_f1'])
```

---

## Critical Notes

### Label Encoding

Labels are shifted for PyTorch compatibility:

| Original | Shifted | Meaning |
|----------|---------|---------|
| -1 | 0 | Down (price decreased) |
| 0 | 1 | Stable (within threshold) |
| +1 | 2 | Up (price increased) |

The shift happens automatically in `LOBSequenceDataset.__getitem__()`.

### Sign Conventions (Schema v2.1)

All directional features follow standard convention:
- `> 0` = BULLISH (buy pressure)
- `< 0` = BEARISH (sell pressure)

**Exception**: `PRICE_IMPACT` (index 47) is unsigned.

### Safety Gates

Always check before using signals:

```python
from lobtrainer.constants import FeatureIndex

# Gate 1: Book validity
if features[i, FeatureIndex.BOOK_VALID] < 0.5:
    continue  # Skip invalid sample

# Gate 2: MBO warmup
if features[i, FeatureIndex.MBO_READY] < 0.5:
    continue  # MBO features not ready
```

---

## Running Tests

```bash
cd lob-model-trainer
pytest tests/ -v

# Run specific test
pytest tests/test_trainer.py -v
```

**Test Coverage**: 14 test modules covering all core functionality.

---

## Scripts

| Script | Purpose |
|--------|---------|
| `train.py` | Training CLI entry point |
| `evaluate_model.py` | Model evaluation |
| `run_baseline_evaluation.py` | Baseline comparison |
| `validate_export.py` | Validate dataset export |

```bash
# Train with config
python scripts/train.py --config configs/experiments/nvda_tlob_h10_v1.yaml

# Evaluate model
python scripts/evaluate_model.py --checkpoint outputs/best.pt

# Run baseline comparison
python scripts/run_baseline_evaluation.py --data-dir ../data/exports/nvda_11month_complete
```

---

## Related Libraries

| Library | Purpose |
|---------|---------|
| `lob-dataset-analyzer` | Dataset analysis and statistics |
| `lob-models` | Neural network architectures (DeepLOB, TLOB) |
| `feature-extractor-MBO-LOB` | Rust pipeline for feature extraction |
| `MBO-LOB-reconstructor` | LOB state reconstruction from MBO data |

---

## Documentation

- `CODEBASE.md` - Comprehensive technical reference for this library
- `configs/README_configs.md` - Complete configuration reference

---

## Version History

| Version | Schema | Changes |
|---------|--------|---------|
| **0.4.0** | 2.1 | Monitoring callbacks, experiment tracking, feature presets, config cleanup |
| 0.3.0 | 2.1 | Strategy-aware metrics, Focal Loss, TLOB support |
| 0.2.0 | 2.1 | Training infrastructure, multi-horizon support |
| 0.1.0 | 2.0 | Initial release |

---

*Last updated: January 13, 2026*
