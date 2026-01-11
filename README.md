# LOB Model Trainer

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-orange.svg)](https://pytorch.org/)

Machine learning experimentation for limit order book price prediction.

**Version**: 0.3.0 | **Schema**: 2.1

## What's New in v0.3.0

- **Strategy-Aware Metrics**: `MetricsCalculator` understands labeling strategy semantics (TLOB, Triple Barrier, Opportunity)
- **Focal Loss**: Handle class imbalance with `FocalLoss` and `BinaryFocalLoss`
- **TLOB Model Support**: Transformer LOB model via `lob-models` integration
- **Task/Loss Configuration**: New `task_type`, `loss_type`, `labeling_strategy` config fields

## Quick Start

```bash
# Install using uv (recommended)
uv venv && uv pip install -e ".[dev]"

# Or using pip
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Train a model
python scripts/train.py --config configs/baseline_lstm.yaml

# Run statistical analysis
python scripts/run_complete_streaming_analysis.py \
    --data-dir ../data/exports/nvda_balanced \
    --symbol NVDA
```

## Overview

This package provides tools for training and evaluating ML models on LOB feature data exported from the `feature-extractor-MBO-LOB` Rust pipeline.

### Data Contract (Schema v2.1)

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

## Installation

```bash
cd lob-model-trainer
pip install -e ".[dev]"
```

## Training a Model

### 1. Load Data (Aligned Format)

```python
from lobtrainer.data import load_split_data, LOBSequenceDataset

# Load training data (aligned format: *_sequences.npy)
train_days = load_split_data("../data/exports/nvda_balanced", "train")
print(f"Loaded {len(train_days)} training days")
print(f"Total sequences: {sum(d.num_sequences for d in train_days)}")

# Multi-horizon support
if train_days[0].is_multi_horizon:
    print(f"Horizons: {train_days[0].horizons}")  # e.g., [10, 20, 50, 100, 200]
```

### 2. Create Dataset and DataLoader

```python
from lobtrainer.data import LOBSequenceDataset, create_dataloaders

# Option A: Direct dataset creation (aligned format)
dataset = LOBSequenceDataset(
    days=train_days,
    horizon_idx=0,  # First horizon (H=10)
    transform=normalizer,  # Optional
)

# Option B: Automatic setup with dataloaders
loaders = create_dataloaders(
    data_dir="../data/exports/nvda_balanced",
    batch_size=128,
    horizon_idx=0,  # Which horizon for multi-horizon labels
)

# Access loaders
train_loader = loaders['train']
val_loader = loaders['val']
```

### 3. Train with Trainer Class

```python
from lobtrainer import create_trainer, set_seed
from lobtrainer.training import EarlyStopping, ModelCheckpoint

# Set seed for reproducibility
set_seed(42)

# Create trainer from config
trainer = create_trainer(
    "configs/baseline_lstm.yaml",
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

### 4. Access Features by Index

```python
from lobtrainer.constants import FeatureIndex, FEATURE_COUNT

# Access specific features
ofi = features[:, FeatureIndex.TRUE_OFI]           # 84
microprice = features[:, FeatureIndex.WEIGHTED_MID_PRICE]  # 46
book_valid = features[:, FeatureIndex.BOOK_VALID]  # 92

# Safety check before using signals
valid_mask = book_valid > 0.5
ofi_valid = ofi[valid_mask]
```

## Configuration

All experiments are configured via YAML files:

```yaml
# configs/baseline_lstm.yaml
name: baseline_lstm
description: "Baseline LSTM for LOB price prediction"

data:
  data_dir: "../data/exports/nvda_balanced"
  feature_count: 98
  horizon_idx: 0  # Which horizon for multi-horizon labels
  
  normalization:
    strategy: zscore_per_day
    exclude_features: [93]  # TIME_REGIME (categorical)
  
  num_classes: 3
  cache_in_memory: true

model:
  model_type: lstm
  input_size: 98
  hidden_size: 64
  num_layers: 2
  dropout: 0.2
  lstm_bidirectional: false
  lstm_attention: false

train:
  batch_size: 128
  learning_rate: 1.0e-3
  epochs: 100
  early_stopping_patience: 10
  use_class_weights: true  # Handle class imbalance
  seed: 42

output_dir: outputs/baseline_lstm
```

See `configs/` for more examples:
- `deeplob_benchmark.yaml` - DeepLOB benchmark mode (Zhang et al. 2019)
- `deeplob_benchmark_h100.yaml` - DeepLOB with horizon=100 (paper setting)
- `experiments/nvda_h10_weighted_v1.yaml` - Experiment config for NVDA h=10
- `archive/baseline_lstm.yaml` - Legacy LSTM baseline (archived)

## Key Modules

### `lobtrainer.constants`

Feature index mapping (Schema v2.1):

```python
from lobtrainer.constants import (
    FeatureIndex, SignalIndex, FEATURE_COUNT, SCHEMA_VERSION,
    LABEL_DOWN, LABEL_STABLE, LABEL_UP,
    SHIFTED_LABEL_NAMES,  # For PyTorch labels {0, 1, 2}
    LOB_ASK_PRICES, LOB_ASK_SIZES, LOB_BID_PRICES, LOB_BID_SIZES,
)

assert FEATURE_COUNT == 98
assert SCHEMA_VERSION == 2.1
assert FeatureIndex.TRUE_OFI == 84
assert FeatureIndex.BOOK_VALID == 92
```

### `lobtrainer.data`

Data loading and PyTorch datasets:

```python
from lobtrainer.data import (
    DayData,              # Container for one day's data
    LOBSequenceDataset,   # For LSTM/Transformer (requires aligned format)
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
    create_model,      # Factory function from config
    LOBMODELS_AVAILABLE,  # True if lob-models package installed (for DeepLOB)
)

# DeepLOB via create_model (requires lob-models package)
from lobtrainer.config import ModelType, DeepLOBMode
model = create_model(ModelConfig(
    model_type=ModelType.DEEPLOB,
    deeplob_mode=DeepLOBMode.BENCHMARK,
))
```

### `lobtrainer.training`

Training infrastructure:

```python
from lobtrainer.training import (
    Trainer,                      # Main training class
    EarlyStopping,                # Stop when metric plateaus
    ModelCheckpoint,              # Save best model
    MetricLogger,                 # Log metrics to file
    compute_classification_report,   # Detailed metrics
    compute_trading_metrics,         # Trading-specific metrics
    compute_transition_accuracy,     # Accuracy on label transitions
)
```

### `lobtrainer.config`

Configuration management:

```python
from lobtrainer.config import (
    ExperimentConfig, DataConfig, ModelConfig, TrainConfig,
    load_config, save_config,
)
```

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

As of Schema v2.1, **all directional features follow standard convention**:
- `> 0` = BULLISH (buy pressure)
- `< 0` = BEARISH (sell pressure)

**Exception**: `PRICE_IMPACT` (index 47) is unsigned - cannot determine direction.

> **Note**: Previous workarounds like `OPPOSITE_SIGN_FEATURES` and `get_corrected_net_trade_flow()` have been removed. Sign conventions are now fixed in the Rust preprocessing pipeline.

### Safety Gates

Always check before using signals:

```python
# Gate 1: Book validity
if features[i, FeatureIndex.BOOK_VALID] < 0.5:
    continue  # Skip invalid sample

# Gate 2: MBO warmup
if features[i, FeatureIndex.MBO_READY] < 0.5:
    continue  # MBO features not ready

# Gate 3: Feed quality
if features[i, FeatureIndex.INVALIDITY_DELTA] > 0:
    continue  # Feed had problems
```

### Aligned vs Legacy Format

This library supports two export formats from the Rust pipeline:

| Format | Files | Shape | Description |
|--------|-------|-------|-------------|
| **Aligned** (recommended) | `*_sequences.npy` | `(N_seq, 100, 98)` | Pre-aligned sequences, 1:1 with labels |
| Legacy | `*_features.npy` | `(N_samples, 98)` | Flat features, requires manual alignment |

`LOBSequenceDataset` **requires** aligned format. For legacy format, use `LOBFlatDataset`.

### Multi-Horizon Labels

When using multi-horizon exports (`horizons: [10, 20, 50, 100, 200]`):

```python
# Check if multi-horizon
day.is_multi_horizon  # True
day.num_horizons      # 5
day.horizons          # [10, 20, 50, 100, 200]

# Get labels for specific horizon
labels_h10 = day.get_labels(0)   # Horizon 10
labels_h20 = day.get_labels(1)   # Horizon 20
labels_all = day.get_labels(None)  # All horizons (N, 5)
```

## Running Tests

```bash
cd lob-model-trainer
pytest tests/ -v

# Run specific test module
pytest tests/test_trainer.py -v
```

## Directory Structure

```
lob-model-trainer/
├── src/lobtrainer/
│   ├── constants/       # Feature indices, label encoding (Schema v2.1)
│   ├── config/          # Configuration schema (dataclasses)
│   ├── data/            # Dataset classes, transforms
│   ├── models/          # LSTM, GRU, baselines
│   ├── training/        # Trainer, callbacks, metrics, evaluation
│   ├── analysis/        # Streaming analysis modules
│   └── utils/           # Reproducibility utilities
├── scripts/
│   ├── train.py                         # Training CLI
│   ├── evaluate_model.py                # Evaluation CLI
│   └── run_complete_streaming_analysis.py  # Full dataset analysis
├── configs/             # YAML configuration files
├── tests/               # Unit and integration tests
└── docs/                # Additional documentation
```

## Related Documentation

- `CODEBASE.md` - Comprehensive technical reference
- `docs/ANALYSIS_MODULES_REFERENCE.md` - Analysis module documentation
- `../feature-extractor-MBO-LOB/docs/full-data-pipeline.md` - End-to-end pipeline

## Version History

| Version | Schema | Changes |
|---------|--------|---------|
| 0.3.0 | 2.1 | Strategy-aware metrics, Focal Loss, TLOB support, LabelingStrategy config |
| 0.2.0 | 2.1 | Training infrastructure, multi-horizon support, sign convention fixes |
| 0.1.0 | 2.0 | Initial release with analysis modules |

---

*Last updated: January 11, 2026*
