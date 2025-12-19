# LOB Model Trainer

## Quick Start

```bash
# Install using uv (recommended)
uv venv && uv pip install -e ".[dev]"

# Or using pip
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run signal analysis
python scripts/run_signal_analysis.py
```

Machine learning experimentation for limit order book price prediction.

## Overview

This package provides tools for training and evaluating ML models on LOB feature data exported from the `feature-extractor-MBO-LOB` Rust pipeline.

### Data Contract

| Property | Value |
|----------|-------|
| Features | 98 per sample |
| Layout | `(N, 98)` flat or `(N, T, 98)` sequences |
| Dtype | `float64` |
| Labels | `{0, 1, 2}` = Down, Stable, Up |

### Feature Categories

| Range | Count | Category |
|-------|-------|----------|
| 0-39 | 40 | Raw LOB (10 levels × 4 values) |
| 40-47 | 8 | Derived (spread, microprice, etc.) |
| 48-83 | 36 | MBO (order flow, queue stats) |
| 84-97 | 14 | Trading Signals (OFI, asymmetry, regime) |

See `plan/03-FEATURE-INDEX-MAP-v2.md` for complete index mapping.

## Installation

```bash
cd lob-model-trainer
pip install -e ".[dev]"
```

## Quick Start

### 1. Verify Data Access

```python
from lobtrainer.data import load_split_data

# Load training data
train_days = load_split_data("../data/exports/nvda_98feat", "train")
print(f"Loaded {len(train_days)} training days")
print(f"Total samples: {sum(d.num_samples for d in train_days)}")
```

### 2. Create DataLoader

```python
from lobtrainer.data import LOBSequenceDataset, create_dataloaders
from lobtrainer.data.transforms import ZScoreNormalizer

# Option A: Manual setup
normalizer = ZScoreNormalizer()
# Fit on training data
all_train_features = np.vstack([d.features for d in train_days])
normalizer.fit(all_train_features)

dataset = LOBSequenceDataset(train_days, window_size=100, stride=10, transform=normalizer)

# Option B: Automatic setup
loaders = create_dataloaders("../data/exports/nvda_98feat", batch_size=64)
```

### 3. Access Features by Index

```python
from lobtrainer.constants import FeatureIndex, SignalIndex

# Access specific features
ofi = features[:, FeatureIndex.TRUE_OFI]
microprice = features[:, FeatureIndex.WEIGHTED_MID_PRICE]
book_valid = features[:, FeatureIndex.BOOK_VALID]

# Safety check before using signals
valid_mask = book_valid > 0.5
ofi_valid = ofi[valid_mask]
```

## Configuration

All experiments are configured via YAML files:

```yaml
# configs/baseline_lstm.yaml
name: baseline_lstm
data:
  data_dir: "../data/exports/nvda_98feat"
  feature_count: 98
  sequence:
    window_size: 100
    stride: 10
model:
  model_type: lstm
  hidden_size: 64
  num_layers: 2
train:
  batch_size: 64
  learning_rate: 1e-4
```

See `configs/` for example configurations.

## Key Modules

### `lobtrainer.constants`

Feature index mapping matching the Rust pipeline export:

```python
from lobtrainer.constants import FeatureIndex, FEATURE_COUNT

assert FEATURE_COUNT == 98
assert FeatureIndex.TRUE_OFI == 84
assert FeatureIndex.BOOK_VALID == 92
```

### `lobtrainer.data`

Data loading and preprocessing:

```python
from lobtrainer.data import LOBSequenceDataset, load_split_data
from lobtrainer.data.transforms import ZScoreNormalizer
```

### `lobtrainer.config`

Type-safe configuration with validation:

```python
from lobtrainer.config import ExperimentConfig, load_config

config = load_config("configs/baseline_lstm.yaml")
```

## Critical Notes

### Sign Conventions

| Feature | Index | Issue | Fix |
|---------|-------|-------|-----|
| `NET_TRADE_FLOW` | 56 | Opposite sign | Negate: `-features[:, 56]` |
| `PRICE_IMPACT` | 47 | Unsigned | Don't use for direction |

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

### Normalization

- Always normalize using **training data statistics only**
- Exclude categorical features (TIME_REGIME at index 93)
- Use per-day Z-score for best results

## Running Tests

```bash
cd lob-model-trainer
pytest tests/ -v
```

## Directory Structure

```
lob-model-trainer/
├── src/lobtrainer/
│   ├── constants/       # Feature indices (data contract)
│   ├── config/          # Configuration schema
│   └── data/            # Dataset and transforms
├── tests/               # Unit tests
├── configs/             # Sample configurations
└── notebooks/           # Exploration notebooks
```

## Related Documentation

- `../plan/01-SIGNAL-HIERARCHY.md` - Signal priority and usage
- `../plan/03-FEATURE-INDEX-MAP-v2.md` - Complete feature mapping
- `../full-data-pipeline.md` - End-to-end pipeline documentation

