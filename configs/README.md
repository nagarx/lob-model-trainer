# Configuration Files

This directory contains YAML configuration files for model training experiments.

## Directory Structure

```
configs/
├── README.md                          # This file
├── deeplob_benchmark.yaml            # Base DeepLOB template
├── deeplob_benchmark_h100.yaml       # DeepLOB h=100 (paper benchmark)
├── experiments/                       # Active experiment configs
│   └── nvda_h10_weighted_v1.yaml     # Current experiment
└── archive/                          # Legacy configs (not actively maintained)
    ├── baseline_lstm.yaml
    ├── baseline_lstm_quick.yaml
    ├── lstm_attn_bidir_h20.yaml
    └── xgboost_baseline.yaml
```

## Active Configurations

### `deeplob_benchmark.yaml`
- **Purpose**: Base template for DeepLOB experiments
- **Horizon**: h=10 (default)
- **Features**: 40 LOB features (benchmark mode)
- **Use case**: Starting point for new DeepLOB experiments

### `deeplob_benchmark_h100.yaml`
- **Purpose**: Paper benchmark configuration
- **Horizon**: h=100 (matches DeepLOB paper's k=4 setting)
- **Use case**: Comparing against published results

### `experiments/nvda_h10_weighted_v1.yaml`
- **Purpose**: Class-weighted loss experiment
- **Key change**: `use_class_weights: true`
- **Metrics**: Tracks directional accuracy, Up/Down precision
- **Use case**: Improving performance on imbalanced data

## Creating New Experiments

1. **Copy from template**:
   ```bash
   cp deeplob_benchmark.yaml experiments/nvda_h{HORIZON}_{VARIANT}_v1.yaml
   ```

2. **Modify key parameters**:
   - `name`: Unique experiment identifier
   - `horizon_idx`: 0=h10, 1=h20, 2=h50, 3=h100, 4=h200
   - `output_dir`: `outputs/experiments/{experiment_name}`
   - `use_class_weights`: true/false
   - `epochs`: Start with 10 for quick iteration

3. **Run experiment**:
   ```bash
   python scripts/train.py --config configs/experiments/{config}.yaml
   ```

## Horizon Index Mapping

| `horizon_idx` | Horizon | Time Ahead | Difficulty |
|---------------|---------|------------|------------|
| 0 | h=10 | ~1 second | Easiest |
| 1 | h=20 | ~2 seconds | Easy |
| 2 | h=50 | ~5 seconds | Medium |
| 3 | h=100 | ~10 seconds | Hard (paper) |
| 4 | h=200 | ~20 seconds | Hardest |

## Naming Convention

```
{model}_{stock}_{horizon}_{variant}_v{version}.yaml
```

Examples:
- `nvda_h10_weighted_v1.yaml` - NVDA, h=10, class-weighted loss, version 1
- `nvda_h100_focal_v1.yaml` - NVDA, h=100, focal loss, version 1

## Key Configuration Sections

```yaml
# Data: Where and what to load
data:
  data_dir: "../data/exports/nvda_balanced"
  horizon_idx: 0  # Which prediction horizon
  
# Model: Architecture parameters
model:
  model_type: deeplob
  deeplob_mode: benchmark  # Uses 40 LOB features
  
# Training: How to train
train:
  epochs: 10
  use_class_weights: true  # Handle class imbalance
  learning_rate: 1.0e-4
```

## Dataset Compatibility

All configs should use: `data_dir: "../data/exports/nvda_balanced"`

This dataset contains:
- 98 features per timestep
- Multi-horizon labels [10, 20, 50, 100, 200]
- ~165 trading days (Feb-Sep 2025)
- Pre-normalized with market-structure z-score

