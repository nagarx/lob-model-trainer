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
├── experiments/                           # ✅ ACTIVE configs (3)
│   ├── nvda_tlob_h10_v1.yaml             # TLOB H=10 (short-term)
│   ├── nvda_tlob_h100_v1.yaml            # TLOB H=100 (paper benchmark)
│   └── nvda_tlob_triple_barrier_11mo_v1.yaml  # Triple Barrier
└── archive/                               # 📦 REFERENCE configs (6)
    ├── baseline_lstm.yaml                 # Pure LSTM model
    ├── lstm_attn_bidir_h20.yaml          # LSTM + Attention + Bidirectional
    ├── deeplob_benchmark.yaml            # DeepLOB (Zhang et al. 2019)
    ├── nvda_bigmove_opportunity_v1.yaml  # Opportunity labeling
    ├── nvda_tlob_bigmove_v1.yaml         # TLOB + Opportunity
    └── nvda_tlob_binary_signal_v1.yaml   # Binary classification + Focal Loss
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

## Active Experiment Configs (3)

These are ready to run on the current 234-day dataset.

| Config | Model | Horizon | Labels | Purpose | When to Use |
|--------|-------|---------|--------|---------|-------------|
| `experiments/nvda_tlob_h10_v1.yaml` | TLOB | H=10 | TLOB | Short-term (~1s) | Fast iteration, high accuracy |
| `experiments/nvda_tlob_h100_v1.yaml` | TLOB | H=100 | TLOB | Paper benchmark (~10s) | Compare with DeepLOB paper |
| `experiments/nvda_tlob_triple_barrier_11mo_v1.yaml` | TLOB | H=50 | Triple Barrier | Risk-managed trading | Backtesting, win rate focus |

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
| `tlob` | Transformer with dual attention | All 98 | Berti & Kasneci (2025) |
| `deeplob` | CNN + Inception + LSTM | First 40 (LOB only) | Zhang et al. (2019) |
| `lstm` | Stacked LSTM | All 98 | Hochreiter & Schmidhuber (1997) |
| `gru` | Stacked GRU | All 98 | Cho et al. (2014) |

---

## Labeling Strategies

| `labeling_strategy` | Classes | Class Meanings | Best For |
|---------------------|---------|----------------|----------|
| `tlob` | 3 | 0=Down, 1=Stable, 2=Up | Trend prediction |
| `triple_barrier` | 3 | 0=StopLoss, 1=Timeout, 2=ProfitTarget | Trading decisions |
| `opportunity` | 3 | 0=BigDown, 1=NoOpportunity, 2=BigUp | Big move detection |

---

## Loss Functions

| `loss_type` | When to Use | Class Imbalance Handling |
|-------------|-------------|--------------------------|
| `cross_entropy` | Balanced classes | None |
| `weighted_ce` | Imbalanced classes | Inverse frequency weights |
| `focal` | Severely imbalanced | Down-weights easy examples |

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
  labeling_strategy: tlob  # Options: tlob, triple_barrier
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
  model_type: tlob  # Options: tlob, deeplob, lstm, gru
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
  loss_type: weighted_ce  # Options: cross_entropy, weighted_ce, focal
  use_class_weights: true
  task_type: multiclass   # Options: multiclass, binary_signal

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

## Dataset Schema (v2.1)

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
