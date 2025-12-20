# Analysis Documentation

> **Last Updated**: 2025-12-20  
> **Symbol**: NVDA  
> **Dataset**: nvda_98feat (98 features, 16 trading days)

---

## Analysis Scripts

All analysis is performed via reusable Python scripts in `scripts/`:

| Script | Purpose | Output |
|--------|---------|--------|
| `run_data_overview.py` | Data validation, quality checks | `nvda_data_overview.json` |
| `run_label_analysis.py` | Label distribution, autocorrelation, transitions | `nvda_label_analysis.json` |
| `run_signal_analysis.py` | Signal stats, stationarity, correlation, PCA/VIF, predictive power | `signal_analysis_results.json` |
| `run_temporal_dynamics.py` | Autocorrelation, lead-lag, predictive decay | `nvda_temporal_dynamics.json` |
| `run_generalization.py` | Day-to-day variance, walk-forward validation | `nvda_generalization.json` |

### Usage

```bash
# Analyze any symbol (example: NVDA)
cd lob-model-trainer
source .venv/bin/activate

python scripts/run_data_overview.py --data-dir ../data/exports/nvda_98feat --symbol NVDA
python scripts/run_label_analysis.py --data-dir ../data/exports/nvda_98feat --symbol NVDA --save-figures
python scripts/run_signal_analysis.py --data-dir ../data/exports/nvda_98feat --symbol NVDA
python scripts/run_temporal_dynamics.py --data-dir ../data/exports/nvda_98feat --symbol NVDA --save-figures
python scripts/run_generalization.py --data-dir ../data/exports/nvda_98feat --symbol NVDA --save-figures
```

---

## Output Files

### JSON Results
| File | Description |
|------|-------------|
| `nvda_data_overview.json` | Data validation, sample counts, quality metrics |
| `nvda_label_analysis.json` | Label distribution, autocorrelation, transitions, regime stats |
| `nvda_temporal_dynamics.json` | Signal autocorrelation, lead-lag relationships, predictive decay |
| `nvda_generalization.json` | Day-to-day stability, walk-forward validation results |
| `signal_analysis_results.json` | Comprehensive signal analysis (stationarity, correlation, PCA, VIF, predictive power) |

### CSV Results
| File | Description |
|------|-------------|
| `signal_predictive_metrics.csv` | Signal ranking by Pearson r, Spearman r, AUC, Mutual Information |

---

## Figures

All figures are in `figures/` directory:

| Figure | Description |
|--------|-------------|
| `nvda_label_distribution.png` | Bar chart of label distribution |
| `nvda_label_autocorrelation.png` | ACF of labels (shows 97% lag-1 correlation) |
| `nvda_transition_matrix.png` | Label transition probabilities |
| `nvda_signal_autocorrelation.png` | Signal persistence (half-life analysis) |
| `nvda_predictive_decay.png` | How signal-label correlation fades with lag |
| `nvda_day_correlations.png` | Signal-label correlation by trading day |
| `nvda_walk_forward.png` | Walk-forward validation accuracy |
| `nvda_label_distribution_by_day.png` | Label balance per day |

---

## Quick Reference: Key Findings (NVDA)

### Signal Ranking

| Rank | Signal | Pearson r | Use Case |
|------|--------|-----------|----------|
| **#1** | `true_ofi` | +0.045 | **Primary feature** |
| **#2** | `executed_pressure` | +0.031 | Redundant with OFI |
| **#3** | `depth_asymmetry` | **-0.027** | **CONTRARIAN** (opposite sign) |
| #4 | `fragility_score` | +0.019 | Independent, book structure |
| #5 | `cancel_asymmetry` | +0.015 | Independent, order flow |

### Model Architecture Guidance

| Question | Finding | Recommendation |
|----------|---------|----------------|
| Sequence model justified? | ✅ YES | Use LSTM/Transformer |
| Optimal lookback | 50 samples | window=100, stride=10 |
| Walk-forward accuracy | 40% | +7% over random baseline |
| Most stable signals | `executed_pressure`, `true_ofi` | Prioritize in production |
| Signals to avoid | `signed_mp_delta_bps` | Near-zero predictive power |

### Feature Selection

```python
# PRIMARY - Use these
primary = ['true_ofi', 'depth_asymmetry']

# SECONDARY - Add if needed
secondary = ['fragility_score', 'cancel_asymmetry']

# AVOID - Redundant or low value
avoid = ['depth_norm_ofi', 'trade_asymmetry', 'signed_mp_delta_bps']
```

---

## Regenerating Analysis

To regenerate all analysis for a new symbol:

```bash
# Set symbol and data path
SYMBOL=NVDA
DATA_DIR=../data/exports/nvda_98feat

# Run all analyses
python scripts/run_data_overview.py --data-dir $DATA_DIR --symbol $SYMBOL
python scripts/run_label_analysis.py --data-dir $DATA_DIR --symbol $SYMBOL --save-figures
python scripts/run_signal_analysis.py --data-dir $DATA_DIR --symbol $SYMBOL
python scripts/run_temporal_dynamics.py --data-dir $DATA_DIR --symbol $SYMBOL --save-figures
python scripts/run_generalization.py --data-dir $DATA_DIR --symbol $SYMBOL --save-figures
```

---

## Next Steps (Phase 2B)

Based on Phase 2A findings:

1. **Train baseline XGBoost** with features: `[true_ofi, depth_asymmetry, fragility_score]`
2. **Train LSTM** with window=100, temporal splits by day
3. **Evaluate on realistic metrics**: Precision@K, walk-forward accuracy
4. **Consider regime-specific models** (Close regime has 2× predictive power)

