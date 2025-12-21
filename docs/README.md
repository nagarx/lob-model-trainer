# Analysis Documentation

> **Last Updated**: 2025-12-21 (Post-Alignment Fix, 205 tests passing)  
> **Symbol**: NVDA  
> **Dataset**: nvda_98feat_full (98 features, **165 trading days**, 17M samples)  
> **Data Quality**: ✅ CLEAN (0 NaN, 0 Inf)

---

## Documentation Files

| Document | Purpose |
|----------|---------|
| **[ANALYSIS_MODULES_REFERENCE.md](ANALYSIS_MODULES_REFERENCE.md)** | **Comprehensive reference** for all analysis modules, functions, formulas, and interpretation |
| `nvda_complete_analysis.json` | Full analysis results (streaming) |
| `signal_predictive_metrics.csv` | Signal ranking by predictive power |

---

## Analysis Scripts

All analysis is performed via reusable Python scripts in `scripts/`:

| Script | Purpose | Output | Memory |
|--------|---------|--------|--------|
| `run_data_overview.py` | Data validation, quality checks | `nvda_data_overview.json` | ~2 GB |
| `run_label_analysis.py` | Label distribution, autocorrelation, transitions | `nvda_label_analysis.json` | ~2 GB |
| `run_signal_analysis.py` | Signal stats, stationarity, correlation, PCA/VIF | `signal_analysis_results.json` | ~2 GB |
| `run_temporal_dynamics.py` | Autocorrelation, lead-lag, predictive decay | `nvda_temporal_dynamics.json` | ~2 GB |
| `run_generalization.py` | Day-to-day variance, walk-forward validation | `nvda_generalization.json` | ~2 GB |
| **`run_streaming_analysis.py`** | Memory-efficient streaming analysis | `nvda_streaming_*.json` | **< 1 GB** |
| **`run_complete_streaming_analysis.py`** | **ALL analyses** via streaming | `nvda_complete_analysis.json` | **< 1 GB** |

### Recommended Usage (Large Datasets)

```bash
# For datasets > 50 days, use streaming analysis:
cd lob-model-trainer
.venv/bin/python scripts/run_complete_streaming_analysis.py \
    --data-dir ../data/exports/nvda_98feat_full \
    --symbol NVDA
```

### Standard Usage (Small Datasets)

```bash
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

### JSON Results (Full Dataset - 165 Days)

| File | Description |
|------|-------------|
| `nvda_complete_analysis.json` | **Comprehensive streaming analysis** (all results) |
| `nvda_streaming_overview.json` | Dataset profile, quality, label distribution |
| `nvda_streaming_labels.json` | Label ACF, transition matrix, per-day stats |
| `nvda_streaming_signals.json` | Signal mean/std statistics |
| `nvda_streaming_days.json` | Per-day breakdown |

### Legacy Results (16-Day Sample)

| File | Description |
|------|-------------|
| `nvda_data_overview.json` | Data validation, sample counts, quality metrics |
| `nvda_label_analysis.json` | Label distribution, autocorrelation, transitions |
| `nvda_temporal_dynamics.json` | Signal autocorrelation, lead-lag relationships |
| `nvda_generalization.json` | Day-to-day stability, walk-forward validation |
| `signal_analysis_results.json` | Stationarity, correlation, PCA, VIF |

### CSV Results

| File | Description |
|------|-------------|
| `signal_predictive_metrics.csv` | Signal ranking by Pearson r, Spearman r, AUC, MI |

---

## Figures

All figures are in `figures/` directory:

| Figure | Description |
|--------|-------------|
| `nvda_label_distribution.png` | Bar chart of label distribution |
| `nvda_label_autocorrelation.png` | ACF of labels (shows 70% lag-1 correlation) |
| `nvda_transition_matrix.png` | Label transition probabilities (97% diagonal) |
| `nvda_signal_autocorrelation.png` | Signal persistence (half-life analysis) |
| `nvda_predictive_decay.png` | How signal-label correlation fades with lag |
| `nvda_day_correlations.png` | Signal-label correlation by trading day |
| `nvda_walk_forward.png` | Walk-forward validation accuracy |
| `nvda_label_distribution_by_day.png` | Label balance per day |

---

## Quick Reference: Key Findings (NVDA - 165 Days)

### Dataset Summary

| Metric | Value |
|--------|-------|
| Total Days | 165 |
| Date Range | Feb 3, 2025 → Sep 29, 2025 |
| Total Samples | 16,996,000 |
| Total Labels | 1,703,657 |
| Data Quality | ✅ 100% clean (0 NaN, 0 Inf) |

### Label Distribution

| Label | Count | Percentage |
|-------|-------|------------|
| Down | 476,822 | 28.0% |
| Stable | 744,587 | **43.7%** |
| Up | 482,248 | 28.3% |
| **Imbalance Ratio** | 1.56 | Mild |

### Critical Finding: Label Autocorrelation

| Lag | ACF | Interpretation |
|-----|-----|----------------|
| 1 | **0.703** | Very strong clustering |
| 2 | 0.459 | Still significant |
| 5 | -0.006 | Decayed to noise |

### Transition Matrix (97% Persistence)

| From → | Down | Stable | Up |
|--------|------|--------|-----|
| **Down** | **97.2%** | 2.8% | 0.0% |
| **Stable** | 2.1% | **95.8%** | 2.1% |
| **Up** | 0.0% | 2.8% | **97.2%** |

**Critical Insight**: A "predict same as last" baseline achieves ~96% accuracy. **The model must learn to predict TRANSITIONS** to add value.

### Signal Ranking (by Correlation with Labels)

| Rank | Signal | Correlation | Stable Across Days? |
|------|--------|-------------|---------------------|
| #1 | `signed_mp_delta_bps` | +0.0071 | ❌ No (ratio=0.02) |
| #2 | `cancel_asymmetry` | -0.0063 | ❌ No |
| #3 | `fragility_score` | +0.0063 | ❌ No |
| #4 | `depth_norm_ofi` | +0.0047 | ✅ Yes (ratio=0.32) |
| #5 | `true_ofi` | -0.0009 | ✅ Yes (ratio=0.32) |

**Note**: Correlations are very weak (< 1%) but this is **expected for HFT**. Alpha is in subtle patterns, not linear relationships.

### Signal Persistence (Autocorrelation)

| Signal | Lag-1 ACF | Half-Life | Interpretation |
|--------|-----------|-----------|----------------|
| signed_mp_delta_bps | **0.402** | N/A | Very persistent |
| fragility_score | **0.341** | N/A | Persistent |
| cancel_asymmetry | 0.176 | 33 | Slow decay |
| depth_asymmetry | 0.144 | N/A | Moderate |
| true_ofi | 0.025 | 3 | Fast decay |

### Model Architecture Guidance

| Question | Finding | Recommendation |
|----------|---------|----------------|
| Sequence model justified? | ✅ YES | High lag-1 ACF (0.70) |
| Optimal lookback | 10-20 samples | Half-life ~3-5 |
| Primary task | Transition detection | 97% persistence = trivial to predict steady state |
| Key features | depth_norm_ofi, true_ofi | Most stable across days |
| Loss function | Focal or transition-weighted | Focus on regime changes |
| Baseline to beat | **96% accuracy** | From transition matrix |

### Feature Selection

```python
# PRIMARY - Most stable across days
primary = ['depth_norm_ofi', 'true_ofi', 'depth_asymmetry']

# SECONDARY - Add for diversity
secondary = ['executed_pressure', 'fragility_score']

# AVOID - Unstable or low value
avoid = ['signed_mp_delta_bps']  # Low stability ratio (0.02)
```

---

## Memory Efficiency

For the 165-day dataset:

| Approach | Peak Memory | Time |
|----------|-------------|------|
| Bulk loading | **20+ GB** ❌ | ~30 min |
| Streaming analysis | **< 1 GB** ✅ | ~9 min |

The streaming module uses:
- **Welford's online algorithm** for incremental mean/variance
- **One day at a time** processing
- **float32** instead of float64 (50% reduction)
- Explicit `gc.collect()` after each day

---

## Regenerating Analysis

To regenerate all analysis for a new symbol:

```bash
# Set symbol and data path
SYMBOL=AAPL
DATA_DIR=../data/exports/aapl_98feat_full

# Run complete streaming analysis (recommended)
.venv/bin/python scripts/run_complete_streaming_analysis.py \
    --data-dir $DATA_DIR \
    --symbol $SYMBOL
```

---

## Next Steps (Phase 3: Model Development)

Based on Phase 2A findings:

1. **Architecture**: LSTM or Transformer with window=100
2. **Primary features**: `[depth_norm_ofi, true_ofi, depth_asymmetry]`
3. **Loss function**: Focal loss or transition-weighted cross-entropy
4. **Evaluation**: Focus on transition detection, not raw accuracy
5. **Baseline**: Must beat 96% (predict-last) to add value
6. **Validation**: Walk-forward (train on N days, test on day N+1)
