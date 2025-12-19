# Key Findings: Phase 2A Complete Analysis

> **Date**: 2025-12-20  
> **Symbol**: NVDA  
> **Dataset**: nvda_98feat (98 features, 11 training days, 1.05M samples)

---

## Executive Summary

Phase 2A analysis is complete. Key findings:

| Finding | Value | Implication |
|---------|-------|-------------|
| **Best Predictor** | `true_ofi` (r=+0.045) | Use as primary feature |
| **Contrarian Signal** | `depth_asymmetry` (r=-0.027) | Opposite of expected - valuable |
| **Redundant Pairs** | 4 pairs (|r|>0.5) | Avoid multicollinearity |
| **Extreme Kurtosis** | `executed_pressure` (58,899) | Requires robust methods |
| **Label Persistence** | 97% autocorrelation | Sequence models essential |

---

## 1. Signal Distribution Analysis

### Statistical Properties

| Signal | Skewness | Kurtosis | % Outliers | Assessment |
|--------|----------|----------|------------|------------|
| `true_ofi` | +0.51 | **82.9** | 1.4% | Heavy-tailed |
| `depth_norm_ofi` | +0.03 | 2.4 | 0.8% | Near-normal |
| `executed_pressure` | **+130.0** | **58,899** | 0.2% | **EXTREME** |
| `signed_mp_delta_bps` | -0.72 | 50.1 | 2.5% | Heavy-tailed |
| `trade_asymmetry` | +0.07 | -0.8 | 0.0% | Near-normal |
| `cancel_asymmetry` | -0.24 | 6.7 | 2.3% | Moderate tails |
| `fragility_score` | **+2.66** | 8.0 | 3.1% | Right-skewed |
| `depth_asymmetry` | +0.20 | 3.1 | 1.3% | Slight tails |

### Key Insight: Extreme Values in `executed_pressure`

The `executed_pressure` signal has:
- Skewness: +130 (extremely right-skewed)
- Kurtosis: 58,899 (normal = 0)

This indicates rare but extreme trade imbalances. **Recommendation**: Consider clipping or log-transformation before modeling.

---

## 2. Signal Correlation Analysis

### Redundant Signal Pairs

| Signal 1 | Signal 2 | Correlation | Action |
|----------|----------|-------------|--------|
| `true_ofi` | `depth_norm_ofi` | **+0.655** | Keep `true_ofi`, drop `depth_norm_ofi` |
| `depth_norm_ofi` | `trade_asymmetry` | **+0.637** | - |
| `true_ofi` | `executed_pressure` | +0.553 | Consider keeping both |
| `true_ofi` | `trade_asymmetry` | +0.536 | Drop `trade_asymmetry` |

### Independent Signals

| Signal | Max |r| with Others | Status |
|--------|---------------------|--------|
| `fragility_score` | 0.07 | **Independent** |
| `cancel_asymmetry` | 0.12 | **Independent** |
| `depth_asymmetry` | 0.24 | Mostly independent |

---

## 3. Signal Predictive Power

### Ranking by |Pearson r|

| Rank | Signal | r | AUC_up | AUC_down | Sign OK |
|------|--------|---|--------|----------|---------|
| **#1** | `true_ofi` | **+0.045** | 0.510 | 0.516 | ✓ |
| **#2** | `executed_pressure` | +0.031 | 0.507 | 0.504 | ✓ |
| **#3** | `depth_asymmetry` | **-0.027** | 0.476 | 0.478 | ✗ |
| #4 | `fragility_score` | +0.019 | 0.499 | 0.515 | ? |
| #5 | `trade_asymmetry` | +0.018 | 0.510 | 0.509 | ✓ |
| #6 | `cancel_asymmetry` | +0.015 | 0.517 | 0.497 | ✓ |
| #7 | `depth_norm_ofi` | +0.011 | 0.505 | 0.511 | ✓ |
| #8 | `signed_mp_delta_bps` | -0.001 | 0.503 | 0.491 | ✗ |

### Contrarian Signal: depth_asymmetry

```
E[depth_asymmetry | Down]:   +0.051  (more bid depth before Down)
E[depth_asymmetry | Stable]: +0.018  
E[depth_asymmetry | Up]:     -0.018  (less bid depth before Up)
```

**Interpretation**: Informed traders may hide sell orders in the bid, creating the illusion of support.

---

## 4. Feature Selection Recommendations

### Final Feature Groups

**GROUP A - USE (Primary)**
```python
primary_features = [
    'true_ofi',        # Best predictor, theoretically grounded
    'depth_asymmetry', # Contrarian signal (use raw, let model learn)
]
```

**GROUP B - CONSIDER (Independent)**
```python
moderate_features = [
    'fragility_score',    # Book structure, independent
    'cancel_asymmetry',   # Order flow, independent
]
```

**GROUP C - AVOID (Redundant)**
```python
redundant_features = [
    'depth_norm_ofi',    # r=0.66 with true_ofi
    'trade_asymmetry',   # r=0.54 with true_ofi
    'executed_pressure', # r=0.55 with true_ofi (but may keep for extreme events)
]
```

**GROUP D - SKIP (Low Value)**
```python
skip_features = [
    'signed_mp_delta_bps',  # Near-zero predictive power
]
```

---

## 5. Modeling Recommendations

### Architecture
1. **Use sequence models** (LSTM, Transformer) - 97% label autocorrelation
2. **Split by day** (not random) - prevent label leakage
3. **Consider regime-specific models** - Close regime has 2x predictive power

### Preprocessing
1. **Clip `executed_pressure`** - extreme kurtosis (58,899)
2. **Keep `depth_asymmetry` as-is** - let model learn contrarian relationship
3. **Exclude redundant features** - avoid multicollinearity

### Baseline Models to Try
```python
# Flat model (ignores sequence)
XGBoost with features: [true_ofi, depth_asymmetry, fragility_score, cancel_asymmetry]

# Sequence model
LSTM with window=100, stride=10
Input: [true_ofi, depth_asymmetry] per timestep
```

---

## 6. Analysis Code

Analysis is now in reusable Python modules:

```bash
# Run complete signal analysis
python scripts/run_signal_analysis.py --data-dir ../data/exports/nvda_98feat
```

Modules available:
- `lobtrainer.analysis.data_loading` - Load and align data
- `lobtrainer.analysis.signal_stats` - Distribution analysis
- `lobtrainer.analysis.signal_correlations` - Correlation/redundancy
- `lobtrainer.analysis.predictive_power` - Predictive metrics

---

## 7. Next Steps

1. **Phase 2B**: Train baseline models (XGBoost, LogReg) with optimal features
2. **Notebook 05**: Temporal dynamics analysis (signal autocorrelation, lead-lag)
3. **Notebook 06**: Regime-specific analysis (per market session)
4. **Model Training**: LSTM with proper temporal splits

---

## Appendix: Raw Results

Results saved to:
- `docs/signal_distribution_stats.csv`
- `docs/signal_predictive_metrics.csv`
- `docs/signal_analysis_results.json`

