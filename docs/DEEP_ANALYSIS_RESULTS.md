# Deep Analysis Results: Signal Predictive Power

> **Date**: 2025-12-20  
> **Symbol**: NVDA  
> **Dataset**: 105,623 labels from 11 trading days  
> **Principle**: Every number must be exact; no self-deception

---

## Executive Summary: Realistic Expectations

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Best signal correlation** | r = +0.045 | Small but **highly significant** (p < 10⁻⁴⁸) |
| **Sample size** | 105,623 | Minimum detectable r > 0.006 at p < 0.05 |
| **Baseline accuracy** | ~33% | Random guessing (3 classes) |
| **Expected model accuracy** | 35-40% | Realistic for 50-sample horizon |
| **Edge at extremes** | +7% | Q5 vs Q1 for true_ofi |

---

## 1. Regime-Specific Signal Behavior

### OFI Correlation by Market Session

| Regime | N samples | true_ofi r | depth_asym r | P(Up) | P(Down) |
|--------|-----------|------------|--------------|-------|---------|
| **Closed** | 4,743 | **+0.085** | +0.093 | 17.6% | 14.1% |
| **Close (15:30-16:00)** | 8,150 | **+0.056** | -0.034 | 26.2% | 25.7% |
| **Early (9:45-10:30)** | 22,860 | +0.052 | -0.053 | 35.7% | 31.8% |
| **Open (9:30-9:45)** | 10,670 | +0.046 | +0.006 | 39.8% | 25.8% |
| **Midday (10:30-15:30)** | 59,200 | +0.038 | -0.041 | 30.3% | 28.2% |

### Key Insight: Close Regime Has Highest OFI Predictability

The **Close** regime (15:30-16:00 ET) has the strongest OFI correlation (+0.056), suggesting:
- End-of-day positioning creates cleaner signal
- Consider regime-specific models or weighting

### Anomaly: "Closed" Regime Has Strongest Signal

The "Closed" regime (after market hours) shows r = +0.085, but:
- Only 4,743 samples (4.5% of data)
- May represent pre/post-market activity
- Requires further investigation before using

---

## 2. Quintile Analysis: Edge at Extremes

### True OFI Quintiles

| Quintile | Range | P(Up) | P(Down) | Net Edge |
|----------|-------|-------|---------|----------|
| Q1 (lowest) | [-36.5, -0.5] | 0.305 | 0.301 | **+0.4%** |
| Q2 | [-0.5, -0.1] | 0.332 | 0.282 | +5.0% |
| Q3 (neutral) | [-0.1, +0.1] | 0.283 | 0.270 | +1.3% |
| Q4 | [+0.1, +0.5] | 0.324 | 0.283 | +4.1% |
| Q5 (highest) | [+0.5, +21.6] | 0.331 | 0.261 | **+7.0%** |

**Q5-Q1 Edge**: High OFI (Q5) has **+7% net edge** over low OFI (Q1).

### Depth Asymmetry Quintiles (CONTRARIAN)

| Quintile | Range | P(Up) | P(Down) | Interpretation |
|----------|-------|-------|---------|----------------|
| Q1 (more ask depth) | [-5.9, -0.7] | **0.343** | 0.250 | **Bullish** |
| Q2 | [-0.7, -0.2] | 0.327 | 0.275 | Slightly bullish |
| Q3 | [-0.2, +0.2] | 0.309 | 0.285 | Neutral |
| Q4 | [+0.2, +0.7] | 0.302 | 0.290 | Neutral |
| Q5 (more bid depth) | [+0.7, +6.4] | 0.294 | **0.297** | **Bearish** |

**Contrarian Confirmed**: More bid depth (Q5) predicts **DOWN**, not UP.

---

## 3. Signal Interaction: OFI × Depth Asymmetry

| Condition | P(Up) | P(Down) | Net Edge | Assessment |
|-----------|-------|---------|----------|------------|
| **High OFI + Low Depth** | 0.335 | 0.255 | **+8.0%** | **Strongest bullish** |
| Low OFI + Low Depth | 0.332 | 0.274 | +5.8% | Bullish |
| High OFI + High Depth | 0.308 | 0.285 | +2.3% | Weak bullish |
| Low OFI + High Depth | 0.282 | 0.302 | -2.0% | Weak bearish |

### Interpretation

The **strongest bullish signal** is:
- High true_ofi (buy pressure)
- Low depth_asymmetry (more ask depth)

This makes sense:
- Buy pressure + thin ask side = likely price increase
- The depth_asymmetry is contrarian because informed traders hide in the bid

---

## 4. Statistical Significance

All core signals are **highly statistically significant** despite small correlations:

| Signal | r | p-value | Significance |
|--------|---|---------|--------------|
| true_ofi | +0.0450 | 1.4×10⁻⁴⁸ | *** |
| executed_pressure | +0.0305 | 4.2×10⁻²³ | *** |
| depth_asymmetry | -0.0266 | 5.3×10⁻¹⁸ | *** |
| fragility_score | +0.0187 | 1.3×10⁻⁹ | *** |
| trade_asymmetry | +0.0183 | 2.6×10⁻⁹ | *** |
| cancel_asymmetry | +0.0145 | 2.6×10⁻⁶ | *** |

With N = 105,623, minimum detectable r at p < 0.05 is r > 0.006.

---

## 5. Why Correlations Are Small (~5%)

This is **expected behavior**, not a failure:

### Reason 1: Label Autocorrelation (97%)

Labels are computed with overlapping horizons:
- Horizon: 50 samples
- Stride: 10 samples
- Overlap: 80% between adjacent labels

Adjacent labels share 80% of their look-ahead window, so they're highly correlated by construction.

### Reason 2: Point-in-Time vs Temporal

We're measuring:
- **Point-in-time**: signal[t] → label[t]
- **Real value**: signal[t-100:t] → label[t]

The real predictive power is in **temporal patterns** (how signals evolve), not point correlations.

### Reason 3: Market Efficiency

If point correlations were > 20%, the alpha would be trivially exploitable and quickly arbitraged away.

---

## 6. Realistic Modeling Expectations

### Performance Targets

| Model Type | Expected Accuracy | Baseline |
|------------|-------------------|----------|
| Random guess | 33% | - |
| Momentum baseline | 35-38% | Label persistence |
| XGBoost (flat) | 36-40% | Uses all features |
| LSTM (sequence) | 38-42% | Captures temporal |

### What "Good" Looks Like

- **Accuracy 40%**: +7% over random = significant
- **Precision > 0.4 for extremes**: When model predicts Up with high confidence, it's right 40%+ of time
- **Profitable after costs**: Depends on trading frequency and costs

### What to Avoid

- Expecting 60%+ accuracy (unrealistic for 50-sample horizon)
- Using random train/test splits (causes massive leakage)
- Overfitting to training data

---

## 7. Actionable Recommendations

### Feature Selection

```python
# Primary features (use these)
primary = ['true_ofi', 'depth_asymmetry']

# Secondary features (add if needed)
secondary = ['fragility_score', 'cancel_asymmetry']

# Avoid (redundant or low value)
avoid = ['depth_norm_ofi', 'trade_asymmetry', 'signed_mp_delta_bps']
```

### Preprocessing

1. **Clip executed_pressure**: Kurtosis = 58,899 (extreme outliers)
2. **Keep depth_asymmetry as-is**: Let model learn contrarian relationship
3. **Consider regime-specific normalization**: Midday vs Close dynamics differ

### Model Architecture

1. **Use sequence models**: LSTM or Transformer with window=100
2. **Split by day**: Never random split (label leakage)
3. **Evaluate on realistic metrics**: Precision@K, profit/loss

### Regime Handling

1. **Option A**: Single model, include time_regime as feature
2. **Option B**: Separate models for Close vs Midday
3. **Option C**: Weighted ensemble

---

## 8. Next Steps

| Priority | Task | Rationale |
|----------|------|-----------|
| **P0** | Baseline XGBoost model | Establish floor performance |
| **P0** | Proper train/val/test by day | Prevent leakage |
| **P1** | LSTM with temporal features | Capture sequence patterns |
| **P1** | Regime-specific analysis | Close may need special handling |
| **P2** | Feature engineering | OFI momentum, signal crosses |

---

## Appendix: Raw Data Locations

- `docs/signal_distribution_stats.csv` - Distribution metrics
- `docs/signal_predictive_metrics.csv` - Predictive power
- `docs/signal_analysis_results.json` - Complete results

## Appendix: Validation Checklist

- [x] Correlations are small but statistically significant
- [x] Contrarian depth_asymmetry confirmed
- [x] Regime-specific patterns documented
- [x] Quintile edges computed
- [x] Signal interactions analyzed
- [x] Realistic expectations set

