# Key Findings: Label Analysis (Phase 2A)

> **Date**: 2025-12-19  
> **Symbol**: NVDA  
> **Dataset**: nvda_98feat (98 features, 16 trading days)

---

## Executive Summary

### Critical Finding: Label Persistence is By Design

The labels show **extremely high autocorrelation** (97% same-label transitions), which initially appears suspicious but is actually **expected behavior** from the TLOB-style labeling approach:

| Parameter | Value |
|-----------|-------|
| Horizon | 50 samples |
| Stride | 10 samples |
| Overlap | 40/50 = 80% of horizon overlaps between adjacent labels |

**Implication**: Adjacent labels share 80% of their look-ahead window, so they naturally correlate. This is NOT a bug - it's a design choice that:
- ✅ Enables sequence models to learn temporal patterns
- ✅ Preserves data efficiency (5× more labels than non-overlapping)
- ⚠️ Requires careful train/val/test splits to prevent leakage

---

## 1. Label Distribution

| Label | Count | Percentage |
|-------|-------|------------|
| Down (-1) | 29,491 | 27.9% |
| Stable (0) | 42,848 | 40.6% |
| Up (1) | 33,284 | 31.5% |

**Class imbalance ratio**: 1.45× (well-balanced, no weighting needed)

---

## 2. Autocorrelation

| Lag | ACF Value | Interpretation |
|-----|-----------|----------------|
| 1 | **+0.972** | Almost perfect correlation |
| 5 | +0.870 | Still very high |
| 10 | +0.751 | Significant |
| 20 | +0.542 | Moderate |
| 50 | +0.034 | Near zero (horizon boundary) |

**Pattern**: ACF decays as lag approaches horizon (50), confirming the overlapping window effect.

---

## 3. Transition Matrix

```
           To:     Down    Stable      Up
From:
  Down          97.1%     2.9%     0.0%
  Stable         2.0%    95.9%     2.1%
  Up             0.0%     2.7%    97.3%
```

**Key observations**:
- Diagonal dominates (high persistence)
- Down↔Up transitions are essentially zero (price must pass through Stable)
- Mean run length: 29 labels, max: 326 labels

---

## 4. Signal-Label Correlations

| Signal | Correlation | Direction | Rank |
|--------|-------------|-----------|------|
| `true_ofi` | +0.051 | Bullish OFI → Up | **#1** |
| `trade_asymmetry` | +0.031 | More ask trades → Up | #2 |
| `depth_asymmetry` | -0.029 | **Contrarian** | #3 |
| `depth_norm_ofi` | +0.021 | Same as true_ofi | #4 |
| `fragility_score` | +0.015 | Concentrated → Up | #5 |

**Important**: `depth_asymmetry` is negative, meaning **more bid depth predicts DOWN**. This is counterintuitive but may reflect informed traders hiding in the bid.

---

## 5. Regime Analysis

| Regime | OFI Correlation | Up % | Down % | Note |
|--------|-----------------|------|--------|------|
| Close | **+0.090** | 26.2% | 25.6% | **Highest predictability** |
| Early | +0.052 | 35.6% | 31.8% | Good |
| Open | +0.052 | 39.8% | 26.0% | More up-biased |
| Midday | +0.042 | 30.3% | 28.2% | Weakest |
| Closed | +0.044 | 17.6% | 13.9% | Most stable |

**Key insight**: The Close regime (15:30-16:00 ET) has nearly **2× the predictive power** of Midday. Consider regime-specific models.

---

## 6. Modeling Recommendations

### Must Do
1. **Use sequence models** (LSTM, Transformer) - the temporal structure is critical
2. **Split data by day** (not random) to prevent label leakage
3. **Evaluate on non-overlapping predictions** for realistic performance

### Should Consider
1. **Regime-specific models** - Close regime is most predictable
2. **Focus on `true_ofi`** as primary feature (highest correlation)
3. **Investigate `depth_asymmetry`** - contrarian signal may be valuable

### Avoid
1. Random train/test splits (would cause massive leakage)
2. Overweighting Midday samples (lower predictability)
3. Expecting high point-in-time correlations (5% is realistic)

---

## 7. Next Steps

Based on these findings:

1. **Notebook 04 (Signal Predictive Power)**: Deeper analysis of each signal
2. **Notebook 06 (Regime Analysis)**: Explore regime-specific modeling
3. **Baseline Models**: Start with XGBoost on aligned features
4. **Sequence Models**: LSTM with proper temporal splits

---

## Appendix: Run Length Statistics

| Metric | Value |
|--------|-------|
| Number of runs | 272 (for first day) |
| Mean run length | 29.1 labels |
| Median run length | 19.5 labels |
| Max run length | 326 labels |
| 95th percentile | 74 labels |

**Interpretation**: Price trends persist for ~29 labels on average, which at stride=10 means ~290 samples or roughly 1-2 minutes of market time (depending on activity).

