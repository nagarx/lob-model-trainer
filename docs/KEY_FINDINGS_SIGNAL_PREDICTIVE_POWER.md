# Key Findings: Signal Predictive Power (Phase 2A)

> **Date**: 2025-12-19  
> **Symbol**: NVDA  
> **Dataset**: nvda_98feat (98 features, 16 trading days, 105,623 labels)

---

## Executive Summary

### Signal Ranking by Predictive Power

| Rank | Signal | Pearson r | Direction | Sign Consistent | Key Insight |
|------|--------|-----------|-----------|-----------------|-------------|
| **#1** | `true_ofi` | **+0.0450** | → Up | ✓ | Best overall predictor |
| **#2** | `executed_pressure` | +0.0305 | → Up | ✓ | Trade-based signal |
| **#3** | `depth_asymmetry` | **-0.0266** | → Down | ✗ | **CONTRARIAN** |
| **#4** | `fragility_score` | +0.0187 | → Up | ? | Book structure |
| **#5** | `trade_asymmetry` | +0.0183 | → Up | ✓ | Redundant with OFI |
| **#6** | `cancel_asymmetry` | +0.0145 | → Up | ✓ | Order flow signal |
| **#7** | `depth_norm_ofi` | +0.0112 | → Up | ✓ | Redundant with OFI |
| **#8** | `signed_mp_delta_bps` | -0.0008 | → Down | ✗ | Near zero |

---

## 1. Primary Findings

### true_ofi is the Best Predictor
- **Pearson r = +0.0450** (highest among all signals)
- AUC_up = 0.510, AUC_down = 0.516
- Consistent with Cont et al. (2014): order flow imbalance predicts returns
- Positive OFI → expect price UP

### depth_asymmetry is Contrarian
- **Pearson r = -0.0266** (opposite of expected sign)
- Expected: More bid depth → price support → Up
- **Observed**: More bid depth → price DOWN

**Interpretation**:
```
E[depth_asymmetry | Down]:   +0.0507  (more bid depth before Down)
E[depth_asymmetry | Stable]: +0.0183  
E[depth_asymmetry | Up]:     -0.0179  (less bid depth before Up)
```

This suggests **informed traders hide in the bid**:
- Large bid depth may indicate sellers positioning
- Retail sees "support" but informed traders know better

---

## 2. Signal Redundancy

Several signals are highly correlated and provide similar information:

| Pair | Correlation | Implication |
|------|-------------|-------------|
| `true_ofi` ↔ `depth_norm_ofi` | **+0.655** | Highly redundant |
| `true_ofi` ↔ `executed_pressure` | +0.553 | Moderately redundant |
| `true_ofi` ↔ `trade_asymmetry` | +0.536 | Moderately redundant |
| `depth_norm_ofi` ↔ `trade_asymmetry` | **+0.637** | Highly redundant |

**Recommendation**: Use `true_ofi` as the primary OFI signal. Avoid including both `true_ofi` and `depth_norm_ofi` in the same model.

---

## 3. Quintile Analysis (depth_asymmetry)

| Quintile | Condition | P(Up) | P(Down) | Net Edge |
|----------|-----------|-------|---------|----------|
| Bottom 20% | More ask depth (< -0.72) | **0.343** | 0.250 | +9.3% for Up |
| Top 20% | More bid depth (> 0.74) | 0.294 | 0.297 | Neutral |

**Key Insight**: Extreme ask depth (negative depth_asymmetry) predicts Up with 9.3% edge over Down.

---

## 4. Feature Selection Recommendations

### Group A: Primary Features (Use These)
| Signal | Reason |
|--------|--------|
| `true_ofi` | Best linear predictor (r=+0.045), theoretically grounded |
| `depth_asymmetry` | **Contrarian** signal (r=-0.027), provides independent information |

### Group B: Moderate Value (Consider Carefully)
| Signal | Reason |
|--------|--------|
| `fragility_score` | Book structure, low correlation with OFI |
| `cancel_asymmetry` | Order flow signal, moderate predictive power |

### Group C: Redundant with true_ofi (Avoid in Same Model)
| Signal | Correlation with true_ofi | Reason |
|--------|---------------------------|--------|
| `depth_norm_ofi` | r=+0.655 | Same base signal, redundant |
| `executed_pressure` | r=+0.553 | Trade-based, overlaps with OFI |
| `trade_asymmetry` | r=+0.536 | Overlaps with OFI |

### Group D: Low Value (Skip)
| Signal | Reason |
|--------|--------|
| `signed_mp_delta_bps` | Near-zero correlation (r=-0.001) |

---

## 5. Modeling Implications

### Why are correlations so weak (~5%)?

1. **High label autocorrelation (97%)**: Labels are highly persistent due to overlapping horizons
2. **Point-in-time limitation**: We're measuring signal-label correlation at single points
3. **The real value is in temporal patterns**: Sequence models should capture trend onset

### Recommended Approach

```
Model Architecture:
├── Use sequence models (LSTM, Transformer)
├── Focus on temporal patterns, not point correlations
├── Include depth_asymmetry as CONTRARIAN feature
│   └── Option A: Negate it in preprocessing
│   └── Option B: Let model learn the relationship
└── Consider interaction features:
    └── true_ofi × depth_asymmetry (divergence signal)
```

### Feature Engineering Ideas

1. **OFI-Depth Divergence**: When OFI is positive but depth_asymmetry is also positive → conflicting signals
2. **Signal Momentum**: Change in signals over the sequence window
3. **Regime-Specific**: Separate models or features for Open/Midday/Close

---

## 6. Statistical Significance

With 105,623 samples, correlations of |r| > 0.006 are statistically significant (p < 0.05).

All top 6 signals are statistically significant:
- `true_ofi`: r = 0.0450 → **highly significant**
- `depth_asymmetry`: r = -0.0266 → **highly significant**
- `executed_pressure`: r = 0.0305 → **highly significant**

---

## 7. Next Steps

Based on these findings:

1. **Notebook 05 (Temporal Dynamics)**: Analyze signal persistence and lead-lag
2. **Notebook 06 (Regime Analysis)**: Test if depth_asymmetry contrarian effect is regime-specific
3. **Baseline Model**: XGBoost with [true_ofi, depth_asymmetry, fragility_score]
4. **Sequence Model**: LSTM focusing on OFI temporal patterns

---

## Appendix: Raw Metrics

```
Signal                     Pearson r   Spearman r     AUC_up   AUC_down  Sign OK
--------------------------------------------------------------------------------
true_ofi                     +0.0450      +0.0243     0.5100     0.5164        ✓
depth_norm_ofi               +0.0112      +0.0145     0.5050     0.5108        ✓
executed_pressure            +0.0305      +0.0104     0.5073     0.5035        ✓
signed_mp_delta_bps          -0.0008      -0.0049     0.5029     0.4912        ✗
trade_asymmetry              +0.0183      +0.0177     0.5099     0.5090        ✓
cancel_asymmetry             +0.0145      +0.0141     0.5172     0.4966        ✓
fragility_score              +0.0187      +0.0122     0.4994     0.5145        ?
depth_asymmetry              -0.0266      -0.0430     0.4764     0.4778        ✗
```

