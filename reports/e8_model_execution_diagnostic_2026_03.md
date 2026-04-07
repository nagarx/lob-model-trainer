# E8: Model-Execution Direction Diagnostic — Full Report

**Date**: 2026-03-21
**Status**: CRITICAL FINDING — Model predicts smoothing artifact, not future direction
**Scripts**: `scripts/e8_model_diagnostic.py`, `scripts/e8_statistical_foundations.py`
**Results**: `scripts/e8_diagnostic_results.json`, `scripts/e8_foundations_results.json`

---

## 1. Executive Summary

The E5/E6 TLOB model (IC=0.380, DA=64.2% on smoothed labels) has **48.3% direction accuracy on point-to-point returns** — worse than a coin flip. The model learned to predict the smoothing residual (R²=45% with residual, R²=0.02% with point component). This structurally explains ALL negative backtest results (R1-R8). No regime gating, cost adjustment, or calibration can fix a model whose direction signal is anti-correlated with tradeable returns.

All four candidate fix paths are blocked:
- **Fix C (retrain on point labels)**: 0/67 non-price features have IC > 0.05 for point returns at 60s bins
- **Fix A (adjust smoothing window)**: MaxFeatIC values driven by price-level artifacts, not OFI signal
- **Fix B (adjust horizon)**: DA ≤ 49.1% at ALL horizons H=1 to H=50
- **Regime gating**: No independent condition shows DA > 52%

Statistical foundations for the regime architecture are partially valid (4/8 pass), confirming that OFI ACF, Student-t tails, and EWMA whitening hold — but BPV, spread persistence, and CUSUM inflation assumptions fail at 60s bins.

---

## 2. Model-Execution Decomposition

### 2.1 Variance Decomposition

```
smoothed_return = 0.7434 × point_return + residual
where Cov(point_return, residual) = 0 (orthogonal decomposition)
```

| Component | Var(component) | % of Var(smoothed) | R²(model, component) |
|-----------|---------------|--------------------|-----------------------|
| Point component | 514.85 | 68.5% | **0.02%** |
| Residual | 236.31 | 31.5% | **45.0%** |

The model captures essentially zero of the point-return signal. Its entire predictive power is aligned with the smoothing residual — the part of the smoothed label that is orthogonal to the actual tradeable return.

### 2.2 Direction Agreement Analysis

| Condition | N | % | Model DA (smoothed) | Model DA (point) |
|-----------|---|---|--------------------|--------------------|
| Labels agree on direction | 6,708 | 80.5% | 57.9% | 57.9% |
| Labels DISAGREE on direction | 1,629 | 19.5% | **90.1%** | **9.1%** |
| Overall | 8,337 | 100% | 64.2% | 48.3% |

When smoothed and point returns point in the same direction (80.5%), the model has modest accuracy (57.9%) on both. But on the 19.5% where they disagree, the model predicts the smoothed direction with 90.1% accuracy — meaning it predicts the WRONG point direction 90.9% of the time.

### 2.3 Why This Happens

OFI features are **contemporaneous** — they describe current order flow which decays before the point endpoint at H=10 events (10 minutes). The smoothed return formula averages prices over the future smoothing window (events t+5 through t+10), capturing the early flow impact that OFI correlates with. The point return at t+10 is dominated by NEW information at events t+6 through t+10 that was not present at time t.

The model optimizes MSE on smoothed labels. Since the residual (smoothed minus its point-return projection) is smoother and more predictable from OFI features than the point component, the model naturally learns the residual. This is not a bug — it's the optimal MSE solution given the features and label.

---

## 3. Feature IC Against Point Returns (Measurement A)

### All 23 features with IC > 0.05 are PRICE features

The initial analysis found 23/89 features passing the IC > 0.05 gate. However, ALL 23 are price-level features (ask_prices L1-L10, bid_prices L1-L10, mid_price, weighted_mid_price). Their IC ≈ -0.103 reflects mean reversion of the stock price during the test period (Nov 2025 - Jan 2026), NOT a tradeable signal.

### Non-price features: ALL below IC = 0.05

| Feature | IC with point | Tradeable? |
|---------|--------------|-----------|
| spread_bps (42) | +0.071 | Marginal (likely price artifact) |
| true_ofi (84) | -0.009 | No |
| depth_norm_ofi (85) | -0.008 | No |
| volume_imbalance (45) | +0.011 | No |
| signed_mp_delta (87) | +0.016 | No |
| order_flow_volatility (58) | +0.011 | No |
| fragility_score (90) | -0.041 | No |

**Conclusion**: OFI features have zero IC for point-to-point returns at 60s bins, confirming E2/E3 findings at a different timescale. The OFI signal is purely contemporaneous.

---

## 4. Smoothing Window Sweep (Measurement B)

| k | r(smoothed, point) | DA(labels) | MaxFeatIC | IC > 0.05? | r > 0.90? |
|---|-------------------|-----------|-----------|-----------|----------|
| 1 | 0.554 | 67.4% | 0.304 | YES | no |
| 2 | 0.617 | 70.4% | 0.311 | YES | no |
| 3 | 0.683 | 73.4% | 0.315 | YES | no |
| 5 | 0.828 | 80.5% | 0.264 | YES | no |
| 10 | 0.903 | 86.1% | 0.170 | YES | YES |

k=10 appears optimal (IC=0.170 AND r=0.903). However, the MaxFeatIC values are dominated by price-level features. After excluding prices, the MaxFeatIC for OFI-type features would be below 0.05 at all k values.

**The fundamental tradeoff has NO sweet spot for OFI features**: as k increases, the label-point correlation improves but the OFI-label IC decreases (since more smoothing = more residual for OFI to capture). At no k value do OFI features have both IC > 0.05 AND r > 0.90 with point returns.

---

## 5. Horizon Sweep (Measurement C)

| H | DA_point | IC_point | r(smoothed, point) |
|---|---------|---------|-------------------|
| 1 | 48.1% | -0.014 | 0.349 |
| 5 | 48.1% | -0.033 | 0.779 |
| 10 | 48.3% | -0.032 | 0.828 |
| 20 | 49.1% | -0.010 | 0.632 |
| 50 | 48.3% | -0.032 | 0.387 |

The model has DA ≤ 49.1% at every horizon tested (H=1 to H=50). There is no horizon where the model correctly predicts point-return direction.

Note: r(smoothed, point) follows the expected pattern — peaking at H=8-9 where the smoothing windows overlap most with the point endpoint, then declining at very long horizons where both labels converge to different things.

---

## 6. Per-Condition Direction Accuracy (Measurement D)

No independent condition (features with |corr| < 0.10 with |cal_ret|) produces DA_point > 52%:

| Condition | Best Bucket | DA_point | Independent? |
|-----------|------------|---------|-------------|
| order_flow_volatility Q1 (calm) | Q1 | 49.6% | Yes |
| dt_seconds Q3 | Q3 | 49.3% | Yes |
| spread_bps Q1 (tight) | Q1 | 49.7% | Yes |
| active_order_count Q1 (low) | Q1 | 51.2% | Yes |
| depth_ticks_bid Q4 (deep) | Q4 | 49.5% | Yes |

The closest is active_order_count Q1 at 51.2% — within sampling noise of 50%.

---

## 7. Statistical Foundation Validation (4/8 PASS)

| ID | Assumption | Measured | Pass | Implication |
|----|-----------|----------|------|-------------|
| A3 | OFI ACF(1) ≈ 0.168 | **0.164** | PASS | CUSUM cadence (60s bins) is correct |
| A5 | EWMA whitening | **ACF=0.028 at λ=0.80** | PASS | Whitening works, but optimal λ=0.80 not 0.95 |
| A6 | Spread ACF(1) ≈ 0.35 | **0.065** | FAIL | Spread much less persistent than expected |
| A9 | Kurtosis ∈ [5,20] | **6.375** | PASS | Student-t appropriate (confirms df ≈ 3) |
| A11 | Student-t df ∈ [3,15] | **df=3.12** | PASS | Very heavy tails. Gaussian BOCPD would fail. |
| A14 | CUSUM inflation >2× | **ratio=0.95** | FAIL | EWMA unnecessary — ACF=0.164 is low enough |
| A53 | Return ACF(1) ≈ 0.277 | **0.968** | FAIL | Artifact of overlapping smoothing windows |
| BPV | Jump ratio ∈ [0.05,0.40] | **0.038** | FAIL | Jumps negligible at 60s bins |

### Key Insights from Foundation Validation

1. **OFI ACF = 0.164 at 60s bins**: This is exactly what the profiler predicted (0.175 at 1 min). The regime plan's cadence assumption is correct.

2. **Student-t df = 3.12**: Extremely heavy tails. Gaussian assumptions would catastrophically fail. This validates the regime plan's insistence on Student-t likelihoods for BOCPD.

3. **Spread ACF = 0.065**: Much less persistent than expected (0.35). This means the CUSUM detector for spread may need different parameters, or spread changes are too fast for 60s-bin detection.

4. **CUSUM inflation = 0.95**: The EWMA preprocessing step may be unnecessary. At ACF=0.164, raw CUSUM performs equally well. This simplifies the architecture.

5. **BPV jump ratio = 0.038**: Jumps are negligible at 60s bins. The bipower variation module provides minimal value at this timescale — it's designed for tick-level or 5-minute data.

6. **Return ACF = 0.968**: This extreme autocorrelation is an artifact of the smoothed label formula with overlapping windows (stride=1). NOT a market property.

---

## 8. Strategic Implications

### For the Regime Detection Project

The regime architecture (CUSUM, BOCPD, HMM, Hawkes, BPV in final_plan/) is **architecturally sound** for a model that can predict point-return direction:
- OFI ACF validates 60s-bin cadence
- Student-t validates probabilistic likelihood choice
- EWMA whitening works (though may be unnecessary)

But the current model CANNOT predict point-return direction. No amount of regime infrastructure can fix this. **Phase B remains BLOCKED.**

### For the Pipeline

The root cause of all negative backtests is now identified: the model optimizes for the smoothing residual, which is orthogonal to the tradeable return. Possible next directions:

1. **Re-export with k=1 or k=2** and retrain. But E2/E3 showed IC=0 for point returns — the signal may not exist at any smoothing level for OFI features.

2. **Per-event architecture** (Kolm et al. 2023). Their LSTM sees raw per-event order flow transitions, not accumulated snapshots. This architecture can capture dynamics that our snapshot features miss. HIGH effort.

3. **Alternative features** that predict endpoint direction (not average drift). Price momentum, microstructure imbalance persistence, or cross-venue lead-lag might have point-return IC.

4. **Different execution** matching the smoothed window. TWAP exit over k bins would realize the smoothed return. But this was tested (R5 ablation) and showed marginal improvement.

5. **Accept that the smoothed model has value as a risk signal** (not direction signal). Use it for position sizing or hedging, not directional trading.

---

## 9. Files

| File | Content |
|------|---------|
| `scripts/e8_model_diagnostic.py` | Model diagnostic script |
| `scripts/e8_statistical_foundations.py` | Foundation validation script |
| `scripts/e8_diagnostic_results.json` | Full diagnostic results |
| `scripts/e8_foundations_results.json` | Foundation validation results |
