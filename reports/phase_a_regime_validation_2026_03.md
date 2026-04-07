# Phase A: Regime Hypothesis Validation — Deep Validation Report

**Date**: 2026-03-21
**Status**: INCONCLUSIVE (original PASS verdict withdrawn)
**Script**: `scripts/validate_regime_hypothesis.py`
**Results**: `scripts/phase_a_results.json`
**Experiment**: E7 in EXPERIMENT_INDEX.md

---

## 1. Executive Summary

Phase A tested whether simple regime gating improves model win rate using a simplified trade PnL model. The best strategy (dno_top_15pct) showed +5.9pp improvement with Bonferroni significance. However, three independent validation agents found that:

1. **The improvement is entirely mediated through prediction magnitude** — depth_norm_ofi (DNO) adds zero independent information after controlling for |calibrated_returns|
2. **The simplified PnL model is biased** — it mechanically favors any filter that selects larger |labels|, giving OPPOSITE results from the full 0DTE backtester
3. **The E6 full backtester validates that magnitude filtering DECREASES win rate** — the exact opposite of what simplified PnL shows

**Verdict**: The simplified PnL model is the wrong tool for regime validation. Phase A is INCONCLUSIVE, not PASS.

---

## 2. Baseline Verification (CONFIRMED)

| Metric | Claimed | Independently Verified |
|--------|---------|----------------------|
| Win rate | 61.0% (0.6099) | 0.6099 |
| DA | 64.2% (0.6416) | 0.6416 |
| IC (Spearman) | 0.380 | 0.3800 |
| N samples | 8,337 | 8,337 |
| Cost | 1.4 bps | Correct (Deep ITM breakeven, IBKR) |

All training quartile boundaries verified to 4+ decimal places against 39,749 training samples.

---

## 3. Critical Finding: DNO is a Proxy for Prediction Magnitude

### 3.1 Correlation

`corr(|depth_norm_ofi|, |calibrated_returns|)` = 0.305-0.350 (Spearman). This means |DNO| is moderately correlated with the model's confidence.

### 3.2 Head-to-Head Comparison (top 15% selectivity)

| Metric | DNO top 15% | |cal_ret| top 15% | Baseline |
|--------|-------------|-------------------|----------|
| N | 1,251 | 1,251 | 8,337 |
| Win rate | 66.9% | **75.0%** | 61.0% |
| DA | 68.7% | **76.9%** | 64.2% |
| IC | 0.488 | **0.533** | 0.380 |
| Mean |label| | 31.32 bps | 26.32 bps | 19.09 bps |

Magnitude filtering dominates DNO on every metric in simplified PnL.

### 3.3 Exclusive Subset Analysis (The Smoking Gun)

| Subset | N | WR | DA |
|--------|---|----|----|
| DNO only (not in |cal| top 15%) | 783 | **61.3%** | **63.2%** |
| |Cal| only (not in DNO top 15%) | 783 | 74.2% | 76.4% |
| Both filters | 468 | 76.3% | 77.8% |
| Baseline | 8,337 | 61.0% | 64.2% |

When DNO selects samples that magnitude filtering does NOT select, performance is AT BASELINE. DNO adds nothing that |cal_ret| doesn't already capture.

### 3.4 Partial Correlations (Definitive)

After controlling for |calibrated_returns|:
- |DNO| → directional accuracy: **r = 0.001**
- |DNO| → win rate: **r = 0.009**

These are effectively zero. DNO's apparent signal quality improvement is entirely mediated through its correlation with |cal_ret|.

---

## 4. Critical Finding: Simplified PnL Model is Biased

### 4.1 The Bias Mechanism

The formula `trade_pnl = sign(cal_ret) * labels - 1.4` produces `win = trade_pnl > 0`. For a win, we need `sign(cal_ret) * labels > 1.4`, which requires both:
- Correct direction: `sign(cal_ret) == sign(labels)`
- Sufficient magnitude: `|labels| > 1.4 bps`

Any filter that selects samples with larger |labels| mechanically increases WR because larger moves more easily exceed the fixed 1.4 bps cost. Both |DNO| and |cal_ret| correlate with |labels|, so both "work" in this model.

### 4.2 Simplified PnL vs Full 0DTE Backtester

| Threshold Filter | Simplified PnL WR | Full 0DTE WR (E6 R8) |
|-----------------|-------------------|----------------------|
| None (all) | 61.0% | 50.6% (at 2 bps) |
| |cal_ret| > P85 | **75.0%** | — |
| |cal_ret| > 20 bps | ~85% (estimated) | **45.5%** |

**The two models give OPPOSITE conclusions.** In simplified PnL, magnitude filtering dramatically improves WR. In the full 0DTE backtester (with delta, theta, BSM, option spread), magnitude filtering DECREASES WR.

### 4.3 Why They Diverge

The full 0DTE backtester includes:
- **Theta decay**: BSM theta costs ~$0.42/min per contract. For a 10-minute hold, that's $4.20 — larger than spread or commission costs
- **Delta exposure**: Deep ITM delta=0.95 attenuates the equity move
- **Option spread**: Real OPRA half-spread, not a fixed bps cost

These costs are NOT proportional to equity move magnitude. A sample with large |labels| doesn't proportionally benefit from larger option profits because the costs (especially theta) are fixed per unit time.

### 4.4 Implication for Phase A

Since DNO gating correlates with |labels| (mean 31.3 bps vs 19.1 bps, ratio=1.64x) through the same mechanism as magnitude filtering, the +5.9pp improvement in simplified PnL almost certainly does NOT transfer to the full 0DTE backtester.

---

## 5. Per-Day Stability (VERIFIED with caveats)

### 5.1 Full Per-Day Table

| Day | N_total | N_gated | WR_base | WR_gated | Delta_pp |
|-----|---------|---------|---------|----------|----------|
| 20251114 | 245 | 64 | 58.0% | 62.5% | +4.5 |
| 20251117 | 245 | 55 | 63.3% | 65.5% | +2.2 |
| 20251118 | 245 | 67 | 58.4% | 64.2% | +5.8 |
| 20251119 | 245 | 68 | 60.4% | 69.1% | +8.7 |
| 20251120 | 245 | 147 | 68.2% | 68.7% | +0.5 |
| 20251121 | 245 | 128 | 67.3% | 65.6% | -1.7 |
| 20251124 | 245 | 61 | 60.8% | 62.3% | +1.5 |
| 20251125 | 245 | 50 | 59.2% | 56.0% | -3.2 |
| 20251126 | 245 | 26 | 66.9% | 69.2% | +2.3 |
| 20251128* | 126 | 7 | 55.6% | 85.7% | +30.2 |
| 20251201 | 245 | 20 | 54.7% | 70.0% | +15.3 |
| 20251202 | 245 | 41 | 65.3% | 75.6% | +10.3 |
| 20251203 | 245 | 35 | 55.1% | 54.3% | -0.8 |
| 20251204 | 245 | 35 | 65.3% | 77.1% | +11.8 |
| 20251205 | 245 | 28 | 60.0% | 71.4% | +11.4 |
| 20251208 | 245 | 33 | 61.6% | 84.8% | +23.2 |
| 20251209 | 245 | 22 | 48.6% | 63.6% | +15.1 |
| 20251210 | 245 | 16 | 63.3% | 50.0% | -13.3 |
| 20251211 | 245 | 23 | 61.2% | 65.2% | +4.0 |
| 20251212 | 245 | 43 | 60.8% | 72.1% | +11.3 |
| 20251215 | 245 | 27 | 60.0% | 63.0% | +3.0 |
| 20251216 | 245 | 24 | 59.2% | 58.3% | -0.9 |
| 20251217 | 245 | 35 | 61.2% | 65.7% | +4.5 |
| 20251218 | 245 | 21 | 63.3% | 81.0% | +17.7 |
| 20251219 | 245 | 14 | 64.5% | 71.4% | +6.9 |
| 20251222 | 245 | 11 | 54.3% | 54.5% | +0.3 |
| 20251223 | 245 | 7 | 65.3% | 85.7% | +20.4 |
| 20251224* | 126 | 3 | 58.7% | — | SKIP |
| 20251226 | 245 | 15 | 68.2% | 80.0% | +11.8 |
| 20251229 | 245 | 10 | 65.3% | 70.0% | +4.7 |
| 20251230 | 245 | 0 | 53.1% | — | SKIP |
| 20251231 | 245 | 5 | 55.1% | 60.0% | +4.9 |
| 20260102 | 245 | 34 | 65.3% | 70.6% | +5.3 |
| 20260105 | 245 | 31 | 57.6% | 51.6% | -5.9 |
| 20260106 | 245 | 45 | 66.1% | 68.9% | +2.8 |

\* Half-day. 2 days skipped (N_gated < 5): 20251224, 20251230.

### 5.2 Summary Statistics

| Metric | Value |
|--------|-------|
| Mean improvement | +6.50pp |
| Median improvement | +4.69pp |
| Std | 8.58pp |
| Sign test p-value | 0.000162 |
| Days positive | 27/33 (81.8%) |
| Days with N_gated < 30 | 15/33 |
| Bootstrap 95% CI (mean) | [+3.69, +9.59]pp |

### 5.3 Caveats

- 6 outlier days (|delta|>15pp) inflate the mean. Without them: +4.82pp (still positive)
- 2 outlier days (20251128, 20251223) have only 7 gated samples — statistically meaningless
- 15 of 33 days have N_gated < 30, making per-day WR estimates noisy
- The P85 threshold was computed from test percentiles (mild data snooping, but sensitivity analysis shows stable results from threshold 30 to 100)

---

## 6. Spread Feature Anomaly

### Root Cause

NVDA stock price appreciated from ~$143 mean (training) to ~$183 mean (test). Since spread is discrete (always a multiple of $0.01 tick size), the bps value of 1 tick shifted:

- Training: 1 tick = ~0.70 bps (above Q25 = 0.5846)
- Test: 1 tick = ~0.55 bps (below Q25 = 0.5846)

Both periods are ~88% 1-tick and ~12% 2-tick spread. The BPS value changed, but the tick count didn't.

### Impact

87% of test samples fall in Q1 (should be ~25%). Q2 has 65 samples, Q3 has 0. Spread provides zero gating selectivity.

### Fix for Future Work

Use tick-count spread (spread / tick_size) or percentile rank instead of absolute bps.

---

## 7. DNO Quartile Distribution Shift

Training |DNO| boundaries: Q25=6.37, Q50=14.20, Q75=27.15

| Test Quartile | Count | % | Expected |
|---------------|-------|---|----------|
| Q1 | 1,435 | 17.2% | 25% |
| Q2 | 1,550 | 18.6% | 25% |
| Q3 | 1,848 | 22.2% | 25% |
| Q4 | 3,504 | 42.0% | 25% |

Test |DNO| distribution shifted ~60% higher than training (test mean 33.7 vs training mean 21.0). Q4 is less selective than designed (42% vs 25%).

---

## 8. Revised Assessment and Next Steps

### What Phase A Actually Proved

1. **Regime-conditional IC varies**: IC ranges 0.299 to 0.446 across DNO quartiles — conditions DO exist where the model is more reliable
2. **The model's own confidence dominates**: |cal_ret| is a better quality signal than any external feature
3. **Simplified PnL is the wrong tool**: Mechanically biased toward larger |labels|, gives opposite results from full backtester
4. **Spread_bps is non-stationary**: Tick structure effect makes bps form unusable

### What Must Happen Before Phase B

1. **Test DNO gating in the full 0DTE backtester** — the only valid evaluation
2. **Test |cal_ret| thresholds** in full backtester as a control (E6 showed this hurts, but should be re-verified at different thresholds)
3. **Test TRUE regime features** not correlated with |cal_ret| — e.g., spread tick count, time regime, order flow volatility independent of magnitude
4. **Only proceed to Phase B** if the full backtester shows regime gating improves option WR

### Key Question for the Regime Project

The regime detection architecture (CUSUM, BOCPD, HMM, Hawkes, Bipower) produces features that are designed to be INDEPENDENT of the model's predictions. Features like CUSUM alarm state, BOCPD run-length, HMM state, and Hawkes intensity ratio should NOT correlate with |cal_ret|. If they don't, they could provide genuinely independent regime information that Phase A's |DNO| proxy could not.

The question is: do market microstructure conditions (independent of model confidence) predict trade quality? Phase A cannot answer this because it only tested features that are model inputs (and thus correlated with model confidence).

---

## 9. Methodological Lesson

**Never use simplified PnL (`sign(pred) * label - cost`) to evaluate regime/gating strategies.** The formula has a built-in bias: any filter correlated with |labels| mechanically produces higher WR because larger moves more easily cover fixed costs. The full 0DTE backtester is the only valid evaluation tool for trading strategy decisions.

This lesson applies to all future experiments where regime/gating filters are evaluated.
