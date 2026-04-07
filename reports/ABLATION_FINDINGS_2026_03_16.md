# Simple Model Ablation Findings

> **Cost correction (2026-03-18)**: Deep ITM breakeven was corrected from 0.7-0.8 bps to **1.4 bps** (spread was missing from the original calculation). See `IBKR-transactions-trades/COST_AUDIT_2026_03.md`.

**Date:** 2026-03-16
**Status:** All results independently validated (sklearn R-squared, scipy IC, cross-checked against TLOB experiment)
**Purpose:** Answer the fundamental question: how much of the TLOB's R-squared=0.464 comes from the architecture vs the features?

---

## 1. Ablation Results (All Validated)

| Level | Model | Params | R-squared | IC | DA | MAE |
|-------|-------|--------|-----------|-----|-----|-----|
| L0 | IC-Weighted Composite | 5 | -0.487 | 0.342 | 0.620 | 3.58 |
| **L1** | **Temporal Ridge** | **53** | **0.324** | **0.616** | **0.722** | **2.75** |
| L2 | Ridge + Polynomial | 68 | 0.324 | 0.616 | 0.722 | 2.75 |
| **L3** | **GradientBoosting** | **200 trees** | **0.397** | **0.617** | **0.723** | **2.58** |
| L3b | GradBoost (raw 128) | 200 trees | 0.137 | 0.446 | 0.654 | 3.13 |
| ref | Ridge (128, last-step) | 128 | 0.170 | 0.433 | 0.651 | -- |
| ref | DEPTH_NORM_OFI only | 1 | 0.107 | 0.335 | 0.620 | -- |
| ref | **TLOB T=100** | **693K** | **0.464** | **0.677** | **0.749** | **2.43** |

### Validation Checks

| Check | Result |
|-------|--------|
| L1 R-squared manual = sklearn = saved | 0.324011 (exact match) |
| Test samples = TLOB test samples | Both 50,724 |
| Labels source = same H10 smoothed avg | Confirmed |
| Temporal feature[0] = raw DEPTH_NORM_OFI | allclose=True |
| No NaN/Inf in temporal features | Confirmed |

---

## 2. What We Learned

### Finding 1: Temporal Ridge (53 features) captures 91% of TLOB IC

| Metric | Temporal Ridge | TLOB | Ratio |
|--------|---------------|------|-------|
| IC | 0.616 | 0.677 | **91.0%** |
| DA | 0.722 | 0.749 | **96.4%** |
| R-squared | 0.324 | 0.464 | **69.8%** |

The rank-ordering ability (IC) of a 53-parameter Ridge regression with hand-crafted temporal features is within 9% of a 693,000-parameter TLOB transformer. Directional accuracy is within 3.6%.

The R-squared gap is larger (0.140) because Ridge cannot capture the nonlinear variance reduction that TLOB provides. But for trading decisions, IC and DA matter more than R-squared.

### Finding 2: GradientBoosting captures 91.1% of TLOB IC and 85.6% of R-squared

GradientBoosting on the same 53 temporal features achieves R-squared=0.397 (vs TLOB 0.464). The remaining gap of 0.067 R-squared is what TLOB's attention mechanism adds beyond gradient-boosted trees.

Critically, GradientBoosting IC (0.617) is nearly identical to Ridge IC (0.616). This means **the nonlinear interactions captured by gradient boosting improve variance explanation (R-squared) but NOT rank-ordering (IC)**. For trading, this suggests the linear temporal features contain most of the actionable information.

### Finding 3: Polynomial interactions add zero value

L2 (Ridge + degree-2 polynomials) = L1 (Ridge). R-squared: 0.324324 vs 0.324011. The difference is noise. The 15 additional polynomial interaction features provide no incremental predictive power.

### Finding 4: Temporal structure is critical (L3b vs L3)

GradBoost on raw 128 features (last timestep only): R-squared=0.137, IC=0.446.
GradBoost on 53 temporal features: R-squared=0.397, IC=0.617.

The temporal features (rolling means, slopes, rate-of-change) nearly **triple** the R-squared and add 38% more IC. This proves the temporal dimension contains genuine information -- the model needs to see how features evolve over the sequence window, not just their current values.

### Finding 5: The TLOB adds real but modest value

The full picture of what each component contributes:

| Component | R-squared Added | IC Added | What It Does |
|-----------|----------------|----------|--------------|
| Raw last-timestep features | 0.137 | 0.446 | Current market state |
| + Temporal engineering | +0.187 | +0.170 | How features evolve (rolling means, slopes) |
| + Nonlinear (GradBoost) | +0.073 | +0.001 | Nonlinear interactions (but NOT better ranking) |
| + TLOB attention | +0.067 | +0.060 | Learned temporal attention patterns |
| **Total (TLOB)** | **0.464** | **0.677** | |

The TLOB's attention mechanism adds the final 14.4% of R-squared (0.067) and 8.9% of IC (0.060) on top of gradient-boosted temporal features. This is real value, but it comes at 3,465x more parameters (693K vs 200 trees).

---

## 3. TWAP Backtest Results

| Config | Trades | Option Return | vs Point-to-Point |
|--------|--------|---------------|--------------------|
| min=5 bps, w=5 | 1,799 | -6.67% | +11% better than -7.53% |
| min=5 bps, w=10 | 1,315 | -5.56% | +26% better |
| min=8 bps, w=5 | 214 | -0.93% | Same as point-to-point |
| min=8 bps, w=10 | 196 | -1.04% | Slightly worse |

TWAP execution marginally improves over point-to-point at lower thresholds but does not achieve profitability. The cost structure (4.7 bps ATM breakeven) dominates when average predicted returns are ~3 bps.

---

## 4. Fine-Grained Data Findings

| Metric | event_count=1000 (current) | event_count=200 (fine-grained) |
|--------|---------------------------|-------------------------------|
| Samples per day | ~1,000 | ~10,000 |
| H1 DEPTH_NORM_OFI IC | N/A | 0.045 |
| H1 Ridge R-squared | N/A | 0.002 |
| Point-return signal | Zero at H10 (10,000 events) | Very weak at H1 (200 events) |

Even at 5x finer granularity, point-return signal is IC=0.045 -- far weaker than smoothed-average signal IC=0.309 at coarser sampling. The fine-grained direction confirmed: **OFI signal exists only in the smoothed average, not in point-to-point returns, at any practical sampling rate we've tested.**

---

## 5. Infrastructure Built

| Module | What Was Added | Status |
|--------|---------------|--------|
| `lob-models/src/lobmodels/layers/cvml.py` | CVML (Li et al. ICLR 2025): 5-layer dilated causal Conv1D | Tested, 49K params |
| `lob-models/src/lobmodels/losses/gmadl.py` | GMADL (Michankov et al. 2024): directional loss | Tested, verified gradients |
| `lob-models/src/lobmodels/losses/pinball.py` | Pinball loss for quantile regression | Tested |
| CVML export from `lobmodels.layers` | `from lobmodels.layers import CVML` | Wired |
| GMADL in `compute_loss()` | `regression_loss_type: gmadl` works in TLOB, DeepLOB, MLPLOB | Wired |
| `lob-backtester/src/lobbacktest/strategies/twap.py` | TWAP execution strategy | Implemented, tested |
| `lob-dataset-analyzer/...effective_horizon.py` | EffectiveHorizonAnalyzer | Built, run |
| `lob-dataset-analyzer/...representation_comparison.py` | OrderFlowRepresentationAnalyzer | Built, run |
| Fine-grained export config | `event_count=200`, horizons [1,2,5,10,20] | Exported (2.7M sequences) |

---

## 6. Implications for Next Steps

### 6.1 The Simple Model Path Is Viable

Temporal Ridge with 53 features achieves IC=0.616 (91% of TLOB). For a "few trades when readable" strategy, this IC is sufficient:
- It trains in milliseconds (vs 7 min/epoch for TLOB)
- It's fully interpretable (Ridge coefficients tell you exactly what matters)
- It can be retrained daily for regime adaptation
- It produces continuous predictions for magnitude-based filtering

**Recommended experiment:** Backtest the Temporal Ridge model with RegressionStrategy at various thresholds. If it approaches TLOB's backtest performance, the simpler model wins.

### 6.2 The Deep Learning Premium Is Real But Small

TLOB adds IC=0.060 over GradientBoosting (0.677 vs 0.617). This 9% premium comes from learned temporal attention patterns that the 53 hand-crafted features don't capture. Whether this premium is worth the complexity depends on:
- Trading frequency: at very high frequency, 9% IC improvement compounds
- Latency: Ridge inference is nanoseconds; TLOB is milliseconds
- Adaptability: Ridge can retrain daily; TLOB requires GPU training

### 6.3 The Tradability Problem Remains

Both TWAP and point-to-point backtests are negative at all thresholds except near-breakeven at 8-10 bps. The core issue is unchanged: the model predicts smoothed-average returns that aren't directly tradeable with standard execution.

The remaining paths to profitability:
1. **ARCX** -- lower costs (VWES=1.1 bps vs 1.97 bps), stronger OFI (r=0.688 vs 0.577)
2. **Deep ITM options** -- 0.7 bps breakeven vs 4.7 bps ATM
3. **Readability hybrid** -- Classification gate (93.88% DA at high conviction) + regression magnitude
4. **Multi-scale OFI** -- MBO profiler shows OFI R-squared increases from 33% to 50% at longer scales

### 6.4 What NOT to Do

- Do NOT pursue fine-grained point-return regression -- signal is IC=0.045, too weak
- Do NOT add more model architectures -- the ablation shows features matter more than architecture
- Do NOT build more statistical analyzers -- we have 28, the marginal value is zero
- Do NOT refactor -- the codebase is functional; experiments are the priority

---

## Appendix: 53 Temporal Features

| Group | Count | Features |
|-------|-------|----------|
| Last-timestep signals | 5 | DEPTH_NORM_OFI, TRUE_OFI, EXECUTED_PRESSURE, NET_TRADE_FLOW, VOLUME_IMBALANCE |
| Rolling means (3 signals x 3 windows) | 9 | mean(t-5), mean(t-10), mean(t-20) for top 3 |
| Rolling slopes (3 x 3) | 9 | Linear regression slope over 5, 10, 20 windows |
| Rate-of-change (3 x 3) | 9 | value[t] - value[t-w] for w=5,10,20 |
| Cross-feature products | 6 | OFI*TRUE_OFI, OFI*PRESS, OFI*VOL_IMB, etc. |
| Context | 5 | SPREAD_BPS, VOLUME_IMBALANCE, NET_ORDER_FLOW, MID_PRICE, DT_SECONDS |
| Regime indicators | 2 | TIME_REGIME, DT_SECONDS |
| Rolling volatility | 5 | std(OFI, t-10), std(TRUE_OFI, t-10), std(PRESS, t-10), RV(10), RV(5) |
| OFI momentum | 3 | mean(short) - mean(long) for 3 signals |

---

## Appendix: Reproducibility

```bash
cd lob-model-trainer
python scripts/run_simple_model_ablation.py --data-dir ../data/exports/nvda_xnas_128feat_regression
```

Results saved to `outputs/experiments/simple_model_ablation/ablation_results.json`.

---

*Generated: 2026-03-16 | Test set: 50,724 samples | All metrics independently validated*
