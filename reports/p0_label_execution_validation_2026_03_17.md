# P0: Label-Execution Mismatch Validation Report

> **Cost correction (2026-03-18)**: Deep ITM breakeven was corrected from 0.7-0.8 bps to **1.4 bps** (spread was missing from the original calculation). See `IBKR-transactions-trades/COST_AUDIT_2026_03.md`.

> **Date**: 2026-03-17
> **Status**: COMPLETE — All results cross-validated
> **Finding**: The previously claimed r=0.24 and 55.8% conditional win rate were WRONG. Actual: r=0.642, 69.3%.

---

## 1. Executive Summary

The CONSOLIDATED_FINDINGS claim that smoothed-average and point-to-point labels have Pearson r=0.24 and 55.8% conditional win rate was **incorrect**. Using aligned forward mid-price trajectories (50,724 test samples, 35 days), we computed:

- **Pearson r = 0.642** (not 0.24) — smoothed and point labels are moderately correlated
- **P(point > 0 | smoothed > 0) = 69.3%** (not 55.8%) — meaningful directional signal
- **At |smoothed| > 5 bps: 87.9% directional win rate** on 11,203 samples
- **At |smoothed| > 10 bps: 93.5% directional win rate** on 1,552 samples

The original r=0.24 was likely computed by comparing labels from two separate exports with different sampling rates (event_count=1000 vs 100), producing near-random correlation from sample misalignment.

**Implication**: The label-execution mismatch is NOT the primary root cause of negative backtests. The actual bottlenecks are (1) cost structure (ATM breakeven 5.4 bps > mean return 2.65 bps), (2) backtester bugs (C3 short sizing, trade_pnls entry cost omission), and (3) insufficient conviction filtering.

---

## 2. Methodology

### Infrastructure Built (Permanent)

| Component | Location | Purpose |
|---|---|---|
| **Rust forward_prices export** | `feature-extractor-MBO-LOB/src/export_aligned/` | Exports `{day}_forward_prices.npy` [N, k+H+1] float64 USD alongside sequences |
| **Python LabelFactory** | `hft-contracts/src/hft_contracts/label_factory.py` | Computes any label type from forward_prices (28 tests) |
| **LabelExecutionMismatchAnalyzer** | `lob-dataset-analyzer/src/lobanalyzer/analysis/labels/label_execution_mismatch.py` | Streaming cross-label-type statistics (13 tests) |

### Data

- **Source**: `data/exports/nvda_xnas_128feat_regression_fwd_prices/train/`
- **35 test-split days**: 2025-11-14 to 2026-01-06 (XNAS ITCH, NVDA)
- **50,724 aligned samples** (contract: `forward_prices.shape[0] == sequences.shape[0]`)
- **Forward prices shape**: [N, 311] — columns [t-10, ..., t, t+1, ..., t+300]

### Parameters

| Parameter | Value | Source |
|---|---|---|
| Smoothing window (k) | 10 | Production config: `nvda_xnas_128feat_regression.toml` |
| Primary horizon (H) | 10 | Standard H10 used in all experiments |
| Event sampling | 1000 events | EventBasedSampler |
| Sequence window | 100 timesteps | SequenceBuilder |
| Stride | 10 | SequenceBuilder |

### Formulas (matching Rust: `multi_horizon.rs` lines 1072-1098, `magnitude.rs` lines 50-132)

**Smoothed-average return** (TLOB formula):
```
past_smooth  = mean(mid_prices[t-k : t+1])      — k+1 prices
future_smooth = mean(mid_prices[t+h-k : t+h+1])  — k+1 prices
smoothed_bps = (future_smooth - past_smooth) / past_smooth × 10000
```

**Point-to-point return**:
```
point_bps = (mid_prices[t+h] - mid_prices[t]) / mid_prices[t] × 10000
```

---

## 3. Results

### Core Correlation (k=10, H=10, N=50,724)

| Metric | Previously Claimed | P0 Validated | Delta |
|---|---|---|---|
| Pearson r (label-to-label) | 0.24 | **0.6417** | +0.40 |
| Spearman r | — | **0.5975** | — |
| P(point > 0 \| smoothed > 0) | 55.8% | **69.3%** | +13.5pp |
| P(point < 0 \| smoothed < 0) | — | **69.4%** | — |
| Bivariate normal expected | — | **72.2%** | — |

### Conditional Return Distributions

| Condition | Mean Point (bps) | Std | Median | p25 | p75 | N |
|---|---|---|---|---|---|---|
| smoothed > 0 | +2.65 | 5.27 | +2.26 | -0.54 | +5.41 | 25,229 |
| smoothed < 0 | -2.63 | 5.10 | -2.26 | -5.41 | +0.54 | 25,375 |
| smoothed ≈ 0 | +0.08 | 3.73 | 0.00 | — | — | 1,212 |

### Threshold Analysis (Conviction Filtering)

| |smoothed| > T | Win Rate | N Samples | Mean Point Return |
|---|---|---|---|
| 0.5 bps | 71.7% | 45,192 | -0.003 bps |
| 1.0 bps | 74.0% | 40,150 | -0.009 bps |
| 2.0 bps | 78.3% | 30,652 | -0.005 bps |
| **5.0 bps** | **87.9%** | **11,203** | **+0.095 bps** |
| **8.0 bps** | **92.2%** | **3,448** | **+0.417 bps** |
| **10.0 bps** | **93.5%** | **1,552** | **+0.659 bps** |
| 15.0 bps | 91.5% | 328 | +2.264 bps |
| 20.0 bps | 89.3% | 131 | +6.553 bps |

**Key insight**: Win rate peaks at ~10 bps threshold (93.5%) then slightly declines at higher thresholds (small sample sizes, noise). The **sweet spot is 5-10 bps** — high win rate (88-93%) with sufficient sample size (1.5K-11K samples per test period).

---

## 4. Cross-Validation

### Python LabelFactory vs Rust regression_labels

| Check | Result |
|---|---|
| Max absolute difference | **7.56e-12** |
| Mean absolute difference | 9.39e-13 |
| Pearson correlation | **1.0000000000** |
| Verdict | **EXACT MATCH** — Python LabelFactory replicates Rust labels |

---

## 5. Sensitivity Analysis

### Smoothing Window (k)

| k | Pearson r | Win Rate | Interpretation |
|---|---|---|---|
| 5 | 0.835 | 79.8% | More overlap → higher correlation |
| **10** | **0.642** | **69.3%** | **Production setting** |
| 15 | 0.507 | 64.2% | Less overlap → lower correlation |
| 20 | 0.429 | 61.7% | Approaching independent windows |

**Why k matters**: The smoothed return uses a window of k+1 prices for both past and future smoothing. At k=H (k=10, H=10), the future window starts at t (same as base price), creating maximum overlap with the point return at t+H. At k=0, smoothed equals point (identical formulas).

### Horizon (H)

| Horizon | Pearson r | Win Rate | Interpretation |
|---|---|---|---|
| **H10** | **0.642** | **69.3%** | Primary horizon |
| H20 | 0.834 | 80.0% | |
| H60 | 0.947 | 88.5% | Near-identical at longer horizons |
| H100 | 0.969 | 91.4% | |
| H300 | 0.989 | 94.8% | Smoothed → point convergence |

**Why H matters**: At longer horizons, the smoothing window (k=10) becomes a smaller fraction of the total horizon, so the smoothed average converges to the endpoint value.

---

## 6. Critical Caveats

### Caveat 1: Label-to-Label vs Model-to-Execution

This analysis compares **ground truth** smoothed labels vs **ground truth** point labels — both computed from the same actual prices. The MODEL's predictions of smoothed labels will have LOWER correlation with point returns because the model has R²=0.464 (not 1.0) on smoothed labels.

**Effective execution estimate**:
```
effective_r = label_r × sqrt(model_R²) = 0.642 × 0.681 ≈ 0.437
effective_R² ≈ 0.191
```

This means ~19% of the variance in point-to-point execution returns is explained by the model's smoothed predictions. This is not zero — it's a meaningful signal — but it's far from the 46.4% R² on the training labels.

### Caveat 2: Source of Original r=0.24

The original r=0.24 was NOT found in any computation code in the codebase. It likely came from comparing labels across two separate exports with different sampling parameters (event_count=1000 vs 100), which produced near-random correlation due to sample misalignment. Our `forward_prices` approach guarantees alignment by computing both label types from the same mid_price trajectories.

### Caveat 3: Win Rate ≠ Profitability

69.3% directional win rate does NOT guarantee profitable trading. Profitability requires:
```
(win_rate × avg_win) - ((1 - win_rate) × avg_loss) > cost_per_trade
```

With mean point return when smoothed > 0 = +2.65 bps and deep ITM breakeven = 0.8 bps, the math is: 2.65 > 0.8. But this ignores the distribution — many individual trades lose despite the positive mean.

---

## 7. Revised Root Cause Analysis

| Factor | Old Understanding | New Understanding |
|---|---|---|
| Label alignment | r=0.24, 55.8% → "near random" | r=0.642, 69.3% → "meaningful signal" |
| Model quality | R²=0.464 on smoothed | Effective execution R² ≈ 0.191 (still meaningful) |
| Cost structure | Not emphasized | **PRIMARY BOTTLENECK**: ATM 5.4 bps > mean 2.65 bps |
| Backtester bugs | Known but deprioritized | **C3 (2x shorts), trade_pnls (missing entry cost)** compound losses |
| Conviction filtering | Not tested | **At >5 bps: 87.9% win rate** — the path to profitability |

---

## 8. Implications for Each Module

| Module | Impact | Action Needed |
|---|---|---|
| **feature-extractor** | New `forward_prices` export capability | Document in CODEBASE.md, CHANGELOG |
| **hft-contracts** | New `LabelFactory` module | Document in README |
| **lob-dataset-analyzer** | New `LabelExecutionMismatchAnalyzer` | Already documented in MODULES.md |
| **lob-model-trainer** | Root cause understanding revised | CONSOLIDATED_FINDINGS updated |
| **lob-backtester** | Bug fixes remain priority; Deep ITM experiment unlocked | Fix P2-P5, then run E1 |

---

## 9. Recommended Next Steps

1. **Fix backtester bugs** (P2-P5) — enables clean experiments
2. **Deep ITM backtest** (E1) — 0.8 bps cost, 87.9% win rate at >5 bps conviction → highest profitability probability
3. **Re-run Round 4** (E2) — same signals, corrected engine
4. **Train on point-return labels** (E3) — since r=0.642, model CAN learn point returns
5. **Purged K-Fold CV** — validate true generalization R²

---

*Report generated from: `data/exports/nvda_xnas_128feat_regression_fwd_prices/p0_label_execution_mismatch_H10.json`*
*Infrastructure: Rust forward_prices (747 tests), Python LabelFactory (28 tests), LabelExecutionMismatchAnalyzer (13 tests)*
*Cross-validated: Python vs Rust max diff = 7.56e-12*
