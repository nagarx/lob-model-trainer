# Consolidated Experiment Findings -- March 2026

**Purpose:** Single authoritative reference for all regression experiment results, validated findings, and strategic lessons. This document is the starting point for any future experiment design.

**Last updated:** 2026-04-05
**Validation status:** All metrics independently verified. E13 results verified to 10+ decimal places by independent agent computation. Universality study (10 stocks) and E16 extreme event study added 2026-04-05.

---

## 1. Experiment Inventory

### 1.1 Regression Experiments (Completed, Validated)

| ID | Model | Label Type | Horizon | Params | Test R-squared | Test IC | Test DA | Status |
|----|-------|-----------|---------|--------|---------------|---------|---------|--------|
| REG-01 | TLOB (2L, h=32) T=100 | Smoothed avg | H10 | 693K | **0.4642** | **0.6766** | **0.7494** | Best model |
| REG-02 | TLOB (2L, h=32) T=20 | Smoothed avg | H10 | 94K | 0.4385 (val) | 0.6773 (val) | 0.7479 (val) | Ablation |
| REG-03 | HMHP-R (3 horizons) | Smoothed avg | H10 primary | 171K | 0.4535 | 0.6706 | 0.7476 | Multi-horizon |
| REG-04 | TLOB (2L, h=32) T=100 | **Point-return** | H10 | 693K | **-0.0000** | **0.0008** | **0.4991** | **FAILED** |
| E4 | TLOB (2L, h=32) T=20 | Smoothed avg | H60 (5min) | 92K | 0.015 | 0.136 | 0.544 | Time-based 5s bins |
| **E5** | **TLOB (2L, h=32) T=20** | **Smoothed avg** | **H10 (10min)** | **92K** | **0.124** | **0.380** | **0.640** | **Time-based 60s bins, BEST tradeable** |
| E6 | TLOB (calibrated) | Regression | H10 (10min) | 60s bins cal x3.73 | 92K | 0.124 | 0.380 | 0.640 | -0.85% Deep ITM |
| E12 | Off-exch threshold | **Point-return** | H30 (30min) | P85 threshold | 0 | +6.90 bps | +0.178 (feat) | 0.521 | **NOT TRADEABLE** (p=0.12, val negative) |
| **E13** | **MBO threshold** | **Point-return** | **H60 (60min)** | **P85 threshold** | **0** | **+10.56 bps** | **+0.530 (feat)** | **0.549** | **p=0.032 (significant), OOS IC holds** |
| Univ. | Multi-stock IC gate | Point-return | H10, H60 | 10 stocks | 0 | N/A | 14/280=5% FPR | N/A | H0 confirmed: universal, not NVDA-specific |
| E16 | Extreme events | Point-return | H1-H60 | 10 stocks | 0 | N/A | 15 survive FDR | N/A | Sign-inconsistent, 50% val→test flip |

### 1.2 Analytical Baselines (Test Set, 50,724 Samples)

| Baseline | H10 R-squared | H10 IC | H10 DA |
|----------|--------------|--------|--------|
| Persistence (return_t = return_{t-1}) | -0.377 | 0.264 | 0.591 |
| Linear Ridge (128 features) | 0.170 | 0.433 | 0.651 |
| Single feature (DEPTH_NORM_OFI) | 0.107 | 0.335 | 0.620 |

### 1.3 Regression Backtests (IBKR 0DTE Costs)

| Hold | Threshold | Trades | Option Return |
|------|-----------|--------|---------------|
| 10 events | 0.7 bps | 4,270 | -19.75% |
| 10 events | 5.0 bps | 1,799 | -7.53% |
| 10 events | 10.0 bps | 54 | -0.35% |
| 60 events | 3.0 bps | 775 | -2.71% |
| 60 events | 10.0 bps | 45 | -0.77% |

**E5 Backtests (Round 7, 60s bins, H10=10min, Deep ITM delta=0.95):**

| Hold | Threshold | Trades | Option Return | Win Rate |
|------|-----------|--------|---------------|----------|
| 10 events (10min) | 0.7 bps | 740 | **-1.93%** | 40.1% |
| 10 events (10min) | 8.0 bps | 594 | -1.37% | 37.0% |

**E5 improvement over prior**: IC +180% (0.380 vs 0.136), return +1.75pp (-1.93% vs -3.68%). Still negative.

**E6 Backtest (Round 8, calibrated predictions, Deep ITM delta=0.95):**

| Hold | Threshold | Trades | Option Return | Win Rate |
|------|-----------|--------|---------------|----------|
| R8 (E6) | 2.0 bps (calibrated) | 741 | -0.85% | 50.6% |

### 1.4 Classification Reference (from Prior HMHP Experiments)

| Model | H10 Acc | Dir Acc (high conv) | Signal Rate |
|-------|---------|---------------------|-------------|
| HMHP 128-feat XNAS | 59.62% | 93.88% | 51.5% |
| HMHP 128-feat ARCX | 58.79% | 97.21% | 23.3% |
| HMHP 40-feat XNAS (readability) | 58.67% | 95.50% | 28.6% |

---

## 2. Validated Technical Findings

### Finding 1: OFI Features Predict Smoothed-Average Returns, NOT Point-to-Point Returns

**This is the single most important finding from the entire regression experiment series.**

| Label Type | DEPTH_NORM_OFI IC | DEPTH_NORM_OFI R-squared | TLOB Model R-squared |
|-----------|-------------------|------------------------|---------------------|
| Smoothed average (mean of next k events) | 0.309 | 0.092 | **0.464** |
| Point-to-point (price at t+k) | -0.005 | 0.0005 | **0.000** (no learning) |

**Why this happens:** OFI measures current order flow pressure. It predicts the average drift during the next 10 events (the direction the pressure pushes during those events), but NOT the specific price 10 events later (which depends on new information arriving in events 2-10).

**Correlation between smoothed and point-to-point labels:**

~~Pearson = 0.24. When smoothed says "up," point-to-point is positive only 55.8% of the time.~~

**CORRECTED (P0 Validation, 2026-03-17):** Using `LabelFactory` on aligned `forward_prices.npy` (510K samples, 35 test days, k=10, H=10):

| Metric | Previously Claimed | P0 Validated (k=10, H=10) |
|---|---|---|
| Pearson r (label-to-label) | 0.24 | **0.640** |
| P(point > 0 \| smoothed > 0) | 55.8% | **69.7%** |
| P(point > 0 \| \|smoothed\| > 5 bps) | — | **87.9%** (114K samples) |
| P(point > 0 \| \|smoothed\| > 10 bps) | — | **92.2%** (17K samples) |

The original r=0.24 was likely computed from **misaligned** data (comparing labels from two separate exports with different sampling parameters, event_count=1000 vs 100). The forward_prices approach guarantees alignment by computing both label types from the **same** mid-price trajectories.

**Critical caveat:** This measures **label-to-label** correlation (smoothed GROUND TRUTH vs point GROUND TRUTH), NOT model predictions vs execution. The effective execution correlation is lower: label_r × sqrt(model_R²) ≈ 0.640 × 0.681 ≈ 0.436. The signal transfers to execution, but with attenuation proportional to model accuracy.

**Sensitivity to smoothing window k:** r depends strongly on k (k=5→0.84, k=10→0.64, k=15→0.51). At k=10 (production config), the correlation is moderate but meaningful. At longer horizons, convergence is near-perfect (H60→0.95, H300→0.99).

**Revised implication (P0):** The label-execution mismatch is smaller than originally diagnosed at the LABEL level. High-conviction smoothed LABELS predict point-to-point direction with 88%+ accuracy.

**CRITICAL UPDATE (E1, 2026-03-17):** However, the MODEL's predictions have **r=0.013** correlation with actual consecutive-sequence price returns (essentially zero). Despite R²=0.464 on smoothed labels, the model has NO predictive power on tradeable returns. The model approximates the smoothed-average FORMULA, not future price movements.

| What | Correlation with consecutive price return |
|---|---|
| H10 smoothed label (ground truth) | r=0.562 (signal exists in labels) |
| TLOB model prediction | r=0.013 (no signal in model predictions) |

**Root cause of all negative backtests:** The model learns a statistical artifact (the smoothed average formula), not a tradeable signal. Training on point-return labels (`return_type = "point_return"`) is required to force the model to predict actual price movements.

Source: `BACKTESTER_AUDIT_PLAN.md § P0`, E1 experiment, `data/exports/nvda_xnas_128feat_regression_fwd_prices/p0_label_execution_mismatch_H10.json`

### Finding 1b: MBO Features HAVE Point-Return Signal at H=60 (E13, 2026-03-29)

**E8's conclusion that "0/67 features have IC > 0.05 for point returns" was horizon-limited.** E8 tested only H=10. E13 tested all 8 horizons [1,2,3,5,10,20,30,60] with the full 5-path framework and found:

| Feature | IC(point, H=60) | CF Ratio | Stability | Mechanism |
|---------|-----------------|----------|-----------|-----------|
| spread_bps (42) | **+0.530** | 0.95 | 100% | Spread level predicts returns |
| total_ask_volume (44) | -0.182 | 0.53 | 100% | Supply pressure |
| true_ofi (84) | **-0.146** | 1.09 | 100% | Price impact decay (mean-reversion) |
| volume_imbalance (45) | +0.126 | **0.01** | 100% | Pure forward bid/ask imbalance |
| depth_norm_ofi (85) | -0.123 | 1.31 | 100% | Normalized OFI |

**OFI sign reversal:** true_ofi IC changes from +0.26 (smoothed, contemporaneous) to -0.146 (point, mean-reversion). Over 60 minutes, OFI predicts return REVERSAL — the price impact decays and partially reverses. This is consistent with the price impact literature (Bouchaud et al., 2004).

**volume_imbalance CF=0.01** means essentially PURE forward signal — the bid/ask volume imbalance at time t predicts the point return from t to t+60 with near-zero contemporaneous contamination.

**Walk-forward (threshold strategy):** t=2.15, p=0.032 (statistically significant). BUT: fixed threshold fails OOS due to spread distribution shift.

**Walk-forward (standardized Ridge, 5 core features):** mean IC=0.139, per-fold stability=1.073 (3.3× E12's off-exchange 0.33), block stability=2.865. Alpha-robust (5/5 alphas). Coefficients stable: spread_bps β=+9.01 dominates (CV=0.26, zero sign flips). BUT: OOS pooled IC collapses — val=-0.001, test=+0.066 (not significant).

**CRITICAL FINDING — Within-Day vs Cross-Day Signal Decomposition:**

| Component | Walk-Forward | Val (OOS) | Test (OOS) | Interpretation |
|---|---|---|---|---|
| Within-day ranking IC | +0.139 | **+0.070** (t=2.76) | **+0.127** (t=6.34) | Model ranks correctly WITHIN each day |
| Between-day level correlation | — | -0.211 | — | Model assigns WRONG levels ACROSS days |
| Pooled IC (combines both) | ~0.007 | -0.001 | +0.066 | Between-day error cancels within-day signal |
| Calibration slope (pred→actual) | — | **-1.109** | **-0.111** | Negative: absolute predictions inverted |

The signal is **within-day only**. Val per-day IC=0.070 (t=2.76 after ACF correction), test=0.127 (t=6.34). All 5 decision gates pass → **first valid (non-look-ahead) GO verdict.** Grinold E[r] = +1.90 bps (val) / +5.41 bps (test) net of 1.4 bps cost — signal theoretically exceeds trading costs.

**Per-day IC distribution (E13 Phase 6, 2026-03-29):**

| Metric | Val (35 days) | Test (35 days) |
|---|---|---|
| Per-day mean IC | 0.070 (t=2.76 corrected) | 0.127 (t=6.34) |
| Per-day median IC | 0.074 | 0.125 |
| Fraction positive | 74% | 89% |
| IC [Q25, Q75] | [-0.000, +0.128] | [+0.043, +0.213] |
| IC autocorrelation | +0.166 (mild) | -0.071 (none) |
| DA (per-day mean) | 0.498 | 0.508 |
| Grinold E[r] | 3.30 bps (+1.90 net) | 6.81 bps (+5.41 net) |

**Caveats:** (1) Negative calibration slope — model's absolute values are wrong, only within-day ranking works. (2) Val DOWN-day IC=0.046 (below 0.05 threshold), but test DOWN-day IC=0.130 (symmetric). (3) Val long/short spread=-3.34 bps — sign-based trading fails on val. (4) DA near 50% — model predicts rank, not direction.

**CRITICAL: spread_bps alone massively outperforms the 5-feature Ridge (CONFIRMED E13 Phase 7, 2026-03-29).**

| Signal | Walk-Forward IC | Val Per-Day IC (OOS) | Test Per-Day IC (OOS) | Q-Spread (bps) |
|---|---|---|---|---|
| **spread_bps alone** | **0.564** | **+0.511** | **+0.601** | **+64.3** |
| 5-feature Ridge | 0.139 | +0.070 | +0.127 | +6.2 |

The Ridge DESTROYS 86% of spread_bps's within-day signal. Verdict: **SPREAD_PRIMARY** — spread_bps rank is the primary backtester signal. Deep validation confirmed:
- **Purely directional**: IC(spread, return)=+0.511, IC(spread, |return|)=-0.082. spread predicts DIRECTION, not magnitude.
- **NOT inflated by overlap**: stride-30 IC = 0.502 (same as stride-1 IC = 0.511). spread ACF(1) ≈ 0 — independent observations.
- **Bilateral**: DOWN-day IC = 0.578 > UP-day IC = 0.409 (val). Test: symmetric (0.60/0.60).
- **Temporally robust**: 94% of val days and 97% of test days have positive IC. Monthly IC: all months 0.40-0.63.
- **Label timing clean**: k=5 offset verified. 5-minute gap between feature and label start. No data leakage.

**Key lessons:** (1) Test at MULTIPLE horizons, not just H=10 (E8's error). (2) Use same IC metric for in-sample and OOS comparison (per-day mean IC, not pooled). (3) Block stability is ~sqrt(N) inflated — report per-fold as primary. (4) Per-day full-day z-scoring has LOOK-AHEAD BIAS (79.6% future data at t=50). Causal variants (expanding, trailing) collapse. Raw features with Ridge pooled-training standardization are the correct causal approach (stability=1.07). (5) Always verify within-day normalization with causal variants before claiming GO. (6) **Compare single-feature per-day IC to multi-feature model per-day IC.** When the model is WORSE, its pooled optimization is destroying within-day signal. (7) **Negative calibration slope + positive per-day IC** = rank-based trading mandatory, sign-based will fail. (8) **Validate IC mechanism**: IC(feature, |return|) ≈ 0 means directional, not volatility. (9) **Subsample at stride=30** to verify overlap isn't inflating IC.

Source: E13 in EXPERIMENT_INDEX.md, `hft-feature-evaluator/outputs/feature_oos_analysis/results.json` (SPREAD_PRIMARY)

### Finding 2: Signal Is Structural and Robust

- Walk-forward IC stability: 8.07 (mean IC / std IC) [smoothed labels, E5]
- Regime shifts: 0 / 158 folds [smoothed labels, E5]
- Per-day test R-squared: all 35 days positive, range [0.331, 0.546], CV=0.077 [smoothed labels]
- Monthly OFI-return correlation std: 0.04-0.06 (from MBO profiler, 12 months)
- **E13 point-return addition:** spread_bps sign flip rate 1-2% (UP/DOWN days), IC positive ALL 26 weeks, drift-adjusted IC loss 0.0%. Signal is equally robust for point returns at H=60.

### Finding 3: Compact Models Suffice

| Configuration | Params | Test R-squared | IC | DA |
|--------------|--------|---------------|-----|-----|
| T=100, 2 layers | 693K | 0.464 | 0.677 | 0.749 |
| T=20, 2 layers | 94K | ~0.411 | 0.674 | 0.750 |
| HMHP-R multi-horizon | 171K | 0.454 | 0.671 | 0.748 |

T=20 retains 99.6% of IC with 7.4x fewer parameters. Multi-horizon (HMHP-R) does not improve H10. Signal half-life is 5 timesteps.

### Finding 4: Multi-Horizon Regression Hurts

HMHP-R (R-squared=0.454) slightly underperforms single-horizon TLOB (R-squared=0.464). Reason: persistence dominates H60 (R-squared=0.78 baseline) and H300 (R-squared=0.957 baseline). The shared encoder gets pulled toward persistence-matching rather than innovation-capturing.

### Finding 5: Model Predictions Are Conservative

- Target std: 4.686 bps, prediction std: 3.096 bps (ratio 0.66)
- Prediction range: [-27.7, +32.3] vs target range: [-91.6, +159.8]
- Residual mean: -0.023 (zero bias)
- Residual ACF(1): 0.069 (low serial correlation)

### Finding 6: Backtests Show Signal-Execution Gap

Model directional accuracy: 74.9% (on smoothed labels). Backtest win rate: ~38% (point-to-point execution). Root cause: Finding 1 -- the label does not match the execution.

At high conviction (8-10 bps threshold), backtests approach breakeven (-0.35% to -0.93%). This suggests profitability is achievable if the execution strategy aligns with the smoothed-average prediction.

---

## 3. Feature Signal Hierarchy (from Statistical Analysis)

### 3.1 Top Features at H10 (Smoothed Return, Test Set)

| Rank | Feature | Index | R-squared | IC | DA |
|------|---------|-------|-----------|-----|-----|
| 1 | DEPTH_NORM_OFI | 85 | 0.107 | 0.335 | 0.620 |
| 2 | VOLUME_IMBALANCE | 45 | 0.099 | -0.346 | 0.377 |
| 3 | TRUE_OFI | 84 | 0.066 | 0.342 | 0.620 |
| 4 | NET_TRADE_FLOW | 56 | 0.058 | 0.264 | 0.600 |
| 5 | TRADE_ASYMMETRY | 88 | 0.058 | 0.264 | 0.600 |
| 6 | EXECUTED_PRESSURE | 86 | 0.050 | 0.288 | 0.600 |

### 3.2 Feature Predictive Decay

IC drops from 0.33 at lag 0 to 0.00 at lag 10 within the sequence window. Half-life: 5 timesteps.

### 3.3 Level vs Change

All top OFI features strongly prefer level over change (11x ratio). The model should process raw feature levels, NOT first differences.

### 3.4 Regime Conditioning

| Condition | Regime Dependence Score | Top Feature Stable? |
|-----------|------------------------|---------------------|
| Time of day | 0.23 (moderate) | Yes -- OFI dominant in all regimes |
| Volatility | 0.22 (moderate) | Yes -- but delta should scale with vol |
| Spread | 0.22 (moderate) | Yes |
| Activity | 0.22 (moderate) | Yes -- high activity R-squared 52% higher |

---

## 4. Configuration Reference

### 4.1 Best Regression Model (REG-01)

```yaml
# lob-model-trainer/configs/experiments/nvda_tlob_128feat_regression_h10.yaml
model:
  model_type: tlob
  input_size: 128
  task_type: regression
  regression_loss_type: huber
  regression_loss_delta: 5.0
  tlob_hidden_dim: 32
  tlob_num_layers: 2
  tlob_num_heads: 2
  tlob_use_bin: true

train:
  task_type: regression
  batch_size: 128
  learning_rate: 5.0e-4
  weight_decay: 0.01
  scheduler: cosine
  gradient_clip_norm: 1.0
  seed: 42

data:
  data_dir: "../data/exports/nvda_xnas_128feat_regression"
  labeling_strategy: regression  # smoothed_return (default)
  horizon_idx: 0                 # H10
  normalization: hybrid
```

### 4.2 Feature Extractor Config

```toml
# feature-extractor-MBO-LOB/configs/nvda_xnas_128feat_regression.toml
[labels]
strategy = "regression"
horizons = [10, 60, 300]
# return_type NOT specified → defaults to SmoothedReturn
# SmoothedReturn: r = (1/k) * sum(m_{t+i} - m_t)/m_t * 10000
```

### 4.3 Huber Delta Calibration

| Horizon | Return Std (train) | IQR | Recommended Delta |
|---------|-------------------|-----|-------------------|
| H10 | 6.73 bps | 6.66 bps | 5.0 bps |
| H60 | 19.83 bps | 21.44 bps | 10.7 bps |
| H300 | 44.38 bps | 49.03 bps | 24.5 bps |

### 4.4 IBKR Cost Model (316 Real NVDA Option Fills + 19 Stock Fills)

Source: `IBKR-transactions-trades/COST_AUDIT_2026_03.md` (316 fills), `IBKR_REAL_WORLD_TRADING_REPORT.md` (318 fills).

**Option Costs (Quick Flip, No Theta):**

| Component | ATM Call | ATM Put | Deep ITM (delta=0.95) |
|-----------|---------|---------|----------|
| Half-spread (OPRA median) | $0.015 | $0.010 | $0.005 |
| Full spread RT (per contract) | $3.00 | $2.00 | $1.00 |
| Commission RT | $1.40 | $1.40 | $1.40 |
| **Total RT (no theta)** | **$4.40** | **$3.40** | **$2.40** |
| **Breakeven (no theta)** | **4.9 bps** | **3.8 bps** | **1.4 bps** |

Commission: $0.70/contract median (TIERED pricing). 0DTE-specific: $0.63/contract ($1.26 RT). 99% limit orders.

**60-Minute Hold Costs (E13 Phase 8 Strategy):**

| Component | ATM Call | Deep ITM | **Equity (100 shares)** |
|-----------|---------|----------|------------------------|
| Spread RT | $3.00 | $1.00 | ~$1.00 |
| Commission RT | $1.40 | $1.40 | ~$0.30 (TIERED, limit) |
| BSM Theta (60 min) | $25.20 | $6.34 | **$0** |
| **Total RT** | **$29.60** | **$8.74** | **$1.30** |
| **Breakeven** | **33.6 bps** | **5.2 bps** | **0.7 bps** |

BSM theta uses ATM approximation (N'(0)=0.399). For deep ITM (delta=0.95), N'(d1)=0.103 → actual theta is 3.87x lower than ATM. The $6.34 value is the CORRECTED deep ITM theta. Stock commission from 19 IBKR fills: $0.35 median per fill, $0.06/share median. TIERED rate: $0.0035/share.

**Key lesson (E13 Phase 8):** For 60-minute holds, theta is 73% of deep ITM cost and 85% of ATM cost. Equity trading eliminates theta entirely. At 8 bps gross alpha per trade, equity (breakeven 0.7 bps) retains 91% of alpha; deep ITM (5.2 bps) retains 35%; ATM (33.6 bps) is impossible.

---

## 5. What We Learned (Lessons for Future Experiments)

### Lesson 1: Always validate label-execution alignment BEFORE running experiments
We trained a model to R-squared=0.464 on smoothed labels, then discovered in backtesting that smoothed labels don't match point-to-point execution. This could have been caught by computing the correlation between label types before training.

### Lesson 2: Point-to-point returns at short horizons have near-zero feature correlation
OFI features predict average drift, not endpoints. This is a physical limitation of flow-based features at event-based sampling. Point-to-point prediction requires either (a) much longer horizons where the average and endpoint converge, or (b) different features that predict specific future states.

### Lesson 3: Compact models work as well as large ones
T=20 with 94K params retains 99.6% of rank-ordering ability. The 5M-param TLOB that caused OOM was 50x oversized. Start small, scale up only if needed.

### Lesson 4: Multi-horizon hurts when persistence dominates
HMHP-R's shared encoder is pulled toward persistence-matching at H60/H300, degrading H10. Multi-horizon regression requires persistence-subtracted (innovation) targets.

### Lesson 5: High-conviction classification IS the viable trading approach
HMHP at agreement=1.0, confirmation>0.65 achieves 93.88% directional accuracy. This readability gate, combined with proper holding and cost management, is closer to profitability than regression magnitude prediction.

### Lesson 6: Per-level OF is not useful in cumulative-per-window architecture (2026-03-17)
Kolm et al.'s 20-dim per-level OF (bid/ask separate) has IC=0.0001 for point-returns at event_count=100. The cumulative-per-window representation destroys the per-event temporal dynamics that Kolm's LSTM exploits. Scalar DEPTH_NORM_OFI (IC=0.070) works because depth normalization creates a mean-reverting signal. See `reports/kolm_of_experiment_2026_03_17.md`.

### Lesson 7: Shorter horizons DO increase point-return IC (2026-03-17)
Scalar DEPTH_NORM_OFI IC for point-returns: 0.005 at event_count=1000, 0.070 at event_count=100 (14x improvement). The signal exists at short horizons but is weak (~51.5% directional accuracy, not tradeable alone).

### Lesson 8: ARCX + fine-grained does NOT fix the point-return IC problem (2026-03-17)

E3 tested the "three converging advantages" hypothesis: ARCX (OFI r=0.688), lower cost (1.10 bps), and fine-grained sampling (event_count=100). Result: **0/93 features have IC > 0.05 for point-return labels at ANY horizon (H10, H60, H300).** Best IC = 0.035 (signed_mp_delta_bps at H10). ARCX IC is ~40% better than XNAS E2 (0.035 vs 0.025) but still far below the 0.05 threshold. This CONCLUSIVELY eliminates the exchange/resolution variation approach. The problem is architectural: accumulated LOB/MBO snapshot features cannot predict point-to-point returns. See `EXPERIMENT_INDEX.md § E3`.

### Lesson 9: OFI persistence is REAL but NOT accessible through event-based sampling (2026-03-17)

F1 validated that our accumulated-event-window OFI (feature indices 84/85) has **zero autocorrelation** between consecutive samples: ACF = 0.021 at K=1, decreasing to 0.000 at K=50, across all 233 days. The profiler's ACF=0.266 at 5-minute scale reflects **fixed-time integration** — a fundamentally different measurement. Our event-based architecture (1000 events/window, non-overlapping) produces statistically independent OFI values. To access OFI persistence, we need: (1) time-based sampling, (2) rolling OFI with exponential decay, or (3) per-event architecture. See `EXPERIMENT_INDEX.md § F1`.

### Lesson 10: Readability hybrid does not beat pure regression (2026-03-16)
The ReadabilityHybridStrategy (HMHP direction + Ridge magnitude) at best config (-2.67%) performs WORSE than pure Ridge at 10bps threshold (-1.14%). Both models predict smoothed-average returns; layering a direction gate adds no value. See BACKTEST_INDEX Round 5.

### Lesson 11: Post-hoc calibration improves win rate but model lacks magnitude ranking (2026-03-20)
**Lesson 11: Post-hoc calibration improves win rate but model lacks magnitude ranking.** E6 variance-matching calibration (scale x3.73) improved win rate from 40.1% to 50.6% (+10.5pp) and best return from -1.93% to -0.85%. However, higher prediction thresholds DECREASE win rate (50.6% at 2 bps -> 45.5% at 20 bps), proving the model's magnitude predictions are uninformative. The model predicts DIRECTION (DA=64%) but cannot distinguish large moves from small ones. The E5 report's label-level threshold analysis (90.8% win rate at |label|>10 bps) does NOT transfer to model predictions. Next approaches must improve magnitude ranking, not just scale.

### Lesson 12: Simplified PnL model gives OPPOSITE results from full 0DTE backtester (2026-03-21)
**Lesson 12: Never use `sign(pred) * label - cost` for regime/gating validation.** Phase A (E7) tested regime gating with a simplified PnL model and found +5.9pp WR improvement for |depth_norm_ofi| gating. Deep validation revealed two critical flaws: (1) DNO gating is a proxy for prediction magnitude (partial r=0.001 after controlling for |cal_ret|), and (2) the simplified PnL model has a built-in bias toward any filter that selects larger |labels| — magnitude filtering gives WR=75.0% in simplified PnL but DECREASES option WR from 50.6% to 45.5% in the full 0DTE backtester (E6 R8). The two models give OPPOSITE conclusions. Any future regime/filter evaluation MUST use the full backtester, not simplified PnL. Regime-conditional IC does vary (0.299-0.446 across DNO quartiles), but whether this translates to real trading improvement is unknown without full backtester testing. See `reports/phase_a_regime_validation_2026_03.md`.

### Lesson 13: Model predicts smoothing residual, not point direction — ROOT CAUSE of all negative backtests (2026-03-21)
**Lesson 13: The E5/E6 TLOB model has DA=48.3% on point returns (below random).** E8 decomposition shows the model captures the smoothing residual (R²=45% with residual, R²=0.02% with point return). When smoothed and point labels disagree on direction (19.5% of samples), the model follows the smoothing artifact with 90.1% fidelity and predicts point direction correctly only 9.1% of the time. 0/67 non-price features have IC > 0.05 for point returns at 60s bins. DA <= 49.1% at ALL horizons H=1 to H=50, and no conditioning strategy achieves DA > 52%. The 23 features with IC > 0.05 in initial analysis were ALL price-level artifacts (stock price mean reversion). ALL four fix paths (label alignment, feature engineering, horizon adjustment, conditioning) are blocked. The model requires fundamental redesign — any future model must be trained on labels whose direction aligns with tradeable execution. See `EXPERIMENT_INDEX.md § E8`.

### Lesson 14: Off-exchange features — no signal at H=10, but genuine signal at H=1 and H=60 (2026-03-21, UPDATED 2026-03-22)
**Lesson 14 (REVISED): At H=10 (10 min), 0/11 off-exchange features pass IC > 0.05.** E9 tested 11 features from XNAS.BASIC CMBP-1 (35 test days, 8,337 samples). Best at H=10: subpenny_intensity IC=+0.048 (marginal). Mroib IC=+0.021, ACF(1)=0.050 (no persistence). Cross-sectional weekly Mroib does NOT transfer to intraday single-stock.

**HOWEVER, E9 cross-validation horizon sweep revealed genuine signal at different horizons:**
- **trf_signed_imbalance IC=+0.103 at H=1 (1 min)** — the strongest point-return predictive signal found in 15 experiments. Bootstrap 95% CI: [+0.019, +0.060] at H=10. Partially contemporaneous (lagged IC drops from 0.040 to 0.014 at lag=1).
- **subpenny_intensity IC=+0.104 at H=60 (1 hr)** — INCREASES with horizon (slow-moving state variable, NOT contemporaneous). Remarkably stable across lags.
- **dark_share IC=+0.051 at H=1** — short-term regime signal, sign-flips at H=60 (mean-reverts).

The original E9 conclusion evaluated only H=10. The signals exist but at different timescales than initially tested. Architecture plan created: `off-exchange-approach/` (7 design docs, 6,982 lines). `basic-quote-processor` Rust crate designed with 34 features. See `off-exchange-approach/04_FEATURE_SPECIFICATION.md` Section 10 for full cross-validation results.

---

## 6. Next Steps (Priority Order, Updated 2026-04-03)

| Priority | Direction | Rationale |
|----------|-----------|-----------|
| ~~1~~ | ~~Re-run raw Ridge with per-day OOS IC~~ | **DONE (E13 Phase 6, 2026-03-29).** Val per-day IC=0.070 (t=2.76 corrected), test=0.127 (t=6.34). All 5 gates pass → **GO**. Grinold E[r]=+1.90 bps net of cost. |
| ~~1~~ | ~~Verify spread_bps single-feature OOS per-day IC~~ | **DONE (E13 Phase 7, 2026-03-29).** CONFIRMED: val IC=+0.511, test IC=+0.601. Purely directional (IC(abs)≈0). Robust to stride-30 subsampling. Bilateral (DOWN-day IC=0.578). Verdict: **SPREAD_PRIMARY**. |
| ~~1~~ | ~~0DTE backtester with spread_bps rank signal~~ | **DONE (E13 Phase 8, 2026-03-29).** Signal has GENUINE alpha: gross equity PnL = +$1,390 (174 trades, val). BUT 0DTE theta overwhelms: ATM $21/trade vs $8/trade alpha. Deep ITM corrected: +$434 (+0.43%). **Test: ALL configs negative.** Root cause: 0DTE 60-min theta, NOT signal quality. |
| ~~1~~ | ~~Equity-level backtest (no options, no theta)~~ | **DONE (E13 Phase 9, 2026-03-30).** Trailing-rank signal + IBKR equity costs ($1.70 RT, 0.97 bps BE). Val: +$69 (+0.07%, noise). **Test: ALL configs negative.** Root cause: IC=0.51 is cross-sectional (245 bins/day), NOT temporal (4 traded bins/day). At traded bins: IC collapses to ~0, DA=48%, quintile spread inverts. **E13 CONCLUDED — spread_bps is not tradeable.** |
| ~~1~~ | ~~Fundamental reassessment: new signal sources~~ | **PARTIALLY DONE (E14, 2026-03-30).** Off-exchange signals tested with E13 Lesson 33 gate filter. 6 features × 3 horizons × 2 splits. Bootstrap CIs on ALL stride-60 ICs cross zero. subpenny_intensity sign-flips between val and test. **No off-exchange feature has statistically significant temporal IC at the trading cadence.** |
| ~~2~~ | ~~Off-exchange signals at optimal horizons~~ | **DONE (E14, 2026-03-30).** subpenny_intensity (ACF(60)=0.35-0.41, persistent) and bbo_update_rate (ACF(60)=0.47-0.60, most persistent) both passed individual gates on test-only but ALL bootstrap CIs cross zero. Per-day IC SNR = 0.1-0.2. Test-only passes are noise. |
| ~~1~~ | ~~Long-horizon intraday: morning signal → afternoon return (E15)~~ | **DONE (E15, 2026-04-04). ALL FAIL — In-sample artifact, negative OOS. Independently validated.** Full-sample ACF(1)=-0.27 at H=300 (return std=180 bps, NOT 44 bps). "Bet against yesterday" P&L=+5345 bps full-sample. BUT: **train P&L=+6011, val=-233, test=-909 bps.** All profit in-sample. Driven by April 8-9 tariff outlier (43% of ACF signal) + fat tails (kurtosis=29). Without top-20 days: P&L=-1998. Morning OFI partial IC=-0.21 (full-sample) but ZERO walk-forward IC (Model B IC=-0.009). Model C had look-ahead bias. **E15 Lesson: per-split validation FIRST. Fat tails make full-sample stats unreliable.** |
| ~~1~~ | ~~Cross-asset / longer-horizon signals~~ | **PARTIALLY DONE (Universality Study, 2026-04-05).** 10 NASDAQ stocks tested at 60s cadence. 14/280 = exactly 5% FPR. Zero cross-stock consistency. Finding is universal. Cross-asset lead-lag (NVDA vs AMD/QQQ) remains untested. |
| ~~1~~ | ~~Extreme event / rare-signal approach~~ | **DONE (E16, 2026-04-05).** Tail-conditional returns (top/bottom 2-10%, 5 features) across 10 stocks. 15/1,656 survive BH FDR but sign-inconsistent and 50% val→test sign flips. Not tradeable. |
| 1 | **Cross-asset lead-lag** | NVDA MBO flow → QQQ/SMH return prediction. Cross-asset information propagation is slower. Requires multi-asset data. |
| 2 | **Streaming/co-location architecture** | OFI IC=0.86 concurrent is a massive signal. Requires sub-ms latency to capture. Infrastructure gap, not signal gap. |
| 3 | **Accept: single-stock directional microstructure prediction is not retail-viable** | **17 experiments (E1-E16 + universality), 49 lessons, 10 stocks.** Every statistical escape hatch closed: linear IC, MI, dCor, TE, regime-conditional, extreme-conditional. OFI lag-1 r < 0.006 at ALL scales (1s-5min). |
| ~~4~~ | ~~Per-day z-score experiment~~ | **DONE then RETRACTED (E13 Phase 4-5, 2026-03-29).** Full-day z-score GO was look-ahead bias. Trailing z-score (K=60) collapses to stability=0.20. Expanding (causal) collapses to -0.19. Per-day z-scoring is NOT the fix. |
| ~~5~~ | ~~Trailing-window z-score~~ | **DONE (E13 Phase 5).** Trailing K=60 stability=0.20. K=30: 0.02. K=120: -0.09. All fail. Signal does NOT survive causal within-day normalization. |
| ~~4~~ | ~~Fix label-execution alignment (E8 ROOT CAUSE)~~ | **PARTIALLY RESOLVED by E13.** E8's root cause (model predicts smoothing residual) is bypassed by using point-return labels directly. E13 proved MBO features have signal with point returns at H=60. The label-execution gap still matters for comparing model R² to backtest P&L, but the fundamental "zero signal" barrier is broken. |
| ~~5~~ | ~~Off-exchange GradBoost model~~ | **CONCLUDED (E12, 2026-03-28).** Off-exchange spread_bps IC=0.178 is real but NOT tradeable. One-sided, noise-dominated, regime-dependent (crisis months). Val/test negative. GradBoost IC=0.036. MBO signal is 3× stronger and bilateral — pursue MBO instead. |
| ~~6~~ | ~~Alternative data sources (F4)~~ | **TESTED/MARGINAL (E9-E12).** Off-exchange features have IC for point returns but fail OOS tradeability test. Options flow untested but lower priority than MBO model. |
| ~~7~~ | ~~Time-based sampling~~ | **DONE (2026-03-18).** Sampler trait, TimeBasedSampler, CompositeSampler implemented. e5_timebased_60s export is the basis for E13. |
| ~~8~~ | ~~ARCX cross-exchange (E3)~~ | **FAILED (2026-03-17).** 0/93 features IC>0.05. |
| ~~9~~ | ~~Per-level OF at shorter horizons~~ | **FAILED (2026-03-17).** Kolm OF IC=0.0001. |
| ~~10-13~~ | ~~Previous failed approaches~~ | See EXPERIMENT_INDEX.md lessons 1-18. |

---

### Finding 7: Zero Predictive IC Is Universal Across 10 NASDAQ Stocks (2026-04-05)

**Universality Study**: 10 stocks spanning the full NASDAQ microstructure spectrum (spread 1.5-12 bps, volume 1.5M-25M/day, price $12-$525, beta 0.55-1.85). Each exported at 60s bins, 134 days, 98 features. IC gate: 14 non-price features vs point returns at H=10 and H=60.

**Result: 14/280 tests "pass" = exactly 5.0% — the expected false positive rate.** Zero features pass for 2+ stocks. All cross-stock stability ratios < 2.0 (best: 0.870, threshold: 2.0). The finding is NOT NVDA-specific.

**Mechanism confirmed**: OFI has IC=0.73-0.86 with concurrent returns (t-1→t) but IC=0.00-0.03 with future returns (t→t+h) across PEP, HOOD, MRNA. The smoothed return label's past window creates IC=0.30-0.39 with the smoothing residual — a measurement artifact, not genuine prediction. OFI ACF at 60s is 0.004-0.116 across stocks (essentially zero persistence).

Source: `hft-feature-evaluator/reports/UNIVERSALITY_STUDY_2026_04.md`

### Finding 8: Extreme Event Tail-Conditional Returns Are Marginal (E16, 2026-04-05)

**E16**: Tested whether MBO features at extreme percentiles (top/bottom 2-10%) predict non-zero forward returns that aggregate IC would average away. 5 features × 3 percentiles × 2 tails × 6 horizons × 10 stocks × 2 splits = 3,600 tests. Per-day block bootstrap CIs (2000 resamples).

**Result: 15/1,656 survive BH FDR at α=0.10 (test split).** The 9.6% raw hit rate (vs 5% expected) suggests weak tail effects exist, but:
- Sign-inconsistent across stocks (spread_bps bottom: PEP negative, FANG positive)
- 50% val→test sign flips (8 of 16 conditions significant in both splits)
- Most effects 0.5-2 bps (barely above 0.7 bps equity cost)
- 3 cross-stock consistent conditions, all marginal

**Implication**: The "rare event signal" escape hatch is substantially closed. Aggregate IC=0 is not hiding large tradeable tail effects.

Source: `hft-feature-evaluator/outputs/e16_extreme_events/e16_results.json`

### Complete Evidence Stack (Updated 2026-04-05)

| Escape Hatch | Experiment | Result | Status |
|---|---|---|---|
| Linear IC (aggregate) | E2, E3, E8, Universality | IC=0 for all features at 60s cadence | **CLOSED** |
| Non-linear (MI, dCor) | E13 Path 2 | 0/89 features | **CLOSED** |
| Transfer entropy | E13 Path 3b | 0 pairs | **CLOSED** |
| Regime-conditional IC | E13 Path 4 | 86/89 pass but cross-sectional only | **CLOSED** (E13 Phase 9) |
| Deep learning on smoothed | REG-01 to E6 | R²=0.464, DA=48.3% on point returns | **CLOSED** (E8) |
| ARCX + fine-grained | E3 | 0/93 IC>0.05 | **CLOSED** |
| Off-exchange | E9, E12, E14 | Bootstrap CIs cross zero OOS | **CLOSED** |
| Long-horizon morning→afternoon | E15 | In-sample artifact | **CLOSED** |
| Multi-stock universality | Universality Study | 14/280 = 5% FPR, 10 stocks | **CLOSED** |
| Extreme events at tails | E16 | 15 FDR-survive but unstable | **CLOSED** |
| Shorter cadence (30s, 15s, 5s) | Profiler multi-scale | lag-1 r < 0.006 at ALL scales | **CLOSED** |

---

## Appendix: Data Provenance

| Artifact | Location | Description |
|----------|----------|-------------|
| Raw MBO data | `data/XNAS_ITCH/NVDA/mbo_2025-02-03_to_2026-01-07/` | 239 files, .mbo.dbn.zst |
| Smoothed regression export | `data/exports/nvda_xnas_128feat_regression/` | 13 GB, 233 days, 266,608 sequences |
| Point-return regression export | `data/exports/nvda_xnas_128feat_regression_pointreturn/` | 13 GB, 233 days, 266,841 sequences |
| TLOB T=100 checkpoint | `outputs/experiments/nvda_tlob_128feat_regression_h10/checkpoints/best.pt` | Epoch 6, 693K params |
| HMHP-R checkpoint | `outputs/experiments/nvda_hmhp_regression_h10_primary/checkpoints/best.pt` | Epoch 16, 171K params |
| Regression signals | `outputs/experiments/nvda_tlob_128feat_regression_h10/signals/test/` | 50,724 samples |
| Backtest results | `lob-backtester/outputs/backtests/tlob_regression_h10_*.json` | H10 and H60 hold |
| Statistical analysis | `lob-dataset-analyzer/outputs/regression_deep_train/` | 7 JSON reports |
| MBO profiler | `mbo-statistical-profiler/output_xnas_full/` | 13 tracker JSONs |
| E3 ARCX IC gate export | `data/exports/e3_ic_gate/` | 35 days, 395,967 seqs, event_count=100 |
| E3 IC results | `data/exports/e3_ic_gate/ic_results.json` | Full per-feature IC for H10/H60/H300 |
| IBKR calibration | `IBKR-transactions-trades/IBKR_REAL_WORLD_TRADING_REPORT.md` | 318 fills |
| MBO point-return export | `data/exports/e5_timebased_60s_point_return/` | Derived from forward_prices, 8 horizons |
| Off-exchange export | `data/exports/basic_nvda_60s/` | 233 days, 34 features, point returns |
| E10 classification | `hft-feature-evaluator/classification_table_lean.json` | Off-exchange 5-path results |
| E11 classification | `hft-feature-evaluator/classification_table_mbo_lean.json` | MBO SmoothedReturn 5-path results |
| E13 classification | `hft-feature-evaluator/classification_table_mbo_point_return_lean.json` | MBO PointReturn 5-path results |
| E12 pre-training analysis | `hft-feature-evaluator/outputs/pre_training_analysis/` | 10-domain off-exchange analysis |
| E12 signal diagnostics | `hft-feature-evaluator/outputs/signal_diagnostics/` | 7-diagnostic off-exchange suite |
| E13 signal diagnostics | `hft-feature-evaluator/outputs/signal_diagnostics_mbo/` | 7-diagnostic MBO suite |

---

| Universality exports | `data/exports/universality_{symbol}_60s/` (10 stocks) | 134 days each, 60s bins, 98 features |
| Universality IC results | `hft-feature-evaluator/outputs/universality_{symbol}_ic/` | Per-stock IC gate results |
| Universality consolidated | `hft-feature-evaluator/outputs/universality_consolidated_results.json` | Cross-stock summary |
| Universality report | `hft-feature-evaluator/reports/UNIVERSALITY_STUDY_2026_04.md` | Full analysis |
| E16 results | `hft-feature-evaluator/outputs/e16_extreme_events/e16_results.json` | Extreme event study |
| E16 script | `hft-feature-evaluator/scripts/e16_extreme_event_study.py` | Reproducible analysis |

---

*This document supersedes individual experiment reports in `reports/`. For detailed per-experiment methodology, see those files. For the experiment ledger, see `EXPERIMENT_INDEX.md`.*
