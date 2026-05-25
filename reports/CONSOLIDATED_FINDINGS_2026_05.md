# Consolidated Experiment Findings -- May 2026

**Purpose:** Single authoritative reference for all regression experiment results, validated findings, anti-patterns, architectural hardening shipped, and strategic lessons through 2026-05-05. This document is the starting point for any future experiment design and supersedes `CONSOLIDATED_FINDINGS_2026_03.md` (last updated 2026-04-08, did not reflect R9-R15 / Phase Y / Phase Z / Stage 8 / 4-agent post-compact audit).

**Last updated:** 2026-05-05 (post-compact 7-agent investigation)
**Validation status:** Quantitative metrics through E16 + Universality Study independently verified prior to 2026-04-08. **NEW (post-Phase-O cycle):** R9-R15 metrics empirically reproduced via canonical Phase Q.6.5.B chain on v3p0 baseline corpus. Phase Y producer-side validated with bit-exact `model_config_hash=de47c0ef49abc0ef...` between Phase X.1 v2 checkpoint sidecar and Phase Y `signal_metadata.json`. **CRITICAL CAVEAT (#PY-8):** R9-R15 metrics are single-point estimates without bootstrap CIs; rankings are NOT statistically validated.

> **Reading order**: this file is the up-to-date narrative reference. For ground-truth ledger entries, see `lob-model-trainer/EXPERIMENT_INDEX.md` and `lob-backtester/BACKTEST_INDEX.md`. For session-state and recommended next actions, see `NEW_SESSION_HANDOFF_2026_05_05.md`. For deferred items, see `PHASE_P_BACKLOG.md`. For 8 active forensic-audit bugs that may invalidate prior results, see `lob-model-trainer/reports/TRAINING_PIPELINE_FORENSIC_AUDIT_2026_04_26.md`.

---

## 0. Document Map

| Section | Topic |
|---|---|
| 1 | Experiment inventory (E1-E16 + R8-R15 + Phase Q.6.5 Stages 1-8 — full table) |
| 2 | Validated technical findings (Findings 1-8, preserved from 2026-03) |
| 3 | Feature signal hierarchy (smoothed return, lagged behavior, regime conditioning) |
| 4 | Configuration reference (best-of TLOB, IBKR cost calibration, Huber δ table) |
| 5 | Lessons (50+ at 2026-04-08 + Lessons 50-75 from Phase Q.6.5 Stages 1-8) |
| 6 | What NOT To Do — anti-patterns (preserved + 15 new caveats) |
| 7 | Pipeline architecture validation — Phase O Cycle 1 / Phase Y / Phase Z / Stage 8 (NEW) |
| 8 | Statistical-rigor caveat (NEW — #PY-8) |
| 9 | Feature evaluation gap (NEW — #PY-9) |
| 10 | Forensic-audit bug ledger (NEW — #PY-10) |
| 11 | Standing process mandates (NEW) |
| 12 | Next steps (REVISED per `NEW_SESSION_HANDOFF_2026_05_05.md` §8) |
| Appendix | Data provenance (UPDATED with v3p0 paths + R-series signal directories) |
| References | Cross-doc links |

---

## 1. Experiment Inventory

### 1.1 Pre-Phase-O Regression Experiments (Completed, Validated through 2026-04-08)

| ID | Model | Label Type | Horizon | Params | Test R² | Test IC | Test DA | Status |
|----|-------|-----------|---------|--------|---------|---------|---------|--------|
| REG-01 | TLOB (2L, h=32) T=100 | Smoothed avg | H10 | 693K | **0.4642** | **0.6766** | **0.7494** | Best event-based 128-feat |
| REG-02 | TLOB (2L, h=32) T=20 | Smoothed avg | H10 | 94K | 0.4385 (val) | 0.6773 (val) | 0.7479 (val) | Ablation |
| REG-03 | HMHP-R (3 horizons) | Smoothed avg | H10 primary | 171K | 0.4535 | 0.6706 | 0.7476 | Multi-horizon |
| REG-04 | TLOB T=100 | Point-return | H10 | 693K | -0.0000 | 0.0008 | 0.4991 | **FAILED** |
| E4 | TLOB T=20 | Smoothed avg | H60 (5min) | 92K | 0.015 | 0.136 | 0.544 | 5s bins |
| **E5** | **TLOB T=20** | **Smoothed avg** | **H10 (10min)** | **92K** | **0.124** | **0.380** | **0.640** | **60s bins, BEST tradeable pre-O** |
| E6 | TLOB calibrated x3.73 | Smoothed avg | H10 (10min) | 92K | 0.124 | 0.380 | 0.640 | -0.85% Deep ITM |
| E12 | Off-exch threshold | Point-return | H30 | P85 | +6.90 bps IC | 0.178 (feat) | 0.521 | NOT TRADEABLE |
| **E13** | **MBO threshold** | **Point-return** | **H60** | **P85** | **+10.56 bps** | **+0.530 (feat)** | **0.549** | p=0.032; cross-sectional only |
| Univ. | Multi-stock IC gate | Point-return | H10/H60 | 10 stocks | 14/280=5% FPR | -- | -- | H0 confirmed |
| E16 | Extreme events | Point-return | H1-H60 | 10 stocks | 15 survive FDR | -- | -- | sign-flip 50% val→test |

### 1.2 Post-Phase-O v3p0 Regression Experiments (Phase Q.6.5 Stages 1-8 / R9-R15, 2026-05-04 to 2026-05-05)

All trained on `e5_timebased_60s_v3p0` (NVDA XNAS, 230 days post Phase O Cycle 1, 60s bins, 98 features, 8,085 test samples, schema 3.0).

| Stage / R# | Model | Loss | Params | Test IC | Test R² | Test DA | Best OptRet | At Threshold | Trades | WR | Status |
|------------|-------|------|--------|---------|---------|---------|-------------|--------------|--------|----|----|
| Stage 1 | TemporalRidge sklearn | -- (Ridge) | -- | 0.3289 | 0.1037 | 0.6206 | -0.46% | max_conv_20bps | 175 | -- | ✅ first sklearn v3p0 |
| **R9 / Stage 2** | TLOB compact no-CVML | Huber δ=12.6 | 92,690 | **0.3747** | **0.1379** | **0.6419** | -1.39% | very_high_10bps | 473 | -- | ✅ first PyTorch v3p0 |
| R10 / Stage 3 | TLOB+CVML | Huber δ=12.6 | 120,179 | 0.3464 | 0.1164 | 0.6294 | +0.56% | high_conv_8bps | 561 | -- | reproduces "CVML doesn't transfer" |
| R11 / Stage 4 | TLOB+GMADL+CVML | GMADL a=10 b=1.5 | 120,179 | -0.0054 | -0.0013 | 0.5014 | 0.00% | (no trades) | 0 | N/A | NEG CONTROL — mean-collapse |
| R12 / Stage 5 | TLOB calibrated (Stage 2 ckpt) | Huber + variance_match | 92,690 | 0.3747 | 0.1379 | 0.6419 | -3.07% | very_high_10bps | 698 | 47.0% | calibration scale=3.174x |
| R13 / Stage 6 | HMHP-R cascading [H10/H60/H300] | Huber, weights 0.50/0.25/0.15/0.10 | 169,239 | **0.3561** (H10) | 0.1147 | 0.6302 | -1.06% | max_conv_20bps | 48 | 39.6% | first HMHP-R v3p0 + Phase S `mean`-pool |
| **R14 / Stage 7** | **TemporalGradBoost sklearn** | **Huber α=0.9** | **200 trees** | **0.2842** | **0.0796** | **0.5948** | **-0.04%** | **max_conv_20bps** | **128** | **50.0%** | **STRONGEST P&L** (sample-of-1) |
| **R15 / Stage 8** | TLOB v3p0 export-only re-run of R9 | (no retrain) | 92,690 | 0.3747 | 0.1379 | 0.6419 | -1.39% | very_high_10bps | 473 | -- | ✅ Phase Y producer-side validated bit-exact |

**Cross-stage observation (challenges "higher IC → better P&L"):** TemporalGradBoost has the LOWEST headline IC (0.2842) of any non-failure stage but the BEST OptRet (-0.04% essentially break-even). Phase Y composability LOCKED across 5 axes (data / architectural / loss-tuning / horizons-set / calibration) by R9-R13 fingerprint discrimination. **However**, ALL R9-R15 results are SAMPLE-OF-1 single-point estimates. See §8 (Statistical Rigor Caveat).

### 1.3 Analytical Baselines (Test Set, pre-Phase-O 50,724 samples; v3p0 8,085 samples)

| Baseline | H10 R² | H10 IC | H10 DA | Notes |
|----------|--------|--------|--------|-------|
| Persistence (return_t = return_{t-1}) | -0.377 | 0.264 | 0.591 | pre-Phase-O 128-feat |
| Linear Ridge (128 features) | 0.170 | 0.433 | 0.651 | pre-Phase-O |
| Single feature (DEPTH_NORM_OFI) | 0.107 | 0.335 | 0.620 | pre-Phase-O |
| TemporalRidge sklearn v3p0 (53 temporal features) | 0.104 | **0.329** | 0.621 | post-Phase-O 98-feat — Stage 1 |

### 1.4 Pre-Phase-O Backtests (Reference)

**E5 Round 7** (60s bins, H10=10 min, Deep ITM δ=0.95):

| Hold | Threshold | Trades | OptRet | WR |
|------|-----------|--------|--------|----|
| 10 events | 0.7 bps | 740 | -1.93% | 40.1% |
| 10 events | 8.0 bps | 594 | -1.37% | 37.0% |

**E6 Round 8** (calibrated, Deep ITM δ=0.95):

| Hold | Threshold | Trades | OptRet | WR |
|------|-----------|--------|--------|----|
| -- | 2.0 bps cal | 741 | -0.85% | 50.6% |

### 1.5 Classification Reference (HMHP, pre-Phase-O)

| Model | H10 Acc | Dir Acc (high conv) | Signal Rate |
|-------|---------|---------------------|-------------|
| HMHP 128-feat XNAS | 59.62% | 93.88% | 51.5% |
| HMHP 128-feat ARCX | 58.79% | 97.21% | 23.3% |
| HMHP 40-feat XNAS (readability) | 58.67% | 95.50% | 28.6% |

---

## 2. Validated Technical Findings (Preserved from 2026-04-08 + cross-checked 2026-05-05)

### Finding 1: OFI Features Predict Smoothed-Average Returns, NOT Point-to-Point Returns

**This remains the single most important finding from the entire regression series.**

| Label Type | DEPTH_NORM_OFI IC | DEPTH_NORM_OFI R² | TLOB Model R² (event-based) |
|-----------|-------------------|-------------------|---------------------------|
| Smoothed average | 0.309 | 0.092 | **0.464** |
| Point-to-point | -0.005 | 0.0005 | **0.000** |

**Mechanism:** OFI measures contemporaneous order flow pressure. It predicts the average drift during the next k events (the direction the pressure pushes), but NOT the specific endpoint price (which depends on new information arriving in events 2 through k).

**P0 Validated correlation (corrected 2026-03-17 from misaligned 0.24 to aligned 0.642):**

| Metric | Value |
|---|---|
| Pearson r (label-to-label, k=10, H=10) | **0.640** |
| P(point > 0 \| smoothed > 0) | **69.7%** |
| P(point > 0 \| \|smoothed\| > 5 bps) | **87.9%** (114K samples) |
| P(point > 0 \| \|smoothed\| > 10 bps) | **92.2%** (17K samples) |

**Critical caveat:** This is LABEL-to-LABEL correlation, not model-to-execution. Effective execution correlation ≈ 0.640 × √R² ≈ 0.436.

**E1 critical finding (2026-03-17):** Despite REG-01 R²=0.464 on smoothed labels, the model's predictions have **r=0.013** with consecutive-sequence price returns — essentially zero. The model approximates the smoothed-average FORMULA, not future price movements. Training on `point_return` labels is required to force prediction of actual price movements.

Source: `BACKTESTER_AUDIT_PLAN.md` (now at `lob-backtester/BACKTESTER_AUDIT_PLAN.md`), E1 experiment, P0 validation report.

**Wiki references (theory backbone consulted):** hft-wiki `theory:order_flow_imbalance_family` consolidates the Cont/Kukanov/Stoikov 2014 OFI + Xu/Gould/Howison 2019 MLOFI + Kolm/Turiel/Westray 2023 DOFI lineage with 3 verbatim Cont equations + page_refs; the entry's anti-drift caveat block documents this finding (concurrent r=0.577 vs predictive r<0.006) as the canonical refutation of any naive predictive-OFI claim.

### Finding 1b: MBO Features HAVE Point-Return Signal at H=60 (Cross-Sectional Only) — E13

E8 concluded "0/67 features have IC > 0.05 for point returns" testing only H=10. E13 expanded to all 8 horizons and found:

| Feature | IC(point, H=60) | CF Ratio | Stability | Mechanism |
|---------|-----------------|----------|-----------|-----------|
| spread_bps (42) | **+0.530** | 0.95 | 100% | Spread level predicts returns |
| total_ask_volume (44) | -0.182 | 0.53 | 100% | Supply pressure |
| true_ofi (84) | **-0.146** | 1.09 | 100% | Price impact decay (mean-reversion) |
| volume_imbalance (45) | +0.126 | **0.01** | 100% | Pure forward bid/ask imbalance |
| depth_norm_ofi (85) | -0.123 | 1.31 | 100% | Normalized OFI |

**OFI sign reversal:** true_ofi flips from +0.26 (smoothed, contemporaneous) to -0.146 (point, mean-reversion) over 60 minutes — consistent with Bouchaud et al. 2004 price impact decay literature.

**Within-day vs cross-day decomposition** (E13 Phase 6, 2026-03-29):

| Component | Walk-Forward | Val (OOS) | Test (OOS) | Interpretation |
|---|---|---|---|---|
| Within-day ranking IC | +0.139 | +0.070 (t=2.76) | +0.127 (t=6.34) | Model ranks correctly WITHIN each day |
| Between-day level correlation | -- | -0.211 | -- | Model assigns WRONG levels ACROSS days |
| Pooled IC | ~0.007 | -0.001 | +0.066 | Between-day error cancels within-day signal |

**spread_bps single-feature beats 5-feature Ridge** (E13 Phase 7):

| Signal | Walk-Forward IC | Val Per-Day IC (OOS) | Test Per-Day IC (OOS) |
|---|---|---|---|
| **spread_bps alone** | **0.564** | **+0.511** | **+0.601** |
| 5-feature Ridge | 0.139 | +0.070 | +0.127 |

Ridge destroys 86% of spread_bps's within-day signal. **However** (E13 Phase 9): At the 4 traded bins/day cadence, IC collapses to ~0 and DA=48%; quintile spread inverts. **Result: spread_bps is NOT tradeable**, confirming cross-sectional ≠ temporal.

**Wiki references:** hft-wiki `theory:order_flow_imbalance_family` (the Cont 2014 lineage's price-impact-decay literature — Bouchaud et al. 2004 sister-citation in the entry's body — predicts the OFI sign-reversal observed here over 60-minute horizons).

### Finding 2: Signal Is Structural (Smoothed Labels)

Walk-forward IC stability: 8.07 (mean IC / std IC); 0/158 fold regime shifts; per-day test R² all 35 days positive (range [0.331, 0.546], CV=0.077). E13 spread_bps: positive IC ALL 26 weeks; sign flip rate 1-2% (UP/DOWN days).

### Finding 3: Compact Models Suffice

| Configuration | Params | Test R² | IC | DA |
|--------------|--------|---------|-----|-----|
| T=100, 2 layers (event-based) | 693K | 0.464 | 0.677 | 0.749 |
| T=20, 2 layers (event-based) | 94K | ~0.411 | 0.674 | 0.750 |
| HMHP-R multi-horizon | 171K | 0.454 | 0.671 | 0.748 |
| **TLOB compact v3p0 60s/98-feat** | **92,690** | **0.138** | **0.375** | **0.642** |

T=20 retains 99.6% of event-based IC with 7.4× fewer params. Time-based 60s 98-feat baseline IC=0.375 is comparable to event-based when normalized for sampling regime.

### Finding 4: Multi-Horizon Regression Hurts H10

Pre-Phase-O HMHP-R R²=0.454 < TLOB R²=0.464 (event-based, single-horizon TLOB wins).
Post-Phase-O R13 HMHP-R IC=0.356 < R9 TLOB IC=0.375 (Δ=-0.019, same direction but tighter margin on time-based v3p0).

Reason: persistence dominates H60 (R²=0.78) and H300 (R²=0.957); shared encoder pulled toward persistence-matching rather than innovation-capturing.

**Wiki references:** hft-wiki `theory:tlob_dual_attention` (Berti & Kasneci 2025 dual-attention transformer — the load-bearing architecture for HMHP-R encoder); `theory:bin_bilinear_normalization` (Tran et al. ICPR 2020 BiN — the load-bearing per-feature normalization layer fronting TLOB's input).

### Finding 5: Model Predictions Are Conservative

Target std 4.686 bps, prediction std 3.096 bps (ratio 0.66). Prediction range [-27.7, +32.3] vs target [-91.6, +159.8]. Residual mean -0.023 (zero bias), ACF(1) 0.069. **Pattern reproduced on v3p0**: Stage 5 calibration computed scale_factor=3.174× (pred std 8.72 → target std 27.68 bps).

### Finding 6: Backtests Show Signal-Execution Gap

Model directional accuracy 74.9% on smoothed labels; backtest win rate ~38% on point-to-point execution. Root cause: Finding 1 (label-execution mismatch). At high conviction (8-10 bps threshold), backtests approach breakeven (-0.35% to -0.93% pre-Phase-O).

**Post-Phase-O confirmation:** R14 GradBoost achieves 50.0% WR @ max_conv_20bps with -0.04% OptRet (128 trades) — closest to break-even on v3p0. Sample-of-1; statistical validation pending.

### Finding 7: Zero Predictive IC Is Universal Across 10 NASDAQ Stocks (2026-04-05)

10 stocks, 60s bins, 134 days each, 98 features. **14/280 tests pass = exactly 5.0%** (expected null FPR). Zero features pass for 2+ stocks. Cross-stock stability ratios all < 2.0.

OFI concurrent IC=0.73-0.86; predictive IC=0.00-0.03. Smoothed return label past-window creates IC=0.30-0.39 with smoothing residual — measurement artifact, not genuine prediction. OFI ACF at 60s = 0.004-0.116 (essentially zero persistence).

Source: `hft-feature-evaluator/reports/UNIVERSALITY_STUDY_2026_04.md`.

**Wiki references:** hft-wiki `theory:order_flow_imbalance_family` (the universality of zero predictive IC at 60s cadence across all 10 stocks is the canonical refutation locked in the entry's body); `theory:dcor_szekely_2007` (E13 Path 2 non-linear independence screening — Szekely-Rizzo-Bakirov 2007 — yielded 0/89 features for cross-stock subset, confirming linear-IC=0 is not hiding a non-linear signal).

### Finding 8: Extreme Event Tail-Conditional Returns Are Marginal (E16)

5 features × 3 percentiles × 2 tails × 6 horizons × 10 stocks × 2 splits = 3,600 tests with per-day block bootstrap CIs.
**15/1,656 survive BH FDR at α=0.10** (test split). 9.6% raw hit rate vs 5% expected suggests weak tail effects, but:
- Sign-inconsistent across stocks (spread_bps: PEP negative, FANG positive)
- 50% val→test sign flips (8/16 conditions significant in both splits)
- Most effects 0.5-2 bps (barely above 0.7 bps equity cost)

**Implication**: rare-event signal escape hatch substantially closed. Aggregate IC=0 not hiding tradeable tail effects.

**Wiki references:** hft-wiki `theory:vpin_easley_toxicity` (Easley/López de Prado/O'Hara 2012 VPIN — the load-bearing theoretical framework for extreme-event toxicity measurement; pipeline anchor XNAS=0.298 vs ARCX=0.079 contextualizes the per-stock variation reported above).

### Complete Evidence Stack (Updated 2026-05-05)

| Escape Hatch | Experiment | Result | Status |
|---|---|---|---|
| Linear IC (aggregate) | E2, E3, E8, Universality | IC=0 for all features at 60s cadence | **CLOSED** |
| Non-linear (MI, dCor) | E13 Path 2 | 0/89 features | **CLOSED** |
| Transfer entropy | E13 Path 3b | 0 pairs | **CLOSED** |
| Regime-conditional IC | E13 Path 4 | 86/89 pass but cross-sectional only | **CLOSED** (E13 Phase 9) |
| Deep learning on smoothed | REG-01 to E6, R9-R13 | R²=0.46 / IC=0.375 reproduces; DA on point returns ≤ 49% | **CLOSED** (E8 + R9 reproduces) |

**Wiki references for Evidence Stack:** Non-linear independence screening (E13 Path 2 row): hft-wiki `theory:dcor_szekely_2007` (Szekely-Rizzo-Bakirov 2007 distance correlation, Theorem 3(i) iff-independence characterization). Regime-conditional IC (E13 Path 4 row): hft-wiki `theory:bocpd_adams_mackay_2007` (Adams-MacKay 2007 Bayesian Online Changepoint Detection — sister paradigm to CUSUM for regime-shift identification; sleeper status disclosed in entry body). Deep learning on smoothed (REG-01 to R13 row): hft-wiki `theory:tlob_dual_attention` (TLOB IS the architecture that captures the smoothing-residual artifact per E8 root-cause analysis).
| ARCX + fine-grained | E3 | 0/93 IC>0.05 | **CLOSED** |
| Off-exchange | E9, E12, E14 | Bootstrap CIs cross zero OOS | **CLOSED** |
| Long-horizon morning→afternoon | E15 | In-sample artifact | **CLOSED** |
| Multi-stock universality | Universality Study | 14/280 = 5% FPR, 10 stocks | **CLOSED** |
| Extreme events at tails | E16 | 15 FDR-survive but unstable | **CLOSED** |
| Shorter cadence (30s, 15s, 5s) | Profiler multi-scale | lag-1 r < 0.006 at ALL scales | **CLOSED** |
| **Architectural variants on v3p0** | R9-R13 | TLOB / TLOB+CVML / HMHP-R / GradBoost: best OptRet -0.04% (sample-of-1) | **NEW POST-PHASE-O — pending statistical rigor (§8)** |

---

## 3. Feature Signal Hierarchy (from Statistical Analysis)

### 3.1 Top Features at H10 (Smoothed Return, Test Set)

| Rank | Feature | Index | R² | IC | DA |
|------|---------|-------|----|----|----|
| 1 | DEPTH_NORM_OFI | 85 | 0.107 | 0.335 | 0.620 |
| 2 | VOLUME_IMBALANCE | 45 | 0.099 | -0.346 | 0.377 |
| 3 | TRUE_OFI | 84 | 0.066 | 0.342 | 0.620 |
| 4 | NET_TRADE_FLOW | 56 | 0.058 | 0.264 | 0.600 |
| 5 | TRADE_ASYMMETRY | 88 | 0.058 | 0.264 | 0.600 |
| 6 | EXECUTED_PRESSURE | 86 | 0.050 | 0.288 | 0.600 |

### 3.2 Feature Predictive Decay

IC drops from 0.33 at lag 0 to 0.00 at lag 10 within the sequence window. Half-life: 5 timesteps.

### 3.3 Level vs Change

All top OFI features prefer level over change (11× ratio). Process raw feature levels, NOT first differences.

### 3.4 Regime Conditioning

| Condition | Regime Dependence Score | Top Feature Stable? |
|-----------|------------------------|---------------------|
| Time of day | 0.23 | OFI dominant in all regimes |
| Volatility | 0.22 | Yes — but δ should scale |
| Spread | 0.22 | Yes |
| Activity | 0.22 | High activity R² 52% higher |

**See §9** (Feature Evaluation Gap) for the major caveat: 30 of 50 experimental features (indices 98-147) have NEVER been formally IC-evaluated. Existing FeatureSet registry entries are HAND-CURATED, not produced by canonical Phase 4 producer path.

---

## 4. Configuration Reference

### 4.1 Best Pre-Phase-O Regression Model (REG-01)

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
  labeling_strategy: regression  # smoothed_return default
  horizon_idx: 0                 # H10
  normalization: hybrid
```

### 4.2 Best Post-Phase-O Regression Model (R9 / Stage 2 v3p0 baseline)

```yaml
# lob-model-trainer/configs/experiments/nvda_first_pytorch_v3p0.yaml
# (inherits from e5_60s_huber_nocvml.yaml; only output_dir differs)
data:
  data_dir: "../data/exports/e5_timebased_60s_v3p0"  # 230 days, schema 3.0
model:
  model_type: tlob
  input_size: 98
  regression_loss_delta: 12.6  # Calibrated from kurtosis=26.5 (E5 SWEEP)
  tlob_hidden_dim: 32
  tlob_use_cvml: false
```

Test metrics: IC=0.3747, R²=0.1379, DA=0.6419, best OptRet=-1.39% @ very_high_10bps (473 trades).

### 4.3 Strongest Post-Phase-O Trading P&L (R14 / Stage 7)

```yaml
# lob-model-trainer/configs/experiments/nvda_first_temporal_gradboost_v3p0.yaml
model:
  model_type: temporal_gradboost  # framework=sklearn
  n_estimators: 200
  max_depth: 5
  learning_rate: 0.05
  subsample: 0.8
  min_samples_leaf: 50
  loss_type: huber
  huber_alpha: 0.9  # sklearn alpha quantile, NOT bps
data:
  data_dir: "../data/exports/e5_timebased_60s_v3p0"
  rolling_windows: [3, 5, 10]  # adapted for 60s/98-feat (was [5,10,20] for 128-feat)
  signal_indices: [85, 84, 86, 56, 45]
```

Test metrics: IC=**0.2842** (lower than Ridge 0.329) but OptRet=**-0.04% @ max_conv_20bps** (128 trades, 50.0% WR — STRONGEST P&L in cycle).

### 4.4 Huber Delta Calibration

| Sampling | Horizon | Return Std (train) | Kurtosis | Recommended δ |
|----------|---------|-------------------|----------|---------------|
| Event-based (1000 evt) | H10 | 6.73 bps | ~50 | 5.0 bps |
| Event-based (1000 evt) | H60 | 19.83 bps | ~50 | 10.7 bps |
| Event-based (1000 evt) | H300 | 44.38 bps | ~50 | 24.5 bps |
| **Time-based 60s** | **H10** | **27.68 bps** | **26.5** | **12.6 bps** |

**IMPORTANT**: copying δ from event-based to time-based produces wrong threshold; always re-calibrate from per-export kurtosis.

### 4.5 IBKR Cost Model (316 NVDA Option Fills + 19 Stock Fills)

**Quick-Flip 0DTE (no theta):**

| Component | ATM Call | ATM Put | Deep ITM (δ=0.95) |
|-----------|---------|---------|-------------------|
| Half-spread (OPRA median) | $0.015 | $0.010 | $0.005 |
| Full RT spread (per contract) | $3.00 | $2.00 | $1.00 |
| Commission RT | $1.40 | $1.40 | $1.40 |
| **Total RT** | **$4.40** | **$3.40** | **$2.40** |
| **Breakeven** | **4.9 bps** | **3.8 bps** | **1.4 bps** |

Source: `IBKR-transactions-trades/COST_AUDIT_2026_03.md` (316 option fills); `IBKR_REAL_WORLD_TRADING_REPORT.md` (318 fills).

**60-Minute Hold Costs:**

| Component | ATM Call | Deep ITM | Equity (100 shares) |
|-----------|---------|----------|---------------------|
| Spread RT | $3.00 | $1.00 | ~$1.00 |
| Commission RT | $1.40 | $1.40 | ~$0.30 (TIERED) |
| BSM Theta (60 min) | $25.20 | $6.34 | $0 |
| **Total RT** | **$29.60** | **$8.74** | **$1.30** |
| **Breakeven** | **33.6 bps** | **5.2 bps** | **0.7 bps** |

Equity trading eliminates theta. At 8 bps gross alpha: equity (BE 0.7 bps) retains 91%; deep ITM (5.2 bps) 35%; ATM (33.6 bps) impossible.

---

## 5. Lessons (Pre-Phase-O 1-49 Cross-Referenced; New Lessons 50-75 from Phase Q.6.5)

### Pre-Phase-O Lessons (1-49 — full text in `EXPERIMENT_INDEX.md` and `CONSOLIDATED_FINDINGS_2026_03.md`)

Key lessons cited frequently:

- **Lesson 1**: Always validate label-execution alignment BEFORE training (E1 root cause).
- **Lesson 2**: Point-to-point returns at short horizons have near-zero feature correlation.
- **Lesson 3**: Compact models work as well as large ones (T=20 retains 99.6% of T=100 IC).
- **Lesson 5**: High-conviction classification IS the viable trading approach (HMHP 93.88% DA).
- **Lesson 8**: ARCX + fine-grained does NOT fix point-return IC.
- **Lesson 11**: Calibration improves WR but model lacks magnitude ranking.
- **Lesson 12**: Simplified PnL gives OPPOSITE results from full 0DTE backtester.
- **Lesson 13**: Model predicts smoothing residual, not point direction (E8 ROOT CAUSE).
- **Lesson 14 (REVISED)**: Off-exchange has IC at H=1 / H=60 but NOT H=10.
- **Lesson 35 (E15)**: Per-split validation FIRST. Fat tails make full-sample stats unreliable. April 8-9 outlier drove 43% of ACF.
- **Lesson 47 (Universality)**: Zero predictive IC is universal across 10 stocks (5.0% FPR exactly).
- **Lesson 48**: All statistical relationship types between MBO features and point returns at 60s are now closed.
- **Lesson 49**: Reducing sampling cadence below 60s will not help (OFI lag-1 r < 0.006 at ALL scales 1s-5min).

### Post-Phase-O Lessons (50-75 — full text in `EXPERIMENT_INDEX.md:1899-2163`)

#### Stage 1 (TemporalRidge sklearn v3p0) lessons not numbered separately — embedded in Stages 2-3 lessons.

#### Stage 2 (R9: TLOB compact v3p0) — 2026-05-04 night

- **Lesson 50**: Phase Q.6.5 + Phase X.2.A.1+A.2 + Phase Q+S+X.1 v2 closures EMPIRICALLY VALIDATED end-to-end. Test metrics on v3p0 reproduce pre-Phase-O E5 R7 baseline within ±5pp/±10pp/±5pp tolerance (R²/IC/DA all within band on first attempt — NO corrupt-module propagation across the 4-cycle refactor).

- **Lesson 51**: Determinism + reproducibility chain validated. `compatibility_fingerprint` for the same (config + data) deterministically produces identical 64-hex SHA-256 across in-process AND canonical-script runs (per Phase X.1 v2 + Phase Q.6.5.A SSoT design). Cross-experiment Phase Y composability is now structurally locked for sklearn AND pytorch.

- **Lesson 52**: Defense-in-depth Phase X.2.A.2 SSoT shim works as intended. The trainer's `_validate_day_metadata` shim correctly delegates to `hft_contracts.validation.validate_day_metadata`. 230 days of v3p0 schema=3.0 metadata pass; the shim would fail-loud on any pre-Phase-O schema=2.2 metadata. Architectural pattern: validate at boundary, fail-loud per hft-rules §8.

- **Lesson 53**: Cosmetic finding (NOT a blocker, logged for Phase X.3 silent-default sweep): `signal_metadata.json::compatibility.horizons` falls back to classification defaults `[10,20,50,100,200]` when `data.labels.horizons` is empty. Affects BOTH sklearn AND pytorch fingerprints. Does NOT affect training (regression labels[:, 0] = H10 = 10 minutes correct). Phase X.3 candidate.

#### Stage 3 (R10: TLOB+CVML v3p0) — 2026-05-04 night

- **Lesson 54**: CVML on v3p0 EMPIRICALLY REPRODUCES CLAUDE.md prior finding "CVML doesn't transfer to low-dim/small-sample regime". CVML test_ic=0.3464 < no-CVML 0.3747 (Δ=-0.028) confirms slightly worse — same direction as CLAUDE.md (CVML 0.373 vs baseline 0.380 prior).

- **Lesson 55**: Phase Y composability EMPIRICALLY VERIFIED via live experiment fingerprint differentiation. Stage 2 (no-CVML) and Stage 3 (CVML) on IDENTICAL data + IDENTICAL labels + IDENTICAL normalization produce IDENTICAL `compatibility_fingerprint=67c8ff36...` but DIFFERENT `model_config_hash` (`de47c0ef...` vs `3ced8443...`). Future Phase Y `experiment_provenance_hash` will correctly differentiate sklearn vs pytorch vs CVML-toggle experiments.

- **Lesson 56**: Pre-flight 4-agent adversarial validation gate caught a non-blocker (false-negative attribute name) that would have been a debugging time-sink. The parallel adversarial validation is not just rigor — it actively improved correctness.

#### Stage 4 (R11: TLOB+GMADL+CVML negative control) — 2026-05-05 morning

- **Lesson 57**: GMADL a=10, b=1.5 EMPIRICALLY REPRODUCES the documented "complete failure, mean-collapse" mode on v3p0. Predictions std=0.000077 bps (only 6 unique values across 8,085 samples) — textbook mean-collapse. Pipeline correctly produces a CORRECT NEGATIVE CONTROL.

- **Lesson 58**: Phase X.1 v2 `_LOSS_TUNING_KEYS` denylist correctness EMPIRICALLY VERIFIED IN PRODUCTION. Stage 3 (Huber) and Stage 4 (GMADL) produce IDENTICAL `model_config_hash=3ced8443...` despite different loss functions. Combined with Stage 2-vs-3 architectural-axis verification (Lesson 55), Phase Y composability is now FULLY VALIDATED across BOTH the architectural axis AND the loss-tuning axis.

- **Lesson 59**: EarlyStopping + ModelCheckpoint(save_best_only=True) protected the checkpoint from late-epoch corruption. Stage 4 collapsed essentially from epoch 1 (best); patience=5 fired at epoch 6. CLAUDE.md predicted "loss inverts at epoch 16" — Stage 4 collapsed earlier but the pipeline correctly halted training and preserved best weights.

- **Lesson 60**: SignalManifest does NOT validate prediction variance/all-zeros (Agent 4 flagged informational). Stage 4's predictions have std=0.000077 bps yet signal_metadata validates and exports cleanly. Backtester correctly produced 0 trades when |pred|=0.9 < 1.4 bps cost gate. **Phase X.3 candidate**: add `prediction_stats.std` minimum threshold to SignalManifest validation per hft-rules §8.

- **Lesson 61**: Pre-flight 4-agent adversarial validation gate was extremely valuable. **Agent 3 ran the EXACT independent simulation that predicted Stage 4's hashes BEFORE training started** — empirical Stage 4 hashes EXACTLY matched Agent 3's predictions, providing falsifiable hypothesis test.

#### Stage 5 (R12: TLOB calibrated variance-match) — 2026-05-05 morning

- **Lesson 62**: Calibration code path EMPIRICALLY VALIDATED end-to-end via canonical scripts. `--calibrate variance_match` correctly invokes `lobtrainer/calibration/variance.py:294-295`; produces `calibrated_returns.npy`; embeds `calibration_method: "variance_match"` in compat block; backtester auto-detects via `BacktestData.from_signal_dir` per Phase II D10 fix at `vectorized.py:180-199`. ZERO new SSoT primitives needed.

- **Lesson 63**: CLAUDE.md Lesson 51 "calibration improves WR but lacks magnitude ranking" REPRODUCED on v3p0. Stage 5 backtest (R12): trades fire at ALL thresholds; WR 44-47% vs R9 uncalibrated 40.1%; but best OptRet still negative (-3.07% at very_high_10bps); higher thresholds (15-20 bps) PRODUCE WORSE results, confirming model lacks true magnitude-ranking.

- **Lesson 64**: Phase II compat_fingerprint correctly differentiates calibrated vs uncalibrated artifacts. Stage 2 (calibration_method=None) → `67c8ff36...`; Stage 5 (variance_match) → `9a72a760...`. The `calibration_method` field is correctly NOT in `_LOSS_TUNING_KEYS` denylist — it's a SIGNAL-side artifact axis, not a loss-tuning training axis.

#### Stage 6 (R13: HMHP-R v3p0) — 2026-05-05 morning

- **Lesson 65**: Pre-flight 4-agent adversarial validation gate caught a CRITICAL BLOCKER bug. Agent 2 module-wiring audit identified `schema.py:1758-1761` silently dropping `hmhp_loss_weights` for `hmhp_regression` model_type (only classification branch propagated). Bug had been latent since Phase A.5+ migrations. Affected NEW Stage 6 YAML AND 2 existing production HMHP-R configs. Fixed same-cycle: 13-line surgical change. **Without the pre-flight gate, Stage 6 would have trained with auto-adjusted uniform weights, NOT the documented H10-primary weighting** — empirical results would have been valid for the wrong experiment.

- **Lesson 66**: Phase S `pool_mode` field EMPIRICALLY WIRED + ARCHITECTURALLY PROVEN. `hmhp_cascade_regression.yaml:30` sets `hmhp_pool_mode: mean`; resolves through trainer schema bridge → lobmodels HMHPRegressor constructor at `_apply_pooling(shared_repr, "mean")`. Stage 6 successfully trained with Phase S `mean`-pool — first live training validation since Phase S shipped 2026-05-04.

- **Lesson 67**: Phase Y composability EMPIRICALLY VERIFIED across ALL 4 axes by combining R9-R13:
  - **Data axis** — same data → same compat_fp `67c8ff36...` (R9/R10/R11) OR same data + different calibration_method → different compat_fp (R12=`9a72a760...`)
  - **Architectural axis** — TLOB no-CVML vs TLOB+CVML produces different model_config_hash (`de47c0ef...` vs `3ced8443...`)
  - **Loss-tuning axis** — TLOB+CVML+Huber vs TLOB+CVML+GMADL produces SAME model_config_hash (`3ced8443...` for both — denylist works)
  - **Horizons-set axis** (NEW from Stage 6) — TLOB classification fallback `[10,20,50,100,200]` vs HMHP-R explicit `[10,60,300]` produces different compat_fp (`67c8ff36...` vs `cdd723ae...`)
  All 4 axes deterministically separable AND composable.

- **Lesson 68**: HMHP-R competitive with TLOB on v3p0 — challenges CLAUDE.md "TLOB > HMHP-R on H10" finding for time-based 60s/98-feat regime. Same direction (TLOB still wins) but tighter margin. HMHP-R adds value via multi-horizon outputs (H60 IC=0.1408, H300 IC=0.0820) + agreement_ratio.npy. Production-ready on v3p0.

#### Stage 7 (R14: TemporalGradBoost sklearn) — 2026-05-05

- **Lesson 69**: Sklearn pipeline path generalization VALIDATED across 2 sklearn models. Stage 1 (TemporalRidge) + Stage 7 (TemporalGradBoost) both train + export + backtest end-to-end through canonical Phase Q.5 dispatch + Phase Q.6 SimpleModelTrainer.from_config + Phase Q.6.5.A signal_metadata SSoT chain. Future sklearn ablations (XGBoost-direct, LightGBM, RandomForest) inherit the contract for free.

- **Lesson 70**: **STRONGEST EMPIRICAL FINDING OF THE CYCLE**: TemporalGradBoost on v3p0 produces the BEST OptRet across all 7 stages despite the LOWEST headline IC (0.2842) of any non-failure stage. **Best OptRet=-0.04% at max_conv_20bps (128 trades, 50.00% WR — near break-even)** vs Stage 1 Ridge (-0.46%, IC=0.329). **Δ IC=-0.045 but Δ OptRet=+0.42pp BETTER for GradBoost** — explicit ablation showing IC and trading utility can DIVERGE. Hypothesis: GradBoost's discrete tree decisions produce sharper directional predictions at high-conviction quantiles.

- **Lesson 71**: 50.00% WR @ max_conv_20bps is the highest WR in the post-Phase-O cycle for a NEAR-BREAKEVEN regime. Combined with -0.04% OptRet (cost-gate barely losing), Stage 7 is the closest to profitable trading we've seen on v3p0. **Caveat**: sample-of-1 evaluation on test split — would need walk-forward + out-of-sample bootstrap before claiming production trading viability.

#### Stage 8 (R15: Phase Y producer-side validation) — 2026-05-05

- **Lesson 72**: **PHASE Y PRODUCER-SIDE EMPIRICALLY VALIDATED**. The model_config_hash emitted in `signal_metadata.json` (Phase Y Stage 1, commit `879a77d`) is BIT-EXACT to the value embedded in the checkpoint sidecar (Phase X.1 v2). Both producers use the same `compute_model_config_hash` SSoT at `lobtrainer.training.compatibility:298` filtering `_LOSS_TUNING_KEYS` — empirical bit-exact match (`de47c0ef49abc0ef5d9d69efe1d4003a8b9551f24d5e6574b77f52fc041ecbb4` in both places) proves SSoT discipline holds end-to-end on real data.

- **Lesson 73**: **PHASE C.1 HORIZONS TRUTH-PIN EMPIRICALLY VALIDATED**. Loading R9's pre-Phase-C.1 checkpoint via current `Trainer.setup()` triggers `CheckpointConfigMismatchWarning` showing horizons drift `(10, 60, 300)` (post-truth-pin, correct) vs `(10, 20, 50, 100, 200)` (R9's pre-truth-pin, WRONG — classification defaults from silent-fallback at compatibility.py:233 that Phase C.1 deleted). **Implication for R9-R14**: their stored compatibility_fingerprints reflect WRONG horizons. Cross-experiment composability queries via `hft-ops ledger list --compatibility-fp 67c8ff36...` would silently group records with the wrong-horizons fingerprint. See PHASE_P_BACKLOG.md `#PY-6`.

- **Lesson 74**: Stage 8 reproduces R9 metrics + R9 best OptRet bit-exactly, validating that Phase Y deployment + Phase C.1 truth-pin do NOT alter MODEL behavior — they only correct the IDENTITY/PROVENANCE side of the contract. Same checkpoint, same data, same horizons feeding the model loss function = same metrics. **This separation is the architectural invariant Phase C.1+Y was designed to preserve**: provenance correctness without computation drift.

- **Lesson 75**: Stage 8 wall-clock (~5s for export-only on MPS) is the SHORTEST validated empirical probe in the cycle. Pattern documented for future use: "validate Phase Y producer changes by re-running export only on existing checkpoints" — burns ~5s instead of ~5-10min for re-train. Useful for future Phase Y producer-side iteration without re-training compute cost.

---

## 6. What NOT To Do — Anti-Patterns Table

### 6.1 Anti-Patterns Preserved from 2026-04-08 (Empirically Validated)

| Failed Approach | Result | Why |
|-----------------|--------|-----|
| Point-return regression at H1-H5 | IC=0.045 | Signal too weak at fine grain |
| Point-return regression at H10 | R²=0.000, IC=0.001 | OFI has zero correlation with point returns |
| ARCX fine-grained point-return (E3) | Best IC=0.035, 0/93 IC>0.05 | Even ARCX + event_count=100 cannot predict point returns |
| Multi-horizon regression for H10 | R²=0.454 (worse) | H60/H300 persistence pulls encoder away |
| Hybrid classification+regression gate | -2.67% | Readability gate adds zero value |
| TWAP execution | Marginal improvement | Still negative |
| 5M-param TLOB | OOM | No benefit over 693K-param model |
| Per-level Kolm OF (cumulative window) | IC=0.000 for all 20 features | Cumulative-sum window destroys per-event dynamics |
| Adding more analyzers | Diminishing returns | Core 34 sufficient |
| CVML front-end on 98 features | IC=0.373 (no improvement over 0.380) | Doesn't transfer to low-dim / small-sample regime — REPRODUCED on v3p0 (R10) |
| GMADL loss a=10, b=1.5 | IC=0.007, DA=49.8% (failure) | Loss inverts, mean prediction collapse — REPRODUCED on v3p0 (R11) |
| Prediction magnitude filtering | WR +10pp but \|pred\|>10 bps WORSE | Model predicts direction, not magnitude — REPRODUCED on v3p0 (R12) |
| Simplified PnL for regime validation | +5.9pp simplified, OPPOSITE in full backtester | Use full backtester, not simplified |
| Smoothed labels for point-to-point trading | DA=48.3% on point returns (below random) | Model learns smoothing residual, not future direction (E8) |
| Off-exchange features at H=10 only | 0/11 features IC>0.05 at H=10 | But IC>0.05 at H=1 / H=60 — wrong horizon tested initially |
| Long-horizon intraday morning→afternoon (E15) | Train +6011, val -233, test -909 bps | April 8-9 outlier drove 43% of ACF; in-sample artifact |
| Extreme events (E16) | 15/1656 BH FDR pass, 93% sign-flip rate | No persistent edge |
| Universality study (10 stocks) | 14/280 = 5.0% FPR (exact) | Universal — not NVDA-specific |

### 6.2 Anti-Patterns Discovered Post-Phase-O (2026-05-04 to 2026-05-05) — 15 NEW

| New Anti-Pattern | Source | Why It's Wrong |
|------------------|--------|----------------|
| Don't trust "+X% positive result" without bootstrap CI | #PY-8 | R10 +0.56% (561 trades) likely sampling noise. R14 -0.04% sample-of-1. R9-R15 ZERO statistical rigor. |
| Don't trust R9-R14 stored `compatibility_fingerprint` | #PY-6, Lesson 73 | Pre-Phase-C.1 horizons drift. Stored fps used wrong horizons `[10,20,50,100,200]` (classification defaults from silent-fallback at compatibility.py:233 that Phase C.1 deleted). New records use correct `[10,60,300]`. |
| Don't run new model variants before triaging 8 forensic-audit bugs (N1-N8) | #PY-10 | Adding more potentially-contaminated metrics doesn't help. N4 (HMHP-R hardcoded horizons[0]) + N6 (calibrated metrics report RAW) + N7 (normalization not bound to checkpoint) + N5 (HMHP/HMHP-R encoder pooling) may invalidate prior R-series interpretations. |
| Don't conflate bit-exact reproduction with statistical validation | Lesson 74 | R15 = R9 metrics proves Phase Y/C.1 don't BREAK anything; does NOT prove R9's IC=0.375 is meaningful. Bit-exact identity ≠ statistical significance. |
| Don't assume "higher IC → better P&L" | Lesson 70 | R14 GradBoost: lowest IC (0.284) but BEST OptRet (-0.04%). Ridge IC=0.329 → -0.46% (Δ +0.42pp WORSE despite +0.045 IC). Cross-sectional correlation and trading utility can diverge. |
| Don't trust calibrated metrics in pre-Phase-Z signal_metadata | N6 forensic audit | Forensic audit N6: "Calibrated metrics report RAW predictions". **Only R12 is the ACTIVE retroactive contaminator on existing v3p0 R-series** (R8 is pre-Phase-O). IC/Pearson/DA are calibration-invariant per audit `:813-814`; only MAE/RMSE/R² affected on R12. |
| Don't trust `_LOSS_TUNING_KEYS` boundary by design alone — verify empirically | Lesson 58 | Stage 4 (GMADL) FIRST EMPIRICAL PROOF that `gmadl_a + gmadl_b + regression_loss_type` denylist works (Stage 3 Huber and Stage 4 GMADL produce identical model_config_hash). Without empirical verification this would remain "by design" only. |
| Don't trust `signal_metadata.compatibility.horizons` when `data.labels.horizons` is empty | Lesson 53 | Falls back to classification defaults `[10,20,50,100,200]`. Cosmetic on metrics but corrupts cross-experiment composability queries. Phase X.3 candidate. |
| Don't deploy SignalManifest without prediction-variance threshold check | Lesson 60 | Stage 4 GMADL had std=0.000077 bps (textbook mean-collapse) yet signal_metadata validated cleanly. Cost-gate filtering caught it ($\|pred\|=0.9 < 1.4$ bps), but defense-in-depth `prediction_stats.std` minimum check is the correct design per hft-rules §8. |
| Don't gate `loss_weights` propagation by `if mt == "hmhp"` | Lesson 65 (resolved) | Schema bridge bug at `schema.py:1758-1761` silently dropped `hmhp_loss_weights` for `hmhp_regression`. Latent since Phase A.5+ migrations. **Fixed Stage 6 same-cycle**. Without pre-flight gate, would have trained on wrong weights. |
| Don't trust hand-curated FeatureSet registry as canonical | #PY-9 | All 3 entries at `contracts/feature_sets/*.json` are HAND-CURATED MIRRORS. NONE produced by `hft-ops evaluate --save-feature-set` (canonical Phase 4 producer path). Empty `source_profile_hash` + empty `data_export`. |
| Don't rely on stale CONSOLIDATED_FINDINGS_2026_03 for v3p0 | #PY-7 | Last updated 2026-04-08; excludes R9-R15 + Phase Y/Z/Stage 8 + 4-agent audit. Use `EXPERIMENT_INDEX.md` + `BACKTEST_INDEX.md` as authoritative for v3p0. **This document (CONSOLIDATED_FINDINGS_2026_05.md) supersedes the 2026-03 version.** |
| Don't run paired bootstrap on R9-R14 IC without re-pairing first | NEW_SESSION_HANDOFF §8.1.2 | `compare_sweep_statistical` raises `ValueError` when `regression_labels_sha256` differs across child records. R9-R14 horizons vary ([10,20,50,100,200] for R9 vs [10,60,300] for R12+R13) → different label tensor shape → different SHA → fail-loud. Requires Stage-8-style re-export-then-pair FIRST. |
| Don't run permutation-importance on a loaded checkpoint without authoring a new script | NEW_SESSION_HANDOFF §8.1.3 | `PermutationImportanceCallback.on_train_end` requires fully-constructed Trainer. Loaded checkpoint requires NEW `scripts/analysis/permutation_importance_from_checkpoint.py` (~2 hr). Phase 8C-α infrastructure shipped + tested; never used on R9-R15. |
| Don't launch Phase 4 ablation as parallel 4-way MPS | NEW_SESSION_HANDOFF §8.2 | "4-parallel TLOB on MPS" unsubstantiated. Serialize. ~30 min × 12 configs (TLOB v3p0 + TemporalRidge + GradBoost) × point_return × H={1,2,5,10}. |

---

## 7. Pipeline Architecture Validation (NEW — Post-Phase-O Cycle Hardening)

### 7.1 Phase O Cycle 1 — v3p0 Baseline Establishment (2026-05-04)

**4 v3p0 baseline corpora** re-extracted with reconstructor v0.2.1 + extractor c62a1c0 + hft-contracts 311bdbf:

| Corpus | Path | Days | Schema | Sequences |
|--------|------|------|--------|-----------|
| `e5_timebased_60s_v3p0` | `data/exports/e5_timebased_60s_v3p0/` | 230 (3 fail-loud per hft-rules §8) | 3.0 | 136,902 |
| `e5_timebased_30s_v3p0` | `data/exports/e5_timebased_30s_v3p0/` | 233 | 3.0 | 278,055 |
| `e4_timebased_5s_v3p0` | `data/exports/e4_timebased_5s_v3p0/` | 233 | 3.0 | 1,579,579 |
| `nvda_xnas_128feat_regression_fwd_prices_v3p0` | `data/exports/nvda_xnas_128feat_regression_fwd_prices_v3p0/` | 35 | 3.0 | 51,809 |

**Two product improvements over OLD pre-Phase-O exports:**
1. **+21% MORE training data** on 164/233 days (Phase O B.2 fix correctly resets book on session-boundary Clears)
2. **3 silently-corrupt short half-sessions now fail-loud** (20250703 / 20251128 / 20251224 — H300 needed 311 prices, sessions only had 209)

**Pre-Phase-O exports** preserved as historical archive at original paths; will FAIL `validate_export_contract()` per `hft-contracts/.../validation.py:115-120`. Per CLAUDE.md long-term-engineering principle: NO legacy validator added; new training uses ONLY v3p0 baseline.

### 7.2 Phase Q.6.5 — Training-Pipeline Completion (2026-05-04 night)

8 findings closed (F-16, F-18, N-4, N-5, N-6 [legacy ID], N-7, N-9, N-16). Architectural changes:

- NEW SSoT `lobtrainer.training.compatibility.feature_set_ref_to_dict` consolidating 3-site duplication
- NEW `Trainer.export_signals(split, *, output_dir, calibration) → Path` Protocol method
- BaseTrainer Protocol unified `load_checkpoint(path, load_optimizer=True)` signature
- `scripts/export_signals.py` collapsed to thin wrapper using `create_trainer + setup + load_checkpoint + trainer.export_signals` polymorphic chain
- CVTrainer rejects sklearn frameworks at construction (was silent k-fold sklearn crash)

### 7.3 Phase X.2.A.1 + A.2 — `validate_day_metadata` SSoT Consolidation (2026-05-04 night)

`hft_contracts.validation.validate_day_metadata` lifted as canonical Class A SSoT. Trainer's `_validate_day_metadata` shim at `dataset.py:60-101` delegates to the canonical helper. **Empirically verified Stage 2** (Lesson 52): 230 days of v3p0 schema=3.0 metadata pass; would fail-loud on any pre-Phase-O schema=2.2 metadata.

### 7.4 Phase Y Stage 1 + Stage 2 — Producer + Composer Wiring (2026-05-05)

**Stage 1** (lob-model-trainer commit `879a77d`): `build_signal_metadata` accepts `model_config_hash` kwarg + emits at `signal_metadata.json` root (not just inside `compatibility` block).

**Stage 2** (hft-ops commit `8839432`): `_record_experiment` harvester picks up `model_config_hash` from signal_metadata; `experiment_provenance_hash` composer at `hft_contracts.experiment_record._compose_experiment_provenance_hash` produces `sha256(data_export_fp + feature_set_content_hash + compat_fp + model_config_hash)` 4-input composition. CLI flag `--provenance-hash` for cross-experiment composability filters.

**Stage 8 / R15 empirical validation (Lesson 72-75):** `model_config_hash=de47c0ef49abc0ef5d9d69efe1d4003a8b9551f24d5e6574b77f52fc041ecbb4` BIT-EXACT match between Phase X.1 v2 checkpoint sidecar AND Phase Y signal_metadata.json. Same checkpoint, same data, same horizons feeding model = same metrics + same backtest results (-1.39% @ very_high_10bps reproduces R9 exactly). **Provenance correctness without computation drift is the architectural invariant Phase C.1+Y was designed to preserve.**

### 7.5 Phase Z Half-Ship Closures (2026-05-05)

**Z.1 / #PY-1 RESOLVED** (lob-model-trainer commit `6cae122`): wired `validate_idx_97_reserved` validator (orphan from Phase D ship gap). Also Phase Y composability lock (regression test asserting Stage 8 invariants).

**Z.2 / #PY-5 RESOLVED** (lob-models commit `09c04ab`): HMHP-R `use_confirmation` consumer-side gate (was silent-ignored at lobmodels-side; now construction-time validation matches schema.py contract).

### 7.6 Final Session State (2026-05-05)

| Repo | HEAD SHA | Tests Pass |
|------|----------|------------|
| hft-contracts | `bb06f65` | 540 |
| lob-backtester | `4b9deef` | 386 + 8 skip |
| lob-models | `09c04ab` | 809 + 25 skip |
| hft-ops | `8839432` | 669 |
| lob-model-trainer | `88cdd1d` | 1604 + 73 skip + 1 xfail |

All 5 CIs GREEN. All working trees clean. ZERO regressions cumulative across 12 atomic commits in 4 cycles.

---

## 8. Statistical-Rigor Caveat (NEW — #PY-8)

**The R9-R15 ranking shown in §1.2 is a single-point-estimate ordering. It is NOT statistically validated.**

Empirical state on 2026-05-05 (per NEW_SESSION_HANDOFF §2.2 7-agent investigation):

- **ZERO bootstrap CIs**, ZERO walk-forward variance, ZERO cross-fold validation, ZERO permutation tests, ZERO null-distribution comparisons applied to R9-R15.
- `BACKTEST_INDEX.md:715` already qualitatively flags R10's "+0.56% likely within sampling noise".
- `BACKTEST_INDEX.md:508` already flags R14's -0.04% as "Sample-of-1 test-split result. Walk-forward bootstrap + out-of-sample replication required".
- All `hft-metrics` primitives EXIST and are tested + import-ready: `block_bootstrap_ci`, `block_permutation`, `pairwise_paired_bootstrap_compare`, `purged_kfold_split`, `compare_sweep_statistical`. **UNUSED on R9-R15**.
- `hft-ops sweep compare` adapter SHIPPED (Phase V.B.4b) but NEVER RUN on R9-R15.
- Phase Y composability filters BUILT (`hft-ops ledger list --provenance-hash`) but UNAPPLIED.

**Decisive precedent**: E15 long-horizon study had ACF=−0.27 driven 43% by April 8-9 outlier alone. Universality Study showed 14/280 BH FDR pass = exactly 5.0% null FPR. The team has documented precedent for why single-point estimates mislead.

**Implications:**
- R10's +0.56% likely sampling noise.
- R14's -0.04% sample-of-1 — could easily flip sign on bootstrap.
- R13's IC=0.356 vs R9's IC=0.375 (Δ=−0.019) may be statistically indistinguishable.
- Without bootstrap CIs, EVERY "ranking" of R9-R15 is suspect.

**Critical structural blocker for paired bootstrap on R9-R14 IC**: `compare_sweep_statistical` at `hft-ops/src/hft_ops/ledger/statistical_compare.py:362-383` raises `ValueError` when `regression_labels_sha256` differs across child records. R9 horizons `[10,20,50,100,200]` vs R12+R13 horizons `[10,60,300]` produce different label tensor SHA → fail-loud on first invocation. **Requires Stage-8-style re-export-then-pair before paired bootstrap is even possible** (per NEW_SESSION_HANDOFF §8.2 Day 4-6 step 3a).

---

## 9. Feature Evaluation Gap (NEW — #PY-9)

50 experimental features (indices 98-147) span 5 groups. Empirical state on 2026-05-05:

| Group | Indices | Count | Status |
|-------|---------|-------|--------|
| kolm_of | 128-147 | 20 | **Evaluated** (`reports/kolm_of_experiment_2026_03_17.md`: all 20 IC ≈ 0) |
| mlofi | 116-127 | 12 | **NEVER evaluated** |
| institutional_v2 | 98-105 | 8 | **NEVER evaluated** |
| volatility | 106-111 | 6 | **NEVER evaluated** |
| seasonality | 112-115 | 4 | **NEVER evaluated** |

**30 features with ZERO formal IC evaluation.** No 148-feature whole-set IC scan exists.

**FeatureSet registry status** (`contracts/feature_sets/*.json`):
- All 3 entries are HAND-CURATED MIRRORS with `produced_by.tool = "hft-ops/scripts/migrate_feature_presets_to_registry.py"`
- Empty `source_profile_hash` + empty `data_export`
- **NONE was produced by `hft-ops evaluate --save-feature-set`** (the canonical Phase 4 producer path)

**ZERO permutation importance artifacts** despite Phase 8C-α infrastructure being shipped + tested:
- `hft-ops/ledger/feature_importance/` directory does NOT exist
- `lob-model-trainer/outputs/experiments/*/feature_importance/` does NOT exist for any of 28 v3p0 / HMHP / Ridge experiments

**Implication**: We don't know which features carry the IC=0.375 in R9. We don't know if any of the 30 unevaluated experimental features have predictive signal. The "feature evaluation is solid" assumption was wrong.

---

## 10. Forensic-Audit Bug Ledger (NEW — #PY-10)

**Source**: `lob-model-trainer/reports/TRAINING_PIPELINE_FORENSIC_AUDIT_2026_04_26.md` (234 KB / 26K words; 13-agent audit Apr 25 + 7-agent validation Apr 26).

8 bugs flagged ACTIVE or IMMINENT in production code paths:

| ID | Bug | Affected R-series | Status (post-compact 7-agent revision 2026-05-05) |
|----|-----|-------------------|------------------------------------------|
| N1 | InputContract `_base:` resolution preflight gap | All 7 stages (theoretical) | Real but NOT a "first live `hft-ops run` unblocker" (gate passed 2026-04-23 per CLAUDE.md L1037). Surgical fix in §12 Day 0 step 0d. |
| N2 | --resume epoch counter | Silent if no resume tested | RESUME-time bug; R9-R15 trained from scratch — DORMANT |
| N3 | EarlyStopping state not restored on resume | R9-R14 (theoretical at training-time) | RESUME-time bug; R9-R15 trained from scratch — DORMANT |
| N4 | HMHP-R primary metrics hardcode horizons[0] | R13 metrics interpretation | **DOESN'T FIRE for R13** (R13 used horizon_idx=0; trigger config `nvda_hmhp_regressor_h60.yaml` "not run"). Surgical fix in §12 Day 1-2 step 1a. |
| N5 | HMHP/HMHP-R encoder pooling inconsistency | R9 vs R13 comparison confounded | **CLOSED** (Phase S `_apply_pooling` SSoT shipped commit `4cbdc39`) |
| N6 | Calibrated metrics report RAW predictions (not calibrated) | **R12 only** on v3p0 (R8 pre-Phase-O excluded from current cycle scope) | **ONLY ACTIVE retroactive contaminator on existing v3p0 R-series**. Only R12 calibrated MAE/RMSE/R²; IC/Pearson/DA are calibration-invariant per audit `:813-814`. Surgical fix in §12 Day 1-2 step 1b. |
| N7 | Normalization not bound to checkpoint | Silent drift on resume / re-export | DORMANT-PRIMED. Defer to Phase X.4 strict-mode promotion. |
| N8 | TLOB final-flatten differs from Berti & Kasneci 2025 | R9, R10, R12 (TLOB-family) | DORMANT-PRIMED. Defer to "if cross-codebase reproduction needed". |

Plus 9 DORMANT bugs + 8 cross-module boundary findings + 5 REFUTED claims + 52 explicitly CLEARED items in the forensic audit.

**Note**: original NEW_SESSION_HANDOFF §2.4 framing "8 active forensic-audit bugs" was inaccurate. Post-compact revision (banner in §0 of NEW_SESSION_HANDOFF_2026_05_05.md) corrects: only N6 actively contaminates existing R9-R15 metrics (only R12; only MAE/RMSE/R²; not IC/DA).

---

## 11. Standing Process Mandates (NEW)

These are reaffirmed for all subsequent work and must be enforced for every commit:

1. **ULTRATHINK before every move** — analyze the problem fully before editing
2. **/effort max** — Opus 4.7 1M context at maximum reasoning effort
3. **MANDATORY pre-commit adversarial validation gate** per saved feedback memory `feedback_final_adversarial_validation_round.md`. Pre-commit dispatch parallel adversarial agents to verify the change.
4. **Always dispatch parallel adversarial agents at every stage**. Pre-flight agents predict. Mid-impl agents review. Pre-commit agents validate.
5. **Ground-truth code over docs** — docs may be stale or misleading (this document and CONSOLIDATED_FINDINGS_2026_03 are exemplars; the 2026-03 doc was 27 days stale on 2026-05-05). Always verify file:line citations against current code.
6. **Commit only when explicitly requested** — never auto-commit even after a fix passes tests.
7. **Build for years** — long-term design over quick fixes. No hidden fragility. Refactor early; refactor later is exponentially more expensive.
8. **Document every discovered issue** — future agents must inherit the finding.
9. **Reuse-first per hft-rules §0** — no new SSoTs when existing ones cover.
10. **Goal: enable many experiments empirically/precise/traceable/trackable/monitorable.**

**Pre-commit checklist** (per saved feedback memory):
- Re-verify file:line citations against current code (this doc may shift)
- Run module test suite + verify zero regressions
- Dispatch parallel adversarial agents to validate the change
- Get user explicit "commit + push" before each push
- Update CHANGELOG.md / module CODEBASE.md if user-visible
- Update PHASE_P_BACKLOG.md when closing items

---

## 12. Next Steps (REVISED PLAN — supersedes 2026-03 §6)

Per `NEW_SESSION_HANDOFF_2026_05_05.md` §8 (post-compact 7-agent investigation revised plan, ~10-12 days):

### Day 0 — Precondition cleanups (~6-8 hr serial)

| Step | Task | Effort |
|------|------|--------|
| 0a | PHASE_P_BACKLOG hygiene — strikethrough #PY-1 + #PY-5 (both RESOLVED) | 5 min |
| **0b** | **CONSOLIDATED_FINDINGS_2026_03.md refresh → bump to `CONSOLIDATED_FINDINGS_2026_05.md` (THIS file) + flip CLAUDE.md mandate pointer at line 14** | **3-4 hr (DELIVERED IN THIS DOC)** |
| 0c | **REFRAMED — #PY-4 was misframed**: empirical verification 2026-05-05 found R9-R14 have NO ledger records on disk (canonical-scripts path bypasses `_record_experiment`). PHASE_P_BACKLOG #PY-4 entry rewritten with corrected disposition (defer entirely; canonical scripts → ledger registration is out-of-scope architectural cleanup). Day 4-6 paired bootstrap will use signal-dir-based pure-function approach instead of ledger adapter. | 10 min docs |
| 0d | N1 surgical fix at 3 sites — call `resolve_inheritance` before reading `model.model_type` in `hft-ops/src/hft_ops/stages/contract_preflight.py:287-304` + `stages/training.py:115-122,200-209` | 1-3 hr |

### Day 1-2 — N-bug closures + Phase 4 launch (~10-12 hr SERIAL)

| Step | Task | Effort |
|------|------|--------|
| 1a | **N4 fix** — replace `horizons[0]` literal with `LabelsConfig.validate_primary_horizon_idx_for(n_horizons)` SSoT call at `lob-model-trainer/src/lobtrainer/training/strategies/hmhp_regression.py:157,254`. Metric-reporting-only — does NOT change `model_config_hash`. | 1.5-2 hr |
| 1b | **N6 fix** — patch metrics array source from raw `pr` to `calibration_result["calibrated"]` when present at `lob-model-trainer/src/lobtrainer/export/exporter.py:594-600`. Adds `metrics_raw` companion. R12 was contaminated; future R-experiments save correct metrics. | 2-3 hr |
| 1c | **Phase 4 SERIAL launch**: 12 configs (TLOB v3p0 + TemporalRidge + GradBoost) × point_return × H={1,2,5,10}. ~30 min × 12 + sklearn 5 min × 4. | 3-4 hr wall-clock |

### Day 3 — Phase 4 collection + permutation script (~3-5 hr)

| Step | Task | Effort |
|------|------|--------|
| 2a | Collect Phase 4 results into EXPERIMENT_INDEX.md + log per-config DA at H={1,2,5,10} | 30 min |
| 2b | Author NEW `permutation_importance_from_checkpoint.py` using pure `hft_metrics.bootstrap.compute_permutation_importance` primitive on Trainer.setup() + load_checkpoint() chain | 2 hr |
| 2c | Run permutation importance on R9 — first feature-importance artifact ever produced | 1 hr |

### Day 4-6 — Statistical Rigor Floor with paired-pair re-export (~2-3 days)

| Step | Task | Effort |
|------|------|--------|
| 3a | **CRITICAL precondition**: Stage-8-style re-export R9-R14 checkpoints against canonical e5_timebased_60s_v3p0 test split via Phase Q.6.5.B `Trainer.export_signals` → produces shared `regression_labels.npy` per #PY-6 cutover. Forces byte-identical `regression_labels_sha256` across all paired records. | 4-6 hr re-export |
| 3b | Pairwise paired bootstrap on R9-R14 IC at shared test split via `compare_sweep_statistical` — 15 pairwise CI rows + BH FDR q-values | 2 hr |
| 3c | Block bootstrap CI on each round's OptRet at very_high_10bps (independent — doesn't need pairing) | 2 hr |
| 3d | Pre-committed decision protocol: when CI on R9-R14 OptRet crosses zero (likely per #PY-8), conclude "rankings statistically meaningless"; document in EXPERIMENT_INDEX | 30 min |

### Day 6-9 — Feature Coverage informed by Phase 4 (~3-5 days)

| Step | Task | Effort |
|------|------|--------|
| 4a-i | NEW-CRITICAL-1 fix — drop schema "2.2" fallback; raise on missing in `hft-feature-evaluator/src/hft_evaluator/data/loader.py:265,237` | 30 min |
| 4a-ii | BLOCKING-2 fix — raise instead of return "unknown" | 1 hr |
| 4b | 148-feature whole-set IC scan AGAINST WINNING LABEL TYPE from Phase 4 (smoothed if DA ≤ 49%; point if DA > 52%) at H10/H60/H300 | 4 hr |
| 4c | First evaluator-derived FeatureSet via canonical `hft-ops evaluate --save-feature-set <name>_v1 --applies-to-assets NVDA --applies-to-horizons <H>` — closes the canonical Phase 4 producer path that was never exercised (#PY-9 main item) | 1 hr |

### Day 9-12 — Wrap-up + Decision Gate (~3 days)

| Step | Task | Effort |
|------|------|--------|
| 5a | NEW-CRITICAL-2 / BACKBONE FIND-C5 LabelFactory cross-language parity test — author golden fixture in hft-statistics + integration test in hft-contracts | 2-3 hr |
| 5b | Phase X.4 audit script — unblocks `strict_config` promotion gate clock | 2-3 hr |
| 5c | Document remaining N-bugs (N2, N3, N7, N8) with IMPACT ANALYSIS in PHASE_P_BACKLOG sub-items — they don't contaminate existing artifacts; deferred to specific cycles | 1 hr |
| 5d | Decision Gate per pre-committed protocol: paradigm-closure / dedicated cycle / per-horizon CI | 2-3 hr |

### Decision Gates Flowchart (per hft-rules §13)

```
                           [START Day 0: PRECONDITIONS]
                                    |
                       4 cleanups: hygiene + docs + (drop 0c) + N1
                                    |
                       [Day 1-2: N4 + N6 + Phase 4 SERIAL launch]
                                    |
                       [Day 3: Phase 4 collection + perm-import script]
                                    |
                                Phase 4 results in
                                    |
                  /-----------------+-----------------\
             (a) ALL DA ≤ 49%                  (b) ANY DA > 52%
                  |                                    |
       [Day 4-6: Stat Rigor]              [Reframe: positive finding]
       paired bootstrap +                          |
       block bootstrap +                  [Dedicated cycle on positive]
       permutation R9                               |
                  |                                  |
       /----------+----------\                       |
   (i) all CIs cross zero  (ii) some signif          |
       |                       |                     |
   [Day 6-9: Coverage         [Day 6-9: Coverage     |
    + smoothed labels]         + smoothed labels     |
       |                                             |
   [Day 9-12: WRAP-UP]                                |
   paradigm-closure                                   |
   publish negative                                   |
   paradigm-closure +                                 |
   pivot recommendation                               |
   (ARCX / sub-second / event-cond)                   |
                                                     |
                                            [Document positive finding;
                                             new cycle on it]
```

### What is explicitly DEFERRED

- N2 / N3 (RESUME-time bugs): defer to dedicated resume-aware cycle when --resume becomes regular workflow
- N7 (normalization-not-bound): defer to Phase X.4 strict-mode promotion
- N8 (TLOB final-flatten ordering): defer to "if cross-codebase reproduction needed" cycle
- D9 RNG state in checkpoint: dormant; composes with N2; defer with N2
- F4 BacktestData NaN→0 substitution: silent on equity-derivative only; defer to dedicated backtester correctness cycle
- F8 PermutationImportanceCallback silent failure: address opportunistically in 2c
- Phase X.5 Pydantic migration of lob-models BaseConfig: ~16-20 hr architectural cycle; independent
- Phase X.2.A.3-A.6 boundary discipline migrations (analyzer + evaluator + backtester): mechanical migration; future docs+migration sweep cycle
- 5,776-day NVDA corpus re-extraction to v3p0: operational; scheduled per Phase S corpus inventory
- PIPELINE_ARCHITECTURE.md body refresh + DOCUMENTATION_INDEX.md / pipeline-version-pin drift: dedicated docs cycle ~6-8 hr

---

## Appendix: Data Provenance

### Pre-Phase-O Data Sources (Preserved as Historical Archive)

| Artifact | Location | Description |
|----------|----------|-------------|
| Raw MBO data | `data/XNAS_ITCH/NVDA/mbo_2025-02-03_to_2026-01-07/` | 239 files, .mbo.dbn.zst |
| Smoothed regression export (PRE-O — schema 2.2) | `data/exports/nvda_xnas_128feat_regression/` | 13 GB, 233 days, 266,608 sequences |
| Point-return regression export | `data/exports/nvda_xnas_128feat_regression_pointreturn/` | 13 GB, 233 days, 266,841 sequences |
| TLOB T=100 checkpoint (REG-01) | `outputs/experiments/nvda_tlob_128feat_regression_h10/checkpoints/best.pt` | 693K params |
| HMHP-R checkpoint (REG-03) | `outputs/experiments/nvda_hmhp_regression_h10_primary/checkpoints/best.pt` | 171K params |
| Pre-Phase-O E5 60s export | `data/exports/e5_timebased_60s/` | 233 days, schema 2.2 — ARCHIVED, will FAIL strict validators |
| Off-exchange export | `data/exports/basic_nvda_60s/` | 233 days, 34 features |
| MBO point-return export | `data/exports/e5_timebased_60s_point_return/` | Derived from forward_prices |
| Universality exports | `data/exports/universality_{symbol}_60s/` (10 stocks) | 134 days each |

### v3p0 Baseline Data Sources (Phase O Cycle 1, 2026-05-04 — recommended for new work)

| Property | `e5_timebased_60s_v3p0` | `e5_timebased_30s_v3p0` | `e4_timebased_5s_v3p0` | `nvda_xnas_128feat_regression_fwd_prices_v3p0` |
|---|---|---|---|---|
| Path | `data/exports/e5_timebased_60s_v3p0/` | `data/exports/e5_timebased_30s_v3p0/` | `data/exports/e4_timebased_5s_v3p0/` | `data/exports/nvda_xnas_128feat_regression_fwd_prices_v3p0/` |
| Days emitted | 230 (3 fail-loud) | 233 | 233 | 35 |
| Total sequences | 136,902 | 278,055 | 1,579,579 | 51,809 |
| Disk size | 656 MB | 2.0 GB | 14 GB | 2.5 GB |
| Sequence shape | `(N, 20, 98)` float32 | `(N, 20, 98)` | `(N, 20, 98)` | `(N, 100, 128)` |
| Bin size | 60s time-based | 30s time-based | 5s time-based | event-based 1000 events/sample |
| Feature set | 98 stable (0-97) | 98 | 98 | 128 (98 + experimental groups) |
| Regression labels | `(N, 3)` float64 bps for H={10,60,300} | `(N, 3)` | `(N, 3)` | `(N, 3)` |
| Forward prices | `(N, 306)` float64 USD | `(N, 306)` | `(N, 306)` | `(N, 311)` |
| Schema version | 3.0 | 3.0 | 3.0 | 3.0 |
| Random seed | 42 | 42 | 42 | 42 |
| Provenance | extractor c62a1c0, reconstructor v0.2.1, hft-contracts 311bdbf | same | same | same |

### R9-R15 Signal Output Directories (Phase Q.6.5 Stages 1-8)

| Stage / R# | Signal Directory | Notes |
|------------|------------------|-------|
| Stage 1 (TemporalRidge sklearn) | `outputs/experiments/nvda_temporal_ridge_h10_v3p0/signals/test/` | First sklearn v3p0 |
| **R9 / Stage 2** (TLOB compact) | `outputs/experiments/nvda_first_pytorch_v3p0/signals/test/` | First PyTorch v3p0; baseline for R15 re-export |
| R10 / Stage 3 (TLOB+CVML) | `outputs/experiments/nvda_first_pytorch_v3p0_cvml/signals/test/` | -- |
| R11 / Stage 4 (TLOB+GMADL+CVML neg control) | `outputs/experiments/nvda_first_pytorch_v3p0_gmadl_cvml/signals/test/` | Mean-collapse |
| R12 / Stage 5 (TLOB calibrated) | `outputs/experiments/nvda_first_pytorch_v3p0/signals/test_calibrated/` | Re-uses R9 ckpt; only export differs |
| R13 / Stage 6 (HMHP-R) | `outputs/experiments/nvda_first_hmhp_r_v3p0/signals/test/` | First HMHP-R v3p0 + Phase S `mean`-pool |
| R14 / Stage 7 (TemporalGradBoost sklearn) | `outputs/experiments/nvda_first_temporal_gradboost_v3p0/signals/test/` | STRONGEST P&L of cycle |
| R15 / Stage 8 (TLOB Phase Y validation) | `outputs/experiments/nvda_first_pytorch_v3p0/signals/test_stage8_phase_y/` | Phase Y producer-side validation; bit-exact R9 reproduction |

### Cost Calibration

| Artifact | Location | Description |
|----------|----------|-------------|
| IBKR calibration (316 fills) | `IBKR-transactions-trades/COST_AUDIT_2026_03.md` | OPRA + IBKR commissions + BSM theta |
| IBKR full report (318 fills) | `IBKR-transactions-trades/IBKR_REAL_WORLD_TRADING_REPORT.md` | Stock fills + option fills |

### Statistical Analysis

| Artifact | Location | Description |
|----------|----------|-------------|
| Statistical analysis (pre-O) | `lob-dataset-analyzer/outputs/regression_deep_train/` | 7 JSON reports |
| MBO profiler | `mbo-statistical-profiler/output_xnas_full/` | 13 tracker JSONs |
| E13 classification | `hft-feature-evaluator/classification_table_mbo_point_return_lean.json` | MBO PointReturn 5-path |
| E13 signal diagnostics | `hft-feature-evaluator/outputs/signal_diagnostics_mbo/` | 7-diagnostic MBO suite |
| Universality consolidated | `hft-feature-evaluator/outputs/universality_consolidated_results.json` | Cross-stock summary |
| Universality report | `hft-feature-evaluator/reports/UNIVERSALITY_STUDY_2026_04.md` | Full analysis |
| E16 results | `hft-feature-evaluator/outputs/e16_extreme_events/e16_results.json` | Extreme event study |

---

## References — Cross-Doc Links

| Reference | Location | Purpose |
|-----------|----------|---------|
| **NEW_SESSION_HANDOFF_2026_05_05.md** | `/Users/knight/code_local/HFT-pipeline-v2/NEW_SESSION_HANDOFF_2026_05_05.md` | Authoritative session handoff (supersedes `_2026_05_04.md`) |
| **PHASE_P_BACKLOG.md** | `/Users/knight/code_local/HFT-pipeline-v2/PHASE_P_BACKLOG.md` | Deferred items #PY-1 through #PY-10 + phase-N items |
| **TRAINING_PIPELINE_FORENSIC_AUDIT_2026_04_26.md** | `/Users/knight/code_local/HFT-pipeline-v2/lob-model-trainer/reports/TRAINING_PIPELINE_FORENSIC_AUDIT_2026_04_26.md` | 234 KB / 26K words; 8 N1-N8 bug ledger; deps for §10 |
| **EXPERIMENT_INDEX.md** | `/Users/knight/code_local/HFT-pipeline-v2/lob-model-trainer/EXPERIMENT_INDEX.md` | Living per-experiment ledger; authoritative for E1-E16 + R8-R15 |
| **BACKTEST_INDEX.md** | `/Users/knight/code_local/HFT-pipeline-v2/lob-backtester/BACKTEST_INDEX.md` | Living per-backtest ledger; authoritative for R-series numerics |
| **CONSOLIDATED_FINDINGS_2026_03.md** | `/Users/knight/code_local/HFT-pipeline-v2/lob-model-trainer/reports/CONSOLIDATED_FINDINGS_2026_03.md` | **STALE** — historical pre-Phase-O findings (last 2026-04-08); preserved for E1-E16 reference but does NOT include R9-R15 |
| **BACKTESTER_AUDIT_PLAN.md** | `/Users/knight/code_local/HFT-pipeline-v2/lob-backtester/BACKTESTER_AUDIT_PLAN.md` | Moved 2026-05-04 from monorepo root → backtester repo; P0 label-execution finding source |
| **CLAUDE.md** | `/Users/knight/code_local/HFT-pipeline-v2/CLAUDE.md` | Root project rules + Validated Findings + auto-loaded session orientation |
| **PIPELINE_ARCHITECTURE.md** | `/Users/knight/code_local/HFT-pipeline-v2/PIPELINE_ARCHITECTURE.md` | ~4,020-line authoritative technical reference (deep dive) |
| **DOCUMENTATION_INDEX.md** | `/Users/knight/code_local/HFT-pipeline-v2/DOCUMENTATION_INDEX.md` | Master doc navigation map |
| **hft-rules.md** | `/Users/knight/code_local/HFT-pipeline-v2/.claude/rules/hft-rules.md` | Project rules (auto-loaded; §0 reuse-first, §13 research discipline, §14 multi-source data) |
| **feedback_final_adversarial_validation_round.md** | `~/.claude/projects/.../memory/feedback_final_adversarial_validation_round.md` | Saved feedback memory: MANDATORY pre-commit adversarial validation gate |

---

*This document supersedes `CONSOLIDATED_FINDINGS_2026_03.md` and is the authoritative consolidated reference for the HFT pipeline as of 2026-05-05. Per CLAUDE.md long-term-engineering principle: build for years; document failures as precisely as successes; independent metric validation mandatory; decision gates at each phase. For ground-truth code state and ledger entries, see EXPERIMENT_INDEX.md + BACKTEST_INDEX.md.*
