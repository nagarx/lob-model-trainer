# Experiment Index

**Living ledger of all training experiments.** Updated after every experiment completes.

**Current best classification:** HMHP 128-feat XNAS (H10 test accuracy 59.62%, directional accuracy 93.88% at high conviction)
**Current best regression:** TLOB 128-feat Regression H10 (test R²=0.464, IC=0.677, DA=74.9%)

**Consolidated findings:** `reports/CONSOLIDATED_FINDINGS_2026_03.md` -- START HERE. All validated metrics, lessons learned, and next steps.

**Detailed reports:** `reports/` directory
- `reports/CONSOLIDATED_FINDINGS_2026_03.md` -- Authoritative cross-pipeline reference (all experiments + findings)
- `reports/ABLATION_FINDINGS_2026_03_16.md` -- Simple model ablation ladder (L0-L4) + TWAP backtest
- `reports/RESEARCH_IMPLEMENTATION_PLAN.md` -- Research-driven implementation plan (10 papers)
- `reports/regression_series_2026_03_15.md` -- Regression experiment series (3 experiments + baselines + backtests)
- `reports/hmhp_128feat_2026_03_13.md` -- HMHP 128-feat classification (XNAS + ARCX)
- `reports/tlob_regression_2026_03_15.md` -- TLOB regression pipeline validation

---

## Validation Experiments

### P0: Label-Execution Mismatch Validation (2026-03-17)

| Field | Value |
|---|---|
| **Hypothesis** | CONSOLIDATED_FINDINGS claims smoothed vs point-return labels have r=0.24 and 55.8% conditional win rate. Is this correct? |
| **Method** | Export forward_prices.npy from Rust (aligned mid-price trajectories), compute both label types via `LabelFactory`, compare |
| **Data** | 35 test-split days (2025-11-14 to 2026-01-06), 510K samples, XNAS ITCH, 128 features |
| **Parameters** | H=10, k=10 (smoothing_window), matching production config |
| **Infrastructure** | New: Rust forward_prices export, Python LabelFactory (hft-contracts), LabelExecutionMismatchAnalyzer (lob-dataset-analyzer) |
| **Report** | `data/exports/nvda_xnas_128feat_regression_fwd_prices/p0_label_execution_mismatch_H10.json` |
| **Status** | Completed |

**Results:**

| Metric | Claimed | Validated | Delta |
|---|---|---|---|
| Pearson r (smoothed vs point labels) | 0.24 | **0.640** | +0.40 |
| P(point > 0 \| smoothed > 0) | 55.8% | **69.7%** | +13.9pp |
| P(point > 0 \| \|smoothed\| > 5 bps) | — | **87.9%** | — |
| P(point > 0 \| \|smoothed\| > 10 bps) | — | **92.2%** | — |

**Lesson**: The original r=0.24 was likely computed from misaligned data (two exports with different event_count sampling). The `forward_prices` approach guarantees alignment by computing both label types from the same mid-price trajectories. The label-execution mismatch is SMALLER than originally diagnosed — the primary bottleneck is cost structure, not label misalignment.

**Caveat**: This measures label-to-label correlation, not model prediction vs execution. Effective execution r ≈ 0.640 × sqrt(0.464) ≈ 0.436.

### E1: Deep ITM Backtest (2026-03-17)

| Field | Value |
|---|---|
| **Hypothesis** | TLOB regression model produces positive P&L at deep ITM (delta=0.95, 1.4 bps breakeven) with conviction filtering |
| **Method** | ExperimentRunner with sweep over min_return_bps [0.7, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0] |
| **Data** | 50,724 test samples, 35 days, XNAS ITCH, TLOB 128-feat regression H10 signals |
| **Config** | delta=0.95, commission=$0.63, hold_events=10, position_size=0.1 |
| **Status** | **FAIL — All thresholds negative** |

**Results (Deep ITM, delta=0.95):**

| Threshold | Trades | Equity Return | Option Return |
|---|---|---|---|
| 0.7 bps | ~4,000 | -13.6% | -11.4% |
| 5.0 bps | ~1,800 | -5.8% | -4.1% |
| 8.0 bps | ~200 | -0.5% | -0.2% |
| 10.0 bps | ~50 | -0.4% | -0.5% |

**Root Cause Analysis (TWO issues found):**

1. **Holding duration 10x too long (fixable):** `hold_events=10` means 10 sequence steps = 100 samples. H10 predicts 10 samples ahead. Correct: `hold_events=1` (1 sequence step = 10 samples = H10). Verified: H10 label correlates with consecutive-sequence return at r=0.562, but with 10-step return at only r=0.183.

2. **Model predictions have ZERO execution correlation (fundamental):** Model prediction vs consecutive price return: **r=0.013** (essentially random). Despite R²=0.464 on smoothed labels, the model has NO predictive power on actual tradeable price returns. The model learns to replicate the smoothed-average FORMULA, not to predict future prices.

**Lesson**: The smoothed-average label is a function of prices (a rolling average). High R² on this label means the model approximates the formula well — but this approximation has no causal relationship with FUTURE price movements. Training on point-return labels or direct price prediction is required.

**Next Steps**: Train on point-return labels (`return_type = "point_return"`) to force the model to predict actual price movements, not the smoothed average.

### E2: Point-Return Training Validation (2026-03-17)

| Field | Value |
|---|---|
| **Hypothesis** | Training TLOB on point-return labels produces r > 0.05 with actual price returns |
| **Method** | Pre-training IC validation: compute feature IC on point-return labels BEFORE training (Rule 13) |
| **Data** | 163 train + 35 val + 35 test days, `nvda_xnas_128feat_regression_pointreturn` export |
| **Status** | **FAIL — Cancelled before training (IC gate failed)** |

**Results (Feature IC on Point-Return Labels):**

| Feature | IC (point-return) | IC (smoothed) | Ratio |
|---|---|---|---|
| depth_norm_ofi | +0.007 | +0.108 | 0.07x |
| true_ofi | +0.013 | +0.109 | 0.12x |
| trade_asymmetry | +0.015 | +0.103 | 0.15x |
| **Best feature** | **0.025** | **0.11** | **0.23x** |
| Features with \|IC\| > 0.05 | **0 of 128** | **9 of 128** | — |

**Decision gate (Rule 13):** "If no feature has IC > 0.05 for the target label, no model will help." Zero features pass. Training cancelled.

**Lesson:** OFI features are **contemporaneous** — they describe the current microstructure state (what the return IS), not what the future price WILL BE. This is consistent with the validated finding "OFI predictive power (lag-1): r < 0.006 at ALL scales" from the mbo-statistical-profiler.

**Implication:** No LOB/MBO/OFI-based model can predict point-to-point returns at H10 (10 samples ≈ 3-5 seconds). The signal exists only in the smoothed-average label (which describes current state, not future state). Profitability requires either:
1. **Per-event architecture** (Kolm LSTM on raw OF transitions, not accumulated features)
2. **Longer horizons** where OFI persistence (ACF(1)=0.266) provides weak predictive signal
3. **Different data sources** (options flow, dark pool, cross-venue signals)

### E3: ARCX Fine-Grained Point-Return Regression (2026-03-17)

| Field | Value |
|---|---|
| **Hypothesis** | ARCX fine-grained sampling (event_count=100) with point-return labels produces feature IC > 0.05, enabling a model with R² > 0.05 on actual price returns |
| **Rationale** | Three converging advantages: ARCX OFI r=0.688 (+14pp vs XNAS), ARCX cost 1.10 bps (-79%), Kolm showed IC=0.070 at event_count=100 (10x vs 1000) |
| **Decision Gates** | IC gate (>0.05) → Training gate (R²>0.05) → Execution gate (r>0.05 with returns) → Profitability gate (option return > 0%) |
| **Config** | `configs/e3_ic_gate.toml` — ARCX, event_count=100, T=20, stride=5, point_return, H=[10,60,300], 98 features |
| **Data** | 35 ARCX test-split days (2025-11-14 to 2026-01-06), 395,967 sequences, shape [N, 20, 98] |
| **Evidence Base** | P0 (label r=0.642), E1 (model r=0.013), E2 (IC gate fail at XNAS 1000-event), Kolm (IC=0.070 at 100-event), profiler (ARCX r=0.688) |
| **IC Results** | `data/exports/e3_ic_gate/ic_results.json` |
| **Status** | **FAIL — IC gate failed, terminated before training** |

**Results (Feature IC on ARCX Point-Return Labels at event_count=100):**

| Horizon | Best Feature | Best IC | Features \|IC\| > 0.05 | Verdict |
|---|---|---|---|---|
| H10 | signed_mp_delta_bps | +0.035 | 0 / 93 | FAIL |
| H60 | net_cancel_flow | +0.013 | 0 / 93 | FAIL |
| H300 | ask_price_L10 | -0.033 | 0 / 93 | FAIL |

**Top features at H10 (ARCX, event_count=100, point-return):**

| Feature | IC (ARCX point) | IC (XNAS point, E2) | IC (XNAS smoothed) |
|---|---|---|---|
| signed_mp_delta_bps | +0.035 | — | — |
| ask_size_L1 | -0.025 | — | — |
| bid_size_L1 | +0.023 | — | — |
| executed_pressure | +0.021 | — | — |
| true_ofi | +0.017 | +0.013 | +0.109 |
| depth_norm_ofi | +0.016 | +0.007 | +0.309 |

**Key finding**: ARCX + fine-grained = slightly BETTER than XNAS for point-return IC (best 0.035 vs 0.025), but still far below the 0.05 threshold. The three converging advantages (stronger OFI, lower cost, finer granularity) improved IC by ~40% relative to E2, but from a base so low that it remains insufficient.

**Critical insight**: Kolm's IC=0.070 was a **contemporaneous** measurement (lag-0 correlation between scalar OFI and return at the same sampling point). Our IC gate measures **predictive** IC (features at time t vs return at time t+H). These are fundamentally different quantities. OFI is contemporaneous even at ARCX, even at fine granularity.

**Lesson**: The ARCX + fine-grained hypothesis is CONCLUSIVELY eliminated. No combination of exchange (XNAS or ARCX), sampling resolution (event_count=100 or 1000), or feature set (40-128 features) produces IC > 0.05 for point-return labels. The problem is not exchange quality, sampling resolution, or feature engineering — it is that **accumulated LOB/MBO features at snapshot-level have no predictive power for future point-to-point returns at any tested horizon (H10, H60, H300).**

**What remains viable:**
- **F1**: OFI persistence prediction — predict OFI[t+1] from features[t], then use contemporaneous OFI-return relationship
- **F2**: Cross-venue lead-lag — XNAS flow may lead ARCX price
- **F3**: Event-level (not snapshot) architecture — raw order transitions, not accumulated statistics
- **F4**: Alternative data sources (options flow, dark pool, retail flow)

### F1: OFI Persistence Prediction (2026-03-17)

| Field | Value |
|---|---|
| **Hypothesis** | OFI has persistence at 4-10 minute timescales (profiler ACF=0.266 at 5m), enabling a two-step chain: predict OFI[t+H] from features[t], then map predicted OFI to expected return via contemporaneous OFI-return r=0.577-0.707 |
| **Method** | Phase 0: Validate TRUE_OFI ACF(1) at K=[1,5,10,13,26,50] sequence steps across 233 days. Phase 1: Feature IC gate for OFI target. |
| **Data** | 233 days XNAS, `nvda_xnas_128feat_regression` export, 266,608 sequences, event_count=1000, stride=10 |
| **Results** | `data/exports/f1_ofi_persistence_analysis.json` |
| **Status** | **FAIL — Phase 0 failed, terminated before any training** |

**Results (TRUE_OFI ACF(1) across 233 days):**

| Timescale | K (steps) | Mean ACF | Median ACF | Threshold | Verdict |
|---|---|---|---|---|---|
| 23 seconds | 1 | **+0.021** | +0.014 | > 0.08 | FAIL |
| 1.9 minutes | 5 | **+0.013** | +0.009 | > 0.08 | FAIL |
| 3.8 minutes | 10 | **+0.004** | +0.003 | > 0.08 | FAIL |
| 5.0 minutes | 13 | **+0.003** | +0.000 | > 0.08 | FAIL |
| 10.0 minutes | 26 | **+0.003** | +0.003 | > 0.08 | FAIL |
| 19.2 minutes | 50 | **+0.000** | -0.002 | > 0.08 | FAIL |

**Phase 1**: OFI(t) → OFI(t+10) IC = **0.005** (threshold: 0.05). Zero features have IC > 0.05 for OFI target.

**Root cause**: Our feature extractor accumulates OFI over non-overlapping 1000-event windows. Each window's accumulated OFI is statistically independent of adjacent windows. The profiler's ACF=0.266 at 5-minute scale reflects **fixed-time integration** over all events in each interval — a fundamentally different measurement that captures sustained flow direction. Our event-based sampling architecture destroys the persistence signal.

**Lesson**: OFI persistence exists in the underlying continuous order flow process (validated by the profiler at fixed time intervals), but it is NOT accessible through our current snapshot-based feature extraction architecture. The OFI feature (index 84/85) represents a window-aggregated statistic that has zero inter-sample autocorrelation. To access OFI persistence:
1. **Time-based sampling** (fixed 5-minute intervals, not event-count) — requires implementing the P2 pending work item in the feature extractor
2. **Per-event architecture** (Kolm LSTM on raw order transitions) — bypasses the snapshot aggregation entirely
3. **Rolling OFI** (cumulative sum with exponential decay, not per-window reset) — a new feature that would preserve inter-sample persistence

### E4: Time-Based Sampling OFI Validation (2026-03-18)

| Field | Value |
|---|---|
| **Hypothesis** | Time-based sampling (5-second intervals) recovers OFI persistence (ACF > 0.08) that event-based sampling destroys (ACF=0.021), enabling features with IC > 0.05 at 5-minute horizon |
| **Rationale** | Profiler uses `resample_to_grid()` with fixed time bins → ACF=0.266 at 5m. Our event-based `sample_and_reset()` every 1000 events → ACF=0.021. Same formula, different sampling. |
| **Method** | Phase 0: Export 233 days with `TimeBasedSampler` at 5s intervals. Phase 1: Validate 4 decision gates (ACF, IC, return std, walk-forward). Phase 2: Baselines (Ridge, AR). Phase 3: TLOB training H60. Phase 4: Backtest. |
| **Config** | `feature-extractor-MBO-LOB/configs/nvda_xnas_98feat_timebased_5s.toml` |
| **Data** | 233 XNAS days (2025-02-03 to 2026-01-06), 98 features, 5s time bins, window_size=20 (100s), stride=1, horizons=[10,60,300] |
| **Export** | `data/exports/e4_timebased_5s/` — train: 163 days (971,947 seqs), val: 35 days (206,493 seqs), test: 35 days (218,163 seqs) = 1,396,603 total |
| **Validation Script** | `lob-dataset-analyzer/scripts/e4_validate_ofi_persistence.py` |
| **Validation Results** | `lob-dataset-analyzer/outputs/e4_validation/e4_gate_results.json` (train), `outputs/e4_validation_val/` (val), `outputs/e4_validation_test/` (test) |
| **Phase 2 Report** | `reports/e4_phase2_baselines_2026_03.md` |
| **Comprehensive Analysis** | `lob-dataset-analyzer/reports/E4_COMPREHENSIVE_ANALYSIS_2026_03.md` — 14 analyzers × 3 splits, cross-validated |
| **Phase 3 Config** | `configs/experiments/e4_tlob_h60.yaml` — TLOB 2L/32H/2Heads, 92.7K params, Huber delta=7.3, T=20, F=98, horizon_idx=1 |
| **Phase 3+4 Report** | `reports/e4_training_backtest_2026_03.md` — training, backtest, lessons |
| **Status** | **COMPLETE — All 4 phases. Test IC=0.136 (+12% vs Ridge). Backtest: Deep ITM best=-3.68% at 5bps, 45% win rate. NOT profitable.** |

**Decision Gates (Rule 13):**

| Gate | Metric | Threshold | Train | Val | Test | Status |
|------|--------|-----------|-------|-----|------|--------|
| **G1** | TRUE_OFI ACF(1) | > 0.08 | 0.043 | 0.032 | 0.072 | **FAIL** (3.3x improvement vs event-based 0.021; per-day mean=0.070) |
| **G2** | Best feature IC at H60 | > 0.05 | **0.083** | **0.086** | **0.089** | **PASS** — TRUE_OFI + DEPTH_NORM_OFI |
| **G3** | H60 return std (bps) | > 5.0 | 29.9 | 16.4 | 20.8 | **PASS** |
| **G4** | Walk-forward IC stability | > 2.0 | 3.04 | 2.96 | **3.87** | **PASS** |

**G1 failure analysis:**
- Concatenated ACF (multi-day) = 0.043 due to day-boundary contamination
- Per-day mean ACF = 0.070 (37% of days have ACF > 0.08)
- 3.3x improvement over event-based (0.021), but below 0.08 target
- Profiler ACF=0.266 uses interval-OFI (flow per time bin); our feature is instantaneous OFI — fundamentally different quantities

**Why proceed despite G1 failure:**
- G2 (IC) is the direct measure of predictive power — G1 (ACF) was a proxy
- IC=0.083-0.089 is consistent across all 3 splits (NOT overfitting)
- This IC was **ZERO** in all prior event-based experiments (E1, E2, E3, F1)
- Walk-forward stability increases on test (3.87) — signal is structural
- The plan's G1 pivot ("try 30s/1min intervals") is less urgent than pursuing the validated IC signal

**Top features at H60 (all splits consistent):**

| Feature | Train IC | Val IC | Test IC |
|---------|----------|--------|---------|
| true_ofi (84) | +0.083 | +0.086 | +0.089 |
| depth_norm_ofi (85) | +0.082 | +0.086 | +0.088 |
| executed_pressure (86) | +0.044 | +0.037 | +0.035 |
| volume_imbalance (45) | — | -0.037 | -0.045 |

**Key design decisions:**
- `time_interval_ns = 5_000_000_000` (5 seconds) — profiler ACF=0.124 at 5s, 0.266 at 5m
- `window_size = 20` — 20 × 5s = 100 seconds of history (T=20 retains 99.6% of IC)
- `horizons = [10, 60, 300]` — H60 = 5min matches profiler's ACF=0.266 target scale
- `normalization = "none"` — raw features for statistical validation first
- 98 features (no experimental) — simple for first time-based run

**Phase 2 Results (Baseline Models at H60, test split):**

| Model | Params | R² | IC | DA | MAE (bps) |
|-------|--------|----|----|----|----|
| DEPTH_NORM_OFI only | 1 | 0.009 | 0.089 | 0.535 | — |
| Ridge (98 raw feat) | 98 | 0.010 | 0.091 | 0.531 | — |
| **TemporalRidge (53 feat)** | **53** | **0.013** | **0.121** | **0.543** | **12.5** |
| TemporalGradBoost (200 trees) | ~200 | -0.145 | 0.073 | 0.533 | 14.7 |

**Phase 2 Decision Gates:**
- B1: TemporalRidge R² > 0.02 → **FAIL** (0.013, but R² is misleading for kurtosis=36.5)
- B2: TemporalRidge IC > single-feat IC (0.087) → **PASS** (IC=0.121, +36%)
- B3: TemporalRidge R² > 1.5× Ridge → **MARGINAL** (1.3×)
- **Decision**: PROCEED to Phase 3 based on IC=0.121 (t=56.6, statistically significant)

**Phase 2 Statistical Characterization:**
- Recommended Huber delta for H60: **7.4 bps** (from ReturnDistributionAnalyzer)
- Feature IC half-life: **5 timesteps = 25 seconds** (signal in most recent data)
- Walk-forward IC stability ratio: **15.2** (zero regime shifts, 158 folds)
- Universally strong across all regimes: DEPTH_NORM_OFI, TRUE_OFI

**Phase 3 Results (TLOB Training at H60):**

| Metric | Val (epoch 1) | Test | TemporalRidge | vs Baseline |
|--------|---------------|------|---------------|-------------|
| IC | 0.143 | **0.136** | 0.121 | +12.4% |
| R2 | 0.016 | 0.015 | 0.013 | +17.7% |
| DA | 0.548 | 0.544 | 0.543 | +0.2% |

Training: 92,690 params, best at epoch 1, overfitting after (IC degrades from 0.143 to 0.061 by epoch 9).

**Phase 4 Results (Backtest, 0DTE Options):**

| Threshold | ATM Return | Deep ITM Return | ATM Win% | Deep ITM Win% |
|-----------|-----------|-----------------|----------|---------------|
| 0.7 bps | -19.8% | -14.2% | 27.4% | 38.0% |
| 2.0 bps | -15.1% | -10.7% | 32.4% | 41.7% |
| 5.0 bps | -5.3% | **-3.7%** | 36.3% | **45.0%** |

**Lesson:** Time-based sampling is the key architectural change that unlocked predictive signal. TLOB adds 12% IC over Ridge. Deep ITM options are consistently better than ATM (+5-6pp return, +9-11pp win rate). But IC=0.136 is insufficient for profitability — model direction accuracy (38-45%) is below the ~50% needed to overcome 1.4 bps deep ITM breakeven. Signal is real but too weak for profitable execution with current cost structure.

### E5: Time-Bin Sweep — OFI Continuation (2026-03-18)

| Field | Value |
|---|---|
| **Hypothesis** | Larger time-bins (15-120s vs E4's 5s) increase OFI persistence and IC at tradeable horizons (5-30 min), enabling profitability at Deep ITM cost (1.4 bps) |
| **Method** | Phase 0: Export 233 days at 4 bin sizes (15s, 30s, 60s, 120s). Compute IC surface (bin × horizon → IC mapped to wall-clock time). Select optimal tradeable configuration. |
| **Data** | 4 exports: e5_timebased_{15,30,60}s + E4 5s baseline. 98 features, horizons=[10,60,300], smoothing=5 |
| **Config** | `e5_timebased_{15,30,60,120}s.toml` — only 4 TOML fields differ from E4 |
| **Sweep Script** | `lob-dataset-analyzer/scripts/e5_sweep_validation.py` |
| **Sweep Results** | `lob-dataset-analyzer/outputs/e5_sweep/e5_sweep_combined.json` |
| **Phase 0 Report** | `lob-dataset-analyzer/reports/E5_SWEEP_RESULTS_2026_03.md` |
| **Status** | **COMPLETE — ALL 4 PHASES. IC=0.380 (highest in pipeline). Backtest: -1.93% Deep ITM (still negative).** |

**Phase 0 Results — IC Surface (bin × horizon → IC, mapped to wall-clock time):**

| Bin | H10 IC | H10 time | H60 IC | H60 time | H300 IC | Tradeable best |
|-----|--------|---------|--------|---------|---------|---------------|
| **5s** (E4) | 0.237 | 50s | 0.082 | 5m | 0.079 | H60=5m (IC=0.082) |
| **15s** | 0.235 | 2.5m | 0.078 | 15m | 0.068 | H60=15m (IC=0.078) |
| **30s** | 0.240 | 5m | 0.079 | 30m | 0.066 | H10=5m (IC=0.240) |
| **60s** | **0.248** | **10m** | 0.086 | 60m | 0.065 | **H10=10m (IC=0.248)** |
| 120s | FAILED | — | — | — | — | Need 311 samples/day for H300, only 195 |

**CRITICAL: Horizons are EVENT-based.** H10 at 60s bins = 10×60s = 10 minutes wall time. H300 at 60s bins = 300×60s = 5 hours (not tradeable). The tradeable IC is found at H10 for larger bins.

**Decision Gate**: IC > 0.08 at any tradeable horizon (5-30 min) → **PASS** (0.248 >> 0.08)

**Key Discovery**: H10 at 30-60s bins maps to the **same 5-10 min time window** as H60 at 5s bins — but with **3x higher IC** (0.24 vs 0.08). Larger bins accumulate more OFI per sample, making each sample more informative. This is the single most important finding of E5.

**120s Failure**: At 120s bins, 6.5h trading day / 120s = ~195 samples per day. Regression labels need 2×smoothing_window + max_horizon + 1 = 2×5 + 300 + 1 = 311 samples. 195 < 311 → export fails. Constrains maximum bin size.

**Export Manifests**:

| Export | Train seqs | Val seqs | Test seqs | Total |
|--------|-----------|---------|-----------|-------|
| e5_timebased_15s | 313,190 | 67,103 | 67,304 | 447,597 |
| e5_timebased_30s | 132,150 | 28,437 | 28,044 | 188,631 |
| e5_timebased_60s | 39,749 | 8,574 | 8,337 | 56,660 |

**Lesson**: The bin size is a critical hyperparameter that was previously fixed at event_count (event-based) or 5s (E4 time-based). Sweeping bin sizes reveals that OFI IC is primarily a function of the OFI accumulation per sample, not the temporal horizon. At 60s bins, each sample integrates ~12,000 MBO events (vs ~1,000 at 5s), producing features with 3x more predictive power. The cost ratio at 60s/H10 is 4.8% (vs 8.5% at E4 H60), making profitability achievable at DA > 52%.

**Phase 1 Baselines (60s bins, H10 = 10 min, test split):**

| Model | Params | R² | IC | DA | MAE (bps) |
|-------|--------|-----|------|------|-----------|
| DEPTH_NORM_OFI only | 1 | — | 0.255 | — | — |
| **TemporalRidge** | **53** | **0.068** | **0.306** | **0.613** | **18.4** |
| TemporalGradBoost | ~200 | -0.040 | 0.257 | 0.589 | 19.1 |

All Phase 1 gates PASS. Huber delta calibrated: 12.6 bps (kurtosis=26.5). Analyzer: `lob-dataset-analyzer/outputs/e5_60s_analysis_train/` (14 reports).

**Phase 2 Training (5 runs, CVML + GMADL ablation):**

| Run | Config | Bin | CVML | Loss | Best Ep | Test IC | Test DA | Test R² | Status |
|-----|--------|-----|------|------|---------|---------|---------|---------|--------|
| 1 | e5_60s_huber_nocvml | 60s | No | Huber δ=12.6 | 4 | **0.380** | **0.640** | 0.124 | **BEST** |
| 2 | e5_60s_huber_cvml | 60s | Yes (49) | Huber δ=12.6 | 6 | 0.373 | 0.640 | 0.121 | PASS |
| 3 | e5_60s_gmadl_cvml | 60s | Yes (49) | GMADL a=10 b=1.5 | 12 | 0.007 | 0.498 | -0.001 | **FAIL** |
| 4 | e5_30s_huber_nocvml | 30s | No | Huber δ=15.1 | 6 | 0.379 | 0.637 | 0.132 | PASS |
| 5 | e5_30s_huber_cvml | 30s | Yes (49) | Huber δ=15.1 | 4 | 0.380 | 0.641 | 0.128 | PASS |

**Phase 2 Decision Gates:**
- G_IC: Test IC=0.380 > Ridge IC=0.306 → **PASS (+24%)**
- G_DA: Test DA=0.640 > 0.52 break-even → **PASS (+12pp above break-even)**
- G_GMADL: GMADL DA=0.498 < Huber DA=0.640 → **FAIL (GMADL loss diverges)**
- G_OVERFIT: Val-Test IC gap = 1.5% → **PASS**

**Phase 2 Lessons:**
1. **CVML adds nothing**: IC=0.373 (CVML) vs 0.380 (no CVML) — within noise. Li et al. CVML (+244% R²) doesn't transfer to our 98-feature / 40K-sample regime. The 5 Conv1D layers add 28K params but no signal. Feature space is already well-structured (LOB prices/sizes are naturally ordered).
2. **GMADL completely fails**: Loss function inverted at epoch 16, model collapsed to mean prediction (DA=49.8%). The direction-weighted sigmoid loss landscape is incompatible with Huber-tuned training dynamics. Needs separate investigation with different a/b and learning rate schedules.
3. **30s bins match 60s bins**: Despite lower sweep IC (0.240 vs 0.248), 30s models achieve identical test IC (0.379-0.380). The 4x more training data (132K vs 39K) compensates exactly. Both bin sizes are viable.
4. **Simplest model wins**: Run 1 (no CVML, Huber, 92K params) is the best or tied-best across all metrics.
5. **Best at epoch 4-6**: Unlike E4 (best at epoch 1), E5 models improve for 4-6 epochs before overfitting. The 60s bins provide enough temporal structure for multi-epoch learning.

**Comparison with E4:**

| Metric | E4 (5s/H60) | **E5 (60s/H10)** | Improvement |
|--------|-------------|-------------------|-------------|
| Test IC | 0.136 | **0.380** | **+180%** |
| Test DA | 0.544 | **0.640** | **+9.6pp** |
| Test R² | 0.015 | **0.124** | **+7.3x** |
| Best epoch | 1 | 4 | Model learns longer |
| Backtest (Deep ITM) | -3.68% | TBD (Phase 3) | — |

**Phase 3 Backtest (Round 7, 2026-03-19):**

Model: `e5_60s_huber_nocvml` (IC=0.380, DA=64.0%, 92K params). Hold: 10 events × 60s = **10 minutes**.

| Threshold | Deep ITM Return | Deep ITM Win% | ATM Return |
|-----------|----------------|---------------|-----------|
| 0.7 bps | **-1.93%** | **40.1%** | -3.07% |
| 2.0 bps | -3.85% | 38.0% | -4.07% |
| 8.0 bps | -1.37% | 37.0% | -2.43% |
| 10.0 bps | -5.10% | 36.0% | -4.14% |

Phase 3 gate: option return > 0% → **FAIL** (best = -1.37% at 8 bps Deep ITM)
Backtest results: `lob-backtester/outputs/backtests/e5_round7/`
BACKTEST_INDEX: Round 7 documented.

**Final E5 Lesson**: Time-bin optimization (5s → 60s) improved IC by 180% (0.136 → 0.380) and DA by +9.6pp (54.4% → 64.0%), but backtest win rate dropped by 4.9pp (45.0% → 40.1%). The signal is dramatically stronger but still doesn't translate to profitable execution. **Root cause**: smoothed-average labels (training objective) ≠ point-to-point returns (execution reality). Model achieves 64% accuracy on "did the average go up" but only 40% accuracy on "is the price higher 10 minutes later."

**What E5 validated**:
1. Larger time bins (60s) accumulate more OFI per sample → 3x higher feature IC
2. CVML (Li et al. ICLR 2025) adds zero value on 98-feature / 40K-sample regime
3. GMADL loss (Michankov et al. 2024) fails to converge with a=10, b=1.5
4. Simplest model wins: no CVML, Huber loss, 92K params, best at epoch 4
5. 30s bins match 60s bins (4x more data compensates for lower IC)

**What E5 did NOT solve**:
- Label-execution mismatch remains the #1 bottleneck (DA=64% on labels → 40% execution win rate)
- Point-return labels have zero IC (E2/E3 validated) — no known path to direct prediction

**Phase 2+3 Report**: `reports/e5_phase2_training_2026_03.md`
**Best checkpoint**: `outputs/experiments/e5_60s_huber_nocvml/checkpoints/best.pt`
**Signal export**: `outputs/experiments/e5_60s_huber_nocvml/signals/test/`

---

### E6: Prediction Calibration + High-Conviction Filtering (2026-03-19)

| Field | Value |
|---|---|
| **Hypothesis** | Model predictions are 3.73x too conservative (std=7.35 vs target=27.41 bps). Variance-matching calibration rescales to target std, enabling conviction-based filtering. At |calibrated|>10 bps, win rate should reach ~90% (from E5 report §7.1 threshold analysis on labels). |
| **Method** | Post-hoc variance-matching calibration: `calibrated = (pred - pred_mean) * (target_std / pred_std) + target_mean`. No retraining. Preserves IC (linear transform). Then sweep thresholds from 1.4 to 20 bps. |
| **Data** | E5 60s test split (8,337 sequences, 35 days). Same data as R7. |
| **Model** | E5 best: `e5_60s_huber_nocvml` (epoch 4, IC=0.380, DA=0.640, 92K params). No retraining. |
| **Config** | `configs/experiments/e6_calibrated_conviction.yaml` |
| **Calibration Module** | `src/lobtrainer/calibration/variance.py` (18 tests, all pass) |
| **Status** | **COMPLETE — FAIL. Win rate improved to 50.6% (from 40.1%), but no threshold achieves profitability. Best: -0.85% at 2 bps.** |

**Calibration Validation:**

| Metric | Before | After | Expected |
|--------|--------|-------|----------|
| Prediction std | 7.35 bps | 27.41 bps | 27.41 ✓ |
| IC (Spearman) | 0.3800 | 0.3800 | Unchanged ✓ |
| Scale factor | — | 3.73 | 27.41/7.35 ✓ |
| Above |pred|>10 bps | 24.4% | 76.1% | Increased ✓ |
| Above |pred|>20 bps | 0.0% | 54.0% | Increased ✓ |

**Backtest Results (Round 8, Deep ITM, delta=0.95):**

| Threshold | Trades | Win Rate | Option Return |
|-----------|--------|----------|---------------|
| 1.4 bps | 742 | 48.0% | -2.87% |
| **2.0 bps** | **741** | **50.6%** | **-0.85%** |
| 3.0 bps | 740 | 45.7% | -5.06% |
| 5.0 bps | 736 | 48.2% | -3.40% |
| 8.0 bps | 724 | 47.9% | -5.95% |
| 10.0 bps | 717 | 47.7% | -6.85% |
| 15.0 bps | 698 | 47.7% | -3.28% |
| 20.0 bps | 670 | 45.5% | -5.99% |

**Decision Gates:**
- G1 (IC > 0.35): PASS (IC=0.380, exactly preserved)
- G2 (win rate > 55% at 10 bps): **FAIL** (47.7% at 10 bps)
- G3 (option return > 0%): **FAIL** (best = -0.85%)

**Root Cause Analysis:**

The calibration successfully corrected prediction magnitudes (std 7.35 → 27.41 bps), and overall win rate improved from 40.1% (R7) to 50.6% (R8 at 2 bps). However, **higher thresholds DECREASE win rate** (50.6% at 2 bps → 45.5% at 20 bps). This reveals a critical insight:

**The model's magnitude predictions do not rank well.** The model predicts DIRECTION correctly (DA=64% on smoothed labels) but its prediction of HOW MUCH the return will be is nearly uncorrelated with actual magnitude. Filtering on calibrated magnitude removes trades approximately randomly, not selectively. The E5 report's threshold analysis (§7.1) showed 90.8% win rate at |smoothed|>10 bps — but that was based on the TRUE LABELS, not model predictions. The model cannot replicate label-level magnitude discrimination.

**Effective r breakdown:**
- Label-to-point: r=0.831 (from E5 report — labels predict execution well)
- Model-to-label: IC=0.380 (model predicts labels moderately well)
- Model-to-execution: r ≈ 0.380 × 0.831 = 0.316 (attenuated)
- Threshold filtering on model predictions: no magnitude ordering → no conviction gain

**What E6 proved:**
1. Variance calibration works mechanically (std matched, IC preserved)
2. Calibration alone improves win rate by +10pp (40→50%) — meaningful
3. BUT model predictions lack magnitude ranking — filtering on |pred| does not select better trades
4. The label-level threshold analysis does NOT transfer to model predictions
5. The R7→R8 win rate improvement (40%→50%) is likely from the mean centering (shifting from -0.32 to -0.17 bps), not from magnitude rescaling

**Lesson**: Post-hoc calibration is necessary but NOT sufficient. The bottleneck is now model magnitude quality, not prediction scale. Next approaches should either: (a) train a model that better ranks return magnitudes (e.g., quantile regression), (b) use an auxiliary model for magnitude estimation, or (c) abandon magnitude-based filtering and use alternative conviction signals (e.g., prediction consistency across nearby samples, regime-conditional confidence).

**Files:**
- Calibration module: `src/lobtrainer/calibration/variance.py` (18 tests)
- Calibrated signals: `outputs/experiments/e6_calibrated_conviction/signals/test/`
- Backtest: `lob-backtester/outputs/backtests/e6_round8/e6_round8_calibrated_deep_itm.json`

---

### E7: Phase A — Regime Hypothesis Validation (2026-03-21)

| Field | Value |
|---|---|
| **Hypothesis** | Simple rule-based regime gating (using depth_norm_ofi, spread_bps, volume_imbalance, order_flow_volatility quartiles) improves simplified model win rate by >= 3pp, validating the regime detection architecture (final_plan/) before Rust implementation. |
| **Method** | Simplified trade PnL: `trade_pnl = sign(calibrated_returns) * regression_labels - 1.4 bps`. 17 gating strategies evaluated (4 single-feature, 3 two-feature combos, 3 multi-feature combos, 7 percentile sweeps). Decision gate: WR improvement >= 3pp + Wilson 95% CI non-overlapping + 60% days improved. |
| **Data** | E6 calibrated signals (8,337 test, 35 days) + E5 test features (98 feat, unnormalized) + E5 train features (163 days, 39,749 samples for quartile boundaries). |
| **Model** | E5 best: TLOB 92K params, IC=0.380, DA=0.640 (same as E6, no retraining). |
| **Script** | `scripts/validate_regime_hypothesis.py` |
| **Results JSON** | `scripts/phase_a_results.json` |
| **Status** | **INCONCLUSIVE — original PASS verdict withdrawn after deep validation** |

**Initial Result (before validation):**

Best strategy `dno_top_15pct` (|depth_norm_ofi| >= P85): WR=66.9% (+5.9pp), Bonferroni significant, 27/33 days improved. All 3 decision gate criteria met.

**Deep Validation Findings (3 independent agents):**

| Finding | Severity | Detail |
|---------|----------|--------|
| DNO gating is proxy for prediction magnitude | CRITICAL | corr(|dno|, |cal_ret|) = 0.305-0.350. After partialing out |cal_ret|, DNO partial correlation with WR = 0.009 (zero). DNO-exclusive samples (selected by DNO but NOT |cal_ret|): WR=61.3%, DA=63.2% — at baseline. |
| Simplified PnL model is biased | CRITICAL | `sign(pred)*label - cost` mechanically favors filters selecting larger |labels|. |cal_ret| top 15% gives WR=75.0% in simplified PnL, but E6 R8 full 0DTE backtester shows magnitude filtering DECREASES option WR (50.6% at 2 bps → 45.5% at 20 bps). Simplified model gives OPPOSITE conclusions from real backtester. |
| Spread feature broken | HIGH | Stock price appreciation ($143 train → $183 test) shifted 1-tick bps value below Q25. 87% of test data in Q1, zero gating selectivity. |
| DNO quartiles skewed | MEDIUM | Q4 contains 42% of test data (not 25%) due to distribution shift. |
| High per-day variance | MEDIUM | Std=8.58pp > mean=6.50pp. 15/33 days have N_gated < 30. |

**Why INCONCLUSIVE (not FAIL):**
- Regime-conditional IC does vary (0.299 to 0.446 across dno quartiles) — conditions DO exist where the model is more reliable
- The simplified PnL model cannot distinguish regime signal from magnitude artifact — it's the wrong tool
- Need full 0DTE backtester to determine if regime gating helps after real costs

**What Phase A proved:**
1. Model signal quality is NOT uniform — regime conditioning has theoretical potential
2. The model's own confidence (|cal_ret|) is the strongest quality signal, dominating all external features
3. Spread_bps is non-stationary across stock price changes (tick structure effect) — unusable in bps form
4. Simplified PnL (`sign*label-cost`) is a biased proxy for real trading — mechanically favors filters selecting larger |labels|. Must use full 0DTE backtester for regime validation.
5. The E6 finding that magnitude filtering DECREASES option WR (in the full backtester) still stands — the validation agent's claim it was "wrong" was tested with simplified PnL, not the full backtester

**Lesson**: Never use simplified PnL (`sign(pred) * label - cost`) to evaluate regime/gating strategies. Any filter correlated with |labels| appears to "work" in simplified PnL but may fail in the full backtester. The correct evaluation tool is the full 0DTE backtester with real cost structure.

**Next steps:**
1. Test DNO gating in the FULL 0DTE backtester (not simplified PnL)
2. Compare to E6 R8 ungated baseline (50.6% option WR)
3. Only proceed to Phase B if full backtester validates improvement

**Validation report**: `reports/phase_a_regime_validation_2026_03.md`

### E8: Model Diagnostic + Statistical Foundations (2026-03-21)

| Field | Value |
|---|---|
| **Hypothesis** | Determine whether the E5/E6 TLOB model can be fixed (via label alignment, feature engineering, horizon adjustment, or conditioning) to achieve DA > 52% on point-to-point returns. Separately, validate 8 statistical assumptions (A3/A5/A6/A9/A11/A14/A53/BPV) underlying the regime detection architecture. |
| **Method** | Script 1: Model diagnostic — decompose model predictions vs smoothed residual vs point returns, sweep horizons H=1..50, test all conditioning strategies, audit 23 features with IC>0.05. Script 2: Statistical foundations — test OFI persistence, EWMA whitening, spread persistence, tail distribution, CUSUM inflation, return ACF, jump prevalence at 60s bins. |
| **Data** | E5/E6 test signals (8,337 samples, 35 days, 60s bins), E5 features (98 features, unnormalized). |
| **Model** | E5 best: TLOB 92K params, IC=0.380, DA=0.640 (same model as E6/E7, no retraining). |
| **Scripts** | `scripts/e8_model_diagnostic.py`, `scripts/e8_statistical_foundations.py` |
| **Status** | **CRITICAL — ALL four fix paths blocked. Model requires fundamental redesign.** |

**Script 1 Results — Model Diagnostic:**

| Test | Result | Detail |
|---|---|---|
| DA on point returns | **48.3%** | Below random (50%). Model actively predicts WRONG direction for point returns. |
| Model captures smoothing residual | R²(model, residual)=**45%**, R²(model, point)=**0.02%** | Model predicts the smoothed-average formula's residual (difference between smoothed and point return), NOT point-to-point price direction. |
| Disagreement analysis | When smoothed and point labels disagree on direction (19.5% of samples): model DA on smoothed=**90.1%**, on point=**9.1%** | On the samples that matter for trading (where label types disagree), the model follows the smoothing artifact with 90% fidelity. |
| Feature IC for point returns | **0/67** non-price features have IC > 0.05 at 60s bins | No signal exists in LOB/MBO/OFI features for point-return prediction at this timescale. |
| Horizon sweep DA | DA <= **49.1%** at ALL horizons H=1 to H=50 | No horizon fixes the problem. The model cannot predict point-return direction at any lookahead. |
| Conditional DA | No condition has DA > **52%** on point returns | Neither volatility, time regime, spread state, nor OFI magnitude rescues point-return prediction. |
| Price-level IC audit | The 23 features with IC > 0.05 in initial analysis were ALL price-level artifacts | Mean reversion of stock price trend, not microstructure signal. Non-actionable. |

**Verdict:** ALL four candidate fix paths (label alignment, feature engineering, horizon adjustment, conditioning) are blocked. The model predicts the smoothing residual, not point-to-point direction. This is the ROOT CAUSE of all 8 negative backtest rounds. Fundamental model/label redesign required.

**Script 2 Results — Statistical Foundations (4/8 pass):**

| Test ID | Name | Result | Detail |
|---|---|---|---|
| A3 | OFI ACF at 60s | **PASS** | ACF(1) = 0.164 (validates CUSUM cadence for regime detection) |
| A5 | EWMA whitening | **PASS** | Optimal lambda=0.80, residual ACF=0.028 (effective decorrelation) |
| A6 | Spread persistence | **FAIL** | ACF = 0.065 (expected 0.35). Spread NOT persistent enough for regime conditioning at 60s. |
| A9 | Heavy tails | **PASS** | Kurtosis = 6.375 (confirms non-Gaussian, Student-t appropriate) |
| A11 | Student-t fit | **PASS** | df = 3.12 (very heavy tails, extreme returns more frequent than Gaussian) |
| A14 | CUSUM inflation | **FAIL** | Inflation ratio = 0.95 (EWMA pre-whitening unnecessary at ACF=0.164 — CUSUM works directly) |
| A53 | Return ACF | **FAIL** | ACF = 0.968 (label construction artifact from overlapping smoothing windows, NOT market structure) |
| BPV | Jump prevalence | **FAIL** | Jump ratio = 0.038 (jumps negligible at 60s bins — no jump-detection value) |

**Statistical foundations implications:**
- OFI persistence at 60s is real (A3 PASS) — regime detection via CUSUM on raw OFI is viable
- EWMA whitening works but is unnecessary (A14) — CUSUM can operate on raw OFI directly
- Spread is NOT persistent at 60s (A6 FAIL) — cannot use spread as a regime feature (confirms E7 spread failure)
- Return ACF=0.968 is a labeling artifact (A53 FAIL) — smoothing windows overlap, creating artificial autocorrelation. This does NOT reflect market-level return persistence.
- Jumps negligible at 60s (BPV FAIL) — jump detection adds no value at this sampling frequency

**Lesson**: The E5/E6 TLOB model has R²=0.124 and DA=64.0% on smoothed labels, but **DA=48.3% on point returns** (below random). The model predicts the smoothing residual — the difference between the smoothed-average label and the actual point return — with R²=45%. This smoothing residual has NO trading value. All 8 negative backtest rounds are explained by this single root cause. Any future model must be trained and evaluated on labels that match the execution horizon (point returns or a label type whose direction aligns with tradeable price moves >95% of the time).

**Next steps**: Model requires fundamental redesign. Options: (1) direct point-return prediction with different architecture/features, (2) much longer horizons where smoothed and point returns converge (H60+), (3) abandon mid-price prediction entirely and predict execution-relevant quantities (e.g., fill probability, queue position).

### E9: Off-Exchange Signal Validation (2026-03-21)

| Field | Value |
|---|---|
| **Hypothesis** | Off-exchange data (TRF prints, BJZZ retail order imbalance, dark pool share, VPIN) provides orthogonal predictive signal for NVDA point returns at 60s bins, exceeding IC > 0.05 threshold. Cross-sectional weekly Mroib finding (Barardehi et al. 2021) may transfer to intraday time-series prediction. |
| **Method** | BJZZ retail identification + midpoint signing on XNAS.BASIC CMBP-1 data. 11 off-exchange features computed at 60s bins: trf_volume_share, dark_share, retail_share, retail_signed_imbalance (Mroib), |Mroib|, trf_signed_imbalance, subpenny_intensity, odd_lot_ratio, trf_trade_intensity, retail_size_ratio, vpin_proxy. IC gate: Spearman rank correlation with H=10 point-to-point returns from E5 forward_prices. |
| **Data** | 35 test days (20251114-20260106), 8,337 samples, XNAS.BASIC CMBP-1 (TRF + on-exchange prints). |
| **Script** | `scripts/e9_offexchange_signal_validation.py` |
| **Status** | **RESOLVED — `basic-quote-processor` implemented (412 tests, Phases 1-5). Cross-validation at optimal horizons: trf_signed_imbalance IC=+0.103 at H=1, subpenny_intensity IC=+0.104 at H=60. Phase 6 Python IC validation pending.** |

**Results — Off-Exchange Feature IC (H=10 point returns, 60s bins):**

| Feature | IC | p-value | Assessment |
|---|---|---|---|
| subpenny_intensity | +0.048 | 1.2e-05 | Best, marginal (below 0.05 threshold) |
| trf_signed_imbalance | +0.040 | 3.0e-04 | Below threshold |
| dark_share | +0.035 | 1.5e-03 | Below threshold |
| retail_signed_imbalance (Mroib) | +0.021 | 0.058 | Far below threshold |
| |Mroib| | -0.005 | -- | Useless |

**Baseline and persistence:**

| Metric | Value |
|---|---|
| MBO true_ofi IC (baseline) | -0.009 (confirms E8: no on-exchange signal either) |
| Mroib ACF(1) | 0.050 (essentially no persistence — contemporaneous like OFI) |
| Dark share ACF(1) | 0.42 (persistent but low IC) |
| Dark share mean | 0.79 |
| TRF trades as % of all trades | 68% |
| TRF volume as % of total | 61% |
| BJZZ retail ID rate | 45.3% |
| Retail trades per 60s bin | ~2,400 |

**Verdict:** INVESTIGATE. 0/11 off-exchange features pass the IC > 0.05 gate. The subpenny_intensity signal (IC=0.048) is marginal and warrants investigation at longer horizons but is unlikely to exceed the threshold. The cross-sectional weekly Mroib finding (Barardehi et al. 2021) does NOT transfer to intraday time-series prediction for a single mega-cap stock. All microstructure signals -- both on-exchange (OFI, IC=-0.009) and off-exchange (Mroib IC=+0.021, TRF imbalance IC=+0.040) -- are contemporaneous for NVDA. Adding 84% of previously unseen volume (TRF prints) does not create predictive signal.

**Lesson (14)**: Off-exchange features (TRF prints, retail order imbalance, dark share, VPIN) have IC < 0.05 for NVDA point returns at 60s bins. The cross-sectional weekly Mroib finding does NOT transfer to intraday time-series prediction for a single mega-cap stock. All microstructure signals -- both on-exchange (OFI) and off-exchange (Mroib, TRF imbalance) -- are contemporaneous for NVDA. Adding 84% of previously unseen volume does not create predictive signal.

---

## Feature Evaluation Experiments

### E10: Off-Exchange 5-Path Feature Evaluation (2026-03-27)

| Field | Value |
|---|---|
| **Hypothesis** | Which of the 34 off-exchange features have robust, reproducible signal with point returns across multiple statistical tests and market conditions? |
| **Method** | 5-path evaluation framework: (1) IC screening with per-day aggregation + t-test + BH, (2) dCor+MI with subsampled pooled permutation tests + BH, (3a) temporal IC via rolling features, (3b) transfer entropy with subsampled pooled permutation, (4) regime-conditional IC with tercile conditioning, (5) JMI forward selection. Stability: 20 bootstrap subsamples of Layer 1. Holdout: 20 days with bootstrap CI confirmation. |
| **Data** | `data/exports/basic_nvda_60s/train` (166 days, 60s bins, 34 features, **point_return** labels at 8 horizons [1,2,3,5,10,20,30,60]) |
| **Config** | `hft-feature-evaluator/configs/offexchange_34feat_lean.yaml` (100 dCor perms, 1000 subsample, 50 MI perms, 20 bootstraps) |
| **Tool** | hft-feature-evaluator v0.1.0 (14 modules, 162 tests) |
| **Runtime** | 138 min |
| **Status** | **COMPLETE** |

**Results:**

| Tier | Count | Key Features |
|---|---|---|
| **STRONG-KEEP** | 2 | spread_bps (IC=+0.30, all 5 paths, stability=100%), bbo_update_rate (IC=+0.33, 4 paths, stability=100%) |
| KEEP | 12 | trf_volume, total_volume, trade_count, subpenny_intensity, lit_volume, size_concentration, spread_change_rate, block_trade_ratio, bin_trade_count, bin_trf_trade_count, session_progress, odd_lot_ratio |
| INVESTIGATE | 4 | mean_trade_size, time_since_burst, retail_volume_fraction, trf_lit_volume_ratio |
| DISCARD | 10 | trf_signed_imbalance (stability=10%), mroib, inv_inst_direction, bid/ask_pressure, bvc_imbalance, retail_trade_rate, trf_burst_intensity, quote_imbalance |
| Excluded | 6 | 4 categorical + 2 zero-variance (trf_vpin, lit_vpin) |

**Key findings:**
1. Nonlinear dependence dominates: Path 2 (dCor+MI) detects 100 (feature, horizon) pairs; Path 1 (IC) detects only 6. Off-exchange features have overwhelmingly nonlinear relationships with returns.
2. Transfer entropy found zero signal (0 pairs) — features describe current state, not predict future.
3. trf_signed_imbalance is DISCARD (stability=10%) despite E9's IC=0.103. Signal is unstable across bootstrap subsamples.
4. Critical statistical finding: dCor has +0.102 bias, TE has +0.051 bias at n=308. Both use subsampled pooled permutation tests instead of per-day t-test (which would produce 100% false positives).

**Lesson (15):** Off-exchange features have predominantly nonlinear dependence with point returns. Only 2 of 28 evaluable features (spread_bps, bbo_update_rate) pass rigorous multi-path + stability evaluation. Linear models (Ridge) will miss most off-exchange signal. Features from E9 (trf_signed_imbalance IC=0.103) are DISCARD when properly evaluated with stability selection — the IC=0.103 was not reproducible across market conditions. The 5-path framework with dCor bias correction provides a more reliable signal quality assessment than single-metric IC screening.

**Report:** `hft-feature-evaluator/EVALUATION_REPORT.md`, output: `hft-feature-evaluator/classification_table_lean.json`

---

### E11: MBO 5-Path Feature Evaluation (2026-03-28)

| Field | Value |
|---|---|
| **Hypothesis** | Which of the 98 stable MBO features have robust signal with SmoothedReturn labels? Does the nonlinear dominance finding from E10 hold for MBO features? |
| **Method** | Same 5-path framework as E10. |
| **Data** | `data/exports/e5_timebased_60s/train` (163 days, 60s bins, 98 features, **SmoothedReturn** labels at 3 horizons [10,60,300]) |
| **Config** | `hft-feature-evaluator/configs/mbo_98feat_lean.yaml` (100 dCor perms, 1000 subsample, 50 MI perms, 20 bootstraps) |
| **Tool** | hft-feature-evaluator v0.1.0 |
| **Runtime** | 173 min |
| **Status** | **COMPLETE — E8 CAVEAT: labels are SmoothedReturn, NOT point-to-point. Results are upper bound on tradeable signal.** |

**Results:**

| Tier | Count | Key Features |
|---|---|---|
| **STRONG-KEEP** | 38 | All 20 prices, 10 bid sizes, 4 ask sizes, true_ofi, depth_norm_ofi, spread_bps, mid_price, volumes |
| KEEP | 10 | Various sizes and derived features |
| INVESTIGATE | 0 | — |
| DISCARD | 41 | All MBO-specific order flow features (add/cancel/trade rates, size distributions, queue metrics, institutional signals) |
| Excluded | 9 | 4 categorical + 5 zero-variance (avg_queue_position, queue_size_ahead, modification_score, iceberg_proxy, invalidity_delta) |

**Key findings:**
1. MBO is the OPPOSITE of off-exchange: Path 1 (linear IC) detects 102 pairs; Path 2 (dCor+MI) detects 0. MBO signal is overwhelmingly linear.
2. Price levels have IC = -0.82 with H=300 SmoothedReturn — this is the level/mean-reversion artifact, NOT tradeable signal.
3. rolling_mean is the best metric for 53/89 features — temporal trajectory > instantaneous value. Confirms ablation finding (temporal features boost R^2 by 2.9x).
4. 41 features DISCARD: ALL MBO-specific order flow details (rates, sizes, queues, institutional) have unstable signal (stability < 35%).
5. 38 STRONG-KEEP is inflated by SmoothedReturn labels and price-level artifacts. With point-return labels, the number would likely drop to 5-10.

**Lesson (16):** MBO features have predominantly linear dependence with SmoothedReturn labels — fundamentally opposite to off-exchange features which show nonlinear dependence with point returns. This difference may reflect the label type (SmoothedReturn amplifies linear autocorrelation) or the feature nature (LOB prices are monotonic in returns). ALL MBO-specific microstructure features (41 of 89) are DISCARD — their signal is not reproducible across market conditions, even with the more favorable SmoothedReturn labels. The core signal carriers are prices (level effect), OFI, and spread — the same 6-10 features identified in the ablation study. The 38 STRONG-KEEP count is an artifact of SmoothedReturn; a point-return evaluation is needed for trading decisions.

**Report:** `hft-feature-evaluator/MBO_EVALUATION_REPORT.md`, output: `hft-feature-evaluator/classification_table_mbo_lean.json`

---

### E12: Off-Exchange Pre-Training Analysis + Signal Diagnostics (2026-03-28)

| Field | Value |
|---|---|
| **Hypothesis** | Can the E10 STRONG-KEEP features (spread_bps IC=0.178, bbo_update_rate IC=0.113 at H=60) be captured by Ridge/GradBoost models and converted to profitable trades via a threshold-based long-only strategy? |
| **Method** | (1) 10-domain pre-training analysis: return distribution, feature distributions, full IC matrix, partial dependence, feature interactions, temporal analysis, redundancy, regime IC, walk-forward Ridge (136 daily folds), GradBoost 3×3 grid search, cost-adjusted tradability. (2) 7-diagnostic signal suite: multi-horizon threshold backtest (H=1-60 × P70-P95), val/test OOS backtest, regime robustness, trade distribution, P&L curve, power analysis, session conditioning. |
| **Data** | `data/exports/basic_nvda_60s/` — Train: 166 days (2025-02-03 to 2025-09-30), Val: 32 days (Oct-Nov 2025), Test: 35 days (Nov 2025-Jan 2026). 34 features, point_return labels at 8 horizons. |
| **Config** | No model config — pure statistical analysis scripts. GradBoost grid: max_depth=[3,5,7], lr=[0.01,0.05,0.1], n_estimators=300. |
| **Tool** | `hft-feature-evaluator/scripts/pre_training_analysis.py` (10 domains, 2.1 min) + `hft-feature-evaluator/scripts/archive/signal_diagnostics.py` (7 diagnostics, 3 sec) — **signal_diagnostics archived Phase 6 6D (2026-04-17), still runnable at new path** |
| **Status** | **COMPLETE — ALL GATES FAILED for model training. Signal is real but not tradeable.** |

**Decision Gates:**

| Gate | Metric | Value | Threshold | Verdict |
|---|---|---|---|---|
| G1 | Best feature \|IC\| at H=60 | **0.178** (spread_bps) | > 0.05 | **PASS** |
| G2 | Return std at H=60 | **79.4 bps** | > 5.0 | **PASS** |
| G3 | Walk-forward IC stability | **0.33** (Ridge, 136 folds) | > 2.0 | **FAIL** |
| G4 | Baseline DA | **0.529** (Ridge in-sample) | > 0.52 | **PASS (marginal)** |

**Pre-Training Analysis Key Findings (10 domains):**

1. **Return distribution H=60**: mean=+3.2 bps, std=79.4 bps, kurtosis=20.7, Huber delta=31.8 bps.
2. **Redundancy**: trade_count ≡ bin_trade_count (r=1.000), trf_volume/total_volume r=0.976. 3 features pruned → 11 effective features.
3. **IC matrix**: spread_bps IC increases monotonically with horizon (H=1: 0.035 → H=60: 0.178). bbo_update_rate similar pattern (0.022 → 0.113).
4. **Partial dependence NON-MONOTONIC**: spread_bps has dips at D4 (-5.4 bps) and D8 (-4.0 bps). No monotone constraints appropriate.
5. **Feature interactions**: session_progress × spread_bps is the ONLY synergistic pair (synergy=+0.048). All other pairs redundant.
6. **Temporal features add ZERO signal**: rolling_mean, rolling_slope, rate_of_change do NOT improve IC for any feature. This is opposite to MBO where temporal features tripled R².
7. **Walk-forward Ridge**: 136 daily folds, mean IC=0.075, std=0.23, stability=0.33, 50/136 folds negative.
8. **GradBoost (9-config grid)**: Best IC=0.036 (depth=7, lr=0.1), R²=-0.155. ALL configs have negative R². GradBoost WORSE than Ridge.
9. **Ridge on val**: IC=-0.048 (anti-correlated). Ridge on test: IC=+0.032 (barely positive).

**Signal Diagnostics Key Findings (7 diagnostics):**

1. **Multi-horizon**: Best combination is H=30 P85 (t=1.55, p=0.123). Short horizons H=1-5 are NEGATIVE (spread conditioning hurts). Signal only positive at H≥10.
2. **Val/Test OOS**: **ALL negative.** Val: -8.30 bps (fixed), -9.99 bps (rolling). Test: -3.42 bps (fixed), -5.70 bps (rolling). Unconditional also negative (NVDA drift reversed Oct 2025-Jan 2026).
3. **Regime robustness**: IC is robust — UP days IC=+0.194, DOWN days IC=+0.159 (p≈0.000), zero drift loss. Monthly IC positive ALL 8 months (+0.08 to +0.27). Signal IS real.
4. **One-sidedness**: IC(ret>0)=+0.201, IC(ret<0)=-0.001 at H=60. At shorter horizons, IC(ret<0) is actually NEGATIVE (H=5: -0.154, H=10: -0.120), meaning spread predicts LARGER losses at short horizons.
5. **P&L concentration**: 69% of in-sample profit from April 2025 (+1146 bps) and August 2025 (+670 bps) — crisis/high-vol months with extreme spread widening.
6. **Power**: 1,250 trades needed for significance, we have 382. Need ~478 days (~2 years).
7. **Session conditioning**: Midday (33-67%) is best session (mean_net=+13.02 bps, t=1.58) but small sample (133 trades).

**Root Cause Analysis — Why Feature IC=0.178 Doesn't Translate to Profitable Trades:**

1. **Feature IC ≠ Model IC**: E10's IC=0.178 is a univariate rank correlation (spread_bps vs returns). The model must learn this relationship from finite training data and generalize OOS. Ridge captures IC≈0.085 in-sample but IC=-0.048 on val.
2. **Regime dependence**: In-sample profits driven by crisis months (April 2025 tariff sell-off, August volatility) where extreme spread widening coincided with directional moves. This pattern did not recur in the calmer Oct 2025-Jan 2026 period.
3. **One-sided signal**: spread_bps predicts positive return magnitude (IC(ret>0)=+0.201) but not negative return magnitude (IC(ret<0)=-0.001). A symmetric regression model can't exploit this asymmetry.
4. **Noise dominance**: Return std=79-130 bps per trade vs expected signal=7-14 bps. SNR=0.07-0.12 per trade. Need 1,250+ trades for significance.
5. **Temporal features don't help**: Unlike MBO (where temporal features tripled R²), off-exchange features are already 60s aggregates. Rolling mean/slope of aggregates adds nothing.

**Lesson (17):** A statistically significant feature IC (0.178, p=4.4e-15, 100% bootstrap stability) does NOT guarantee tradeable alpha. The off-exchange spread_bps signal is genuine (robust across up/down days, drift-independent, stable across all 8 months) but (a) regime-dependent in its profitability (crisis months only), (b) one-sided (predicts positive return magnitude only), (c) swamped by noise (SNR≈0.07 per trade at H=30), and (d) fails OOS (val/test both negative). The 5-path evaluation framework correctly identifies signal presence but cannot determine tradeability — that requires walk-forward threshold backtesting with non-overlapping trades and OOS validation. Feature IC > 0.05 is a necessary but NOT sufficient condition for trading profitability.

**What NOT to repeat:**
- Do NOT train GradBoost/Ridge on these off-exchange features with point-return regression labels — GradBoost grid search (9 configs) achieved max IC=0.036 with negative R².
- Do NOT use temporal rolling features for off-exchange data — they add zero signal (unlike MBO where they triple R²).
- Do NOT assume in-sample positive P&L generalizes — 69% of profit came from 2 crisis months.

**Reports:**
- `hft-feature-evaluator/outputs/pre_training_analysis/PRE_TRAINING_ANALYSIS_REPORT.md` (10 domains)
- `hft-feature-evaluator/outputs/pre_training_analysis/analysis.json` (machine-readable)
- `hft-feature-evaluator/outputs/signal_diagnostics/SIGNAL_DIAGNOSTICS_REPORT.md` (7 diagnostics)
- `hft-feature-evaluator/outputs/signal_diagnostics/diagnostics.json` (machine-readable)

---

### E13: MBO 5-Path Evaluation + Signal Diagnostics with Point-Return Labels (2026-03-29)

| Field | Value |
|---|---|
| **Hypothesis** | Do MBO features have nonlinear dependence with point returns that E8's Spearman-IC-only methodology missed? Can the 5-path framework detect signal that E8 declared absent? |
| **Method** | (1) Computed point-return labels from existing forward_prices (no Rust re-export): `pr = (fp[:, k+H] - fp[:, k]) / fp[:, k] * 10000` at 8 horizons [1,2,3,5,10,20,30,60]. (2) Ran full 5-path evaluation (IC+dCor+MI+temporal+TE+regime+JMI+stability). (3) Ran 7-diagnostic signal suite (multi-horizon threshold backtest, val/test OOS, regime robustness, P&L, power). |
| **Data** | `data/exports/e5_timebased_60s_point_return/` — derived from `e5_timebased_60s` forward_prices. Train: 163 days, Val: 35, Test: 35. 98 features, PointReturn labels at 8 horizons. |
| **Config** | `hft-feature-evaluator/configs/mbo_98feat_point_return_lean.yaml` (100 dCor perms, 1000 subsample, 20 bootstraps) |
| **Tool** | hft-feature-evaluator v0.1.0 (5-path, 789 min) + `scripts/signal_diagnostics_mbo.py` (7 diagnostics, 3 sec) |
| **Status** | **IC=0.51 IS REAL BUT NOT TRADEABLE (Phase 9). IC measures within-day cross-sectional ranking (245 bins), NOT temporal prediction at trading frequency (4 bins/day). At traded bins: IC collapses to ~0, DA=48% (below random), quintile spread inverts. Equity+IBKR+trailing-rank: val +$69, test ALL negative. Signal is a microstructural property, not a tradeable edge.** |

**5-Path Evaluation Results:**

| Path | Pairs/Features | Key Finding |
|---|---|---|
| Path 1 (linear IC + BH) | **209** (feature, horizon) pairs | 3.5× more than E11's SmoothedReturn (102) — point returns have MORE linear signal |
| Path 2 (dCor + MI) | **0** pairs | Zero nonlinear signal — MBO is overwhelmingly linear (same as E11) |
| Path 3a (temporal IC) | **52** features | Rolling features add value |
| Path 3b (transfer entropy) | **0** pairs | No Granger-causal structure |
| Path 4 (regime IC) | **1,356** triplets | Widespread regime-conditional signal |
| Path 5 (JMI) | **56** features selected | Rich feature set |

**Classification: 30 STRONG-KEEP, 6 KEEP, 1 INVESTIGATE, 52 DISCARD (89 evaluated, 9 excluded)**

**Non-Price STRONG-KEEP Features (the tradeable candidates):**

| Feature | Index | IC at H=60 | CF Ratio | Stability | Signal Type |
|---|---|---|---|---|---|
| **spread_bps** | 42 | **+0.530** | 0.95 | 100% | Spread → returns (3× off-exchange) |
| **total_ask_volume** | 44 | **-0.182** | 0.53 | 100% | Ask supply → lower returns (forward) |
| **true_ofi** | 84 | **-0.146** | 1.09 | 100% | OFI mean-reversion at H=60 |
| **volume_imbalance** | 45 | **+0.126** | **0.01** | 100% | Bid>Ask → positive return (**pure forward**) |
| **depth_norm_ofi** | 85 | **-0.123** | 1.31 | 100% | Normalized OFI |
| ask_size_l4 | 14 | -0.128 | 0.57 | 100% | Deep book size |
| ask_size_l5 | 15 | -0.128 | 0.58 | 100% | Deep book size |
| ask_size_l7 | 17 | -0.111 | 0.65 | 100% | Deep book size |

**E8 Was Wrong About OFI**: E8 found true_ofi IC=-0.009 at H=10 and declared zero signal. E13 finds IC=-0.146 at H=60 (16× stronger). The signal exists but only at longer horizons (price impact decay takes 60 minutes, not 10). OFI sign REVERSES between smoothed labels (IC=+0.26, contemporaneous) and point returns (IC=-0.146, mean-reversion).

**Signal Diagnostics Results:**

| Metric | Off-Exchange (E12) | **MBO (E13)** |
|---|---|---|
| Best config | H=30, P85 | **H=60, P85** |
| Mean net return | +6.90 bps (not significant) | **+10.56 bps (p=0.032)** |
| **t-statistic** | 1.55 | **2.15** |
| **Statistically significant?** | NO (p=0.123) | **YES (p=0.032)** |
| Trades | 382 | 359 |
| Win rate | 52.1% | **54.9%** |
| Cumulative P&L | +2,636 bps | **+3,792 bps** |
| Positive weeks | 5/7 months | **18/26 weeks** |
| Max drawdown | -989 bps | **-752 bps** |
| Projected annual Sharpe | 2.03 | **2.83** |
| **Val OOS** | **-8.30 bps** | **-11.33 bps** |
| **Test OOS** | **-3.42 bps** | **-3.34 bps** |

**Regime Robustness (dramatically stronger than off-exchange):**

| Metric | Off-Exchange (E12) | **MBO (E13)** |
|---|---|---|
| UP-day IC | +0.194 | **+0.549** |
| DOWN-day IC | +0.159 | **+0.529** |
| Sign flip rate (UP) | 26.1% | **2.2%** |
| Sign flip rate (DOWN) | 24.3% | **1.4%** |
| IC(ret>0) at H=60 | +0.201 | **+0.434** |
| IC(ret<0) at H=60 | **-0.001** (one-sided!) | **+0.317** (both sides!) |

**E13 Phase 2: Standardized Ridge Walk-Forward Gate Test (2026-03-29)**

| Gate | Metric | Value | Threshold | Verdict |
|---|---|---|---|---|
| G1 | Walk-forward IC stability (block-averaged) | **2.865** | > 2.0 | **PASS** |
| G1 | Walk-forward IC stability (per-fold) | **1.073** | > 2.0 | **FAIL** |
| G2 | OOS prediction IC (test) | 0.066 (t=0.78) | > 0.10, t > 2.0 | **FAIL** |
| G3 | Walk-forward demeaned DA | **0.529** | > 0.52 | **PASS** |
| G4 | Alpha robustness (n alphas with stability > 2.0) | **5/5** | ≥ 2 | **PASS** |
| G5 | OOS profitable accuracy at 1.4 bps | 0.508 | > 0.50 | **PASS** |

**Ridge Walk-Forward Results (5 core features, α=1000, 133 daily folds):**

| Metric | In-Sample WF | Val (OOS) | Test (OOS) |
|---|---|---|---|
| Prediction IC (mean daily) | **0.139** | +0.070 (per-day) | — |
| Prediction IC (pooled) | ~0.007 | **-0.001** | **+0.066** |
| Demeaned DA | **0.529** | 0.504 | 0.508 |
| R² | -0.245 | -0.010 | -0.003 |
| Profitable DA (1.4 bps) | — | 0.498 | 0.508 |

**Subset comparison (α=1.0):**

| Subset | Features | Mean IC | Per-Fold Stability | Block Stability | Regime Shifts |
|---|---|---|---|---|---|
| **A (5 core)** | spread_bps, total_ask_volume, volume_imbalance, true_ofi, depth_norm_ofi | **0.141** | **1.084** | **2.857** | 19/133 |
| B (8 extended) | A + 3 ask sizes | 0.132 | 1.084 | 2.567 | 20/133 |
| C (14 full) | B + 6 KEEP | 0.124 | 0.806 | 1.821 | 29/133 |

**Key: Fewer features is better.** Adding sizes and KEEP features HURTS — increases regime shifts and reduces IC.

**Standardized Ridge coefficients (stable across 133 folds):**

| Feature | Mean β (std) | CV | Sign Flips | Interpretation |
|---|---|---|---|---|
| **spread_bps** | **+9.01** | 0.26 | 0 | Dominant predictor (wider spread → higher return) |
| true_ofi | -1.02 | 0.74 | 0 | OFI mean-reversion (consistent with E13 IC=-0.146) |
| total_ask_volume | -0.73 | 0.38 | 2 | Ask supply pressure |
| volume_imbalance | +0.42 | 0.49 | 6 | Bid/ask imbalance |
| depth_norm_ofi | +0.11 | 2.95 | 1 | Unstable — suppressor variable (sign opposite to raw IC) |

IC trend: slope=+0.00014, p=0.63 — **stable** (no signal degradation over time).

**Critical Discovery — Within-Day vs Cross-Day Signal Decomposition:**

The validation revealed that the walk-forward (per-day IC) and OOS (pooled IC) metrics measure DIFFERENT things:

| Component | In-Sample | Val (OOS) | Interpretation |
|---|---|---|---|
| **Within-day ranking IC** | +0.139 | **+0.070** | Model ranks correctly WITHIN each day |
| **Between-day level correlation** | — | **-0.211** | Model assigns WRONG levels ACROSS days |
| **Pooled IC** (combines both) | ~0.007 | **-0.001** | Between-day error cancels within-day signal |

The model's dominant spread_bps coefficient (+9.01 standardized) causes it to systematically predict higher returns on wide-spread days. Within each day, this ranking is correct (wider spread → higher relative return). But ACROSS days, high-spread days do NOT have higher MEAN returns — the spread-return relationship is a within-day phenomenon, not a between-day phenomenon. The fixed-coefficient Ridge cannot distinguish "high spread within today's distribution" from "today has wider spreads than yesterday."

Walk-forward P&L: **-1,397 bps** over 662 trades (long/short by prediction sign), win rate=46.2%.

**Lesson (18, REVISED):** MBO features HAVE genuine point-return signal at H=60 with 100% stability and strong walk-forward IC (0.139, per-fold stability 1.073 — 3.3× E12's 0.33). BUT the signal is **within-day only**: features rank bins correctly within a day but do NOT predict between-day return levels. A fixed-coefficient Ridge collapses OOS (pooled IC=-0.001 on val) because it cannot adapt to day-to-day spread regime shifts. The critical decomposition is: **within-day ranking (IC~0.07-0.14) works; between-day level prediction (correlation=-0.21) does not.** This is a fundamentally different failure from E12 (where the signal itself was weak) — here the signal is strong but non-stationary in its absolute relationship.

**Lesson (19):** Block-averaged stability metrics (grouping N daily folds into blocks) are mechanically inflated by ~sqrt(N) relative to per-fold stability. A block stability of 2.87 corresponds to a per-fold stability of only 1.07. Always report BOTH metrics and use per-fold as the primary gate. Similarly, walk-forward mean-of-daily-ICs and OOS pooled-across-days-IC are INCOMPATIBLE metrics — the former captures within-day ranking, the latter mixes within-day and between-day effects. Use the same metric for both in-sample and OOS evaluation.

**E13 Phase 4: Per-Day Z-Scored Ridge Walk-Forward — FIRST GO VERDICT (2026-03-29)**

| Gate | Metric | Without Z-Score | **With Z-Score** | Verdict |
|---|---|---|---|---|
| G1 | Block stability | 2.865 | **2.696** | **PASS** |
| G2 | OOS per-day IC (test) | — | **+0.145 (t=5.80)** | **PASS** |
| G3 | Demeaned DA | 0.529 | **0.535** | **PASS** |
| G4 | Alpha robust | 5/5 | **5/5** | **PASS** |
| G5 | Profitable DA (test) | 0.508 | **0.525** | **PASS** |

**Per-Day Z-Scored Ridge Results (5 core features, α=100, 133 daily folds):**

| Metric | In-Sample WF | Val (OOS) | Test (OOS) |
|---|---|---|---|
| Per-day mean IC | **0.132** | **+0.078 (t=2.89)** | **+0.145 (t=5.80)** |
| Pooled IC | — | +0.051 | +0.098 |
| Demeaned DA | **0.535** | 0.537 | 0.548 |
| Profitable DA (1.4 bps) | — | 0.505 | 0.525 |
| Walk-forward P&L | **+3,411 bps** | +208 bps | +68 bps |

**What changed**: 3 lines of code added per-day z-scoring in `load_all_data()`:
```python
day_mu = np.mean(feat_raw, axis=0)
day_sigma = np.std(feat_raw, axis=0)
feat_z = (feat_raw - day_mu) / day_sigma
```
This removes between-day feature distribution shift (spread mean 0.84 → 0.58 bps over 8 months) so the model sees only within-day relative position. The Ridge coefficients now mean "1σ increase in today's spread → expected β bps return" — a relationship that IS stationary across days.

**Standardized coefficients (stable across 133 folds):**

| Feature | Mean β | CV | Sign Flips | Stable? |
|---|---|---|---|---|
| true_ofi | **-5.90** | 0.20 | 0 | YES |
| total_ask_volume | **-3.11** | 0.12 | 0 | YES |
| spread_bps | **+2.83** | 0.13 | 0 | YES |
| depth_norm_ofi | +1.94 | 0.31 | 0 | YES |
| volume_imbalance | +0.03 | 12.85 | 16 | **NO** (should drop) |

IC trend: slope=+0.00022, p=0.44 — stable (no degradation).

**Independently verified**: All numbers reproduced to full precision by independent computation. No bugs found.

**Caveats (validated, do not invalidate GO):**

1. **Per-day z-scoring uses full-day data.** Acceptable for offline walk-forward but NOT directly implementable in real-time. Production requires trailing-window z-score (e.g., last 60 minutes). Untested.
2. **82.8% long bias.** The positive intercept (= training mean return) makes most predictions positive. Value comes from correctly identifying ~18% of trades as short candidates (short trade mean return: -6.9 bps actual, correctly captured).
3. **OOS P&L is marginal.** Test mean trade: +0.40 bps. Val: +1.19 bps. Barely above cost (1.4 bps). In-sample P&L (+3,411) benefits from training-period positive drift.
4. **Subset A (5 features) remains best.** Adding features still hurts. volume_imbalance is unstable (CV=12.85) and should be dropped in production.

**Lesson (20):** Per-day feature z-scoring is the key to converting within-day microstructure signal to OOS-viable predictions. The 3-line change fixed what 12 prior experiments could not. The within-day ranking (IC=0.07-0.15 OOS) is real and capturable when between-day distribution shift is removed. Without z-scoring, the model learns "absolute spread level → return" which is non-stationary. With z-scoring, it learns "relative spread position within today → return" which IS stationary. This is the first approach in the pipeline's history to pass all 5 decision gates simultaneously.

**Lesson (21):** When walk-forward in-sample P&L is dramatically different from OOS P&L (+3,411 vs +68-208 bps), investigate (a) prediction sign bias (82% long in-sample during positive-drift period), (b) intercept dominance, and (c) the difference between within-day ranking value vs between-day level prediction. OOS P&L is the honest metric.

**What NOT to repeat:**
- Do NOT test MBO features for point returns at H=10 only — signal builds over 60 minutes (E8's error)
- Do NOT use fixed absolute thresholds for OOS trading — spread distribution drifts 30-40% over months
- Do NOT conclude "zero signal" from Spearman IC alone — the 5-path framework found 209 pairs that IC missed
- Do NOT use block-averaged stability as primary gate — it is mechanically inflated by ~sqrt(block_size)
- Do NOT compare per-day mean IC (walk-forward) to pooled IC (OOS) — they measure different things
- Do NOT add more features hoping to improve Ridge — 5 core features beat 8 and 14 feature sets
- ~~Do NOT skip per-day feature z-scoring~~ **RETRACTED (Phase 5):** Per-day full-day z-score has look-ahead bias
- Do NOT trust in-sample P&L alone — OOS P&L is 16-50× smaller due to drift and sign bias
- Do NOT use within-day normalization without testing causal variants (expanding, trailing)

**E13 Phase 5: Trailing Z-Score Test + 4-Strategy Comparison (2026-03-29)**

| Strategy | Mean IC | Per-Fold Stability | Block Stability | Regime Shifts | Future Data? |
|---|---|---|---|---|---|
| **Raw (no z-score, Ridge internal std)** | **0.139** | **1.073** | **2.865** | **20/133** | **NO** |
| Full-day z-score | 0.133 | 1.047 | 2.696 | 23/133 | **YES (79.6% at t=50)** |
| Trailing K=60 | 0.036 | 0.195 | 0.397 | 62/133 | NO |
| Expanding (causal) z-score | -0.026 | -0.190 | -0.372 | 89/133 | NO |

**Phase 4 GO verdict RETRACTED.** Independent validation revealed:
1. Full-day z-score at sample t uses mean/std from ALL 245 daily samples. At t=50, **79.6% is future data**. This is not implementable in real-time.
2. The causal equivalent (expanding z-score using [0..t]) collapses to stability=-0.190 — proving the full-day benefit came from look-ahead, not signal.
3. Trailing K=60 (real-time implementable) collapses to stability=0.195 — signal destroyed.
4. **Raw features with Ridge's internal pooled-training standardization are the BEST causal approach** (stability=1.073 per-fold, 2.865 block, 20 shifts). This uses ONLY past training data for standardization — fully causal.

**HOWEVER: Raw Ridge per-day OOS IC was NEVER computed.** Phase 3 only reported pooled OOS IC (val=-0.001, test=+0.066). The metric mismatch (pooled IC ≠ per-day IC) means the val=-0.001 may mask a positive per-day IC (as happened with full-day z-score: pooled=-0.001 but per-day=+0.078). **The raw Ridge per-day OOS IC is the CRITICAL missing number.**

**Lesson (20, REVISED):** Per-day full-day z-scoring introduces look-ahead bias — at time t, using the full day's mean/std includes 79.6% future information (at t=50 of 245). The expanding (causal) equivalent collapses to negative stability. The correct causal approach is Ridge's internal standardization using pooled HISTORICAL training statistics. **Always test causal variants (expanding, trailing) when within-day normalization shows surprising results.**

**Lesson (22, NEW):** When a normalization strategy dramatically improves results (stability 1.07 → 2.70), verify the information set: does the transform use data available at prediction time? Full-day z-scoring "works" because it secretly reveals the day's feature distribution, which contains information about the day's returns. This is a subtle form of look-ahead that passes naive walk-forward tests (which check train/test DAY boundaries but not within-day temporal ordering).

**E13 Phase 6: Raw Ridge Per-Day OOS IC — GO (2026-03-29)**

Added per-day IC computation to `ridge_walkforward_mbo.py` (schema v2) with correct dual-offset indexing (fixes latent bug in z-score script), Grinold's Law, calibration, per-day DA/profitable_da, long/short returns, UP/DOWN-day IC. Results independently verified via separate numpy Ridge implementation.

| Metric | Val (35 days) | Test (35 days) | Gate |
|---|---|---|---|
| **Per-day mean IC** | **0.0704** (t=3.26, corrected 2.76) | **0.1265** (t=6.34) | G2: > 0.05 AND t > 2.0 → **PASS** |
| Per-day IC std / median | 0.1278 / 0.074 | 0.1181 / 0.125 | — |
| Frac positive | 74% (26/35) | 89% (31/35) | — |
| Per-day IC [Q25, Q75] | [-0.000, +0.128] | [+0.043, +0.213] | — |
| Walk-forward stability (block) | 2.865 | — | G1: > 2.0 → PASS |
| Walk-forward demeaned DA | 0.529 | — | G3: > 0.52 → PASS |
| Alpha robustness | 5/5 | — | G4: >= 2 → PASS |
| OOS profitable DA (1.4 bps) | 0.498 | 0.508 | G5: > 0.50 → PASS (test) |
| Grinold E[r] = IC × σ_r | 3.30 bps (net +1.90) | 6.81 bps (net +5.41) | — |
| Pooled IC (reference) | -0.001 | +0.066 | — |

**Verdict: GO — all 5 gates pass.** First valid (non-look-ahead) GO in E13.

**IC Autocorrelation Check:** Val ACF(1)=+0.166 → effective N_eff=25.1, corrected t=2.76 (still > 2.0). Test ACF(1)=-0.071, no correction needed.

**Caveats (independently validated):**
1. **Negative calibration slope** — val=-1.109, test=-0.111. Higher predictions → lower actuals at POOLED level. Model's absolute values are wrong; only within-day RANKING works. Trading MUST use rank-based positioning, NOT sign-based.
2. **Val DOWN-day IC = 0.046** (below 0.05) vs UP-day IC = 0.108. Test balanced (UP=0.124, DOWN=0.130). Val has directional asymmetry.
3. **Val long/short spread = -3.34 bps** — when model says "go long" (pred > 0), actual returns are more negative than "go short." Test spread = +5.70 bps (correct direction). Sign-based trading fails on val.
4. **DA near 50%** — val per-day DA=0.498, test=0.508. Model can't predict direction, only rank.

**CRITICAL DISCOVERY — spread_bps dominance:**

| Signal | Walk-Forward IC | Val Per-Day IC | Val/WF Ratio |
|---|---|---|---|
| spread_bps alone (single-feature Ridge) | **0.564** | **~0.51** | ~90% |
| 5-feature Ridge model | 0.139 | 0.070 | 50% |

The 5-feature Ridge **DESTROYS 86% of spread_bps's within-day signal**. The other 4 features (true_ofi, total_ask_volume, volume_imbalance, depth_norm_ofi) actively hurt within-day ranking when combined via a globally-fit Ridge. Root cause: Ridge optimizes POOLED loss (between + within day), and between-day feature-return relationships CONFLICT with within-day relationships. The Ridge learns a compromise that is suboptimal for within-day prediction.

**Lesson (23, NEW):** When a multi-feature model achieves lower per-day IC than a single-feature baseline, the model's pooled-loss optimization is conflicting with within-day signal structure. Always compare single-feature per-day IC to the model's per-day IC. The simpler signal may be superior for within-day trading.

**Lesson (24, NEW):** Negative calibration slope (pred vs actual) at the pooled level, combined with positive per-day IC, diagnoses the within-day / between-day split definitively. It means: rankings correct within days, absolute levels wrong across days. For execution: use within-day rank (quintile/decile positioning), never use raw prediction sign or magnitude.

**Lesson (25, NEW):** For the 0DTE backtester, test spread_bps rank directly alongside the Ridge model. If spread_bps alone outperforms, the model adds negative value for this signal structure.

**E13 Phase 7: Feature OOS Signal Analysis — SPREAD_PRIMARY (2026-03-29)**

Computed per-day OOS IC for all 5 core features individually + Ridge model across all 70 OOS days. Deep validation: signed vs absolute IC decomposition, non-overlapping subsampling, spread persistence, label timing (k=5 offset verified), UP/DOWN conditioning, concept drift.

**Feature Ranking (val per-day |IC|):**

| Rank | Signal | Val Per-Day IC | Test Per-Day IC | Q-Spread (bps) | Grinold Net (bps) |
|---|---|---|---|---|---|
| **1** | **spread_bps** | **+0.511** | **+0.601** | **+64.3** | **+22.6** |
| 2 | total_ask_volume | -0.105 | -0.136 | -7.0 | — |
| 3 | volume_imbalance | +0.071 | +0.096 | +3.9 | — |
| 4 | ridge_5feat | +0.070 | +0.127 | +6.2 | +1.9 |
| 5 | true_ofi | -0.062 | -0.097 | -3.1 | — |
| 6 | depth_norm_ofi | -0.059 | -0.086 | -3.3 | — |

**Decision gate:** spread_bps val per-day IC = 0.511 > 0.30 → **SPREAD_PRIMARY**.

**Deep Validation Results (independently verified):**

| Check | Result | Significance |
|---|---|---|
| Signed vs absolute IC | IC(spread, return)=+0.511, IC(spread, \|return\|)=-0.082 | **Purely DIRECTIONAL** (not volatility proxy) |
| Non-overlapping IC (stride=30) | IC=+0.502 (8 samples/day) | Overlap does NOT inflate IC |
| spread_bps ACF(1) | ≈ -0.001 to -0.030 | Independent observations (no persistence) |
| UP/DOWN day IC (val) | UP=0.409 (14d), DOWN=**0.578** (21d) | **Bilateral** — stronger on DOWN days |
| UP/DOWN day IC (test) | UP=0.603 (21d), DOWN=0.598 (14d) | Symmetric |
| Negative IC days | 2/35 val, 1/35 test | >94% consistency |
| Monthly stability | All months 0.40-0.63 | No regime breakdown |
| Label timing | k=5 verified (5-min gap, no leakage) | Feature at T, label from T+5 to T+65 |
| Concept drift (val) | slope=-0.006, p=0.093 | Borderline |
| Concept drift (test) | slope=+0.001, p=0.843 | Stable |

**Head-to-head:** spread_bps beats Ridge on **94% of val days** and **100% of test days**. Ridge model wins on 0% of days.

**Quintile spread:** Val Q5-Q1 = +64.3 bps (Q1=-39.1, Q5=+25.2). Test Q5-Q1 = +86.1 bps. Monotonic progression across all 5 quintiles.

**Normalization:** spread_bps values are RAW bps (range 0.53-1.13, not z-scored). Rank-based IC is unaffected by normalization status.

**Lesson (26, NEW):** Validate IC mechanism with IC(feature, |return|). If IC(abs) ≈ 0 but IC(signed) >> 0, the signal is purely DIRECTIONAL — spread does NOT predict volatility, it predicts return direction. This eliminates the "volatility × drift" artifact hypothesis.

**Lesson (27, NEW):** Verify IC robustness to subsampling at stride=30 (one sample per 30 minutes). If IC(stride=30) ≈ IC(stride=1), the 98.3% return overlap at H=60 is NOT inflating the IC. Combined with ACF(1) ≈ 0 for the feature, each observation contributes independent information.

**Lesson (28, NEW):** The Ridge model ranks LAST among the 5 individual features for val per-day IC (0.070 < 0.511). When a model underperforms its own dominant input, pooled loss optimization is actively harmful. For within-day microstructure signals, raw feature ranking may be the optimal strategy.

**E13 Phase 8: 0DTE Backtester — Signal Has Alpha, Costs Consume It (2026-03-29)**

Per-day backtesting (correct for 0DTE expiry). 3 z-score variants (global, warmup-30min, expanding) + Ridge model. Deep ITM (delta=0.95). Threshold sweep [0.0, 0.5, 1.0, 1.5, 2.0] z-score units. Long-short and long-only. Post-hoc theta correction (ATM overestimates deep ITM by 3.87x, correction ratio 0.742).

**P&L Decomposition (spread_warmup z=0 LS, val, 174 trades):**

| Component | Value |
|---|---|
| Gross equity P&L (signal alpha) | **+$1,390** |
| ATM theta (backtester, overestimated) | -$3,698 ($21.25/trade) |
| Reported PnL | -$2,308 (-2.31%) |
| Deep ITM theta correction | +$2,742 |
| **Corrected PnL (realistic deep ITM)** | **+$434** (+0.43%) |
| Net alpha per trade | ~$2.50 |

**Best Corrected Results:**

| Signal | Split | Config | Corrected Return | Trades | WR |
|---|---|---|---|---|---|
| spread_warmup | val | z=0.0 LS | **+0.43%** | 174 | 31.0% |
| spread_expanding | val | z=0.0 LS | +0.14% | 174 | 29.3% |
| ridge_5feat | val | z=3.0 LS | +0.24% | 136 | 29.4% |
| spread_expanding | test | z=0.5 LO | -0.15% | 82 | 30.5% |
| All test configs | | | Negative | | |

**Validated Root Causes (independently verified):**

1. **Theta overwhelms alpha**: $21.25/trade ATM theta vs $8/trade gross alpha. Even corrected deep ITM theta ($5.49/trade) consumes most of the signal.
2. **Z-score structurally biased SHORT**: 83% of post-warmup samples have z < 0 (spread narrows after opening). Strategy shorts 91% of the time. Confirmed NOT a bug — real intraday spread compression.
3. **IC ≠ P&L**: IC=0.51 measures ranking across all 245 overlapping bins per day. Backtester trades at 3-4 non-overlapping bins. IC at traded bins = 0.44 (still strong) but entry-bin returns average -5.1 bps (better than random -8.4 bps by only 3.3 bps).
4. **Training→val regime shift**: Training spread mean=0.837, val=0.577 (-0.66 std shift). Global z-score failure is expected. Per-day z-scores (warmup, expanding) partially fix this but don't change the structural SHORT bias.
5. **Intraday spread is NOT "wide morning, narrow afternoon"**: Hourly quarters show flat spread (0.56-0.59 bps). The warmup effect is subtle but sufficient to bias z-scores negative.

**Lesson (29, NEW):** High per-day IC (ranking metric) does NOT guarantee tradeable P&L. IC=0.51 across 245 overlapping bins measures "if you could optimally rank all bins, the top outperforms the bottom by 64 bps." But trading constraints (non-overlapping holds, fixed entry timing, directional z-score) extract only a fraction of this theoretical alpha. Gross equity alpha was +$8/trade, but costs were +$21/trade (ATM) or +$16/trade (deep ITM corrected).

**Lesson (30, NEW):** 0DTE theta is the DOMINANT cost for 60-minute holds, not spread or commission. ATM theta: $24.53/contract at 120min-to-close. Deep ITM (delta=0.95) actual theta: $6.33/contract (3.87x lower than ATM, which the backtester uses). For future backtest design: either (a) fix the theta function for deep ITM, (b) reduce hold period to minimize theta, or (c) trade equity instead of options.

**Lesson (31, NEW):** The gross equity P&L (before theta) was **POSITIVE** (+$1,390, 174 trades). This means spread_bps has real predictive value at the equity level. The bottleneck is 0DTE option execution costs, not signal quality. Equity-level trading (no theta) should be explored.

**Lesson (32, NEW):** Causal intraday z-scoring (warmup, expanding) produces structurally biased direction because spread_bps has intraday compression (slightly wider at open, narrower rest of day). This is NOT a normalization artifact — it's a real microstructure pattern. The IC signal is about RANKING (wider-than-peers → higher return), not about being above/below a fixed baseline.

**E13 Phase 9: Equity Backtest + Trailing-Rank Signal — NOT TRADEABLE (2026-03-30)**

Added trailing-rank signal (percentile of spread within trailing 60-min window, balanced ~50% BUY/SELL), IBKR equity costs ($1.70 RT, 0.97 bps breakeven), equity P&L reporting. Per-day backtesting. 5 signal variants × 5-10 thresholds × 2 direction modes × 2 splits.

**The Definitive Root Cause — IC ≠ Tradeable P&L:**

| Metric | All Bins (245/day) | Traded Bins (4/day) | Degradation |
|---|---|---|---|
| Val per-day IC | 0.511 | **0.331** | -35% |
| Pooled IC (val) | — | **-0.048** | Near zero |
| Direction accuracy (val) | — | **48.2%** | Below random |
| Quintile spread (val) | +64.3 bps | **-5.11 bps** | INVERTED |
| Long mean return (val) | — | **-4.12 bps** | Wrong direction |
| Short mean return (val) | — | **+7.53 bps** | Shorts win |

**IC=0.51 measures within-day cross-sectional ranking across 245 overlapping bins.** It says: "if you could simultaneously rank all 245 bins and go long the widest-spread ones, you'd make money." But trading requires temporal prediction at 4 non-overlapping entry points per day. At those points, the IC collapses, direction inverts, and quintile spreads reverse.

**Best Results (val equity P&L, IBKR costs):**

| Signal | Config | Equity Return | Trades | Equity WR | Buys/Sells |
|---|---|---|---|---|---|
| trailing_rank_ibkr | z=0.0 LS | **+0.07%** (+$69) | 174 | 40.7% | 52/122 |
| trailing_rank_ibkr | z=0.6 LS | -0.25% | 108 | 51.2% | 45/63 |
| **All test configs** | | **Negative** | | | |

**Trailing rank DID fix the direction balance** — 38% positive (vs 8-17% for z-scores). But balanced direction with no temporal signal produces random P&L.

**Val +$69 is noise** — $0.40/trade on 174 trades, well within SE of ~$500 (random P&L variation). Not statistically distinguishable from zero.

**Lesson (33, DEFINITIVE):** Per-day IC measures within-day cross-sectional ranking across all overlapping bins. This is a DIFFERENT quantity from temporal predictive power at sparse non-overlapping entry points. A feature can have IC=0.51 (strong cross-sectional ranking) while having DA=48% (below random at trading frequency). The quintile spread of +64 bps at all bins INVERTS to -5 bps at traded bins. **IC is necessary but NOT sufficient for tradeability. The IC must be validated at the actual trading frequency and cadence before any backtest.**

**Lesson (34, NEW):** With H=60 (60-min returns) and hold=60 (60-min holding), only ~4 non-overlapping trades per day are possible. This gives ~140 trades per 35-day split. With return std=59 bps, the SE per quintile (28 trades) is ±11 bps — making all quintile differences statistically insignificant. **Sample size at the trading cadence is too small for any reliable signal extraction.**

**Lesson (35, NEW):** The "gross equity alpha was positive" finding from Phase 8 (+$1,390) used XNAS costs (3.27 bps/trade) which include exchange-level slippage. With IBKR costs (0.97 bps/trade) and the SAME z-score signal, val equity P&L was also positive (+$69 with trailing rank). But test was negative for ALL configs. **Val-positive-test-negative is the hallmark of noise, not signal.**

**E13 CONCLUDED.** The spread_bps IC=0.51 is a real microstructural property of NVDA but is NOT tradeable with any tested strategy (z-score, trailing rank, Ridge model) at any tested cost level (XNAS, IBKR, 0DTE options) on any split (val positive is noise, test ALL negative).

**Reports:**
- `hft-feature-evaluator/classification_table_mbo_point_return_lean.json` (E13 5-path classification)
- `hft-feature-evaluator/outputs/signal_diagnostics_mbo/diagnostics.json` (7 threshold diagnostics)
- `hft-feature-evaluator/outputs/ridge_walkforward_mbo/results.json` (Raw Ridge v2, per-day IC)
- `hft-feature-evaluator/outputs/ridge_walkforward_mbo_zscore/results.json` (Full-day z-scored, **GO RETRACTED**)
- `hft-feature-evaluator/outputs/ridge_walkforward_mbo_trailing/results.json` (Trailing z-scored, NO-GO)
- `hft-feature-evaluator/outputs/feature_oos_analysis/results.json` (6-signal OOS analysis, **SPREAD_PRIMARY**)
- `lob-backtester/outputs/backtests/spread_bps_signal/results.json` (0DTE backtest, **ALPHA EXISTS, COSTS KILL**)

---

## Regression Experiments

### TLOB 128-feat Regression H10 (FIRST REGRESSION EXPERIMENT)

| Field | Value |
|---|---|
| Config | `configs/experiments/nvda_tlob_128feat_regression_h10.yaml` |
| Output | `outputs/experiments/nvda_tlob_128feat_regression_h10/` |
| Report | `reports/tlob_regression_2026_03_15.md` |
| Model | TLOB (2 layers, hidden=32, 2 heads, regression head) |
| Data | `nvda_xnas_128feat_regression` (233 days, XNAS ITCH) |
| Features | All 128 (98 stable + 30 experimental) |
| Labeling | Regression (continuous bps, TLOB smoothed-average) |
| Horizon | H10 only (index 0) |
| Loss | Huber (delta=5.0 bps) |
| Training | batch=128, lr=5e-4, cosine, weight_decay=0.01, grad_clip=1.0, seed=42 |
| Epochs | 15 (best at epoch 6, early stopping at 14) |
| Parameters | 693,190 |
| **Test R²** | **0.4642** |
| **Test IC** | **0.6766** |
| **Test DA** | **0.7494** |
| Test MAE | 2.43 bps |
| Test RMSE | 3.43 bps |
| Test profitable accuracy | 0.9263 (>5 bps moves) |
| Baseline (Linear Ridge) | R²=0.170, IC=0.433 |
| **Improvement over linear** | **R² 2.73x, IC 1.56x** |
| Per-day R² range | [0.331, 0.546], all positive |
| Validation status | All metrics independently verified; zero data leakage confirmed |

### TLOB T=20 Sequence Ablation

| Field | Value |
|---|---|
| Output | `outputs/experiments/nvda_tlob_128feat_regression_h10_T20/` |
| Model | TLOB (2 layers, hidden=32, 2 heads) -- same as T=100 |
| Sequence | Last 20 of 100 timesteps (sliced at training time) |
| Parameters | 93,710 (7.4x smaller than T=100) |
| Epochs | 15 (best at epoch 2) |
| **Test R²** | **0.4114** (88.6% of T=100) |
| **Test IC** | **0.6742** (99.6% of T=100) |
| **Test DA** | **0.7497** (identical to T=100) |
| Test MAE | 2.49 bps |
| **Finding** | IC and DA identical; R² gap of 0.053 confirms signal half-life ~5 timesteps |

### HMHP-R Multi-Horizon Regression (H10-Primary)

| Field | Value |
|---|---|
| Config | `configs/experiments/nvda_hmhp_regression_h10_primary.yaml` |
| Output | `outputs/experiments/nvda_hmhp_regression_h10_primary/` |
| Model | HMHP-R (TLOB encoder, 3 cascading decoders, gate fusion, confirmation) |
| Horizons | [10, 60, 300] |
| Loss weights | H10=0.50, H60=0.25, H300=0.15, consistency=0.10 |
| Parameters | 171,379 |
| Epochs | 20 (best at epoch 16) |
| **Test R²** | **0.4535** |
| **Test IC** | **0.6706** |
| **Test DA** | **0.7476** |
| Test MAE | 2.45 bps |
| **Finding** | Multi-horizon did NOT improve H10; persistence at H60/H300 hurts shared encoder |

### Regression Backtests (IBKR 0DTE)

| Variant | Hold | Threshold | Trades | Option Return |
|---------|------|-----------|--------|---------------|
| H10 hold | 10 events | 0.7 bps | 4,270 | -19.75% |
| H10 hold | 10 events | 5.0 bps | 1,799 | -7.53% |
| H10 hold | 10 events | 10.0 bps | 54 | -0.35% |
| H60 hold | 60 events | 3.0 bps | 775 | -2.71% |
| H60 hold | 60 events | 10.0 bps | 45 | -0.77% |

**Critical finding**: Model's 74.9% directional accuracy on smoothed labels translates to ~38% execution win rate due to label-execution mismatch (smoothed average != point-to-point tradeable return).

### TLOB Point-Return Regression H10 (FAILED -- Zero Signal)

| Field | Value |
|---|---|
| Config | `configs/experiments/nvda_tlob_128feat_regression_pointreturn_h10.yaml` |
| Data | `nvda_xnas_128feat_regression_pointreturn` (return_type=point_return) |
| Model | TLOB (2 layers, hidden=32, 2 heads) -- identical to REG-01 |
| **Result** | **R²=-0.0000, IC=0.0008, DA=0.4991 after 3 epochs -- model learned NOTHING** |
| **Root cause** | Point-to-point H10 returns have ZERO correlation with features (DEPTH_NORM_OFI IC=-0.005, R²=0.0005) |
| **Key finding** | OFI features predict smoothed-average returns (IC=0.309) but NOT point-to-point returns (IC=-0.005). The smoothed avg and point return are only 0.24 correlated. See `reports/CONSOLIDATED_FINDINGS_2026_03.md` Finding 1. |
| Status | Stopped at epoch 2, output cleaned up |

### Simple Model Ablation Ladder (2026-03-16)

| Field | Value |
|---|---|
| Script | `scripts/archive/run_simple_model_ablation.py` — **archived Phase 6 6D (2026-04-17)**, runnable at new path |
| Output | `outputs/experiments/simple_model_ablation/ablation_results.json` |
| Report | `reports/ABLATION_FINDINGS_2026_03_16.md` |
| Data | Same smoothed-average data as REG-01 (50,724 test samples) |
| **L0: IC-Weighted** | R²=-0.487, IC=0.342 (5 params) |
| **L1: Temporal Ridge** | **R²=0.324, IC=0.616 (53 params) -- 91% of TLOB IC** |
| **L2: Ridge+Poly** | R²=0.324, IC=0.616 (68 params) -- polynomial adds nothing |
| **L3: GradBoost** | **R²=0.397, IC=0.617 (200 trees) -- 85.6% of TLOB R²** |
| **L3b: GradBoost(raw128)** | R²=0.137, IC=0.446 -- temporal structure triples R² |
| **Key finding** | Temporal Ridge captures 91% of TLOB IC with 0.008% of parameters. Deep learning adds real but modest value (9% IC, 14% R²). |
| Status | Validated (all metrics independently verified) |

### TWAP Backtest (2026-03-16)

| Field | Value |
|---|---|
| Script | inline (TWAPStrategy in `lobbacktest/strategies/twap.py`) |
| Model | TLOB smoothed-average (best checkpoint, R²=0.464) |
| **Best result** | min=5bps, w=10: **-5.56%** (vs -7.53% point-to-point) |
| **High conviction** | min=8bps, w=5: **-0.93%** (same as point-to-point) |
| **Key finding** | TWAP improves over point-to-point by 11-26% but does not achieve profitability |
| Status | Completed |

### Temporal Ridge Full Pipeline (2026-03-16)

| Field | Value |
|---|---|
| Config | `configs/experiments/nvda_temporal_ridge_h10.yaml` |
| Output | `outputs/experiments/nvda_temporal_ridge_h10/` |
| Script | `scripts/archive/run_simple_training.py` — **archived Phase 6 6D (2026-04-17)**, runnable at new path |
| Model | TemporalRidge(alpha=1.0, features=53) -- sklearn Ridge |
| **Test R-squared** | **0.324** |
| **Test IC** | **0.616** |
| **Test DA** | **0.722** |
| Test MAE | 2.75 bps |
| Parameters | 54 (53 coefficients + 1 intercept) |
| Fit time | 0.9 seconds |
| Backtest (10bps, h10) | -1.14% option return (333 trades) |
| Status | Full pipeline validated: config -> train -> evaluate -> export -> backtest |

### Temporal GradBoost Full Pipeline (2026-03-16)

| Field | Value |
|---|---|
| Config | `configs/experiments/nvda_temporal_gradboost_h10.yaml` |
| Output | `outputs/experiments/nvda_temporal_gradboost_h10/` |
| Model | TemporalGradBoost(trees=200, depth=5, features=53) -- sklearn GBR |
| **Test R-squared** | **0.397** |
| **Test IC** | **0.617** |
| **Test DA** | **0.723** |
| Test MAE | 2.58 bps |
| Parameters | ~6,400 (200 trees * 32 leaves) |
| Fit time | 171.5 seconds |
| Status | Full pipeline validated |

### Kolm Per-Level OF Experiment (2026-03-17)

| Field | Value |
|---|---|
| Config | `feature-extractor-MBO-LOB/configs/nvda_xnas_kolm_of_regression.toml` |
| Output | `data/exports/nvda_xnas_kolm_of_regression/` (28 GB, 233 days) |
| Report | `reports/kolm_of_experiment_2026_03_17.md` |
| Features | 136 (98 stable + 8 inst + 6 vol + 4 season + 20 kolm_of) |
| event_count | 100 (~195ms/sample, H1-H3 within Kolm effective range) |
| Horizons | [1, 2, 3, 5] (point-return, bps) |
| Sequences | Train: 1,695,700 / Val: 542,958 / Test: 521,348 |
| **Kolm OF IC (H1)** | **0.0001** (20 features, all indistinguishable from zero) |
| **Scalar OFI IC (H1)** | **0.0703** (DEPTH_NORM_OFI) |
| **Ridge (Kolm OF only)** | R²=-0.0002, IC=0.0001 (zero predictive power) |
| **Ridge (scalar OFI only)** | R²=0.0044, IC=0.0730 (weak but real) |
| **Ridge (full 136 features)** | R²=-0.0073, IC=0.1059 |
| **Decision gate** | **FAIL** (Kolm OF IC=0.0001 < threshold 0.05) |
| **Phase 5 (training)** | **CANCELLED** |
| **Root cause** | Cumulative-per-window OF destroys the per-event temporal dynamics that Kolm's LSTM exploits. Scalar OFI works because depth normalization creates a mean-reverting signal. |
| **Key finding** | Per-level OF is not useful in cumulative-per-window architecture. Shorter horizons (event_count=100) DO increase scalar OFI IC from 0.005 to 0.070 for point-returns. |
| Status | Completed, documented, infrastructure retained |

---

## Classification Experiments (128-feature, profiler-aligned)

These experiments use the current 128-feature, 233-day, profiler-aligned dataset.

### HMHP 128-feat XNAS H10 Primary

| Field | Value |
|---|---|
| Config | `configs/experiments/nvda_hmhp_128feat_h10_primary.yaml` |
| Output | `outputs/experiments/nvda_hmhp_128feat_xnas_h10/` |
| Model | HMHP (TLOB encoder, full cascade, gate fusion, confirmation) |
| Data | `nvda_xnas_128feat` (233 days, XNAS ITCH) |
| Features | 119 active (analysis_ready_128 preset: 128 minus 9 dead) |
| Labeling | TLOB (smoothed endpoint return) |
| Horizons | [10, 60, 300] |
| Loss weights | H10=0.50, H60=0.25, H300=0.10, consistency=0.05, final=0.10 |
| Training | batch=256, lr=5e-4, cosine, weight_decay=0.01, grad_clip=1.0, seed=42 |
| Epochs | 30 (ran to completion; best val_loss=0.8451 at epoch 27) |
| Parameters | 170,980 |
| **Val accuracy** | **59.19%** |
| **Test accuracy** | **59.62%** |
| Test H10 | 59.62% |
| Test H60 | 41.22% |
| Test H300 | 38.18% |
| Test macro F1 | 0.5915 |
| Test agreement ratio | 91.80% |
| Test directional acc | 63.80% |
| Train-val gap | 1.10pp (minimal overfitting) |
| **Confidence analysis** | |
| agree=1.0 (test) | 76.5% of samples, 63.08% acc, 89.52% dir acc |
| agree=1.0 + confirm>0.65 (test) | 51.5% of samples, 68.72% acc, 93.88% dir acc |
| agree<1.0 (test) | 23.5% of samples, 48.38% acc (near random) |
| Lesson | MLOFI adds +1.4pp over 116-feat prior. Confirmation ceiling at 0.667 limits thresholding. |

### HMHP 128-feat ARCX H10

| Field | Value |
|---|---|
| Config | `configs/experiments/nvda_hmhp_128feat_arcx_h10.yaml` |
| Output | `outputs/experiments/nvda_hmhp_128feat_arcx_h10/` |
| Model | Same architecture as XNAS |
| Data | `nvda_arcx_128feat` (233 days, ARCX PILLAR) |
| Epochs | 23 (early stopped; best val_loss=0.8447 at epoch 15) |
| **Val accuracy** | **59.73%** |
| **Test accuracy** | **58.79%** |
| Test H10 | 58.79% |
| Test H60 | 41.79% |
| Test H300 | 38.11% |
| Test macro F1 | 0.5858 |
| Test agreement ratio | 91.99% |
| Test directional acc | 63.82% |
| **Confidence analysis** | |
| agree=1.0 (test) | 76.7% of samples, 62.20% acc, 89.03% dir acc |
| agree=1.0 + confirm>0.65 (test) | 23.3% of samples, **78.63% acc, 97.21% dir acc** |
| agree<1.0 (test) | 23.3% of samples, 47.70% acc (near random) |
| Lesson | ARCX high-conviction subset is smaller (23% vs 52%) but much stronger (78.6% vs 68.7%). Thinner book = sharper signal. |

### XGBoost Baseline XNAS H60

| Field | Value |
|---|---|
| Config | `configs/experiments/nvda_xgboost_baseline_h60.yaml` |
| Output | `outputs/xgboost_baseline_xnas_h60/` |
| Model | XGBoost (500 trees, depth=6) |
| Data | `nvda_xnas_128feat` |
| Features | 119 active (analysis_ready_128), flattened single snapshot |
| Labeling | TLOB, horizon index 3 (H60) |
| **Test accuracy** | **37.99%** |
| Test macro F1 | 0.3647 |
| Lesson | Single-snapshot XGBoost cannot capture temporal signal. HMHP adds +21.6pp via temporal context. Top features: depth_norm_ofi, trade_asymmetry, executed_pressure. |

### XGBoost Baseline ARCX H60

| Field | Value |
|---|---|
| Config | `configs/experiments/nvda_xgboost_baseline_arcx_h60.yaml` |
| Output | `outputs/xgboost_baseline_arcx_h60/` |
| Model | XGBoost (500 trees, depth=6) |
| Data | `nvda_arcx_128feat` |
| **Test accuracy** | **37.53%** |
| Test macro F1 | 0.3548 |
| Lesson | Cross-exchange XGBoost consistency (0.5pp delta). Confirms signal is NVDA-intrinsic. |

### HMHP 40-feat XNAS H10 (Feature Selection + Readability Test)

| Field | Value |
|---|---|
| Config | `configs/experiments/nvda_hmhp_40feat_h10.yaml` |
| Output | `outputs/experiments/nvda_hmhp_40feat_h10/` |
| Model | HMHP (TLOB encoder, full cascade, gate fusion, confirmation) |
| Data | `nvda_xnas_128feat` (233 days, XNAS ITCH) |
| Features | 40 (short_term_40 preset: OFI core + trade flow + L1-L2 LOB) |
| Labeling | TLOB (smoothed endpoint return) |
| Horizons | [10, 60, 300] |
| Epochs | 30 (ran to completion; best val_loss=0.8472 at epoch 24; process killed during post-training finalization) |
| Parameters | 165,766 |
| **Val H10** | **59.13%** (epoch 24) |
| **Val H10 (last)** | **59.22%** (epoch 29) |
| Val H60 | 40.97% |
| Val agreement | 93.75% |
| Train-val gap | 0.55pp (minimal overfitting) |
| **Readability analysis (test)** | |
| agree=1.0 (test) | 82.0% of samples, 62.18% acc, 89.26% dir acc |
| agree=1.0 + confirm>0.65 (test) | 38.1% of samples, **74.29% acc, 95.33% dir acc** |
| **FULL_READABILITY: high_conf+1tick+dir (test)** | **28.6% of samples, 74.08% acc, 95.50% dir acc** |
| agree<1.0 (test) | 18.0% of samples, 46.34% acc (random) |
| Lesson | **40-feat model has HIGHER readability-conditioned accuracy than 119-feat model** (95.50% vs 94.07% directional at FULL_READABILITY). Fewer features = less noise = sharper confidence signal. Preferred model for readability-first strategy. |
| Note | Process killed by OOM during post-training finalization. Best checkpoint saved. No test evaluation or training_history.json. Results from log + confidence_analysis.json. |

---

## Prior Experiments (116-feature or older)

These experiments used older datasets (98 or 116 features) and are retained for historical comparison.

### HMHP Multi-Horizon v1 (116-feat)

| Field | Value |
|---|---|
| Config | `configs/experiments/nvda_hmhp_multihorizon_v1.yaml` |
| Data | `nvda_116feat_full_analysis` (116 features, no MLOFI) |
| Horizons | [10, 20, 50, 100, 200] |
| Epochs | 16 (early stopped; best at epoch 15) |
| **Val accuracy** | **59.29%** (H10: 59.32%) |
| Val H20 | 47.67% |
| Val H50 | 42.15% |
| Val agreement | 91.34% |
| Lesson | First successful HMHP run. Benchmark for 128-feat comparison. |

### Short-Term HMHP v1 (40-feat, H10+H20)

| Field | Value |
|---|---|
| Config | `configs/experiments/nvda_short_term_hmhp_v1.yaml` |
| Data | `nvda_116feat_full_analysis`, short_term_40 preset |
| Features | 40 (evidence-based short-term subset) |
| Horizons | [10, 20] |
| Epochs | 19 (early stopped; best at epoch 17) |
| **Val accuracy** | **58.67%** (H10: 58.67%) |
| Val agreement | 98.61% |
| Lesson | 40-feature model nearly matches 116-feature model at H10 (-0.65pp). Very high agreement with only 2 horizons. |

### HMHP Triple Barrier v1 (98-feat)

| Field | Value |
|---|---|
| Config | `configs/experiments/nvda_hmhp_triple_barrier_v1.yaml` |
| Data | `nvda_11month_triple_barrier` |
| Horizons | [50, 100, 200] |
| Epochs | 1 (only 1 epoch completed) |
| Val accuracy | 98.56% (suspiciously high — likely majority-class collapse to Timeout) |
| Lesson | Triple Barrier labels with HMHP cascade produce degenerate predictions. Label semantics break the cascade assumption. |

### HMHP Triple Barrier Vol-Scaled (98-feat)

| Field | Value |
|---|---|
| Config | `configs/experiments/nvda_hmhp_triple_barrier_volscaled.yaml` |
| Data | `nvda_11month_triple_barrier_volscaled` |
| Epochs | 10 (early stopped) |
| Val accuracy | 50.33% (near random) |
| Lesson | Volatility-scaled barriers did not fix the cascade-label mismatch. Confirms Triple Barrier is architecturally misaligned with HMHP. |

### DeepLOB Opportunity v1 (98-feat)

| Field | Value |
|---|---|
| Config | stored in output `config.yaml` |
| Model | DeepLOB |
| Data | `nvda_bigmove` (Opportunity labels) |
| Features | 40 (input_size) |
| Epochs | 16 (early stopped; best at epoch 8) |
| Val accuracy | 58.47% |
| Lesson | DeepLOB on Opportunity labels performs reasonably. Opportunity labeling is viable for directional models. |

### DeepLOB H10 Weighted v1 (98-feat)

| Field | Value |
|---|---|
| Config | stored in output `config.yaml` |
| Model | DeepLOB |
| Data | `nvda_balanced` |
| Epochs | 10 |
| Val accuracy | 58.11% |
| Lesson | Class weighting on DeepLOB shows modest improvement over unweighted. |

### TLOB Single-Horizon Experiments

| Experiment | Features | Horizon | Best Val Acc | Lesson |
|---|---|---|---|---|
| nvda_tlob_repo_match_h50 | 40 | H50 | 53.17% | TLOB on 40-feature LOB data; modest performance |
| nvda_tlob_raw_h50_v1 | 98 | H50 | 41.11% | Raw features without normalization tuning |
| nvda_tlob_98feat_h100 | 98 | H100 | 39.56% | H100 is near signal floor |
| nvda_tlob_v2_h100 | 40 | H100 | 38.83% | H100 with fewer features |
| nvda_tlob_h50_v1 | 98 | H50 | 35.39% | Early experiment, poor normalization |
| nvda_tlob_bigmove_v1 | 98 | bigmove | 76.59% | Opportunity labels, high NoOpp class → inflated accuracy |

---

## Cross-Experiment Insights

### Model Architecture Ranking (H10 on XNAS)

| Model | H10 Val Acc | Features | Readability Dir Acc (test) | Preferred? |
|---|---|---|---|---|
| **HMHP 40-feat** | **59.13%** | 40 | **95.50%** | **YES (readability-first)** |
| HMHP 119-feat | 59.19% | 119 | 94.07% | Best raw accuracy |
| HMHP 116-feat | ~58.2% | 116 | N/A (not computed) | Superseded |
| XGBoost 119-feat | N/A | 119 | N/A | 38% at H60 (no temporal) |

**Key finding:** The 40-feature model achieves 95.50% directional accuracy at full readability gate (vs 94.07% for 119-feat). Fewer features = purer confidence signal = better readability detection.

### Labeling Strategy Findings

| Strategy | HMHP Result | Diagnosis |
|---|---|---|
| **TLOB** | 59.62% H10 | Works well with cascade. Current best. Correct objective for readability-first strategy. |
| Triple Barrier | 98.56% or 50.33% | Degenerate. Label semantics break cascade. |
| Opportunity | **Cancelled** | Wrong objective. Detects magnitude (big moves), not readability (reliable direction). |

### Readability-Conditioned Accuracy (Test Sets)

The agreement_ratio is a learned **market readability score**. When all horizons agree, the microstructure is in a coherent, readable state. When they disagree, the market is unreadable.

**119-feature model (XNAS + ARCX):**

| Market State | Filter | XNAS Acc | XNAS Dir Acc | ARCX Acc | ARCX Dir Acc |
|---|---|---|---|---|---|
| **Readable + confident** | agree=1.0 + confirm>0.65 | **68.72%** | **93.88%** | **78.63%** | **97.21%** |
| **Readable** | agree=1.0 | 63.08% | 89.52% | 62.20% | 89.03% |
| **Unreadable** | agree<1.0 | 48.38% | 80.59% | 47.70% | 82.32% |
| All samples | no filter | 59.63% | 88.55% | 58.82% | 88.40% |

**40-feature model vs 119-feature model (XNAS test, with spread conditioning):**

| Gate | 119-feat Rate | 119-feat Dir Acc | 40-feat Rate | 40-feat Dir Acc |
|---|---|---|---|---|
| All samples | 100% | 88.55% | 100% | 88.67% |
| agree=1.0 | 76.5% | 89.52% | 82.0% | 89.26% |
| high_conf | 51.5% | 93.88% | 38.1% | **95.33%** |
| high_conf + 1tick | 41.1% | 94.07% | 30.1% | **95.50%** |
| **FULL_READABILITY** | **33.9%** | **94.07%** | **28.6%** | **95.50%** |

The 40-feature model is more selective (28.6% vs 33.9% trade rate) but more accurate when it signals (95.50% vs 94.07% directional). For a readability-first strategy that trades only when sure, this is the optimal model.

---

## Strategic Framework: Readability-First Trading

**Objective:** Trade only when the microstructure is readable — when our signals reliably predict direction. Even a tiny move is sufficient because 0DTE ATM options provide leverage.

**Why not Opportunity (big-move detection)?**
- Opportunity labels ask "will the price move > threshold?" — this is magnitude prediction
- Magnitude prediction is harder (requires both direction AND size accuracy)
- A 10 bps move the model is 97% sure about is more valuable than a 50 bps move at 60% confidence
- OFI is contemporaneous (confirms current state), not a leading indicator of big moves
- The model already achieves 97.2% directional accuracy on readable states — this IS the signal

**Why TLOB labeling is correct:**
- TLOB predicts *direction* (Up/Stable/Down) — exactly what we need
- The HMHP cascade naturally produces a readability signal (agreement_ratio)
- Confidence filtering turns the directional model into a readability detector
- No architecture changes needed; the existing model already solves the right problem

**Trading rules derived from this framework:**
1. Pre-filter: spread must be 1 tick (OFI r=0.546 at 1-tick vs 0.365 at 2-tick)
2. Readability gate: agreement_ratio must be 1.0 (all horizons agree)
3. Confidence gate: confirmation_score must be > 0.65
4. Trade direction: model's predicted class (Up -> buy call, Down -> buy put)
5. Never trade when agreement < 1.0 (market is unreadable, accuracy drops to ~48%)

---

## Hybrid Backtest Experiments

### Readability Hybrid: HMHP Classification + Ridge Regression (2026-03-16)

| Field | Value |
|---|---|
| Strategy | `ReadabilityHybridStrategy` (dual gate: readability + magnitude) |
| Classification model | HMHP 40-feat (95.50% directional accuracy at full readability gate) |
| Regression model | TemporalRidge (IC=0.616, DA=72.2%, 54 params) |
| Signal source | `outputs/experiments/hybrid_readability_ridge_h10/signals/test/` |
| Backtest output | `lob-backtester/outputs/backtests/hybrid_readability_ridge_h10.json` |
| Samples | 50,724 (identical prices, verified) |
| Configurations swept | 48 (2 agreement x 3 confidence x 4 min_return x 2 hold_events) |
| **Best result** | **-2.67%** (agree>=1.0, conf>0.65, \|ret\|>=5bps, hold=60, 701 trades, 42.8% win rate) |

**Gate distribution (50,724 samples):**

| Gate | Pass Count | Pass Rate |
|---|---|---|
| agreement >= 1.0 | 41,619 | 82.0% |
| confirmation > 0.65 | 19,340 | 38.1% |
| \|predicted_return\| >= 3 bps | 20,347 | 40.1% |
| \|predicted_return\| >= 5 bps | 8,751 | 17.3% |
| directional (Up or Down) | 36,404 | 71.8% |
| spread <= 1.05 bps | 38,988 | 76.9% |
| **full readability gate** | **14,497** | **28.6%** |
| **readability + \|ret\|>=3 bps** | **10,204** | **20.1%** |
| **readability + \|ret\|>=5 bps** | **5,094** | **10.0%** |
| **readability + \|ret\|>=8 bps** | **1,116** | **2.2%** |

**Top 5 non-zero configurations (by option return):**

| Config | Trades | Option Return | Win Rate |
|---|---|---|---|
| agree=1.0, conf>0.65, \|ret\|>=5bps, h=60 | 701 | **-2.67%** | 42.8% |
| agree=1.0, conf>0.65, \|ret\|>=1bps, h=60 | 786 | -2.97% | 42.9% |
| agree=1.0, conf>0.65, \|ret\|>=8bps, h=10 | 804 | -3.03% | 41.9% |
| agree=1.0, conf>0.50, \|ret\|>=5bps, h=60 | 714 | -3.11% | 42.7% |
| agree=1.0, conf>0.50, \|ret\|>=8bps, h=10 | 842 | -3.41% | 41.1% |

**Key findings:**

1. **Hybrid does NOT beat pure regression**: Best hybrid -2.67% vs pure Ridge at 10bps -1.14% (333 trades). The magnitude gate alone (Ridge) outperforms the dual gate.
2. **agreement=0.9 and agreement=1.0 produce identical results**: All near-agreement samples already have agreement=1.0 -- the HMHP agreement distribution is bimodal (either 1.0 or significantly lower).
3. **confidence > 0.80 produces ZERO trades**: The HMHP 40-feat model's confirmation scores never exceed 0.80 in the test set (max ~0.667).
4. **Fundamental issue unchanged**: Both models predict smoothed-average returns, but execution is point-to-point. Adding a classification gate to a regression signal does not fix the label-execution mismatch.
5. **Readability gate is not additive with regression**: The readability gate filters for confident DIRECTION, but the regression already captures direction (DA=72.2%). The intersection does not improve over the regression's magnitude filter alone.

---

### E14: Off-Exchange Signal Gate Check — Lesson 33 Filter (2026-03-30)

| Field | Value |
|---|---|
| **Hypothesis** | Off-exchange features with high temporal persistence (ACF(60) > 0.30) may avoid E13's failure mode (cross-sectional IC collapsing at trading cadence). subpenny_intensity (ACF(1)=0.889) is the primary candidate. |
| **Method** | Three-gate filter from E13 lessons: G1 (ACF(60) > 0.30), G2 (pooled stride-60 IC > 0.05), G3 (pooled lag-1 IC > 0.03). 6 features × 3 horizons × 2 splits. Bootstrap 95% CI on all stride-60 ICs. |
| **Data** | `data/exports/basic_nvda_60s/` — 32 val days (9,856 samples), 35 test days (10,780 samples), 34 features, 8 horizons. |
| **Script** | `hft-feature-evaluator/scripts/offexchange_gate_check.py` |
| **Status** | **ALL FAIL — No tradeable off-exchange signal. Bootstrap CIs on ALL stride-60 ICs cross zero.** |

**Gate Results:**

| Feature | Horizon | ACF(60) | G1 | IC(s=1) val/test | IC(s=60) val/test | G2 | Lag-1 IC val/test | G3 |
|---|---|---|---|---|---|---|---|---|
| **subpenny_intensity** | H=60 | 0.28/0.41 | val:fail/test:pass | -0.064/+0.037 | -0.068/+0.073 | val:fail/test:pass | -0.070/+0.078 | val:fail/test:pass |
| **bbo_update_rate** | H=60 | 0.47/0.60 | PASS | +0.042/+0.132 | +0.072/+0.077 | PASS | -0.011/+0.055 | val:fail/test:pass |
| trf_signed_imbalance | H=1 | -0.021/-0.007 | FAIL | — | — | — | — | — |
| dark_share | H=1 | 0.010/-0.018 | FAIL | — | — | — | — | — |
| spread_bps (off-exch) | H=60 | 0.098/0.173 | FAIL | — | — | — | — | — |
| quote_imbalance | H=1 | -0.003/0.013 | FAIL | — | — | — | — | — |

**Critical: Bootstrap 95% CIs on stride-60 IC (pooled, ~160-175 samples):**

| Feature | Split | IC(s=60) | 95% CI | Contains Zero? |
|---|---|---|---|---|
| subpenny_intensity | val | -0.068 | [-0.234, +0.088] | **YES** |
| subpenny_intensity | test | +0.073 | [-0.072, +0.222] | **YES** |
| bbo_update_rate | val | +0.072 | [-0.075, +0.217] | **YES** |
| bbo_update_rate | test | +0.077 | [-0.081, +0.241] | **YES** |

**ALL four CIs include zero. No feature has statistically significant temporal IC at the trading cadence.**

**subpenny_intensity SIGN FLIP:** Val per-day IC mean = **-0.064** (41% positive days), test = **+0.037** (60% positive days). The signal direction reverses between October-November (val) and November-January (test). This is a regime change, not a stable signal.

**Per-day IC is noise-dominated:** subpenny_intensity per-day IC std = 0.38 vs mean ±0.06 (SNR = 0.16). bbo_update_rate std = 0.32-0.35 vs mean 0.04-0.13 (SNR = 0.12-0.38). The signal-to-noise ratio at the feature level is effectively zero.

**Lesson (36, NEW):** Bootstrap CI on pooled stride-60 IC is the DEFINITIVE tradeability test. If the 95% CI crosses zero, the feature has no statistically significant temporal prediction at the trading cadence. All four tested CIs crossed zero — no off-exchange feature is tradeable at H=60.

**Lesson (37, NEW):** Test-only gate passes are noise, not signal. subpenny_intensity passed all 3 gates on test but FAILED on val (with sign flip). This confirms E13 lesson 35: split-specific passes without cross-split confirmation are unreliable. Bootstrap CIs confirmed: both splits' stride-60 ICs are indistinguishable from zero.

**Lesson (38, NEW):** Per-day IC std 8-10x larger than mean (SNR ≈ 0.1-0.2) means the feature contributes essentially random per-day predictions. No model or strategy can extract signal from SNR < 0.5 with 30-35 day evaluation windows.

---

## Config Archive Status

Configs not associated with any output directory (never run or results deleted):

- `nvda_logistic_baseline_h10.yaml` — not run
- `nvda_logistic_baseline_v1.yaml` — not run
- `nvda_logistic_signals_h10.yaml` — not run
- `nvda_logistic_signals_v1.yaml` — not run
- `nvda_tlob_h100_v1.yaml` — not run (separate from nvda_tlob_98feat_h100)
- `nvda_tlob_triple_barrier_11mo_v1.yaml` — not run
- `nvda_hmhp_128feat_opportunity_h10.yaml` — cancelled (wrong objective: magnitude detection instead of readability detection)
- `nvda_hmhp_triple_barrier_calibrated` — output exists but no training_history.json (incomplete run)
- `nvda_hmhp_40feat_h60_profit8bps` — output exists (completed), not previously indexed
- `nvda_hmhp_40feat_h60_profit8bps_regression.yaml` — not run (regression variant of profit8bps)
- `nvda_hmhp_regressor_h60.yaml` — not run (HMHP-R H60-primary, superseded by H10-primary)

---

### E15: Long-Horizon Intraday — Morning Signal to Afternoon Return (2026-04-04)

| Field | Value |
|---|---|
| **Hypothesis** | Morning microstructure features (spread, OFI) predict afternoon returns across 233 trading days. Between-day temporal signal with 1 trade/day. Fundamentally different from E1-E14 (within-day cross-sectional). |
| **Method** | 3-phase analysis: (1) daily aggregation (13 morning features × 3 windows × 4 horizons), (2) baseline-first IC analysis with confound control (partial IC after controlling yesterday's return + day-of-week), (3) walk-forward Ridge with 3 models (A=baseline, B=morning-only, C=combined). |
| **Data** | `data/exports/e5_timebased_60s/` — 233 days, 56,660 sequences, 98 MBO features, forward_prices (N,306). Morning windows: 30/45/60 min. Afternoon horizons: 120/180/240/300 bins (2-5 hours). |
| **Script** | `hft-feature-evaluator/scripts/archive/e15_morning_signal_analysis.py` — **archived Phase 6 6D (2026-04-17)**, runnable at new path |
| **Output** | `hft-feature-evaluator/outputs/e15_morning_signal/` |
| **Status** | **ALL FAIL — In-sample artifact, negative out-of-sample. Independently validated.** |

**Headline: Afternoon return ACF(1) = -0.27 is REAL but NOT TRADEABLE (same pattern as E1-E14)**

| Horizon | Return Std (bps) | ACF(1) | In-Sample IC | In-Sample P&L (bps) |
|---------|-----------------|--------|--------------|---------------------|
| H=120 (2hr) | 108 | -0.11 (p=0.11) | 0.115 | +931 |
| H=180 (3hr) | 142 | -0.20 (p=0.003) | 0.144 | +3,612 |
| H=240 (4hr) | 152 | -0.23 (p=0.0004) | 0.193 | +4,246 |
| H=300 (5hr) | 180 | -0.27 (p<0.0001) | 0.165 | +5,345 |

**CRITICAL: Independent validation revealed the +5,345 bps is entirely in-sample:**

| Split | Days | ACF(1) | P&L (bps) | Status |
|-------|------|--------|-----------|--------|
| Train | 163 | **-0.34** | **+6,011** | Profitable (in-sample) |
| Val | 35 | -0.09 | **-233** | NEGATIVE |
| Test | 35 | -0.05 | **-909** | NEGATIVE |
| Val+Test | 70 | ~-0.07 | **-1,142** | NEGATIVE |

**Root causes of in-sample mirage:**
1. **Single outlier dominates**: April 8-9 2025 (tariff chaos/rebound) contributes 43% of ACF. Without it: ACF drops from -0.264 to -0.131.
2. **Tail concentration**: Without top-20 profitable days: P&L = -1,998 bps. Strategy is net-negative excluding extreme events.
3. **Temporal instability**: First half ACF=-0.34, second half=-0.14. Rolling 60-day ACF swings from -0.47 to +0.11.
4. **Winsorized ACF is weak**: At 5%/95% clip: ACF=-0.149 (p=0.024). Marginal.

**Morning Features: ZERO temporal predictive power**

| Feature | Partial IC (full-sample) | Walk-Forward IC | Verdict |
|---------|--------------------------|-----------------|---------|
| depth_norm_ofi_sum (w=45 H=300) | -0.210 | N/A (Model B IC=-0.009) | Full-sample partial IC does NOT survive walk-forward |
| true_ofi_sum (w=45 H=300) | -0.198 | N/A (same Model B) | Same failure |

Model B (morning-only): IC=-0.009, DA=48.5%. ZERO standalone temporal power (consistent with E1-E14).
Model C (combined): IC=0.134, but feature selection had LOOK-AHEAD BIAS (features selected from full-sample Phase 2C). Model C result is untrustworthy.

**Lessons:**

- **39**: Daily afternoon return mean-reversion (ACF=-0.27) is statistically real in the full 233-day sample. BUT it is temporally unstable (halves from first to second half), driven by fat tails (kurtosis=29) and a single extreme event (April 8-9 tariff). **The ACF is a sample property, not a tradeable signal.**
- **40**: Return std at daily horizon is 108-180 bps (NOT 44 bps intra-day estimate). Breakeven IC=0.006-0.009. But low breakeven doesn't help if the signal is unstable.
- **41**: Morning microstructure features have ZERO standalone walk-forward IC (Model B IC=-0.009). Full-sample partial IC of 0.21 does NOT guarantee OOS temporal prediction. **This confirms E1-E14: microstructure features are contemporaneous, not predictive, even at the daily horizon.**
- **42**: In-sample P&L can be massively misleading with fat-tailed returns (kurtosis=29). Top-20 days (+7,343 bps) overwhelm remaining 212 days (-1,998 bps). ALWAYS validate on held-out data.
- **43**: Per-split validation is MANDATORY for any signal with daily observations. The val+test ACF (~-0.07) is an order of magnitude weaker than the train ACF (-0.34). This should have been the FIRST check, not a post-hoc validation.
- **44**: Feature selection from full-sample results into walk-forward is look-ahead bias. Must be inside the loop.

---

### Universality Study: Multi-Stock IC Gate (2026-04-05)

| Field | Value |
|---|---|
| **Hypothesis** | Is the zero IC for point returns at 60s cadence specific to NVDA, or universal across NASDAQ stocks with diverse microstructure characteristics? |
| **Method** | (1) Selected 10 stocks spanning full NASDAQ spectrum: CRSP ($49, 1.6M vol, 12 bps spread) to HOOD ($45, 25M vol, 4 bps spread). (2) Exported all 10 at 60s bins using identical E5 config (98 features, H=[10,60], k=5). (3) Computed stride-60 Spearman IC between 14 non-price features and point returns (from forward_prices) on OOS (val+test). (4) Bootstrap 95% CIs (2000 resamples). (5) Cross-stock consistency check. |
| **Data** | `data/exports/universality_{symbol}_60s/` — 10 stocks × 134 days each. Raw MBO from `XNAS_ITCH/{SYMBOL}/mbo_2025-07-01_to_2026-01-09/`. |
| **Configs** | `feature-extractor-MBO-LOB/configs/universality_{symbol}_60s.toml` (10 configs) |
| **Script** | `hft-feature-evaluator/scripts/universality_ic_gate.py` |
| **Output** | `hft-feature-evaluator/outputs/universality_{symbol}_ic/ic_gate_results.json` (10 files), `hft-feature-evaluator/outputs/universality_consolidated_results.json` |
| **Report** | `hft-feature-evaluator/reports/UNIVERSALITY_STUDY_2026_04.md` |
| **Status** | **H0 CONFIRMED — Zero features have predictive IC across any stock.** |

**Results:**

| Stock | Price | Vol/Day | Spread (bps) | H10 Pass | H60 Pass |
|---|---|---|---|---|---|
| CRSP | $49 | 1.6M | 11.8 | 0/14 | 0/14 |
| PEP | $155 | 5M | 1.5 | 0/14 | 0/14 |
| IBKR | $190 | 1.5M | 5.1 | 0/14 | 0/14 |
| ZM | $72 | 3M | 2.7 | 1/14 | 0/14 |
| ISRG | $525 | 1.5M | 6.3 | 1/14 | 2/14 |
| FANG | $170 | 2M | 8.5 | 0/14 | 0/14 |
| DKNG | $42 | 10M | 2.5 | 0/14 | 0/14 |
| MRNA | $40 | 12M | 5.9 | 3/14 | 2/14 |
| HOOD | $45 | 25M | 4.0 | 2/14 | 0/14 |
| SNAP | $12 | 20M | 10.8 | 0/14 | 3/14 |
| **Total** | | | | **7/140** | **7/140** |

**Combined: 14/280 = 5.0% — exactly the expected false positive rate at 95% CI.**

- 0 features pass for 2+ stocks (zero cross-stock consistency)
- All stability ratios < 2.0 (best: 0.870, threshold: 2.0)
- Sign reversals confirm noise: MRNA depth_norm_ofi IC=-0.170 vs SNAP IC=+0.138

**Feature Extractor Investigation (same session):**
Comprehensive code audit verified: (1) features are raw (not normalized), (2) forward_prices alignment correct (fp_base matches mid_price feature to f64 precision), (3) sequence-label alignment correct, (4) OFI accumulation semantics correct.

**Contemporaneous vs Predictive IC (proven across PEP, HOOD, MRNA):**

| Target | IC with true_ofi | Interpretation |
|---|---|---|
| Concurrent return (t-1 → t) | **+0.73 to +0.86** | OFI describes what IS happening |
| Smoothing residual (smoothed - point) | **+0.30 to +0.39** | OFI predicts the label artifact |
| Point return (t → t+1, 1 min) | **-0.01 to -0.03** | Zero predictive power |
| Point return (t → t+10, 10 min) | **-0.001 to -0.02** | Zero predictive power |

**Lessons:**

- **45**: Multi-stock universality (10 stocks, 134 days each, spread range 1.5-12 bps, volume range 1.5M-25M) confirms E8's NVDA finding is NOT stock-specific. MBO features have zero predictive IC for point returns at 60s cadence universally.
- **46**: The smoothed return label's past window `mean(mid[t-5..t+1])` mechanically leaks concurrent OFI into the training target. OFI has IC=0.30-0.39 with the smoothing residual, and IC=0.00 with the point-return component. This is the mathematical root cause of all smoothed-label R² that fails to translate to backtest P&L. 97-99% of OFI's information content is consumed within the same 60s bin.

---

### E16: Extreme Event Conditional Return Analysis (2026-04-05)

| Field | Value |
|---|---|
| **Hypothesis** | When MBO features are at extreme percentiles (top/bottom 2-10%), the expected forward point return is non-zero and exceeds equity trading costs (0.7 bps). Tests the "rare event signal" escape hatch that aggregate IC would average away. |
| **Method** | (1) Compute feature percentile thresholds from TRAIN split (causal — no look-ahead). (2) Apply to OOS (val and test SEPARATELY). (3) Measure conditional mean return at H=1,3,5,10,20,60 with per-day block bootstrap CIs (2000 resamples). (4) BH FDR correction at α=0.10 across all tests. (5) Cross-stock sign consistency. (6) Val→test stability check. |
| **Data** | 10 universality exports (134 days each, 60s bins, 98 features, point returns from forward_prices). 5 features × 3 percentiles × 2 tails × 6 horizons × 10 stocks × 2 splits = 3,600 tests. |
| **Features** | true_ofi (84), depth_norm_ofi (85), spread_bps (42), volume_imbalance (45), fragility_score (90) |
| **Script** | `hft-feature-evaluator/scripts/archive/e16_extreme_event_study.py` — **archived Phase 6 6D (2026-04-17)**, runnable at new path |
| **Output** | `hft-feature-evaluator/outputs/e16_extreme_events/e16_results.json` |
| **Status** | **MARGINAL — 15 survive FDR but sign-inconsistent and val→test unstable. Not tradeable.** |

**Results (test split only):**

| Metric | Value |
|---|---|
| Total test-split tests | 1,656 |
| CI excludes zero (before FDR) | 159 (9.6% vs 5% expected) |
| Surviving BH FDR (α=0.10) | 15 |
| Cross-stock consistent (3+ stocks, same sign) | 3 conditions |
| Val→test sign flips (of conditions significant in both) | 50% |

**15 FDR-surviving results show:**
- Sign inconsistency: spread_bps bottom → PEP negative, FANG positive (opposite)
- Small magnitudes: most 0.5-2 bps (barely above 0.7 bps equity cost)
- CRSP's large H=60 OFI effect (-26 bps) shows val→test sign flip

**3 cross-stock consistent conditions (before FDR):**
1. spread_bps P10 top H=1: -1.0 bps avg (3 stocks) — barely above cost
2. true_ofi P2 top H=5: +7.5 bps avg (3 stocks: CRSP, DKNG, HOOD) — but CRSP val→test flips
3. volume_imbalance P5 bottom H=1: -0.6 bps avg (3 stocks) — below cost

**Val→test stability: 8 same-sign, 8 sign-flips** of 16 conditions significant in both splits. The 50% sign-flip rate is indistinguishable from random.

**Lessons:**

- **47**: Tail-conditional returns show a marginal departure from null (9.6% hit rate vs 5% expected), suggesting weak tail effects exist. But after FDR correction, the surviving 15 results are sign-inconsistent across stocks and 50% val→test unstable — not tradeable.
- **48**: All statistical relationship types between MBO features and point returns at 60s cadence are now tested and closed: linear IC (E2/E3/E8/universality), MI/dCor (E13 Path 2 = 0), transfer entropy (E13 Path 3b = 0), regime-conditional (E13 Path 4 = cross-sectional only), extreme events (E16 = marginal/unstable). No escape hatch remains.
- **49**: Reducing sampling cadence below 60s will not help. The MBO profiler found OFI lag-1 predictive r < 0.006 at ALL scales from 1 second to 5 minutes (233 NVDA days). Price impact incorporation occurs in milliseconds — no retail-accessible timescale can capture it.
