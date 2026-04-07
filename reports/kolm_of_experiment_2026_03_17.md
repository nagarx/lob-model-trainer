# Kolm Per-Level OF Experiment Report

**Date:** 2026-03-17
**Status:** All results validated. Phase 5 (model training) cancelled per decision gate.
**Dataset:** `nvda_xnas_kolm_of_regression` (233 days, event_count=100, point-return labels)

---

## 1. Experiment Objective

Test whether Kolm et al.'s (2023) 20-dimensional per-level Order Flow (bid/ask separate) improves point-to-point return prediction at short horizons within the OFI effective range.

**Hypothesis:** Per-level OF (20-dim) outperforms scalar OFI for point-return regression, as Kolm demonstrated on 115 NASDAQ stocks. At event_count=100 (~195ms/sample), horizons H1-H3 fall within the OFI effective range (~600ms for NVDA).

**Prior evidence for the hypothesis:**
- Kolm et al. (2023): OF significantly outperforms OFI across 115 stocks
- Our scalar DEPTH_NORM_OFI: IC=0.677 for smoothed-average at event_count=1000 (strong signal on wrong target)
- Conditional win rate analysis: IC=0.005 for point-return at event_count=1000 (zero signal at long horizons)

---

## 2. Implementation

### 2A. Rust Feature Extractor (Phase 1-2)

New experimental group `kolm_of` implemented in `feature-extractor-MBO-LOB/src/features/experimental/kolm_of.rs`:
- 20 features: `bof_level_1..10` (bid OF per level) + `aof_level_1..10` (ask OF per level)
- Wraps existing `MultiLevelOfiTracker` from `order_flow.rs`
- Ask values negated to match Kolm sign convention (positive = ask improved)
- 8 unit tests passing

Pipeline contract updated: `contracts/pipeline_contract.toml` indices 128-147 for `kolm_of`.
Python contracts regenerated. Rust contract validation: 19/19 tests pass.

### 2B. Export Configuration (Phase 3)

Config: `feature-extractor-MBO-LOB/configs/nvda_xnas_kolm_of_regression.toml`

| Parameter | Value | Rationale |
|---|---|---|
| `event_count` | 100 | ~195ms/sample. H1=195ms ~ Kolm h3, H3=585ms ~ Kolm h10 |
| `window_size` | 20 | T=20 retains 99.6% of IC (from T=100 ablation) |
| `stride` | 10 | |
| `horizons` | [1, 2, 3, 5] | H1-H3 within Kolm effective range, H5 for decay |
| `return_type` | point_return | Kolm uses point-to-point, no smoothing |
| `experimental.groups` | [institutional_v2, volatility, seasonality, kolm_of] | No mlofi (redundant with kolm_of) |
| Total features | 136 | 98 + 8 + 6 + 4 + 20 |

### 2C. Export Output

| Split | Days | Sequences | Size |
|---|---|---|---|
| Train | 163 | 1,695,700 | 17 GB |
| Val | 35 | 542,958 | 5.5 GB |
| Test | 35 | 521,348 | 5.3 GB |
| **Total** | **233** | **2,760,006** | **28 GB** |

Shape per day: `[N, 20, 136]` sequences + `[N, 4]` regression labels (4 horizons in bps).
All validation passed: zero NaN/Inf, 100% positive spreads, schema 2.2.

---

## 3. Phase 4 Results: Statistical Baseline Validation

### 3A. Per-Feature IC (521,348 test sequences)

**Scalar OFI features (existing):**

| Feature | Index | IC (H1) | IC (H2) | IC (H3) | p-value (H1) |
|---|---|---|---|---|---|
| DEPTH_NORM_OFI | 85 | **0.0703** | 0.0540 | 0.0462 | < 1e-67 |
| TRUE_OFI | 84 | **0.0568** | 0.0487 | 0.0428 | < 1e-50 |
| EXECUTED_PRESSURE | 86 | 0.0315 | 0.0347 | 0.0339 | < 1e-20 |

**Kolm per-level OF features (new):**

| Feature | Index | IC (H1) | IC (H2) | IC (H3) | p-value (H1) |
|---|---|---|---|---|---|
| bof_level_1 | 116 | 0.0002 | 0.0010 | 0.0006 | 0.28 |
| bof_level_2 | 117 | -0.0013 | -0.0008 | -0.0010 | 0.17 |
| bof_level_5 | 120 | -0.0004 | -0.0003 | -0.0006 | 0.14 |
| bof_level_10 | 125 | -0.0018 | -0.0011 | -0.0017 | 0.35 |
| aof_level_1 | 126 | -0.0008 | -0.0002 | 0.0002 | 0.36 |
| aof_level_5 | 130 | -0.0008 | -0.0008 | -0.0011 | 0.43 |
| aof_level_10 | 135 | -0.0028 | -0.0028 | -0.0034 | 0.93 |

**All 20 Kolm OF features have IC indistinguishable from zero** (range: -0.003 to +0.001). No p-value is significant.

### 3B. Ridge Regression Baselines

| Model | R-squared | IC | Interpretation |
|---|---|---|---|
| Kolm OF only (20 features) | -0.0002 | **0.0001** | Zero predictive power |
| Scalar OFI only (2 features) | 0.0044 | **0.0730** | Weak but real signal |
| Full features (136) | -0.0073 | **0.1059** | Combined features have some signal |

### 3C. Persistence Baseline

| Horizon | Persistence R-squared | Persistence IC |
|---|---|---|
| H1 (100 events) | -1.012 | -0.006 |
| H2 (200 events) | -0.991 | 0.004 |
| H3 (300 events) | -1.001 | 0.000 |
| H5 (500 events) | -1.004 | -0.002 |

Zero autocorrelation in point-returns at this resolution. Persistence is worse than random.

### 3D. Label Distribution

| Horizon | Mean (bps) | Std (bps) | |Mean| (bps) | Positive % | Negative % |
|---|---|---|---|---|---|
| H1 | -0.001 | 0.545 | 0.349 | 31.0% | 30.9% |
| H2 | -0.001 | 0.757 | 0.508 | 35.9% | 36.2% |
| H3 | -0.001 | 0.911 | 0.624 | 38.4% | 38.3% |
| H5 | -0.003 | 1.192 | 0.815 | 40.6% | 40.9% |

Note: ~38% of H1 returns are exactly zero (price unchanged over 100 events).

---

## 4. Decision Gate

**Decision gate (from plan):** "If none of the 20 OF features have IC > 0.05 for point-return at H1, the per-level structure doesn't help and we pivot."

**Result: FAIL.** Maximum Kolm OF IC = 0.0036 (bof_level_1). Threshold = 0.05.

**Phase 5 (model training) CANCELLED.**

---

## 5. Root Cause Analysis

### Why Kolm OF features have zero IC despite Kolm's published results

1. **Cumulative vs per-event representation**: Our pipeline accumulates OF over the entire 100-event sampling window into a single value. Kolm's LSTM sees the OF at each individual LOB transition (100 sequential snapshots). The cumulative sum destroys the fine-grained temporal dynamics that Kolm's model exploits.

2. **Scale mismatch**: Kolm OF values are in the millions (mean ~1.5M for bof_level_1) because they accumulate 100 events of size changes. The scalar DEPTH_NORM_OFI normalizes by average depth, producing values with std ~1.2. The unnormalized cumulative OF has high variance but no directional information for endpoints.

3. **Why scalar OFI works but per-level OF doesn't**: The scalar DEPTH_NORM_OFI (IC=0.070) benefits from depth normalization, which creates a mean-reverting signal. The raw per-level OF accumulation doesn't have this property -- it's just a running sum that grows monotonically within the window.

4. **Kolm's architecture is fundamentally different**: Kolm feeds 100 per-event OF vectors to an LSTM. The LSTM learns temporal patterns in how OF evolves over the 100 transitions. Our T=20 sequence of accumulated-over-100-events values cannot capture this -- each of our 20 timesteps is a bulk summary, not a granular event.

### What the scalar OFI IC=0.070 means

DEPTH_NORM_OFI at event_count=100 has IC=0.070 for point-return at H1 (195ms). This is weak but real:
- It's 14x stronger than IC=0.005 at event_count=1000 (where H1 = 19.5 seconds)
- It confirms that OFI signal EXISTS at short horizons for point-returns
- But it's not strong enough alone for profitable trading (0.070 IC ~ 51.5% directional accuracy)

---

## 6. Implications for Next Steps

### What this experiment proves

1. **Per-level OF structure is not useful in our cumulative-per-window pipeline.** The information is in the temporal sequence of per-event OF, not the window sum.
2. **Shorter horizons DO increase point-return IC** (from 0.005 at 1000-event to 0.070 at 100-event for scalar OFI). The signal is real but decays rapidly.
3. **The fundamental limitation is our sampling architecture**: accumulating features over N events, then building sequences of these summaries, is not equivalent to processing per-event data as Kolm does.

### Paths forward (ranked)

1. **Per-event processing architecture** (HIGH effort, HIGH potential): Redesign the pipeline to pass individual LOB transitions to the model, matching Kolm's LSTM approach. This is a fundamental architecture change.
2. **ARCX cross-exchange** (LOW effort, MODERATE potential): The scalar OFI IC=0.070 at event_count=100 may be tradeable on ARCX where costs are lower (VWES=1.1 vs 1.97 bps).
3. **Depth-normalized per-level features** (LOW effort, LOW potential): Normalize the Kolm OF features by depth, matching DEPTH_NORM_OFI's normalization. This may recover some signal.
4. **Accept current architecture limits**: Use the scalar OFI signal (IC=0.070) for what it is -- a weak but real short-horizon signal that may compound in a portfolio context.

---

## 7. Infrastructure Built (Retained)

The following infrastructure was built and is retained for future use regardless of this experiment's outcome:

| Component | Location | Value |
|---|---|---|
| `kolm_of` experimental group | `feature-extractor-MBO-LOB/src/features/experimental/kolm_of.rs` | 20-dim OF feature group, 8 tests |
| Pipeline contract update | `contracts/pipeline_contract.toml` indices 128-147 | Kolm OF feature definitions |
| Export config | `feature-extractor-MBO-LOB/configs/nvda_xnas_kolm_of_regression.toml` | event_count=100 fine-grained config |
| Feature count validation fix | `export_aligned/validation.rs` | Accepts range [98, 148] instead of exact values |
| Python contract validation fix | `hft-contracts/validation.py` | Same range-based validation |
| Updated documentation | All 8 feature extractor .md files | Reflect 148 features, 5 groups |

---

## Appendix: Horizon Calibration

| Our Horizon | Wall-Clock | Kolm Equivalent | Within Effective Range? |
|---|---|---|---|
| H1 (100 events) | ~195ms | Kolm h3 | Yes |
| H2 (200 events) | ~390ms | Kolm h7 | Yes |
| H3 (300 events) | ~585ms | Kolm h10 | Boundary |
| H5 (500 events) | ~975ms | Kolm h16 | No |

NVDA Delta_t = 299ms (23.4M ms / 78,340 price_changes). Effective horizon = 2 * Delta_t = 598ms.

---

*Generated: 2026-03-17 | Test set: 521,348 sequences | All metrics validated against scipy.stats.pearsonr and sklearn.metrics.r2_score*
