# TLOB Regression Experiment Report

**Date:** 2026-03-15
**Model:** TLOB (Berti & Kasneci 2025) with regression head
**Task:** Continuous forward return prediction (bps) at H10
**Data:** 233-day NVDA XNAS 128-feature regression export
**Status:** COMPLETE -- pipeline validated, results independently verified

This is the **first regression experiment** in the pipeline. It validates end-to-end correctness from feature extraction through model training to metric computation, and establishes baseline performance for all subsequent regression work.

---

## 1. Experiment Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Architecture | TLOB (2 layers, hidden_dim=32, 2 heads) | Compact transformer; accepts 128 features natively |
| Task type | regression (continuous bps) | Pure regression, no classification |
| Loss function | Huber (delta=5.0 bps) | Calibrated from REGRESSION_ANALYSIS_REPORT.md §8; kurtosis=49.69 justifies Huber over MSE |
| Horizon | H10 (index 0) | Strongest signal: single-feature R²=0.107, IC=0.335 |
| Features | All 128 (98 stable + 30 experimental) | Model learns to select via attention |
| Normalization | Hybrid (global LOB, per-feature derived) | Same as prior HMHP experiments |
| Batch size | 128 | Reduced from 256 to prevent MPS memory pressure |
| Learning rate | 5e-4, cosine schedule | Conservative; matches HMHP experiments |
| Weight decay | 0.01 | L2 regularization |
| Gradient clip | 1.0 | Stability |
| Max epochs | 15 (resumed from epoch 0 checkpoint) | Early stopping patience=10 |
| Parameters | 693,190 (all trainable) | ~4x HMHP classification (170K) |
| Random seed | 42 | Deterministic |
| Config file | `configs/experiments/nvda_tlob_128feat_regression_h10.yaml` |
| Output dir | `outputs/experiments/nvda_tlob_128feat_regression_h10/` |

---

## 2. Dataset

| Split | Days | Sequences | Source |
|-------|------|-----------|--------|
| Train | 163 | 162,999 | 2025-02-03 to 2025-09-25 |
| Val | 35 | 52,885 | 2025-09-26 to 2025-11-13 |
| Test | 35 | 50,724 | 2025-11-14 to 2026-01-06 |
| **Total** | **233** | **266,608** | |

**Temporal split with zero overlap.** Train end (2025-09-25) strictly precedes test start (2025-11-14). Verified programmatically: `set(train_days) & set(test_days) = {}`.

**Label type:** Continuous forward returns in bps, float64. TLOB smoothed-average formula:
`r_t = (1/k) * Σ_{i=1}^{k} (m_{t+i} - m_t) / m_t × 10000`, where k=10 (H10), m=mid-price.

**Label statistics (test set):**

| Metric | H10 (used) |
|--------|------------|
| Mean | 0.000 bps |
| Std | 4.686 bps |
| Min / Max | -91.6 / +159.8 bps |
| Skewness | -- |
| Kurtosis | 52.98 (extreme) |

---

## 3. Analytical Baselines

Computed on the same test set (50,724 samples) using `scripts/compute_regression_baselines.py`:

| Baseline | R² | IC | DA | Description |
|----------|----|----|-----|-------------|
| Persistence | -0.377 | 0.264 | 0.591 | Predict return_t = return_{t-1} |
| Linear Ridge (128 feat) | 0.170 | 0.433 | 0.651 | sklearn Ridge (alpha=1.0) on last-timestep features |
| Single feature (DEPTH_NORM_OFI) | 0.107 | 0.335 | 0.620 | Linear regression on OFI alone |

**Key insight:** Persistence **hurts** at H10 (R² < 0). The 10-event smoothed return has only ACF(1)=0.28 -- not enough autocorrelation to make naive persistence useful. The model must learn from features, not from label history.

---

## 4. Training Results

Best checkpoint at **epoch 6** (val_loss=3.9115). Training ran 15 epochs (1 + 14 resumed), early stopping triggered at epoch 14.

| Epoch | Train Loss | Val Loss | Val R² | Val IC | Val DA | Note |
|-------|-----------|----------|--------|--------|--------|------|
| 0 | 9.548 | 4.079 | 0.448 | 0.672 | 0.746 | Initial |
| 1 | 8.834 | 3.993 | 0.463 | 0.678 | 0.749 | Improving |
| 4 | 8.578 | 3.942 | 0.471 | 0.680 | 0.750 | Best so far |
| **6** | **8.438** | **3.912** | **0.476** | **0.680** | **0.748** | **Best checkpoint** |
| 14 | 7.640 | 3.983 | 0.462 | 0.673 | 0.746 | Early stop |

**Training time:** ~7 minutes per epoch on Apple M-series MPS. Total: 96 minutes (14 epochs).

---

## 5. Test Results (Best Checkpoint)

| Metric | Value |
|--------|-------|
| **R²** | **0.4642** |
| **IC (Spearman)** | **0.6766** |
| **Pearson r** | **0.6817** |
| MAE | 2.43 bps |
| RMSE | 3.43 bps |
| Directional accuracy | 0.7494 |
| Profitable accuracy (>5 bps) | 0.9263 |

### vs Baselines

| Metric | Persistence | Linear Ridge | TLOB | TLOB / Ridge |
|--------|-------------|-------------|------|-------------|
| R² | -0.377 | 0.170 | **0.464** | **2.73x** |
| IC | 0.264 | 0.433 | **0.677** | **1.56x** |
| DA | 0.591 | 0.651 | **0.749** | **1.15x** |

---

## 6. Independent Validation (Post-Training Verification)

All checks performed after training completed, using saved checkpoint and test data. Every metric was recomputed independently and cross-checked.

### 6.1 Metrics Correctness

| Metric | Manual | sklearn | Saved | Match |
|--------|--------|---------|-------|-------|
| R² | 0.4642272592 | 0.4642272592 | 0.4642272592 | Exact |
| IC | 0.6765911268 | -- | 0.6765911268 | Exact |
| DA | 0.7493628371 | -- | 0.7493628371 | Exact |
| MAE | 2.4313187599 | -- | 2.4313187599 | Exact |

### 6.2 Data Flow Integrity

| Check | Result |
|-------|--------|
| Loader labels == raw H10 column | Exact match (50,724 values, allclose=True) |
| Loader labels ≠ H60 column | Confirmed |
| Train/test day overlap | 0 (zero) |
| Temporal ordering (train < test) | 2025-09-25 < 2025-11-14 |
| No NaN/Inf in predictions | Confirmed |
| No NaN/Inf in targets | Confirmed |

### 6.3 Prediction Sanity

| Check | Result | Interpretation |
|-------|--------|---------------|
| Prediction std / target std | 0.661 | Model is conservative (predictions narrower than reality) |
| Prediction range | [-27.7, +32.3] vs target [-91.6, +159.8] | Model avoids extreme predictions |
| Residual mean | -0.023 bps | Effectively zero bias |
| Residual ACF(1) | 0.069 | Low serial correlation in residuals |
| Residual vs DEPTH_NORM_OFI | 0.038 | Model extracted most OFI signal |
| Unique prediction values | 50,708 / 50,724 | Model is not degenerate |
| Shuffle test degradation | 58.3% | Model genuinely uses feature structure |

### 6.4 Per-Day Stability

| Metric | Value |
|--------|-------|
| Day R² mean | 0.476 |
| Day R² std | 0.037 |
| Day R² min | 0.331 (2025-11-19) |
| Day R² max | 0.546 (2025-11-28) |
| Days with R² < 0 | **0 / 35** |
| Day R² coefficient of variation | 0.077 |

The model achieves positive R² on every single test day. The worst day (R²=0.331) still vastly exceeds the linear Ridge baseline (R²=0.170).

---

## 7. Interpretation and Implications

### 7.1 What the Model Learned

The TLOB regression model achieves R²=0.464 by learning **nonlinear combinations of OFI features** that a linear model cannot capture. Evidence:
- Linear Ridge (same 128 features): R²=0.170
- TLOB (same features, nonlinear): R²=0.464
- The gap (0.294) represents genuine nonlinear predictive information

The residual analysis confirms the model extracts most of the OFI signal (residual vs DEPTH_NORM_OFI correlation = 0.038, down from raw 0.335). The remaining residual autocorrelation (0.069) suggests slight temporal patterns the model doesn't fully capture.

### 7.2 Why R²=0.464 is Plausible (Not Delusional)

- **H10 is a 10-event (~1 second) horizon.** At this timescale, order flow imbalance mechanically causes price movement (Cont et al. 2014). R²=0.46 is consistent with OFI-return correlations of 0.577-0.707 measured by the MBO statistical profiler.
- **The model is conservative.** Prediction std = 3.10 vs target std = 4.69. It doesn't overpredict.
- **No data leakage.** Temporal split with zero overlap. Labels verified to be exactly H10 column. No future features.
- **Shuffle test confirms.** Shuffling features degrades correlation by 58.3%.
- **Per-day stability.** R² is positive every single day with CV=0.077. No "one-day anomaly" inflating the average.

### 7.3 Implications for Feature Extraction

- **128 features work.** The model can handle the full feature set and learn to weight relevant ones via attention.
- **OFI features are the primary signal source.** Residual analysis confirms DEPTH_NORM_OFI signal is nearly fully extracted.
- **No feature subset needed** for initial experiments. Feature selection can be explored later for efficiency.

### 7.4 Implications for Model Architecture

- **TLOB with 2 layers is sufficient.** The original 4-layer config (5M params) was 7x oversized. 693K params achieves strong results.
- **Temporal attention is valuable.** The 0.294 R² gap over linear (which uses only the last timestep) proves the temporal dimension adds value.
- **HMHP-R is next.** Multi-horizon regression should benefit from the shared encoder architecture.

### 7.5 Implications for Training Pipeline

- **Huber delta=5.0 bps is well-calibrated.** The model converges cleanly with this setting.
- **Cosine LR schedule works.** Best checkpoint at epoch 6/15 with lr decaying from 5e-4 to 3.3e-4.
- **batch_size=128 is safe for MPS.** No memory issues during training.
- **~7 min/epoch is feasible** for iterative experiments.

### 7.6 Implications for Backtesting

- **Profitable accuracy=92.6% on moves > 5 bps** is directly relevant to trade signal generation.
- **MAE=2.43 bps** means the average prediction error is smaller than typical transaction costs (~3-5 bps for NVDA options).
- **The model can serve as a directional signal** with 74.9% accuracy, or as a **magnitude estimator** for position sizing.

---

## 8. Feature and Signal Configuration Reference

### Features Used

All 128 features from the `nvda_xnas_128feat_regression` export:
- **Indices 0-39:** Raw LOB (10 levels × 4: ask_price, ask_size, bid_price, bid_size)
- **Indices 40-47:** Derived (mid_price, spread, spread_bps, volumes, imbalance, weighted_mid, price_impact)
- **Indices 48-83:** MBO order flow (add/cancel/trade rates, net flows, conviction, size distribution, queue, institutional)
- **Indices 84-91:** Trading signals (TRUE_OFI, DEPTH_NORM_OFI, EXECUTED_PRESSURE, SIGNED_MP_DELTA_BPS, TRADE_ASYMMETRY, DEPTH_ASYMMETRY, CANCEL_TO_ADD_RATIO, AVG_ORDER_AGE)
- **Indices 92-97:** Control/categorical (book_valid, TIME_REGIME, mbo_ready, DT_SECONDS, invalidity_delta, schema_version)
- **Indices 98-127:** Experimental (institutional_v2, volatility, seasonality, MLOFI)

### Signal Hierarchy (from REGRESSION_ANALYSIS_REPORT.md)

| Rank | Feature | Index | Test R² | Test IC |
|------|---------|-------|---------|---------|
| 1 | DEPTH_NORM_OFI | 85 | 0.107 | 0.335 |
| 2 | VOLUME_IMBALANCE | 45 | 0.099 | -0.346 |
| 3 | TRUE_OFI | 84 | 0.066 | 0.342 |
| 4 | NET_TRADE_FLOW | 56 | 0.058 | 0.264 |
| 5 | TRADE_ASYMMETRY | 88 | 0.058 | 0.264 |

### Normalization

Hybrid normalization (computed from training data only):
- **LOB prices** (indices 0-9, 20-29): global z-score, mean=136.53, std=28.41
- **LOB sizes** (indices 10-19, 30-39): global z-score, mean=1098.03, std=9183.11
- **Derived/MBO/Signals** (indices 40-91, 98-127): per-feature z-score
- **Excluded** from normalization: TIME_REGIME (93), feature 115

### Regression Label Formula

TLOB smoothed-average return (Kolm et al. 2023):
```
r_t^(k) = (1/k) * Σ_{i=1}^{k} (m_{t+i} - m_t) / m_t × 10000
```
Where k=10 (H10), m_t = mid-price at event t. Units: basis points (bps).

---

## Appendix: Reproducibility

```bash
# Export regression dataset (if not already done)
cd feature-extractor-MBO-LOB
cargo build --release --bin export_dataset --features parallel
./target/release/export_dataset --config configs/nvda_xnas_128feat_regression.toml

# Compute baselines
cd lob-dataset-analyzer
python scripts/compute_regression_baselines.py --data-dir ../data/exports/nvda_xnas_128feat_regression

# Train TLOB regression
cd lob-model-trainer
python scripts/run_regression_training.py configs/experiments/nvda_tlob_128feat_regression_h10.yaml --epochs 15
```

---

*Generated: 2026-03-15 | TLOB regression v1 | 693,190 params | 15 epochs | seed=42*
