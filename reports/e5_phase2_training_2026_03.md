# E5 Phase 2+3: Training + Backtest Results

**Date**: 2026-03-19 | **Experiment**: E5 Phases 2-3 | **Status**: COMPLETE

---

## 1. Training Configuration (5 Runs)

All runs: TLOB 2L/h=32/2Heads, T=20, seed=42, cosine scheduler, AdamW(lr=5e-4, wd=0.01), grad_clip=1.0, early_stopping patience=5.

| Run | Config | Bin | Features | CVML | Loss | Huber δ | Params | Train Seqs |
|-----|--------|-----|----------|------|------|---------|--------|-----------|
| 1 | e5_60s_huber_nocvml | 60s | 98 | No | Huber | 12.6 | 92,690 | 39,749 |
| 2 | e5_60s_huber_cvml | 60s | 98→49 | Yes | Huber | 12.6 | 120,179 | 39,749 |
| 3 | e5_60s_gmadl_cvml | 60s | 98→49 | Yes | GMADL a=10 b=1.5 | — | 120,179 | 39,749 |
| 4 | e5_30s_huber_nocvml | 30s | 98 | No | Huber | 15.1 | 92,690 | 132,150 |
| 5 | e5_30s_huber_cvml | 30s | 98→49 | Yes | Huber | 15.1 | 120,179 | 132,150 |

Huber deltas calibrated empirically: 60s H10 kurtosis=26.5 → δ=12.6 bps; 30s H10 kurtosis=12.6 → δ=15.1 bps.

---

## 2. Results

| Run | Best Ep | Val IC | Val DA | Test IC | Test DA | Test R² | Test MAE | Time (s) | Status |
|-----|---------|--------|--------|---------|---------|---------|----------|---------|--------|
| **1** | **4** | **0.374** | **0.635** | **0.380** | **0.640** | **0.124** | **17.8** | 224 | **BEST** |
| 2 | 6 | 0.363 | 0.635 | 0.373 | 0.640 | 0.121 | 17.8 | 315 | PASS |
| 3 | 12 | 0.024 | 0.521 | 0.007 | 0.498 | -0.001 | 19.1 | 476 | **FAIL** |
| 4 | 6 | 0.382 | 0.638 | 0.379 | 0.637 | 0.132 | 11.5 | 982 | PASS |
| 5 | 4 | 0.385 | 0.642 | 0.380 | 0.641 | 0.128 | 11.6 | 841 | PASS |

**Baselines** (from Phase 1):
- TemporalRidge: IC=0.306, DA=0.613 (53 params)
- DEPTH_NORM_OFI: IC=0.255 (1 param)

**All Huber models beat baseline**: IC=0.373-0.380 > 0.306 (+22-24%). DA=0.637-0.641 > 0.613 (+2.4-2.8pp).

---

## 3. CVML Analysis

**Hypothesis**: CVML (Li et al. ICLR 2025) mixes features across LOB dimensions via 5 dilated causal Conv1D layers, reducing 98→49 features. Achieved +244.9% R² on MPRF benchmark.

**Result**: No improvement.
- 60s: IC=0.373 (CVML) vs 0.380 (no CVML) — CVML is **worse by 1.8%**
- 30s: IC=0.380 (CVML) vs 0.379 (no CVML) — within noise

**Why CVML doesn't help on our data**:
1. **Feature dimensionality is already low** (98 features vs 128+ in MPRF paper)
2. **Small effective training set** (~4K independent samples at 60s, ~13K at 30s) — insufficient for the 28K additional CVML parameters to learn meaningful cross-feature patterns
3. **Features are naturally structured** (LOB prices/sizes are ordered by level) — Conv1D mixing may disrupt this structure rather than enhance it

---

## 4. GMADL Failure Analysis

**Hypothesis**: GMADL (Michankov et al. 2024) optimizes directional accuracy by penalizing wrong-direction predictions more than wrong magnitudes. Expected to improve DA beyond Huber.

**Result**: Complete failure (IC=0.007, DA=49.8% — worse than random).

**Root cause**: GMADL loss became negative around epoch 16, inverting the optimization gradient. The model converged to predicting near-zero values (mean prediction), achieving DA≈50% (random).

**Technical details**:
- GMADL = (-1) × (sigmoid(a × R × R_hat) - 0.5) × |R|^b
- With a=10, b=1.5: the sigmoid is very sharp, creating near-binary {-0.5, +0.5} direction scores
- Combined with |R|^1.5 magnitude weighting, the loss landscape has steep gradients that cause instability
- Early stopping on val_loss doesn't help because GMADL loss can go negative (reward > penalty)

**Recommendation**: If retrying GMADL, use a=2-5 (softer sigmoid), b=1.0 (no magnitude upweighting), lower learning rate (1e-5), and separate early stopping metric (val_DA, not val_loss).

---

## 5. 30s vs 60s Bins

| Metric | 60s bins (Run 1) | 30s bins (Run 4) | Winner |
|--------|-----------------|-----------------|--------|
| Sweep IC (Phase 0) | 0.248 | 0.240 | 60s (+3.3%) |
| Test IC (trained) | 0.380 | 0.379 | **Tied** |
| Test DA | 0.640 | 0.637 | 60s (+0.3pp) |
| Test R² | 0.124 | 0.132 | **30s** (+6.5%) |
| Train sequences | 39,749 | 132,150 | **30s** (3.3x) |
| Training time | 224s | 982s | 60s (4.4x faster) |
| Best epoch | 4 | 6 | 60s (faster convergence) |

**Conclusion**: 30s bins compensate for lower per-sample IC with 3.3x more training data. Both achieve identical test IC (0.379-0.380). Use 60s for faster iteration; use 30s for more robust training.

---

## 6. Comparison with E4

| Metric | E4 (5s/H60=5min) | E5 (60s/H10=10min) | Improvement |
|--------|------------------|---------------------|-------------|
| Feature IC (sweep) | 0.082 | 0.248 | **+202%** |
| Test IC (trained) | 0.136 | 0.380 | **+180%** |
| Test DA | 0.544 | 0.640 | **+9.6pp** |
| Test R² | 0.015 | 0.124 | **+7.3x** |
| Best epoch | 1 | 4 | More training headroom |
| Backtest (Deep ITM) | -3.68% | -1.93% | **+1.75pp** |
| Backtest win rate | 45.0% | 40.1% | -4.9pp (worse) |

**Key insight**: Upstream metrics (IC, DA, R²) improved dramatically. But backtest win rate **decreased** because the 10-minute hold exposes positions to more price movement than E4's ~1-minute hold.

---

## 7. Phase 3 Backtest (Round 7)

**Model**: Run 1 (e5_60s_huber_nocvml), best checkpoint at epoch 4.
**Signal export**: 8,337 test sequences. Prediction std=7.35 bps, actual return std=27.4 bps.
**Hold**: 10 events × 60s = 10 minutes.

### Deep ITM (delta=0.95, breakeven=1.4 bps)

| Threshold | Trades | Return | Win Rate | ProfitFactor |
|-----------|--------|--------|----------|--------------|
| 0.7 bps | 740 | -1.93% | 40.1% | 0.622 |
| 8.0 bps | 594 | -1.37% | 37.0% | 0.635 |

### ATM (delta=0.50, breakeven=4.9 bps)

| Threshold | Trades | Return |
|-----------|--------|--------|
| 0.7 bps | 740 | -3.07% |
| 8.0 bps | 594 | -2.43% |

**Decision gate**: Option return > 0% → **FAIL** (best = -1.37% at 8bps Deep ITM)

---

## 8. Root Cause: Label-Execution Mismatch Persists

DA=64.0% measures accuracy on smoothed-average labels:
- Label: (1/5) × sum_{i=6}^{10} (mid[t+i] - mid[t]) / mid[t] × 10000

Execution measures point-to-point return:
- P&L: (mid[t+10] - mid[t]) / mid[t] × 10000

These have correlation r≈0.64 (P0 validation). So DA=64% on smoothed → ~55% on point-to-point → 40% after cost drag and model conservatism (prediction std=7.35 bps vs actual 27.4 bps).

**The 60s time-bin optimization dramatically improved the signal but did NOT solve the fundamental label-execution alignment problem.**

---

## 9. Lessons for Future Experiments

| What Worked | What Didn't | Don't Repeat |
|-------------|-------------|--------------|
| 60s bins → 3x IC | CVML on 98 features | CVML on small feature sets |
| Huber loss with calibrated δ | GMADL a=10 b=1.5 | GMADL without extensive tuning |
| Early stopping patience=5 | Expecting DA to match execution | Treating smoothed DA as tradeable |
| 30s bins = viable alternative | Larger model (CVML +28K params) | Adding complexity to weak signal |
| Simplest model wins | — | — |

---

## Source Files

| File | Description |
|------|-------------|
| `configs/experiments/e5_60s_huber_nocvml.yaml` | Best model config |
| `outputs/experiments/e5_60s_huber_nocvml/checkpoints/best.pt` | Best checkpoint |
| `outputs/experiments/e5_60s_huber_nocvml/signals/test/` | Signal export |
| `outputs/experiments/e5_60s_huber_nocvml/training_history.json` | Per-epoch metrics |
| `../lob-backtester/outputs/backtests/e5_round7/` | Backtest JSONs |
| `../lob-dataset-analyzer/reports/E5_SWEEP_RESULTS_2026_03.md` | Phase 0 sweep |
