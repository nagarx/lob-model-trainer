# E4 Training + Backtest Report (2026-03-18)

**Experiment**: E4 — Time-Based Sampling OFI Validation
**Phases**: 0 (export) → 1 (validation) → 2 (baselines) → 3 (TLOB training) → 4 (backtest)
**Status**: COMPLETE — All phases executed. Signal validated. Profitability NOT achieved.

---

## 1. Training Configuration (Phase 3)

| Parameter | Value | Source |
|-----------|-------|--------|
| Config | `configs/experiments/e4_tlob_h60.yaml` | |
| Model | TLOB (2 layers, hidden=32, 2 heads) | Proven on event-based H10 |
| Parameters | 92,690 | 7.5x smaller than T=100 (693K) |
| Input shape | [B, 20, 98] | T=20 × 5s = 100 seconds, 98 features |
| Target | H60 (horizon_idx=1, 5-minute returns) | IC=0.085-0.091 at H60 |
| Loss | Huber (delta=7.3 bps) | Calibrated from E4 H60 kurtosis=33 |
| Optimizer | AdamW (lr=5e-4, weight_decay=0.01) | Same as REG-01 |
| Training | 11 epochs, early stop at epoch 11 (best=epoch 1) | |
| Checkpoint | `outputs/experiments/e4_tlob_h60/checkpoints/best.pt` | Epoch 1 weights |

## 2. Training Results (Epoch-by-Epoch)

| Epoch | Train Loss | Val Loss | Val IC | Val R2 | Val DA | Val MAE |
|-------|-----------|----------|--------|--------|--------|---------|
| **1** | 72.28 | **59.23** | **+0.143** | **+0.016** | **0.548** | 11.01 |
| 2 | 71.99 | 59.29 | +0.129 | +0.016 | 0.545 | 11.02 |
| 3 | 71.55 | 59.56 | +0.106 | +0.012 | 0.536 | 11.06 |
| 4 | 71.01 | 59.77 | +0.092 | +0.006 | 0.531 | 11.09 |
| 5 | 70.36 | 60.25 | +0.080 | -0.010 | 0.526 | 11.16 |
| 6 | 69.63 | 60.53 | +0.068 | -0.019 | 0.525 | 11.20 |
| 7 | 68.88 | 60.19 | +0.074 | -0.008 | 0.527 | 11.15 |
| 8 | 68.22 | 60.52 | +0.067 | -0.014 | 0.522 | 11.20 |
| 9 | 67.57 | 61.02 | +0.061 | -0.033 | 0.517 | 11.28 |
| 10 | 66.94 | 60.94 | +0.075 | -0.030 | 0.526 | 11.26 |
| 11 | 66.29 | 60.53 | +0.071 | -0.018 | 0.524 | 11.21 |

**Pattern**: Train loss decreases monotonically (72.28 → 66.29). Val IC PEAKS at epoch 1 (+0.143) then monotonically degrades. Best model = epoch 1 (early stopping restored best weights).

**Root cause of immediate overfitting**:
1. Only 2 of 98 features have IC > 0.05 at H60 — 92.7K params for 2 effective signals
2. Stride=1 creates 95% feature overlap between consecutive samples — 971K samples but ~6K independent observations/day
3. Model captures the useful pattern in one gradient pass, then memorizes training noise

## 3. Test Metrics

| Metric | Val (epoch 1) | **Test** | TemporalRidge (baseline) | Improvement |
|--------|---------------|----------|--------------------------|-------------|
| **IC** | 0.143 | **0.136** | 0.121 | +12.4% |
| R2 | 0.016 | 0.015 | 0.013 | +17.7% |
| DA | 0.548 | 0.544 | 0.543 | +0.2% |
| MAE | 11.01 bps | 12.43 bps | 12.46 bps | -0.2% |
| Pearson | 0.139 | 0.132 | — | — |
| Profitable Acc | 0.565 | 0.563 | — | — |

**Decision gates**:
- G5: Test IC > 0.121 → **PASS** (0.136, +12.4%)
- G6: Test R2 > 0.013 → **PASS** (0.015, +17.7%)
- G7: |Val-Test IC gap| < 10% → **PASS** (4.9%)
- G8: Best epoch < 30 → **PASS** (epoch 1, but concerning)

## 4. Backtest Results (Phase 4)

### ATM Options (delta=0.50, half-spread=$0.015, breakeven=4.9 bps)

| Threshold | Trades | 0DTE Return | Win Rate | Avg P&L/Trade |
|-----------|--------|-------------|----------|---------------|
| 0.7 bps | 3,145 | **-19.81%** | 27.4% | -$6.30 |
| 2.0 bps | 2,488 | **-15.06%** | 32.4% | -$6.05 |
| 3.0 bps | 2,153 | **-13.68%** | 32.2% | -$6.35 |
| 5.0 bps | 763 | **-5.25%** | 36.3% | -$6.88 |
| 8+ bps | 0 | — | — | — |

### Deep ITM Options (delta=0.95, half-spread=$0.005, breakeven=1.4 bps)

| Threshold | Trades | 0DTE Return | Win Rate | Avg P&L/Trade |
|-----------|--------|-------------|----------|---------------|
| 0.7 bps | 3,145 | **-14.24%** | 38.0% | -$4.53 |
| 2.0 bps | 2,488 | **-10.71%** | 41.7% | -$4.30 |
| 3.0 bps | 2,153 | **-10.73%** | 41.5% | -$4.98 |
| 5.0 bps | 763 | **-3.68%** | 45.0% | -$4.82 |
| 8+ bps | 0 | — | — | — |

### ATM vs Deep ITM Comparison

| Threshold | ATM Return | Deep ITM Return | Improvement | ATM Win% | Deep ITM Win% |
|-----------|-----------|-----------------|-------------|----------|---------------|
| 0.7 bps | -19.8% | -14.2% | +5.6 pp | 27.4% | 38.0% |
| 2.0 bps | -15.1% | -10.7% | +4.4 pp | 32.4% | 41.7% |
| 3.0 bps | -13.7% | -10.7% | +3.0 pp | 32.2% | 41.5% |
| 5.0 bps | -5.3% | **-3.7%** | +1.6 pp | 36.3% | **45.0%** |

## 5. Comparison with Prior Backtest Rounds

| Round | Model | Horizon | Best Option Return | Best Win Rate | Notes |
|-------|-------|---------|--------------------|--------------| ------|
| R1 | HMHP classification | H10 | -4.50% | 42.8% | First backtest |
| R4 | TLOB regression (event) | H10 | -0.35% | ~40% | Event-based, 54 trades |
| R5 | HMHP+Ridge hybrid | H60 | -2.67% | 42.8% | Readability gate |
| **R6** | **TLOB regression (time)** | **H60** | **-3.68%** | **45.0%** | **E4 deep ITM** |

E4 result (-3.68% at 45% win rate) is comparable to R5 (-2.67% at 42.8% win rate). Neither achieves profitability.

## 6. Root Cause Analysis

### Why the model has positive IC but negative P&L

1. **IC measures ranking, not direction**: IC=0.136 means the model correctly RANKS 56.8% of return pairs. But at a fixed threshold (e.g., 5 bps), directional accuracy is only 45%.

2. **Model prediction range is narrow**: std=1.78 bps (range: -4.8 to +7.3 bps). Actual returns: std=20.5 bps. The model is extremely conservative — it never predicts large moves.

3. **Asymmetric loss**: When the model is wrong, losses are proportional to the full return magnitude (~20 bps std). When right, gains are also proportional. But at 45% accuracy, cumulative losses exceed gains.

4. **Cost structure**: Even deep ITM breakeven (1.4 bps) consumes a significant fraction of the model's small predicted returns (mean |pred| ≈ 2.5 bps at 0.7 bps threshold).

### Why training overfits after epoch 1

- **Signal is thin**: Only 2 features (TRUE_OFI, DEPTH_NORM_OFI) have IC > 0.05 at H60
- **Data overlap**: Stride=1 with T=20 means 95% overlap between consecutive samples
- **Overparameterization**: 92.7K parameters for essentially 2 input signals

## 7. What We Learned (Lessons for EXPERIMENT_INDEX)

### Validated
1. **Time-based sampling IS the fix**: IC went from ZERO (event-based) to 0.083-0.089 at H60. This was the architectural breakthrough.
2. **TLOB adds marginal value over Ridge**: +12% IC (0.136 vs 0.121), confirming temporal attention captures patterns Ridge misses.
3. **Deep ITM is consistently better**: +5-6 pp return, +9-11 pp win rate vs ATM options.
4. **Signal is stable**: Walk-forward IC stability=15.2, cross-split CV=2-3%.

### Not sufficient
5. **IC=0.136 is insufficient for profitability**: Even at deep ITM (1.4 bps breakeven), 45% win rate at 5 bps threshold cannot overcome costs.
6. **Model predictions are too conservative**: std=1.78 bps vs actual std=20.5 bps. The model hedges, producing small predictions that are costly relative to execution costs.
7. **Epoch 1 = best**: No further training helps. Signal is captured instantly.

### What NOT to repeat
8. **Do NOT train more epochs on weak-signal data** — epoch 1 is optimal when only 2 features have IC > 0.05.
9. **Do NOT use stride=1 with overlapping labels** — creates massive effective duplication that inflates apparent sample size without adding information.

## 8. Next Directions

| Direction | Rationale | Feasibility |
|-----------|-----------|-------------|
| **H300 (25-min horizon)** | Longer hold = more return accumulation, but IC decays to 0.03 | Medium (IC too low?) |
| **Larger time bins (30s, 60s)** | More OFI persistence per bin, fewer but more independent samples | High |
| **Alternative data** (options flow, dark pools) | New signal sources orthogonal to MBO OFI | High effort |
| **Smaller model** (hidden=16, 1 layer) | Less overfitting on weak signal | Easy to test |
| **Lower learning rate** (1e-5) | Slower convergence may find better optimum | Easy to test |
| **Stride=10 or subsample** | Reduce overlap, force truly independent samples | Easy to test |
| **GMADL loss** (directional reward) | Optimize for direction, not magnitude | Medium |

---

**Source files**:
- Training: `outputs/experiments/e4_tlob_h60/training_history.json`
- Checkpoint: `outputs/experiments/e4_tlob_h60/checkpoints/best.pt`
- Signals: `outputs/experiments/e4_tlob_h60/signals/test/`
- ATM backtest: `lob-backtester/outputs/backtests/e4_tlob_h60_atm.json`
- Deep ITM backtest: `lob-backtester/outputs/backtests/e4_tlob_h60_deep_itm.json`
- Cost audit: `IBKR-transactions-trades/COST_AUDIT_2026_03.md`
