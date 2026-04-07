#!/usr/bin/env python3
"""
E4 Baseline Models: Temporal Ridge + GradBoost on time-based H60 data.

Extends the ablation ladder to the E4 time-based export targeting H60
(5-minute returns). Adapts the proven temporal feature engineering
(53 features, IC=0.616 at H10) to the new sampling resolution.

Key differences from prior ablation:
- T=20 (not T=100), stride=1 (not 10), 98 features (not 128)
- Target: H60 (index 1) = 5 minutes, not H10
- Time-based sampling at 5-second intervals

Decision gates:
    Ridge R² > 0.02 at H60: deep learning justified
    TemporalRidge IC > single-feature IC (0.087): temporal features add value
    TemporalRidge R² > Ridge R² × 1.5: temporal structure matters

Usage:
    python scripts/e4_baselines.py \
        --data-dir ../data/exports/e4_timebased_5s \
        --horizon-idx 1

Reference: EXPERIMENT_INDEX.md § E4, ABLATION_FINDINGS_2026_03_16.md
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor

# Feature indices (all within 98-feature layout)
DEPTH_NORM_OFI = 85
TRUE_OFI = 84
EXECUTED_PRESSURE = 86
NET_TRADE_FLOW = 56
VOLUME_IMBALANCE = 45
NET_ORDER_FLOW = 54
TRADE_ASYMMETRY = 88
SPREAD_BPS = 42
MID_PRICE = 40
DT_SECONDS = 95
TIME_REGIME = 93

TOP_SIGNALS = [DEPTH_NORM_OFI, TRUE_OFI, EXECUTED_PRESSURE, NET_TRADE_FLOW, VOLUME_IMBALANCE]
TOP_3 = [DEPTH_NORM_OFI, TRUE_OFI, EXECUTED_PRESSURE]

HORIZON_NAMES = {0: "H10", 1: "H60", 2: "H300"}


def load_data(data_dir: Path, split: str, horizon_idx: int, max_days: int = None):
    """Load sequences and regression labels for a specific horizon."""
    split_dir = data_dir / split
    metas = sorted(split_dir.glob("*_metadata.json"))
    if max_days:
        metas = metas[:max_days]
    all_seqs, all_labels = [], []
    for mf in metas:
        m = json.load(open(mf))
        day = m["day"]
        seq = np.load(split_dir / f"{day}_sequences.npy", mmap_mode="r")
        reg = np.load(split_dir / f"{day}_regression_labels.npy")
        all_seqs.append(seq)
        all_labels.append(reg[:, horizon_idx])
    return np.concatenate(all_seqs, axis=0), np.concatenate(all_labels, axis=0)


def engineer_temporal_features(sequences: np.ndarray) -> np.ndarray:
    """Extract 53 hand-crafted temporal features from [N, T, F] sequences.

    Adapted from ablation ladder. Works with any T >= 20.

    Feature groups:
        5 last-timestep signals
        9 rolling means (3 signals × 3 windows)
        9 rolling slopes (3 signals × 3 windows)
        9 rate-of-change (3 signals × 3 windows)
        6 cross-feature products
        5 context features
        2 regime indicators
        5 volatility features
        3 momentum features
    Total: 53
    """
    N, T, F = sequences.shape
    features = np.zeros((N, 53), dtype=np.float64)
    col = 0

    last = sequences[:, -1, :].astype(np.float64)

    # 5 last-timestep signal values
    for idx in TOP_SIGNALS:
        features[:, col] = last[:, idx]
        col += 1

    # 9 rolling means
    for idx in TOP_3:
        sig = sequences[:, :, idx].astype(np.float64)
        for w in [5, 10, 20]:
            w_eff = min(w, T)
            features[:, col] = sig[:, -w_eff:].mean(axis=1)
            col += 1

    # 9 rolling slopes
    for idx in TOP_3:
        sig = sequences[:, :, idx].astype(np.float64)
        for w in [5, 10, 20]:
            w_eff = min(w, T)
            window = sig[:, -w_eff:]
            x = np.arange(w_eff, dtype=np.float64)
            x_mean = x.mean()
            ss_xx = ((x - x_mean) ** 2).sum()
            if ss_xx > 1e-12:
                slopes = ((window - window.mean(axis=1, keepdims=True)) * (x - x_mean)).sum(axis=1) / ss_xx
            else:
                slopes = np.zeros(N)
            features[:, col] = slopes
            col += 1

    # 9 rate-of-change
    for idx in TOP_3:
        sig = sequences[:, :, idx].astype(np.float64)
        for w in [5, 10, 20]:
            w_eff = min(w, T)
            features[:, col] = sig[:, -1] - sig[:, -w_eff]
            col += 1

    # 6 cross-feature products
    pairs = [
        (DEPTH_NORM_OFI, TRUE_OFI), (DEPTH_NORM_OFI, EXECUTED_PRESSURE),
        (DEPTH_NORM_OFI, VOLUME_IMBALANCE), (TRUE_OFI, NET_TRADE_FLOW),
        (TRUE_OFI, EXECUTED_PRESSURE), (EXECUTED_PRESSURE, VOLUME_IMBALANCE),
    ]
    for i, j in pairs:
        features[:, col] = last[:, i] * last[:, j]
        col += 1

    # 5 context features
    for idx in [SPREAD_BPS, VOLUME_IMBALANCE, NET_ORDER_FLOW, MID_PRICE, DT_SECONDS]:
        features[:, col] = last[:, idx]
        col += 1

    # 2 regime
    features[:, col] = last[:, TIME_REGIME]; col += 1
    features[:, col] = last[:, DT_SECONDS]; col += 1

    # 3 volatility: std of top-3 signals over window=10
    for idx in TOP_3:
        sig = sequences[:, :, idx].astype(np.float64)
        w_eff = min(10, T)
        features[:, col] = sig[:, -w_eff:].std(axis=1)
        col += 1

    # 2 realized vol
    mid = sequences[:, :, MID_PRICE].astype(np.float64)
    mid_safe = np.where(mid < 1e-6, 1.0, mid)
    log_ret = np.diff(np.log(mid_safe), axis=1)
    for w in [10, 5]:
        w_eff = min(w, log_ret.shape[1])
        features[:, col] = log_ret[:, -w_eff:].std(axis=1) * np.sqrt(w_eff)
        col += 1

    # 3 momentum: short-term vs long-term mean
    for idx in TOP_3:
        sig = sequences[:, :, idx].astype(np.float64)
        short_w = min(5, T)
        long_w = min(20, T)
        features[:, col] = sig[:, -short_w:].mean(axis=1) - sig[:, -long_w:].mean(axis=1)
        col += 1

    assert col == 53, f"Expected 53 features, got {col}"
    return features


# Metrics: single source of truth via hft-metrics adapter (Rule 0)
from lobtrainer.training.regression_metrics import (
    r_squared,
    information_coefficient as spearman_ic,
    directional_accuracy,
    mean_absolute_error as mae,
)


def main():
    parser = argparse.ArgumentParser(description="E4 Baseline Models")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--horizon-idx", type=int, default=1, help="Horizon index (0=H10, 1=H60, 2=H300)")
    parser.add_argument("--max-train-days", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    horizon_name = HORIZON_NAMES.get(args.horizon_idx, f"H{args.horizon_idx}")

    print(f"\n{'='*70}")
    print(f"  E4 BASELINE MODELS — {horizon_name}")
    print(f"  Data: {args.data_dir}")
    print(f"{'='*70}\n")

    # Load data
    t0 = time.time()
    print("Loading train split...")
    train_seqs, train_labels = load_data(args.data_dir, "train", args.horizon_idx, args.max_train_days)
    print(f"  Train: {train_seqs.shape[0]:,} samples, shape {train_seqs.shape}")

    print("Loading test split...")
    test_seqs, test_labels = load_data(args.data_dir, "test", args.horizon_idx)
    print(f"  Test:  {test_seqs.shape[0]:,} samples, shape {test_seqs.shape}")

    print(f"  Load time: {time.time()-t0:.1f}s\n")
    print(f"  Train labels: mean={train_labels.mean():.4f}, std={train_labels.std():.2f} bps")
    print(f"  Test labels:  mean={test_labels.mean():.4f}, std={test_labels.std():.2f} bps\n")

    results = {
        "experiment": "E4 Baselines",
        "horizon": horizon_name,
        "horizon_idx": args.horizon_idx,
        "train_samples": int(train_seqs.shape[0]),
        "test_samples": int(test_seqs.shape[0]),
        "models": {},
    }

    # =========================================================================
    # L1: Temporal Ridge (53 features)
    # =========================================================================
    print(f"--- L1: Temporal Ridge (53 features) ---")
    t1 = time.time()

    train_tf = engineer_temporal_features(train_seqs)
    test_tf = engineer_temporal_features(test_seqs)

    # Handle NaN/Inf
    train_tf = np.nan_to_num(train_tf, nan=0.0, posinf=0.0, neginf=0.0)
    test_tf = np.nan_to_num(test_tf, nan=0.0, posinf=0.0, neginf=0.0)

    ridge = Ridge(alpha=1.0)
    ridge.fit(train_tf, train_labels)
    pred_ridge = ridge.predict(test_tf)

    r2 = r_squared(test_labels, pred_ridge)
    ic = spearman_ic(test_labels, pred_ridge)
    da = directional_accuracy(test_labels, pred_ridge)
    m = mae(test_labels, pred_ridge)

    print(f"  R²  = {r2:.6f}")
    print(f"  IC  = {ic:.4f}")
    print(f"  DA  = {da:.4f}")
    print(f"  MAE = {m:.2f} bps")
    print(f"  Time: {time.time()-t1:.1f}s")

    # Top coefficients
    coef_idx = np.argsort(np.abs(ridge.coef_))[::-1][:10]
    print(f"  Top 10 coefficients:")
    for rank, ci in enumerate(coef_idx, 1):
        print(f"    {rank:2d}. feature {ci:3d}  coef={ridge.coef_[ci]:+.6f}")
    print()

    results["models"]["temporal_ridge"] = {
        "params": 53, "r2": r2, "ic": ic, "da": da, "mae": m,
    }

    # =========================================================================
    # L2: Temporal GradBoost (200 trees)
    # =========================================================================
    print(f"--- L2: Temporal GradBoost (200 trees) ---")
    t2 = time.time()

    # Subsample for speed if >200K samples
    if len(train_tf) > 200000:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(train_tf), 200000, replace=False)
        train_tf_sub = train_tf[idx]
        train_labels_sub = train_labels[idx]
    else:
        train_tf_sub = train_tf
        train_labels_sub = train_labels

    gb = GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        loss="huber", alpha=0.9, random_state=42,
        subsample=0.8, min_samples_leaf=50,
    )
    gb.fit(train_tf_sub, train_labels_sub)
    pred_gb = gb.predict(test_tf)

    r2_gb = r_squared(test_labels, pred_gb)
    ic_gb = spearman_ic(test_labels, pred_gb)
    da_gb = directional_accuracy(test_labels, pred_gb)
    m_gb = mae(test_labels, pred_gb)

    print(f"  R²  = {r2_gb:.6f}")
    print(f"  IC  = {ic_gb:.4f}")
    print(f"  DA  = {da_gb:.4f}")
    print(f"  MAE = {m_gb:.2f} bps")
    print(f"  Time: {time.time()-t2:.1f}s")

    # Top feature importances
    imp_idx = np.argsort(gb.feature_importances_)[::-1][:10]
    print(f"  Top 10 feature importances:")
    for rank, fi in enumerate(imp_idx, 1):
        print(f"    {rank:2d}. feature {fi:3d}  importance={gb.feature_importances_[fi]:.4f}")
    print()

    results["models"]["temporal_gradboost"] = {
        "params": "~200 trees", "r2": r2_gb, "ic": ic_gb, "da": da_gb, "mae": m_gb,
    }

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"{'='*70}")
    print(f"  SUMMARY — {horizon_name}")
    print(f"{'='*70}")
    print(f"  {'Model':<30s} {'R²':>10s} {'IC':>10s} {'DA':>8s} {'MAE':>8s}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
    for name, m in results["models"].items():
        print(f"  {name:<30s} {m['r2']:10.6f} {m['ic']:10.4f} {m['da']:8.4f} {m['mae']:8.2f}")

    # Decision gates
    print(f"\n  Decision Gates:")
    b1 = r2 > 0.02
    b2 = ic > 0.087  # single-feature IC at H60
    b3 = r2 > 0.0096 * 1.5  # > 1.5x Ridge R²
    print(f"    B1: Ridge R² > 0.02 → {'PASS' if b1 else 'FAIL'} (R²={r2:.6f})")
    print(f"    B2: TemporalRidge IC > single-feat IC (0.087) → {'PASS' if b2 else 'FAIL'} (IC={ic:.4f})")
    print(f"    B3: TemporalRidge R² > 1.5× Ridge R² (0.0144) → {'PASS' if b3 else 'FAIL'} (R²={r2:.6f})")

    results["decision_gates"] = {
        "B1_ridge_r2": {"threshold": 0.02, "value": r2, "passed": bool(b1)},
        "B2_temporal_ic": {"threshold": 0.087, "value": ic, "passed": bool(b2)},
        "B3_temporal_vs_ridge": {"threshold": 0.0096 * 1.5, "value": r2, "passed": bool(b3)},
    }

    print(f"{'='*70}\n")

    # Save results
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out_file = args.output_dir / f"e4_baselines_{horizon_name}.json"
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to: {out_file}")


if __name__ == "__main__":
    main()
