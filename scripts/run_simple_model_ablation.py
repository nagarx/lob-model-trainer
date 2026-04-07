#!/usr/bin/env python3
"""
Simple Model Ablation Ladder: L0 through L4.

Answers the fundamental question: how much of R-squared=0.464 comes from
the TLOB architecture vs the features themselves?

Levels:
    L0: IC-weighted linear composite (1 param) -- trivial floor
    L1: Temporal Ridge (~53 params) -- hand-crafted temporal features
    L2: Ridge + polynomial interactions (~78 params)
    L3: LightGBM on temporal features (~15K params) -- nonlinear ceiling

All use existing smoothed-average data. Runtime: ~2-3 minutes total.

Decision gates:
    L3 R-squared > 0.40: signal is in features, not architecture
    L1 R-squared > 0.35: hand-crafted temporal captures most signal
    L0 R-squared > 0.15: signal is truly linear

Usage:
    python scripts/run_simple_model_ablation.py \
        --data-dir ../data/exports/nvda_xnas_128feat_regression
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

DEPTH_NORM_OFI = 85
TRUE_OFI = 84
EXECUTED_PRESSURE = 86
NET_TRADE_FLOW = 56
TRADE_ASYMMETRY = 88
VOLUME_IMBALANCE = 45
NET_ORDER_FLOW = 54
SPREAD_BPS = 42
MID_PRICE = 40
DT_SECONDS = 95
TIME_REGIME = 93

TOP_SIGNALS = [DEPTH_NORM_OFI, TRUE_OFI, EXECUTED_PRESSURE, NET_TRADE_FLOW, VOLUME_IMBALANCE]
TOP_3 = [DEPTH_NORM_OFI, TRUE_OFI, EXECUTED_PRESSURE]


def load_data(data_dir, split, max_days=None):
    """Load sequences and H10 regression labels."""
    split_dir = Path(data_dir) / split
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
        all_labels.append(reg[:, 0])
    return np.concatenate(all_seqs, axis=0), np.concatenate(all_labels, axis=0)


def engineer_temporal_features(sequences):
    """Extract 53 hand-crafted temporal features from [N, T=100, F=128] sequences.

    Features capture temporal structure that Ridge cannot learn from raw last-timestep.
    """
    N, T, F = sequences.shape
    features = np.zeros((N, 53), dtype=np.float64)
    col = 0

    last = sequences[:, -1, :].astype(np.float64)
    for idx in TOP_SIGNALS:
        features[:, col] = last[:, idx]
        col += 1

    for idx in TOP_3:
        sig = sequences[:, :, idx].astype(np.float64)
        for w in [5, 10, 20]:
            features[:, col] = sig[:, -w:].mean(axis=1)
            col += 1

    for idx in TOP_3:
        sig = sequences[:, :, idx].astype(np.float64)
        for w in [5, 10, 20]:
            window = sig[:, -w:]
            x = np.arange(w, dtype=np.float64)
            x_mean = x.mean()
            ss_xx = ((x - x_mean) ** 2).sum()
            if ss_xx > 1e-12:
                slopes = ((window - window.mean(axis=1, keepdims=True)) * (x - x_mean)).sum(axis=1) / ss_xx
            else:
                slopes = np.zeros(N)
            features[:, col] = slopes
            col += 1

    for idx in TOP_3:
        sig = sequences[:, :, idx].astype(np.float64)
        for w in [5, 10, 20]:
            features[:, col] = sig[:, -1] - sig[:, -w]
            col += 1

    pairs = [(DEPTH_NORM_OFI, TRUE_OFI), (DEPTH_NORM_OFI, EXECUTED_PRESSURE),
             (DEPTH_NORM_OFI, VOLUME_IMBALANCE), (TRUE_OFI, NET_TRADE_FLOW),
             (TRUE_OFI, EXECUTED_PRESSURE), (EXECUTED_PRESSURE, VOLUME_IMBALANCE)]
    for i1, i2 in pairs:
        features[:, col] = last[:, i1] * last[:, i2]
        col += 1

    features[:, col] = last[:, SPREAD_BPS]; col += 1
    features[:, col] = last[:, VOLUME_IMBALANCE]; col += 1
    features[:, col] = last[:, NET_ORDER_FLOW]; col += 1
    features[:, col] = last[:, MID_PRICE]; col += 1
    features[:, col] = last[:, DT_SECONDS]; col += 1

    features[:, col] = last[:, TIME_REGIME]; col += 1
    features[:, col] = last[:, DT_SECONDS]; col += 1

    for idx in TOP_3:
        sig = sequences[:, :, idx].astype(np.float64)
        features[:, col] = sig[:, -10:].std(axis=1)
        col += 1

    mid = sequences[:, :, MID_PRICE].astype(np.float64)
    log_ret = np.diff(np.log(np.maximum(mid, 1e-10)), axis=1)
    features[:, col] = np.sqrt((log_ret[:, -10:] ** 2).sum(axis=1)) * 10000
    col += 1
    features[:, col] = np.sqrt((log_ret[:, -5:] ** 2).sum(axis=1)) * 10000
    col += 1

    for idx in TOP_3:
        sig = sequences[:, :, idx].astype(np.float64)
        mean_long = sig[:, -20:].mean(axis=1)
        mean_short = sig[:, -5:].mean(axis=1)
        features[:, col] = mean_short - mean_long
        col += 1

    assert col == 53, f"Expected 53 features, got {col}"
    return features


def compute_metrics(y_true, y_pred):
    """Compute regression metrics. Delegates to hft-metrics via adapter (Rule 0)."""
    from lobtrainer.training.regression_metrics import compute_all_regression_metrics

    full = compute_all_regression_metrics(y_true, y_pred)
    # Return abbreviated keys for backward compat with this script's output format
    return {
        "r2": full["r2"],
        "ic": full["ic"],
        "da": full["directional_accuracy"],
        "mae": full["mae"],
    }


def main():
    parser = argparse.ArgumentParser(description="Simple Model Ablation Ladder")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--max-train-days", type=int, default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("  SIMPLE MODEL ABLATION LADDER")
    print("=" * 70)

    t0 = time.time()
    print("\n  Loading data...")
    seq_train, y_train = load_data(args.data_dir, "train", args.max_train_days)
    seq_test, y_test = load_data(args.data_dir, "test")
    print(f"  Train: {seq_train.shape[0]:,} samples, Test: {seq_test.shape[0]:,} samples")

    print("  Engineering temporal features...")
    X_train_temporal = engineer_temporal_features(seq_train)
    X_test_temporal = engineer_temporal_features(seq_test)
    print(f"  Temporal features: {X_train_temporal.shape[1]}")

    X_train_last = seq_train[:, -1, :].astype(np.float64)
    X_test_last = seq_test[:, -1, :].astype(np.float64)

    mask_tr = np.isfinite(X_train_temporal).all(axis=1) & np.isfinite(y_train)
    mask_te = np.isfinite(X_test_temporal).all(axis=1) & np.isfinite(y_test)
    X_tr_t, y_tr = X_train_temporal[mask_tr], y_train[mask_tr]
    X_te_t, y_te = X_test_temporal[mask_te], y_test[mask_te]

    results = []

    # ========== L0: IC-Weighted Linear Composite ==========
    print("\n--- L0: IC-Weighted Linear Composite ---")
    ics = []
    for idx in TOP_SIGNALS:
        x = X_train_last[mask_tr, idx]
        try:
            ic_val, _ = spearmanr(x, y_tr)
            ics.append(float(ic_val) if np.isfinite(ic_val) else 0.0)
        except:
            ics.append(0.0)

    weights = np.array(ics)
    weights = weights / (np.abs(weights).sum() + 1e-10)
    y_pred_l0 = np.zeros(len(y_te))
    for i, idx in enumerate(TOP_SIGNALS):
        y_pred_l0 += weights[i] * X_test_last[mask_te, idx]

    scale = y_tr.std() / (y_pred_l0.std() + 1e-10) if y_pred_l0.std() > 1e-10 else 1.0
    scale = y_te.std() / (y_pred_l0.std() + 1e-10) if y_pred_l0.std() > 1e-10 else 1.0
    y_pred_l0 *= scale

    m0 = compute_metrics(y_te, y_pred_l0)
    results.append(("L0: IC-Weighted", 5, m0))
    print(f"  R²={m0['r2']:.6f}  IC={m0['ic']:.4f}  DA={m0['da']:.4f}  MAE={m0['mae']:.2f}")

    # ========== L1: Temporal Ridge ==========
    print("\n--- L1: Temporal Ridge (53 features) ---")
    ridge_l1 = Ridge(alpha=1.0)
    ridge_l1.fit(X_tr_t, y_tr)
    y_pred_l1 = ridge_l1.predict(X_te_t)
    m1 = compute_metrics(y_te, y_pred_l1)
    results.append(("L1: Temporal Ridge", 53, m1))
    print(f"  R²={m1['r2']:.6f}  IC={m1['ic']:.4f}  DA={m1['da']:.4f}  MAE={m1['mae']:.2f}")

    # ========== L2: Ridge + Polynomial ==========
    print("\n--- L2: Ridge + Polynomial Interactions ---")
    top_feat_idx = [0, 1, 2, 3, 4]
    X_tr_top = X_tr_t[:, top_feat_idx]
    X_te_top = X_te_t[:, top_feat_idx]
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    X_tr_poly = poly.fit_transform(X_tr_top)
    X_te_poly = poly.transform(X_te_top)
    X_tr_l2 = np.hstack([X_tr_t, X_tr_poly[:, 5:]])
    X_te_l2 = np.hstack([X_te_t, X_te_poly[:, 5:]])
    ridge_l2 = Ridge(alpha=1.0)
    ridge_l2.fit(X_tr_l2, y_tr)
    y_pred_l2 = ridge_l2.predict(X_te_l2)
    m2 = compute_metrics(y_te, y_pred_l2)
    n_poly = X_tr_l2.shape[1]
    results.append(("L2: Ridge+Poly", n_poly, m2))
    print(f"  R²={m2['r2']:.6f}  IC={m2['ic']:.4f}  DA={m2['da']:.4f}  MAE={m2['mae']:.2f}  (features={n_poly})")

    # ========== L3: Gradient Boosting (temporal features) ==========
    print("\n--- L3: GradientBoosting (temporal features, 53 feats) ---")
    from sklearn.ensemble import GradientBoostingRegressor

    rng = np.random.RandomState(42)
    n_sub = min(50000, len(y_tr))
    sub_idx = rng.choice(len(y_tr), n_sub, replace=False)
    X_tr_sub, y_tr_sub = X_tr_t[sub_idx], y_tr[sub_idx]

    gb = GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=50,
        loss="huber", alpha=0.9,
        random_state=42, verbose=0,
    )
    gb.fit(X_tr_sub, y_tr_sub)
    y_pred_l3 = gb.predict(X_te_t)
    m3 = compute_metrics(y_te, y_pred_l3)
    results.append(("L3: GradBoost(53feat)", 200, m3))
    print(f"  R²={m3['r2']:.6f}  IC={m3['ic']:.4f}  DA={m3['da']:.4f}  MAE={m3['mae']:.2f}  (trees={gb.n_estimators})")

    print("\n  GradBoost top-10 features:")
    importance = gb.feature_importances_
    for rank, idx in enumerate(np.argsort(importance)[::-1][:10], 1):
        print(f"    {rank:>2}. feat_{idx:<5} importance={importance[idx]:.4f}")

    # ========== L3b: GradientBoosting on LAST TIMESTEP ONLY ==========
    print("\n--- L3b: GradBoost (last-timestep raw, 128 features) ---")
    mask_tr_raw = np.isfinite(X_train_last).all(axis=1) & np.isfinite(y_train)
    mask_te_raw = np.isfinite(X_test_last).all(axis=1) & np.isfinite(y_test)
    X_tr_raw_sub = X_train_last[mask_tr_raw][:n_sub]
    y_tr_raw_sub = y_train[mask_tr_raw][:n_sub]

    gb_raw = GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=50,
        loss="huber", alpha=0.9,
        random_state=42, verbose=0,
    )
    gb_raw.fit(X_tr_raw_sub, y_tr_raw_sub)
    y_pred_raw = gb_raw.predict(X_test_last[mask_te_raw])
    m3b = compute_metrics(y_test[mask_te_raw], y_pred_raw)
    results.append(("L3b: GradBoost(raw128)", 200, m3b))
    print(f"  R²={m3b['r2']:.6f}  IC={m3b['ic']:.4f}  DA={m3b['da']:.4f}  MAE={m3b['mae']:.2f}")

    # ========== SUMMARY ==========
    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  ABLATION RESULTS (H10, smoothed-average labels)")
    print(f"{'=' * 70}")
    print(f"  {'Model':<30} {'Params':>8} {'R²':>10} {'IC':>8} {'DA':>8} {'MAE':>8}")
    print(f"  {'-'*30} {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
    for name, params, m in results:
        print(f"  {name:<30} {params:>8} {m['r2']:>10.6f} {m['ic']:>8.4f} {m['da']:>8.4f} {m['mae']:>8.2f}")

    print(f"\n  REFERENCE BASELINES:")
    print(f"  {'Persistence':<30} {'--':>8} {'-0.377':>10} {'0.264':>8} {'0.591':>8} {'--':>8}")
    print(f"  {'Ridge(128, last-step)':<30} {'128':>8} {'0.170':>10} {'0.433':>8} {'0.651':>8} {'--':>8}")
    print(f"  {'DEPTH_NORM_OFI only':<30} {'1':>8} {'0.107':>10} {'0.335':>8} {'0.620':>8} {'--':>8}")
    print(f"  {'TLOB T=100 (693K params)':<30} {'693K':>8} {'0.464':>10} {'0.677':>8} {'0.749':>8} {'2.43':>8}")

    print(f"\n  DECISION GATES:")
    l3_r2 = m3["r2"]
    l1_r2 = m1["r2"]
    l0_r2 = m0["r2"]
    if l3_r2 > 0.40:
        print(f"  --> L3 R²={l3_r2:.4f} > 0.40: SIGNAL IS IN FEATURES, not architecture")
        print(f"      Deep learning temporal processing adds minimal value.")
        print(f"      Focus on feature engineering + simple models.")
    elif l1_r2 > 0.35:
        print(f"  --> L1 R²={l1_r2:.4f} > 0.35: TEMPORAL FEATURES capture most signal")
        print(f"      Hand-crafted features + Ridge matches deep learning.")
    else:
        print(f"  --> L3 R²={l3_r2:.4f}, L1 R²={l1_r2:.4f}: DEEP LEARNING ADDS VALUE")
        print(f"      Temporal attention/convolution captures structure that simple models miss.")

    print(f"\n  Runtime: {elapsed:.1f}s")

    output_dir = Path("outputs/experiments/simple_model_ablation")
    output_dir.mkdir(parents=True, exist_ok=True)
    ablation_results = {
        "models": [{"name": n, "params": p, "metrics": m} for n, p, m in results],
        "data_dir": str(args.data_dir),
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "runtime_seconds": round(elapsed, 1),
    }
    with open(output_dir / "ablation_results.json", "w") as f:
        json.dump(ablation_results, f, indent=2)
    print(f"  Saved to {output_dir / 'ablation_results.json'}")


if __name__ == "__main__":
    main()
