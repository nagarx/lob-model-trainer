#!/usr/bin/env python3
"""
XGBoost Baseline Training Script.

Trains an XGBoost classifier on flat (single-snapshot) features from the
128-feature MBO exports. Uses the last timestep of each sequence as the
feature vector, matching XGBoostLOB's default pooling strategy.

This script reuses the trainer's data loading infrastructure but bypasses
the gradient-based PyTorch training loop — XGBoost uses its own fit() API.

Usage:
    python scripts/train_xgboost_baseline.py --config configs/experiments/nvda_xgboost_baseline_h60.yaml

    python scripts/train_xgboost_baseline.py \\
        --data-dir ../data/exports/nvda_xnas_128feat \\
        --horizon-idx 3 \\
        --output-dir outputs/xgboost_baseline_h60
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lobtrainer.data.dataset import load_split_data, DayData
from lobtrainer.constants.feature_presets import FEATURE_PRESETS


def load_config(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def flatten_days(
    days: List[DayData],
    horizon_idx: int,
    feature_indices: Optional[Tuple[int, ...]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten day data into (X, y) arrays for XGBoost.

    Uses the last timestep of each sequence (features field from DayData).
    Selects a single horizon from multi-horizon labels.
    """
    all_x, all_y = [], []
    for day in days:
        x = day.features
        if feature_indices is not None:
            valid = [i for i in feature_indices if i < x.shape[1]]
            x = x[:, valid]

        y = day.get_labels(horizon_idx=horizon_idx)
        if y.ndim > 1:
            y = y[:, 0]

        all_x.append(x)
        all_y.append(y)

    X = np.concatenate(all_x, axis=0).astype(np.float64)
    y = np.concatenate(all_y, axis=0).astype(np.int64)

    # Shift labels from {-1, 0, 1} to {0, 1, 2} for XGBoost
    if y.min() < 0:
        y = y + 1

    return X, y


def compute_normalization(X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-feature mean and std from training data only."""
    mean = np.nanmean(X_train, axis=0)
    std = np.nanstd(X_train, axis=0)
    std[std < 1e-10] = 1.0
    return mean, std


def normalize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    result = (X - mean) / std
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return result


def get_feature_names(feature_indices: Tuple[int, ...]) -> Dict[int, str]:
    """Get feature names from hft_contracts for reporting."""
    try:
        from lobanalyzer.constants import get_feature_name
        return {idx: get_feature_name(idx) for idx in feature_indices}
    except ImportError:
        return {idx: f"feature_{idx}" for idx in feature_indices}


def evaluate(model: Any, X: np.ndarray, y: np.ndarray, split_name: str) -> Dict:
    """Evaluate model and return metrics dict."""
    from sklearn.metrics import (
        accuracy_score, f1_score, confusion_matrix, classification_report,
    )

    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    f1_macro = f1_score(y, y_pred, average="macro")
    f1_per_class = f1_score(y, y_pred, average=None).tolist()
    cm = confusion_matrix(y, y_pred).tolist()

    class_names = ["Down", "Stable", "Up"]
    report = classification_report(y, y_pred, target_names=class_names, output_dict=True)

    print(f"\n  {split_name} Results:")
    print(f"    Accuracy: {acc:.4f} ({acc*100:.1f}%)")
    print(f"    F1 Macro: {f1_macro:.4f}")
    for i, name in enumerate(class_names):
        print(f"    F1 {name}: {f1_per_class[i]:.4f}")
    print(f"    Confusion Matrix:")
    for row in cm:
        print(f"      {row}")

    return {
        "split": split_name,
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_per_class": {name: float(f1_per_class[i]) for i, name in enumerate(class_names)},
        "confusion_matrix": cm,
        "classification_report": report,
        "n_samples": int(len(y)),
    }


def extract_feature_importances(
    model: Any,
    feature_indices: Tuple[int, ...],
    feature_names: Dict[int, str],
) -> List[Dict]:
    """Extract and rank feature importances from the XGBoost model."""
    importances = model.feature_importances_
    if importances is None:
        return []

    ranked = []
    for local_idx, importance in enumerate(importances):
        global_idx = feature_indices[local_idx] if local_idx < len(feature_indices) else local_idx
        ranked.append({
            "rank": 0,
            "index": int(global_idx),
            "name": feature_names.get(global_idx, f"feature_{global_idx}"),
            "importance": float(importance),
        })

    ranked.sort(key=lambda x: x["importance"], reverse=True)
    for i, item in enumerate(ranked):
        item["rank"] = i + 1

    return ranked


def main():
    parser = argparse.ArgumentParser(description="XGBoost Baseline Training")
    parser.add_argument("--config", type=Path, help="YAML config file")
    parser.add_argument("--data-dir", type=Path, help="Data directory (overrides config)")
    parser.add_argument("--horizon-idx", type=int, default=None, help="Horizon index (overrides config)")
    parser.add_argument("--feature-preset", type=str, default=None, help="Feature preset name")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory")
    args = parser.parse_args()

    # Load config (CLI args override YAML)
    cfg = {}
    if args.config and args.config.exists():
        cfg = load_config(args.config)

    data_dir = args.data_dir or Path(cfg.get("data", {}).get("data_dir", "../data/exports/nvda_xnas_128feat"))
    horizon_idx = args.horizon_idx if args.horizon_idx is not None else cfg.get("data", {}).get("horizon_idx", 3)
    feature_preset = args.feature_preset or cfg.get("data", {}).get("feature_preset", "analysis_ready_128")
    output_dir = args.output_dir or Path(cfg.get("output", {}).get("output_dir", "outputs/xgboost_baseline"))

    model_cfg = cfg.get("model", {})
    n_estimators = model_cfg.get("n_estimators", 500)
    max_depth = model_cfg.get("max_depth", 6)
    learning_rate = model_cfg.get("learning_rate", 0.05)
    subsample = model_cfg.get("subsample", 0.8)
    colsample_bytree = model_cfg.get("colsample_bytree", 0.8)
    early_stopping_rounds = model_cfg.get("early_stopping_rounds", 50)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve feature preset
    if feature_preset in FEATURE_PRESETS:
        feature_indices = FEATURE_PRESETS[feature_preset]
    else:
        feature_indices = tuple(range(128))
        print(f"  Warning: preset '{feature_preset}' not found, using all 128 features")

    feature_names = get_feature_names(feature_indices)

    print("=" * 70)
    print("  XGBOOST BASELINE TRAINING")
    print("=" * 70)
    print(f"  Data:       {data_dir}")
    print(f"  Horizon:    index {horizon_idx}")
    print(f"  Preset:     {feature_preset} ({len(feature_indices)} features)")
    print(f"  XGBoost:    n_estimators={n_estimators}, max_depth={max_depth}, lr={learning_rate}")
    print(f"  Output:     {output_dir}")
    print()

    # Load data
    t0 = time.time()
    print("  Loading train data...")
    train_days = load_split_data(data_dir, split="train", validate=False)
    print(f"    {len(train_days)} days loaded")

    print("  Loading val data...")
    val_days = load_split_data(data_dir, split="val", validate=False)
    print(f"    {len(val_days)} days loaded")

    print("  Loading test data...")
    test_days = load_split_data(data_dir, split="test", validate=False)
    print(f"    {len(test_days)} days loaded")

    # Flatten and select features
    print("\n  Flattening sequences (last timestep)...")
    X_train, y_train = flatten_days(train_days, horizon_idx, feature_indices)
    X_val, y_val = flatten_days(val_days, horizon_idx, feature_indices)
    X_test, y_test = flatten_days(test_days, horizon_idx, feature_indices)

    print(f"    Train: {X_train.shape} → {len(y_train)} samples")
    print(f"    Val:   {X_val.shape} → {len(y_val)} samples")
    print(f"    Test:  {X_test.shape} → {len(y_test)} samples")

    # Normalize using train stats only
    print("\n  Computing normalization from train data...")
    mean, std = compute_normalization(X_train)
    X_train = normalize(X_train, mean, std)
    X_val = normalize(X_val, mean, std)
    X_test = normalize(X_test, mean, std)

    # Label distribution
    for name, y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        unique, counts = np.unique(y, return_counts=True)
        dist = {int(u): int(c) for u, c in zip(unique, counts)}
        print(f"    {name} labels: {dist}")

    load_time = time.time() - t0
    print(f"\n  Data loaded in {load_time:.1f}s")

    # Create and train XGBoost model
    print("\n  Training XGBoost...")
    t1 = time.time()

    from lobmodels import XGBoostLOB, XGBoostLOBConfig

    xgb_config = XGBoostLOBConfig(
        num_features=X_train.shape[1],
        num_classes=3,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        early_stopping_rounds=early_stopping_rounds,
    )

    model = XGBoostLOB(xgb_config)
    fit_result = model.fit(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        verbose=True,
    )

    train_time = time.time() - t1
    print(f"\n  Training complete in {train_time:.1f}s")
    print(f"    Best iteration: {fit_result.best_iteration}")
    print(f"    Best score: {fit_result.best_score:.6f}")

    # Evaluate
    print("\n  Evaluating...")
    train_metrics = evaluate(model, X_train, y_train, "Train")
    val_metrics = evaluate(model, X_val, y_val, "Validation")
    test_metrics = evaluate(model, X_test, y_test, "Test")

    # Feature importances
    print("\n  Extracting feature importances...")
    importances = extract_feature_importances(model, feature_indices, feature_names)

    print(f"\n  Top 20 Features by Importance:")
    for item in importances[:20]:
        print(f"    [{item['index']:3d}] {item['name']:<30s} importance={item['importance']:.4f}")

    # Save results
    results = {
        "experiment": cfg.get("experiment", {"name": "XGBoost Baseline"}),
        "config": {
            "data_dir": str(data_dir),
            "horizon_idx": horizon_idx,
            "feature_preset": feature_preset,
            "n_features": int(X_train.shape[1]),
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "early_stopping_rounds": early_stopping_rounds,
        },
        "training": {
            "n_train": int(len(y_train)),
            "n_val": int(len(y_val)),
            "n_test": int(len(y_test)),
            "best_iteration": fit_result.best_iteration,
            "best_score": float(fit_result.best_score),
            "train_time_seconds": float(train_time),
            "load_time_seconds": float(load_time),
        },
        "metrics": {
            "train": train_metrics,
            "validation": val_metrics,
            "test": test_metrics,
        },
        "feature_importances": importances,
        "provenance": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "script": "train_xgboost_baseline.py",
            "xgboost_version": __import__("xgboost").__version__,
        },
    }

    results_path = output_dir / "xgboost_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")

    # Save model
    model_path = output_dir / "model.json"
    model.save(str(model_path))
    print(f"  Model saved to {model_path}")

    # Save normalization stats
    norm_path = output_dir / "normalization_stats.json"
    with open(norm_path, "w") as f:
        json.dump({
            "mean": mean.tolist(),
            "std": std.tolist(),
            "feature_indices": list(feature_indices),
        }, f)
    print(f"  Normalization stats saved to {norm_path}")

    print(f"\n{'=' * 70}")
    print(f"  EXPERIMENT COMPLETE")
    print(f"  Test Accuracy: {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.1f}%)")
    print(f"  Test F1 Macro: {test_metrics['f1_macro']:.4f}")
    print(f"  Total Time: {load_time + train_time:.1f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
