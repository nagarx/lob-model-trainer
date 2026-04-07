#!/usr/bin/env python3
"""
DEPRECATED: Use export_signals.py instead.

  python scripts/export_signals.py --config <yaml> --checkpoint <pt> --calibrate variance_match

Export TLOB regression signals for backtesting.

Runs the trained TLOB regression model on the test split and exports
continuous bps return predictions alongside ground truth.

Output files (contract: pipeline_contract.toml [signals.regression]):
    predicted_returns.npy     [N] float64 — predicted bps returns (H10)
    regression_labels.npy     [N] float64 — actual bps returns (H10)
    spreads.npy               [N] float64 — spread_bps at last timestep
    prices.npy                [N] float64 — mid_price at last timestep
    signal_metadata.json      provenance, metrics, model info

Usage:
    python scripts/export_tlob_regression_signals.py \
        --config configs/experiments/nvda_tlob_128feat_regression_h10.yaml \
        --checkpoint outputs/experiments/nvda_tlob_128feat_regression_h10/checkpoints/best.pt \
        --output-dir outputs/experiments/nvda_tlob_128feat_regression_h10/signals/test/
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

MID_PRICE_INDEX = 40
SPREAD_BPS_INDEX = 42


def main():
    parser = argparse.ArgumentParser(description="Export TLOB Regression Signals")
    parser.add_argument("--config", type=str, required=True, help="Experiment YAML config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint .pt")
    parser.add_argument("--output-dir", type=str, required=True, help="Signal output directory")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--calibrate", type=str, default="none",
        choices=["none", "variance_match"],
        help="Post-hoc prediction calibration method. 'variance_match' rescales "
             "predictions to match target return distribution variance (preserves IC).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  TLOB REGRESSION SIGNAL EXPORT")
    print("=" * 60)

    from lobtrainer.config.schema import ExperimentConfig
    from lobtrainer.training.trainer import Trainer
    from lobtrainer.training.regression_metrics import compute_all_regression_metrics

    config = ExperimentConfig.from_yaml(args.config)
    trainer = Trainer(config)
    trainer.setup()

    ckpt = torch.load(args.checkpoint, weights_only=False, map_location=trainer.device)
    trainer.model.load_state_dict(ckpt["model_state_dict"])
    trainer.model.eval()
    print(f"  Model: {trainer.model.name}, {sum(p.numel() for p in trainer.model.parameters()):,} params")
    print(f"  Device: {trainer.device}")
    print(f"  Checkpoint epoch: {ckpt.get('epoch', '?')}")

    if args.split == "test":
        loader = trainer._test_loader
    elif args.split == "val":
        loader = trainer._val_loader
    else:
        loader = trainer._train_loader

    all_preds, all_targets = [], []
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(trainer.device)
            outputs = trainer.model(features)
            all_preds.append(outputs.predictions.cpu().numpy())
            all_targets.append(labels.numpy())

    predicted_returns = np.concatenate(all_preds).astype(np.float64)
    regression_labels = np.concatenate(all_targets).astype(np.float64)
    n_total = len(predicted_returns)
    print(f"  Predictions: {n_total:,} samples")

    data_dir = Path(config.data.data_dir)
    split_dir = data_dir / args.split
    meta_files = sorted(split_dir.glob("*_metadata.json"))

    all_spreads, all_prices = [], []
    for mf in meta_files:
        with open(mf) as f:
            m = json.load(f)
        day = m["day"]
        raw_seqs = np.load(split_dir / f"{day}_sequences.npy", mmap_mode="r")
        all_spreads.append(raw_seqs[:, -1, SPREAD_BPS_INDEX].astype(np.float64))
        all_prices.append(raw_seqs[:, -1, MID_PRICE_INDEX].astype(np.float64))

    spreads = np.concatenate(all_spreads)
    prices = np.concatenate(all_prices)

    assert len(spreads) == n_total, f"Spread length {len(spreads)} != predictions {n_total}"
    assert len(prices) == n_total, f"Price length {len(prices)} != predictions {n_total}"

    np.save(output_dir / "predicted_returns.npy", predicted_returns)
    np.save(output_dir / "regression_labels.npy", regression_labels)
    np.save(output_dir / "spreads.npy", spreads)
    np.save(output_dir / "prices.npy", prices)

    # --- Calibration ---
    calibration_stats = None
    if args.calibrate != "none":
        from lobtrainer.calibration import calibrate_variance, VarianceCalibrationConfig
        cal_config = VarianceCalibrationConfig(
            method=args.calibrate,
            compute_from_labels=True,
        )
        cal_result = calibrate_variance(predicted_returns, regression_labels, cal_config)
        np.save(output_dir / "calibrated_returns.npy", cal_result.calibrated)
        calibration_stats = cal_result.to_dict()
        print(f"\n  Calibration ({args.calibrate}):")
        print(f"    Scale factor: {cal_result.scale_factor:.4f}")
        print(f"    Raw std:      {cal_result.pred_std:.4f} bps")
        print(f"    Calibrated:   {float(np.std(cal_result.calibrated)):.4f} bps")
        print(f"    Target std:   {cal_result.target_std:.4f} bps")

    metrics = compute_all_regression_metrics(regression_labels, predicted_returns)

    metadata = {
        "model_type": "tlob_regression",
        "model_name": trainer.model.name,
        "parameters": sum(p.numel() for p in trainer.model.parameters()),
        "horizon": f"H{[10, 60, 300][config.data.horizon_idx]}",
        "horizon_idx": config.data.horizon_idx,
        "total_samples": n_total,
        "split": args.split,
        "checkpoint": args.checkpoint,
        "config": args.config,
        "metrics": {k: round(v, 6) for k, v in metrics.items()},
        "prediction_stats": {
            "mean": round(float(predicted_returns.mean()), 4),
            "std": round(float(predicted_returns.std()), 4),
            "min": round(float(predicted_returns.min()), 2),
            "max": round(float(predicted_returns.max()), 2),
        },
        "spread_stats": {
            "mean": round(float(spreads.mean()), 4),
            "median": round(float(np.median(spreads)), 4),
            "p90": round(float(np.percentile(spreads, 90)), 4),
        },
        "calibration": calibration_stats,
    }
    with open(output_dir / "signal_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Exported to {output_dir}")
    print(f"  predicted_returns.npy  [{n_total:,}] mean={predicted_returns.mean():+.3f}, std={predicted_returns.std():.3f}")
    print(f"  regression_labels.npy  [{n_total:,}] mean={regression_labels.mean():+.3f}, std={regression_labels.std():.3f}")
    print(f"  spreads.npy            [{n_total:,}] mean={spreads.mean():.3f} bps")
    print(f"  prices.npy             [{n_total:,}] mean=${prices.mean():.2f}")
    print(f"\n  Metrics: R²={metrics['r2']:.4f}, IC={metrics['ic']:.4f}, DA={metrics['directional_accuracy']:.4f}")


if __name__ == "__main__":
    main()
