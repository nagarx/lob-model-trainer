#!/usr/bin/env python3
"""
DEPRECATED: Use export_signals.py instead.

  python scripts/export_signals.py --config <yaml> --checkpoint <pt>

Export HMHP-R regression signals for backtesting.

Runs the trained regression model on the test split and exports
continuous bps return predictions alongside ground truth.

Output files (in --output-dir):
    predicted_returns.npy     [N, H] float64 — predicted bps returns per horizon
    regression_labels.npy     [N, H] float64 — actual bps returns per horizon
    spreads.npy               [N]   float64 — spread_bps from features
    prices.npy                [N]   float64 — mid_price from features
    signal_metadata.json      provenance, metrics, model info

Usage:
    python scripts/export_regression_signals.py \\
        --checkpoint outputs/experiments/nvda_hmhp_regressor_h60/checkpoints/best.pt \\
        --data-dir ../data/exports/nvda_xnas_128feat_regression \\
        --output-dir outputs/experiments/nvda_hmhp_regressor_h60/signals/test/
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


MID_PRICE_INDEX = 40
SPREAD_BPS_INDEX = 42
STRIDE = 10


def main():
    parser = argparse.ArgumentParser(description="Export Regression Signals")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--feature-preset", type=str, default="short_term_40")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  REGRESSION SIGNAL EXPORT")
    print("=" * 60)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Data: {args.data_dir}")
    print(f"  Split: {args.split}")

    from lobtrainer.data import load_split_data, LOBSequenceDataset
    from lobtrainer.data.feature_selector import create_feature_selector
    from lobtrainer.data.normalization import HybridNormalizer

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    model_state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", {}))

    from lobmodels.models.hmhp_regressor import create_hmhp_regressor
    model_config = config.get("model", {})
    model = create_hmhp_regressor(
        num_features=model_config.get("input_size", 40),
        horizons=model_config.get("hmhp_horizons", [10, 60, 300]),
        encoder_type=model_config.get("hmhp_encoder_type", "tlob"),
        hidden_dim=model_config.get("hmhp_encoder_hidden_dim", 64),
        num_encoder_layers=model_config.get("hmhp_num_encoder_layers", 2),
        decoder_hidden_dim=model_config.get("hmhp_decoder_hidden_dim", 32),
        state_dim=model_config.get("hmhp_state_dim", 32),
    )
    model.load_state_dict(model_state)
    model.eval()
    print(f"  Model: {model.name}, {model.num_parameters:,} params")

    days = load_split_data(args.data_dir, args.split, validate=True)
    print(f"  Loaded {len(days)} days")

    selector = create_feature_selector(args.feature_preset, 128)
    feature_indices = selector.indices

    norm_path = Path(args.data_dir) / "hybrid_normalization_stats.json"
    normalizer = HybridNormalizer.load(str(norm_path)) if norm_path.exists() else None

    all_preds = []
    all_labels = []
    all_spreads = []
    all_prices = []

    horizons = model_config.get("hmhp_horizons", [10, 60, 300])

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)

    with torch.no_grad():
        for day in days:
            n = day.num_sequences
            for start in range(0, n, args.batch_size):
                end_idx = min(start + args.batch_size, n)
                batch_seqs = day.sequences[start:end_idx].copy()

                if normalizer:
                    for i in range(len(batch_seqs)):
                        batch_seqs[i] = normalizer.transform(batch_seqs[i])

                batch_seqs = batch_seqs[:, :, feature_indices]
                x = torch.from_numpy(batch_seqs).float().to(device)

                output = model(x)

                preds_per_h = []
                for h in horizons:
                    preds_per_h.append(output.horizon_predictions[h].squeeze(-1).cpu().numpy())
                all_preds.append(np.stack(preds_per_h, axis=-1))

                raw_seqs = day.sequences[start:end_idx]
                all_spreads.append(raw_seqs[:, -1, SPREAD_BPS_INDEX].astype(np.float64))
                all_prices.append(raw_seqs[:, -1, MID_PRICE_INDEX].astype(np.float64))

                reg_labels_path = Path(args.data_dir) / args.split / f"{day.date}_regression_labels.npy"
                if reg_labels_path.exists():
                    reg_labels = np.load(reg_labels_path)
                    all_labels.append(reg_labels[start:end_idx])

    predicted_returns = np.concatenate(all_preds, axis=0)
    spreads = np.concatenate(all_spreads, axis=0)
    prices = np.concatenate(all_prices, axis=0)

    np.save(output_dir / "predicted_returns.npy", predicted_returns)
    np.save(output_dir / "spreads.npy", spreads)
    np.save(output_dir / "prices.npy", prices)

    if all_labels:
        regression_labels = np.concatenate(all_labels, axis=0)
        np.save(output_dir / "regression_labels.npy", regression_labels)

    metadata = {
        "model_type": "hmhp_regression",
        "horizons": horizons,
        "total_samples": len(predicted_returns),
        "split": args.split,
        "checkpoint": args.checkpoint,
        "feature_preset": args.feature_preset,
    }
    with open(output_dir / "signal_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Exported {len(predicted_returns):,} samples to {output_dir}")
    for i, h in enumerate(horizons):
        p = predicted_returns[:, i]
        print(f"  H{h}: mean={p.mean():+.2f} bps, std={p.std():.2f}, "
              f"|mean|={np.abs(p).mean():.2f} bps")


if __name__ == "__main__":
    main()
