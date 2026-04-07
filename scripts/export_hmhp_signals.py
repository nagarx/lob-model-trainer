#!/usr/bin/env python3
"""
DEPRECATED: Use export_signals.py instead.

  python scripts/export_signals.py --config <yaml> --checkpoint <pt>

Export HMHP model signals for backtesting.

Runs inference on a data split and saves per-sample arrays:
  predictions.npy     [N] argmax class (0=Down, 1=Stable, 2=Up)
  agreement_ratio.npy [N] float in [0.333, 1.0]
  confirmation_score.npy [N] float in [0, 0.667]
  spreads.npy         [N] raw spread_bps from feature index 42
  prices.npy          [N] raw mid prices from feature index 40
  labels.npy          [N] ground truth for primary horizon
  signal_metadata.json  provenance info

The backtester (lob-backtester) consumes these arrays without needing
torch or lobmodels -- separation of inference and backtesting.

Usage:
    python scripts/export_hmhp_signals.py \\
        --experiment outputs/experiments/nvda_hmhp_40feat_h10 \\
        --split test \\
        --output outputs/experiments/nvda_hmhp_40feat_h10/signals/test/
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'lob-models' / 'src'))

from lobtrainer.config.schema import ExperimentConfig
from lobtrainer.training.trainer import Trainer

try:
    from hft_contracts import SIGNAL_SPREAD_FEATURE_INDEX, SIGNAL_PRICE_FEATURE_INDEX
    SPREAD_BPS_INDEX = SIGNAL_SPREAD_FEATURE_INDEX
    MID_PRICE_INDEX = SIGNAL_PRICE_FEATURE_INDEX
except ImportError:
    SPREAD_BPS_INDEX = 42
    MID_PRICE_INDEX = 40


@torch.no_grad()
def export_signals(
    trainer: Trainer,
    loader: torch.utils.data.DataLoader,
    horizons: list,
    data_dir: str,
    split: str,
    output_dir: Path,
):
    """Run inference and save all signal arrays to output_dir."""

    model = trainer.model
    model.eval()
    device = trainer.device
    first_h = horizons[0]

    all_preds = []
    all_labels = []
    all_agreement = []
    all_confirmation = []

    for features, label_dict in loader:
        features = features.to(device)
        output = model(features)

        all_preds.append(output.logits.argmax(dim=1).cpu().numpy())
        all_labels.append(label_dict[first_h].numpy())
        all_agreement.append(output.agreement.squeeze(-1).cpu().numpy())
        all_confirmation.append(output.confidence.squeeze(-1).cpu().numpy())

    predictions = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    agreement = np.concatenate(all_agreement)
    confirmation = np.concatenate(all_confirmation)

    split_dir = Path(data_dir) / split
    day_files = sorted(split_dir.glob("*_sequences.npy"))

    raw_spreads = []
    raw_prices = []
    for day_file in day_files:
        seqs = np.load(day_file, mmap_mode='r')
        raw_spreads.append(seqs[:, -1, SPREAD_BPS_INDEX].copy())
        raw_prices.append(seqs[:, -1, MID_PRICE_INDEX].copy())

    spreads = np.concatenate(raw_spreads)[:len(predictions)]
    prices = np.concatenate(raw_prices)[:len(predictions)]

    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "predictions.npy", predictions)
    np.save(output_dir / "labels.npy", labels)
    np.save(output_dir / "agreement_ratio.npy", agreement)
    np.save(output_dir / "confirmation_score.npy", confirmation)
    np.save(output_dir / "spreads.npy", spreads)
    np.save(output_dir / "prices.npy", prices)

    n = len(predictions)
    dir_mask = np.isin(predictions, [0, 2])
    agree_mask = agreement == 1.0

    metadata = {
        "exported_at": datetime.now().isoformat(),
        "split": split,
        "horizons": horizons,
        "primary_horizon": first_h,
        "total_samples": int(n),
        "data_dir": str(data_dir),
        "predictions_distribution": {
            "Down": int((predictions == 0).sum()),
            "Stable": int((predictions == 1).sum()),
            "Up": int((predictions == 2).sum()),
        },
        "agreement_distribution": {
            "full_agreement": int(agree_mask.sum()),
            "partial": int((~agree_mask).sum()),
        },
        "confirmation_percentiles": {
            "p25": float(np.percentile(confirmation, 25)),
            "p50": float(np.percentile(confirmation, 50)),
            "p75": float(np.percentile(confirmation, 75)),
            "p99": float(np.percentile(confirmation, 99)),
        },
        "spread_percentiles_bps": {
            "p25": float(np.percentile(spreads, 25)),
            "p50": float(np.percentile(spreads, 50)),
            "p90": float(np.percentile(spreads, 90)),
        },
        "directional_rate": float(dir_mask.mean()),
    }

    with open(output_dir / "signal_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Exported {n:,} samples to {output_dir}")
    print(f"  Predictions: Down={metadata['predictions_distribution']['Down']:,}, "
          f"Stable={metadata['predictions_distribution']['Stable']:,}, "
          f"Up={metadata['predictions_distribution']['Up']:,}")
    print(f"  Full agreement: {metadata['agreement_distribution']['full_agreement']:,} "
          f"({100 * metadata['agreement_distribution']['full_agreement'] / n:.1f}%)")
    print(f"  Directional rate: {metadata['directional_rate']:.1%}")


def main():
    parser = argparse.ArgumentParser(description="Export HMHP signals for backtesting")
    parser.add_argument("--experiment", type=str, required=True,
                        help="Path to experiment output directory")
    parser.add_argument("--split", type=str, default="test",
                        help="Data split to export (val/test)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: <experiment>/signals/<split>/)")
    args = parser.parse_args()

    exp_dir = Path(args.experiment)
    config_path = exp_dir / "config.yaml"
    checkpoint_path = exp_dir / "checkpoints" / "best.pt"

    if not checkpoint_path.exists():
        print(f"ERROR: No checkpoint at {checkpoint_path}")
        sys.exit(1)

    print("=" * 60)
    print("  HMHP Signal Export for Backtesting")
    print("=" * 60)

    config = ExperimentConfig.from_yaml(str(config_path))
    horizons = config.model.hmhp_horizons

    output_dir = Path(args.output) if args.output else exp_dir / "signals" / args.split
    print(f"  Experiment: {exp_dir}")
    print(f"  Split: {args.split}")
    print(f"  Output: {output_dir}")

    trainer = Trainer(config)
    trainer.setup()
    trainer.load_checkpoint(checkpoint_path, load_optimizer=False)

    loader = getattr(trainer, f"_{args.split}_loader", None)
    if loader is None:
        print(f"ERROR: No loader for split '{args.split}'")
        sys.exit(1)

    export_signals(
        trainer, loader, horizons,
        config.data.data_dir, args.split, output_dir,
    )

    print("  Done.")


if __name__ == "__main__":
    main()
