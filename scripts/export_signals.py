#!/usr/bin/env python3
"""
Unified signal export script for all model types.

Replaces the 3 separate export scripts:
  - export_hmhp_signals.py (classification)
  - export_regression_signals.py (HMHP-R multi-horizon)
  - export_tlob_regression_signals.py (single-horizon regression)

Uses the Trainer pipeline for normalization and feature selection,
ensuring consistency with the training data flow. Extracts raw
spread/price from disk for backtester compatibility.

Usage:
    python scripts/export_signals.py \\
        --config configs/experiments/my_experiment.yaml \\
        --checkpoint outputs/experiments/my_experiment/checkpoints/best.pt \\
        --split test \\
        --output-dir outputs/experiments/my_experiment/signals/test

    # With variance calibration (regression only):
    python scripts/export_signals.py \\
        --config ... --checkpoint ... --calibrate variance_match

Produces backtester-compatible signal files:
    prices.npy, spreads.npy, predictions.npy (classification),
    predicted_returns.npy (regression), signal_metadata.json, etc.
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("export_signals")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export model signals for backtesting.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", required=True, type=str,
        help="Path to experiment YAML config file.",
    )
    parser.add_argument(
        "--checkpoint", required=True, type=str,
        help="Path to model checkpoint .pt file.",
    )
    parser.add_argument(
        "--split", default="test", choices=["val", "test"],
        help="Data split to export (default: test).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: <experiment>/signals/<split>/).",
    )
    parser.add_argument(
        "--calibrate", default="none", choices=["none", "variance_match"],
        help="Calibration method for regression outputs (default: none).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override batch size from config (for memory management).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    from lobtrainer.config import ExperimentConfig
    from lobtrainer.training.trainer import Trainer
    from lobtrainer.export import SignalExporter

    # Load config
    config = ExperimentConfig.from_yaml(args.config)

    # Override batch size if specified
    if args.batch_size is not None:
        config.train.batch_size = args.batch_size

    # Setup trainer (creates model, normalizer, dataloaders)
    trainer = Trainer(config, callbacks=[])
    trainer.setup()

    # Load checkpoint
    trainer.load_checkpoint(args.checkpoint, load_optimizer=False)
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    # Determine output directory
    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir)

    # Export
    exporter = SignalExporter(trainer, calibration=args.calibrate)
    result = exporter.export(split=args.split, output_dir=output_dir)

    # Summary
    print(f"\n{'='*60}")
    print(f"Signal Export Complete")
    print(f"{'='*60}")
    print(f"  Split:       {args.split}")
    print(f"  Samples:     {result.n_samples:,}")
    print(f"  Signal type: {result.signal_type}")
    print(f"  Output:      {result.output_dir}")
    print(f"  Files:       {', '.join(result.files_written)}")
    if result.metadata.get("metrics"):
        m = result.metadata["metrics"]
        print(f"  Metrics:     R²={m.get('r2', 'N/A'):.4f}, IC={m.get('ic', 'N/A'):.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
