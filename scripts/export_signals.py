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

    import json

    from lobtrainer.config import ExperimentConfig
    from lobtrainer.training import create_trainer

    # Load config
    config = ExperimentConfig.from_yaml(args.config)

    # Phase A.5.3f.1 hardening (2026-04-24): TrainConfig is a frozen Pydantic
    # BaseModel (A.5.3e). Phase A.5.3i (2026-04-24 KEYSTONE):
    # ExperimentConfig itself is now also frozen. Two-layer mutation:
    # (a) build the new TrainConfig via inner model_copy (re-fires TrainConfig
    #     validators), (b) swap the new TrainConfig into the outer
    #     ExperimentConfig via outer model_copy (re-fires ExperimentConfig
    #     validators including T13 auto-derive).
    if args.batch_size is not None:
        _new_train = config.train.model_copy(update={"batch_size": args.batch_size})
        config = config.model_copy(update={"train": _new_train})

    # Phase Q.6.5.B (2026-05-04 night): F-16 closure — use create_trainer
    # framework dispatch instead of direct Trainer() instantiation. Pre-fix
    # this site silently broke for sklearn-registered models (temporal_ridge,
    # temporal_gradboost) — TemporalRidgeConfig.__init__ would raise on
    # `features` kwarg passed by params. Now sklearn dispatch returns
    # SimpleModelTrainer.from_config(config) and the rest of this script
    # operates polymorphically through the BaseTrainer Protocol.
    trainer = create_trainer(config, callbacks=[])
    trainer.setup()

    # Phase Q.6.5.B (2026-05-04 night): unified load_checkpoint signature.
    # PyTorch path uses load_optimizer=False to skip optimizer state restore
    # (signal export is inference-only). Sklearn path documents this kwarg
    # as a no-op (no optimizer in the pickle). Closes N-6 signature drift.
    trainer.load_checkpoint(args.checkpoint, load_optimizer=False)
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    # Determine output directory
    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir)

    # Phase Q.6.5.B (2026-05-04 night): export via BaseTrainer Protocol method.
    # Both Trainer (PyTorch — delegates to SignalExporter) and
    # SimpleModelTrainer (sklearn — emits in-memory predictions + Phase X.1.A
    # CompatibilityContract block via build_signal_metadata) satisfy the
    # method, so this single polymorphic call works for ALL registered model
    # types. The previous direct SignalExporter usage was PyTorch-only.
    signal_dir = trainer.export_signals(
        args.split,
        output_dir=output_dir,
        calibration=args.calibrate,
    )

    # Summary — read from signal_metadata.json + directory listing.
    # Pre-Q.6.5.B used the SignalExporter ExportResult dataclass directly;
    # post-Q.6.5.B the Protocol method returns just the Path so we reconstruct
    # the summary by reading the produced metadata. Same UX, polymorphic.
    metadata_path = signal_dir / "signal_metadata.json"
    metadata: dict = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    files_written = sorted(p.name for p in signal_dir.iterdir() if p.is_file())

    print(f"\n{'='*60}")
    print(f"Signal Export Complete")
    print(f"{'='*60}")
    print(f"  Split:       {args.split}")
    print(f"  Samples:     {metadata.get('total_samples', 'N/A'):,}" if isinstance(metadata.get('total_samples'), int) else f"  Samples:     N/A")
    print(f"  Signal type: {metadata.get('signal_type', 'N/A')}")
    print(f"  Output:      {signal_dir}")
    print(f"  Files:       {', '.join(files_written)}")
    if metadata.get("compatibility_fingerprint"):
        # Phase II + Phase Q.6.5.A surface — confirm the fingerprint is present
        # so operators see Phase Y composability is intact.
        print(f"  Compat FP:   {metadata['compatibility_fingerprint'][:16]}...")
    if metadata.get("metrics"):
        m = metadata["metrics"]
        r2 = m.get('r2')
        ic = m.get('ic')
        if r2 is not None and ic is not None:
            print(f"  Metrics:     R²={r2:.4f}, IC={ic:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
