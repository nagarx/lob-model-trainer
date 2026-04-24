#!/usr/bin/env python3
"""
Training script for LOB price prediction models.

This script provides a command-line interface for training models
using configuration files.

Usage:
    # Train with config file
    python scripts/train.py --config configs/baseline_lstm.yaml
    
    # Override specific parameters
    python scripts/train.py --config configs/baseline_lstm.yaml \\
        --epochs 50 --batch-size 128 --output-dir outputs/experiment1
    
    # Resume from checkpoint
    python scripts/train.py --config configs/baseline_lstm.yaml \\
        --resume outputs/checkpoints/best.pt

Design principles (RULE.md):
- Configuration-driven: All parameters via config file or CLI
- Reproducible: Explicit seed management
- Comprehensive logging: Track all experiments
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

# Add parent directory to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Phase 1.4: emit a deprecation warning when invoked directly (not via hft-ops).
# The warning is suppressed when HFT_OPS_ORCHESTRATED=1 is set in the env
# (hft-ops sets this before invoking trainer subprocesses).
sys.path.insert(0, str(Path(__file__).parent))  # for _hft_ops_compat
from _hft_ops_compat import warn_if_not_orchestrated
warn_if_not_orchestrated(
    script_name="train.py",
    suggestion=(
        "Use 'hft-ops run <manifest>' to launch experiments. The manifest "
        "should reference this trainer config via stages.training.config "
        "(legacy) or stages.training.trainer_config (inline)."
    ),
)

from lobtrainer import create_trainer, set_seed
from lobtrainer.config import load_config, ExperimentConfig, save_config
from lobtrainer.training import (
    EarlyStopping,
    ModelCheckpoint,
    MetricLogger,
    ProgressCallback,
)


def setup_logging(output_dir: Path, log_level: str = "INFO") -> None:
    """Configure logging to console and file."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_{timestamp}.log"
    
    # Configure logging
    level = getattr(logging, log_level.upper())
    
    # Create handlers
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(
        logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', '%H:%M:%S')
    )
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)  # Always log everything to file
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    )
    
    # Configure root logger
    logging.root.setLevel(logging.DEBUG)
    logging.root.addHandler(console_handler)
    logging.root.addHandler(file_handler)
    
    logging.info(f"Logging to {log_file}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train LOB price prediction models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic training
    python scripts/train.py --config configs/baseline_lstm.yaml
    
    # With custom output directory
    python scripts/train.py --config configs/baseline_lstm.yaml \\
        --output-dir outputs/my_experiment
    
    # Resume from checkpoint
    python scripts/train.py --config configs/baseline_lstm.yaml \\
        --resume outputs/checkpoints/best.pt
        """,
    )
    
    # Required arguments
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to experiment configuration file (YAML or JSON)",
    )
    
    # Optional overrides
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--learning-rate", "--lr",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed",
    )
    
    # Checkpoint options
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    
    # Logging options
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar",
    )
    
    # Evaluation options
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation (requires --resume)",
    )
    
    return parser.parse_args()


def _dump_test_metrics(
    result: Any,
    output_dir: Path,
    prefix: str = "test_",
) -> Optional[Path]:
    """Persist final test-split evaluation metrics to ``test_metrics.json``.

    Phase 7 Stage 7.4 Round 4 item #6 (BLOCKER). The PyTorch ``Trainer`` flow
    calls ``trainer.evaluate('test')`` for every run but only logged the
    summary — the metrics were never persisted. Consequence: the 7 ``test_*``
    keys added to ``ExperimentRecord.index_entry()`` whitelist in Round 1
    were silently dead for every TLOB / HMHP / HMHP-R run. Round 4 closes
    this gap by mirroring the ``SimpleModelTrainer.save`` convention
    (simple_trainer.py:223-225): a flat ``{test_<key>: float}`` JSON file
    consumed downstream by ``hft_ops.stages.training._capture_training_metrics``.

    Handles BOTH return shapes of ``trainer.evaluate``:
      - ``ClassificationMetrics`` → flattened via ``.to_dict()``.
      - Regression / HMHP-R ``Dict[str, Any]`` → used directly.

    Non-scalar values (confusion matrices, per-class dicts, numpy arrays)
    are dropped — the whitelist at ``ExperimentRecord.index_entry`` only
    surfaces scalars and non-scalars would break the flat-key contract.
    """
    if hasattr(result, "to_dict") and callable(result.to_dict):
        flat = result.to_dict()
    elif isinstance(result, dict):
        flat = result
    else:
        return None

    prefixed: Dict[str, float] = {}
    for key, value in flat.items():
        if isinstance(value, bool) or value is None:
            continue
        try:
            prefixed[f"{prefix}{key}"] = float(value)
        except (TypeError, ValueError):
            # Drop non-scalars (arrays, dicts, strings) — only scalars land
            # in the index_entry() whitelist projection.
            continue

    if not prefixed:
        return None

    output_path = output_dir / "test_metrics.json"
    with open(output_path, "w") as f:
        json.dump(prefixed, f, indent=2, sort_keys=True)
    return output_path


def _safe_summary(metrics: Any) -> str:
    """Best-effort string rendering of evaluate() result.

    ``ClassificationMetrics`` has ``.summary()``; regression strategies
    return ``Dict[str, Any]`` which lacks it. Without the guard, the log
    line at the end of ``scripts/train.py`` raises ``AttributeError`` on
    every regression run (the enclosing ``except ValueError`` does NOT
    catch it). Guard lets the persist step land regardless.
    """
    if hasattr(metrics, "summary") and callable(metrics.summary):
        try:
            return metrics.summary()
        except Exception:  # pragma: no cover — defensive
            pass
    if isinstance(metrics, dict):
        return "\n".join(
            f"  {k}: {v:.6f}" if isinstance(v, (int, float))
            and not isinstance(v, bool)
            else f"  {k}: {v}"
            for k, v in sorted(metrics.items())
        )
    return str(metrics)


def apply_overrides(config: ExperimentConfig, args) -> ExperimentConfig:
    """Apply command-line overrides to configuration.

    Phase A.5.3f.1 hardening (2026-04-24): TrainConfig is a frozen Pydantic
    BaseModel (A.5.3e commit 7c91170). Direct field assignment raises
    ValidationError. Every ``python scripts/train.py --epochs N`` would
    have crashed before this fix landed. Accumulate all train overrides
    into a dict and apply via SafeBaseModel.model_copy(update=...) which
    re-runs validators (including cross-field task↔loss compatibility).

    Phase A.5.3g (2026-04-24): DataConfig migrated to frozen Pydantic
    BaseModel. ``config.data.data_dir = ...`` now raises. Same
    model_copy(update=...) pattern applied to the single DataConfig CLI
    override (``--data-dir``).

    Phase A.5.3i (2026-04-24 KEYSTONE): ExperimentConfig itself migrated
    to frozen Pydantic BaseModel. Top-level ``config.output_dir = ...``
    also raises. Same pattern applied — ``config = config.model_copy(
    update={"output_dir": ...})`` re-fires all ExperimentConfig
    validators (including T13 auto-derive + T9 deprecation warnings).

    NOTE: ``config.data = config.data.model_copy(...)`` at the START of
    this function is still a TOP-LEVEL FIELD ASSIGNMENT on
    ExperimentConfig (``config.data = X``) which now ALSO raises under
    frozen=True. Refactored to use the same ``config.model_copy`` idiom
    with an outer accumulator so all overrides are applied atomically.
    """
    from typing import Any, Dict

    # Accumulate all overrides at both levels; apply in one model_copy
    # per frozen layer.
    _data_overrides: Dict[str, Any] = {}
    if args.data_dir is not None:
        _data_overrides["data_dir"] = args.data_dir

    _top_overrides: Dict[str, Any] = {}
    if _data_overrides:
        # Build new DataConfig first (its own validators fire), then stash
        # the new DataConfig in the outer overrides dict for atomic
        # ExperimentConfig.model_copy below.
        _top_overrides["data"] = config.data.model_copy(update=_data_overrides)

    if args.output_dir is not None:
        _top_overrides["output_dir"] = args.output_dir

    _train_overrides: Dict[str, Any] = {}
    if args.epochs is not None:
        _train_overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        _train_overrides["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        _train_overrides["learning_rate"] = args.learning_rate
    if args.seed is not None:
        _train_overrides["seed"] = args.seed
    if _train_overrides:
        # Same two-layer pattern: build the new TrainConfig (inner
        # validators fire), stash in top-overrides dict for atomic
        # ExperimentConfig.model_copy below.
        _top_overrides["train"] = config.train.model_copy(update=_train_overrides)

    if _top_overrides:
        config = config.model_copy(update=_top_overrides)

    return config


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    config = load_config(str(config_path))
    
    # Apply command-line overrides
    config = apply_overrides(config, args)
    
    # Setup output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(output_dir, args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting experiment: {config.name}")
    logger.info(f"Output directory: {output_dir}")
    
    # Save config to output directory for reproducibility
    config_save_path = output_dir / "config.yaml"
    save_config(config, str(config_save_path))
    logger.info(f"Saved config to {config_save_path}")
    
    # Set random seed
    set_seed(config.train.seed)
    logger.info(f"Set random seed: {config.train.seed}")
    
    # Create callbacks
    callbacks = [
        EarlyStopping(
            patience=config.train.early_stopping_patience,
            metric='val_loss',
            mode='min',
            restore_best_weights=True,
        ),
        ModelCheckpoint(
            save_dir=output_dir / 'checkpoints',
            metric='val_loss',
            mode='min',
            save_best_only=True,
            max_checkpoints=3,
        ),
        MetricLogger(
            log_every_n_batches=None,  # Only log epoch end
            log_to_file=True,
            log_file=output_dir / 'training_history.json',
        ),
    ]
    
    if not args.no_progress:
        callbacks.append(ProgressCallback())
    
    # Create trainer
    trainer = create_trainer(config, callbacks=callbacks)
    
    # Resume from checkpoint if specified
    if args.resume:
        checkpoint_path = Path(args.resume)
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            sys.exit(1)
        trainer.load_checkpoint(checkpoint_path)
        logger.info(f"Resumed from checkpoint: {checkpoint_path}")
    
    # Evaluation only mode
    if args.eval_only:
        if not args.resume:
            logger.error("--eval-only requires --resume to specify model checkpoint")
            sys.exit(1)
        
        logger.info("Running evaluation only")

        # Evaluate on all splits
        for split in ['train', 'val', 'test']:
            try:
                metrics = trainer.evaluate(split)
                if split == 'test':
                    written = _dump_test_metrics(metrics, output_dir)
                    if written is not None:
                        logger.info(f"Saved test metrics to {written}")
                logger.info(f"\n{split.upper()} Results:\n{_safe_summary(metrics)}")
            except ValueError as e:
                logger.warning(f"Could not evaluate {split}: {e}")

        return
    
    # Run training
    logger.info("Starting training...")
    train_result = trainer.train()
    
    logger.info(
        f"Training completed: {train_result['total_epochs']} epochs, "
        f"best val_loss={train_result['best_val_metric']:.6f} "
        f"at epoch {train_result['best_epoch']}"
    )
    
    # Final evaluation
    logger.info("\n" + "=" * 60)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 60)
    
    for split in ['val', 'test']:
        try:
            metrics = trainer.evaluate(split)
            if split == 'test':
                written = _dump_test_metrics(metrics, output_dir)
                if written is not None:
                    logger.info(f"Saved test metrics to {written}")
            logger.info(f"\n{split.upper()} Results:\n{_safe_summary(metrics)}")
        except ValueError as e:
            logger.warning(f"Could not evaluate {split}: {e}")
    
    # Save final model
    final_model_path = output_dir / 'checkpoints' / 'final.pt'
    trainer.save_checkpoint(final_model_path)
    logger.info(f"Saved final model to {final_model_path}")
    
    logger.info(f"\nExperiment completed. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()

