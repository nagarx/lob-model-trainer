"""
Command-line interface for LOB Model Trainer.

Provides a production-ready entry point for training and evaluation with:
- Configuration-driven experiments (YAML/JSON)
- Command-line overrides for quick experiments
- Checkpoint resume capability
- Optional monitoring and diagnostics
- Proper file logging for experiment tracking

Design principles (RULE.md):
- Configuration-driven: All parameters via config file or CLI overrides
- Reproducible: Explicit seed management and config saving
- Comprehensive: Supports all training features

Usage:
    # Basic training
    lobtrainer train configs/experiments/nvda_tlob_h50_v1.yaml
    
    # With overrides
    lobtrainer train configs/experiments/nvda_tlob_h50_v1.yaml --epochs 20 --lr 5e-5
    
    # Resume from checkpoint
    lobtrainer train configs/experiments/nvda_tlob_h50_v1.yaml --resume outputs/checkpoints/best.pt
    
    # Evaluate only
    lobtrainer evaluate configs/experiments/nvda_tlob_h50_v1.yaml --checkpoint outputs/checkpoints/best.pt
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from lobtrainer.config import load_config, save_config, ExperimentConfig
from lobtrainer.training.trainer import Trainer
from lobtrainer.training.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    MetricLogger,
    ProgressCallback,
)
from lobtrainer.training.monitoring import (
    GradientMonitor,
    LearningRateTracker,
    TrainingDiagnostics,
    PerClassMetricsTracker,
)
from lobtrainer.utils.reproducibility import set_seed


# =============================================================================
# Logging Setup
# =============================================================================


def setup_logging(
    level: str = "INFO",
    output_dir: Optional[Path] = None,
) -> logging.Logger:
    """
    Configure logging to console and optionally to file.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        output_dir: If provided, also log to file in output_dir/logs/
    
    Returns:
        Logger instance for the CLI module
    """
    log_level = getattr(logging, level.upper())
    
    # Console handler with clean format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all, handlers filter
    
    # Clear existing handlers to avoid duplicates
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    
    # File handler if output_dir provided
    if output_dir is not None:
        log_dir = output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"train_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root_logger.addHandler(file_handler)
        
        # Log the log file path
        logging.info(f"Logging to file: {log_file}")
    
    return logging.getLogger(__name__)


# =============================================================================
# Config Override Application
# =============================================================================


def apply_overrides(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    """
    Apply command-line overrides to configuration.
    
    Args:
        config: Base configuration loaded from file
        args: Parsed command-line arguments
    
    Returns:
        Modified configuration with overrides applied
    """
    # Data overrides
    if getattr(args, 'data_dir', None) is not None:
        config.data.data_dir = args.data_dir
    
    # Training overrides
    if getattr(args, 'epochs', None) is not None:
        config.train.epochs = args.epochs
    
    if getattr(args, 'batch_size', None) is not None:
        config.train.batch_size = args.batch_size
    
    if getattr(args, 'learning_rate', None) is not None:
        config.train.learning_rate = args.learning_rate
    
    if getattr(args, 'seed', None) is not None:
        config.train.seed = args.seed
    
    # Output overrides
    if getattr(args, 'output_dir', None) is not None:
        config.output_dir = args.output_dir
    
    return config


# =============================================================================
# Callback Factory
# =============================================================================


def create_callbacks(
    config: ExperimentConfig,
    output_dir: Path,
    no_progress: bool = False,
    enable_monitoring: bool = False,
) -> list:
    """
    Create the standard set of callbacks for training.
    
    Args:
        config: Experiment configuration
        output_dir: Directory for outputs (checkpoints, logs, etc.)
        no_progress: If True, disable progress bar
        enable_monitoring: If True, add advanced monitoring callbacks
    
    Returns:
        List of callback instances
    """
    callbacks = [
        # Core callbacks (always enabled)
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
    
    # Progress callback (optional, enabled by default)
    if not no_progress:
        callbacks.append(ProgressCallback())
    
    # Advanced monitoring callbacks (optional)
    if enable_monitoring:
        task_type = getattr(config, 'task_type', 'multiclass')
        callbacks.extend([
            GradientMonitor(
                log_every_n_batches=100,
                warn_threshold_low=1e-7,
                warn_threshold_high=100.0,
                save_history=True,
            ),
            LearningRateTracker(save_history=True),
            TrainingDiagnostics(
                alert_on_nan=True,
                stagnation_patience=5,
            ),
        ])
        if task_type != 'regression':
            callbacks.append(PerClassMetricsTracker(save_history=True))
    
    return callbacks


# =============================================================================
# Train Command
# =============================================================================


def train_command(args: argparse.Namespace) -> int:
    """
    Execute training command.
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        return 1
    
    config = load_config(str(config_path))
    
    # Apply command-line overrides
    config = apply_overrides(config, args)
    
    # Setup output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging (to console and file)
    logger = setup_logging(args.log_level, output_dir)
    
    logger.info(f"=" * 60)
    logger.info(f"LOB Model Trainer - Starting Experiment")
    logger.info(f"=" * 60)
    logger.info(f"Experiment: {config.name}")
    logger.info(f"Model type: {config.model.model_type.value}")
    logger.info(f"Data directory: {config.data.data_dir}")
    logger.info(f"Feature count: {config.data.feature_count}")
    logger.info(f"Output directory: {output_dir}")
    
    # Save config for reproducibility
    config_save_path = output_dir / "config.yaml"
    save_config(config, str(config_save_path))
    logger.info(f"Saved config to: {config_save_path}")
    
    # Set random seed for reproducibility
    set_seed(config.train.seed)
    logger.info(f"Random seed: {config.train.seed}")
    
    # Create callbacks
    callbacks = create_callbacks(
        config=config,
        output_dir=output_dir,
        no_progress=getattr(args, 'no_progress', False),
        enable_monitoring=getattr(args, 'monitoring', False),
    )
    logger.info(f"Callbacks enabled: {[type(c).__name__ for c in callbacks]}")
    
    # Create trainer
    trainer = Trainer(config, callbacks=callbacks)
    
    # Resume from checkpoint if specified
    if getattr(args, 'resume', None):
        checkpoint_path = Path(args.resume)
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return 1
        trainer.load_checkpoint(checkpoint_path)
        logger.info(f"Resumed from checkpoint: {checkpoint_path}")
    
    # Run training
    logger.info(f"-" * 60)
    logger.info("Starting training...")
    logger.info(f"-" * 60)
    
    result = trainer.train()
    
    logger.info(f"-" * 60)
    logger.info(
        f"Training completed: {result['total_epochs']} epochs in "
        f"{result['total_time_seconds']:.1f}s"
    )
    logger.info(
        f"Best val_loss: {result['best_val_metric']:.6f} at epoch {result['best_epoch']}"
    )
    logger.info(f"-" * 60)
    
    # Final evaluation on validation and test sets
    logger.info(f"=" * 60)
    logger.info("FINAL EVALUATION")
    logger.info(f"=" * 60)
    
    test_result = None
    for split in ['val', 'test']:
        try:
            eval_result = trainer.evaluate(split)
            if split == 'test':
                test_result = eval_result
            if hasattr(eval_result, 'summary'):
                logger.info(f"\n{split.upper()} Results:\n{eval_result.summary()}")
            elif isinstance(eval_result, dict):
                logger.info(f"\n{split.upper()} Results (regression):")
                for k, v in sorted(eval_result.items()):
                    if isinstance(v, float):
                        logger.info(f"  {k}: {v:.6f}")
        except ValueError as e:
            logger.warning(f"Could not evaluate {split}: {e}")
    
    # Save final model
    final_model_path = output_dir / 'checkpoints' / 'final.pt'
    final_model_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(final_model_path)
    logger.info(f"Saved final model to: {final_model_path}")
    
    # Register experiment in the experiment registry
    manifest_name = getattr(args, 'manifest', None)
    try:
        from lobtrainer.experiments.result import ExperimentResult
        from lobtrainer.experiments.registry import ExperimentRegistry
        
        registry_dir = output_dir.parent / '_registry'
        registry = ExperimentRegistry(registry_dir)
        
        exp_result = ExperimentResult.from_trainer(trainer, test_metrics=test_result)
        exp_result.checkpoint_path = str(final_model_path)
        exp_result.output_dir = str(output_dir)
        if manifest_name:
            exp_result.tags = list(getattr(exp_result, 'tags', []) or [])
            exp_result.tags.append(f"manifest:{Path(manifest_name).stem}")
        exp_id = registry.register(exp_result)
        logger.info(f"Registered experiment in registry: {exp_id}")
    except Exception as e:
        logger.warning(f"Failed to register experiment (non-fatal): {e}")

    # Update hft-ops ledger if manifest provided
    if manifest_name:
        try:
            import yaml as _yaml
            manifest_path = Path(manifest_name)
            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifest_data = _yaml.safe_load(f)
                manifest_exp_name = manifest_data.get("experiment", {}).get("name", "unknown")
                ledger_path = manifest_path.parent.parent / "ledger" / "runs"
                ledger_path.mkdir(parents=True, exist_ok=True)
                import json as _json
                record = {
                    "experiment_name": manifest_exp_name,
                    "stage": "training",
                    "status": "completed",
                    "output_dir": str(output_dir),
                    "checkpoint": str(final_model_path),
                    "best_val_loss": result.get("best_val_metric", None),
                    "total_epochs": result.get("total_epochs", None),
                    "manifest": str(manifest_path),
                }
                record_path = ledger_path / f"{manifest_exp_name}_training.json"
                with open(record_path, "w") as f:
                    _json.dump(record, f, indent=2, default=str)
                logger.info(f"Updated hft-ops ledger: {record_path}")
        except Exception as e:
            logger.warning(f"Failed to update hft-ops ledger (non-fatal): {e}")
    
    logger.info(f"=" * 60)
    logger.info(f"Experiment completed. Outputs saved to: {output_dir}")
    logger.info(f"=" * 60)
    
    return 0


# =============================================================================
# Evaluate Command
# =============================================================================


def evaluate_command(args: argparse.Namespace) -> int:
    """
    Execute evaluation command.
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        return 1
    
    config = load_config(str(config_path))
    
    # Setup logging (console only for evaluation)
    logger = setup_logging(args.log_level, output_dir=None)
    
    logger.info(f"Loading config from: {config_path}")
    logger.info(f"Experiment: {config.name}")
    
    # Create trainer (no callbacks needed for evaluation)
    trainer = Trainer(config, callbacks=[])
    
    # Load checkpoint (required for evaluation)
    if not args.checkpoint:
        logger.error("--checkpoint is required for evaluation")
        return 1
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return 1
    
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    trainer.load_checkpoint(checkpoint_path, load_optimizer=False)
    
    # Evaluate on specified split(s)
    splits = args.split.split(',') if ',' in args.split else [args.split]
    
    for split in splits:
        split = split.strip()
        logger.info(f"Evaluating on {split} set...")
        
        try:
            eval_result = trainer.evaluate(split)
            if hasattr(eval_result, 'summary'):
                logger.info(f"\n{split.upper()} Results:\n{eval_result.summary()}")
            elif isinstance(eval_result, dict):
                logger.info(f"\n{split.upper()} Results (regression):")
                for k, v in sorted(eval_result.items()):
                    if isinstance(v, float):
                        logger.info(f"  {k}: {v:.6f}")
        except ValueError as e:
            logger.error(f"Could not evaluate {split}: {e}")
            return 1
    
    return 0


# =============================================================================
# Argument Parser
# =============================================================================


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all commands and options."""
    parser = argparse.ArgumentParser(
        prog="lobtrainer",
        description="LOB Model Trainer - Train and evaluate limit order book prediction models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic training
    lobtrainer train configs/experiments/nvda_tlob_h50_v1.yaml
    
    # Training with overrides
    lobtrainer train configs/experiments/nvda_tlob_h50_v1.yaml --epochs 20 --lr 5e-5
    
    # Training with monitoring enabled
    lobtrainer train configs/experiments/nvda_tlob_h50_v1.yaml --monitoring
    
    # Resume training from checkpoint
    lobtrainer train configs/experiments/nvda_tlob_h50_v1.yaml --resume outputs/checkpoints/best.pt
    
    # Evaluate a trained model
    lobtrainer evaluate configs/experiments/nvda_tlob_h50_v1.yaml --checkpoint outputs/checkpoints/best.pt
    
    # Evaluate on multiple splits
    lobtrainer evaluate configs/experiments/nvda_tlob_h50_v1.yaml --checkpoint outputs/checkpoints/best.pt --split val,test
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # =========================================================================
    # Train command
    # =========================================================================
    train_parser = subparsers.add_parser(
        "train",
        help="Train a model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Required arguments
    train_parser.add_argument(
        "config",
        type=str,
        help="Path to experiment configuration file (YAML or JSON)",
    )
    
    # Optional overrides (frequently changed parameters)
    override_group = train_parser.add_argument_group("Config Overrides")
    override_group.add_argument(
        "--data-dir",
        type=str,
        default=None,
        metavar="PATH",
        help="Override data directory",
    )
    override_group.add_argument(
        "--output-dir",
        type=str,
        default=None,
        metavar="PATH",
        help="Override output directory",
    )
    override_group.add_argument(
        "--epochs",
        type=int,
        default=None,
        metavar="N",
        help="Override number of training epochs",
    )
    override_group.add_argument(
        "--batch-size",
        type=int,
        default=None,
        metavar="N",
        help="Override batch size",
    )
    override_group.add_argument(
        "--learning-rate", "--lr",
        type=float,
        default=None,
        dest="learning_rate",
        metavar="LR",
        help="Override learning rate",
    )
    override_group.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="N",
        help="Override random seed",
    )
    
    # Checkpoint options
    checkpoint_group = train_parser.add_argument_group("Checkpoint")
    checkpoint_group.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="PATH",
        help="Resume training from checkpoint file",
    )
    
    # Callback options
    callback_group = train_parser.add_argument_group("Callbacks")
    callback_group.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar",
    )
    callback_group.add_argument(
        "--monitoring",
        action="store_true",
        help="Enable advanced monitoring (gradients, LR, diagnostics)",
    )
    
    # Logging options
    logging_group = train_parser.add_argument_group("Logging")
    logging_group.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    orchestration_group = train_parser.add_argument_group("Orchestration")
    orchestration_group.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to hft-ops experiment manifest YAML (links training to unified experiment ID)",
    )
    
    train_parser.set_defaults(func=train_command)
    
    # =========================================================================
    # Evaluate command
    # =========================================================================
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a trained model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    eval_parser.add_argument(
        "config",
        type=str,
        help="Path to experiment configuration file (YAML or JSON)",
    )
    eval_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        metavar="PATH",
        help="Path to model checkpoint file (required)",
    )
    eval_parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Data split(s) to evaluate on, comma-separated (default: test)",
    )
    eval_parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    
    eval_parser.set_defaults(func=evaluate_command)
    
    return parser


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.", file=sys.stderr)
        return 130
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
