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
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add parent directory to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

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


def apply_overrides(config: ExperimentConfig, args) -> ExperimentConfig:
    """Apply command-line overrides to configuration."""
    if args.data_dir is not None:
        config.data.data_dir = args.data_dir
    
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    
    if args.epochs is not None:
        config.train.epochs = args.epochs
    
    if args.batch_size is not None:
        config.train.batch_size = args.batch_size
    
    if args.learning_rate is not None:
        config.train.learning_rate = args.learning_rate
    
    if args.seed is not None:
        config.train.seed = args.seed
    
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
                logger.info(f"\n{split.upper()} Results:\n{metrics.summary()}")
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
            logger.info(f"\n{split.upper()} Results:\n{metrics.summary()}")
        except ValueError as e:
            logger.warning(f"Could not evaluate {split}: {e}")
    
    # Save final model
    final_model_path = output_dir / 'checkpoints' / 'final.pt'
    trainer.save_checkpoint(final_model_path)
    logger.info(f"Saved final model to {final_model_path}")
    
    logger.info(f"\nExperiment completed. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()

