"""
Command-line interface for LOB Model Trainer.

Provides the main entry point for training and evaluation.

Usage:
    lobtrainer train configs/experiments/nvda_tlob_h50_v1.yaml
    lobtrainer evaluate configs/experiments/nvda_tlob_h50_v1.yaml --checkpoint path/to/model.pt
"""

import argparse
import logging
import sys
from pathlib import Path

from lobtrainer.config import load_config
from lobtrainer.training.trainer import Trainer


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the CLI."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def train_command(args: argparse.Namespace) -> int:
    """Execute training command."""
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return 1
    
    logger.info(f"Loading config from: {config_path}")
    config = load_config(str(config_path))
    
    logger.info(f"Starting experiment: {config.name}")
    logger.info(f"Model type: {config.model.model_type}")
    logger.info(f"Data directory: {config.data.data_dir}")
    
    trainer = Trainer(config)
    trainer.train()
    
    # Optionally evaluate on test set after training
    if args.evaluate_after:
        logger.info("Evaluating on test set...")
        metrics = trainer.evaluate("test")
        logger.info(f"Test metrics: {metrics}")
    
    logger.info("Training complete!")
    return 0


def evaluate_command(args: argparse.Namespace) -> int:
    """Execute evaluation command."""
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return 1
    
    logger.info(f"Loading config from: {config_path}")
    config = load_config(str(config_path))
    
    trainer = Trainer(config)
    
    # Load checkpoint if provided
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return 1
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        trainer.load_checkpoint(checkpoint_path)
    
    # Evaluate on specified split
    logger.info(f"Evaluating on {args.split} set...")
    metrics = trainer.evaluate(args.split)
    
    logger.info(f"Evaluation metrics ({args.split}):")
    for name, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {name}: {value:.4f}")
        else:
            logger.info(f"  {name}: {value}")
    
    return 0


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="lobtrainer",
        description="LOB Model Trainer - Train and evaluate limit order book models",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument(
        "config",
        type=str,
        help="Path to experiment configuration file (YAML)",
    )
    train_parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    train_parser.add_argument(
        "--evaluate-after",
        action="store_true",
        help="Evaluate on test set after training completes",
    )
    train_parser.set_defaults(func=train_command)
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument(
        "config",
        type=str,
        help="Path to experiment configuration file (YAML)",
    )
    eval_parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to model checkpoint file",
    )
    eval_parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Data split to evaluate on (default: test)",
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


def main() -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
