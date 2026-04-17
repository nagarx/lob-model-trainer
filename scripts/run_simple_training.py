#!/usr/bin/env python3
"""
Unified simple model training CLI.

Trains non-PyTorch models (Ridge, GradBoost) through the full pipeline:
config -> load data -> engineer features -> fit -> evaluate -> export signals -> save.

Output follows the same contract as PyTorch experiments so the backtester
works unchanged.

Usage:
    python scripts/run_simple_training.py configs/experiments/nvda_temporal_ridge_h10.yaml
    python scripts/run_simple_training.py configs/experiments/nvda_temporal_gradboost_h10.yaml
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Phase 1.4 deprecation hook
sys.path.insert(0, str(Path(__file__).parent))
from _hft_ops_compat import warn_if_not_orchestrated
warn_if_not_orchestrated(
    script_name="run_simple_training.py",
    suggestion=(
        "Use 'hft-ops run <manifest>' with a unified manifest referencing "
        "this simple-trainer config."
    ),
)

from lobtrainer.training.simple_trainer import SimpleModelTrainer


def main():
    parser = argparse.ArgumentParser(description="Simple Model Training")
    parser.add_argument("config", type=Path, help="YAML config file")
    parser.add_argument("--no-export", action="store_true", help="Skip signal export")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    log_level = config.get("log_level", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info(f"  SIMPLE MODEL TRAINING: {config.get('name', 'unnamed')}")
    logger.info("=" * 60)

    trainer = SimpleModelTrainer(
        data_dir=config["data"]["data_dir"],
        model_type=config["model"]["model_type"],
        model_config={k: v for k, v in config["model"].items() if k != "model_type" and k != "features"},
        feature_config=config["model"].get("features", {}),
        horizon_idx=config["data"].get("horizon_idx", 0),
        output_dir=config.get("output_dir", "outputs/experiments/simple_model"),
    )

    trainer.setup()
    trainer.train()

    test_metrics = trainer.evaluate()
    logger.info("--- TEST RESULTS ---")
    for k, v in sorted(test_metrics.items()):
        logger.info(f"  {k:<25}: {v:.6f}")

    if not args.no_export:
        trainer.export_signals("test")

    trainer.save()

    with open(args.config) as f:
        config_copy = yaml.safe_load(f)
    output_dir = Path(config.get("output_dir", "outputs/experiments/simple_model"))
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config_copy, f, default_flow_style=False)

    logger.info(f"\nExperiment complete: {output_dir}")


if __name__ == "__main__":
    main()
