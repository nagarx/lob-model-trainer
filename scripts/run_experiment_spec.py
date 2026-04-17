#!/usr/bin/env python3
"""Run a complete experiment from an ExperimentSpec YAML (T14).

End-to-end orchestrator: validate → signal quality gate → train → record.

Usage:
    python scripts/run_experiment_spec.py experiments/e17_fusion.yaml
    python scripts/run_experiment_spec.py experiments/e17_fusion.yaml --dry-run
    python scripts/run_experiment_spec.py experiments/e17_fusion.yaml --skip-gates

Reference: plan/EXPERIMENTATION_FIRST_ARCHITECTURE.md §20
"""

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

# Ensure lob-model-trainer/src is on the path
_SCRIPT_DIR = Path(__file__).resolve().parent
_TRAINER_SRC = _SCRIPT_DIR.parent / "src"
if str(_TRAINER_SRC) not in sys.path:
    sys.path.insert(0, str(_TRAINER_SRC))

# Phase 1.4 deprecation hook — this script is obsoleted by hft-ops manifests
# + Phase 2 validation stage (which supersedes internal spec gate logic).
sys.path.insert(0, str(_SCRIPT_DIR))
from _hft_ops_compat import warn_if_not_orchestrated
warn_if_not_orchestrated(
    script_name="run_experiment_spec.py",
    suggestion=(
        "Use 'hft-ops run <manifest>' instead. The Phase 2 validation stage "
        "supersedes this script's internal IC gate. ExperimentSpec YAMLs can "
        "be converted to hft-ops manifests."
    ),
)

from lobtrainer.config.experiment_spec import ExperimentSpec

logger = logging.getLogger(__name__)


def run_experiment_spec(
    spec: ExperimentSpec,
    dry_run: bool = False,
    skip_gates: bool = False,
) -> dict:
    """Execute a full experiment from an ExperimentSpec.

    Stages:
        1. Validate spec (config generation dry-run)
        2. Load training data
        3. Signal quality gate (mandatory unless skip_gates)
        4. Training (Trainer or CVTrainer)
        5. Evaluation
        6. Record results

    Args:
        spec: Parsed ExperimentSpec.
        dry_run: If True, validate and gate only (no training).
        skip_gates: If True, skip pre-training gates.

    Returns:
        Dict with experiment results and metadata.
    """
    report = {
        "experiment_name": spec.experiment.name,
        "hypothesis": spec.experiment.hypothesis,
        "tags": spec.experiment.tags,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "stages_completed": [],
        "status": "pending",
    }

    # Stage 1: Validate
    logger.info("=" * 60)
    logger.info("Experiment: %s", spec.experiment.name)
    logger.info("Hypothesis: %s", spec.experiment.hypothesis)
    logger.info("=" * 60)

    validation_warnings = spec.validate()
    for w in validation_warnings:
        logger.warning("Validation warning: %s", w)
    report["stages_completed"].append("validate")
    report["validation_warnings"] = validation_warnings

    config = spec.to_experiment_config()
    logger.info("ExperimentConfig generated successfully")
    logger.info("  data_dir: %s", config.data.data_dir)
    logger.info("  model: %s (input_size=%d)", config.model.model_type.value, config.model.input_size)
    logger.info("  epochs: %d", config.train.epochs)

    if config.data.sources:
        logger.info("  sources: %s", [(s.name, s.role) for s in config.data.sources])

    # Stage 2: Load training data (for gate)
    if not skip_gates and spec.gates.signal_quality.enabled:
        logger.info("\n--- Stage 2: Loading training data for signal gate ---")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            if config.data.sources is not None:
                from lobtrainer.data.bundle import load_split_bundles
                from lobtrainer.data.sources import DataSource
                sources = [
                    DataSource(name=s.name, data_dir=s.data_dir, role=s.role)
                    for s in config.data.sources
                ]
                train_days = load_split_bundles(
                    sources, "train", labels_config=config.data.labels
                )
            else:
                from lobtrainer.data.dataset import load_split_data
                train_days = load_split_data(
                    config.data.data_dir, "train",
                    labels_config=config.data.labels,
                )
        logger.info("Loaded %d training days", len(train_days))
        report["stages_completed"].append("data_loaded")
        report["n_train_days"] = len(train_days)

        # Stage 3: Signal quality gate
        logger.info("\n--- Stage 3: Signal quality gate ---")
        from lobtrainer.training.gates import run_signal_quality_gate

        gate_result = run_signal_quality_gate(
            train_days,
            horizon_idx=config.data.labels.primary_horizon_idx or 0,
            min_ic=spec.gates.signal_quality.min_ic,
            min_features_passing=spec.gates.signal_quality.min_features_passing,
        )
        report["stages_completed"].append("signal_quality_gate")
        report["gate_result"] = {
            "passed": gate_result.passed,
            "message": gate_result.message,
        }

        if not gate_result.passed:
            report["status"] = "gate_failed"
            logger.error("EXPERIMENT STOPPED: %s", gate_result.message)
            report["finished_at"] = datetime.now(timezone.utc).isoformat()
            return report

    elif skip_gates:
        logger.info("\n--- Skipping gates (--skip-gates) ---")

    if dry_run:
        report["status"] = "dry_run"
        logger.info("\n--- Dry run complete (no training) ---")
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        return report

    # Stage 4: Training
    logger.info("\n--- Stage 4: Training ---")
    from lobtrainer.training.trainer import Trainer

    if config.cv is not None:
        from lobtrainer.training.cv_trainer import CVTrainer
        logger.info("CV mode: %d folds, embargo=%d", config.cv.n_splits, config.cv.embargo_days)
        cv_trainer = CVTrainer(config)
        cv_results = cv_trainer.run()
        report["stages_completed"].append("cv_training")
        report["cv_summary"] = cv_results.summary()
        report["metrics"] = cv_results.mean_metrics
        report["metrics_std"] = cv_results.std_metrics
        logger.info("\n%s", cv_results.summary())
    else:
        trainer = Trainer(config)
        trainer.train()
        report["stages_completed"].append("training")

        # Stage 5: Evaluate
        logger.info("\n--- Stage 5: Evaluation ---")
        try:
            val_metrics = trainer.evaluate("val")
            if isinstance(val_metrics, dict):
                report["metrics"] = val_metrics
            elif hasattr(val_metrics, "to_dict"):
                # ClassificationMetrics has .to_dict() for structured metrics
                report["metrics"] = val_metrics.to_dict()
            else:
                report["metrics"] = {"val_result": str(val_metrics)}
            report["stages_completed"].append("evaluation")
        except Exception as e:
            logger.warning("Evaluation failed: %s", e)
            report["metrics"] = {}

    report["status"] = "completed"
    report["finished_at"] = datetime.now(timezone.utc).isoformat()

    # Stage 6: Save report (append stage BEFORE writing for consistency)
    report["stages_completed"].append("report_saved")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "experiment_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("\nExperiment report saved to %s", report_path)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Run experiment from ExperimentSpec YAML"
    )
    parser.add_argument("spec_yaml", help="Path to ExperimentSpec YAML")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate and gate only, no training",
    )
    parser.add_argument(
        "--skip-gates", action="store_true",
        help="Skip pre-training gates",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    spec = ExperimentSpec.from_yaml(args.spec_yaml)
    report = run_experiment_spec(
        spec,
        dry_run=args.dry_run,
        skip_gates=args.skip_gates,
    )

    if report["status"] == "completed":
        logger.info("\n✓ Experiment completed successfully")
    elif report["status"] == "gate_failed":
        logger.error("\n✗ Experiment stopped: gate failed")
        sys.exit(1)
    elif report["status"] == "dry_run":
        logger.info("\n✓ Dry run completed")


if __name__ == "__main__":
    main()
