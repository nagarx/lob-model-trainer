"""Purged K-Fold cross-validation trainer (T11).

Wraps Trainer for running K temporal folds with embargo periods.
Creates a fresh Trainer per fold to ensure complete state reset
(model weights, optimizer, scheduler, callbacks).

Resolves known issue C3: "No cross-validation in trainer."

Reference: de Prado (2018) AFML Chapter 7.
"""

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from lobtrainer.config.schema import CVConfig, ExperimentConfig
from lobtrainer.data.dataset import load_split_data

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    """Metrics and metadata from one CV fold."""

    fold_idx: int
    train_dates: List[str]
    val_dates: List[str]
    val_metrics: Dict[str, float]
    train_metrics: Dict[str, float]
    best_epoch: int


@dataclass
class CVResults:
    """Aggregated results across all K folds."""

    folds: List[FoldResult]
    config_name: str = ""

    @property
    def mean_metrics(self) -> Dict[str, float]:
        """Mean of each metric across folds."""
        if not self.folds:
            return {}
        all_keys = set()
        for f in self.folds:
            all_keys.update(f.val_metrics.keys())
        result = {}
        for key in sorted(all_keys):
            values = [
                f.val_metrics[key]
                for f in self.folds
                if key in f.val_metrics
            ]
            if values:
                result[key] = sum(values) / len(values)
        return result

    @property
    def std_metrics(self) -> Dict[str, float]:
        """Std of each metric across folds."""
        if not self.folds:
            return {}
        mean = self.mean_metrics
        result = {}
        for key, m in mean.items():
            values = [
                f.val_metrics[key]
                for f in self.folds
                if key in f.val_metrics
            ]
            if len(values) >= 2:
                variance = sum((v - m) ** 2 for v in values) / (len(values) - 1)
                result[key] = variance ** 0.5
            else:
                result[key] = 0.0
        return result

    def summary(self) -> str:
        """Formatted summary: metric ± std across folds."""
        lines = [f"CV Results ({len(self.folds)} folds):"]
        mean = self.mean_metrics
        std = self.std_metrics
        for key in sorted(mean):
            lines.append(f"  {key}: {mean[key]:.4f} ± {std.get(key, 0):.4f}")
        return "\n".join(lines)


class CVTrainer:
    """Purged K-Fold cross-validation wrapper around Trainer.

    Loads all train+val days once, then for each fold:
    1. Splits days into fold-specific train/val via purged_kfold_split
    2. Creates a fresh Trainer with preloaded_days (mandatory per Decision 9)
    3. Trains from scratch with fold-specific seed
    4. Evaluates on the fold's val set
    5. Collects metrics

    The test set (if present on disk) is NEVER touched during CV.

    Args:
        config: ExperimentConfig with data_dir, model, train parameters.
        n_splits: Number of temporal folds. Default 5.
        embargo_days: Days after each val block excluded from training.
            Default 1.

    Usage:
        >>> cv = CVTrainer(config, n_splits=5, embargo_days=1)
        >>> results = cv.run()
        >>> print(results.summary())
    """

    _UNSET = object()

    def __init__(
        self,
        config: ExperimentConfig,
        n_splits: "int | object" = _UNSET,
        embargo_days: "int | object" = _UNSET,
    ):
        self.config = config

        # Resolve: explicit arg > CVConfig > hard default
        if config.cv is not None:
            self.n_splits = (
                n_splits if n_splits is not self._UNSET
                else config.cv.n_splits
            )
            self.embargo_days = (
                embargo_days if embargo_days is not self._UNSET
                else config.cv.embargo_days
            )
        else:
            self.n_splits = 5 if n_splits is self._UNSET else n_splits
            self.embargo_days = 1 if embargo_days is self._UNSET else embargo_days

    def run(self) -> CVResults:
        """Run K-fold cross-validation.

        Returns:
            CVResults with per-fold metrics and aggregated statistics.
        """
        from hft_metrics.purged_cv import purged_kfold_split
        from lobtrainer.training.trainer import Trainer

        data_dir = self.config.data.data_dir
        labels_config = self.config.data.labels

        # 1. Load ALL train + val days (test stays held out)
        logger.info("CV: loading all train+val days from %s", data_dir)
        all_days = []
        for split in ["train", "val"]:
            try:
                split_days = load_split_data(
                    data_dir, split,
                    labels_config=labels_config,
                    validate=True, lazy=False,
                )
                all_days.extend(split_days)
                logger.info("  %s: %d days loaded", split, len(split_days))
            except FileNotFoundError:
                logger.info("  %s: not found, skipping", split)

        if not all_days:
            raise ValueError(
                f"No train/val data found at {data_dir}. "
                "CV requires at least train/ and optionally val/ directories."
            )

        # Sort by date (YYYYMMDD sorts chronologically)
        all_days.sort(key=lambda d: d.date)
        all_dates = [d.date for d in all_days]
        date_to_day = {d.date: d for d in all_days}

        logger.info(
            "CV: %d total days (%s to %s), %d splits, embargo=%d",
            len(all_dates), all_dates[0], all_dates[-1],
            self.n_splits, self.embargo_days,
        )

        # 2. Generate purged fold splits
        folds = purged_kfold_split(
            all_dates, self.n_splits, self.embargo_days
        )

        # 3. Run each fold
        fold_results: List[FoldResult] = []
        for fold_idx, (train_dates, val_dates) in enumerate(folds):
            logger.info(
                "=== CV Fold %d/%d: train=%d days, val=%d days ===",
                fold_idx + 1, self.n_splits,
                len(train_dates), len(val_dates),
            )

            train_days = [date_to_day[d] for d in train_dates]
            val_days = [date_to_day[d] for d in val_dates]

            fold_config = self._build_fold_config(fold_idx)

            trainer = Trainer(
                fold_config,
                preloaded_days={"train": train_days, "val": val_days},
            )
            trainer.train()

            # Evaluate on fold's val set
            val_metrics = trainer.evaluate("val")
            if not isinstance(val_metrics, dict):
                val_metrics = {"val_metric": float(val_metrics)}

            train_loss = trainer.state.history[-1] if trainer.state.history else {}
            if isinstance(train_loss, dict):
                train_metrics = train_loss
            else:
                train_metrics = {"train_loss": float(train_loss)}

            fold_results.append(FoldResult(
                fold_idx=fold_idx,
                train_dates=train_dates,
                val_dates=val_dates,
                val_metrics=val_metrics,
                train_metrics=train_metrics,
                best_epoch=trainer.state.best_epoch,
            ))

            logger.info(
                "  Fold %d result: best_epoch=%d, val_metrics=%s",
                fold_idx, trainer.state.best_epoch,
                {k: f"{v:.4f}" for k, v in val_metrics.items()
                 if isinstance(v, (int, float))},
            )

        results = CVResults(fold_results, config_name=self.config.name)
        logger.info("\n%s", results.summary())
        return results

    def _build_fold_config(self, fold_idx: int) -> ExperimentConfig:
        """Build per-fold config with distinct output_dir and seed."""
        fold_cfg = copy.deepcopy(self.config)
        fold_cfg.output_dir = str(
            Path(self.config.output_dir) / f"cv_fold_{fold_idx}"
        )
        fold_cfg.train.seed = self.config.train.seed + fold_idx
        fold_cfg.name = f"{self.config.name}_fold{fold_idx}"
        return fold_cfg
