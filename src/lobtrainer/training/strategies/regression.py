"""
Regression training strategy.

Handles standard single-horizon regression (TLOB-R, DeepLOB-R, etc.)
using model.compute_loss() with regression targets.

Batch format: (features [B,T,F], regression_target [B] float32)
Loss: model.compute_loss(output, regression_targets=target)
Metrics: R-squared, IC, MAE, RMSE, directional accuracy, profitable accuracy
"""

import logging
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lobtrainer.training.strategy import BatchResult, TrainingStrategy
from lobtrainer.training.regression_metrics import compute_all_regression_metrics

logger = logging.getLogger(__name__)


class RegressionStrategy(TrainingStrategy):
    """Training strategy for single-horizon regression tasks.

    Uses model.compute_loss() for training, compute_all_regression_metrics
    for validation/evaluation.
    """

    @property
    def requires_regression_targets(self) -> bool:
        return True

    # =========================================================================
    # Core Methods
    # =========================================================================

    def process_batch(self, model: nn.Module, batch_data: tuple) -> BatchResult:
        """Process one regression batch.

        Args:
            model: Model in train mode.
            batch_data: (features [B,T,F], regression_target [B]).

        Returns:
            BatchResult with regression loss.
        """
        features, labels = batch_data
        features = features.to(self.device)
        regression_target = labels.to(self.device).float()

        output = model(features)
        loss, _ = model.compute_loss(output, regression_targets=regression_target)

        return BatchResult(
            loss=loss,
            batch_size=features.size(0),
            metrics={"loss": loss.item()},
        )

    def aggregate_epoch_metrics(
        self,
        results: List[BatchResult],
        total_samples: int,
    ) -> Dict[str, float]:
        """Aggregate batch results into epoch metrics.

        Returns:
            {'train_loss': float}
        """
        if total_samples == 0:
            return {"train_loss": 0.0}

        total_loss = sum(r.metrics["loss"] * r.batch_size for r in results)
        return {"train_loss": total_loss / total_samples}

    @torch.no_grad()
    def validate(self, model: nn.Module, loader: DataLoader) -> Dict[str, float]:
        """Validate with regression metrics (R^2, IC, MAE, DA, etc.).

        Returns dict with val_loss and val_-prefixed regression metrics.
        """
        model.eval()

        total_loss = 0.0
        total_samples = 0
        all_preds = []
        all_targets = []

        for features, labels in loader:
            features = features.to(self.device)
            regression_target = labels.to(self.device).float()

            output = model(features)
            loss, _ = model.compute_loss(output, regression_targets=regression_target)

            batch_size = features.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            preds = output.predictions
            all_preds.append(preds.detach().cpu().numpy())
            all_targets.append(labels.numpy())

        if total_samples == 0:
            return {"val_loss": float("inf")}

        avg_loss = total_loss / total_samples
        y_pred = np.concatenate(all_preds).ravel()
        y_true = np.concatenate(all_targets).ravel()

        result = {"val_loss": avg_loss}
        reg_metrics = compute_all_regression_metrics(y_true, y_pred, prefix="val_")
        result.update(reg_metrics)

        return result

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
        split: str,
    ) -> Dict[str, Any]:
        """Full evaluation with regression metrics (no prefix).

        Args:
            model: Model to evaluate.
            loader: Data loader for the split.
            split: Split name for logging.

        Returns:
            Dict with R^2, IC, MAE, RMSE, DA, profitable accuracy.
        """
        model.eval()

        all_preds = []
        all_targets = []

        for features, labels in loader:
            features = features.to(self.device)
            output = model(features)

            preds = output.predictions
            all_preds.append(preds.detach().cpu().numpy())
            all_targets.append(labels.numpy())

        y_pred = np.concatenate(all_preds).ravel()
        y_true = np.concatenate(all_targets).ravel()

        results = compute_all_regression_metrics(y_true, y_pred, prefix="")

        logger.info(
            f"Evaluation [{split}]: R\u00b2={results.get('r2', 0):.4f}, "
            f"IC={results.get('ic', 0):.4f}, "
            f"MAE={results.get('mae', 0):.4f}, "
            f"DA={results.get('directional_accuracy', 0):.4f}"
        )

        return results

    @torch.no_grad()
    def predict(
        self,
        model: nn.Module,
        features: torch.Tensor,
        return_proba: bool = False,
    ) -> np.ndarray:
        """Predict continuous return values.

        Args:
            model: Model for inference.
            features: Input tensor [B, T, F] on device.
            return_proba: Ignored for regression.

        Returns:
            np.ndarray of predicted returns [B].
        """
        model.eval()
        output = model(features)
        return output.predictions.cpu().numpy()
