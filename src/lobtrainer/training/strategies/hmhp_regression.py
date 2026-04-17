"""
HMHP Regression training strategy.

Handles multi-horizon regression with the Hierarchical Multi-Horizon
Regressor. Regression targets are {horizon: bps_tensor} dicts.
Per-horizon losses and regression quality metrics are tracked.

Batch format: (features [B,T,F], labels_dict {h: [B]}, regression_targets {h: [B]})
Loss: model.compute_loss(output, regression_targets=reg_dict)
Metrics: per-horizon R^2, IC, MAE, RMSE, DA, profitable accuracy
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lobtrainer.training.strategy import BatchResult, TrainingStrategy
from lobtrainer.training.regression_metrics import compute_all_regression_metrics

logger = logging.getLogger(__name__)


class HMHPRegressionStrategy(TrainingStrategy):
    """Training strategy for HMHP multi-horizon regression.

    The HMHP-R model produces per-horizon continuous predictions.
    Regression targets are required (raises ValueError if missing).
    """

    @property
    def requires_dict_labels(self) -> bool:
        return True

    @property
    def requires_regression_targets(self) -> bool:
        return True

    @property
    def horizon_idx(self) -> Optional[int]:
        """None: return all horizons as dict."""
        return None

    @property
    def horizons(self) -> list:
        return self.config.model.hmhp_horizons

    # =========================================================================
    # Core Methods
    # =========================================================================

    def process_batch(self, model: nn.Module, batch_data: tuple) -> BatchResult:
        """Process one HMHP regression batch.

        Args:
            model: HMHP-R model in train mode.
            batch_data: (features, labels_dict, regression_targets_dict).
                labels_dict is present but unused; regression_targets required.

        Returns:
            BatchResult with total loss and per-horizon losses.

        Raises:
            ValueError: If regression targets are missing from batch.
        """
        # T10: detect sample weights (scalar tensor) vs regression targets (dict)
        import torch
        sample_weights = None
        regression_targets = None
        if len(batch_data) >= 3:
            third = batch_data[2]
            if isinstance(third, dict):
                regression_targets = {
                    h: t.to(self.device) for h, t in third.items()
                }
            elif isinstance(third, torch.Tensor) and third.ndim == 1:
                sample_weights = third
        if len(batch_data) >= 4:
            fourth = batch_data[3]
            if isinstance(fourth, torch.Tensor) and fourth.ndim == 1:
                sample_weights = fourth

        if regression_targets is None:
            raise ValueError(
                "HMHP_REGRESSION requires regression targets but batch "
                "has no dict element. Check dataset return_regression_targets."
            )

        features = batch_data[0]
        features = features.to(self.device)
        if sample_weights is not None:
            sample_weights = sample_weights.to(self.device)
        output = model(features)

        loss, loss_components = model.compute_loss(
            output, regression_targets=regression_targets
        )
        if sample_weights is not None:
            loss = loss * sample_weights.mean()

        batch_size = features.size(0)
        metrics: Dict[str, float] = {"loss": loss.item()}

        for key, value in loss_components.items():
            if key.startswith("H"):
                metrics[f"{key}_loss"] = value.item()

        return BatchResult(
            loss=loss,
            batch_size=batch_size,
            metrics=metrics,
        )

    def aggregate_epoch_metrics(
        self,
        results: List[BatchResult],
        total_samples: int,
    ) -> Dict[str, float]:
        """Aggregate into epoch metrics with per-horizon losses.

        Returns:
            Dict with train_loss, train_h{x}_loss for each horizon.
        """
        if total_samples == 0:
            return {"train_loss": 0.0}

        total_loss = sum(r.metrics["loss"] * r.batch_size for r in results)
        result: Dict[str, float] = {"train_loss": total_loss / total_samples}

        for h in self.horizons:
            key = f"H{h}_loss"
            total = sum(
                r.metrics.get(key, 0.0) * r.batch_size for r in results
            )
            result[f"train_h{h}_loss"] = total / total_samples

        return result

    @torch.no_grad()
    def validate(self, model: nn.Module, loader: DataLoader) -> Dict[str, float]:
        """HMHP regression validation with per-horizon regression metrics.

        Returns dict with val_loss, per-horizon losses, and regression quality
        metrics (R^2, IC, MAE, RMSE, DA) for each horizon. Primary horizon
        metrics are also surfaced without horizon prefix for early-stopping.
        """
        model.eval()
        horizons = self.horizons
        primary_horizon = horizons[0]

        total_loss = 0.0
        total_samples = 0
        horizon_losses = {h: 0.0 for h in horizons}
        horizon_preds: Dict[int, list] = {h: [] for h in horizons}
        horizon_targets: Dict[int, list] = {h: [] for h in horizons}

        for batch_data in loader:
            if len(batch_data) == 3:
                features, _labels, regression_targets = batch_data
                regression_targets = {
                    h: t.to(self.device) for h, t in regression_targets.items()
                }
            else:
                features, _labels = batch_data
                regression_targets = None

            features = features.to(self.device)
            output = model(features)

            if regression_targets is not None:
                loss, loss_components = model.compute_loss(
                    output, regression_targets=regression_targets
                )

                batch_size = features.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                for h in horizons:
                    h_key = f"H{h}"
                    if h_key in loss_components:
                        horizon_losses[h] += (
                            loss_components[h_key].item() * batch_size
                        )

                for h in horizons:
                    if h in output.horizon_predictions:
                        horizon_preds[h].append(
                            output.horizon_predictions[h]
                            .squeeze(-1)
                            .cpu()
                            .numpy()
                        )
                    if h in regression_targets:
                        horizon_targets[h].append(
                            regression_targets[h].cpu().numpy()
                        )

        if total_samples == 0:
            return {"val_loss": float("inf")}

        avg_loss = total_loss / total_samples
        result: Dict[str, float] = {"val_loss": avg_loss}

        for h in horizons:
            result[f"val_h{h}_loss"] = horizon_losses[h] / total_samples

        # Primary horizon metrics (without horizon prefix for early-stopping)
        if horizon_preds[primary_horizon] and horizon_targets[primary_horizon]:
            y_pred = np.concatenate(horizon_preds[primary_horizon]).ravel()
            y_true = np.concatenate(horizon_targets[primary_horizon]).ravel()
            primary_metrics = compute_all_regression_metrics(
                y_true, y_pred, prefix="val_"
            )
            result.update(primary_metrics)

        # Per-horizon regression metrics
        for h in horizons:
            if horizon_preds[h] and horizon_targets[h]:
                y_pred = np.concatenate(horizon_preds[h]).ravel()
                y_true = np.concatenate(horizon_targets[h]).ravel()
                h_metrics = compute_all_regression_metrics(
                    y_true, y_pred, prefix=f"val_h{h}_"
                )
                result.update(h_metrics)

        return result

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
        split: str,
    ) -> Dict[str, Any]:
        """Full evaluation with per-horizon regression metrics (no prefix).

        Primary horizon metrics surfaced at top level. Per-horizon metrics
        use h{H}_ prefix.

        Returns:
            Dict with r2, ic, mae, etc. + h{H}_r2, h{H}_ic, etc.
        """
        model.eval()
        horizons = self.horizons
        primary_horizon = horizons[0]

        horizon_preds: Dict[int, list] = {h: [] for h in horizons}
        horizon_targets: Dict[int, list] = {h: [] for h in horizons}

        for batch_data in loader:
            if len(batch_data) == 3:
                features, _labels, regression_targets = batch_data
                regression_targets = {
                    h: t.to(self.device) for h, t in regression_targets.items()
                }
            else:
                features, _labels = batch_data
                regression_targets = None

            features = features.to(self.device)
            output = model(features)

            for h in horizons:
                if h in output.horizon_predictions:
                    horizon_preds[h].append(
                        output.horizon_predictions[h].squeeze(-1).cpu().numpy()
                    )
                if regression_targets is not None and h in regression_targets:
                    horizon_targets[h].append(
                        regression_targets[h].cpu().numpy()
                    )

        results: Dict[str, Any] = {}

        for h in horizons:
            if horizon_preds[h] and horizon_targets[h]:
                y_pred = np.concatenate(horizon_preds[h]).ravel()
                y_true = np.concatenate(horizon_targets[h]).ravel()

                h_metrics = compute_all_regression_metrics(
                    y_true, y_pred, prefix=f"h{h}_"
                )
                results.update(h_metrics)

                if h == primary_horizon:
                    primary_metrics = compute_all_regression_metrics(
                        y_true, y_pred, prefix=""
                    )
                    results.update(primary_metrics)

        if "r2" in results:
            logger.info(
                f"Evaluation [{split}]: R\u00b2={results['r2']:.4f}, "
                f"IC={results.get('ic', 0):.4f}, "
                f"MAE={results.get('mae', 0):.4f}, "
                f"DA={results.get('directional_accuracy', 0):.4f}"
            )

        for h in horizons:
            if f"h{h}_r2" in results:
                logger.info(
                    f"  H{h}: R\u00b2={results[f'h{h}_r2']:.4f}, "
                    f"IC={results.get(f'h{h}_ic', 0):.4f}, "
                    f"MAE={results.get(f'h{h}_mae', 0):.4f}"
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
            model: HMHP-R model for inference.
            features: Input tensor [B, T, F] on device.
            return_proba: Ignored for regression.

        Returns:
            np.ndarray of predicted returns [B].
        """
        model.eval()
        output = model(features)
        return output.predictions.cpu().numpy()
