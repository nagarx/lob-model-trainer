"""
HMHP Classification training strategy.

Handles multi-horizon classification with the Hierarchical Multi-Horizon
Predictor. Labels are {horizon: class_tensor} dicts. Per-horizon losses,
accuracy, agreement ratio, and consistency loss are tracked.

Batch format: (features [B,T,F], labels {h: [B]}, optional regression_targets {h: [B]})
Loss: model.compute_loss(output, labels_dict, regression_targets=reg_dict)
Metrics: per-horizon accuracy/loss, agreement ratio, confirmation score, consistency loss
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lobtrainer.training.strategy import BatchResult, TrainingStrategy

logger = logging.getLogger(__name__)


class HMHPClassificationStrategy(TrainingStrategy):
    """Training strategy for HMHP multi-horizon classification.

    The HMHP model receives all horizons simultaneously and produces:
    - Per-horizon logits (horizon_logits)
    - Ensemble logits (logits)
    - Agreement ratio and confirmation score
    - Consistency loss as an auxiliary training signal
    """

    @property
    def requires_dict_labels(self) -> bool:
        return True

    @property
    def requires_regression_targets(self) -> bool:
        """True if HMHP has regression heads enabled."""
        return (
            getattr(self.config.model, "hmhp_use_regression", False)
            or self.config.model.model_type.value == "hmhp_regression"
        )

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
        """Process one HMHP classification batch.

        Args:
            model: HMHP model in train mode.
            batch_data: (features, labels_dict) or (features, labels_dict, reg_targets_dict).

        Returns:
            BatchResult with total loss, per-horizon losses, accuracy, agreement.
        """
        # T10: detect sample weights (scalar tensor) vs regression targets (dict)
        sample_weights = None
        regression_targets = None
        if len(batch_data) >= 3:
            # Element 2 is regression targets (dict) or sample weight (tensor)
            third = batch_data[2]
            if isinstance(third, dict):
                regression_targets = {
                    h: t.to(self.device) for h, t in third.items()
                }
            elif isinstance(third, torch.Tensor) and third.ndim == 1:
                sample_weights = third
        if len(batch_data) >= 4:
            # Element 3 is sample weight when regression targets are at [2]
            fourth = batch_data[3]
            if isinstance(fourth, torch.Tensor) and fourth.ndim == 1:
                sample_weights = fourth
        if len(batch_data) < 3:
            pass  # just (features, labels)

        features = batch_data[0]
        labels = batch_data[1]

        features = features.to(self.device)
        labels = {h: l.to(self.device) for h, l in labels.items()}
        if sample_weights is not None:
            sample_weights = sample_weights.to(self.device)

        output = model(features)
        total_batch_loss, loss_components = model.compute_loss(
            output, labels, regression_targets=regression_targets
        )
        if sample_weights is not None:
            total_batch_loss = total_batch_loss * sample_weights.mean()

        batch_size = features.size(0)

        # Per-horizon losses
        metrics: Dict[str, float] = {"loss": total_batch_loss.item()}
        for key, value in loss_components.items():
            if key.startswith("H"):
                metrics[f"{key}_loss"] = value.item()
            elif key == "consistency":
                metrics["consistency_loss"] = value.item()

        # Accuracy from ensemble prediction vs first horizon labels
        predictions = output.logits.argmax(dim=1)
        first_h = self.horizons[0]
        correct = (predictions == labels[first_h]).sum().item()
        metrics["correct_count"] = correct

        # Agreement ratio
        metrics["agreement_mean"] = output.agreement.mean().item()

        return BatchResult(
            loss=total_batch_loss,
            batch_size=batch_size,
            metrics=metrics,
        )

    def aggregate_epoch_metrics(
        self,
        results: List[BatchResult],
        total_samples: int,
    ) -> Dict[str, float]:
        """Aggregate into epoch metrics with per-horizon detail.

        Returns:
            Dict with train_loss, train_accuracy, train_h{x}_loss,
            train_consistency_loss, train_agreement_ratio.
        """
        if total_samples == 0:
            return {"train_loss": 0.0}

        total_loss = sum(r.metrics["loss"] * r.batch_size for r in results)
        total_correct = sum(r.metrics["correct_count"] for r in results)

        result: Dict[str, float] = {
            "train_loss": total_loss / total_samples,
            "train_accuracy": total_correct / total_samples,
        }

        # Per-horizon losses
        for h in self.horizons:
            key = f"H{h}_loss"
            total = sum(
                r.metrics.get(key, 0.0) * r.batch_size for r in results
            )
            result[f"train_h{h}_loss"] = total / total_samples

        # Consistency loss
        total_consistency = sum(
            r.metrics.get("consistency_loss", 0.0) * r.batch_size for r in results
        )
        result["train_consistency_loss"] = total_consistency / total_samples

        # Agreement ratio
        total_agreement = sum(
            r.metrics.get("agreement_mean", 0.0) * r.batch_size for r in results
        )
        result["train_agreement_ratio"] = total_agreement / total_samples

        return result

    @torch.no_grad()
    def validate(self, model: nn.Module, loader: DataLoader) -> Dict[str, float]:
        """HMHP-specific validation with per-horizon metrics.

        Returns dict with val_loss, val_accuracy, per-horizon loss/accuracy,
        val_agreement_ratio, val_confirmation_score, val_consistency_loss.
        """
        model.eval()
        horizons = self.horizons

        total_loss = 0.0
        total_samples = 0
        horizon_losses = {h: 0.0 for h in horizons}
        horizon_correct = {h: 0 for h in horizons}
        final_correct = 0
        agreement_sum = 0.0
        confirmation_sum = 0.0
        consistency_loss_sum = 0.0

        for batch_data in loader:
            if len(batch_data) == 3:
                features, labels, regression_targets = batch_data
                regression_targets = {
                    h: t.to(self.device) for h, t in regression_targets.items()
                }
            else:
                features, labels = batch_data
                regression_targets = None

            features = features.to(self.device)
            labels = {h: l.to(self.device) for h, l in labels.items()}

            output = model(features)
            total_batch_loss, loss_components = model.compute_loss(
                output, labels, regression_targets=regression_targets
            )

            batch_size = features.size(0)
            total_loss += total_batch_loss.item() * batch_size
            total_samples += batch_size

            for h in horizons:
                h_key = f"H{h}"
                if h_key in loss_components:
                    horizon_losses[h] += loss_components[h_key].item() * batch_size
                h_preds = output.horizon_logits[h].argmax(dim=1)
                horizon_correct[h] += (h_preds == labels[h]).sum().item()

            final_preds = output.logits.argmax(dim=1)
            first_h = horizons[0]
            final_correct += (final_preds == labels[first_h]).sum().item()

            agreement_sum += output.agreement.sum().item()
            confirmation_sum += output.confidence.sum().item()

            if "consistency" in loss_components:
                consistency_loss_sum += (
                    loss_components["consistency"].item() * batch_size
                )

        if total_samples == 0:
            return {"val_loss": float("inf")}

        result = {
            "val_loss": total_loss / total_samples,
            "val_accuracy": final_correct / total_samples,
            "val_agreement_ratio": agreement_sum / total_samples,
            "val_confirmation_score": confirmation_sum / total_samples,
            "val_consistency_loss": consistency_loss_sum / total_samples,
        }

        for h in horizons:
            result[f"val_h{h}_loss"] = horizon_losses[h] / total_samples
            result[f"val_h{h}_accuracy"] = horizon_correct[h] / total_samples

        return result

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
        split: str,
    ) -> Any:
        """Full evaluation returning ClassificationMetrics with HMHP extras.

        Returns ClassificationMetrics with per-horizon accuracy, agreement,
        and confirmation score injected into strategy_metrics.
        """
        from lobtrainer.training.metrics import MetricsCalculator

        model.eval()
        horizons = self.horizons
        first_h = horizons[0]

        all_predictions = []
        all_labels = []
        per_horizon_correct = {h: 0 for h in horizons}
        per_horizon_total = 0
        agreement_sum = 0.0
        confirmation_sum = 0.0

        for batch_data in loader:
            if len(batch_data) == 3:
                features, labels, _regression_targets = batch_data
            else:
                features, labels = batch_data
            features = features.to(self.device)
            labels = {h: l.to(self.device) for h, l in labels.items()}
            output = model(features)

            final_preds = output.logits.argmax(dim=1)
            all_predictions.append(final_preds.cpu().numpy())
            all_labels.append(labels[first_h].cpu().numpy())

            batch_size = features.size(0)
            per_horizon_total += batch_size
            for h in horizons:
                h_preds = output.horizon_logits[h].argmax(dim=1)
                per_horizon_correct[h] += (h_preds == labels[h]).sum().item()

            agreement_sum += output.agreement.sum().item()
            confirmation_sum += output.confidence.sum().item()

        y_pred = np.concatenate(all_predictions)
        y_true = np.concatenate(all_labels)

        strategy = getattr(self.config.data, "labeling_strategy", "opportunity")
        if hasattr(strategy, "value"):
            strategy = strategy.value

        metrics_calculator = MetricsCalculator(strategy, self.config.model.num_classes)
        metrics = metrics_calculator.compute(y_pred, y_true)

        logger.info(
            f"Evaluation [{split}]: accuracy={metrics.accuracy:.4f}, "
            f"macro_f1={metrics.macro_f1:.4f}"
        )

        # Inject HMHP-specific metrics
        for h in horizons:
            h_acc = per_horizon_correct[h] / per_horizon_total
            metrics.strategy_metrics[f"h{h}_accuracy"] = h_acc
            logger.info(f"  H{h} accuracy: {h_acc:.4f}")

        metrics.strategy_metrics["agreement_ratio"] = (
            agreement_sum / per_horizon_total
        )
        metrics.strategy_metrics["confirmation_score"] = (
            confirmation_sum / per_horizon_total
        )
        logger.info(
            f"  Agreement ratio: {agreement_sum / per_horizon_total:.4f}, "
            f"Confirmation score: {confirmation_sum / per_horizon_total:.4f}"
        )

        if metrics.strategy_metrics:
            strategy_str = ", ".join(
                f"{k}={v:.4f}"
                for k, v in list(metrics.strategy_metrics.items())[:5]
            )
            logger.info(f"  Strategy metrics: {strategy_str}")

        return metrics

    @torch.no_grad()
    def predict(
        self,
        model: nn.Module,
        features: torch.Tensor,
        return_proba: bool = False,
    ) -> np.ndarray:
        """Predict using ensemble (final) logits.

        Args:
            model: HMHP model for inference.
            features: Input tensor [B, T, F] on device.
            return_proba: If True, return softmax probabilities [B, C].

        Returns:
            np.ndarray of predictions [B] or probabilities [B, C].
        """
        model.eval()
        output = model(features)
        logits = output.logits

        if return_proba:
            return torch.softmax(logits, dim=1).cpu().numpy()
        else:
            return logits.argmax(dim=1).cpu().numpy()
