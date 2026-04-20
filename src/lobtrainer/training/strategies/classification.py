"""
Classification training strategy.

Handles standard single-horizon classification (TLOB, DeepLOB, MLPLOB,
LogisticLOB, LSTM, GRU) with external criterion (CrossEntropy/Focal).

Batch format: (features [B,T,F], labels [B] int64)
Loss: model.compute_loss(output, labels=labels) or criterion(logits, labels)
Metrics: accuracy, macro_f1, per-class precision, strategy-specific metrics
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lobtrainer.training.strategy import BatchResult, TrainingStrategy

logger = logging.getLogger(__name__)


class ClassificationStrategy(TrainingStrategy):
    """Training strategy for single-horizon classification tasks.

    Handles multiclass (3-class) and binary_signal (2-class) classification.
    Creates and owns the loss criterion (CrossEntropy or Focal with class weights).
    """

    def __init__(self, config, device):
        super().__init__(config, device)
        self._criterion: Optional[nn.Module] = None
        # P0-V1-N1 fix (Phase I.A, 2026-04-20): mirror criterion with reduction='none'
        # built once at initialize() to eliminate the per-batch mutate-save-restore
        # pattern that was unsafe under num_workers > 0 (race on self._criterion.reduction)
        # AND under validation-during-training (re-entrance on the same criterion instance).
        self._criterion_unreduced: Optional[nn.Module] = None

    def initialize(self, train_loader: DataLoader, model: nn.Module) -> None:
        """Compute class weights and create criterion (+ unreduced mirror for weighted loss)."""
        self._criterion = self._create_criterion(train_loader)
        # Build unreduced mirror ONCE per epoch. Shares the same class_weights / focal
        # gamma / alpha as the primary criterion — no per-sample-weighting semantic drift.
        self._criterion_unreduced = self._build_unreduced_mirror(self._criterion)

    @staticmethod
    def _build_unreduced_mirror(primary: nn.Module) -> nn.Module:
        """Construct a reduction='none' criterion mirroring ``primary``'s configuration.

        Invariant: the returned criterion must produce per-sample loss that, when
        averaged (with equal weights), equals the primary criterion's scalar loss.
        This is the reduction-contract for per-sample weighting in process_batch.
        """
        from lobtrainer.training.loss import FocalLoss
        if isinstance(primary, FocalLoss):
            return FocalLoss(
                alpha=getattr(primary, "alpha", None),
                gamma=getattr(primary, "gamma", 2.0),
                reduction="none",
            )
        # nn.CrossEntropyLoss path (weighted or unweighted)
        weight = getattr(primary, "weight", None)
        return nn.CrossEntropyLoss(weight=weight, reduction="none")

    @property
    def criterion(self) -> Optional[nn.Module]:
        """Loss function (created in initialize)."""
        return self._criterion

    # =========================================================================
    # Criterion Creation
    # =========================================================================

    def _create_criterion(self, train_loader: DataLoader) -> nn.Module:
        """Create loss function with class weights from training data.

        Supports:
            - cross_entropy: Unweighted CE
            - weighted_ce: CE with inverse-frequency class weights
            - focal: Focal loss for class imbalance
        """
        from lobtrainer.config import LossType, TaskType
        from lobtrainer.training.loss import FocalLoss

        cfg = self.config.train
        loss_type = getattr(cfg, "loss_type", LossType.WEIGHTED_CE)
        task_type = getattr(cfg, "task_type", TaskType.MULTICLASS)

        if task_type == TaskType.BINARY_SIGNAL:
            num_classes = 2
            class_names = ["NoSignal", "Signal"]
        else:
            num_classes = 3
            class_names = ["Down", "Stable", "Up"]

        if loss_type == LossType.CROSS_ENTROPY:
            logger.info(f"Using unweighted CrossEntropyLoss ({num_classes} classes)")
            return nn.CrossEntropyLoss()

        # Compute class counts from training data
        class_counts = self._compute_class_counts(train_loader, num_classes, class_names)

        if loss_type == LossType.FOCAL:
            gamma = getattr(cfg, "focal_gamma", 2.0)
            weights = None
            if class_counts is not None:
                total = class_counts.sum()
                if total > 0:
                    weights = total / (num_classes * class_counts.clamp(min=1))
                    weights = weights.to(self.device)
            logger.info(f"Using FocalLoss(gamma={gamma}, alpha={weights})")
            return FocalLoss(gamma=gamma, alpha=weights)

        # LossType.WEIGHTED_CE (default)
        if class_counts is not None:
            total = class_counts.sum()
            if total > 0:
                weights = total / (float(num_classes) * class_counts.clamp(min=1))
                weights = weights.to(self.device)
                weight_str = ", ".join(
                    f"{name}={weights[i]:.2f}" for i, name in enumerate(class_names)
                )
                logger.info(f"Using class weights: {weight_str}")
                return nn.CrossEntropyLoss(weight=weights)

        logger.info("Using unweighted CrossEntropyLoss (fallback)")
        return nn.CrossEntropyLoss()

    def _compute_class_counts(
        self,
        train_loader: DataLoader,
        num_classes: int,
        class_names: list,
    ) -> Optional[torch.Tensor]:
        """Count samples per class from training data."""
        try:
            class_counts = torch.zeros(num_classes)
            for batch in train_loader:
                labels = batch[1]  # works for both 2-tuple and 3-tuple (T10)
                for c in range(num_classes):
                    class_counts[c] += (labels == c).sum().item()

            total = class_counts.sum()
            if total > 0:
                for i, name in enumerate(class_names):
                    pct = class_counts[i] / total * 100
                    logger.info(f"  Class {name}: {int(class_counts[i]):,} ({pct:.1f}%)")
            return class_counts
        except Exception as e:
            logger.warning(f"Could not compute class counts: {e}")
            return None

    # =========================================================================
    # Core Methods
    # =========================================================================

    def process_batch(self, model: nn.Module, batch_data: tuple) -> BatchResult:
        """Process one classification batch.

        Args:
            model: Model in train mode.
            batch_data: (features [B,T,F], labels [B]) or
                (features [B,T,F], labels [B], sample_weights [B]).

        Returns:
            BatchResult with cross-entropy loss and correct count.
        """
        if len(batch_data) == 3:
            features, labels, sample_weights = batch_data
            features = features.to(self.device)
            labels = labels.to(self.device)
            sample_weights = sample_weights.to(self.device)

            output = model(features)
            # P0-V1-N1 fix (Phase I.A, 2026-04-20): use the prebuilt reduction='none'
            # mirror criterion instead of mutating self._criterion.reduction per-batch.
            # The old mutate-save-restore pattern was unsafe under num_workers>0
            # (concurrent process_batch calls corrupted the saved_reduction handshake)
            # AND under validation-during-training (re-entrance on same instance).
            # Mirror shares class weights / focal gamma+alpha with primary — same semantics.
            if self._criterion_unreduced is None:
                # Defensive: initialize() should have populated this. Rebuild lazily to
                # preserve backward compatibility with callers that skip initialize().
                self._criterion_unreduced = self._build_unreduced_mirror(self._criterion)
            loss_unreduced = self._criterion_unreduced(output.logits, labels)
            loss = (loss_unreduced * sample_weights).mean()
        else:
            features, labels = batch_data
            features = features.to(self.device)
            labels = labels.to(self.device)

            output = model(features)
            loss = self._criterion(output.logits, labels)

        logits = output.logits
        predictions = logits.argmax(dim=1)
        correct = (predictions == labels).sum().item()

        return BatchResult(
            loss=loss,
            batch_size=features.size(0),
            metrics={
                "loss": loss.item(),
                "correct_count": correct,
            },
        )

    def aggregate_epoch_metrics(
        self,
        results: List[BatchResult],
        total_samples: int,
    ) -> Dict[str, float]:
        """Aggregate batch results into epoch metrics.

        Returns:
            {'train_loss': float, 'train_accuracy': float}
        """
        if total_samples == 0:
            return {"train_loss": 0.0, "train_accuracy": 0.0}

        total_loss = sum(r.metrics["loss"] * r.batch_size for r in results)
        total_correct = sum(r.metrics["correct_count"] for r in results)

        return {
            "train_loss": total_loss / total_samples,
            "train_accuracy": total_correct / total_samples,
        }

    @torch.no_grad()
    def validate(self, model: nn.Module, loader: DataLoader) -> Dict[str, float]:
        """Validate with strategy-aware classification metrics.

        Returns dict with val_loss, val_accuracy, val_macro_f1,
        per-class precision, and labeling-strategy-specific metrics.
        """
        from lobtrainer.training.metrics import MetricsCalculator

        model.eval()

        total_loss = 0.0
        total_samples = 0
        all_preds = []
        all_labels = []

        for features, labels in loader:
            features = features.to(self.device)
            labels = labels.to(self.device)

            output = model(features)
            logits = output.logits
            loss = self._criterion(logits, labels)

            total_loss += loss.item() * features.size(0)
            total_samples += features.size(0)

            predictions = logits.argmax(dim=1)
            all_preds.append(predictions.cpu())
            all_labels.append(labels.cpu())

        if total_samples == 0:
            return {"val_loss": float("inf")}

        avg_loss = total_loss / total_samples
        y_pred = torch.cat(all_preds)
        y_true = torch.cat(all_labels)

        strategy = getattr(self.config.data, "labeling_strategy", "opportunity")
        if hasattr(strategy, "value"):
            strategy = strategy.value

        num_classes = self.config.data.num_classes
        metrics_calculator = MetricsCalculator(strategy, num_classes)
        metrics = metrics_calculator.compute(y_pred, y_true, loss=avg_loss)

        result = {
            "val_loss": avg_loss,
            "val_accuracy": metrics.accuracy,
            "val_macro_f1": metrics.macro_f1,
        }

        for class_id, precision in metrics.per_class_precision.items():
            if class_id < len(metrics.class_names):
                name = metrics.class_names[class_id].lower()
                result[f"val_{name}_precision"] = precision

        for key, value in metrics.strategy_metrics.items():
            result[f"val_{key}"] = value

        # Backward compatibility for TLOB and OPPORTUNITY
        if strategy in ("tlob", "opportunity"):
            if "directional_accuracy" in metrics.strategy_metrics:
                result["val_directional_accuracy"] = metrics.strategy_metrics[
                    "directional_accuracy"
                ]
            if "signal_rate" in metrics.strategy_metrics:
                result["val_signal_rate"] = metrics.strategy_metrics["signal_rate"]

        return result

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
        split: str,
    ) -> Any:
        """Full evaluation returning ClassificationMetrics.

        Args:
            model: Model to evaluate.
            loader: Data loader for the split.
            split: Split name for logging.

        Returns:
            ClassificationMetrics with accuracy, f1, per-class metrics.
        """
        from lobtrainer.training.metrics import MetricsCalculator

        model.eval()

        all_predictions = []
        all_labels = []

        for features, labels in loader:
            features = features.to(self.device)
            output = model(features)
            logits = output.logits
            predictions = logits.argmax(dim=1).cpu().numpy()
            all_predictions.append(predictions)
            all_labels.append(labels.numpy())

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
        """Predict class labels or probabilities.

        Args:
            model: Model for inference.
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
