"""
Strategy-aware metrics for LOB model training.

This module provides metrics that understand the semantic meaning of different
labeling strategies, enabling accurate monitoring and evaluation.

Design principles (per RULE.md):
- Metrics match the labeling strategy semantics
- All formulas are documented with clear definitions
- No misleading metrics that game high numbers

Usage:
    >>> from lobtrainer.training.metrics import MetricsCalculator
    >>> from lobtrainer.config import LabelingStrategy
    >>> 
    >>> calculator = MetricsCalculator(LabelingStrategy.TRIPLE_BARRIER)
    >>> metrics = calculator.compute(predictions, labels)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor


# =============================================================================
# Class Name Constants (per labeling strategy)
# =============================================================================

OPPORTUNITY_CLASS_NAMES = ["BigDown", "NoOpportunity", "BigUp"]
TRIPLE_BARRIER_CLASS_NAMES = ["StopLoss", "Timeout", "ProfitTarget"]
TLOB_CLASS_NAMES = ["Down", "Stable", "Up"]
BINARY_CLASS_NAMES = ["NoSignal", "Signal"]


def get_class_names(strategy: str, num_classes: int = 3) -> List[str]:
    """
    Get human-readable class names for a labeling strategy.
    
    Args:
        strategy: Labeling strategy name (opportunity, triple_barrier, tlob)
        num_classes: Number of classes (2 for binary, 3 for multiclass)
    
    Returns:
        List of class names indexed by class ID.
    """
    if num_classes == 2:
        return BINARY_CLASS_NAMES
    
    strategy = strategy.lower()
    if strategy == "opportunity":
        return OPPORTUNITY_CLASS_NAMES
    elif strategy == "triple_barrier":
        return TRIPLE_BARRIER_CLASS_NAMES
    elif strategy == "tlob":
        return TLOB_CLASS_NAMES
    else:
        return [f"Class_{i}" for i in range(num_classes)]


# =============================================================================
# Metrics Dataclass
# =============================================================================


@dataclass
class ClassificationMetrics:
    """
    Comprehensive classification metrics with strategy-aware semantics.
    
    All metrics are computed per-class AND aggregated.
    """
    
    # Overall metrics
    accuracy: float = 0.0
    """Overall accuracy: correct / total."""
    
    loss: float = 0.0
    """Average loss value."""
    
    # Per-class metrics (indexed by class ID)
    per_class_precision: Dict[int, float] = field(default_factory=dict)
    """Precision per class: TP / (TP + FP)."""
    
    per_class_recall: Dict[int, float] = field(default_factory=dict)
    """Recall per class: TP / (TP + FN)."""
    
    per_class_f1: Dict[int, float] = field(default_factory=dict)
    """F1 score per class: 2 * (P * R) / (P + R)."""
    
    per_class_count: Dict[int, int] = field(default_factory=dict)
    """Sample count per class (ground truth)."""
    
    per_class_predicted_count: Dict[int, int] = field(default_factory=dict)
    """Predicted count per class."""
    
    # Macro averages
    macro_precision: float = 0.0
    """Macro-averaged precision (mean of per-class)."""
    
    macro_recall: float = 0.0
    """Macro-averaged recall (mean of per-class)."""
    
    macro_f1: float = 0.0
    """Macro-averaged F1 (mean of per-class F1)."""
    
    # Confusion matrix (flattened)
    confusion_matrix: Optional[np.ndarray] = None
    """Confusion matrix: confusion_matrix[true][pred]."""
    
    # Strategy-specific metrics
    strategy_metrics: Dict[str, float] = field(default_factory=dict)
    """Strategy-specific metrics (e.g., win_rate for triple_barrier)."""
    
    # Class names for reporting
    class_names: List[str] = field(default_factory=list)
    """Human-readable class names."""
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to flat dictionary for logging."""
        result = {
            "accuracy": self.accuracy,
            "loss": self.loss,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
        }
        
        # Add per-class metrics with class names
        for class_id, precision in self.per_class_precision.items():
            name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
            result[f"{name.lower()}_precision"] = precision
            
        for class_id, recall in self.per_class_recall.items():
            name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
            result[f"{name.lower()}_recall"] = recall
            
        for class_id, f1 in self.per_class_f1.items():
            name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
            result[f"{name.lower()}_f1"] = f1
        
        # Add strategy-specific metrics
        result.update(self.strategy_metrics)
        
        return result
    
    def summary(self) -> str:
        """Generate human-readable summary string."""
        lines = [
            f"Accuracy: {self.accuracy:.4f}",
            f"Macro F1: {self.macro_f1:.4f}",
        ]
        
        # Per-class breakdown
        for class_id in sorted(self.per_class_precision.keys()):
            name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class {class_id}"
            p = self.per_class_precision.get(class_id, 0.0)
            r = self.per_class_recall.get(class_id, 0.0)
            f1 = self.per_class_f1.get(class_id, 0.0)
            count = self.per_class_count.get(class_id, 0)
            pred_count = self.per_class_predicted_count.get(class_id, 0)
            lines.append(f"  {name}: P={p:.3f} R={r:.3f} F1={f1:.3f} (n={count}, pred={pred_count})")
        
        # Strategy metrics
        if self.strategy_metrics:
            lines.append("Strategy metrics:")
            for key, value in self.strategy_metrics.items():
                lines.append(f"  {key}: {value:.4f}")
        
        return "\n".join(lines)


# =============================================================================
# Metrics Calculator
# =============================================================================


class MetricsCalculator:
    """
    Strategy-aware metrics calculator.
    
    Computes metrics that match the semantic meaning of the labeling strategy,
    providing meaningful insights for model evaluation.
    
    Args:
        strategy: Labeling strategy (opportunity, triple_barrier, tlob)
        num_classes: Number of classes (default 3)
    
    Example:
        >>> calc = MetricsCalculator("triple_barrier")
        >>> metrics = calc.compute(preds, labels)
        >>> print(metrics.strategy_metrics["win_rate"])
    """
    
    def __init__(self, strategy: str, num_classes: int = 3):
        self.strategy = strategy.lower()
        self.num_classes = num_classes
        self.class_names = get_class_names(strategy, num_classes)
    
    def compute(
        self,
        predictions: Tensor,
        labels: Tensor,
        loss: Optional[float] = None,
        probabilities: Optional[Tensor] = None,
    ) -> ClassificationMetrics:
        """
        Compute comprehensive metrics.
        
        Args:
            predictions: Predicted class indices [N]
            labels: Ground truth class indices [N]
            loss: Optional loss value to include
            probabilities: Optional class probabilities [N, num_classes] for confidence
        
        Returns:
            ClassificationMetrics with all computed values
        """
        # Convert to numpy if needed
        if isinstance(predictions, Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(labels, Tensor):
            labels = labels.cpu().numpy()
        
        predictions = predictions.astype(np.int64)
        labels = labels.astype(np.int64)
        
        n_samples = len(labels)
        if n_samples == 0:
            return ClassificationMetrics(class_names=self.class_names)
        
        # Basic accuracy
        correct = (predictions == labels).sum()
        accuracy = correct / n_samples
        
        # Confusion matrix
        confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        for true_label, pred_label in zip(labels, predictions):
            if 0 <= true_label < self.num_classes and 0 <= pred_label < self.num_classes:
                confusion[true_label, pred_label] += 1
        
        # Per-class metrics
        per_class_precision = {}
        per_class_recall = {}
        per_class_f1 = {}
        per_class_count = {}
        per_class_predicted_count = {}
        
        for c in range(self.num_classes):
            # True positives, false positives, false negatives
            tp = confusion[c, c]
            fp = confusion[:, c].sum() - tp  # Other classes predicted as c
            fn = confusion[c, :].sum() - tp  # Class c predicted as other
            
            # Precision
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            # Recall
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            # F1
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            per_class_precision[c] = precision
            per_class_recall[c] = recall
            per_class_f1[c] = f1
            per_class_count[c] = int(confusion[c, :].sum())
            per_class_predicted_count[c] = int(confusion[:, c].sum())
        
        # Macro averages
        macro_precision = np.mean(list(per_class_precision.values()))
        macro_recall = np.mean(list(per_class_recall.values()))
        macro_f1 = np.mean(list(per_class_f1.values()))
        
        # Strategy-specific metrics
        strategy_metrics = self._compute_strategy_metrics(
            confusion, predictions, labels, probabilities
        )
        
        return ClassificationMetrics(
            accuracy=float(accuracy),
            loss=float(loss) if loss is not None else 0.0,
            per_class_precision=per_class_precision,
            per_class_recall=per_class_recall,
            per_class_f1=per_class_f1,
            per_class_count=per_class_count,
            per_class_predicted_count=per_class_predicted_count,
            macro_precision=float(macro_precision),
            macro_recall=float(macro_recall),
            macro_f1=float(macro_f1),
            confusion_matrix=confusion,
            strategy_metrics=strategy_metrics,
            class_names=self.class_names,
        )
    
    def _compute_strategy_metrics(
        self,
        confusion: np.ndarray,
        predictions: np.ndarray,
        labels: np.ndarray,
        probabilities: Optional[Tensor],
    ) -> Dict[str, float]:
        """Compute strategy-specific metrics based on labeling semantics."""
        metrics = {}
        
        if self.strategy == "triple_barrier":
            metrics.update(self._triple_barrier_metrics(confusion, predictions, labels))
        elif self.strategy == "opportunity":
            metrics.update(self._opportunity_metrics(confusion, predictions, labels))
        elif self.strategy == "tlob":
            metrics.update(self._tlob_metrics(confusion, predictions, labels))
        
        # Signal rate (how often model predicts non-neutral class)
        if self.num_classes == 3:
            # Classes 0 and 2 are "signal" classes in all strategies
            signal_predictions = (predictions == 0) | (predictions == 2)
            metrics["signal_rate"] = signal_predictions.mean()
            
            # Ground truth signal rate
            signal_labels = (labels == 0) | (labels == 2)
            metrics["true_signal_rate"] = signal_labels.mean()
        
        return metrics
    
    def _triple_barrier_metrics(
        self,
        confusion: np.ndarray,
        predictions: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, float]:
        """
        Triple Barrier specific metrics.
        
        Classes: 0=StopLoss, 1=Timeout, 2=ProfitTarget
        
        Note: Per-class precision/recall are already computed by the base calculator.
        This method adds ONLY metrics specific to Triple Barrier trading semantics.
        """
        metrics = {}
        
        # Decisive rate: how often do we predict non-Timeout?
        decisive_preds = (predictions == 0) | (predictions == 2)
        metrics["decisive_prediction_rate"] = decisive_preds.mean()
        
        # Alias for consistency with other strategies
        metrics["signal_rate"] = metrics["decisive_prediction_rate"]
        
        # True decisive rate in labels
        decisive_labels = (labels == 0) | (labels == 2)
        metrics["true_decisive_rate"] = decisive_labels.mean()
        
        # Alias for consistency
        metrics["true_signal_rate"] = metrics["true_decisive_rate"]
        
        # Win rate among decisive samples (ground truth)
        # "Of all samples where a barrier WAS hit, how many were profit targets?"
        if decisive_labels.sum() > 0:
            wins = (labels[decisive_labels] == 2).sum()
            metrics["true_win_rate"] = wins / decisive_labels.sum()
        else:
            metrics["true_win_rate"] = 0.0
        
        # Predicted trade win rate (KEY TRADING METRIC)
        # "Of samples where we predicted decisive (trade), what fraction were actually wins?"
        if decisive_preds.sum() > 0:
            predicted_decisive_labels = labels[decisive_preds]
            actual_wins = (predicted_decisive_labels == 2).sum()
            metrics["predicted_trade_win_rate"] = actual_wins / decisive_preds.sum()
        else:
            metrics["predicted_trade_win_rate"] = 0.0
        
        return metrics
    
    def _opportunity_metrics(
        self,
        confusion: np.ndarray,
        predictions: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, float]:
        """
        Opportunity detection specific metrics.
        
        Classes: 0=BigDown, 1=NoOpportunity, 2=BigUp
        
        Note: Per-class precision/recall are already computed by the base calculator.
        This method adds ONLY metrics specific to opportunity detection semantics.
        """
        metrics = {}
        
        # Signal rate: how often do we predict an opportunity (BigUp or BigDown)?
        opportunity_preds = (predictions == 0) | (predictions == 2)
        opportunity_labels = (labels == 0) | (labels == 2)
        
        metrics["signal_rate"] = opportunity_preds.mean()
        metrics["true_signal_rate"] = opportunity_labels.mean()
        
        # Legacy names for backward compatibility
        metrics["opportunity_prediction_rate"] = metrics["signal_rate"]
        metrics["true_opportunity_rate"] = metrics["true_signal_rate"]
        
        # Directional accuracy (KEY METRIC)
        # "When we predict an opportunity (direction), are we right?"
        if opportunity_preds.sum() > 0:
            directional_preds = predictions[opportunity_preds]
            directional_labels = labels[opportunity_preds]
            directional_correct = (directional_preds == directional_labels).sum()
            metrics["directional_accuracy"] = directional_correct / opportunity_preds.sum()
        else:
            metrics["directional_accuracy"] = 0.0
        
        return metrics
    
    def _tlob_metrics(
        self,
        confusion: np.ndarray,
        predictions: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, float]:
        """
        TLOB/DeepLOB specific metrics.
        
        Classes: 0=Down, 1=Stable, 2=Up
        
        Note: Per-class precision/recall are already computed by the base calculator.
        This method adds ONLY metrics specific to TLOB trading semantics.
        """
        metrics = {}
        
        # Signal rate: how often do we predict non-Stable (i.e., make a directional call)?
        directional_preds = (predictions == 0) | (predictions == 2)
        metrics["signal_rate"] = directional_preds.mean()
        
        # True signal rate (directional labels in ground truth)
        directional_labels = (labels == 0) | (labels == 2)
        metrics["true_signal_rate"] = directional_labels.mean()
        
        # Directional accuracy (KEY METRIC)
        # "When we predict a direction (Up or Down), are we right?"
        if directional_preds.sum() > 0:
            directional_pred_labels = labels[directional_preds]
            directional_pred_preds = predictions[directional_preds]
            correct = (directional_pred_preds == directional_pred_labels).sum()
            metrics["directional_accuracy"] = correct / directional_preds.sum()
        else:
            metrics["directional_accuracy"] = 0.0
        
        return metrics


# =============================================================================
# Convenience Functions
# =============================================================================


def compute_metrics(
    predictions: Tensor,
    labels: Tensor,
    strategy: str = "triple_barrier",
    num_classes: int = 3,
    loss: Optional[float] = None,
) -> ClassificationMetrics:
    """
    Convenience function to compute strategy-aware metrics.
    
    Args:
        predictions: Predicted class indices [N]
        labels: Ground truth class indices [N]
        strategy: Labeling strategy name
        num_classes: Number of classes
        loss: Optional loss value
    
    Returns:
        ClassificationMetrics instance
    """
    calculator = MetricsCalculator(strategy, num_classes)
    return calculator.compute(predictions, labels, loss)


def compute_confusion_matrix(
    predictions: Tensor,
    labels: Tensor,
    num_classes: int = 3,
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        predictions: Predicted class indices [N]
        labels: Ground truth class indices [N]
        num_classes: Number of classes
    
    Returns:
        Confusion matrix [num_classes, num_classes] where [i,j] = count of true=i, pred=j
    """
    if isinstance(predictions, Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, Tensor):
        labels = labels.cpu().numpy()
    
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(labels, predictions):
        if 0 <= true_label < num_classes and 0 <= pred_label < num_classes:
            confusion[true_label, pred_label] += 1
    
    return confusion


# =============================================================================
# Backward Compatibility: Per-Class Metrics Dataclass
# =============================================================================


@dataclass 
class PerClassMetrics:
    """Per-class metrics (backward compatibility)."""
    
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    support: int = 0


# =============================================================================
# Backward Compatibility Functions
# =============================================================================


def compute_accuracy(predictions: Tensor, labels: Tensor) -> float:
    """Compute accuracy (backward compatibility)."""
    if isinstance(predictions, Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, Tensor):
        labels = labels.cpu().numpy()
    return float((predictions == labels).mean())


def compute_classification_report(
    predictions: Tensor,
    labels: Tensor,
    num_classes: int = 3,
    class_names: Optional[List[str]] = None,
) -> Dict[str, PerClassMetrics]:
    """Compute per-class classification report (backward compatibility)."""
    if isinstance(predictions, Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, Tensor):
        labels = labels.cpu().numpy()
    
    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]
    
    report = {}
    confusion = compute_confusion_matrix(predictions, labels, num_classes)
    
    for c in range(num_classes):
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        name = class_names[c] if c < len(class_names) else f"class_{c}"
        report[name] = PerClassMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            support=int(confusion[c, :].sum()),
        )
    
    return report


def compute_trading_metrics(
    predictions: Tensor,
    labels: Tensor,
    num_classes: int = 3,
) -> Dict[str, float]:
    """
    Compute trading-specific metrics (backward compatibility).
    
    Returns signal rate, directional accuracy, up/down precision.
    """
    if isinstance(predictions, Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, Tensor):
        labels = labels.cpu().numpy()
    
    metrics = {}
    
    # Signal rate
    if num_classes == 3:
        signal_preds = (predictions == 0) | (predictions == 2)
        metrics["signal_rate"] = float(signal_preds.mean())
        
        # Directional accuracy
        if signal_preds.sum() > 0:
            directional_correct = (predictions[signal_preds] == labels[signal_preds]).sum()
            metrics["directional_accuracy"] = float(directional_correct / signal_preds.sum())
        else:
            metrics["directional_accuracy"] = 0.0
        
        # Up/Down precision
        up_preds = predictions == 2
        down_preds = predictions == 0
        
        if up_preds.sum() > 0:
            metrics["up_precision"] = float(((predictions == 2) & (labels == 2)).sum() / up_preds.sum())
        else:
            metrics["up_precision"] = 0.0
            
        if down_preds.sum() > 0:
            metrics["down_precision"] = float(((predictions == 0) & (labels == 0)).sum() / down_preds.sum())
        else:
            metrics["down_precision"] = 0.0
    
    return metrics


def compute_transition_accuracy(
    predictions: Tensor,
    labels: Tensor,
) -> float:
    """
    Compute accuracy on label transitions (backward compatibility).
    
    Only evaluates samples where label changed from previous sample.
    """
    if isinstance(predictions, Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, Tensor):
        labels = labels.cpu().numpy()
    
    if len(labels) < 2:
        return 0.0
    
    # Find transitions
    transitions = np.where(labels[1:] != labels[:-1])[0] + 1
    
    if len(transitions) == 0:
        return 0.0
    
    return float((predictions[transitions] == labels[transitions]).mean())
