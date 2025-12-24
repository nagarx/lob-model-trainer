"""
Evaluation metrics for LOB price prediction.

Metrics are designed for 3-class classification (Down, Stable, Up).

Label encoding:
    -1 = Down (bearish)
     0 = Stable (neutral)
    +1 = Up (bullish)

Design principles (RULE.md):
- Consistent metric computation across all models
- Handle edge cases (empty predictions, missing classes)
- Provide both aggregate and per-class metrics
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix as sklearn_confusion_matrix,
    classification_report as sklearn_classification_report,
)

from lobtrainer.constants import (
    LABEL_DOWN,
    LABEL_STABLE,
    LABEL_UP,
    LABEL_NAMES,
    SHIFTED_LABEL_DOWN,
    SHIFTED_LABEL_STABLE,
    SHIFTED_LABEL_UP,
    SHIFTED_LABEL_NAMES,
    get_label_name,
)


# =============================================================================
# Dataclasses for structured results
# =============================================================================


@dataclass
class PerClassMetrics:
    """Metrics for a single class."""
    
    label: int
    """Label value (-1, 0, 1)."""
    
    name: str
    """Label name (Down, Stable, Up)."""
    
    precision: float
    """Precision for this class."""
    
    recall: float
    """Recall for this class."""
    
    f1: float
    """F1 score for this class."""
    
    support: int
    """Number of true instances of this class."""


@dataclass
class ClassificationMetrics:
    """
    Comprehensive classification metrics.
    
    Includes:
    - Overall accuracy
    - Per-class precision, recall, F1
    - Macro and weighted averages
    - Confusion matrix
    """
    
    accuracy: float
    """Overall accuracy (correct / total)."""
    
    macro_precision: float
    """Unweighted mean precision across classes."""
    
    macro_recall: float
    """Unweighted mean recall across classes."""
    
    macro_f1: float
    """Unweighted mean F1 across classes."""
    
    weighted_precision: float
    """Support-weighted mean precision."""
    
    weighted_recall: float
    """Support-weighted mean recall."""
    
    weighted_f1: float
    """Support-weighted mean F1."""
    
    per_class: List[PerClassMetrics]
    """Metrics for each class."""
    
    confusion_matrix: np.ndarray
    """Confusion matrix of shape (n_classes, n_classes)."""
    
    n_samples: int
    """Total number of samples evaluated."""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "accuracy": self.accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "weighted_precision": self.weighted_precision,
            "weighted_recall": self.weighted_recall,
            "weighted_f1": self.weighted_f1,
            "per_class": {
                pc.name: {
                    "precision": pc.precision,
                    "recall": pc.recall,
                    "f1": pc.f1,
                    "support": pc.support,
                }
                for pc in self.per_class
            },
            "confusion_matrix": self.confusion_matrix.tolist(),
            "n_samples": self.n_samples,
        }
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Accuracy: {self.accuracy:.4f} ({self.n_samples} samples)",
            f"Macro F1: {self.macro_f1:.4f}",
            f"Weighted F1: {self.weighted_f1:.4f}",
            "",
            "Per-class metrics:",
        ]
        for pc in self.per_class:
            lines.append(
                f"  {pc.name:>6}: P={pc.precision:.3f} R={pc.recall:.3f} "
                f"F1={pc.f1:.3f} (n={pc.support})"
            )
        return "\n".join(lines)


# =============================================================================
# Core metric functions
# =============================================================================


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute accuracy.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        Accuracy score in [0, 1]
    """
    return float(accuracy_score(y_true, y_pred))


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Label order (default: [-1, 0, 1])
    
    Returns:
        Confusion matrix of shape (n_classes, n_classes)
        Row i = true class i, Column j = predicted class j
    """
    if labels is None:
        labels = [LABEL_DOWN, LABEL_STABLE, LABEL_UP]
    
    return sklearn_confusion_matrix(y_true, y_pred, labels=labels)


def compute_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[int]] = None,
    shifted: bool = False,
) -> ClassificationMetrics:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Label order. Default depends on `shifted`:
            - shifted=False: [-1, 0, 1] (original encoding)
            - shifted=True: [0, 1, 2] (PyTorch encoding)
        shifted: If True, use shifted label names (0=Down, 1=Stable, 2=Up).
            Set this to True when using labels directly from PyTorch DataLoader
            which shifts {-1,0,1} to {0,1,2} for CrossEntropyLoss.
    
    Returns:
        ClassificationMetrics with all metrics
    
    Note:
        The label names in per_class metrics will correctly show:
        - shifted=False: -1="Down", 0="Stable", 1="Up"
        - shifted=True:   0="Down", 1="Stable", 2="Up"
    """
    # Set default labels based on encoding
    if labels is None:
        if shifted:
            labels = [SHIFTED_LABEL_DOWN, SHIFTED_LABEL_STABLE, SHIFTED_LABEL_UP]
        else:
            labels = [LABEL_DOWN, LABEL_STABLE, LABEL_UP]
    
    # Select correct label names mapping
    label_names = SHIFTED_LABEL_NAMES if shifted else LABEL_NAMES
    
    # Handle edge case of empty arrays
    if len(y_true) == 0 or len(y_pred) == 0:
        return ClassificationMetrics(
            accuracy=0.0,
            macro_precision=0.0,
            macro_recall=0.0,
            macro_f1=0.0,
            weighted_precision=0.0,
            weighted_recall=0.0,
            weighted_f1=0.0,
            per_class=[],
            confusion_matrix=np.zeros((len(labels), len(labels)), dtype=np.int64),
            n_samples=0,
        )
    
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Macro averages (unweighted)
    macro_precision = precision_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
    
    # Weighted averages (by support)
    weighted_precision = precision_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0)
    weighted_recall = recall_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0)
    
    # Per-class metrics
    per_class_precision = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    
    # Support (count of true instances per class)
    unique, counts = np.unique(y_true, return_counts=True)
    support_dict = dict(zip(unique, counts))
    
    per_class = []
    for i, label in enumerate(labels):
        per_class.append(PerClassMetrics(
            label=label,
            name=label_names.get(label, str(label)),
            precision=float(per_class_precision[i]),
            recall=float(per_class_recall[i]),
            f1=float(per_class_f1[i]),
            support=int(support_dict.get(label, 0)),
        ))
    
    # Confusion matrix
    cm = sklearn_confusion_matrix(y_true, y_pred, labels=labels)
    
    return ClassificationMetrics(
        accuracy=float(accuracy),
        macro_precision=float(macro_precision),
        macro_recall=float(macro_recall),
        macro_f1=float(macro_f1),
        weighted_precision=float(weighted_precision),
        weighted_recall=float(weighted_recall),
        weighted_f1=float(weighted_f1),
        per_class=per_class,
        confusion_matrix=cm,
        n_samples=len(y_true),
    )


# =============================================================================
# Specialized metrics for trading
# =============================================================================


def compute_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute directional accuracy (ignoring Stable class).
    
    This measures how well the model predicts Up vs Down,
    excluding samples where true label is Stable.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        Directional accuracy (accuracy on non-Stable samples)
    """
    mask = y_true != LABEL_STABLE
    if mask.sum() == 0:
        return 0.0
    return float(accuracy_score(y_true[mask], y_pred[mask]))


def compute_trading_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute trading-specific metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        Dict with:
        - directional_accuracy: Accuracy on non-Stable samples
        - up_precision: Precision when predicting Up
        - down_precision: Precision when predicting Down
        - signal_rate: Fraction of non-Stable predictions
    """
    # Directional accuracy
    directional_mask = y_true != LABEL_STABLE
    directional_acc = accuracy_score(y_true[directional_mask], y_pred[directional_mask]) if directional_mask.sum() > 0 else 0.0
    
    # Up precision: when we predict Up, how often is it correct?
    up_pred_mask = y_pred == LABEL_UP
    up_precision = (y_true[up_pred_mask] == LABEL_UP).mean() if up_pred_mask.sum() > 0 else 0.0
    
    # Down precision: when we predict Down, how often is it correct?
    down_pred_mask = y_pred == LABEL_DOWN
    down_precision = (y_true[down_pred_mask] == LABEL_DOWN).mean() if down_pred_mask.sum() > 0 else 0.0
    
    # Signal rate: how often do we make a directional prediction?
    signal_rate = (y_pred != LABEL_STABLE).mean()
    
    return {
        "directional_accuracy": float(directional_acc),
        "up_precision": float(up_precision),
        "down_precision": float(down_precision),
        "signal_rate": float(signal_rate),
    }


# =============================================================================
# Transition-based metrics (for temporal evaluation)
# =============================================================================


def compute_transition_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute accuracy conditioned on label transitions.
    
    This measures performance on samples where the label changed
    from the previous timestep (harder to predict).
    
    Args:
        y_true: True labels (temporally ordered)
        y_pred: Predicted labels
    
    Returns:
        Dict with:
        - overall_accuracy: Standard accuracy
        - transition_accuracy: Accuracy on samples where y[t] != y[t-1]
        - stable_accuracy: Accuracy on samples where y[t] == y[t-1]
    """
    overall_acc = accuracy_score(y_true, y_pred)
    
    # Identify transitions
    transitions = np.concatenate([[False], y_true[1:] != y_true[:-1]])
    
    # Transition accuracy (harder samples)
    transition_mask = transitions
    transition_acc = accuracy_score(y_true[transition_mask], y_pred[transition_mask]) if transition_mask.sum() > 0 else 0.0
    
    # Stable accuracy (easier samples)
    stable_mask = ~transitions
    stable_acc = accuracy_score(y_true[stable_mask], y_pred[stable_mask]) if stable_mask.sum() > 0 else 0.0
    
    return {
        "overall_accuracy": float(overall_acc),
        "transition_accuracy": float(transition_acc),
        "stable_accuracy": float(stable_acc),
        "transition_rate": float(transitions.mean()),
    }

