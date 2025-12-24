"""
Model evaluation framework for LOB price prediction.

Provides:
- evaluate_model: Evaluate any model with comprehensive metrics
- evaluate_naive_baseline: Special evaluation for naive previous-label baseline
- BaselineReport: Structured report comparing model vs baselines

Design principles (RULE.md):
- Consistent evaluation across all models
- Include naive baselines for comparison
- Comprehensive reporting for experiment tracking
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
import logging
import numpy as np

from lobtrainer.models.baselines import BaseModel, NaivePreviousLabel, NaiveClassPrior
from lobtrainer.training.metrics import (
    compute_accuracy,
    compute_classification_report,
    compute_trading_metrics,
    compute_transition_accuracy,
    ClassificationMetrics,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Evaluation Functions
# =============================================================================


def evaluate_model(
    model: BaseModel,
    X: np.ndarray,
    y: np.ndarray,
    name: Optional[str] = None,
) -> ClassificationMetrics:
    """
    Evaluate a model on given data.
    
    Args:
        model: Model with predict() method
        X: Features
        y: True labels
        name: Optional name for logging
    
    Returns:
        ClassificationMetrics with all metrics
    """
    model_name = name or model.name
    
    # Get predictions
    y_pred = model.predict(X)
    
    # Compute metrics
    metrics = compute_classification_report(y, y_pred)
    
    logger.info(f"{model_name}: accuracy={metrics.accuracy:.4f}, macro_f1={metrics.macro_f1:.4f}")
    
    return metrics


def evaluate_naive_baseline(
    y_true: np.ndarray,
    split_name: str = "test",
) -> Dict[str, ClassificationMetrics]:
    """
    Evaluate naive baselines on given labels.
    
    Baselines evaluated:
    1. NaiveClassPrior: Always predict most common class
    2. NaivePreviousLabel: Predict previous label (requires temporal ordering)
    
    Args:
        y_true: True labels (must be temporally ordered for previous-label baseline)
        split_name: Name for logging
    
    Returns:
        Dict with 'class_prior' and 'previous_label' metrics
    """
    results = {}
    
    # 1. Class prior baseline
    class_prior = NaiveClassPrior()
    # Fit on data to get class distribution
    class_prior.fit(y_true.reshape(-1, 1), y_true)  # X doesn't matter
    y_pred_prior = class_prior.predict(y_true.reshape(-1, 1))
    results['class_prior'] = compute_classification_report(y_true, y_pred_prior)
    
    logger.info(
        f"[{split_name}] ClassPrior: accuracy={results['class_prior'].accuracy:.4f}"
    )
    
    # 2. Previous-label baseline (requires temporal ordering)
    # Predict: y_pred[i] = y_true[i-1]
    y_pred_prev = np.concatenate([[y_true[0]], y_true[:-1]])  # Shift by 1
    results['previous_label'] = compute_classification_report(y_true, y_pred_prev)
    
    logger.info(
        f"[{split_name}] PreviousLabel: accuracy={results['previous_label'].accuracy:.4f}"
    )
    
    return results


# =============================================================================
# Baseline Report
# =============================================================================


@dataclass
class BaselineReport:
    """
    Comprehensive report comparing model against baselines.
    
    This report answers: "Is my model actually learning something,
    or is it just exploiting label autocorrelation?"
    """
    
    model_name: str
    """Name of the evaluated model."""
    
    split: str
    """Data split (train/val/test)."""
    
    n_samples: int
    """Number of samples evaluated."""
    
    # Model metrics
    model_metrics: ClassificationMetrics
    """Full metrics for the model."""
    
    # Baseline metrics
    class_prior_metrics: ClassificationMetrics
    """Metrics for class prior baseline."""
    
    previous_label_metrics: ClassificationMetrics
    """Metrics for previous-label baseline."""
    
    # Computed comparisons
    beats_class_prior: bool = False
    """True if model accuracy > class prior accuracy."""
    
    beats_previous_label: bool = False
    """True if model accuracy > previous label accuracy."""
    
    improvement_over_prior: float = 0.0
    """Absolute improvement over class prior (percentage points)."""
    
    improvement_over_previous: float = 0.0
    """Absolute improvement over previous label (percentage points)."""
    
    def __post_init__(self):
        """Compute derived fields."""
        self.beats_class_prior = self.model_metrics.accuracy > self.class_prior_metrics.accuracy
        self.beats_previous_label = self.model_metrics.accuracy > self.previous_label_metrics.accuracy
        
        self.improvement_over_prior = (
            (self.model_metrics.accuracy - self.class_prior_metrics.accuracy) * 100
        )
        self.improvement_over_previous = (
            (self.model_metrics.accuracy - self.previous_label_metrics.accuracy) * 100
        )
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"{'=' * 60}",
            f"BASELINE REPORT: {self.model_name}",
            f"{'=' * 60}",
            f"Split: {self.split} ({self.n_samples} samples)",
            "",
            "Accuracy Comparison:",
            f"  Model:         {self.model_metrics.accuracy:.4f} ({self.model_metrics.accuracy*100:.1f}%)",
            f"  Class Prior:   {self.class_prior_metrics.accuracy:.4f} ({self.class_prior_metrics.accuracy*100:.1f}%)",
            f"  Prev Label:    {self.previous_label_metrics.accuracy:.4f} ({self.previous_label_metrics.accuracy*100:.1f}%)",
            "",
            "Model vs Baselines:",
            f"  vs Class Prior:   {self.improvement_over_prior:+.2f}pp {'✅ BEATS' if self.beats_class_prior else '❌ LOSES'}",
            f"  vs Prev Label:    {self.improvement_over_previous:+.2f}pp {'✅ BEATS' if self.beats_previous_label else '❌ LOSES'}",
            "",
            "Per-Class F1 (Model):",
        ]
        
        for pc in self.model_metrics.per_class:
            lines.append(f"  {pc.name:>6}: {pc.f1:.3f}")
        
        lines.append("")
        
        if self.beats_previous_label:
            lines.append("✅ Model shows real predictive value (beats persistence baseline)")
        else:
            lines.append("⚠️  Model may just be exploiting label autocorrelation")
        
        lines.append(f"{'=' * 60}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_name": self.model_name,
            "split": self.split,
            "n_samples": self.n_samples,
            "model": self.model_metrics.to_dict(),
            "baselines": {
                "class_prior": self.class_prior_metrics.to_dict(),
                "previous_label": self.previous_label_metrics.to_dict(),
            },
            "comparison": {
                "beats_class_prior": self.beats_class_prior,
                "beats_previous_label": self.beats_previous_label,
                "improvement_over_prior_pp": self.improvement_over_prior,
                "improvement_over_previous_pp": self.improvement_over_previous,
            },
        }
    
    def save(self, path: Union[str, Path]) -> None:
        """Save report to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def create_baseline_report(
    model: BaseModel,
    X: np.ndarray,
    y: np.ndarray,
    split: str = "test",
) -> BaselineReport:
    """
    Create comprehensive baseline comparison report.
    
    Args:
        model: Trained model to evaluate
        X: Features
        y: True labels (must be temporally ordered)
        split: Split name for logging
    
    Returns:
        BaselineReport with full comparison
    """
    # Evaluate model
    model_metrics = evaluate_model(model, X, y)
    
    # Evaluate baselines
    baseline_metrics = evaluate_naive_baseline(y, split)
    
    return BaselineReport(
        model_name=model.name,
        split=split,
        n_samples=len(y),
        model_metrics=model_metrics,
        class_prior_metrics=baseline_metrics['class_prior'],
        previous_label_metrics=baseline_metrics['previous_label'],
    )


# =============================================================================
# Extended evaluation (trading metrics, temporal analysis)
# =============================================================================


def full_evaluation(
    model: BaseModel,
    X: np.ndarray,
    y: np.ndarray,
    split: str = "test",
) -> Dict:
    """
    Full evaluation with all metrics.
    
    Includes:
    - Standard classification metrics
    - Trading-specific metrics
    - Transition analysis
    - Baseline comparison
    
    Args:
        model: Trained model
        X: Features
        y: True labels (temporally ordered)
        split: Split name
    
    Returns:
        Dict with comprehensive evaluation results
    """
    # Get predictions
    y_pred = model.predict(X)
    
    # Standard metrics
    classification = compute_classification_report(y, y_pred)
    
    # Trading metrics
    trading = compute_trading_metrics(y, y_pred)
    
    # Transition analysis
    transitions = compute_transition_accuracy(y, y_pred)
    
    # Baseline comparison
    baseline_report = create_baseline_report(model, X, y, split)
    
    return {
        "split": split,
        "n_samples": len(y),
        "model_name": model.name,
        "classification": classification.to_dict(),
        "trading": trading,
        "transitions": transitions,
        "baseline_comparison": baseline_report.to_dict(),
    }

