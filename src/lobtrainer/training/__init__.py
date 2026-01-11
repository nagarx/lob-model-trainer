"""
Training infrastructure for LOB models.

Provides:
- Trainer: Configuration-driven training loop with callbacks
- Callbacks: EarlyStopping, ModelCheckpoint, MetricLogger
- Metrics: Accuracy, precision, recall, F1, confusion matrix
- Evaluation: Model evaluation with comprehensive reporting

Design principles (RULE.md):
- Configuration-driven training
- Deterministic with seed management
- Extensible via callbacks
- Comprehensive reporting for experiment tracking

Usage:
    >>> from lobtrainer.training import Trainer, create_trainer
    >>> from lobtrainer.config import load_config
    >>> 
    >>> config = load_config("configs/baseline_lstm.yaml")
    >>> trainer = create_trainer(config)
    >>> trainer.train()
    >>> metrics = trainer.evaluate("test")
"""

from lobtrainer.training.metrics import (
    # New strategy-aware metrics
    MetricsCalculator,
    ClassificationMetrics,
    compute_metrics,
    get_class_names,
    TRIPLE_BARRIER_CLASS_NAMES,
    OPPORTUNITY_CLASS_NAMES,
    TLOB_CLASS_NAMES,
    BINARY_CLASS_NAMES,
    # Backward compatibility
    compute_accuracy,
    compute_classification_report,
    compute_confusion_matrix,
    compute_trading_metrics,
    compute_transition_accuracy,
    PerClassMetrics,
)

from lobtrainer.training.evaluation import (
    evaluate_model,
    evaluate_naive_baseline,
    create_baseline_report,
    full_evaluation,
    BaselineReport,
)

from lobtrainer.training.callbacks import (
    Callback,
    CallbackList,
    EarlyStopping,
    ModelCheckpoint,
    MetricLogger,
    ProgressCallback,
)

from lobtrainer.training.trainer import (
    Trainer,
    TrainingState,
    create_trainer,
)

from lobtrainer.training.loss import (
    FocalLoss,
    BinaryFocalLoss,
    create_focal_loss,
)

__all__ = [
    # Core Trainer
    "Trainer",
    "TrainingState",
    "create_trainer",
    # Loss Functions
    "FocalLoss",
    "BinaryFocalLoss",
    "create_focal_loss",
    # Callbacks
    "Callback",
    "CallbackList",
    "EarlyStopping",
    "ModelCheckpoint",
    "MetricLogger",
    "ProgressCallback",
    # Strategy-aware Metrics (NEW)
    "MetricsCalculator",
    "compute_metrics",
    "get_class_names",
    "TRIPLE_BARRIER_CLASS_NAMES",
    "OPPORTUNITY_CLASS_NAMES",
    "TLOB_CLASS_NAMES",
    "BINARY_CLASS_NAMES",
    # Metrics (backward compatibility)
    "compute_accuracy",
    "compute_classification_report",
    "compute_confusion_matrix",
    "compute_trading_metrics",
    "compute_transition_accuracy",
    "ClassificationMetrics",
    "PerClassMetrics",
    # Evaluation
    "evaluate_model",
    "evaluate_naive_baseline",
    "create_baseline_report",
    "full_evaluation",
    "BaselineReport",
]
