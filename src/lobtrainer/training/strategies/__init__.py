"""
Concrete training strategies for task-specific training logic.

Each strategy encapsulates batch processing, validation, evaluation,
and prediction for one training paradigm.
"""

from lobtrainer.training.strategies.classification import ClassificationStrategy
from lobtrainer.training.strategies.regression import RegressionStrategy
from lobtrainer.training.strategies.hmhp_classification import (
    HMHPClassificationStrategy,
)
from lobtrainer.training.strategies.hmhp_regression import HMHPRegressionStrategy

__all__ = [
    "ClassificationStrategy",
    "RegressionStrategy",
    "HMHPClassificationStrategy",
    "HMHPRegressionStrategy",
]
