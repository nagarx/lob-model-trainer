"""
Training strategy abstraction for task-specific training logic.

The Strategy Pattern separates task-specific behavior (batch processing,
validation metrics, evaluation, prediction) from the Trainer orchestrator
(epochs, callbacks, scheduling, checkpointing, data pipeline).

4 concrete strategies:
    - ClassificationStrategy: Standard classification with external criterion
    - RegressionStrategy: Standard regression via model.compute_loss
    - HMHPClassificationStrategy: Multi-horizon classification with dict labels
    - HMHPRegressionStrategy: Multi-horizon regression with per-horizon metrics

Design principles (hft-rules.md):
    - Configuration-driven (Rule 5): Strategy selected from config
    - Single responsibility (Rule 4): Each strategy handles one task type
    - Testable (Rule 6): Strategies testable in isolation with mock model/data
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# =============================================================================
# Batch Result
# =============================================================================


@dataclass
class BatchResult:
    """Result from processing one training batch.

    The Trainer calls strategy.process_batch() which returns this. The Trainer
    then calls loss.backward(), clips gradients, and steps the optimizer.

    Attributes:
        loss: Live tensor for .backward(). NOT detached.
        batch_size: Number of samples in this batch (for weighted averaging).
        metrics: Strategy-specific per-batch values for epoch aggregation.
            Convention: values are per-batch MEANS (loss, agreement_mean, etc.)
            except keys ending in '_count' which are absolute SUMS (correct_count).
    """

    loss: torch.Tensor
    batch_size: int
    metrics: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# Strategy ABC
# =============================================================================


class TrainingStrategy(ABC):
    """Abstract base for task-specific training logic.

    The Trainer orchestrates the outer loop (epochs, callbacks, scheduling).
    The Strategy handles the inner logic (batch processing, metric computation,
    validation loops, evaluation, prediction).

    Strategy lifecycle:
        1. create_strategy(config, device) -> strategy instance
        2. strategy.initialize(train_loader, model) -> one-time setup
        3. Per epoch: strategy.process_batch() -> BatchResult (called per batch)
        4. Per epoch: strategy.aggregate_epoch_metrics() -> epoch dict
        5. Per epoch: strategy.validate() -> val metrics dict
        6. End: strategy.evaluate() -> final metrics
        7. Inference: strategy.predict() -> numpy predictions

    Args:
        config: ExperimentConfig with all settings.
        device: torch.device for tensor placement.
    """

    def __init__(self, config: "ExperimentConfig", device: torch.device):
        self.config = config
        self.device = device

    # =========================================================================
    # Data Pipeline Properties
    # =========================================================================
    # Read by Trainer._create_dataloaders to configure label format.

    @property
    def requires_dict_labels(self) -> bool:
        """True if dataset should return labels as {horizon: tensor} dict."""
        return False

    @property
    def requires_regression_targets(self) -> bool:
        """True if dataset should return regression target tensors."""
        return False

    @property
    def horizon_idx(self) -> Optional[int]:
        """Which horizon to select. None = return all horizons (HMHP)."""
        # T9: read from LabelsConfig (always populated by DataConfig.__post_init__).
        # Defensive: verify the attribute is a real int/None, not a MagicMock.
        labels = getattr(self.config.data, "labels", None)
        if labels is not None:
            idx = getattr(labels, "primary_horizon_idx", None)
            if isinstance(idx, (int, type(None))):
                return idx
        return getattr(self.config.data, "horizon_idx", 0)

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def initialize(
        self,
        train_loader: DataLoader,
        model: nn.Module,
    ) -> None:
        """Called once after dataloaders are created.

        Override to perform one-time setup (e.g., compute class weights
        and create criterion for classification).

        Args:
            train_loader: Training data loader.
            model: The model (already on device).
        """

    # =========================================================================
    # Core Abstract Methods
    # =========================================================================

    @abstractmethod
    def process_batch(
        self,
        model: nn.Module,
        batch_data: tuple,
    ) -> BatchResult:
        """Process one training batch: forward pass + loss computation.

        The Trainer handles optimizer.zero_grad(), loss.backward(),
        gradient clipping, and optimizer.step().

        Args:
            model: Model in train mode, on device.
            batch_data: Tuple from DataLoader (format varies by strategy).

        Returns:
            BatchResult with live loss tensor and per-batch metrics.
        """
        ...

    @abstractmethod
    def aggregate_epoch_metrics(
        self,
        results: List[BatchResult],
        total_samples: int,
    ) -> Dict[str, float]:
        """Aggregate per-batch results into epoch-level metrics.

        Called once per epoch after all batches are processed.

        Args:
            results: List of BatchResult from all batches.
            total_samples: Total number of samples processed.

        Returns:
            Dict with 'train_loss' and strategy-specific metrics.
        """
        ...

    @abstractmethod
    def validate(
        self,
        model: nn.Module,
        loader: DataLoader,
    ) -> Dict[str, float]:
        """Run full validation loop.

        Must return at least 'val_loss' for early stopping compatibility.

        Args:
            model: Model (strategy sets eval mode internally).
            loader: Validation data loader.

        Returns:
            Dict with 'val_loss' and strategy-specific validation metrics.
        """
        ...

    @abstractmethod
    def evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
        split: str,
    ) -> Any:
        """Run full evaluation on a split.

        Args:
            model: Model (strategy sets eval mode internally).
            loader: Data loader for the split.
            split: Split name ('train', 'val', 'test') for logging.

        Returns:
            ClassificationMetrics for classification strategies,
            Dict[str, Any] for regression strategies.
        """
        ...

    @abstractmethod
    def predict(
        self,
        model: nn.Module,
        features: torch.Tensor,
        return_proba: bool = False,
    ) -> np.ndarray:
        """Make predictions on raw features.

        Args:
            model: Model (strategy sets eval mode).
            features: Input tensor [B, T, F] already on device.
            return_proba: If True, return class probabilities (classification only).

        Returns:
            np.ndarray of predictions.
        """
        ...


# =============================================================================
# Strategy Factory
# =============================================================================


def create_strategy(
    config: "ExperimentConfig",
    device: torch.device,
) -> TrainingStrategy:
    """Create the appropriate training strategy from config.

    Dispatch logic:
        - HMHP model_type -> HMHPClassificationStrategy
        - HMHP_REGRESSION model_type -> HMHPRegressionStrategy
        - REGRESSION task_type -> RegressionStrategy
        - Otherwise -> ClassificationStrategy

    Args:
        config: ExperimentConfig with model and train settings.
        device: Device for tensor placement.

    Returns:
        Concrete TrainingStrategy instance.
    """
    from lobtrainer.config import ModelType, TaskType

    model_type = config.model.model_type
    task_type = config.train.task_type

    if model_type == ModelType.HMHP:
        from lobtrainer.training.strategies.hmhp_classification import (
            HMHPClassificationStrategy,
        )

        return HMHPClassificationStrategy(config, device)

    elif model_type == ModelType.HMHP_REGRESSION:
        from lobtrainer.training.strategies.hmhp_regression import (
            HMHPRegressionStrategy,
        )

        return HMHPRegressionStrategy(config, device)

    elif task_type == TaskType.REGRESSION:
        from lobtrainer.training.strategies.regression import RegressionStrategy

        return RegressionStrategy(config, device)

    else:
        from lobtrainer.training.strategies.classification import (
            ClassificationStrategy,
        )

        return ClassificationStrategy(config, device)
