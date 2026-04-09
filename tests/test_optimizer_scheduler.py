"""
Tests for optimizer and scheduler creation in the Trainer.

Tests verify the _create_optimizer and _create_scheduler dispatch logic
added in Phase 4 Fix 2. Previously, AdamW was hardcoded; now the optimizer
is config-driven via config.train.optimizer field.

Test approach: call Trainer._create_optimizer(mock_trainer) with a MagicMock
that has the required attributes. This tests the dispatch logic without
constructing a full Trainer instance (validated in V5).

Design Principles (hft-rules.md):
    - Configuration-driven (Rule 5): 3 optimizer types, 4 scheduler types
    - Fail fast (Rule 5): unknown optimizer raises ValueError
    - Assertions explain WHAT failed and WHY (Rule 6)
"""

import pytest
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from unittest.mock import MagicMock

from lobtrainer.training.trainer import Trainer


# =============================================================================
# Helpers
# =============================================================================


def _make_trainer_mock(optimizer="adamw", lr=0.001, wd=0.01, **scheduler_kw):
    """Build mock trainer with attributes required by _create_optimizer/scheduler."""
    model = nn.Linear(10, 3)

    mock = MagicMock()
    mock.model = model  # Real model for .parameters()
    mock.config.train.optimizer = optimizer
    mock.config.train.learning_rate = lr
    mock.config.train.weight_decay = wd

    # Scheduler fields
    mock.config.train.scheduler = scheduler_kw.get("scheduler", "none")
    mock.config.train.epochs = scheduler_kw.get("epochs", 50)
    mock.config.train.scheduler_step_size = scheduler_kw.get("step_size", 30)
    mock.config.train.scheduler_gamma = scheduler_kw.get("gamma", 0.1)

    return mock


# =============================================================================
# TestCreateOptimizer
# =============================================================================


class TestCreateOptimizer:
    """Tests for Trainer._create_optimizer dispatch."""

    def test_adamw_default(self):
        """optimizer='adamw' -> AdamW with correct lr and weight_decay."""
        mock = _make_trainer_mock(optimizer="adamw", lr=0.001, wd=0.01)
        opt = Trainer._create_optimizer(mock)
        assert isinstance(opt, AdamW), (
            f"Expected AdamW, got {type(opt).__name__}"
        )
        assert opt.defaults["lr"] == 0.001
        assert opt.defaults["weight_decay"] == 0.01

    def test_adam_selected(self):
        """optimizer='adam' -> Adam."""
        mock = _make_trainer_mock(optimizer="adam", lr=0.0005, wd=0.0)
        opt = Trainer._create_optimizer(mock)
        assert isinstance(opt, Adam), f"Expected Adam, got {type(opt).__name__}"
        assert opt.defaults["lr"] == 0.0005

    def test_sgd_with_momentum(self):
        """optimizer='sgd' -> SGD with momentum=0.9 (hardcoded)."""
        mock = _make_trainer_mock(optimizer="sgd")
        opt = Trainer._create_optimizer(mock)
        assert isinstance(opt, SGD), f"Expected SGD, got {type(opt).__name__}"
        assert opt.defaults["momentum"] == 0.9, (
            f"SGD momentum should be 0.9, got {opt.defaults['momentum']}"
        )

    def test_unknown_raises_valueerror(self):
        """Unknown optimizer string raises ValueError with clear message."""
        mock = _make_trainer_mock(optimizer="rmsprop")
        with pytest.raises(ValueError, match="Unknown optimizer"):
            Trainer._create_optimizer(mock)


# =============================================================================
# TestCreateScheduler
# =============================================================================


class TestCreateScheduler:
    """Tests for Trainer._create_scheduler dispatch."""

    def _make_optimizer(self):
        """Create a simple optimizer for scheduler tests."""
        model = nn.Linear(10, 3)
        return SGD(model.parameters(), lr=0.01)

    def test_cosine_scheduler(self):
        """scheduler='cosine' -> CosineAnnealingLR."""
        mock = _make_trainer_mock(scheduler="cosine", epochs=100)
        opt = self._make_optimizer()
        sched = Trainer._create_scheduler(mock, opt)
        assert isinstance(sched, CosineAnnealingLR), (
            f"Expected CosineAnnealingLR, got {type(sched).__name__}"
        )

    def test_step_scheduler(self):
        """scheduler='step' -> StepLR with configured step_size and gamma."""
        mock = _make_trainer_mock(scheduler="step", step_size=20, gamma=0.5)
        opt = self._make_optimizer()
        sched = Trainer._create_scheduler(mock, opt)
        assert isinstance(sched, StepLR), (
            f"Expected StepLR, got {type(sched).__name__}"
        )

    def test_plateau_scheduler(self):
        """scheduler='plateau' -> ReduceLROnPlateau."""
        mock = _make_trainer_mock(scheduler="plateau")
        opt = self._make_optimizer()
        sched = Trainer._create_scheduler(mock, opt)
        assert isinstance(sched, ReduceLROnPlateau), (
            f"Expected ReduceLROnPlateau, got {type(sched).__name__}"
        )

    def test_unknown_returns_none(self):
        """Unknown scheduler returns None (graceful fallback with warning)."""
        mock = _make_trainer_mock(scheduler="cyclical")
        opt = self._make_optimizer()
        sched = Trainer._create_scheduler(mock, opt)
        assert sched is None, f"Unknown scheduler should return None, got {sched}"
