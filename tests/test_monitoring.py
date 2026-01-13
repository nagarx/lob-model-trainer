"""
Tests for training monitoring callbacks.

Tests GradientMonitor, LearningRateTracker, TrainingDiagnostics, and PerClassMetricsTracker:
- Gradient computation correctness
- NaN/Inf detection
- Learning rate tracking
- Training health checks
- Per-class metric tracking

RULE.md compliance:
- NaN/Inf handling is explicit
- Assertions explain WHAT failed and WHY
- Edge cases tested
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import tempfile
import json
from unittest.mock import MagicMock, patch

from lobtrainer.training.monitoring import (
    GradientStats,
    GradientMonitor,
    LearningRateTracker,
    TrainingDiagnostics,
    PerClassMetricsTracker,
    HealthCheckResult,
    create_standard_monitoring,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 3),
    )


@pytest.fixture
def mock_trainer(simple_model, tmp_path):
    """Create a mock trainer object."""
    trainer = MagicMock()
    trainer.model = simple_model
    trainer._model = simple_model
    trainer._optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)
    trainer.optimizer = trainer._optimizer
    trainer.config = MagicMock()
    trainer.config.output_dir = str(tmp_path)
    trainer.output_dir = tmp_path
    return trainer


# =============================================================================
# GradientStats Tests
# =============================================================================


class TestGradientStats:
    """Test GradientStats dataclass."""
    
    def test_healthy_gradients(self):
        """Normal gradients should be healthy."""
        stats = GradientStats(
            total_norm=1.5,
            max_norm=0.5,
            min_norm=0.01,
            mean_norm=0.1,
            num_zero_grads=0,
            num_nan_grads=0,
            num_inf_grads=0,
        )
        
        assert stats.is_healthy, "Normal gradients should be healthy"
    
    def test_nan_grads_unhealthy(self):
        """NaN gradients should be unhealthy."""
        stats = GradientStats(
            total_norm=1.5,
            max_norm=0.5,
            min_norm=0.01,
            mean_norm=0.1,
            num_zero_grads=0,
            num_nan_grads=1,  # Has NaN
            num_inf_grads=0,
        )
        
        assert not stats.is_healthy, "NaN gradients should be unhealthy"
    
    def test_inf_grads_unhealthy(self):
        """Inf gradients should be unhealthy."""
        stats = GradientStats(
            total_norm=1.5,
            max_norm=0.5,
            min_norm=0.01,
            mean_norm=0.1,
            num_zero_grads=0,
            num_nan_grads=0,
            num_inf_grads=1,  # Has Inf
        )
        
        assert not stats.is_healthy, "Inf gradients should be unhealthy"
    
    def test_exploding_grads_unhealthy(self):
        """Very large gradients should be unhealthy."""
        stats = GradientStats(
            total_norm=5000,  # Way too large
            max_norm=100,
            min_norm=0.01,
            mean_norm=50,
            num_zero_grads=0,
            num_nan_grads=0,
            num_inf_grads=0,
        )
        
        assert not stats.is_healthy, "Exploding gradients should be unhealthy"
    
    def test_to_dict(self):
        """to_dict should return all fields."""
        stats = GradientStats(
            total_norm=1.5,
            max_norm=0.5,
            min_norm=0.01,
            mean_norm=0.1,
            num_zero_grads=2,
            num_nan_grads=0,
            num_inf_grads=0,
        )
        
        d = stats.to_dict()
        
        assert d["total_norm"] == 1.5
        assert d["num_zero_grads"] == 2
        assert "is_healthy" in d


# =============================================================================
# GradientMonitor Tests
# =============================================================================


class TestGradientMonitorInit:
    """Test GradientMonitor initialization."""
    
    def test_default_init(self):
        """Default initialization should work."""
        monitor = GradientMonitor()
        
        assert monitor.log_every_n_batches is None
        assert monitor.warn_threshold_low == 1e-7
        assert monitor.warn_threshold_high == 100.0
        assert monitor.track_per_layer is False
    
    def test_custom_thresholds(self):
        """Custom thresholds should be accepted."""
        monitor = GradientMonitor(
            warn_threshold_low=1e-10,
            warn_threshold_high=50.0,
        )
        
        assert monitor.warn_threshold_low == 1e-10
        assert monitor.warn_threshold_high == 50.0


class TestGradientMonitorCompute:
    """Test gradient statistics computation."""
    
    def test_compute_stats_after_backward(self, simple_model):
        """Should compute gradient stats after backward pass."""
        monitor = GradientMonitor()
        
        # Forward and backward pass
        x = torch.randn(4, 10)
        y = torch.randint(0, 3, (4,))
        output = simple_model(x)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        
        stats = monitor.compute_gradient_stats(simple_model)
        
        assert stats.total_norm > 0, "Should have non-zero gradient norm"
        assert stats.num_nan_grads == 0, "Should have no NaN gradients"
        assert stats.num_inf_grads == 0, "Should have no Inf gradients"
    
    def test_compute_stats_no_gradients(self, simple_model):
        """Should handle model with no gradients."""
        monitor = GradientMonitor()
        
        # No backward pass, no gradients
        stats = monitor.compute_gradient_stats(simple_model)
        
        assert stats.total_norm == 0.0, "Should have zero norm without gradients"
    
    def test_track_per_layer(self, simple_model):
        """Should track per-layer norms if enabled."""
        monitor = GradientMonitor(track_per_layer=True)
        
        # Forward and backward pass
        x = torch.randn(4, 10)
        y = torch.randint(0, 3, (4,))
        output = simple_model(x)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        
        stats = monitor.compute_gradient_stats(simple_model)
        
        assert len(stats.layer_norms) > 0, "Should have per-layer norms"


class TestGradientMonitorCallbacks:
    """Test GradientMonitor callback methods."""
    
    def test_on_epoch_start_resets_stats(self, mock_trainer):
        """on_epoch_start should reset batch stats."""
        monitor = GradientMonitor()
        monitor.trainer = mock_trainer
        
        # Simulate some stats
        monitor._batch_stats = [GradientStats(1, 1, 1, 1, 0, 0, 0)]
        
        monitor.on_epoch_start(0)
        
        assert len(monitor._batch_stats) == 0
    
    def test_on_batch_end_collects_stats(self, mock_trainer):
        """on_batch_end should collect gradient stats."""
        monitor = GradientMonitor()
        monitor.trainer = mock_trainer
        
        # Do backward pass to create gradients
        x = torch.randn(4, 10)
        y = torch.randint(0, 3, (4,))
        output = mock_trainer.model(x)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        
        monitor.on_batch_end(0, {"loss": 0.5})
        
        assert len(monitor._batch_stats) == 1
    
    def test_on_epoch_end_summarizes(self, mock_trainer):
        """on_epoch_end should summarize epoch stats."""
        monitor = GradientMonitor()
        monitor.trainer = mock_trainer
        
        # Add some batch stats
        monitor._batch_stats = [
            GradientStats(1.0, 0.5, 0.1, 0.3, 0, 0, 0),
            GradientStats(1.5, 0.6, 0.2, 0.4, 0, 0, 0),
        ]
        
        monitor.on_epoch_end(0, {})
        
        assert len(monitor._history) == 1
        assert "grad_norm_mean" in monitor._history[0]


# =============================================================================
# LearningRateTracker Tests
# =============================================================================


class TestLearningRateTracker:
    """Test LearningRateTracker callback."""
    
    def test_tracks_lr_on_epoch_start(self, mock_trainer):
        """Should track learning rate at epoch start."""
        tracker = LearningRateTracker()
        tracker.trainer = mock_trainer
        
        tracker.on_epoch_start(0)
        
        assert len(tracker.history) == 1
        assert tracker.history[0]["epoch"] == 0
        assert tracker.history[0]["lr"] == 0.01  # Initial LR
    
    def test_tracks_multiple_epochs(self, mock_trainer):
        """Should track LR across multiple epochs."""
        tracker = LearningRateTracker()
        tracker.trainer = mock_trainer
        
        for epoch in range(3):
            tracker.on_epoch_start(epoch)
        
        assert len(tracker.history) == 3
    
    def test_saves_history(self, mock_trainer):
        """Should save history on train_end."""
        tracker = LearningRateTracker(save_history=True)
        tracker.trainer = mock_trainer
        
        tracker.on_epoch_start(0)
        tracker.on_train_end()
        
        history_path = mock_trainer.output_dir / "learning_rate_history.json"
        assert history_path.exists()


# =============================================================================
# TrainingDiagnostics Tests
# =============================================================================


class TestHealthCheckResult:
    """Test HealthCheckResult dataclass."""
    
    def test_healthy_result(self):
        """Healthy result should have no issues."""
        result = HealthCheckResult(
            is_healthy=True,
            issues=[],
            warnings=[],
        )
        
        assert result.is_healthy
        assert len(result.issues) == 0
    
    def test_unhealthy_result(self):
        """Unhealthy result should have issues."""
        result = HealthCheckResult(
            is_healthy=False,
            issues=["Critical error"],
            warnings=["Warning message"],
        )
        
        assert not result.is_healthy
        assert len(result.issues) == 1


class TestTrainingDiagnosticsInit:
    """Test TrainingDiagnostics initialization."""
    
    def test_default_init(self):
        """Default initialization should work."""
        diag = TrainingDiagnostics()
        
        assert diag.alert_on_nan is True
        assert diag.stagnation_patience == 5
        assert diag.divergence_threshold == 2.0
    
    def test_custom_params(self):
        """Custom parameters should be accepted."""
        diag = TrainingDiagnostics(
            alert_on_nan=False,
            stagnation_patience=10,
        )
        
        assert diag.alert_on_nan is False
        assert diag.stagnation_patience == 10


class TestTrainingDiagnosticsNaN:
    """Test NaN detection."""
    
    def test_detects_nan_loss(self):
        """Should detect NaN loss."""
        diag = TrainingDiagnostics(alert_on_nan=True)
        
        with pytest.raises(ValueError, match="NaN loss"):
            diag.on_batch_end(0, {"loss": float('nan')})
    
    def test_detects_inf_loss(self):
        """Should detect Inf loss."""
        diag = TrainingDiagnostics(alert_on_nan=True)
        
        with pytest.raises(ValueError, match="Inf loss"):
            diag.on_batch_end(0, {"loss": float('inf')})
    
    def test_no_alert_when_disabled(self):
        """Should not raise when alert_on_nan=False."""
        diag = TrainingDiagnostics(alert_on_nan=False)
        
        # Should not raise
        diag.on_batch_end(0, {"loss": float('nan')})


class TestTrainingDiagnosticsHealthChecks:
    """Test training health checks."""
    
    def test_tracks_loss_history(self):
        """Should track loss history."""
        diag = TrainingDiagnostics()
        diag.on_train_start()
        
        for epoch in range(3):
            diag.on_epoch_end(epoch, {"train_loss": 1.0 - epoch * 0.2})
        
        assert len(diag._loss_history) == 3
    
    def test_detects_stagnation(self):
        """Should detect accuracy stagnation."""
        diag = TrainingDiagnostics(stagnation_patience=2)
        diag.on_train_start()
        
        # Train with no improvement
        for epoch in range(5):
            diag.on_epoch_end(epoch, {"val_accuracy": 0.5})
        
        # Should have warnings about stagnation
        warnings = [
            w 
            for h in diag._health_history 
            for w in h.warnings 
            if "stagnation" in w.lower()
        ]
        assert len(warnings) > 0
    
    def test_detects_divergence(self):
        """Should detect loss divergence."""
        diag = TrainingDiagnostics(divergence_threshold=1.5)
        diag.on_train_start()
        
        # Initial loss
        diag.on_epoch_end(0, {"train_loss": 1.0})
        # Much higher loss (diverging)
        diag.on_epoch_end(1, {"train_loss": 3.0})
        
        # Should have divergence warning
        warnings = [
            w 
            for h in diag._health_history 
            for w in h.warnings 
            if "divergence" in w.lower()
        ]
        assert len(warnings) > 0


class TestTrainingDiagnosticsReport:
    """Test diagnostic report generation."""
    
    def test_saves_report(self, mock_trainer):
        """Should save diagnostic report."""
        diag = TrainingDiagnostics()
        diag.trainer = mock_trainer
        diag.on_train_start()
        
        # Simulate training
        for epoch in range(3):
            diag.on_epoch_end(epoch, {
                "train_loss": 1.0 - epoch * 0.1,
                "val_loss": 1.1 - epoch * 0.1,
                "val_accuracy": 0.5 + epoch * 0.05,
            })
        
        diag.on_train_end()
        
        report_path = mock_trainer.output_dir / "training_diagnostics.json"
        assert report_path.exists()
        
        with open(report_path) as f:
            report = json.load(f)
        
        assert "loss_history" in report


# =============================================================================
# PerClassMetricsTracker Tests
# =============================================================================


class TestPerClassMetricsTracker:
    """Test PerClassMetricsTracker callback."""
    
    def test_tracks_per_class_metrics(self, mock_trainer):
        """Should track per-class metrics from logs."""
        tracker = PerClassMetricsTracker()
        tracker.trainer = mock_trainer
        
        # Simulate validation with per-class metrics
        tracker.on_validation_end(0, {
            "down_precision": 0.6,
            "down_recall": 0.5,
            "stable_precision": 0.8,
            "stable_recall": 0.9,
        })
        
        assert len(tracker.history) == 1
        assert "down_precision" in tracker.history[0]
    
    def test_saves_history(self, mock_trainer):
        """Should save history on train_end."""
        tracker = PerClassMetricsTracker(save_history=True)
        tracker.trainer = mock_trainer
        
        tracker.on_validation_end(0, {"down_precision": 0.6})
        tracker.on_train_end()
        
        history_path = mock_trainer.output_dir / "per_class_metrics_history.json"
        assert history_path.exists()


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestCreateStandardMonitoring:
    """Test create_standard_monitoring function."""
    
    def test_returns_list(self):
        """Should return list of callbacks."""
        callbacks = create_standard_monitoring()
        
        assert isinstance(callbacks, list)
        assert len(callbacks) > 0
    
    def test_includes_gradient_monitor(self):
        """Should include GradientMonitor by default."""
        callbacks = create_standard_monitoring()
        
        has_gradient_monitor = any(
            isinstance(cb, GradientMonitor) for cb in callbacks
        )
        assert has_gradient_monitor
    
    def test_includes_diagnostics(self):
        """Should include TrainingDiagnostics by default."""
        callbacks = create_standard_monitoring()
        
        has_diagnostics = any(
            isinstance(cb, TrainingDiagnostics) for cb in callbacks
        )
        assert has_diagnostics
    
    def test_custom_gradient_log_interval(self):
        """Should respect gradient_log_every parameter."""
        callbacks = create_standard_monitoring(gradient_log_every=50)
        
        gradient_monitor = next(
            cb for cb in callbacks if isinstance(cb, GradientMonitor)
        )
        assert gradient_monitor.log_every_n_batches == 50
    
    def test_exclude_diagnostics(self):
        """Should exclude diagnostics if requested."""
        callbacks = create_standard_monitoring(include_diagnostics=False)
        
        has_diagnostics = any(
            isinstance(cb, TrainingDiagnostics) for cb in callbacks
        )
        assert not has_diagnostics


# =============================================================================
# Integration Tests
# =============================================================================


class TestMonitoringIntegration:
    """Test monitoring in a realistic training scenario."""
    
    def test_full_monitoring_pipeline(self, simple_model, tmp_path):
        """Test all monitors working together."""
        # Create monitors
        gradient_monitor = GradientMonitor()
        lr_tracker = LearningRateTracker()
        diagnostics = TrainingDiagnostics()
        per_class = PerClassMetricsTracker()
        
        # Create mock trainer
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
        
        trainer = MagicMock()
        trainer.model = simple_model
        trainer._model = simple_model
        trainer._optimizer = optimizer
        trainer.optimizer = optimizer
        trainer.config = MagicMock()
        trainer.config.output_dir = str(tmp_path)
        trainer.output_dir = tmp_path
        
        # Set trainer for all monitors (Callback.trainer is set directly by Trainer)
        for monitor in [gradient_monitor, lr_tracker, diagnostics, per_class]:
            monitor.trainer = trainer
        
        # Simulate training loop
        diagnostics.on_train_start()
        
        for epoch in range(3):
            gradient_monitor.on_epoch_start(epoch)
            lr_tracker.on_epoch_start(epoch)
            
            # Simulate batch
            x = torch.randn(4, 10)
            y = torch.randint(0, 3, (4,))
            
            optimizer.zero_grad()
            output = simple_model(x)
            loss = nn.CrossEntropyLoss()(output, y)
            loss.backward()
            optimizer.step()
            
            gradient_monitor.on_batch_end(0, {"loss": loss.item()})
            diagnostics.on_batch_end(0, {"loss": loss.item()})
            
            # Epoch end
            metrics = {
                "train_loss": loss.item(),
                "val_loss": loss.item() + 0.1,
                "val_accuracy": 0.5 + epoch * 0.1,
                "down_precision": 0.5,
            }
            
            gradient_monitor.on_epoch_end(epoch, metrics)
            diagnostics.on_epoch_end(epoch, metrics)
            per_class.on_validation_end(epoch, metrics)
            
            scheduler.step()
        
        # End training
        gradient_monitor.on_train_end()
        lr_tracker.on_train_end()
        diagnostics.on_train_end()
        per_class.on_train_end()
        
        # Verify outputs were saved
        assert (tmp_path / "gradient_history.json").exists()
        assert (tmp_path / "learning_rate_history.json").exists()
        assert (tmp_path / "training_diagnostics.json").exists()
        assert (tmp_path / "per_class_metrics_history.json").exists()
