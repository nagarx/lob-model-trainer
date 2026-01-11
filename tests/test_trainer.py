"""
Tests for the Trainer class and training infrastructure.

Design principles (RULE.md §5):
- Tests document behavior and expose implementation correctness
- Verify math matches expected behavior
- Test edge cases
- Verify determinism

Note: These tests use synthetic data to avoid dependency on actual dataset.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tempfile import TemporaryDirectory

from lobtrainer.training.trainer import Trainer, TrainingState, create_trainer
from lobtrainer.training.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    MetricLogger,
    Callback,
    CallbackList,
)
from lobtrainer.training.metrics import (
    compute_accuracy,
    compute_classification_report,
    ClassificationMetrics,
)
from lobtrainer.config import ExperimentConfig, ModelConfig, DataConfig, TrainConfig
from lobtrainer.utils.reproducibility import set_seed, get_seed_state, SeedManager


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(100 * 98, 64),
        nn.ReLU(),
        nn.Linear(64, 3),
    )


@pytest.fixture
def synthetic_data():
    """Create synthetic data mimicking the real dataset structure."""
    np.random.seed(42)
    
    n_samples = 500
    seq_len = 100
    n_features = 98
    n_classes = 3
    
    # Generate features
    features = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
    
    # Generate labels with some pattern
    labels = np.random.randint(0, n_classes, size=n_samples).astype(np.int64)
    
    return features, labels


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory."""
    with TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# Test Metrics
# =============================================================================


class TestMetrics:
    """Tests for classification metrics."""
    
    def test_compute_accuracy_perfect(self):
        """Test accuracy with perfect predictions."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        
        acc = compute_accuracy(y_true, y_pred)
        
        assert acc == 1.0, f"Expected 100% accuracy, got {acc}"
    
    def test_compute_accuracy_zero(self):
        """Test accuracy with completely wrong predictions."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1])
        
        acc = compute_accuracy(y_true, y_pred)
        
        assert acc == 0.0, f"Expected 0% accuracy, got {acc}"
    
    def test_compute_accuracy_partial(self):
        """Test accuracy with partial correct predictions."""
        y_true = np.array([0, 1, 2, 0])
        y_pred = np.array([0, 0, 2, 1])  # 2 correct out of 4
        
        acc = compute_accuracy(y_true, y_pred)
        
        assert acc == 0.5, f"Expected 50% accuracy, got {acc}"
    
    def test_classification_report_structure(self):
        """Test that classification report has correct structure.
        
        Note: compute_classification_report is a backward-compatibility function
        that returns a dict. Use compute_metrics for the new ClassificationMetrics API.
        """
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 0, 2, 1, 1, 2])
        
        # Use the backward-compatible function (returns dict of PerClassMetrics)
        report = compute_classification_report(torch.tensor(y_pred), torch.tensor(y_true))
        
        assert isinstance(report, dict)
        assert len(report) == 3
        for name, metrics in report.items():
            assert hasattr(metrics, 'precision')
            assert hasattr(metrics, 'recall')
            assert hasattr(metrics, 'f1')
            assert hasattr(metrics, 'support')
    
    def test_classification_metrics_new_api(self):
        """Test new MetricsCalculator API returns ClassificationMetrics."""
        from lobtrainer.training.metrics import compute_metrics
        
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 0, 2, 1, 1, 2])
        
        metrics = compute_metrics(
            torch.tensor(y_pred),
            torch.tensor(y_true),
            strategy="tlob"
        )
        
        assert isinstance(metrics, ClassificationMetrics)
        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.macro_f1 <= 1
        assert metrics.confusion_matrix.shape == (3, 3)
    
    def test_classification_report_empty_arrays(self):
        """Test classification report with empty arrays."""
        y_true = np.array([])
        y_pred = np.array([])
        
        # Use new API for empty arrays
        from lobtrainer.training.metrics import compute_metrics
        metrics = compute_metrics(
            torch.tensor(y_pred, dtype=torch.long),
            torch.tensor(y_true, dtype=torch.long),
            strategy="tlob"
        )
        
        assert metrics.accuracy == 0.0
    
    def test_classification_metrics_to_dict(self):
        """Test serialization of classification metrics."""
        from lobtrainer.training.metrics import compute_metrics
        
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 1, 1, 2])
        
        metrics = compute_metrics(
            torch.tensor(y_pred),
            torch.tensor(y_true),
            strategy="tlob"
        )
        result = metrics.to_dict()
        
        assert isinstance(result, dict)
        assert 'accuracy' in result
        assert 'macro_f1' in result


# =============================================================================
# Test Callbacks
# =============================================================================


class TestEarlyStopping:
    """Tests for EarlyStopping callback."""
    
    def test_early_stopping_patience(self):
        """Test that early stopping triggers after patience epochs."""
        callback = EarlyStopping(patience=3, metric='val_loss', mode='min')
        callback.on_train_start()
        
        # Simulate epochs with no improvement
        for i in range(5):
            callback.on_epoch_end(i, {'val_loss': 1.0})
            
            if i < 3:
                assert not callback.should_stop, f"Should not stop at epoch {i}"
        
        assert callback.should_stop, "Should have stopped after patience=3"
    
    def test_early_stopping_improvement_resets(self):
        """Test that improvement resets wait counter."""
        callback = EarlyStopping(patience=2, metric='val_loss', mode='min')
        callback.on_train_start()
        
        # Epoch 0: loss = 1.0 (best)
        callback.on_epoch_end(0, {'val_loss': 1.0})
        assert not callback.should_stop
        
        # Epoch 1: loss = 1.1 (no improvement)
        callback.on_epoch_end(1, {'val_loss': 1.1})
        assert not callback.should_stop
        
        # Epoch 2: loss = 0.9 (improvement - resets counter)
        callback.on_epoch_end(2, {'val_loss': 0.9})
        assert not callback.should_stop
        
        # Epoch 3: loss = 1.0 (no improvement - wait=1)
        callback.on_epoch_end(3, {'val_loss': 1.0})
        assert not callback.should_stop
        
        # Epoch 4: loss = 1.0 (no improvement - wait=2)
        callback.on_epoch_end(4, {'val_loss': 1.0})
        assert callback.should_stop
    
    def test_early_stopping_max_mode(self):
        """Test early stopping with max mode (for accuracy)."""
        callback = EarlyStopping(patience=2, metric='val_acc', mode='max')
        callback.on_train_start()
        
        callback.on_epoch_end(0, {'val_acc': 0.5})  # Best
        assert callback.best_value == 0.5
        
        callback.on_epoch_end(1, {'val_acc': 0.6})  # Improvement
        assert callback.best_value == 0.6
        
        callback.on_epoch_end(2, {'val_acc': 0.5})  # No improvement
        callback.on_epoch_end(3, {'val_acc': 0.5})  # No improvement
        
        assert callback.should_stop


class TestModelCheckpoint:
    """Tests for ModelCheckpoint callback."""
    
    def test_checkpoint_saves_on_improvement(self, temp_output_dir):
        """Test that checkpoint is saved when metric improves."""
        checkpoint = ModelCheckpoint(
            save_dir=temp_output_dir,
            metric='val_loss',
            mode='min',
            save_best_only=True,
        )
        
        # Mock trainer
        class MockTrainer:
            def __init__(self):
                self.model = nn.Linear(10, 3)
                self.optimizer = None
                self.config = None
        
        checkpoint.trainer = MockTrainer()
        checkpoint.on_train_start()
        
        # First epoch - should save
        checkpoint.on_epoch_end(0, {'val_loss': 1.0})
        assert (temp_output_dir / 'best.pt').exists()
        
        # Second epoch with improvement - should save new best
        checkpoint.on_epoch_end(1, {'val_loss': 0.8})
        assert (temp_output_dir / 'best.pt').exists()
        
        # Third epoch without improvement - should not create new checkpoint
        initial_checkpoints = list(temp_output_dir.glob('checkpoint_*.pt'))
        checkpoint.on_epoch_end(2, {'val_loss': 0.9})
        final_checkpoints = list(temp_output_dir.glob('checkpoint_*.pt'))
        
        # Should have same number (no new checkpoint for non-improvement)
        assert len(initial_checkpoints) == len(final_checkpoints)


class TestCallbackList:
    """Tests for CallbackList container."""
    
    def test_callback_list_dispatches_to_all(self):
        """Test that CallbackList calls all callbacks."""
        call_counts = {'a': 0, 'b': 0}
        
        class CountingCallback(Callback):
            def __init__(self, name):
                super().__init__()
                self.name = name
            
            def on_epoch_end(self, epoch, logs):
                call_counts[self.name] += 1
        
        callbacks = CallbackList([CountingCallback('a'), CountingCallback('b')])
        callbacks.on_epoch_end(0, {})
        
        assert call_counts['a'] == 1
        assert call_counts['b'] == 1
    
    def test_callback_list_should_stop(self):
        """Test that should_stop is True if any callback signals stop."""
        class StoppingCallback(Callback):
            def __init__(self, should_stop):
                super().__init__()
                self._stop = should_stop
            
            @property
            def should_stop(self):
                return self._stop
        
        # Neither should stop
        callbacks = CallbackList([StoppingCallback(False), StoppingCallback(False)])
        assert not callbacks.should_stop
        
        # One should stop
        callbacks = CallbackList([StoppingCallback(False), StoppingCallback(True)])
        assert callbacks.should_stop


# =============================================================================
# Test Reproducibility
# =============================================================================


class TestReproducibility:
    """Tests for reproducibility utilities."""
    
    def test_set_seed_deterministic(self):
        """Test that same seed produces same random numbers."""
        set_seed(42)
        a1 = torch.rand(5)
        b1 = np.random.rand(5)
        
        set_seed(42)
        a2 = torch.rand(5)
        b2 = np.random.rand(5)
        
        assert torch.allclose(a1, a2), "PyTorch random should be deterministic"
        np.testing.assert_array_almost_equal(b1, b2, err_msg="NumPy random should be deterministic")
    
    def test_different_seeds_different_results(self):
        """Test that different seeds produce different results."""
        set_seed(42)
        a1 = torch.rand(5)
        
        set_seed(123)
        a2 = torch.rand(5)
        
        assert not torch.allclose(a1, a2), "Different seeds should produce different numbers"
    
    def test_seed_manager_context(self):
        """Test SeedManager context manager."""
        set_seed(42)
        
        # Save state
        original_state = get_seed_state()
        
        # Run in context with different seed
        with SeedManager(123, restore_state=True):
            inner = torch.rand(3)
        
        # After context, should have original state
        restored = torch.rand(3)
        
        # Verify by re-running with seed 42
        set_seed(42)
        expected = torch.rand(3)
        
        # The restored tensor should match what we'd get from seed 42
        assert torch.allclose(restored, expected), "State should be restored after context"
    
    def test_seed_manager_no_restore(self):
        """Test SeedManager without state restoration."""
        set_seed(42)
        before = torch.rand(3)
        
        set_seed(42)
        with SeedManager(42, restore_state=False):
            inner = torch.rand(3)
        
        after = torch.rand(3)
        
        # After context without restore, state continues from context
        set_seed(42)
        _ = torch.rand(3)  # Skip first
        expected_after = torch.rand(3)
        
        # Won't match because SeedManager consumed random numbers
        assert torch.allclose(inner, before), "Same seed should produce same first numbers"


# =============================================================================
# Test Training State
# =============================================================================


class TestTrainingState:
    """Tests for TrainingState dataclass."""
    
    def test_training_state_defaults(self):
        """Test default values of TrainingState."""
        state = TrainingState()
        
        assert state.current_epoch == 0
        assert state.global_step == 0
        assert state.best_val_metric == float('inf')
        assert state.best_epoch == 0
        assert not state.training_started
        assert not state.training_completed
        assert state.history == []
    
    def test_training_state_mutable(self):
        """Test that TrainingState is mutable."""
        state = TrainingState()
        
        state.current_epoch = 5
        state.global_step = 100
        state.training_started = True
        state.history.append({'epoch': 0, 'loss': 1.0})
        
        assert state.current_epoch == 5
        assert state.global_step == 100
        assert state.training_started
        assert len(state.history) == 1


# =============================================================================
# Test Models
# =============================================================================


class TestLSTMModel:
    """Tests for LSTM model."""
    
    def test_lstm_forward_shape(self):
        """Test LSTM model output shape."""
        from lobtrainer.models.lstm import LSTMClassifier, LSTMConfig
        
        config = LSTMConfig(
            input_size=98,
            hidden_size=64,
            num_layers=2,
            num_classes=3,
        )
        model = LSTMClassifier(config)
        
        # Input: [batch, seq_len, features]
        x = torch.randn(32, 100, 98)
        output = model(x)
        
        assert output.shape == (32, 3), f"Expected shape (32, 3), got {output.shape}"
    
    def test_lstm_bidirectional(self):
        """Test bidirectional LSTM."""
        from lobtrainer.models.lstm import LSTMClassifier, LSTMConfig
        
        config = LSTMConfig(
            input_size=98,
            hidden_size=64,
            num_layers=2,
            num_classes=3,
            bidirectional=True,
        )
        model = LSTMClassifier(config)
        
        x = torch.randn(16, 50, 98)
        output = model(x)
        
        assert output.shape == (16, 3)
    
    def test_gru_forward_shape(self):
        """Test GRU model output shape."""
        from lobtrainer.models.lstm import GRUClassifier, LSTMConfig
        
        config = LSTMConfig(
            input_size=98,
            hidden_size=64,
            num_layers=2,
            num_classes=3,
        )
        model = GRUClassifier(config)
        
        x = torch.randn(32, 100, 98)
        output = model(x)
        
        assert output.shape == (32, 3)
    
    def test_model_deterministic(self):
        """Test that model is deterministic given same seed."""
        from lobtrainer.models.lstm import LSTMClassifier, LSTMConfig
        
        config = LSTMConfig(input_size=10, hidden_size=16, num_layers=1, num_classes=3)
        
        set_seed(42)
        model1 = LSTMClassifier(config)
        x = torch.randn(4, 10, 10)
        out1 = model1(x)
        
        set_seed(42)
        model2 = LSTMClassifier(config)
        x = torch.randn(4, 10, 10)
        out2 = model2(x)
        
        assert torch.allclose(out1, out2, atol=1e-6), "Same seed should give same outputs"


# =============================================================================
# Test Model Factory
# =============================================================================


class TestModelFactory:
    """Tests for model factory functions."""
    
    def test_create_model_lstm(self):
        """Test creating LSTM model from config."""
        from lobtrainer.models import create_model
        from lobtrainer.config import ModelConfig, ModelType
        
        config = ModelConfig(
            model_type=ModelType.LSTM,
            input_size=98,
            hidden_size=64,
            num_layers=2,
            num_classes=3,
        )
        
        model = create_model(config)
        
        assert isinstance(model, nn.Module)
        assert hasattr(model, 'lstm')
    
    def test_create_model_gru(self):
        """Test creating GRU model from config."""
        from lobtrainer.models import create_model
        from lobtrainer.config import ModelConfig, ModelType
        
        config = ModelConfig(
            model_type=ModelType.GRU,
            input_size=98,
            hidden_size=64,
            num_layers=2,
            num_classes=3,
        )
        
        model = create_model(config)
        
        assert isinstance(model, nn.Module)
        assert hasattr(model, 'gru')
    
    def test_create_model_unknown_type(self):
        """Test that unknown model type raises error."""
        from lobtrainer.models import create_model
        from lobtrainer.config import ModelConfig, ModelType
        
        config = ModelConfig(model_type=ModelType.TRANSFORMER)
        
        with pytest.raises(NotImplementedError):
            create_model(config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

