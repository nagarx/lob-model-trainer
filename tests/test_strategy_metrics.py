"""
Tests for strategy-aware metrics module.

These tests verify that the MetricsCalculator correctly computes metrics
for different labeling strategies (triple_barrier, opportunity, tlob).
"""

import numpy as np
import pytest
import torch

from lobtrainer.training.metrics import (
    MetricsCalculator,
    ClassificationMetrics,
    compute_metrics,
    compute_confusion_matrix,
    get_class_names,
    TRIPLE_BARRIER_CLASS_NAMES,
    OPPORTUNITY_CLASS_NAMES,
    TLOB_CLASS_NAMES,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def perfect_predictions():
    """Predictions that perfectly match labels."""
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])
    predictions = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])
    return predictions, labels


@pytest.fixture
def random_predictions():
    """Random predictions with known distribution."""
    np.random.seed(42)
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
    predictions = torch.randint(0, 3, (9,))
    return predictions, labels


@pytest.fixture
def triple_barrier_data():
    """
    Realistic Triple Barrier data.
    
    Classes: 0=StopLoss, 1=Timeout, 2=ProfitTarget
    Distribution: ~25% StopLoss, ~50% Timeout, ~25% ProfitTarget
    """
    np.random.seed(42)
    n_samples = 1000
    
    # Generate realistic distribution
    labels = torch.tensor(
        [0] * 250 +  # StopLoss
        [1] * 500 +  # Timeout
        [2] * 250    # ProfitTarget
    )
    
    # Simulate model that:
    # - Is conservative (often predicts Timeout)
    # - When decisive, is ~60% accurate
    predictions = labels.clone()
    
    # Add some errors
    error_indices = np.random.choice(n_samples, size=300, replace=False)
    for idx in error_indices:
        predictions[idx] = np.random.randint(0, 3)
    
    return predictions, labels


@pytest.fixture
def opportunity_data():
    """
    Realistic Opportunity data.
    
    Classes: 0=BigDown, 1=NoOpportunity, 2=BigUp
    Distribution: ~15% BigDown, ~70% NoOpportunity, ~15% BigUp
    """
    np.random.seed(42)
    n_samples = 1000
    
    labels = torch.tensor(
        [0] * 150 +  # BigDown
        [1] * 700 +  # NoOpportunity
        [2] * 150    # BigUp
    )
    
    predictions = labels.clone()
    error_indices = np.random.choice(n_samples, size=350, replace=False)
    for idx in error_indices:
        predictions[idx] = np.random.randint(0, 3)
    
    return predictions, labels


# =============================================================================
# Test Class Names
# =============================================================================


def test_get_class_names_triple_barrier():
    """Test class names for Triple Barrier strategy."""
    names = get_class_names("triple_barrier", 3)
    assert names == TRIPLE_BARRIER_CLASS_NAMES
    assert names == ["StopLoss", "Timeout", "ProfitTarget"]


def test_get_class_names_opportunity():
    """Test class names for Opportunity strategy."""
    names = get_class_names("opportunity", 3)
    assert names == OPPORTUNITY_CLASS_NAMES
    assert names == ["BigDown", "NoOpportunity", "BigUp"]


def test_get_class_names_tlob():
    """Test class names for TLOB strategy."""
    names = get_class_names("tlob", 3)
    assert names == TLOB_CLASS_NAMES
    assert names == ["Down", "Stable", "Up"]


def test_get_class_names_binary():
    """Test class names for binary classification."""
    names = get_class_names("any", 2)
    assert names == ["NoSignal", "Signal"]


def test_get_class_names_unknown():
    """Test class names for unknown strategy."""
    names = get_class_names("unknown_strategy", 3)
    assert names == ["Class_0", "Class_1", "Class_2"]


# =============================================================================
# Test MetricsCalculator Basic Functionality
# =============================================================================


def test_metrics_calculator_creation():
    """Test MetricsCalculator initialization."""
    calc = MetricsCalculator("triple_barrier", 3)
    assert calc.strategy == "triple_barrier"
    assert calc.num_classes == 3
    assert calc.class_names == TRIPLE_BARRIER_CLASS_NAMES


def test_metrics_calculator_perfect_predictions(perfect_predictions):
    """Test metrics for perfect predictions."""
    predictions, labels = perfect_predictions
    calc = MetricsCalculator("triple_barrier", 3)
    
    metrics = calc.compute(predictions, labels)
    
    # Perfect predictions should have 100% accuracy
    assert metrics.accuracy == 1.0
    
    # Perfect precision and recall for each class
    for class_id in range(3):
        assert metrics.per_class_precision[class_id] == 1.0
        assert metrics.per_class_recall[class_id] == 1.0
        assert metrics.per_class_f1[class_id] == 1.0
    
    # Macro F1 should be 1.0
    assert metrics.macro_f1 == 1.0


def test_metrics_calculator_with_numpy():
    """Test MetricsCalculator works with numpy arrays."""
    predictions = np.array([0, 1, 2, 0, 1, 2])
    labels = np.array([0, 1, 2, 0, 1, 2])
    
    calc = MetricsCalculator("triple_barrier", 3)
    metrics = calc.compute(
        torch.from_numpy(predictions),
        torch.from_numpy(labels)
    )
    
    assert metrics.accuracy == 1.0


def test_metrics_calculator_with_loss():
    """Test MetricsCalculator includes loss in output."""
    predictions = torch.tensor([0, 1, 2])
    labels = torch.tensor([0, 1, 2])
    
    calc = MetricsCalculator("triple_barrier", 3)
    metrics = calc.compute(predictions, labels, loss=0.5)
    
    assert metrics.loss == 0.5


# =============================================================================
# Test Triple Barrier Specific Metrics
# =============================================================================


def test_triple_barrier_metrics(triple_barrier_data):
    """Test Triple Barrier specific metrics are computed correctly."""
    predictions, labels = triple_barrier_data
    calc = MetricsCalculator("triple_barrier", 3)
    
    metrics = calc.compute(predictions, labels)
    
    # Check strategy metrics exist
    # Note: per-class precision/recall/f1 are in per_class_* dicts, not strategy_metrics
    assert "decisive_prediction_rate" in metrics.strategy_metrics
    assert "true_decisive_rate" in metrics.strategy_metrics
    assert "true_win_rate" in metrics.strategy_metrics
    assert "predicted_trade_win_rate" in metrics.strategy_metrics
    assert "signal_rate" in metrics.strategy_metrics
    
    # Per-class metrics are in per_class_precision, not strategy_metrics
    assert 2 in metrics.per_class_precision  # ProfitTarget (class 2)
    assert 0 in metrics.per_class_precision  # StopLoss (class 0)
    
    # Check all values are in valid range
    for key, value in metrics.strategy_metrics.items():
        assert 0.0 <= value <= 1.0, f"{key} = {value} is out of range"


def test_triple_barrier_true_win_rate():
    """Test true_win_rate calculation for Triple Barrier."""
    # All decisive samples are ProfitTarget (perfect win rate)
    labels = torch.tensor([1, 1, 1, 2, 2, 2])  # 3 Timeout, 3 ProfitTarget
    predictions = torch.tensor([1, 1, 1, 2, 2, 2])
    
    calc = MetricsCalculator("triple_barrier", 3)
    metrics = calc.compute(predictions, labels)
    
    # True win rate = ProfitTarget / (ProfitTarget + StopLoss) = 3 / 3 = 1.0
    assert metrics.strategy_metrics["true_win_rate"] == 1.0


def test_triple_barrier_decisive_rate():
    """Test decisive_prediction_rate calculation."""
    # Half predictions are decisive (not Timeout)
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    predictions = torch.tensor([0, 1, 2, 1, 1, 1])  # Only first 3 are decisive
    
    calc = MetricsCalculator("triple_barrier", 3)
    metrics = calc.compute(predictions, labels)
    
    # Decisive predictions: class 0 or 2
    # predictions: [0, 1, 2, 1, 1, 1] -> decisive = [0, 2] = 2/6 = 0.333
    expected_decisive_rate = 2 / 6
    assert abs(metrics.strategy_metrics["decisive_prediction_rate"] - expected_decisive_rate) < 0.01


# =============================================================================
# Test Opportunity Specific Metrics
# =============================================================================


def test_opportunity_metrics(opportunity_data):
    """Test Opportunity specific metrics are computed correctly."""
    predictions, labels = opportunity_data
    calc = MetricsCalculator("opportunity", 3)
    
    metrics = calc.compute(predictions, labels)
    
    # Check strategy metrics exist
    assert "opportunity_prediction_rate" in metrics.strategy_metrics
    assert "true_opportunity_rate" in metrics.strategy_metrics
    assert "directional_accuracy" in metrics.strategy_metrics
    assert "signal_rate" in metrics.strategy_metrics


def test_opportunity_directional_accuracy():
    """Test directional accuracy for Opportunity labeling."""
    # When predicting opportunities, are we getting direction right?
    labels = torch.tensor([0, 2, 0, 2])  # Only opportunity classes
    predictions = torch.tensor([0, 2, 2, 0])  # 50% correct direction
    
    calc = MetricsCalculator("opportunity", 3)
    metrics = calc.compute(predictions, labels)
    
    # All predictions are opportunities (rate = 1.0)
    assert metrics.strategy_metrics["opportunity_prediction_rate"] == 1.0
    
    # Directional accuracy: 2/4 = 0.5
    assert metrics.strategy_metrics["directional_accuracy"] == 0.5


# =============================================================================
# Test TLOB Specific Metrics
# =============================================================================


def test_tlob_metrics():
    """Test TLOB specific metrics are computed correctly."""
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    predictions = torch.tensor([0, 1, 2, 1, 1, 1])
    
    calc = MetricsCalculator("tlob", 3)
    metrics = calc.compute(predictions, labels)
    
    # Check strategy metrics exist
    assert "directional_accuracy" in metrics.strategy_metrics
    assert "signal_rate" in metrics.strategy_metrics
    
    # Per-class precision (Up=class 2, Down=class 0) is in per_class_precision, not strategy_metrics
    assert 2 in metrics.per_class_precision  # Up
    assert 0 in metrics.per_class_precision  # Down


def test_tlob_up_down_precision():
    """Test up/down precision for TLOB labeling."""
    # Perfect Up predictions, no Down predictions
    labels = torch.tensor([0, 0, 2, 2, 1, 1])
    predictions = torch.tensor([1, 1, 2, 2, 1, 1])  # No Down preds, perfect Up
    
    calc = MetricsCalculator("tlob", 3)
    metrics = calc.compute(predictions, labels)
    
    # Up precision: 2/2 = 1.0 (class 2 = Up)
    assert metrics.per_class_precision[2] == 1.0
    
    # Down precision: 0 predictions, so 0.0 (class 0 = Down)
    assert metrics.per_class_precision[0] == 0.0


# =============================================================================
# Test ClassificationMetrics Methods
# =============================================================================


def test_classification_metrics_to_dict(triple_barrier_data):
    """Test ClassificationMetrics.to_dict() method."""
    predictions, labels = triple_barrier_data
    calc = MetricsCalculator("triple_barrier", 3)
    metrics = calc.compute(predictions, labels, loss=0.5)
    
    result = metrics.to_dict()
    
    # Check basic fields
    assert "accuracy" in result
    assert "loss" in result
    assert "macro_f1" in result
    
    # Check per-class metrics with class names (lowercase)
    assert "stoploss_precision" in result
    assert "timeout_precision" in result
    assert "profittarget_precision" in result
    
    # Check strategy metrics (trading-specific, not duplicating per-class)
    assert "true_win_rate" in result
    assert "predicted_trade_win_rate" in result
    assert "decisive_prediction_rate" in result


def test_classification_metrics_summary(triple_barrier_data):
    """Test ClassificationMetrics.summary() method."""
    predictions, labels = triple_barrier_data
    calc = MetricsCalculator("triple_barrier", 3)
    metrics = calc.compute(predictions, labels)
    
    summary = metrics.summary()
    
    # Should contain key information
    assert "Accuracy:" in summary
    assert "Macro F1:" in summary
    assert "StopLoss:" in summary
    assert "Timeout:" in summary
    assert "ProfitTarget:" in summary
    assert "Strategy metrics:" in summary


# =============================================================================
# Test Confusion Matrix
# =============================================================================


def test_confusion_matrix_shape():
    """Test confusion matrix has correct shape."""
    predictions = torch.tensor([0, 1, 2, 0, 1, 2])
    labels = torch.tensor([0, 1, 2, 1, 2, 0])
    
    confusion = compute_confusion_matrix(predictions, labels, num_classes=3)
    
    assert confusion.shape == (3, 3)


def test_confusion_matrix_perfect():
    """Test confusion matrix for perfect predictions."""
    predictions = torch.tensor([0, 1, 2, 0, 1, 2])
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    
    confusion = compute_confusion_matrix(predictions, labels, num_classes=3)
    
    # Diagonal should have all counts
    assert confusion[0, 0] == 2
    assert confusion[1, 1] == 2
    assert confusion[2, 2] == 2
    
    # Off-diagonal should be zero
    assert confusion.sum() == 6
    assert np.trace(confusion) == 6


def test_confusion_matrix_all_wrong():
    """Test confusion matrix when all predictions are wrong."""
    predictions = torch.tensor([1, 2, 0])  # Shifted by 1
    labels = torch.tensor([0, 1, 2])
    
    confusion = compute_confusion_matrix(predictions, labels, num_classes=3)
    
    # Diagonal should be zero
    assert np.trace(confusion) == 0
    
    # Off-diagonal has all counts
    assert confusion[0, 1] == 1  # True 0, predicted 1
    assert confusion[1, 2] == 1  # True 1, predicted 2
    assert confusion[2, 0] == 1  # True 2, predicted 0


# =============================================================================
# Test Convenience Functions
# =============================================================================


def test_compute_metrics_function():
    """Test the compute_metrics convenience function."""
    predictions = torch.tensor([0, 1, 2, 0, 1, 2])
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    
    metrics = compute_metrics(predictions, labels, strategy="triple_barrier")
    
    assert isinstance(metrics, ClassificationMetrics)
    assert metrics.accuracy == 1.0


# =============================================================================
# Test Edge Cases
# =============================================================================


def test_empty_predictions():
    """Test handling of empty predictions."""
    predictions = torch.tensor([])
    labels = torch.tensor([])
    
    calc = MetricsCalculator("triple_barrier", 3)
    metrics = calc.compute(predictions, labels)
    
    # Should return empty metrics without crashing
    assert metrics.accuracy == 0.0


def test_single_sample():
    """Test handling of single sample."""
    predictions = torch.tensor([1])
    labels = torch.tensor([1])
    
    calc = MetricsCalculator("triple_barrier", 3)
    metrics = calc.compute(predictions, labels)
    
    assert metrics.accuracy == 1.0


def test_single_class_only():
    """Test when all samples are from one class."""
    # All samples are Timeout (class 1)
    predictions = torch.tensor([1, 1, 1, 1])
    labels = torch.tensor([1, 1, 1, 1])
    
    calc = MetricsCalculator("triple_barrier", 3)
    metrics = calc.compute(predictions, labels)
    
    assert metrics.accuracy == 1.0
    assert metrics.per_class_precision[1] == 1.0
    # Other classes should have 0 precision (no predictions or support)


def test_no_decisive_predictions():
    """Test when all predictions are Timeout (no trading signals)."""
    predictions = torch.tensor([1, 1, 1, 1, 1])  # All Timeout
    labels = torch.tensor([0, 1, 2, 0, 2])  # Mixed
    
    calc = MetricsCalculator("triple_barrier", 3)
    metrics = calc.compute(predictions, labels)
    
    # Decisive prediction rate should be 0
    assert metrics.strategy_metrics["decisive_prediction_rate"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
