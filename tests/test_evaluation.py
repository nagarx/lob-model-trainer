"""
Tests for model evaluation framework.

Tests:
- evaluate_model function
- evaluate_naive_baseline function
- BaselineReport class
- create_baseline_report function
- full_evaluation function

RULE.md compliance:
- Consistent metrics across all evaluations
- Baseline comparison for model validation
- No data leakage in evaluation
"""

import json
import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from lobtrainer.training.evaluation import (
    evaluate_model,
    evaluate_naive_baseline,
    BaselineReport,
    create_baseline_report,
    full_evaluation,
)
from lobtrainer.models.baselines import (
    BaseModel,
    NaiveClassPrior,
    NaivePreviousLabel,
    LogisticBaseline,
    LogisticBaselineConfig,
)
from lobtrainer.training.metrics import ClassificationMetrics


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_labels():
    """Create sample labels with autocorrelation."""
    np.random.seed(42)
    # Simulate autocorrelated labels (Markov chain)
    labels = [1]  # Start with Stable
    for _ in range(999):
        if np.random.rand() < 0.8:  # 80% chance to stay same
            labels.append(labels[-1])
        else:
            labels.append(np.random.choice([0, 1, 2]))
    return np.array(labels)


@pytest.fixture
def sample_features():
    """Create sample features."""
    np.random.seed(42)
    return np.random.randn(1000, 98)


@pytest.fixture
def trained_naive_prior(sample_features, sample_labels):
    """Create a trained NaiveClassPrior model."""
    model = NaiveClassPrior()
    model.fit(sample_features, sample_labels)
    return model


# =============================================================================
# evaluate_model Tests
# =============================================================================


class TestEvaluateModel:
    """Test evaluate_model function."""
    
    def test_returns_classification_metrics(self, trained_naive_prior, sample_features, sample_labels):
        """Should return ClassificationMetrics object."""
        metrics = evaluate_model(trained_naive_prior, sample_features, sample_labels)
        
        assert isinstance(metrics, ClassificationMetrics)
    
    def test_metrics_contain_required_fields(self, trained_naive_prior, sample_features, sample_labels):
        """Metrics should contain all required fields."""
        metrics = evaluate_model(trained_naive_prior, sample_features, sample_labels)
        
        assert hasattr(metrics, 'accuracy')
        assert hasattr(metrics, 'macro_f1')
        assert hasattr(metrics, 'macro_precision')
        assert hasattr(metrics, 'macro_recall')
    
    def test_accuracy_is_valid(self, trained_naive_prior, sample_features, sample_labels):
        """Accuracy should be between 0 and 1."""
        metrics = evaluate_model(trained_naive_prior, sample_features, sample_labels)
        
        assert 0 <= metrics.accuracy <= 1, (
            f"Accuracy should be in [0, 1], got {metrics.accuracy}"
        )
    
    def test_custom_name(self, trained_naive_prior, sample_features, sample_labels):
        """Should accept custom name."""
        # This shouldn't raise
        metrics = evaluate_model(
            trained_naive_prior, 
            sample_features, 
            sample_labels,
            name="CustomModel"
        )
        
        assert metrics is not None


# =============================================================================
# evaluate_naive_baseline Tests
# =============================================================================


class TestEvaluateNaiveBaseline:
    """Test evaluate_naive_baseline function."""
    
    def test_returns_dict_with_both_baselines(self, sample_labels):
        """Should return dict with class_prior and previous_label."""
        results = evaluate_naive_baseline(sample_labels)
        
        assert 'class_prior' in results
        assert 'previous_label' in results
    
    def test_class_prior_metrics(self, sample_labels):
        """Class prior baseline should have valid metrics."""
        results = evaluate_naive_baseline(sample_labels)
        
        assert isinstance(results['class_prior'], ClassificationMetrics)
        assert 0 <= results['class_prior'].accuracy <= 1
    
    def test_previous_label_metrics(self, sample_labels):
        """Previous label baseline should have valid metrics."""
        results = evaluate_naive_baseline(sample_labels)
        
        assert isinstance(results['previous_label'], ClassificationMetrics)
        assert 0 <= results['previous_label'].accuracy <= 1
    
    def test_previous_label_exploits_autocorrelation(self, sample_labels):
        """Previous label should have higher accuracy for autocorrelated labels."""
        # Our sample_labels have 80% autocorrelation
        results = evaluate_naive_baseline(sample_labels)
        
        # Previous label should beat class prior
        prev_acc = results['previous_label'].accuracy
        prior_acc = results['class_prior'].accuracy
        
        # With 80% autocorrelation, previous label should be significantly higher
        assert prev_acc > prior_acc, (
            f"Previous label ({prev_acc:.3f}) should beat class prior ({prior_acc:.3f}) "
            f"for autocorrelated labels"
        )


# =============================================================================
# BaselineReport Tests
# =============================================================================


class TestBaselineReport:
    """Test BaselineReport dataclass."""
    
    @pytest.fixture
    def mock_metrics(self):
        """Create mock ClassificationMetrics."""
        # Create per-class mocks with proper string names
        per_class = []
        for name, f1 in [("Down", 0.65), ("Stable", 0.80), ("Up", 0.65)]:
            pc = MagicMock()
            pc.name = name  # Ensure name is a string, not MagicMock
            pc.f1 = f1
            per_class.append(pc)
        
        return MagicMock(
            accuracy=0.75,
            macro_f1=0.70,
            macro_precision=0.72,
            macro_recall=0.68,
            per_class=per_class,
            to_dict=lambda: {"accuracy": 0.75, "macro_f1": 0.70}
        )
    
    @pytest.fixture
    def mock_prior_metrics(self):
        """Create mock metrics for class prior."""
        return MagicMock(
            accuracy=0.40,
            macro_f1=0.35,
            to_dict=lambda: {"accuracy": 0.40, "macro_f1": 0.35}
        )
    
    @pytest.fixture
    def mock_prev_metrics(self):
        """Create mock metrics for previous label."""
        return MagicMock(
            accuracy=0.70,
            macro_f1=0.65,
            to_dict=lambda: {"accuracy": 0.70, "macro_f1": 0.65}
        )
    
    def test_computes_beats_flags(self, mock_metrics, mock_prior_metrics, mock_prev_metrics):
        """Should compute beats_class_prior and beats_previous_label."""
        report = BaselineReport(
            model_name="TestModel",
            split="test",
            n_samples=1000,
            model_metrics=mock_metrics,
            class_prior_metrics=mock_prior_metrics,
            previous_label_metrics=mock_prev_metrics,
        )
        
        assert report.beats_class_prior is True  # 0.75 > 0.40
        assert report.beats_previous_label is True  # 0.75 > 0.70
    
    def test_computes_improvement(self, mock_metrics, mock_prior_metrics, mock_prev_metrics):
        """Should compute improvement percentages."""
        report = BaselineReport(
            model_name="TestModel",
            split="test",
            n_samples=1000,
            model_metrics=mock_metrics,
            class_prior_metrics=mock_prior_metrics,
            previous_label_metrics=mock_prev_metrics,
        )
        
        # 0.75 - 0.40 = 0.35 = 35pp
        assert abs(report.improvement_over_prior - 35.0) < 0.1
        # 0.75 - 0.70 = 0.05 = 5pp
        assert abs(report.improvement_over_previous - 5.0) < 0.1
    
    def test_summary_contains_key_info(self, mock_metrics, mock_prior_metrics, mock_prev_metrics):
        """Summary should contain key information."""
        report = BaselineReport(
            model_name="TestModel",
            split="test",
            n_samples=1000,
            model_metrics=mock_metrics,
            class_prior_metrics=mock_prior_metrics,
            previous_label_metrics=mock_prev_metrics,
        )
        
        summary = report.summary()
        
        assert "TestModel" in summary
        assert "test" in summary.lower()
        assert "1000" in summary
    
    def test_to_dict(self, mock_metrics, mock_prior_metrics, mock_prev_metrics):
        """to_dict should return all fields."""
        report = BaselineReport(
            model_name="TestModel",
            split="test",
            n_samples=1000,
            model_metrics=mock_metrics,
            class_prior_metrics=mock_prior_metrics,
            previous_label_metrics=mock_prev_metrics,
        )
        
        d = report.to_dict()
        
        assert d["model_name"] == "TestModel"
        assert d["split"] == "test"
        assert d["n_samples"] == 1000
        assert "model" in d
        assert "baselines" in d
        assert "comparison" in d
    
    def test_save(self, mock_metrics, mock_prior_metrics, mock_prev_metrics):
        """save should write valid JSON."""
        report = BaselineReport(
            model_name="TestModel",
            split="test",
            n_samples=1000,
            model_metrics=mock_metrics,
            class_prior_metrics=mock_prior_metrics,
            previous_label_metrics=mock_prev_metrics,
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            path = f.name
        
        try:
            report.save(path)
            
            with open(path) as f:
                data = json.load(f)
            
            assert data["model_name"] == "TestModel"
        finally:
            Path(path).unlink()


# =============================================================================
# create_baseline_report Tests
# =============================================================================


class TestCreateBaselineReport:
    """Test create_baseline_report function."""
    
    def test_creates_report(self, trained_naive_prior, sample_features, sample_labels):
        """Should create BaselineReport."""
        report = create_baseline_report(
            trained_naive_prior,
            sample_features,
            sample_labels,
        )
        
        assert isinstance(report, BaselineReport)
    
    def test_report_has_correct_n_samples(self, trained_naive_prior, sample_features, sample_labels):
        """Report should have correct sample count."""
        report = create_baseline_report(
            trained_naive_prior,
            sample_features,
            sample_labels,
        )
        
        assert report.n_samples == len(sample_labels)
    
    def test_report_has_model_name(self, trained_naive_prior, sample_features, sample_labels):
        """Report should use model's name."""
        report = create_baseline_report(
            trained_naive_prior,
            sample_features,
            sample_labels,
        )
        
        assert report.model_name == "NaiveClassPrior"


# =============================================================================
# full_evaluation Tests
# =============================================================================


class TestFullEvaluation:
    """Test full_evaluation function."""
    
    def test_returns_dict(self, trained_naive_prior, sample_features, sample_labels):
        """Should return dictionary with all metrics."""
        result = full_evaluation(
            trained_naive_prior,
            sample_features,
            sample_labels,
        )
        
        assert isinstance(result, dict)
    
    def test_contains_required_keys(self, trained_naive_prior, sample_features, sample_labels):
        """Should contain all required keys."""
        result = full_evaluation(
            trained_naive_prior,
            sample_features,
            sample_labels,
        )
        
        assert "split" in result
        assert "n_samples" in result
        assert "model_name" in result
        assert "classification" in result
        assert "trading" in result
        assert "transitions" in result
        assert "baseline_comparison" in result
    
    def test_classification_metrics_valid(self, trained_naive_prior, sample_features, sample_labels):
        """Classification metrics should be valid."""
        result = full_evaluation(
            trained_naive_prior,
            sample_features,
            sample_labels,
        )
        
        assert "accuracy" in result["classification"]
        assert 0 <= result["classification"]["accuracy"] <= 1


# =============================================================================
# Edge Cases
# =============================================================================


class TestEvaluationEdgeCases:
    """Test edge cases in evaluation."""
    
    def test_all_same_label(self):
        """Should handle all same label."""
        labels = np.ones(100, dtype=np.int64)
        features = np.random.randn(100, 10)
        
        model = NaiveClassPrior()
        model.fit(features, labels)
        
        metrics = evaluate_model(model, features, labels)
        
        # Perfect accuracy when all same
        assert metrics.accuracy == 1.0
    
    def test_two_classes_only(self):
        """Should handle only two classes present."""
        labels = np.array([0] * 50 + [1] * 50)
        features = np.random.randn(100, 10)
        
        results = evaluate_naive_baseline(labels)
        
        assert results['class_prior'] is not None
        assert results['previous_label'] is not None
    
    def test_single_sample(self):
        """Should handle single sample (edge case)."""
        labels = np.array([1])
        features = np.random.randn(1, 10)
        
        model = NaiveClassPrior()
        model.fit(features, labels)
        
        metrics = evaluate_model(model, features, labels)
        
        assert metrics.accuracy == 1.0
