"""
Tests for baseline models.

Tests:
- BaseModel interface
- NaivePreviousLabel baseline
- NaiveClassPrior baseline
- LogisticBaseline

RULE.md compliance:
- Baselines establish performance floor
- Deterministic outputs for same inputs
- Proper handling of edge cases
"""

import pytest
import numpy as np

from lobtrainer.models.baselines import (
    BaseModel,
    NaivePreviousLabel,
    NaiveClassPrior,
    LogisticBaseline,
    LogisticBaselineConfig,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_features():
    """Create sample features."""
    np.random.seed(42)
    return np.random.randn(100, 98)


@pytest.fixture
def sample_labels():
    """Create sample labels."""
    np.random.seed(42)
    return np.random.choice([0, 1, 2], size=100)


@pytest.fixture
def imbalanced_labels():
    """Create imbalanced labels (80% class 1)."""
    np.random.seed(42)
    return np.array([1] * 80 + [0] * 10 + [2] * 10)


# =============================================================================
# NaiveClassPrior Tests
# =============================================================================


class TestNaiveClassPrior:
    """Test NaiveClassPrior baseline."""
    
    def test_fit_computes_majority_class(self, sample_features, imbalanced_labels):
        """Should identify majority class."""
        model = NaiveClassPrior()
        model.fit(sample_features, imbalanced_labels)
        
        # Class 1 is majority (80%)
        assert model._class_prior == 1
    
    def test_predict_returns_majority_class(self, sample_features, imbalanced_labels):
        """Should always predict majority class."""
        model = NaiveClassPrior()
        model.fit(sample_features, imbalanced_labels)
        
        predictions = model.predict(sample_features)
        
        assert np.all(predictions == 1), "Should always predict majority class"
    
    def test_predict_correct_shape(self, sample_features, sample_labels):
        """Predictions should have correct shape."""
        model = NaiveClassPrior()
        model.fit(sample_features, sample_labels)
        
        predictions = model.predict(sample_features)
        
        assert predictions.shape == (len(sample_features),)
    
    def test_predict_proba_sums_to_one(self, sample_features, sample_labels):
        """Probabilities should sum to 1."""
        model = NaiveClassPrior()
        model.fit(sample_features, sample_labels)
        
        proba = model.predict_proba(sample_features)
        
        row_sums = proba.sum(axis=1)
        assert np.allclose(row_sums, 1.0), "Probabilities should sum to 1"
    
    def test_predict_before_fit_raises(self, sample_features):
        """Should raise if predict called before fit."""
        model = NaiveClassPrior()
        
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(sample_features)
    
    def test_name_property(self):
        """Should have correct name."""
        model = NaiveClassPrior()
        assert model.name == "NaiveClassPrior"
    
    def test_deterministic(self, sample_features, sample_labels):
        """Same inputs should give same outputs."""
        model1 = NaiveClassPrior()
        model1.fit(sample_features, sample_labels)
        pred1 = model1.predict(sample_features)
        
        model2 = NaiveClassPrior()
        model2.fit(sample_features, sample_labels)
        pred2 = model2.predict(sample_features)
        
        np.testing.assert_array_equal(pred1, pred2)


# =============================================================================
# NaivePreviousLabel Tests
# =============================================================================


class TestNaivePreviousLabel:
    """Test NaivePreviousLabel baseline."""
    
    def test_fit_stores_class_prior(self, sample_features, imbalanced_labels):
        """Should store class prior for first sample."""
        model = NaivePreviousLabel()
        model.fit(sample_features, imbalanced_labels)
        
        assert model._class_prior == 1  # Majority class
    
    def test_predict_returns_prior_for_all(self, sample_features, sample_labels):
        """predict() returns class prior for all samples."""
        model = NaivePreviousLabel()
        model.fit(sample_features, sample_labels)
        
        predictions = model.predict(sample_features)
        
        # Note: basic predict() returns prior, not shifted labels
        # Use predict_with_history for true previous-label behavior
        assert predictions.shape == (len(sample_features),)
    
    def test_predict_with_history(self, sample_features, sample_labels):
        """predict_with_history should return given history."""
        model = NaivePreviousLabel()
        model.fit(sample_features, sample_labels)
        
        # Shift labels by 1 as "previous labels"
        history = np.concatenate([[sample_labels[0]], sample_labels[:-1]])
        
        predictions = model.predict_with_history(sample_features, history)
        
        np.testing.assert_array_equal(predictions, history)
    
    def test_predict_before_fit_raises(self, sample_features):
        """Should raise if predict called before fit."""
        model = NaivePreviousLabel()
        
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(sample_features)
    
    def test_name_property(self):
        """Should have correct name."""
        model = NaivePreviousLabel()
        assert model.name == "NaivePreviousLabel"


# =============================================================================
# LogisticBaseline Tests
# =============================================================================


class TestLogisticBaselineConfig:
    """Test LogisticBaselineConfig dataclass."""
    
    def test_default_values(self):
        """Default values should be reasonable."""
        config = LogisticBaselineConfig()
        
        assert config.C == 1.0
        assert config.max_iter == 1000
        assert config.solver == "lbfgs"
        assert config.class_weight == "balanced"
        assert config.normalize is True
        assert config.random_state == 42
    
    def test_custom_values(self):
        """Custom values should be accepted."""
        config = LogisticBaselineConfig(
            C=0.1,
            max_iter=500,
            normalize=False,
        )
        
        assert config.C == 0.1
        assert config.max_iter == 500
        assert config.normalize is False


class TestLogisticBaseline:
    """Test LogisticBaseline model."""
    
    def test_fit_2d_features(self, sample_features, sample_labels):
        """Should fit on 2D features."""
        model = LogisticBaseline()
        model.fit(sample_features, sample_labels)
        
        assert model._fitted is True
    
    def test_fit_3d_features(self, sample_labels):
        """Should handle 3D features (sequences) by using last timestep."""
        # Create sequence features
        features_3d = np.random.randn(100, 10, 98)  # (samples, seq_len, features)
        
        model = LogisticBaseline()
        model.fit(features_3d, sample_labels)
        
        assert model._fitted is True
    
    def test_predict_shape(self, sample_features, sample_labels):
        """Predictions should have correct shape."""
        model = LogisticBaseline()
        model.fit(sample_features, sample_labels)
        
        predictions = model.predict(sample_features)
        
        assert predictions.shape == (len(sample_features),)
    
    def test_predict_valid_classes(self, sample_features, sample_labels):
        """Predictions should be valid class labels."""
        model = LogisticBaseline()
        model.fit(sample_features, sample_labels)
        
        predictions = model.predict(sample_features)
        
        unique_preds = np.unique(predictions)
        unique_labels = np.unique(sample_labels)
        
        # Predictions should be subset of training labels
        assert all(p in unique_labels for p in unique_preds)
    
    def test_predict_proba_shape(self, sample_features, sample_labels):
        """Probabilities should have correct shape."""
        model = LogisticBaseline()
        model.fit(sample_features, sample_labels)
        
        proba = model.predict_proba(sample_features)
        
        n_classes = len(np.unique(sample_labels))
        assert proba.shape == (len(sample_features), n_classes)
    
    def test_predict_proba_sums_to_one(self, sample_features, sample_labels):
        """Probabilities should sum to 1."""
        model = LogisticBaseline()
        model.fit(sample_features, sample_labels)
        
        proba = model.predict_proba(sample_features)
        
        row_sums = proba.sum(axis=1)
        assert np.allclose(row_sums, 1.0)
    
    def test_handles_nan(self, sample_labels):
        """Should handle NaN values in features."""
        features = np.random.randn(100, 98)
        features[0, 0] = np.nan
        features[50, 50] = np.inf
        
        model = LogisticBaseline()
        # Should not raise during fit (removes invalid samples)
        model.fit(features, sample_labels)
        
        # Should not raise during predict (replaces with 0)
        predictions = model.predict(features)
        
        assert predictions.shape == (100,)
        assert np.all(np.isfinite(predictions))
    
    def test_feature_importance(self, sample_features, sample_labels):
        """Should compute feature importance."""
        model = LogisticBaseline()
        model.fit(sample_features, sample_labels)
        
        importance = model.feature_importance
        
        assert importance.shape == (98,)
        assert np.all(importance >= 0)  # Absolute values
    
    def test_coef_property(self, sample_features, sample_labels):
        """Should expose coefficients."""
        model = LogisticBaseline()
        model.fit(sample_features, sample_labels)
        
        coef = model.coef_
        
        n_classes = len(np.unique(sample_labels))
        assert coef.shape[0] == n_classes
        assert coef.shape[1] == 98
    
    def test_predict_before_fit_raises(self, sample_features):
        """Should raise if predict called before fit."""
        model = LogisticBaseline()
        
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(sample_features)
    
    def test_name_property(self):
        """Should have correct name."""
        model = LogisticBaseline()
        assert model.name == "LogisticRegression"
    
    def test_no_normalization(self, sample_features, sample_labels):
        """Should work without normalization."""
        config = LogisticBaselineConfig(normalize=False)
        model = LogisticBaseline(config)
        
        model.fit(sample_features, sample_labels)
        predictions = model.predict(sample_features)
        
        assert predictions.shape == (100,)
    
    def test_deterministic_with_seed(self, sample_features, sample_labels):
        """Same seed should give same results."""
        config = LogisticBaselineConfig(random_state=42)
        
        model1 = LogisticBaseline(config)
        model1.fit(sample_features, sample_labels)
        pred1 = model1.predict(sample_features)
        
        model2 = LogisticBaseline(config)
        model2.fit(sample_features, sample_labels)
        pred2 = model2.predict(sample_features)
        
        np.testing.assert_array_equal(pred1, pred2)


# =============================================================================
# BaseModel Interface Tests
# =============================================================================


class TestBaseModelInterface:
    """Test that all models implement BaseModel interface correctly."""
    
    @pytest.mark.parametrize("model_class", [
        NaiveClassPrior,
        NaivePreviousLabel,
        LogisticBaseline,
    ])
    def test_has_fit_method(self, model_class):
        """All models should have fit method."""
        model = model_class()
        assert hasattr(model, 'fit')
        assert callable(model.fit)
    
    @pytest.mark.parametrize("model_class", [
        NaiveClassPrior,
        NaivePreviousLabel,
        LogisticBaseline,
    ])
    def test_has_predict_method(self, model_class):
        """All models should have predict method."""
        model = model_class()
        assert hasattr(model, 'predict')
        assert callable(model.predict)
    
    @pytest.mark.parametrize("model_class", [
        NaiveClassPrior,
        NaivePreviousLabel,
        LogisticBaseline,
    ])
    def test_has_name_property(self, model_class):
        """All models should have name property."""
        model = model_class()
        assert hasattr(model, 'name')
        assert isinstance(model.name, str)
        assert len(model.name) > 0
    
    @pytest.mark.parametrize("model_class", [
        NaiveClassPrior,
        NaivePreviousLabel,
        LogisticBaseline,
    ])
    def test_fit_returns_self(self, model_class, sample_features, sample_labels):
        """fit should return self."""
        model = model_class()
        result = model.fit(sample_features, sample_labels)
        
        assert result is model


# =============================================================================
# Edge Cases
# =============================================================================


class TestBaselineEdgeCases:
    """Test edge cases for baselines."""
    
    def test_single_class(self):
        """Should handle single class."""
        features = np.random.randn(50, 10)
        labels = np.ones(50, dtype=np.int64)
        
        model = NaiveClassPrior()
        model.fit(features, labels)
        
        predictions = model.predict(features)
        
        np.testing.assert_array_equal(predictions, 1)
    
    def test_binary_classification(self):
        """Should handle binary classification."""
        features = np.random.randn(100, 10)
        labels = np.array([0] * 50 + [1] * 50)
        
        model = LogisticBaseline()
        model.fit(features, labels)
        
        predictions = model.predict(features)
        
        assert set(predictions) <= {0, 1}
    
    def test_large_number_of_classes(self):
        """Should handle many classes."""
        np.random.seed(42)
        features = np.random.randn(500, 20)
        labels = np.random.choice(range(10), size=500)
        
        model = LogisticBaseline()
        model.fit(features, labels)
        
        predictions = model.predict(features)
        
        assert predictions.shape == (500,)
    
    def test_high_dimensional_features(self):
        """Should handle high-dimensional features."""
        np.random.seed(42)
        features = np.random.randn(200, 500)  # More features than samples
        labels = np.random.choice([0, 1, 2], size=200)
        
        # Logistic regression might not converge well, but shouldn't crash
        config = LogisticBaselineConfig(max_iter=100)
        model = LogisticBaseline(config)
        
        model.fit(features, labels)
        predictions = model.predict(features)
        
        assert predictions.shape == (200,)
