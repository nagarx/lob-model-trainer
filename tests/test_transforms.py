"""
Tests for feature transforms and normalization.
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest

from lobtrainer.data.transforms import (
    FeatureStatistics,
    compute_statistics,
    ZScoreNormalizer,
    PerDayNormalizer,
)
from lobtrainer.constants import FEATURE_COUNT, FeatureIndex


class TestFeatureStatistics:
    """Test FeatureStatistics dataclass."""
    
    def test_creation(self):
        """Basic statistics creation."""
        n_features = 98
        stats = FeatureStatistics(
            mean=np.zeros(n_features),
            std=np.ones(n_features),
            min=np.zeros(n_features),
            max=np.ones(n_features),
            median=np.zeros(n_features),
            q25=np.zeros(n_features),
            q75=np.ones(n_features),
            count=1000,
        )
        assert stats.count == 1000
        assert len(stats.mean) == 98
    
    def test_iqr_property(self):
        """IQR should be q75 - q25."""
        n_features = 98
        stats = FeatureStatistics(
            mean=np.zeros(n_features),
            std=np.ones(n_features),
            min=np.zeros(n_features),
            max=np.ones(n_features),
            median=np.ones(n_features) * 0.5,
            q25=np.ones(n_features) * 0.25,
            q75=np.ones(n_features) * 0.75,
            count=1000,
        )
        np.testing.assert_array_almost_equal(stats.iqr, np.ones(n_features) * 0.5)
    
    def test_serialization_round_trip(self):
        """Statistics should survive JSON round trip."""
        n_features = 98
        stats = FeatureStatistics(
            mean=np.random.randn(n_features),
            std=np.abs(np.random.randn(n_features)) + 0.1,
            min=np.random.randn(n_features),
            max=np.random.randn(n_features),
            median=np.random.randn(n_features),
            q25=np.random.randn(n_features),
            q75=np.random.randn(n_features),
            count=1000,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stats.json"
            stats.save(path)
            loaded = FeatureStatistics.load(path)
        
        np.testing.assert_array_almost_equal(loaded.mean, stats.mean)
        np.testing.assert_array_almost_equal(loaded.std, stats.std)
        assert loaded.count == stats.count


class TestComputeStatistics:
    """Test compute_statistics function."""
    
    def test_basic_statistics(self):
        """Compute correct mean and std."""
        np.random.seed(42)
        n_samples, n_features = 1000, 98
        
        # Generate data with known statistics
        mean = np.arange(n_features, dtype=np.float64)
        std = np.ones(n_features) * 2.0
        features = np.random.randn(n_samples, n_features) * std + mean
        
        # Add book_valid column (all valid) - this is a special column
        features[:, FeatureIndex.BOOK_VALID] = 1.0
        # Update expected mean for book_valid
        mean[FeatureIndex.BOOK_VALID] = 1.0
        
        stats = compute_statistics(features)
        
        # Check mean is close (within 5 std errors for most features)
        # Use larger tolerance due to sample variance
        for i in range(n_features):
            expected_error = std[i] / np.sqrt(n_samples)
            actual_diff = abs(stats.mean[i] - mean[i])
            tolerance = 5 * expected_error + 0.2  # Generous tolerance
            assert actual_diff < tolerance, (
                f"Feature {i}: expected mean {mean[i]:.4f}, got {stats.mean[i]:.4f}, "
                f"diff {actual_diff:.4f} > tolerance {tolerance:.4f}"
            )
    
    def test_handles_nan_values(self):
        """Statistics should handle NaN values."""
        features = np.ones((100, 98))
        features[:, FeatureIndex.BOOK_VALID] = 1.0
        features[0, 0] = np.nan  # Single NaN
        
        stats = compute_statistics(features)
        
        # Mean of feature 0 should be computed ignoring NaN
        assert np.isfinite(stats.mean[0])
    
    def test_excludes_invalid_samples(self):
        """Invalid samples (book_valid=0) should be excluded."""
        features = np.ones((100, 98))
        features[:50, FeatureIndex.BOOK_VALID] = 0.0  # First 50 invalid
        features[50:, FeatureIndex.BOOK_VALID] = 1.0  # Last 50 valid
        features[:50, 0] = 1000.0  # Invalid samples have different value
        features[50:, 0] = 1.0  # Valid samples have value 1
        
        stats = compute_statistics(features)
        
        # Mean should be close to 1.0, not affected by invalid samples
        assert abs(stats.mean[0] - 1.0) < 0.01
        assert stats.count == 50  # Only valid samples counted


class TestZScoreNormalizer:
    """Test ZScoreNormalizer."""
    
    def test_basic_normalization(self):
        """Z-score should produce zero mean and unit variance."""
        np.random.seed(42)
        features = np.random.randn(1000, 98) * 10 + 5
        features[:, FeatureIndex.BOOK_VALID] = 1.0
        features[:, FeatureIndex.TIME_REGIME] = 2.0  # Should not be normalized
        
        normalizer = ZScoreNormalizer()
        normalized = normalizer.fit_transform(features)
        
        # Check mean is near zero for normalized features
        for i in range(98):
            if i == FeatureIndex.TIME_REGIME:
                # Should be unchanged
                assert abs(normalized[:, i].mean() - 2.0) < 0.01
            elif i != FeatureIndex.BOOK_VALID:
                assert abs(normalized[:, i].mean()) < 0.1
                assert abs(normalized[:, i].std() - 1.0) < 0.1
    
    def test_excludes_time_regime(self):
        """Time regime should not be normalized."""
        features = np.ones((100, 98))
        features[:, FeatureIndex.BOOK_VALID] = 1.0
        features[:, FeatureIndex.TIME_REGIME] = 2.0
        
        normalizer = ZScoreNormalizer()
        normalized = normalizer.fit_transform(features)
        
        # Time regime should be unchanged
        np.testing.assert_array_equal(
            normalized[:, FeatureIndex.TIME_REGIME],
            features[:, FeatureIndex.TIME_REGIME],
        )
    
    def test_clipping(self):
        """Extreme values should be clipped."""
        features = np.zeros((100, 98))
        features[:, FeatureIndex.BOOK_VALID] = 1.0
        features[0, 0] = 1000.0  # Extreme outlier
        
        normalizer = ZScoreNormalizer(clip_value=5.0)
        normalized = normalizer.fit_transform(features)
        
        # All values should be within clip range (except excluded features)
        for i in range(98):
            if i not in normalizer.exclude_features:
                assert normalized[:, i].max() <= 5.0
                assert normalized[:, i].min() >= -5.0
    
    def test_inverse_transform(self):
        """Inverse should recover original values (approximately)."""
        np.random.seed(42)
        features = np.random.randn(100, 98) * 10 + 5
        features[:, FeatureIndex.BOOK_VALID] = 1.0
        
        normalizer = ZScoreNormalizer(clip_value=None)  # No clipping for exact inverse
        normalized = normalizer.fit_transform(features)
        recovered = normalizer.inverse_transform(normalized)
        
        # Should recover original values
        np.testing.assert_array_almost_equal(recovered, features, decimal=10)
    
    def test_serialization(self):
        """Normalizer should survive save/load."""
        np.random.seed(42)
        features = np.random.randn(100, 98)
        features[:, FeatureIndex.BOOK_VALID] = 1.0
        
        normalizer = ZScoreNormalizer()
        normalizer.fit(features)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "normalizer.json"
            normalizer.save(path)
            loaded = ZScoreNormalizer.load(path)
        
        # Loaded normalizer should produce same results
        test_features = np.random.randn(10, 98)
        test_features[:, FeatureIndex.BOOK_VALID] = 1.0
        
        original_result = normalizer.transform(test_features)
        loaded_result = loaded.transform(test_features)
        
        np.testing.assert_array_almost_equal(original_result, loaded_result)
    
    def test_not_fitted_raises(self):
        """Transform before fit should raise."""
        normalizer = ZScoreNormalizer()
        with pytest.raises(RuntimeError, match="not fitted"):
            normalizer.transform(np.ones((10, 98)))
    
    def test_3d_input(self):
        """Should handle 3D input (sequences)."""
        np.random.seed(42)
        features_2d = np.random.randn(1000, 98) * 10 + 5
        features_2d[:, FeatureIndex.BOOK_VALID] = 1.0
        
        normalizer = ZScoreNormalizer()
        normalizer.fit(features_2d)
        
        # 3D input: (batch, sequence, features)
        features_3d = np.random.randn(10, 100, 98) * 10 + 5
        features_3d[..., FeatureIndex.BOOK_VALID] = 1.0
        
        normalized = normalizer.transform(features_3d)
        assert normalized.shape == (10, 100, 98)

