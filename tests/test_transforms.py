"""
Tests for feature transforms and utilities.
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest

from lobtrainer.data.transforms import (
    FeatureStatistics,
    compute_statistics,
)
from lobtrainer.constants import FeatureIndex


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
