#!/usr/bin/env python3
"""
Comprehensive tests for data_overview.py.

Tests validate:
1. Data quality checks (NaN, Inf)
2. Categorical feature validation
3. Label distribution computation
4. Signal statistics
"""

import pytest
import numpy as np
import sys
from pathlib import Path
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lobtrainer.analysis.data_overview import (
    compute_data_quality,
    compute_label_distribution,
    validate_categorical_feature,
    compute_all_categorical_validations,
    compute_signal_statistics,
    DataQuality,
    LabelDistribution,
    CategoricalValidation,
    SignalStatistics,
)
from lobtrainer.constants import LABEL_UP, LABEL_DOWN, LABEL_STABLE


class TestDataQuality:
    """Tests for data quality computation."""
    
    def test_clean_data_no_issues(self):
        """Clean data should have no NaN or Inf."""
        np.random.seed(42)
        features = np.random.randn(1000, 98)
        
        quality = compute_data_quality(features)
        
        assert quality.nan_count == 0, f"Expected 0 NaN, got {quality.nan_count}"
        assert quality.inf_count == 0, f"Expected 0 Inf, got {quality.inf_count}"
        assert quality.pct_nan == 0.0
        assert quality.pct_inf == 0.0
        assert quality.is_clean
    
    def test_detects_nan_values(self):
        """Should detect NaN values."""
        features = np.random.randn(1000, 98)
        features[0, 0] = np.nan
        features[100, 50] = np.nan
        
        quality = compute_data_quality(features)
        
        assert quality.nan_count == 2, f"Expected 2 NaN, got {quality.nan_count}"
        assert quality.pct_nan > 0
        assert not quality.is_clean
    
    def test_detects_inf_values(self):
        """Should detect Inf values."""
        features = np.random.randn(1000, 98)
        features[0, 0] = np.inf
        features[100, 50] = -np.inf
        
        quality = compute_data_quality(features)
        
        assert quality.inf_count == 2, f"Expected 2 Inf, got {quality.inf_count}"
        assert not quality.is_clean
    
    def test_shape_info(self):
        """Should report correct shape."""
        features = np.random.randn(1234, 98)
        
        quality = compute_data_quality(features)
        
        # total_values = n_samples * n_features
        assert quality.total_values == 1234 * 98


class TestLabelDistribution:
    """Tests for label distribution computation."""
    
    def test_counts_sum_to_total(self):
        """Label counts should sum to total."""
        labels = np.array([1, 0, -1, 1, 0, -1, 0])
        
        dist = compute_label_distribution(labels)
        
        total = dist.up_count + dist.stable_count + dist.down_count
        assert total == len(labels)
    
    def test_percentages_sum_to_100(self):
        """Percentages should sum to 100."""
        labels = np.random.choice([-1, 0, 1], size=1000)
        
        dist = compute_label_distribution(labels)
        
        total_pct = dist.up_pct + dist.stable_pct + dist.down_pct
        assert abs(total_pct - 100.0) < 0.01
    
    def test_known_distribution(self):
        """Test with known distribution."""
        labels = np.array([1]*25 + [0]*50 + [-1]*25)  # 25/50/25
        
        dist = compute_label_distribution(labels)
        
        assert dist.up_pct == 25.0
        assert dist.stable_pct == 50.0
        assert dist.down_pct == 25.0


class TestCategoricalValidation:
    """Tests for categorical feature validation."""
    
    def test_binary_feature_valid(self):
        """Binary feature with 0/1 should be valid."""
        from lobtrainer.constants import FeatureIndex
        np.random.seed(42)
        features = np.random.randn(1000, 98)
        
        # Set BOOK_VALID to binary values
        features[:, FeatureIndex.BOOK_VALID] = np.random.choice([0.0, 1.0], size=1000)
        
        # API: validate_categorical_feature(features, name, index, expected_values)
        result = validate_categorical_feature(
            features, 
            name="book_valid",
            index=FeatureIndex.BOOK_VALID, 
            expected_values=[0.0, 1.0],
        )
        
        assert result.is_valid, f"Binary 0/1 should be valid: {result.unique_values}"
    
    def test_invalid_values_detected(self):
        """Should detect unexpected values."""
        from lobtrainer.constants import FeatureIndex
        features = np.random.randn(1000, 98)
        
        # Set to unexpected values
        features[:, FeatureIndex.BOOK_VALID] = 5.0
        
        result = validate_categorical_feature(
            features,
            name="book_valid",
            index=FeatureIndex.BOOK_VALID,
            expected_values=[0.0, 1.0],
        )
        
        assert not result.is_valid


class TestSignalStatistics:
    """Tests for signal statistics computation."""
    
    def test_returns_stats_per_signal(self):
        """Should return stats for each core signal."""
        np.random.seed(42)
        features = np.random.randn(1000, 98)
        
        stats = compute_signal_statistics(features)
        
        # Should have stats for core signals (indices 84-91)
        assert len(stats) >= 8, f"Expected >= 8 signal stats, got {len(stats)}"
        assert all(isinstance(s, SignalStatistics) for s in stats)
    
    def test_stats_are_finite(self):
        """All statistics should be finite."""
        np.random.seed(42)
        features = np.random.randn(1000, 98)
        
        stats = compute_signal_statistics(features)
        
        for s in stats:
            assert np.isfinite(s.mean), f"{s.name} mean not finite"
            assert np.isfinite(s.std), f"{s.name} std not finite"
    
    def test_std_positive(self):
        """Standard deviation should be positive."""
        np.random.seed(42)
        features = np.random.randn(1000, 98)
        
        stats = compute_signal_statistics(features)
        
        for s in stats:
            assert s.std >= 0, f"{s.name} std should be >= 0"


class TestComputeAllCategoricalValidations:
    """Tests for batch categorical validation."""
    
    def test_returns_validations_for_all_categorical(self):
        """Should validate all categorical features."""
        from lobtrainer.constants import FeatureIndex
        np.random.seed(42)
        features = np.random.randn(1000, 98)
        
        # Set categorical features to valid values
        features[:, FeatureIndex.BOOK_VALID] = np.random.choice([0.0, 1.0], size=1000)
        features[:, FeatureIndex.TIME_REGIME] = np.random.choice([0.0, 1.0, 2.0, 3.0], size=1000)
        features[:, FeatureIndex.MBO_READY] = np.random.choice([0.0, 1.0], size=1000)
        
        validations = compute_all_categorical_validations(features)
        
        # Should have validations for at least 3 categorical features
        assert len(validations) >= 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

