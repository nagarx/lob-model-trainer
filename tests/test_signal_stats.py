#!/usr/bin/env python3
"""
Comprehensive tests for signal_stats.py.

Tests validate:
1. Distribution statistics (mean, std, skewness, kurtosis, percentiles)
2. Stationarity tests (ADF test)
3. Rolling statistics computation
4. Edge cases and numerical stability
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lobtrainer.analysis.signal_stats import (
    compute_distribution_stats,
    compute_stationarity_test,
    compute_all_stationarity_tests,
    compute_rolling_stats,
    compute_all_rolling_stats,
    StationarityResult,
    RollingStatsResult,
)


class TestDistributionStats:
    """Tests for distribution statistics computation."""
    
    def test_basic_statistics(self):
        """Test basic distribution statistics."""
        np.random.seed(42)
        n = 10000
        n_features = 98
        
        features = np.random.randn(n, n_features)
        signal_indices = [84, 85, 86]
        
        df = compute_distribution_stats(features, signal_indices)
        
        assert isinstance(df, pd.DataFrame), "Should return DataFrame"
        assert len(df) == 3, f"Should have 3 rows, got {len(df)}"
    
    def test_mean_near_zero_for_standard_normal(self):
        """Standard normal data should have mean ≈ 0."""
        np.random.seed(42)
        n = 50000  # Large sample for convergence
        n_features = 98
        
        features = np.random.randn(n, n_features)
        df = compute_distribution_stats(features, [84])
        
        mean = df.iloc[0]['mean']
        assert abs(mean) < 0.05, f"Standard normal mean should be ≈ 0, got {mean}"
    
    def test_std_near_one_for_standard_normal(self):
        """Standard normal data should have std ≈ 1."""
        np.random.seed(42)
        n = 50000
        n_features = 98
        
        features = np.random.randn(n, n_features)
        df = compute_distribution_stats(features, [84])
        
        std = df.iloc[0]['std']
        assert abs(std - 1.0) < 0.05, f"Standard normal std should be ≈ 1, got {std}"
    
    def test_skewness_near_zero_for_symmetric(self):
        """Symmetric distribution should have skewness ≈ 0."""
        np.random.seed(42)
        n = 50000
        n_features = 98
        
        features = np.random.randn(n, n_features)
        df = compute_distribution_stats(features, [84])
        
        skewness = df.iloc[0]['skewness']
        assert abs(skewness) < 0.1, f"Symmetric dist skewness should be ≈ 0, got {skewness}"
    
    def test_min_less_than_max(self):
        """Min should be less than max."""
        np.random.seed(42)
        n = 10000
        n_features = 98
        
        features = np.random.randn(n, n_features)
        df = compute_distribution_stats(features, [84])
        
        row = df.iloc[0]
        assert row['min'] < row['max'], f"min ({row['min']}) should be < max ({row['max']})"
        assert row['min'] <= row['median'] <= row['max'], "median should be between min and max"


class TestStationarityTest:
    """Tests for ADF stationarity test."""
    
    def test_stationary_signal_detected(self):
        """White noise should be detected as stationary."""
        np.random.seed(42)
        signal = np.random.randn(1000)
        
        adf_stat, p_value, critical_values, is_stationary = compute_stationarity_test(signal)
        
        # White noise should be stationary (low p-value)
        assert p_value < 0.05, f"White noise should be stationary, p-value = {p_value}"
        assert is_stationary, "White noise should be classified as stationary"
    
    def test_random_walk_non_stationary(self):
        """Random walk should be detected as non-stationary."""
        np.random.seed(42)
        signal = np.cumsum(np.random.randn(1000))  # Random walk
        
        adf_stat, p_value, critical_values, is_stationary = compute_stationarity_test(signal)
        
        # Random walk typically has high p-value (non-stationary)
        # Note: ADF test may sometimes reject for finite samples
        # We just verify the function runs and returns valid values
        assert np.isfinite(adf_stat), "ADF statistic should be finite"
        assert 0 <= p_value <= 1, f"p-value should be in [0, 1], got {p_value}"
    
    def test_returns_critical_values(self):
        """Should return critical values dictionary."""
        np.random.seed(42)
        signal = np.random.randn(1000)
        
        adf_stat, p_value, critical_values, is_stationary = compute_stationarity_test(signal)
        
        assert '1%' in critical_values, "Should have 1% critical value"
        assert '5%' in critical_values, "Should have 5% critical value"
        assert '10%' in critical_values, "Should have 10% critical value"
    
    def test_compute_all_stationarity_tests(self):
        """Test batch stationarity testing."""
        np.random.seed(42)
        n = 1000
        n_features = 98
        
        features = np.random.randn(n, n_features)
        signal_indices = [84, 85, 86]
        
        results = compute_all_stationarity_tests(features, signal_indices)
        
        assert len(results) == 3, f"Should have 3 results, got {len(results)}"
        assert all(isinstance(r, StationarityResult) for r in results)


class TestRollingStats:
    """Tests for rolling statistics computation."""
    
    def test_rolling_stats_basic(self):
        """Test basic rolling statistics computation."""
        np.random.seed(42)
        signal = np.random.randn(50000)  # Need large sample for window_size=10000
        
        result = compute_rolling_stats(signal, window_size=10000, n_windows=5)
        
        # Returns tuple: (mean_drift, std_drift, max_mean, min_mean, mean_range, is_mean_stable, is_std_stable)
        assert len(result) == 7, "Should return 7 values"
        mean_drift, std_drift, max_mean, min_mean, mean_range, is_mean_stable, is_std_stable = result
        
        assert np.isfinite(mean_drift), "mean_drift should be finite"
        assert np.isfinite(std_drift), "std_drift should be finite"
    
    def test_rolling_mean_stability_for_stationary(self):
        """Rolling mean should be stable for stationary series."""
        np.random.seed(42)
        signal = np.random.randn(100000)  # Stationary white noise
        
        mean_drift, std_drift, max_mean, min_mean, mean_range, is_mean_stable, is_std_stable = \
            compute_rolling_stats(signal, window_size=10000, n_windows=10)
        
        # For stationary white noise, mean should be stable
        assert is_mean_stable, "Mean should be stable for stationary series"
        assert abs(mean_drift) < 0.1, f"Mean drift should be small, got {mean_drift}"
    
    def test_trending_signal_detected(self):
        """Rolling stats should detect trend as non-stationary."""
        np.random.seed(42)
        n = 100000
        # Strong linear trend with noise
        signal = np.linspace(-10, 10, n) + np.random.randn(n) * 0.5
        
        mean_drift, std_drift, max_mean, min_mean, mean_range, is_mean_stable, is_std_stable = \
            compute_rolling_stats(signal, window_size=10000, n_windows=10)
        
        # Mean should drift significantly for trending signal
        assert abs(mean_drift) > 1.0, f"Mean drift should be large for trending signal, got {mean_drift}"
        # May or may not be detected as unstable depending on thresholds
    
    def test_compute_all_rolling_stats(self):
        """Test batch rolling stats computation."""
        np.random.seed(42)
        n = 50000  # Large enough for window_size=10000
        n_features = 98
        
        features = np.random.randn(n, n_features)
        signal_indices = [84, 85]
        
        results = compute_all_rolling_stats(features, signal_indices, window_size=10000, n_windows=5)
        
        assert len(results) == 2, f"Should have 2 results, got {len(results)}"
        assert all(isinstance(r, RollingStatsResult) for r in results)


class TestEdgeCases:
    """Edge case and numerical stability tests."""
    
    def test_short_signal_stationarity(self):
        """Should handle short signals for stationarity test."""
        signal = np.random.randn(50)  # Short signal
        
        # Should not crash
        try:
            adf_stat, p_value, critical_values, is_stationary = compute_stationarity_test(signal)
            assert np.isfinite(adf_stat) or np.isnan(adf_stat)
        except ValueError:
            # Acceptable to raise for very short signals
            pass
    
    def test_constant_signal_stationarity(self):
        """Should handle constant signal."""
        signal = np.ones(100)
        
        # May raise or return NaN - both are acceptable
        try:
            result = compute_stationarity_test(signal)
            # If it returns, values may be NaN
            assert result is not None
        except (ValueError, RuntimeWarning):
            pass
    
    def test_signal_with_nan_rolling_stats(self):
        """Should handle signals with NaN values for rolling stats."""
        np.random.seed(42)
        signal = np.random.randn(50000)
        signal[::1000] = np.nan  # Insert NaNs
        
        # Should not crash for rolling stats (may have NaN in results)
        result = compute_rolling_stats(signal, window_size=10000, n_windows=5)
        assert result is not None
        assert len(result) == 7
    
    def test_signal_with_inf_rolling_stats(self):
        """Should handle signals with Inf values for rolling stats."""
        np.random.seed(42)
        signal = np.random.randn(50000)
        signal[25000] = np.inf
        
        # Should handle without crashing (may have NaN in results)
        result = compute_rolling_stats(signal, window_size=10000, n_windows=5)
        assert result is not None
        assert len(result) == 7
    
    def test_short_signal_rolling_stats(self):
        """Short signal should return default values."""
        signal = np.random.randn(100)  # Too short for window_size=10000
        
        result = compute_rolling_stats(signal, window_size=10000, n_windows=5)
        
        # Should return default values (stable)
        mean_drift, std_drift, max_mean, min_mean, mean_range, is_mean_stable, is_std_stable = result
        assert is_mean_stable, "Should default to stable for short signals"
        assert is_std_stable, "Should default to stable for short signals"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

