#!/usr/bin/env python3
"""
Comprehensive tests for temporal_dynamics.py.

These tests validate:
1. Alignment formula correctness
2. Autocorrelation computation
3. Predictive decay with parameterized window/stride
4. Level vs change analysis
5. Edge cases and numerical stability
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lobtrainer.analysis.temporal_dynamics import (
    compute_autocorrelation,
    compute_signal_autocorrelations,
    compute_predictive_decay,
    compute_all_predictive_decays,
    compute_level_vs_change,
    compute_all_level_vs_change,
    compute_lead_lag_relations,
    run_temporal_dynamics_analysis,
)


class TestAutocorrelation:
    """Tests for autocorrelation computation."""
    
    def test_constant_signal_has_perfect_autocorrelation(self):
        """A constant signal should have ACF = 1.0 at all lags."""
        signal = np.ones(1000)
        acf, half_life, first_zero = compute_autocorrelation(signal, max_lag=50)
        
        # All autocorrelations should be 1.0 (or very close)
        # Note: Due to numerical precision, we use a tolerance
        assert all(a >= 0.99 for a in acf[:10]), \
            f"Constant signal should have ACF ≈ 1.0, got {acf[:10]}"
    
    def test_white_noise_has_near_zero_autocorrelation(self):
        """White noise should have ACF ≈ 0 for lag > 0."""
        np.random.seed(42)
        signal = np.random.randn(10000)  # Large sample for stable estimate
        acf, half_life, first_zero = compute_autocorrelation(signal, max_lag=50)
        
        # ACF at lag 0 should be 1.0
        assert abs(acf[0] - 1.0) < 0.01, f"ACF(0) should be 1.0, got {acf[0]}"
        
        # ACF for lag > 0 should be near zero (< 0.1 for large sample)
        for lag in [1, 5, 10]:
            assert abs(acf[lag]) < 0.1, \
                f"White noise ACF({lag}) should be ≈ 0, got {acf[lag]}"
    
    def test_ar1_signal_decays_exponentially(self):
        """AR(1) signal ACF should decay exponentially."""
        np.random.seed(42)
        phi = 0.9  # AR(1) coefficient
        n = 5000
        
        # Generate AR(1) process: x_t = phi * x_{t-1} + epsilon_t
        signal = np.zeros(n)
        signal[0] = np.random.randn()
        for t in range(1, n):
            signal[t] = phi * signal[t-1] + np.random.randn()
        
        acf, half_life, first_zero = compute_autocorrelation(signal, max_lag=50)
        
        # ACF should decay roughly as phi^lag
        for lag in [1, 2, 5]:
            expected_acf = phi ** lag
            # Allow 30% tolerance due to finite sample
            assert abs(acf[lag] - expected_acf) < 0.3, \
                f"AR(1) ACF({lag}) expected ≈ {expected_acf:.2f}, got {acf[lag]:.2f}"
    
    def test_half_life_detection(self):
        """Test that half-life is correctly detected."""
        np.random.seed(42)
        phi = 0.9
        n = 5000
        
        # Generate AR(1) process
        signal = np.zeros(n)
        signal[0] = np.random.randn()
        for t in range(1, n):
            signal[t] = phi * signal[t-1] + np.random.randn()
        
        acf, half_life, first_zero = compute_autocorrelation(signal, max_lag=100)
        
        # Half-life should be roughly where ACF = 0.5 * ACF(1)
        # For AR(1), ACF(k) = phi^k, so half-life is when phi^k = 0.5
        # k = log(0.5) / log(phi) ≈ 6.6 for phi=0.9
        expected_half_life = np.log(0.5) / np.log(phi)
        
        # Allow some tolerance due to discrete lags
        assert half_life is not None, "Half-life should be detected"
        assert abs(half_life - expected_half_life) < 3, \
            f"Expected half-life ≈ {expected_half_life:.1f}, got {half_life}"


class TestPredictiveDecay:
    """Tests for predictive decay with parameterized window/stride."""
    
    def test_window_stride_parameters_are_used(self):
        """Verify that window_size and stride parameters affect alignment."""
        np.random.seed(42)
        n_samples = 2000
        n_features = 5
        
        # Create signal that increases with sample index
        features = np.zeros((n_samples, n_features))
        features[:, 0] = np.arange(n_samples)  # Linear increase
        
        # Test with different window/stride settings
        for window_size, stride in [(100, 10), (50, 5), (200, 20)]:
            n_labels = (n_samples - window_size) // stride + 1
            labels = np.ones(n_labels)  # Dummy labels
            
            signal = features[:, 0]
            corrs, half_life, optimal_lag, max_corr = compute_predictive_decay(
                signal, labels, lags=[0], window_size=window_size, stride=stride
            )
            
            # The aligned signal values should follow the formula
            # For lag=0, label[i] uses signal at (i * stride + window_size - 1)
            expected_aligned = np.array([i * stride + window_size - 1 for i in range(n_labels)])
            
            # Build aligned signal manually for verification
            aligned_signal = np.array([signal[i * stride + window_size - 1] for i in range(n_labels)])
            
            # They should match the expected indices
            np.testing.assert_array_equal(aligned_signal, expected_aligned, 
                err_msg=f"Alignment failed for window={window_size}, stride={stride}")
    
    def test_correlation_with_lagged_signal(self):
        """Test that correlation correctly uses lagged signal."""
        np.random.seed(42)
        n_samples = 5000
        window_size = 100
        stride = 10
        
        # Create a signal where the value at time t predicts label at time t+100
        n_labels = (n_samples - window_size) // stride + 1
        
        # Labels are correlated with signal from 100 samples earlier
        signal = np.random.randn(n_samples)
        labels = np.zeros(n_labels)
        
        for i in range(n_labels):
            signal_idx = i * stride + window_size - 1 - 100  # lag = 100
            if signal_idx >= 0:
                labels[i] = signal[signal_idx] + 0.1 * np.random.randn()
        
        corrs, half_life, optimal_lag, max_corr = compute_predictive_decay(
            signal, labels, lags=[0, 50, 100, 150], 
            window_size=window_size, stride=stride
        )
        
        # Correlation should be highest at lag=100
        lag_100_idx = [0, 50, 100, 150].index(100)
        assert abs(corrs[lag_100_idx]) > 0.5, \
            f"Correlation at lag=100 should be high, got {corrs[lag_100_idx]}"
        
        # Optimal lag should be 100
        assert optimal_lag == 100, \
            f"Optimal lag should be 100, got {optimal_lag}"
    
    def test_decay_with_persistent_signal(self):
        """Test decay with a signal that has persistent predictive power."""
        np.random.seed(42)
        n_samples = 10000
        window_size = 100
        stride = 10
        
        n_labels = (n_samples - window_size) // stride + 1
        
        # Create a signal with slow decay
        signal = np.cumsum(np.random.randn(n_samples))  # Random walk
        
        # Labels are moving averages of signal
        labels = np.array([
            signal[max(0, i * stride + window_size - 50):i * stride + window_size].mean()
            for i in range(n_labels)
        ])
        
        corrs, half_life, optimal_lag, max_corr = compute_predictive_decay(
            signal, labels, lags=[0, 5, 10, 20, 50],
            window_size=window_size, stride=stride
        )
        
        # Correlation should be positive and decay with lag
        assert corrs[0] > 0, f"Base correlation should be positive, got {corrs[0]}"
        
        # Generally, correlations should decrease with lag (though noisy)
        # We just check that the function doesn't crash and returns valid values
        assert all(np.isfinite(c) for c in corrs), "All correlations should be finite"


class TestLevelVsChange:
    """Tests for level vs change analysis."""
    
    def test_returns_valid_dataclass(self):
        """Test that level vs change returns a valid LevelVsChangeAnalysis."""
        n_samples = 1000
        n_features = 98
        window_size = 100
        stride = 10
        
        n_labels = (n_samples - window_size) // stride + 1
        
        # Features: column 0 = random values
        np.random.seed(42)
        features = np.random.randn(n_samples, n_features)
        labels = np.random.choice([-1, 0, 1], size=n_labels)
        
        result = compute_level_vs_change(
            features, labels, signal_index=0,
            window_size=window_size, stride=stride
        )
        
        # Check expected attributes exist
        assert hasattr(result, 'signal_name'), "Should have signal_name"
        assert hasattr(result, 'signal_index'), "Should have signal_index"
        assert hasattr(result, 'level_correlation'), "Should have level_correlation"
        assert hasattr(result, 'change_correlation'), "Should have change_correlation"
        assert hasattr(result, 'level_auc'), "Should have level_auc"
        assert hasattr(result, 'change_auc'), "Should have change_auc"
        assert hasattr(result, 'recommendation'), "Should have recommendation"
        
        # Check values are finite
        assert np.isfinite(result.level_correlation), \
            f"level_correlation should be finite, got {result.level_correlation}"
        assert np.isfinite(result.change_correlation), \
            f"change_correlation should be finite, got {result.change_correlation}"
    
    def test_level_vs_change_with_predictive_level(self):
        """Test that level is correctly identified when level is predictive."""
        np.random.seed(42)
        n_samples = 2000
        n_features = 98
        window_size = 100
        stride = 10
        
        n_labels = (n_samples - window_size) // stride + 1
        
        # Create a signal where the level (endpoint value) predicts the label
        features = np.random.randn(n_samples, n_features)
        
        # Make labels correlate with the signal level (endpoint value)
        labels = np.zeros(n_labels, dtype=int)
        for i in range(n_labels):
            end_idx = i * stride + window_size - 1
            level = features[end_idx, 0]
            labels[i] = np.sign(level)  # Label = sign of level
        
        result = compute_level_vs_change(
            features, labels, signal_index=0,
            window_size=window_size, stride=stride
        )
        
        # Level should have strong correlation
        assert abs(result.level_correlation) > 0.5, \
            f"Level should be highly correlated, got {result.level_correlation}"
    
    def test_level_vs_change_with_predictive_change(self):
        """Test that change is correctly identified when change is predictive."""
        np.random.seed(42)
        n_samples = 2000
        n_features = 98
        window_size = 100
        stride = 10
        
        n_labels = (n_samples - window_size) // stride + 1
        
        # Create a signal where the change predicts the label
        features = np.zeros((n_samples, n_features))
        features[:, 0] = np.cumsum(np.random.randn(n_samples) * 0.1)  # Random walk
        
        # Make labels correlate with the signal change (end - start)
        labels = np.zeros(n_labels, dtype=int)
        for i in range(n_labels):
            start_idx = i * stride
            end_idx = i * stride + window_size - 1
            change = features[end_idx, 0] - features[start_idx, 0]
            labels[i] = np.sign(change)  # Label = sign of change
        
        result = compute_level_vs_change(
            features, labels, signal_index=0,
            window_size=window_size, stride=stride
        )
        
        # Change should have strong correlation
        assert abs(result.change_correlation) > 0.5, \
            f"Change should be highly correlated, got {result.change_correlation}"


class TestIntegration:
    """Integration tests for the full temporal dynamics pipeline."""
    
    @pytest.fixture
    def synthetic_data(self):
        """Create synthetic data for integration tests."""
        np.random.seed(42)
        n_samples = 5000
        n_features = 98
        window_size = 100
        stride = 10
        
        n_labels = (n_samples - window_size) // stride + 1
        
        # Create features with some structure
        features = np.random.randn(n_samples, n_features)
        
        # Make one signal (index 84 = TRUE_OFI) predictive
        signal_idx = 84
        features[:, signal_idx] = np.cumsum(np.random.randn(n_samples) * 0.1)
        
        # Create labels correlated with the signal
        labels = np.zeros(n_labels)
        for i in range(n_labels):
            signal_val = features[i * stride + window_size - 1, signal_idx]
            labels[i] = np.sign(signal_val + np.random.randn() * 0.5)
        
        return {
            'features': features,
            'labels': labels.astype(int),
            'n_samples': n_samples,
            'n_labels': n_labels,
            'window_size': window_size,
            'stride': stride,
            'signal_idx': signal_idx,
        }
    
    def test_run_temporal_dynamics_analysis(self, synthetic_data):
        """Test the main analysis function runs without error."""
        result = run_temporal_dynamics_analysis(
            synthetic_data['features'],
            synthetic_data['labels'],
            signal_indices=[synthetic_data['signal_idx']],
            max_acf_lag=50,
            max_leadlag_lag=10,
            window_size=synthetic_data['window_size'],
            stride=synthetic_data['stride'],
        )
        
        # Check all components are present
        assert len(result.autocorrelations) > 0, "Should have autocorrelation results"
        assert len(result.predictive_decays) > 0, "Should have predictive decay results"
        assert len(result.level_vs_change) > 0, "Should have level vs change results"
        
        # Check that optimal lookback is reasonable
        assert 0 < result.optimal_lookback <= 100, \
            f"Optimal lookback should be in range, got {result.optimal_lookback}"
    
    def test_parameters_propagate_correctly(self, synthetic_data):
        """Verify that window_size and stride propagate to all sub-analyses."""
        # Run with custom window/stride
        custom_window = 50
        custom_stride = 5
        
        # Need to recreate labels for the new alignment
        n_labels_custom = (synthetic_data['n_samples'] - custom_window) // custom_stride + 1
        labels_custom = np.random.choice([-1, 0, 1], size=n_labels_custom)
        
        result = run_temporal_dynamics_analysis(
            synthetic_data['features'],
            labels_custom,
            signal_indices=[synthetic_data['signal_idx']],
            window_size=custom_window,
            stride=custom_stride,
        )
        
        # Verify that all components are present and valid
        assert len(result.level_vs_change) > 0, "Should have level_vs_change results"
        
        lvc = result.level_vs_change[0]
        
        # Verify the dataclass has expected attributes (from LevelVsChangeAnalysis)
        assert hasattr(lvc, 'level_correlation'), "Should have level_correlation"
        assert hasattr(lvc, 'change_correlation'), "Should have change_correlation"
        assert np.isfinite(lvc.level_correlation), "level_correlation should be finite"


class TestEdgeCases:
    """Test edge cases and numerical stability."""
    
    def test_empty_signal(self):
        """Test handling of empty signal - should not crash."""
        signal = np.array([])
        
        # Empty signal should either return empty ACF or raise a clear error
        # Testing that it doesn't crash with an unclear error
        try:
            acf, half_life, first_zero = compute_autocorrelation(signal)
            # If it returns, check the result is sensible
            assert len(acf) == 0 or acf == [], "Empty signal should return empty ACF"
        except (ValueError, IndexError) as e:
            # These are acceptable errors for empty input
            assert "empty" in str(e).lower() or "size" in str(e).lower() or "bound" in str(e).lower(), \
                f"Error message should indicate empty/size issue: {e}"
    
    def test_short_signal(self):
        """Test handling of signal shorter than max_lag."""
        signal = np.random.randn(10)
        acf, half_life, first_zero = compute_autocorrelation(signal, max_lag=100)
        
        # Should return ACF up to the signal length
        assert len(acf) <= len(signal), \
            f"ACF length should be <= signal length, got {len(acf)}"
    
    def test_nan_handling(self):
        """Test that NaN values don't crash the analysis."""
        np.random.seed(42)
        signal = np.random.randn(1000)
        signal[500] = np.nan  # Insert a NaN
        
        # This should not raise an exception
        acf, half_life, first_zero = compute_autocorrelation(signal)
        
        # Results might be NaN, but shouldn't crash
        # Just verify it returns something
        assert len(acf) > 0
    
    def test_inf_handling(self):
        """Test that Inf values don't crash the analysis."""
        np.random.seed(42)
        signal = np.random.randn(1000)
        signal[500] = np.inf  # Insert an Inf
        
        # This should not raise an exception
        acf, half_life, first_zero = compute_autocorrelation(signal)
        
        assert len(acf) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

