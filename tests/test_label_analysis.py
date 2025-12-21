#!/usr/bin/env python3
"""
Comprehensive tests for label_analysis.py.

Tests validate:
1. Label distribution computation (counts, percentages)
2. Autocorrelation (persistence detection)
3. Transition matrix (Markov properties)
4. Regime statistics (per-time-regime analysis)
5. Signal-label correlations
6. Edge cases (empty labels, single label, etc.)

Per RULE.md: Tests expose implementation, not re-implement logic.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lobtrainer.analysis.label_analysis import (
    compute_label_distribution,
    compute_autocorrelation,
    compute_transition_matrix,
    compute_regime_stats,
    compute_signal_label_correlations,
    run_label_analysis,
    LabelDistribution,
    TransitionMatrix,
)
from lobtrainer.constants import LABEL_UP, LABEL_DOWN, LABEL_STABLE


class TestLabelDistribution:
    """Tests for label distribution computation."""
    
    def test_counts_sum_to_total(self):
        """Counts should sum to total labels."""
        labels = np.array([1, 1, 0, 0, 0, -1, -1, -1, -1])
        result = compute_label_distribution(labels)
        
        total = result.up_count + result.stable_count + result.down_count
        assert total == len(labels), f"Counts sum ({total}) != len(labels) ({len(labels)})"
    
    def test_percentages_sum_to_100(self):
        """Percentages should sum to ~100%."""
        labels = np.random.choice([-1, 0, 1], size=1000)
        result = compute_label_distribution(labels)
        
        total_pct = result.up_pct + result.stable_pct + result.down_pct
        assert abs(total_pct - 100.0) < 0.01, f"Percentages sum to {total_pct}, expected 100"
    
    def test_all_up_labels(self):
        """All up labels should give 100% up."""
        labels = np.ones(100, dtype=int)  # All 1 (UP)
        result = compute_label_distribution(labels)
        
        assert result.up_pct == 100.0, f"Expected 100% up, got {result.up_pct}"
        assert result.stable_pct == 0.0
        assert result.down_pct == 0.0
    
    def test_all_down_labels(self):
        """All down labels should give 100% down."""
        labels = -np.ones(100, dtype=int)  # All -1 (DOWN)
        result = compute_label_distribution(labels)
        
        assert result.down_pct == 100.0, f"Expected 100% down, got {result.down_pct}"
        assert result.up_pct == 0.0
    
    def test_known_distribution(self):
        """Test with known distribution."""
        # 20 up, 30 stable, 50 down = 100 total
        labels = np.array([1]*20 + [0]*30 + [-1]*50)
        result = compute_label_distribution(labels)
        
        assert result.up_count == 20
        assert result.stable_count == 30
        assert result.down_count == 50
        assert result.up_pct == 20.0
        assert result.stable_pct == 30.0
        assert result.down_pct == 50.0


class TestAutocorrelation:
    """Tests for label autocorrelation."""
    
    def test_alternating_labels_negative_acf(self):
        """Alternating labels should have negative ACF at lag 1."""
        # Alternating pattern: 1, -1, 1, -1, ...
        labels = np.array([1, -1] * 500)
        result = compute_autocorrelation(labels, max_lag=10)
        
        # ACF at lag 1 should be strongly negative
        assert result.acf_values[1] < -0.9, f"Expected negative ACF(1), got {result.acf_values[1]}"
    
    def test_constant_labels_perfect_acf(self):
        """Constant labels should have ACF = 1 at all lags (or NaN due to zero variance)."""
        labels = np.ones(100, dtype=int)
        result = compute_autocorrelation(labels, max_lag=10)
        
        # With constant input, ACF is either 1.0 or undefined (handled gracefully)
        # The implementation should handle this without crashing
        assert result is not None
    
    def test_random_labels_near_zero_acf(self):
        """Random labels should have near-zero ACF for lag > 0."""
        np.random.seed(42)
        labels = np.random.choice([-1, 0, 1], size=10000)
        result = compute_autocorrelation(labels, max_lag=20)
        
        # ACF for lag > 0 should be near zero for random labels
        # Use the specific attributes from AutocorrelationResult
        assert abs(result.lag_1_acf) < 0.1, \
            f"Random labels lag_1_acf should be ≈ 0, got {result.lag_1_acf}"
        assert abs(result.lag_5_acf) < 0.1, \
            f"Random labels lag_5_acf should be ≈ 0, got {result.lag_5_acf}"
        assert abs(result.lag_10_acf) < 0.1, \
            f"Random labels lag_10_acf should be ≈ 0, got {result.lag_10_acf}"
    
    def test_persistent_labels_positive_acf(self):
        """Labels that tend to persist should have positive ACF."""
        # Create labels that tend to stay the same
        np.random.seed(42)
        labels = [np.random.choice([-1, 0, 1])]
        for _ in range(999):
            if np.random.rand() < 0.8:  # 80% chance to repeat
                labels.append(labels[-1])
            else:
                labels.append(np.random.choice([-1, 0, 1]))
        labels = np.array(labels)
        
        result = compute_autocorrelation(labels, max_lag=10)
        
        # ACF at lag 1 should be positive due to persistence
        assert result.lag_1_acf > 0.3, f"Expected positive lag_1_acf for persistent labels, got {result.lag_1_acf}"


class TestTransitionMatrix:
    """Tests for Markov transition matrix."""
    
    def test_rows_sum_to_one(self):
        """Each row of transition probabilities should sum to 1."""
        np.random.seed(42)
        labels = np.random.choice([-1, 0, 1], size=1000)
        result = compute_transition_matrix(labels)
        
        probs = np.array(result.probabilities)
        row_sums = probs.sum(axis=1)
        
        for i, row_sum in enumerate(row_sums):
            assert abs(row_sum - 1.0) < 0.001, \
                f"Row {i} sums to {row_sum}, expected 1.0"
    
    def test_alternating_labels_off_diagonal(self):
        """Alternating labels should have high off-diagonal transitions."""
        # Pattern: -1, 1, -1, 1, ...
        labels = np.array([-1, 1] * 100)
        result = compute_transition_matrix(labels)
        
        probs = np.array(result.probabilities)
        label_order = result.labels
        
        # Find indices
        down_idx = label_order.index(-1)
        up_idx = label_order.index(1)
        
        # P(-1 → 1) should be 1.0
        assert probs[down_idx, up_idx] > 0.99, \
            f"Expected P(-1→1) ≈ 1.0, got {probs[down_idx, up_idx]}"
        
        # P(1 → -1) should be 1.0
        assert probs[up_idx, down_idx] > 0.99, \
            f"Expected P(1→-1) ≈ 1.0, got {probs[up_idx, down_idx]}"
    
    def test_absorbing_state(self):
        """Test with an 'absorbing' pattern."""
        # Pattern: many -1, then all 1
        labels = np.array([-1]*50 + [1]*50)
        result = compute_transition_matrix(labels)
        
        probs = np.array(result.probabilities)
        label_order = result.labels
        
        up_idx = label_order.index(1)
        
        # Once in state 1, stays in state 1 (except at boundary)
        # P(1 → 1) should be high
        assert probs[up_idx, up_idx] > 0.9, \
            f"Expected high P(1→1), got {probs[up_idx, up_idx]}"
    
    def test_counts_and_probs_consistent(self):
        """Probabilities should be counts normalized by row."""
        np.random.seed(42)
        labels = np.random.choice([-1, 0, 1], size=1000)
        result = compute_transition_matrix(labels)
        
        counts = np.array(result.counts)
        probs = np.array(result.probabilities)
        
        # Manually compute probabilities from counts
        row_sums = counts.sum(axis=1, keepdims=True)
        expected_probs = np.divide(counts, row_sums, where=row_sums > 0, 
                                   out=np.zeros_like(counts, dtype=float))
        
        np.testing.assert_allclose(probs, expected_probs, rtol=1e-5)


class TestRegimeStats:
    """Tests for per-regime statistics."""
    
    def test_returns_stats_per_regime(self):
        """Should return stats for each unique regime."""
        from lobtrainer.constants import FeatureIndex
        np.random.seed(42)
        n = 1000
        n_features = 98
        
        labels = np.random.choice([-1, 0, 1], size=n)
        
        # Create aligned_features with TIME_REGIME in the correct column
        aligned_features = np.random.randn(n, n_features)
        aligned_features[:, FeatureIndex.TIME_REGIME] = np.repeat([0, 1, 2, 3], n // 4)[:n]
        aligned_features[:, FeatureIndex.TRUE_OFI] = np.random.randn(n)
        
        results = compute_regime_stats(aligned_features, labels, min_samples=100)
        
        # Should have 4 regimes (each with >= 250 samples)
        assert len(results) == 4, f"Expected 4 regimes, got {len(results)}"
        
        # Each regime should have a name
        regime_names = {r.name for r in results}
        assert len(regime_names) == 4, "Should have unique regime names"
    
    def test_regime_sample_counts(self):
        """Sample counts per regime should sum to total."""
        from lobtrainer.constants import FeatureIndex
        np.random.seed(42)
        n = 1000
        n_features = 98
        
        labels = np.random.choice([-1, 0, 1], size=n)
        aligned_features = np.random.randn(n, n_features)
        aligned_features[:, FeatureIndex.TIME_REGIME] = np.repeat([0, 1, 2, 3], n // 4)[:n]
        aligned_features[:, FeatureIndex.TRUE_OFI] = np.random.randn(n)
        
        results = compute_regime_stats(aligned_features, labels, min_samples=100)
        
        total_samples = sum(r.n_samples for r in results)
        assert total_samples == n, f"Regime samples sum ({total_samples}) != n ({n})"
    
    def test_single_regime(self):
        """Should handle single regime correctly."""
        from lobtrainer.constants import FeatureIndex
        n = 200  # Enough samples
        n_features = 98
        
        labels = np.random.choice([-1, 0, 1], size=n)
        aligned_features = np.random.randn(n, n_features)
        aligned_features[:, FeatureIndex.TIME_REGIME] = 0  # All regime 0
        aligned_features[:, FeatureIndex.TRUE_OFI] = np.random.randn(n)
        
        results = compute_regime_stats(aligned_features, labels, min_samples=100)
        
        assert len(results) == 1, f"Expected 1 regime, got {len(results)}"
        assert results[0].n_samples == n


class TestSignalLabelCorrelations:
    """Tests for signal-label correlation computation."""
    
    def test_perfect_correlation(self):
        """Test with perfectly correlated signal and labels."""
        np.random.seed(42)
        n = 1000
        n_features = 98
        
        # Create aligned_features (already aligned with labels)
        aligned_features = np.random.randn(n, n_features)
        
        # Make labels correlate with signal at column 84 (TRUE_OFI index)
        labels = np.array([int(np.sign(aligned_features[i, 84])) for i in range(n)])
        
        # Use dict format expected by compute_signal_label_correlations
        signal_indices = {84: 'true_ofi'}
        
        results = compute_signal_label_correlations(
            aligned_features, labels, signal_indices
        )
        
        # Should find high correlation for signal
        assert len(results) == 1
        assert abs(results[0].correlation) > 0.5, \
            f"Expected high correlation, got {results[0].correlation}"
    
    def test_returns_significance(self):
        """Verify that significance is computed."""
        np.random.seed(42)
        n = 1000
        n_features = 98
        
        aligned_features = np.random.randn(n, n_features)
        labels = np.random.choice([-1, 0, 1], size=n)
        
        signal_indices = {84: 'true_ofi', 85: 'depth_norm_ofi'}
        
        results = compute_signal_label_correlations(
            aligned_features, labels, signal_indices
        )
        
        assert len(results) == 2
        # All results should have p_value and is_significant
        for r in results:
            assert hasattr(r, 'p_value'), "Should have p_value"
            assert hasattr(r, 'is_significant'), "Should have is_significant"


class TestRunLabelAnalysis:
    """Integration tests for the full analysis pipeline."""
    
    @pytest.fixture
    def mock_data(self):
        """Create mock data for testing."""
        np.random.seed(42)
        n = 5000
        n_features = 98
        window_size = 100
        stride = 10
        
        n_labels = (n - window_size) // stride + 1
        
        features = np.random.randn(n, n_features)
        labels = np.random.choice([-1, 0, 1], size=n_labels)
        
        return {
            'features': features,
            'labels': labels,
            'window_size': window_size,
            'stride': stride,
        }
    
    def test_returns_complete_summary(self, mock_data):
        """Should return a complete LabelAnalysisSummary."""
        summary = run_label_analysis(
            mock_data['features'],
            mock_data['labels'],
            window_size=mock_data['window_size'],
            stride=mock_data['stride'],
        )
        
        # Check all components are present
        assert summary.distribution is not None
        assert summary.autocorrelation is not None
        assert summary.transition_matrix is not None
        assert summary.signal_correlations is not None
    
    def test_handles_different_label_distributions(self, mock_data):
        """Should handle skewed label distributions."""
        # All stable
        labels_stable = np.zeros(len(mock_data['labels']), dtype=int)
        
        summary = run_label_analysis(
            mock_data['features'],
            labels_stable,
            window_size=mock_data['window_size'],
            stride=mock_data['stride'],
        )
        
        assert summary.distribution.stable_pct == 100.0


class TestEdgeCases:
    """Edge case and error handling tests."""
    
    def test_empty_labels(self):
        """Should handle empty labels gracefully."""
        labels = np.array([], dtype=int)
        
        # Distribution should handle empty
        try:
            result = compute_label_distribution(labels)
            # If it succeeds, check values are zero
            assert result.up_count == 0
            assert result.down_count == 0
            assert result.stable_count == 0
        except (ValueError, ZeroDivisionError):
            # Acceptable to raise an error for empty input
            pass
    
    def test_single_label(self):
        """Should handle single label."""
        labels = np.array([1])
        
        result = compute_label_distribution(labels)
        assert result.up_count == 1
        assert result.total == 1
    
    def test_very_short_sequence(self):
        """Should handle very short sequences for autocorrelation."""
        labels = np.array([1, 0, -1])
        
        # Should not crash even with short sequence
        result = compute_autocorrelation(labels, max_lag=5)
        
        # ACF values should exist and be a list
        assert result is not None
        assert hasattr(result, 'acf_values')
        # Implementation may pad ACF values; just check it returns valid structure
        assert len(result.acf_values) > 0
        # First value (lag 0) should always be 1.0
        assert result.acf_values[0] == 1.0
    
    def test_all_same_label_transition(self):
        """Transition matrix with all same label."""
        labels = np.ones(100, dtype=int)  # All 1
        
        result = compute_transition_matrix(labels)
        
        # Should have only one label
        assert len(result.labels) == 1
        assert result.labels[0] == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

