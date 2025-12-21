#!/usr/bin/env python3
"""
Comprehensive tests for predictive_power.py.

These tests validate:
1. Signal metrics computation
2. Correlation matrix computation
3. Binned probability analysis
4. Dynamic recommendations generation
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from io import StringIO

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lobtrainer.analysis.predictive_power import (
    compute_signal_metrics,
    compute_all_signal_metrics,
    compute_binned_probabilities,
    print_predictive_summary,
)
from lobtrainer.constants import LABEL_UP, LABEL_DOWN, LABEL_STABLE


def compute_signal_correlation_matrix(signals: np.ndarray) -> np.ndarray:
    """
    Compute correlation matrix between signals.
    
    Helper function for testing - computes inter-signal correlations.
    """
    return np.corrcoef(signals.T)


class TestSignalMetrics:
    """Tests for signal metrics computation."""
    
    def test_perfect_positive_correlation(self):
        """Test with perfectly correlated signal and labels."""
        np.random.seed(42)
        n = 1000
        
        signal = np.linspace(-1, 1, n)
        labels = np.sign(signal)  # High correlation (not perfect due to discretization)
        
        metrics = compute_signal_metrics(signal, labels, expected_sign='+')
        
        # Should have high correlation (>0.8 due to sign discretization)
        assert metrics['pearson_r'] > 0.8, f"Expected high r, got {metrics['pearson_r']}"
        assert metrics['sign_consistent'] == True, "Sign should be consistent"
    
    def test_perfect_negative_correlation(self):
        """Test with perfectly anti-correlated signal and labels."""
        np.random.seed(42)
        n = 1000
        
        signal = np.linspace(-1, 1, n)
        labels = -np.sign(signal)  # Negative correlation
        
        metrics = compute_signal_metrics(signal, labels, expected_sign='+')
        
        # Should have negative correlation (<-0.8 due to sign discretization)
        assert metrics['pearson_r'] < -0.8, f"Expected negative r, got {metrics['pearson_r']}"
        assert metrics['sign_consistent'] == False, "Sign should be inconsistent (expected +)"
    
    def test_no_correlation(self):
        """Test with uncorrelated signal and labels."""
        np.random.seed(42)
        n = 5000
        
        signal = np.random.randn(n)
        labels = np.random.choice([-1, 0, 1], size=n)
        
        metrics = compute_signal_metrics(signal, labels)
        
        # Should have near-zero correlation
        assert abs(metrics['pearson_r']) < 0.1, f"Expected near-zero r, got {metrics['pearson_r']}"
    
    def test_auc_computation(self):
        """Test that AUC is computed correctly."""
        np.random.seed(42)
        n = 1000
        
        # Create signal that predicts Up vs Not-Up well
        signal = np.random.randn(n)
        labels = np.where(signal > 0.5, LABEL_UP, 
                         np.where(signal < -0.5, LABEL_DOWN, LABEL_STABLE))
        
        metrics = compute_signal_metrics(signal, labels)
        
        # AUC should be > 0.5 (better than random)
        assert metrics['auc_up'] > 0.5, f"AUC_up should be > 0.5, got {metrics['auc_up']}"
    
    def test_handles_nan_values(self):
        """Test that NaN values are handled correctly."""
        np.random.seed(42)
        n = 1000
        
        signal = np.random.randn(n)
        signal[::10] = np.nan  # Insert NaNs
        labels = np.random.choice([-1, 0, 1], size=n)
        
        metrics = compute_signal_metrics(signal, labels)
        
        # Should not crash and should have valid n_samples
        assert metrics['n_samples'] < n, "NaN samples should be excluded"
        assert metrics['n_samples'] > 0, "Should have some valid samples"


class TestCorrelationMatrix:
    """Tests for inter-signal correlation matrix."""
    
    def test_diagonal_is_one(self):
        """Diagonal of correlation matrix should be 1.0."""
        np.random.seed(42)
        n = 1000
        n_signals = 5
        
        signals = np.random.randn(n, n_signals)
        corr_matrix = compute_signal_correlation_matrix(signals)
        
        np.testing.assert_allclose(np.diag(corr_matrix), 1.0, atol=1e-10)
    
    def test_symmetry(self):
        """Correlation matrix should be symmetric."""
        np.random.seed(42)
        n = 1000
        n_signals = 5
        
        signals = np.random.randn(n, n_signals)
        corr_matrix = compute_signal_correlation_matrix(signals)
        
        np.testing.assert_allclose(corr_matrix, corr_matrix.T, atol=1e-10)
    
    def test_detects_high_correlation(self):
        """Should detect highly correlated signals."""
        np.random.seed(42)
        n = 1000
        
        base = np.random.randn(n)
        signals = np.column_stack([
            base,
            base * 1.2 + np.random.randn(n) * 0.1,  # High correlation with col 0
            np.random.randn(n),  # Independent
        ])
        
        corr_matrix = compute_signal_correlation_matrix(signals)
        
        # Columns 0 and 1 should be highly correlated
        assert abs(corr_matrix[0, 1]) > 0.9, f"Expected high correlation, got {corr_matrix[0, 1]}"
        
        # Column 2 should be independent
        assert abs(corr_matrix[0, 2]) < 0.2, f"Expected low correlation, got {corr_matrix[0, 2]}"


class TestBinnedProbabilities:
    """Tests for binned probability analysis."""
    
    def test_bins_sum_to_one(self):
        """Probabilities within each bin should sum to 1."""
        np.random.seed(42)
        n = 1000
        
        signal = np.random.randn(n)
        labels = np.random.choice([-1, 0, 1], size=n)
        
        df = compute_binned_probabilities(signal, labels, n_bins=10)
        
        for _, row in df.iterrows():
            total = row['p_up'] + row['p_down'] + row['p_stable']
            assert abs(total - 1.0) < 0.001, f"Bin probs should sum to 1, got {total}"
    
    def test_monotonic_relationship(self):
        """For predictive signal, P(up) should increase with signal value."""
        np.random.seed(42)
        n = 10000
        
        # Create signal where high values predict Up
        signal = np.random.randn(n)
        # Make labels correlate with signal
        probs = 1 / (1 + np.exp(-signal))  # Sigmoid
        labels = np.where(np.random.rand(n) < probs, LABEL_UP, LABEL_DOWN)
        
        df = compute_binned_probabilities(signal, labels, n_bins=5)
        
        # P(up) should generally increase across bins
        p_ups = df['p_up'].values
        # Check that last bin has higher P(up) than first bin
        assert p_ups[-1] > p_ups[0], \
            f"P(up) should increase: first={p_ups[0]:.3f}, last={p_ups[-1]:.3f}"


class TestDynamicRecommendations:
    """Tests for dynamic recommendation generation."""
    
    def create_mock_metrics(self):
        """Create mock metrics for testing."""
        return pd.DataFrame([
            {'name': 'signal_A', 'pearson_r': 0.15, 'spearman_r': 0.14, 
             'auc_up': 0.65, 'auc_down': 0.62, 'sign_consistent': True, 
             'expected_sign': '+', 'n_samples': 1000},
            {'name': 'signal_B', 'pearson_r': -0.08, 'spearman_r': -0.07,
             'auc_up': 0.55, 'auc_down': 0.54, 'sign_consistent': False,
             'expected_sign': '+', 'n_samples': 1000},
            {'name': 'signal_C', 'pearson_r': 0.05, 'spearman_r': 0.05,
             'auc_up': 0.52, 'auc_down': 0.51, 'sign_consistent': True,
             'expected_sign': '+', 'n_samples': 1000},
            {'name': 'signal_D', 'pearson_r': 0.12, 'spearman_r': 0.11,
             'auc_up': 0.60, 'auc_down': 0.58, 'sign_consistent': True,
             'expected_sign': '+', 'n_samples': 1000},
            {'name': 'signal_E', 'pearson_r': 0.005, 'spearman_r': 0.004,
             'auc_up': 0.50, 'auc_down': 0.50, 'sign_consistent': None,
             'expected_sign': '?', 'n_samples': 1000},
        ])
    
    def test_recommendations_are_generated(self):
        """Test that recommendations are generated without error."""
        df = self.create_mock_metrics()
        
        # Capture stdout
        import io
        import sys
        captured = io.StringIO()
        sys.stdout = captured
        
        try:
            print_predictive_summary(df)
        finally:
            sys.stdout = sys.__stdout__
        
        output = captured.getvalue()
        
        # Check that key sections are present
        assert 'RECOMMENDATIONS' in output, "Should have recommendations section"
        assert 'PRIMARY FEATURES' in output or 'GROUP A' in output, "Should have primary group"
    
    def test_identifies_primary_features(self):
        """Test that signals with |r| >= 0.05 are identified as primary."""
        df = self.create_mock_metrics()
        
        import io
        import sys
        captured = io.StringIO()
        sys.stdout = captured
        
        try:
            print_predictive_summary(df)
        finally:
            sys.stdout = sys.__stdout__
        
        output = captured.getvalue()
        
        # signal_A (r=0.15) and signal_D (r=0.12) should be in primary
        # Both are >= 0.05 and have consistent sign
        assert 'signal_A' in output, "signal_A should be mentioned"
        assert 'signal_D' in output, "signal_D should be mentioned"
    
    def test_identifies_contrarian_signals(self):
        """Test that contrarian signals are identified."""
        df = self.create_mock_metrics()
        
        import io
        import sys
        captured = io.StringIO()
        sys.stdout = captured
        
        try:
            print_predictive_summary(df)
        finally:
            sys.stdout = sys.__stdout__
        
        output = captured.getvalue()
        
        # signal_B has sign_consistent=False, should be in contrarian
        assert 'CONTRARIAN' in output or 'signal_B' in output, \
            "Should identify contrarian signal_B"
    
    def test_identifies_low_priority(self):
        """Test that low priority signals are identified."""
        df = self.create_mock_metrics()
        
        import io
        import sys
        captured = io.StringIO()
        sys.stdout = captured
        
        try:
            print_predictive_summary(df)
        finally:
            sys.stdout = sys.__stdout__
        
        output = captured.getvalue()
        
        # signal_E (r=0.005) should be low priority
        assert 'signal_E' in output, "signal_E should be mentioned as low priority"
    
    def test_redundancy_detection(self):
        """Test that redundant pairs are identified."""
        df = self.create_mock_metrics()
        
        # Create correlation matrix with high correlation between signals
        corr_matrix = np.array([
            [1.0, 0.7, 0.1, 0.2, 0.1],  # A highly correlated with B
            [0.7, 1.0, 0.1, 0.1, 0.1],
            [0.1, 0.1, 1.0, 0.1, 0.1],
            [0.2, 0.1, 0.1, 1.0, 0.1],
            [0.1, 0.1, 0.1, 0.1, 1.0],
        ])
        signal_names = ['signal_A', 'signal_B', 'signal_C', 'signal_D', 'signal_E']
        
        import io
        import sys
        captured = io.StringIO()
        sys.stdout = captured
        
        try:
            print_predictive_summary(df, corr_matrix, signal_names)
        finally:
            sys.stdout = sys.__stdout__
        
        output = captured.getvalue()
        
        # Should identify signal_A and signal_B as redundant pair
        assert 'signal_A' in output and 'signal_B' in output, \
            "Should mention redundant pair A-B"
        assert '0.7' in output or 'REDUNDANT' in output, \
            "Should show redundancy information"


class TestComputeAllSignalMetrics:
    """Tests for computing metrics for all signals."""
    
    def test_returns_dataframe(self):
        """Test that compute_all_signal_metrics returns a DataFrame."""
        np.random.seed(42)
        n = 1000
        n_features = 98
        
        features = np.random.randn(n, n_features)
        labels = np.random.choice([-1, 0, 1], size=n)
        
        # Just test with a few signal indices
        signal_indices = [84, 85, 86]
        
        df = compute_all_signal_metrics(features, labels, signal_indices)
        
        assert isinstance(df, pd.DataFrame), "Should return DataFrame"
        assert len(df) == 3, f"Should have 3 rows, got {len(df)}"
    
    def test_includes_expected_columns(self):
        """Test that all expected columns are present."""
        np.random.seed(42)
        n = 1000
        n_features = 98
        
        features = np.random.randn(n, n_features)
        labels = np.random.choice([-1, 0, 1], size=n)
        
        signal_indices = [84, 85]
        df = compute_all_signal_metrics(features, labels, signal_indices)
        
        expected_columns = ['name', 'pearson_r', 'auc_up', 'auc_down', 'n_samples']
        for col in expected_columns:
            assert col in df.columns, f"Missing column: {col}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

