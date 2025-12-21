#!/usr/bin/env python3
"""
Comprehensive tests for signal_correlations.py.

Tests validate:
1. Correlation matrix computation
2. Redundant pair detection
3. PCA analysis
4. VIF computation
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lobtrainer.analysis.signal_correlations import (
    compute_signal_correlation_matrix,
    find_redundant_pairs,
    compute_pca_analysis,
    compute_vif,
    PCAResult,
    VIFResult,
)


class TestCorrelationMatrix:
    """Tests for signal correlation matrix."""
    
    def test_diagonal_is_one(self):
        """Diagonal of correlation matrix should be 1.0."""
        np.random.seed(42)
        features = np.random.randn(1000, 98)
        signal_indices = [84, 85, 86, 87]
        
        corr_matrix, names = compute_signal_correlation_matrix(features, signal_indices)
        
        for i in range(len(signal_indices)):
            assert abs(corr_matrix[i, i] - 1.0) < 1e-10, \
                f"Diagonal element [{i},{i}] should be 1.0, got {corr_matrix[i, i]}"
    
    def test_symmetry(self):
        """Correlation matrix should be symmetric."""
        np.random.seed(42)
        features = np.random.randn(1000, 98)
        signal_indices = [84, 85, 86]
        
        corr_matrix, names = compute_signal_correlation_matrix(features, signal_indices)
        
        np.testing.assert_allclose(corr_matrix, corr_matrix.T, atol=1e-10)
    
    def test_returns_signal_names(self):
        """Should return signal names."""
        np.random.seed(42)
        features = np.random.randn(1000, 98)
        signal_indices = [84, 85]
        
        corr_matrix, names = compute_signal_correlation_matrix(features, signal_indices)
        
        assert len(names) == 2
        assert all(isinstance(n, str) for n in names)
    
    def test_detects_high_correlation(self):
        """Should detect highly correlated signals."""
        np.random.seed(42)
        n = 1000
        
        base = np.random.randn(n)
        features = np.zeros((n, 98))
        features[:, 84] = base
        features[:, 85] = base * 1.2 + np.random.randn(n) * 0.1  # Highly correlated
        features[:, 86] = np.random.randn(n)  # Independent
        
        corr_matrix, names = compute_signal_correlation_matrix(features, [84, 85, 86])
        
        # Signals 84 and 85 should be highly correlated
        assert abs(corr_matrix[0, 1]) > 0.9, \
            f"Expected high correlation, got {corr_matrix[0, 1]}"
        
        # Signal 86 should be independent
        assert abs(corr_matrix[0, 2]) < 0.2


class TestRedundantPairs:
    """Tests for redundant pair detection."""
    
    def test_finds_redundant_pairs(self):
        """Should find pairs with |r| > threshold."""
        np.random.seed(42)
        n = 1000
        
        # Create correlated signals
        base = np.random.randn(n)
        features = np.zeros((n, 98))
        features[:, 84] = base
        features[:, 85] = base + np.random.randn(n) * 0.1  # r > 0.9
        features[:, 86] = np.random.randn(n)  # Independent
        
        # First compute correlation matrix
        corr_matrix, signal_names = compute_signal_correlation_matrix(features, [84, 85, 86])
        
        # Then find redundant pairs
        pairs = find_redundant_pairs(corr_matrix, signal_names, threshold=0.5)
        
        # Should find at least one redundant pair
        assert len(pairs) >= 1, "Should find redundant pair between 84 and 85"
    
    def test_no_redundant_pairs_when_independent(self):
        """Should not find pairs when all signals are independent."""
        np.random.seed(42)
        features = np.random.randn(1000, 98)  # All independent
        
        corr_matrix, signal_names = compute_signal_correlation_matrix(features, [84, 85, 86])
        pairs = find_redundant_pairs(corr_matrix, signal_names, threshold=0.9)
        
        # Should find no highly correlated pairs
        assert len(pairs) == 0, f"Expected no pairs, found {len(pairs)}"
    
    def test_returns_correlation_value(self):
        """Pairs should include correlation value as dict."""
        np.random.seed(42)
        n = 1000
        
        base = np.random.randn(n)
        features = np.zeros((n, 98))
        features[:, 84] = base
        features[:, 85] = base  # Perfect correlation
        
        corr_matrix, signal_names = compute_signal_correlation_matrix(features, [84, 85])
        pairs = find_redundant_pairs(corr_matrix, signal_names, threshold=0.5)
        
        assert len(pairs) >= 1
        # Each pair is a dict with signal_1, signal_2, correlation
        pair = pairs[0]
        assert 'signal_1' in pair
        assert 'signal_2' in pair
        assert 'correlation' in pair
        assert abs(pair['correlation']) > 0.9


class TestPCAAnalysis:
    """Tests for PCA analysis."""
    
    def test_returns_pca_result(self):
        """Should return PCAResult."""
        np.random.seed(42)
        features = np.random.randn(1000, 98)
        
        result = compute_pca_analysis(features, [84, 85, 86, 87])
        
        assert isinstance(result, PCAResult)
    
    def test_explained_variance_sums_to_one(self):
        """Explained variance ratios should sum to 1."""
        np.random.seed(42)
        features = np.random.randn(1000, 98)
        
        result = compute_pca_analysis(features, [84, 85, 86, 87])
        
        total = sum(result.explained_variance_ratio)
        assert abs(total - 1.0) < 0.01, f"Variance ratios sum to {total}, expected 1.0"
    
    def test_n_components_matches_signals(self):
        """Number of components should match number of signals."""
        np.random.seed(42)
        features = np.random.randn(1000, 98)
        signal_indices = [84, 85, 86]
        
        result = compute_pca_analysis(features, signal_indices)
        
        assert len(result.explained_variance_ratio) == len(signal_indices)
    
    def test_correlated_signals_fewer_components(self):
        """Correlated signals should concentrate variance in fewer components."""
        np.random.seed(42)
        n = 1000
        
        # Create correlated signals
        base = np.random.randn(n)
        features = np.zeros((n, 98))
        features[:, 84] = base
        features[:, 85] = base + np.random.randn(n) * 0.1
        features[:, 86] = base + np.random.randn(n) * 0.1
        features[:, 87] = np.random.randn(n)
        
        result = compute_pca_analysis(features, [84, 85, 86, 87])
        
        # First component should explain most variance (due to correlation)
        assert result.explained_variance_ratio[0] > 0.5, \
            f"First PC should explain >50% variance, got {result.explained_variance_ratio[0]}"


class TestVIF:
    """Tests for Variance Inflation Factor computation."""
    
    def test_returns_vif_results(self):
        """Should return VIFResult for each signal."""
        np.random.seed(42)
        features = np.random.randn(1000, 98)
        signal_indices = [84, 85, 86]
        
        results = compute_vif(features, signal_indices)
        
        assert len(results) == 3
        assert all(isinstance(r, VIFResult) for r in results)
    
    def test_independent_signals_low_vif(self):
        """Independent signals should have VIF ≈ 1."""
        np.random.seed(42)
        features = np.random.randn(1000, 98)  # All independent
        signal_indices = [84, 85, 86]
        
        results = compute_vif(features, signal_indices)
        
        for r in results:
            assert r.vif < 2.0, f"{r.signal_name} VIF should be low for independent signals, got {r.vif}"
            assert not r.is_problematic
    
    def test_correlated_signals_high_vif(self):
        """Highly correlated signals should have high VIF."""
        np.random.seed(42)
        n = 1000
        
        # Create multicollinear signals
        base = np.random.randn(n)
        features = np.zeros((n, 98))
        features[:, 84] = base
        features[:, 85] = base + np.random.randn(n) * 0.01  # Nearly identical
        features[:, 86] = np.random.randn(n)  # Independent
        
        results = compute_vif(features, [84, 85, 86])
        
        # Signals 84 and 85 should have high VIF
        vif_84 = next(r for r in results if r.signal_index == 84)
        vif_85 = next(r for r in results if r.signal_index == 85)
        
        assert vif_84.vif > 5, f"VIF for signal 84 should be high, got {vif_84.vif}"
        assert vif_85.vif > 5, f"VIF for signal 85 should be high, got {vif_85.vif}"


class TestEdgeCases:
    """Edge case tests."""
    
    def test_single_signal_pca(self):
        """PCA with single signal should work."""
        np.random.seed(42)
        features = np.random.randn(1000, 98)
        
        result = compute_pca_analysis(features, [84])
        
        # Single signal should explain 100% variance
        assert len(result.explained_variance_ratio) == 1
        assert abs(result.explained_variance_ratio[0] - 1.0) < 0.01
    
    def test_short_feature_array(self):
        """Should handle short feature arrays."""
        np.random.seed(42)
        features = np.random.randn(50, 98)  # Short
        
        # Should not crash
        corr_matrix, names = compute_signal_correlation_matrix(features, [84, 85])
        assert corr_matrix.shape == (2, 2)
    
    def test_constant_signal(self):
        """Should handle constant signal gracefully."""
        np.random.seed(42)
        features = np.random.randn(1000, 98)
        features[:, 84] = 5.0  # Constant
        
        # May have warnings but shouldn't crash
        try:
            corr_matrix, names = compute_signal_correlation_matrix(features, [84, 85])
            # NaN correlation is acceptable for constant signal
        except (ValueError, RuntimeWarning):
            pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

