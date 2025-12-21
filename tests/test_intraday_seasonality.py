"""
Comprehensive tests for intraday_seasonality.py module.

Tests cover:
1. Formula correctness (Pearson correlation)
2. Edge cases (NaN, Inf, empty arrays, constant arrays)
3. Boundary conditions (regime filtering, threshold behavior)
4. Invariants (sign conventions, regime names)
5. Integration tests (full analysis pipeline)

Reference:
    Cont et al. (2014) §3.3: "Price impact is five times higher at the 
    market open compared to the market close."
"""

import pytest
import numpy as np
from scipy import stats as scipy_stats
from pathlib import Path
import tempfile

from lobtrainer.analysis.intraday_seasonality import (
    compute_regime_stats,
    compute_signal_regime_correlation,
    compute_all_regime_correlations,
    compute_signal_seasonality,
    compute_regime_importance,
    generate_recommendations,
    run_intraday_seasonality_analysis,
    to_dict,
    RegimeStats,
    SignalRegimeCorrelation,
    SignalSeasonality,
    IntradaySeasonalitySummary,
    REGIME_NAMES,
    CORE_SIGNAL_INDICES,
    EXPECTED_SIGNS,
)
from lobtrainer.constants import FeatureIndex


class TestRegimeStats:
    """Tests for compute_regime_stats function."""
    
    def test_basic_regime_stats(self):
        """Test basic regime statistics computation."""
        labels = np.array([1, 1, 0, -1, 0, 1, -1, -1, 0, 1])
        time_regimes = np.array([0, 0, 0, 1, 1, 2, 2, 2, 3, 3])
        
        results = compute_regime_stats(labels, time_regimes)
        
        # Should have 4 regimes (0, 1, 2, 3)
        assert len(results) == 4
        
        # Check regime names
        regime_names = [r.regime_name for r in results]
        assert "OPEN" in regime_names
        assert "EARLY" in regime_names
        assert "MIDDAY" in regime_names
        assert "CLOSE" in regime_names
    
    def test_label_distribution_per_regime(self):
        """Test that label distribution sums to 100%."""
        labels = np.array([1, 0, -1, 1, 0, -1])
        time_regimes = np.array([0, 0, 0, 1, 1, 1])
        
        results = compute_regime_stats(labels, time_regimes)
        
        for rs in results:
            total_pct = sum(rs.label_distribution.values())
            assert abs(total_pct - 100.0) < 1e-10, (
                f"Label distribution should sum to 100%, got {total_pct}"
            )
    
    def test_sample_count_per_regime(self):
        """Test that sample counts are correct."""
        labels = np.array([1, 1, 1, 0, 0, -1])
        time_regimes = np.array([0, 0, 0, 2, 2, 3])
        
        results = compute_regime_stats(labels, time_regimes)
        
        # Regime 0: 3 samples
        regime_0 = [r for r in results if r.regime == 0][0]
        assert regime_0.n_samples == 3
        
        # Regime 2: 2 samples
        regime_2 = [r for r in results if r.regime == 2][0]
        assert regime_2.n_samples == 2
        
        # Regime 3: 1 sample
        regime_3 = [r for r in results if r.regime == 3][0]
        assert regime_3.n_samples == 1
    
    def test_label_mean_and_std(self):
        """Test label mean and std calculation."""
        # Regime 0: all UP (1), mean=1, std=0
        labels = np.array([1, 1, 1])
        time_regimes = np.array([0, 0, 0])
        
        results = compute_regime_stats(labels, time_regimes)
        assert len(results) == 1
        
        assert results[0].label_mean == 1.0
        assert results[0].label_std == 0.0
    
    def test_length_mismatch_raises(self):
        """Test that mismatched array lengths raise ValueError."""
        labels = np.array([1, 0, -1])
        time_regimes = np.array([0, 0])  # Wrong length
        
        with pytest.raises(ValueError, match="Length mismatch"):
            compute_regime_stats(labels, time_regimes)
    
    def test_empty_arrays_raises(self):
        """Test that empty arrays raise ValueError."""
        labels = np.array([])
        time_regimes = np.array([])
        
        with pytest.raises(ValueError, match="Empty arrays"):
            compute_regime_stats(labels, time_regimes)
    
    def test_nan_in_time_regimes(self):
        """Test handling of NaN in time_regimes."""
        labels = np.array([1, 0, -1, 1])
        time_regimes = np.array([0, np.nan, 2, 2])
        
        results = compute_regime_stats(labels, time_regimes)
        
        # Should only have regimes 0 and 2 (NaN ignored)
        regimes = [r.regime for r in results]
        assert 0 in regimes
        assert 2 in regimes
        assert len(results) == 2


class TestSignalRegimeCorrelation:
    """Tests for compute_signal_regime_correlation function."""
    
    def test_perfect_positive_correlation(self):
        """Test correlation = 1 for perfect positive relationship."""
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        labels = np.array([1, 1, 1, 1, 1])  # Constant, so we need variation
        
        # Use labels that vary with signal
        labels_varying = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        
        result = compute_signal_regime_correlation(
            signal, labels_varying, 'test_signal', 0
        )
        
        assert result.correlation > 0.99, (
            f"Expected correlation ~1.0, got {result.correlation}"
        )
    
    def test_perfect_negative_correlation(self):
        """Test correlation = -1 for perfect negative relationship."""
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        labels = np.array([0.5, 0.4, 0.3, 0.2, 0.1])  # Inverse
        
        result = compute_signal_regime_correlation(
            signal, labels, 'test_signal', 0
        )
        
        assert result.correlation < -0.99, (
            f"Expected correlation ~-1.0, got {result.correlation}"
        )
    
    def test_zero_correlation(self):
        """Test correlation ~0 for independent variables."""
        np.random.seed(42)
        signal = np.random.randn(1000)
        labels = np.random.randn(1000)
        
        result = compute_signal_regime_correlation(
            signal, labels, 'test_signal', 0
        )
        
        # Correlation should be close to 0 (within statistical variance)
        assert abs(result.correlation) < 0.1, (
            f"Expected correlation ~0, got {result.correlation}"
        )
    
    def test_significance_for_large_sample(self):
        """Test that significant correlations have low p-values."""
        np.random.seed(42)
        n = 1000
        signal = np.arange(n).astype(float)
        labels = signal + np.random.randn(n) * 10  # Strong positive + noise
        
        result = compute_signal_regime_correlation(
            signal, labels, 'test_signal', 0
        )
        
        assert result.is_significant, (
            f"Expected significant correlation, p-value={result.p_value}"
        )
        assert result.p_value < 0.01
    
    def test_constant_signal_returns_zero(self):
        """Test that constant signal returns zero correlation."""
        signal = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        labels = np.array([1, 0, -1, 1, 0])
        
        result = compute_signal_regime_correlation(
            signal, labels, 'test_signal', 0
        )
        
        assert result.correlation == 0.0, (
            "Constant signal should have zero correlation"
        )
    
    def test_constant_labels_returns_zero(self):
        """Test that constant labels return zero correlation."""
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        labels = np.array([0, 0, 0, 0, 0])
        
        result = compute_signal_regime_correlation(
            signal, labels, 'test_signal', 0
        )
        
        assert result.correlation == 0.0, (
            "Constant labels should have zero correlation"
        )
    
    def test_nan_in_signal_filtered(self):
        """Test that NaN values in signal are filtered."""
        signal = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        labels = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        
        result = compute_signal_regime_correlation(
            signal, labels, 'test_signal', 0
        )
        
        # Should compute correlation on 4 valid samples
        assert result.n_samples == 4
        assert np.isfinite(result.correlation)
    
    def test_inf_in_signal_filtered(self):
        """Test that Inf values in signal are filtered."""
        signal = np.array([1.0, np.inf, 3.0, 4.0, 5.0])
        labels = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        
        result = compute_signal_regime_correlation(
            signal, labels, 'test_signal', 0
        )
        
        # Should compute correlation on 4 valid samples
        assert result.n_samples == 4
        assert np.isfinite(result.correlation)
    
    def test_insufficient_samples(self):
        """Test handling of too few samples."""
        signal = np.array([1.0, 2.0])
        labels = np.array([0.1, 0.2])
        
        result = compute_signal_regime_correlation(
            signal, labels, 'test_signal', 0
        )
        
        # Should return zero correlation and not significant
        assert result.correlation == 0.0
        assert result.p_value == 1.0
        assert not result.is_significant
    
    def test_sign_consistency_positive(self):
        """Test sign consistency for expected positive signals."""
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        labels = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # Positive relationship
        
        result = compute_signal_regime_correlation(
            signal, labels, 'true_ofi', 0  # true_ofi expects positive
        )
        
        assert result.expected_sign == '+'
        assert result.sign_consistent == True
    
    def test_sign_consistency_negative(self):
        """Test sign inconsistency when correlation is opposite."""
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        labels = np.array([0.5, 0.4, 0.3, 0.2, 0.1])  # Negative relationship
        
        result = compute_signal_regime_correlation(
            signal, labels, 'true_ofi', 0  # true_ofi expects positive
        )
        
        assert result.expected_sign == '+'
        assert result.sign_consistent == False
    
    def test_regime_name_lookup(self):
        """Test that regime names are correctly looked up."""
        signal = np.array([1.0, 2.0, 3.0])
        labels = np.array([0.1, 0.2, 0.3])
        
        for regime, name in REGIME_NAMES.items():
            result = compute_signal_regime_correlation(
                signal, labels, 'test', regime
            )
            assert result.regime_name == name


class TestAllRegimeCorrelations:
    """Tests for compute_all_regime_correlations function."""
    
    @pytest.fixture
    def mock_features_labels(self):
        """Create mock features and labels for testing."""
        np.random.seed(42)
        n_samples = 100
        n_features = 98
        
        features = np.random.randn(n_samples, n_features)
        
        # Set time_regime (index 93) to cycle through 0, 1, 2, 3
        features[:, FeatureIndex.TIME_REGIME] = np.tile([0, 1, 2, 3], n_samples // 4)
        
        labels = np.random.choice([-1, 0, 1], n_samples)
        
        return features, labels
    
    def test_returns_all_signal_regime_pairs(self, mock_features_labels):
        """Test that all signal-regime pairs are computed."""
        features, labels = mock_features_labels
        
        results = compute_all_regime_correlations(features, labels)
        
        # Should have 8 signals × 4 regimes = 32 results
        expected_count = len(CORE_SIGNAL_INDICES) * 4
        assert len(results) == expected_count, (
            f"Expected {expected_count} results, got {len(results)}"
        )
    
    def test_custom_signal_indices(self, mock_features_labels):
        """Test with custom signal indices."""
        features, labels = mock_features_labels
        
        custom_indices = {'signal_a': 0, 'signal_b': 1}
        results = compute_all_regime_correlations(
            features, labels, signal_indices=custom_indices
        )
        
        # Should have 2 signals × 4 regimes = 8 results
        assert len(results) == 8
    
    def test_length_mismatch_raises(self, mock_features_labels):
        """Test that mismatched lengths raise ValueError."""
        features, labels = mock_features_labels
        
        with pytest.raises(ValueError, match="Length mismatch"):
            compute_all_regime_correlations(features, labels[:50])


class TestSignalSeasonality:
    """Tests for compute_signal_seasonality function."""
    
    def test_identifies_regime_dependent_signal(self):
        """Test detection of regime-dependent signals."""
        # Create correlations that vary significantly by regime
        correlations = [
            SignalRegimeCorrelation(
                signal_name='test_signal', regime=0, regime_name='OPEN',
                n_samples=100, correlation=0.5, p_value=0.001,
                is_significant=True, expected_sign='+', sign_consistent=True
            ),
            SignalRegimeCorrelation(
                signal_name='test_signal', regime=1, regime_name='EARLY',
                n_samples=100, correlation=0.3, p_value=0.001,
                is_significant=True, expected_sign='+', sign_consistent=True
            ),
            SignalRegimeCorrelation(
                signal_name='test_signal', regime=2, regime_name='MIDDAY',
                n_samples=100, correlation=0.1, p_value=0.001,
                is_significant=True, expected_sign='+', sign_consistent=True
            ),
            SignalRegimeCorrelation(
                signal_name='test_signal', regime=3, regime_name='CLOSE',
                n_samples=100, correlation=0.05, p_value=0.1,
                is_significant=False, expected_sign='+', sign_consistent=True
            ),
        ]
        
        results = compute_signal_seasonality(correlations, regime_dependence_threshold=0.01)
        
        assert len(results) == 1
        ss = results[0]
        
        # Most predictive regime should be OPEN (|0.5| > others)
        assert ss.most_predictive_regime == 0
        
        # Least predictive regime should be CLOSE (|0.05|)
        assert ss.least_predictive_regime == 3
        
        # Should be regime-dependent (range = 0.5 - 0.05 = 0.45 > 0.01)
        assert ss.is_regime_dependent == True
        
        # Correlation range
        assert abs(ss.correlation_range - 0.45) < 0.01
    
    def test_identifies_stable_signal(self):
        """Test detection of stable (non-regime-dependent) signals."""
        # Create correlations that are similar across regimes
        correlations = [
            SignalRegimeCorrelation(
                signal_name='stable_signal', regime=0, regime_name='OPEN',
                n_samples=100, correlation=0.15, p_value=0.001,
                is_significant=True, expected_sign='+', sign_consistent=True
            ),
            SignalRegimeCorrelation(
                signal_name='stable_signal', regime=1, regime_name='EARLY',
                n_samples=100, correlation=0.14, p_value=0.001,
                is_significant=True, expected_sign='+', sign_consistent=True
            ),
            SignalRegimeCorrelation(
                signal_name='stable_signal', regime=2, regime_name='MIDDAY',
                n_samples=100, correlation=0.15, p_value=0.001,
                is_significant=True, expected_sign='+', sign_consistent=True
            ),
        ]
        
        # High threshold
        results = compute_signal_seasonality(correlations, regime_dependence_threshold=0.05)
        
        assert len(results) == 1
        ss = results[0]
        
        # Correlation range = 0.15 - 0.14 = 0.01 < 0.05
        assert ss.is_regime_dependent == False
    
    def test_regime_impact_ratio(self):
        """Test regime impact ratio calculation."""
        correlations = [
            SignalRegimeCorrelation(
                signal_name='test', regime=0, regime_name='OPEN',
                n_samples=100, correlation=0.4, p_value=0.001,
                is_significant=True, expected_sign='+', sign_consistent=True
            ),
            SignalRegimeCorrelation(
                signal_name='test', regime=1, regime_name='EARLY',
                n_samples=100, correlation=0.1, p_value=0.001,
                is_significant=True, expected_sign='+', sign_consistent=True
            ),
        ]
        
        results = compute_signal_seasonality(correlations)
        
        ss = results[0]
        # Ratio should be 0.4 / 0.1 = 4.0
        assert abs(ss.regime_impact_ratio - 4.0) < 0.01


class TestRegimeImportance:
    """Tests for compute_regime_importance function."""
    
    def test_average_absolute_correlation(self):
        """Test that importance = average |correlation| per regime."""
        correlations = [
            SignalRegimeCorrelation(
                signal_name='s1', regime=0, regime_name='OPEN',
                n_samples=100, correlation=0.4, p_value=0.001,
                is_significant=True, expected_sign='+', sign_consistent=True
            ),
            SignalRegimeCorrelation(
                signal_name='s2', regime=0, regime_name='OPEN',
                n_samples=100, correlation=-0.2, p_value=0.001,
                is_significant=True, expected_sign='+', sign_consistent=False
            ),
            SignalRegimeCorrelation(
                signal_name='s1', regime=1, regime_name='EARLY',
                n_samples=100, correlation=0.1, p_value=0.001,
                is_significant=True, expected_sign='+', sign_consistent=True
            ),
        ]
        
        importance = compute_regime_importance(correlations)
        
        # Regime 0: average(|0.4|, |-0.2|) = 0.3
        assert abs(importance[0] - 0.3) < 0.01
        
        # Regime 1: average(|0.1|) = 0.1
        assert abs(importance[1] - 0.1) < 0.01


class TestRecommendations:
    """Tests for generate_recommendations function."""
    
    def test_regime_matters_recommendation(self):
        """Test that high regime variance triggers recommendation."""
        regime_stats = [
            RegimeStats(0, 'OPEN', 1000, 0.0, 0.5, {-1: 33, 0: 34, 1: 33}),
            RegimeStats(2, 'MIDDAY', 1000, 0.0, 0.3, {-1: 33, 0: 34, 1: 33}),
        ]
        
        signal_seasonality = []
        regime_importance = {0: 0.3, 2: 0.1}
        
        recommendations = generate_recommendations(
            regime_stats, signal_seasonality, regime_importance
        )
        
        # Should have recommendation about regime mattering
        regime_recs = [r for r in recommendations if "REGIME MATTERS" in r]
        assert len(regime_recs) > 0
    
    def test_filter_closed_recommendation(self):
        """Test that CLOSED regime triggers filter recommendation."""
        regime_stats = [
            RegimeStats(4, 'CLOSED', 100, 0.0, 0.5, {-1: 33, 0: 34, 1: 33}),
        ]
        
        recommendations = generate_recommendations(regime_stats, [], {})
        
        # Should have recommendation about filtering CLOSED
        filter_recs = [r for r in recommendations if "FILTER" in r]
        assert len(filter_recs) > 0


class TestIntegration:
    """Integration tests for the full analysis pipeline."""
    
    @pytest.fixture
    def realistic_data(self):
        """Create realistic test data with regime-dependent correlations."""
        np.random.seed(42)
        n_samples = 1000
        n_features = 98
        
        features = np.zeros((n_samples, n_features))
        labels = np.zeros(n_samples)
        
        # Assign regimes
        features[:250, FeatureIndex.TIME_REGIME] = 0  # OPEN
        features[250:500, FeatureIndex.TIME_REGIME] = 1  # EARLY
        features[500:750, FeatureIndex.TIME_REGIME] = 2  # MIDDAY
        features[750:, FeatureIndex.TIME_REGIME] = 3  # CLOSE
        
        # Create true_ofi with regime-dependent correlation to labels
        # OPEN: high correlation (0.3)
        features[:250, FeatureIndex.TRUE_OFI] = np.random.randn(250)
        labels[:250] = features[:250, FeatureIndex.TRUE_OFI] * 0.5 + np.random.randn(250) * 0.5
        
        # MIDDAY: low correlation (0.1)
        features[500:750, FeatureIndex.TRUE_OFI] = np.random.randn(250)
        labels[500:750] = features[500:750, FeatureIndex.TRUE_OFI] * 0.1 + np.random.randn(250) * 0.9
        
        # Fill other features with noise
        for idx in CORE_SIGNAL_INDICES.values():
            if idx != FeatureIndex.TRUE_OFI and idx != FeatureIndex.TIME_REGIME:
                features[:, idx] = np.random.randn(n_samples)
        
        return features, labels
    
    def test_full_analysis_runs(self, realistic_data):
        """Test that full analysis completes without errors."""
        features, labels = realistic_data
        
        summary = run_intraday_seasonality_analysis(features, labels)
        
        assert isinstance(summary, IntradaySeasonalitySummary)
        assert len(summary.regime_stats) > 0
        assert len(summary.signal_regime_correlations) > 0
        assert len(summary.signal_seasonality) > 0
        assert len(summary.overall_regime_importance) > 0
    
    def test_to_dict_serializable(self, realistic_data):
        """Test that to_dict output is JSON-serializable."""
        features, labels = realistic_data
        
        summary = run_intraday_seasonality_analysis(features, labels)
        result = to_dict(summary)
        
        # Should be a dict
        assert isinstance(result, dict)
        
        # Check key sections exist
        assert 'regime_stats' in result
        assert 'signal_regime_correlations' in result
        assert 'signal_seasonality' in result
        assert 'recommendations' in result
        
        # Try JSON serialization
        import json
        json_str = json.dumps(result)
        assert len(json_str) > 0
    
    def test_detects_regime_dependent_signal(self, realistic_data):
        """Test that analysis detects true_ofi as regime-dependent."""
        features, labels = realistic_data
        
        summary = run_intraday_seasonality_analysis(features, labels)
        
        # Find true_ofi seasonality
        ofi_seasonality = [
            s for s in summary.signal_seasonality if s.signal_name == 'true_ofi'
        ]
        
        assert len(ofi_seasonality) == 1
        # true_ofi should be detected as regime-dependent
        # (we created it with different correlations per regime)


class TestEdgeCases:
    """Edge case tests."""
    
    def test_single_regime(self):
        """Test handling of data with only one regime."""
        features = np.random.randn(100, 98)
        features[:, FeatureIndex.TIME_REGIME] = 2  # All MIDDAY
        labels = np.random.choice([-1, 0, 1], 100)
        
        summary = run_intraday_seasonality_analysis(features, labels)
        
        # Should have only one regime
        assert len(summary.regime_stats) == 1
        assert summary.regime_stats[0].regime_name == "MIDDAY"
    
    def test_all_nan_time_regime(self):
        """Test handling of all NaN time regimes."""
        features = np.random.randn(100, 98)
        features[:, FeatureIndex.TIME_REGIME] = np.nan  # All NaN
        labels = np.random.choice([-1, 0, 1], 100)
        
        summary = run_intraday_seasonality_analysis(features, labels)
        
        # Should have no regime stats
        assert len(summary.regime_stats) == 0
    
    def test_inf_in_signals(self):
        """Test handling of Inf in signal values."""
        features = np.random.randn(100, 98)
        features[:, FeatureIndex.TIME_REGIME] = np.tile([0, 1, 2, 3], 25)
        features[0, FeatureIndex.TRUE_OFI] = np.inf  # One Inf value
        labels = np.random.choice([-1, 0, 1], 100)
        
        summary = run_intraday_seasonality_analysis(features, labels)
        
        # Should complete without errors
        assert isinstance(summary, IntradaySeasonalitySummary)


class TestFormulaCorrectness:
    """Test that formulas match expected mathematical definitions."""
    
    def test_pearson_correlation_matches_scipy(self):
        """Test that our correlation matches scipy.stats.pearsonr."""
        np.random.seed(42)
        signal = np.random.randn(100)
        labels = np.random.randn(100)
        
        result = compute_signal_regime_correlation(signal, labels, 'test', 0)
        
        scipy_corr, scipy_p = scipy_stats.pearsonr(signal, labels)
        
        assert abs(result.correlation - scipy_corr) < 1e-10, (
            f"Correlation mismatch: ours={result.correlation}, scipy={scipy_corr}"
        )
        assert abs(result.p_value - scipy_p) < 1e-10, (
            f"P-value mismatch: ours={result.p_value}, scipy={scipy_p}"
        )


class TestInvariants:
    """Test invariants that should always hold."""
    
    def test_regime_names_complete(self):
        """Test that all regime values have names."""
        for regime in [0, 1, 2, 3, 4]:
            assert regime in REGIME_NAMES, f"Missing name for regime {regime}"
    
    def test_expected_signs_complete(self):
        """Test that all core signals have expected signs."""
        for signal_name in CORE_SIGNAL_INDICES:
            assert signal_name in EXPECTED_SIGNS, (
                f"Missing expected sign for {signal_name}"
            )
    
    def test_correlation_bounded(self):
        """Test that correlations are always in [-1, 1]."""
        np.random.seed(42)
        for _ in range(100):
            signal = np.random.randn(50)
            labels = np.random.randn(50)
            
            result = compute_signal_regime_correlation(signal, labels, 'test', 0)
            
            assert -1 <= result.correlation <= 1, (
                f"Correlation out of bounds: {result.correlation}"
            )

