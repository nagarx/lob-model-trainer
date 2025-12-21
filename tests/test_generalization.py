#!/usr/bin/env python3
"""
Comprehensive tests for generalization.py.

Tests validate:
1. Day statistics computation
2. Signal day-to-day variance
3. Walk-forward validation
4. Generalization assessment
"""

import pytest
import numpy as np
import sys
from pathlib import Path
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lobtrainer.analysis.generalization import (
    compute_day_statistics,
    compute_signal_day_stats,
    walk_forward_validation,
    run_generalization_analysis,
    align_features_for_day,
    DayStatistics,
    SignalDayStats,
    WalkForwardResult,
    GeneralizationSummary,
)
from lobtrainer.constants import LABEL_UP, LABEL_DOWN, LABEL_STABLE


class TestAlignFeaturesForDay:
    """Tests for feature alignment function."""
    
    def test_alignment_formula(self):
        """Test that alignment uses correct formula."""
        n_samples = 500
        n_features = 98
        window_size = 100
        stride = 10
        
        n_labels = (n_samples - window_size) // stride + 1
        
        # Features: column 0 = sample index
        features = np.zeros((n_samples, n_features))
        features[:, 0] = np.arange(n_samples)
        
        aligned = align_features_for_day(features, n_labels, window_size, stride)
        
        assert aligned.shape == (n_labels, n_features)
        
        # First aligned feature should be from index window_size - 1 = 99
        assert aligned[0, 0] == 99, f"Expected 99, got {aligned[0, 0]}"
        
        # Second aligned feature should be from index stride + window_size - 1 = 109
        assert aligned[1, 0] == 109, f"Expected 109, got {aligned[1, 0]}"


class TestDayStatistics:
    """Tests for per-day statistics computation."""
    
    @pytest.fixture
    def mock_days(self):
        """Create mock day data with correct keys."""
        np.random.seed(42)
        
        days = []
        for i, (date, n_samples) in enumerate([
            ('20250101', 1000),
            ('20250102', 1200),
            ('20250103', 800),
        ]):
            n_labels = 91  # (n_samples - 100) // 10 + 1 for default params
            labels = np.random.choice([-1, 0, 1], size=n_labels)
            features = np.random.randn(n_samples, 98)  # Must have 'features' key
            days.append({
                'date': date,
                'labels': labels,
                'features': features,
            })
        
        return days
    
    def test_returns_one_stat_per_day(self, mock_days):
        """Should return one DayStatistics per day."""
        stats = compute_day_statistics(mock_days)
        
        assert len(stats) == 3, f"Expected 3 stats, got {len(stats)}"
        assert all(isinstance(s, DayStatistics) for s in stats)
    
    def test_percentages_sum_to_100(self, mock_days):
        """Label percentages should sum to 100."""
        stats = compute_day_statistics(mock_days)
        
        for s in stats:
            total = s.label_up_pct + s.label_down_pct + s.label_stable_pct
            assert abs(total - 100.0) < 0.1, f"Percentages sum to {total}, expected 100"
    
    def test_sample_counts_match(self, mock_days):
        """Sample counts should match input."""
        stats = compute_day_statistics(mock_days)
        
        for s, day in zip(stats, mock_days):
            assert s.n_labels == len(day['labels']), \
                f"n_labels mismatch: {s.n_labels} vs {len(day['labels'])}"


class TestSignalDayStats:
    """Tests for signal day-to-day variance analysis."""
    
    @pytest.fixture
    def mock_days_for_signal_stats(self):
        """Create mock day data with features key (not aligned_features)."""
        from lobtrainer.constants import FeatureIndex
        np.random.seed(42)
        
        days = []
        for i, (date, n_samples) in enumerate([
            ('20250101', 1000),
            ('20250102', 1100),
            ('20250103', 1050),
            ('20250104', 1020),
        ]):
            n_features = 98
            n_labels = (n_samples - 100) // 10 + 1  # Match alignment formula
            
            # Create raw features (will be aligned internally)
            features = np.random.randn(n_samples, n_features)
            labels = np.random.choice([-1, 0, 1], size=n_labels)
            
            days.append({
                'date': date,
                'features': features,
                'labels': labels,
            })
        
        return days
    
    def test_returns_stats_per_signal(self, mock_days_for_signal_stats):
        """Should return stats for each signal index."""
        signal_indices = [84, 85, 86]
        
        stats = compute_signal_day_stats(mock_days_for_signal_stats, signal_indices)
        
        assert len(stats) == 3, f"Expected 3 signal stats, got {len(stats)}"
        assert all(isinstance(s, SignalDayStats) for s in stats)
    
    def test_stability_score_computed(self, mock_days_for_signal_stats):
        """Should compute stability score."""
        signal_indices = [84]
        
        stats = compute_signal_day_stats(mock_days_for_signal_stats, signal_indices)
        
        assert len(stats) == 1
        assert hasattr(stats[0], 'stability_score')
        assert np.isfinite(stats[0].stability_score)
    
    def test_correlations_per_day(self, mock_days_for_signal_stats):
        """Should have correlation for each day."""
        signal_indices = [84]
        
        stats = compute_signal_day_stats(mock_days_for_signal_stats, signal_indices)
        
        assert len(stats) == 1
        assert len(stats[0].correlations) == len(mock_days_for_signal_stats)


class TestWalkForwardValidation:
    """Tests for walk-forward validation."""
    
    @pytest.fixture
    def mock_days_for_walk_forward(self):
        """Create mock days for walk-forward (with 'features' key)."""
        from lobtrainer.constants import FeatureIndex
        np.random.seed(42)
        
        days = []
        for i in range(10):  # 10 days
            n_samples = 1000
            n_features = 98
            n_labels = (n_samples - 100) // 10 + 1  # 91 labels
            
            features = np.random.randn(n_samples, n_features)
            labels = np.random.choice([-1, 0, 1], size=n_labels)
            
            days.append({
                'date': f'202501{i+1:02d}',
                'features': features,
                'labels': labels,
            })
        
        return days
    
    def test_returns_walk_forward_results(self, mock_days_for_walk_forward):
        """Should return WalkForwardResult for each test day."""
        signal_indices = [84]
        min_train_days = 3
        
        results = walk_forward_validation(
            mock_days_for_walk_forward, signal_indices, min_train_days
        )
        
        # Should have (n_days - min_train_days) results
        expected = len(mock_days_for_walk_forward) - min_train_days
        assert len(results) == expected, f"Expected {expected} results, got {len(results)}"
        assert all(isinstance(r, WalkForwardResult) for r in results)
    
    def test_train_days_increase(self, mock_days_for_walk_forward):
        """Training days (list) should increase in length for each result."""
        signal_indices = [84]
        min_train_days = 3
        
        results = walk_forward_validation(
            mock_days_for_walk_forward, signal_indices, min_train_days
        )
        
        for i, r in enumerate(results):
            # train_days is a list of date strings
            expected_train_count = min_train_days + i
            assert len(r.train_days) == expected_train_count, \
                f"Result {i}: expected {expected_train_count} train days, got {len(r.train_days)}"
    
    def test_accuracy_in_valid_range(self, mock_days_for_walk_forward):
        """Prediction accuracy should be in [0, 1]."""
        signal_indices = [84]
        
        results = walk_forward_validation(mock_days_for_walk_forward, signal_indices, 3)
        
        for r in results:
            assert 0 <= r.prediction_accuracy <= 1, \
                f"Accuracy {r.prediction_accuracy} not in [0, 1]"


class TestRunGeneralizationAnalysis:
    """Integration tests for full generalization analysis."""
    
    @pytest.fixture
    def mock_data_dir(self, tmp_path):
        """Create mock dataset directory."""
        from lobtrainer.constants import FeatureIndex
        np.random.seed(42)
        
        train_dir = tmp_path / 'train'
        train_dir.mkdir()
        
        # Create 5 days of data
        for i in range(5):
            date = f'202501{i+1:02d}'
            n_samples = 1000
            n_features = 98
            
            features = np.random.randn(n_samples, n_features).astype(np.float32)
            n_labels = 91  # (1000 - 100) // 10 + 1
            labels = np.random.choice([-1, 0, 1], size=n_labels)
            
            np.save(train_dir / f'{date}_features.npy', features)
            np.save(train_dir / f'{date}_labels.npy', labels)
        
        return tmp_path
    
    def test_returns_summary(self, mock_data_dir):
        """Should return complete GeneralizationSummary."""
        summary = run_generalization_analysis(
            mock_data_dir, 'train', 
            signal_indices=[84, 85],
            window_size=100, stride=10
        )
        
        assert isinstance(summary, GeneralizationSummary)
        assert summary.day_statistics is not None
        assert summary.signal_day_stats is not None
        assert summary.walk_forward_results is not None
    
    def test_has_recommendations(self, mock_data_dir):
        """Should provide recommendations."""
        summary = run_generalization_analysis(
            mock_data_dir, 'train',
            signal_indices=[84],
            window_size=100, stride=10
        )
        
        assert summary.recommendations is not None
        assert isinstance(summary.recommendations, list)
    
    def test_overall_stability_score(self, mock_data_dir):
        """Should compute overall stability score."""
        summary = run_generalization_analysis(
            mock_data_dir, 'train',
            signal_indices=[84],
            window_size=100, stride=10
        )
        
        assert np.isfinite(summary.overall_stability_score)


class TestEdgeCases:
    """Edge case tests."""
    
    def test_too_few_days_for_walk_forward(self):
        """Should handle too few days gracefully."""
        np.random.seed(42)
        
        # Only 2 days, but min_train_days=3
        days = [
            {'date': '20250101', 'features': np.random.randn(1000, 98), 
             'labels': np.random.choice([-1, 0, 1], 91)},
            {'date': '20250102', 'features': np.random.randn(1000, 98),
             'labels': np.random.choice([-1, 0, 1], 91)},
        ]
        
        results = walk_forward_validation(days, [84], min_train_days=3)
        
        # Should return empty list
        assert len(results) == 0
    
    def test_single_day(self):
        """Should handle single day for day statistics."""
        np.random.seed(42)
        
        days = [
            {'date': '20250101', 'features': np.random.randn(100, 98),
             'labels': np.ones(10)},
        ]
        
        stats = compute_day_statistics(days)
        assert len(stats) == 1
        assert stats[0].label_up_pct == 100.0  # All labels are 1 (UP)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

