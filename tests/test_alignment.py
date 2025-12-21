"""
Comprehensive tests for feature-label alignment.

These tests verify that:
1. Single-day alignment is mathematically correct
2. Multi-day alignment (load_split_aligned) handles day boundaries correctly
3. The alignment produces correct 1:1 correspondence for any dataset

Run with: pytest tests/test_alignment.py -v
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lobtrainer.analysis.data_loading import (
    align_features_with_labels,
    load_split_aligned,
    load_split,
    WINDOW_SIZE,
    STRIDE,
)


class TestSingleDayAlignment:
    """Tests for align_features_with_labels on single-day data."""
    
    def test_basic_alignment_formula(self):
        """
        Test that alignment follows the formula:
        feat_idx = i * stride + window_size - 1
        """
        # Create simple test data where feature[i] = i
        n_samples = 1000
        n_features = 10
        features = np.arange(n_samples * n_features).reshape(n_samples, n_features)
        
        # Expected number of labels
        window_size = 100
        stride = 10
        expected_n_labels = (n_samples - window_size) // stride + 1
        
        aligned = align_features_with_labels(
            features, expected_n_labels, window_size, stride
        )
        
        # Verify shape
        assert aligned.shape == (expected_n_labels, n_features), \
            f"Shape mismatch: expected ({expected_n_labels}, {n_features}), got {aligned.shape}"
        
        # Verify alignment formula for first few labels
        for i in range(min(5, expected_n_labels)):
            expected_feat_idx = i * stride + window_size - 1
            # First column of features is the row index * n_features
            expected_first_col = expected_feat_idx * n_features
            actual_first_col = aligned[i, 0]
            
            assert actual_first_col == expected_first_col, \
                f"Label {i}: expected feature from row {expected_feat_idx} " \
                f"(first_col={expected_first_col}), got first_col={actual_first_col}"
    
    def test_label_0_uses_feature_99(self):
        """
        With window_size=100, stride=10:
        label[0] should use feature[99] (end of window [0, 100))
        """
        n_samples = 500
        n_features = 5
        features = np.arange(n_samples * n_features).reshape(n_samples, n_features)
        
        n_labels = (n_samples - 100) // 10 + 1
        aligned = align_features_with_labels(features, n_labels, 100, 10)
        
        # Label 0 should correspond to feature row 99
        expected = features[99]
        np.testing.assert_array_equal(
            aligned[0], expected,
            err_msg="Label 0 should use feature[99] with window_size=100"
        )
    
    def test_label_1_uses_feature_109(self):
        """
        With window_size=100, stride=10:
        label[1] should use feature[109] (end of window [10, 110))
        """
        n_samples = 500
        n_features = 5
        features = np.arange(n_samples * n_features).reshape(n_samples, n_features)
        
        n_labels = (n_samples - 100) // 10 + 1
        aligned = align_features_with_labels(features, n_labels, 100, 10)
        
        # Label 1 should correspond to feature row 109
        expected = features[109]
        np.testing.assert_array_equal(
            aligned[1], expected,
            err_msg="Label 1 should use feature[109] with window_size=100, stride=10"
        )
    
    def test_last_label_uses_last_valid_feature(self):
        """
        The last label should use a feature within bounds.
        """
        n_samples = 1000
        n_features = 5
        features = np.arange(n_samples * n_features).reshape(n_samples, n_features)
        
        window_size = 100
        stride = 10
        n_labels = (n_samples - window_size) // stride + 1
        
        aligned = align_features_with_labels(features, n_labels, window_size, stride)
        
        # Last label index
        last_label_idx = n_labels - 1
        expected_feat_idx = last_label_idx * stride + window_size - 1
        
        assert expected_feat_idx < n_samples, \
            f"Last feature index {expected_feat_idx} should be < {n_samples}"
        
        expected = features[expected_feat_idx]
        np.testing.assert_array_equal(
            aligned[last_label_idx], expected,
            err_msg=f"Last label should use feature[{expected_feat_idx}]"
        )
    
    def test_different_window_stride(self):
        """Test with non-default window and stride values."""
        n_samples = 500
        n_features = 3
        features = np.arange(n_samples * n_features).reshape(n_samples, n_features)
        
        window_size = 50
        stride = 5
        n_labels = (n_samples - window_size) // stride + 1
        
        aligned = align_features_with_labels(features, n_labels, window_size, stride)
        
        # Label 0 should use feature[49] (end of window [0, 50))
        expected = features[49]
        np.testing.assert_array_equal(aligned[0], expected)
        
        # Label 1 should use feature[54] (end of window [5, 55))
        expected = features[54]
        np.testing.assert_array_equal(aligned[1], expected)


class TestMultiDayAlignment:
    """Tests for load_split_aligned with multi-day data."""
    
    @pytest.fixture
    def temp_dataset(self):
        """Create a temporary dataset with known properties."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            train_dir = data_dir / 'train'
            train_dir.mkdir()
            
            window_size = 100
            stride = 10
            
            # Day 1: 1000 samples → 91 labels
            day1_samples = 1000
            day1_labels = (day1_samples - window_size) // stride + 1
            day1_features = np.arange(day1_samples * 98, dtype=np.float32).reshape(day1_samples, 98)
            day1_labels_arr = np.ones(day1_labels, dtype=np.int8)  # All label=1
            
            np.save(train_dir / '20250101_features.npy', day1_features)
            np.save(train_dir / '20250101_labels.npy', day1_labels_arr)
            
            # Day 2: 1200 samples → 111 labels (different length!)
            day2_samples = 1200
            day2_labels = (day2_samples - window_size) // stride + 1
            # Offset by 1,000,000 to distinguish from day 1
            day2_features = np.arange(1_000_000, 1_000_000 + day2_samples * 98, dtype=np.float32).reshape(day2_samples, 98)
            day2_labels_arr = np.ones(day2_labels, dtype=np.int8) * 2  # All label=2
            
            np.save(train_dir / '20250102_features.npy', day2_features)
            np.save(train_dir / '20250102_labels.npy', day2_labels_arr)
            
            # Day 3: 800 samples → 71 labels
            day3_samples = 800
            day3_labels = (day3_samples - window_size) // stride + 1
            day3_features = np.arange(2_000_000, 2_000_000 + day3_samples * 98, dtype=np.float32).reshape(day3_samples, 98)
            day3_labels_arr = np.ones(day3_labels, dtype=np.int8) * 3  # All label=3
            
            np.save(train_dir / '20250103_features.npy', day3_features)
            np.save(train_dir / '20250103_labels.npy', day3_labels_arr)
            
            yield {
                'data_dir': data_dir,
                'days': [
                    {'date': '20250101', 'n_samples': day1_samples, 'n_labels': day1_labels, 'label_value': 1},
                    {'date': '20250102', 'n_samples': day2_samples, 'n_labels': day2_labels, 'label_value': 2},
                    {'date': '20250103', 'n_samples': day3_samples, 'n_labels': day3_labels, 'label_value': 3},
                ],
                'total_labels': day1_labels + day2_labels + day3_labels,
            }
    
    def test_load_split_aligned_shape(self, temp_dataset):
        """Test that aligned features have same length as labels."""
        data = load_split_aligned(temp_dataset['data_dir'], 'train')
        
        assert data['features'].shape[0] == len(data['labels']), \
            f"Features ({data['features'].shape[0]}) must equal labels ({len(data['labels'])})"
        
        assert data['features'].shape[0] == temp_dataset['total_labels'], \
            f"Total labels mismatch: expected {temp_dataset['total_labels']}, got {data['features'].shape[0]}"
    
    def test_load_split_aligned_day_boundaries(self, temp_dataset):
        """Test that day boundaries are correctly tracked."""
        data = load_split_aligned(temp_dataset['data_dir'], 'train')
        
        expected_boundaries = []
        current = 0
        for day in temp_dataset['days']:
            expected_boundaries.append((current, current + day['n_labels']))
            current += day['n_labels']
        
        assert data['day_boundaries'] == expected_boundaries, \
            f"Day boundaries mismatch: expected {expected_boundaries}, got {data['day_boundaries']}"
    
    def test_day1_alignment_correct(self, temp_dataset):
        """Test that Day 1 features are correctly aligned."""
        data = load_split_aligned(temp_dataset['data_dir'], 'train')
        
        day1_info = temp_dataset['days'][0]
        start, end = data['day_boundaries'][0]
        
        # Day 1 label 0 should use feature[99] of Day 1
        # Day 1 features start at 0, so feature[99] has first column = 99 * 98
        expected_first_col_label0 = 99 * 98
        actual_first_col_label0 = data['features'][start, 0]
        
        assert actual_first_col_label0 == expected_first_col_label0, \
            f"Day 1 label 0: expected first_col={expected_first_col_label0}, got {actual_first_col_label0}"
        
        # All Day 1 labels should have value 1
        assert np.all(data['labels'][start:end] == 1), \
            "All Day 1 labels should be 1"
    
    def test_day2_alignment_correct(self, temp_dataset):
        """Test that Day 2 features are correctly aligned (not drifted)."""
        data = load_split_aligned(temp_dataset['data_dir'], 'train')
        
        start, end = data['day_boundaries'][1]
        
        # Day 2 label 0 should use feature[99] of Day 2
        # Day 2 features start at 1,000,000, so feature[99] has first column = 1,000,000 + 99 * 98
        expected_first_col_label0 = 1_000_000 + 99 * 98
        actual_first_col_label0 = data['features'][start, 0]
        
        assert actual_first_col_label0 == expected_first_col_label0, \
            f"Day 2 label 0: expected first_col={expected_first_col_label0}, got {actual_first_col_label0}"
        
        # All Day 2 labels should have value 2
        assert np.all(data['labels'][start:end] == 2), \
            "All Day 2 labels should be 2"
    
    def test_day3_alignment_correct(self, temp_dataset):
        """Test that Day 3 features are correctly aligned."""
        data = load_split_aligned(temp_dataset['data_dir'], 'train')
        
        start, end = data['day_boundaries'][2]
        
        # Day 3 label 0 should use feature[99] of Day 3
        expected_first_col_label0 = 2_000_000 + 99 * 98
        actual_first_col_label0 = data['features'][start, 0]
        
        assert actual_first_col_label0 == expected_first_col_label0, \
            f"Day 3 label 0: expected first_col={expected_first_col_label0}, got {actual_first_col_label0}"
        
        # All Day 3 labels should have value 3
        assert np.all(data['labels'][start:end] == 3), \
            "All Day 3 labels should be 3"
    
    def test_global_alignment_would_be_wrong(self, temp_dataset):
        """
        Demonstrate that using global alignment on concatenated data is WRONG.
        
        This is the bug we're fixing!
        
        The drift calculation:
        - Day 1: 1000 samples, 91 labels
        - Day 2 label 0 (global label 91) should use feature[99] of Day 2
        - In concatenated array: index 1000 + 99 = 1099
        - Global formula gives: 91 * 10 + 99 = 1009
        - Drift = 1099 - 1009 = 90 samples!
        """
        # Load raw (unaligned) data
        raw_data = load_split(temp_dataset['data_dir'], 'train')
        
        # Apply global alignment (THE BUG)
        global_aligned = align_features_with_labels(
            raw_data['features'], 
            len(raw_data['labels']),
            WINDOW_SIZE,
            STRIDE,
        )
        
        # Load correctly aligned data
        correct_data = load_split_aligned(temp_dataset['data_dir'], 'train')
        
        # Day 2 starts at label index = day1_n_labels = 91
        day2_start_label = temp_dataset['days'][0]['n_labels']
        
        # Calculate expected values
        day1_samples = temp_dataset['days'][0]['n_samples']  # 1000
        
        # CORRECT: Day 2 label 0 uses feature[99] of Day 2
        # In concatenated array: day1_samples + 99 = 1000 + 99 = 1099
        correct_raw_index = day1_samples + 99
        
        # GLOBAL (wrong): feat_idx = day2_start_label * stride + window_size - 1
        # = 91 * 10 + 99 = 1009
        global_raw_index = day2_start_label * STRIDE + WINDOW_SIZE - 1
        
        # The drift
        drift = correct_raw_index - global_raw_index  # 1099 - 1009 = 90
        
        assert drift == 90, f"Expected drift of 90, got {drift}"
        
        # Verify the actual feature values differ
        global_day2_label0_feat = global_aligned[day2_start_label, 0]
        correct_day2_label0_feat = correct_data['features'][day2_start_label, 0]
        
        # They should be DIFFERENT (by drift * 98 since each row has 98 features)
        expected_difference = drift * 98  # 90 * 98 = 8820
        actual_difference = correct_day2_label0_feat - global_day2_label0_feat
        
        assert abs(actual_difference - expected_difference) < 1, \
            f"Feature difference should be ~{expected_difference}, got {actual_difference}"
        
        # Verify correct alignment uses exact expected value
        # Day 2 features start at 1,000,000 + (row * 98)
        # Row 99 of Day 2: first_col = 1,000,000 + 99 * 98 = 1,009,702
        expected_correct_value = 1_000_000 + 99 * 98
        assert correct_day2_label0_feat == expected_correct_value, \
            f"Correct alignment should give {expected_correct_value}, got {correct_day2_label0_feat}"


class TestAlignmentInvariant:
    """Tests for alignment invariants that should always hold."""
    
    def test_label_count_matches_formula(self):
        """
        Label count should match formula: (n_samples - window_size) // stride + 1
        """
        for n_samples in [100, 500, 1000, 10000]:
            for window_size in [50, 100, 200]:
                for stride in [5, 10, 20]:
                    if n_samples < window_size:
                        continue
                    
                    expected_labels = (n_samples - window_size) // stride + 1
                    features = np.zeros((n_samples, 10))
                    
                    aligned = align_features_with_labels(
                        features, expected_labels, window_size, stride
                    )
                    
                    assert aligned.shape[0] == expected_labels, \
                        f"n={n_samples}, w={window_size}, s={stride}: " \
                        f"expected {expected_labels} labels, got {aligned.shape[0]}"
    
    def test_aligned_features_within_bounds(self):
        """All aligned features should come from within the feature array bounds."""
        n_samples = 1000
        n_features = 10
        features = np.arange(n_samples * n_features).reshape(n_samples, n_features)
        
        window_size = 100
        stride = 10
        n_labels = (n_samples - window_size) // stride + 1
        
        aligned = align_features_with_labels(features, n_labels, window_size, stride)
        
        # All aligned values should be in range [0, n_samples * n_features)
        max_possible = n_samples * n_features - 1
        assert aligned.max() <= max_possible, \
            f"Aligned features contain values > {max_possible}"
        
        assert aligned.min() >= 0, \
            "Aligned features contain negative values"


class TestStreamingAlignment:
    """
    Tests for streaming module alignment functions.
    
    These tests verify that iter_days_aligned and align_features_for_day
    produce correctly aligned feature-label pairs.
    """
    
    @pytest.fixture
    def mock_streaming_data(self, tmp_path):
        """Create mock data for streaming tests."""
        from lobtrainer.analysis.streaming import WINDOW_SIZE, STRIDE
        
        train_dir = tmp_path / 'train'
        train_dir.mkdir()
        
        # Create 2 simple days
        days = []
        for i, (n_samples, label_val) in enumerate([(500, 1), (600, -1)]):
            date = f"2025-01-0{i+1}"
            
            # Features: column 0 = sample index, column 1 = day index
            features = np.zeros((n_samples, 98), dtype=np.float32)
            features[:, 0] = np.arange(n_samples)  # Sample index
            features[:, 1] = i  # Day index
            
            # Labels
            n_labels = (n_samples - WINDOW_SIZE) // STRIDE + 1
            labels = np.full(n_labels, label_val, dtype=np.int32)
            
            # Save files
            np.save(train_dir / f"{date}_features.npy", features)
            np.save(train_dir / f"{date}_labels.npy", labels)
            
            days.append({
                'date': date,
                'n_samples': n_samples,
                'n_labels': n_labels,
                'label_val': label_val,
            })
        
        return {
            'data_dir': tmp_path,
            'days': days,
        }
    
    def test_align_features_for_day_shape(self):
        """Test that align_features_for_day produces correct shape."""
        from lobtrainer.analysis.streaming import align_features_for_day, WINDOW_SIZE, STRIDE
        
        n_samples = 1000
        n_features = 98
        n_labels = (n_samples - WINDOW_SIZE) // STRIDE + 1
        
        features = np.arange(n_samples * n_features).reshape(n_samples, n_features).astype(np.float32)
        aligned = align_features_for_day(features, n_labels)
        
        assert aligned.shape == (n_labels, n_features), \
            f"Expected shape ({n_labels}, {n_features}), got {aligned.shape}"
    
    def test_align_features_for_day_formula(self):
        """Test that align_features_for_day uses correct formula."""
        from lobtrainer.analysis.streaming import align_features_for_day, WINDOW_SIZE, STRIDE
        
        n_samples = 500
        n_features = 98
        n_labels = (n_samples - WINDOW_SIZE) // STRIDE + 1
        
        # Features: column 0 = sample index
        features = np.zeros((n_samples, n_features), dtype=np.float32)
        features[:, 0] = np.arange(n_samples)
        
        aligned = align_features_for_day(features, n_labels)
        
        # For label i, the aligned feature should come from index i * stride + window_size - 1
        for i in range(min(5, n_labels)):  # Check first 5
            expected_idx = i * STRIDE + WINDOW_SIZE - 1
            actual_idx = aligned[i, 0]
            assert actual_idx == expected_idx, \
                f"Label {i}: expected feature from idx {expected_idx}, got {actual_idx}"
    
    def test_iter_days_aligned_yields_correct_pairs(self, mock_streaming_data):
        """Test that iter_days_aligned yields aligned feature-label pairs."""
        from lobtrainer.analysis.streaming import iter_days_aligned, WINDOW_SIZE, STRIDE
        
        data_dir = mock_streaming_data['data_dir']
        days_info = mock_streaming_data['days']
        
        day_num = 0
        for day in iter_days_aligned(data_dir, 'train'):
            expected = days_info[day_num]
            
            # Check shape
            assert day.features.shape[0] == day.n_pairs, \
                f"Day {day.date}: features rows ({day.features.shape[0]}) != n_pairs ({day.n_pairs})"
            
            assert len(day.labels) == day.n_pairs, \
                f"Day {day.date}: labels len ({len(day.labels)}) != n_pairs ({day.n_pairs})"
            
            # Check label values match expected
            assert np.all(day.labels == expected['label_val']), \
                f"Day {day.date}: expected label {expected['label_val']}, got unique {np.unique(day.labels)}"
            
            # Check alignment: column 1 should all be the day index
            assert np.all(day.features[:, 1] == day_num), \
                f"Day {day.date}: expected day_idx {day_num} in column 1"
            
            # Check alignment: column 0 should follow the formula
            expected_first_idx = WINDOW_SIZE - 1
            actual_first_idx = day.features[0, 0]
            assert actual_first_idx == expected_first_idx, \
                f"Day {day.date}: first aligned feature should be from idx {expected_first_idx}, got {actual_first_idx}"
            
            day_num += 1
        
        assert day_num == 2, f"Expected 2 days, got {day_num}"
    
    def test_iter_days_aligned_matches_load_split_aligned(self, mock_streaming_data):
        """Test that streaming and bulk loading produce identical results."""
        from lobtrainer.analysis.streaming import iter_days_aligned
        from lobtrainer.analysis.data_loading import load_split_aligned
        
        data_dir = mock_streaming_data['data_dir']
        
        # Bulk load
        bulk_data = load_split_aligned(data_dir, 'train')
        
        # Stream load
        streamed_features = []
        streamed_labels = []
        for day in iter_days_aligned(data_dir, 'train'):
            streamed_features.append(day.features)
            streamed_labels.append(day.labels)
        
        streamed_features = np.vstack(streamed_features)
        streamed_labels = np.concatenate(streamed_labels)
        
        # Compare
        assert streamed_features.shape == bulk_data['features'].shape, \
            f"Shape mismatch: streaming {streamed_features.shape} vs bulk {bulk_data['features'].shape}"
        
        # Note: dtypes may differ (float32 vs float64), so we compare with tolerance
        np.testing.assert_allclose(
            streamed_features, 
            bulk_data['features'],
            rtol=1e-5,
            err_msg="Streaming and bulk features should match"
        )
        
        np.testing.assert_array_equal(
            streamed_labels,
            bulk_data['labels'],
            err_msg="Streaming and bulk labels should match"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

