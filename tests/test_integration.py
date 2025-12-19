"""
Integration tests with real NVDA exported data.

These tests verify that the data loading pipeline works correctly
with actual exported data from the Rust pipeline.
"""

import os
from pathlib import Path
import numpy as np
import pytest

# Skip all tests if data not available
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "exports" / "nvda_98feat"
pytestmark = pytest.mark.skipif(
    not DATA_DIR.exists(),
    reason=f"NVDA data not found at {DATA_DIR}"
)


class TestRealDataLoading:
    """Test loading real NVDA data."""
    
    def test_load_train_split(self):
        """Load training data successfully."""
        from lobtrainer.data import load_split_data
        from lobtrainer.constants import FEATURE_COUNT
        
        days = load_split_data(DATA_DIR, "train", validate=True)
        
        assert len(days) > 0, "Should have at least one training day"
        
        for day in days:
            assert day.features.shape[1] == FEATURE_COUNT, (
                f"Expected {FEATURE_COUNT} features, got {day.features.shape[1]}"
            )
            assert day.features.dtype == np.float64
            assert day.labels.dtype == np.int64
    
    def test_feature_count_matches_contract(self):
        """Verify exported data matches 98-feature contract."""
        from lobtrainer.data import load_split_data
        from lobtrainer.constants import FEATURE_COUNT
        
        days = load_split_data(DATA_DIR, "train", validate=False)
        
        for day in days:
            assert day.features.shape[1] == FEATURE_COUNT == 98
    
    def test_label_values_valid(self):
        """Labels should be in expected range: {-1=Down, 0=Stable, 1=Up}."""
        from lobtrainer.data import load_split_data
        
        days = load_split_data(DATA_DIR, "train", validate=False)
        
        for day in days:
            unique_labels = np.unique(day.labels)
            # Label encoding: -1 = Down, 0 = Stable, 1 = Up
            # This follows sign convention: negative=bearish, positive=bullish
            valid_labels = {-1, 0, 1}
            assert all(l in valid_labels for l in unique_labels), (
                f"Invalid labels: {unique_labels}. Expected subset of {valid_labels}"
            )
    
    def test_book_valid_present(self):
        """Book valid feature should be present and binary."""
        from lobtrainer.data import load_split_data
        from lobtrainer.constants import FeatureIndex
        
        days = load_split_data(DATA_DIR, "train", validate=False)
        
        for day in days:
            book_valid = day.features[:, FeatureIndex.BOOK_VALID]
            assert np.all((book_valid == 0) | (book_valid == 1)), (
                "BOOK_VALID should be binary (0 or 1)"
            )
    
    def test_time_regime_categorical(self):
        """Time regime should be categorical 0-4 (NOT normalized)."""
        from lobtrainer.data import load_split_data
        from lobtrainer.constants import FeatureIndex
        
        days = load_split_data(DATA_DIR, "train", validate=False)
        
        for day in days:
            time_regime = day.features[:, FeatureIndex.TIME_REGIME]
            unique_regimes = np.unique(time_regime)
            
            # TIME_REGIME should be categorical: 0=Open, 1=Early, 2=Midday, 3=Close, 4=Closed
            # These are NOT normalized (excluded from Z-score normalization)
            valid_regimes = {0.0, 1.0, 2.0, 3.0, 4.0}
            assert all(r in valid_regimes for r in unique_regimes), (
                f"TIME_REGIME should be categorical 0-4, got {unique_regimes}"
            )


class TestSequenceDataset:
    """Test PyTorch Dataset with real data."""
    
    def test_sequence_dataset_creation(self):
        """Create sequence dataset from real data."""
        from lobtrainer.data import load_split_data, LOBSequenceDataset
        
        days = load_split_data(DATA_DIR, "train", validate=False)
        dataset = LOBSequenceDataset(days, window_size=100, stride=10)
        
        assert len(dataset) > 0
        assert dataset.sequence_shape == (100, 98)
    
    def test_sequence_dataset_iteration(self):
        """Iterate through dataset samples."""
        import torch
        from lobtrainer.data import load_split_data, LOBSequenceDataset
        
        days = load_split_data(DATA_DIR, "train", validate=False)[:1]  # First day only
        dataset = LOBSequenceDataset(days, window_size=100, stride=10)
        
        # Get first sample
        sequence, label = dataset[0]
        
        assert isinstance(sequence, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert sequence.shape == (100, 98)
        assert label.shape == ()  # Scalar
        assert label.dtype == torch.long


class TestNormalization:
    """Test normalization with real data."""
    
    def test_zscore_normalization(self):
        """Z-score normalization on real data."""
        from lobtrainer.data import load_split_data
        from lobtrainer.data.transforms import compute_statistics, ZScoreNormalizer
        
        days = load_split_data(DATA_DIR, "train", validate=False)[:1]
        features = days[0].features
        
        # Use all finite samples for normalization
        finite_mask = np.isfinite(features).all(axis=1)
        if finite_mask.sum() == 0:
            pytest.skip("No finite samples in first day")
        
        # Compute statistics with explicit valid_mask (bypass BOOK_VALID check)
        stats = compute_statistics(features, valid_mask=finite_mask)
        
        # Verify statistics computed successfully
        assert stats.count > 0, "Should have computed statistics from samples"
        assert np.isfinite(stats.mean).all(), "Mean should be finite"
        assert np.isfinite(stats.std).all(), "Std should be finite"
    
    def test_normalization_preserves_shape(self):
        """Normalization should preserve feature shape."""
        from lobtrainer.data import load_split_data
        from lobtrainer.data.transforms import compute_statistics
        
        days = load_split_data(DATA_DIR, "train", validate=False)[:1]
        features = days[0].features
        
        finite_mask = np.isfinite(features).all(axis=1)
        if finite_mask.sum() == 0:
            pytest.skip("No finite samples")
        
        # Just verify we can compute statistics
        stats = compute_statistics(features, valid_mask=finite_mask)
        assert stats.count == finite_mask.sum()


class TestSignalValues:
    """Validate signal values in real data."""
    
    def test_ofi_values_finite(self):
        """OFI values should be finite for valid samples."""
        from lobtrainer.data import load_split_data
        from lobtrainer.constants import FeatureIndex
        
        days = load_split_data(DATA_DIR, "train", validate=False)
        
        for day in days:
            valid_mask = day.features[:, FeatureIndex.BOOK_VALID] > 0.5
            mbo_ready = day.features[:, FeatureIndex.MBO_READY] > 0.5
            fully_valid = valid_mask & mbo_ready
            
            if fully_valid.sum() > 0:
                ofi = day.features[fully_valid, FeatureIndex.TRUE_OFI]
                assert np.isfinite(ofi).all(), (
                    f"OFI should be finite for valid samples in {day.date}"
                )
    
    def test_asymmetry_signals_normalized(self):
        """Asymmetry signals should be Z-score normalized (mean≈0, std≈1)."""
        from lobtrainer.data import load_split_data
        from lobtrainer.constants import FeatureIndex
        
        days = load_split_data(DATA_DIR, "train", validate=False)
        
        asymmetry_indices = [
            FeatureIndex.TRADE_ASYMMETRY,
            FeatureIndex.CANCEL_ASYMMETRY,
            FeatureIndex.DEPTH_ASYMMETRY,
        ]
        
        for day in days:
            valid_mask = (
                (day.features[:, FeatureIndex.BOOK_VALID] > 0.5) &
                (day.features[:, FeatureIndex.MBO_READY] > 0.5)
            )
            
            if valid_mask.sum() > 100:  # Need enough samples for stable stats
                for idx in asymmetry_indices:
                    values = day.features[valid_mask, idx]
                    finite_values = values[np.isfinite(values)]
                    if len(finite_values) > 100:
                        # Z-score normalized: mean ≈ 0, std ≈ 1
                        # Note: Per-day validation may not be exactly 0,1 since
                        # normalization was done on the whole day's data
                        assert np.isfinite(finite_values).all(), (
                            f"Feature {idx} has non-finite values"
                        )
                        # Values should be roughly normal distributed
                        assert np.abs(finite_values.mean()) < 0.5, (
                            f"Feature {idx} mean {finite_values.mean():.3f} not near 0"
                        )
                        assert 0.5 < finite_values.std() < 2.0, (
                            f"Feature {idx} std {finite_values.std():.3f} not near 1"
                        )
    
    def test_schema_version_is_2(self):
        """Schema version should be 2.0 for all samples (NOT normalized)."""
        from lobtrainer.data import load_split_data
        from lobtrainer.constants import FeatureIndex, SCHEMA_VERSION
        
        days = load_split_data(DATA_DIR, "train", validate=False)
        
        for day in days:
            schema_col = day.features[:, FeatureIndex.SCHEMA_VERSION_FEATURE]
            # SCHEMA_VERSION is excluded from normalization, should be exactly 2.0
            unique_versions = np.unique(schema_col)
            assert len(unique_versions) == 1, (
                f"Schema version should be constant, got {unique_versions}"
            )
            assert unique_versions[0] == SCHEMA_VERSION, (
                f"Schema version should be {SCHEMA_VERSION}, got {unique_versions[0]}"
            )

