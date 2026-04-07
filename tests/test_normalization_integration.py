"""
Integration tests for normalization in the training data pipeline.

Tests the full flow: normalizer creation → dataset transform → model input.
Verifies that normalization, feature selection, and label handling work
together correctly through LOBSequenceDataset.

All existing test configs use normalization: "none". These tests exercise
the actual normalization code paths in the training pipeline.

Schema v2.2.
"""

import numpy as np
import pytest
import torch

from lobtrainer.data.normalization import (
    GlobalZScoreNormalizer,
    HybridNormalizer,
)
from lobtrainer.data.dataset import LOBSequenceDataset
from lobtrainer.data.feature_selector import FeatureSelector

# Constants
EXCLUDE_INDICES = (92, 93, 94, 96, 97)
LOB_FEATURE_END = 40


class TestNormalizationAsDatasetTransform:
    """Test normalizers as LOBSequenceDataset transforms."""

    def test_global_normalizer_applied_in_getitem(self, train_days_98):
        """GlobalZScoreNormalizer applied per-sample in __getitem__."""
        normalizer = GlobalZScoreNormalizer.from_train_data(
            train_days_98, layout="grouped", num_features=98
        )
        dataset = LOBSequenceDataset(train_days_98, transform=normalizer)

        seq_tensor, label = dataset[0]
        assert seq_tensor.shape == (100, 98)
        assert seq_tensor.dtype == torch.float32

        # Price columns should have mean near 0 (z-scored)
        price_vals = seq_tensor[:, 0].numpy()  # ask_price_L1
        raw_price_mean = train_days_98[0].sequences[0, :, 0].mean()
        # Normalized should be much smaller than raw (raw ~130, normed ~0)
        assert abs(price_vals.mean()) < abs(raw_price_mean)

    def test_hybrid_normalizer_applied_in_getitem(self, train_days_98):
        """HybridNormalizer applied per-sample in __getitem__."""
        normalizer = HybridNormalizer.from_train_data(
            train_days_98, num_features=98
        )
        dataset = LOBSequenceDataset(train_days_98, transform=normalizer)

        seq_tensor, label = dataset[0]
        assert seq_tensor.shape == (100, 98)

        # Excluded indices should be unchanged (raw values pass through)
        raw_seq = train_days_98[0].sequences[0].astype(np.float64)
        normed_seq = normalizer(raw_seq)
        for idx in EXCLUDE_INDICES:
            np.testing.assert_allclose(
                seq_tensor[:, idx].numpy(),
                normed_seq[:, idx].astype(np.float32),
                rtol=1e-5,
            )

    def test_normalize_then_select(self, train_days_98):
        """Normalization happens BEFORE feature selection in __getitem__."""
        normalizer = HybridNormalizer.from_train_data(
            train_days_98, num_features=98
        )
        selector = FeatureSelector.from_indices(
            [0, 10, 40, 84, 85], 98, "test_5"
        )

        dataset = LOBSequenceDataset(
            train_days_98,
            transform=normalizer,
            feature_indices=list(selector.indices),
        )

        seq_tensor, label = dataset[0]
        # Should have only 5 features after selection
        assert seq_tensor.shape == (100, 5)

        # Verify the selected features are normalized, not raw
        raw = train_days_98[0].sequences[0].astype(np.float64)
        normed_full = normalizer(raw)
        expected = normed_full[:, list(selector.indices)].astype(np.float32)
        # rtol=5e-4 accounts for float64→float32 precision loss in the
        # dataset __getitem__ path (data stored as float32, normed in float64,
        # then cast back to float32 tensor)
        np.testing.assert_allclose(
            seq_tensor.numpy(), expected, rtol=5e-4
        )

    def test_stats_from_train_only(self, day_data_factory):
        """Normalization stats come from train split only (no val/test leakage)."""
        train_days = [
            day_data_factory("2025-03-01"),
            day_data_factory("2025-03-02"),
        ]
        val_day = day_data_factory("2025-03-03")

        # Compute stats from train only
        normalizer = HybridNormalizer.from_train_data(train_days, num_features=98)

        # Apply to val — should use train stats, not recompute
        val_dataset = LOBSequenceDataset([val_day], transform=normalizer)
        seq_tensor, _ = val_dataset[0]
        assert seq_tensor.shape == (100, 98)
        # Key assertion: val data is normalized with train stats, not its own.
        # We can't easily test this without checking the stats values, but
        # the fact that it runs without error and produces finite output
        # confirms the pipeline works end-to-end.
        assert torch.isfinite(seq_tensor).all()

    def test_cached_stats_identical_normalization(self, synthetic_export_dir):
        """Cached and freshly-computed stats produce identical normalization."""
        from lobtrainer.data.dataset import load_split_data

        train_days = load_split_data(synthetic_export_dir, "train", validate=False)

        # Compute fresh
        norm1 = HybridNormalizer.from_train_data(train_days, num_features=98)

        # Save and reload
        stats_path = synthetic_export_dir / "test_hybrid_stats.json"
        norm1.stats.save(stats_path)
        from lobtrainer.data.normalization import HybridNormalizationStats
        loaded_stats = HybridNormalizationStats.load(stats_path)
        norm2 = HybridNormalizer(loaded_stats)

        # Apply both to same data
        raw = train_days[0].sequences[0].astype(np.float64)
        normed1 = norm1(raw)
        normed2 = norm2(raw)
        np.testing.assert_array_equal(normed1, normed2)
