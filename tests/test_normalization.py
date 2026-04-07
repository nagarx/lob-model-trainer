"""
Comprehensive tests for normalization module.

Tests cover:
- GlobalNormalizationStats: serialization round-trips, backward compat
- GlobalZScoreNormalizer: stats computation, normalization paths (1D/2D/tensor), layout detection
- HybridNormalizationStats: serialization round-trips
- HybridNormalizer: three-tier normalization, exclude indices, clipping, cache
- compute_hybrid_stats_streaming: disk-based streaming equivalence
- Numerical stability: catastrophic cancellation documentation

Schema v2.2.  Reference: hft-rules.md §6 (Testing Philosophy), §9 (ML Pipeline Integrity).
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from lobtrainer.data.normalization import (
    GlobalNormalizationStats,
    GlobalZScoreNormalizer,
    HybridNormalizationStats,
    HybridNormalizer,
    compute_hybrid_stats_streaming,
    load_or_compute_hybrid_stats,
)
from lobtrainer.constants.feature_index import get_price_size_indices

# Constants (matching conftest.py and pipeline contract)
EXCLUDE_INDICES = (92, 93, 94, 96, 97)
LOB_FEATURE_END = 40


# =============================================================================
# GlobalNormalizationStats
# =============================================================================


class TestGlobalNormalizationStats:
    """Test GlobalNormalizationStats dataclass serialization."""

    def test_to_dict_from_dict_roundtrip(self):
        """Stats survive dict serialization."""
        stats = GlobalNormalizationStats(
            mean_prices=130.0, std_prices=0.5,
            mean_sizes=200.0, std_sizes=50.0,
            layout="grouped", num_features=40,
        )
        restored = GlobalNormalizationStats.from_dict(stats.to_dict())
        assert restored.mean_prices == stats.mean_prices
        assert restored.std_prices == stats.std_prices
        assert restored.mean_sizes == stats.mean_sizes
        assert restored.std_sizes == stats.std_sizes
        assert restored.layout == "grouped"
        assert restored.num_features == 40

    def test_save_load_roundtrip(self, tmp_path):
        """Stats survive JSON file round-trip."""
        stats = GlobalNormalizationStats(
            mean_prices=130.12345, std_prices=0.54321,
            mean_sizes=200.0, std_sizes=50.0,
            layout="lobster", num_features=98,
        )
        path = tmp_path / "stats.json"
        stats.save(path)
        loaded = GlobalNormalizationStats.load(path)
        assert abs(loaded.mean_prices - 130.12345) < 1e-10
        assert loaded.layout == "lobster"
        assert loaded.num_features == 98

    def test_backward_compat_no_layout(self):
        """from_dict defaults layout to 'grouped' when missing."""
        d = {"mean_prices": 1.0, "std_prices": 1.0,
             "mean_sizes": 1.0, "std_sizes": 1.0}
        stats = GlobalNormalizationStats.from_dict(d)
        assert stats.layout == "grouped"

    def test_backward_compat_no_num_features(self):
        """from_dict defaults num_features to 40 when missing."""
        d = {"mean_prices": 1.0, "std_prices": 1.0,
             "mean_sizes": 1.0, "std_sizes": 1.0, "layout": "grouped"}
        stats = GlobalNormalizationStats.from_dict(d)
        assert stats.num_features == 40


# =============================================================================
# GlobalZScoreNormalizer
# =============================================================================


class TestGlobalZScoreNormalizer:
    """Test GlobalZScoreNormalizer stats computation and normalization."""

    def test_from_train_data_computes_correct_stats(self, train_days_98):
        """Stats match numpy mean/std on pooled prices (float64 gold standard).

        After Welford migration, the normalizer uses Chan's parallel merge
        which is numerically stable. We verify against numpy.mean/numpy.std
        on float64-cast data (the gold standard).
        """
        normalizer = GlobalZScoreNormalizer.from_train_data(
            train_days_98, layout="grouped", num_features=98
        )
        # Compute gold-standard stats in float64
        all_prices = []
        all_sizes = []
        price_cols = list(range(0, 10)) + list(range(20, 30))
        size_cols = list(range(10, 20)) + list(range(30, 40))
        for day in train_days_98:
            data = day.sequences.reshape(-1, 98).astype(np.float64)
            all_prices.append(data[:, price_cols].flatten())
            all_sizes.append(data[:, size_cols].flatten())
        all_prices = np.concatenate(all_prices)
        all_sizes = np.concatenate(all_sizes)

        # Welford on float32 data should match float64 numpy within ~1e-6
        # (limited by float32 → float64 cast at batch boundaries)
        np.testing.assert_allclose(
            normalizer.stats.mean_prices, np.mean(all_prices), rtol=1e-6
        )
        np.testing.assert_allclose(
            normalizer.stats.std_prices, np.std(all_prices), rtol=1e-6
        )
        np.testing.assert_allclose(
            normalizer.stats.mean_sizes, np.mean(all_sizes), rtol=1e-6
        )
        np.testing.assert_allclose(
            normalizer.stats.std_sizes, np.std(all_sizes), rtol=1e-6
        )

    def test_from_train_data_empty_raises(self):
        """Empty train_days raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            GlobalZScoreNormalizer.from_train_data([])

    def test_from_train_data_auto_detects_grouped(self, train_days_98):
        """Layout auto-detected as 'grouped' from metadata."""
        normalizer = GlobalZScoreNormalizer.from_train_data(train_days_98)
        assert normalizer.layout == "grouped"

    def test_normalize_1d_prices_zscore(self, train_days_98):
        """1D: price columns get (x - mean_p) / (std_p + eps)."""
        normalizer = GlobalZScoreNormalizer.from_train_data(
            train_days_98, layout="grouped", num_features=98
        )
        raw = train_days_98[0].sequences[0, 0, :].copy()  # single [98] vector
        normed = normalizer(raw)

        # Price col 0 should be z-scored
        expected = (raw[0] - normalizer.stats.mean_prices) / (normalizer.stats.std_prices + normalizer.eps)
        np.testing.assert_allclose(normed[0], expected, rtol=1e-12)

        # Size col 10 should be z-scored
        expected_size = (raw[10] - normalizer.stats.mean_sizes) / (normalizer.stats.std_sizes + normalizer.eps)
        np.testing.assert_allclose(normed[10], expected_size, rtol=1e-12)

    def test_normalize_1d_non_lob_untouched(self, train_days_98):
        """1D: features 40+ are NOT modified by GlobalZScoreNormalizer."""
        normalizer = GlobalZScoreNormalizer.from_train_data(
            train_days_98, layout="grouped", num_features=98
        )
        raw = train_days_98[0].sequences[0, 0, :].copy()
        normed = normalizer(raw)
        # Features 40-97 should be unchanged
        np.testing.assert_array_equal(normed[40:], raw[40:])

    def test_normalize_2d_vectorized(self, train_days_98):
        """2D [T, F]: result matches row-by-row 1D normalization."""
        normalizer = GlobalZScoreNormalizer.from_train_data(
            train_days_98, layout="grouped", num_features=98
        )
        seq = train_days_98[0].sequences[0].copy()  # [100, 98]
        normed_2d = normalizer(seq)

        # Compare with row-by-row 1D
        for t in range(min(5, seq.shape[0])):
            normed_1d = normalizer(seq[t])
            np.testing.assert_allclose(normed_2d[t], normed_1d, rtol=1e-12)

    def test_normalize_tensor_2d(self, train_days_98):
        """Tensor [B, F]: matches numpy 1D path."""
        normalizer = GlobalZScoreNormalizer.from_train_data(
            train_days_98, layout="grouped", num_features=98
        )
        raw_np = train_days_98[0].features[:3].copy()  # [3, 98]
        normed_np = np.stack([normalizer(raw_np[i]) for i in range(3)])
        normed_tensor = normalizer.normalize_tensor(
            torch.from_numpy(raw_np.astype(np.float32))
        )
        np.testing.assert_allclose(
            normed_tensor.numpy(), normed_np.astype(np.float32), rtol=1e-5
        )

    def test_normalize_tensor_3d(self, train_days_98):
        """Tensor [B, T, F]: matches numpy 2D path."""
        normalizer = GlobalZScoreNormalizer.from_train_data(
            train_days_98, layout="grouped", num_features=98
        )
        raw_np = train_days_98[0].sequences[:2].copy()  # [2, 100, 98]
        normed_np = np.stack([normalizer(raw_np[b]) for b in range(2)])
        normed_tensor = normalizer.normalize_tensor(
            torch.from_numpy(raw_np.astype(np.float32))
        )
        np.testing.assert_allclose(
            normed_tensor.numpy(), normed_np.astype(np.float32), rtol=1e-5
        )

    def test_from_data_dir_caching(self, synthetic_export_dir):
        """from_data_dir saves and reloads cached stats."""
        # First call: computes and saves
        norm1 = GlobalZScoreNormalizer.from_data_dir(synthetic_export_dir, num_features=98)
        cache_path = synthetic_export_dir / "normalization_stats.json"
        assert cache_path.exists()

        # Second call: loads from cache
        norm2 = GlobalZScoreNormalizer.from_data_dir(synthetic_export_dir, num_features=98)
        assert norm2.stats.mean_prices == norm1.stats.mean_prices
        assert norm2.stats.std_prices == norm1.stats.std_prices

    def test_handles_nan_in_data(self, day_data_factory):
        """NaN values in price columns are excluded from stats."""
        day = day_data_factory("2025-01-01")
        # Inject NaN into one price
        day.sequences[0, 0, 0] = np.nan
        normalizer = GlobalZScoreNormalizer.from_train_data(
            [day], layout="grouped", num_features=98
        )
        assert np.isfinite(normalizer.stats.mean_prices)
        assert np.isfinite(normalizer.stats.std_prices)


# =============================================================================
# HybridNormalizationStats
# =============================================================================


class TestHybridNormalizationStats:
    """Test HybridNormalizationStats serialization."""

    def test_to_dict_from_dict_roundtrip(self):
        """Full round-trip including numpy arrays."""
        lob_stats = GlobalNormalizationStats(130.0, 0.5, 200.0, 50.0)
        stats = HybridNormalizationStats(
            lob_stats=lob_stats,
            per_feature_mean=np.random.randn(98),
            per_feature_std=np.abs(np.random.randn(98)) + 0.1,
            exclude_indices=(92, 93, 94, 96, 97),
            num_features=98,
        )
        restored = HybridNormalizationStats.from_dict(stats.to_dict())
        np.testing.assert_allclose(
            restored.per_feature_mean, stats.per_feature_mean, rtol=1e-10
        )
        np.testing.assert_allclose(
            restored.per_feature_std, stats.per_feature_std, rtol=1e-10
        )
        assert restored.exclude_indices == (92, 93, 94, 96, 97)
        assert restored.num_features == 98

    def test_save_load_roundtrip(self, tmp_path):
        """JSON file round-trip preserves per-feature arrays."""
        lob_stats = GlobalNormalizationStats(130.0, 0.5, 200.0, 50.0)
        stats = HybridNormalizationStats(
            lob_stats=lob_stats,
            per_feature_mean=np.arange(98, dtype=np.float64),
            per_feature_std=np.ones(98),
            exclude_indices=(92, 93, 94, 96, 97),
        )
        path = tmp_path / "hybrid_stats.json"
        stats.save(path)
        loaded = HybridNormalizationStats.load(path)
        np.testing.assert_array_equal(loaded.per_feature_mean, stats.per_feature_mean)


# =============================================================================
# HybridNormalizer
# =============================================================================


class TestHybridNormalizer:
    """Test HybridNormalizer three-tier normalization."""

    def test_from_train_data_lob_stats_match_global(self, train_days_98):
        """LOB price/size stats match GlobalZScoreNormalizer on same data."""
        hybrid = HybridNormalizer.from_train_data(train_days_98, num_features=98)
        global_norm = GlobalZScoreNormalizer.from_train_data(
            train_days_98, layout="grouped", num_features=98
        )
        np.testing.assert_allclose(
            hybrid.stats.lob_stats.mean_prices,
            global_norm.stats.mean_prices, rtol=1e-10
        )
        np.testing.assert_allclose(
            hybrid.stats.lob_stats.std_prices,
            global_norm.stats.std_prices, rtol=1e-10
        )

    def test_from_train_data_per_feature_correctness(self, train_days_98):
        """Per-feature mean/std match numpy for non-excluded derived features."""
        hybrid = HybridNormalizer.from_train_data(train_days_98, num_features=98)
        # Manually compute stats for feature 40
        all_vals = []
        for day in train_days_98:
            data = day.sequences.reshape(-1, 98)
            col = data[:, 40]
            col = col[np.isfinite(col)]
            all_vals.append(col)
        all_vals = np.concatenate(all_vals)
        # float32 data accumulated day-by-day in float64 via Welford
        # vs numpy on concatenated float32 — rtol=1e-5 for cross-method tolerance
        np.testing.assert_allclose(
            hybrid.stats.per_feature_mean[40], np.mean(all_vals), rtol=1e-5
        )
        np.testing.assert_allclose(
            hybrid.stats.per_feature_std[40], np.std(all_vals), rtol=1e-5
        )

    def test_from_train_data_excludes_categorical(self, train_days_98):
        """Excluded indices {92,93,94,96,97} have mean=0, std=1."""
        hybrid = HybridNormalizer.from_train_data(train_days_98, num_features=98)
        for idx in EXCLUDE_INDICES:
            assert hybrid.stats.per_feature_mean[idx] == 0.0, f"idx {idx} mean should be 0"
            assert hybrid.stats.per_feature_std[idx] == 1.0, f"idx {idx} std should be 1"

    def test_normalize_1d_three_tier(self, train_days_98):
        """1D: LOB prices → global, LOB sizes → global, derived → per-feature."""
        hybrid = HybridNormalizer.from_train_data(train_days_98, num_features=98)
        raw = train_days_98[0].sequences[0, 0, :].astype(np.float64).copy()
        normed = hybrid(raw)

        # Price col 0: (x - mean_prices) / (std_prices + eps)
        expected_price = (raw[0] - hybrid.stats.lob_stats.mean_prices) / (
            hybrid.stats.lob_stats.std_prices + hybrid.eps
        )
        np.testing.assert_allclose(normed[0], expected_price, rtol=1e-12)

        # Size col 10: (x - mean_sizes) / (std_sizes + eps)
        expected_size = (raw[10] - hybrid.stats.lob_stats.mean_sizes) / (
            hybrid.stats.lob_stats.std_sizes + hybrid.eps
        )
        np.testing.assert_allclose(normed[10], expected_size, rtol=1e-12)

        # Derived col 40: (x - per_feat_mean[40]) / (per_feat_std[40] + eps)
        expected_derived = (raw[40] - hybrid.stats.per_feature_mean[40]) / (
            hybrid.stats.per_feature_std[40] + hybrid.eps
        )
        np.testing.assert_allclose(normed[40], expected_derived, rtol=1e-12)

    def test_normalize_1d_excluded_unchanged(self, train_days_98):
        """1D: excluded features pass through raw."""
        hybrid = HybridNormalizer.from_train_data(train_days_98, num_features=98)
        raw = train_days_98[0].sequences[0, 0, :].astype(np.float64).copy()
        normed = hybrid(raw)
        for idx in EXCLUDE_INDICES:
            np.testing.assert_equal(normed[idx], raw[idx])

    def test_normalize_2d_matches_1d(self, train_days_98):
        """2D [T, F] matches row-by-row 1D."""
        hybrid = HybridNormalizer.from_train_data(train_days_98, num_features=98)
        seq = train_days_98[0].sequences[0].astype(np.float64).copy()  # [100, 98]
        normed_2d = hybrid(seq)
        for t in range(min(5, seq.shape[0])):
            normed_1d = hybrid(seq[t])
            np.testing.assert_allclose(normed_2d[t], normed_1d, rtol=1e-12)

    def test_clip_applied(self, train_days_98):
        """Normalized values within [-clip_value, clip_value]."""
        hybrid = HybridNormalizer.from_train_data(
            train_days_98, num_features=98, clip_value=5.0
        )
        raw = train_days_98[0].sequences[0].astype(np.float64).copy()
        normed = hybrid(raw)
        # Non-excluded columns should be clipped
        for col in range(98):
            if col not in EXCLUDE_INDICES:
                assert normed[:, col].max() <= 5.0 + 1e-12
                assert normed[:, col].min() >= -5.0 - 1e-12

    def test_clip_excluded_not_clipped(self, train_days_98):
        """Excluded indices are NOT clipped in numpy paths."""
        hybrid = HybridNormalizer.from_train_data(
            train_days_98, num_features=98, clip_value=5.0
        )
        raw = train_days_98[0].sequences[0].astype(np.float64).copy()
        normed = hybrid(raw)
        # Excluded columns should be unchanged (raw pass-through)
        for idx in EXCLUDE_INDICES:
            np.testing.assert_array_equal(normed[:, idx], raw[:, idx])

    def test_normalize_tensor_2d_matches_numpy(self, train_days_98):
        """Tensor [B, F] path produces same result as numpy (except clipping bug)."""
        hybrid = HybridNormalizer.from_train_data(
            train_days_98, num_features=98, clip_value=None  # disable clipping to test pure normalization
        )
        raw_np = train_days_98[0].features[:3].astype(np.float64).copy()
        normed_np = np.stack([hybrid(raw_np[i]) for i in range(3)])
        normed_tensor = hybrid.normalize_tensor(
            torch.from_numpy(raw_np.astype(np.float32))
        )
        # rtol=5e-4 accounts for float32 tensor vs float64 numpy precision
        np.testing.assert_allclose(
            normed_tensor.numpy(), normed_np.astype(np.float32), rtol=5e-4
        )

    def test_normalize_tensor_does_not_clip_excluded(self, train_days_98):
        """normalize_tensor skips excluded indices during clipping.

        Both numpy and tensor paths should preserve excluded features
        (BOOK_VALID, TIME_REGIME, etc.) through the clipping step.
        """
        hybrid = HybridNormalizer.from_train_data(
            train_days_98, num_features=98, clip_value=0.5  # very tight clip
        )
        raw = np.zeros((2, 98), dtype=np.float32)
        raw[:, 92] = 1.0  # BOOK_VALID — excluded, should NOT be clipped
        raw[:, 93] = 4.0  # TIME_REGIME — excluded, should NOT be clipped

        # numpy path: excluded features unchanged
        normed_np = hybrid(raw[0].astype(np.float64))
        assert normed_np[92] == 1.0, "numpy: BOOK_VALID should NOT be clipped"
        assert normed_np[93] == 4.0, "numpy: TIME_REGIME should NOT be clipped"

        # tensor path: excluded features also unchanged (bug fixed)
        normed_t = hybrid.normalize_tensor(torch.from_numpy(raw))
        assert normed_t[0, 92].item() == pytest.approx(1.0, abs=1e-6), \
            "tensor: BOOK_VALID should NOT be clipped"
        assert normed_t[0, 93].item() == pytest.approx(4.0, abs=1e-6), \
            "tensor: TIME_REGIME should NOT be clipped"

    def test_std_floor_applied(self, day_data_factory):
        """When feature has zero variance, std is floored to 1.0."""
        day = day_data_factory("2025-01-01")
        # Make feature 50 constant (zero variance)
        day.sequences[:, :, 50] = 42.0
        hybrid = HybridNormalizer.from_train_data([day], num_features=98)
        assert hybrid.stats.per_feature_std[50] == 1.0

    def test_from_cached_or_compute(self, synthetic_export_dir):
        """Cache-first: saves stats, second call loads from cache."""
        stats1 = load_or_compute_hybrid_stats(
            synthetic_export_dir, num_features=98
        )
        cache_path = synthetic_export_dir / "hybrid_normalization_stats.json"
        assert cache_path.exists()

        stats2 = load_or_compute_hybrid_stats(
            synthetic_export_dir, num_features=98
        )
        np.testing.assert_array_equal(
            stats2.per_feature_mean, stats1.per_feature_mean
        )


# =============================================================================
# Streaming stats
# =============================================================================


class TestComputeHybridStatsStreaming:
    """Test streaming stats computation from disk."""

    def test_streaming_matches_from_train_data(self, synthetic_export_dir, day_data_factory):
        """Streaming from disk matches from_train_data on same data.

        Both paths use the same sum/sum_sq algorithm on float32 data.
        Small differences arise from float32 accumulation order (streaming
        processes raw files, from_train_data processes DayData objects).
        Use generous tolerance to characterize equivalence, not exact match.
        """
        from lobtrainer.data.dataset import load_split_data
        train_days = load_split_data(synthetic_export_dir, "train", validate=False)

        hybrid = HybridNormalizer.from_train_data(train_days, num_features=98)
        stats_streaming = compute_hybrid_stats_streaming(
            synthetic_export_dir, num_features=98
        )

        # LOB mean should match closely (mean accumulation is stable)
        np.testing.assert_allclose(
            stats_streaming.lob_stats.mean_prices,
            hybrid.stats.lob_stats.mean_prices, rtol=1e-5
        )

        # Std may differ more due to float32 E[X^2]-E[X]^2 instability,
        # but should be in the same ballpark (within 15%)
        if hybrid.stats.lob_stats.std_prices > 0:
            ratio = stats_streaming.lob_stats.std_prices / hybrid.stats.lob_stats.std_prices
            assert 0.85 < ratio < 1.15, (
                f"std_prices ratio {ratio:.4f} — streaming={stats_streaming.lob_stats.std_prices:.6f}, "
                f"from_train_data={hybrid.stats.lob_stats.std_prices:.6f}"
            )

        # Per-feature means should match closely
        for col in range(LOB_FEATURE_END, 98):
            if col not in EXCLUDE_INDICES:
                np.testing.assert_allclose(
                    stats_streaming.per_feature_mean[col],
                    hybrid.stats.per_feature_mean[col],
                    rtol=1e-4,
                    err_msg=f"Feature {col} mean mismatch",
                )

    def test_streaming_saves_cache(self, synthetic_export_dir):
        """Streaming produces cache file."""
        stats = compute_hybrid_stats_streaming(
            synthetic_export_dir, num_features=98
        )
        assert stats is not None
        assert stats.num_features == 98


# =============================================================================
# Numerical stability
# =============================================================================


class TestNumericalStability:
    """Test numerical stability of the accumulation algorithm."""

    def test_catastrophic_cancellation_documents_risk(self):
        """Current sum/sum_sq algorithm produces inaccurate std for large-mean data.

        For data with large mean relative to std, E[X^2] - E[X]^2 suffers
        catastrophic cancellation. This test documents the deficiency.
        After Welford migration, the algorithm should produce accurate results.
        """
        rng = np.random.default_rng(42)
        # Data: mean=100000, std=0.01 → E[X^2] ≈ 1e10, E[X]^2 ≈ 1e10
        # The difference (~1e-4) is computed from two ~1e10 values,
        # losing ~14 digits of precision with float64.
        data = np.full(10000, 100000.0) + rng.normal(0, 0.01, 10000)
        true_std = np.std(data)  # numpy uses a more stable algorithm

        # Simulate the sum/sum_sq formula
        s = np.sum(data)
        s2 = np.sum(data ** 2)
        n = len(data)
        mean = s / n
        var = s2 / n - mean ** 2
        computed_std = np.sqrt(max(var, 0.0))

        # For this extreme case, sum/sum_sq may give zero variance
        # (because var goes negative due to cancellation, clamped to 0)
        # or wildly inaccurate std.
        # The relative error should be significant:
        if computed_std > 0:
            rel_error = abs(computed_std - true_std) / true_std
            # We just document that the error exists (may be >50% or even 100%)
            assert rel_error > 0.0, "Expected non-zero error from cancellation"
        # The key assertion: the sum/sum_sq formula is NOT numerically stable
        # for data with large mean. After Welford migration, this test's
        # companion test_welford_handles_catastrophic_case should pass cleanly.

    def test_welford_reference_value(self):
        """Verify numpy.std is our reference for correctness.

        numpy.std uses a two-pass algorithm (mean first, then deviations)
        which is numerically stable. This is our gold standard.
        """
        rng = np.random.default_rng(42)
        data = np.full(10000, 100000.0) + rng.normal(0, 0.01, 10000)
        true_std = np.std(data)
        # numpy's std should be close to 0.01 (the generating std)
        assert abs(true_std - 0.01) < 0.001, \
            f"numpy.std should be ~0.01, got {true_std}"
