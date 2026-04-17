"""Tests for T12 multi-source fusion: sources.py + bundle.py.

Covers: DataSource, SourceDay, normalize_date, DayBundle, _align_sources,
to_fused_day_data, load_split_bundles. Uses both synthetic fixtures
and real NVDA E5 60s + BASIC 60s data.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from lobtrainer.data.sources import DataSource, SourceDay, normalize_date


# =============================================================================
# DataSource
# =============================================================================


class TestDataSource:
    def test_valid_primary(self):
        ds = DataSource(name="mbo", data_dir="/tmp", role="primary")
        assert ds.name == "mbo"
        assert ds.role == "primary"

    def test_valid_auxiliary(self):
        ds = DataSource(name="basic", data_dir="/tmp", role="auxiliary")
        assert ds.role == "auxiliary"

    def test_invalid_role_raises(self):
        with pytest.raises(ValueError, match="primary.*auxiliary"):
            DataSource(name="x", data_dir="/tmp", role="invalid")

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name"):
            DataSource(name="", data_dir="/tmp")

    def test_empty_data_dir_raises(self):
        with pytest.raises(ValueError, match="data_dir"):
            DataSource(name="x", data_dir="")


# =============================================================================
# SourceDay
# =============================================================================


class TestSourceDay:
    def test_auto_detect_shape(self):
        seq = np.random.randn(10, 20, 34).astype(np.float64)
        sd = SourceDay(name="basic", date="20250203", sequences=seq, features=seq[:, -1, :])
        assert sd.n_features == 34
        assert sd.window_size == 20
        assert sd.n_sequences == 10

    def test_explicit_shape_override(self):
        seq = np.random.randn(5, 15, 50).astype(np.float64)
        sd = SourceDay(name="x", date="20250203", sequences=seq, features=seq[:, -1, :],
                       n_features=50, window_size=15)
        assert sd.n_features == 50
        assert sd.window_size == 15


# =============================================================================
# normalize_date
# =============================================================================


class TestNormalizeDate:
    def test_hyphenated(self):
        assert normalize_date("2025-02-03") == "20250203"

    def test_compact(self):
        assert normalize_date("20250203") == "20250203"

    def test_no_hyphens_passthrough(self):
        assert normalize_date("20251231") == "20251231"


# =============================================================================
# _align_sources
# =============================================================================


class TestAlignSources:
    def _make_source_day(self, name, n, f=34, t=20):
        seq = np.random.randn(n, t, f).astype(np.float64)
        return SourceDay(name=name, date="20250203", sequences=seq, features=seq[:, -1, :])

    def test_equal_n_no_trim(self):
        from lobtrainer.data.bundle import _align_sources
        pri = self._make_source_day("mbo", 100, f=98)
        aux = self._make_source_day("basic", 100, f=34)
        aligned = _align_sources(pri, aux)
        assert aligned[0].n_sequences == 100
        assert aligned[1].n_sequences == 100

    def test_primary_smaller_trims_auxiliary(self):
        from lobtrainer.data.bundle import _align_sources
        pri = self._make_source_day("mbo", 50, f=98)
        aux = self._make_source_day("basic", 80, f=34)
        aligned = _align_sources(pri, aux)
        assert aligned[0].n_sequences == 50
        assert aligned[1].n_sequences == 50

    def test_auxiliary_smaller_trims_primary(self):
        from lobtrainer.data.bundle import _align_sources
        pri = self._make_source_day("mbo", 80, f=98)
        aux = self._make_source_day("basic", 50, f=34)
        aligned = _align_sources(pri, aux)
        assert aligned[0].n_sequences == 50
        assert aligned[1].n_sequences == 50

    def test_zero_sequences_raises(self):
        from lobtrainer.data.bundle import _align_sources
        pri = self._make_source_day("mbo", 0, f=98)
        aux = self._make_source_day("basic", 50, f=34)
        with pytest.raises(ValueError, match="0 sequences"):
            _align_sources(pri, aux)


# =============================================================================
# DayBundle.to_fused_day_data
# =============================================================================


class TestToFusedDayData:
    def _make_bundle(self, n=50, f_pri=98, f_aux=34, t=20):
        from lobtrainer.data.bundle import DayBundle
        pri_seq = np.random.randn(n, t, f_pri).astype(np.float64)
        aux_seq = np.random.randn(n, t, f_aux).astype(np.float64)
        pri = SourceDay(name="mbo", date="20250203", sequences=pri_seq, features=pri_seq[:, -1, :])
        aux = SourceDay(name="basic", date="20250203", sequences=aux_seq, features=aux_seq[:, -1, :])
        labels = np.zeros(n, dtype=np.int64)
        return DayBundle(
            date="20250203", primary_source="mbo",
            sources={"mbo": pri, "basic": aux},
            labels=labels,
        )

    def test_fused_shape(self):
        bundle = self._make_bundle(n=50, f_pri=98, f_aux=34)
        day = bundle.to_fused_day_data()
        assert day.sequences.shape == (50, 20, 132)
        assert day.features.shape == (50, 132)

    def test_primary_first_in_default_order(self):
        bundle = self._make_bundle(n=10, f_pri=3, f_aux=2)
        day = bundle.to_fused_day_data()
        # First 3 features = primary, last 2 = auxiliary
        np.testing.assert_array_equal(
            day.sequences[:, :, :3],
            bundle.sources["mbo"].sequences,
        )
        np.testing.assert_array_equal(
            day.sequences[:, :, 3:],
            bundle.sources["basic"].sequences,
        )

    def test_custom_source_order(self):
        bundle = self._make_bundle(n=10, f_pri=3, f_aux=2)
        day = bundle.to_fused_day_data(source_order=["basic", "mbo"])
        # Now basic first
        np.testing.assert_array_equal(
            day.sequences[:, :, :2],
            bundle.sources["basic"].sequences,
        )

    def test_feature_indices_selection(self):
        bundle = self._make_bundle(n=10, f_pri=5, f_aux=3)
        day = bundle.to_fused_day_data(
            feature_indices={"mbo": [0, 2], "basic": [1]},
        )
        assert day.sequences.shape == (10, 20, 3)  # 2 from mbo + 1 from basic

    def test_mismatched_window_size_raises(self):
        from lobtrainer.data.bundle import DayBundle
        pri_seq = np.random.randn(10, 20, 5).astype(np.float64)
        aux_seq = np.random.randn(10, 100, 3).astype(np.float64)  # different T
        pri = SourceDay(name="mbo", date="20250203", sequences=pri_seq, features=pri_seq[:, -1, :])
        aux = SourceDay(name="basic", date="20250203", sequences=aux_seq, features=aux_seq[:, -1, :])
        bundle = DayBundle(
            date="20250203", primary_source="mbo",
            sources={"mbo": pri, "basic": aux},
            labels=np.zeros(10, dtype=np.int64),
        )
        with pytest.raises(ValueError, match="window sizes"):
            bundle.to_fused_day_data()

    def test_missing_source_raises(self):
        bundle = self._make_bundle()
        with pytest.raises(KeyError, match="nonexistent"):
            bundle.to_fused_day_data(source_order=["mbo", "nonexistent"])

    def test_labels_preserved(self):
        bundle = self._make_bundle(n=10)
        bundle.labels = np.array([1, 0, -1, 1, 0, -1, 1, 0, -1, 0], dtype=np.int64)
        day = bundle.to_fused_day_data()
        np.testing.assert_array_equal(day.labels, bundle.labels)


# =============================================================================
# Real NVDA data integration (T12 end-to-end)
# =============================================================================


class TestRealDataFusion:
    """Integration tests using real E5 60s + BASIC 60s exports."""

    @pytest.fixture
    def real_sources(self):
        mbo_dir = Path("../data/exports/e5_timebased_60s")
        basic_dir = Path("../data/exports/basic_nvda_60s")
        if not mbo_dir.exists() or not basic_dir.exists():
            pytest.skip("Real export data not available")
        return [
            DataSource(name="mbo", data_dir=str(mbo_dir), role="primary"),
            DataSource(name="basic", data_dir=str(basic_dir), role="auxiliary"),
        ]

    def test_load_split_bundles_real_data(self, real_sources):
        """Load 1 real fused day and verify shapes."""
        from lobtrainer.config.schema import LabelsConfig
        from lobtrainer.data.bundle import load_split_bundles

        lc = LabelsConfig(
            source="forward_prices", task="regression",
            horizons=[10], primary_horizon_idx=0,
        )
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            days = load_split_bundles(real_sources, "train", labels_config=lc)[:1]

        assert len(days) == 1
        day = days[0]
        assert day.sequences.shape[2] == 132  # 98 MBO + 34 BASIC
        assert day.sequences.shape[1] == 20   # both use window_size=20
        assert day.features.shape[1] == 132
        assert day.labels is not None
        assert day.is_aligned

    def test_alignment_preserves_primary_n(self, real_sources):
        """Fused N should equal primary source's N (MBO has fewer)."""
        from lobtrainer.config.schema import LabelsConfig
        from lobtrainer.data.bundle import load_split_bundles
        from lobtrainer.data.dataset import load_split_data

        lc = LabelsConfig(source="forward_prices", task="regression", horizons=[10])
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fused = load_split_bundles(real_sources, "train", labels_config=lc)[:1]
            mbo_only = load_split_data(str(real_sources[0].data_dir), "train", labels_config=lc)[:1]

        assert fused[0].sequences.shape[0] == mbo_only[0].sequences.shape[0]
