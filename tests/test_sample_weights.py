"""Unit tests for lobtrainer.data.sample_weights (Audit 2026-05-27 Batch 4).

Tests the trainer-side wrapper that resolves horizon from metadata +
LabelsConfig and delegates to hft_metrics.sample_weights. The upstream
formula (de Prado AFML eq. 4.2) is tested in hft-metrics; these tests
cover the RESOLUTION logic and edge cases specific to the trainer.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lobtrainer.data.sample_weights import (
    _get_horizons_from_metadata,
    _resolve_horizon,
    compute_sample_weights_for_day,
)


# ---------------------------------------------------------------------------
# _get_horizons_from_metadata
# ---------------------------------------------------------------------------

class TestGetHorizonsFromMetadata:

    def test_none_metadata(self):
        assert _get_horizons_from_metadata(None) is None

    def test_top_level_horizons(self):
        assert _get_horizons_from_metadata({"horizons": [10, 60, 300]}) == [10, 60, 300]

    def test_labeling_horizons(self):
        meta = {"labeling": {"horizons": [10, 20, 50]}}
        assert _get_horizons_from_metadata(meta) == [10, 20, 50]

    def test_label_config_horizons(self):
        meta = {"label_config": {"horizons": [5, 10]}}
        assert _get_horizons_from_metadata(meta) == [5, 10]

    def test_top_level_takes_priority(self):
        meta = {
            "horizons": [10, 60, 300],
            "labeling": {"horizons": [10, 20, 50]},
        }
        assert _get_horizons_from_metadata(meta) == [10, 60, 300]

    def test_empty_metadata(self):
        assert _get_horizons_from_metadata({}) is None

    def test_labeling_not_dict(self):
        assert _get_horizons_from_metadata({"labeling": "invalid"}) is None


# ---------------------------------------------------------------------------
# _resolve_horizon
# ---------------------------------------------------------------------------

class TestResolveHorizon:

    @staticmethod
    def _make_labels_config(
        primary_horizon_idx=0,
        horizons=None,
        sample_weights="concurrent_overlap",
    ):
        cfg = MagicMock()
        cfg.primary_horizon_idx = primary_horizon_idx
        cfg.horizons = horizons
        cfg.sample_weights = sample_weights
        return cfg

    def test_from_labels_config_horizons(self):
        cfg = self._make_labels_config(primary_horizon_idx=1, horizons=(10, 60, 300))
        assert _resolve_horizon(None, cfg) == 60

    def test_from_labels_config_horizons_idx_0(self):
        cfg = self._make_labels_config(primary_horizon_idx=0, horizons=(10, 60, 300))
        assert _resolve_horizon(None, cfg) == 10

    def test_from_metadata_when_config_horizons_empty(self):
        cfg = self._make_labels_config(primary_horizon_idx=1, horizons=())
        meta = {"horizons": [10, 60, 300]}
        assert _resolve_horizon(meta, cfg) == 60

    def test_from_metadata_when_config_horizons_none(self):
        cfg = self._make_labels_config(primary_horizon_idx=0, horizons=None)
        meta = {"horizons": [10, 60, 300]}
        assert _resolve_horizon(meta, cfg) == 10

    def test_idx_out_of_bounds_returns_none(self):
        cfg = self._make_labels_config(primary_horizon_idx=5, horizons=(10, 60, 300))
        assert _resolve_horizon(None, cfg) is None

    def test_idx_none_hmhp_mode_returns_max(self):
        cfg = self._make_labels_config(primary_horizon_idx=None, horizons=(10, 60, 300))
        assert _resolve_horizon(None, cfg) == 300

    def test_idx_none_no_horizons_returns_none(self):
        cfg = self._make_labels_config(primary_horizon_idx=None, horizons=None)
        assert _resolve_horizon(None, cfg) is None

    def test_idx_none_uses_metadata_max(self):
        cfg = self._make_labels_config(primary_horizon_idx=None, horizons=None)
        meta = {"horizons": [10, 60, 300]}
        assert _resolve_horizon(meta, cfg) == 300

    def test_no_metadata_no_config_horizons(self):
        cfg = self._make_labels_config(primary_horizon_idx=0, horizons=None)
        assert _resolve_horizon(None, cfg) is None


# ---------------------------------------------------------------------------
# compute_sample_weights_for_day
# ---------------------------------------------------------------------------

class TestComputeSampleWeightsForDay:

    @staticmethod
    def _make_labels_config(method="concurrent_overlap", **kw):
        cfg = MagicMock()
        cfg.sample_weights = method
        cfg.primary_horizon_idx = kw.get("primary_horizon_idx", 0)
        cfg.horizons = kw.get("horizons", (10, 60, 300))
        return cfg

    def test_method_none_returns_none(self):
        cfg = self._make_labels_config(method="none")
        assert compute_sample_weights_for_day(100, {}, cfg) is None

    def test_unknown_method_returns_none(self):
        cfg = self._make_labels_config(method="exotic_weighting")
        assert compute_sample_weights_for_day(100, {}, cfg) is None

    def test_zero_samples_returns_none(self):
        cfg = self._make_labels_config()
        assert compute_sample_weights_for_day(0, {}, cfg) is None

    def test_unresolvable_horizon_returns_none(self):
        cfg = self._make_labels_config(horizons=None)
        assert compute_sample_weights_for_day(100, None, cfg) is None

    def test_produces_float64_array(self):
        cfg = self._make_labels_config(horizons=(10,))
        weights = compute_sample_weights_for_day(50, {}, cfg)
        assert weights is not None
        assert weights.dtype == np.float64
        assert weights.shape == (50,)

    def test_mean_approximately_one(self):
        cfg = self._make_labels_config(horizons=(10,))
        weights = compute_sample_weights_for_day(200, {}, cfg)
        assert weights is not None
        assert abs(weights.mean() - 1.0) < 0.01

    def test_single_sample(self):
        cfg = self._make_labels_config(horizons=(10,))
        weights = compute_sample_weights_for_day(1, {}, cfg)
        assert weights is not None
        assert weights.shape == (1,)
        assert np.isfinite(weights).all()

    def test_stride_parameter_passed_through(self):
        cfg = self._make_labels_config(horizons=(10,))
        with patch("hft_metrics.sample_weights.compute_sample_weights") as mock_fn:
            mock_fn.return_value = np.ones(100, dtype=np.float64)
            compute_sample_weights_for_day(100, {}, cfg, stride=10)
            mock_fn.assert_called_once_with(100, 10, 10)
