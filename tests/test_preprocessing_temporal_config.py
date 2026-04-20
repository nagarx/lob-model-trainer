"""Phase IV (2026-04-20) — tests for the domain-layer temporal-config factory.

Locks the binding from ``hft_contracts`` enums → ``hft_metrics.TemporalFeatureConfig``
field values. The factory must produce configs that pass ``TemporalFeatureConfig``
construction-time validation AND engineer_features end-to-end on plausible fixtures.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("hft_metrics")
pytest.importorskip("hft_contracts")
from hft_metrics import engineer_features, TemporalFeatureConfig

from lobtrainer.data.preprocessing import for_basic_pipeline, for_mbo_lob


class TestMBOFactory:
    def test_returns_temporal_feature_config(self):
        cfg = for_mbo_lob()
        assert isinstance(cfg, TemporalFeatureConfig)

    def test_mbo_defaults_match_bare_constructor(self):
        """for_mbo_lob() bit-equivalent to TemporalFeatureConfig() — both produce the same dict."""
        a = for_mbo_lob().to_dict()
        b = TemporalFeatureConfig().to_dict()
        assert a == b, f"Factory config diverges from default:\n  factory: {a}\n  default: {b}"

    def test_mbo_runs_on_F128_fixture(self):
        """Factory config runs engineer_features cleanly on MBO-scaled data."""
        seq = np.random.RandomState(7).standard_normal((2, 30, 128)).astype(np.float64)
        seq[:, :, 40] = np.abs(seq[:, :, 40]) + 100.0  # mid-price positive
        out = engineer_features(seq, for_mbo_lob())
        assert out.shape == (2, 53)
        assert np.all(np.isfinite(out))


class TestBasicPipelineFactory:
    def test_returns_config_with_regime_disabled(self):
        # SB-4 (Phase II hardening, 2026-04-20): `signal_indices` + `mid_price_idx`
        # are REQUIRED. Prior silent defaults [0] produced self-correlated garbage.
        cfg = for_basic_pipeline(signal_indices=[5, 12, 18], mid_price_idx=0)
        assert cfg.include_regime is False
        assert cfg.time_regime_idx is None
        assert cfg.dt_seconds_idx is None

    def test_runs_on_F34_fixture_no_crash(self):
        """BASIC F=34 data must flow through engineer_features without IndexError (FRESH-1 fix).

        SB-4: signal_indices now required — test supplies explicit picks.
        """
        seq = np.random.RandomState(11).standard_normal((2, 20, 34)).astype(np.float64)
        seq[:, :, 0] = np.abs(seq[:, :, 0]) + 100.0  # mid_price-like
        cfg = for_basic_pipeline(signal_indices=[5, 12, 18], mid_price_idx=0)
        out = engineer_features(seq, cfg)
        assert out.shape[0] == 2
        assert np.all(np.isfinite(out))

    def test_runs_with_overrides(self):
        """Caller can override signals/context/cross_pairs for custom BASIC schemas."""
        seq = np.random.RandomState(11).standard_normal((2, 20, 34)).astype(np.float64)
        seq[:, :, 0] = np.abs(seq[:, :, 0]) + 100.0
        cfg = for_basic_pipeline(
            signal_indices=[5, 12, 18],
            top_k_signals=2,
            context_indices=[0, 1, 2],
            mid_price_idx=0,
        )
        out = engineer_features(seq, cfg)
        assert out.shape[0] == 2
        assert np.all(np.isfinite(out))

    def test_out_of_bounds_signal_override_still_raises(self):
        """Factory's defaults are safe; caller overrides still bounds-checked."""
        seq = np.random.RandomState(11).standard_normal((2, 20, 34)).astype(np.float64)
        cfg = for_basic_pipeline(signal_indices=[99], mid_price_idx=0)
        with pytest.raises(ValueError, match="out of bounds"):
            engineer_features(seq, cfg)

    # --- Phase II hardening SB-4 (2026-04-20): fail-fast on missing mandatory args ---

    def test_missing_signal_indices_raises(self):
        """Prior silent default [0] produced self-correlated garbage — now fail-fast."""
        with pytest.raises(ValueError, match="signal_indices is required"):
            for_basic_pipeline()

    def test_missing_mid_price_idx_raises(self):
        """mid_price_idx is required for realized-vol feature — no silent default."""
        with pytest.raises(ValueError, match="mid_price_idx is required"):
            for_basic_pipeline(signal_indices=[5, 12, 18])

    def test_fail_fast_error_message_guides_user(self):
        """Error message must point to the domain-layer enum for semantic construction."""
        with pytest.raises(ValueError) as excinfo:
            for_basic_pipeline()
        msg = str(excinfo.value)
        assert "OffExchangeFeatureIndex" in msg or "signal_indices=[5, 12, 18]" in msg, (
            f"Error must guide user to explicit construction, got: {msg}"
        )

    def test_empty_context_indices_default_does_not_emit_garbage(self):
        """Default context_indices is now [] — no silent garbage context features."""
        seq = np.random.RandomState(11).standard_normal((2, 20, 34)).astype(np.float64)
        seq[:, :, 0] = np.abs(seq[:, :, 0]) + 100.0
        cfg = for_basic_pipeline(signal_indices=[5, 12, 18], mid_price_idx=0)
        assert cfg.context_indices == []
        out = engineer_features(seq, cfg)
        assert out.shape[0] == 2
        assert np.all(np.isfinite(out))
