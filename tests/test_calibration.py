"""
Tests for prediction calibration module.

Validates variance-matching calibration with hand-calculated expected values.
Every formula is verified: IC preservation, variance matching, mean centering.

Reference:
    E5 Comprehensive Statistical Report §1.3, §7.1, §8.1
"""

import numpy as np
import pytest
from scipy.stats import spearmanr

from lobtrainer.calibration.variance import (
    VarianceCalibrationConfig,
    CalibrationResult,
    calibrate_variance,
)


class TestVarianceCalibrationFormula:
    """Verify the calibration formula with hand-calculated values."""

    def test_scale_factor_formula(self):
        """scale_factor = target_std / pred_std.

        For E5: 27.41 / 7.35 = 3.7292...
        """
        target_std = 27.41
        pred_std = 7.35
        expected_scale = target_std / pred_std
        assert abs(expected_scale - 3.7292) < 0.001

    def test_calibration_preserves_ic(self):
        """Spearman IC must be EXACTLY preserved (linear transform)."""
        rng = np.random.RandomState(42)
        n = 1000
        target = rng.randn(n) * 27.41
        predictions = 0.38 * target + rng.randn(n) * 5.0  # IC ~ 0.38

        ic_before = spearmanr(predictions, target)[0]

        result = calibrate_variance(predictions, target)
        ic_after = spearmanr(result.calibrated, target)[0]

        assert abs(ic_before - ic_after) < 1e-10, (
            f"IC changed from {ic_before:.10f} to {ic_after:.10f}. "
            "Linear transform must preserve Spearman rank correlation exactly."
        )

    def test_calibration_matches_target_std(self):
        """Calibrated predictions std must equal target std."""
        rng = np.random.RandomState(42)
        predictions = rng.randn(500) * 7.35 - 0.32  # E5-like predictions
        labels = rng.randn(500) * 27.41 - 0.167      # E5-like labels

        result = calibrate_variance(predictions, labels)

        calibrated_std = float(np.std(result.calibrated))
        label_std = float(np.std(labels))
        assert abs(calibrated_std - label_std) < 0.01, (
            f"Calibrated std={calibrated_std:.4f}, target std={label_std:.4f}. "
            "Variance matching failed."
        )

    def test_calibration_matches_target_mean(self):
        """Calibrated predictions mean must equal target mean."""
        rng = np.random.RandomState(42)
        predictions = rng.randn(500) * 7.35 - 0.32
        labels = rng.randn(500) * 27.41 - 0.167

        result = calibrate_variance(predictions, labels)

        calibrated_mean = float(np.mean(result.calibrated))
        label_mean = float(np.mean(labels))
        assert abs(calibrated_mean - label_mean) < 0.01, (
            f"Calibrated mean={calibrated_mean:.4f}, target mean={label_mean:.4f}. "
            "Mean centering failed."
        )

    def test_known_scale_factor(self):
        """With known std ratio, verify exact scale factor."""
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # std=1.414
        labels = np.array([10.0, 20.0, 30.0, 40.0, 50.0])    # std=14.14

        result = calibrate_variance(predictions, labels)
        expected_scale = float(np.std(labels) / np.std(predictions))

        assert abs(result.scale_factor - expected_scale) < 1e-10, (
            f"Scale factor {result.scale_factor} != expected {expected_scale}"
        )
        assert abs(result.scale_factor - 10.0) < 1e-10, (
            f"Expected scale=10.0 (14.14/1.414), got {result.scale_factor}"
        )


class TestVarianceCalibrationConfig:
    """Test configuration options."""

    def test_default_config_values(self):
        """Default config matches E5 test statistics."""
        config = VarianceCalibrationConfig()
        assert config.method == "variance_match"
        assert config.target_std_bps == 27.41
        assert config.target_mean_bps == -0.167
        assert config.compute_from_labels is True

    def test_none_method_returns_copy(self):
        """method='none' returns identical predictions."""
        predictions = np.array([1.0, 2.0, 3.0])
        config = VarianceCalibrationConfig(method="none")
        result = calibrate_variance(predictions, config=config)

        np.testing.assert_array_equal(result.calibrated, predictions)
        assert result.scale_factor == 1.0

    def test_static_target_when_compute_from_labels_false(self):
        """When compute_from_labels=False, use static config values."""
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        config = VarianceCalibrationConfig(
            compute_from_labels=False,
            target_std_bps=10.0,
            target_mean_bps=5.0,
        )
        result = calibrate_variance(predictions, config=config)

        assert result.target_std == 10.0
        assert result.target_mean == 5.0
        assert abs(float(np.std(result.calibrated)) - 10.0) < 0.01

    def test_compute_from_labels_overrides_static(self):
        """When compute_from_labels=True, labels override static values."""
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        labels = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        config = VarianceCalibrationConfig(
            compute_from_labels=True,
            target_std_bps=1.0,  # This should be overridden
            target_mean_bps=0.0,  # This should be overridden
        )
        result = calibrate_variance(predictions, labels, config)

        assert result.target_std == float(np.std(labels))  # Overridden
        assert result.target_mean == float(np.mean(labels))  # Overridden


class TestVarianceCalibrationEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_predictions_raises(self):
        """Empty predictions array raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            calibrate_variance(np.array([]))

    def test_zero_variance_predictions_raises(self):
        """Constant predictions raise ValueError."""
        with pytest.raises(ValueError, match="near-zero variance"):
            calibrate_variance(np.array([5.0, 5.0, 5.0, 5.0]))

    def test_compute_from_labels_without_labels_raises(self):
        """compute_from_labels=True without labels raises ValueError."""
        config = VarianceCalibrationConfig(compute_from_labels=True)
        with pytest.raises(ValueError, match="labels is None"):
            calibrate_variance(np.array([1.0, 2.0, 3.0]), config=config)

    def test_2d_labels_raises_value_error(self):
        """Phase A (2026-04-23): 2-D labels now raise per hft-rules §8 fail-loud.

        Previously (pre-Phase-A), ``calibrate_variance`` silently collapsed
        multi-horizon ``labels[:, 0]`` — this masked caller-side bugs where
        the caller's ``primary_horizon_idx`` was non-zero, silently
        mis-calibrating every HMHP-R experiment with a non-H0 primary horizon.

        Post-Phase-A, multi-horizon slicing is the CALLER's responsibility
        (see ``SignalExporter._apply_calibration``). Passing 2-D labels
        directly is now a program bug and raises ``ValueError``.
        """
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        labels_2d = np.column_stack([
            np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
            np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        ])
        with pytest.raises(ValueError, match="expects 1-D labels"):
            calibrate_variance(predictions, labels_2d)

    def test_2d_predictions_raises_value_error(self):
        """Phase A (2026-04-23): 2-D predictions also raise per strict 1-D contract."""
        preds_2d = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        labels = np.array([10.0, 20.0, 30.0])
        with pytest.raises(ValueError, match="expects 1-D predictions"):
            calibrate_variance(preds_2d, labels)

    def test_context_kwarg_propagates_to_result(self):
        """Phase A.5.5 (2026-04-24): ``metadata=`` kwarg renamed to ``context=``
        + typed via ``CalibrationContext`` TypedDict. Observability surface
        preserved for multi-method calibrators (quantile, isotonic, conformal).

        Field on ``CalibrationResult`` also renamed: ``metadata`` → ``context``.
        Wire-format JSON key in ``to_dict()`` output PRESERVED as ``"metadata"``
        (plan v4 round-2 refinement — don't mix Python semantic rename with
        wire-format rename).
        """
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        labels = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        provenance = {"primary_horizon_idx": 1, "method_variant": "variance_match"}
        result = calibrate_variance(predictions, labels, context=provenance)
        # Python attribute is ``context``
        assert result.context == provenance
        # Wire-format JSON key is preserved as ``"metadata"``
        assert result.to_dict()["metadata"] == provenance

    def test_context_none_omits_metadata_key_in_to_dict(self):
        """Phase A.5.5: when ``context=None`` (default), ``to_dict()`` must
        NOT emit a ``metadata`` key — preserves pre-A.5.5 wire-format
        (downstream consumers that key-check via ``if "metadata" in d`` see
        no change)."""
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        labels = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        result = calibrate_variance(predictions, labels)  # no context=
        assert result.context is None
        assert "metadata" not in result.to_dict()

    def test_context_to_dict_is_deep_copy_safe(self):
        """Phase A.5.5 bug #4 regression lock: to_dict() shallow copies
        the context TypedDict. Since CalibrationContext contains only
        FLAT primitive-typed fields (by TypedDict schema), shallow copy
        is safe by construction — no nested-dict aliasing hazard.

        Lock: mutating the returned dict's 'metadata' key MUST NOT affect
        the original CalibrationResult.context (no shared reference)."""
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        labels = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        provenance = {"primary_horizon_idx": 2}
        result = calibrate_variance(predictions, labels, context=provenance)
        dumped = result.to_dict()
        # Mutate the dumped dict's metadata
        dumped["metadata"]["primary_horizon_idx"] = 999
        # Original context is UNAFFECTED (shallow copy is safe because
        # TypedDict values are flat primitives)
        assert result.context["primary_horizon_idx"] == 2

    def test_calibration_context_typeddict_import_available(self):
        """Phase A.5.5: CalibrationContext is importable from the
        package surface. Future consumers can explicitly annotate call
        sites for IDE / mypy type-checking."""
        from lobtrainer.calibration import CalibrationContext
        # TypedDict IS a dict subclass at runtime; instantiate as a dict
        ctx: CalibrationContext = {"primary_horizon_idx": 0}
        assert isinstance(ctx, dict)
        assert ctx["primary_horizon_idx"] == 0

    def test_determinism(self):
        """Same inputs produce identical outputs."""
        rng = np.random.RandomState(42)
        predictions = rng.randn(100) * 7.35
        labels = rng.randn(100) * 27.41

        r1 = calibrate_variance(predictions, labels)
        r2 = calibrate_variance(predictions, labels)

        np.testing.assert_array_equal(r1.calibrated, r2.calibrated)
        assert r1.scale_factor == r2.scale_factor


class TestCalibrationResult:
    """Test CalibrationResult serialization."""

    def test_to_dict_serializable(self):
        """to_dict() produces JSON-serializable output."""
        import json
        rng = np.random.RandomState(42)
        predictions = rng.randn(100) * 7.35
        labels = rng.randn(100) * 27.41

        result = calibrate_variance(predictions, labels)
        d = result.to_dict()

        serialized = json.dumps(d)
        assert len(serialized) > 10

    def test_to_dict_has_all_fields(self):
        """to_dict() includes all calibration statistics."""
        rng = np.random.RandomState(42)
        result = calibrate_variance(rng.randn(100) * 7.35, rng.randn(100) * 27.41)
        d = result.to_dict()

        required = [
            "scale_factor", "pred_mean", "pred_std",
            "target_mean", "target_std", "n_samples",
            "calibrated_mean", "calibrated_std",
            "calibrated_min", "calibrated_max",
        ]
        for key in required:
            assert key in d, f"Missing key: {key}"
            assert np.isfinite(d[key]), f"Non-finite value for {key}: {d[key]}"


class TestE5RealisticScenario:
    """Test with E5-realistic prediction and label distributions."""

    def test_e5_scale_factor(self):
        """E5: pred_std=7.35, target_std=27.41 → scale=3.73."""
        rng = np.random.RandomState(42)
        predictions = rng.randn(8337) * 7.35 - 0.32   # E5 test predictions
        labels = rng.randn(8337) * 27.41 - 0.167       # E5 test labels

        result = calibrate_variance(predictions, labels)

        expected_scale = 27.41 / 7.35
        # Scale factor computed from actual data, not config values
        actual_scale = float(np.std(labels)) / float(np.std(predictions))
        assert abs(result.scale_factor - actual_scale) < 0.01

    def test_e5_calibrated_enables_threshold_filtering(self):
        """After calibration, significant fraction of predictions exceed 10 bps."""
        rng = np.random.RandomState(42)
        predictions = rng.randn(8337) * 7.35 - 0.32
        labels = rng.randn(8337) * 27.41 - 0.167

        # Before calibration: few predictions above 10 bps
        above_10_raw = (np.abs(predictions) > 10.0).sum()
        above_10_raw_pct = above_10_raw / len(predictions)

        result = calibrate_variance(predictions, labels)

        # After calibration: many more above 10 bps
        above_10_cal = (np.abs(result.calibrated) > 10.0).sum()
        above_10_cal_pct = above_10_cal / len(result.calibrated)

        assert above_10_cal_pct > above_10_raw_pct, (
            f"Calibration should increase fraction above 10 bps. "
            f"Raw: {above_10_raw_pct:.1%}, Calibrated: {above_10_cal_pct:.1%}"
        )
        # With Gaussian, ~68% within 1 std, so ~32% above 1 std
        # At 10 bps with std=27.4, that's ~10/27.4 = 0.365 std → ~71% above
        assert above_10_cal_pct > 0.5, (
            f"Expected >50% above 10 bps after calibration, got {above_10_cal_pct:.1%}"
        )
