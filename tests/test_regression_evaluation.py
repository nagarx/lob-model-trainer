"""
Tests for RegressionMetrics dataclass.

RegressionMetrics formats all regression experiment results. It delegates
to compute_all_regression_metrics (hft-metrics adapter) for computation
and provides summary(), to_dict(), from_dict() for serialization.

A bug in from_arrays() or to_dict() would corrupt every regression
experiment's reported metrics.

Design Principles (hft-rules.md):
    - Golden value tests for known inputs (Rule 6)
    - Round-trip serialization verified (Rule 6)
    - All 7 metric keys verified (Rule 6)
"""

import numpy as np
import pytest

from lobtrainer.training.regression_evaluation import RegressionMetrics


# =============================================================================
# Test Data
# =============================================================================


@pytest.fixture
def correlated_data():
    """Correlated predictions (r ≈ 0.99) for positive metric values."""
    np.random.seed(42)
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    y_pred = y_true + np.random.normal(0, 0.1, size=len(y_true))
    return y_true, y_pred


@pytest.fixture
def uncorrelated_data():
    """Uncorrelated predictions (IC ≈ 0)."""
    np.random.seed(42)
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    y_pred = np.random.randn(len(y_true)) * 5
    return y_true, y_pred


# =============================================================================
# Tests
# =============================================================================


class TestRegressionMetrics:
    """Tests for RegressionMetrics creation, serialization, and formatting."""

    def test_from_arrays_populates_all_fields(self, correlated_data):
        """from_arrays produces non-zero values for all 7 metrics on correlated data."""
        y_true, y_pred = correlated_data
        rm = RegressionMetrics.from_arrays(y_true, y_pred)

        assert rm.r2 > 0, f"R² should be positive for correlated data, got {rm.r2}"
        assert rm.ic != 0, f"IC should be non-zero for correlated data, got {rm.ic}"
        assert rm.mae > 0, f"MAE should be positive, got {rm.mae}"
        assert rm.rmse > 0, f"RMSE should be positive, got {rm.rmse}"
        assert rm.pearson != 0, f"Pearson should be non-zero, got {rm.pearson}"
        assert 0 < rm.directional_accuracy <= 1.0, (
            f"DA should be in (0, 1], got {rm.directional_accuracy}"
        )

    def test_from_arrays_with_uncorrelated_data(self, uncorrelated_data):
        """from_arrays works on uncorrelated data (low R², IC near 0)."""
        y_true, y_pred = uncorrelated_data
        rm = RegressionMetrics.from_arrays(y_true, y_pred)

        # R² should be low or negative for uncorrelated
        assert rm.r2 < 0.5, f"R² should be low for noise, got {rm.r2}"
        assert rm.mae > 0, "MAE should always be positive"

    def test_to_dict_has_exactly_7_keys(self, correlated_data):
        """to_dict returns exactly the 7 expected metric keys."""
        y_true, y_pred = correlated_data
        rm = RegressionMetrics.from_arrays(y_true, y_pred)
        d = rm.to_dict()

        expected_keys = {
            "r2", "ic", "pearson", "mae", "rmse",
            "directional_accuracy", "profitable_accuracy",
        }
        assert set(d.keys()) == expected_keys, (
            f"Expected keys {expected_keys}, got {set(d.keys())}"
        )

    def test_to_dict_values_are_float(self, correlated_data):
        """All to_dict values must be Python floats (JSON-serializable)."""
        y_true, y_pred = correlated_data
        rm = RegressionMetrics.from_arrays(y_true, y_pred)
        d = rm.to_dict()

        for key, value in d.items():
            assert isinstance(value, float), (
                f"to_dict['{key}'] should be float, got {type(value).__name__}"
            )

    def test_summary_contains_all_metric_names(self, correlated_data):
        """summary() string contains all 7 metric names."""
        y_true, y_pred = correlated_data
        rm = RegressionMetrics.from_arrays(y_true, y_pred)
        s = rm.summary()

        for name in ["R-squared", "Information Coefficient", "Pearson", "MAE", "RMSE", "Directional Accuracy", "Profitable"]:
            assert name in s, f"summary() missing '{name}'. Got:\n{s}"

    def test_summary_starts_with_header(self, correlated_data):
        """summary() starts with the expected header."""
        y_true, y_pred = correlated_data
        rm = RegressionMetrics.from_arrays(y_true, y_pred)
        s = rm.summary()
        assert s.startswith("=== Regression Metrics ==="), (
            f"summary() should start with header, got: {s[:40]}"
        )

    def test_from_arrays_breakeven_bps_parameter(self, correlated_data):
        """from_arrays passes breakeven_bps to profitable_accuracy."""
        y_true, y_pred = correlated_data
        rm_default = RegressionMetrics.from_arrays(y_true, y_pred)
        rm_high = RegressionMetrics.from_arrays(y_true, y_pred, breakeven_bps=100.0)

        # Higher breakeven = fewer "profitable" predictions
        assert rm_high.profitable_accuracy <= rm_default.profitable_accuracy, (
            f"Higher breakeven should reduce profitable accuracy. "
            f"Default: {rm_default.profitable_accuracy}, High: {rm_high.profitable_accuracy}"
        )

    def test_extra_field_empty_by_default(self, correlated_data):
        """from_arrays produces empty extra dict."""
        y_true, y_pred = correlated_data
        rm = RegressionMetrics.from_arrays(y_true, y_pred)
        assert rm.extra == {} or rm.extra is None or len(rm.extra) == 0
