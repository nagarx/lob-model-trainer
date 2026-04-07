"""
Tests for regression_metrics.py.

Validates all regression metrics with known inputs and edge cases.
"""

import numpy as np
import pytest

from lobtrainer.training.regression_metrics import (
    r_squared,
    information_coefficient,
    pearson_correlation,
    mean_absolute_error,
    root_mean_squared_error,
    directional_accuracy,
    profitable_accuracy,
    compute_all_regression_metrics,
)


class TestRSquared:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert abs(r_squared(y, y) - 1.0) < 1e-10

    def test_mean_prediction(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean = np.full_like(y, y.mean())
        assert abs(r_squared(y, mean)) < 1e-10

    def test_worse_than_mean(self):
        y = np.array([1.0, 2.0, 3.0])
        pred = np.array([10.0, -5.0, 20.0])
        assert r_squared(y, pred) < 0.0

    def test_constant_target(self):
        y = np.array([5.0, 5.0, 5.0])
        pred = np.array([4.0, 5.0, 6.0])
        assert r_squared(y, pred) == 0.0


class TestInformationCoefficient:
    def test_perfect_rank_correlation(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert abs(information_coefficient(y, y) - 1.0) < 1e-10

    def test_perfect_negative_rank(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert abs(information_coefficient(y, -y) - (-1.0)) < 1e-10

    def test_too_few_samples(self):
        assert information_coefficient(np.array([1.0, 2.0]), np.array([3.0, 4.0])) == 0.0


class TestPearsonCorrelation:
    def test_perfect_linear(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        pred = y * 2.0 + 1.0
        assert abs(pearson_correlation(y, pred) - 1.0) < 1e-10

    def test_too_few_samples(self):
        assert pearson_correlation(np.array([1.0]), np.array([2.0])) == 0.0


class TestMAE:
    def test_zero_error(self):
        y = np.array([1.0, 2.0, 3.0])
        assert abs(mean_absolute_error(y, y)) < 1e-10

    def test_known_value(self):
        """MAE of constant-zero truth vs [1, -1, 2] = mean([1, 1, 2]) = 4/3."""
        y = np.array([0.0, 0.0, 0.0])
        pred = np.array([1.0, -1.0, 2.0])
        expected = np.abs(y - pred).mean()  # 4/3
        assert abs(mean_absolute_error(y, pred) - expected) < 1e-10


class TestRMSE:
    def test_zero_error(self):
        y = np.array([1.0, 2.0, 3.0])
        assert abs(root_mean_squared_error(y, y)) < 1e-10

    def test_known_value(self):
        """RMSE of constant-zero truth vs [1, -1, 2] = sqrt(mean([1, 1, 4])) = sqrt(2)."""
        y = np.array([0.0, 0.0, 0.0])
        pred = np.array([1.0, -1.0, 2.0])
        expected = np.sqrt(np.mean((y - pred) ** 2))  # sqrt(2)
        assert abs(root_mean_squared_error(y, pred) - expected) < 1e-10


class TestDirectionalAccuracy:
    def test_perfect_direction(self):
        y = np.array([1.0, -2.0, 3.0, -4.0])
        pred = np.array([0.5, -0.1, 100.0, -0.01])
        assert abs(directional_accuracy(y, pred) - 1.0) < 1e-10

    def test_opposite_direction(self):
        """All 3 predictions have wrong sign -> DA = 0.0."""
        y = np.array([1.0, -2.0, 3.0])
        pred = np.array([-1.0, 2.0, -0.5])
        assert abs(directional_accuracy(y, pred)) < 1e-10

    def test_zeros_excluded(self):
        y = np.array([0.0, 0.0])
        pred = np.array([1.0, -1.0])
        assert directional_accuracy(y, pred) == 0.5


class TestProfitableAccuracy:
    def test_all_profitable(self):
        y = np.array([10.0, -10.0, 20.0])
        pred = np.array([5.0, -5.0, 15.0])
        assert abs(profitable_accuracy(y, pred, breakeven_bps=5.0) - 1.0) < 1e-10

    def test_none_profitable_wrong_direction(self):
        y = np.array([10.0, -10.0])
        pred = np.array([-5.0, 5.0])
        assert abs(profitable_accuracy(y, pred, breakeven_bps=5.0)) < 1e-10

    def test_no_large_moves(self):
        y = np.array([0.1, -0.1])
        pred = np.array([0.05, -0.05])
        assert profitable_accuracy(y, pred, breakeven_bps=5.0) == 0.0


class TestComputeAll:
    def test_returns_all_keys(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pred = y * 0.9 + 0.1
        metrics = compute_all_regression_metrics(y, pred)
        assert "r2" in metrics
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "pearson" in metrics
        assert "directional_accuracy" in metrics
        assert "profitable_accuracy" in metrics

    def test_prefix(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pred = y + 0.1
        metrics = compute_all_regression_metrics(y, pred, prefix="val_h60_")
        assert "val_h60_r2" in metrics
        assert "val_h60_mae" in metrics

    def test_all_values_finite(self):
        np.random.seed(42)
        y = np.random.randn(100)
        pred = y * 0.5 + np.random.randn(100) * 0.3
        metrics = compute_all_regression_metrics(y, pred)
        for k, v in metrics.items():
            assert np.isfinite(v), f"Non-finite value for {k}: {v}"
