"""
Phase X.3 Empirical Trust — Phase A regression tests.

Tests the NaN-loud train loop discipline shipped 2026-05-05:
  - TrainingDivergedError raised at Trainer._train_epoch:902 (PRE-backward)
  - TrainingDivergedError raised at TrainingDiagnostics.on_batch_end (post-hoc)
  - MonitorMetricUndefined raised at EarlyStopping._is_better
  - MonitorMetricUndefined raised at ModelCheckpoint._is_better
  - ModelCheckpoint.on_epoch_end warn+returns on missing metric (no spurious save)
  - DegenerateFeatureError raised at transforms.compute_statistics(strict=True)
  - Trainer.__init__ auto-registers TrainingDiagnostics(alert_on_nan=True)
  - Auto-register skips when user supplies their own TrainingDiagnostics

These tests lock the design — preventing accidental regressions to the silent-
zero / silent-NaN behavior caught by the 2026-05-05 multi-agent audit.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import numpy as np
import pytest

from lobtrainer.training.callbacks import EarlyStopping, ModelCheckpoint
from lobtrainer.training.exceptions import (
    DegenerateFeatureError,
    MonitorMetricUndefined,
    TrainingDivergedError,
)
from lobtrainer.training.monitoring import TrainingDiagnostics
from lobtrainer.data.transforms import compute_statistics


class TestTrainingDivergedError:
    """TrainingDivergedError carries full context for debugging."""

    def test_carries_epoch_batch_loss_step_fields(self):
        e = TrainingDivergedError(
            epoch=3, batch=42, loss_value=float("nan"), global_step=128
        )
        assert e.epoch == 3
        assert e.batch == 42
        assert math.isnan(e.loss_value)
        assert e.global_step == 128
        # Message must include all context (operator should debug from message alone)
        msg = str(e)
        assert "epoch=3" in msg
        assert "batch=42" in msg
        assert "global_step=128" in msg

    def test_inherits_runtime_error(self):
        # User code catching RuntimeError should also catch TrainingDivergedError
        e = TrainingDivergedError(0, 0, float("inf"), 0)
        assert isinstance(e, RuntimeError)


class TestMonitorMetricUndefined:
    def test_carries_metric_and_value(self):
        e = MonitorMetricUndefined(metric="val_loss", value=float("nan"))
        assert e.metric == "val_loss"
        assert math.isnan(e.value)
        assert "val_loss" in str(e)

    def test_inherits_runtime_error(self):
        e = MonitorMetricUndefined("val_ic", float("nan"))
        assert isinstance(e, RuntimeError)


class TestDegenerateFeatureError:
    def test_all_nan_reason(self):
        e = DegenerateFeatureError(feature_indices=[5, 12], reason="all_nan")
        assert e.feature_indices == [5, 12]
        assert e.reason == "all_nan"
        assert "all-NaN" in str(e)
        assert "[5, 12]" in str(e)

    def test_zero_variance_reason_with_eps(self):
        e = DegenerateFeatureError(
            feature_indices=[97], reason="zero_variance", eps=1e-8
        )
        assert e.eps == 1e-8
        assert "zero variance" in str(e)
        assert "1e-08" in str(e)

    def test_inherits_value_error(self):
        e = DegenerateFeatureError([0], "all_nan")
        assert isinstance(e, ValueError)


class TestEarlyStoppingNaNGuard:
    """EarlyStopping._is_better raises MonitorMetricUndefined on non-finite."""

    def test_min_mode_raises_on_nan(self):
        es = EarlyStopping(metric="val_loss", mode="min", patience=3)
        with pytest.raises(MonitorMetricUndefined) as exc_info:
            es._is_better(float("nan"), 0.5)
        assert exc_info.value.metric == "val_loss"
        assert math.isnan(exc_info.value.value)

    def test_max_mode_raises_on_nan(self):
        es = EarlyStopping(metric="val_ic", mode="max", patience=3)
        with pytest.raises(MonitorMetricUndefined):
            es._is_better(float("nan"), 0.0)

    def test_min_mode_raises_on_pos_inf(self):
        es = EarlyStopping(metric="val_loss", mode="min", patience=3)
        with pytest.raises(MonitorMetricUndefined):
            es._is_better(float("inf"), 0.5)

    def test_min_mode_raises_on_neg_inf(self):
        es = EarlyStopping(metric="val_loss", mode="min", patience=3)
        with pytest.raises(MonitorMetricUndefined):
            es._is_better(float("-inf"), 0.5)

    def test_finite_value_works_normally(self):
        es = EarlyStopping(metric="val_loss", mode="min", patience=3)
        # Improvement (lower is better in min mode)
        assert es._is_better(0.3, 0.5) is True
        # No improvement
        assert es._is_better(0.6, 0.5) is False


class TestModelCheckpointNaNGuard:
    """ModelCheckpoint._is_better raises MonitorMetricUndefined on non-finite."""

    def test_min_mode_raises_on_nan(self):
        mc = ModelCheckpoint(save_dir="/tmp/test", metric="val_loss", mode="min")
        with pytest.raises(MonitorMetricUndefined):
            mc._is_better(float("nan"), 0.5)

    def test_max_mode_raises_on_inf(self):
        mc = ModelCheckpoint(save_dir="/tmp/test", metric="val_ic", mode="max")
        with pytest.raises(MonitorMetricUndefined):
            mc._is_better(float("-inf"), 0.0)

    def test_finite_value_works_normally(self):
        mc = ModelCheckpoint(save_dir="/tmp/test", metric="val_loss", mode="min")
        assert mc._is_better(0.3, 0.5) is True
        assert mc._is_better(0.6, 0.5) is False


class TestModelCheckpointMissingMetric:
    """ModelCheckpoint.on_epoch_end warns + returns when metric missing.

    Pre-Phase-X.3, ``logs.get(self.metric, 0.0)`` silently returned 0.0 →
    for mode='min' this trivially beat ``best_value=inf`` → spurious "best"
    checkpoint saved. Now warns + returns early without saving.
    """

    def test_missing_metric_warns_and_skips_save(self, tmp_path, caplog):
        import logging
        mc = ModelCheckpoint(
            save_dir=str(tmp_path), metric="val_loss", mode="min", save_best_only=True
        )
        # Mock trainer (won't be called since we skip)
        mc.trainer = MagicMock()
        # Logs do NOT contain val_loss
        with caplog.at_level(logging.WARNING, logger="lobtrainer.training.callbacks"):
            mc.on_epoch_end(epoch=0, logs={"train_loss": 0.5})
        # Verify warning was emitted
        assert any("not found in logs" in rec.message for rec in caplog.records)
        # Verify NO checkpoint was saved (trainer.model.state_dict not called)
        mc.trainer.model.state_dict.assert_not_called()


class TestTransformsComputeStatistics:
    """transforms.compute_statistics fail-loud on all-NaN columns when strict=True."""

    def test_strict_true_raises_on_all_nan_column(self):
        # 100 samples, 5 features; feature 2 is all-NaN
        features = np.random.randn(100, 5)
        features[:, 2] = np.nan
        # book_valid mask: use all-finite-row mask via passing valid_mask
        valid_mask = np.ones(100, dtype=bool)
        with pytest.raises(DegenerateFeatureError) as exc_info:
            compute_statistics(features, valid_mask=valid_mask, strict=True)
        assert exc_info.value.reason == "all_nan"
        assert 2 in exc_info.value.feature_indices

    def test_strict_false_legacy_path_warns(self):
        features = np.random.randn(100, 5)
        features[:, 2] = np.nan
        valid_mask = np.ones(100, dtype=bool)
        with pytest.warns(DeprecationWarning, match="strict=False"):
            stats = compute_statistics(features, valid_mask=valid_mask, strict=False)
        # Legacy path: silent fabrication produces mean=0, std=1 for all-NaN column
        assert stats.mean[2] == 0.0
        assert stats.std[2] == 1.0

    def test_strict_true_default_unchanged_for_normal_data(self):
        # Normal data: no all-NaN columns → no raise, no warning
        np.random.seed(42)
        features = np.random.randn(1000, 5) * 2 + 1
        valid_mask = np.ones(1000, dtype=bool)
        # Default strict=True (Phase X.3); should compute normally
        stats = compute_statistics(features, valid_mask=valid_mask)
        assert stats.count == 1000
        assert np.isfinite(stats.mean).all()
        assert np.isfinite(stats.std).all()


class TestTrainerAutoRegisterTrainingDiagnostics:
    """Trainer.__init__ auto-registers TrainingDiagnostics(alert_on_nan=True)."""

    def test_default_callbacks_include_training_diagnostics(self, tmp_path):
        from lobtrainer.config import load_config
        from lobtrainer.training import create_trainer

        c = load_config("configs/experiments/nvda_first_pytorch_v3p0.yaml")
        # Override output_dir to avoid clobbering live experiments
        c = c.model_copy(update={"output_dir": str(tmp_path)})
        t = create_trainer(c, callbacks=[])
        cb_names = [type(cb).__name__ for cb in t.callbacks.callbacks]
        assert "TrainingDiagnostics" in cb_names

        # Verify alert_on_nan is True
        diag = [cb for cb in t.callbacks.callbacks if type(cb).__name__ == "TrainingDiagnostics"][0]
        assert diag.alert_on_nan is True

    def test_user_supplied_diagnostics_skips_auto_register(self, tmp_path):
        """If user passes their own TrainingDiagnostics, don't double-register."""
        from lobtrainer.config import load_config
        from lobtrainer.training import create_trainer

        c = load_config("configs/experiments/nvda_first_pytorch_v3p0.yaml")
        c = c.model_copy(update={"output_dir": str(tmp_path)})
        # User supplies their own with alert_on_nan=False (research mode)
        user_diag = TrainingDiagnostics(alert_on_nan=False)
        t = create_trainer(c, callbacks=[user_diag])
        diag_callbacks = [cb for cb in t.callbacks.callbacks if type(cb).__name__ == "TrainingDiagnostics"]
        # Exactly ONE — user's, not auto-registered duplicate
        assert len(diag_callbacks) == 1
        assert diag_callbacks[0] is user_diag
        assert diag_callbacks[0].alert_on_nan is False


class TestTrainingDiagnosticsExceptionType:
    """TrainingDiagnostics.on_batch_end raises TrainingDivergedError (not ValueError)."""

    def test_nan_loss_raises_training_diverged(self):
        diag = TrainingDiagnostics(alert_on_nan=True)
        with pytest.raises(TrainingDivergedError) as exc_info:
            diag.on_batch_end(0, {"loss": float("nan")})
        assert math.isnan(exc_info.value.loss_value)
        assert exc_info.value.batch == 0
        # Without trainer set, epoch/global_step default to -1
        assert exc_info.value.epoch == -1
        assert exc_info.value.global_step == -1

    def test_inf_loss_raises_training_diverged(self):
        diag = TrainingDiagnostics(alert_on_nan=True)
        with pytest.raises(TrainingDivergedError) as exc_info:
            diag.on_batch_end(0, {"loss": float("inf")})
        assert math.isinf(exc_info.value.loss_value)

    def test_with_trainer_context_includes_epoch_and_step(self):
        diag = TrainingDiagnostics(alert_on_nan=True)
        # Mock trainer with state
        mock_trainer = MagicMock()
        mock_trainer.state.current_epoch = 5
        mock_trainer.state.global_step = 250
        diag.trainer = mock_trainer

        with pytest.raises(TrainingDivergedError) as exc_info:
            diag.on_batch_end(7, {"loss": float("nan")})
        assert exc_info.value.epoch == 5
        assert exc_info.value.batch == 7
        assert exc_info.value.global_step == 250
