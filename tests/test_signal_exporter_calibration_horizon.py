"""Phase A (2026-04-23) regression tests for exporter caller-side horizon slicing.

Locks the invariants from Phase A C2 (bugs #6 and #6b): ``SignalExporter`` must
use :func:`resolve_labels_config` to read the canonical
``primary_horizon_idx`` from ``config.data.labels`` when slicing multi-horizon
predictions / labels for either:

    (#6)  calibration (``_apply_calibration``)
    (#6b) stats metadata (``_build_metadata`` prediction_stats + regression metrics)

Pre-Phase-A, both sites hardcoded ``[:, 0]`` — silently mis-calibrating and
mis-reporting for every HMHP-R experiment with ``primary_horizon_idx != 0``.
Post-Phase-A, :func:`calibrate_variance` is strict 1-D (raises ``ValueError``
on 2-D input), so the caller-side fix in the exporter is enforced end-to-end
by a type-contract boundary.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import pytest

_hft_contracts = pytest.importorskip("hft_contracts")

from lobtrainer.config.schema import (
    DataConfig,
    ExperimentConfig,
    LabelsConfig,
    ModelConfig,
    TrainConfig,
)


def _build_config_with_primary_horizon_idx(idx: int) -> ExperimentConfig:
    """Minimal ExperimentConfig with multi-horizon labels + configurable primary idx."""
    return ExperimentConfig(
        data=DataConfig(
            feature_count=98,
            labels=LabelsConfig(
                horizons=[10, 60, 300],
                primary_horizon_idx=idx,
                task="regression",
            ),
        ),
        model=ModelConfig(model_type="tlob", input_size=98, num_classes=3),
        train=TrainConfig(epochs=1),
    )


def _build_inference(
    predicted_returns_2d: np.ndarray,
    regression_labels_2d: np.ndarray,
) -> dict:
    """Shape-conformant inference dict (mimics ``SignalExporter._run_inference``)."""
    return {
        "signal_type": "regression",
        "n_samples": predicted_returns_2d.shape[0],
        "predicted_returns": predicted_returns_2d,
        "regression_labels": regression_labels_2d,
        "horizons": [10, 60, 300],
    }


class _DummyModel:
    """Minimal model stand-in exposing ``.parameters()`` (count used by metadata)."""

    def __init__(self, param_count: int = 0):
        self._param_count = param_count
        self.__class__.__name__ = "DummyModel"  # populates metadata.model_name

    def parameters(self):
        return iter(())  # zero parameters — sum(p.numel()) = 0


class _DummyStrategy:
    """Non-HMHP strategy stand-in — triggers the generic regression branch."""


class _DummyTrainer:
    """Trainer stand-in with the attributes ``SignalExporter`` probes in its
    calibration + metadata paths. Keeps test setup independent of the heavy
    Trainer.setup() path (data loading, model construction, etc.).
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.model = _DummyModel()
        self.strategy = _DummyStrategy()


class TestExporterCalibrationSlicesByPrimaryHorizonIdx:
    """Bug #6: ``_apply_calibration`` must use ``primary_horizon_idx`` from config,
    not hardcoded 0, when reducing 2-D predictions/labels to 1-D.
    """

    @pytest.mark.parametrize("primary_idx", [0, 1, 2])
    def test_calibration_uses_configured_primary_horizon_idx(
        self, primary_idx: int,
    ) -> None:
        """Target-std of calibration matches the selected column's std.

        Per-column target std:
          col 0: std=14.14 (values 10..50, step 10)
          col 1: std=1414.21 (values 1000..5000, step 1000)
          col 2: std=141421.4 (values 100000..500000, step 100000)

        If the exporter hardcodes column 0, ``target_std`` is ~14.14 regardless
        of the configured ``primary_idx``. Post-Phase-A, it tracks the column.
        """
        from lobtrainer.export.exporter import SignalExporter

        config = _build_config_with_primary_horizon_idx(primary_idx)
        exporter = SignalExporter(_DummyTrainer(config), calibration="variance_match")  # type: ignore[arg-type]

        # Construct a 3-column (N=5) predictions+labels matrix with per-column
        # stds differing by orders of magnitude to make wrong-column selection
        # trivially detectable in the assertion.
        preds_2d = np.column_stack([
            np.linspace(0.5, 2.5, 5),      # col 0 small
            np.linspace(50, 250, 5),       # col 1 medium
            np.linspace(5000, 25000, 5),   # col 2 large
        ])
        labels_2d = np.column_stack([
            np.linspace(10.0, 50.0, 5),                  # col 0 std ≈ 14.14
            np.linspace(1000.0, 5000.0, 5),              # col 1 std ≈ 1414.21
            np.linspace(100000.0, 500000.0, 5),          # col 2 std ≈ 141421.36
        ])
        inference = _build_inference(preds_2d, labels_2d)

        result = exporter._apply_calibration(inference)
        assert result is not None
        expected_target_std = float(np.std(labels_2d[:, primary_idx]))
        assert result["target_std"] == pytest.approx(expected_target_std), (
            f"Calibration target_std ({result['target_std']}) does not match "
            f"std of labels column {primary_idx} ({expected_target_std}). "
            f"Suggests the exporter is still hardcoding [:, 0] or a different "
            f"column instead of threading primary_horizon_idx from config."
        )
        # Metadata provenance propagates (forward-compat for Phase B calibrators)
        assert result["metadata"] == {"primary_horizon_idx": primary_idx}

    def test_default_primary_horizon_idx_zero_preserves_old_behavior(self) -> None:
        """Regression guard: when primary_horizon_idx=0 (the default), behavior
        is observationally identical to the pre-Phase-A hardcoded-0 path.

        This locks the backward-compat promise for the >99% of experiments
        that use the default horizon.
        """
        from lobtrainer.export.exporter import SignalExporter

        config = _build_config_with_primary_horizon_idx(0)
        exporter = SignalExporter(_DummyTrainer(config), calibration="variance_match")  # type: ignore[arg-type]

        preds_2d = np.column_stack([
            np.linspace(0.5, 2.5, 5),
            np.linspace(50, 250, 5),
        ])
        labels_2d = np.column_stack([
            np.linspace(10.0, 50.0, 5),
            np.linspace(1000.0, 5000.0, 5),
        ])
        inference = _build_inference(preds_2d, labels_2d)
        result = exporter._apply_calibration(inference)
        assert result is not None
        # Target_std matches column 0 — unchanged from pre-Phase-A for default case.
        assert result["target_std"] == pytest.approx(float(np.std(labels_2d[:, 0])))


class TestExporterMetadataStatsSliceByPrimaryHorizonIdx:
    """Bug #6b: ``_build_metadata`` prediction_stats + regression metrics must
    slice by configured ``primary_horizon_idx``, not hardcoded 0. These stats
    are published to ``signal_metadata.json`` and consumed by
    ``hft-ops statistical_compare`` — wrong column silently misrepresents the
    experiment's primary horizon.
    """

    @pytest.mark.parametrize("primary_idx", [0, 1, 2])
    def test_prediction_stats_use_configured_primary_horizon_idx(
        self, primary_idx: int, tmp_path,
    ) -> None:
        """prediction_stats.mean/std match the configured column's stats."""
        from lobtrainer.export.exporter import SignalExporter

        config = _build_config_with_primary_horizon_idx(primary_idx)
        exporter = SignalExporter(_DummyTrainer(config), calibration="none")  # type: ignore[arg-type]

        preds_2d = np.column_stack([
            np.linspace(0.5, 2.5, 5),
            np.linspace(50, 250, 5),
            np.linspace(5000, 25000, 5),
        ])
        labels_2d = np.column_stack([
            np.linspace(10.0, 50.0, 5),
            np.linspace(1000.0, 5000.0, 5),
            np.linspace(100000.0, 500000.0, 5),
        ])
        inference = _build_inference(preds_2d, labels_2d)

        # _build_metadata takes (inference, raw, split, output_dir, calibration_result).
        # raw is a dataclass-like object exposing spread/price numpy arrays; we
        # construct a tiny stand-in (only .spread_stats / .price_stats / fields
        # accessed by _build_metadata are needed; see the real
        # RawFeatureExtractor.extract() return shape).
        class _RawStub:
            spreads = np.zeros(5, dtype=np.float64)
            prices = np.ones(5, dtype=np.float64) * 100.0
            n_samples = 5

        meta = exporter._build_metadata(
            inference=inference,
            raw=_RawStub(),
            split="val",
            output_dir=tmp_path,
            calibration_result=None,
        )
        ps = meta.get("prediction_stats", {})
        expected_mean = float(np.mean(preds_2d[:, primary_idx]))
        expected_std = float(np.std(preds_2d[:, primary_idx]))
        assert ps.get("mean") == pytest.approx(expected_mean), (
            f"prediction_stats.mean doesn't match column {primary_idx} "
            f"(expected {expected_mean}, got {ps.get('mean')})"
        )
        assert ps.get("std") == pytest.approx(expected_std)

    def test_default_primary_horizon_idx_zero_stats_preserves_old_behavior(
        self, tmp_path,
    ) -> None:
        """Regression guard — default primary_horizon_idx=0 gives identical
        prediction_stats to the pre-Phase-A hardcoded-0 case."""
        from lobtrainer.export.exporter import SignalExporter

        config = _build_config_with_primary_horizon_idx(0)
        exporter = SignalExporter(_DummyTrainer(config), calibration="none")  # type: ignore[arg-type]

        preds_2d = np.column_stack([
            np.linspace(0.5, 2.5, 5),
            np.linspace(50, 250, 5),
        ])
        labels_2d = np.column_stack([
            np.linspace(10.0, 50.0, 5),
            np.linspace(1000.0, 5000.0, 5),
        ])
        inference = _build_inference(preds_2d, labels_2d)

        class _RawStub:
            spreads = np.zeros(5, dtype=np.float64)
            prices = np.ones(5, dtype=np.float64) * 100.0
            n_samples = 5

        meta = exporter._build_metadata(
            inference=inference,
            raw=_RawStub(),
            split="val",
            output_dir=tmp_path,
            calibration_result=None,
        )
        assert meta["prediction_stats"]["mean"] == pytest.approx(
            float(np.mean(preds_2d[:, 0]))
        )
