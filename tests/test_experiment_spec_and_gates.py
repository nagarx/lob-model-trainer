"""Tests for T14 ExperimentSpec + signal quality gate.

Covers: ExperimentSpec.from_yaml, to_experiment_config, validate,
SignalQualityGateConfig, run_signal_quality_gate. Uses synthetic
fixtures and real NVDA E5 60s data.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml

from lobtrainer.config.experiment_spec import (
    CostGateConfig,
    ExperimentMetadata,
    ExperimentSpec,
    GateConfig,
    SignalQualityGateConfig,
)
from lobtrainer.config.schema import ExperimentConfig
from lobtrainer.training.gates import GateResult, run_signal_quality_gate


# =============================================================================
# ExperimentSpec.from_yaml
# =============================================================================


class TestExperimentSpecFromYaml:
    def _write_yaml(self, data, tmp_path):
        p = tmp_path / "spec.yaml"
        with open(p, "w") as f:
            yaml.dump(data, f)
        return str(p)

    def test_full_spec_parses(self, tmp_path):
        data = {
            "experiment": {"name": "Test", "hypothesis": "H1", "tags": ["a"]},
            "data": {"feature_count": 98, "normalization": {"strategy": "none"}},
            "model": {"model_type": "tlob", "input_size": 98},
            "train": {"epochs": 5},
            "gates": {"signal_quality": {"enabled": True, "min_ic": 0.1}},
            "cv": {"n_splits": 3},
            "output_dir": "/tmp/test",
        }
        spec = ExperimentSpec.from_yaml(self._write_yaml(data, tmp_path))
        assert spec.experiment.name == "Test"
        assert spec.experiment.hypothesis == "H1"
        assert spec.gates.signal_quality.min_ic == 0.1
        assert spec.cv is not None
        assert spec.output_dir == "/tmp/test"

    def test_missing_sections_use_defaults(self, tmp_path):
        data = {"data": {"feature_count": 98}, "model": {"input_size": 98}}
        spec = ExperimentSpec.from_yaml(self._write_yaml(data, tmp_path))
        assert spec.experiment.name == "unnamed"
        assert spec.gates.signal_quality.enabled is True
        assert spec.cv is None

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            ExperimentSpec.from_yaml("/nonexistent/path.yaml")

    def test_non_dict_yaml_raises(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("- item1\n- item2\n")
        with pytest.raises(ValueError, match="must be a dict"):
            ExperimentSpec.from_yaml(str(p))


# =============================================================================
# ExperimentSpec.to_experiment_config
# =============================================================================


class TestToExperimentConfig:
    def test_produces_valid_config(self):
        spec = ExperimentSpec(
            experiment=ExperimentMetadata(name="test"),
            data={"feature_count": 98, "normalization": {"strategy": "none"}},
            model={"model_type": "tlob", "input_size": 0},
            train={"epochs": 5},
        )
        config = spec.to_experiment_config()
        assert isinstance(config, ExperimentConfig)
        assert config.model.input_size == 98  # auto-derived (T13)
        assert config.train.epochs == 5

    def test_with_cv_config(self):
        spec = ExperimentSpec(
            data={"feature_count": 98},
            model={"input_size": 98},
            cv={"n_splits": 3, "embargo_days": 2},
        )
        config = spec.to_experiment_config()
        assert config.cv is not None
        assert config.cv.n_splits == 3
        assert config.cv.embargo_days == 2

    def test_with_labels_config(self):
        spec = ExperimentSpec(
            data={
                "feature_count": 98,
                "labels": {"source": "forward_prices", "task": "regression"},
            },
            model={"input_size": 98},
        )
        config = spec.to_experiment_config()
        assert config.data.labels.source == "forward_prices"
        assert config.data.labels.task == "regression"

    def test_with_multi_source(self):
        spec = ExperimentSpec(
            data={
                "feature_count": 132,
                "normalization": {"strategy": "none"},
                "sources": [
                    {"name": "mbo", "data_dir": "/tmp", "role": "primary", "feature_count": 98},
                    {"name": "basic", "data_dir": "/tmp", "role": "auxiliary", "feature_count": 34},
                ],
            },
            model={"input_size": 0},
        )
        config = spec.to_experiment_config()
        assert config.model.input_size == 132
        assert len(config.data.sources) == 2


# =============================================================================
# ExperimentSpec.validate
# =============================================================================


class TestValidate:
    def test_valid_spec_no_errors(self):
        spec = ExperimentSpec(
            experiment=ExperimentMetadata(name="E1", hypothesis="Test H"),
            data={"feature_count": 98},
            model={"input_size": 98},
        )
        warnings = spec.validate()
        assert isinstance(warnings, list)

    def test_unnamed_experiment_warns(self):
        spec = ExperimentSpec(
            data={"feature_count": 98},
            model={"input_size": 98},
        )
        warnings = spec.validate()
        assert any("unnamed" in w.lower() for w in warnings)

    def test_no_hypothesis_warns(self):
        spec = ExperimentSpec(
            experiment=ExperimentMetadata(name="E1"),
            data={"feature_count": 98},
            model={"input_size": 98},
        )
        warnings = spec.validate()
        assert any("hypothesis" in w.lower() for w in warnings)

    def test_empty_data_raises(self):
        spec = ExperimentSpec(model={"input_size": 98})
        with pytest.raises(ValueError, match="data section is empty"):
            spec.validate()

    def test_bad_min_ic_raises(self):
        spec = ExperimentSpec(
            data={"feature_count": 98},
            model={"input_size": 98},
            gates=GateConfig(signal_quality=SignalQualityGateConfig(min_ic=-0.1)),
        )
        with pytest.raises(ValueError, match="min_ic must be > 0"):
            spec.validate()


# =============================================================================
# run_signal_quality_gate
# =============================================================================


class TestSignalQualityGate:
    def _make_day_with_signal(self, n=100, n_features=10, ic_strength=0.5):
        """Create a DayData-like object with known signal strength."""
        from lobtrainer.data.dataset import DayData

        rng = np.random.default_rng(42)
        labels = rng.standard_normal(n)
        features = np.column_stack([
            labels * ic_strength + rng.standard_normal(n) * (1 - ic_strength)
            for _ in range(n_features)
        ])
        reg_labels = labels.reshape(-1, 1)
        return DayData(
            date="20250203",
            features=features.astype(np.float64),
            labels=np.zeros(n, dtype=np.int64),
            regression_labels=reg_labels.astype(np.float64),
            is_aligned=True,
        )

    def _make_day_random(self, n=100, n_features=10):
        """Create a DayData-like with pure random (no signal)."""
        from lobtrainer.data.dataset import DayData

        rng = np.random.default_rng(99)
        return DayData(
            date="20250203",
            features=rng.standard_normal((n, n_features)).astype(np.float64),
            labels=np.zeros(n, dtype=np.int64),
            regression_labels=rng.standard_normal((n, 1)).astype(np.float64),
            is_aligned=True,
        )

    def test_passes_with_signal(self):
        days = [self._make_day_with_signal(ic_strength=0.8)]
        result = run_signal_quality_gate(days, min_ic=0.05)
        assert result.passed is True
        assert "PASSED" in result.message

    def test_fails_with_random(self):
        days = [self._make_day_random()]
        result = run_signal_quality_gate(days, min_ic=0.5)
        assert result.passed is False
        assert "FAILED" in result.message

    def test_empty_days_fails(self):
        result = run_signal_quality_gate([], min_ic=0.05)
        assert result.passed is False

    def test_no_regression_labels_fails(self):
        from lobtrainer.data.dataset import DayData
        day = DayData(
            date="20250203",
            features=np.random.randn(10, 5).astype(np.float64),
            labels=np.zeros(10, dtype=np.int64),
            regression_labels=None,
            is_aligned=True,
        )
        result = run_signal_quality_gate([day], min_ic=0.05)
        assert result.passed is False

    def test_gate_result_has_feature_ics(self):
        days = [self._make_day_with_signal(n_features=5)]
        result = run_signal_quality_gate(days, min_ic=0.05)
        assert len(result.details) == 5

    @pytest.mark.xfail(
        strict=False,
        reason="Phase G G.6.A bumped SchemaVersion 2.2 → 3.0 (MAJOR per "
        "CLAUDE.md root rule). Legacy production exports at "
        "data/exports/e5_timebased_60s are at schema '2.2' and correctly "
        "rejected by validate_schema_version. Re-export of these legacy "
        "exports is gated on Phase G+1 (re-export execution + scripts/ "
        "trigger infrastructure); test will pass once Phase G+1 ships and "
        "operator runs the regen. Until then this xfail is the structural "
        "marker that the legacy data needs re-export at schema 3.0.",
    )
    def test_real_e5_60s_data(self):
        """Signal quality gate on real NVDA E5 60s data should pass."""
        import warnings
        from lobtrainer.config.schema import LabelsConfig
        from lobtrainer.data.dataset import load_split_data

        data_dir = Path("../data/exports/e5_timebased_60s")
        if not data_dir.exists():
            pytest.skip("Real E5 60s data not available")

        lc = LabelsConfig(source="forward_prices", task="regression",
                          horizons=[10], primary_horizon_idx=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            days = load_split_data(str(data_dir), "train", labels_config=lc)[:5]

        result = run_signal_quality_gate(days, min_ic=0.05)
        assert result.passed is True, f"Real data gate should pass: {result.message}"
