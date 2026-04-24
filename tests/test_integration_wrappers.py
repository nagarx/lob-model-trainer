"""Tests for T10/T11 trainer integration wrappers + schema gaps.

Covers: compute_sample_weights_for_day, _resolve_horizon, CVTrainer,
CVResults, FoldResult, LabelsConfig.sample_weights validation,
CVConfig validation, SourceConfig validation, DataConfig.sources
validation, ForwardPriceContract invariant.
"""

import copy

import numpy as np
import pytest

from lobtrainer.config.schema import (
    CVConfig,
    DataConfig,
    ExperimentConfig,
    LabelsConfig,
    LabelingStrategy,
    ModelConfig,
    SourceConfig,
)


# =============================================================================
# Schema validation gaps
# =============================================================================


class TestLabelsConfigSampleWeights:
    def test_valid_none(self):
        lc = LabelsConfig(sample_weights="none")
        assert lc.sample_weights == "none"

    def test_valid_concurrent_overlap(self):
        lc = LabelsConfig(sample_weights="concurrent_overlap")
        assert lc.sample_weights == "concurrent_overlap"

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="sample_weights"):
            LabelsConfig(sample_weights="invalid_method")


class TestCVConfigValidation:
    def test_valid_defaults(self):
        cv = CVConfig()
        assert cv.n_splits == 5
        assert cv.embargo_days == 1

    def test_n_splits_too_small_raises(self):
        with pytest.raises(ValueError, match="n_splits must be >= 2"):
            CVConfig(n_splits=1)

    def test_negative_embargo_raises(self):
        with pytest.raises(ValueError, match="embargo_days must be >= 0"):
            CVConfig(embargo_days=-1)


class TestSourceConfigValidation:
    def test_valid_primary(self):
        sc = SourceConfig(name="mbo", data_dir="/tmp", role="primary")
        assert sc.feature_count == 0  # default

    def test_with_feature_count(self):
        sc = SourceConfig(name="mbo", data_dir="/tmp", feature_count=98)
        assert sc.feature_count == 98

    def test_negative_feature_count_raises(self):
        """Phase A.5.3d (2026-04-24): SourceConfig is now Pydantic BaseModel
        via SafeBaseModel; `raise ValueError(...)` in @model_validator is
        auto-wrapped as pydantic.ValidationError. match=substring still fires."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="feature_count"):
            SourceConfig(name="x", data_dir="/tmp", feature_count=-1)

    def test_invalid_role_raises(self):
        """Phase A.5.3d: error message now cites sorted(_VALID_ROLES) which
        places 'auxiliary' BEFORE 'primary' alphabetically. Updated the match
        pattern to the stable substring 'must be one of'."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="role must be one of"):
            SourceConfig(name="x", data_dir="/tmp", role="bad")


class TestDataConfigSourcesValidation:
    def test_no_primary_raises(self):
        with pytest.raises(ValueError, match="role='primary'"):
            DataConfig(sources=[
                SourceConfig(name="a", data_dir="/tmp", role="auxiliary"),
                SourceConfig(name="b", data_dir="/tmp", role="auxiliary"),
            ])

    def test_two_primaries_raises(self):
        with pytest.raises(ValueError, match="role='primary'"):
            DataConfig(sources=[
                SourceConfig(name="a", data_dir="/tmp", role="primary"),
                SourceConfig(name="b", data_dir="/tmp", role="primary"),
            ])

    def test_duplicate_names_raises(self):
        with pytest.raises(ValueError, match="Duplicate"):
            DataConfig(sources=[
                SourceConfig(name="mbo", data_dir="/tmp1", role="primary"),
                SourceConfig(name="mbo", data_dir="/tmp2", role="auxiliary"),
            ])

    def test_valid_sources(self):
        dc = DataConfig(sources=[
            SourceConfig(name="mbo", data_dir="/tmp1", role="primary"),
            SourceConfig(name="basic", data_dir="/tmp2", role="auxiliary"),
        ])
        assert len(dc.sources) == 2


class TestForwardPriceContractInvariant:
    def test_valid_invariant(self):
        from hft_contracts.label_factory import ForwardPriceContract
        c = ForwardPriceContract(smoothing_window_offset=5, max_horizon=300, n_columns=306)
        assert c.n_columns == 306  # 5 + 300 + 1

    def test_invalid_invariant_raises(self):
        from hft_contracts.label_factory import ForwardPriceContract
        with pytest.raises(ValueError, match="invariant violated"):
            ForwardPriceContract(smoothing_window_offset=5, max_horizon=300, n_columns=999)


# =============================================================================
# T10: compute_sample_weights_for_day
# =============================================================================


class TestComputeSampleWeightsForDay:
    def test_returns_none_when_disabled(self):
        from lobtrainer.data.sample_weights import compute_sample_weights_for_day
        lc = LabelsConfig(sample_weights="none")
        result = compute_sample_weights_for_day(100, {"horizons": [10]}, lc)
        assert result is None

    def test_returns_weights_when_enabled(self):
        from lobtrainer.data.sample_weights import compute_sample_weights_for_day
        lc = LabelsConfig(
            sample_weights="concurrent_overlap",
            primary_horizon_idx=0,
        )
        meta = {"horizons": [10, 60, 300]}
        result = compute_sample_weights_for_day(100, meta, lc, stride=1)
        assert result is not None
        assert result.shape == (100,)
        assert np.isclose(result.mean(), 1.0, atol=1e-10)

    def test_hmhp_mode_uses_max_horizon(self):
        from lobtrainer.data.sample_weights import _resolve_horizon
        lc = LabelsConfig(primary_horizon_idx=None)  # HMHP
        meta = {"horizons": [10, 60, 300]}
        h = _resolve_horizon(meta, lc)
        assert h == 300  # max of [10, 60, 300]

    def test_resolve_horizon_from_labeling_key(self):
        from lobtrainer.data.sample_weights import _resolve_horizon
        lc = LabelsConfig(primary_horizon_idx=1)
        meta = {"labeling": {"horizons": [10, 60, 300]}}
        h = _resolve_horizon(meta, lc)
        assert h == 60  # index 1

    def test_resolve_horizon_out_of_bounds_returns_none(self):
        from lobtrainer.data.sample_weights import _resolve_horizon
        lc = LabelsConfig(primary_horizon_idx=99)
        meta = {"horizons": [10]}
        h = _resolve_horizon(meta, lc)
        assert h is None

    def test_zero_samples_returns_none(self):
        from lobtrainer.data.sample_weights import compute_sample_weights_for_day
        lc = LabelsConfig(sample_weights="concurrent_overlap")
        result = compute_sample_weights_for_day(0, {"horizons": [10]}, lc)
        assert result is None


# =============================================================================
# T11: CVTrainer + CVResults
# =============================================================================


class TestCVTrainerInit:
    def test_default_values(self):
        from lobtrainer.training.cv_trainer import CVTrainer
        cfg = ExperimentConfig()
        cv = CVTrainer(cfg)
        assert cv.n_splits == 5
        assert cv.embargo_days == 1

    def test_explicit_overrides_defaults(self):
        from lobtrainer.training.cv_trainer import CVTrainer
        cfg = ExperimentConfig()
        cv = CVTrainer(cfg, n_splits=3, embargo_days=2)
        assert cv.n_splits == 3
        assert cv.embargo_days == 2

    def test_cvconfig_overrides_defaults(self):
        from lobtrainer.training.cv_trainer import CVTrainer
        cfg = ExperimentConfig(cv=CVConfig(n_splits=7, embargo_days=3))
        cv = CVTrainer(cfg)
        assert cv.n_splits == 7
        assert cv.embargo_days == 3

    def test_explicit_overrides_cvconfig(self):
        from lobtrainer.training.cv_trainer import CVTrainer
        cfg = ExperimentConfig(cv=CVConfig(n_splits=7, embargo_days=3))
        cv = CVTrainer(cfg, n_splits=4, embargo_days=0)
        assert cv.n_splits == 4
        assert cv.embargo_days == 0


class TestBuildFoldConfig:
    def test_output_dir_per_fold(self):
        from lobtrainer.training.cv_trainer import CVTrainer
        cfg = ExperimentConfig(output_dir="/tmp/base")
        cv = CVTrainer(cfg)
        fold_cfg = cv._build_fold_config(2)
        assert "cv_fold_2" in fold_cfg.output_dir

    def test_seed_per_fold(self):
        """Phase A.5.3e (2026-04-24): TrainConfig is now frozen Pydantic.
        Phase A.5.3i (2026-04-24 KEYSTONE): ExperimentConfig is now also
        frozen — cannot assign ``cfg.train = X`` either. Two-layer
        model_copy pattern matches cli.py + CVTrainer._build_fold_config.
        """
        from lobtrainer.training.cv_trainer import CVTrainer
        cfg = ExperimentConfig()
        cfg = cfg.model_copy(update={
            "train": cfg.train.model_copy(update={"seed": 42}),
        })
        cv = CVTrainer(cfg)
        fold0 = cv._build_fold_config(0)
        fold1 = cv._build_fold_config(1)
        assert fold0.train.seed == 42
        assert fold1.train.seed == 43

    def test_name_per_fold(self):
        from lobtrainer.training.cv_trainer import CVTrainer
        cfg = ExperimentConfig(name="experiment")
        cv = CVTrainer(cfg)
        fold = cv._build_fold_config(3)
        assert fold.name == "experiment_fold3"


class TestCVResults:
    def test_mean_metrics(self):
        from lobtrainer.training.cv_trainer import CVResults, FoldResult
        results = CVResults(folds=[
            FoldResult(0, [], [], {"val_ic": 0.3, "val_loss": 1.0}, {}, 5),
            FoldResult(1, [], [], {"val_ic": 0.5, "val_loss": 0.8}, {}, 8),
        ])
        mean = results.mean_metrics
        assert np.isclose(mean["val_ic"], 0.4)
        assert np.isclose(mean["val_loss"], 0.9)

    def test_std_metrics(self):
        from lobtrainer.training.cv_trainer import CVResults, FoldResult
        results = CVResults(folds=[
            FoldResult(0, [], [], {"val_ic": 0.3}, {}, 5),
            FoldResult(1, [], [], {"val_ic": 0.5}, {}, 8),
        ])
        std = results.std_metrics
        assert std["val_ic"] > 0

    def test_empty_folds(self):
        from lobtrainer.training.cv_trainer import CVResults
        results = CVResults(folds=[])
        assert results.mean_metrics == {}
        assert results.std_metrics == {}

    def test_summary_format(self):
        from lobtrainer.training.cv_trainer import CVResults, FoldResult
        results = CVResults(folds=[
            FoldResult(0, [], [], {"val_ic": 0.4}, {}, 5),
        ])
        s = results.summary()
        assert "1 folds" in s
        assert "val_ic" in s
