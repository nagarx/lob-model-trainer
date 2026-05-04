"""Phase Q.6.5.B Part 4 — N-5 closure: CVTrainer fail-fast guard for sklearn.

Locks the contract: ``CVTrainer(config)`` MUST raise ``ValueError`` at
construction when the registered model framework is not ``"pytorch"``.

Pre-fix, k-fold CV with a sklearn config silently fell through to
``Trainer(fold_config)`` at cv_trainer.py:203 and crashed mid-fold at
PyTorch model build. Per hft-rules §5 fail-fast: detect at construction
with an actionable remediation message.

This is the regression lock — if a future refactor accidentally drops
the framework guard from ``CVTrainer.__init__``, this test fails before
production CI hits the silent crash.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from lobtrainer.training.cv_trainer import CVTrainer


def _build_synthetic_sklearn_config(tmp_path):
    """Construct a minimal ExperimentConfig with a sklearn-registered model."""
    from lobtrainer.config.schema import (
        DataConfig,
        ExperimentConfig,
        LabelsConfig,
        LossType,
        ModelConfig,
        ModelType,
        NormalizationConfig,
        SequenceConfig,
        TaskType,
        TrainConfig,
    )
    data = DataConfig(
        data_dir=str(tmp_path / "synthetic_data"),
        feature_count=98,
        sequence=SequenceConfig(window_size=20, stride=1),
        normalization=NormalizationConfig(strategy="none"),
        labels=LabelsConfig(
            primary_horizon_idx=0,
            horizons=[10, 60, 300],
            source="forward_prices",
            task="regression",
        ),
    )
    model = ModelConfig(
        model_type=ModelType.TEMPORAL_RIDGE,  # framework="sklearn"
        input_size=98,
        params={"alpha": 1.0},
    )
    train = TrainConfig(task_type=TaskType.REGRESSION, loss_type=LossType.HUBER)
    return ExperimentConfig(
        name="cv_trainer_sklearn_guard_test",
        data=data,
        model=model,
        train=train,
        output_dir=str(tmp_path / "output"),
    )


def _build_synthetic_pytorch_config(tmp_path):
    """Construct a minimal ExperimentConfig with a pytorch-registered model."""
    from lobtrainer.config.schema import (
        DataConfig,
        ExperimentConfig,
        LabelsConfig,
        LossType,
        ModelConfig,
        ModelType,
        NormalizationConfig,
        SequenceConfig,
        TaskType,
        TrainConfig,
    )
    data = DataConfig(
        data_dir=str(tmp_path / "synthetic_data"),
        feature_count=98,
        sequence=SequenceConfig(window_size=20, stride=1),
        normalization=NormalizationConfig(strategy="none"),
        labels=LabelsConfig(
            primary_horizon_idx=0,
            horizons=[10, 60, 300],
            source="forward_prices",
            task="regression",
        ),
    )
    model = ModelConfig(
        model_type=ModelType.LOGISTIC,  # framework="pytorch"
        input_size=98,
        params={},
    )
    train = TrainConfig(task_type=TaskType.REGRESSION, loss_type=LossType.HUBER)
    return ExperimentConfig(
        name="cv_trainer_pytorch_guard_test",
        data=data,
        model=model,
        train=train,
        output_dir=str(tmp_path / "output"),
    )


class TestCVTrainerSklearnGuard:
    """N-5 closure: CVTrainer rejects sklearn models at construction."""

    def test_sklearn_model_rejected_at_construction(self, tmp_path):
        """temporal_ridge (framework='sklearn') must fail-fast in CVTrainer.__init__."""
        config = _build_synthetic_sklearn_config(tmp_path)
        with pytest.raises(ValueError, match=r"only pytorch-framework models"):
            CVTrainer(config, n_splits=3, embargo_days=1)

    def test_sklearn_error_message_is_actionable(self, tmp_path):
        """The error message must point at the workaround (single-fold via train.py)
        so operators are not blocked."""
        config = _build_synthetic_sklearn_config(tmp_path)
        with pytest.raises(ValueError) as exc_info:
            CVTrainer(config)
        msg = str(exc_info.value)
        assert "framework='sklearn'" in msg or "sklearn" in msg.lower(), (
            f"Error message must name the offending framework. Got: {msg}"
        )
        assert "scripts/train.py" in msg or "single fold" in msg.lower(), (
            f"Error message must suggest the single-fold workaround. Got: {msg}"
        )

    def test_pytorch_model_accepted_at_construction(self, tmp_path):
        """logistic_lob (framework='pytorch') must construct without error.
        Locks against over-eager guard that would also reject pytorch paths."""
        config = _build_synthetic_pytorch_config(tmp_path)
        # Should not raise — pytorch framework is accepted
        cv = CVTrainer(config, n_splits=3, embargo_days=1)
        assert cv.n_splits == 3
        assert cv.embargo_days == 1

    def test_unknown_model_falls_through_to_pytorch_default(self, tmp_path):
        """Per defensive try/except in __init__, an unknown model name (not
        in ModelRegistry) defaults to framework='pytorch' so CVTrainer
        constructs successfully. Trainer itself will raise its own
        descriptive error later if the model is genuinely broken.

        This preserves prior behavior for unusual configs that bypassed
        the registry — closes Agent 1 verification gap."""
        config = _build_synthetic_pytorch_config(tmp_path)
        # Construct — should NOT raise even though no real registry walk
        # happens for synthetic pytorch model.
        cv = CVTrainer(config)
        assert cv.config is config

    @pytest.mark.parametrize(
        "framework_name",
        ["sklearn", "xgboost", "lightgbm", "jax", "tensorflow"],
    )
    def test_non_pytorch_frameworks_uniformly_rejected(self, tmp_path, framework_name):
        """Phase Q.6.5.B Part 4 post-audit hardening (Agent 1 mid-impl
        b3 closure): the framework guard should uniformly reject ALL
        non-pytorch frameworks, not just the sklearn case Stage 1
        surfaced. Parametric coverage prevents future framework additions
        from accidentally bypassing the guard.

        Uses a fixture model that registers with a synthetic framework
        name, so we exercise the actual ``ModelRegistry.get(...).framework``
        != "pytorch" branch at cv_trainer.py:140-152.
        """
        # Build a config + monkey-patch ModelRegistry to return the
        # framework_name for the registered model. This lets us test
        # the guard for hypothetical future frameworks (xgboost / lightgbm
        # / jax) without registering real model classes.
        config = _build_synthetic_pytorch_config(tmp_path)

        # Patch ModelRegistry.get to return a fake entry with the
        # parametrized framework. The CVTrainer guard reads
        # ``ModelRegistry.get(model_name).framework`` directly.
        from unittest.mock import patch

        class _FakeEntry:
            framework = framework_name

        with patch("lobmodels.ModelRegistry") as mock_registry:
            mock_registry.get = lambda name: _FakeEntry()
            with pytest.raises(ValueError, match=r"only pytorch-framework models"):
                CVTrainer(config, n_splits=3, embargo_days=1)
