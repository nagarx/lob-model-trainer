"""Phase Q.5 regression tests — `create_trainer` framework-aware dispatch.

Pre-Q.5 ``create_trainer`` always returned ``Trainer``, regardless of
``model_type``. For sklearn-registered models (TemporalRidge,
TemporalGradBoost), this caused TypeError at construction or
AttributeError at ``model.parameters()`` calls.

Post-Q.5 ``create_trainer`` inspects ``ModelRegistry.get(name).framework``
and returns the appropriate trainer:
  * ``framework="pytorch"`` → ``Trainer``
  * ``framework="sklearn"`` → ``SimpleModelTrainer`` (via ``from_config``)

These tests parametrize over the live registry to lock the contract.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from lobmodels import ModelRegistry
from lobtrainer.training import BaseTrainer, create_trainer
from lobtrainer.training.simple_trainer import SimpleModelTrainer
from lobtrainer.training.trainer import Trainer


def _build_minimal_config(
    tmp_path: Path,
    *,
    model_type: str,
    params: dict | None = None,
    feature_count: int = 98,
    input_size: int | None = None,
    window_size: int = 20,
):
    """Build a minimal ExperimentConfig for dispatch testing.

    Doesn't require real data on disk because ``create_trainer``
    itself doesn't call ``setup()`` — only the dispatch is exercised.
    """
    from lobtrainer.config.schema import (
        ExperimentConfig, DataConfig, ModelConfig, TrainConfig,
        LabelsConfig, SequenceConfig, NormalizationConfig,
        ModelType, TaskType, LossType,
    )
    if params is None:
        params = {}
    if input_size is None:
        input_size = feature_count

    data = DataConfig(
        data_dir=str(tmp_path / "fake_data"),
        feature_count=feature_count,
        sequence=SequenceConfig(window_size=window_size, stride=1),
        normalization=NormalizationConfig(strategy="none"),
        labels=LabelsConfig(
            primary_horizon_idx=0,
            horizons=[10, 60, 300],
            source="forward_prices",
            task="regression",
        ),
    )
    model = ModelConfig(
        model_type=getattr(ModelType, model_type.upper()),
        input_size=input_size,
        params=params,
    )
    train = TrainConfig(task_type=TaskType.REGRESSION, loss_type=LossType.HUBER)
    return ExperimentConfig(
        name=f"test_{model_type}",
        data=data,
        model=model,
        train=train,
        output_dir=str(tmp_path / "output"),
    )


# ---------------------------------------------------------------------------
# Sklearn dispatch — TemporalRidge + TemporalGradBoost
# ---------------------------------------------------------------------------


class TestSklearnDispatch:
    """Sklearn models route through SimpleModelTrainer."""

    def test_temporal_ridge_routes_to_simple_trainer(self, tmp_path):
        config = _build_minimal_config(
            tmp_path, model_type="temporal_ridge",
            params={"alpha": 1.0},
        )
        trainer = create_trainer(config)
        assert isinstance(trainer, SimpleModelTrainer), (
            f"Expected SimpleModelTrainer for temporal_ridge, got {type(trainer)}"
        )
        assert isinstance(trainer, BaseTrainer), (
            "Returned trainer must satisfy BaseTrainer Protocol"
        )

    def test_temporal_gradboost_routes_to_simple_trainer(self, tmp_path):
        config = _build_minimal_config(
            tmp_path, model_type="temporal_gradboost",
            params={"n_estimators": 50, "max_depth": 3},
        )
        trainer = create_trainer(config)
        assert isinstance(trainer, SimpleModelTrainer)
        assert isinstance(trainer, BaseTrainer)

    def test_sklearn_drops_callbacks_with_info_log(self, tmp_path, caplog):
        """Sklearn trainer doesn't run callbacks; create_trainer drops them
        with an INFO log so the caller sees what was discarded."""
        from lobtrainer.training.callbacks import EarlyStopping
        config = _build_minimal_config(
            tmp_path, model_type="temporal_ridge",
            params={"alpha": 1.0},
        )
        callbacks = [EarlyStopping(patience=5, metric="val_loss", mode="min")]

        with caplog.at_level(logging.INFO, logger="lobtrainer.training.trainer"):
            trainer = create_trainer(config, callbacks=callbacks)

        assert isinstance(trainer, SimpleModelTrainer)
        # The INFO log should mention "Dropping" + count.
        dropped_msgs = [r for r in caplog.records if "Dropping" in r.message]
        assert dropped_msgs, (
            f"Expected an INFO log about dropping callbacks, got: "
            f"{[r.message for r in caplog.records]}"
        )

    def test_sklearn_from_config_round_trip(self, tmp_path):
        """The SimpleModelTrainer returned by create_trainer carries a
        reference to the original ExperimentConfig (Phase Q.6 traceability)."""
        config = _build_minimal_config(
            tmp_path, model_type="temporal_ridge",
            params={"alpha": 1.0},
        )
        trainer = create_trainer(config)
        # SimpleModelTrainer.from_config sets self.config for traceability.
        assert trainer.config is config


# ---------------------------------------------------------------------------
# PyTorch dispatch — preserves all existing PyTorch trainer behavior
# ---------------------------------------------------------------------------


class TestPyTorchDispatch:
    """PyTorch models route through Trainer (preserved from pre-Q.5)."""

    @pytest.mark.parametrize(
        "model_type",
        # Phase X.1 v2 (2026-05-04): "mlplob" added to ModelType enum
        # (closes F-3). Pre-X.1 v2 was registered in lob-models but absent
        # from the trainer's enum.
        ["lstm", "gru", "logistic", "deeplob", "tlob", "mlplob"],
    )
    def test_pytorch_models_route_to_trainer(self, tmp_path, model_type):
        """All registered PyTorch models with a ModelType enum entry
        route to Trainer."""
        config = _build_minimal_config(tmp_path, model_type=model_type)
        trainer = create_trainer(config)
        assert isinstance(trainer, Trainer), (
            f"Expected Trainer for {model_type}, got {type(trainer)}"
        )

    def test_mlplob_params_builder_returns_arch_keys(self, tmp_path):
        """Phase X.1 v2 X.1.I: MLPLOB params builder mirrors lobmodels.MLPLOBConfig.

        Verifies the trainer schema bridge produces a dict whose keys match
        what ``MLPLOBConfig(**params)`` accepts (no missing required keys,
        no extra-rejected keys).
        """
        config = _build_minimal_config(tmp_path, model_type="mlplob")
        params = config.model._build_params_from_legacy()
        # Required architectural keys for lobmodels.MLPLOBConfig
        for key in (
            "num_features", "num_classes", "hidden_dim", "num_layers",
            "mlp_expansion", "use_bin", "dropout", "dataset_type", "task_type",
        ):
            assert key in params, f"MLPLOB params builder missing key: {key}"
        # Verify lobmodels.MLPLOBConfig accepts the params dict
        from lobmodels.config.base import MLPLOBConfig
        cfg = MLPLOBConfig(**{k: v for k, v in params.items() if k in MLPLOBConfig.__dataclass_fields__})
        assert cfg.hidden_dim == 40  # default from trainer mlplob_hidden_dim
        assert cfg.num_layers == 3

    def test_pytorch_default_callbacks_attached(self, tmp_path):
        """PyTorch path retains the default-callbacks block."""
        config = _build_minimal_config(tmp_path, model_type="lstm")
        trainer = create_trainer(config)
        # CallbackList has `.callbacks: List[Callback]` (not iterable
        # directly) — assert the underlying list is non-empty.
        assert trainer.callbacks is not None
        assert len(trainer.callbacks.callbacks) > 0


# ---------------------------------------------------------------------------
# Registry coverage — every registered model must be reachable
# ---------------------------------------------------------------------------


class TestRegistryCoverage:
    """Ground-truth lock: every registered model has a known framework
    and dispatch must succeed."""

    def test_all_registered_models_have_known_framework(self):
        """Each registered model declares framework ∈ {"pytorch", "sklearn"}.

        Locks the registry so future @register additions can't silently
        introduce a third framework that the dispatch doesn't handle."""
        valid_frameworks = {"pytorch", "sklearn"}
        for name in ModelRegistry.list_models():
            entry = ModelRegistry.get(name)
            assert entry.framework in valid_frameworks, (
                f"Model '{name}' has framework '{entry.framework}'; "
                f"expected one of {valid_frameworks}. If adding a new "
                f"framework, extend `create_trainer` first."
            )

    def test_framework_field_used_for_dispatch(self):
        """Sanity: the registry has at least one of each framework so
        the dispatch test parametrization covers both branches."""
        frameworks_seen = {
            ModelRegistry.get(n).framework for n in ModelRegistry.list_models()
        }
        assert "pytorch" in frameworks_seen
        assert "sklearn" in frameworks_seen
