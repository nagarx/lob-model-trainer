"""Phase 8C-α Stage C.1 trainer wire-in (2026-04-20 post-audit round-2).

Tests the `PermutationImportanceCallback` + factories
(`make_pytorch_predict_fn`, `make_metric_fn_for_task`) in isolation
from a real training loop. A true end-to-end trainer test belongs in a
separate integration file; these tests exercise the callback's
internal flow with a minimal mock trainer + tiny synthetic data.

Coverage:
  - Factory: predict_fn handles tuple outputs (HMHP-style) correctly
  - Factory: metric_fn_for_task dispatches by task_type
  - Callback: gate respected (disabled config → no-op)
  - Callback: end-to-end happy path writes a valid FeatureImportanceArtifact
  - Callback: graceful error when trainer state is missing
  - Config: ExperimentConfig parses `importance:` dict → ImportanceConfig
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pytest
import torch

from hft_contracts import FeatureImportanceArtifact
from lobtrainer.config.schema import ExperimentConfig
from lobtrainer.training.callbacks import CallbackList
from lobtrainer.training.importance.callback import (
    PermutationImportanceCallback,
    _extract_eval_tensors,
    make_metric_fn_for_task,
    make_pytorch_predict_fn,
)
from lobtrainer.training.importance.config import ImportanceConfig


# ---------------------------------------------------------------------------
# Fixtures: a tiny linear model + a tiny synthetic loader
# ---------------------------------------------------------------------------


class _TinyLinearModel(torch.nn.Module):
    """Minimal PyTorch model: flattens (N, T, F) → linear → (N, H).

    Used to drive predict_fn + metric_fn through a realistic forward
    pass without pulling in the full LOB model zoo.
    """

    def __init__(self, n_features: int, seq_len: int, out_dim: int = 1):
        super().__init__()
        self.fc = torch.nn.Linear(seq_len * n_features, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, T, F) → (N, T*F) → (N, out_dim)
        return self.fc(x.reshape(x.shape[0], -1))


class _TinyTupleModel(torch.nn.Module):
    """Model with tuple-output interface (mimics HMHP)."""

    def __init__(self, n_features: int, seq_len: int):
        super().__init__()
        self.primary = torch.nn.Linear(seq_len * n_features, 1)
        self.aux = torch.nn.Linear(seq_len * n_features, 1)

    def forward(self, x: torch.Tensor):
        x_flat = x.reshape(x.shape[0], -1)
        return (self.primary(x_flat), self.aux(x_flat), x_flat.mean(dim=1, keepdim=True))


def _make_tiny_loader(N: int = 20, T: int = 5, F: int = 3, seed: int = 0):
    """Produce a single-batch DataLoader-like iterator yielding (X, y)."""
    rng = np.random.RandomState(seed)
    X = rng.randn(N, T, F).astype(np.float32)
    y = X.mean(axis=1)[:, 0:1] * 2.0 + 0.1 * rng.randn(N, 1).astype(np.float32)
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)

    # Yield a single batch to simulate a DataLoader.
    class _SingleBatchLoader:
        def __iter__(self):
            yield (X_tensor, y_tensor)
    return _SingleBatchLoader()


class _MockTrainer:
    """Minimal trainer stand-in exposing the attributes the callback reads."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        output_dir: Path,
        test_loader: Any,
        config: ExperimentConfig,
        val_loader: Optional[Any] = None,
    ):
        self.model = model
        self.device = device
        self.output_dir = output_dir
        self._test_loader = test_loader
        self._val_loader = val_loader
        self.config = config


# ---------------------------------------------------------------------------
# predict_fn factory
# ---------------------------------------------------------------------------


class TestMakePytorchPredictFn:
    def test_simple_linear_output(self):
        """Linear model returns (N, H); predict_fn passes through."""
        m = _TinyLinearModel(n_features=3, seq_len=5, out_dim=1)
        device = torch.device("cpu")
        predict = make_pytorch_predict_fn(m, device)
        X = np.random.RandomState(0).randn(10, 5, 3).astype(np.float32)
        preds = predict(X)
        assert preds.shape == (10, 1), f"expected (10, 1), got {preds.shape}"
        assert preds.dtype in (np.float32, np.float64)

    def test_tuple_output_model_returns_primary(self):
        """HMHP-style model returns tuple; predict_fn extracts element[0]."""
        m = _TinyTupleModel(n_features=3, seq_len=5)
        device = torch.device("cpu")
        predict = make_pytorch_predict_fn(m, device)
        X = np.random.RandomState(0).randn(7, 5, 3).astype(np.float32)
        preds = predict(X)
        # Primary output shape (N, 1) — aux + confirmation not included
        assert preds.shape == (7, 1), (
            f"Tuple-model predict_fn must extract element[0]; got shape {preds.shape}"
        )

    def test_deterministic_given_weights(self):
        """Same weights + same input → same preds (model.eval + no_grad)."""
        torch.manual_seed(42)
        m = _TinyLinearModel(n_features=3, seq_len=5)
        device = torch.device("cpu")
        predict = make_pytorch_predict_fn(m, device)
        X = np.random.RandomState(0).randn(5, 5, 3).astype(np.float32)
        p1 = predict(X)
        p2 = predict(X)
        np.testing.assert_array_equal(p1, p2)


# ---------------------------------------------------------------------------
# metric_fn factory
# ---------------------------------------------------------------------------


class TestMakeMetricFnForTask:
    def test_regression_task_returns_ic(self):
        """Regression metric_fn returns spearman_ic on primary horizon."""
        metric = make_metric_fn_for_task("regression", primary_horizon_idx=0)
        rng = np.random.RandomState(0)
        y = rng.randn(50, 3).astype(np.float64)
        preds_signal = y + 0.01 * rng.randn(50, 3)  # near-perfect prediction
        preds_noise = rng.randn(50, 3)
        ic_signal = metric(preds_signal, y)
        ic_noise = metric(preds_noise, y)
        assert ic_signal > 0.9, f"signal IC should be ~1.0, got {ic_signal}"
        assert abs(ic_noise) < 0.3, f"noise IC should be near zero, got {ic_noise}"

    def test_classification_task_returns_accuracy(self):
        """Classification metric_fn returns argmax-accuracy."""
        metric = make_metric_fn_for_task("classification")
        # Perfect predictions
        logits = np.array([
            [1.0, 0.0, 0.0],  # argmax → 0
            [0.0, 1.0, 0.0],  # argmax → 1
            [0.0, 0.0, 1.0],  # argmax → 2
        ])
        y_perfect = np.array([0, 1, 2])
        assert metric(logits, y_perfect) == 1.0
        # Wrong predictions
        y_wrong = np.array([2, 0, 1])
        assert metric(logits, y_wrong) == 0.0

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError, match="Unknown task_type"):
            make_metric_fn_for_task("nonsense")


# ---------------------------------------------------------------------------
# _extract_eval_tensors
# ---------------------------------------------------------------------------


class TestExtractEvalTensors:
    def test_stacks_batches_correctly(self):
        X1 = torch.zeros(5, 5, 3)
        y1 = torch.zeros(5, 1)
        X2 = torch.ones(7, 5, 3)
        y2 = torch.ones(7, 1)

        class _Loader:
            def __iter__(self):
                yield (X1, y1)
                yield (X2, y2)

        X_np, y_np = _extract_eval_tensors(_Loader())
        assert X_np.shape == (12, 5, 3)
        assert y_np.shape == (12, 1)
        assert X_np[:5].mean() == 0.0
        assert X_np[5:].mean() == 1.0

    def test_handles_sample_weighted_3tuple(self):
        """Batches of shape (X, y, sample_weights) — weights discarded."""
        X = torch.zeros(4, 3, 2)
        y = torch.zeros(4, 1)
        sw = torch.ones(4)

        class _Loader:
            def __iter__(self):
                yield (X, y, sw)

        X_np, y_np = _extract_eval_tensors(_Loader())
        assert X_np.shape == (4, 3, 2)
        assert y_np.shape == (4, 1)

    def test_empty_loader_raises(self):
        class _Empty:
            def __iter__(self):
                return iter([])

        with pytest.raises(ValueError, match="ZERO batches"):
            _extract_eval_tensors(_Empty())


# ---------------------------------------------------------------------------
# PermutationImportanceCallback — gate + end-to-end happy path
# ---------------------------------------------------------------------------


class TestPermutationImportanceCallback:
    def test_gate_disabled_is_noop(self, tmp_path: Path):
        """Gate check: disabled config → no-op, no artifact written."""
        cfg_imp = ImportanceConfig(enabled=False)
        callback = PermutationImportanceCallback(cfg_imp)
        trainer = _MockTrainer(
            model=_TinyLinearModel(3, 5),
            device=torch.device("cpu"),
            output_dir=tmp_path,
            test_loader=_make_tiny_loader(),
            config=ExperimentConfig(name="test", output_dir=str(tmp_path)),
        )
        # Register via callback list so trainer is set
        CallbackList([callback]).set_trainer(trainer)

        callback.on_train_end()

        assert not (tmp_path / "feature_importance_v1.json").exists()

    def test_happy_path_writes_valid_artifact(self, tmp_path: Path):
        """End-to-end: enabled callback on mock trainer writes a valid
        FeatureImportanceArtifact JSON that loads back correctly."""
        cfg_imp = ImportanceConfig(
            enabled=True,
            n_permutations=5,  # small for test speed
            n_seeds=1,
            subsample=-1,  # use full (N=20)
            seed=42,
            eval_split="test",
        )
        callback = PermutationImportanceCallback(cfg_imp)

        # Minimal config with regression task_type inference
        config = ExperimentConfig(
            name="test_happy", output_dir=str(tmp_path),
        )
        # Phase A.5.3h (2026-04-24): ModelConfig is now frozen Pydantic.
        # Phase A.5.3i (2026-04-24 KEYSTONE): ExperimentConfig is now also
        # frozen. Two-layer pattern — build the new ModelConfig via inner
        # model_copy, swap into outer ExperimentConfig via outer model_copy.
        # Use ModelType.HMHP_REGRESSION (.value='hmhp_regression', contains
        # the substring 'regression' that the task-type resolver heuristic
        # keys on — matches legacy behavior without relying on a synthetic
        # string that ModelType Enum would reject under strict).
        config = config.model_copy(update={
            "model": config.model.model_copy(
                update={"model_type": "hmhp_regression"}
            ),
        })
        trainer = _MockTrainer(
            model=_TinyLinearModel(n_features=3, seq_len=5, out_dim=1),
            device=torch.device("cpu"),
            output_dir=tmp_path,
            test_loader=_make_tiny_loader(N=20, T=5, F=3, seed=0),
            config=config,
        )
        CallbackList([callback]).set_trainer(trainer)

        callback.on_train_end()

        # Assert artifact file exists
        artifact_path = tmp_path / "feature_importance_v1.json"
        assert artifact_path.exists(), (
            f"Callback must write feature_importance_v1.json to output_dir; "
            f"contents of {tmp_path}: {list(tmp_path.iterdir())}"
        )

        # Load back + verify structure
        reloaded = FeatureImportanceArtifact.load(artifact_path)
        assert reloaded.schema_version == "2"
        assert reloaded.method == "permutation"
        assert len(reloaded.features) == 3  # n_features from X.shape[-1]
        for feat in reloaded.features:
            assert np.isfinite(feat.importance_mean)

    def test_callback_failure_does_not_raise(self, tmp_path: Path, caplog):
        """If importance computation fails (e.g., degenerate data), the
        callback logs + swallows — training run is NOT killed by an
        observation-tier failure.
        """
        import logging as _logging
        cfg_imp = ImportanceConfig(
            enabled=True,
            n_permutations=5,
            n_seeds=1,
            subsample=-1,
            seed=42,
        )
        callback = PermutationImportanceCallback(cfg_imp)

        # Trainer with MISSING test_loader → callback fails to resolve
        # eval split → logs warning, does not raise.
        config = ExperimentConfig(name="test_fail", output_dir=str(tmp_path))
        trainer = _MockTrainer(
            model=_TinyLinearModel(3, 5),
            device=torch.device("cpu"),
            output_dir=tmp_path,
            test_loader=None,  # missing!
            config=config,
        )
        CallbackList([callback]).set_trainer(trainer)

        # Should not raise
        with caplog.at_level(_logging.WARNING):
            callback.on_train_end()

        # Should have logged a warning about missing loader
        warn_messages = [r.message for r in caplog.records
                         if r.levelno >= _logging.WARNING]
        assert any("loader is" in m.lower() or "none" in m.lower()
                   for m in warn_messages), (
            f"Expected warning about missing eval loader. Got: {warn_messages}"
        )


# ---------------------------------------------------------------------------
# ExperimentConfig integration — YAML-round-trip (dict → ImportanceConfig)
# ---------------------------------------------------------------------------


class TestExperimentConfigImportanceField:
    def test_default_none(self):
        """Default ExperimentConfig has importance=None (zero overhead)."""
        c = ExperimentConfig()
        assert c.importance is None

    def test_dict_coerces_to_importance_config(self):
        """YAML-load sim: passing a dict gets coerced to ImportanceConfig
        via __post_init__._coerce_importance."""
        c = ExperimentConfig(importance={"enabled": True, "n_permutations": 77})
        assert c.importance is not None
        assert c.importance.enabled is True
        assert c.importance.n_permutations == 77
        # Defaults preserved on other fields
        assert c.importance.method == "permutation"

    def test_pre_constructed_passes_through(self):
        """Pre-constructed ImportanceConfig survives __post_init__ unchanged."""
        imp = ImportanceConfig(enabled=True, seed=999)
        c = ExperimentConfig(importance=imp)
        assert c.importance is imp
        assert c.importance.seed == 999

    def test_invalid_type_raises(self):
        """Garbage input raises ValidationError with actionable message.

        Phase A.5.3i (2026-04-24 KEYSTONE): the ``_coerce_importance(self)``
        helper (which raised TypeError) was replaced by a
        ``@field_validator(mode='before')`` on ExperimentConfig. Per
        Pydantic convention, validators raise ``ValueError`` (converted
        to ``ValidationError`` by Pydantic) rather than TypeError —
        unifies the failure mode across every config boundary (callers
        catch ValidationError uniformly).
        """
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="must be None, a dict, or"):
            ExperimentConfig(importance=42)  # int, not valid
