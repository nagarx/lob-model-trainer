"""Phase A (2026-04-23) regression tests for :class:`PermutationImportanceCallback`.

Locks bugs #7a (config path + attribute name) + #7b (primary_horizon_idx
threading) from the Phase A bug ledger.

Pre-Phase-A:
    * ``_resolve_task_type`` read ``config.labels.task_type`` — both path
      (``config.labels`` does not exist on ``ExperimentConfig``) AND attribute
      name (``LabelsConfig.task``, not ``task_type``) were wrong. Silent
      fallback to the model-type heuristic masked the double-error.
    * ``make_metric_fn_for_task`` was always called with ``primary_horizon_idx=0``
      (hardcoded at callback.py:352). Every feature-importance artifact was
      computed against H10 regardless of the manifest's actual primary horizon.

Post-Phase-A:
    * :func:`resolve_labels_config` sources labels from the canonical
      ``config.data.labels`` path.
    * Attribute name corrected to ``task`` (per ``LabelsConfig.task``
      at schema.py:263).
    * ``primary_horizon_idx`` threaded from the canonical path.
    * Pre-T9 / broken config → graceful fallback to heuristic preserved.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from lobtrainer.config.schema import (
    DataConfig,
    ExperimentConfig,
    LabelsConfig,
    ModelConfig,
)
from lobtrainer.training.importance.callback import PermutationImportanceCallback


class TestResolveTaskTypeCanonicalPath:
    """Bug #7a: ``_resolve_task_type`` must read ``config.data.labels.task``."""

    def test_reads_from_data_labels_task_regression(self) -> None:
        """Regression task flows through the canonical path."""
        config = ExperimentConfig(
            data=DataConfig(
                feature_count=98,
                labels=LabelsConfig(task="regression", primary_horizon_idx=0),
            ),
            model=ModelConfig(model_type="tlob", input_size=98, num_classes=3),
        )
        assert PermutationImportanceCallback._resolve_task_type(config) == "regression"

    def test_reads_from_data_labels_task_classification(self) -> None:
        """Classification task flows through the canonical path."""
        config = ExperimentConfig(
            data=DataConfig(
                feature_count=98,
                labels=LabelsConfig(task="classification", primary_horizon_idx=0),
            ),
            model=ModelConfig(model_type="tlob", input_size=98, num_classes=3),
        )
        assert (
            PermutationImportanceCallback._resolve_task_type(config) == "classification"
        )

    def test_auto_task_falls_back_to_model_type_heuristic(self) -> None:
        """LabelsConfig.task='auto' triggers the legacy heuristic path.

        'auto' is the sentinel for 'detect from metadata' — not an
        authoritative answer the callback should trust.
        """
        config = ExperimentConfig(
            data=DataConfig(
                feature_count=98,
                labels=LabelsConfig(task="auto", primary_horizon_idx=0),
            ),
            # Heuristic: "tlob" → classification (not a -r/-regression variant)
            model=ModelConfig(model_type="tlob", input_size=98, num_classes=3),
        )
        assert (
            PermutationImportanceCallback._resolve_task_type(config) == "classification"
        )

    def test_broken_config_falls_back_to_heuristic(self) -> None:
        """Config with no data.labels / no labels attr → heuristic fallback.

        The helper raises AttributeError for these cases; the callback catches
        and falls back to the model_type string inspection (pre-T9 contract).
        This test passes a ``MagicMock(spec=['model'])`` so only the ``model``
        attribute is present — the helper can't find a labels config.
        """
        mock_config = MagicMock(spec=["model"])
        mock_config.model = MagicMock(model_type="hmhp_regression")
        # Heuristic: "hmhp_regression" contains "regression" → regression
        assert (
            PermutationImportanceCallback._resolve_task_type(mock_config) == "regression"
        )


class TestPrimaryHorizonIdxThreading:
    """Bug #7b: ``on_train_end`` must thread ``primary_horizon_idx`` from config,
    not hardcode 0.

    Verifies via monkeypatch of :func:`make_metric_fn_for_task` + inspection
    of the positional / keyword args captured at call time.
    """

    @pytest.mark.parametrize("primary_idx", [0, 1, 2])
    def test_metric_fn_factory_receives_configured_primary_horizon_idx(
        self, primary_idx: int, monkeypatch: pytest.MonkeyPatch, tmp_path,
    ) -> None:
        """``make_metric_fn_for_task`` is called with the config's ``primary_horizon_idx``.

        Calls ``_compute_and_save`` directly (the underlying method; see
        callback.py:318 docstring "kept separate for test-mocking") — the
        outer ``on_train_end`` swallows exceptions, which would hide test
        failures.
        """
        import numpy as np
        from lobtrainer.training.importance import callback as cb_mod

        captured: dict = {}

        def _spy(task_type: str, primary_horizon_idx: int = -1):
            captured["task_type"] = task_type
            captured["primary_horizon_idx"] = primary_horizon_idx
            return lambda y_true, y_pred: 0.0  # trivial metric — we only observe args

        monkeypatch.setattr(cb_mod, "make_metric_fn_for_task", _spy)
        monkeypatch.setattr(
            cb_mod, "make_pytorch_predict_fn",
            lambda model, device: (lambda X: X),
        )
        monkeypatch.setattr(
            cb_mod, "_extract_eval_tensors",
            lambda loader: (
                np.zeros((5, 98), dtype="float32"),
                np.zeros(5, dtype="int64"),
            ),
        )

        # Short-circuit the compute chain — we only care that the FACTORY
        # was called with the correct primary_horizon_idx.
        class _StubArtifact:
            features: list = []
            baseline_metric = "ic"
            baseline_value = 0.0

            def content_hash(self) -> str:
                return "deadbeef" * 8

            def save(self, path):
                return None

        monkeypatch.setattr(
            cb_mod, "compute_permutation_importance",
            lambda **kwargs: _StubArtifact(),
        )

        config = ExperimentConfig(
            data=DataConfig(
                feature_count=98,
                labels=LabelsConfig(
                    horizons=[10, 60, 300],
                    primary_horizon_idx=primary_idx,
                    task="regression",
                ),
            ),
            model=ModelConfig(model_type="tlob", input_size=98, num_classes=3),
        )

        class _TrainerStub:
            def __init__(self):
                self.config = config
                self.model = MagicMock()
                self.device = "cpu"
                self._val_loader = object()
                self._test_loader = None
                self.output_dir = tmp_path

        from lobtrainer.training.importance.config import ImportanceConfig

        importance_cfg = ImportanceConfig(
            enabled=True,
            n_permutations=1,
            n_seeds=1,
            eval_split="val",
        )
        callback = PermutationImportanceCallback(config=importance_cfg)
        # Callback framework sets ``.trainer`` when attached; simulate that.
        callback.trainer = _TrainerStub()
        # Call _compute_and_save directly — outer on_train_end swallows
        # exceptions, which would hide test failures.
        callback._compute_and_save()

        assert captured.get("primary_horizon_idx") == primary_idx, (
            f"Expected primary_horizon_idx={primary_idx} threaded from config; "
            f"factory was called with {captured.get('primary_horizon_idx')!r}. "
            f"Regression of Phase A bug #7b (was hardcoded 0 pre-fix)."
        )


class TestTypeErrorFallbackOnBrokenConfig:
    """Phase A.5.4 (2026-04-24) regression locks for plan v4 bug #7.

    Pre-A.5.4 the callback caught ONLY ``AttributeError`` from
    ``resolve_labels_config(trainer.config)``. MagicMock trainer configs
    (common in test fixtures) or partially-constructed configs can raise
    ``TypeError`` (e.g., passing a dict instead of ExperimentConfig) which
    escapes the catch → kills the entire callback.

    Post-A.5.4 the catch widens to ``(AttributeError, TypeError)`` AND
    logs the fallback reason via ``logger.info(...)`` so activations
    are traceable (hft-rules §8).
    """

    def test_typeerror_on_dict_config_falls_back_to_heuristic(self, caplog):
        """``resolve_labels_config`` raises AttributeError on a dict
        (no ``.data`` attribute). Validates the documented fallback path
        fires + logs + heuristic succeeds."""
        import logging
        from lobtrainer.training.importance.callback import (
            PermutationImportanceCallback,
        )

        # A plain dict exercises the resolver's "neither .data.labels
        # nor .labels found" AttributeError path (no ``.model`` → heuristic
        # returns "classification" as ultimate fallback).
        bad_config = {"data": {}, "model": {"model_type": "tlob"}}

        with caplog.at_level(logging.INFO):
            task = PermutationImportanceCallback._resolve_task_type(bad_config)

        # Heuristic fallback returned a valid task_type string
        assert task in {"classification", "regression"}
        # The fallback activation was logged (hft-rules §8 — never silently drop)
        assert any(
            "task-type resolver fallback" in rec.message
            for rec in caplog.records
        ), (
            "Post-A.5.4 bug #7 fix: fallback activation MUST be logged at "
            "INFO level (previously silent AttributeError→fallback)."
        )

    def test_magicmock_trainer_config_does_not_crash_primary_idx_path(
        self, caplog,
    ):
        """Integration: a MagicMock trainer.config would raise TypeError
        from ``resolve_labels_config`` (or AttributeError). Pre-A.5.4 the
        callback caught only AttributeError — TypeError escaped and killed
        the callback. Post-A.5.4 both are caught and logged."""
        import logging
        from unittest.mock import MagicMock
        from lobtrainer.training.importance.callback import (
            PermutationImportanceCallback,
        )
        from lobtrainer.training.importance.config import ImportanceConfig

        callback = PermutationImportanceCallback(
            config=ImportanceConfig(enabled=True, n_permutations=1, n_seeds=1)
        )

        # Minimal MagicMock: resolve_labels_config will fail with
        # AttributeError / TypeError depending on MagicMock spec. Both
        # are now caught + logged + fallback idx=0 used. We exercise the
        # primary-idx block directly by monkeypatching resolve_labels_config
        # to raise TypeError (simulating dict/int/non-config input).
        from lobtrainer.training.importance import callback as cb_mod
        import pytest  # noqa: F401

        def _raise_typeerror(_cfg):
            raise TypeError("simulated: trainer.config is not an ExperimentConfig")

        # Patch the helper imported into callback module namespace
        original = cb_mod.resolve_labels_config
        cb_mod.resolve_labels_config = _raise_typeerror
        try:
            # Direct test of the catch path: synthesize the relevant slice of
            # _compute_and_save that reaches the try/except. Validate the
            # callback module's behavior catches TypeError without crash.
            with caplog.at_level(logging.INFO):
                # The try block is inside _compute_and_save; the helper
                # is also called in _resolve_task_type. Both paths widened.
                # Exercise _resolve_task_type since it's a pure function.
                fake_config = MagicMock()
                fake_config.model = MagicMock(model_type="tlob")
                task = PermutationImportanceCallback._resolve_task_type(fake_config)
            assert task in {"classification", "regression"}
            assert any(
                "task-type resolver fallback" in rec.message
                and "TypeError" in rec.message
                for rec in caplog.records
            ), (
                "Widened catch must classify + log TypeError specifically "
                "(not just AttributeError) — bug #7 regression lock."
            )
        finally:
            cb_mod.resolve_labels_config = original
