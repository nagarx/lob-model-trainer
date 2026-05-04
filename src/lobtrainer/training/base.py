"""BaseTrainer Protocol — common lifecycle for any trainer in the pipeline.

Phase Q (2026-05-04): formalizes the de-facto contract that
``Trainer`` (PyTorch) and ``SimpleModelTrainer`` (sklearn) already
satisfy. This Protocol is the type the framework-aware factory
``create_trainer`` returns regardless of which concrete trainer is
selected based on the registered ``framework`` field
(``"pytorch"`` → ``Trainer``, ``"sklearn"`` → ``SimpleModelTrainer``).

Design:
- Structural typing via ``typing.Protocol``: no inheritance forced;
  any class with the matching method signatures conforms automatically.
- ``@runtime_checkable`` enables ``isinstance(trainer, BaseTrainer)`` for
  test-time assertions (the runtime check verifies attribute presence
  only, NOT signatures, per typing.Protocol semantics).
- The ``config`` attribute is typed as ``Any`` to avoid pulling
  ``ExperimentConfig`` into ``base.py`` (which would create a circular
  import via ``lobtrainer.training.trainer``).

Why Protocol over ABC: future trainer backends (XGBoost-direct,
LightGBM, JAX, remote-distributed) can satisfy the contract without
touching ``base.py`` or being forced into a base-class hierarchy.
Inheritance would couple every backend to a Python class definition;
structural typing decouples them.

Reference: PEP 544 (Protocols), PEP 604 (typing.runtime_checkable).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Protocol, runtime_checkable


__all__ = ["BaseTrainer"]


@runtime_checkable
class BaseTrainer(Protocol):
    """Common lifecycle hooks for any trainer reachable through
    ``create_trainer``.

    Phase Q.6.5.B (2026-05-04 night): contract extended with
    ``export_signals(split, *, output_dir, calibration) -> Path``.
    Both ``Trainer`` (PyTorch — delegates to ``SignalExporter``) and
    ``SimpleModelTrainer`` (sklearn — direct in-memory predict +
    metadata emit via ``build_signal_metadata`` + Phase X.1.A
    ``build_compatibility_contract`` SSoT) satisfy the new method.
    This unifies the entry-point dispatch surface so
    ``scripts/export_signals.py`` and the orchestrator
    (``hft-ops run``) can use one polymorphic call.

    The contract is the FIVE methods that ``scripts/train.py`` and
    ``scripts/export_signals.py`` call polymorphically: ``train``,
    ``evaluate(split)``, ``save_checkpoint``, ``load_checkpoint``,
    ``export_signals``. Sufficient for the canonical entry-point
    dispatch use case.

    Out of scope for the Protocol (intentionally):
    - ``setup`` — internal lifecycle; ``train`` is responsible for
      ensuring it runs first if needed.

    Note: ``runtime_checkable`` ``isinstance(obj, BaseTrainer)`` checks
    method-name presence only (per typing.Protocol semantics).
    Signature equivalence is the responsibility of static type checkers,
    not runtime.
    """

    def train(self) -> Dict[str, Any]:
        """Run training to convergence.

        Returns:
            Dictionary of training-result metadata. PyTorch returns
            ``{"total_epochs", "best_val_metric", "best_epoch", ...}``.
            Sklearn returns the same shape with sentinel epoch values
            (``total_epochs=1``, ``best_epoch=0``) so callers that read
            those keys generically still work.
        """
        ...

    def evaluate(self, split: str = "test") -> Dict[str, float]:
        """Evaluate the trained model on the named split.

        Args:
            split: One of ``"train"`` / ``"val"`` / ``"test"``.

        Returns:
            Metrics for that split (regression: r2/ic/mae/...;
            classification: accuracy/macro_f1/...).
        """
        ...

    def save_checkpoint(self, path: Optional[Path] = None) -> Path:
        """Persist trainer state.

        PyTorch writes ``best.pt`` (torch.save with model_state_dict
        + optimizer_state_dict + epoch + metrics); sklearn writes
        ``best.pkl`` (pickle with fitted estimator + config).

        Args:
            path: Override default location. ``None`` uses
                ``<output_dir>/checkpoints/best.{pt|pkl}``.

        Returns:
            Absolute path actually written.
        """
        ...

    def load_checkpoint(
        self,
        path: Path,
        load_optimizer: bool = True,
    ) -> None:
        """Restore trainer state from a checkpoint file.

        Phase Q.6.5.B (2026-05-04 night): signature unified with
        ``Trainer.load_checkpoint``. The ``load_optimizer`` kwarg is a
        no-op on the sklearn path (``SimpleModelTrainer`` has no optimizer
        to load — explicit no-op rather than TypeError-on-call). Closes
        N-6 signature drift surfaced by Q.6.5.A audit.

        Args:
            path: Checkpoint path. PyTorch reads ``.pt`` torch.save dict;
                sklearn reads ``.pkl`` pickle + ``.config.json`` sidecar.
            load_optimizer: PyTorch-only — when ``False``, skip the
                optimizer state load (saves time during inference-only
                use cases like signal export). Sklearn ignores this
                kwarg (no optimizer state in the pickle).

        Out of scope for Phase Q (deferred to Phase T): restoring
        callback state (early-stopping patience, plateau counters)
        and RNG state across torch / numpy / random.
        """
        ...

    def export_signals(
        self,
        split: str = "test",
        *,
        output_dir: Optional[Path] = None,
        calibration: str = "none",
    ) -> Path:
        """Export predicted signals + ``signal_metadata.json`` to
        ``<output_dir or config.output_dir>/signals/<split>/``.

        Phase Q.6.5.B (2026-05-04 night): added to the Protocol to close
        the historical asymmetry where ``SimpleModelTrainer`` exposed
        ``export_signals`` directly while ``Trainer`` required manually
        constructing ``SignalExporter``. Now both paths satisfy the
        same Protocol method.

        PyTorch path delegates to
        ``lobtrainer.export.exporter.SignalExporter`` which runs
        inference through ``trainer.get_loader(split)``. Sklearn path
        emits predictions from in-memory ``self._X_test`` / ``_y_test``
        / ``_spreads_test`` / ``_prices_test`` populated by ``setup()``.

        Args:
            split: Data split — ``"val"`` or ``"test"``. Training split
                is refused (PyTorch path: DataLoader drop_last=True
                alignment mismatch; sklearn path: train arrays not
                exposed). Sklearn currently restricts to ``"test"``
                only — extending to ``"val"`` is a follow-up extension
                (val arrays loaded but ``_spreads_val`` / ``_prices_val``
                not yet extracted).
            output_dir: Override default. ``None`` uses
                ``<config.output_dir>/signals/<split>/``.
            calibration: ``"none"`` (default) or ``"variance_match"``
                (regression-only; classification is no-op + WARN).
                Sklearn currently rejects non-``"none"`` values per
                hft-rules §5 fail-fast — variance_match is not yet
                wired for the sklearn pipeline.

        Returns:
            Output directory path (``Path``).

        Raises:
            ValueError: Invalid split or unsupported calibration value.
            RuntimeError: setup() not called (PyTorch DataLoader
                missing for the target split).
        """
        ...
