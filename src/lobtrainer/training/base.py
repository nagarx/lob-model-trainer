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

    The contract is the four methods that ``scripts/train.py`` calls
    polymorphically: ``train``, ``evaluate(split)``, ``save_checkpoint``,
    ``load_checkpoint``. These are sufficient for the entry-point dispatch
    use case.

    Out of scope for the Protocol (intentionally):
    - ``setup`` — internal lifecycle; ``train`` is responsible for
      ensuring it runs first if needed.
    - ``export_signals`` — sklearn ``SimpleModelTrainer`` exposes it
      directly; the PyTorch path uses the separate
      ``scripts/export_signals.py`` + ``SignalExporter``. Not part
      of the unified lifecycle today.

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

    def load_checkpoint(self, path: Path) -> None:
        """Restore trainer state from a checkpoint file.

        Out of scope for Phase Q (deferred to Phase T): restoring
        callback state (early-stopping patience, plateau counters)
        and RNG state across torch / numpy / random.
        """
        ...
