"""Phase Q.4 regression tests — `BaseTrainer` Protocol conformance.

Verifies that both production trainer classes (`Trainer`, PyTorch path;
`SimpleModelTrainer`, sklearn path) satisfy the `BaseTrainer` Protocol.
This is the structural-typing foundation for the framework-aware
factory dispatch landing in Q.5.

Per typing.Protocol with @runtime_checkable, `isinstance(obj, Protocol)`
checks method-NAME presence only — signature equivalence requires a
static type checker (mypy, pyright). These runtime tests are sufficient
for the dispatch use case: `create_trainer` only needs to know which
methods exist; the caller (scripts/train.py, hft-ops) calls them by name.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lobtrainer.training import BaseTrainer
from lobtrainer.training.simple_trainer import SimpleModelTrainer
from lobtrainer.training.trainer import Trainer


class TestProtocolConformance:
    """Q.4: production trainers structurally satisfy BaseTrainer."""

    def test_trainer_class_satisfies_protocol(self) -> None:
        """The PyTorch `Trainer` exposes all five required methods.

        Phase Q.6.5.B (2026-05-04 night): contract extended with
        ``export_signals`` so PyTorch ``Trainer`` and sklearn
        ``SimpleModelTrainer`` are at full Protocol parity (Q1 closure).

        Uses ``__new__`` to avoid invoking ``__init__`` (which would
        require an ExperimentConfig). The runtime_checkable Protocol
        check verifies method-name presence on the class, not behavior.
        """
        instance = Trainer.__new__(Trainer)
        assert isinstance(instance, BaseTrainer), (
            "Trainer must implement: train, evaluate, "
            "save_checkpoint, load_checkpoint, export_signals"
        )

    def test_simple_model_trainer_class_satisfies_protocol(self) -> None:
        """The sklearn `SimpleModelTrainer` exposes all five required methods.

        Q.6 added `save_checkpoint` / `load_checkpoint` aliases on top
        of the legacy `save()` method. Phase Q.6.5.B added
        ``export_signals`` to the Protocol contract. This test is the
        gate that catches any regression in that surface.
        """
        instance = SimpleModelTrainer.__new__(SimpleModelTrainer)
        assert isinstance(instance, BaseTrainer), (
            f"SimpleModelTrainer must implement: train, evaluate, "
            f"save_checkpoint, load_checkpoint, export_signals. "
            f"Currently exposes: {sorted(name for name in dir(SimpleModelTrainer) if not name.startswith('_'))}"
        )

    @pytest.mark.parametrize(
        "missing_method",
        ["train", "evaluate", "save_checkpoint", "load_checkpoint", "export_signals"],
    )
    def test_class_missing_a_method_fails_protocol(self, missing_method: str) -> None:
        """Any class missing ONE of the five required methods must fail.

        Locks the contract — if a future refactor accidentally removes
        a method from `Trainer` or `SimpleModelTrainer`, the conformance
        tests fail before the production dispatch hits the missing method.
        """
        all_methods = {
            "train": lambda self: {},
            "evaluate": lambda self, split="test": {},
            "save_checkpoint": lambda self, path=None: Path("/tmp"),
            "load_checkpoint": lambda self, path: None,
            "export_signals": lambda self, split="test", *, output_dir=None, calibration="none": Path("/tmp"),
        }
        del all_methods[missing_method]

        # Construct a class with all-but-one of the methods.
        TestClass = type("TestClass", (object,), all_methods)

        instance = TestClass()
        assert not isinstance(instance, BaseTrainer), (
            f"Class missing '{missing_method}' must NOT satisfy "
            f"BaseTrainer; structural check should reject it."
        )

    def test_class_with_all_methods_passes(self) -> None:
        """Sanity: a class with exactly the five required methods passes."""
        all_methods = {
            "train": lambda self: {},
            "evaluate": lambda self, split="test": {},
            "save_checkpoint": lambda self, path=None: Path("/tmp"),
            "load_checkpoint": lambda self, path: None,
            "export_signals": lambda self, split="test", *, output_dir=None, calibration="none": Path("/tmp"),
        }
        TestClass = type("TestClass", (object,), all_methods)
        instance = TestClass()
        assert isinstance(instance, BaseTrainer)
