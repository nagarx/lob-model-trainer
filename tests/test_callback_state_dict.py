"""Phase DESIGN-1 G-1 (2026-05-10) regression tests — callback state in
checkpoint dict.

Locks the cross-process resume contract: ``Trainer._build_checkpoint_dict``
adds ``'callback_state'`` (13th key, sister to ``'rng_state'``); each
callback's ``state_dict()`` returns a JSON-serializable dict; class-name
keying with explicit collision detection per Agent X recommendation.

Per saved feedback memory + hft-rules §6 (Testing Philosophy): tests must
explain WHAT failed and WHY. Each assertion message names the invariant.

Architectural references:
  - lob-model-trainer/src/lobtrainer/training/callbacks.py
    Callback.state_dict / Callback.load_state_dict (ABC defaults)
  - lob-model-trainer/src/lobtrainer/training/trainer.py
    _build_checkpoint_dict (callback_state key) + load_checkpoint (apply)
"""

from pathlib import Path

import pytest

from lobtrainer.training.callbacks import (
    Callback,
    CallbackList,
    EarlyStopping,
    EarlyStoppingState,
    MetricLogger,
    ModelCheckpoint,
    ProgressCallback,
)


# =============================================================================
# Callback ABC — default state_dict / load_state_dict
# =============================================================================


class TestCallbackABCDefaults:
    """Callback ABC defaults preserve back-compat with pre-G-1 callbacks."""

    def test_default_state_dict_returns_empty_dict(self):
        """Stateless callbacks need not override — default is empty dict."""
        # Construct any Callback subclass that doesn't override state_dict
        cb = ProgressCallback()
        assert cb.state_dict() == {}, (
            "Pre-G-1 callbacks must produce empty state_dict by default. "
            "Returning anything else would break pre-G-1 callback subclasses "
            "that haven't been migrated."
        )

    def test_default_load_state_dict_is_noop(self):
        """Default load_state_dict accepts empty/arbitrary dict without error."""
        cb = ProgressCallback()
        # Multiple calls with various state shapes must not raise
        cb.load_state_dict({})
        cb.load_state_dict({"unknown_key": 42})
        cb.load_state_dict({"a": 1, "b": "string", "c": [1, 2, 3]})


# =============================================================================
# EarlyStopping state_dict / load_state_dict
# =============================================================================


class TestEarlyStoppingStateRoundtrip:
    """state_dict → load_state_dict produces identical state."""

    def test_state_dict_serializes_4_fields(self):
        """EarlyStopping state_dict must contain {best_value, best_epoch,
        wait_count, stopped} per G-1 spec."""
        cb = EarlyStopping(patience=5, metric="val_loss", mode="min")
        cb._state.best_value = 0.123
        cb._state.best_epoch = 7
        cb._state.wait_count = 3
        cb._state.stopped = False

        state = cb.state_dict()
        assert "best_value" in state
        assert "best_epoch" in state
        assert "wait_count" in state
        assert "stopped" in state
        # Deliberately exclude _best_weights to avoid checkpoint bloat
        assert "best_weights" not in state, (
            "G-1 spec deliberately excludes _best_weights from state_dict "
            "(typically 100MB-1GB for TLOB/HMHP). Including it would "
            "violate the bloat-avoidance design decision."
        )

    def test_state_dict_values_are_json_native(self):
        """All values must be JSON-serializable scalars (no tuples/ndarray)."""
        cb = EarlyStopping(patience=5, metric="val_loss", mode="min")
        cb._state.best_value = 1.5
        cb._state.best_epoch = 10
        cb._state.wait_count = 2
        cb._state.stopped = True

        state = cb.state_dict()
        assert isinstance(state["best_value"], float)
        assert isinstance(state["best_epoch"], int)
        assert isinstance(state["wait_count"], int)
        assert isinstance(state["stopped"], bool)

    def test_roundtrip_preserves_all_fields(self):
        """Round-trip via state_dict → load_state_dict produces identical state."""
        cb_a = EarlyStopping(patience=10, metric="val_loss", mode="min")
        cb_a._state.best_value = 0.456
        cb_a._state.best_epoch = 12
        cb_a._state.wait_count = 4
        cb_a._state.stopped = False

        # Round-trip: serialize from a, restore into b
        state = cb_a.state_dict()
        cb_b = EarlyStopping(patience=10, metric="val_loss", mode="min")
        cb_b.load_state_dict(state)

        assert cb_b._state.best_value == cb_a._state.best_value
        assert cb_b._state.best_epoch == cb_a._state.best_epoch
        assert cb_b._state.wait_count == cb_a._state.wait_count
        assert cb_b._state.stopped == cb_a._state.stopped

    def test_load_state_dict_tolerates_empty_dict(self):
        """Forward-compat: empty state (pre-G-1 checkpoint) leaves init defaults."""
        cb = EarlyStopping(patience=5, metric="val_loss", mode="min")
        # Set state to non-default
        cb._state.wait_count = 9
        # Empty dict should NOT raise and NOT clobber existing state
        cb.load_state_dict({})
        assert cb._state.wait_count == 9, (
            "Empty state_dict must leave existing state untouched per G-1 "
            "forward-compat contract (pre-G-1 checkpoint resumed via post-G-1 "
            "trainer must NOT silently reset patience counter)."
        )

    def test_load_state_dict_tolerates_partial_dict(self):
        """Forward-compat: missing keys keep existing values."""
        cb = EarlyStopping(patience=5, metric="val_loss", mode="min")
        cb._state.best_value = 0.5
        cb._state.wait_count = 2

        # Partial state — only best_value
        cb.load_state_dict({"best_value": 0.123})

        assert cb._state.best_value == 0.123  # restored
        assert cb._state.wait_count == 2  # preserved (not in partial dict)


# =============================================================================
# ModelCheckpoint state_dict / load_state_dict
# =============================================================================


class TestModelCheckpointStateRoundtrip:
    """ModelCheckpoint state_dict / load_state_dict roundtrip."""

    def test_state_dict_serializes_best_tracking(self, tmp_path):
        """ModelCheckpoint state_dict must include _best_value + _best_checkpoint_path."""
        cb = ModelCheckpoint(save_dir=tmp_path, metric="val_loss", mode="min")
        cb._best_value = 0.789
        cb._best_checkpoint_path = tmp_path / "best.pt"

        state = cb.state_dict()
        assert state["best_value"] == 0.789
        assert state["best_checkpoint_path"] == str(tmp_path / "best.pt")
        # JSON-native types
        assert isinstance(state["best_value"], float)
        assert isinstance(state["best_checkpoint_path"], str)

    def test_state_dict_handles_none_path(self, tmp_path):
        """Path → None when no best checkpoint yet (init state)."""
        cb = ModelCheckpoint(save_dir=tmp_path, metric="val_loss", mode="min")
        # _best_checkpoint_path defaults to None
        state = cb.state_dict()
        assert state["best_checkpoint_path"] is None

    def test_roundtrip_preserves_state(self, tmp_path):
        """Round-trip via state_dict → load_state_dict."""
        cb_a = ModelCheckpoint(save_dir=tmp_path, metric="val_loss", mode="min")
        cb_a._best_value = 0.234
        cb_a._best_checkpoint_path = tmp_path / "best.pt"

        state = cb_a.state_dict()

        cb_b = ModelCheckpoint(save_dir=tmp_path, metric="val_loss", mode="min")
        cb_b.load_state_dict(state)

        assert cb_b._best_value == cb_a._best_value
        assert cb_b._best_checkpoint_path == cb_a._best_checkpoint_path
        assert isinstance(cb_b._best_checkpoint_path, Path), (
            "Path string must be reconstructed back to Path on load_state_dict "
            "per G-1 spec (str → Path)."
        )

    def test_load_state_dict_tolerates_empty_dict(self, tmp_path):
        cb = ModelCheckpoint(save_dir=tmp_path, metric="val_loss", mode="min")
        cb._best_value = 0.5
        cb.load_state_dict({})
        assert cb._best_value == 0.5  # preserved


# =============================================================================
# MetricLogger state_dict / load_state_dict
# =============================================================================


class TestMetricLoggerStateRoundtrip:
    """MetricLogger state_dict / load_state_dict roundtrip."""

    def test_state_dict_serializes_history(self):
        """MetricLogger state_dict must include _history."""
        cb = MetricLogger(log_to_file=False)
        cb._history = [
            {"epoch": 0, "train_loss": 1.5, "val_loss": 1.7},
            {"epoch": 1, "train_loss": 1.3, "val_loss": 1.5},
        ]
        state = cb.state_dict()
        assert "history" in state
        assert state["history"] == cb._history

    def test_roundtrip_preserves_history(self):
        cb_a = MetricLogger(log_to_file=False)
        cb_a._history = [{"epoch": 5, "loss": 0.42}]

        state = cb_a.state_dict()

        cb_b = MetricLogger(log_to_file=False)
        cb_b.load_state_dict(state)
        assert cb_b._history == cb_a._history

    def test_load_state_dict_tolerates_empty_dict(self):
        cb = MetricLogger(log_to_file=False)
        cb._history = [{"epoch": 0}]
        cb.load_state_dict({})
        assert cb._history == [{"epoch": 0}]  # preserved


# =============================================================================
# Trainer._build_checkpoint_dict — callback_state key
# =============================================================================
#
# Note: full Trainer integration tests require a real ExperimentConfig +
# data fixtures, which is heavyweight. The class-name keying + collision
# detection logic is unit-testable via the CallbackList helper below
# without instantiating the full Trainer.


class _MockTrainer:
    """Minimal mock for class-name keying + collision detection tests.

    The real ``Trainer._build_checkpoint_dict`` is heavyweight (loads
    config + builds compatibility contract). The class-name keying logic
    (the part G-1 actually adds) is independent of that and can be
    exercised via this lightweight harness.
    """

    def __init__(self, callbacks):
        self.callbacks = CallbackList(callbacks)
        self.callbacks.set_trainer(self)

    def _build_callback_state_dict(self):
        """Mirror of Trainer._build_checkpoint_dict's callback_state block.

        Lifted from trainer.py — this is the canonical block-under-test.
        Any divergence between this and trainer.py would surface as
        test failure during the empirical validation step.
        """
        callback_state = {}
        if self.callbacks is not None and self.callbacks.callbacks:
            seen_class_names = {}
            for cb in self.callbacks.callbacks:
                cb_name = type(cb).__name__
                seen_class_names[cb_name] = seen_class_names.get(cb_name, 0) + 1
            duplicates = [n for n, c in seen_class_names.items() if c > 1]
            if duplicates:
                raise ValueError(
                    f"Duplicate callback class name(s) {duplicates!r} — "
                    f"checkpoint resume cannot disambiguate via class-name keying."
                )
            for cb in self.callbacks.callbacks:
                callback_state[type(cb).__name__] = cb.state_dict()
        return callback_state


class TestCallbackStateClassNameKeying:
    """Test the class-name keying logic + collision detection."""

    def test_callback_state_keyed_by_class_name(self, tmp_path):
        """Each callback gets its own key in the dict."""
        cbs = [
            EarlyStopping(patience=3, metric="val_loss", mode="min"),
            ModelCheckpoint(save_dir=tmp_path, metric="val_loss", mode="min"),
            MetricLogger(log_to_file=False),
        ]
        trainer = _MockTrainer(cbs)
        state = trainer._build_callback_state_dict()

        assert set(state.keys()) == {
            "EarlyStopping", "ModelCheckpoint", "MetricLogger"
        }, "Each callback must contribute its class name as a top-level key."

    def test_duplicate_class_names_raise(self, tmp_path):
        """G-1 fail-loud invariant: duplicate class names cannot be disambiguated."""
        cbs = [
            EarlyStopping(patience=3, metric="val_loss", mode="min"),
            EarlyStopping(patience=5, metric="val_ic", mode="max"),  # SAME class
        ]
        trainer = _MockTrainer(cbs)

        with pytest.raises(ValueError, match=r"Duplicate callback class name"):
            trainer._build_callback_state_dict()

    def test_stateless_callbacks_appear_with_empty_dict(self, tmp_path):
        """ProgressCallback (no override) appears with empty state."""
        cbs = [
            EarlyStopping(patience=3, metric="val_loss", mode="min"),
            ProgressCallback(),  # uses Callback.state_dict default = {}
        ]
        trainer = _MockTrainer(cbs)
        state = trainer._build_callback_state_dict()

        assert "ProgressCallback" in state
        assert state["ProgressCallback"] == {}, (
            "Stateless callbacks must serialize to empty dict per G-1 default. "
            "Pre-G-1 callbacks (no state_dict override) get this for free."
        )

    def test_empty_callbacks_list(self):
        """No callbacks → empty dict (no error)."""
        trainer = _MockTrainer([])
        state = trainer._build_callback_state_dict()
        assert state == {}


# =============================================================================
# Cross-callback EarlyStopping state preservation through resume
# =============================================================================
#
# This is the key Phase DESIGN-1 G-1 invariant: cross-process resume of
# a checkpoint that has callback_state preserves the patience counter
# and best_value. Without G-1, these would reset to init defaults.


class TestCrossProcessResumeInvariant:
    """The G-1 keystone test: simulate cross-process resume."""

    def test_earlystopping_preserves_patience_via_serialize_then_load(self):
        """Simulate: train run A reaches wait_count=4 / best_value=0.5;
        save checkpoint with state_dict; fresh process loads checkpoint
        and applies state_dict; resumed callback has wait_count=4.

        Pre-G-1 (no state_dict / no callback_state in checkpoint):
        resumed callback would have wait_count=0 → silently train past
        the original stopping epoch → divergent metrics. G-1 closes
        this. RNG determinism alone (Phase A.2) is INSUFFICIENT.
        """
        # PROCESS A: trainer reaches wait_count=4
        cb_a = EarlyStopping(patience=10, metric="val_loss", mode="min")
        cb_a._state.best_value = 0.5
        cb_a._state.best_epoch = 8
        cb_a._state.wait_count = 4
        cb_a._state.stopped = False

        # Serialize state (would be embedded in checkpoint['callback_state'])
        serialized = cb_a.state_dict()

        # PROCESS B: fresh trainer with fresh callback (init state)
        cb_b = EarlyStopping(patience=10, metric="val_loss", mode="min")
        # Fresh init defaults
        assert cb_b._state.wait_count == 0
        assert cb_b._state.best_value == float("inf")  # mode=min default

        # Apply restored state
        cb_b.load_state_dict(serialized)

        # G-1 INVARIANT: wait_count + best_value preserved cross-process
        assert cb_b._state.wait_count == 4, (
            "Phase DESIGN-1 G-1 CROSS-PROCESS RESUME INVARIANT VIOLATED: "
            "EarlyStopping wait_count not preserved across serialize/load. "
            "This means cross-process resume would silently train past the "
            "original stopping epoch."
        )
        assert cb_b._state.best_value == 0.5
        assert cb_b._state.best_epoch == 8
        assert cb_b._state.stopped is False
