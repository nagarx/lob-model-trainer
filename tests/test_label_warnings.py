"""Tests for the Phase 3b E8 smoothed_return run-entry nudge (label_warnings)."""

import warnings

import pytest

from lobtrainer.training.label_warnings import (
    warn_if_smoothed_return,
    _reset_smoothed_return_nudge,
)


@pytest.fixture(autouse=True)
def _reset_guard():
    """Reset the once-per-process guard around every test."""
    _reset_smoothed_return_nudge()
    yield
    _reset_smoothed_return_nudge()


def test_warns_on_smoothed_return():
    with pytest.warns(UserWarning, match="E8 trap"):
        emitted = warn_if_smoothed_return("smoothed_return")
    assert emitted is True


def test_silent_on_point_return():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would raise
        emitted = warn_if_smoothed_return("point_return")
    assert emitted is False


def test_silent_on_other_return_types():
    for rt in ("mean_return", "peak_return", "anything_else"):
        _reset_smoothed_return_nudge()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert warn_if_smoothed_return(rt) is False


def test_idempotent_once_per_process():
    """CVTrainer spawns K fresh Trainers in one run; only the first warns."""
    with pytest.warns(UserWarning):
        assert warn_if_smoothed_return("smoothed_return") is True
    # second call (fold 2): guarded -> no warning, returns False
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a 2nd warning would raise
        assert warn_if_smoothed_return("smoothed_return") is False


def test_force_bypasses_guard():
    with pytest.warns(UserWarning):
        warn_if_smoothed_return("smoothed_return")
    with pytest.warns(UserWarning):  # force re-emits despite the guard
        assert warn_if_smoothed_return("smoothed_return", force=True) is True


def test_message_cites_finding_and_point_return():
    with pytest.warns(UserWarning) as rec:
        warn_if_smoothed_return("smoothed_return", force=True)
    msg = str(rec[0].message)
    assert "FINDING-001" in msg
    assert "point" in msg.lower()


def test_trainer_train_invokes_nudge_before_setup(monkeypatch):
    """Wiring: Trainer.train() calls the nudge with the config's return_type,
    before setup() — proven by spying the helper and aborting setup()."""
    from lobtrainer.config.schema import ExperimentConfig
    from lobtrainer.training import trainer as trainer_mod

    cfg = ExperimentConfig.from_dict({
        "name": "nudge_wiring",
        "data": {
            "data_dir": "/nonexistent",
            "feature_count": 40,
            "labeling_strategy": "tlob",
            "num_classes": 3,
            "horizon_idx": 1,
            "feature_preset": "lob_only",
            "sequence": {"window_size": 20, "stride": 10},
            "normalization": {"strategy": "none"},
        },
        "model": {
            "model_type": "deeplob", "input_size": 40, "num_classes": 3,
            "task_type": "regression", "dropout": 0.0, "deeplob_mode": "benchmark",
        },
        "train": {
            "batch_size": 16, "learning_rate": 1e-3, "epochs": 1,
            "task_type": "regression", "loss_type": "huber",
            "num_workers": 0, "pin_memory": False, "seed": 42,
        },
        "output_dir": "/tmp/nudge_wiring",
    })
    # The regression default return_type IS smoothed_return (the E8-prone case).
    assert cfg.data.labels.return_type == "smoothed_return"

    trainer = trainer_mod.Trainer(cfg)
    seen = {}
    monkeypatch.setattr(
        trainer_mod, "warn_if_smoothed_return",
        lambda rt, **kw: (seen.update(rt=rt), True)[1],
    )

    def _abort_setup():
        raise RuntimeError("STOP_AFTER_NUDGE")

    monkeypatch.setattr(trainer, "setup", _abort_setup)

    with pytest.raises(RuntimeError, match="STOP_AFTER_NUDGE"):
        trainer.train()
    assert seen.get("rt") == "smoothed_return"
