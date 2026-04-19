"""Unit tests for ``scripts/train.py`` test-metrics persistence helpers.

Phase 7 Stage 7.4 Round 4 item #6 (BLOCKER). The PyTorch Trainer flow
did not persist ``test_metrics.json``, silently deading the 7 test_*
keys added to ``ExperimentRecord.index_entry()`` whitelist in Round 1.

These tests lock the new ``_dump_test_metrics`` + ``_safe_summary``
helpers added to ``scripts/train.py`` against both return shapes of
``trainer.evaluate('test')``:

  * ``ClassificationMetrics`` dataclass → flattened via ``.to_dict()``.
  * Regression / HMHP-R ``Dict[str, Any]`` → used directly.

The helpers are tested in isolation here. An end-to-end integration
covering trainer → disk → ``_capture_training_metrics`` →
``index_entry`` lives in ``hft-ops/tests/test_training_stage.py``
(existing Round 1 coverage) since that boundary is outside this repo.
"""

from __future__ import annotations

import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


@pytest.fixture(scope="module")
def train_script_module():
    """Import ``scripts/train.py`` as a module for helper-function access.

    The script lives outside the installed ``lobtrainer`` package, so we
    load it via ``importlib`` against its on-disk path. The module's
    import-time side effects (``warn_if_not_orchestrated``) are
    suppressed by setting ``HFT_OPS_ORCHESTRATED=1`` before load.
    """
    import os

    os.environ["HFT_OPS_ORCHESTRATED"] = "1"
    train_py = _SCRIPTS_DIR / "train.py"
    # The script prepends parent/src and scripts/ to sys.path — do the same
    # before load so its own internal imports resolve.
    sys.path.insert(0, str(_SCRIPTS_DIR.parent / "src"))
    sys.path.insert(0, str(_SCRIPTS_DIR))
    spec = spec_from_file_location("train_script_under_test", train_py)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeClassificationMetrics:
    """Minimal stand-in that mimics the ``.to_dict()`` contract."""

    def __init__(self, payload):
        self._payload = payload

    def to_dict(self):
        return dict(self._payload)

    def summary(self):
        return "fake-summary"


class TestDumpTestMetricsDict:
    """``_dump_test_metrics`` with regression ``Dict[str, Any]`` input."""

    def test_writes_flat_prefixed_json(self, tmp_path, train_script_module):
        result = {
            "r2": 0.124,
            "ic": 0.380,
            "mae": 5.67,
            "directional_accuracy": 0.640,
        }

        written = train_script_module._dump_test_metrics(result, tmp_path)

        assert written == tmp_path / "test_metrics.json"
        loaded = json.loads(written.read_text())
        assert loaded == {
            "test_directional_accuracy": 0.640,
            "test_ic": 0.380,
            "test_mae": 5.67,
            "test_r2": 0.124,
        }

    def test_empty_dict_returns_none_no_file(self, tmp_path, train_script_module):
        written = train_script_module._dump_test_metrics({}, tmp_path)

        assert written is None
        assert not (tmp_path / "test_metrics.json").exists()

    def test_drops_non_scalars(self, tmp_path, train_script_module):
        # Numpy arrays / lists / nested dicts would not survive JSON
        # serialization AND would break the flat-key contract.
        result = {
            "r2": 0.5,
            "confusion_matrix": [[1, 0], [0, 1]],
            "per_class_breakdown": {"up": 0.6, "down": 0.4},
            "dataset_name": "NVDA_2026",  # string, non-numeric
        }

        written = train_script_module._dump_test_metrics(result, tmp_path)

        assert written is not None
        loaded = json.loads(written.read_text())
        assert loaded == {"test_r2": 0.5}

    def test_skips_bool_values(self, tmp_path, train_script_module):
        # ``bool`` is-a int in Python — the helper's explicit
        # ``isinstance(x, bool)`` check must drop them.
        result = {"r2": 0.5, "converged": True, "early_stopped": False}

        written = train_script_module._dump_test_metrics(result, tmp_path)
        loaded = json.loads(written.read_text())
        assert loaded == {"test_r2": 0.5}


class TestDumpTestMetricsClassification:
    """``_dump_test_metrics`` with ``.to_dict()``-bearing object."""

    def test_flattens_via_to_dict(self, tmp_path, train_script_module):
        metrics = _FakeClassificationMetrics({
            "accuracy": 0.596,
            "macro_f1": 0.421,
            "macro_precision": 0.450,
            "loss": 1.02,
        })

        written = train_script_module._dump_test_metrics(metrics, tmp_path)

        loaded = json.loads(written.read_text())
        assert loaded == {
            "test_accuracy": 0.596,
            "test_loss": 1.02,
            "test_macro_f1": 0.421,
            "test_macro_precision": 0.450,
        }


class TestDumpTestMetricsEdgeCases:
    """Unrecognized input shapes must not raise."""

    def test_returns_none_for_string_input(self, tmp_path, train_script_module):
        assert train_script_module._dump_test_metrics("bad", tmp_path) is None
        assert not (tmp_path / "test_metrics.json").exists()

    def test_returns_none_for_none_input(self, tmp_path, train_script_module):
        assert train_script_module._dump_test_metrics(None, tmp_path) is None

    def test_custom_prefix(self, tmp_path, train_script_module):
        written = train_script_module._dump_test_metrics(
            {"ic": 0.3}, tmp_path, prefix="final_",
        )
        loaded = json.loads(written.read_text())
        assert loaded == {"final_ic": 0.3}


class TestSafeSummary:
    """``_safe_summary`` must render both shapes without raising."""

    def test_uses_summary_method_when_available(self, train_script_module):
        result = train_script_module._safe_summary(
            _FakeClassificationMetrics({"accuracy": 0.5}),
        )
        assert result == "fake-summary"

    def test_falls_back_to_key_value_lines_for_dict(self, train_script_module):
        result = train_script_module._safe_summary({"ic": 0.38, "r2": 0.12})
        # Sorted keys → deterministic output
        assert "ic: 0.380000" in result
        assert "r2: 0.120000" in result

    def test_survives_dict_with_non_numeric(self, train_script_module):
        result = train_script_module._safe_summary(
            {"ic": 0.38, "note": "skipped_val"},
        )
        assert "ic: 0.380000" in result
        assert "note: skipped_val" in result

    def test_returns_str_repr_for_unknown(self, train_script_module):
        result = train_script_module._safe_summary(42)
        assert result == "42"
