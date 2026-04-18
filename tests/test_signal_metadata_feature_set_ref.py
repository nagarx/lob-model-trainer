"""Phase 4 Batch 4c.4: trainer-side signal_metadata.json feature_set_ref writer.

Locks:
1. `build_signal_metadata(feature_set_ref=...)` includes it in output.
2. Absent kwarg → key NOT emitted (backward compat — legacy consumers see no drift).
3. `_feature_set_ref_dict(data_config)` converts `_feature_set_ref_resolved`
   tuple `(name, hash)` to the `{name, content_hash}` dict shape.
4. Absent/None private cache → None return (harvester-side then omits the key).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import pytest

from lobtrainer.export.metadata import build_signal_metadata
from lobtrainer.export.exporter import _feature_set_ref_dict


class TestBuildSignalMetadataWithRef:
    def test_feature_set_ref_included_when_provided(self):
        ref = {"name": "momentum_v1", "content_hash": "a" * 64}
        meta = build_signal_metadata(
            model_type="tlob",
            model_name="test",
            parameters=100,
            signal_type="regression",
            split="test",
            total_samples=1000,
            checkpoint="/tmp/ckpt",
            feature_set_ref=ref,
        )
        assert meta.get("feature_set_ref") == ref

    def test_feature_set_ref_absent_when_none(self):
        meta = build_signal_metadata(
            model_type="tlob",
            model_name="test",
            parameters=100,
            signal_type="regression",
            split="test",
            total_samples=1000,
            checkpoint="/tmp/ckpt",
            feature_set_ref=None,
        )
        assert "feature_set_ref" not in meta, (
            "Absent field should NOT be in output (matches feature_preset convention)."
        )


class TestFeatureSetRefDictHelper:
    """`_feature_set_ref_dict` converts DataConfig._feature_set_ref_resolved
    (an Optional[Tuple[str, str]]) to the JSON-shape dict."""

    def test_tuple_to_dict(self):
        @dataclass
        class FakeCfg:
            _feature_set_ref_resolved: Optional[Tuple[str, str]] = None

        cfg = FakeCfg(_feature_set_ref_resolved=("momentum_v1", "a" * 64))
        assert _feature_set_ref_dict(cfg) == {
            "name": "momentum_v1",
            "content_hash": "a" * 64,
        }

    def test_none_to_none(self):
        @dataclass
        class FakeCfg:
            _feature_set_ref_resolved: Optional[Tuple[str, str]] = None

        cfg = FakeCfg(_feature_set_ref_resolved=None)
        assert _feature_set_ref_dict(cfg) is None

    def test_missing_attr_safe(self):
        class FakeCfg:
            pass  # no _feature_set_ref_resolved attribute at all

        # `getattr(cfg, "_feature_set_ref_resolved", None)` returns None
        # gracefully (legacy config, pre-4c.1 shape).
        assert _feature_set_ref_dict(FakeCfg()) is None
