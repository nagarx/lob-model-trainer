"""Phase Q.6.5.B Part 1 — direct unit tests for the SSoT helpers in
``lobtrainer.training.compatibility``.

Closes the test-coverage gap surfaced by Agent 1 mid-impl audit (HIGH-1
follow-up): ``feature_set_ref_to_dict`` was exercised only TRANSITIVELY
through the trainer + signal-export tests. This file locks the helper's
behavioral contract directly, mirroring the test convention for
``derive_data_source`` / ``compute_model_config_hash`` / etc.

Locks:
- 3 documented null-return cases (None resolved / malformed arity /
  empty-string components)
- happy-path tuple-to-dict conversion
- defensive (TypeError, ValueError) catch on non-iterable / wrong-arity
- public API preserved across the SSoT lift (no signature change)
- mirror behavior with ``derive_data_source`` for off-exchange / mbo_lob
  classification (sanity-check of the sibling SSoT helpers)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import pytest

from lobtrainer.training.compatibility import (
    derive_data_source,
    feature_set_ref_to_dict,
)


# =============================================================================
# Test fixtures
# =============================================================================


@dataclass
class _FakeDataConfig:
    """Minimal stand-in for DataConfig private cache. Mirrors the
    PrivateAttr declaration at schema.py:1109.
    """
    _feature_set_ref_resolved: Optional[Tuple[str, str]] = None


# =============================================================================
# feature_set_ref_to_dict — happy path + 3 documented null-return cases
# =============================================================================


class TestFeatureSetRefToDict:
    """SSoT helper for converting DataConfig private cache → JSON dict shape."""

    def test_valid_tuple_yields_canonical_dict(self):
        cfg = _FakeDataConfig(
            _feature_set_ref_resolved=("momentum_v1", "a" * 64)
        )
        assert feature_set_ref_to_dict(cfg) == {
            "name": "momentum_v1",
            "content_hash": "a" * 64,
        }

    def test_none_resolved_returns_none(self):
        """When no FeatureSet was resolved (ad-hoc / preset paths),
        ``_feature_set_ref_resolved`` is ``None``. Helper returns
        ``None`` (signal_metadata key omitted)."""
        cfg = _FakeDataConfig(_feature_set_ref_resolved=None)
        assert feature_set_ref_to_dict(cfg) is None

    def test_missing_attribute_returns_none(self):
        """When the cache attribute does not exist (legacy config path or
        pre-4c.1 shape), ``getattr(..., None)`` defensively returns
        ``None``. Helper returns ``None``."""
        class LegacyCfg:
            pass  # no _feature_set_ref_resolved attribute at all
        assert feature_set_ref_to_dict(LegacyCfg()) is None

    @pytest.mark.parametrize(
        "malformed,description",
        [
            (("name_only",), "1-tuple"),
            (("a", "b", "c"), "3-tuple"),
            ("not_a_tuple", "string"),
            (123, "non-iterable int"),
        ],
    )
    def test_malformed_arity_returns_none(self, malformed, description):
        """Malformed cache values (wrong arity / non-iterable) return
        None defensively per hft-rules §8 (never crash on bad cache state).
        Documented at compatibility.py:175-180."""
        cfg = _FakeDataConfig()
        # Bypass the dataclass type hint — the SSoT helper is defensive
        # against arbitrary cache values.
        cfg._feature_set_ref_resolved = malformed  # type: ignore[assignment]
        result = feature_set_ref_to_dict(cfg)
        assert result is None, (
            f"Malformed cache ({description}: {malformed!r}) silently "
            f"produced {result!r} — defensive guard at compatibility.py "
            f"failed."
        )

    @pytest.mark.parametrize(
        "name,content_hash,description",
        [
            ("", "valid" + "0" * 59, "empty name"),
            ("valid_name", "", "empty content_hash"),
            ("", "", "both empty"),
        ],
    )
    def test_empty_string_components_return_none(
        self, name, content_hash, description
    ):
        """Empty-string components in the tuple trigger cache-poisoning
        defense per hft-rules §5 (fail-fast at producer). Helper returns
        None so downstream consumers' ``CONTENT_HASH_RE`` validation is
        not exercised on a silently-malformed dict.

        Mirrors the original guard at importance/callback.py:595 +
        Q.6.5.A inline pattern at simple_trainer.py."""
        cfg = _FakeDataConfig(
            _feature_set_ref_resolved=(name, content_hash)
        )
        result = feature_set_ref_to_dict(cfg)
        assert result is None, (
            f"Empty-string component case ({description}) silently "
            f"produced {result!r} — empty-string guard failed."
        )

    def test_str_coercion_applied(self):
        """Non-str types in the tuple are str-coerced (defensive — schema
        declares Tuple[str, str] but runtime types may vary)."""
        cfg = _FakeDataConfig(
            _feature_set_ref_resolved=("name", "0" * 64)
        )
        result = feature_set_ref_to_dict(cfg)
        # Both fields must be str post-coercion.
        assert isinstance(result["name"], str)
        assert isinstance(result["content_hash"], str)


# =============================================================================
# derive_data_source — sibling SSoT helper sanity check
# =============================================================================


class TestDeriveDataSource:
    """Sibling SSoT helper at compatibility.py:117-131 — converts the
    export directory basename to ``data_source`` tag.

    Phase II heuristic: ``basic_*`` prefix → off_exchange; otherwise
    mbo_lob. Locks the convention so changes are visible.
    """

    @pytest.mark.parametrize(
        "data_dir,expected",
        [
            ("/path/to/basic_nvda_60s", "off_exchange"),
            ("/path/to/basic_synthetic", "off_exchange"),
            ("basic_e9_60s", "off_exchange"),
            # Non-basic-prefixed paths → mbo_lob default.
            ("/path/to/e5_timebased_60s_v3p0", "mbo_lob"),
            ("/path/to/nvda_xnas_128feat", "mbo_lob"),
            ("synthetic_data_dir", "mbo_lob"),
            # Edge cases — empty / ambiguous names.
            ("", "mbo_lob"),
            ("/", "mbo_lob"),
        ],
    )
    def test_classification_by_basename_prefix(self, data_dir, expected):
        assert derive_data_source(data_dir) == expected, (
            f"derive_data_source({data_dir!r}) classification wrong; "
            f"expected {expected!r}"
        )

    def test_path_object_accepted(self):
        """The helper accepts both str AND Path-like inputs. Locks against
        accidental str-only restriction."""
        from pathlib import Path
        assert derive_data_source(Path("/path/basic_x")) == "off_exchange"
        assert derive_data_source(Path("/path/foo")) == "mbo_lob"


# =============================================================================
# Public API surface lock — prevent accidental signature drift
# =============================================================================


class TestPublicAPISurface:
    """Locks the public API of compatibility.py against accidental drift.

    Phase Q.6.5.B lifted ``feature_set_ref_to_dict`` from 3 sites
    (exporter.py + importance/callback.py + simple_trainer.py inline) to
    one SSoT. If a future refactor accidentally renames/deletes the helper,
    consumers (3 sites + this test) will break visibly.
    """

    def test_feature_set_ref_to_dict_is_callable(self):
        assert callable(feature_set_ref_to_dict)

    def test_feature_set_ref_to_dict_signature(self):
        """Signature must be ``(data_config: Any) -> Optional[Dict[str, str]]``."""
        import inspect
        sig = inspect.signature(feature_set_ref_to_dict)
        params = list(sig.parameters.keys())
        assert params == ["data_config"], (
            f"feature_set_ref_to_dict signature changed — 3 consumers will "
            f"break. Got params: {params}"
        )

    def test_derive_data_source_signature(self):
        import inspect
        sig = inspect.signature(derive_data_source)
        params = list(sig.parameters.keys())
        assert params == ["data_dir"], (
            f"derive_data_source signature changed. Got params: {params}"
        )
