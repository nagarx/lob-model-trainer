"""Tests for DataConfig feature-selection fields + mutual exclusion
(Phase 4 Batch 4c.1)."""

from __future__ import annotations

import warnings

import pytest

from lobtrainer.config.schema import DataConfig


# ---------------------------------------------------------------------------
# feature_set field
# ---------------------------------------------------------------------------


class TestFeatureSetField:
    def test_default_is_none(self):
        c = DataConfig()
        assert c.feature_set is None

    def test_accepts_valid_name(self):
        c = DataConfig(feature_set="momentum_hft_v1")
        assert c.feature_set == "momentum_hft_v1"

    def test_empty_string_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            DataConfig(feature_set="")

    def test_whitespace_only_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            DataConfig(feature_set="   ")

    def test_forward_slash_rejected(self):
        with pytest.raises(ValueError, match="path separators"):
            DataConfig(feature_set="foo/bar")

    def test_backslash_rejected(self):
        with pytest.raises(ValueError, match="path separators"):
            DataConfig(feature_set="foo\\bar")


# ---------------------------------------------------------------------------
# feature_sets_dir — explicit override for registry auto-detection
# ---------------------------------------------------------------------------


class TestFeatureSetsDirOverride:
    """Phase 4 Batch 4c hardening: allow DataConfig to override the
    trainer's auto-detected FeatureSet registry directory. Enables test
    isolation, multi-registry workflows, and running from CWDs outside
    the monorepo."""

    def test_default_is_none(self):
        # When None, trainer falls back to find_feature_sets_dir walk-up.
        assert DataConfig().feature_sets_dir is None

    def test_explicit_override_accepted(self):
        c = DataConfig(feature_sets_dir="/tmp/custom_registry")
        assert c.feature_sets_dir == "/tmp/custom_registry"

    def test_not_in_mutual_exclusion(self):
        # feature_sets_dir is orthogonal to the 3 selection fields
        # (it's about WHERE to look, not WHAT to select). Setting it
        # alongside any selection field must be allowed.
        c = DataConfig(feature_set="x_v1", feature_sets_dir="/tmp/r")
        assert c.feature_set == "x_v1"
        assert c.feature_sets_dir == "/tmp/r"


# ---------------------------------------------------------------------------
# Mutual exclusion across 4 selection fields
# ---------------------------------------------------------------------------


class TestMutualExclusion:
    """At most ONE of {feature_set, feature_indices, feature_preset} may
    be set on DataConfig.

    Phase 4 Batch 4c revised the original plan's priority-based
    precedence to explicit mutual exclusion, because priorities silently
    discard a user's override in one direction — either direction.
    Explicit mutual exclusion forces users to clear other fields via
    YAML null when they intend to override.

    Note (Phase 4 Batch 4c hardening, 2026-04-15): ``feature_set_per_horizon``
    was removed from this mutual-exclusion set entirely — it returns in
    Batch 4d alongside the HMHP ``feature_attention`` activation. Until
    then, any HMHP per-horizon selection uses a uniform ``feature_set``.
    """

    def test_feature_set_plus_feature_preset_raises(self):
        with pytest.raises(ValueError, match="At most one"):
            DataConfig(
                feature_set="x_v1",
                feature_preset="short_term_40",
            )

    def test_feature_set_plus_feature_indices_raises(self):
        with pytest.raises(ValueError, match="At most one"):
            DataConfig(
                feature_set="x_v1",
                feature_indices=[0, 5, 12],
            )

    def test_feature_preset_plus_feature_indices_raises(self):
        # This was the pre-4c check; verify still works.
        with pytest.raises(ValueError, match="At most one"):
            DataConfig(
                feature_preset="short_term_40",
                feature_indices=[0, 5, 12],
            )

    def test_error_message_names_all_active_fields(self):
        with pytest.raises(ValueError) as excinfo:
            DataConfig(
                feature_set="x_v1",
                feature_preset="short_term_40",
            )
        err = str(excinfo.value)
        assert "feature_set" in err
        assert "feature_preset" in err
        assert "2" in err  # count

    def test_error_message_suggests_null_override(self):
        with pytest.raises(ValueError) as excinfo:
            DataConfig(
                feature_set="x_v1",
                feature_indices=[0, 5],
            )
        err = str(excinfo.value)
        assert "null" in err  # migration hint

    def test_exactly_zero_selection_fields_ok(self):
        # No selection field set → trainer uses all features (default).
        c = DataConfig()
        assert c.feature_set is None
        assert c.feature_preset is None
        assert c.feature_indices is None

    def test_exactly_one_selection_field_ok(self):
        # Each single field, in isolation, is allowed.
        DataConfig(feature_set="x_v1")
        # feature_preset emits DeprecationWarning; suppress for this test.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            DataConfig(feature_preset="short_term_40")
        DataConfig(feature_indices=[0, 5, 12])


# ---------------------------------------------------------------------------
# feature_preset DeprecationWarning (Phase 4 Batch 4c, 2026-04-15)
# ---------------------------------------------------------------------------


class TestFeaturePresetDeprecation:
    """`feature_preset` is scheduled for removal on the 4-month 3-step
    schedule. Batch 4c installs the initial DeprecationWarning.
    Escalation on 2026-06-15; ImportError on 2026-08-15."""

    def test_warning_fires_on_valid_preset(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            DataConfig(feature_preset="short_term_40")
            dep_warnings = [
                wi for wi in w if issubclass(wi.category, DeprecationWarning)
            ]
            assert len(dep_warnings) == 1

    def test_warning_message_contains_migration_hint(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            DataConfig(feature_preset="short_term_40")
            dep = [wi for wi in w if issubclass(wi.category, DeprecationWarning)][0]
            msg = str(dep.message)
            assert "feature_set" in msg  # migration target
            assert "hft-ops evaluate" in msg  # producer command
            assert "2026-08-15" in msg  # hard-error date

    def test_no_warning_when_feature_preset_is_none(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            DataConfig()
            dep_warnings = [
                wi for wi in w if issubclass(wi.category, DeprecationWarning)
            ]
            assert dep_warnings == []

    def test_no_warning_when_only_feature_set_is_set(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            DataConfig(feature_set="momentum_hft_v1")
            dep_warnings = [
                wi for wi in w if issubclass(wi.category, DeprecationWarning)
            ]
            assert dep_warnings == []

    def test_unknown_preset_still_raises_valueerror(self):
        # Validation still applies — unknown presets fail immediately
        # (before the deprecation warning fires, since the order in
        # __post_init__ is: preset-name-validation → warning).
        with pytest.raises(ValueError, match="Unknown feature_preset"):
            DataConfig(feature_preset="nonexistent_preset")


# ---------------------------------------------------------------------------
# Private runtime cache fields
# ---------------------------------------------------------------------------


class TestPrivateCacheFields:
    """Phase 4 R3 "runtime cache, not YAML mutation" pattern: the
    resolver populates these at dataloader construction; they are NOT
    serialized back to YAML via asdict."""

    def test_cache_default_is_none(self):
        c = DataConfig(feature_set="x_v1")
        assert c._feature_indices_resolved is None
        assert c._feature_set_ref_resolved is None

    def test_resolver_does_not_mutate_user_facing_fields(self):
        # R3 invariant: the resolver populates the private cache
        # WITHOUT mutating the user's `feature_set`. Round-trip
        # preservation of the on-disk YAML depends on this — we don't
        # want `feature_set` silently replaced by `feature_indices` at
        # load time.
        c = DataConfig(feature_set="x_v1")
        # Simulate the resolver writing to the private cache:
        c._feature_indices_resolved = [0, 5, 12]
        c._feature_set_ref_resolved = ("x_v1", "a" * 64)
        # User field untouched:
        assert c.feature_set == "x_v1"
        # Cache populated:
        assert c._feature_indices_resolved == [0, 5, 12]
        assert c._feature_set_ref_resolved == ("x_v1", "a" * 64)


# ---------------------------------------------------------------------------
# Phase 4 Batch 4c hardening: ExperimentConfig.to_dict() must filter
# private cache fields (those starting with "_")
# ---------------------------------------------------------------------------


class TestToDictFiltersPrivateFields:
    """Agent 1's H1 finding: the private cache leaked through
    ``dataclasses.asdict()`` and thus through ``ExperimentConfig.to_dict()``
    / ``to_yaml()`` / ``to_json()``. Hardening fix filters `_`-prefixed
    fields at the dataclass serialization boundary so the on-disk YAML
    round-trip is preserved (feature_set in → feature_set out, no cache
    leakage).

    These tests LOCK that invariant. If someone removes the filter, the
    R3 invariant breaks and the producer's source-of-truth (the YAML
    file the user wrote) would silently corrupt after training.
    """

    def _make_experiment_config(self):
        from lobtrainer.config import ExperimentConfig
        # Default ExperimentConfig has DataConfig with no feature_set.
        return ExperimentConfig(name="test_fs_todict")

    def test_clean_config_has_no_private_fields(self):
        cfg = self._make_experiment_config()
        d = cfg.to_dict()
        # Recurse into data:
        assert "_feature_indices_resolved" not in d.get("data", {})
        assert "_feature_set_ref_resolved" not in d.get("data", {})
        # Also not at any other level:
        def _walk(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    assert not k.startswith("_"), (
                        f"Private field '{k}' leaked into to_dict() output"
                    )
                    _walk(v)
            elif isinstance(obj, list):
                for item in obj:
                    _walk(item)
        _walk(d)

    def test_resolved_cache_does_not_leak(self):
        # Simulate the trainer's post-resolve state: user set feature_set,
        # resolver populated the cache. to_dict() must hide the cache.
        cfg = self._make_experiment_config()
        cfg.data.feature_set = "momentum_v1"
        cfg.data._feature_indices_resolved = [0, 5, 12, 84, 85]
        cfg.data._feature_set_ref_resolved = ("momentum_v1", "a" * 64)

        d = cfg.to_dict()
        # User field preserved:
        assert d["data"]["feature_set"] == "momentum_v1"
        # Cache filtered:
        assert "_feature_indices_resolved" not in d["data"]
        assert "_feature_set_ref_resolved" not in d["data"]

    def test_yaml_round_trip_preserves_feature_set(self, tmp_path):
        # End-to-end R3: write to YAML, read back, verify the cache
        # state never crosses the serialization boundary.
        from lobtrainer.config import ExperimentConfig

        cfg = self._make_experiment_config()
        cfg.data.feature_set = "test_v1"
        cfg.data._feature_indices_resolved = [0, 5, 12]

        yaml_path = tmp_path / "roundtrip.yaml"
        cfg.to_yaml(str(yaml_path))
        text = yaml_path.read_text()
        assert "_feature_indices_resolved" not in text
        assert "_feature_set_ref_resolved" not in text
        assert "feature_set: test_v1" in text

    def test_cache_excluded_from_compare(self):
        # Two configs with the same user fields but different cache
        # state should compare equal — the resolver-populated cache is
        # an implementation detail, not part of semantic identity.
        c1 = DataConfig(feature_set="x_v1")
        c2 = DataConfig(feature_set="x_v1")
        c2._feature_indices_resolved = [0, 5, 12]
        c2._feature_set_ref_resolved = ("x_v1", "a" * 64)
        assert c1 == c2

    def test_cache_is_mutable(self):
        # field(init=False) + dataclass default => settable post-init.
        c = DataConfig(feature_set="x_v1")
        c._feature_indices_resolved = [0, 5, 12]
        assert c._feature_indices_resolved == [0, 5, 12]


# ---------------------------------------------------------------------------
# T13 auto-derivation of model.input_size (Phase 6 6A.1, 2026-04-17)
# ---------------------------------------------------------------------------


class TestT13InputSizeWithFeatureSet:
    """Phase 6 6A.1 regression guard — `data.feature_set` + `model.input_size=0`
    must raise a clear ValueError at config-load time. Auto-derivation is NOT
    available for feature_set because the resolver needs filesystem / registry
    context unavailable at `__post_init__`. Proper architectural fix (reorder
    `setup()` to run resolver before model construction) is deferred to
    Phase 7 lobtrainer-core split.

    Prior state (pre-6A.1): `__post_init__` silently used `feature_count` as
    input_size; trainer's model was built with wrong dim; crash deep in
    `setup()` at model constructor OR at `trainer.py:413` "input_size != resolved".

    Post-6A.1: __post_init__ raises with guidance ("set model.input_size
    explicitly OR use feature_preset/feature_indices for auto-derivation").
    """

    def _make_cfg(self, *, feature_set: str | None = None, input_size: int = 0):
        from lobtrainer.config.schema import ExperimentConfig, ModelConfig
        return ExperimentConfig(
            name="t13_feature_set_test",
            data=DataConfig(
                feature_count=98,
                feature_set=feature_set,
            ),
            model=ModelConfig(
                model_type="tlob",
                input_size=input_size,
                num_classes=3,
                task_type="regression",
            ),
        )

    def test_feature_set_with_input_size_zero_raises(self):
        """The error message must guide the user to set input_size OR switch
        to feature_preset/feature_indices (not just say "invalid")."""
        with pytest.raises(ValueError, match="feature_set.*model\\.input_size"):
            self._make_cfg(feature_set="nvda_98_stable_v1", input_size=0)

    def test_feature_set_with_explicit_input_size_accepted(self):
        """When input_size is set explicitly (non-zero), __post_init__ must
        NOT touch it. _create_dataloaders validates match at runtime."""
        cfg = self._make_cfg(feature_set="nvda_98_stable_v1", input_size=40)
        assert cfg.model.input_size == 40, (
            "__post_init__ must leave explicit input_size alone for feature_set"
        )

    def test_feature_preset_still_auto_derives(self):
        """Regression: auto-derivation for feature_preset still works
        (Phase 6 fix scoped only to feature_set case)."""
        from lobtrainer.config.schema import ExperimentConfig, ModelConfig
        # Use a known preset that exists; short_term_40 → len 40 indices.
        cfg = ExperimentConfig(
            name="t13_preset_test",
            data=DataConfig(
                feature_count=98,
                feature_preset="short_term_40",
            ),
            model=ModelConfig(
                model_type="tlob",
                input_size=0,  # auto-derive
                num_classes=3,
                task_type="regression",
            ),
        )
        # Auto-derived from preset (resolved at __post_init__ time).
        assert cfg.model.input_size == 40, (
            "feature_preset auto-derivation regressed (must still work)"
        )


# ---------------------------------------------------------------------------
# Phase 6 6A.5: from_yaml empty-file guard
# ---------------------------------------------------------------------------


class TestFromYamlEmptyFile:
    """Phase 6 6A.5 regression guard — empty YAML files must not crash.
    Prior behavior: `yaml.safe_load(f)` returned None; `resolve_inheritance(None, ...)`
    crashed on `data.pop('_partial')` with AttributeError. Post-fix: `or {}`
    defensive fallback returns a default-valued ExperimentConfig.
    """

    def test_empty_file_no_crash(self, tmp_path):
        """Completely empty YAML → default ExperimentConfig."""
        from lobtrainer.config.schema import ExperimentConfig
        p = tmp_path / "empty.yaml"
        p.write_text("")
        # Must NOT raise AttributeError on None
        cfg = ExperimentConfig.from_yaml(str(p))
        assert cfg is not None

    def test_null_yaml_no_crash(self, tmp_path):
        """Explicit `null` YAML → default ExperimentConfig."""
        from lobtrainer.config.schema import ExperimentConfig
        p = tmp_path / "null.yaml"
        p.write_text("null\n")
        cfg = ExperimentConfig.from_yaml(str(p))
        assert cfg is not None

    def test_triple_dash_only_no_crash(self, tmp_path):
        """Just the YAML doc-start marker → default ExperimentConfig."""
        from lobtrainer.config.schema import ExperimentConfig
        p = tmp_path / "dash.yaml"
        p.write_text("---\n")
        cfg = ExperimentConfig.from_yaml(str(p))
        assert cfg is not None
