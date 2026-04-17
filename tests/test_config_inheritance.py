"""
Tests for config inheritance via _base: key.

Validates deep_merge(), resolve_inheritance(), and ExperimentConfig.from_yaml()
with YAML config inheritance.
"""

from pathlib import Path

import pytest
import yaml

from lobtrainer.config.merge import deep_merge, resolve_inheritance
from lobtrainer.config import ExperimentConfig


# =============================================================================
# deep_merge unit tests
# =============================================================================


class TestDeepMerge:
    """Tests for deep_merge()."""

    def test_scalar_override(self):
        """Override replaces scalar values."""
        base = {"a": 1, "b": 2}
        override = {"b": 99}
        result = deep_merge(base, override)
        assert result == {"a": 1, "b": 99}

    def test_nested_dict_merge(self):
        """Nested dicts are merged recursively, preserving unmodified keys."""
        base = {"data": {"feature_count": 98, "stride": 1}}
        override = {"data": {"feature_count": 40}}
        result = deep_merge(base, override)
        assert result == {"data": {"feature_count": 40, "stride": 1}}

    def test_list_replaces(self):
        """Lists are replaced entirely, not appended."""
        base = {"tags": ["a", "b", "c"]}
        override = {"tags": ["x"]}
        result = deep_merge(base, override)
        assert result == {"tags": ["x"]}

    def test_none_overrides(self):
        """None in override explicitly sets value to None."""
        base = {"feature_preset": "mbo_98"}
        override = {"feature_preset": None}
        result = deep_merge(base, override)
        assert result == {"feature_preset": None}

    def test_new_key_added(self):
        """Keys in override not in base are added."""
        base = {"a": 1}
        override = {"b": 2}
        result = deep_merge(base, override)
        assert result == {"a": 1, "b": 2}

    def test_does_not_mutate_inputs(self):
        """Neither base nor override dict is modified."""
        base = {"data": {"x": 1}}
        override = {"data": {"y": 2}}
        base_copy = {"data": {"x": 1}}
        override_copy = {"data": {"y": 2}}

        deep_merge(base, override)

        assert base == base_copy
        assert override == override_copy

    def test_deeply_nested(self):
        """Three levels of nesting merge correctly."""
        base = {"a": {"b": {"c": 1, "d": 2}}}
        override = {"a": {"b": {"c": 99}}}
        result = deep_merge(base, override)
        assert result == {"a": {"b": {"c": 99, "d": 2}}}


# =============================================================================
# resolve_inheritance tests
# =============================================================================


class TestResolveInheritance:
    """Tests for resolve_inheritance()."""

    def test_no_base_passthrough(self, tmp_path):
        """Config without _base is returned unchanged."""
        data = {"name": "test", "data": {"feature_count": 98}}
        config_path = tmp_path / "child.yaml"
        config_path.touch()

        result = resolve_inheritance(data, config_path)
        assert result == {"name": "test", "data": {"feature_count": 98}}

    def test_single_level_inheritance(self, tmp_path):
        """Child inherits from base, overriding specific fields."""
        base_data = {
            "name": "base",
            "data": {"feature_count": 98, "stride": 1},
            "train": {"batch_size": 128},
        }
        base_path = tmp_path / "base.yaml"
        with open(base_path, "w") as f:
            yaml.dump(base_data, f)

        child_data = {
            "_base": "base.yaml",
            "name": "child",
            "train": {"batch_size": 64},
        }
        child_path = tmp_path / "child.yaml"

        result = resolve_inheritance(child_data, child_path)

        assert result["name"] == "child"
        assert result["data"]["feature_count"] == 98  # inherited
        assert result["data"]["stride"] == 1  # inherited
        assert result["train"]["batch_size"] == 64  # overridden

    def test_chained_inheritance(self, tmp_path):
        """A -> B -> C chain resolves correctly."""
        # C (grandparent)
        c_data = {"name": "C", "a": 1, "b": 2, "c": 3}
        c_path = tmp_path / "c.yaml"
        with open(c_path, "w") as f:
            yaml.dump(c_data, f)

        # B (parent, inherits C)
        b_data = {"_base": "c.yaml", "name": "B", "b": 20}
        b_path = tmp_path / "b.yaml"
        with open(b_path, "w") as f:
            yaml.dump(b_data, f)

        # A (child, inherits B)
        a_data = {"_base": "b.yaml", "name": "A", "a": 100}
        a_path = tmp_path / "a.yaml"

        result = resolve_inheritance(a_data, a_path)

        assert result["name"] == "A"  # from A
        assert result["a"] == 100  # overridden by A
        assert result["b"] == 20  # overridden by B
        assert result["c"] == 3  # inherited from C

    def test_cycle_detection_raises(self, tmp_path):
        """Circular _base references raise ValueError."""
        a_data = {"_base": "b.yaml", "name": "A"}
        a_path = tmp_path / "a.yaml"
        with open(a_path, "w") as f:
            yaml.dump(a_data, f)

        b_data = {"_base": "a.yaml", "name": "B"}
        b_path = tmp_path / "b.yaml"
        with open(b_path, "w") as f:
            yaml.dump(b_data, f)

        child_data = {"_base": "a.yaml", "name": "child"}
        child_path = tmp_path / "child.yaml"

        with pytest.raises(ValueError, match="cycle"):
            resolve_inheritance(child_data, child_path)

    def test_missing_base_raises(self, tmp_path):
        """Missing base file raises FileNotFoundError."""
        data = {"_base": "nonexistent.yaml", "name": "test"}
        config_path = tmp_path / "child.yaml"

        with pytest.raises(FileNotFoundError, match="nonexistent.yaml"):
            resolve_inheritance(data, config_path)

    def test_relative_path_resolution(self, tmp_path):
        """_base path resolves relative to config file directory."""
        # Create subdirectory structure
        bases_dir = tmp_path / "bases"
        bases_dir.mkdir()
        experiments_dir = tmp_path / "experiments"
        experiments_dir.mkdir()

        base_data = {"name": "base", "value": 42}
        base_path = bases_dir / "shared.yaml"
        with open(base_path, "w") as f:
            yaml.dump(base_data, f)

        # _base path is relative to the child config's directory
        child_data = {"_base": "../bases/shared.yaml", "name": "child"}
        child_path = experiments_dir / "exp.yaml"

        result = resolve_inheritance(child_data, child_path)
        assert result["name"] == "child"
        assert result["value"] == 42

    def test_empty_string_base_raises(self, tmp_path):
        """Empty _base string raises ValueError."""
        data = {"_base": "", "name": "test"}
        config_path = tmp_path / "child.yaml"

        with pytest.raises(ValueError, match="non-empty"):
            resolve_inheritance(data, config_path)

    def test_depth_limit_raises(self, tmp_path):
        """Exceeding max inheritance depth raises ValueError."""
        # Create a chain of 12 configs (exceeds limit of 10)
        for i in range(12):
            if i == 0:
                data = {"name": f"level_{i}", "depth": i}
            else:
                data = {"_base": f"level_{i-1}.yaml", "name": f"level_{i}"}
            path = tmp_path / f"level_{i}.yaml"
            with open(path, "w") as f:
                yaml.dump(data, f)

        child_data = {"_base": "level_11.yaml", "name": "child"}
        child_path = tmp_path / "child.yaml"

        with pytest.raises(ValueError, match="depth"):
            resolve_inheritance(child_data, child_path)

    def test_base_key_removed_from_result(self, tmp_path):
        """_base key does not appear in the resolved result."""
        base_data = {"name": "base", "value": 1}
        base_path = tmp_path / "base.yaml"
        with open(base_path, "w") as f:
            yaml.dump(base_data, f)

        child_data = {"_base": "base.yaml", "name": "child"}
        child_path = tmp_path / "child.yaml"

        result = resolve_inheritance(child_data, child_path)
        assert "_base" not in result


# =============================================================================
# Integration tests with ExperimentConfig
# =============================================================================


class TestFromYamlInheritance:
    """Integration tests for ExperimentConfig.from_yaml() with inheritance."""

    def test_from_yaml_without_inheritance(self, tmp_path):
        """Existing configs without _base still work."""
        config_data = {
            "name": "test",
            "data": {"feature_count": 98},
            "model": {"model_type": "tlob", "input_size": 98, "num_classes": 3},
            "train": {"task_type": "multiclass"},
        }
        config_path = tmp_path / "test.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        cfg = ExperimentConfig.from_yaml(str(config_path))
        assert cfg.name == "test"
        assert cfg.data.feature_count == 98

    def test_from_yaml_with_inheritance(self, tmp_path):
        """Full round-trip: base + child YAML -> merged ExperimentConfig."""
        base_data = {
            "name": "base",
            "data": {"feature_count": 98},
            "model": {
                "model_type": "tlob",
                "input_size": 98,
                "num_classes": 3,
                "dropout": 0.1,
            },
            "train": {"task_type": "multiclass", "batch_size": 128},
        }
        base_path = tmp_path / "base.yaml"
        with open(base_path, "w") as f:
            yaml.dump(base_data, f)

        child_data = {
            "_base": "base.yaml",
            "name": "child_experiment",
            "train": {"batch_size": 64},
        }
        child_path = tmp_path / "child.yaml"
        with open(child_path, "w") as f:
            yaml.dump(child_data, f)

        cfg = ExperimentConfig.from_yaml(str(child_path))
        assert cfg.name == "child_experiment"  # overridden
        assert cfg.data.feature_count == 98  # inherited
        assert cfg.train.batch_size == 64  # overridden
        assert cfg.model.dropout == 0.1  # inherited

    def test_cross_validation_fires_after_merge(self, tmp_path):
        """input_size / feature_count mismatch after merge raises ValueError."""
        base_data = {
            "data": {"feature_count": 98},
            "model": {"model_type": "tlob", "input_size": 98, "num_classes": 3},
            "train": {"task_type": "multiclass"},
        }
        base_path = tmp_path / "base.yaml"
        with open(base_path, "w") as f:
            yaml.dump(base_data, f)

        # Child changes feature_count but forgets to update input_size
        child_data = {
            "_base": "base.yaml",
            "name": "mismatch",
            "data": {"feature_count": 40},
        }
        child_path = tmp_path / "child.yaml"
        with open(child_path, "w") as f:
            yaml.dump(child_data, f)

        with pytest.raises(ValueError, match="model.input_size"):
            ExperimentConfig.from_yaml(str(child_path))

    def test_from_dict_rejects_base_key(self):
        """from_dict() raises ValueError if _base key is present."""
        data = {"_base": "base.yaml", "name": "test"}
        with pytest.raises(ValueError, match="_base key found"):
            ExperimentConfig.from_dict(data)


# =============================================================================
# Real config validation
# =============================================================================


class TestRealConfigInheritance:
    """Tests using actual E5 config files (verifies real-world correctness)."""

    @pytest.fixture
    def configs_dir(self):
        """Path to the actual configs directory."""
        path = Path(__file__).parent.parent / "configs"
        if not path.exists():
            pytest.skip("configs directory not found")
        return path

    # test_e5_base_loads_standalone REMOVED 2026-04-15: monolith
    # `bases/e5_tlob_regression.yaml` retired after all 5 E5 consumers
    # migrated to multi-base composition (Batch 1a). The monolith is no
    # longer a loadable standalone — its content decomposed into 4 partial
    # bases under `bases/{models,datasets,labels,train}/`. Equivalent
    # coverage: `test_multi_base_inheritance.py` + `test_merge_v1_parity.py`
    # (byte-identical parity on every E5 migrated config).

    def test_e5_cvml_inherits_via_multibase(self, configs_dir):
        """E5 60s Huber CVML config correctly composes its 4-base chain.

        Post-Batch-1 this config uses `_base: [models/tlob_compact_regression,
        datasets/nvda_e5_60s, labels/regression_huber, train/regression_default]`.
        Inherited field values below come from the multi-base chain
        (models/tlob_compact_bare → tlob_compact_regression for model fields;
        datasets/nvda_e5_60s for data; etc.).

        This test verifies the end-to-end from_yaml → dacite path on the
        canonical migrated E5 config, complementing the byte-identical
        resolved-dict parity guaranteed by `test_merge_v1_parity.py`.
        """
        child_path = configs_dir / "experiments" / "e5_60s_huber_cvml.yaml"
        if not child_path.exists():
            pytest.skip("E5 CVML config not found")
        cfg = ExperimentConfig.from_yaml(str(child_path))

        # Overridden in child
        assert cfg.name == "E5_60s_Huber_CVML"
        assert cfg.model.tlob_use_cvml is True
        assert cfg.model.tlob_cvml_out_channels == 49

        # Inherited via multi-base composition
        # (originally from the monolith; now from the decomposed 4 bases)
        assert cfg.data.feature_count == 98
        assert cfg.model.tlob_hidden_dim == 32
        assert cfg.model.tlob_num_layers == 2
        assert cfg.train.batch_size == 128
        assert cfg.train.learning_rate == 5e-4
