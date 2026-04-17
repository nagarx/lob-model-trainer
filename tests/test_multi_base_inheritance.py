"""Tests for v2 multi-base inheritance (``_base: list[str]``).

These tests cover the NEW v2 semantics — list-form ``_base``. The existing
``tests/test_config_inheritance.py`` covers v1 semantics and is locked by
``tests/test_merge_v1_parity.py``.

Scope:
- List form validation (empty / non-string / null element rejection)
- Merge order (left-to-right; later overrides earlier; child overrides all)
- Diamond resolution (A, B both inherit from C; resolved via _base: [A, B])
- Transitive chains through list
- Cycle detection across list branches
- Depth guard through list branches
- Partial-base sentinel protection
- Real E5 4-base composition reproduces the monolith output
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from lobtrainer.config.merge import deep_merge, is_partial_base, resolve_inheritance


# -----------------------------------------------------------------------------
# List-form validation
# -----------------------------------------------------------------------------


class TestBaseRefValidation:
    """Malformed _base: values are rejected with clear error messages."""

    def test_list_with_single_string_equivalent_to_string(self, tmp_path):
        """_base: ["a.yaml"] behaves exactly like _base: "a.yaml"."""
        base_data = {"name": "base", "x": 1, "y": 2}
        base_path = tmp_path / "base.yaml"
        with open(base_path, "w") as f:
            yaml.dump(base_data, f)

        # Child A uses string form; child B uses single-element list form
        child_a = {"_base": "base.yaml", "x": 99}
        child_b = {"_base": ["base.yaml"], "x": 99}

        res_a = resolve_inheritance(child_a, tmp_path / "child_a.yaml")
        res_b = resolve_inheritance(child_b, tmp_path / "child_b.yaml")

        assert res_a == res_b == {"name": "base", "x": 99, "y": 2}

    def test_empty_list_raises(self, tmp_path):
        data = {"_base": [], "name": "child"}
        with pytest.raises(ValueError, match="_base list cannot be empty"):
            resolve_inheritance(data, tmp_path / "child.yaml")

    def test_list_with_empty_string_element_raises(self, tmp_path):
        data = {"_base": ["a.yaml", ""], "name": "child"}
        with pytest.raises(ValueError, match="non-empty strings"):
            resolve_inheritance(data, tmp_path / "child.yaml")

    def test_list_with_null_element_raises(self, tmp_path):
        data = {"_base": ["a.yaml", None], "name": "child"}
        with pytest.raises(ValueError, match="non-empty strings"):
            resolve_inheritance(data, tmp_path / "child.yaml")

    def test_list_with_single_null_raises(self, tmp_path):
        data = {"_base": [None], "name": "child"}
        with pytest.raises(ValueError, match="non-empty strings"):
            resolve_inheritance(data, tmp_path / "child.yaml")

    def test_list_with_int_element_raises(self, tmp_path):
        data = {"_base": ["a.yaml", 42], "name": "child"}
        with pytest.raises(ValueError, match="non-empty strings"):
            resolve_inheritance(data, tmp_path / "child.yaml")

    def test_base_as_int_raises(self, tmp_path):
        data = {"_base": 42, "name": "child"}
        with pytest.raises(ValueError, match="_base must be str or list"):
            resolve_inheritance(data, tmp_path / "child.yaml")

    def test_base_as_dict_raises(self, tmp_path):
        data = {"_base": {"a": 1}, "name": "child"}
        with pytest.raises(ValueError, match="_base must be str or list"):
            resolve_inheritance(data, tmp_path / "child.yaml")


# -----------------------------------------------------------------------------
# Multi-base merge order (core semantic)
# -----------------------------------------------------------------------------


class TestMultiBaseMergeOrder:
    """Multi-base merge: B overrides A; C overrides B; child overrides all."""

    def test_two_bases_later_wins(self, tmp_path):
        """_base: [A, B] — B's values override A's."""
        a = {"name": "A", "foo": 1, "a_only": "a"}
        b = {"name": "B", "foo": 2, "b_only": "b"}
        (tmp_path / "A.yaml").write_text(yaml.dump(a))
        (tmp_path / "B.yaml").write_text(yaml.dump(b))

        child = {"_base": ["A.yaml", "B.yaml"], "child_only": "c"}
        result = resolve_inheritance(child, tmp_path / "child.yaml")

        assert result == {
            "name": "B",       # B overrides A
            "foo": 2,          # B overrides A
            "a_only": "a",     # inherited from A (not in B)
            "b_only": "b",     # inherited from B
            "child_only": "c", # child-only
        }

    def test_child_overrides_all_bases(self, tmp_path):
        """Child-level keys always win, regardless of base order."""
        a = {"x": 1}
        b = {"x": 2}
        (tmp_path / "A.yaml").write_text(yaml.dump(a))
        (tmp_path / "B.yaml").write_text(yaml.dump(b))

        child = {"_base": ["A.yaml", "B.yaml"], "x": 99}
        result = resolve_inheritance(child, tmp_path / "child.yaml")

        assert result["x"] == 99

    def test_three_bases_left_to_right(self, tmp_path):
        """_base: [A, B, C] — merge A, then B over A, then C over B, then child."""
        (tmp_path / "A.yaml").write_text(yaml.dump({"x": 1, "a": "A"}))
        (tmp_path / "B.yaml").write_text(yaml.dump({"x": 2, "b": "B"}))
        (tmp_path / "C.yaml").write_text(yaml.dump({"x": 3, "c": "C"}))

        child = {"_base": ["A.yaml", "B.yaml", "C.yaml"], "child": "CHILD"}
        result = resolve_inheritance(child, tmp_path / "child.yaml")

        assert result == {
            "x": 3,        # C wins (last in list)
            "a": "A", "b": "B", "c": "C",
            "child": "CHILD",
        }

    def test_nested_dict_merges_across_bases(self, tmp_path):
        """Nested dicts recursively merge across multi-base list."""
        (tmp_path / "A.yaml").write_text(yaml.dump({
            "model": {"type": "tlob", "dropout": 0.1, "only_a": "A"},
        }))
        (tmp_path / "B.yaml").write_text(yaml.dump({
            "model": {"dropout": 0.2, "only_b": "B"},
            "data": {"feature_count": 98},
        }))
        child = {
            "_base": ["A.yaml", "B.yaml"],
            "model": {"dropout": 0.3, "child_override": "C"},
        }
        result = resolve_inheritance(child, tmp_path / "child.yaml")

        assert result == {
            "model": {
                "type": "tlob",            # from A
                "dropout": 0.3,            # child overrides both bases
                "only_a": "A",             # from A
                "only_b": "B",             # from B
                "child_override": "C",     # child-only
            },
            "data": {"feature_count": 98}, # from B
        }

    def test_list_value_is_replaced_not_merged_across_bases(self, tmp_path):
        """Lists REPLACE (invariant from v1): last-setter wins."""
        (tmp_path / "A.yaml").write_text(yaml.dump({"tags": ["a", "b"]}))
        (tmp_path / "B.yaml").write_text(yaml.dump({"tags": ["x"]}))

        child = {"_base": ["A.yaml", "B.yaml"]}
        result = resolve_inheritance(child, tmp_path / "child.yaml")
        assert result["tags"] == ["x"]   # B replaced A; child did not touch

        child_with_tags = {"_base": ["A.yaml", "B.yaml"], "tags": ["final"]}
        result = resolve_inheritance(child_with_tags, tmp_path / "child.yaml")
        assert result["tags"] == ["final"]  # child replaces all


# -----------------------------------------------------------------------------
# Diamond resolution
# -----------------------------------------------------------------------------


class TestDiamondResolution:
    """Diamond: A, B both inherit from C; child inherits via [A, B]."""

    def test_diamond_c_effectively_merged_twice_deep(self, tmp_path):
        """
        C → {x: "C", c_only: "C"}
        A inherits C, adds {a_only: "A", x: "A"}
        B inherits C, adds {b_only: "B"}  (does not override x)
        child has _base: [A, B]

        Resolution trace:
          load A: resolve_inheritance on A → deep_merge(C, A_own)
                = {x: "A", c_only: "C", a_only: "A"}
          load B: resolve_inheritance on B → deep_merge(C, B_own)
                = {x: "C", c_only: "C", b_only: "B"}
          merged_base: deep_merge({}, A_resolved) then deep_merge(that, B_resolved)
                     = deep_merge(A_resolved, B_resolved)
                     = {x: "C" (B_resolved wins), c_only: "C",
                        a_only: "A" (from A, B didn't override),
                        b_only: "B"}
          child: deep_merge(merged_base, child_own) → same (child has no overrides)

        Expected: {x: "C", c_only: "C", a_only: "A", b_only: "B"}

        KEY NUANCE: B did NOT override x, so x came from C via B — and
        B_resolved's x="C" wins over A_resolved's x="A" in the final merge.
        This is the "most specific parent wins for shared fields" behavior.
        """
        (tmp_path / "C.yaml").write_text(yaml.dump({"x": "C", "c_only": "C"}))
        (tmp_path / "A.yaml").write_text(yaml.dump({
            "_base": "C.yaml", "x": "A", "a_only": "A",
        }))
        (tmp_path / "B.yaml").write_text(yaml.dump({
            "_base": "C.yaml", "b_only": "B",
        }))

        child = {"_base": ["A.yaml", "B.yaml"]}
        result = resolve_inheritance(child, tmp_path / "child.yaml")

        assert result == {
            "x": "C",              # B_resolved wins; B_resolved got x="C" from C
            "c_only": "C",
            "a_only": "A",
            "b_only": "B",
        }

    def test_diamond_with_child_override_wins_all(self, tmp_path):
        """Child explicitly overriding the shared key wins regardless of branches."""
        (tmp_path / "C.yaml").write_text(yaml.dump({"x": "C"}))
        (tmp_path / "A.yaml").write_text(yaml.dump({"_base": "C.yaml", "x": "A"}))
        (tmp_path / "B.yaml").write_text(yaml.dump({"_base": "C.yaml", "x": "B"}))

        child = {"_base": ["A.yaml", "B.yaml"], "x": "CHILD"}
        result = resolve_inheritance(child, tmp_path / "child.yaml")
        assert result["x"] == "CHILD"


# -----------------------------------------------------------------------------
# Cycle detection & depth limits through list
# -----------------------------------------------------------------------------


class TestCycleDetectionThroughList:
    """Cycles via multi-base must be detected."""

    def test_direct_cycle_through_list(self, tmp_path):
        """child → [A]; A → [child]. child.yaml MUST exist on disk for the
        cycle to be detectable (A's recursion loads child.yaml from disk)."""
        child_path = tmp_path / "child.yaml"
        # Write child to disk so A can load it — THEN cycle detection fires
        child_path.write_text(yaml.dump({"_base": ["A.yaml"], "name": "child"}))
        (tmp_path / "A.yaml").write_text(
            yaml.dump({"_base": ["child.yaml"], "from_A": 1})
        )

        # Load the in-memory copy and kick off resolve
        with open(child_path) as f:
            child_data = yaml.safe_load(f)
        with pytest.raises(ValueError, match="cycle"):
            resolve_inheritance(child_data, child_path)

    def test_cycle_through_second_branch(self, tmp_path):
        """child → [A, B]; B → [child] (only B cycles — A is fine).

        child.yaml must exist on disk so B can load it before the cycle
        check at B's recursion entry point triggers."""
        child_path = tmp_path / "child.yaml"
        child_path.write_text(
            yaml.dump({"_base": ["A.yaml", "B.yaml"], "name": "child"})
        )
        (tmp_path / "A.yaml").write_text(yaml.dump({"a_only": 1}))
        (tmp_path / "B.yaml").write_text(
            yaml.dump({"_base": ["child.yaml"], "from_B": 1})
        )

        with open(child_path) as f:
            child_data = yaml.safe_load(f)
        with pytest.raises(ValueError, match="cycle"):
            resolve_inheritance(child_data, child_path)


class TestDepthLimitThroughList:
    """Depth cap of 10 is per chain; enforced through list branches too."""

    def test_deep_chain_through_single_branch_raises(self, tmp_path):
        """child → [deep]; deep has 12-level inheritance chain — should raise."""
        # Build chain: level_0 (no _base) → level_1 inherits level_0 → ... → level_11
        for i in range(12):
            if i == 0:
                data = {"name": f"level_{i}", "depth": i}
            else:
                data = {"_base": f"level_{i-1}.yaml", "name": f"level_{i}"}
            with open(tmp_path / f"level_{i}.yaml", "w") as f:
                yaml.dump(data, f)

        child = {"_base": ["level_11.yaml"], "name": "child"}
        with pytest.raises(ValueError, match="depth"):
            resolve_inheritance(child, tmp_path / "child.yaml")


# -----------------------------------------------------------------------------
# Partial-base sentinel protection
# -----------------------------------------------------------------------------


class TestPartialBaseSentinel:
    """_partial: true on a base file = standalone-invalid (base-only marker)."""

    def test_is_partial_base_detects_sentinel(self, tmp_path):
        partial = tmp_path / "partial.yaml"
        partial.write_text(yaml.dump({"_partial": True, "model": {"dropout": 0.1}}))
        assert is_partial_base(partial) is True

    def test_is_partial_base_absent(self, tmp_path):
        normal = tmp_path / "normal.yaml"
        normal.write_text(yaml.dump({"model": {"dropout": 0.1}}))
        assert is_partial_base(normal) is False

    def test_is_partial_base_explicit_false(self, tmp_path):
        no_partial = tmp_path / "no.yaml"
        no_partial.write_text(yaml.dump({"_partial": False, "model": {}}))
        assert is_partial_base(no_partial) is False

    def test_is_partial_base_missing_file(self, tmp_path):
        assert is_partial_base(tmp_path / "nope.yaml") is False

    def test_partial_stripped_from_merged_result(self, tmp_path):
        """When a child inherits from a partial base, _partial is stripped.

        This ensures the merged config that goes to dacite never sees
        _partial as a field (which would raise unknown-field errors).
        """
        (tmp_path / "partial.yaml").write_text(
            yaml.dump({"_partial": True, "model": {"type": "tlob"}})
        )
        child = {"_base": "partial.yaml", "name": "child"}
        result = resolve_inheritance(child, tmp_path / "child.yaml")

        assert "_partial" not in result
        assert result == {"model": {"type": "tlob"}, "name": "child"}

    def test_partial_stripped_from_top_level_child(self, tmp_path):
        """Even a child with _partial at top level gets it stripped on resolve.

        (Researchers should not do this at child level; the from_yaml caller
        detects it first and raises. This test documents the resolve-layer
        hygiene: strip the marker so it never reaches dacite.)
        """
        child = {"_partial": True, "name": "weird_child"}
        result = resolve_inheritance(child, tmp_path / "weird.yaml")
        assert result == {"name": "weird_child"}


# -----------------------------------------------------------------------------
# from_yaml integration: partial bases raise descriptive errors
# -----------------------------------------------------------------------------


class TestFromYamlPartialBaseProtection:
    """Loading a partial base directly via from_yaml raises a clear error."""

    def test_from_yaml_on_partial_base_raises(self, tmp_path):
        from lobtrainer.config import ExperimentConfig

        partial = tmp_path / "tlob_compact.yaml"
        partial.write_text(yaml.dump({
            "_partial": True,
            "model": {"model_type": "tlob", "input_size": 98, "num_classes": 3},
        }))

        with pytest.raises(ValueError, match="Partial base config cannot be loaded standalone"):
            ExperimentConfig.from_yaml(str(partial))


# -----------------------------------------------------------------------------
# Input mutation pattern (v1 invariant)
# -----------------------------------------------------------------------------


class TestInputMutationPattern:
    """_base pop-on-read preserved from v1 merge.py:91."""

    def test_base_is_popped_from_input(self, tmp_path):
        (tmp_path / "base.yaml").write_text(yaml.dump({"x": 1}))
        data = {"_base": "base.yaml", "name": "child"}

        resolve_inheritance(data, tmp_path / "child.yaml")

        # The ORIGINAL input dict had _base removed (v1 mutation pattern)
        assert "_base" not in data

    def test_base_as_list_is_also_popped(self, tmp_path):
        """Same behavior for the v2 list form."""
        (tmp_path / "A.yaml").write_text(yaml.dump({"a": 1}))
        (tmp_path / "B.yaml").write_text(yaml.dump({"b": 1}))
        data = {"_base": ["A.yaml", "B.yaml"], "name": "child"}

        resolve_inheritance(data, tmp_path / "child.yaml")
        assert "_base" not in data


# -----------------------------------------------------------------------------
# Real-world: E5 4-base composition reproduces monolith
# -----------------------------------------------------------------------------
# This was historically a single skipped test placeholder awaiting Phase 3.4.
# Coverage now lives in two places:
#   - `test_base_axis_ownership.py::TestChainedInheritancePurity` — locks the
#     bare/regression field split so the chained inheritance pattern is
#     mechanically protected.
#   - `test_merge_v1_parity.py::TestV1V2Parity::test_v2_matches_golden_fixture`
#     — verifies every migrated config (including all 5 E5 + E6) resolves
#     byte-identical to its pre-migration golden JSON.
# The placeholder test was removed post-Batch-1 as it added no coverage
# beyond these two files (R1/C1 regression guard + universal parity).
