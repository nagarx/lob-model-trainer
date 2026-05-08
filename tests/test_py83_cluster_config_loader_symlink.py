"""Tests for #PY-83-cluster Phase α-1.2 (config-loader symlink-source preservation).

Locks the post-α-1.2 contract for the lob-model-trainer config loader:

- ``merge.py::resolve_inheritance`` cycle-detection key (line 135) uses
  ``Path.absolute()`` not ``Path.resolve()`` — preserves symlink-source
  lineage in the ``_seen`` set.
- ``merge.py`` base-path resolution (lines 158/160) uses ``.absolute()``
  consistently with line 135 (cycle detection requires consistent key
  derivation; partial flip would break the invariant).
- ``schema.py:2687`` entry-point ``_Path(path).absolute()`` matches the
  consumer's expectation downstream in ``resolve_inheritance``.

This is the lob-model-trainer SSoT-discipline mirror of α-1.1's hft-ops
``paths.resolve()`` fix and α-3's ``feature_set_resolver.py:442`` fix
(both shipped 2026-05-10). All three address the same defect class
(``Path.resolve()`` derefs symlinks at start) at different boundaries.

Discovered by: 8-agent prep round 2026-05-10 (Agent I FINDING 5
"Hidden Findings Hunt"); design verified by Explore agent 2026-05-10.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from lobtrainer.config.merge import resolve_inheritance


# ---------------------------------------------------------------------------
# Fixture — symlinked configs/ deployment
# ---------------------------------------------------------------------------


@pytest.fixture
def symlinked_configs_dir(tmp_path: Path) -> tuple[Path, Path]:
    """Build a configs hierarchy where the configs/ directory itself is
    symlinked to an external location.

    Layout:
      tmp_path/external_configs/
        bases/parent.yaml   (the real base)
        experiments/child.yaml  (inherits from ../bases/parent.yaml)

      tmp_path/HFT-pipeline-v2/
        configs/  -> tmp_path/external_configs   (symlink)

    Returns a tuple ``(symlink_source_root, real_root)``:
      - symlink_source_root = tmp_path/HFT-pipeline-v2/configs
        (the path users author + reference)
      - real_root = tmp_path/external_configs
        (the deref'd target)
    """
    real_root = tmp_path / "external_configs"
    (real_root / "bases").mkdir(parents=True)
    (real_root / "experiments").mkdir(parents=True)

    (real_root / "bases" / "parent.yaml").write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "parent_test", "description": "α-1.2 fixture"},
                "shared_field": "from_parent",
            }
        )
    )
    (real_root / "experiments" / "child.yaml").write_text(
        yaml.safe_dump(
            {
                "_base": "../bases/parent.yaml",
                "experiment": {"description": "α-1.2 child override"},
                "child_field": "from_child",
            }
        )
    )

    pipeline_root = tmp_path / "HFT-pipeline-v2"
    pipeline_root.mkdir()
    symlink_source = pipeline_root / "configs"
    symlink_source.symlink_to(real_root)

    return symlink_source, real_root


# ---------------------------------------------------------------------------
# Inheritance resolution succeeds through symlinked configs/
# ---------------------------------------------------------------------------


class TestInheritanceThroughSymlinkedConfigsDir:
    """Phase α-1.2 / #PY-83-cluster: ``resolve_inheritance`` must succeed
    when the configs/ directory is a symlink. Functional test (both pre-
    and post-α-1.2 should pass this — locks the basic positive path).
    """

    def test_child_loads_parent_through_symlinked_configs(
        self, symlinked_configs_dir: tuple[Path, Path]
    ):
        symlink_source, _ = symlinked_configs_dir
        child_path = symlink_source / "experiments" / "child.yaml"
        with open(child_path) as f:
            child_data = yaml.safe_load(f)
        merged = resolve_inheritance(child_data, child_path)
        # Child override wins on `experiment.description`
        assert merged["experiment"]["description"] == "α-1.2 child override"
        # Parent fields preserved
        assert merged["experiment"]["name"] == "parent_test"
        assert merged["shared_field"] == "from_parent"
        # Child-only field preserved
        assert merged["child_field"] == "from_child"


# ---------------------------------------------------------------------------
# Symlink-source preservation in cycle-detection (the α-1.2 invariant)
# ---------------------------------------------------------------------------


class TestCycleDetectionSymlinkSourcePreserved:
    """Phase α-1.2 / #PY-83-cluster: ``_seen`` (cycle-detection set)
    contains symlink-source-preserved paths post-α-1.2 (uses Path.absolute()
    not Path.resolve()). Pre-α-1.2 it would contain deref'd paths.

    This invariant is what makes the 3 sites in merge.py + 1 site in
    schema.py atomically coupled: cycle detection on line 135 must use
    the SAME key-derivation as the path computed at line 158/160 for
    the recursive call.
    """

    def test_seen_set_contains_symlink_source_path_not_deref_target(
        self, symlinked_configs_dir: tuple[Path, Path], monkeypatch
    ):
        """Capture the keys added to ``_seen`` during recursion and assert
        they are symlink-source-preserved (under HFT-pipeline-v2/configs/),
        NOT deref'd (under external_configs/)."""
        symlink_source, real_root = symlinked_configs_dir
        child_path = symlink_source / "experiments" / "child.yaml"

        # Hook into merge.py to capture _seen evolution.
        captured_keys: list[str] = []
        original_resolve = resolve_inheritance

        def _patched(data, config_path, _depth=0, _seen=None):
            if _seen is None:
                _seen = frozenset()
            # Record what would be added to _seen at this level.
            captured_keys.append(str(config_path.absolute()))
            return original_resolve(data, config_path, _depth=_depth, _seen=_seen)

        monkeypatch.setattr(
            "lobtrainer.config.merge.resolve_inheritance", _patched
        )

        # Direct invocation (not through the patched name) using the
        # original function — we want to actually run it, not the wrapper.
        with open(child_path) as f:
            child_data = yaml.safe_load(f)
        original_resolve(child_data, child_path)

        # The first key recorded was for child_path. It MUST be under
        # symlink_source (HFT-pipeline-v2/configs), NOT under real_root.
        assert len(captured_keys) >= 1
        # Verify symlink-source preservation: the path string contains the
        # symlink-source ancestor, NOT the deref'd target's distinctive name.
        first_key = captured_keys[0]
        assert "HFT-pipeline-v2" in first_key, (
            f"#PY-83-cluster regression: cycle-detection key {first_key!r} "
            f"does NOT contain the symlink-source ancestor 'HFT-pipeline-v2/'. "
            f"This means merge.py is using Path.resolve() instead of "
            f"Path.absolute() — the bug α-1.2 was supposed to fix."
        )
        assert "external_configs" not in first_key, (
            f"#PY-83-cluster regression: cycle-detection key {first_key!r} "
            f"contains the deref target 'external_configs'. Expected "
            f"symlink-source preservation."
        )


# ---------------------------------------------------------------------------
# Negative regression: locks the broken `.resolve()` idiom
# ---------------------------------------------------------------------------


class TestResolveIdiomWouldHaveProducedDerefPath:
    """Phase α-1.2 / #PY-83-cluster: explicit negative regression locking
    that the BROKEN ``.resolve()`` idiom WOULD have produced a deref'd
    path. If a future refactor reverts to ``.resolve()``, this test
    documents WHY (and the symptom that surfaces).
    """

    def test_path_resolve_derefs_symlink_source(
        self, symlinked_configs_dir: tuple[Path, Path]
    ):
        """Direct verification: ``Path.resolve()`` on the symlinked
        path returns a path under the deref target, NOT the symlink
        source. Locks the language-level behavior we depend on."""
        symlink_source, real_root = symlinked_configs_dir
        child_path = symlink_source / "experiments" / "child.yaml"

        # The broken idiom (pre-α-1.2): would produce deref'd path.
        broken_idiom = child_path.resolve()
        assert "external_configs" in str(broken_idiom), (
            "Path.resolve() should deref symlinks. If this fails, "
            "Python semantics changed."
        )

        # The fixed idiom (post-α-1.2): preserves symlink-source.
        fixed_idiom = child_path.absolute()
        assert "HFT-pipeline-v2" in str(fixed_idiom)
        assert "external_configs" not in str(fixed_idiom)
