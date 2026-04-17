"""
Config inheritance utilities for YAML experiment configs.

Provides deep dict merging and ``_base:`` inheritance resolution, enabling
experiment configs to inherit from base configs and override only the
fields that differ.

Semantics:
    - Dicts: recursively merged (base keys preserved unless overridden)
    - Lists: REPLACED entirely (not appended)
    - Scalars / None: override wins
    - ``_base: "path.yaml"`` — single base (v1 form, backward compat)
    - ``_base: ["a.yaml", "b.yaml", ...]`` — multi-base list (v2 form);
      merged left-to-right; each successive base overrides the previous;
      child overrides all accumulated bases.
    - ``_partial: true`` at the top level — sentinel marking a base that is
      standalone-invalid (only meaningful when composed with peer bases).
      Stripped silently during inheritance resolution so it never reaches
      the downstream dataclass validator. Callers (``ExperimentConfig.from_yaml``)
      check the RAW input YAML for this sentinel before calling us and raise
      a descriptive error if a partial base is loaded directly.

Reference: Follows the TypeScript ``extends`` / Tailwind ``@config`` pattern.

History:
    v1 (pre-2026-04-15): single-string ``_base:`` only. Archived to
    ``archive/merge-v1/merge.py`` (see ARCHIVE_README.md) after Phase 3 of
    the training-pipeline-architecture refactor promoted multi-base
    composition to a first-class feature. v2 preserves every v1 invariant
    byte-identically (verified by ``tests/test_merge_v1_parity.py`` against
    39 golden fixtures) and adds list-form ``_base`` support plus the
    ``_partial`` sentinel convention.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


_MAX_INHERITANCE_DEPTH = 10


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base``. Returns a new dict.

    Dicts are recursively merged. All other types (scalars, lists, None)
    in ``override`` replace the base value entirely.

    Neither input is mutated (base is shallow-copied; override is iterated).

    Args:
        base: Base configuration dict.
        override: Override dict whose values take precedence.

    Returns:
        New dict with merged values.

    Invariants (unchanged from v1; locked by ``test_config_inheritance.py``):
        - lists REPLACE (not append)
        - None explicitly overrides
        - new keys in override are added
        - dict ↔ scalar override: scalar replaces the whole subtree
    """
    result = base.copy()
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def resolve_inheritance(
    data: dict[str, Any],
    config_path: Path,
    *,
    _seen: frozenset[str] | None = None,
    _depth: int = 0,
) -> dict[str, Any]:
    """Resolve ``_base`` inheritance chain, returning a fully merged dict.

    Accepts either:
    - **v1 form**: ``_base: "path.yaml"`` — single-parent inheritance. Paths
      resolved relative to the directory of the config file containing the
      ``_base`` key.
    - **v2 form**: ``_base: ["a.yaml", "b.yaml", ...]`` — multi-base composition.
      Each entry is loaded (transitively resolving its own ``_base`` first),
      then merged left-to-right: each successive base overrides the previous;
      this config's own keys finally override everything.

    The ``_base`` key is removed from the returned dict. Cycle detection
    covers ALL transitively visited paths. Depth is capped at
    ``_MAX_INHERITANCE_DEPTH`` (= 10) per chain.

    The ``_partial: true`` sentinel is silently stripped from the merged
    result (it is a base-file marker, not user data).

    Args:
        data: Parsed YAML data dict (may contain a ``_base`` key). MUTATED:
            ``_base`` is popped from the input dict (pattern preserved from v1).
        config_path: Absolute path to the config file (for relative resolution).
        _seen: Visited config paths for cycle detection (internal).
        _depth: Current recursion depth (internal).

    Returns:
        Merged dict with ``_base`` and ``_partial`` removed.

    Raises:
        ValueError: If ``_base`` is malformed, a cycle is detected, or depth
            exceeds ``_MAX_INHERITANCE_DEPTH``.
        FileNotFoundError: If a referenced base config file does not exist.
    """
    if _seen is None:
        _seen = frozenset()

    # Entry-level depth + cycle checks — preserved from v1 merge.py:78-89.
    if _depth > _MAX_INHERITANCE_DEPTH:
        raise ValueError(
            f"Config inheritance depth exceeds {_MAX_INHERITANCE_DEPTH}. "
            f"Chain: {_seen}"
        )

    config_str = str(config_path.resolve())
    if config_str in _seen:
        raise ValueError(
            f"Config inheritance cycle detected: {config_path} "
            f"already visited. Chain: {sorted(_seen)}"
        )

    # Strip _partial silently (it is a base-file directive, not data).
    data.pop("_partial", None)

    # Mutation pattern preserved from v1 merge.py:91 — pop _base from input.
    base_ref = data.pop("_base", None)
    if base_ref is None:
        return data

    base_refs = _validate_base_ref(base_ref)

    # Merge left-to-right. Each base is recursively resolved (its own _base
    # expanded), then deep-merged onto the accumulator. Child overrides last.
    merged_base: dict[str, Any] = {}
    for b in base_refs:
        base_path_raw = Path(b)
        base_path = (
            base_path_raw.resolve()
            if base_path_raw.is_absolute()
            else (config_path.parent / base_path_raw).resolve()
        )
        if not base_path.exists():
            raise FileNotFoundError(
                f"Base config not found: {base_path} "
                f"(referenced by _base: '{b}' in {config_path})"
            )

        with open(base_path) as f:
            base_data = yaml.safe_load(f) or {}

        # Recurse — _seen includes THIS config so a base that references
        # back to us is detected at the base's own entry check.
        base_data = resolve_inheritance(
            base_data,
            base_path,
            _seen=_seen | {config_str},
            _depth=_depth + 1,
        )
        merged_base = deep_merge(merged_base, base_data)

    # Child overrides all accumulated bases (matches v1 "child over base" direction).
    return deep_merge(merged_base, data)


def is_partial_base(path: str | Path) -> bool:
    """Return True if the YAML at ``path`` declares ``_partial: true`` at top level.

    Used by ``ExperimentConfig.from_yaml`` to detect researchers who
    accidentally load a partial base directly (e.g., ``bases/models/tlob.yaml``)
    and raise a descriptive error instead of a confusing dacite failure
    deep in the validator.

    Returns False if the file doesn't exist, can't be parsed, or has no
    ``_partial: true`` marker.
    """
    p = Path(path)
    if not p.exists():
        return False
    try:
        with open(p) as f:
            raw = yaml.safe_load(f) or {}
    except Exception:  # noqa: BLE001
        return False
    return isinstance(raw, dict) and raw.get("_partial") is True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_base_ref(base_ref: Any) -> list[str]:
    """Validate ``_base:`` value and return a normalized ``list[str]``.

    Accepts (see truth table in plan §3.3):
        - ``str`` non-empty → ``[str]``
        - ``list[str]`` non-empty, every element a non-empty string → as-is

    Raises:
        ValueError: for any malformed form (empty list, wrong type,
            element that is None / non-string / empty string).
    """
    if isinstance(base_ref, str):
        if not base_ref.strip():
            raise ValueError("_base must be a non-empty file path string")
        return [base_ref]
    if isinstance(base_ref, list):
        if not base_ref:
            raise ValueError("_base list cannot be empty")
        for i, b in enumerate(base_ref):
            if b is None or not isinstance(b, str) or not b.strip():
                raise ValueError(
                    f"_base list must contain non-empty strings; "
                    f"got {b!r} at index {i}"
                )
        return list(base_ref)
    raise ValueError(
        f"_base must be str or list[str], got {type(base_ref).__name__}"
    )
