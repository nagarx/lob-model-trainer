"""
Config inheritance utilities for YAML experiment configs.

Provides deep dict merging and _base: inheritance resolution,
enabling experiment configs to inherit from base configs and
override only the fields that differ.

Semantics:
    - Dicts: recursively merged (base keys preserved unless overridden)
    - Lists: REPLACED entirely (not appended)
    - Scalars/None: override wins
    - _base paths: resolved relative to the config file containing the key

Reference: Follows the TypeScript extends / Tailwind @config pattern.
"""

from pathlib import Path
from typing import Any

import yaml


_MAX_INHERITANCE_DEPTH = 10


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Returns a new dict.

    Dicts are recursively merged. All other types (scalars, lists, None)
    in override replace the base value entirely.

    Args:
        base: Base configuration dict.
        override: Override dict whose values take precedence.

    Returns:
        New dict with merged values. Neither input is modified.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
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
    """Resolve _base inheritance chain, returning a fully merged dict.

    If the data dict contains a ``_base`` key, loads the referenced YAML
    file, recursively resolves its own ``_base`` (if any), then deep-merges
    this config's values on top of the resolved base. The ``_base`` key is
    removed from the returned dict.

    Args:
        data: Parsed YAML data dict (may contain a _base key).
        config_path: Absolute path to the config file (for relative resolution).
        _seen: Visited config paths for cycle detection (internal).
        _depth: Current recursion depth (internal).

    Returns:
        Merged dict with _base removed.

    Raises:
        ValueError: If _base is empty, a cycle is detected, or depth exceeds limit.
        FileNotFoundError: If the referenced base config file does not exist.
    """
    if _seen is None:
        _seen = frozenset()

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

    base_ref = data.pop("_base", None)
    if base_ref is None:
        return data

    if not isinstance(base_ref, str) or not base_ref.strip():
        raise ValueError(
            f"_base must be a non-empty file path string, "
            f"got {base_ref!r} in {config_path}"
        )

    # Resolve path relative to the config file's directory
    base_path = Path(base_ref)
    if not base_path.is_absolute():
        base_path = (config_path.parent / base_path).resolve()
    else:
        base_path = base_path.resolve()

    if not base_path.exists():
        raise FileNotFoundError(
            f"Base config not found: {base_path} "
            f"(referenced by _base: '{base_ref}' in {config_path})"
        )

    with open(base_path) as f:
        base_data = yaml.safe_load(f)

    # Recursively resolve the base's own _base (if any)
    base_data = resolve_inheritance(
        base_data,
        base_path,
        _seen=_seen | {config_str},
        _depth=_depth + 1,
    )

    # Child overrides base
    return deep_merge(base_data, data)
