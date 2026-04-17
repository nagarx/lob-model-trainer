"""Pre-migration golden snapshot generator for Phase 3 config-composition refactor.

Usage:
    python tests/fixtures/golden/generate_snapshots.py

This captures every current config's resolved dict via v1 ``resolve_inheritance``
as a JSON fixture. The output becomes the non-negotiable acceptance criterion
for Phase 3a: the new multi-base ``merge.py`` MUST reproduce these fixtures
bit-identically (content AND error-path) for every non-migrated config.

Fixtures written to ``tests/fixtures/golden/{experiments,bases}/<name>.json``
mirroring the ``configs/`` directory structure.

Each fixture has one of two forms:
    {"status": "ok", "resolved": {...}}                  # successful resolution
    {"status": "error", "exc_type": "...", "message": "..."}  # expected failure

Scope:
    - IN: configs/experiments/*.yaml (40 files)
    - IN: configs/bases/*.yaml (1 file — the pre-decomposition monolith)
    - EXCLUDED: configs/archive/** (legacy datasets, out of Phase 3 scope)
    - EXCLUDED: XGBoost configs (different schema, bypass ExperimentConfig.from_yaml)

Keying: relative path from ``configs/`` root (CWD-independent).
Serialization: ``json.dumps(..., sort_keys=True, indent=2, default=str)``.

Design: deterministic; re-running produces identical output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

# Path gymnastics: this script lives at
#   lob-model-trainer/tests/fixtures/golden/generate_snapshots.py
# The trainer source is at
#   lob-model-trainer/src/lobtrainer/...
# We want to import lobtrainer.config.merge to use v1 resolve_inheritance.
_TRAINER_ROOT = Path(__file__).resolve().parents[3]  # lob-model-trainer/
_CONFIGS_ROOT = _TRAINER_ROOT / "configs"
_GOLDEN_ROOT = Path(__file__).resolve().parent

# Ensure the trainer package is importable
_SRC = _TRAINER_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import yaml  # noqa: E402

from lobtrainer.config.merge import resolve_inheritance  # noqa: E402


# Configs excluded from golden fixtures per plan §3.0 / §3.6
_EXCLUDED_EXPERIMENTS = {
    # XGBoost configs use a different schema (experiment:+output: nesting)
    # and bypass ExperimentConfig.from_yaml. Out of Phase 3 scope.
    "nvda_xgboost_baseline_h60.yaml",
    "nvda_xgboost_baseline_arcx_h60.yaml",
}


def _snapshot_one(yaml_path: Path) -> Dict[str, Any]:
    """Resolve one config via v1 ``resolve_inheritance`` and capture the result.

    Returns one of:
        {"status": "ok", "resolved": {...dict...}}
        {"status": "error", "exc_type": "ClassName", "message": "..."}
    """
    try:
        with open(yaml_path, "r") as f:
            raw = yaml.safe_load(f)
        if raw is None:
            raw = {}
        resolved = resolve_inheritance(raw, yaml_path.resolve())
        # Round-trip through JSON to catch non-serializable values early
        # default=str mirrors the plan's serialization spec
        roundtrip = json.loads(
            json.dumps(resolved, sort_keys=True, default=str)
        )
        return {"status": "ok", "resolved": roundtrip}
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "exc_type": type(exc).__name__,
            "message": str(exc),
        }


def _write_fixture(rel_key: str, snapshot: Dict[str, Any]) -> Path:
    """Write a fixture under tests/fixtures/golden/<rel_key>.json."""
    # rel_key is like "experiments/e5_60s_huber_cvml.yaml"
    # → fixture at golden/experiments/e5_60s_huber_cvml.json
    out_path = _GOLDEN_ROOT / rel_key
    out_path = out_path.with_suffix(".json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(snapshot, f, sort_keys=True, indent=2, default=str)
        f.write("\n")
    return out_path


def generate_all() -> Dict[str, Dict[str, Any]]:
    """Generate golden fixtures for every in-scope config.

    Returns:
        dict mapping rel_key → snapshot (for reporting).
    """
    results: Dict[str, Dict[str, Any]] = {}

    # (1) All experiments (minus excluded)
    for yaml_path in sorted((_CONFIGS_ROOT / "experiments").glob("*.yaml")):
        if yaml_path.name in _EXCLUDED_EXPERIMENTS:
            continue
        rel_key = str(yaml_path.relative_to(_CONFIGS_ROOT))
        snap = _snapshot_one(yaml_path)
        _write_fixture(rel_key, snap)
        results[rel_key] = snap

    # (2) All pre-existing bases (currently: 1 monolith)
    for yaml_path in sorted((_CONFIGS_ROOT / "bases").glob("*.yaml")):
        rel_key = str(yaml_path.relative_to(_CONFIGS_ROOT))
        snap = _snapshot_one(yaml_path)
        _write_fixture(rel_key, snap)
        results[rel_key] = snap

    return results


def _report(results: Dict[str, Dict[str, Any]]) -> None:
    """Print a summary of snapshot outcomes."""
    ok = [k for k, v in results.items() if v["status"] == "ok"]
    err = [(k, v) for k, v in results.items() if v["status"] == "error"]

    print(f"Generated {len(results)} golden fixtures under {_GOLDEN_ROOT}")
    print(f"  OK: {len(ok)}")
    print(f"  ERROR: {len(err)}")
    if err:
        print("\nFailing configs (captured in fixtures with exc_type+message):")
        for k, v in err:
            print(f"  {k}: {v['exc_type']}: {v['message'][:100]}")


if __name__ == "__main__":
    results = generate_all()
    _report(results)
