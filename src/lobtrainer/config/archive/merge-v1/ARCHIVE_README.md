# Archived merge.py (v1)

**Archived**: 2026-04-15
**Reason**: Superseded by v2 `merge.py` with `_base: list[str]` multi-base composition support (Phase 3 of training-pipeline-architecture refactor).
**Validation**: v2 reproduces **byte-identical** resolved dicts on all 38 in-scope pre-migration configs (verified by `tests/test_merge_v1_parity.py` via 39 golden JSON fixtures under `tests/fixtures/golden/`).

## What's Here

| File | Contents | Lines |
|------|----------|-------|
| `merge.py` | Original `deep_merge()` + `resolve_inheritance()` (single-string `_base:` only) | 127 |

## How to Reference

This archive is **read-only**. It is NOT importable as a Python package (no `__init__.py` deliberately) and will NOT be discovered by pytest collection.

To look up how the v1 implementation resolved a config:
1. Open this file: `merge.py`
2. The v2 equivalent at `lob-model-trainer/src/lobtrainer/config/merge.py` preserves every v1 invariant and adds `_base: list[str]` support.

If you need to run the v1 function in a test (e.g., the parity test), load it via `importlib.util.spec_from_file_location`:

```python
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "archive_merge_v1",
    Path(__file__).parent / "src/lobtrainer/config/archive/merge-v1/merge.py",
)
v1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v1)

resolved = v1.resolve_inheritance(data, config_path)
```

## Key API Mappings (v1 → v2)

| v1 API | v2 API | Change |
|---|---|---|
| `resolve_inheritance(data, config_path, _seen=None, _depth=0)` | `resolve_inheritance(data, config_path, _seen=None, _depth=0)` | **Same signature**; `_base:` now accepts `str \| list[str]` |
| `deep_merge(base, override)` | `deep_merge(base, override)` | **Identical** (preserved for back-compat; tests import it directly) |
| `_MAX_INHERITANCE_DEPTH = 10` | `_MAX_INHERITANCE_DEPTH = 10` | **Unchanged** (depth guard + cycle detection preserved) |

## Why v1 Was Retired

- Single-string-only `_base:` required a monolithic "full-stack" base covering all 4 axes (model, dataset, label, train) — leading to duplicated content across experiment families.
- After decomposing `bases/e5_tlob_regression.yaml` into `bases/{models,datasets,labels,train}/*.yaml`, child configs need to inherit from multiple orthogonal bases — which v1 cannot express.
- Other validated invariants (cycle detection, depth cap, deep-merge, list-replace, relative-path resolution, mutation-on-pop) are all preserved in v2 bit-identically.

## Related Documents

- `/Users/knight/code_local/HFT-pipeline-v2/lob-model-trainer/MERGE_MIGRATION_PLAN.md` — step-by-step migration ledger (this file's parallel)
- `/Users/knight/.claude/plans/gentle-brewing-quail.md` §3.1 — plan specification
- `/Users/knight/code_local/HFT-pipeline-v2/feature-extractor-MBO-LOB/archive/monolith-v1/ARCHIVE_README.md` — precedent this archive follows
