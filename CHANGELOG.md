# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.7.0] — 2026-04-24 — Phase A.5 Scope D (Pydantic Migration)

Major cycle: migrate every config class in the 9-class hierarchy from
``@dataclass`` + ``dacite`` to Pydantic v2 ``SafeBaseModel``. Retires 4
bug classes at the TYPE layer and closes 7 plan v4 bugs across the
trainer / exporter / callback surfaces.

Key architectural wins — all active at construction time:
  1. Silent mutation — ``frozen=True`` raises on every field assignment.
  2. Extra-field acceptance — ``extra="forbid"`` rejects typos at
     ``model_validate`` time (e.g., ``horizen_idx`` for ``horizon_idx``).
  3. Canonical-path-drift — ``config.labels`` fails with AttributeError
     at attribute access (only ``config.data.labels`` is declared).
  4. Silent-None field access — every field is typed; validators catch
     missing-required + type mismatches at construction time.

### Added
- **``lobtrainer.config.base.SafeBaseModel``** — shared Pydantic v2 base
  packaging ``frozen=True + extra="forbid" + strict=True`` via
  ``ConfigDict`` + ``model_copy(update=...)`` override that re-fires
  validators.
- **``LabelsConfig.validate_primary_horizon_idx_for(n_horizons)``** —
  SSoT primitive for horizon-index validation. Raises on negative idx
  (Python negative-indexing silently picks last-N) and out-of-bounds
  idx. Parametric helper ``_validate_horizon_idx_for`` supports future
  horizon fields via one-line wrappers.
- **``CalibrationContext`` TypedDict** (``lobtrainer.calibration.variance``
  inline) — typed observability surface for ``CalibrationResult``.
  Two initial fields: ``primary_horizon_idx``, ``method_variant``.
  Re-exported from ``lobtrainer.calibration``.
- **9 TestXxxPydantic regression-test classes** covering every migrated
  class (LabelsConfig, SequenceConfig, NormalizationConfig, SourceConfig,
  TrainConfig, CVConfig, DataConfig, ModelConfig, ExperimentConfig).
- **Live-YAML corpus regression test** — loads EVERY non-partial-base
  YAML under ``configs/`` to catch post-A.5.3i operator-facing schema
  regressions at CI time.

### Changed
- **``ExperimentConfig.from_dict``** body rewritten from dacite to
  ``cls.model_validate(data)``. Pydantic v2 handles nested BaseModel
  construction natively — every sub-config validator fires.
- **``CalibrationResult.metadata`` → ``CalibrationResult.context``** +
  **``calibrate_variance(..., metadata=...)`` kwarg → ``context=``**
  (plan v4 Python-side rename). JSON wire-format key PRESERVED as
  ``"metadata"`` in ``to_dict()`` output — downstream consumers reading
  ``signal_metadata.json["calibration"]["metadata"]`` unchanged.
- **``CVTrainer._build_fold_config``** — single atomic
  ``config.model_copy(update={...})`` replaces ``copy.deepcopy`` +
  field-assignment.
- **CLI override pattern** — all 3 production sites (cli.py,
  scripts/train.py, scripts/export_signals.py) migrated to unified
  two-layer ``model_copy`` pattern.
- **Callback ``except AttributeError`` widened** to
  ``except (AttributeError, TypeError)`` at 2 sites — fallback
  activations now log at INFO level (hft-rules §8).
- **Exporter slicing sites** use ``validate_primary_horizon_idx_for``
  SSoT method instead of ``... or 0`` coalesce.

### Removed
- **``dacite>=1.8``** runtime dependency. Subsequent ``pip install -e .``
  does not fetch it.
- **``_coerce_importance(config)``** module-level helper (ExperimentConfig
  ``@field_validator`` replaces it).
- **``_PYDANTIC_TYPE_HOOKS``** module-level dict (ex-dacite bridge table,
  no consumer post-migration).

### Fixed
- **Plan v4 bug #2**: exporter slicing now validates idx via SSoT method
  before slicing (pre-A.5.4 negative idx silently picked last-N column).
- **Plan v4 bug #4**: ``CalibrationResult.to_dict()`` shallow-copy
  nested-dict aliasing hazard structurally eliminated by TypedDict.
- **Plan v4 bug #5**: silent ``primary_horizon=None`` fallback now emits
  WARN log with actionable diagnostic.
- **Plan v4 bug #7**: callback catches TypeError alongside AttributeError.
- **5 pre-existing YAML schema bugs** surfaced by strict Pydantic —
  temporal_ridge / temporal_gradboost / xgboost YAMLs had fields outside
  declared ModelConfig schema that dacite silently dropped. All fixed
  to nest under ``model.params:``.

### Migration Guide

**Pre-A.5 → A.5.0+**:

Callers that mutated config fields in place must migrate to
``model_copy(update={...})``:

```python
# Before:
config.output_dir = "/new/path"
config.train.epochs = 50

# After:
config = config.model_copy(update={
    "output_dir": "/new/path",
    "train": config.train.model_copy(update={"epochs": 50}),
})
```

Callers using ``calibrate_variance(metadata=...)`` kwarg or
``CalibrationResult.metadata`` attribute must rename to ``context``
(JSON wire-format key in ``to_dict()`` output still uses ``"metadata"``):

```python
# Before:
result = calibrate_variance(preds, labels, metadata={...})
provenance = result.metadata

# After:
result = calibrate_variance(preds, labels, context={...})
provenance = result.context
```

### Commit chain (10 atomic commits on main)

- A.5.3a LabelsConfig (``1507b87``)
- A.5.3b SequenceConfig (``f32288f``)
- A.5.3c NormalizationConfig (``52516e5``)
- A.5.3d SourceConfig (``f54a838``)
- A.5.3e TrainConfig (``7c91170``)
- A.5.3f CVConfig (``26f6f2a``)
- A.5.3f.1 post-audit (``8e9d312``)
- A.5.3g DataConfig (``dd23333``)
- A.5.3h ModelConfig (``dd2bf20``)
- A.5.3i ExperimentConfig KEYSTONE (``d3d35d2``)
- A.5.4 horizon_idx SSoT method + hardening (``aedecbe``)
- A.5.5 CalibrationContext TypedDict (``bb5b566``)
- A.5.6 docs + E2E + hft-contracts bug #6 round-trip (this commit)

---


## Versioning Policy

- **v0.x** (current): minor versions may contain breaking changes.
  All breaking changes are documented in this file with migration notes.
- **v1.0** (planned): public API frozen. Strict semver thereafter:
  major bump for breaking changes, minor for additive, patch for fixes.

## Deprecation Policy

When an API is to be removed:
1. The minor release adding the replacement marks the old API with
   `DeprecationWarning` and links to the replacement.
2. The next-next minor release removes the old API.
3. Removals are documented in the [Removed] section of the relevant version.

---

## [0.4.0] — 2026-04-15

First logged release. Prior versions (0.1.x–0.3.x) are not documented here;
see `git log` and `EXPERIMENT_INDEX.md` for historical context.

Phase 3 of the training-pipeline-architecture refactor: config composition
moves from the pre-existing single-base monolith (`configs/bases/e5_tlob_regression.yaml`)
to a 4-axis axis-partitioned base hierarchy, with full byte-identity preserved
for all migrated configs.

### Added
- **`_base` multi-base composition** — `config/merge.py::resolve_inheritance` now
  accepts `_base: str | list[str]`. List form merges left-to-right (each
  successive base overrides the previous); child config overrides all accumulated
  bases. Single-string form preserved unchanged.
- **21 axis-partitioned base configs** under `configs/bases/` across 4 axes:
  - `models/` (5): `tlob_compact_bare`, `tlob_compact_regression`,
    `tlob_paper_classification`, `hmhp_cascade_bare`, `hmhp_cascade_regression`
  - `datasets/` (8): `nvda_e4_5s`, `nvda_e5_30s`, `nvda_e5_60s`,
    `nvda_xnas_128feat_full`, `nvda_40feat_short_term`,
    `nvda_98feat_triple_barrier`, `nvda_98feat_zscore_per_day`,
    `nvda_40feat_tlob_repo`
  - `labels/` (4): `regression_huber`, `tlob_smoothed`, `opportunity`,
    `triple_barrier_volscaled`
  - `train/` (4): `regression_default`, `classification_default`,
    `classification_triple_barrier`, `tlob_paper_classification_train`
- **Chained-inheritance patterns** (2, lock-tested by
  `tests/test_base_axis_ownership.py::TestChainedInheritancePurity`):
  - `tlob_compact_bare` → `tlob_compact_regression` (regression adds cvml
    defaults). E4 TLOB uses `bare` directly; E5/E6 use the full chain for
    byte-identity with pre-migration goldens.
  - `hmhp_cascade_bare` → `hmhp_cascade_regression` (regression adds
    `model_type: hmhp_regression` + `hmhp_regression_loss_type: huber`).
    HMHP classification/triple-barrier use `bare` directly; HMHP regression
    uses the chain.
- **`_partial: true` sentinel** — bases declaring this at top level are
  standalone-invalid (only valid when composed with peer bases).
  `config/merge.py::is_partial_base` detects and raises a descriptive error
  if `ExperimentConfig.from_yaml()` is called on a partial base directly.
- **`configs/bases/README.md`** — 4-axis ownership rule, chained-inheritance
  patterns, `_partial: true` convention, full per-axis base inventory.
- **`MERGE_MIGRATION_PLAN.md`** — Phase 3 migration ledger (mirrors
  `feature-extractor-MBO-LOB/MONOLITH_MIGRATION_PLAN.md` precedent).
- **25 of 42 in-scope experiment configs migrated** to axis-composed form
  across 3 batches (Batch 1: E4×1 + E5×5 + E6×1 = 7; Batch 2: HMHP×11 = 11;
  Batch 3: TLOB classif×7 = 7). 17 configs remain standalone by design
  (baselines×7, XGBoost×2, archive×6, niche HMHP×2 —
  see `MERGE_MIGRATION_PLAN.md`).

### Changed
- **`train.loss_type` ownership refinement (Batch 2)**: moved from
  `models/` → `labels/` axis. The field is **task-coupled**
  (regression → `huber`, tlob → `weighted_ce`, triple_barrier → `focal`),
  not model-coupled. HMHP cascade shares one model base across all three
  loss types instead of requiring three near-duplicate HMHP model bases.
  All migrated configs' resolved dicts unchanged (Batch 1 byte-identity
  preserved, Batches 2–3 golden-file parity passed).

### Removed
- **`configs/bases/e5_tlob_regression.yaml`** (monolith base) — retired at
  end of Batch 1 after all 5 E5 consumers migrated to the axis-composed
  form (`_base: [models/tlob_compact_regression, datasets/nvda_e5_60s,
  labels/regression_huber, train/regression_default]`) with byte-identical
  resolved dicts.

### Archived
- **`src/lobtrainer/config/archive/merge-v1/`** — v1 `merge.py` (single-string
  `_base:` only, 127 LOC) preserved read-only. Not importable as a package
  (no `__init__.py` by design; follows `feature-extractor-MBO-LOB/archive/monolith-v1/`
  precedent). Loaded via `importlib.util.spec_from_file_location` only from
  `tests/test_merge_v1_parity.py` for bit-for-bit regression checks against
  the in-scope pre-migration golden fixtures. See `ARCHIVE_README.md` in
  that directory.

### Fixed
- **(hft-ops companion fix, ledger-conflation guard)**:
  `hft-ops/src/hft_ops/ledger/dedup.py::compute_fingerprint` now resolves
  `_base:` inheritance in trainer YAMLs (both file-based `config:` and
  inline `trainer_config:` paths) before hashing, via a torch-free
  `importlib.util.spec_from_file_location` load of `lobtrainer.config.merge`.
  Hard errors (cycle, depth, malformed `_base`) propagate; soft I/O errors
  (`OSError`, `YAMLError`) log a warning and return an empty dict (matching
  the pre-existing fail-safe).

  **Why it matters**: pre-fix, mutating a shared base (e.g., changing
  `bases/train/regression_default.yaml: epochs 30 → 40`) left every
  dependent experiment's fingerprint unchanged — every pre/post-change
  run was silently conflated in the ledger. Post-fix fingerprints are
  **content-addressed** over the resolved effective dict, so base mutations
  correctly invalidate dependent-experiment fingerprints.

  Regression test: `hft-ops/tests/test_fingerprint_base_mutation.py`.

### Tests
- **21 pre-existing tests** (pytest-collected) in `tests/test_config_inheritance.py`
  continue to pass unchanged (all v1 semantics preserved: depth cap=10,
  entry-level cycle detection, list-REPLACE, None overrides, pop-on-read
  mutation, relative-path resolution). Static count `grep "def test_"` = 22;
  pytest discovers 21 (one method skipped at collection).
- **New tests**:
  - `tests/test_multi_base_inheritance.py` — list-form merge semantics,
    diamond resolution, left-to-right order.
  - `tests/test_merge_v1_parity.py` — byte-identity parity between v1
    (loaded via `importlib.util.spec_from_file_location` from
    `archive/merge-v1/merge.py`) and v2 on every in-scope pre-migration
    golden JSON fixture.
  - `tests/test_base_axis_ownership.py` — mechanical enforcement of the
    §3.4 ownership rule (every top-level dotted-key appears in exactly
    one axis directory); includes `TestChainedInheritancePurity` locking
    the TLOB compact and HMHP cascade chained patterns.
  - `tests/test_migrated_configs_e2e.py` — auto-discovers migrated configs
    and verifies `ExperimentConfig.from_yaml` loads each without error;
    meta-tests guarantee every E-family / HMHP-family / TLOB-classif-family
    member is migrated.
  - `hft-ops/tests/test_fingerprint_base_mutation.py` — content-addressed
    fingerprint regression guards for the companion hft-ops fix above.

### Notes
- Phase 3 work is scoped to config-composition infrastructure only. Next
  phase (Phase 4 — FeatureSet Registry) will land feature-set resolution
  via content-addressed JSON registry.
