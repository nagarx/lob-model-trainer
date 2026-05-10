# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased] — Phase Y / γ-1 LITE / #PY-88 Phase 2 (2026-05-10 evening) — sklearn `LabelsConfig.return_type` axis closure + bundled top-level `return_type` emission

**Fixed (#PY-88 — sklearn label dispatch honors `return_type`)**

Pre-#PY-88, `simple_trainer._load_split` read cached `*_regression_labels.npy`
regardless of `LabelsConfig.return_type`. γ-1 LITE empirical gate 2026-05-09
night confirmed 6 sklearn arms produced **bit-exact identical metrics**
across the `point_return / smoothed_return` axis values exercised by
`hft-ops/experiments/sweeps/cycle5_multi_arm.yaml` (the bug applies
symmetrically to all 4 values in `_VALID_RETURN_TYPES = {smoothed_return,
point_return, mean_return, peak_return}` per `config/schema.py:325-327`)
— proving the data-path was bypassing the configured return_type.

This release closes the cosmetic-axis bug at the architectural source by
mirroring the PyTorch path's 3-way dispatch on `LabelsConfig.source` ∈
`{"auto", "precomputed", "forward_prices"}` (per `_VALID_SOURCES` enum at
`config/schema.py:322-324`):

- `src/lobtrainer/training/simple_trainer.py` — extended `_load_split` with
  `labels_config: Optional[LabelsConfig] = None` keyword-only kwarg + NEW
  helper `_resolve_labels_for_day` (~140 LOC) implementing 3-way dispatch:
  - **Branch 1** (`labels_config is None` OR `source == "precomputed"`):
    legacy cached `*_regression_labels.npy` path. Preserves pre-#PY-88
    sklearn behavior + back-compat for flat-keyword `SimpleModelTrainer(...)`
    construction (no `from_config`).
  - **Branch 2** (`source == "forward_prices"`): ALWAYS recompute via
    `LabelFactory.multi_horizon(forward_prices, horizons, k, return_type)`.
    Mid-impl HIGH-1 fix added explicit `ValueError` BEFORE
    `from_metadata` call when metadata lacks `forward_prices.exported=True`
    — mirrors PyTorch path's `dataset.py:657-661` exactly for cross-trainer
    UX symmetry (pre-fix sklearn raised `KeyError`, breaking parity).
  - **Branch 3** (`source == "auto"`): recompute IF metadata declares
    `forward_prices.exported=True` AND fp.npy exists; else fall back to
    cached labels (legacy v2.2 export back-compat).
- `setup()` lines 498-526 — passes `labels_config=self.config.data.labels`
  to all 3 `_load_split` calls; legacy flat-keyword path passes None.
- `k = ForwardPriceContract.smoothing_window_offset` ALWAYS from per-day
  metadata (Bug-B2 invariant; mirrors PyTorch `dataset.py:421-422`).

**Added (Step 3 — top-level `return_type` cosmetic emission)**

- `src/lobtrainer/export/metadata.py::build_signal_metadata` — new optional
  `return_type: Optional[str] = None` kwarg. When provided, emits at
  `signal_metadata.json` top level for human-visible operator queries.
  The `compatibility_fingerprint` already encodes this opaquely via
  `compute_label_strategy_hash`'s `model_dump()` payload — top-level
  string field aids backtester / ledger / dashboard filtering by
  return_type axis without parsing the nested compatibility block.
- `src/lobtrainer/training/simple_trainer.py::export_signals` (sklearn
  path) + `src/lobtrainer/export/exporter.py::SignalExporter._build_metadata`
  (PyTorch path) — both pass `return_type=labels_config.return_type`.

**Tests added (27 new tests)**

- `tests/test_simple_trainer.py:simple_data_dir` — fixture upgraded with
  `*_forward_prices.npy` shape `(N, 306)` float64 (pseudo-random walk) +
  metadata `forward_prices: {exported, smoothing_window_offset, max_horizon,
  n_columns}` block + `horizons` list. Locks the `ForwardPriceContract`
  invariant `n_columns == k + max_H + 1 = 306` (k=5, max_H=300).
- `tests/test_simple_trainer_signal_metadata_compat.py:synthetic_data_dir`
  + inline `basic_synthetic_60s` fixture in `TestSklearnDataSourceTagging
  ::test_basic_prefix_data_dir_yields_off_exchange_tag` — same upgrade.
- **NEW** `tests/test_simple_trainer_return_type.py` (27 tests across 5
  classes): `TestSourceDispatch` (5 tests covering Branches 1/2/3) +
  `TestReturnTypeDispatch` (cosmetic-axis bug fix proof + parametric
  parity over 4 return_types × 3 horizons = 12 cases) +
  `TestHorizonsTruthPin` (LabelsConfig.horizons override) +
  `TestFailLoudGuards` (mid-impl MED-2 closure: 4 fail-loud guard tests
  for missing fp.npy / missing block / block exported=false / unknown
  source) + `TestSignalMetadataReturnTypeEmission` (parametric over 4
  return_types verifying top-level field emission).

**Test counts**: **1747 passed + 73 skipped** (was 1720 + 73 — net +27).

**Cross-cycle bundle context**

This release is Commit 3 of 3 in the Phase Y / γ-1 LITE close-out bundle:

1. **hft-contracts `89ca163`** (PUSHED 2026-05-10) — `INDEX_SCHEMA_VERSION`
   1.5.0→1.6.0 + `model_config_hash` top-level mirror in
   `ExperimentRecord.index_entry()` projection (#PY-94 reframed closure).
2. **hft-ops `097e83c`** (PUSHED 2026-05-10) — `--model-config-hash` CLI
   filter on `ledger list` + `ExperimentLedger.filter` kwarg.
3. **THIS lob-model-trainer commit** (#PY-88 sklearn return_type closure +
   Step 3 top-level emission) + bundled hft-ops Agent C ledger gaps
   closure (#PY-95 + #PY-96 + #PY-97 — see hft-ops/CHANGELOG.md).

**Empirical validation gate** (Step 6 γ-1 LITE re-run, ~75-120 min compute):
12-arm sweep verifies 6 sklearn arms produce 6 DISTINCT compatibility_fingerprint
values (vs 2 pre-fix) + per-arm test_ic differing across return_type axis (vs
bit-exact identical pre-fix). Closes #PY-88 with empirical proof.

---

## [Unreleased] — HYBRID Phase α-1.2 (2026-05-10) — config-loader symlink-source preservation (#PY-83-cluster)

**Fixed (#PY-83-cluster — α-1.2 follow-up to α-3 / #PY-79)**

- `src/lobtrainer/config/merge.py::resolve_inheritance` — 3 `.resolve()`
  calls flipped to `.absolute()` (preserves symlink-source lineage):
  - line 135 (cycle-detection key derivation in `_seen` set)
  - line 158 (absolute-base path resolution)
  - line 160 (relative-base path resolution via `config_path.parent / `)
- `src/lobtrainer/config/schema.py:2687` — `_Path(path).resolve()` →
  `.absolute()`. This is the upstream entry-point for
  `resolve_inheritance`; its output propagates into merge.py's cycle-
  detection AND parent-relative base lookup, so all 4 sites MUST flip
  together (cycle-detection consistency invariant).

**Rationale**

Per α-3 / #PY-79 lesson (closed in commit `c232cf3`): when the configs/
or any ancestor is a symlink, `Path.resolve()` derefs the symlink at
start, which can:
- Produce inconsistent cycle-detection keys (deref'd vs symlink-source
  paths disagree on equality)
- Emit confusing diagnostics that cite deref'd paths the user never
  authored
- Diverge from the rest of the lob-models / lob-model-trainer / hft-ops
  ecosystem now uniformly using `.absolute()` per α-1.1 / α-3 / α-1.2

`Path.absolute()` is purely lexical (no FS access; never derefs);
mirrors α-3 fix philosophy. Sister to α-1.1 hft-ops `paths.resolve()`
fix (#PY-83) shipped in hft-ops commit `a00a799`.

**Kept (intentional)**

- `src/lobtrainer/analysis/stat_rigor/ci.py:444` — `signals_dir.resolve()`
  KEPT by design + intent comment added. This populates
  `signal_export_output_dir` in the metadata overlay used for
  compatibility-fingerprint matching; canonicalizing across symlinks
  IS desirable here (the (a) "needs symlink-DEREF" case from the audit).

**Tests**

- NEW `tests/test_py83_cluster_config_loader_symlink.py` (3 tests in 3
  classes covering inheritance-through-symlinked-configs / cycle-detection
  symlink-source preservation / negative regression locking the broken
  `.resolve()` idiom).

**Test result**: 1720 passed, 73 skipped, 0 failures (was 1713 baseline).

**Discovered by**: 8-agent prep round 2026-05-10 (Agent I FINDING 5
"Hidden Findings Hunt"); design verified by Explore agent same date.

## [Unreleased] — HYBRID Phase α-3 (2026-05-10) — symlink-safe pipeline-root detection (#PY-79)

**Fixed (#PY-79)**

- `src/lobtrainer/data/feature_set_resolver.py:442`
  `Path(anchor).resolve()` → `Path(anchor).absolute()`. Pre-α-3,
  `find_feature_sets_dir` walked up from a deref'd anchor — when
  `data/` was symlinked to an external mount (e.g.
  `/Volumes/WD_Black/HFT-data/`), the walk began under the deref
  target where no `contracts/pipeline_contract.toml` exists in any
  ancestor, so auto-detection failed with `FeatureSetResolverError`.
- `src/lobtrainer/training/trainer.py:482-485` — caller-side fix:
  `Path(cfg_data.data_dir).resolve()` + `Path(cfg_data.feature_sets_dir).resolve()`
  both flipped to `.absolute()`. Required because caller's prior
  `.resolve()` would have defeated the in-function `.absolute()` fix
  (the symlink-source would already be lost before the function ran).

**Tests**

- NEW `tests/test_feature_set_resolver.py::TestFindFeatureSetsDirSymlinkSafe`
  (3 tests covering positive walk-through-symlinked-data-dir / negative
  regression locking broken `.resolve()` idiom / edge case anchor-IS-
  the-symlink).

**Discovered by**: 7-agent prep round 2026-05-10 (Agent C1 ground-truth);
implemented in commit `c232cf3` per HYBRID Phase α design.

## [Unreleased] — Cycle 2.5b (2026-05-07) — Cross-config horizon-LIST mismatch validator

Defense-in-depth atop the existing IDX-only cross-config invariant at
`schema.py:2407-2450` (Cycle 1b.2). Closes Issue 2 of Cycle 2.5 hardening
flagged by V1-V6 + V7-V12 6-agent re-validation rounds.

**Why**: Pre-Cycle-2.5b, `ExperimentConfig._validate_all` only verified
that `model.hmhp_primary_horizon_idx == data.labels.primary_horizon_idx`
(idx-equality). Two configs with DIFFERENT `model.hmhp_horizons` and
`data.labels.horizons` lists could pass silently when both indices were
in range, producing silent gradient corruption in HMHP cascade decoders
(loss = sum_h(loss_h), so non-primary heads also matter — V12 Agent Y
finding).

**Changes**:
- `src/lobtrainer/config/schema.py:2452-2494` — new full-tuple-equality
  check inside the existing HMHP cross-config block. Skipped when either
  list is empty (defers to auto-resolve at `trainer.py:850-866` +
  `simple_trainer.py:240-260`). Inserted INSIDE the existing
  `if _mt_str in ("hmhp", "hmhp_regression"):` gate — non-HMHP models
  unaffected.
- `tests/test_hmhp_primary_horizon_idx_bridge.py` — NEW
  `TestCrossConfigHorizonListMismatch` class with 12 tests:
  identical-pass / permutation-at-resolved-idx / permutation-off-idx /
  different-lengths / completely-different / empty-labels-skip /
  empty-model-skip / both-empty-skip / tlob-skip / auto-align-then-pass /
  auto-align-then-permutation-raises / error-message-traceability.
  Plus new helper `_build_hmhp_config_with_separate_horizons` for
  independently-controlled horizons across model + labels fields.

**Test counts**: trainer 1701 → 1713 passed (+12 new), 73 skipped,
1 xfailed. Zero regressions.

**Per hft-rules**: §5 fail-fast (precise + actionable error message),
§8 never silently accept corrupt data, §0 reuse-first (cross-references
LabelsConfig dup-check at `schema.py:411-413` + HMHPConfig dup-check
at `lob-models/.../config/base.py:2093+` as third defensive layer).

**Risk profile**: HIGH-GAP, no current trigger — verified by V12
hidden-interaction auditor 2026-05-07: all 12 production HMHP fixtures
+ 4 production HMHP YAMLs use `data.labels.horizons` empty (auto-resolve
path) so the new check skips on existing workloads. Defense-in-depth
against future YAML edits / programmatic library use.

**Coordinated cycle**: shipped alongside Cycle 2.5a in lob-models
(HMHPConfig duplicate-rejection validator at `config/base.py:2093+`)
mirroring the LabelsConfig discipline at `schema.py:411-413`.

## [0.7.1] — 2026-04-27 — REV 3.1 Phase G G.6.A→G.8 (SchemaVersion 2.2 → 3.0 MAJOR)

REV 3.1 cycle Phase G consumer-side cascade. NO trainer-internal logic
changes — purely fixture + documentation cascade flowing the producer-side
SchemaVersion bump from `feature-extractor-MBO-LOB` (2.2 → 3.0 MAJOR per
CLAUDE.md root rule: any modification to stable features 0-97 = BREAKING).

**Trainer-side changes**:
- `tests/conftest.py:112` — fixture `data[:, 97] = 2.2` flipped to `0.0`
  per Phase G.1 RESERVED (in-NPY emission DROPPED; JSON metadata is the
  canonical schema_version SSoT post-Phase-G.1).
- `tests/test_phase2_signal_metadata_emit.py:30,31` — `_fixture_contract`
  defaults `contract_version="3.0"` + `schema_version="3.0"`.
- `tests/test_feature_set_resolver.py:46-65` — golden hash
  `GOLDEN_HASH_98F_5_12 = 1fc8d7f8dee2c9a07617c995ba492b2cb14cb81e1857d8b57fd7cb5f888480d2`
  (regenerated via SSoT `hft_contracts.feature_sets.hashing.compute_feature_set_hash`
  from the bumped contract_version 3.0).
- `tests/test_feature_preset_migration.py` — 4 cascade sites updating
  `expected_contract_version="3.0"`.
- `tests/test_sources_and_bundle.py:245` + `tests/test_experiment_spec_and_gates.py:249`
  — NEW `@pytest.mark.xfail(strict=False, reason=...)` markers on 2 tests
  that exercise `data/exports/e5_timebased_60s` legacy NPYs at schema 2.2
  (correctly REJECTED post-G.6.A by `validate_schema_version`). The xfail
  markers are the structural reminder that re-export of legacy data is
  gated on Phase G+1 (re-export execution + scripts/ trigger
  infrastructure); test will pass once Phase G+1 ships and operator runs
  the regen.
- 5 production-config FeatureSet content_hashes regenerated via
  `hft_contracts.feature_sets.hashing.compute_feature_set_hash` cascade
  (G.6.D ship — locked across 3 production JSONs at
  `contracts/feature_sets/`).
- README.md banner: Version 0.4.0 → 0.7.1 + Schema 2.2 → 3.0 + new v0.7.1
  history row + corrected SCHEMA_VERSION assertion to use string
  comparison `assert SCHEMA_VERSION == "3.0"` (matches `_generated.py`).
- CODEBASE.md banner: Schema 2.2 → 3.0 with Phase G G.6.A bump rationale
  inline.

**Test counts post-cascade**: trainer **1367 passed + 65 skipped + 2
xfailed** (xfailed = legacy-corpus xfail markers per Phase G+1 deferral);
broader pipeline aggregate **3,676 tests passing** across 5 modules
(Rust 800 + analyzer 358 + hft-contracts 518 + trainer 1367 + hft-ops 633).
ZERO regressions across all 5 module test suites.

**Validation provenance**: 17+ fresh-eyes adversarial agents (V1-V6
round-1 + W1-W6 round-2 + post-G.6 cumulative audit + G.6.F cascade
completeness + G.6.A→G.8 closure verification + G.9 investigation).
Standing user mandate "ULTRATHINK + parallel adversarial agents +
/effort max" upheld throughout.

**Deferred** (not in this trainer release):
- **Phase G.9** (FIND-H9 cross-language `config_hash` fold — Rust-side
  `feature-extractor-MBO-LOB` change): investigation surfaced
  implementation complexity beyond cycle scope; deferred to dedicated
  mini-cycle for fresh adversarial validation.
- **Phase G+1** (re-export execution): legacy `data/exports/*` NPYs at
  schema "2.2" remain on disk; xfail markers in `test_sources_and_bundle.py`
  + `test_experiment_spec_and_gates.py` are the structural reminders.

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
