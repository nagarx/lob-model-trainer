# merge.py Migration Plan (v1 → v2)

> **Purpose**: Step-by-step plan for retiring the v1 `merge.py` (single-string `_base:` only) and landing the v2 implementation (`_base: str | list[str]`, plus downstream base-decomposition + fingerprint-fix).
>
> **Status**: EXECUTING
>
> **Created**: 2026-04-15 (Phase 3 of training-pipeline-architecture refactor)
>
> **Parent plan**: `/Users/knight/.claude/plans/gentle-brewing-quail.md` § Phase 3 (REVISED v3)
>
> **Precedent**: `/Users/knight/code_local/HFT-pipeline-v2/feature-extractor-MBO-LOB/MONOLITH_MIGRATION_PLAN.md` — the structural model for this document.

---

## Executive Summary

v1 `merge.py` (127 LOC) supports single-string `_base: "path.yaml"` only. After three parallel validation agents rejected OmegaConf in favor of a minimal hand-rolled extension, v2 adds `_base: list[str]` multi-base composition with left-to-right merge and child-overrides-all semantics. Every v1 invariant (cycle detection, depth cap = 10, list-REPLACE, deep-merge, relative-path resolution, pop-on-read mutation, deterministic JSON ordering) is preserved bit-identically.

**Goal**: Ship v2 `merge.py` + retire v1 to `archive/merge-v1/`, then decompose the sole monolith base (`configs/bases/e5_tlob_regression.yaml`) into 17 orthogonal slices across 4 axes (models/datasets/labels/train), then migrate ~21 experiment configs in 3 batches (revised from original ~36/4 — Batch 4 baselines cancelled; see §Scope Revision). Plus CRITICAL companion fix to `hft-ops/ledger/dedup.py` (§3.3b) so the ledger fingerprint reflects the resolved effective dict rather than the pre-inheritance YAML.

**Risk**: MEDIUM. v1 has ONE production caller (`schema.py:1323`) + two test imports. The migration surface is small but touches every existing config indirectly (via the parity test across 41 golden fixtures). Failure mode: a silently-different resolved dict post-migration would cause ledger-fingerprint drift or training-loss divergence.

**Approach**: ADDITIVE. Archive first (zero code loss), then add multi-base support in-place (new merge.py exports same public API), gate every downstream change on bit-identical output.

---

## Current State (2026-04-15 ground truth)

### What v1 provides

| File | Lines | Purpose |
|---|---|---|
| `src/lobtrainer/config/merge.py` | 127 | `deep_merge()` + `resolve_inheritance()` with single-string `_base:` |
| `tests/test_config_inheritance.py` | 364 | **22 test methods** (grep-verified) covering merge semantics, inheritance, error paths |
| `configs/bases/e5_tlob_regression.yaml` | 64 | Monolith base covering all 4 axes — the ONLY pre-existing base |

### Callers of `resolve_inheritance` / `deep_merge`

Grep-verified 2026-04-15:

| Path | Line | Usage |
|---|---|---|
| `src/lobtrainer/config/schema.py` | 1323 | `from lobtrainer.config.merge import resolve_inheritance` in `ExperimentConfig.from_yaml()` |
| `tests/test_config_inheritance.py` | 13 | `from lobtrainer.config.merge import deep_merge, resolve_inheritance` |

Indirect coupling:

| Path | Line | Coupling |
|---|---|---|
| `hft-ops/src/hft_ops/stages/training.py` | 108-152 | `_absolutize_inline_base_paths` knows about the `_base` key name; handles str AND list forms (Phase 2b prep) |
| `hft-ops/src/hft_ops/ledger/dedup.py` | 117-120 | `_load_config_as_dict` raw-loads trainer YAMLs; §3.3b fix will route trainer YAMLs through `resolve_inheritance` |

---

## Archive Target

```
lob-model-trainer/src/lobtrainer/config/archive/merge-v1/
├── ARCHIVE_README.md       ← Written 2026-04-15 (monolith template)
└── merge.py                 ← Copy of v1, mtime preserved (Apr 11 00:28)
```

No `__init__.py` — archive is deliberately non-importable as a package (matches `feature-extractor-MBO-LOB/archive/monolith-v1/` precedent). Tests that need v1 for parity use `importlib.util.spec_from_file_location` (see `tests/test_merge_v1_parity.py`, Phase 3.7).

---

## Migration Phases

### Phase 3.0 — Pre-migration golden fixtures (✅ COMPLETE 2026-04-15)

Generated 39 JSON fixtures at `tests/fixtures/golden/` via `tests/fixtures/golden/generate_snapshots.py`:
- 38 OK: resolved dicts captured for every in-scope config
- 1 ERROR: `nvda_tlob_triple_barrier_11mo_v1.yaml` (pre-existing YAML parse error — ScannerError at line 124) captured with `exc_type` + `message`

Excluded (per plan §3.6):
- 2 XGBoost configs (different schema, bypass `ExperimentConfig.from_yaml`)
- `configs/archive/*.yaml` (6 legacy files, out of Phase 3 scope)

### Phase 3.1 — Archive v1 (✅ IN PROGRESS 2026-04-15)

```
cp -p src/lobtrainer/config/merge.py \
     src/lobtrainer/config/archive/merge-v1/merge.py
# Write ARCHIVE_README.md
# Write MERGE_MIGRATION_PLAN.md (this file)
```

Original `merge.py` **stays in place** during 3.3 — no broken state between commits. v2 rewrites it atomically.

### Phase 3.2 — Documentation ledger (✅ THIS FILE)

### Phase 3.3 — Extend merge.py with multi-base

~20 LOC delta to `src/lobtrainer/config/merge.py`. New:
- Validates `_base: str | list[str]` exhaustively (see §3.3 validity table in plan)
- Multi-base merges left-to-right; child overrides all
- Detects `_partial: true` sentinel on top-level base dicts and raises a descriptive error if loaded via `ExperimentConfig.from_yaml()` (prevents researchers from accidentally loading a partial base standalone)

Tests added (to `tests/test_config_inheritance.py`, ~10 new tests) + new test files:
- `tests/test_merge_v1_parity.py` — loads archive merge.py via `importlib.util.spec_from_file_location`, diffs against new merge.py on every fixture
- `tests/test_multi_base_inheritance.py` — diamond resolution + 4-base composition for E5 equivalent
- `tests/test_base_axis_ownership.py` — mechanical §3.4 ownership rule enforcement

### Phase 3.3b — Ledger fingerprint fix (CRITICAL)

Change to `hft-ops/src/hft_ops/ledger/dedup.py`:
- `compute_fingerprint` calls `resolve_inheritance` (lazily imported from lobtrainer) when loading trainer YAMLs
- Inline `trainer_config` dicts with `_base:` also resolve before fingerprinting
- Regression guard: `hft-ops/tests/test_fingerprint_base_mutation.py` asserts base mutation changes dependent-experiment fingerprint

### Phase 3.4 — Decompose monolith base

Create 17 orthogonal bases:
- `bases/models/` (7): tlob_compact_regression, tlob_paper_classification, hmhp_cascade_{classification, regression}, hmhp_triple_barrier, logistic_baseline, temporal_ridge
- `bases/datasets/` (5): nvda_xnas_128feat, nvda_xnas_98feat, nvda_e5_60s, nvda_e5_30s, nvda_40feat_short_term
- `bases/labels/` (3): tlob_smoothed, regression_huber, triple_barrier_volscaled
- `bases/train/` (2): regression_default, classification_default

Ownership table enforced by `test_base_axis_ownership.py`. No field straddles axes.

### Phase 3.5 — Progressive migration (4 batches)

Each batch is a set of per-config commits. Per-config gate: golden-file parity. Per-batch gate: CPU-deterministic 2-epoch smoke trainer run on a named canonical config (±1.0% val_loss tolerance).

| Batch | Configs | Canonical smoke config | Status |
|---|---|---|---|
| **1a** | **E5×5 + E6** (6) | `e5_60s_huber_cvml.yaml` | ✅ **COMPLETE 2026-04-15** — all 6 byte-identical parity |
| **1b** | **E4 TLOB** (1) | `e4_tlob_h60.yaml` | ✅ **COMPLETE 2026-04-15** — byte-identical via `tlob_compact_bare` base |
| **2** | **HMHP** (11) | `nvda_hmhp_128feat_h10_primary.yaml` | ✅ **COMPLETE 2026-04-15** — all 11 byte-identical. 10 new bases added. Bases target (17) reached. |
| 3 | TLOB classif (7) | `nvda_tlob_h10_v1.yaml` | ⏳ pending |
| ~~4~~ | ~~Baselines (7)~~ | — | ❌ **CANCELLED** — see Scope Revision below |

#### Batch 1a Completion Record (2026-04-15)

Configs migrated (all byte-identical parity verified):

1. `e5_60s_huber_cvml.yaml` — single-base `_base: "../bases/e5_tlob_regression.yaml"` → 4-base list
2. `e5_60s_huber_nocvml.yaml` — single-base → 4-base list
3. `e5_60s_gmadl_cvml.yaml` — single-base → 4-base list (+ GMADL loss overrides)
4. `e5_30s_huber_cvml.yaml` — single-base → 4-base list (+ 30s dataset, delta=15.1 override)
5. `e5_30s_huber_nocvml.yaml` — single-base → 4-base list (+ 30s dataset, delta=15.1 override)
6. `e6_calibrated_conviction.yaml` — **standalone 77 LOC → 4-base list 26 LOC** (-66% LOC; the clearest migration win)

Bases added during Batch 1a:
- `bases/datasets/nvda_e5_30s.yaml` (bringing total bases to 5 of 17)

Tests added to support migration:
- `test_merge_v1_parity.py` updated with `_is_v1_compatible_yaml()` helper — auto-skip v1-archive tests for migrated configs (which use list-form `_base:` that v1 cannot parse). `test_v2_matches_golden_fixture` remains the universal correctness gate.

#### Batch 1b Completion Record (2026-04-15) — `tlob_compact_bare` chained inheritance

**Problem encountered**: E4 TLOB's standalone YAML omitted `model.tlob_use_cvml` and `model.tlob_cvml_out_channels` (letting dataclass defaults apply). Using `bases/models/tlob_compact_regression.yaml` (which inherited these from the pre-Phase-3 monolith) would ADD these fields to the resolved dict — changing the fingerprint despite zero behavioral change.

**Solution applied**: Chained inheritance via a new `bases/models/tlob_compact_bare.yaml` (without cvml defaults). `bases/models/tlob_compact_regression.yaml` was refactored to use `_base: "tlob_compact_bare.yaml"` + adds cvml default fields on top. E5/E6 consumers are unchanged (still resolve to identical dicts — verified by re-running all parity tests: **117 passed, 12 skipped**). E4 TLOB uses `tlob_compact_bare.yaml` directly — byte-identical golden-fixture match achieved.

**Result**: +1 base (6 of 17 total now — chain: `tlob_compact_bare` → `tlob_compact_regression`), +1 dataset base (`nvda_e4_5s.yaml`).

#### Batch 2 Completion Record (2026-04-15) — HMHP family

11 configs migrated byte-identically (golden-fixture parity verified):

1. `nvda_hmhp_triple_barrier_v1.yaml` — canonical TB, 223 LOC → 54 LOC multi-base
2. `nvda_hmhp_triple_barrier_volscaled.yaml` — variant TB with volatility-scaled barriers
3. `nvda_hmhp_triple_barrier_calibrated.yaml` — variant TB with per-horizon calibration
4. `nvda_hmhp_128feat_h10_primary.yaml` — canonical 128-feat classification
5. `nvda_hmhp_128feat_arcx_h10.yaml` — ARCX data variant
6. `nvda_hmhp_128feat_opportunity_h10.yaml` — uses new `labels/opportunity.yaml`
7. `nvda_hmhp_40feat_h10.yaml` — 40-feat preset H10 primary
8. `nvda_hmhp_40feat_h60_profit8bps.yaml` — H60 primary, num_workers override
9. `nvda_hmhp_40feat_h60_profit8bps_regression.yaml` — rare `hmhp_use_regression: true` child override
10. `nvda_hmhp_regressor_h60.yaml` — first `hmhp_regression` enum via chained base
11. `nvda_hmhp_regression_h10_primary.yaml` — edge case (no preset, input_size=128): 3-base composition + inline dataset

Bases added: 10 new → total 17 of 17 (plan §3.4 target reached)
- `models/hmhp_cascade_bare.yaml` + `models/hmhp_cascade_regression.yaml` (chained)
- `datasets/nvda_xnas_128feat_full.yaml`, `datasets/nvda_40feat_short_term.yaml`, `datasets/nvda_98feat_triple_barrier.yaml`
- `labels/tlob_smoothed.yaml`, `labels/opportunity.yaml`, `labels/triple_barrier_volscaled.yaml`
- `train/classification_default.yaml`, `train/classification_triple_barrier.yaml`

Ownership refinement (transparent to Batch 1):
- `train.loss_type` moved from `models/` to `labels/` (task-coupled, not model-coupled).
- `models/tlob_compact_bare.yaml` no longer sets `train.loss_type: huber` — moved to `labels/regression_huber.yaml`.
- All Batch 1 parity tests still pass (resolved dicts unchanged; only resolution path changed).

Tests added:
- `test_all_hmhp_family_migrated` + `test_niche_hmhp_configs_excluded_as_designed` in `tests/test_migrated_configs_e2e.py`
- `test_hmhp_cascade_bare_model_type_is_hmhp`, `test_hmhp_cascade_bare_excludes_regression_fields`, `test_hmhp_cascade_regression_adds_regression_fields_only`, `test_triple_barrier_label_omits_horizon_idx` in `tests/test_base_axis_ownership.py`

Trainer test totals: 208 pass + 36 skip (up from 192 + 14 after Batch 1 hardening).

#### Batch 3 Completion Record (2026-04-15) — TLOB paper-classification family + monolith retirement

**7 configs migrated byte-identically** + **monolith retired** in the same commit set (per validated user decision after 2-agent brainstorm: Migrate all 7 + Retire monolith first).

Configs migrated:

1. `nvda_tlob_h10_v1.yaml` — canonical H10, 4-base clean composition (no per-child overrides)
2. `nvda_tlob_h100_v1.yaml` — H100 sibling of h10_v1, overrides `horizon_idx: 3`
3. `nvda_tlob_h50_v1.yaml` — H50, normalization=none, dataset inline (3-base + inline)
4. `nvda_tlob_raw_h50_v1.yaml` — raw-export variant, dataset inline, horizon_idx=2
5. `nvda_tlob_98feat_h100.yaml` — hybrid norm, exclude=[], dataset inline, dropout=0.0 override
6. `nvda_tlob_repo_match_h50.yaml` — official TLOB-repo reproduction, uses new `nvda_40feat_tlob_repo` base, dropout=0.0
7. `nvda_tlob_v2_h100.yaml` — V2 dynamic-threshold, `nvda_40feat_tlob_repo` base, dropout=0.0

Bases added: 4 new → total **21 axis-partitioned bases** (exceeded plan target of 17 due to appropriate granularity in dataset axis):
- `models/tlob_paper_classification.yaml` — paper-spec TLOB (hidden=64, layers=4, heads=1) — distinct from `tlob_compact_bare` regression variant
- `datasets/nvda_98feat_zscore_per_day.yaml` — 2-config shared base (h10_v1 + h100_v1)
- `datasets/nvda_40feat_tlob_repo.yaml` — 2-config shared base (repo_match_h50 + v2_h100) with `normalization=global_zscore, clip_value=null`
- `train/tlob_paper_classification_train.yaml` — distinct training regime (batch=64, lr=1e-4, epochs=50 vs HMHP classif's 256/5e-4/30)

**Dropout split handled without chained inheritance**: 4 configs use base default `dropout: 0.1`, 3 TLOB-repo-match variants (v2_h100, 98feat_h100, repo_match_h50) override to `dropout: 0.0` via child. Cleaner than another chain level.

**Monolith `bases/e5_tlob_regression.yaml` RETIRED** (plan §3.9 success criterion #10 ✅):
- File + golden fixture deleted
- `test_e5_base_loads_standalone` removed (loaded monolith directly — no longer relevant)
- `test_e5_cvml_inherits_base` renamed → `test_e5_cvml_inherits_via_multibase`, docstring updated
- `test_exactly_one_pre_existing_base` → `test_zero_pre_existing_bases_after_monolith_retirement`
- `test_monolith_excluded_from_this_check` → `test_monolith_retired` (asserts NON-existence going forward)

**P0 documentation bug FIXED**: `configs/bases/README.md:46` ownership table now correctly shows `train.loss_type` under `labels/` (task-coupled) rather than `models/`. Also added `train.focal_gamma` to train/ ownership explicitly.

Hardening tests added:
- `test_all_tlob_classif_family_migrated` in `test_migrated_configs_e2e.py`
- `test_tlob_paper_classification_excludes_cvml_fields` + `test_tlob_paper_classification_excludes_regression_fields` in `test_base_axis_ownership.py`

Trainer test totals: **211 pass + 50 skip** (up from 208+36 after Batch 2). hft-ops unchanged: 160 pass.

**Phase 3.5 migration status**: 25 configs migrated (7 E-family + 11 HMHP + 7 TLOB classif). 17 configs standalone by design (7 baselines + 2 XGBoost + 6 archive + 2 niche HMHP). Total Phase 3.5 coverage: 25/42 in-scope configs migrated (59%).

#### Batch 4 Cancellation — Scope Revision (2026-04-15)

The original plan included **Batch 4 — Baselines** (logistic×4, temporal_ridge×2, temporal_gradboost×1 = 7 configs). **CANCELLED** after analyzing their actual structure:

- `e4_temporal_ridge_h60.yaml` (53 LOC) and `nvda_temporal_ridge_h10.yaml` (55 LOC) are already small standalone configs with ridge-unique fields (`model.alpha`, `model.features.signal_indices`, `features.rolling_windows`, etc.) that don't share structure with other model families.
- Their resolved dicts are SPARSE (don't include TLOB-specific fields, training defaults, etc.).
- Migrating them to multi-base would either (a) require hyper-minimal bases specific to each baseline, or (b) introduce many unwanted default fields, breaking byte-identity.

**Decision**: Keep baseline configs standalone. Multi-base composition is most valuable when multiple configs share a LARGE common structure (E5 family: 60+ shared lines; TLOB classification variants: 120+ shared lines). Baselines already have the minimum-viable-config pattern; composing them via bases adds no value and potentially reduces clarity.

**Revised migration scope**: **25 configs** (final actual; estimated ~28 at time of scope revision, was originally planned as ~36) across Batches 1-3. Baselines (7 configs) + XGBoost (2) + archive (6) + 2 niche HMHP = **17 configs stay standalone by design**. 25 migrated + 17 standalone = 42 in-scope total. This is an improvement per "long-term benefit": simpler configs stay simple, shared-structure families benefit from composition.

### Monolith Retirement Policy

The pre-existing `bases/e5_tlob_regression.yaml` is no longer referenced by any active child config (all 5 E5 configs migrated to multi-base form). HOWEVER, it is kept until Batch 1 is FULLY complete (including 1b) as a safety net. At end of true Batch 1 completion:

1. Delete `bases/e5_tlob_regression.yaml`
2. Delete `tests/fixtures/golden/bases/e5_tlob_regression.json`
3. Remove or update `test_e5_base_loads_standalone` and `test_e5_cvml_inherits_base` in `test_config_inheritance.py` (currently they reference the monolith directly)
4. Update `test_merge_v1_parity.py::TestFixtureCompleteness::test_exactly_one_pre_existing_base` → `test_zero_pre_existing_bases` (or remove)

---

## Validation Protocol

At the end of Phase 3, every item in the `§3.9 Verification Gate` of the parent plan must pass. Key checkpoints:

1. ✅ Pre-migration fixtures generated (§3.0 done)
2. [ ] Archive structure mirrors `feature-extractor-MBO-LOB/archive/monolith-v1/` (no `__init__.py`, ARCHIVE_README, in-file pointer in new merge.py)
3. [ ] `test_merge_v1_parity.py` green: all 39 fixtures produce byte-identical output under v2 merge.py
4. [ ] All 22 existing `test_config_inheritance.py` tests pass unchanged
5. [ ] New tests green (~35 tests added)
6. [ ] `test_base_axis_ownership.py` passes (mechanical §3.4 enforcement)
7. [ ] `test_fingerprint_base_mutation.py` passes (§3.3b regression guard)
8. [ ] `test_inline_multibase_e2e.py` passes (hft-ops end-to-end integration)
9. [ ] All trainer tests still pass (baseline count to be captured at migration time)
10. [ ] All 158 hft-ops tests still pass (post-§3.3b baseline; Batch 1 added +3 fingerprint regression tests bringing to 160; Phase 3.5 hardening added +2 content-address invariants bringing to 162)
11. [ ] Per-batch smoke trainer runs within ±1.0% val_loss threshold

---

## Risk Mitigation

| Risk | Mitigation |
|---|---|
| v2 silently produces different resolved dict | `test_merge_v1_parity.py` catches via byte-diff on 39 fixtures |
| Ledger fingerprint drift on base mutation | §3.3b fix + `test_fingerprint_base_mutation.py` regression guard |
| Field straddles multiple base axes | `test_base_axis_ownership.py` mechanical enforcement |
| Partial base accidentally loaded standalone | `_partial: true` sentinel + descriptive error in `ExperimentConfig.from_yaml()` |
| hft-ops inline trainer_config with multi-base breaks | `test_inline_multibase_e2e.py` covers end-to-end path |
| Smoke run val_loss variance from CUDA non-determinism | CPU-deterministic mode + fixed fixture + ±1.0% tolerance |
| Per-batch failure cascades | Per-config commits; atomic rollback of individual migrations |
| Trainer tests unaccounted for | Run full suite at each migration step; revert on any failure |

---

## Success Criteria

Phase 3 is DONE when:

1. `merge.py` v2 at `src/lobtrainer/config/merge.py` with multi-base support
2. v1 at `src/lobtrainer/config/archive/merge-v1/merge.py` + ARCHIVE_README
3. 17 orthogonal bases under `configs/bases/{models,datasets,labels,train}/`
4. `configs/bases/e5_tlob_regression.yaml` DELETED (monolith retired)
5. ~21 experiment configs migrated to multi-base form (revised scope: 7 done in Batch 1 + 14 HMHP pending in Batch 2 + 7 TLOB classif pending in Batch 3; 7 baselines CANCELLED per §Scope Revision above)
6. 35+ new tests added, all green
7. All existing tests still green (trainer + hft-ops)
8. `hft-ops/ledger/dedup.py` updated for fingerprint-via-resolution
9. Documentation: `CODEBASE.md`, `README_configs.md`, `bases/README.md`, `CHANGELOG.md` (new), `PIPELINE_ARCHITECTURE.md` all updated
10. Memory entry `project_training_pipeline_phase3.md` written
