# Consumer Fix Plan — lob-model-trainer (2026-05-04)

**Created**: post Phase O Cycle 1 close + producer-readiness investigation
**Read first** when starting a new session focused on this repo
**Cross-references**: monorepo-root `NEW_SESSION_HANDOFF_2026_05_04.md` (since archived → root `.archive/2026-05-08-pre-cycle-4-hygiene/`) + `PHASE_P_BACKLOG.md` + sibling `lob-models/CONSUMER_FIX_PLAN_2026_05_04.md`

---

## 1. Context

**Producer state (READY)**: Phase O Cycle 1 SHIPPED 2026-05-04 to all 3 producer remotes (feature-extractor `c62a1c0`, reconstructor `28a9a22`/`v0.2.1`, hft-contracts `311bdbf`). All CI green. Bit-exact preserved (`GOLDEN_HASH_SEQUENCES_NPY=0x8bebad9b09b564cd`, `GOLDEN_HASH_LABELS_NPY=0x5dcf907068fcadcc`).

**4 NEW v3p0 exports consumable now** (all at `schema_version=3.0`, `contract_version=3.0`, idx 97=0.0 RESERVED, lob.book_clears=1/day):
- `data/exports/e5_timebased_60s_v3p0/` — 98 feat, 60s bins, 230 days (3 fail-loud), 136K seq, 656MB
- `data/exports/e5_timebased_30s_v3p0/` — 98 feat, 30s bins, 233 days, 278K seq, 2.0GB
- `data/exports/e4_timebased_5s_v3p0/` — 98 feat, 5s bins, 233 days, 1.58M seq, 14GB
- `data/exports/nvda_xnas_128feat_regression_fwd_prices_v3p0/` — 128 feat, 35 days, 51K seq, 2.5GB

**Pre-Phase-O exports preserved as historical archive** at original paths (`data/exports/e5_timebased_60s/` etc.) — `schema_version="2.2"`, will fail strict validators per `hft-contracts/.../validation.py:115-120`.

**This repo's state**: HEAD on Phase A.5 Scope D v2 (Pydantic v2 migration COMPLETE), v0.7.0; 1434 tests collected. Trainer was last RUN end-to-end **2026-03-19** — 46+ days of compounded code drift between consumer and producer untested live.

**Critical asymmetry**: Producer is now strict and consumer is permissive — this is the inversion of what we want. The consumer has known warn-and-skip silent-fallback gates around `schema_version` checks that allow the +21% data and clean v3.0 baseline to coexist with stale 2.2 metadata silently if both directories ever get mixed.

---

## 2. Critical action items (CRITICAL — block training on v3p0 baseline)

| ID | File:Line | Current behavior | Required behavior | Test | Effort |
|---|---|---|---|---|---|
| **C-1** | `src/lobtrainer/training/simple_trainer.py:50-74` | `_load_split` directly `np.load`s sequences + `_regression_labels.npy`. ZERO call to `validate_export_contract`, ZERO `schema_version` check, ZERO `contract_version` check. TemporalRidge + TemporalGradBoost run BLIND on whatever v2.2/v3.0 mix is on disk. | Wire `_validate_day_metadata(metadata, day)` (already at `dataset.py:60-87`) into `_load_split` per-day OR (better) replace ad-hoc loader with `load_day_data(...)` from `dataset.py:497-733` which already validates. Validation must run BEFORE first numpy load (fail-fast). | New `tests/test_simple_trainer_validation.py`: synthetic dir with mixed schema_version → assert `ContractError` raised before training. | **2-3 hr** including refactor + test |
| **C-2** | `src/lobtrainer/data/dataset.py:60-87` (`_validate_day_metadata`) | When `metadata is None`: silent return (line 74-75). When `schema_version` key missing: log warning + return (line 77-83). Skips actual `validate_export_contract` call entirely on missing-key path. **The Phase O v3.0 baseline does emit schema_version, so this gate WILL fire correctly on v3p0** — but if any one day in a mixed corpus loses metadata.json (partial export, NFS lag), corruption proceeds silently. | Promote both branches to fail-loud `raise ContractError(...)`. The "warn + continue" semantics were a bridge for legacy 1.0 exports — now archived. Phase N pivot decision rejected escape hatches because they break `compatibility_fingerprint` traceability. | New `tests/test_dataset_strict_validation.py`: missing metadata → `ContractError`; missing schema_version → `ContractError`. Update `test_sources_and_bundle.py:245-255` xfail (Phase N pivot deferred this — now ready). | **1.5 hr** |
| **C-3** | `src/lobtrainer/data/dataset.py:859-873` (`load_split_data` calling `_validate_day_metadata`) | Day-1-only schema validation (per Forensic Audit D6 finding, status DORMANT). Validates first day's metadata + skips remaining 232 days. Combined with C-2's silent-fallback, an export clean on day 1 + corrupted on day 50 silently trains. | Validate **every** day's metadata, not just day 1. Cost is negligible (~233 dict-ops). | Extend `test_dataset_strict_validation.py` with synthetic 5-day dir where day 3 is corrupted → `ContractError` mentions day 3. | **45 min** |
| **C-4** (cross-cutting → also affects backtester) | `lob-backtester/src/lobbacktest/data/loader.py:214` and `:293` | Both sites: `if metadata.get("schema_version"):` — TRUTHINESS check. If `schema_version=""` or `0` or missing → silently skips validation entirely. | Change `metadata.get("schema_version")` truthiness to `"schema_version" in metadata` (presence check), then call validator unconditionally. Producer ALWAYS emits this field at v3.0; absence is itself a contract violation. | New backtester `tests/test_loader_strict_validation.py`. | **45 min** (separate repo — see Cross-cutting §8) |

---

## 3. High-priority action items (HIGH — silent-corruption surface)

| ID | File:Line | Current behavior | Required behavior | Effort |
|---|---|---|---|---|
| **H-1** | `src/lobtrainer/training/simple_trainer.py:207-218` and `src/lobtrainer/export/metadata.py:13-149` (`build_signal_metadata`) | Both producer-of-signal sites OMIT `schema_version` and `contract_version` from `signal_metadata.json`. Only Phase II `compatibility` block added (and only when explicit). Backtester downstream (Phase N D2) cannot fail-loud on missing schema_version because every signal directory ever produced lacks it. **This is BLOCKING-1 from PHASE_N_VALIDATION_FINDINGS §3.** | Inject `"schema_version": str(hft_contracts.SCHEMA_VERSION), "contract_version": str(hft_contracts.SCHEMA_VERSION)` at top of `build_signal_metadata` AND at `simple_trainer.py:207`. Both sources of signal_metadata.json must converge on same field set. | **1.5 hr** (new tests + 2-site producer-side patch) |
| **H-2** | `src/lobtrainer/data/normalization.py:72-106` (`_welford_update_batch` + `_welford_init_scalar` + `_welford_finalize`) | Trainer hand-rolls Chan's parallel-merge Welford algorithm (cites "Chan, Golub, LeVeque 1979"). `hft-metrics/src/hft_metrics/welford.py:62-106` defines `RunningStats.update_batch` citing same paper. **Mean-merge formulas algebraically identical but NOT bit-equivalent under IEEE-754** (POST_PHASE_M_VALIDATION CONFIRMED-3). Two divergent code paths → if either is bug-fixed in isolation, exported normalization stats diverge. | Replace `_welford_*` helpers with `RunningStats` from `hft-metrics`. Already a runtime dep (`pyproject.toml:28`). Add precision-equivalence regression test. | **3 hr** (refactor + golden test + verify HybridNormalizer + GlobalZScoreNormalizer paths still produce identical exports) |
| **H-3** | `src/lobtrainer/training/trainer.py:807-810` + `:1007-1033` (N2 from Forensic Audit) | `for epoch in range(cfg.epochs)` always starts at 0; `load_checkpoint()` at `:1026` reads `checkpoint.get('epoch', 0)` into `self.state.current_epoch` but the for loop ignores it. `--resume` re-runs N redundant epochs. Status: CRITICAL ACTIVE. | `for epoch in range(self.state.current_epoch, cfg.epochs):` with explicit log of resume offset. New `tests/test_resume_epoch_alignment.py`. | **2 hr** including determinism re-verification |
| **H-4** | `src/lobtrainer/training/trainer.py:1007-1033` (N3 from Forensic Audit) | `load_checkpoint` does NOT restore `EarlyStopping`/`ModelCheckpoint`/`MetricLogger` callback state. Resume re-arms early-stopping from scratch, re-counts plateau-patience, re-overwrites best.pt. Status: CRITICAL ACTIVE. | Serialize callback `.state_dict()` into checkpoint dict; restore on load. | **3 hr** |
| **H-5** | (lob-models cross-cutting; see lob-models/CONSUMER_FIX_PLAN §M-1) | HMHP vs HMHP-R encoder pooling inconsistency (cosine 0.18 between paired outputs). | Add cross-model parity test in `test_hmhp_regression_training.py` that asserts encoder pool reproducibility within tolerance after lob-models fix lands. | **2 hr trainer-side** (after lob-models fix) |
| **H-6** | `src/lobtrainer/training/strategies/hmhp_regression.py:157,254` (N4 from Forensic Audit) — VERIFIED ground-truth location 2026-05-04 (NOT trainer.py as some prior docs claimed) | HMHP-R primary metrics use `horizons[0]` not `primary_horizon_idx` (`primary_horizon = horizons[0]` literal at line 157). Silent metric drift between training + ledger + backtester views of "primary horizon". Status: CRITICAL ACTIVE. | Use `cfg_data.labels.primary_horizon_idx` consistently across HMHP-R metric pathway. Update `test_hmhp_regression_metrics.py` to assert `primary_horizon_idx=2` produces metrics from `horizons[2]`. | **1.5 hr** |
| **H-7** | `src/lobtrainer/calibration/variance.py` + `src/lobtrainer/export/exporter.py` (N6) | Calibrated metrics in signal_metadata.json describe RAW predictions. Calibration applied AFTER metric computation. Backtester reads `metrics.r2` thinking it's calibrated, but it's raw. Status: CRITICAL ACTIVE. | Either (a) recompute metrics on calibrated predictions before stamping into signal_metadata.json, OR (b) tag the metrics block with `"computed_on": "raw"` discriminator. Update `test_signal_export_inference.py` to assert metric provenance. | **2 hr** |

---

## 4. Medium-priority action items (MEDIUM — discipline + observability)

| ID | File:Line | Current behavior | Required behavior | Effort |
|---|---|---|---|---|
| **M-1** | `src/lobtrainer/training/trainer.py:339-560` (`_create_dataloaders`) | C-10 from POST_PHASE_M_VALIDATION: god method, 221 LoC, 8 cross-cutting concerns (feature selection, horizon resolution, transform composition, label mapping, num_workers, mmap, FeatureSet resolution, lazy loading). | Extract subhelpers: `_resolve_features()`, `_resolve_horizons()`, `_build_transforms()`, `_build_label_pipeline()`, leave `_create_dataloaders` as orchestrator. | **6-8 hr** |
| **M-2** | `src/lobtrainer/training/trainer.py:154-173` and `dataset.py:1547-1557` | F2/F9 + REFUTED-2 site 2 from POST_PHASE_M_VALIDATION: `except ValueError:` is too broad — would silently swallow HMHP config errors. | Narrow to sentinel `LegacyFormatFallback` exception class. | **1 hr** |
| **M-3** | `configs/bases/datasets/nvda_e5_60s.yaml:11`, `nvda_e5_30s.yaml:16`, `nvda_e4_5s.yaml:11` | All 3 base configs hardcode `data_dir: "../data/exports/e5_timebased_60s"` etc. — pointing at the LEGACY v2.2 archive, NOT the v3p0 baseline. | (See §6 v3p0 baseline migration plan below) | **30 min per config** |
| **M-4** | `configs/bases/datasets/nvda_xnas_128feat_full.yaml` | No `data_dir` set (relies on child config). 128-feat flow needs new base pointing at `nvda_xnas_128feat_regression_fwd_prices_v3p0`. | New file `nvda_xnas_128feat_v3p0.yaml`. | **30 min** |
| **M-5** | `src/lobtrainer/data/dataset.py` (load_split_data lazy/mmap path) | The 230-day `e5_60s_v3p0` train manifest. Phase P task #319: 3-day fail-loud (233-230=3). Consumer needs to know: does the trainer's "expected day count" assertion still pass? Likely NOT, since `e5_60s` (legacy) had 233 days of train+val+test. v3p0 e5_60s_v3p0 train has 230 because of fail-loud days. | Add tolerance margin to any "day count must equal X" asserts; document the 3-day fail-loud as expected. Cross-reference Phase O EXPORT_INDEX.md. | **1 hr** |
| **M-6** | `src/lobtrainer/data/dataset.py:1530-1534` (REFUTED-2 site 1) | `FileNotFoundError` on missing split: HAS `logger.info` — legitimate factory pattern. NOT silent. Cleared by Forensic Audit. | No action — keep as-is. Document audit-cleared status in CODEBASE.md. | **15 min docs** |
| **M-7** | `src/lobtrainer/data/sample_weights.py` (D4 from Forensic Audit) | Sample weights computed pre-trim → mean drifts after multi-source align. Status: DORMANT (only fires with multi-source). | Apply trim FIRST then compute weights to preserve mean=1.0 invariant. | **2 hr** |
| **M-8** | `src/lobtrainer/data/normalization.py` (D5 from Forensic Audit) | `HybridNormalizer.DEFAULT_EXCLUDE_INDICES` missing index 95 (`DT_SECONDS`). Status: DORMANT (configs override). | Add 95 to default. | **15 min** |

---

## 5. Low-priority items (LOW — cosmetic/docs)

- **L-1**: `CODEBASE.md:4` — version banner says `0.7.0` (Phase A.5 Scope D v2) but no Phase O context. Add Phase O cycle pointer once consumer-side fixes ship.
- **L-2**: `src/lobtrainer/data/dataset.py:4` — docstring says "Rust pipeline Schema v2.2" — STALE. Update to v3.0.
- **L-3**: `configs/README_configs.md:270, 367` — references `nvda_11month_complete` paths (legacy). Update to v3p0 paths or document deprecation.
- **L-4**: `src/lobtrainer/data/dataset.py:859-873` — comment "Day-1-only" should be removed once C-3 lands.
- **L-5**: F2 from POST_PHASE_M_VALIDATION: `train.py:_dump_test_metrics` `except ValueError` only — narrow OR explicitly document the catch shape.

---

## 6. v3p0 baseline migration plan (concrete steps)

### Step 6.1 — Update 4 base dataset YAMLs (atomic, single commit)

| File | Old `data_dir` | New `data_dir` | Notes |
|---|---|---|---|
| `configs/bases/datasets/nvda_e5_60s.yaml:11` | `"../data/exports/e5_timebased_60s"` | `"../data/exports/e5_timebased_60s_v3p0"` | 60s base, all e5_60s descendants benefit |
| `configs/bases/datasets/nvda_e5_30s.yaml:16` | `"../data/exports/e5_timebased_30s"` | `"../data/exports/e5_timebased_30s_v3p0"` | 30s family |
| `configs/bases/datasets/nvda_e4_5s.yaml:11` | `"../data/exports/e4_timebased_5s"` | `"../data/exports/e4_timebased_5s_v3p0"` | 5s family |
| New file: `configs/bases/datasets/nvda_xnas_128feat_regression_fwd_prices_v3p0.yaml` | (n/a) | `"../data/exports/nvda_xnas_128feat_regression_fwd_prices_v3p0"` | Mirror `nvda_xnas_128feat_full.yaml` structure with feature_count=128, FeatureSet `nvda_analysis_ready_119_src128_v1` |

### Step 6.2 — Per-experiment config audit

Run grep: `grep -rn "data_dir.*data/exports" configs/experiments/`. There are **22 standalone experiment configs** (per CODEBASE.md §2.190-204) that bypass `_base:` inheritance and hardcode `data_dir`. Each needs a per-experiment decision: (a) migrate to v3p0, (b) preserve as-is for legacy reproducibility, (c) document deprecation. Recommend: keep all standalone configs frozen; only migrate the 25 multi-base experiments + add explicit v3p0 variants.

### Step 6.3 — Retire dead xfail

`tests/test_sources_and_bundle.py:245-255` — `@pytest.mark.xfail` annotation cites "Phase G+1" as gate. Phase O Cycle 1 (= Phase G+1+L+M+N pivot) has now shipped. The test should now PASS. Remove xfail; if test fails, the v3p0 schema_version=3.0 export ISN'T being recognized — investigate.

### Step 6.4 — Document the legacy/v3p0 split

Add `configs/bases/datasets/MIGRATION_v3p0.md` or extend `CHANGELOG.md` with a v0.7.1 row enumerating: (a) the 4 path swaps; (b) the 3-day fail-loud expected delta in `e5_60s_v3p0`; (c) the +21% data per Phase O B.2 fix; (d) the legacy archive sunset policy.

### Step 6.5 — Decide legacy validator policy

Per the producer-readiness session decision: **STAY with no legacy validator.** Rationale: (a) `hft-contracts.validation.py:115-120` already raises `ContractError` on schema_version mismatch — clean fail-loud without adding code path; (b) Phase N Refinement-4 explicitly REJECTED env-var escape hatches because they break compatibility_fingerprint traceability; (c) the legacy 5,547 days at v2.2 are preserved at original paths for read-only re-extraction reference and can be loaded with a one-shot script if ever needed.

**However**, audit `configs/archive/` (6 legacy reference configs) — they will now FAIL strict validation if anyone runs them. Mark each as `# ARCHIVED — requires schema 2.2 corpus` in YAML comment.

### Step 6.6 — v3p0-specific observability flags

Per Phase P task #318 (manifest split.X.sequences ~2x inflation): if any consumer-side validator checks `manifest["split"]["train"]["sequences"]` numerically, it will see inflated counts. Recommendation: add a one-liner note in `dataset.py` near `load_split_data` that inflated manifest sequence counts are expected post-Phase-O and should not be used for any "actual sample count" check; use `np.load` shape directly.

---

## 7. Cross-references to producer docs

| Producer change | Producer doc | Consumer impact |
|---|---|---|
| `PRODUCER_DIAGNOSTICS_SCHEMA_VERSION` 2.8.0 → 2.9.0 (B-3 lob.book_clears canonical name) | producer CHANGELOGs | Trainer doesn't consume diagnostics directly; safe. |
| `SchemaVersion` UNCHANGED at 3.0 | producer CHANGELOGs + this doc | **Critical for v3p0 baseline**: producers + consumers agree on 3.0; no consumer changes needed at the schema-level. |
| `dataset_manifest.json` adds `skipped_days` top-level key | producer CHANGELOGs | Consumer-side: optional. If trainer ever reads manifest directly (it doesn't currently), enumerate the new field. |
| Action::Clear exempt from is_system_message filter (B.2a + B.2b) | producer CHANGELOGs + EXPORT_INDEX.md | This is the +21% data delta; consumer-side changes none required, but baseline-comparison scripts that compare v2.2 vs v3.0 metrics will see different numbers — flag with banner in `EXPERIMENT_INDEX.md`. |
| F-9 EXPORT_INDEX banner + POST-DISCIPLINE section | `feature-extractor-MBO-LOB/EXPORT_INDEX.md` | Operator guidance: re-extract or identify Clear-days for any pre-Phase-O analysis. Cite this banner in any trainer doc referencing legacy exports. |

---

## 8. Cross-cutting concerns (require coordinated multi-repo work)

### XC-1: Cross-language LabelFactory parity (Phase N B1-B3, deferred)

- **Where**: `feature-extractor-MBO-LOB/.../regression.rs:52,127` (Rust) vs `hft-contracts/.../label_factory.py:238,273,308,351` (Python).
- **Trainer impact**: When trainer's `dataset.py:_compute_labels_from_forward_prices` (line 373-494) calls `LabelFactory.multi_horizon`, it computes labels **in Python**. The Rust producer ALSO emits `_regression_labels.npy` from its own LabelFactory. **There is currently no fixture round-trip test that asserts both produce identical bytes for the same forward_prices input.**
- **Risk**: If a researcher uses `labels_config.source="forward_prices"` (consumer-side compute), labels diverge from the Rust-emitted file. The choice is documented but the divergence is not asserted.
- **Fix**: Add `tests/test_label_factory_cross_language_parity.py` in this repo that loads a v3p0 day's `_forward_prices.npy` + computes labels via Python `LabelFactory.multi_horizon` + asserts byte-for-byte match with `_regression_labels.npy`.
- **Effort**: 2 hr.

### XC-2: `canonical_hash` mislabel docs

- The function lives in `hft_contracts.canonical_hash` and was promoted in trainer 6B.2 (per `CODEBASE.md:23`) as the SSoT for content hashing. **Search proves it's Python-only**: no Rust producer calls into a SHA-256 of the same domain bytes. Documentation occasionally implies cross-language SSoT — it isn't.
- **Fix**: Edit `hft-contracts/src/hft_contracts/canonical_hash.py` docstring to remove any "cross-language SSoT" claim; tag as "Python content-hash for FeatureSet integrity verification, NOT a cross-language algorithmic invariant."
- **Effort**: 30 min docs.

### XC-3: Welford parity (H-2 above; bridges to producer)

- Producer's Welford: `MBO-LOB-reconstructor/.../hft-statistics/src/welford.rs` (Rust).
- Consumer Python: `hft-metrics/src/hft_metrics/welford.py` AND `lob-model-trainer/src/lobtrainer/data/normalization.py:72-106` (duplicate).
- **3 implementations** of Chan/Golub/LeVeque algorithm. Trainer-side dedupe (H-2) brings it to 2. The Rust↔Python boundary is intentional (different language) but should still be parity-tested if normalization stats EVER round-trip Rust→Python or vice-versa.
- **Fix sequencing**: H-2 first (2-impl); add cross-language fixture test (Phase Q candidate). Trainer-side fix is 3 hr; full cross-language parity is +4 hr.

### XC-4: Phase-II compatibility_fingerprint chain (Phase N pivot Refinement-4)

- Backtester's `SignalManifest` validates a 3-way fingerprint check. Trainer's `signal_metadata.json` producers (H-1) currently lack `schema_version` — backtester silently passes Phase II tampering detection but not version-skew detection.
- **Fix coordination**: H-1 in trainer (add fields) + C-4 in backtester (consume + assert). MUST land atomically across 2 repos in same PR cycle to prevent producing signals that backtester then rejects.
- **Effort**: 1.5 hr trainer + 45 min backtester + integration test = ~3 hr.

### XC-5: Hft-ops "unknown" sentinel (PHASE_N_VALIDATION_FINDINGS BLOCKING-2)

- `hft-ops/src/hft_ops/feature_sets/producer.py:284-300` returns string "unknown" if pipeline schema lacks both `contract_version` and `schema_version`. Upstream of trainer's FeatureSet resolver (`feature_set_resolver.py`).
- **Trainer impact**: when trainer resolves a FeatureSet via `resolve_feature_set(...)`, the `expected_contract_version` is currently `_CURRENT_CONTRACT_VERSION` (correct). But if a FeatureSet was AUTHORED with `contract_version="unknown"` (because hft-ops couldn't determine it), then `_compute_content_hash` aliases drift across actual contract versions silently.
- **Fix**: hft-ops side — change to raise; trainer side — `feature_set_resolver.py` should raise on `contract_version in {"unknown", ""}`.
- **Effort**: 1 hr hft-ops + 30 min trainer = 1.5 hr.

### XC-6: lob-dataset-analyzer swallows ContractError (BLOCKING-3)

- `lob-dataset-analyzer/src/lobanalyzer/streaming/session.py:419-440` catches `ContractError` and downgrades to warning. Upstream/sibling of the trainer pipeline — not directly consumer of v3p0 but emits diagnostics that may feed into experiment provenance.
- **Fix**: Re-raise. 2-line change.
- **Effort**: 15 min.

### XC-7: Test coverage atomic gate

After all above ship, run:
```bash
cd lob-model-trainer && pytest tests/ -v -W error::DeprecationWarning --tb=short
cd ../lob-models && pytest tests/ -v -W error::DeprecationWarning --tb=short
cd ../lob-backtester && pytest tests/ -v
cd ../hft-contracts && pytest tests/ -v
```
Expected: lob-model-trainer ~1450 (+15-20 new validation tests), lob-models ~745 (+5-9), lob-backtester ~362, hft-contracts unchanged. CI must be GREEN before any live `hft-ops run` invocation.

### XC-8: Live experiment after fixes

The pipeline has not run end-to-end since 2026-03-19 (74+ days). Many of the dormant findings will only surface at runtime. **Recommendation per Forensic Audit §1**: "Fix N1 (single-line). Run an experiment via `hft-ops run`. Then come back for the multi-day fix cycle." Translated to consumer-side: ship C-1, C-2, C-3, H-1, M-3 (v3p0 dataset path swap), then RUN one e5_60s_v3p0 TemporalRidge experiment end-to-end before tackling the rest.

---

## 9. Effort summary

| Tier | Total effort | Items |
|---|---|---|
| **Minimum viable to train on v3p0** | ~10 hr | C-1, C-2, C-3, H-1, M-3, M-4, retire xfail (Step 6.3) |
| **Full critical + high** | ~25 hr | + H-2, H-3, H-4, H-5, H-6, H-7, M-1 (lob-models), C-4 |
| **Polish + cross-cutting** | ~40 hr | + all medium + cross-cutting |
| **One-time live verification** | +2-3 hr | XC-7 + XC-8 (run actual experiment) |

**Recommended next-session sequence**: C-1 → C-2 → C-3 → M-3 → M-4 → retire xfail → H-1 (with backtester C-4 coordination) → live experiment → tackle rest.

---

## 10. Source documents

This plan was compiled from:
- `lob-model-trainer/reports/TRAINING_PIPELINE_FORENSIC_AUDIT_2026_04_26.md` (moved into this repo's reports/ from monorepo root) — N1-N8 CRITICAL ACTIVE bugs, 9 DORMANT, 8 NEW Cross-Module
- Producer-readiness investigation Round 1-4 findings (11 agents, 2026-05-04)
- (Deleted) `POST_PHASE_M_VALIDATION_2026_05_02.md` — NEW-CRITICAL-1/2 + BLOCKING-1/2/3 (extracted into `PHASE_P_BACKLOG.md`)
- (Deleted) `PHASE_N_VALIDATION_FINDINGS_2026_05_02.md` — Section 8 inventory (~50 silent-fallback patterns, categorized into 7 categories — see PHASE_P_BACKLOG)
- Round 2 Agent C: confirmed ZERO operational impact for #318 (no consumer reads manifest.split.X.sequences)
