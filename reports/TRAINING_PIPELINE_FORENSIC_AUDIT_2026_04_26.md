# Training Pipeline Forensic Audit & Validation Report

**Scope**: lob-model-trainer + lob-models + cross-repo dependencies (hft-contracts, hft-ops, lob-backtester, hft-feature-evaluator, hft-metrics)
**Compiled**: 2026-04-26
**Status**: Authoritative cross-pipeline audit reference. Use this document — not prior agent reports — as the technical baseline for any future fix cycle.
**File location**: monorepo-root local-only (NOT under any git repo). Shared agent-coordination state.
**Document length**: ~26,000 words (extremely technical, citation-rich).

---

## CLOSURE STATUS BANNER (UPDATED 2026-05-07 post-#PY-63 cycle)

**THIS DOCUMENT IS A 2026-04-26 SNAPSHOT.** Multiple findings have been closed since via Phase 1 INTEGRITY GATE (POST-COMPACT-5 + POST-COMPACT-6) + ancillary cycles. Section bodies below are PRESERVED VERBATIM as the original audit narrative; closure status is tracked HERE at the top.

| Finding | Status | Closure Commit | Notes |
|---|---|---|---|
| **N1** InputContract preflight `_base:` | **CLOSED** | hft-ops `7904112` (Day 0 0d) | Multi-base manifests now resolve correctly; AST torch-free regression test added |
| **N2** `--resume` epoch ignored on resume | **CLOSED** | lob-model-trainer `576c217` (Phase 1) | `train()` loop respects `state.current_epoch`; original audit "epoch ignored" claim was VERIFIED REAL (not stale as a prior agent suggested) |
| **N3** EarlyStopping/ModelCheckpoint/MetricsLogger state on resume | **PARTIAL CLOSE** | Phase X.1.K minimum-viable (skip `on_train_start` reset on resume via `_resumed_from_checkpoint` sentinel; LOCAL UNCOMMITTED in Phase X.1 v2 working trees, ancestor of current HEAD) | Full `state_dict()/load_state_dict()` Protocol on Callback ABC + RNG capture deferred to Phase X.2.B |
| **N4** HMHP-R hardcoded `horizons[0]` | **CLOSED** | lob-model-trainer `ff5eb26` (Phase 1) | + 3 #PY-43 sister sites (cluster) closed in same commit; `primary_horizon_idx` SSoT used at all 5 sites |
| **N5** HMHP/HMHP-R encoder pool inconsistency | **CLOSED** | lob-models `4cbdc39` (Phase S 2026-05-04) | `pool_mode: Literal["last","mean"]` field + `_apply_pooling` SSoT helper; `hmhp_cascade_regression.yaml` migrated to `pool_mode: "mean"` for HMHP-R back-compat. **F-13 HIGH** structurally closed in Phase X.1 v2 via CompatibilityContract embedding (mismatch detection at load_checkpoint time) |
| **N6** Calibrated metrics describe RAW predictions | **CLOSED** | lob-model-trainer `bc3c0ee` (Phase 1) | Calibrated metrics now report on the calibrated array; `_apply_calibration` flow re-wired |
| **N7** Normalization stats not bound to checkpoint | **CLOSED** | lob-model-trainer `0cf9867` (Phase 1, POST-COMPACT-6 night) | Normalization-stats SHA bound to checkpoint; resume-time mismatch raises. Sklearn N7 sidecar binding deferred per #PY-53 architectural rationale |
| **N8** TLOB final-flatten ordering | **CLOSED** | lob-models `1bea036` (Phase 1) | Berti & Kasneci 2025 reference alignment; pretrained checkpoint loadability restored |

**ADJACENT CLOSURES** (Phase Q + S + Q.6.5 + X.1 v2 + X.2.A.1+A.2 + Phase Y/Z + Stage 8 cycles, 2026-05-04 → 2026-05-05):
- F-1 `BaseConfig.from_dict` recursive — Phase X.1.B
- F-3 MLPLOB ModelType enum + ModelConfig fields — Phase X.1.I
- F-12 pre-Phase-S HMHP-R checkpoint pool drift — Phase S documentation + Phase X.1 v2 detection
- F-13 HMHP-R programmatic checkpoint silent-pool-mismatch — Phase X.1 v2 (HIGH)
- F-16 sklearn pipeline broken via canonical scripts — Phase Q.6.5.B
- F-18 sklearn signal_metadata missing compatibility block — Phase Q.6.5.A (Phase Y prereq)
- N4 hardcoded horizons[0] sister sites (#PY-43 cluster) — Phase 1 ff5eb26

**OPEN ITEMS** (post 2026-05-07):
- **N3 full Protocol** + RNG capture/restore — deferred to Phase X.2.B (Boundary Discipline Cycle, designed but not started)
- **N7 sklearn-side sidecar binding** — deferred per #PY-53 architectural rationale
- 36 LOW/MEDIUM-impact silent-NaN sites — F-cycle (deferred from #PY-63 closure)
- 4 NEW lob-models silent-NaN sites discovered Round 1 (2026-05-07): `temporal_ridge.py:79` + `temporal_gradboost.py:105` (silent fit() NaN-row drop), `temporal_ridge.py:84-87` + `temporal_gradboost.py:117-120` (predict() no-raise on NaN). HIGH-adjacent, downstream-mitigated by `simple_trainer.py:464` save-boundary + `regression_metrics → r_squared` validation-boundary; tracked as #PY-64 in PHASE_P_BACKLOG.md
- F-cycle scope expansion to producer-side Rust crates (MBO-LOB-analyzer + feature-extractor + reconstructor — Round 2 Agent G D7 finding); tracked as #PY-65

**Most recent cycle**: #PY-63 silent-NaN cluster closure (5 commits SHIPPED LOCAL 2026-05-07: hft-contracts `f63eaf6`, lob-model-trainer `69f09bd`, lob-backtester `257c2ac`, hft-metrics `72b1aa5`, hft-ops `45149eb`). See `PHASE_P_BACKLOG.md` §#PY-63 closure marker + `PY63_CYCLE_HANDOFF_2026_05_07.md` for full narrative.

**For current state**, see CLAUDE.md root banner + PHASE_P_BACKLOG.md. **For new findings after this banner's date**, consult the per-cycle handoff docs at monorepo root.

---

## Document Purpose and Use

This document is the **canonical technical reference** for findings from a 13-agent forensic audit + validation cycle conducted 2026-04-25 through 2026-04-26 on the training pipeline (the heart of the HFT pipeline-v2 monorepo).

**The cycle had two phases**:

1. **Forensic audit (2026-04-25)** — 6 agents in parallel, each scoped to a specific layer of the training pipeline (Data Pipeline, Training Loop, Model Architecture, Calibration & Signal Export, Configuration, Gates & Importance). Each produced a forensic report with severity-ranked findings.

2. **Validation round (2026-04-26)** — 7 agents with fresh eyes independently re-verified each prior claim by reading the actual code, running reproducers, and tracing data flow. The validation refuted 5 prior claims as FALSE, downgraded several from CRITICAL-ACTIVE to DORMANT, and surfaced 9 NEW findings the prior round had missed entirely (1 CRITICAL: N1 InputContract preflight `_base:` bug; 8 Cross-Module boundary findings F2-F9).

**This document captures the synthesis of both phases** with extreme technical precision. It is intended to:

- **Prevent future LLM agents from being misled** by stale documentation, incomplete prior agent reports, or rediscovering claims that have already been refuted.
- **Provide complete root-cause and impact analysis** for every confirmed bug so a fix cycle can be planned without re-investigating from scratch.
- **Distinguish between active production-impacting bugs, dormant bugs (real but no current trigger), refuted claims (false positives), and cleared items (verified sound)** — so engineering effort is allocated honestly.
- **Document architectural patterns** (cross-cutting root causes) so future work targets the bug class, not just instances.

**This document deliberately does NOT prescribe fixes.** Suggested fix scopes are listed for planning only. Fix sequencing belongs in a separate planning document informed by this audit.

---

## How to Read This Document

### Severity tags

- **CRITICAL — ACTIVE**: bug is currently producing wrong artifacts/behavior in production code paths. Fix before next live experiment.
- **CRITICAL — IMMINENT**: bug is dormant only because a specific control-flow ordering happens to skip it; the next routine pipeline operation triggers silent corruption. Fix before any change that could perturb the ordering.
- **CRITICAL — DORMANT-PRIMED**: bug exists in the code but no current usage path triggers it. Activates when a specific (named) usage pattern occurs.
- **HIGH**: bug produces silent bias, wrong projections, or fragile invariants but is not (yet) corrupting artifacts.
- **MEDIUM**: correctness sound but architectural fragility (e.g., performance bottleneck, brittle convention, observability loss).
- **LOW**: style, documentation drift, defense-in-depth gap.

### Status tags

- **NEW**: surfaced by the validation round (2026-04-26); was missed by the prior audit (2026-04-25).
- **CONFIRMED**: prior audit + validation round both agree the bug is real.
- **PARTIAL**: prior audit was partially correct; validation refined the verdict (e.g., true for some metrics, false for others).
- **REFUTED**: prior audit claimed a bug; validation determined the code is actually correct.
- **CLEARED**: an item explicitly verified to be working correctly by direct code inspection.

### Citation conventions

- Every claim cites `relative/path/file.py:LINE_NUMBER` or `LINE_RANGE`.
- All code blocks contain ACTUAL code copied from the file at the cited lines (NOT paraphrased).
- Reproducer scripts are self-contained Python where possible; conceptual where execution requires GPU/data.
- Validation source is named (V1-V7) so the reader knows which agent confirmed/refuted each claim.

### Document layout

| Section | Contents | Length |
|---|---|---|
| 1. Executive Summary | High-level outcome | ~600 words |
| 2. Methodology | Audit + validation discipline | ~1000 words |
| 3. Verdict Summary Table | One-row-per-finding overview | ~table |
| **4. CRITICAL Findings — Active or Imminent** | 8 entries with full technical depth | ~6000 words |
| **5. DORMANT Findings — Real Bugs with No Production Trigger Today** | 9 entries with activation conditions | ~4500 words |
| **6. Cross-Module Boundary Findings (NEW)** | 8 entries from V7 fresh-eyes round | ~3500 words |
| **7. REFUTED Claims** | 5 entries — what is NOT a bug despite prior claims | ~2500 words |
| **8. Architectural Patterns and Root Causes** | 6 cross-cutting patterns | ~3000 words |
| **9. CLEARED Items — Specifically Verified Sound** | 52 entries across all layers | ~3500 words |
| 10. Production Impact Assessment | Per-experiment, per-config impact map | ~1500 words |
| 11. Suggested Priority Tiers | NOT a fix plan — planning input only | ~800 words |
| Appendix A | Reproducer Scripts | ~varies |
| Appendix B | File:Line Citation Table | ~varies |
| Appendix C | Test Coverage Gaps | ~varies |
| Appendix D | Cross-Reference: Round 1 vs Round 2 Verdicts | ~varies |

---

## 1. Executive Summary

A 13-agent audit + validation cycle examined the training pipeline (lob-model-trainer + lob-models) and its consumer/producer boundaries (hft-contracts, hft-ops, lob-backtester, hft-feature-evaluator, hft-metrics) at the file:line level. The cycle confirmed:

- **8 CRITICAL bugs are ACTIVE or IMMINENT** in production-reachable code paths. 1 of these (N1 InputContract preflight `_base:` bug) was entirely missed by the prior audit and surfaced only in the validation round. N2 (`--resume` epoch counter ignored) was first reported in Round 5 forensic audit and re-confirmed in validation. Several known issues (HMHP/HMHP-R pooling, calibrated metrics describing raw predictions) are real and currently silently impact shipped experiments.

- **9 DORMANT bugs are real but have no current production trigger**. Each has a clearly documented activation condition (e.g., D1 DataConfig.labels staleness activates only when a future CLI flag exposes `labeling_strategy`; D4 sample weights pre-trim activates only with multi-source experiments). Phase A.5.9's deferral decisions on the 3 Pydantic escape hatches were CORRECT — production code does not currently trigger them.

- **8 Cross-Module boundary bugs** (V7 fresh-eyes round) document silent-failure surfaces at trainer↔hft-ops↔backtester boundaries (broad except swallows, non-atomic config writes, RAM-exhaustion in concatenation loaders, etc.). All real, mostly MEDIUM severity, mostly fixable in <1 day each.

- **5 prior claims were REFUTED**. The validation determined that exporter.py:502 1-D fallback `or 0`, ExperimentConfig.input_size staleness, BASIC stride defaulting, PostTrainingGate test/val cascade mismatch, and Spearman-vs-Pearson IC mismatch are NOT bugs — the prior agents misread the code. **Future agents must not re-claim these.**

- **6 Architectural patterns** explain ~80% of the findings: (P1) three independent primary_horizon channels, (P2) Pydantic validator-skip pattern, (P3) resume semantics broken at multiple layers, (P4) calibration producer/consumer drift, (P5) defaults can be silently invalid, (P6) broad except Exception swallows correctness bugs.

- **52 items are explicitly CLEARED** as verified sound (AFML sample-weights formula, HMHP P0-3 guard, HMHP-R FRESH-2 dead-model gate, BiN parameter naming math, optimizer step ordering, atomic_write_json, gate_reports fingerprint exclusion, etc.).

**Top-level recommendation** (not a fix plan, just a planning input): N1 (InputContract preflight `_base:`) is a single-line fix that unblocks the first live experiment via `hft-ops run`. Fix it. Run an experiment. Then come back for the multi-day fix cycle informed by what real data surfaces.

---

## 2. Methodology

### Agent topology — Round 1 (forensic audit, 2026-04-25)

Six agents dispatched in parallel, each with non-overlapping scope:

| Agent | Layer | Scope (modules audited) |
|---|---|---|
| 1 | Data Pipeline | `lob-model-trainer/data/`, `bundle.py`, `dataset.py`, `cv_trainer.py`, sample weights, normalization |
| 2 | Training Loop | `trainer.py`, `strategy.py`, `strategies/*`, `loss.py`, `callbacks.py`, AMP, checkpointing |
| 3 | Model Architecture | `lob-models/models/*`, `layers/*`, `losses/*`, `registry/`, output contracts |
| 4 | Calibration & Signal Export | `calibration/variance.py`, `export/exporter.py`, `metadata.py`, signal_metadata.json |
| 5 | Configuration System | `config/schema.py` (9 SafeBaseModel classes), `base.py`, `paths.py`, `merge.py`, validators |
| 6 | Gates / Importance / Metrics | `post_training_gate.py`, `validation.py`, `fast_gate.py`, importance callbacks |

Each agent received an extensive prompt (~3000 words) with: scope boundaries, specific known-bug patterns to look for (e.g., `[:, 0]` slicing, broad except, validator self-mutation), output format spec, and explicit guardrails (read code, do not trust documentation, do not edit anything, ultrathink).

Each produced a forensic report with severity-ranked findings (CRITICAL/HIGH/MEDIUM/LOW) plus "Cleared" items they specifically verified sound.

### Agent topology — Round 2 (validation, 2026-04-26)

Seven agents dispatched in parallel, organized by **bug class** (cross-cutting) rather than by layer:

| Agent | Bug class scope |
|---|---|
| V1 | primary_horizon_idx recurrence (validates 5 sites + open-ended search across 6 repos) |
| V2 | Pydantic model_copy(update=) staleness (validates 4 sites + executes reproducers) |
| V3 | Checkpoint resume + determinism (validates 3 sites + finds 2 NEW critical bugs) |
| V4 | Model architecture (validates 3 sites with NUMERIC parity tests against TLOB reference) |
| V5 | Data pipeline (validates 7 sites including CV embargo + sample weights + norm stats) |
| V6 | Gates / importance / calibration metrics (validates 7 sites) |
| V7 | Find what we missed — open-ended fresh-eyes audit of cross-module boundaries |

Each validation agent was instructed to be SKEPTICAL of the prior round's claims, run reproducers via `Bash` where feasible, and produce TRUE/PARTIAL/FALSE/DORMANT verdicts with code-level evidence.

### Validation discipline

- **Code is ground truth** — agents must read the actual code at every cited file:line. Documentation (CODEBASE.md, CLAUDE.md, docstrings) was treated as potentially stale.
- **No edits** — both rounds were strictly read-only. No code changes were made.
- **Numeric reproducers** — wherever an empirical claim was made (e.g., "cosine similarity 0.18 between HMHP and HMHP-R pool"), the validation agent ran an actual numeric test and reported real stdout.
- **Honest verdicts** — agents were explicitly instructed that finding the prior audit was WRONG was as valuable as confirming a bug. Five claims were refuted; all are documented in Section 7.
- **Severity calibration** — DORMANT was distinguished from ACTIVE by examining whether any current production code path triggers the bug.

### Coverage gaps

- **No live experiment was run**. All findings are from static code analysis + numeric microbenches. The pipeline has not been exercised end-to-end via `hft-ops run` since 2026-03-19. Some bugs (notably N1) only manifest at runtime.
- **Cross-codebase reference compared structurally**, not via checkpoint loading. TLOB final-flatten ordering was confirmed via numeric microbench but not via attempting to load an official Berti & Kasneci checkpoint.
- **DataLoader worker pickle survival was not exhaustively tested**. SafeBaseModel pickle is locked (Phase A.5.7b); DayBundle pickle was spot-tested (V5) but a full multi-day, num_workers>0 stress test was not run.
- **AMP / mixed-precision paths**: not in scope (verified absent from the trainer).
- **Multi-GPU / distributed training**: not exercised.

These gaps are itemized in Appendix C.

---

## 3. Verdict Summary Table

| ID | Title | Severity | Status | Layer |
|---|---|---|---|---|
| **N1** | InputContract preflight does NOT resolve `_base:` inheritance | CRITICAL | ACTIVE NEW | Cross-Module |
| **N2** | `--resume` epoch counter ignored — resume re-runs N redundant epochs | CRITICAL | ACTIVE | Training Loop |
| **N3** | EarlyStopping/ModelCheckpoint/MetricLogger state lost on load_checkpoint | CRITICAL | ACTIVE | Training Loop |
| **N4** | HMHP-R primary metrics use `horizons[0]` not `primary_horizon_idx` | CRITICAL | ACTIVE | Training Loop |
| **N5** | HMHP vs HMHP-R encoder pooling inconsistency (P0-9) | CRITICAL | ACTIVE | Model Architecture |
| **N6** | Calibrated metrics in signal_metadata.json describe RAW predictions | CRITICAL | ACTIVE | Calibration & Export |
| **N7** | Normalization stats NOT bound to checkpoint (re-export hazard) | CRITICAL | DORMANT-PRIMED | Data Pipeline |
| **N8** | TLOB final-flatten ordering differs from Berti & Kasneci reference | CRITICAL | DORMANT-PRIMED | Model Architecture |
| **D1** | DataConfig.labels staleness on `model_copy(update={"labeling_strategy"})` | HIGH | DORMANT | Configuration |
| **D2** | ModelConfig.params staleness on `model_copy(update={"tlob_hidden_dim"})` | HIGH | DORMANT | Configuration |
| **D3** | PrivateAttr cache lost on `model_copy(update=...)` non-empty branch | HIGH | DORMANT | Configuration |
| **D4** | Sample weights computed pre-trim, mean drifts after multi-source align | HIGH | DORMANT | Data Pipeline |
| **D5** | HybridNormalizer fallback DEFAULT_EXCLUDE_INDICES missing index 95 | LOW | DORMANT | Data Pipeline |
| **D6** | Day-1-only schema validation in load_split_data | MEDIUM | DORMANT | Data Pipeline |
| **D7** | Stability NaN on perfectly stable feature → silent gate FAIL | LOW | DORMANT | Gates |
| **D8** | Zero `prior_best_value` → unconditional gate PASS regardless of current | LOW | DORMANT | Gates |
| **D9** | RNG state NOT captured in checkpoint (resume deterministic-but-misaligned) | HIGH | DORMANT (composes with N2) | Training Loop |
| **F2** | train.py final-evaluate catches only ValueError | HIGH | NEW | Training Loop |
| **F3** | hft-ops `_capture_training_metrics` swallows JSON corruption | HIGH | NEW | Cross-Module |
| **F4** | BacktestData validates only `prices`, NaN propagates | HIGH | NEW | Cross-Module |
| **F5** | Backtester DataLoader concatenates ALL train days into RAM | MEDIUM | NEW | Cross-Module |
| **F6** | train.py:save_config write is non-atomic | MEDIUM | NEW | Training Loop |
| **F7** | hft-ops `_apply_overrides` non-atomic YAML write | MEDIUM | NEW | Cross-Module |
| **F8** | PermutationImportanceCallback swallows all post-train failures | MEDIUM | NEW | Importance |
| **F9** | classification.py:151 broad except masks class-count failures | LOW | NEW | Training Loop |
| **F-1** | exporter.py:502 1-D fallback `or 0` idiom-trap | — | REFUTED | Calibration |
| **F-2** | ExperimentConfig.input_size staleness on model_copy(update=) | — | REFUTED | Configuration |
| **F-3** | BASIC export stride=1 default breaking AFML | — | REFUTED | Data Pipeline |
| **F-4** | PostTrainingGate test/val primary metric mismatch | — | REFUTED | Gates |
| **F-5** | Spearman vs Pearson IC mismatch between callback and trainer | — | REFUTED | Gates |

**Total**: 8 CRITICAL ACTIVE/IMMINENT + 9 DORMANT + 8 NEW Cross-Module + 5 REFUTED + 52 CLEARED.

---

## 4. CRITICAL Findings — Active or Imminent

This section documents 8 critical bugs that are ACTIVE in the production trainer / model / exporter / orchestrator code, or are DORMANT-PRIMED such that the next routine pipeline operation triggers silent corruption. Every claim has been verified by reading the actual code at each cited line during this audit pass; where prior audit findings disagreed with the code-as-it-stands, the verdict has been adjusted in the entry. Each finding is technically deep enough that a future fix cycle can plan an intervention without re-investigating from scratch.

---

### N1. InputContract pre-flight does NOT resolve `_base:` YAML inheritance — every multi-base manifest silently fails the model_type check at preflight time

**Severity**: CRITICAL — ACTIVE
**Status**: NEW (validation-discovered)
**Layer**: Cross-Module / Configuration
**First reported**: Round 7 validation 2026-04-26

**Files and lines**:
- `hft-ops/src/hft_ops/stages/contract_preflight.py:287-311` — `preflight_trainer_config` reads YAML and indexes into `cfg["model"]["model_type"]` directly.
- `hft-ops/src/hft_ops/stages/training.py:100-124` — `_apply_overrides` calls `yaml.safe_load` then `_apply_overrides_to_dict` then `yaml.dump`, with no `resolve_inheritance` call.
- `lob-model-trainer/src/lobtrainer/config/merge.py:85-118` — the canonical `resolve_inheritance(data, config_path)` resolver that EVERY trainer-side load goes through (`schema.py::ExperimentConfig.from_yaml`).
- `lob-model-trainer/configs/bases/models/tlob_compact_bare.yaml:16` — `model_type: tlob` lives in the BASE.
- `lob-model-trainer/configs/bases/models/hmhp_cascade_bare.yaml:21` — `model_type: hmhp` lives in the BASE.
- `lob-model-trainer/configs/bases/models/hmhp_cascade_regression.yaml:20` — `model_type: hmhp_regression` lives in the BASE.
- `lob-model-trainer/configs/experiments/e5_60s_huber_cvml.yaml:9-26` — production leaf YAML using `_base:` list and NOT setting `model.model_type` locally.
- `lob-model-trainer/configs/experiments/nvda_hmhp_40feat_h60_profit8bps_regression.yaml:17` — likewise, `_base:` form.

**What the code actually does**:

`hft-ops/src/hft_ops/stages/contract_preflight.py:287-304`:

```python
with open(trainer_config_path) as f:
    cfg = yaml.safe_load(f) or {}

if not isinstance(cfg, dict):
    raise ValueError(
        f"Input contract pre-flight: trainer config at {trainer_config_path!r} "
        f"is not a mapping (got {type(cfg).__name__})."
    )

model_block = cfg.get("model") or {}
data_block = cfg.get("data") or {}

model_name = model_block.get("model_type")
if not model_name or not isinstance(model_name, str):
    raise ValueError(
        f"Input contract pre-flight: trainer config missing "
        f"model.model_type (got {model_name!r})."
    )
```

`hft-ops/src/hft_ops/stages/training.py:114-124`:

```python
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f) or {}

_apply_overrides_to_dict(cfg, overrides)

output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

return output_path
```

`lob-model-trainer/configs/experiments/e5_60s_huber_cvml.yaml:9-26`:

```yaml
_base:
  - "../bases/models/tlob_compact_regression.yaml"
  - "../bases/datasets/nvda_e5_60s.yaml"
  - "../bases/labels/regression_huber.yaml"
  - "../bases/train/regression_default.yaml"

name: E5_60s_Huber_CVML
description: |
  E5 Phase 2 CVML test ...
output_dir: outputs/experiments/e5_60s_huber_cvml
log_level: INFO
tags: [e5, nvda, regression, h10, 60s, huber, cvml]

# Per-child overrides: enable CVML
model:
  tlob_use_cvml: true
  tlob_cvml_out_channels: 49
```

Trace: `_apply_overrides` reads YAML with `yaml.safe_load`, applies dotted-key overrides from the manifest, and writes back via `yaml.dump`. It NEVER calls `resolve_inheritance`. So the materialized "effective" trainer YAML retains the literal `_base: [...]` list and the `model:` block contains ONLY the per-child overrides (`tlob_use_cvml: true`, `tlob_cvml_out_channels: 49`) — `model_type` is absent because it lives in `bases/models/tlob_compact_bare.yaml:16`. The pre-flight then opens this materialized YAML, reads `cfg["model"]["model_type"]`, finds `None`, and raises `ValueError("Input contract pre-flight: trainer config missing model.model_type (got None).")` BEFORE the trainer subprocess is launched.

Meanwhile the trainer subprocess itself works fine because `lobtrainer.config.load_config` calls `resolve_inheritance` at load time (`schema.py:ExperimentConfig.from_yaml`).

**Why this is a bug**:

The pre-flight stage's purpose is to catch contract mismatches BEFORE GPU compute is spent on a doomed run. It currently does the opposite: every multi-base manifest (every leaf experiment in `configs/experiments/` that uses `_base:` — which is the canonical pattern documented in `configs/bases/README.md`) fails preflight on a bookkeeping artifact rather than reaching the actual contract check. Per hft-rules §0 reuse-first and §11 "docs reflect code exactly": the project ships `lobtrainer.config.load_config` precisely so that every consumer goes through the same SSoT loader. Pre-flight bypasses it.

**Reproducer**:

```python
# Run from any directory
import yaml, json
from pathlib import Path

p = Path("lob-model-trainer/configs/experiments/e5_60s_huber_cvml.yaml")
cfg = yaml.safe_load(p.read_text())
print("Pre-flight reads:", json.dumps({"model_type": cfg.get("model", {}).get("model_type")}, indent=2))
# -> {"model_type": null}

from lobtrainer.config.merge import resolve_inheritance
resolved = resolve_inheritance(cfg, p)
print("After resolve_inheritance:", resolved["model"]["model_type"])
# -> tlob
```

**Empirical evidence**: a fresh read of `e5_60s_huber_cvml.yaml:24-26` confirms the leaf `model:` block contains only `{tlob_use_cvml: true, tlob_cvml_out_channels: 49}`. `tlob_compact_bare.yaml:16` (transitively pulled via `tlob_compact_regression.yaml`) contains `model_type: tlob`. `_apply_overrides` does not merge bases. Therefore the materialized YAML has a `model:` block with no `model_type` key.

**Production impact**:
- Configs affected: ALL leaf YAMLs under `lob-model-trainer/configs/experiments/` that use `_base:` form. From a quick filesystem grep this includes every E5/E6 manifest and every HMHP manifest currently active. This is the canonical production pattern after Phase 3.
- Observable manifestation: `hft-ops run <manifest>` calls preflight first; preflight raises `ValueError`; the run aborts; the operator sees a misleading "missing model.model_type" message even though the manifest is well-formed. Every fingerprint comparison + every gate downstream is blocked.
- Cross-reference: `EXPERIMENT_INDEX.md` cites E5_60s_Huber_NoCVML, E5_60s_Huber_CVML, HMHP variants. None of these can run end-to-end through the orchestrator until this is fixed.

**Root cause analysis**:

Phase V.A.8 MVP (2026-04-21) introduced `_INPUT_CONTRACTS` pre-flight to catch invalid model+feature_count combinations at validation-time rather than after subprocess launch. The MVP author tested only against a single-file (no `_base:`) fixture in `tests/test_contract_preflight.py`. The architectural distinction between "raw author-time YAML" and "resolved effective config" was missed: `lobtrainer.config.load_config` is the SSoT loader and the preflight bypasses it. The bug landed silent because no live `hft-ops run` invocation occurred between Phase V.A.8 ship and Frame 5 Task 1 (2026-04-23, also unable to reach the gate due to OTHER orchestrator integration bugs N#2/#5 of that cycle which masked this one).

**Suggested fix scope (NOT EXECUTED — for future planning)**:
- Minimal patch: change `cfg = yaml.safe_load(f) or {}` at `contract_preflight.py:288` to `from lobtrainer.config import load_config; cfg = load_config(trainer_config_path).to_dict()` — but `load_config` pulls torch (forbidden in hft-ops per the L2.6 AST regression test). Instead: `from lobtrainer.config.merge import resolve_inheritance; cfg = resolve_inheritance(yaml.safe_load(open(trainer_config_path)), trainer_config_path)`. The `merge` module is torch-free.
- Principled fix: extract a torch-free `lobtrainer.config.merge` import shim in `hft-contracts` (sibling to `atomic_io`/`canonical_hash`) so cross-module consumers can resolve inheritance without pulling the trainer's torch dependency tree. Add an AST-based torch-free regression test on `contract_preflight.py` that imports `_resolve_inheritance` and proves no torch landed.
- Architectural retirement: Phase VI snapshot architecture replaces the hardcoded `_INPUT_CONTRACTS` table with a `lobmodels.registry._snapshot.json` consumed via `importlib.resources`. As part of that move, also relocate the YAML resolution to a torch-free `lobtrainer.config.resolver` package re-exported from `hft_contracts` so every cross-module consumer sees the same resolved view.

---

### N2. `--resume` epoch counter from checkpoint is IGNORED — train loop always starts at `range(0, cfg.epochs)`

**Severity**: CRITICAL — ACTIVE
**Status**: CONFIRMED (prior audit + validation)
**Layer**: Training Loop
**First reported**: Round 5 forensic 2026-04-25, re-validated Round 7

**Files and lines**:
- `lob-model-trainer/src/lobtrainer/training/trainer.py:807-848` — main `train()` loop.
- `lob-model-trainer/src/lobtrainer/training/trainer.py:1007-1033` — `load_checkpoint()`.
- `lob-model-trainer/src/lobtrainer/training/trainer.py:1026` — `self.state.current_epoch = checkpoint.get('epoch', 0)`.
- `lob-model-trainer/src/lobtrainer/training/trainer.py:808-809` — `for epoch in range(cfg.epochs): self.state.current_epoch = epoch`.

**What the code actually does**:

`trainer.py:807-810`:

```python
try:
    for epoch in range(cfg.epochs):
        self.state.current_epoch = epoch
        self.callbacks.on_epoch_start(epoch)
```

`trainer.py:1025-1033`:

```python
        # Restore state
        self.state.current_epoch = checkpoint.get('epoch', 0)
        self.state.global_step = checkpoint.get('global_step', 0)
        
        if 'state' in checkpoint:
            self.state.best_val_metric = checkpoint['state'].get('best_val_metric', float('inf'))
            self.state.best_epoch = checkpoint['state'].get('best_epoch', 0)
        
        logger.info(f"Loaded checkpoint from {path} (epoch {self.state.current_epoch})")
```

Trace: a typical `--resume` flow calls `Trainer.load_checkpoint(...)` BEFORE `Trainer.train()`. `load_checkpoint` correctly populates `self.state.current_epoch` to the saved epoch (e.g., 4). Then `train()` enters its loop at line 808 — `for epoch in range(cfg.epochs):` — which always iterates `0, 1, 2, ..., cfg.epochs - 1` regardless of `state.current_epoch`. Line 809 IMMEDIATELY overwrites `state.current_epoch` with `epoch=0`. The model and optimizer state are already loaded from the checkpoint (model is at epoch 4, optimizer is at the corresponding momentum state) but the loop counter says we're at epoch 0. The trainer redoes "epochs 0..4" (with already-trained-to-4 weights) then continues with "epochs 5..N-1".

**Why this is a bug**:

Resume semantics by universal convention (PyTorch `state_dict` workflows, Hugging Face Trainer, TensorFlow `tf.train.Checkpoint`) is "continue from the next unfinished epoch." `cfg.epochs` is defined as the TOTAL number of epochs to train (not "additional epochs after this resume"). The current code semantics require `cfg.epochs` to mean "total" but the loop re-traverses the entire range — they are incompatible. Per hft-rules §7 (determinism + reset semantics): "Stateful components must define and test reset semantics (what resets, what persists, and why)." Resume-load-then-train has no test (`find -name '*resume*'` returned 0 hits in `lob-model-trainer/tests/`).

**Reproducer**:

```python
# Conceptual — full reproduction requires a tiny synthetic dataset.
trainer = create_trainer(cfg)  # cfg.epochs = 10
trainer.setup()
trainer.load_checkpoint("path/to/epoch_4.pt")  # state.current_epoch = 4
result = trainer.train()
# Expected: 5 more epochs trained (5..9). result["total_epochs"] == 10.
# Actual: 10 epochs trained (0..9). The model already-trained-to-4 sees its
#         own optimizer state from epoch 4 carried into "epoch 0" of the
#         new run; gradient updates compound on already-trained weights;
#         RNG is freshly seeded so data shuffling differs from the
#         first-run epochs 0..4.
```

**Empirical evidence**: `train()` returns `total_epochs = self.state.current_epoch + 1` (`trainer.py:862`). After a `--resume` with `cfg.epochs=10`, this evaluates to 10 (the loop naturally exits at epoch 9). The training log records 10 epoch lines, NOT 5. There is no log diff between fresh-train-10-epochs vs resume-from-4-train-10. A test `test_resume_equals_fresh_training` would catch this immediately and does not exist.

**Production impact**:
- Configs affected: any operator manually invoking `lobtrainer train --resume <ckpt>` or any orchestrator path that calls `load_checkpoint` then `train()`. The `--resume` flag exists in `scripts/train.py` argparse.
- Observable manifestation: silent over-training on pre-trained weights. EarlyStopping (N3) is also reset, so the model trains past its previously-found best epoch. `training_history.json` is overwritten by `MetricLogger` on `on_train_start` (N3). `best.pt` may be overwritten by a worse model.
- Cross-reference: no current `EXPERIMENT_INDEX.md` entry uses `--resume`, but Frame 5 Task 1d (deferred) was planning to. The only reason this hasn't manifested in production is that no live resume run has been attempted in 35+ days.

**Root cause analysis**:

The trainer was authored as a single-pass `train()` method without resume support. `load_checkpoint` was added later for inference/eval (the `lobtrainer eval` path) where the loop is never entered. The lack of an explicit resume API + the implicit `load_checkpoint` → `train()` ordering convention created the gap. No `test_resume_*` was written because no production manifest invoked the path.

**Suggested fix scope (NOT EXECUTED — for future planning)**:
- Minimal patch (`trainer.py:808`): change `for epoch in range(cfg.epochs):` to `for epoch in range(self.state.current_epoch, cfg.epochs):`. This makes `--resume` behavior consistent with universal convention.
- Principled fix: introduce explicit `Trainer.resume(checkpoint_path)` method that calls `load_checkpoint`, sets a `_resumed: bool` flag, and `train()` reads the flag to decide the start-epoch. Save also stores a `cfg` snapshot so `--resume` from a checkpoint with `cfg.epochs=5` cannot silently extend to `cfg.epochs=10` without operator confirmation.
- Architectural retirement: adopt PyTorch Lightning's `Trainer.fit(model, ckpt_path=...)` semantics (resume + start-epoch-detection + state restoration as a single atomic operation) and deprecate the manual `load_checkpoint` + `train` chain.

---

### N3. EarlyStopping + ModelCheckpoint + MetricLogger state NOT restored on `load_checkpoint`

**Severity**: CRITICAL — ACTIVE
**Status**: CONFIRMED (prior audit + validation)
**Layer**: Training Loop
**First reported**: Round 5 forensic 2026-04-25

**Files and lines**:
- `lob-model-trainer/src/lobtrainer/training/trainer.py:982-1004` — `save_checkpoint`.
- `lob-model-trainer/src/lobtrainer/training/trainer.py:1007-1033` — `load_checkpoint`.
- `lob-model-trainer/src/lobtrainer/training/callbacks.py:246-255` — `EarlyStopping.on_train_start` resets `best_value, wait_count, best_epoch, _best_weights`.
- `lob-model-trainer/src/lobtrainer/training/callbacks.py:390-400` — `ModelCheckpoint.on_train_start` resets `_best_value, _saved_checkpoints, _best_checkpoint_path`.
- `lob-model-trainer/src/lobtrainer/training/callbacks.py:534-536` — `MetricLogger.on_train_start` resets `_history = []`.

**What the code actually does**:

`trainer.py:989-998`:

```python
checkpoint = {
    'epoch': self.state.current_epoch,
    'global_step': self.state.global_step,
    'model_state_dict': self.model.state_dict(),
    'optimizer_state_dict': self.optimizer.state_dict(),
    'config': self.config.to_dict(),
    'state': {
        'best_val_metric': self.state.best_val_metric,
        'best_epoch': self.state.best_epoch,
    },
}
```

`callbacks.py:246-255` (EarlyStopping):

```python
def on_train_start(self) -> None:
    """Reset state at start of training."""
    if self.mode == 'min':
        self._state.best_value = float('inf')
    else:
        self._state.best_value = float('-inf')
    self._state.best_epoch = 0
    self._state.wait_count = 0
    self._state.stopped = False
    self._best_weights = None
```

`callbacks.py:390-400` (ModelCheckpoint):

```python
def on_train_start(self) -> None:
    """Create save directory."""
    self.save_dir.mkdir(parents=True, exist_ok=True)
    
    # Reset state
    if self.mode == 'min':
        self._best_value = float('inf')
    else:
        self._best_value = float('-inf')
    self._saved_checkpoints = []
    self._best_checkpoint_path = None
```

`callbacks.py:534-536` (MetricLogger):

```python
def on_train_start(self) -> None:
    """Clear history at start of training."""
    self._history = []
```

Trace: `save_checkpoint` saves only `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `epoch`, `global_step`, and the trainer-level `best_val_metric`/`best_epoch`. NO callback state is captured. The `Callback` base class has no `state_dict()`/`load_state_dict()` interface. After `load_checkpoint`, the next call to `train()` invokes `self.callbacks.on_train_start()` (line 802) which iterates each registered callback. Each callback's `on_train_start` RESETS its internal state — `wait_count` to 0, `_history` to `[]`, `_best_value` back to `inf`.

**Why this is a bug**:

Three concrete failure modes, each silent:

1. **EarlyStopping**: Suppose original training stopped at epoch 8 because `wait_count` hit `patience=5` after no improvement since epoch 3. The checkpoint at epoch 4 (the best) is loaded. After resume, `wait_count` resets to 0, `best_value` resets to `inf`. The trainer now has another full `patience=5` epochs of patience. It will train 5+ more epochs before early-stopping fires again. The original run's stopping decision is forgotten.
2. **ModelCheckpoint**: `_best_value = inf` after resume. The first epoch's val_metric is automatically `_is_better` than `inf` → triggers save → `_best_checkpoint_path` overwritten with the resumed-then-trained model's first-epoch state. If the resumed run is worse than the original best, `best.pt` is silently corrupted.
3. **MetricLogger**: `_history = []` after resume. `training_history.json` written at the end of resume contains ONLY the post-resume epochs. The original 0..4 history is lost from the JSON record (still in the original log files but not in the structured artifact).

Per hft-rules §7 ("Stateful components must define and test reset semantics") and §13 ("Document failures as precisely as successes"), this constitutes silent state corruption.

**Reproducer**:

```python
# Conceptual.
# Original training: epoch 4 is best (val_loss=0.10), patience=5,
#                    epochs 5..9 all worse, EarlyStopping fires at 9.
# Save checkpoint at epoch 4.
trainer = create_trainer(cfg)
trainer.load_checkpoint("checkpoint_epoch_4.pt")  # model + optimizer at epoch 4
trainer.train()                                   # cfg.epochs=10
# Observed:
#   - EarlyStopping._state.wait_count restarts at 0 → trains all 10 epochs
#   - ModelCheckpoint._best_value reset to inf → saves a "best.pt" on the
#     very first new epoch (which is post-resume epoch 0, but model is
#     already at epoch 4's weights so val_metric is similar to original
#     best — _is_better fires → overwrites best.pt)
#   - MetricLogger._history = [] → training_history.json has only 10 rows
#     starting from "post-resume epoch 0"
```

**Empirical evidence**: callback `on_train_start` methods all begin with explicit reset assignments (verified above). The `Callback` base class definition has no `state_dict`/`load_state_dict` signature. The checkpoint dict assembled at `trainer.py:989-1004` does not iterate `self.callbacks`.

**Production impact**:
- Configs affected: same as N2 — any `--resume` workflow.
- Observable manifestation: best.pt silently corrupted; training_history.json silently truncated; over-training past previous early-stop point.
- Cross-reference: no `EXPERIMENT_INDEX.md` entry currently records a resumed run; impact is dormant-pending-first-resume but the gap is structural.

**Root cause analysis**:

`Callback` was designed as a stateless interface (event hooks fire, state is incidental). Reset-on-train-start is the simplest "deterministic boot" pattern — but it conflicts with the "save/load state" pattern needed for resume. The two patterns have to be reconciled via an explicit `state_dict`/`load_state_dict` interface; that interface was deferred when callbacks shipped.

**Suggested fix scope (NOT EXECUTED — for future planning)**:
- Minimal patch: add `state_dict() -> Dict` and `load_state_dict(state: Dict) -> None` methods on `Callback` base class with default empty/no-op. Override in `EarlyStopping`/`ModelCheckpoint`/`MetricLogger` to capture/restore their state. Update `save_checkpoint` to call `callbacks_state = {cb.__class__.__name__: cb.state_dict() for cb in self.callbacks._callbacks}` and store under key `"callbacks"`. Update `load_checkpoint` to restore. Update `on_train_start` in each callback to skip reset when `self._loaded_from_checkpoint = True`.
- Principled fix: introduce a `Resumable` ABC in `lobtrainer.training.callbacks` that all stateful callbacks implement; the trainer's checkpoint-save/load enumerates Resumable callbacks explicitly, with versioning. Add `test_resume_equals_fresh_training_with_callbacks` golden test that asserts identical `_history` after `train(epochs=10)` vs `train(epochs=4) → save → load → train(epochs=10)`.
- Architectural retirement: see N2 — adopt Lightning-style explicit resume API.

---

### N4. HMHP-R primary metrics hardcode `horizons[0]` regardless of `primary_horizon_idx` configuration

**Severity**: CRITICAL — ACTIVE
**Status**: CONFIRMED (prior audit + validation)
**Layer**: Training Loop / Model Architecture
**First reported**: Round 5 forensic 2026-04-25 (sister bug to Phase A.5.4 #6)

**Files and lines**:
- `lob-model-trainer/src/lobtrainer/training/strategies/hmhp_regression.py:157` — `primary_horizon = horizons[0]` in `validate()`.
- `lob-model-trainer/src/lobtrainer/training/strategies/hmhp_regression.py:217-223` — primary metrics call.
- `lob-model-trainer/src/lobtrainer/training/strategies/hmhp_regression.py:254` — `primary_horizon = horizons[0]` in `evaluate()`.
- `lob-model-trainer/src/lobtrainer/training/strategies/hmhp_regression.py:294-298` — primary metrics call without prefix.
- `lob-model-trainer/configs/experiments/nvda_hmhp_regressor_h60.yaml` — `horizon_idx: 1` + `_base: hmhp_cascade_regression` (`model_type: hmhp_regression`). **TRUE HMHP-R + non-zero primary horizon: this IS the active N4 trigger.**
- `lob-model-trainer/configs/experiments/nvda_hmhp_40feat_h60_profit8bps_regression.yaml:42` — `horizon_idx: 1` BUT `_base: hmhp_cascade_bare` (`model_type: hmhp`, NOT `hmhp_regression`). Despite the misleading `_regression` suffix in the filename, this config instantiates `HMHPClassificationStrategy`, NOT `HMHPRegressionStrategy`. N4 does NOT fire for this config (validated via direct YAML inspection 2026-04-27).

**What the code actually does**:

`hmhp_regression.py:155-157` (validate method):

```python
model.eval()
horizons = self.horizons
primary_horizon = horizons[0]
```

`hmhp_regression.py:216-223`:

```python
        # Primary horizon metrics (without horizon prefix for early-stopping)
        if horizon_preds[primary_horizon] and horizon_targets[primary_horizon]:
            y_pred = np.concatenate(horizon_preds[primary_horizon]).ravel()
            y_true = np.concatenate(horizon_targets[primary_horizon]).ravel()
            primary_metrics = compute_all_regression_metrics(
                y_true, y_pred, prefix="val_"
            )
            result.update(primary_metrics)
```

`hmhp_regression.py:252-254` (evaluate method):

```python
        model.eval()
        horizons = self.horizons
        primary_horizon = horizons[0]
```

`hmhp_regression.py:293-298`:

```python
                if h == primary_horizon:
                    primary_metrics = compute_all_regression_metrics(
                        y_true, y_pred, prefix=""
                    )
                    results.update(primary_metrics)
```

Trace: `self.horizons` is set from `self.config.model.hmhp_horizons` in the strategy's `__init__`. `hmhp_horizons` is a list like `[10, 60, 300]` in insertion order — practically always ascending, so `horizons[0]` is the smallest horizon (H10). The intended "primary horizon" — the one that early-stopping monitors via the unprefixed `val_ic`/`val_r2`/etc. — is taken from `horizons[0]` regardless of what `data.horizon_idx` (or its A.5.4-canonical successor `data.labels.primary_horizon_idx`) says.

The per-horizon-prefixed metrics ARE correctly computed for every horizon (loops `for h in horizons:` at lines 213-214 and 226-233). Only the unprefixed primary block is wrong.

**Why this is a bug**:

`LabelsConfig.primary_horizon_idx` is the operator-set index that selects which horizon the experiment's primary metric describes. The exporter (`exporter.py:674-680`, post-Phase-A.5.4) correctly slices `pr[:, stats_idx]` with `stats_idx = labels_cfg.validate_primary_horizon_idx_for(n_horizons)`. The HMHP-R training-time strategy bypasses this and hardcodes index 0.

For the manifest `nvda_hmhp_regressor_h60.yaml` (TRUE HMHP-R trigger; `_base: hmhp_cascade_regression`, `model_type: hmhp_regression`, `horizon_idx: 1`):
- `horizons = [10, 60, 300]`
- Intended primary: H60 (index 1)
- `primary_horizon = horizons[0] = 10`
- Reported `val_ic`, `val_r2`, `val_mae`, `val_directional_accuracy` describe H10 quality
- EarlyStopping's `monitor='val_ic'` callback fires on H10 metric trajectory, NOT H60
- The `primary_horizon` reference in the exporter's post-train metadata IS correctly set to H60 (the exporter uses canonical `primary_horizon_idx`), so the SIGNAL files describe H60 but the CHECKPOINT was selected by H10 quality — silent train-vs-export divergence

Per hft-rules §1 ("No hardcoded indices"; "Use a centralized constants module and reference by semantic name"): `horizons[0]` is a hardcoded index. Phase A.5.4 introduced `LabelsConfig.validate_primary_horizon_idx_for(n_horizons)` precisely to retire this pattern in 4 exporter sites + the callback site, but the strategies were missed in that scope.

**Reproducer**:

```python
# Conceptual reproduction:
from lobtrainer.config import load_config
from lobtrainer.training.strategies.hmhp_regression import HMHPRegressionStrategy

cfg = load_config("configs/experiments/nvda_hmhp_regressor_h60.yaml")
print("Primary horizon idx (config):", cfg.data.labels.primary_horizon_idx)
print("HMHP horizons:", cfg.model.hmhp_horizons)
# -> primary_horizon_idx=1, hmhp_horizons=[10, 60, 300]

strategy = HMHPRegressionStrategy(model=..., config=cfg, ...)
print("strategy.horizons[0]:", strategy.horizons[0])
# -> 10  # WRONG — should be 60
```

**Empirical evidence**: validate at `hmhp_regression.py:157` and `:254`. Both literally set `primary_horizon = horizons[0]`. There is no read of `self.config.data.labels.primary_horizon_idx` anywhere in `hmhp_regression.py`.

**Production impact**:
- Configs affected (verified 2026-04-27): `lob-model-trainer/configs/experiments/nvda_hmhp_regressor_h60.yaml` (TRUE HMHP-R + horizon_idx=1 — the immediate trigger). `nvda_hmhp_regression_h10_primary.yaml` (TRUE HMHP-R but horizon_idx=0 — inert under current setting; would activate if `horizon_idx > 0`). The misleadingly-named `nvda_hmhp_40feat_h60_profit8bps_regression.yaml` does NOT trigger N4 because it uses `model_type: hmhp` (auxiliary-regression-head classification), so `HMHPRegressionStrategy` is never instantiated. **N4 actively corrupts unprefixed val_ic/val_r2/val_mae for `nvda_hmhp_regressor_h60.yaml`** the next time it is trained; early-stopping monitors the H10-derived metric while the operator's intent is H60.
- Observable manifestation: silent metric mismatch between training-time `val_*` and export-time `metrics_dict`. Training selects best.pt by H10 quality; backtester evaluates H60. Comparable lag-of-detection as N6.
- Cross-reference: any `EXPERIMENT_INDEX.md` HMHP-R entry with `horizon_idx != 0`. The H60-primary manifest above is the immediate trigger.

**Root cause analysis**:

Phase A.5.4 (2026-04-25) explicitly refactored 4 sites in `exporter.py` and 1 site in `callback.py` to use `LabelsConfig.validate_primary_horizon_idx_for`. The PR description called out exporter slicing sites by name but did not enumerate the strategy modules. Reviewers checked the cited sites and missed sister sites in `lobtrainer.training.strategies.hmhp_regression`. Same root cause as Phase A bug #6 (which led to A.5.4) but in a different module — the `horizons[0]` pattern was inserted independently in HMHP-R's strategy when it was extracted from the monolith.

**Suggested fix scope (NOT EXECUTED — for future planning)**:
- Minimal patch (`hmhp_regression.py:157` and `:254`): replace `primary_horizon = horizons[0]` with:
  ```python
  labels_cfg = self.config.data.labels
  primary_idx = labels_cfg.validate_primary_horizon_idx_for(len(horizons))
  primary_horizon = horizons[primary_idx]
  ```
- Principled fix: extract a strategy-level helper `BaseStrategy._resolve_primary_horizon(horizons: List[int]) -> int` that all multi-horizon strategies call. Add a regression test parametrizing over `(primary_horizon_idx ∈ {0, 1, 2})` × `(horizons = [10, 60, 300])` asserting `validate()`'s `val_*` keys describe the intended horizon by comparing `val_ic` vs `val_h{H}_ic` for the chosen H.
- Architectural retirement: collapse the unprefixed-primary-metric pattern entirely; require all consumers (early-stopping, exporter, ledger) to read prefixed `val_h{primary_horizon}_*` keys. Eliminates the implicit "what is primary" coupling between strategy and consumer.

---

### N5. HMHP vs HMHP-R encoder pooling inconsistency — last-timestep vs mean-over-T (P0-9 STILL OPEN)

**Severity**: CRITICAL — ACTIVE
**Status**: CONFIRMED (prior audit + validation)
**Layer**: Model Architecture
**First reported**: lob-models 2026-04-20 validation report (P0-9); deferred to Phase I.B; re-confirmed Round 6 forensic

**Files and lines**:
- `lob-models/src/lobmodels/models/hmhp.py:276` — HMHP classification decoder pools last timestep.
- `lob-models/src/lobmodels/models/hmhp_regressor.py:111` — HMHP-R regression decoder pools mean over time.
- `lob-models/src/lobmodels/models/hmhp.py:475-499` — `SharedEncoder` consumed identically by both decoders.
- `HMHPConfig` — verified to NOT contain a `pool_mode` field (filesystem grep returned 0 hits).

**What the code actually does**:

`hmhp.py:272-289` (classification HorizonDecoder.forward):

```python
        B = shared_repr.size(0)
        
        # Pool temporal dimension (use last timestep)
        # Shape: [B, T', D] → [B, D]
        pooled = shared_repr[:, -1, :]
        
        # State fusion if receiving from previous horizon
        if self.config.receive_prev_state and prev_state is not None:
            if self.config.state_fusion == StateFusion.GATE:
                fused = self.state_fusion(pooled, prev_state)
            elif self.config.state_fusion == StateFusion.CONCAT:
                fused = self.state_fusion_proj(torch.cat([pooled, prev_state], dim=-1))
            elif self.config.state_fusion == StateFusion.ADD:
                fused = pooled + self.state_proj(prev_state)
```

`hmhp_regressor.py:108-119` (regression decoder.forward):

```python
        B = shared_repr.size(0)
        pooled = shared_repr.mean(dim=1) if shared_repr.dim() == 3 else shared_repr

        if prev_state is not None and self.config.receive_prev_state:
            if self.config.state_fusion == StateFusion.GATE:
                pooled = self.state_fusion(pooled, prev_state)
            elif self.config.state_fusion == StateFusion.CONCAT:
                pooled = self.state_fusion_proj(torch.cat([pooled, prev_state], dim=-1))
            elif self.config.state_fusion == StateFusion.ADD:
                pooled = pooled + self.state_proj(prev_state)
```

Both classes consume `SharedEncoder.forward(...) -> Tensor[B, T', D]` from the same encoder definition (`hmhp.py:475-499`). The pooling step is the FIRST operation in each decoder's forward and is the difference between the two architectures.

**Why this is a bug**:

These two pooling operations are mathematically distinct, NOT minor variants:
- `shared_repr[:, -1, :]` returns the encoder's representation at the LAST timestep — emphasizes recent state, useful when the most recent events carry the most predictive information (TLOB-style).
- `shared_repr.mean(dim=1)` averages across the entire window T — emphasizes the smoothed signal across the lookback, useful when long-range patterns matter (deep-learning-friendly RNN-style).

For a `[B=2, T=100, D=64]` encoder output with deterministic Gaussian seeding, the cosine similarity between the two pools is approximately 0.18 (verified by prior validation runs cited in the lob-models 2026-04-20 P0-9 finding). They are NOT approximately the same vector. The downstream `mlp(pooled)` therefore receives two different "summary" representations.

This means the documented HMHP H10 acc=59.62% vs HMHP-R H10 R²=0.454 comparison (CLAUDE.md §Model Inventory) confounds:
1. Loss function (cross-entropy vs Huber) — variable A
2. Decoder head type (classifier+confidence vs regressor+uncertainty) — variable B
3. **Temporal aggregation (last vs mean)** — variable C — the most architecturally significant of the three

Per hft-rules §1 ("Inter-Module Contracts"; "every module must have a clearly defined interface for how it communicates with other modules") and §6 ("Tests document behavior and expose implementation correctness"): these two decoders share `SharedEncoder` but the architectural choice of pooling is hardcoded inconsistently between them with no `HMHPConfig.pool_mode` field to control it.

**Reproducer**:

```python
import torch

torch.manual_seed(42)
shared_repr = torch.randn(2, 100, 64)

last_pool = shared_repr[:, -1, :]
mean_pool = shared_repr.mean(dim=1)

print("last_pool stats:", last_pool.mean().item(), last_pool.std().item())
print("mean_pool stats:", mean_pool.mean().item(), mean_pool.std().item())

cos = torch.nn.functional.cosine_similarity(
    last_pool.flatten().unsqueeze(0), mean_pool.flatten().unsqueeze(0)
)
print("cosine similarity:", cos.item())
print("torch.allclose:", torch.allclose(last_pool, mean_pool, atol=1e-1))
```

**Empirical evidence**: For Gaussian input, `mean_pool.std()` ≈ 1/sqrt(T) ≈ 0.10 (variance is averaged), while `last_pool.std()` ≈ 1.00 (single sample variance). These are mathematically incomparable summary vectors. Cosine similarity ≈ 0.18 confirms they are not approximately the same direction in [0, 1].

**Production impact**:
- Configs affected: every HMHP and HMHP-R config. Direct ablation comparisons (HMHP_classification.yaml vs HMHP_regression.yaml) are confounded.
- Observable manifestation: the documented finding "HMHP H10 acc=59.62%, HMHP-R R²=0.454" — the gap between the two should not be attributed solely to loss/head differences. A future "switch HMHP-R to last-timestep pool" experiment may reveal the regression head is actually competitive with `pool=last`.
- Cross-reference: `EXPERIMENT_INDEX.md` HMHP H10 entries; `CLAUDE.md` §Model Inventory table comparing the two.

**Root cause analysis**:

`HMHP` was authored first with last-timestep pooling (typical for transformer classification). `HMHP-R` was implemented later as a regression variant — the author chose mean-pooling without explicit justification (likely a default-pattern intuition: "regression on continuous targets averages noise"). No `pool_mode` field was added to `HMHPConfig` to surface the choice. The lob-models 2026-04-20 validation flagged this as P0-9 + Phase I.B was deferred per cycle scope discipline. The deferral persisted because no production manifest forced the issue.

**Suggested fix scope (NOT EXECUTED — for future planning)**:
- Minimal patch: add `pool_mode: Literal["last", "mean"] = "last"` to `HMHPConfig`. Both decoders read `self.config.pool_mode`. Default = `"last"` to preserve HMHP classification behavior. Add a one-line release note that HMHP-R default behavior changes.
- Principled fix: introduce a `lobmodels.layers.TemporalPool` module with pluggable strategies (`last`, `mean`, `attention`, `cls_token`). Both decoders consume `self.pool = TemporalPool.from_config(config.pool_mode)`. Add regression test against the documented HMHP H10 metric to guarantee `pool_mode=last` reproduces the existing acc=59.62% number.
- Architectural retirement: merge HMHP and HMHP-R into a single class with a `task: Literal["classification", "regression"]` field; the head is task-dispatched but the encoder + pooling + state-fusion are unified. Eliminates the bug class structurally.

---

### N6. Calibrated experiments report MAE/RMSE/R² for RAW predictions, not the calibrated array the backtester actually trades

**Severity**: CRITICAL — ACTIVE
**Status**: CONFIRMED (prior audit + validation)
**Layer**: Calibration & Export
**First reported**: Round 5 forensic 2026-04-25, re-validated Round 7

**Files and lines**:
- `lob-model-trainer/src/lobtrainer/export/exporter.py:652-694` — `_build_metadata` computes `metrics_dict`.
- `lob-model-trainer/src/lobtrainer/export/exporter.py:653` — `pr = inference["predicted_returns"]` (raw).
- `lob-model-trainer/src/lobtrainer/export/exporter.py:692-694` — `compute_all_regression_metrics(rl[:, stats_idx], pr[:, stats_idx])` uses RAW.
- `lob-model-trainer/src/lobtrainer/calibration/variance.py:1-21` — `calibrated = (pred - pred_mean) * (target_std / pred_std) + target_mean`.
- `lob-backtester/src/lobbacktest/engine/vectorized.py:180-184` — backtester loads `calibrated_returns.npy` when manifest declares calibration.

**What the code actually does**:

`exporter.py:652-694`:

```python
        if "predicted_returns" in inference:
            pr = inference["predicted_returns"]
            # ... prediction_stats slicing ...
            labels_cfg_stats = resolve_labels_config(config)
            if pr.ndim == 1:
                prediction_stats = {
                    "mean": float(np.mean(pr)),
                    "std": float(np.std(pr)),
                    "min": float(np.min(pr)),
                    "max": float(np.max(pr)),
                }
            else:
                stats_idx = labels_cfg_stats.validate_primary_horizon_idx_for(
                    n_horizons=pr.shape[-1]
                )
                prediction_stats = {
                    "mean": float(np.mean(pr[:, stats_idx])),
                    "std": float(np.std(pr[:, stats_idx])),
                }

            if "regression_labels" in inference:
                rl = inference["regression_labels"]
                try:
                    from lobtrainer.training.regression_metrics import (
                        compute_all_regression_metrics,
                    )
                    if pr.ndim == 1:
                        metrics_dict = compute_all_regression_metrics(rl, pr)
                    else:
                        # stats_idx already validated above for pr.ndim==2 branch.
                        metrics_dict = compute_all_regression_metrics(
                            rl[:, stats_idx], pr[:, stats_idx]
                        )
                except Exception as e:
                    logger.warning(f"Could not compute regression metrics: {e}")
```

`vectorized.py:180-184` (backtester):

```python
        manifest_says_calibrated = (
            manifest is not None and manifest.calibration_method is not None
        )
        if manifest_says_calibrated and (d / "calibrated_returns.npy").exists():
            predicted_returns = np.load(d / "calibrated_returns.npy")
```

Trace: the exporter's `_build_metadata` at line 653 reads `pr = inference["predicted_returns"]` — which is the RAW model output. The `calibration_result` (computed earlier in the export flow) is held separately — `calibration_result["calibrated"]` is the variance-matched array. The metrics computation at line 689/692 always uses `pr`, never `calibration_result["calibrated"]`. The manifest's `metrics` field thus describes the RAW predictions. Meanwhile the backtester (vectorized.py:184) loads `calibrated_returns.npy` when `manifest.calibration_method is not None` — so the manifest's metrics describe a different array than the backtester trades.

**Why this is a bug**:

Variance-matching calibration has the form `c = (p - μ_p) · σ_t/σ_p + μ_t` (linear). This affects metric subsets differently:

- **IC (Spearman rank)**: linear monotone transform preserves rank → IC unchanged ✓
- **Pearson r**: linear transform of one variable preserves correlation → r unchanged ✓
- **DA (sign of pred)**: shifts when the additive constant `μ_t - μ_p · σ_t/σ_p` flips signs across the zero threshold; for zero-mean inputs (the typical case post-Huber training) sign is preserved → near-unchanged ✓
- **MAE = (1/N) Σ |c - y| vs (1/N) Σ |p - y|**: differs by data-dependent amounts. The relationship depends on prediction quality, label kurtosis, and the calibration scale factor — NOT a clean linear scaling. A previous version of this entry incorrectly stated `MAE_calib ≈ scale_factor × MAE_raw` (with worked example "5 → 18.6 bps for E5 scale=3.73x"); that algebraic identity does NOT hold in general. The accurate framing: MAE/RMSE/R² differ between raw and calibrated by amounts that must be computed empirically per dataset; only the rank-based metrics (IC, Pearson r, DA) are calibration-invariant. The bug — that the manifest's metrics block describes a different array than the backtester trades — is real regardless of the precise magnitude of the difference.
- **RMSE = sqrt((1/N) Σ (c - y)²)**: differs by data-dependent amounts. Under variance-matching specifically, the closed form is approximately `RMSE_cal² ≈ 2 σ_y² (1 − r_pearson)` where `σ_y` is the label std and `r_pearson` is the Pearson correlation between calibrated predictions and labels. This is NOT `scale_factor × RMSE_raw`.
- **R² = 1 - SS_res/SS_tot**: SS_res changes asymmetrically under variance-matching because `cov(c, y) = cov(scaled p, y) = scale · cov(p, y)`, but `var(c) = scale² · var(p) = var(y)` post-matching. The closed form for R²_calib differs from R²_raw by an algebraic transformation that depends on the original Pearson r and the scale factor.

Per hft-rules §11 ("Documentation must reflect the current codebase behavior exactly") and §13 ("Independent metric validation is mandatory"): the manifest's metrics block claims to describe the signal that downstream consumers will trade, but it describes a different array.

**Reproducer**:

```python
import json
import numpy as np
from pathlib import Path

# Conceptual — read any E6 calibrated signal directory
sig_dir = Path("data/exports/.../e6_*/test")  # adjust path
meta = json.loads((sig_dir / "signal_metadata.json").read_text())
print("Manifest calibration_method:", meta["calibration_method"])
print("Manifest metrics MAE:", meta["metrics"]["mae"])

cal = np.load(sig_dir / "calibrated_returns.npy")
labels = np.load(sig_dir / "regression_labels.npy")
mae_actual = np.mean(np.abs(cal - labels[:, 0]))  # primary horizon
print("Recomputed MAE on calibrated:", mae_actual)
# Expect: meta["metrics"]["mae"] (raw) DIFFERS from mae_actual (calibrated)
#         by data-dependent amount (not a simple scale-factor multiplier)
#         only IC, Pearson r, DA are calibration-invariant
```

**Empirical evidence**:
- The variance-match formula at `variance.py:7-13` documents `pred_std = 7.35 bps, target_std = 27.41 bps, scale_factor = 3.73` (E5 reference values). The variance-match transform `c = (p - μ_p)·(σ_t/σ_p) + μ_t` linearly scales predictions, but its effect on MAE depends on the joint distribution of predictions and labels — NOT a clean factor-of-3.73 multiplier. The correct empirical procedure is: read both `predicted_returns.npy` and `calibrated_returns.npy` from a real signal directory, compute MAE against `regression_labels.npy`, and observe the actual difference (which will not match `scale_factor × MAE_raw`).
- `exporter.py:653` literally takes `pr = inference["predicted_returns"]` and never reads `calibration_result["calibrated"]` for metric computation.
- IC and DA are correctly described because rank-based metrics are invariant under linear monotone transforms.

**Production impact**:
- Configs affected: E6, R8 (Round 8), and any future calibrated experiment (any manifest with `--calibrate variance_match` or `output.calibrate: variance_match`). Note: a prior version cited 'R9' but BACKTEST_INDEX.md only has Rounds 1-8 — R9 reference removed in REV 2 E7.
- Observable manifestation: `EXPERIMENT_INDEX.md` cells display the MAE/RMSE/R² of the RAW predicted_returns array, while the backtester trades the calibrated_returns array. The two arrays produce different magnitude metrics (the difference is data-dependent and must be measured empirically per experiment, NOT computed from a closed-form scale-factor mapping). Operators making decisions based on manifest MAE/RMSE/R² are misled about the trade signal's magnitude error. Rank-based metrics (IC, Pearson r, DA) are correct because they are calibration-invariant.
- Cross-reference: `EXPERIMENT_INDEX.md` E6 row ("E6 TLOB calibrated, R²=0.124"). The R² describes raw predictions, not what the backtester evaluates.

**Root cause analysis**:

The variance-matching calibration was introduced as a quick patch to address E5's Huber-induced conservatism (predictions too small in magnitude). The implementation correctly emits `calibrated_returns.npy` AND sets `manifest.calibration_method` AND wires the backtester to consume the calibrated file. But the metric computation step in `_build_metadata` was not updated to match — the author assumed (incorrectly) that variance-matching is calibration-invariant for ALL metrics so the manifest's metrics block could continue to describe raw predictions without consequence. The truth: IC, Pearson r, and DA are calibration-invariant under linear monotone transforms (rank or linear are preserved); MAE, RMSE, and R² are NOT (their numeric values change because they describe magnitude/dispersion, which calibration alters).

**Suggested fix scope (NOT EXECUTED — for future planning)**:
- Minimal patch (`exporter.py:653`): when `calibration_result is not None`, set `pr_for_metrics = calibration_result["calibrated"]` and use it for the `compute_all_regression_metrics` call. Keep `pr` for `prediction_stats` (which is descriptive of the model's raw output distribution by design).
- Principled fix: emit BOTH `metrics_raw` AND `metrics_calibrated` in the manifest, with explicit naming. Add a docstring + validator that prevents future drift. The manifest's primary `metrics` becomes a pointer to `metrics_calibrated` when calibration is applied; consumers reading `manifest["metrics"]` get the array-aligned values.
- Architectural retirement: bind `metrics` to `calibration_result["calibrated"]` (or raw if not calibrated) at the SignalManifest contract level — `SignalManifest.validate()` cross-checks `metrics["mae"]` against a recompute on the actual on-disk array. Drift caught at load time.

---

### N7. Normalization stats NOT bound to checkpoint — re-export silently invalidates every prior checkpoint

**Severity**: CRITICAL — DORMANT-PRIMED
**Status**: CONFIRMED (prior audit + validation)
**Layer**: Data Pipeline / Calibration & Export
**First reported**: Round 5 forensic 2026-04-25

**Files and lines**:
- `lob-model-trainer/src/lobtrainer/training/trainer.py:583-608` — `Trainer.setup` reads `<data_dir>/normalization_stats.json` cached by extractor.
- `lob-model-trainer/src/lobtrainer/training/trainer.py:982-1004` — `save_checkpoint` does NOT include stats reference.
- `lob-model-trainer/src/lobtrainer/training/trainer.py:1015-1033` — `load_checkpoint` does NOT verify stats hash.

**What the code actually does**:

`trainer.py:583-609`:

```python
        if norm_strategy == NormalizationStrategy.GLOBAL_ZSCORE:
            # Global Z-score matching TLOB repository
            num_features = self.config.data.feature_count
            stats_path = Path(data_dir) / "normalization_stats.json"
            
            if stats_path.exists():
                # Fast path: load cached stats
                from lobtrainer.data.normalization import GlobalNormalizationStats
                logger.info(f"Loading cached normalization stats from {stats_path}")
                stats = GlobalNormalizationStats.load(stats_path)
                feature_transform = GlobalZScoreNormalizer(
                    stats, 
                    eps=self.config.data.normalization.eps
                )
            else:
                # Slow path: compute stats (uses streaming internally)
                logger.info(f"Computing GlobalZScoreNormalizer stats for {num_features} features...")
                # Load training data with lazy loading for stats computation
                train_days_for_stats = load_split_data(
                    data_dir, "train",
                    labels_config=self.config.data.labels,
                    validate=False, lazy=True,
                )
                feature_transform = GlobalZScoreNormalizer.from_train_data(
                    train_days_for_stats,
                    num_features=num_features,
                    eps=self.config.data.normalization.eps,
                )
                feature_transform.stats.save(stats_path)
                logger.info(f"Saved normalization stats to {stats_path}")
                del train_days_for_stats  # Free memory
```

`trainer.py:989-998` (save_checkpoint, see N3):

```python
checkpoint = {
    'epoch': self.state.current_epoch,
    'global_step': self.state.global_step,
    'model_state_dict': self.model.state_dict(),
    'optimizer_state_dict': self.optimizer.state_dict(),
    'config': self.config.to_dict(),
    'state': {
        'best_val_metric': self.state.best_val_metric,
        'best_epoch': self.state.best_epoch,
    },
}
```

Trace: stats are persisted at `<data_dir>/normalization_stats.json` (sibling to the train/val/test NPYs in the EXPORT directory). The checkpoint dict has no `normalization_stats_path` field, no `normalization_stats_sha256` field. `Trainer.setup()` runs BEFORE any `load_checkpoint()` call (the load_checkpoint method does not call setup) and reads from `data_dir`. If a re-export overwrote `normalization_stats.json` between training-time and inference-time, `Trainer.setup` would silently load the new stats, and the checkpoint's frozen weights would be applied to inputs normalized with the wrong (μ, σ).

**Why this is a bug**:

A neural network checkpoint is a pair (weights, input distribution assumptions). Variance-shifted inputs cause the same weights to produce systematically wrong outputs. For typical normalization stats with `σ` differences of order 10-30% across re-exports (driven by changes in dataset day mix, e.g., adding new days), the resulting prediction distribution shift can completely invalidate model confidence.

Per hft-rules §9 ("Normalization boundaries: Statistics (mean, std) must be computed from training data exclusively. Save stats with checkpoints for inference reproducibility. Apply identically to train/val/test."): the SECOND sentence is the violated invariant. Stats must accompany the checkpoint, not live in `data_dir`.

The bug is currently DORMANT-PRIMED because no re-export has happened in 35+ days (per Frame 5 audit). The first re-export will silently corrupt every prior checkpoint. There is no enforcement preventing this.

**Reproducer**:

```python
# Conceptual.
# 1. Train with data_dir=data/exports/v1/. trainer.setup() computes
#    normalization_stats_v1.json (or loads cached).
# 2. Save checkpoint at epoch 4.
# 3. Re-export with new days: data/exports/v1/ now has
#    normalization_stats_v1.json overwritten with v2 stats.
# 4. Run signal export with checkpoint_v1 (Trainer.setup re-reads stats):
trainer = create_trainer(cfg)  # cfg.data.data_dir = data/exports/v1/
trainer.setup()                # Loads stats_v2 (silently)
trainer.load_checkpoint("checkpoint_v1.pt")  # Loads weights (intended for stats_v1)
# Inference applies (input - μ_v2)/σ_v2 to inputs that the model was
# trained on (input - μ_v1)/σ_v1. Distribution mismatch is silent.
```

**Empirical evidence**:
- Read of `save_checkpoint` (`trainer.py:989-1004`): no `normalization_stats` key, no path reference, no SHA-256.
- `Trainer.setup()` reads `stats_path` from `data_dir` unconditionally; no consistency check against checkpoint.
- The risk surface widens every time a Phase X re-export happens. With recent V.A.4 + Phase 4 work pointing toward more frequent feature-set evolutions, re-exports will become routine.

**Production impact**:
- Configs affected: every checkpoint produced by the trainer to date — silently invalidated on next re-export.
- Observable manifestation: signal_metadata.json and best.pt continue to load without error. Metrics computed at export time use the wrong-normalization signal but with the wrong-distribution model. The backtester trades a corrupt signal. There is NO audit trail in the manifest beyond the bare `data_dir` path.
- Cross-reference: prevention is bound up with Phase 4 FeatureSet Registry and the V.A.4 `compatibility_fingerprint` work — fingerprint-based rejection would catch this if stats were content-addressed.

**Root cause analysis**:

The stats live at `data_dir/normalization_stats.json` because the export pipeline produces them. This optimizes for "pre-compute once per dataset, reuse across many trainings" — efficient but unsafe under re-export. The trainer's separation between `data_dir` and `output_dir` is sound on paper (data is read-only, output is per-experiment), but the assumption that `data_dir` is immutable is not enforced. No one has re-exported during a checkpoint's lifetime yet, so no one has noticed.

**Suggested fix scope (NOT EXECUTED — for future planning)**:
- Minimal patch: in `save_checkpoint`, copy `normalization_stats.json` into `output_dir/checkpoints/` as `normalization_stats_at_train.json`. Compute its SHA-256 and store as `checkpoint["normalization_stats_sha256"]`. In `load_checkpoint`, recompute SHA-256 of current `data_dir/normalization_stats.json`; raise `ContractError` on mismatch with an actionable message ("Re-export detected — checkpoint was trained against stats_v1; current stats are v2. Use the snapshot at <output_dir>/checkpoints/normalization_stats_at_train.json by setting data.normalization.stats_override_path.").
- Principled fix: content-address `normalization_stats.json` as a Class A SSoT artifact (Phase 4 pattern). Stats live at `data/exports/_normalization_cache/<sha256>.json` and `data_dir` carries a pointer (`normalization_stats_ref.json` with `{sha256: ..., path: ..., size: ...}`). The exporter populates the cache; checkpoints reference the SHA-256 directly. Re-export creates a new cache entry rather than overwriting. The `compatibility_fingerprint` (V.A.4) extension naturally absorbs this into the trust column.
- Architectural retirement: eliminate per-day cached stats; compute stats on-the-fly from a content-addressed hash of train days at load time. Strict but most reproducible.

---

### N8. TLOB final-flatten ordering differs from official Berti & Kasneci 2025 reference — pretrained checkpoints unloadable

**Severity**: CRITICAL — DORMANT-PRIMED
**Status**: CONFIRMED (prior audit + validation)
**Layer**: Model Architecture
**First reported**: Round 6 forensic 2026-04-25

**Files and lines**:
- `lob-models/src/lobmodels/models/tlob.py:226-235` — our final-flatten.
- `TLOB/models/tlob.py:117-118` — reference implementation final-flatten.

**What the code actually does**:

Our `tlob.py:226-235`:

```python
        # --- TLOB Blocks ---
        for layer in self.layers:
            x, _ = layer(x, need_weights=False)
            # Permute after each layer (temporal ↔ feature attention)
            x = x.permute(0, 2, 1)

        # --- Task Head ---
        # After the last permute, shape is [B, D', T'] where D' and T' are reduced
        # We need to flatten
        x = x.reshape(x.shape[0], -1)  # [B, flattened_dim]
```

Reference `TLOB/models/tlob.py` (lines around 117-118 in the snippet captured from disk):

```python
        for i in range(len(self.layers)):
            x, att = self.layers[i](x)
            att = att.detach()
            x = x.permute(0, 2, 1)
        x = rearrange(x, 'b s f -> b (f s) 1')              
        x = x.reshape(x.shape[0], -1)
```

Trace: after the final transformer block + final `permute(0, 2, 1)`, the tensor is in shape `[B, X, Y]` (where X and Y depend on the permutation cadence). Both implementations get to the same intermediate shape after the same number of permutes. The difference is in the final flatten:

- **Ours**: `x.reshape(x.shape[0], -1)` — flattens C-order (PyTorch row-major), yielding `[X, Y, X·Y...]` per batch. The element at flat index `k = i·Y + j` is `x[i, j]`.
- **Reference**: `rearrange(x, 'b s f -> b (f s) 1')` — equivalent to `x.permute(0, 2, 1).reshape(B, -1, 1)` — first transposes the last two dims so X becomes inner, THEN flattens. The element at flat index `k = j·X + i` is `x[i, j]`.

These produce DIFFERENT linear orderings of the same elements. Subsequent `final_layers[0]: nn.Linear(in_features=X·Y, out_features=...)` learns weight rows over an ordering of features that differs between the two implementations.

**Why this is a bug**:

For a `[B=1, S=3, F=4]` tensor with values `[[0,1,2,3], [4,5,6,7], [8,9,10,11]]`:
- Ours: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]` (S outer, F inner)
- Reference: `[0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]` (F outer, S inner)
- `torch.equal(...)` = `False`

Implications:
1. **Cross-codebase checkpoint loading**: BROKEN. A pretrained Berti & Kasneci 2025 checkpoint cannot be loaded into our `TLOB`. The state_dict's `final_layers.0.weight` has rows organized differently. A naive load_state_dict succeeds (shapes match) but produces semantically wrong predictions.
2. **Trained-from-scratch**: NOT broken in isolation. The Linear head learns whichever ordering it sees during training; predictions are self-consistent.
3. **Reference reproduction claim**: root `CLAUDE.md` cites Berti & Kasneci 2025 as the architectural reference. Our deviation is undocumented; researchers comparing our results against their published numbers cannot verify architectural identity.

Per hft-rules §0 ("When implementing a published method, the official repository (if available) is the primary reference. Papers often omit edge cases that working code handles. Document any intentional deviations from reference implementations with explicit justification.") — this is an undocumented deviation.

**Reproducer**:

```python
import torch
from einops import rearrange

torch.manual_seed(42)
x = torch.tensor([
    [[0, 1, 2, 3],
     [4, 5, 6, 7],
     [8, 9, 10, 11]]
], dtype=torch.float32)  # [B=1, S=3, F=4]

ours = x.reshape(x.shape[0], -1)
ref = rearrange(x, 'b s f -> b (f s) 1').reshape(x.shape[0], -1)

print("Ours:", ours.tolist())  # [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
print("Ref :", ref.tolist())   # [[0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]]
print("Equal:", torch.equal(ours, ref))  # False
```

**Empirical evidence**: the snippet-comparison above demonstrates the divergence is bit-exact for any input. The reference uses `einops.rearrange` to make the intent explicit (`s` outer, `f` inner); ours uses raw `reshape` whose ordering depends on tensor stride.

**Production impact**:
- Configs affected: every TLOB config (`configs/bases/models/tlob_*.yaml`) and every TLOB experiment (E5, E6, R8, R9, etc.). All such checkpoints currently in production are trained-from-scratch on our ordering, so they are self-consistent — but they cannot be cross-loaded with the official TLOB pretrained weights from FI-2010 reproduction.
- Observable manifestation: silent. The model trains and exports normally on our ordering. Comparison with published Berti & Kasneci numbers requires both implementations to use the same ordering — currently they don't.
- Cross-reference: `EXPERIMENT_INDEX.md` TLOB entries; `CLAUDE.md` §Model Inventory citing TLOB R²=0.464 — the published comparison is on a different architecture in this load-bearing detail.

**Root cause analysis**:

The TLOB implementation in our repo was authored before the third-party `TLOB/` reference was dropped at `/Users/knight/code_local/HFT-pipeline-v2/TLOB/`. At authoring time, the contributor matched the paper's textual description (which is ambiguous about flatten ordering) but did not verify against the official repo. Subsequent comparison reviews caught most architectural details (BiN, dual attention, head structure) but missed this one because both orderings produce a tensor of the right shape.

**Suggested fix scope (NOT EXECUTED — for future planning)**:
- Minimal patch (`tlob.py:235`): change `x = x.reshape(x.shape[0], -1)` to `x = x.permute(0, 2, 1).reshape(x.shape[0], -1)` (matches reference). Note: this is a SHAPE-PRESERVING + semantics-changing edit — every existing TLOB checkpoint trained against our current ordering would need to be retrained or have its `final_layers.0.weight` rows permuted in a one-time migration script.
- Principled fix: introduce `flatten_ordering: Literal["s_outer", "f_outer"] = "f_outer"` on `TLOBConfig` with default `f_outer` (reference). Provide a migration utility `lobmodels.tools.migrate_tlob_flatten` that loads an old checkpoint and permutes weights into the new ordering. Add a regression test that loads a saved fixture from the official TLOB repo and verifies forward-pass output bit-exactness.
- Architectural retirement: replace with explicit `einops.rearrange(x, 'b s f -> b (f s)')` mirroring the reference exactly — eliminates the ambiguity and makes the intent unmistakable to future readers.

---

---

## 5. DORMANT Findings — Real Bugs with No Production Trigger Today

This section documents bugs that have been verified by direct code inspection but cannot manifest under current production usage patterns. Each finding identifies the activation trigger — the specific change in usage, configuration, or data flow that would cause the bug to surface as silent corruption or crash. These are not false positives; they are real correctness gaps that defense-in-depth would close, but which the current call graph happens to route around.

### D1. DataConfig.labels staleness on `model_copy(update={"labeling_strategy": ..., "horizon_idx": ...})`

**Severity**: HIGH
**Status**: DORMANT — no in-process CLI flag or programmatic mutation path exists today for `labeling_strategy` / `horizon_idx`.
**Activation trigger**: any future CLI flag, sweep generator, or notebook-driven hyperparameter scan that calls `cfg.model_copy(update={"labeling_strategy": ...})` or `cfg.model_copy(update={"horizon_idx": ...})` on an already-validated `DataConfig`.

**Files and lines**:
- `lob-model-trainer/src/lobtrainer/config/schema.py:1272-1284` — `DataConfig._validate_all` auto-derive guard
- `lob-model-trainer/src/lobtrainer/config/base.py:329-374` — `SafeBaseModel.model_copy` override

**What the code actually does**:

```python
# schema.py:1272-1284
if self.labels is None:
    if self.labeling_strategy == LabelingStrategy.REGRESSION:
        derived_task = "regression"
    else:
        derived_task = "classification"
    derived = LabelsConfig(
        source="auto",
        task=derived_task,
        primary_horizon_idx=self.horizon_idx,
    )
    object.__setattr__(self, "labels", derived)

return self
```

```python
# base.py:370-374
if update:
    return self.__class__.model_validate(
        {**self.model_dump(), **update}
    )
return super().model_copy(deep=deep)
```

Trace: the validator at schema.py:1272 only re-derives `labels` when `self.labels is None`. After construction, `labels` is populated with the derived `LabelsConfig`. When `model_copy(update={"labeling_strategy": REGRESSION})` is called, `SafeBaseModel.model_copy` at base.py:370-372 calls `model_validate({**self.model_dump(), **update})` — `model_dump()` serializes the previously-derived `labels` field, so the merged dict has a non-None `labels`, the auto-derive branch at schema.py:1272 is skipped, and the new `labels.task` remains the prior `"classification"` value despite the user's intent to switch to regression.

**Why this is a bug**:

The `model_copy` override at base.py:329-374 was specifically added to close ship-blocker bug #1 ("Pydantic v2's default `model_copy` skips validators"). The override fires re-validation, but the auto-derive logic itself uses a pre-existing-non-None gate that wasn't updated to handle the override semantics. Result: `cfg.model_copy(update={...})` with one of the legacy DataConfig fields produces a config whose `labels` block is desynchronized from the intended state. Violates hft-rules §1 ("contracts are explicit and synchronized") and hft-rules §5 ("if a config option exists but is not fully supported, it must fail fast — never silently degrade").

**Why dormant today**:

A `grep -rn "model_copy(update=" hft-ops/src/ lob-model-trainer/scripts/` search shows zero call sites that pass `labeling_strategy` or `horizon_idx` keys. The hft-ops sweep grid expansion happens at the YAML dict level via `copy.deepcopy(manifest)` followed by setattr/dict-level overrides, then materializes the resolved YAML to a temp file at `hft-ops/src/hft_ops/stages/training.py:121-122` and `:208-209`, and the trainer subprocess receives an absolute path and constructs from a fresh dict via `from_yaml` — no `model_copy` ever runs. Similarly, no CLI flag in `lob-model-trainer/scripts/train.py:apply_overrides` exposes these two fields.

**Empirical reproducer**:

```python
from lobtrainer.config.schema import DataConfig, LabelingStrategy
cfg = DataConfig(labeling_strategy=LabelingStrategy.TLOB, horizon_idx=0)
print("Before:", cfg.labels.task, cfg.labels.primary_horizon_idx)
new = cfg.model_copy(update={
    "labeling_strategy": LabelingStrategy.REGRESSION,
    "horizon_idx": 2,
})
print("After:", new.labels.task, new.labels.primary_horizon_idx)
print("Operator intent:", new.labeling_strategy, new.horizon_idx)
```

Expected output:
```
Before: classification 0
After: classification 0    # STALE — should be (regression, 2)
Operator intent: LabelingStrategy.REGRESSION 2
```

**Production impact (if activated)**: silent training of a regression model on classification labels (or vice versa) with the wrong primary horizon. The label adapter would dispatch to the wrong path at `data/dataset.py:642-693`, producing garbage labels that pass shape checks. Discoverable only by inspecting `metrics["task"]` after-the-fact.

**Root cause analysis**: pre-existing-non-None gates are an anti-pattern under `model_copy(update=...)` semantics. The two-phase sequence (construction → mutation → re-derive) requires re-derivation whenever the source field changes, not whenever the derived field is None.

**Suggested fix scope**:
- Minimal patch: in `_validate_all`, check whether `self.labels` is "consistent with current `labeling_strategy` + `horizon_idx`"; re-derive on mismatch.
- Principled fix: deprecate the legacy flat `labeling_strategy`/`horizon_idx` fields and require operators to set `labels:` directly (Phase A.5.9 deferred this).
- Architectural retirement: schema-level `@model_validator(mode="after")` returning `self.model_copy(update={"labels": derived})` — but this requires breaking the recursive-revalidation cycle, which Phase A.5.3g explicitly avoided.

---

### D2. ModelConfig.params staleness on `model_copy(update={"tlob_hidden_dim": ...})`

**Severity**: HIGH
**Status**: DORMANT — no in-process sweep harness uses `model_copy(update=...)` on ModelConfig.
**Activation trigger**: future in-process sweep generator (e.g., `optuna` integration, notebook-driven hyperparameter scan) that mutates flat ModelConfig fields via `model_copy`.

**Files and lines**:
- `lob-model-trainer/src/lobtrainer/config/schema.py:1566-1567` — `ModelConfig._validate_all` `_build_params_from_legacy` guard

**What the code actually does**: same anti-pattern as D1 — a pre-existing-non-None check on `self.params` skips the build path on copy-update.

Trace: legacy flat fields (`tlob_hidden_dim`, `tlob_num_heads`, `tlob_num_layers`, `lstm_hidden_size`, `deeplob_conv_filters`, ~30 total) are designed to populate `self.params` dict via `_build_params_from_legacy` when no explicit `params` block is provided. After first construction, `params` is non-empty. `model_copy(update={"tlob_hidden_dim": 256})` re-validates from `model_dump()` which serializes the populated `params` dict; the build path is skipped; `params["hidden_dim"]` remains the pre-update value while the flat field shows the new value.

**Why this is a bug**: violates the "single source of truth" rule (hft-rules §1). The model factory at `lob-models/` reads from `config.params`, NOT from the flat fields. Net effect: every grid point trains the FIRST point's architecture while the flat fields lie about the actual hyperparameters.

**Why dormant today**: `grep -rln "ExperimentConfig" hft-ops/src/` returns nothing. hft-ops never instantiates ExperimentConfig in-process. All sweep grid expansion is dict-level prior to materialization. The trainer subprocess's `apply_overrides` at `lob-model-trainer/scripts/train.py:346` operates on a fresh-from-YAML config and uses Pydantic's standard `model_validate` for any CLI overrides (see schema.py CLI-keystone refactor in A.5.3i which switched FROM `model_copy(update=...)` TO `model_validate({**dump, **overrides})` as the official pattern — and that pattern, applied at process boundaries, side-steps this bug because the dict is freshly serialized + validated).

**Production impact (if activated)**: silent hyperparameter-frozen sweeps. A 27-cell `{hidden_dim: [64, 128, 256, 512], num_heads: [4, 8, 16]}` grid would train every cell with the FIRST cell's architecture while reports list the intended hyperparameters. The fingerprint at `hft-ops/src/hft_ops/ledger/dedup.py::compute_fingerprint` reads from the resolved dict (which contains the lying flat fields), so each cell would land in a separate ledger record despite identical actual training.

**Suggested fix scope**:
- Minimal patch: invalidate `self.params` to `{}` whenever any tracked flat field is in the `update` dict (requires `model_copy` override extension).
- Principled fix: deprecate the legacy flat fields; require explicit `params:` block (Phase A.5.9 deferred).

---

### D3. PrivateAttr cache lost on `model_copy(update={...})`

**Severity**: MEDIUM
**Status**: DORMANT — all production model_copy callers fire BEFORE Trainer.setup populates the cache.
**Activation trigger**: a future code path that calls `model_copy(update=...)` on a DataConfig AFTER `Trainer.setup` has run.

**Files and lines**:
- `lob-model-trainer/src/lobtrainer/config/base.py:370-374` — branch divergence

**What the code actually does**:

```python
if update:
    return self.__class__.model_validate(
        {**self.model_dump(), **update}
    )
return super().model_copy(deep=deep)
```

Trace: empty-update path delegates to Pydantic's `super().model_copy(deep=False)` which preserves `PrivateAttr` fields. Non-empty-update path calls `model_validate(model_dump())` — Pydantic's `model_dump()` excludes `PrivateAttr` by design (the R3 invariant in Phase 4 that gates feature-set caches OUT of the YAML round-trip surface). The reconstructed instance has the default `None` for PrivateAttr fields.

**Why this is a bug**: violates the R3 invariant's contract. Phase 4 documented "PrivateAttr mutation legal even under frozen=True, stripped from `model_dump()` automatically" — the strip-from-dump is correct for serialization but means `model_copy(update=...)` cannot preserve runtime cache.

**Why dormant today**: all 5 production `model_copy(update=...)` callers (CLI override application in `cli.py`, `train.py`, `export_signals.py`, CV trainer at `cv_trainer.py:256`, schema.py:2238) fire BEFORE `Trainer.setup()` executes, and `Trainer.setup` at `training/trainer.py:416-418` populates `_feature_indices_resolved` and `_feature_set_ref_resolved` AFTER all config mutation is complete. The downstream consumer at `importance/callback.py:509` reads with `getattr(config.data, "_feature_indices_resolved", None)` — graceful degradation by synthesizing `feature_X` names.

**Empirical reproducer**:

```python
from lobtrainer.config.schema import DataConfig
cfg = DataConfig(feature_indices=[0,1,2], feature_count=3)
cfg._feature_indices_resolved = (0,1,2,3,4)
c1 = cfg.model_copy()
print("c1 cache:", c1._feature_indices_resolved)  # (0,1,2,3,4) PRESERVED
c2 = cfg.model_copy(update={"feature_count": 4})
print("c2 cache:", c2._feature_indices_resolved)  # None LOST
```

**Production impact (if activated)**: regression in feature-importance artifact metadata — feature names would degrade from real names (e.g., `mid_price`, `volume_imbalance`) to synthesized `feature_0`, `feature_1`, etc., breaking the Stage C.5 feedback-merge contract that requires `feature_set_ref` for upstream consumption.

**Suggested fix scope**:
- Minimal patch: explicitly copy PrivateAttr fields in the override's update branch via `for attr in self.__private_attributes__: setattr(new_inst, attr, getattr(self, attr))`.
- Principled fix: document this in the model_copy override docstring + add a parametric regression test over PrivateAttr-bearing config classes.

---

### D4. Sample weights computed pre-trim, mean drifts after multi-source alignment

**Severity**: HIGH
**Status**: DORMANT — no multi-source experiments exist in production.
**Activation trigger**: any future T12 multi-source experiment (BASIC + MBO fusion).

**Files and lines**:
- `lob-model-trainer/src/lobtrainer/data/dataset.py:678-693` — `load_day_data` weight computation with full N
- `lob-model-trainer/src/lobtrainer/data/bundle.py:394-398` — slice-without-recompute

**What the code actually does**:

```python
# dataset.py:686-693
n_weight_samples = labels.shape[0] if labels.ndim >= 1 and labels.shape[0] > 0 else 0
if n_weight_samples > 0:
    computed_sample_weights = compute_sample_weights_for_day(
        n_samples=n_weight_samples,
        metadata=metadata,
        labels_config=labels_config,
        stride=export_stride,
    )
```

```python
# bundle.py:394-398
sample_weights=(
    primary_day_data.sample_weights[:n_aligned]
    if primary_day_data.sample_weights is not None
    else None
),
```

Trace: `compute_sample_weights_for_day` at `data/sample_weights.py:67` calls `hft_metrics.sample_weights.compute_sample_weights(n_samples, horizon, stride)`. Per CLAUDE.md §"Sample Weights (T10)", the AFML formula normalizes weights so `mean(weights) = 1.0` over the original N. After multi-source `_align_sources` produces `n_aligned ≤ N`, bundle.py:395 slices to `[:n_aligned]` without recomputing — `mean(weights[:n_aligned]) ≠ 1.0` and the per-sample `u_i = mean(1/c_t for t in [i, t1_i])` includes concurrency counts referencing labels at indices `[n_aligned, N)` that were trimmed from the bundle.

**Why this is a bug**: violates AFML §4.5.1 eq 4.2 — `u_i` MUST be computed over the actual label horizon counts in the trimmed dataset. The concurrent-overlap count `c_t` references labels that no longer exist in the bundle. Violates hft-rules §2 ("zero tolerance for precision errors") and §9 ("feature-label alignment").

**Why dormant today**: `grep -rn "sources:" lob-model-trainer/configs/experiments/` returns no hits. T12 multi-source mode has never been used in production — every active experiment is single-source (MBO-only XNAS or ARCX). The slice-without-recompute path at bundle.py:394-398 is reached only when `len(config.data.sources) > 1`.

**Empirical reproducer** (theoretical computation):

For `N = 1000, horizon = 60, stride = 1`:
- Full-N normalization: `weights = u * (1000 / sum(u))` so `mean(weights[0:1000]) = 1.0` exactly.
- After trim to `n_aligned = 900`: `mean(weights[0:900]) ≈ 1.003` (verified empirically: ~0.3% drift; max abs diff scales with how-correlated `u_i` is with index — first/last samples have lower `c_t`).

**Production impact (if activated)**: silent loss-scaling drift in any multi-source training run. Loss `loss_fn(reduction='none') * weights → .mean()` would have an effective scaling of ~1.003 instead of 1.0, masking convergence diagnostics. The bigger problem is the concurrency-count contamination: `u_i` for samples near the trim boundary references trimmed-away labels, producing biased weights that downweight (or upweight) the boundary samples in a horizon-dependent direction.

**Suggested fix scope**:
- Minimal patch: in bundle.py, after computing `n_aligned`, recompute weights via `compute_sample_weights_for_day(n_samples=n_aligned, ...)` instead of slicing.
- Principled fix: defer weight computation to `bundle.to_fused_day_data` so it always sees the final trimmed N.

---

### D5. HybridNormalizer fallback DEFAULT_EXCLUDE_INDICES missing index 95 (dt_seconds)

**Severity**: LOW
**Status**: DORMANT — hft_contracts is a hard runtime dependency; the ImportError fallback never fires in production.
**Activation trigger**: deployment environment where `hft_contracts` is missing or broken (not a realistic scenario for the monorepo).

**Files and lines**:
- `lob-model-trainer/src/lobtrainer/data/normalization.py:673-677`

**What the code actually does**:

```python
try:
    from hft_contracts import NON_NORMALIZABLE_INDICES as _CONTRACT_NON_NORM
    DEFAULT_EXCLUDE_INDICES: Tuple[int, ...] = tuple(sorted(_CONTRACT_NON_NORM))
except ImportError:
    DEFAULT_EXCLUDE_INDICES: Tuple[int, ...] = (92, 93, 94, 96, 97, 115)
```

Trace: production `hft_contracts.NON_NORMALIZABLE_INDICES = [92, 93, 94, 95, 96, 97, 115]` (7 values, includes `dt_seconds` at index 95). Fallback at line 677 lists only 6 values, missing 95.

**Why this is a bug**: violates hft-rules §1 ("contracts are explicit and synchronized — never duplicated") and §5 ("if a config option exists but is not fully supported, it must fail fast"). The fallback is a duplicated copy of a contract that drifts silently. If activated, dt_seconds (a non-stationary counter) would be normalized against training-set statistics and produce nonsense values for unseen days where the counter range differs.

**Why dormant today**: hft_contracts is a hard dep declared in `lob-model-trainer/pyproject.toml`. ImportError can occur only in a degraded deployment — and at that point, every other module-boundary import would also fail.

**Suggested fix scope**:
- Minimal patch: align the fallback with the canonical 7-value list.
- Principled fix: delete the fallback. Let ImportError propagate per hft-rules §5 (fail-fast). The "graceful degradation" pattern here masks a critical environment misconfiguration.

---

### D6. Day-1-only schema validation in load_split_data

**Severity**: MEDIUM
**Status**: DORMANT — production exports are uniform.
**Activation trigger**: any researcher mixing exports (e.g., partial re-export of recent days into a directory with older 2.1-schema days; mixed-vintage local data).

**Files and lines**:
- `lob-model-trainer/src/lobtrainer/data/dataset.py:840-880`

**What the code actually does**:

```python
days = []
contract_validated = False
expected_feature_count: Optional[int] = None

for data_file in data_files:
    # ... load day_data ...

    if not contract_validated and day_data.metadata is not None:
        _validate_day_metadata(day_data.metadata, date)
        n_feat = day_data.metadata.get("n_features")
        if n_feat is not None:
            expected_feature_count = int(n_feat)
        contract_validated = True

    if validate and not lazy:
        day_data.validate(expected_feature_count=expected_feature_count)

    days.append(day_data)
```

Trace: `contract_validated` flips to True after day 1 (line 877). Subsequent iterations skip the `_validate_day_metadata` block at line 872-877 entirely. Later schema_version-2.1 days would silently load with day 1's expected_feature_count, producing shape-aligned but contract-invalid sequences.

**Why this is a bug**: violates hft-rules §1 ("any contract change requires a version bump and synchronized updates") and §9 ("data lineage — every link must reference the specific artifact and config that produced it"). Day-N validation per export is the only defense against silent vintage drift.

**Why dormant today**: production exports are uniform single-vintage (one export run produces 233 days at one schema_version). The activation is a partial re-export scenario.

**Empirical reproducer**: not feasible without staged vintage data.

**Production impact (if activated)**: silent loading of mixed-vintage data. The most damaging variant: schema 2.1 had different feature semantics at indices 84-91 (trading signals) — a day from 2025 (schema 2.1) mixed with days from 2026 (schema 2.2) would feed the model semantically incompatible features at the same index slots. Discoverable only by inspecting day-by-day metadata after-the-fact.

**Suggested fix scope**:
- Minimal patch: remove the `not contract_validated and` guard at line 872; validate every day's metadata.
- Principled fix: add a `expected_schema_version: str` parameter to `load_split_data` and assert per-day equality.

---

### D7. Stability NaN on perfectly stable feature → `gate_stability=False`

**Severity**: LOW
**Status**: DORMANT — per-fold IC byte-identity essentially never occurs in real data.
**Activation trigger**: synthetic test data with identical IC across folds; degenerate features producing identical IC; test fixtures.

**Files and lines**:
- `hft-feature-evaluator/src/hft_evaluator/fast_gate.py:382-385, 512`

**What the code actually does**:

```python
# fast_gate.py:381-385
std = float(np.std(arr, ddof=1))
if std < 1e-12:
    return float("nan"), n_folds_eff, per_fold_ic
mean_abs = float(abs(np.mean(arr)))
return mean_abs / std, n_folds_eff, per_fold_ic

# fast_gate.py:512
gate_stability = bool(np.isfinite(stability) and stability > thresholds.min_stability)
```

Trace: when per-fold IC is exactly identical across folds (`std == 0`), the function returns NaN to avoid 0/0. The gate at line 512 then rejects NaN — so the gate fails.

**Why this is a bug**: semantic inversion. A perfectly stable feature SHOULD pass the stability gate (std=0 → infinite IR ratio → strongest possible stability), but the code FAILS it. Violates hft-rules §6 ("invariant tests must ensure consistency").

**Why dormant today**: per-fold IC across 5 folds × 1000 samples × float64 spearman is essentially never byte-identical in real market data. The activation requires synthetic data, deterministic constants, or a degenerate feature like `feature = 0.0` everywhere.

**Production impact (if activated)**: false NO-GO on synthetic/test data; CI test fixtures using deterministic data would silently fail the IC gate without reason.

**Suggested fix scope**:
- Minimal patch: when `std < 1e-12 and abs(mean) > 1e-12`, return `+inf` instead of NaN; only return NaN when both are zero.
- Principled fix: replace IR-based stability with a different stability statistic that handles the constant case.

---

### D8. Zero `prior_best_value` → unconditional PASS in PostTrainingGateRunner

**Severity**: LOW
**Status**: DORMANT — exact-zero `test_ic` is a near-zero-measure event in real data.
**Activation trigger**: synthetic baseline test deliberately producing test_ic=0; degenerate experiments where the model collapses to constant prediction.

**Files and lines**:
- `hft-ops/src/hft_ops/stages/post_training_gate.py:763-794`

**What the code actually does**:

```python
if prior_best_value <= 0:
    # Ratio-against-zero-or-negative is ill-defined, but a simple signed
    # inequality IS well-defined and catches regressions. Phase 7 Stage
    # 7.4 post-validation (2026-04-19) fix for A-H2: pre-fix, a new
    # experiment with worse-than-prior negative metric (e.g., test_r2
    # = -0.8 vs prior best = -0.1) was "vacuously passing" — now fails
    # loudly. Zero prior-best is treated as "no signal baseline" and
    # passes (neutral).
    if prior_best_value == 0 or current_value >= prior_best_value:
        return CheckResult(
            name="prior_best_ratio",
            status="pass",
            ...
        )
```

Trace: at line 771, the disjunction `if prior_best_value == 0 or current_value >= prior_best_value` returns PASS unconditionally when `prior_best_value == 0`, regardless of `current_value` — even `current_value = -0.4` (catastrophic regression).

**Why this is a bug**: comment at line 770 documents the intent ("zero = no signal baseline = neutral pass"), but a directional inversion regression (`current_value = -0.4`) should obviously fail, not pass. The disjunction conflates "no historical baseline" (legitimate first-of-kind PASS) with "historical baseline produced no signal but current is worse than no signal."

**Why dormant today**: floating-point exact zero on `test_ic` is essentially impossible — Spearman IC on real returns produces a continuous-valued statistic that rounds to ~1e-300 fluctuation rather than exact 0.0.

**Production impact (if activated)**: a current-experiment producing IC = -0.4 against a prior with IC = 0 would silently PASS the regression gate, masking a directional inversion. Currently impossible due to float precision.

**Suggested fix scope**:
- Minimal patch: change `if prior_best_value == 0` branch to require `current_value >= 0` (sign-positive only).
- Principled fix: replace the special case with a proper "first-of-kind PASS" mechanism keyed on `n_matching == 0` (already returned at line 753-762).

---

### D9. RNG state NOT captured in checkpoint — resume produces deterministic-but-misaligned RNG

**Severity**: HIGH (when composed with N2)
**Status**: DORMANT — only manifests as silent corruption when composed with N2 (epoch counter ignored on resume).
**Activation trigger**: training resume from checkpoint after partial run.

**Files and lines**:
- `lob-model-trainer/src/lobtrainer/training/trainer.py:982-1004` — `save_checkpoint` no RNG fields
- `lob-model-trainer/src/lobtrainer/training/trainer.py:751-752` — unconditional `set_seed` on every setup
- `lob-model-trainer/src/lobtrainer/utils/reproducibility.py:104-129` — `get_seed_state` exists but is NEVER called by trainer

**What the code actually does**:

```python
# trainer.py:982-1004
def save_checkpoint(self, path: Union[str, Path]) -> None:
    checkpoint = {
        'epoch': self.state.current_epoch,
        'global_step': self.state.global_step,
        'model_state_dict': self.model.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        'config': self.config.to_dict(),
        'state': {
            'best_val_metric': self.state.best_val_metric,
            'best_epoch': self.state.best_epoch,
        },
    }

    if self._scheduler is not None:
        checkpoint['scheduler_state_dict'] = self._scheduler.state_dict()

    torch.save(checkpoint, path)
```

```python
# trainer.py:751-752
# Set seed for reproducibility
set_seed(self.config.train.seed)
```

Trace: save_checkpoint omits `torch.get_rng_state()`, `torch.cuda.get_rng_state_all()`, `np.random.get_state()`, and `random.getstate()`. The `get_seed_state` helper at `reproducibility.py:104-129` exists for exactly this purpose but is NEVER called by trainer (`grep -n "get_seed_state\|set_seed_state" lob-model-trainer/src/lobtrainer/training/trainer.py` returns zero hits). Trainer.setup() at line 751-752 calls `set_seed(self.config.train.seed)` UNCONDITIONALLY on every fresh setup, including post-resume.

Resume sequence: load_checkpoint → setup() → set_seed(config.seed) → epoch 1 of resumed training reuses the IDENTICAL shuffle order, dropout mask sequence, and weight-init noise as fresh epoch 1.

**Why this is a bug**: violates hft-rules §7 ("same inputs MUST produce identical outputs across runs" requires resume = continuation, not restart). Resume IS deterministic but MISALIGNED — it reuses the SHUFFLE ORDER of the FIRST training run, so any data shuffle that happens during resume sees data in a non-stratified order biased toward "data the original run already processed in early epochs."

**Why dormant today (standalone)**: resume of a partial run with set_seed reseeding is "deterministic but wrong" — not silently-wrong-result on its own. The dormancy is conditional: composed with N2 (epoch counter ignored on resume), it becomes actual silent corruption because the epoch counter mismatch means the resumed run thinks it's epoch 1 and re-shuffles with seed `epoch=1` bias, biasing gradient updates toward the most-recently-seen data of the prior run.

**Empirical reproducer**:

```python
# Original run
trainer.train()  # epochs 1-5
# Save at epoch 3
trainer.save_checkpoint("ckpt.pt")  # epoch=3, global_step=Y

# Resume
new_trainer = Trainer(config)
new_trainer.load_checkpoint("ckpt.pt")
new_trainer.setup()  # set_seed(42) re-fires
# Epoch 1 of resumed run: identical shuffle order to epoch 1 of original
# But the model has already SEEN this data in original epochs 1-3
```

**Production impact (if activated)**: silent reuse of training data shuffle order across resume boundary. Combined with N2's epoch counter loss, this produces a model trained on biased data ordering that under-explores the dataset's tail.

**Suggested fix scope**:
- Minimal patch: capture RNG state in `save_checkpoint`; restore in `load_checkpoint`; conditionally skip `set_seed` in setup() when resuming. The `reproducibility.py:get_seed_state/set_seed_state` helpers already exist — wire them.
- Principled fix: explicit `Trainer.is_resumed: bool` flag set by `load_checkpoint`; `setup()` checks this flag before deciding whether to reseed.

---

## 6. Cross-Module Boundary Findings — Discovered in Validation Round

This section documents findings discovered in the V7 cross-cutting audit that span module boundaries (trainer ↔ hft-ops ↔ backtester ↔ hft-contracts). Each finding identifies a producer-consumer asymmetry where one module's defensive coding choice silently masks an error in another module's code path.

**Numbering note**: The F-series begins at F2 (NOT F1). During synthesis, what was originally tracked as "F1 — InputContract preflight `_base:` bug" was promoted to CRITICAL ACTIVE severity and renumbered to N1 (Section 4). The subsequent V7 findings retained their F-prefix numbering starting at F2 to preserve traceability with the round-2 validation reports. There is no missing finding — F1 is N1.

### F2. train.py final-evaluate catches only `ValueError`, drops other errors silently

**Severity**: HIGH
**Status**: NEW — discovered during V7 cross-module exception-handling audit.
**Activation trigger**: any non-ValueError exception during test-split evaluation: CUDA OOM (RuntimeError), missing-config-key (KeyError), test split returns empty (IndexError), mock test fixture with missing attribute (AttributeError).

**Files and lines**:
- `lob-model-trainer/scripts/train.py:442-451` — final evaluation loop (post-training)
- `lob-model-trainer/scripts/train.py:454-455` — final.pt save (skipped on exception)
- `lob-model-trainer/scripts/train.py:415-423` — duplicate `except ValueError` pattern in the `evaluate-only` branch (less severe — no `final.pt` follows in that path — but logically the same swallow pattern; a fix should address both sites)

**What the code actually does**:

```python
# train.py:442-451
for split in ['val', 'test']:
    try:
        metrics = trainer.evaluate(split)
        if split == 'test':
            written = _dump_test_metrics(metrics, output_dir)
            if written is not None:
                logger.info(f"Saved test metrics to {written}")
        logger.info(f"\n{split.upper()} Results:\n{_safe_summary(metrics)}")
    except ValueError as e:
        logger.warning(f"Could not evaluate {split}: {e}")

# Save final model
final_model_path = output_dir / 'checkpoints' / 'final.pt'
trainer.save_checkpoint(final_model_path)
```

Trace: only `ValueError` caught at line 450. RuntimeError (CUDA OOM during evaluation), KeyError (mock test config missing key), IndexError (empty test split), AttributeError (object lacks `.summary()`) all PROPAGATE past the for-loop. The function exits abnormally before line 454-455 — `final.pt` never saved + `test_metrics.json` never written.

**Why this is a bug**: violates hft-rules §8 ("never silently drop, clamp, or fix data without recording diagnostics"). The narrow `except ValueError` was likely added to catch an earlier observed bug (label-shape mismatch?) but inverts the priority — the most common evaluation failure modes (CUDA OOM, empty splits) bypass the except entirely and kill the training script before the final checkpoint is saved.

**Cascade across module boundaries**:

```
train.py exception escapes
    → final.pt save SKIPPED + test_metrics.json NEVER WRITTEN
        → hft-ops _capture_training_metrics at training.py:478 finds no test_metrics.json
            → silent zero-key capture
                → PostTrainingGate prior-best-ratio query falls back to best_val_ic
                    → ledger record has empty test_* fields
                        → ledger list --gate-status filters miss this run
```

**Empirical reproducer**:

```python
# Mock test config raises AttributeError
config.test_split = MockBadObject()  # raises AttributeError on .summary()
# Run train.py
# Result: AttributeError during evaluate('test') → escapes the try → final.pt NOT saved
```

**Production impact (if activated)**: silent loss of test-split artifacts after a complete training run. The trained model is on disk (from per-epoch checkpoints) but the standardized `final.pt` + `test_metrics.json` artifacts that downstream stages expect are missing. Discoverable only by inspecting the output directory after-the-fact.

**Root cause analysis**: defensive narrow exception type, likely added without considering the cascade impact on the post-evaluation save path. The except-then-continue pattern should be the LAST step in the function (with ALL artifacts already saved before the try-except), not the second-to-last.

**Suggested fix scope**:
- Minimal patch: broaden to `except Exception as e: logger.exception(...)` AND move the `trainer.save_checkpoint(final_model_path)` call BEFORE the try-except.
- Principled fix: split into two functions — `train()` saves all checkpoints + metrics that don't require evaluation; `evaluate_and_dump()` runs eval with proper error capture but doesn't gate the final.pt save.

---

### F3. hft-ops `_capture_training_metrics` swallows JSON corruption silently

**Severity**: HIGH
**Status**: NEW — discovered during V7 cross-module JSON-handling audit.
**Activation trigger**: SIGKILL during `_dump_test_metrics` write; out-of-disk-space mid-write; concurrent reader of partially-written file; corrupted JSON from prior bug.

**Files and lines**:
- `hft-ops/src/hft_ops/stages/training.py:471-472, 486-487` — bare `except: pass` on JSON read errors
- `lob-model-trainer/scripts/train.py:237-239` — non-atomic write that creates the corruption window

**What the code actually does**:

```python
# training.py:471-472 (training_history.json read)
except (json.JSONDecodeError, OSError):
    pass

# training.py:486-487 (test_metrics.json read)
except (json.JSONDecodeError, OSError):
    pass
```

```python
# train.py:237-239 (non-atomic write)
output_path = output_dir / "test_metrics.json"
with open(output_path, "w") as f:
    json.dump(prefixed, f, indent=2, sort_keys=True)
```

Trace: train.py:238 opens file for write (truncates to 0 bytes), then json.dump writes. If SIGKILL fires between open and json.dump completion, `test_metrics.json` exists with partial JSON. hft-ops `_capture_training_metrics` at training.py:478 finds the file, attempts to parse, hits JSONDecodeError, silently `pass`es, and `result.captured_metrics` does NOT get the test_* keys. Operator sees no error.

**Why this is a bug**: compound violation. (a) train.py:237-239 violates the canonical "atomic write" pattern documented in `hft-contracts/atomic_io.py::atomic_write_json` (tmp + fsync + os.replace). (b) hft-ops `except: pass` violates hft-rules §8 ("never silently drop").

**Cascade**: empty captured_metrics → PostTrainingGate sees zero-valued test_ic → may produce false-PASS or false-SKIPPED → ledger record buys an experiment as "succeeded" when it actually crashed mid-test-write.

**Empirical reproducer**:

```python
# Simulate partial write
test_metrics_file = output_dir / "test_metrics.json"
with open(test_metrics_file, "w") as f:
    f.write('{"test_ic": 0.3')  # Truncated mid-JSON, no closing brace

# Run hft-ops _capture_training_metrics
# Result: silent zero capture, no diagnostic
```

**Production impact (if activated)**: ledger records with apparent success but missing test_* metrics; false PASS through PostTrainingGate; downstream sweep-compare tooling produces nonsense rankings.

**Suggested fix scope**:
- Minimal patch (hft-ops): replace `except: pass` with `except (json.JSONDecodeError, OSError) as e: logger.warning("..."); result.captured_metrics["_metric_capture_errors"] = ["test_metrics.json: " + str(e)]`. Surface to operator via ledger query.
- Minimal patch (trainer): migrate `_dump_test_metrics` at train.py:237-239 to `atomic_write_json` from `hft_contracts.atomic_io`.
- Principled fix: introduce an artifact-validation pass in `_capture_training_metrics` that distinguishes "file absent" (legit) from "file present but unparseable" (corruption — should surface as an error or at least a captured warning).

---

### F4. BacktestData.__post_init__ validates only prices, not predictions/labels

**Severity**: HIGH
**Status**: NEW — discovered during V7 backtester input-validation audit.
**Activation trigger**: any silent NaN/Inf injection in `predictions`, `predicted_returns`, `regression_labels`, `spreads`, `agreement_ratio`, `confirmation_score` arrays at signal-export time.

**Files and lines**:
- `lob-backtester/src/lobbacktest/engine/vectorized.py:66-75` — validation only on `prices`
- `lob-backtester/src/lobbacktest/engine/vectorized.py:445-447` — silent zero-substitution on returns

**What the code actually does**:

```python
# vectorized.py:66-75
def __post_init__(self) -> None:
    """Validate data."""
    if self.prices.ndim != 1:
        raise ValueError(f"prices must be 1D, got shape {self.prices.shape}")
    if len(self.prices) == 0:
        raise ValueError("prices cannot be empty")
    if not np.all(np.isfinite(self.prices)):
        raise ValueError("prices contains NaN or Inf values")
    if np.any(self.prices <= 0):
        raise ValueError("prices must be positive")
```

```python
# vectorized.py:445-447
# Compute returns
returns = np.diff(equity) / equity[:-1]
# Handle division by zero
returns = np.where(np.isfinite(returns), returns, 0.0)
```

Trace: `__post_init__` validates `prices` for NaN/Inf/non-positive. The 6 other arrays (`predictions`, `predicted_returns`, `regression_labels`, `spreads`, `agreement_ratio`, `confirmation_score`) declared at lines 53-64 are NOT validated. NaN/Inf in any of these propagates through the per-sample loop → `equity[i] = cash + position_value` becomes NaN → `np.diff(equity) / equity[:-1]` is NaN → line 447 silently substitutes with 0.0 → the equity-curve metric reports a 0.0 return that masks the actual NaN catastrophe.

**Why this is a bug**: violates hft-rules §8 ("never silently drop, clamp, or fix data") and §9 ("provenance before comparison — verify inputs before computing"). If a producer (signal_export) bug or a partial-write produces NaN in `predicted_returns`, the backtester silently zeros out every affected timestep instead of failing loud.

**Empirical reproducer**:

```python
import numpy as np
from lobbacktest.engine.vectorized import BacktestData
data = BacktestData(
    prices=np.array([100.0, 100.5, 101.0]),
    predicted_returns=np.array([0.001, np.nan, 0.002]),  # NaN at index 1
    regression_labels=np.array([0.001, 0.001, 0.001]),
    spreads=np.array([0.5, 0.5, 0.5]),
)
# Constructs without raising — NaN survives
# Run backtest → returns[1] = NaN/value = NaN → silently → 0.0
# Reported metrics: TotalReturn looks fine, hides catastrophic bug
```

**Production impact (if activated)**: silent loss of P&L during NaN windows. A signal_export bug producing NaN at certain timesteps would be fully masked by the 0.0-substitution. Operator sees a "successful" backtest with realistic but wrong P&L.

**Suggested fix scope**:
- Minimal patch: extend `__post_init__` to validate finiteness on every supplied array (`predictions`, `predicted_returns`, etc.).
- Principled fix: replace silent zero-substitution at line 447 with explicit `if not np.all(np.isfinite(returns)): raise BacktestError(...)` per hft-rules §8. Acknowledge that legitimate division-by-zero (when `equity[i-1] == 0`) is an unrecoverable error — the backtester should not silently absorb it.

---

### F5. Backtester DataLoader concatenates ALL train days into RAM (~16-20GB for NVDA)

**Severity**: MEDIUM (workload-dependent CRITICAL on constrained machines)
**Status**: NEW — discovered during V7 backtester I/O audit.
**Activation trigger**: any backtest invocation on a 100+-day train split with full 98- or 148-feature exports on a memory-constrained machine.

**Files and lines**:
- `lob-backtester/src/lobbacktest/data/loader.py:188-248`

**What the code actually does**:

```python
# loader.py:188-191
for seq_file in seq_files:
    date = seq_file.stem.replace("_sequences", "")

    sequences = np.load(seq_file)
    # ... no mmap_mode, full materialization
```

```python
# loader.py:238-248
all_sequences.append(sequences)
all_labels.append(labels)
all_prices.append(prices)

return LoadedData(
    sequences=np.concatenate(all_sequences, axis=0),
    labels=np.concatenate(all_labels, axis=0),
    prices=np.concatenate(all_prices, axis=0),
    day_boundaries=day_boundaries,
    days=days,
)
```

Trace: `np.load(seq_file)` at line 191 reads the full file into RAM (no `mmap_mode='r'`). For 233-day NVDA at 100,000 sequences/day × 100 timesteps × 98 features × 4 bytes = ~3.7 GB/day for sequences alone. 163 train days × 3.7 GB = 603 GB in theory, but the actual config produces ~100 MB/day after stride/horizon trimming → 163 × 100 MB = 16 GB just for sequences. Plus `np.concatenate` at line 243 requires another 16 GB of contiguous RAM during the concat operation (peak 32 GB transiently).

**Why this is a bug**: violates hft-rules §12 ("hot paths must avoid allocations where possible; keep memory bounded"). The trainer-side dataset.py at line 869-870 supports `lazy=True` and `mmap_mode='r'` — the backtester does not.

**Empirical reproducer**:

```python
# On a machine with 24GB RAM:
loader = DataLoader(export_dir="/path/to/233-day-export")
data = loader.load_split("train")  # OOM crash on day ~120
```

**Production impact (if activated)**: OOM crashes on machines with <32 GB RAM when running backtests against full 163-day train splits. Workaround: the backtest typically runs against test split (35 days × 100 MB = ~3.5 GB, manageable), so the issue activates only when test split is large or train backtests are needed.

**Suggested fix scope**:
- Minimal patch: add `mmap_mode='r'` to `np.load` at line 191 + lazy concatenation via a custom iterator that yields per-day arrays.
- Principled fix: rearchitect `LoadedData` to wrap a list-of-mmaps with virtual indexing, mirroring trainer's `lazy=True` mode in dataset.py.

---

### F6. train.py:save_config write is non-atomic (also: `schema.py:to_json` has same pattern)

**Extension found by V4 audit 2026-04-27**: the same non-atomic `open(w) + yaml.dump` pattern duplicates at `lob-model-trainer/src/lobtrainer/config/schema.py:2361-2364` (`ExperimentConfig.to_json` method). A unified fix should migrate both `save_config` (yaml) and `to_json` (json) to use atomic_write helpers. The existing `hft_contracts.atomic_io.atomic_write_json` SSoT is suitable for the JSON site; a new `atomic_write_yaml` helper would close the YAML site (or both could route through a generic `atomic_write_text(serializer)` helper).

**Severity**: MEDIUM
**Status**: NEW — discovered during V7 atomic-write coverage audit.
**Activation trigger**: SIGKILL during config persistence; out-of-disk-space; concurrent reader (orchestrator polls config.yaml during write).

**Files and lines**:
- `lob-model-trainer/scripts/train.py:360-362` — non-atomic invocation
- `lob-model-trainer/src/lobtrainer/config/schema.py:2356-2359` — `to_yaml` plain open + yaml.dump

**What the code actually does**:

```python
# train.py:360-362
config_save_path = output_dir / "config.yaml"
save_config(config, str(config_save_path))
```

```python
# schema.py:2356-2359
def to_yaml(self, path: str) -> None:
    """Save configuration to YAML file."""
    with open(path, "w") as f:
        yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
```

Trace: line 2358-2359 plain `open(w) + yaml.dump`. SIGKILL between open (truncates to 0) and yaml.dump completion produces a partial YAML at `output_dir/config.yaml`. The hft-ops SignalExportRunner config-resolution logic (3-tier resolver per Phase 7.5-A) reads `<training.output_dir>/config.yaml` as priority 2; partial YAML triggers either (a) parse error → fallback to legacy priority 3 → silent divergence between trained-checkpoint and signal-export config, or (b) successful parse of truncated YAML → wrong config silently used.

**Why this is a bug**: violates the monorepo's "atomic write" convention documented in `hft-contracts/atomic_io.py`. The Phase 7 Stage 7.4 Round 5 audit migrated `ExperimentRecord.save()` and other JSON producers to `atomic_write_json`; YAML producers were not migrated.

**Empirical reproducer**: SIGKILL during a long YAML serialization (large nested dict) leaves a partial file.

**Production impact (if activated)**: orchestrator-level silent miscompose. The downstream signal_export stage uses a stale or partial config that drifts from the actual trained checkpoint's config.

**Suggested fix scope**:
- Minimal patch: replace `to_yaml` body with tmp-write + fsync + os.replace pattern, mirroring `atomic_write_json`.
- Principled fix: extend `hft-contracts/atomic_io.py` with `atomic_write_yaml` SSoT helper; migrate all YAML producers across the monorepo.

---

### F7. hft-ops `_apply_overrides` and `_materialize_inline_config` non-atomic

**Severity**: MEDIUM
**Status**: NEW — discovered during V7 atomic-write coverage audit.
**Activation trigger**: SIGKILL during YAML write; trainer subprocess spawn racing with orchestrator write.

**Files and lines**:
- `hft-ops/src/hft_ops/stages/training.py:121-122` — `_apply_overrides` non-atomic write
- `hft-ops/src/hft_ops/stages/training.py:208-209` — `_materialize_inline_config` non-atomic write

**What the code actually does**:

```python
# training.py:121-122
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
```

```python
# training.py:208-209
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
```

Trace: both helpers materialize the resolved trainer YAML to a temp file path, then immediately spawn the trainer subprocess with that path as `--config`. If the orchestrator process is killed between open(w) and yaml.dump completion, the temp file is partial → trainer subprocess reads partial YAML → parser error → trainer crashes with an unclear stack trace.

**Why this is a bug**: violates atomic-write convention. The race window is small (a few hundred ms for a YAML write) but real, especially under sweep parallelism where the orchestrator spawns many subprocesses rapidly.

**Empirical reproducer**: SIGKILL during a sweep parallel run mid-yaml.dump.

**Production impact (if activated)**: trainer subprocess fails with confusing YAML parser error; ledger record shows "training stage failed" without revealing the actual cause.

**Suggested fix scope**:
- Minimal patch: replace both write blocks with `atomic_write_json`-equivalent (tmp + fsync + os.replace).
- Principled fix: hoist into `hft_contracts.atomic_io.atomic_write_yaml` SSoT (same as F6).

---

### F8. PermutationImportanceCallback swallows all post-train failures (also: `callback.py:557` enum-resolution cousin site)

**Extension found by V5 audit 2026-04-27**: a sibling broad-except at `lob-model-trainer/src/lobtrainer/training/importance/callback.py:557` (in `_resolve_feature_metadata`'s FeatureIndex/ExperimentalFeatureIndex enum-iteration path) silently degrades to `experimental_by_value = {}` if enum iteration ever fails. Severity: LOW (extremely rare — would require import-time/iter-time enum corruption). Fix: same template as F8 — narrow to specific exception types or capture diagnostic to `result.captured_metrics`.

**Severity**: MEDIUM
**Status**: NEW — discovered during V7 callback exception-handling audit.
**Activation trigger**: any failure inside `_compute_and_save` — eval-loader resolution, predict_fn construction, metric_fn task-type dispatch, importance computation, artifact serialization.

**Files and lines**:
- `lob-model-trainer/src/lobtrainer/training/importance/callback.py:294-312`

**What the code actually does**:

```python
def on_train_end(self) -> None:  # noqa: C901 (dispatch complexity intentional)
    """Compute importance on the configured eval split + write the artifact."""
    if not permutation_importance_enabled(self.config):
        logger.debug(
            "PermutationImportanceCallback: gate disabled "
            "(enabled=%s, method=%s); no-op.",
            self.config.enabled, self.config.method,
        )
        return

    try:
        self._compute_and_save()
    except Exception:
        logger.exception(
            "PermutationImportanceCallback: artifact generation "
            "failed. Training run is NOT affected; the artifact "
            "will be missing from this run's output. See traceback "
            "above for root cause."
        )
```

Trace: broad `except Exception` at line 306 catches ALL post-train failures (intentional design — training must never be killed by importance failure per Phase 8C-α "graceful failure observation-tier contract"). The exception is logged via `logger.exception` at line 307 (good — full traceback emitted), but no diagnostic is captured in `result.captured_metrics` for the operator to find via ledger query.

**Why this is a bug**: half-violation of hft-rules §8. The `logger.exception` line is a strong diagnostic for live operators watching the log stream; but for ledger-driven retrospective analysis ("which experiments had silent importance failure?"), there is no machine-queryable signal. Operators must manually grep stage logs to find which runs lost their importance artifact.

**Why dormant in part**: the design choice (training must not die from importance failure) is correct. The defect is in observability granularity, not in the swallow itself.

**Suggested fix scope**:
- Minimal patch: extend `on_train_end` to write a marker file `<output_dir>/importance_failed.json` capturing the exception class + message; have hft-ops `_capture_training_metrics` read this and emit `result.captured_metrics["importance_failed"] = True` for ledger surfacing.
- Principled fix: define a generic "callback observation channel" wired to `result.captured_metrics` so any callback can surface its observability state to the ledger.

---

### F9. classification.py:151 broad except masks class-count failures

**Severity**: LOW
**Status**: NEW — discovered during V7 strategy exception-handling audit.
**Activation trigger**: NaN labels, empty batches, CUDA OOM during DataLoader iteration, mock train_loader missing 2-tuple/3-tuple shape.

**Files and lines**:
- `lob-model-trainer/src/lobtrainer/training/strategies/classification.py:151-153`

**What the code actually does**:

```python
def _compute_class_counts(
    self,
    train_loader: DataLoader,
    num_classes: int,
    class_names: list,
) -> Optional[torch.Tensor]:
    """Count samples per class from training data."""
    try:
        class_counts = torch.zeros(num_classes)
        for batch in train_loader:
            labels = batch[1]  # works for both 2-tuple and 3-tuple (T10)
            for c in range(num_classes):
                class_counts[c] += (labels == c).sum().item()

        total = class_counts.sum()
        if total > 0:
            for i, name in enumerate(class_names):
                pct = class_counts[i] / total * 100
                logger.info(f"  Class {name}: {int(class_counts[i]):,} ({pct:.1f}%)")
        return class_counts
    except Exception as e:
        logger.warning(f"Could not compute class counts: {e}")
        return None
```

Trace: line 151 broad `except Exception` swallows everything from CUDA OOM (RuntimeError) to NaN-corrupted labels (ValueError on isnan check) to empty batches (IndexError) to mock-loader misconfiguration (AttributeError, TypeError). Returns None silently. The downstream consumer at the strategy's loss-construction path then uses `class_counts=None` → uniform class weights → training proceeds with possibly-degenerate weights.

**Why this is a bug**: violates hft-rules §8 ("never silently drop, clamp, or fix data"). The except logs a warning (good — visible in stage log), but does NOT propagate to `result.captured_metrics` and does NOT cause the trainer to fail. Result: a CUDA-OOM during class-count enumeration silently degrades to "skip class weights" + continues training, producing a model trained without class-imbalance correction without informing the operator that the correction was disabled.

**Why low severity**: the immediate consequence (training without class weights) is recoverable — most production classification experiments don't use class weights anyway (most NVDA splits are roughly balanced after threshold calibration). The bug surfaces only when an operator explicitly relies on class-imbalance correction.

**Suggested fix scope**:
- Minimal patch: narrow except to the actually-expected exceptions (e.g., `IndexError`, `KeyError`); let RuntimeError propagate so CUDA OOM kills training instead of silently degrading.
- Principled fix: surface failure via `self.captured_metrics["class_counts_failed"] = str(e)` and document in the strategy's docstring that class weights are best-effort.

---

---

## 7. REFUTED Claims

This section preserves five claims raised by prior round-1 audit agents that re-investigation determined to be FALSE. Each entry shows the original claim, the actual code that refutes it, and why the prior agent was wrong. Future agents who encounter these patterns should NOT re-claim them as bugs without new, contradicting evidence.

---

### F-1. `exporter.py:502` 1-D fallback `or 0` idiom-trap

**Original claim**: Round 1 Agent 4 (HIGH-2): the fallback `primary_idx = labels_cfg.primary_horizon_idx or 0` at `exporter.py:502` is a Python idiom-trap because `0 or 0 == 0` evaluates the same as `None or 0 == 0`. The agent argued this could mask a `primary_horizon_idx == 0` configured-zero value being conflated with a `None` (unset) sentinel, leading to silent off-by-one when a future contract change makes `0` a meaningful index distinct from "unset".

**Verdict**: FALSE — claim is incorrect.

**Verified by**: V4 validation agent (V4 re-traced the 1-D control flow and confirmed `primary_idx` is unused for slicing in this branch).

**Files cited by claim**:
- `lob-model-trainer/src/lobtrainer/export/exporter.py:495-504` — calibration dispatch with 2-D vs 1-D branches.

**What the code actually does**:

```python
# /Users/knight/code_local/HFT-pipeline-v2/lob-model-trainer/src/lobtrainer/export/exporter.py:494-504
labels_cfg = resolve_labels_config(self._trainer.config)
if preds.ndim == 2:
    primary_idx = labels_cfg.validate_primary_horizon_idx_for(
        n_horizons=preds.shape[-1]
    )
    preds_1d = preds[:, primary_idx]
    labels_1d = labels[:, primary_idx] if labels is not None else None
else:
    primary_idx = labels_cfg.primary_horizon_idx or 0
    preds_1d = preds          # NO slicing — preds is already 1-D
    labels_1d = labels         # NO slicing — labels is already 1-D
```

**Why the prior agent was wrong**:

The 1-D branch performs **no slicing**. `primary_idx` is computed but only used as a **metadata field** in the calibration call:

```python
# exporter.py:511-516
cal_result = calibrate_variance(
    preds_1d,
    labels_1d,
    config,
    context={"primary_horizon_idx": primary_idx},  # metadata only
)
```

When `preds.ndim == 1`, the data structure has only one horizon — the architectural invariant `n_horizons == 1` makes `idx == 0` the *only* valid value. The `or 0` is a defensive fallback for the case where `LabelsConfig.primary_horizon_idx` is `None` (unset), and is correct under that invariant.

Furthermore, `LabelsConfig.__post_init__` (post-Phase A.5.3a) already rejects negative `primary_horizon_idx` at construction time via Pydantic validators (`SafeBaseModel` with `extra="forbid"` and field bounds checks). So the only un-handled case is `None → 0`, which is the fallback's purpose. The "0 is meaningful" speculative future change does not exist in the current contract.

**For future agents**: do NOT re-claim this. The 1-D branch does not slice; `primary_idx` is metadata only. Even if `LabelsConfig.primary_horizon_idx == 0` were a meaningful "configured zero" distinct from "unset None", the operational behavior would be identical (no slicing happens) and the metadata recorded in `signal_metadata.json` would correctly read `0` either way. The idiom is fine.

---

### F-2. `ExperimentConfig.input_size` staleness on `model_copy(update={"feature_count": ...})`

**Original claim**: Round 1 Agent 5 (CRIT-3): T13-B `_validate_all` self-mutation at `schema.py:2213` could leak stale input_size on `model_copy(update={"data": new_data_with_changed_feature_count})`. The agent argued that since `_validate_all` only auto-derives when `self.model.input_size == 0`, a subsequent `model_copy(update=...)` carrying the post-derivation populated input_size would silently drift if the new `data.feature_count` differs.

**Verdict**: FALSE — claim is incorrect.

**Verified by**: V2 validation agent (empirical reproduction with Pydantic `model_copy(update=...)` + ValidationError trace).

**Files cited by claim**:
- `lob-model-trainer/src/lobtrainer/config/schema.py:2213` — auto-derive guard `if self.model.input_size == 0`
- `lob-model-trainer/src/lobtrainer/config/schema.py:2246-2251` — fail-loud elif branch.

**What the code actually does**:

```python
# /Users/knight/code_local/HFT-pipeline-v2/lob-model-trainer/src/lobtrainer/config/schema.py:2213-2251
if self.model.input_size == 0:
    # Auto-derive (T13).
    # ... resolves resolved_input_size ...
    _new_model = self.model.model_copy(update={
        "input_size": resolved_input_size,
        "params": _new_params,
    })
    object.__setattr__(self, "model", _new_model)
    _t13_logger.info(
        "Auto-derived model.input_size = %d", resolved_input_size
    )
elif self.model.input_size != resolved_input_size:
    raise ValueError(
        f"model.input_size ({self.model.input_size}) != resolved "
        f"feature count ({resolved_input_size}). "
        f"Set model.input_size: 0 to auto-derive."
    )
```

**Why the prior agent was wrong**:

After construction, `self.model.input_size` is **no longer zero** (auto-derive ran). On a subsequent `cfg.model_copy(update={"data": cfg.data.model_copy(update={"feature_count": 116})})`, Pydantic re-fires `_validate_all` (frozen models with `validate_assignment=True` re-run validators on copy). The condition `self.model.input_size == 0` is False, so control falls through to the `elif` branch. The `elif` compares the populated `model.input_size` against the freshly resolved `resolved_input_size` derived from the new `data.feature_count` — and **raises ValueError** when they diverge.

This is **fail-loud**, not silent. The agent missed that the elif branch is the protection mechanism, not a "validation hole". V2 reproduced the path empirically: a `model_copy(update={"data": ..._with_different_feature_count})` raises `ValidationError` with the diagnostic message above; it does not silently drift.

The pattern is documented in the §A.5.3i KEYSTONE comment block at `schema.py:2216-2233` explaining the two-layer mutation pattern with `object.__setattr__` and why direct assignment / nested `model_copy` would loop.

**For future agents**: do NOT re-claim. The elif at line 2246 is the fail-loud guard. If you suspect drift is possible, reproduce empirically with a model_copy + assert on `cfg.model.input_size == resolved value`. The `ValidationError` will fire BEFORE you reach any silent-drift path.

---

### F-3. BASIC export missing `dataset_manifest.json` defaults stride=1, breaking AFML

**Original claim**: Round 1 Agent 1 (MED-2): BASIC exports have NO `dataset_manifest.json`, so `dataset.py:819-824` defaults `export_stride = 1`. AFML's `compute_sample_weights_for_day` then computes `effective_horizon = ceil(horizon/1) = horizon` instead of `ceil(horizon/10)`. For event-based exports with stride=10 events, this would silently 10x over-weight overlap, distorting all sample-weighted regression training.

**Verdict**: FALSE — claim is incorrect (premise inverted).

**Verified by**: V5 validation agent (file-system inspection of `data/exports/basic_nvda_60s/` confirmed manifest presence; BASIC unit-system audit confirmed stride/horizon coherence).

**Files cited by claim**:
- `lob-model-trainer/src/lobtrainer/data/dataset.py:819-824` — manifest read with `stride = 1` fallback.

**What the code actually does**:

```python
# /Users/knight/code_local/HFT-pipeline-v2/lob-model-trainer/src/lobtrainer/data/dataset.py:818-824
# T10: read export stride from manifest for sample weight computation
export_stride = 1  # safe default (time-based exports)
manifest_path = data_dir / "dataset_manifest.json"
if manifest_path.exists():
    with open(manifest_path) as _mf:
        _manifest = json.load(_mf)
    export_stride = _manifest.get("stride", 1)
```

Inspection of the actual BASIC export reveals the manifest DOES exist:

```json
// /Users/knight/code_local/HFT-pipeline-v2/data/exports/basic_nvda_60s/dataset_manifest.json
{
  "experiment": "basic_nvda_60s",
  "data_source": "XNAS.BASIC",
  "feature_count": 34,
  "sequence_length": 20,
  "stride": 1,
  "bin_size_seconds": 60,
  "horizons": [1, 2, 3, 5, 10, 20, 30, 60],
  ...
}
```

**Why the prior agent was wrong**:

The audit's premise ("BASIC has NO manifest") was inverted. BASIC exports do publish a `dataset_manifest.json`, written by `basic-quote-processor` during the export. The `stride=1` value in the manifest is **correct semantics for the BASIC time-based regime**: in BASIC, `stride` is in **bin units** (one 60-second bin per emission), and `horizon` is also in **bin units** (e.g., `horizon=60` means 60 bins ahead = 60 minutes ahead).

Critically, AFML's formula `effective_horizon = ceil(horizon / stride)` is **unit-agnostic** — it requires only that `stride` and `horizon` be expressed in the same unit. For BASIC: `ceil(60 / 1) = 60` bins of overlap, which is the correct concurrency window. For event-based MBO with `stride=10 events / horizon=10 events`: `ceil(10 / 10) = 1`, also correct.

The unit confusion in the prior audit (assuming `stride=10` was load-bearing for BASIC) is a category error. There is no silent 10× over-weighting in either pipeline because both pipelines maintain stride/horizon in their respective native units.

**For future agents**: do NOT re-claim. BASIC has a manifest with `stride=1` (correct for bin-units). Event-based MBO has a manifest with `stride=10` (correct for event-units). The AFML formula is unit-coherent within each pipeline.

---

### F-4. `PostTrainingGateRunner` test_ic vs best_val_ic apples-to-oranges comparison

**Original claim**: Round 1 Agent 6 (CRIT-4): `_select_primary_metric` cascades through `_PRIMARY_METRIC_FALLBACK_ORDER` from `test_*` keys to `best_val_*` keys. The agent argued: experiment A logs `test_ic=0.05`; experiment B logs only `best_val_ic=0.13` (no test pass). When the gate compares B's prior-best against A, it would compare `0.13 (val) vs 0.05 (test)` — apples-to-oranges, since test and validation distributions differ in shape and difficulty.

**Verdict**: FALSE — claim is incorrect.

**Verified by**: V6 validation agent (re-traced `_find_prior_best_experiment` ledger query path and confirmed metric_name single-selection invariant).

**Files cited by claim**:
- `hft-ops/src/hft_ops/stages/post_training_gate.py:62-84` — `_PRIMARY_METRIC_FALLBACK_ORDER` tuple.
- `hft-ops/src/hft_ops/stages/post_training_gate.py:472-483` — `_select_primary_metric` cascade implementation.

**What the code actually does**:

```python
# /Users/knight/code_local/HFT-pipeline-v2/hft-ops/src/hft_ops/stages/post_training_gate.py:472-483
def _select_primary_metric(
    metrics: Dict[str, float],
    configured_name: str,
) -> tuple[str, Optional[float]]:
    """Return (metric_name, metric_value) per stage config + fallback order."""
    if configured_name:
        # Configured explicitly — use it, even if missing (None value).
        return configured_name, metrics.get(configured_name)
    for candidate in _PRIMARY_METRIC_FALLBACK_ORDER:
        if candidate in metrics:
            return candidate, metrics[candidate]
    return "", None
```

The function returns a single `(metric_name, metric_value)` pair. Downstream, `_find_prior_best_experiment(metric_name=primary_name, ...)` queries the ledger for **that exact metric_name only**. If prior records lack the same key, `prior_best_value=None`, and `_check_prior_best_ratio` returns status `"skipped"` (no baseline available).

**Why the prior agent was wrong**:

The cascade selects ONE metric for the current experiment, and the ledger query uses THAT EXACT metric_name as the lookup key. There is no path where the gate compares experiment A's `test_ic=0.05` against experiment B's `best_val_ic=0.13`. Either:

- Both experiments exposed `test_ic` → both log it → comparison is `test_ic vs test_ic` (apples-to-apples).
- Experiment B exposes only `best_val_ic` → A's lookup for B's `test_ic` returns `None` → `prior_best_value=None` → check `skipped`, not silently mis-compared.

The prior agent appears to have read the cascade as "the gate compares the first available metric across experiments" — it does not. The cascade only resolves the *current experiment's* primary metric; the prior-best lookup is a lookup by name, not a cascade.

**For future agents**: do NOT re-claim. The cascade resolves a single primary metric for the current experiment; the ledger lookup uses that exact key. Mixed-metric comparisons cannot occur. If you suspect a comparison is incorrect, trace `_find_prior_best_experiment(metric_name=...)` and confirm it filters records by `metric_name in record.metrics`.

---

### F-5. `make_metric_fn_for_task` Spearman vs trainer's Pearson `test_ic` mismatch

**Original claim**: Round 1 Agent 6 (HIGH-4): the regression `make_metric_fn_for_task` in `callback.py` uses Spearman IC (`spearman_ic` from hft_metrics). If trainer's `test_ic` is computed via Pearson correlation, the post-training gate would compare permutation-importance Spearman-IC against test-set Pearson-IC — apples-to-oranges.

**Verdict**: FALSE — claim is incorrect.

**Verified by**: V6 validation agent (re-read both call sites in trainer + callback).

**Files cited by claim**:
- `lob-model-trainer/src/lobtrainer/training/importance/callback.py:140-155` — `make_metric_fn_for_task` regression branch.
- `lob-model-trainer/src/lobtrainer/training/regression_metrics.py:29-40` — trainer-side `information_coefficient`.

**What the code actually does**:

Callback site (importance feature evaluator):

```python
# /Users/knight/code_local/HFT-pipeline-v2/lob-model-trainer/src/lobtrainer/training/importance/callback.py:140-155
if task_type == "regression":
    from hft_metrics.ic import spearman_ic

    def regression_metric(preds: np.ndarray, y: np.ndarray) -> float:
        # Multi-horizon preds/y → reduce to primary horizon.
        if preds.ndim > 1 and preds.shape[-1] > 1:
            preds = preds[:, primary_horizon_idx]
        if y.ndim > 1 and y.shape[-1] > 1:
            y = y[:, primary_horizon_idx]
        preds = preds.ravel()
        y = y.ravel()
        # spearman_ic returns (ic, p_value); take the point estimate.
        ic, _p = spearman_ic(preds, y)
        return float(ic)

    return regression_metric
```

Trainer site (test_metrics.json producer):

```python
# /Users/knight/code_local/HFT-pipeline-v2/lob-model-trainer/src/lobtrainer/training/regression_metrics.py:29-40
def information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank IC between predicted and actual returns.

    Wrapper over ``hft_metrics.ic.spearman_ic`` which returns ``(rho, p_value)``.
    This adapter returns rho only for backward compatibility.

    Reference: Grinold & Kahn (2000), "Active Portfolio Management", Ch 6.
    """
    from hft_metrics.ic import spearman_ic

    rho, _ = spearman_ic(y_true, y_pred)
    return rho
```

**Why the prior agent was wrong**:

Both sites use `hft_metrics.ic.spearman_ic` and label the output `ic`. The trainer's metrics dict assembled in `regression_metrics.py:76-86` writes `f"{prefix}ic"` (Spearman) and `f"{prefix}pearson"` (separate Pearson key) — so `test_ic` is unambiguously Spearman, and Pearson lives under `test_pearson`.

The prior agent appears to have assumed `ic` ≡ Pearson by convention. In this codebase, `ic` ≡ Spearman by deliberate de-facto contract (cited in the docstring with reference to Grinold & Kahn). Both the importance callback and the trainer's primary metric agree on Spearman.

**For future agents**: do NOT re-claim. `ic` in this codebase is Spearman everywhere. Pearson lives under `pearson` keys. If future code introduces a Pearson-only `ic` field, that would be a separate bug — but the current state is consistent.

---

## 8. Architectural Patterns and Root Causes

The validation surfaced six cross-cutting structural patterns. Each pattern has multiple instances in this audit, suggesting that instance-level patches (fixing each occurrence) would not stop new instances from appearing as the codebase grows. Pattern-level retirement — restructuring the architecture so the bug class is no longer expressible — is the long-run-engineering response (HFT rule §0).

---

### P1. Three independent `primary_horizon` channels

**Pattern**: The notion of "the primary horizon for this experiment" is encoded in three independent channels with no enforced consistency.

**Instances** (in this audit):
- `LabelsConfig.primary_horizon_idx` (config layer): integer index into the label tensor's horizon axis. Default 0; user-configurable per YAML.
- `hmhp_horizons[0]` (HMHP architectural convention at `lob-models/src/lobmodels/models/hmhp.py:705`): the cascading-decoder backbone treats the **first listed** horizon (NOT "shortest sorted" — the position-0 of `hmhp_horizons` as configured) as the primary. If `hmhp_horizons = [60, 10, 300]`, the model treats `60` as primary regardless of `LabelsConfig.primary_horizon_idx`.
- `sorted(label_dict.keys())[0]` (exporter HMHP-classification convention): when classification labels are emitted as a dict-of-arrays keyed by horizon, the exporter takes `sorted(keys)[0]` as primary.

These three channels coincide ONLY when `hmhp_horizons` is sorted ascending AND `LabelsConfig.primary_horizon_idx == 0`. They diverge silently otherwise: a user who configures `primary_horizon_idx=2` against `hmhp_horizons=[10, 60, 300]` thinks "predict 300", but the HMHP backbone trains on horizon=10 as primary, and the exporter slices column-2 (horizon=300) post-train. The model never optimized for horizon=300; the metrics describe horizon=300; the gate compares both.

**Root cause**:

Phase A.5.4 added `LabelsConfig.validate_primary_horizon_idx_for(n_horizons)` as a SSoT helper, but it is **opt-in**: only 4 call sites currently invoke it (4 exporter slicing sites + the importance callback). The HMHP backbone bakes the "primary = horizons[0]" convention into the cascading-decoder construction at `lob-models/src/lobmodels/models/hmhp.py:705`, bypassing the helper. The exporter HMHP-cls path takes `sorted(label_dict.keys())[0]` independently. There is no architectural force compelling all three channels to read from one source.

**Retirement strategy**:

- **Instance patching** (rejected): annotate each call site with the helper and add an integration assertion test for each model. Brittle; new model architectures would re-invent their own convention.
- **Pattern retirement** (recommended): introduce a `PrimaryHorizonSlicer` value-type bound at config-resolve time. The slicer encapsulates `(horizons_tuple, primary_idx)` and exposes one method `select_primary(tensor) → tensor` that performs the bounds-checked slice. Every consumer (model, exporter, importance, gate, calibration) accepts a `PrimaryHorizonSlicer` rather than a raw int. The HMHP backbone takes a slicer too — its construction must use the slicer's `select_primary` rather than `horizons[0]`. With this refactor, the bug class "horizons-list-and-idx-disagree" becomes architecturally inexpressible.

**Lesson for future architectural work**:

When a piece of business logic ("which horizon is primary?") has multiple call sites that semantically must agree, the SSoT must be **mandatory** (not opt-in) and **type-enforced** (not function-call-enforced). Helper functions that consumers can fail to call are not real SSoTs.

---

### P2. Pydantic validator-skip pattern (`if X is None / if not X`)

**Pattern**: A `@model_validator(mode="after")` auto-derives a derived field only when the field is `None` / empty / zero. After construction, the field is populated, so subsequent `model_copy(update=...)` carrying the dumped value through the new construction sees the populated field, the guard skips, and the derived field is left stale relative to the new input field.

**Instances** (in this audit):
- **T9** `DataConfig.labels` derivation: `if labels is None: derive from labeling_strategy + horizon_idx`. Once labels is populated, a `model_copy(update={"labeling_strategy": NEW})` carries the OLD labels through unchanged.
- **T13** `ModelConfig.params` self-mutation: `if "num_features" not in params: derive from input_size`. Once params has num_features, a `model_copy(update={"input_size": NEW})` leaves stale params.
- **T13-B** `ExperimentConfig.input_size` (REFUTED in F-2 above; this instance is fail-loud, not silent): the elif branch raises ValueError when input_size diverges from resolved feature count, neutralizing the bug class for this specific case.

T9 and T13 are DORMANT in production (no current YAML or training script triggers a `model_copy` that exercises the stale-derived path). Phase A.5.9 deferred the fix correctly under the §0 small-reversible-changes principle. But the bug class is real and would activate the moment a sweep generator or programmatic config-mutation tool calls `model_copy(update=...)` on these fields.

**Root cause**:

The validator-skip idiom `if X is None: derive` mistakes "field is None" for "the derived state is invalid". The two are not equivalent: after the first derivation, `X` is populated even if the input fields it was derived FROM have changed. The validator has no way to detect "X is stale relative to current input fields" because Pydantic v2 `mode="after"` validators receive the post-validation `self` with all fields already populated.

**Retirement strategy**:

- **Instance patching**: replace `if X is None` with `if X is None or X != _expected(self)`. Brittle: requires the validator to recompute the expected value purely as a comparison, which is the same work as the derivation itself.
- **Pattern retirement** (recommended): two structural alternatives:
  - **Pydantic v2 `@computed_field`**: declare derived fields as computed properties bound to the input fields. The Pydantic engine recomputes on every access; no `model_copy` staleness is possible because the field has no stored state.
  - **`mode="before"` validators that drop derived fields when input fields are present**: the validator inspects the raw input dict; if the input fields are present, it deletes any provided derived-field values before the field-construction phase. This forces re-derivation on every `model_validate` (including `model_copy` paths, which re-route through `model_validate`).

The `@computed_field` route is preferred because it eliminates the field-storage entirely, removing any opportunity for staleness.

**Lesson for future architectural work**:

Auto-derivation in `mode="after"` validators is fragile under `model_copy` semantics. Either move derivation to `@computed_field` (preferred — Pydantic-native) or `mode="before"` (pre-construction). Avoid `if X is None` as the gate-condition; it conflates absence with validity.

---

### P3. Resume semantics broken at multiple layers

**Pattern**: `Trainer.train(resume_from=ckpt)` is broken at four independent layers, and these layers compose multiplicatively: any single failure produces silent corruption; together they ensure that every `--resume` invocation produces a model that is structurally distinct from the un-interrupted training trajectory.

**Instances** (in this audit):
- **N2 (V1 agent)**: epoch counter ignored. `Trainer.train()` always starts at epoch=0 even when the checkpoint contains `epoch=5`. The training loop runs `cfg.train.epochs` more epochs than intended, and per-epoch artifacts (checkpoints, history) overwrite epoch-0 artifacts.
- **N3 (V1 agent)**: callback state reset. `EarlyStopping` (patience counter), `ModelCheckpoint` (best-metric tracking), `MetricLogger` (rolling buffers) are all reconstructed fresh on resume. Early-stopping fires in mid-trajectory because the patience counter starts at 0; the "best" model checkpoint overwrites the pre-resume best.
- **D9 + N2 composition**: RNG state misalignment. The DataLoader RNG is re-seeded from `cfg.train.seed` on `resume`, not from the checkpoint's RNG state. Combined with the epoch-zero restart, the resumed run sees a different mini-batch sequence than the un-interrupted run, so loss curves diverge.
- **Bug 9 (V3 agent — partially mitigated)**: optimizer state device portability. The checkpoint serializes `optimizer.state_dict()` with parameters living on the original training device. **Mitigation in current code**: `Trainer.load_checkpoint` calls `torch.load(path, map_location=self.device)` at `trainer.py:1015`, which remaps all tensors in the checkpoint to the target device at load time. PyTorch's `optimizer.load_state_dict` then preserves those device assignments. Standard cross-device transfers (GPU0 → GPU1 or GPU → CPU) work correctly through this idiom. The bug only manifests in narrower scenarios: distributed training (`DDP` / `FSDP`) state, stream-bound CUDA buffers, or custom optimizers with non-tensor state that bypasses `map_location`. The original V3 framing ("resume on a different device leaves buffers on the wrong device, silently corrupting parameter updates") was overstated for the standard case but accurate for distributed/specialized scenarios.

Test coverage of resume across all four layers: 0%. There are no integration tests asserting "resumed run after N epochs produces identical model to un-interrupted N+M-epoch run".

**Root cause**:

The Trainer was built for forward training only. `save_checkpoint` was retro-fitted as a "save weights" feature, not as a "save trajectory state" feature. The checkpoint dict contains hard-coded fields `{"model_state_dict", "optimizer_state_dict", "epoch", "best_metric"}` and assumes callbacks are stateless / reconstructable from config. There is no `Callback.state_dict() / load_state_dict()` interface; callbacks have no contract for serializing their internal state.

**Retirement strategy**:

- **Instance patching**: fix N2 first (start loop at `start_epoch = ckpt["epoch"]`); add manual save/restore for each callback's state in `Trainer.save_checkpoint`. Brittle: every new callback would need a maintainer to remember to wire its state up.
- **Pattern retirement** (recommended): introduce a `Callback.state_dict() → Dict[str, Any]` and `Callback.load_state_dict(d) → None` interface as a base contract. `Trainer.save_checkpoint` walks `self.callbacks` and serializes each callback's state under a key. RNG state (Python `random`, NumPy, PyTorch CPU+CUDA) is captured into the checkpoint as a nested dict. Optimizer state is moved to CPU before serialization and restored to the target device on load. Add an integration test: train for N epochs, save at epoch=N//2, resume, train remaining; assert byte-identical model state to un-interrupted N-epoch reference.

**Lesson for future architectural work**:

Resume is not "load weights and keep going" — it is **trajectory replay**. Every stateful component (callbacks, RNG, optimizer, dataloader-iter-state) must have a serialization contract. If any component lacks one, resume is silently broken.

---

### P4. Calibration producer/consumer drift

**Pattern**: When the producer emits both a primary artifact (raw predictions) and a derived artifact (calibrated predictions), and the producer also emits metadata describing the artifact, the metadata describes the wrong array if not updated synchronously with derivation.

**Instances** (in this audit):
- **B5 (V6 agent)**: exporter writes `calibrated_returns.npy` when `--calibrate variance_match` is passed AND writes `signal_metadata.json` with magnitude metrics (MAE, RMSE, R², |pred|_mean). The metrics are computed on RAW `predicted_returns`, but the backtester loads `calibrated_returns.npy` when the manifest declares calibration. The metadata fields describe a DIFFERENT array than what the backtester trades against.

**Root cause**:

The `metrics` block in `signal_metadata.json` was added at an earlier phase (pre-calibration support). Phase E6 added calibration to the exporter, wired the calibration into a separate file write, and updated the manifest's `calibration` block — but the `metrics` block was never updated to recompute on the calibrated array. There is no architectural force compelling "metrics describe what the backtester actually trades".

**Retirement strategy**:

- **Instance patching**: in the exporter, after calibration, recompute `metrics` from `calibration_result.calibrated`. Brittle: future producers (ensemble averaging, post-prediction filtering) would each need an ad-hoc remember-to-update.
- **Pattern retirement** (recommended): formalize a producer-consumer contract: "metrics in `signal_metadata.json` describe the array the backtester actually trades". Lock with an assertion in `BacktestData.from_signal_dir`: load the predictions array (calibrated if present, raw otherwise), recompute a fingerprint of the metrics dict against that array (e.g., assert `abs(mean(predictions) - metadata["metrics"]["pred_mean"]) < 1e-9`), raise `ContractError` on divergence. The producer is then forced (by failing test) to recompute metrics on the calibrated array.

**Lesson for future architectural work**:

Whenever the producer writes metadata describing an artifact AND the artifact has a derived form (calibration, smoothing, post-filter), the metadata must be locked to the FINAL form of the artifact. A consumer-side assertion is the fastest way to make the contract self-enforcing.

---

### P5. Defaults can be silently invalid

**Pattern**: A configuration field with a default value where the default semantically violates the field's documented behavior.

**Instances** (in this audit):
- **F4 (V4 agent)**: `ImportanceConfig.block_length_samples = 1` default. The docstring says "block-permutation null distribution"; with `block_length_samples=1`, the permutation degenerates to element-wise (every sample is its own block). This is mathematically equivalent to `np.random.permutation`, NOT block-permutation. The null is over-conservative (no autocorrelation preserved), and importance estimates have inflated variance.
- **F7 (V4 agent)**: `PostTrainingGateRunner.min_ratio_vs_prior_best = 0.9` default. The Phase 7 Stage 7.4 spec explicitly stated `0.95` as the conservative default. The 0.9 default is more permissive — a 10% regression silently passes when 5% was the intent.
- **F11 (V5 agent)**: `DEFAULT_EXCLUDE_INDICES` fallback for `NormalizationConfig.exclude_features` lacks index 95 (`dt_seconds`). Index 95 is a non-normalizable feature per the 148-feature contract; normalizing it produces meaningless z-scores for time deltas. The fallback dropped the index by oversight when the categorical-features list was hand-typed.

**Root cause**:

Default values are hand-typed against the field's documented semantics, but the documented semantics aren't enforced by tests. There is no test that says "with the default value of `block_length_samples`, the null distribution must exhibit block-permutation properties (e.g., autocorrelation preserved within blocks)". Without a behavior-check test on the default, the default drifts from the documented contract over multiple PRs.

**Retirement strategy**:

- **Instance patching**: change defaults to known-correct values. Fragile: future defaults may drift again.
- **Pattern retirement** (recommended): every default value gets a "default-semantic-validity" test in the field's owning module. For `block_length_samples`, the test asserts that with the default value AND a synthetic time series with known autocorrelation, the null distribution preserves the autocorrelation. For `min_ratio_vs_prior_best`, the test asserts the value matches the spec'd 0.95 (or wherever the spec lives — TOML changelog entry, etc.). For `DEFAULT_EXCLUDE_INDICES`, the test asserts every non-normalizable feature in the contract is in the default exclude list.

**Lesson for future architectural work**:

A default that "looks reasonable" is not a default that "is correct". Every default should have an assertion test that exercises the field's documented semantics with the default value. Defaults are part of the public contract.

---

### P6. Broad `except Exception` swallows correctness bugs

**Pattern**: A `try / except Exception` (or `except BaseException`) wrapping logic that can fail in multiple distinct ways → log a generic warning + fall back to a safe default → silent observability loss.

**Instances** (in this audit):
- **Phase A `_build_compatibility_contract`**: pre-Phase A C1 fix, this method silently caught any error in label resolution and returned `None`, hiding a 4-day-long bug where every signal export had `compatibility_contract.labels == None`. The fix added explicit handling for the specific failure modes.
- **F8 (V5 agent)**: `PermutationImportanceCallback.on_train_end` wraps the entire artifact-production pipeline in `except Exception: logger.warning(...)`. If the eval loader fails, the metric_fn fails, the predict_fn fails, or the artifact serialization fails — all collapse into one warning. The training run completes "successfully" with no artifact and no diagnostic of which step broke.
- **F9 (V4 agent)**: `classification.py:151` (regression metrics fallback) catches all exceptions when computing metrics on a multi-horizon prediction. Silent fallback to `nan` on shape errors, dtype errors, NaN inputs, etc.
- **F2 (V2 agent — clarified)**: `train.py` final-evaluate at `scripts/train.py:422,450` uses `except ValueError as e: logger.warning(...)` (NOT `except Exception: pass` as a prior version of this section described). The narrow `ValueError` catch lets RuntimeError/KeyError/IndexError/AttributeError propagate, which means `final.pt` and `test_metrics.json` are silently NOT written when `trainer.evaluate()` raises a non-ValueError. The bug class is the same (silent loss of test artifacts) but the catch is narrower than the prior description suggested. Note: the same pattern duplicates at `train.py:415-423` in the `evaluate-only` branch — fix should address both sites.
- **F3 (V3 agent — clarified)**: `hft-ops/stages/training.py:471-487` `_capture_training_metrics` uses `except (json.JSONDecodeError, OSError): pass` (NOT "catches all exceptions" as a prior version described). Narrow catches, but still silent: a partially-written `test_metrics.json` (mid-write SIGKILL) produces JSONDecodeError → silent zero-metric capture → PostTrainingGate falls back to `best_val_ic`. Compounded with `train.py:_dump_test_metrics` at line 237-239 being non-atomic (`open(w) + json.dump`).

**Root cause**:

Defensive coding by junior engineers conflates two distinct cases:
1. **Legitimate boundary error handling**: e.g., "the upstream tool crashed; we want this stage to record FAILED but not crash the whole pipeline".
2. **Lazy error suppression**: e.g., "I don't know what could go wrong here, so I'll swallow everything".

Case (1) is correct when the catch is narrow (`except SubprocessError as e:`) and the failure is RECORDED in observable artifacts (`result.captured_metrics["_failure_kind"] = ...`). Case (2) is HFT rule §8 violation: "Never silently drop, clamp, or fix data without recording diagnostics."

**Retirement strategy**:

- **Instance patching**: tighten each catch site to specific exception types and add `result.captured_metrics["_failure_kind"]` recording. Boring but tractable.
- **Pattern retirement** (recommended): codify the invariant in a project-wide lint rule: `except Exception:` and `except BaseException:` are forbidden EXCEPT (a) in re-raise paths (`except Exception: log_then_raise(...)`); (b) in atomic-write cleanup (the only legitimate use case for catch-all is "ensure tmp file is cleaned up regardless of exception kind"). All other broad catches must be narrowed and must record `_failure_kind`. Add a CI lint pass.

**Scope of pattern in current codebase** (open-ended grep, V5 audit 2026-04-27): **5 representative instances are cited above; ~48 broad `except Exception` / `except BaseException` instances total across `lob-model-trainer/src/`, `lob-models/src/`, and `hft-ops/src/`**. Many are case (1) legitimate boundary handling (e.g., `hft-ops/stages/training.py:379` records `result.error_message = str(e); result.status = StageStatus.FAILED`). Several deserve case-by-case triage:
- `lob-model-trainer/src/lobtrainer/training/importance/callback.py:557` — second broad-except in same file as F8 (FeatureIndex enum-resolution path; silent degradation to synthetic feature names if enum iteration fails)
- `hft-ops/src/hft_ops/ledger/dedup.py:77, 124, 252, 379` — 4 broad-except in fingerprint computation
- `hft-ops/src/hft_ops/ledger/ledger.py:475, 500, 509, 551` — 4 broad-except in record handling
- `hft-ops/src/hft_ops/feature_sets/producer.py:274, 280` — 2 broad-except in feature set production

A single CI lint pass would surface all 48 for individual triage. Each must be evaluated against case (1) vs case (2); narrowing requires per-site judgment.

**Lesson for future architectural work**:

`except Exception` is a code smell. It almost always hides a bug class. The narrow alternative is harder to write but produces software where failures are observable, which is non-negotiable for a research pipeline whose value is empirical findings (silent failures = falsified findings).

---

## 9. CLEARED Items — Specifically Verified Sound

This section preserves items that round-1 audit agents specifically inspected and verified to be working correctly. Future agents should NOT re-investigate these items without new, contradicting evidence (e.g., a regression test failure, a new code path that bypasses the verified invariant).

---

### C1. AFML sample-weights formula `u_i = mean(1/c_t)` (NOT `1/mean(c_t)`)

**Verified by**: V1 validation agent
**Files**: `hft-metrics/src/hft_metrics/sample_weights.py` and `lob-model-trainer/src/lobtrainer/data/sample_weights.py`
**Status**: WORKING CORRECTLY

**Verified invariant**: For each label index `i` with `t1_i = min(i + effective_horizon, N-1)`, the per-label uniqueness is `u_i = mean(1/c_t for t in [i, t1_i])` — the **per-event reciprocal averaged over the label's lifetime**. The reciprocal-of-mean (`1/mean(c_t)`) is mathematically distinct and produces over-weighted long-lived labels (de Prado AFML 4.5.1, eq 4.2). Implementation in both Python sites uses the per-event-reciprocal form. Golden test in `hft-metrics` locks the formula against hand-calculated expected values.

---

### C2. Per-day weight normalization to mean=1.0

**Verified by**: V1 validation agent
**Files**: `lob-model-trainer/src/lobtrainer/data/sample_weights.py`
**Status**: WORKING CORRECTLY

**Verified invariant**: After computing `u_i` per AFML, the weights are normalized via `sample_weight_i = u_i / mean(u_·)` so that `mean(weights) == 1.0` per day. This preserves the loss magnitude invariance (a sample-weighted loss with `weights.mean() = 1` has the same expected scale as the unweighted loss). The normalization is applied per-day independently, so concurrency variance across days does not bias gradient magnitudes.

---

### C3. Train-only normalization stats (no val/test contamination)

**Verified by**: V1 validation agent
**Files**: `lob-model-trainer/src/lobtrainer/data/transforms.py`, `lob-model-trainer/src/lobtrainer/training/trainer.py`
**Status**: WORKING CORRECTLY

**Verified invariant**: Normalization statistics (means, stds, min/max) are computed exclusively from `train` split sequences. The `Trainer.setup` flow calls `_compute_normalization_stats(train_loader)` before constructing val/test loaders. Stats are saved alongside the checkpoint (`stats.npz`) for reproducible inference. Val and test loaders apply the same train-derived stats, guaranteeing no future leakage (HFT rule §9).

---

### C4. Welford streaming stats Chan-Golub-LeVeque parallel merge

**Verified by**: V1 validation agent
**Files**: `hft-statistics/src/welford.rs`
**Status**: WORKING CORRECTLY

**Verified invariant**: The `WelfordAccumulator` parallel-merge implementation follows Chan-Golub-LeVeque (1979) for combining two independent variance estimators, NOT the naive `(sum1 + sum2) / (n1 + n2)` form which loses precision under near-equal means. Locked by `welford_parity.json` cross-language fixture and 4 boundary tests covering empty merges, single-element merges, near-equal means, and large-N stability.

---

### C5. Forward-prices column k anchor reads from contract

**Verified by**: V1 validation agent
**Files**: `hft-contracts/src/hft_contracts/_generated.py` (constant `FORWARD_PRICE_BASE_OFFSET`); `lob-model-trainer/src/lobtrainer/data/dataset.py`
**Status**: WORKING CORRECTLY

**Verified invariant**: The base price for each sequence (column k in `forward_prices.npy`) is read from `ForwardPriceContract.smoothing_window_offset`, NOT hard-coded. T9 ForwardPriceContract is the SSoT; consumers (trainer, backtester, label factory) all import from `hft_contracts`. The contract guarantees `forward_prices[:, k]` is the mid-price at sequence end (anchor), with `forward_prices[:, k+h]` the mid-price at horizon h ahead.

---

### C6. Train/val/test split correctly excludes val/test from norm-stat computation

**Verified by**: V1 validation agent
**Files**: `lob-model-trainer/src/lobtrainer/training/trainer.py`
**Status**: WORKING CORRECTLY

**Verified invariant**: `Trainer._create_dataloaders` constructs the train loader first; `_compute_normalization_stats` is called ONLY on the train loader. Val and test loaders are constructed AFTER stats are finalized, with the same stats applied. There is no path where val/test sequences contribute to the running mean/std accumulators.

---

### C7. HMHP P0-3 zero-gradient guard

**Verified by**: V2 validation agent
**Files**: `lob-models/src/lobmodels/hmhp.py`
**Status**: WORKING CORRECTLY

**Verified invariant**: `HMHPModel.compute_loss` raises `ValueError` when no labels, no regression targets, AND no consistency penalty are configured (the loss would be a constant 0, producing zero-gradient updates that silently train no parameters). The guard is at the head of `compute_loss` and predates the full forward pass, so the failure is fast and unambiguous.

---

### C8. HMHP-R FRESH-2 dead-model gate

**Verified by**: V2 validation agent + V4 numeric reproducer
**Files**: `lob-models/src/lobmodels/models/hmhp_regressor.py:188-215` (`_compute_agreement` method)
**Status**: WORKING CORRECTLY

**Verified invariant**: When the HMHP-R model produces all-zero or all-NaN per-horizon predictions (a "dead model" — typically gradient-collapse or NaN-poison), the `_compute_agreement` routine forces `agreement_score = 0.5` (sentinel "no opinion") via the `nonzero_fraction == 0.0 → dead_mask` override at line 210-215. The `agreement_score` is a confirmation/confidence signal in `[0.5, 1.0]` (NOT a logit/prediction); HMHP-R has NO classification heads — predictions are not zeroed out, they remain whatever the cascading regression decoders produced. The gate prevents downstream confidence metrics from spurious-passing on degenerate outputs. Numeric reproducer (V4) verified all-zero → score 0.5, nonzero_fraction 0.0; all-NaN → same; constant +1 → score 1.0; mixed-sign → score < 1.0. Locked by FRESH-2 unit test.

---

### C9. HMHP `compute_loss` sample_weights threading

**Verified by**: V2 validation agent
**Files**: `lob-models/src/lobmodels/hmhp.py`
**Status**: WORKING CORRECTLY

**Verified invariant**: When `sample_weights` are provided, the per-sample loss is multiplied element-wise BEFORE the consistency penalty is added. The order is: (per-sample loss) × (sample_weights) → reduce → ADD consistency penalty. The consistency penalty (a model-internal regularizer) does not have per-sample semantics, so it is correctly excluded from the weighting. Locked by integration test asserting weighted vs unweighted losses differ only on the per-sample term.

---

### C10. Optimizer step ordering

**Verified by**: V2 validation agent
**Files**: `lob-model-trainer/src/lobtrainer/training/trainer.py`
**Status**: WORKING CORRECTLY

**Verified invariant**: The training step performs `optimizer.zero_grad() → forward → loss.backward() → grad_clip → optimizer.step()` in this exact order. Zero-grad before forward (not after step) is the PyTorch-recommended pattern and avoids stale-gradient accumulation when an exception is thrown mid-step. Gradient clipping is applied AFTER backward and BEFORE step (the only correct location).

---

### C11. `model.eval()` / `model.train()` toggle correctness

**Verified by**: V2 validation agent + V4 re-verification 2026-04-27
**Files**: `lob-model-trainer/src/lobtrainer/training/trainer.py:925-953` (Trainer.evaluate), strategies in `training/strategies/{regression,classification,hmhp_*}.py`
**Status**: WORKING CORRECTLY (functionally safe; invariant weaker than originally claimed)

**Verified invariant**: Every non-training forward pass enters `model.eval()` mode. Strategy `evaluate()` methods (e.g., `regression.py:101`) call `model.eval()` directly. **Note**: there is NO explicit `try/finally` pattern in any strategy or Trainer.evaluate — a prior version of this entry overstated the guarantee. Functional safety is achieved indirectly: `Trainer._train_epoch` at line 877 ALWAYS calls `self.model.train()` at the start of each training epoch, so even if a previous evaluate() left the model in eval mode (no try/finally restore), the next training epoch re-enables train mode. BatchNorm / Dropout layers thus behave correctly during evaluation (no batch-stat updates, no dropout); training restoration is guaranteed by the next-epoch `model.train()` call rather than by exception-safe restoration in evaluate.

---

### C12. `@torch.no_grad()` on all evaluate/validate/predict methods

**Verified by**: V2 validation agent
**Files**: `lob-model-trainer/src/lobtrainer/training/trainer.py`, `lob-model-trainer/src/lobtrainer/export/exporter.py`
**Status**: WORKING CORRECTLY

**Verified invariant**: Every non-training forward pass is wrapped in `@torch.no_grad()` (or `with torch.no_grad():`). This zeros out the autograd graph construction overhead during evaluation and prevents accidental gradient accumulation if the eval routine is called from a context where `requires_grad=True`. Verified across `Trainer.evaluate`, `Trainer.predict`, and `SignalExporter.export`.

---

### C13. `FocalLoss` thread-safety

**Verified by**: V2 validation agent + V6 file-path correction 2026-04-27
**Files**: `lob-model-trainer/src/lobtrainer/training/loss.py:26-110` (FocalLoss class), `lob-model-trainer/src/lobtrainer/training/trainer.py` (constructor call site). **Note**: prior cited path `lob-models/src/lobmodels/losses.py` was wrong — that file does not exist; lob-models has a `losses/` subpackage with `gmadl.py` and `pinball.py` only. FocalLoss lives in the trainer.
**Status**: WORKING CORRECTLY

**Verified invariant**: The `FocalLoss` criterion is constructed once in `Trainer.initialize`, not per batch. State (alpha, gamma) is immutable after construction. The criterion is callable from concurrent dataloader workers without race conditions. Locked by stress test simulating 4 concurrent workers consuming the same criterion instance.

---

### C14. AMP not used (no GradScaler / autocast)

**Verified by**: V2 validation agent
**Files**: `lob-model-trainer/src/lobtrainer/training/trainer.py`
**Status**: WORKING CORRECTLY (deliberately disabled)

**Verified invariant**: The trainer does NOT use `torch.cuda.amp.GradScaler` or `torch.autocast`. All training is full fp32. This is a deliberate design choice for HFT (numerical precision is non-negotiable per HFT rule §2). If AMP is added in future, all loss-magnitude tests must be re-run because mixed-precision changes accumulation order in attention/conv kernels.

---

### C15. TLOB BiN parameter naming convention vs reference

**Verified by**: V3 validation agent + V6 mechanism re-verification 2026-04-27
**Files**: `lob-models/src/lobmodels/layers/normalization.py:55-225` (BiN class). **Note**: prior cited path `lob-models/src/lobmodels/layers.py` was wrong — actual is `layers/normalization.py` (subdir).
**Status**: WORKING CORRECTLY (renamed parameter convention; numerical parity verified bit-exact)

**Verified invariant**: Our BiN has SIX parameters (NOT a `gamma`↔`beta` swap as a prior version of this entry described): `lambda_temporal[F,1]`, `beta_temporal[F,1]`, `lambda_feature[T,1]`, `beta_feature[T,1]`, `gamma_feature` (branch importance), `gamma_temporal` (branch importance). The official TLOB reference uses `l1[t1,1], l2[d1,1], B1, B2, y1, y2`. Our `gamma_feature, gamma_temporal` correspond to the reference's `y1, y2` (branch-weight scalars); our `lambda_*, beta_*` correspond to the reference's `l*, B*` (per-element affine). The mapping operates on opposite axes (lambda_temporal acts in the feature_normalize branch and vice versa) but math is bit-exact. Verified via numerical parity test: max abs diff = 0.0 against the official TLOB reference implementation on identical inputs.

---

### C16. DeepLOB Inception kernel sizes match Zhang et al. 2019

**Verified by**: V3 validation agent + V6 file-path correction 2026-04-27
**Files**: `lob-models/src/lobmodels/layers/inception.py:42-185` (Inception module), `lob-models/src/lobmodels/models/deeplob.py` (caller). **Note**: prior cited path `lob-models/src/lobmodels/deeplob.py` was wrong — actual `models/deeplob.py` (subdir).
**Status**: WORKING CORRECTLY

**Verified invariant**: The Inception module has THREE parallel branches (NOT four as a prior version of this entry incorrectly stated): (a) `branch_short` with `(short_kernel, 1) = (3, 1)` conv via `InceptionBranch`; (b) `branch_medium` with `(medium_kernel, 1) = (5, 1)` conv via `InceptionBranch`; (c) `branch_pool` with `MaxPool(pool_kernel=3)` then `(1, 1)` projection via `InceptionPoolBranch`. Forward at lines 213-216 concatenates exactly three branch outputs (`out_short, out_medium, out_pool`); output channels = `branch_filters × 3` (lines 149, 158). The `1×1` reductions are PER-BRANCH channel-reduction stages WITHIN each conv branch (gated by `use_1x1_reduce=True`), NOT a separate fourth parallel branch — a prior version of this entry conflated reduction-conv with branch count. Default `short_kernel=3, medium_kernel=5` per `config/base.py:187-188`. Matches Zhang et al. 2019, Figure 4 (3 parallel branches, each with its own 1×1 reduction). Verified against the paper's parameter count (~143K).

---

### C17. DeepLOB Conv blocks match paper figure 4

**Verified by**: V3 validation agent + V6 file-path correction 2026-04-27
**Files**: `lob-models/src/lobmodels/models/deeplob.py:130-220`. **Note**: prior cited path was missing the `models/` subdir.
**Status**: WORKING CORRECTLY

**Verified invariant**: The first conv block applies `(1×2)` then `(4×1)` kernels (price-level pairs first, then temporal); the second block applies `(1×2)` then `(4×1)` (level pairs second across the now-halved width); the third applies `(1×10)` to project to 10 features. The leaky-ReLU activations and BatchNorm placements match Zhang Figure 4. Total parameter count matches the paper's reference (~143K).

---

### C18. GMADL formula matches Michankov et al. 2024

**Verified by**: V3 validation agent + V6 mechanism re-verification 2026-04-27
**Files**: `lob-models/src/lobmodels/losses/gmadl.py:40-66`. **Note**: prior cited path was wrong — actual is `losses/gmadl.py` (subdir).
**Status**: WORKING CORRECTLY (mechanism description corrected from prior version)

**Verified invariant**: The Generalized Maximum-Likelihood-Adversarial-Directional Loss matches Michankov et al. 2024, equation 5. Formula at lines 60-64: `loss = -(sigmoid(a * y_true * y_pred) - 0.5) * |y_true|^b`. **Parameter roles** (corrected from a prior version of this entry): `a` is the SIGMOID SHARPNESS on the directional product `y_true * y_pred` (controls how sharply the loss saturates as directional agreement strengthens); `b` is the MAGNITUDE EXPONENT on `|y_true|^b` (controls how strongly large-magnitude targets are weighted). The prior entry described these in reverse. Locked by golden-value test against the paper's reference example.

---

### C19. Pinball loss formula correct

**Verified by**: V3 validation agent + V6 file-path correction 2026-04-27
**Files**: `lob-models/src/lobmodels/losses/pinball.py:24-72`. **Note**: prior cited path was wrong — actual is `losses/pinball.py` (subdir).
**Status**: WORKING CORRECTLY

**Verified invariant**: Pinball quantile loss correctly applies `(q - 1) * residual` for `residual < 0` and `q * residual` for `residual >= 0`. The sign convention (positive residual = under-prediction) is consistent with Koenker & Bassett 1978. Multi-quantile output (predicting multiple quantiles per sample) sums losses over quantile axis. Locked by golden-value test.

---

### C20. Heteroscedastic loss formula correct

**Verified by**: V3 validation agent + V6 mechanism re-verification 2026-04-27
**Files**: `lob-models/src/lobmodels/models/hmhp_regressor.py:370-388`. **Note**: prior cited path `lob-models/src/lobmodels/losses.py` was wrong — heteroscedastic NLL is implemented inline in HMHP-R's compute_loss.
**Status**: WORKING CORRECTLY (parameterization mechanism corrected from prior version)

**Verified invariant**: The heteroscedastic regression loss formula at line 379 is `0.5 * (residual^2)/unc + 0.5 * log(unc)`, matching Kendall & Gal 2017, "What Uncertainties Do We Need". **Parameterization** (corrected from prior version of this entry): `unc` is computed via `nn.Sequential(nn.Linear, nn.Softplus())` — i.e., the network outputs Softplus-transformed positive variance directly, NOT `exp(log_var)` as a prior version claimed. The `clamp(min=EPS)` at line 376 prevents log-of-zero. Functionally equivalent to log-var parameterization (both ensure `unc > 0`); the choice is implementation detail rather than mathematically distinct. Locked by gradient-flow stress test.

---

### C21. TLOB attention layer ordering

**Verified by**: V3 validation agent
**Files**: `lob-models/src/lobmodels/tlob.py`
**Status**: WORKING CORRECTLY

**Verified invariant**: Each TLOB block applies temporal attention FIRST (over the time axis) THEN feature attention (over the feature axis), matching Berti & Kasneci 2025. The pre-LayerNorm convention is applied at each attention site. Verified against the official TLOB reference implementation by per-layer activation comparison on identical inputs (max abs diff < 1e-6).

---

### C22. GRU/LSTM bidirectional final hidden state extraction

**Verified by**: V3 validation agent + V6 file-path correction 2026-04-27
**Files**: `lob-models/src/lobmodels/models/rnn.py:200-208` (GRUClassifier, LSTMClassifier). **Note**: prior cited path `lob-models/src/lobmodels/baselines.py` was wrong — RNN baselines live in `models/rnn.py` (subdir).
**Status**: WORKING CORRECTLY

**Verified invariant**: For bidirectional RNN, the final hidden state for classification is `torch.cat([h_n[-2], h_n[-1]], dim=-1)` — the forward direction's last layer's last hidden + the backward direction's last layer's last hidden. The `[-2:-1]` indexing convention matches PyTorch's `(num_layers * num_directions, batch, hidden)` packed shape per the PyTorch documentation. Common bug: using `h_n[-1]` only would drop the forward-direction signal entirely.

---

### C23. TLOB hyperparameters match Berti & Kasneci 2025

**Verified by**: V3 validation agent
**Files**: `lob-model-trainer/src/lobtrainer/configs/bases/models/tlob_compact_regression.yaml`, `lob-models/src/lobmodels/tlob.py`
**Status**: WORKING CORRECTLY

**Verified invariant**: The base config carries `num_heads=1`, `num_layers=4`, `hidden_dim=40`, matching Berti & Kasneci 2025's reference experiment. (Newer experiments may override these for ablation; the base default is the paper-faithful reference.)

---

### C24. XGBoost early_stopping_rounds RuntimeWarning when no eval_set

**Verified by**: V3 validation agent + V6 file-path correction 2026-04-27
**Files**: `lob-models/src/lobmodels/models/xgboost_model.py` (XGBoostLOB). **Note**: prior cited path was wrong — XGBoostLOB lives in `models/xgboost_model.py`.
**Status**: WORKING CORRECTLY (P0-1 fix verified)

**Verified invariant**: When `early_stopping_rounds > 0` is configured but no `eval_set` is passed to `model.fit`, the wrapper emits a RuntimeWarning citing the misconfiguration. The XGBoost library would otherwise silently ignore early stopping; the wrapper adds the diagnostic per HFT rule §8.

---

### C25. `calibrate_variance` 1-D contract enforcement

**Verified by**: V4 validation agent
**Files**: `lob-model-trainer/src/lobtrainer/calibration/variance.py`
**Status**: WORKING CORRECTLY

**Verified invariant**: `calibrate_variance(preds, labels, config)` raises `ValueError` if `preds.ndim != 1` or `labels.ndim != 1`. The 1-D contract is strict; 2-D inputs cannot silently fall back to a row-wise scalar calibration. Phase A.5.4 added the strict contract specifically to close the silent-mis-calibration bug class (multi-horizon HMHP-R was being calibrated against the wrong horizon column).

---

### C26. `_apply_calibration` 2-D path uses `validate_primary_horizon_idx_for`

**Verified by**: V4 validation agent
**Files**: `lob-model-trainer/src/lobtrainer/export/exporter.py:495-498`
**Status**: WORKING CORRECTLY (Phase A.5.4 fix verified)

**Verified invariant**: When `preds.ndim == 2`, the exporter calls `labels_cfg.validate_primary_horizon_idx_for(n_horizons=preds.shape[-1])` to bounds-check the primary index BEFORE slicing. The validator raises ValueError with diagnostic on negative or out-of-bounds idx, eliminating the prior silent-Python-negative-indexing hazard.

---

### C27. `_build_compatibility_contract` reads labels via `resolve_labels_config`

**Verified by**: V5 validation agent
**Files**: `lob-model-trainer/src/lobtrainer/export/compatibility.py`
**Status**: WORKING CORRECTLY (Phase A C1 fix verified)

**Verified invariant**: The compatibility contract resolves `LabelsConfig` via the canonical `resolve_labels_config(config)` helper, NOT direct attribute access (`config.labels` would raise AttributeError post Phase A.5; the helper handles both legacy and modern paths). The 4-day silent-None bug from pre-Phase A is closed; the contract now correctly populates the labels block in `signal_metadata.json`.

---

### C28. `atomic_write_json` BaseException-safe cleanup

**Verified by**: V5 validation agent
**Files**: `hft-contracts/src/hft_contracts/atomic_io.py`
**Status**: WORKING CORRECTLY (Phase 7.4 Round 5 SB-2 hardening verified)

**Verified invariant**: The atomic-write helper catches `BaseException` (NOT just Exception) and ensures the temp file is cleaned up on any failure mode (KeyboardInterrupt mid-fsync, MemoryError, system signals). The catch-and-cleanup-then-re-raise pattern is the only legitimate use of `except BaseException` per the §0 architectural review. OSError is wrapped as `AtomicWriteError`; other exceptions propagate unchanged.

---

### C29. `CONTENT_HASH_RE` 64-hex regex enforcement on `_harvest_compatibility_fingerprint`

**Verified by**: V5 validation agent
**Files**: `hft-ops/src/hft_ops/stages/signal_export.py`, `hft-contracts/src/hft_contracts/signal_manifest.py`
**Status**: WORKING CORRECTLY (Phase V.A.4 trust column verified)

**Verified invariant**: When the harvester reads `compatibility_fingerprint` from `signal_metadata.json`, it validates the value matches `CONTENT_HASH_RE` (lowercase 64-hex SHA-256). Malformed fingerprints (truncated, mixed-case, wrong length) fail the validator and emit a WARN log (per Phase A.5.1 timestamp_utils policy). The trust column for ledger filtering thus only carries valid fingerprints; corrupted ones are rejected at harvest, not at query time.

---

### C30. `is_after_cutoff` UTC-aware comparator

**Verified by**: V5 validation agent
**Files**: `hft-contracts/src/hft_contracts/timestamp_utils.py`
**Status**: WORKING CORRECTLY (Phase A.5.1 SSoT verified)

**Verified invariant**: `is_after_cutoff(ts_str, cutoff_str)` parses BOTH strings as ISO-8601 UTC-aware datetimes via `parse_iso8601_utc`, then compares the parsed datetime objects. Lexicographic string comparison (the prior implementation) is silently wrong for non-UTC offsets crossing midnight (e.g., `"2026-04-26T00:30:00+02:00"` lex-compares > `"2026-04-25T23:30:00Z"` despite being earlier in absolute time). The SSoT is now used by `_harvest_compatibility_fingerprint` post-cutoff WARN logic.

---

### C31. `compute_label_strategy_hash` Pydantic branch dispatch order

**Verified by**: V5 validation agent + V6 mechanism re-verification 2026-04-27
**Files**: `hft-contracts/src/hft_contracts/compatibility.py:268-330`. **Note**: prior cited path `hft-contracts/src/hft_contracts/canonical_hash.py:318-339` was wrong on TWO counts: (a) the function lives in `compatibility.py` not `canonical_hash.py`; (b) `canonical_hash.py` only has 158 lines so :318-339 is out-of-range.
**Status**: WORKING CORRECTLY (Phase A.5.1 byte-identity preserved)

**Verified invariant**: `compute_label_strategy_hash` uses **duck-typing** `hasattr(labels_config, "model_dump")` (NOT `isinstance(obj, BaseModel)` as a prior version of this entry incorrectly described) to detect Pydantic v2 models BEFORE falling through to `dataclasses.asdict()`. Post Phase A.5.3a, all 9 config classes are BaseModels; the Pydantic branch dispatches to `obj.model_dump(exclude_none=False)` which produces byte-identical JSON to the pre-Phase-A dataclass path. Byte-identity locked by golden test asserting same fingerprint pre/post migration.

---

### C32. Frozen `CompatibilityContract` horizons list→tuple coercion

**Verified by**: V5 validation agent
**Files**: `hft-contracts/src/hft_contracts/signal_manifest.py`
**Status**: WORKING CORRECTLY (Phase II SB-D fix verified)

**Verified invariant**: When the contract is constructed with `horizons=[10, 60, 300]` (list), the `__post_init__` coerces via `object.__setattr__(self, "horizons", tuple(horizons))` to ensure JSON round-trip stability. Lists serialize to JSON arrays; tuples also serialize to JSON arrays; both deserialize to lists by default. The tuple coercion guarantees the equality fingerprint doesn't drift across serialize/deserialize cycles.

---

### C33. `BacktestData.from_signal_dir(validate=False, expected_fields={...})` raises

**Verified by**: V5 validation agent + V6 file-path correction 2026-04-27
**Files**: `lob-backtester/src/lobbacktest/engine/vectorized.py:139-142` (BacktestData.from_signal_dir guard). **Note**: prior cited path `lob-backtester/src/lobbacktest/data/loader.py` was wrong — the guard is in `engine/vectorized.py`, not the loader.
**Status**: WORKING CORRECTLY (Phase II SB-E guard verified)

**Verified invariant**: Calling with `validate=False` AND `expected_fields={...}` simultaneously is contradictory (the caller is asking the loader to skip validation but also providing assertions). The loader raises ValueError up-front, refusing to silently drop the assertion. This is HFT rule §5 fail-fast.

---

### C34. `SignalManifest.validate(expected_fields={})` raises

**Verified by**: V5 validation agent
**Files**: `hft-contracts/src/hft_contracts/signal_manifest.py`
**Status**: WORKING CORRECTLY (Phase II SB-D guard verified)

**Verified invariant**: An empty `expected_fields` dict provides no assertions and would silently no-op the validation. The validate method now raises ValueError on empty dict, refusing to silently no-op. Caller must either pass a populated dict or omit the kwarg entirely (which uses the default full contract).

---

### C35. `SafeBaseModel` auto-registry (`__pydantic_init_subclass__`)

**Verified by**: V6 validation agent
**Files**: `lob-model-trainer/src/lobtrainer/config/base.py`
**Status**: WORKING CORRECTLY (Phase A.5.7a fix verified)

**Verified invariant**: Every subclass of `SafeBaseModel` is auto-registered into `SafeBaseModel._registry: ClassVar[List[type]]` via the `__pydantic_init_subclass__` hook, with a `_`-prefix filter excluding test fixtures. The hand-maintained `_PYDANTIC_CONFIG_CLASSES` list (which previously drifted out of sync with reality) is now a re-export shim `list(SafeBaseModel._registry)`. Drift is structurally impossible.

---

### C36. `_canonical_form()` SSoT aligns `__eq__` and `__hash__`

**Verified by**: V6 validation agent
**Files**: `lob-model-trainer/src/lobtrainer/config/base.py`
**Status**: WORKING CORRECTLY (Phase A.5.7a SB-1 fix verified)

**Verified invariant**: `SafeBaseModel.__hash__` and `SafeBaseModel.__eq__` both delegate to `self._canonical_form()` which returns `json.dumps(self.model_dump(mode="json"), sort_keys=True)`. The Python invariant `a == b ⟹ hash(a) == hash(b)` is now preserved for dict-typed fields (pre-fix, `__hash__` used dict-order-sensitive `repr()` while `__eq__` compared via dict-order-insensitive `__dict__ ==`, breaking the invariant).

---

### C37. `_partial: true` strip in `resolve_inheritance` is unconditional

**Verified by**: V6 validation agent
**Files**: `lob-model-trainer/src/lobtrainer/config/merge.py:143`
**Status**: WORKING CORRECTLY

**Verified invariant**: When `resolve_inheritance` resolves a `_base:` chain, the final composed dict has `_partial: true` deleted unconditionally before `from_dict` is invoked. A base file with `_partial: true` cannot be loaded directly (Pydantic raises on the unknown field due to `extra="forbid"`); the strip-before-from_dict makes the composition path work. Locked by parametric `test_all_partial_bases_rejected_on_direct_load` over all 22 base YAMLs.

---

### C38. `extra="forbid"` rejects typos at `model_validate`

**Verified by**: V6 validation agent
**Files**: `lob-model-trainer/src/lobtrainer/config/base.py`
**Status**: WORKING CORRECTLY

**Verified invariant**: `SafeBaseModel`'s `ConfigDict(extra="forbid")` raises `ValidationError` at `model_validate` time when an unknown field is present. The classic `horizen_idx` typo (instead of `horizon_idx`) is caught at construction with a precise error message identifying the field. This was the closure for one of the four bug-class retirements in Phase A.5 Scope D v2.

---

### C39. `frozen=True` raises on field assignment post-construction

**Verified by**: V6 validation agent
**Files**: `lob-model-trainer/src/lobtrainer/config/base.py`
**Status**: WORKING CORRECTLY

**Verified invariant**: `SafeBaseModel`'s `frozen=True` raises `ValidationError` on `config.train.X = Y` post-construction. Mutable config drift is structurally impossible. The only legitimate mutation paths are `model_copy(update=...)` (which constructs a new immutable instance) and `object.__setattr__` (which is gated behind explicit auto-derivation invariants and is locked by tests).

---

### C40. `@model_validator(mode="after")` returning `self.model_copy(update={...})` is safe

**Verified by**: V6 validation agent
**Files**: `lob-model-trainer/src/lobtrainer/config/schema.py` (ModelConfig validator)
**Status**: WORKING CORRECTLY

**Verified invariant**: Inside an `after`-mode validator, the model state is already validated. Calling `self.model_copy(update={...})` re-fires validators on the updated state, producing a coherent post-mutation instance. The pattern is documented in Phase A.5.3h. The alternative (`object.__setattr__` then return self) would skip the cross-field validators and is only used at the outer ExperimentConfig level where re-firing would loop.

---

### C41. Phase 4 R3 invariant: `PrivateAttr` cache stripped from `model_dump`

**Verified by**: V6 validation agent
**Files**: `lob-model-trainer/src/lobtrainer/config/schema.py` (DataConfig)
**Status**: WORKING CORRECTLY

**Verified invariant**: `DataConfig._feature_indices_resolved` and `_feature_set_ref_resolved` are declared as `PrivateAttr` (not regular fields). Pydantic's `model_dump()` excludes PrivateAttrs by default. YAML round-trips do not leak the resolved cache. Mutation legality under `frozen=True` is preserved because PrivateAttr mutation is the documented escape hatch.

---

### C42. `gate_reports` fingerprint exclusion

**Verified by**: V7 validation agent
**Files**: `hft-ops/src/hft_ops/ledger/dedup.py`, `hft-contracts/src/hft_contracts/experiment_record.py`
**Status**: WORKING CORRECTLY

**Verified invariant**: `compute_fingerprint` excludes `gate_reports` from the fingerprint blob. Gate outcomes are observations (recorded post-stage), not treatments (config inputs). Including them would produce ledger-conflation: the same experiment config with different gate outcomes would fingerprint differently, violating the fingerprint-as-config-identity invariant. Locked by Phase 7.4 Round 5 test asserting fingerprint stability across same-config / different-gate runs.

---

### C43. `artifacts` list fingerprint exclusion

**Verified by**: V7 validation agent
**Files**: `hft-ops/src/hft_ops/ledger/dedup.py`, `hft-contracts/src/hft_contracts/experiment_record.py`
**Status**: WORKING CORRECTLY

**Verified invariant**: `ExperimentRecord.artifacts` (list of post-stage content-addressed artifact refs) is excluded from `compute_fingerprint`. Same rationale as C42: artifacts are observations, not treatments. Locked by `test_artifacts_field_excluded_from_fingerprint` walking the full components dict and asserting the field's absence.

---

### C44. `INDEX_SCHEMA_VERSION` cascade on MAJOR.MINOR mismatch

**Verified by**: V7 validation agent
**Files**: `hft-ops/src/hft_ops/ledger/ledger.py`, `hft-contracts/src/hft_contracts/experiment_record.py`
**Status**: WORKING CORRECTLY

**Verified invariant**: `_load_index` parses the on-disk envelope's `index_schema_version` and compares MAJOR.MINOR to the constant `INDEX_SCHEMA_VERSION` from hft-contracts. On mismatch (e.g., 1.3 vs 1.4), the index is auto-rebuilt from `records/*.json` with a WARN log identifying the rebuild source. The previous silent-omission class (newly-added index_entry whitelist keys not surfacing for old records) is structurally closed.

---

### C45. Phase 8C-α RNG decorrelation across (feature, seed) pairs

**Verified by**: V7 validation agent
**Files**: `lob-model-trainer/src/lobtrainer/training/importance/permutation.py`
**Status**: WORKING CORRECTLY (Agent-D H2 fix verified)

**Verified invariant**: For permutation importance with N features × M seeds, the per-pair RNG seed is `seed_base + local_idx * config.n_seeds + seed_idx`. The cross-feature offset `local_idx * config.n_seeds` ensures feature-A's seed-0 permutation is decorrelated from feature-B's seed-0 permutation. Pre-fix, the importance estimates were rank-correlated across features, biasing the feature-importance ordering.

---

### C46. Phase 8C-α NaN baseline raises ValueError

**Verified by**: V7 validation agent
**Files**: `lob-model-trainer/src/lobtrainer/training/importance/permutation.py`
**Status**: WORKING CORRECTLY (Agent-D M2 fix verified)

**Verified invariant**: When the baseline metric (computed pre-permutation) is NaN, `compute_permutation_importance` raises ValueError with a precise diagnostic ("baseline metric is NaN — model produced unusable predictions on eval split"). Pre-fix, NaN baselines silently propagated through subtractions, producing all-NaN importance estimates. HFT rule §8 fail-loud is now enforced.

---

### C47. Atomic copy via `copy2(tmp) + os.replace(tmp, target)` for content-addressed cells

**Verified by**: V7 validation agent
**Files**: `hft-ops/src/hft_ops/ledger/ledger.py`
**Status**: WORKING CORRECTLY (Agent-C.3 H1 fix verified)

**Verified invariant**: When routing a post-stage artifact into the content-addressed ledger cell (`feature_importance/<sha>.json`), the routing performs `shutil.copy2(src, tmp_path)` followed by `os.replace(tmp_path, target_path)`. The `os.replace` is atomic on POSIX. Pre-fix, `shutil.copy2(src, target_path)` alone is NOT atomic — SIGKILL mid-copy leaves a half-written content-addressed cell, poisoning every future cache lookup. The two-phase write pattern guarantees the target either fully exists or doesn't exist.

---

### C48. `block_permutation` degenerate-null guard

**Verified by**: V7 validation agent
**Files**: `hft-metrics/src/hft_metrics/bootstrap.py`
**Status**: WORKING CORRECTLY (Agent-A H2 fix verified)

**Verified invariant**: `block_permutation(values, block_length, n_permutations, seed)` raises ValueError when `n_blocks < 2` (i.e., `block_length >= len(values) / 2`). With fewer than 2 blocks, every permutation is the identity, producing a degenerate null distribution (all permuted values equal the original). The guard fires upfront, not silently. HFT rule §8.

---

### C49. `gate_status` convention "pass/warn/fail/abort" lowercase

**Verified by**: V7 validation agent
**Files**: `hft-contracts/src/hft_contracts/gate_report.py`
**Status**: WORKING CORRECTLY (Phase 7.4 Round 5 fix verified)

**Verified invariant**: `GateReportDict.status` is constrained to the lowercase set `{"pass", "warn", "fail", "abort"}` via `GATE_STATUS_VALUES`. Any upstream gate runner using a different naming (e.g., `verdict="PASS"`) must adapt at the boundary by lowercasing. The `validation.py` adapter at line ~237 injects the lowercased `status` field BEFORE writing to `captured_metrics["gate_report"]`. The `ledger list --gate-status` filter operates on the canonical lowercase form.

---

### C50. `block_index_permutations` shared primitive in hft_metrics v0.1.3

**Verified by**: V7 validation agent
**Files**: `hft-metrics/src/hft_metrics/bootstrap.py`
**Status**: WORKING CORRECTLY (Phase 8C-α Integration Close-Out)

**Verified invariant**: `block_index_permutations(n, block_length, n_permutations, seed) → ndarray[(P, N), int64]` is the shared index-generation primitive. `block_permutation` consumes it (not duplicates it). Trainer's `compute_permutation_importance` documentedly inlines its own index generator (intentional duplicate per the trade-off in `permutation.py:94-103` — performance-critical hot path doesn't justify the import-time cost across the per-feature loop). The duplicate is documented; the upstream primitive exists for future consumers.

---

### C51. `pairwise_paired_bootstrap_compare` tail-padding fix

**Verified by**: V7 validation agent
**Files**: `hft-metrics/src/hft_metrics/pairwise.py`
**Status**: WORKING CORRECTLY (v0.1.6 fix + v0.1.7 observability verified)

**Verified invariant**: `n_blocks = math.ceil(n / block_length)` over-produces blocks then trims via `[:n]` slice. Pre-v0.1.6, `n_blocks = n // block_length` (floor division) left up to `block_length - 1` samples unresampled on every bootstrap iteration, biasing CIs narrower than the true paired-bootstrap distribution. The fix is locked by `test_no_deterministic_tail_at_n_eq_100_block_51` and `test_index_count_matches_n_at_nondivisible_block`. v0.1.7 added the observability field `PairwiseResult.n_nonfinite_replaced` to surface fallback diagnostics.

---

### C52. `compatibility_fingerprint` top-level field on `ExperimentRecord`

**Verified by**: V7 validation agent
**Files**: `hft-contracts/src/hft_contracts/experiment_record.py`
**Status**: WORKING CORRECTLY (Phase V.A.4 trust column verified)

**Verified invariant**: `ExperimentRecord.compatibility_fingerprint: Optional[str]` is a top-level field validated by `CONTENT_HASH_RE` (lowercase 64-hex SHA-256). It is projected into `index_entry()` for fast `--compatibility-fp` filter queries via `hft-ops ledger list`. The field is OBSERVATION-tier (populated post-stage by the harvester), NOT in the fingerprint blob — gates of the same model architecture against different test sets get different `compatibility_fingerprint` values without conflating in `compute_fingerprint`. `INDEX_SCHEMA_VERSION` was MINOR-bumped 1.3.0 → 1.4.0 on field addition (additive whitelist extension policy).



---

## 10. Production Impact Assessment

### Configs and experiments affected

This section maps each ACTIVE finding to the specific configs / experiments it impacts. Cross-reference with `lob-model-trainer/EXPERIMENT_INDEX.md` and `lob-backtester/BACKTEST_INDEX.md` when assessing past results.

#### Active impact (silent-wrong-result currently in shipped artifacts)

| Finding | Configs impacted | Manifestation |
|---|---|---|
| **N4** HMHP-R `horizons[0]` instead of primary_horizon_idx | `lob-model-trainer/configs/experiments/nvda_hmhp_regressor_h60.yaml` (TRUE HMHP-R + `horizon_idx: 1`). Reported `val_ic`, `val_r2`, `val_mae` describe H10, not H60. Early-stopping fires on H10 quality. Best-checkpoint selected on H10 metric. **NOT** triggered by `nvda_hmhp_40feat_h60_profit8bps_regression.yaml` — that uses `model_type: hmhp` (HMHP-classification) despite filename. | EXPERIMENT_INDEX entries for `nvda_hmhp_regressor_h60.yaml` runs are mis-labeled — unprefixed metrics describe wrong horizon. |
| **N5** HMHP vs HMHP-R pooling inconsistency | All HMHP (classification) runs: last-timestep pool. All HMHP-R (regression) runs: mean-over-T pool. Documented HMHP H10 acc=59.62% vs HMHP-R H10 R²=0.454 comparison. | Cross-task ablations confound (a) loss type, (b) decoder head, (c) **temporal aggregation**. Conclusions about "regression is better/worse than classification for HMHP" are invalid. |
| **N6** Calibrated metrics describe RAW predictions | E6 (Round 8) calibrated runs and any future `--calibrate variance_match` invocation. Specifically: `signal_metadata.json` `metrics.mae`, `metrics.rmse`, `metrics.r2` describe RAW `predicted_returns.npy` while the backtester loads `calibrated_returns.npy`. Magnitude metrics differ by data-dependent amounts (function of prediction quality, label kurtosis, scale factor); for variance-matching, RMSE relationship is approximately `RMSE_cal² ≈ 2σ_y²(1 − r_pearson)` — NOT a linear scale-factor mapping. | EXPERIMENT_INDEX/BACKTEST_INDEX cells citing MAE/RMSE/R² for E6+ calibrated runs describe wrong array. IC, DA, Pearson are correct (rank-preserving under linear monotonic transform). |

#### Imminent impact (next routine operation triggers)

| Finding | Trigger condition | What happens |
|---|---|---|
| **N1** InputContract preflight `_base:` bug | Next `hft-ops run` invocation on any composed experiment (E5_*, HMHP_*, etc.) | Pipeline halts at preflight with `ValueError("missing model.model_type")` BEFORE the GPU subprocess. No experiment can run via the orchestrator until fixed. |
| **N2** `--resume` epoch counter | Any `--resume` invocation | Trainer redoes N already-trained epochs over resumed model. No production run has used --resume since Phase 7.5 dry-run came online (no checkpoint to resume from), but next user attempt corrupts the model. |

#### Dormant-primed impact (specific future trigger)

| Finding | Trigger condition |
|---|---|
| **N7** Norm stats not bound | Any researcher re-exporting dataset between training and `export_signals.py` |
| **N8** TLOB flatten order | Any attempt to load an official Berti & Kasneci checkpoint |
| **D1, D2, D3** Pydantic staleness | Any new CLI flag exposing `labeling_strategy`/`horizon_idx`/`tlob_hidden_dim` etc. via `model_copy(update=...)` programmatic mutation |
| **D4** Sample weights pre-trim | Any multi-source (T12) experiment |
| **D6** Day-1 schema check | Any mixed-export training set |

### Trustworthiness assessment of past artifacts

| Artifact category | Status | Notes |
|---|---|---|
| HMHP-R H60-primary unprefixed metrics | **Wrong** | Describes H10 not H60 (N4) |
| Calibrated MAE/RMSE/R² in signal_metadata.json | **Wrong (describes raw array, not calibrated; difference is data-dependent — see N6)** | E6 / Round 8 + future calibrated runs (N6) |
| Calibrated IC/DA/Pearson | **Correct** | Rank-preserving under linear monotonic transform |
| All HMHP vs HMHP-R cross-task comparisons | **Confounded** | Pooling mismatch (N5) |
| All single-horizon (non-HMHP) regression results | **Trustworthy** | No primary_horizon_idx involvement |
| All resumed-training results | **Untrustworthy** | --resume is broken (N2 + N3) |
| All from-scratch training results | **Trustworthy** | Resume bugs don't apply |

---

## 11. Suggested Priority Tiers (planning input — NOT a fix plan)

This section provides a planning input. Actual fix sequencing must be informed by separate cost-benefit analysis. The user's explicit directive is "no quick fixes; long-term design first" — every recommendation below should be evaluated against that lens.

### Tier 1 — Single-line ship-blocker (unblocks first live experiment)

- **N1 InputContract preflight `_base:` bug** — replace `yaml.safe_load(f)` with a resolution call that handles `_base:` inheritance, then add 1 regression test using a real composed manifest (e.g., `e5_60s_huber_cvml.yaml`). UNBLOCKS first live `hft-ops run`.
  - **Architectural caveat (added Pass 5 from meta-validation 2026-04-27)**: the simplest-looking fix would import `lobtrainer.config.load_config` or `lobtrainer.config.merge.resolve_inheritance` — but doing so introduces a cross-module dependency from hft-ops onto lob-model-trainer that the L2.6 AST regression test was specifically designed to prevent (hft-ops must remain torch-free; importing trainer modules pulls torch via the `__init__.py` side-effects). The principled fix extracts `merge.resolve_inheritance` into `hft-contracts` (which IS torch-free) and has BOTH hft-ops preflight AND trainer load-path import from there. Effort estimates: minimal-but-ugly fix (yaml-only re-implementation in hft-ops) is ~1 hour; principled fix (extract to hft-contracts) is ~3-4 hours including tests and CI verification across 3 repos. Pick based on cycle goals; prior framing of "single-line ~30 min" understated the architectural surface.

### Tier 2 — Active correctness in shipped artifacts (retroactive impact)

- **N6 Calibrated metrics raw vs calibrated** — fix metrics_dict computation to operate on calibrated array when calibration_method is set. Add `metrics_raw` companion field. Past EXPERIMENT_INDEX cells need flagging.
- **N4 HMHP-R primary_horizon threading** — replace `horizons[0]` at 4 sites with `horizons[validate_primary_horizon_idx_for(...)]`. Re-run `nvda_hmhp_regressor_h60.yaml` (the TRUE HMHP-R trigger) to recover correct H60 metrics.

### Tier 3 — Active correctness for future experiments

- **N2 + N3 Resume semantics** — Define `Callback.state_dict()/load_state_dict()` interface; restore RNG state; fix epoch counter. Add `test_resume_equals_fresh_training` parametric test.
- **N5 HMHP/HMHP-R pooling alignment** — add `pool_mode` to HMHPConfig. Pick last vs mean via empirical A/B (Phase I.B.0, deferred). Lock with regression test.

### Tier 4 — Latent / hardening

- **N7 Norm stats binding** — content-address stats; checkpoint stores hash; assert at load.
- **F4 BacktestData NaN validation** — add finite-check on all optional fields. Replace silent-zero substitution with raise.
- **F5 Backtester DataLoader RAM exhaustion** — add `mmap_mode='r'` + lazy-concatenation iterator; rearchitect `LoadedData` for virtual indexing. Empirical correction (V3 audit 2026-04-27): per-day load is ~48.8MB (NOT the 100MB cited in F5 entry); 163 days × 48.8MB ≈ 8GB raw + ~16GB peak during concat transient. The peak claim of "~16-20GB" is still defensible during concat phase.
- **N8 TLOB final-flatten** — only if cross-codebase reproduction matters.
- **block_length_samples=1 default** — change default OR add WARN.

### Tier 5 — Defense-in-depth (defer until informed by experiment)

- D1-D9 dormant findings.
- F2/F3/F6/F7 broad-except + non-atomic-write hardening (with extensions: F2 also covers `train.py:415-423` evaluate-only-branch duplicate; F6 also covers `schema.py:2361-2364` `to_json` sibling).
- F8 + F9 broad-except observability gaps (lower priority than F2/F3 because they only affect post-training importance + class-weight computation, not core training correctness).
- Additional broad-except sites enumerated in P6 scope note (`callback.py:557`, `dedup.py` x4, `ledger.py` x4, `feature_sets/producer.py` x2 — case-by-case triage).
- Architectural retirements (P1-P6) — design work, not patching.

### Out of scope for this audit cycle

- Documentation drift fixes (e.g., `min_ratio_vs_prior_best` 0.9 vs 0.95).
- Test coverage expansion beyond regressions for the above fixes.
- Performance optimizations.

---

## Appendix A — Reproducer Scripts

Each ACTIVE finding has a reproducer in its Section 4 entry. This appendix collects the most consequential ones for quick reference.

```python
# N2: --resume epoch counter ignored
# Run inside /Users/knight/code_local/HFT-pipeline-v2/lob-model-trainer
import torch
from unittest.mock import MagicMock
from lobtrainer.training.trainer import Trainer
# Construct trainer + load checkpoint at epoch 4
# Then call train() with cfg.epochs=10
# Observe: epochs 0-4 re-run on already-trained model
```

```python
# N5: HMHP vs HMHP-R pooling cosine similarity
import torch
torch.manual_seed(0)
shared_repr = torch.randn(2, 100, 64)
pool_hmhp = shared_repr[:, -1, :]
pool_hmhp_r = shared_repr.mean(dim=1)
cos_sim = torch.nn.functional.cosine_similarity(pool_hmhp, pool_hmhp_r, dim=-1)
print(f"Cosine similarity: {cos_sim}")  # expected: tensor([0.177, 0.196])
```

```python
# N6: Calibrated metrics describe raw predictions
# Read any signal_metadata.json from a calibrated run
# Compare metrics.mae to MAE computed on calibrated_returns.npy
import json, numpy as np
manifest = json.load(open("data/exports/.../signal_metadata.json"))
preds_raw = np.load(".../predicted_returns.npy")
preds_cal = np.load(".../calibrated_returns.npy")
labels = np.load(".../regression_labels.npy")
print(f"Manifest MAE: {manifest['metrics']['mae']}")
print(f"Raw MAE:      {np.abs(preds_raw - labels).mean()}")
print(f"Calibrated MAE: {np.abs(preds_cal - labels).mean()}")
# Expected: Manifest MAE matches Raw MAE, NOT Calibrated MAE
```

```python
# D1: DataConfig.labels staleness
from lobtrainer.config.schema import DataConfig, LabelingStrategy
cfg = DataConfig(labeling_strategy=LabelingStrategy.TLOB, horizon_idx=0)
print(cfg.labels.task)  # classification
new = cfg.model_copy(update={"labeling_strategy": LabelingStrategy.REGRESSION, "horizon_idx": 2})
print(new.labels.task)  # STILL classification (STALE)
```

```python
# D2: ModelConfig.params staleness
from lobtrainer.config.schema import ModelConfig, ModelType
cfg = ModelConfig(model_type=ModelType.TLOB, tlob_hidden_dim=64, input_size=98)
print(cfg.params["hidden_dim"])  # 64
new = cfg.model_copy(update={"tlob_hidden_dim": 256})
print(new.params["hidden_dim"])  # STILL 64 (STALE)
```

```python
# D8 (refuted F-4): Demonstrate the gate fallback uses single metric
# (this REFUTES Round 1 Agent 6 CRIT-4)
from hft_ops.stages.post_training_gate import _select_primary_metric
metrics_a = {"test_ic": 0.05, "best_val_ic": 0.20}
metrics_b = {"best_val_ic": 0.13}  # no test_ic
name_a, val_a = _select_primary_metric(metrics_a, configured_name=None)
name_b, val_b = _select_primary_metric(metrics_b, configured_name=None)
# name_a = "test_ic", val_a = 0.05
# name_b = "best_val_ic", val_b = 0.13
# When the gate compares B against A's prior, it queries ledger for "best_val_ic"
# (B's metric_name), so it gets A's best_val_ic=0.20, NOT A's test_ic=0.05.
# The cascade is sound.
```

---

## Appendix B — File:Line Citation Table

This table indexes every cited file:line by finding ID for fast cross-reference.

| File | Lines | Findings |
|---|---|---|
| `hft-ops/src/hft_ops/stages/contract_preflight.py` | 287-311 | N1 |
| `hft-ops/src/hft_ops/stages/training.py` | 100-124, 121-124, 207-209, 332-344, 379, 471-487 | N1, F3, F7 |
| `lob-model-trainer/src/lobtrainer/config/merge.py` | 85-118, 143, 185-204 | N1 |
| `lob-model-trainer/src/lobtrainer/training/trainer.py` | 583-608, 672, 752, 802, 808-809, 877, 982-1004, 1007-1033, 1026 | N2, N3, N7, F2 |
| `lob-model-trainer/src/lobtrainer/training/callbacks.py` | 246-255, 390-400, 534-536, 49-117 | N3 |
| `lob-model-trainer/src/lobtrainer/training/strategies/hmhp_regression.py` | 157, 217-219, 254, 294, 318-337 | N4 |
| `lob-model-trainer/src/lobtrainer/training/strategies/hmhp_classification.py` | 127, 233, 277, 297 | (PARTIAL — convention) |
| `lob-models/src/lobmodels/models/hmhp.py` | 212-218, 276, 451-458, 475-488, 520-530, 705, 747-816, 846-866, 858-866, 874-893 | N5, P3 |
| `lob-models/src/lobmodels/models/hmhp_regressor.py` | 111, 159, 161-163, 185-186, 188-215, 376-380, 390-432 | N5, P3 |
| `lob-model-trainer/src/lobtrainer/export/exporter.py` | 132, 362-364, 401, 408, 411-465, 443-451, 477-478, 494-504, 502, 631-632, 644, 649-652, 666-694, 682-694 | N6, F-1 |
| `lob-models/src/lobmodels/models/tlob.py` | 235 | N8 |
| `TLOB/models/tlob.py` | 117-118 | N8 (reference) |
| `lob-model-trainer/src/lobtrainer/config/schema.py` | 1272-1284, 1566-1567, 2213, 2238, 2246, 2402-2412, 2474-2484, 2522-2549, 2580+, 2623 | D1, D2, F-2 |
| `lob-model-trainer/src/lobtrainer/config/base.py` | 329-374, 370-373, 262-287 | D3 |
| `lob-model-trainer/src/lobtrainer/data/dataset.py` | 117-161, 336-341, 678-693, 819-824, 840-880, 1278, 1311, 1336-1339, 1467-1486 | D4, D6, F-3 |
| `lob-model-trainer/src/lobtrainer/data/bundle.py` | 394-398 | D4 |
| `lob-model-trainer/src/lobtrainer/data/normalization.py` | 97-106, 673-677 | D5 |
| `hft-feature-evaluator/src/hft_evaluator/fast_gate.py` | 345-346, 379-380, 382-385, 501, 512 | D7 |
| `hft-ops/src/hft_ops/stages/post_training_gate.py` | 62-84, 329-338, 477-483, 518-577, 614-625, 710-727, 752-794, 763-794, 846 | D8, F-4 |
| `lob-model-trainer/src/lobtrainer/utils/reproducibility.py` | 90-96, 104-149, 211-267 | D9 |
| `lob-model-trainer/scripts/train.py` | 162-187, 237-239, 317-319, 360-362, 397-403, 422, 446, 450, 455 | F2, F6 |
| `lob-backtester/src/lobbacktest/engine/vectorized.py` | 66-75, 161, 175-200, 180-184, 445-447, 698-722, 721-722 | F4 |
| `lob-backtester/src/lobbacktest/data/loader.py` | 188-248, 93, 134, 107, 146 | F5 |
| `lob-model-trainer/src/lobtrainer/training/importance/callback.py` | 116, 140-155, 215-252, 294-312, 304-311, 366, 422-425, 509, 517, 524-528, 557, 587-597 | F8 |
| `lob-model-trainer/src/lobtrainer/training/strategies/classification.py` | 39, 46, 131-153, 140-143, 151-153, 159-208, 187, 255 | F9 |
| `lob-backtester/scripts/run_regression_backtest.py` | 140-149, 173-181, 237-243, 241 | (PARTIAL DORMANT) |
| `lob-backtester/scripts/run_spread_signal_backtest.py` | 256 | (related to above) |
| `hft-contracts/src/hft_contracts/compatibility.py` | 268-330 | C31 compute_label_strategy_hash Pydantic branch (corrected from prior canonical_hash.py:318-339 which was out-of-range — file has 158 lines) |
| `hft-contracts/src/hft_contracts/signal_manifest.py` | 248-257, 439, 482, 622-625, 670-682 | (cleared) |
| `hft-metrics/src/hft_metrics/purged_cv.py` | 26-29, 88-98 | (DESIGN-CHOICE) |
| `hft-metrics/src/hft_metrics/sample_weights.py` | 67, 117, 119, 144-150, 187-189, 189 | (cleared) |
| `hft-metrics/src/hft_metrics/bootstrap.py` | 113-115, 122-130, 141, 179-180 | (cleared) |
| `lob-model-trainer/src/lobtrainer/training/cv_trainer.py` | 159, 185, 191-227, 256-264 | (cleared embargo + fold isolation) |

---

## Appendix C — Test Coverage Gaps

Tests that DO NOT exist but should, ranked by impact:

| Priority | Test | Would catch | Effort |
|---|---|---|---|
| **P0** | `test_resume_equals_fresh_training` | N2 + N3 + D9 (resume semantics) | High |
| **P0** | `test_preflight_resolves_base_inheritance` | N1 | Low |
| **P0** | `test_calibrated_metrics_describe_calibrated_array` | N6 | Low |
| **P1** | `test_hmhp_r_primary_metric_horizon` (parametric over primary_horizon_idx ∈ {0, 1, 2}) | N4 | Low |
| **P1** | `test_hmhp_pool_mode_consistency` | N5 (after fix adds pool_mode) | Low |
| **P1** | `test_normalization_stats_sha256_locked_to_checkpoint` | N7 | Medium |
| **P2** | `test_model_copy_update_does_not_leak_stale_derived_fields` (parametric over D1/D2/D3 patterns) | D1, D2, D3 | Medium |
| **P2** | `test_compute_label_strategy_hash_byte_stability_across_classvar_extensions` | (defense-in-depth) | Low |
| **P2** | `test_backtest_data_rejects_nan_in_predictions` | F4 | Low |
| **P3** | `test_train_py_writes_test_metrics_atomically` | F2, F3, F6 | Low |
| **P3** | `test_dataloader_concatenation_memory_bounded` | F5 | Medium |
| **P3** | `test_regression_labels_column_order_matches_metadata` | partial-confirm-7 | Low |

---

## Appendix D — Cross-Reference: Round 1 vs Round 2 Verdicts

| Round 1 (forensic) ID | Round 1 verdict | Round 2 (validation) verdict | Disposition |
|---|---|---|---|
| Agent 1 CRIT-1 | CRITICAL | TRUE — DORMANT (HMHP-cls convention) | PARTIAL → renamed N4 (HMHP-R only is ACTIVE) |
| Agent 1 CRIT-2 | CRITICAL (one-sided embargo) | TRUE but DESIGN-CHOICE | Refuted as "bug" — design rationale documented |
| Agent 1 CRIT-3, CRIT-4 | CRITICAL | TRUE DORMANT | D4 |
| Agent 1 CRIT-5 | CRITICAL | TRUE | N7 |
| Agent 1 CRIT-6, 7 | downgraded | confirmed downgraded | (incorporated into D-series) |
| Agent 1 CRIT-8 | HIGH | TRUE DORMANT | D5 |
| Agent 2 C1 | CRITICAL | TRUE (HMHP-R) | N4 |
| Agent 2 C2 | CRITICAL | TRUE | N3 |
| Agent 2 C3 | CRITICAL | PARTIAL (deterministic-but-misaligned) | D9 (composes with N2) |
| Agent 3 C1 | CRITICAL | TRUE numerically | N5 |
| Agent 3 C2 | CRITICAL | TRUE | N5 (Phase III.A deferred) |
| Agent 3 C3 | HIGH | TRUE | N8 |
| Agent 4 C1 | CRITICAL | PARTIAL (HMHP-cls convention, dormant) | (incorporated as PARTIAL) |
| Agent 4 C2 | CRITICAL | TRUE DORMANT (no Round used --primary-horizon-idx flag yet) | (incorporated as DORMANT-PRIMED) |
| Agent 4 C3 | CRITICAL | PARTIAL (IC/DA correct; MAE/RMSE/R² wrong) | N6 |
| Agent 5 CRIT-1, 2 | CRITICAL | TRUE DORMANT | D1, D2 |
| Agent 5 CRIT-3 | CRITICAL | FALSE (fail-loud protected) | F-2 |
| Agent 5 H-3 | HIGH | TRUE DORMANT | D3 |
| Agent 6 CRIT-1 | CRITICAL | TRUE (design-choice w/ trade-off) | (Tier 4) |
| Agent 6 CRIT-3 | CRITICAL | TRUE (silent gate bypass) | (variant of N1 for orchestrator) |
| Agent 6 CRIT-4 | CRITICAL | FALSE | F-4 |
| Agent 6 HIGH-2 | HIGH | FALSE | F-1 |
| Agent 6 HIGH-4 | HIGH | FALSE | F-5 |
| (NEW Round 2) | — | NEW | N1, F2-F9 (N2 was first reported Round 5 forensic; CONFIRMED in Round 2 validation, NOT new — see E5/E6 in Revision History) |

---

## Document Maintenance

### Revision history

- **REV 1 (2026-04-26)**: initial synthesis from 13-agent audit + validation cycle (6 forensic round 1 + 7 validation round 2).
- **REV 2 (2026-04-27)**: post-meta-validation corrections applied. A 7-agent meta-validation round re-verified every claim in the document against actual code. This revision applies the material corrections found by that meta-validation:
  - **E1**: N4 production impact attribution corrected — true HMHP-R trigger config is `nvda_hmhp_regressor_h60.yaml` (uses `_base: hmhp_cascade_regression`, `model_type: hmhp_regression`, `horizon_idx: 1`). The misleadingly-named `nvda_hmhp_40feat_h60_profit8bps_regression.yaml` was previously cited but actually uses `model_type: hmhp` (HMHP-classification with auxiliary regression head, NOT HMHP-R), so `HMHPRegressionStrategy` is never instantiated for it. Verified by direct YAML inspection.
  - **E3**: N6 numerical claim about magnitude metrics "off by ~scale_factor (E5 ≈3.73)" replaced with data-dependent language. The oversimplified worked example "MAE 5 → 18.6 bps" was removed; the correct mathematical relationship under variance-matching is `RMSE_cal² ≈ 2σ_y²(1 − r_pearson)`, not a linear scale-factor mapping. Core claim (calibrated metrics describe wrong array) is unchanged.
  - **E4**: Section 3 ↔ Section 4 severity inconsistencies resolved — N7 verdict table changed IMMINENT → DORMANT-PRIMED to match Section 4 entry; N8 Section 4 entry changed ACTIVE → DORMANT-PRIMED to match Section 3 verdict table.
  - **E5**: N2 verdict table tag corrected from "ACTIVE NEW" to "ACTIVE" (N2 was first reported in Round 5 forensic audit and re-confirmed in validation, NOT new to validation).
  - **E6**: Methodology "11 NEW findings" → "9 NEW findings (1 critical N1 + 8 cross-module F2-F9)". Executive Summary "2 of these missed" → "1 of these missed by the prior audit; N2 was first reported in Round 5".
  - **E7**: Round 9 phantom citation in N6 production impact removed (only Round 8 exists in BACKTEST_INDEX.md).
  - **E8**: Citation table error `canonical_hash.py:318-339` (out of range — file has 158 lines) corrected to `compatibility.py:268-330` (where Pydantic branch actually lives).
  - **C8, C11, C13, C15, C16, C17, C18, C19, C20, C22, C24, C31, C33**: 13 cleared-item descriptions corrected. C8 FRESH-2 terminology ("classification heads" — HMHP-R has none); C11 try/finally claim removed (no such pattern exists; functional safety via `_train_epoch.model.train()`); C15 BiN parameter description rewritten (no gamma↔beta swap; actual rename is reference's `y1, y2` → our `gamma_feature, gamma_temporal`); C18 GMADL `a` and `b` roles unswapped (`a` = sigmoid sharpness, `b` = magnitude exponent); C20 heteroscedastic parameterization corrected (Softplus direct variance, NOT `exp(log_var)`); C31 hash dispatch mechanism corrected (`hasattr(model_dump)` duck-typing, NOT `isinstance(BaseModel)`). Plus 11 file-path drift corrections (post-refactor: lob-models flat → hierarchical subpackages).
  - **P3 Bug 9**: optimizer device portability claim downgraded from full bug to "partially mitigated" — `torch.load(map_location=self.device)` at trainer.py:1015 handles standard cross-device cases; bug is real only for distributed training / stream-bound CUDA buffers.
  - **P6**: F2 + F3 catch types corrected (F2 uses `except ValueError`, NOT `except Exception: pass`; F3 uses `except (json.JSONDecodeError, OSError)`, NOT "catches all exceptions"). Scope expanded — 48 broad-except total instances enumerated across 3 modules (5 representative + 43 worth case-by-case triage).
  - **F2, F6, F8 extensions**: sibling sites added (F2 → `train.py:415-423` evaluate-only-branch duplicate; F6 → `schema.py:2361-2364` `to_json` sibling; F8 → `callback.py:557` enum-resolution cousin).
  - **Tier 5**: F8 + F9 added to coverage (previously missing from any tier listing).
  - **Section 6 numbering note**: F1 → N1 promotion explained (no missing finding — F1 was promoted to CRITICAL during synthesis and renumbered).
  - **Word count**: "~30,000 words" → "~26,000 words" (matches actual `wc -w` count).
  - **Cleared count**: "25+ entries" → "52 entries" in Section 9 layout table reference.

REV 2 corrections were validated by 7 parallel adversarial agents on 2026-04-27, each scoped to a distinct subset (N1-N4, N5-N8, D1-D9, F2-F9, F-1 to F-5 + P1-P6, C-series, framing/appendices). REV 2.1 (also 2026-04-27) applied 9 additional residual fixes found by a final 5-agent meta-validation pass: (5-1) trustworthiness table line 2942 "Wrong by ~scale_factor" + "Rounds 8-9" replaced with data-dependent language; (5-2) N6 production-impact "R9" reference cleaned up; (5-3, 5-4) Executive Summary + Verdict Summary footer "25+ CLEARED" → "52 CLEARED"; (5-5) F5 added to Tier 4 (was missing from any tier); (5-6) Appendix D added to Document Layout table; (5-7) Appendix D "NEW Round 2" row removed N2 (E6 noted N2 was first reported in Round 5, NOT new); (5-8) P1 cited `hmhp.py:705` corrected to `models/hmhp.py:705` (path-drift class consistent with REV 2's C-series corrections); (5-9) Tier 1 N1 "single-line ~30 min" framing supplemented with architectural caveat about cross-module dependency surface and L2.6 torch-free invariant. The 7-agent REV 2 validation found:
- 7 of 8 ACTIVE CRITICAL findings (N1-N3, N5-N8) — fully confirmed, no material correction needed
- 1 of 8 ACTIVE CRITICAL (N4) — substantive bug exists exactly as described, but production-impact attribution was wrong (corrected via E1)
- 9 of 9 DORMANT findings — fully confirmed (all dormancy claims verified by independent grep)
- 8 of 8 V7 NEW findings — fully confirmed (3 with extension sites added)
- 5 of 5 REFUTED claims — verified TRULY refuted (no hidden bugs behind refutations)
- 52 cleared items — all functionally sound; ~13 description inaccuracies corrected
- 6 architectural patterns — 4 confirmed exactly, 2 (P3, P6) refined per E-corrections above

After REV 2.1 (which applied final residual fixes from a 5-agent meta-validation), the document is FROZEN as of 2026-04-27. It captures the state of findings at this point in time. Any future code changes should be reflected by:

1. A SEPARATE document (e.g., `TRAINING_PIPELINE_FORENSIC_AUDIT_2026_05_XX.md`) for a future audit cycle.
2. OR: a "Disposition Update" section appended below as findings are addressed.

Do NOT silently edit this document to reflect post-audit fixes. The frozen baseline is the value.

**Next planned cycle**: after first live experiment surfaces real-data issues. The validation step explicitly recommends running the experiment first, then revisiting this document for fix planning.

---

**Document compiled by**: 13-agent audit + validation cycle (2026-04-25, 2026-04-26) + 7-agent meta-validation revision (2026-04-27, REV 2) + 5-agent final-pass meta-validation (2026-04-27, REV 2.1)
**Synthesis author**: claude-opus-4-7[1m] orchestrator
**Status**: AUTHORITATIVE (REV 2.1) — supersedes prior agent reports and prior revisions of this document

