# Contributing to lob-model-trainer

> **Created**: Cycle 11 (2026-05-27) per Option δ Phase 1 implementation. Per `#PY-NEW-CONSUMPTION-ENFORCEMENT` closure criterion.

This document captures contribution discipline specific to `lob-model-trainer` that is NOT in `CODEBASE.md` (which describes architecture + technical contracts).

## Authoritative References

- **Architecture + contracts**: `CODEBASE.md` (this repo, 81-section reference, 1,636 lines)
- **Experiment cycle reference**: `EXPERIMENT_INDEX.md` (this repo, R-NN ledger)
- **Cross-pipeline findings**: `reports/CONSOLIDATED_FINDINGS_2026_05.md` (this repo)
- **Pipeline architecture deep-dive**: `../PIPELINE_ARCHITECTURE.md` §11 (~400 LOC trainer section)
- **Theoretical backbone**: `../hft-wiki/research/theory/` (18 entries as of 2026-05-27)
- **Wiki consultation playbook**: `../hft-wiki/playbooks/record-experiment-result.md`

## `wiki_consultation:` Discipline (REQUIRED post-Cycle-11)

Every NEW R-NN entry in `EXPERIMENT_INDEX.md` authored after Cycle 11 ship (2026-05-27) MUST include a `**Wiki consultation**` block citing relevant `theory:` / `synthesis:` / `FINDING-` IDs from `hft-wiki`. This is the consumer side of `#PY-NEW-CONSUMPTION-ENFORCEMENT` (TIER 2 HIGH) — currently cycle 7 of UNMET observation window (closure criterion: R ≥ 20% organic sustained ≥ 3 cycles OR TIER 1 escalation at Cycle 13).

### Field Format

```markdown
**Wiki consultation** (REQUIRED — list theory: / synthesis: / FINDING- IDs reviewed before running):
- `theory:<slug>` — <one-line justification, ≥ 20 chars>
- `synthesis:<slug>` — <one-line justification>
- `FINDING-NNN-<slug>` — <known anti-pattern context>

— OR explicit negative-result fallback:

- **None applicable** — queried `hft-wiki list theory --tag=<X>` returned 0 matches against this experiment's substance scope `<X>`.
```

### Requirements

| Requirement | Hard? | Notes |
|---|---|---|
| **Block PRESENCE** in markdown table OR as dedicated `**Wiki consultation**` section | REQUIRED post-Cycle-11 | Validator WARNs if absent (default exit 0) |
| **Block CONTENT** | REQUIRED | Either ≥1 cited ID OR "None applicable" fallback |
| **Justification length** per cite | SOFT ≥ 20 chars | Validator WARNs below threshold |
| **ID resolution** | SOFT (validator opt-in via `--strict`) | Runs `hft-wiki show <id>` per cite; WARN on resolution failure |
| **Citation completeness** | NOT enforced; operator judgment | Cite IDs that ACTUALLY informed design — not symbolic compliance |

### Grandfathering

Pre-Cycle-11 entries (R1-R20 + all Validation P0/E1-E16 / F1 / Phase Q.6.5 / Cycle 10/12) are GRANDFATHERED and exempt. The validator skips with INFO note. Retrofit is OPTIONAL (Cycle 13+ batch retrofit deferred).

### Worked Example

See `../hft-wiki/playbooks/record-experiment-result.md` §"Worked Example — Retrofit R20" for a 4-citation worked example using Cycle 8/9/10 entries (Welford + block-bootstrap + Spearman + FINDING-008).

### Running the Soft Validator

```bash
cd lob-model-trainer
python3 scripts/check_experiment_index_completeness.py            # WARN-not-ERROR (exit 0)
python3 scripts/check_experiment_index_completeness.py --verbose  # per-entry detail
python3 scripts/check_experiment_index_completeness.py --strict   # WARN → exit 1
python3 scripts/check_experiment_index_completeness.py --json     # machine-readable output
```

Run BEFORE every commit that adds a new R-NN entry. Not yet wired to pre-commit hooks (no `.pre-commit-config.yaml` exists in this repo as of Cycle 11; Phase 2 will consider CI integration).

## Failure Modes

- **Fake-compliance**: filling the block with stale/generic cites that don't justify design. **Cycle-close PR review** is the primary detection; `consumption_ratio.py --strict` measures organic-vs-backfill ratio over time.
- **Block schema-vs-impl divergence**: if the validator script's expected format diverges from documented format here, the validator wins (it's the SSoT for what counts as "compliant"). Update this doc to match the validator.
- **Validator unavailable in CI**: validator is opt-in operator-run helper; no CI gate. Operators MUST run manually for now.

## What NOT to Cite

- Hotfix entries (per hft-rules §13 exception clause).
- Pure infrastructure changes (CI fix, dep bump, doc rewording).
- Trivial bug fixes (`fix: typo in comment`).

## Cycle 11 Reasoning

This `wiki_consultation:` discipline was introduced via Cycle 11 Option δ Phase 1 implementation (#PY-NEW-CONSUMPTION-ENFORCEMENT closure attempt) after 6 consecutive cycles of 0% organic wiki consumption in canonical R-NN ledgers. Per `consumption_ratio.py` L106 disclosure, the empirical truth was that 18 substance entries existed in `hft-wiki/research/theory/` but no R-NN cycle organically cited them — Cycles 7/9/10 backfilled cites in `reports/CONSOLIDATED_FINDINGS_2026_05.md` but EXPERIMENT_INDEX / BACKTEST_INDEX / EXPORT_INDEX remained at zero.

Cycle 11 designs this consumer-side forcing function so that Cycle 12 + Cycle 13 R-NN authors can produce ORGANIC citations (not backfilled). If Cycles 12+13 ship < 20% organic, TIER 1 escalation triggers per closure criterion.

## Related Documents

- `EXPERIMENT_INDEX.md` — primary target ledger; per-entry template embedded at top of file
- `../hft-wiki/playbooks/record-experiment-result.md` — operator workflow playbook
- `../hft-wiki/scripts/consumption_ratio.py` — operator-runnable Goodhart trajectory measurement
- `../PHASE_P_BACKLOG.md #PY-NEW-CONSUMPTION-ENFORCEMENT` — closure criterion (root-level local-only)
- `../hft-wiki/meta/2026-05-26-option-delta-design.md` — original design doc (Cycle 9 DRAFT + Cycle 10 L106 in-place revision)
