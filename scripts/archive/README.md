# Trainer Scripts — Archived Fossils (Phase 6 6D, 2026-04-17)

**STATUS**: Experimental fossils — NOT templates for new work.

Per [hft-rules §4](/.claude/rules/hft-rules.md#4-modularity--genericity),
new experiments MUST be authored as:

1. **hft-ops manifests** at `hft-ops/experiments/<name>.yaml` (single runs)
   or `hft-ops/experiments/sweeps/<name>.yaml` (parameter grids), OR
2. **Library modules** at `hft_evaluator.experiments.*` /
   `lobtrainer.experiments.*` with a `run(config) -> result` entry point.

New files under `scripts/` are reserved for:
- Production infra (`train.py`, `export_signals.py`,
  `precompute_norm_stats.py`).
- Data-prep utilities (must begin with the header
  `# DATA PREP UTILITY — not an experiment`).

## Archived Files (this directory)

| File | Original Purpose | Retired In | Replacement |
|---|---|---|---|
| `e4_baselines.py` | E4 baseline sweep (Ridge + GradBoost + XGBoost @ H60 5s) | Phase 6 6D | `hft-ops/experiments/sweeps/e5_phase2_sweep.yaml` (template for new baselines) |
| `e5_baselines.py` | E5 baseline sweep (TemporalRidge / GradBoost / XGBoost @ H10 60s) | Phase 6 6D | Same — use sweep manifests |
| `run_simple_model_ablation.py` | Ad-hoc feature-ablation runner | Phase 6 6D | Authoring sweep manifests under `hft-ops/experiments/sweeps/` |
| `run_simple_training.py` | Single-experiment CLI wrapper (pre-hft-ops) | Phase 6 6D | `hft-ops run <manifest.yaml>` |
| `run_experiment_spec.py` | Declarative ExperimentSpec invoker (Phase 2b) | Phase 6 6D | `hft-ops run` (Phase 2 ValidationStage + full orchestration) |

## Why Archived Rather Than Deleted?

1. **Reproducibility**: each script reproduces a historical experiment
   (E4/E5 series, ablations). Git `git mv` preserves full history so
   the logic is still recoverable.
2. **Teaching value**: the scripts document methods that shipped findings
   in `EXPERIMENT_INDEX.md`. They're citable by `memory/project_*` files.
3. **Preventing imitation**: LLM coders surveying the repo for templates
   will see the fossil header (`# STATUS: experimental fossil — NOT a
   template for new work.`) at the top of each script and be steered
   away from reviving the ad-hoc-script pattern.

## Restoration

If a genuine need arises to re-run an archived script:

```bash
# Check out the script at its archived path:
python lob-model-trainer/scripts/archive/<name>.py --help
```

They are NOT deleted — just relocated. For NEW experiments, author a
manifest under `hft-ops/experiments/` (or `experiments/sweeps/` for
grid search) and let the orchestrator compose + validate + record +
fingerprint-dedup + emit signals + backtest.
