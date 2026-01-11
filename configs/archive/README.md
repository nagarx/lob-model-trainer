# Archived Configurations

These configurations are from earlier development phases. They may be useful as references
but are not actively maintained.

## Files

| Config | Model | Status | Notes |
|--------|-------|--------|-------|
| `baseline_lstm.yaml` | LSTM | ✅ Works | Full LSTM baseline, 100 epochs |
| `baseline_lstm_quick.yaml` | LSTM | ✅ Works | Quick LSTM, 20 epochs |
| `lstm_attn_bidir_h20.yaml` | LSTM+Attention | ✅ Works | Bidirectional with attention |
| `xgboost_baseline.yaml` | XGBoost | ⚠️ Needs update | Uses old dataset path |

## Why Archived

These configs were used during the initial development phase before the focus shifted
to DeepLOB experiments. The DeepLOB architecture more closely follows the research
paper implementation and is the primary focus for NVDA directional prediction.

## Usage

If you need to use these configs, first verify:
1. `data_dir` points to an existing dataset (e.g., `../data/exports/nvda_balanced`)
2. All required model parameters are present in the schema
3. The output directory doesn't conflict with active experiments

## Active Configs

For current experiments, see:
- `../deeplob_benchmark.yaml` - DeepLOB base template
- `../deeplob_benchmark_h100.yaml` - Paper benchmark (h=100)
- `../experiments/` - Active experiment configs

