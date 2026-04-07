# Research Implementation Plan

> **Cost correction (2026-03-18)**: Deep ITM breakeven was corrected from 0.7-0.8 bps to **1.4 bps** (spread was missing from the original calculation). See `IBKR-transactions-trades/COST_AUDIT_2026_03.md`.

**Date:** 2026-03-16
**Source:** `regression_research/RESEARCH_SYNTHESIS_REPORT.md`
**Scope:** Per-repo changes required to implement the research-driven pipeline evolution

---

## Phase 1: Statistical Analysis (Before Any Code Changes)

### 1A. EffectiveHorizonAnalyzer (lob-dataset-analyzer)

**Purpose:** Determine the optimal `event_count` and `horizons` for the feature extractor by measuring at what raw-event horizon OFI signal peaks and decays on our actual NVDA XNAS data.

**Method:** Using raw sequences (T=100, each at 1000-event sampling), compute the correlation between OFI features at the last timestep and mid-price returns at various raw-event offsets within the sequence window.

**Location:** `lob-dataset-analyzer/src/lobanalyzer/analysis/regression/effective_horizon.py`

**Source:** Kolm, Turiel & Westray (2023), Section 4.2 -- effective horizon = ~2 average price changes.

### 1B. OrderFlowRepresentationAnalyzer (lob-dataset-analyzer)

**Purpose:** Compare predictive power of different feature subsets (Raw LOB indices 0-39 vs MBO flow indices 48-56 vs OFI signal indices 84-88) at multiple horizons.

**Method:** For each feature group, compute R-squared, IC, and DA against regression labels.

**Location:** `lob-dataset-analyzer/src/lobanalyzer/analysis/regression/representation_comparison.py`

**Source:** Kolm et al. (2023), Section 4.1; Yang et al. (2024), Table comparing OFI vs LOB.

---

## Phase 2: Infrastructure Changes

### 2A. CVML Layer (lob-models)

**What:** Cross-Variate Mixing Layers -- a plug-in Conv1D front-end that improves regression R-squared by 244.9%.

**Location:** New file `lob-models/src/lobmodels/layers/cvml.py`

**Architecture** (from Li et al. ICLR 2025, reference code `mprf/layers/conv_layer.py`):
- 5 Conv1D layers with kernel_size=2
- Dilation: exponential [1, 2, 4, 8, 16]
- Causal padding (LeftPad1d)
- in_channels=N_features, out_channels=ceil(N/2)
- ReLU activation between layers
- Operates on feature dimension: (B,T,D) -> permute -> Conv1D -> permute back

**Integration:** Add `use_cvml: bool = False` to TLOBConfig. When True, prepend CVML before BiN layer. CVML reduces feature dimension from 128 to 64, which then feeds into BiN/embedding.

**Config:**
```yaml
model:
  use_cvml: true
  cvml_out_channels: 64
```

### 2B. GMADL Loss (lob-models)

**What:** Generalized Mean Absolute Directional Loss -- penalizes wrong-direction predictions, rewards correct-direction predictions weighted by magnitude.

**Location:** Add to `lob-models/src/lobmodels/losses/gmadl.py` or extend existing loss infrastructure.

**Formula** (Michankov et al. 2024):
```python
def gmadl_loss(y_true, y_pred, a=10.0, b=1.5):
    product = y_true * y_pred
    direction_score = torch.sigmoid(a * product) - 0.5
    magnitude_weight = torch.abs(y_true).pow(b)
    loss = (-1.0 * direction_score * magnitude_weight).mean()
    return loss
```

**Parameters:**
- `a=10.0`: sigmoid sharpness (higher = sharper direction penalty)
- `b=1.5`: large-return emphasis (higher = focus on big moves)

**Config:**
```yaml
model:
  regression_loss_type: gmadl
  gmadl_a: 10.0
  gmadl_b: 1.5
```

### 2C. Quantile Regression Head (lob-models -- Phase 4)

**What:** Multi-quantile output head with pinball loss. Predict tau=[0.1, 0.25, 0.5, 0.75, 0.9].

**Location:** `lob-models/src/lobmodels/heads/quantile.py`

**From:** Zhang, Zohren & Roberts (ICML 2019) DeepLOB-QR.

---

## Phase 3: Data Export

### 3A. Fine-Grained Export Config (feature-extractor-MBO-LOB)

**New file:** `configs/nvda_xnas_128feat_regression_finegrained.toml`

Key changes from current `nvda_xnas_128feat_regression.toml`:
```toml
[sampling]
event_count = 100      # Was 1000. 10x finer.

[sequence]
window_size = 100      # Keep same window of 100 timesteps
stride = 5             # Reduce stride to manage overlap

[labels]
strategy = "regression"
horizons = [1, 2, 5, 10, 20]      # Was [10, 60, 300]. Short horizons.
return_type = "point_return"        # Point-return at short horizons
```

**Expected output:** ~2.7M sequences (10x current), ~130 GB. Mitigate with stride=5 (vs stride=10).

**Trade-off analysis:**
- Pro: Horizons align with OFI effective lifetime
- Pro: Point-return should have signal at H1-H5
- Con: 10x data volume, 10x training time
- Mitigation: Use T=20 (94K param model) and batch_size=256

---

## Phase 4: Experiments

### Experiment A: Fine-Grained Baselines
- Run `compute_regression_baselines.py` on fine-grained export
- Establish persistence, linear Ridge, and single-feature R-squared at H1-H20

### Experiment B: TLOB + CVML + Huber on Fine-Grained Data
- Config: TLOB (2L, h=32) with CVML front-end
- Loss: Huber (delta calibrated from fine-grained IQR)
- Compare to Experiment A baselines

### Experiment C: TLOB + CVML + GMADL
- Same architecture as B, replace Huber with GMADL (a=10, b=1.5)
- Compare directional accuracy and backtest profitability

### Experiment D: Quantile Regression (if B/C show promise)
- 5-quantile head with pinball loss
- Risk-aware backtest: trade only when tau=0.1 quantile exceeds 0.7 bps (deep ITM breakeven)

---

## Per-Repository Change Summary

| Repo | Change | Priority | Effort |
|------|--------|----------|--------|
| **lob-dataset-analyzer** | EffectiveHorizonAnalyzer | 1 | Low |
| **lob-dataset-analyzer** | OrderFlowRepresentationAnalyzer | 1 | Low |
| **lob-models** | CVML layer (`layers/cvml.py`) | 2 | Medium |
| **lob-models** | GMADL loss (`losses/gmadl.py`) | 2 | Low |
| **lob-models** | TLOBConfig: `use_cvml`, `cvml_out_channels` | 2 | Low |
| **lob-models** | ModelConfig: `gmadl_a`, `gmadl_b` | 2 | Low |
| **feature-extractor-MBO-LOB** | Fine-grained export config | 3 | Low |
| **lob-model-trainer** | Training config for fine-grained | 3 | Low |
| **lob-model-trainer** | GMADL support in `compute_loss` dispatch | 2 | Low |
| **lob-models** | Quantile head (Phase 4) | 4 | Medium |
| **feature-extractor-MBO-LOB** | Spread-aware return_type (Phase 4) | 4 | Medium |

---

*This plan should be executed in the order listed. Phase 1 (statistical analysis) informs the exact parameters for Phases 2-4. Do not skip Phase 1.*
