# Phase 2A: Exploratory Data Analysis Plan

> **Status**: 📋 PLANNING  
> **Date**: 2025-12-19  
> **Purpose**: Understand signals, labels, and their relationships before model training  
> **Principle**: No modeling until we deeply understand the data

---

## Executive Summary

This phase answers fundamental questions before any model training:

| Question | Why It Matters |
|----------|----------------|
| Which signals actually predict price movement? | Focus modeling effort on informative features |
| What horizon/threshold maximizes predictability? | Optimal label configuration |
| Are signals stable across days and regimes? | Generalization potential |
| Which signals are redundant? | Feature selection, avoid overfitting |
| Do temporal patterns exist in 100-sample window? | Sequence model justification |

**Expected Duration**: 3-5 days of careful analysis  
**Expected Output**: Concrete recommendations for model architecture, feature set, and hyperparameters

---

## Data Overview

### Available Data

| Attribute | Value |
|-----------|-------|
| Symbol | NVDA |
| Date Range | 2025-02-03 to 2025-02-25 (16 trading days) |
| Features | 98 per sample |
| Feature Categories | LOB (40) + Derived (8) + MBO (36) + Signals (14) |
| Labels | {-1: Down, 0: Stable, 1: Up} |
| Label Config | horizon=50, smoothing=10, threshold=8bps |
| Sequence Config | window=100, stride=10 |
| Total Samples | ~1.6M features, ~160K labels |

### Key Signals (Indices 84-97)

| Index | Signal | Description | Expected Range |
|-------|--------|-------------|----------------|
| 84 | `true_ofi` | Cont et al. OFI | (-∞, +∞) |
| 85 | `depth_norm_ofi` | OFI / avg_depth | (-∞, +∞) |
| 86 | `executed_pressure` | trades_ask - trades_bid | (-∞, +∞) |
| 87 | `signed_mp_delta_bps` | Microprice deviation | ~[-100, 100] |
| 88 | `trade_asymmetry` | (ask-bid)/total trades | [-1, 1] |
| 89 | `cancel_asymmetry` | (ask-bid)/total cancels | [-1, 1] |
| 90 | `fragility_score` | concentration/ln(depth) | [0, ∞) |
| 91 | `depth_asymmetry` | (bid-ask)/total depth | [-1, 1] |
| 92 | `book_valid` | Safety gate | {0, 1} |
| 93 | `time_regime` | Market session | {0,1,2,3,4} |
| 94 | `mbo_ready` | Warmup flag | {0, 1} |
| 95 | `dt_seconds` | Sample duration | [0, ∞) |
| 96 | `invalidity_delta` | Feed problems | [0, ∞) |
| 97 | `schema_version` | Version constant | {2} |

---

## Analysis Notebooks

### Notebook 0: Data Overview & Sanity Checks (P0)

**File**: `notebooks/00_data_overview.ipynb`

**Objectives**:
- Load and validate all training data
- Verify feature count, label encoding, data types
- Check for missing values, NaN, Inf
- Basic statistics for each feature category
- Confirm safety gates (book_valid, mbo_ready) are as expected

**Key Outputs**:
- [ ] Total sample count by split (train/val/test)
- [ ] Feature completeness verification
- [ ] Label distribution verification
- [ ] Any data quality issues identified

**Expected Duration**: 1-2 hours

---

### Notebook 1: Label Analysis (P0 - Critical)

**File**: `notebooks/01_label_analysis.ipynb`

**Objectives**:
- Understand label distribution and balance
- Analyze label autocorrelation (clustering of Up/Down)
- Compute label transition probabilities
- **Horizon sensitivity analysis**: Re-label at different horizons
- **Threshold sensitivity analysis**: Re-label at different thresholds
- Find optimal (horizon, threshold) for maximum predictability

**Key Analyses**:

```python
# 1. Label Distribution
for label in [-1, 0, 1]:
    print(f"{label}: {(labels == label).sum() / len(labels):.1%}")

# 2. Label Autocorrelation
from statsmodels.tsa.stattools import acf
label_acf = acf(labels, nlags=50)
# If high autocorrelation → labels are clustered (trends)
# If low autocorrelation → labels are random

# 3. Transition Matrix
transitions = np.zeros((3, 3))
for i in range(len(labels) - 1):
    transitions[labels[i]+1, labels[i+1]+1] += 1
transitions /= transitions.sum(axis=1, keepdims=True)
# P(Up→Up), P(Up→Down), etc.

# 4. Horizon Sensitivity
for horizon in [10, 25, 50, 100, 200]:
    labels_h = compute_labels(mid_prices, horizon=horizon, threshold=0.0008)
    predictability = compute_predictability(signals, labels_h)
    print(f"Horizon {horizon}: Predictability = {predictability:.4f}")

# 5. Threshold Sensitivity
for threshold in [0.0002, 0.0004, 0.0006, 0.0008, 0.0010, 0.0015]:
    labels_t = compute_labels(mid_prices, horizon=50, threshold=threshold)
    balance = min(class_counts) / max(class_counts)
    print(f"Threshold {threshold:.4f}: Balance = {balance:.2f}")
```

**Key Questions**:
- Current config: horizon=50, threshold=8bps. Is this optimal?
- Do labels exhibit mean-reversion or momentum?
- How much does class balance change with threshold?

**Expected Duration**: 3-4 hours

---

### Notebook 2: Signal Distributions (P1)

**File**: `notebooks/02_signal_distributions.ipynb`

**Objectives**:
- Visualize distribution of each signal
- Test for normality (important for some models)
- Analyze tail behavior (heavy tails → robust methods needed)
- Check stationarity (do statistics change over time?)
- Compare distributions across days

**Key Analyses**:

```python
# For each signal in [84, 85, 86, 87, 88, 89, 90, 91]:
for idx, name in signal_indices.items():
    signal = features[:, idx]
    
    # Distribution metrics
    print(f"{name}:")
    print(f"  Mean: {signal.mean():.4f}")
    print(f"  Std: {signal.std():.4f}")
    print(f"  Skewness: {skew(signal):.4f}")
    print(f"  Kurtosis: {kurtosis(signal):.4f}")
    print(f"  % outside 3σ: {(np.abs(signal) > 3).mean():.2%}")
    
    # Normality test
    stat, p = normaltest(signal)
    print(f"  Normal? p={p:.4e}")
    
    # Stationarity test (ADF)
    adf_stat, adf_p, _, _, _, _ = adfuller(signal[:10000])
    print(f"  Stationary? p={adf_p:.4e}")
```

**Visualizations**:
- Histogram with fitted normal overlay
- QQ-plot for normality assessment
- Rolling mean/std over time (stationarity check)
- Box plots by day (day-to-day stability)

**Expected Duration**: 2-3 hours

---

### Notebook 3: Signal Correlations & Redundancy (P1)

**File**: `notebooks/03_signal_correlations.ipynb`

**Objectives**:
- Compute correlation matrix for all 14 signals
- Identify redundant signal pairs (corr > 0.8)
- Perform PCA to find orthogonal factors
- Calculate VIF for multicollinearity assessment
- Cluster signals by similarity

**Key Analyses**:

```python
# 1. Correlation Matrix
signal_features = features[:, 84:98]
corr_matrix = np.corrcoef(signal_features.T)
sns.heatmap(corr_matrix, annot=True, xticklabels=signal_names, yticklabels=signal_names)

# 2. PCA Analysis
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(signal_features)
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)
# How many components explain 95% variance?

# 3. VIF (Variance Inflation Factor)
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = [variance_inflation_factor(signal_features, i) for i in range(14)]
# VIF > 10 indicates severe multicollinearity

# 4. Mutual Information
from sklearn.feature_selection import mutual_info_classif
mi = mutual_info_classif(signal_features, labels)
```

**Expected Findings**:
- `true_ofi` ↔ `depth_norm_ofi`: High correlation (same base signal)
- `trade_asymmetry` ↔ `executed_pressure`: Moderate correlation
- `fragility_score`: Likely independent (book structure, not flow)

**Expected Duration**: 2-3 hours

---

### Notebook 4: Signal Predictive Power (P0 - Critical)

**File**: `notebooks/04_signal_predictive_power.ipynb`

**Objectives**:
- **Primary question**: Which signals predict price movement?
- Compute multiple predictive metrics per signal
- Rank signals by predictive power
- Identify best standalone predictors
- Analyze non-linear relationships

**Key Analyses**:

```python
# For each signal:
for idx, name in signal_indices.items():
    signal = aligned_features[:, idx]
    
    # 1. Pearson Correlation
    corr = np.corrcoef(signal, labels)[0, 1]
    
    # 2. Spearman Correlation (rank-based, robust to outliers)
    spearman = spearmanr(signal, labels)[0]
    
    # 3. AUC for Up vs Not-Up
    y_binary_up = (labels == 1).astype(int)
    auc_up = roc_auc_score(y_binary_up, signal)
    
    # 4. AUC for Down vs Not-Down
    y_binary_down = (labels == -1).astype(int)
    auc_down = roc_auc_score(y_binary_down, -signal)  # Negative for Down
    
    # 5. Mutual Information
    mi = mutual_info_classif(signal.reshape(-1, 1), labels, discrete_features=False)[0]
    
    # 6. Information Gain (bits)
    ig = mi / np.log(2)
    
    print(f"{name}:")
    print(f"  Pearson: {corr:+.4f}")
    print(f"  Spearman: {spearman:+.4f}")
    print(f"  AUC (Up): {auc_up:.4f}")
    print(f"  AUC (Down): {auc_down:.4f}")
    print(f"  MI: {mi:.4f} ({ig:.4f} bits)")
```

**Non-Linear Analysis**:

```python
# Bin signal into deciles and compute label probability per bin
for idx, name in signal_indices.items():
    signal = aligned_features[:, idx]
    deciles = pd.qcut(signal, q=10, labels=False, duplicates='drop')
    
    for d in range(10):
        mask = deciles == d
        p_up = (labels[mask] == 1).mean()
        p_down = (labels[mask] == -1).mean()
        print(f"  Decile {d}: P(Up)={p_up:.3f}, P(Down)={p_down:.3f}")
```

**Expected Output**: Ranked list of signals by predictive power

**Expected Duration**: 4-5 hours

---

### Notebook 5: Temporal Dynamics (P2)

**File**: `notebooks/05_temporal_dynamics.ipynb`

**Objectives**:
- Analyze signal persistence (autocorrelation)
- Find optimal lookback window
- Detect lead-lag relationships between signals
- Determine if signal changes are more predictive than levels
- Understand within-sequence patterns

**Key Analyses**:

```python
# 1. Signal Autocorrelation
for idx, name in signal_indices.items():
    signal = features[:, idx]
    acf_values = acf(signal[:50000], nlags=100)
    half_life = np.argmax(acf_values < 0.5)  # When does correlation drop to 0.5?
    print(f"{name}: Half-life = {half_life} samples")

# 2. Cross-Correlation (Lead-Lag)
from scipy.signal import correlate
for i, name1 in enumerate(signal_names):
    for j, name2 in enumerate(signal_names):
        if i < j:
            xcorr = correlate(signals[:, i], signals[:, j], mode='full')
            lag = np.argmax(xcorr) - len(signals[:, i])
            if abs(lag) > 0:
                print(f"{name1} leads {name2} by {lag} samples")

# 3. Predictive Decay
for lag in [1, 5, 10, 20, 50, 100]:
    lagged_signal = np.roll(signal, lag)
    corr = np.corrcoef(lagged_signal[lag:], labels[:-lag])[0, 1]
    print(f"Lag {lag}: Correlation = {corr:.4f}")

# 4. Signal Level vs Change
signal_level = aligned_features[:, 84]  # true_ofi
signal_change = np.diff(aligned_features[:, 84])
# Which is more predictive?
```

**Expected Duration**: 3-4 hours

---

### Notebook 6: Regime Analysis (P1)

**File**: `notebooks/06_regime_analysis.ipynb`

**Objectives**:
- Compare signal behavior across time regimes
- Measure predictive power per regime
- Determine if regime-specific models are warranted
- Analyze volatility patterns by regime

**Key Analyses**:

```python
# Split by time_regime (index 93)
regimes = {
    0: "Open (9:30-9:45)",
    1: "Early (9:45-10:30)",
    2: "Midday (10:30-15:30)",
    3: "Close (15:30-16:00)",
    4: "Closed"
}

for regime_val, regime_name in regimes.items():
    mask = aligned_features[:, 93] == regime_val
    if mask.sum() < 100:
        continue
    
    regime_signals = aligned_features[mask]
    regime_labels = labels[mask]
    
    print(f"\n{regime_name} (n={mask.sum()}):")
    
    # Label distribution
    for lbl, lbl_name in [(-1, "Down"), (0, "Stable"), (1, "Up")]:
        pct = (regime_labels == lbl).mean() * 100
        print(f"  {lbl_name}: {pct:.1f}%")
    
    # Signal predictive power
    for idx, name in list(signal_indices.items())[:4]:
        signal = regime_signals[:, idx]
        corr = np.corrcoef(signal, regime_labels)[0, 1]
        print(f"  {name} corr: {corr:+.4f}")
```

**Key Questions**:
- Is `true_ofi` more predictive at Open or Midday?
- Should we train separate models per regime?
- Are thresholds regime-dependent?

**Expected Duration**: 3-4 hours

---

### Notebook 7: Feature Engineering (P2)

**File**: `notebooks/07_feature_engineering.ipynb`

**Objectives**:
- Create new features from existing signals
- Test interaction terms
- Aggregate signals over window (mean, max, trend)
- Compute signal momentum/acceleration
- Evaluate engineered features

**Feature Ideas**:

```python
# 1. Signal Momentum (change over time)
ofi_momentum = ofi[t] - ofi[t-10]

# 2. Signal Acceleration
ofi_acceleration = ofi_momentum[t] - ofi_momentum[t-10]

# 3. Interaction Terms
ofi_fragility_interaction = true_ofi * fragility_score

# 4. Regime-Adjusted Signals
ofi_regime_adjusted = true_ofi / regime_typical_volatility[time_regime]

# 5. Window Aggregations
ofi_window_mean = mean(true_ofi[-100:])
ofi_window_max = max(true_ofi[-100:])
ofi_window_trend = linear_regression_slope(true_ofi[-100:])

# 6. Relative Signals
ofi_percentile = percentile_rank(true_ofi, lookback=1000)

# 7. Composite Signals
direction_confidence = (
    normalize(true_ofi) + 
    normalize(trade_asymmetry) + 
    normalize(executed_pressure)
) / 3
```

**Expected Duration**: 4-5 hours

---

### Notebook 8: Optimal Configuration (P3)

**File**: `notebooks/08_optimal_configuration.ipynb`

**Objectives**:
- Synthesize findings from all previous notebooks
- Determine optimal horizon and threshold
- Select best feature subset
- Recommend model architecture
- Document final configuration

**Key Outputs**:
- [ ] Recommended horizon (samples)
- [ ] Recommended threshold (bps)
- [ ] Selected features (indices)
- [ ] Recommended model architecture
- [ ] Per-regime recommendations (if applicable)

**Expected Duration**: 2-3 hours

---

### Notebook 9: Generalization Tests (P3)

**File**: `notebooks/09_generalization_tests.ipynb`

**Objectives**:
- Test findings across different days
- Walk-forward validation
- Cross-symbol testing (if data available)
- Document robustness of findings

**Key Analyses**:

```python
# 1. Day-to-Day Variance
daily_correlations = []
for day in days:
    corr = compute_ofi_label_correlation(day.features, day.labels)
    daily_correlations.append(corr)
print(f"OFI-Label Corr: {np.mean(daily_correlations):.4f} ± {np.std(daily_correlations):.4f}")

# 2. Walk-Forward Validation
for train_end in range(5, 16):  # Days 1-5 train, predict day 6, etc.
    train_days = days[:train_end]
    test_day = days[train_end]
    model = train_simple_model(train_days)
    accuracy = evaluate(model, test_day)
    print(f"Train days 1-{train_end}, Test day {train_end+1}: Accuracy = {accuracy:.2%}")
```

**Expected Duration**: 3-4 hours

---

## Analysis Utilities

### Helper Functions to Create

```python
# lob-model-trainer/src/lobtrainer/analysis/

def load_aligned_data(data_dir: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load features aligned with labels (one feature per label)."""
    pass

def compute_label_at_horizon(mid_prices: np.ndarray, horizon: int, threshold: float) -> np.ndarray:
    """Recompute labels with different horizon/threshold."""
    pass

def signal_predictive_power(signal: np.ndarray, labels: np.ndarray) -> dict:
    """Compute multiple predictive metrics for a signal."""
    return {
        'pearson': ...,
        'spearman': ...,
        'auc_up': ...,
        'auc_down': ...,
        'mutual_info': ...,
    }

def regime_analysis(features: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Compute per-regime statistics."""
    pass
```

---

## Success Criteria

At the end of Phase 2A, we should have:

| Deliverable | Description |
|-------------|-------------|
| **Signal Ranking** | Ordered list of signals by predictive power |
| **Optimal Config** | Recommended (horizon, threshold) pair |
| **Feature Selection** | Which of 98 features to use |
| **Regime Insights** | Per-regime behavior documented |
| **Model Recommendation** | Sequence vs flat, simple vs complex |
| **Generalization Evidence** | Day-to-day stability metrics |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Overfitting to NVDA | Document which findings are symbol-specific |
| Data snooping | Reserve test set, don't peek |
| Spurious correlations | Use multiple metrics, require consistency |
| Non-stationarity | Test on different time periods |
| Survivorship bias | Consider what data we DON'T have |

---

## Timeline

| Day | Focus | Notebooks |
|-----|-------|-----------|
| 1 | Foundation | 00_data_overview, 01_label_analysis |
| 2 | Signals | 02_signal_distributions, 03_signal_correlations |
| 3 | Predictability | 04_signal_predictive_power (main focus) |
| 4 | Dynamics & Regime | 05_temporal_dynamics, 06_regime_analysis |
| 5 | Synthesis | 07_feature_engineering, 08_optimal_configuration, 09_generalization_tests |

---

## Next Phase Preview

Based on Phase 2A findings, Phase 2B will:

1. **Train baseline models** (XGBoost, Logistic Regression) with optimal configuration
2. **Establish performance floor** (random baseline: ~33%, momentum baseline: ~35-40%)
3. **Feature importance analysis** from trained models
4. **Prepare for sequence modeling** (LSTM, Transformer) based on temporal findings

