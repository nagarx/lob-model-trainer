# Analysis Modules Reference

**Version**: 1.0  
**Last Updated**: 2025-12-20  
**Location**: `lob-model-trainer/src/lobtrainer/analysis/`

This document provides comprehensive documentation for all analysis modules in the LOB Model Trainer. Each module is designed to be reusable across different datasets and symbols, following the principles in `.cursor/rules/hft-rules/RULE.md`.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Module: data_loading](#2-module-data_loading)
3. [Module: streaming](#3-module-streaming)
4. [Module: data_overview](#4-module-data_overview)
5. [Module: label_analysis](#5-module-label_analysis)
6. [Module: signal_stats](#6-module-signal_stats)
7. [Module: signal_correlations](#7-module-signal_correlations)
8. [Module: predictive_power](#8-module-predictive_power)
9. [Module: temporal_dynamics](#9-module-temporal_dynamics)
10. [Module: generalization](#10-module-generalization)
11. [Output Files Reference](#11-output-files-reference)
12. [Interpretation Guide](#12-interpretation-guide)

---

## 1. Architecture Overview

### Design Principles

1. **Memory Efficiency**: Two access patterns available:
   - **Bulk loading** (`data_loading`): Loads entire splits into memory. Fast but memory-intensive.
   - **Streaming** (`streaming`): Processes one day at a time. O(1) memory regardless of dataset size.

2. **Symbol-Agnostic**: All functions work with any symbol exported from the Rust pipeline.

3. **Dataclass-Based Results**: All analysis results are structured as dataclasses for type safety and JSON serialization.

4. **Separation of Computation and Display**: Every module has:
   - `compute_*()` functions → Return dataclasses
   - `print_*()` functions → Format for console output

### Module Dependencies

```
data_loading ←── data_overview
     ↑              ↑
     │              └── label_analysis
     │                       ↑
     │                       └── signal_stats
streaming                         ↑
     │                            └── signal_correlations
     │                                     ↑
     └──────────────────────────────────── predictive_power
                                                  ↑
                                                  └── temporal_dynamics
                                                           ↑
                                                           └── generalization
```

### Memory Usage Guide

| Dataset Size | Recommended Approach | Peak Memory |
|--------------|---------------------|-------------|
| < 20 days | Bulk loading | ~2 GB |
| 20-100 days | Mixed | ~4 GB |
| > 100 days | **Streaming only** | < 1 GB |

---

## 2. Module: data_loading

**File**: `data_loading.py` (220 lines)  
**Purpose**: Load and align features with labels from exported NumPy files.

### Constants

```python
WINDOW_SIZE = 100  # Samples per sequence (matches Rust export)
STRIDE = 10        # Samples between sequence starts
CORE_SIGNAL_INDICES = [84, 85, 86, 87, 88, 89, 90, 91]  # Primary trading signals
```

### Key Functions

#### `load_split(data_dir: Path, split_name: str) -> Dict`

Loads all data for a single split (train/val/test).

**Returns**:
```python
{
    'features': np.ndarray,  # (N_samples, 98)
    'labels': np.ndarray,    # (N_labels,)
    'n_days': int,
    'dates': List[str],
}
```

**Usage**:
```python
data = load_split(Path("data/exports/nvda_98feat"), "train")
features = data['features']  # Shape: (N, 98)
labels = data['labels']      # Shape: (M,) where M < N
```

#### `align_features_with_labels(features, n_labels, window_size, stride) -> np.ndarray`

Aligns sample-level features with sequence-level labels.

**Formula**:
```
For label[i], the corresponding feature is at:
feat_idx = i * stride + window_size - 1

This is the LAST feature in the sequence window [i*stride, i*stride + window_size)
```

**Why This Matters**: Labels are computed at sequence boundaries, not per-sample. This function extracts the correct feature vector for each label.

#### `get_signal_info() -> Dict[int, Dict]`

Returns metadata for all 14 signal features (indices 84-97).

**Example**:
```python
{
    84: {
        'name': 'true_ofi',
        'description': 'Cont et al. Order Flow Imbalance',
        'type': 'continuous',
        'expected_sign': '+',  # Positive OFI → expect Up label
    },
    # ... 13 more signals
}
```

---

## 3. Module: streaming

**File**: `streaming.py` (657 lines)  
**Purpose**: Memory-efficient analysis for large datasets (100+ days).

### Design Principle

> "Process data in chunks/days, never all at once. Use generators instead of lists."

### Key Classes

#### `DayData`

```python
@dataclass
class DayData:
    date: str
    features: np.ndarray  # (N, 98)
    labels: np.ndarray    # (M,)
    n_samples: int
    n_labels: int
    
    @property
    def memory_bytes(self) -> int
```

#### `RunningStats`

Welford's online algorithm for computing mean and variance.

**Reference**: Welford, B. P. (1962). "Note on a method for calculating corrected sums of squares and products"

```python
@dataclass
class RunningStats:
    n: int = 0
    mean: float = 0.0
    M2: float = 0.0  # Sum of squared deviations
    
    def update(self, x: float) -> None:
        # Numerically stable, single-pass update
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
    
    @property
    def variance(self) -> float:
        return self.M2 / self.n if self.n > 0 else 0.0
```

**Why Welford's Algorithm?**
- Numerically stable (no catastrophic cancellation)
- Single-pass (O(n) time)
- Constant memory O(1)
- Supports merging (for parallel computation)

### Key Functions

#### `iter_days(data_dir, split, dtype=np.float32, mmap_mode=None)`

Generator that yields one day at a time (RAW features, NOT aligned with labels).

**Usage**:
```python
for day in iter_days(Path("data/exports/nvda"), "train"):
    # day.features has N samples, day.labels has M labels (N > M)
    # Use for signal autocorrelation, statistics, etc.
    # DO NOT use for signal-label correlation!
```

**Options**:
- `dtype=np.float32`: 50% memory reduction vs float64
- `mmap_mode='r'`: Memory-mapped files (even more efficient)

#### `iter_days_aligned(data_dir, split, window_size=100, stride=10, dtype=np.float32)` ⭐ **PREFERRED**

Generator that yields CORRECTLY ALIGNED feature-label pairs per day.

**CRITICAL**: Use this for ANY analysis involving signal-label relationships!

**Usage**:
```python
for day in iter_days_aligned(data_dir, 'train'):
    # day.features[i] corresponds EXACTLY to day.labels[i]
    corr = np.corrcoef(day.features[:, TRUE_OFI], day.labels)[0, 1]
```

**Formula**:
```
For label[i], the aligned feature is at:
    feat_idx = i * stride + window_size - 1
```

**Why this matters**:
- Crude subsampling (`features[::step]`) breaks alignment
- Day boundary handling is critical for multi-day analysis
- ~30x improvement in correlation accuracy (0.003 → 0.10)

#### `compute_streaming_overview(data_dir, symbol, dtype)`

Computes dataset overview with streaming.

**Memory**: O(n_features) - constant regardless of dataset size.

**Returns**:
```python
{
    'total_days': int,
    'total_samples': int,
    'total_labels': int,
    'date_range': (str, str),
    'data_quality': {
        'nan_count': int,
        'inf_count': int,
        'is_clean': bool,
    },
    'label_distribution': {
        'down_pct': float,
        'stable_pct': float,
        'up_pct': float,
    },
}
```

#### `compute_streaming_label_analysis(data_dir, split, max_samples_for_acf=100000)`

Streams label analysis including autocorrelation and transition matrix.

**Note**: Labels are small enough to collect in memory (1 byte per sample), so ACF can be computed exactly. Subsampling is applied for very large datasets.

---

## 4. Module: data_overview

**File**: `data_overview.py` (633 lines)  
**Purpose**: Data validation, quality checks, and dataset profiling.

### Key Dataclasses

#### `DataQuality`

```python
@dataclass
class DataQuality:
    total_values: int
    finite_count: int
    nan_count: int
    inf_count: int
    pct_finite: float
    columns_with_nan: List[int]
    columns_with_inf: List[int]
    
    @property
    def is_clean(self) -> bool:
        return self.nan_count == 0 and self.inf_count == 0
```

#### `LabelDistribution`

```python
@dataclass
class LabelDistribution:
    total: int
    down_count: int
    stable_count: int
    up_count: int
    imbalance_ratio: float  # max_class / min_class
    
    @property
    def is_balanced(self) -> bool:
        return self.imbalance_ratio < 1.5
```

#### `CategoricalValidation`

Validates categorical features against expected values.

```python
@dataclass
class CategoricalValidation:
    name: str  # e.g., 'book_valid', 'time_regime'
    index: int
    unique_values: List[float]
    expected_values: Optional[List[float]]
    is_valid: bool
    message: str
```

### Key Functions

#### `validate_file_structure(data_dir) -> Dict[str, FileInventory]`

Discovers and validates all data files across splits.

**Checks**:
- Feature files exist (`*_features.npy`)
- Label files exist (`*_labels.npy`)
- Files are paired correctly

#### `compute_all_categorical_validations(features) -> List[CategoricalValidation]`

Validates all categorical features:

| Feature | Index | Expected Values |
|---------|-------|-----------------|
| book_valid | 92 | [0.0, 1.0] |
| time_regime | 93 | [0.0, 1.0, 2.0, 3.0, 4.0] |
| mbo_ready | 94 | [0.0, 1.0] |
| invalidity_delta | 96 | Any non-negative |
| schema_version | 97 | [2.0] (current version) |

#### `generate_dataset_summary(data_dir, symbol) -> DatasetSummary`

Generates comprehensive summary including all validations and statistics.

**Output Example** (`nvda_data_overview.json`):
```json
{
  "symbol": "NVDA",
  "date_range": ["20250203", "20250929"],
  "total_days": 165,
  "total_samples": 16996000,
  "data_quality": {
    "is_clean": true,
    "nan_count": 0
  },
  "label_distribution": {
    "imbalance_ratio": 1.56
  }
}
```

---

## 5. Module: label_analysis

**File**: `label_analysis.py` (536 lines)  
**Purpose**: Comprehensive analysis of label characteristics.

### Key Analyses

#### 1. Label Distribution

```python
def compute_label_distribution(labels: np.ndarray) -> LabelDistribution
```

**Metrics**:
- Counts and percentages for each class
- Imbalance ratio (max/min)
- Majority/minority class identification

#### 2. Autocorrelation (ACF)

```python
def compute_autocorrelation(labels: np.ndarray, max_lag: int = 100) -> AutocorrelationResult
```

**Formula**:
```
ACF(k) = Cov(label_t, label_{t+k}) / Var(label)
```

**Interpretation**:
| ACF(1) | Meaning |
|--------|---------|
| > 0.1 | Strong positive: Labels cluster (trends persist) |
| > CI | Weak positive: Some persistence |
| < -CI | Negative: Mean-reversion |
| ≈ 0 | No autocorrelation: Random labels |

**95% CI for white noise**: ±1.96/√n

#### 3. Transition Matrix

```python
def compute_transition_matrix(labels: np.ndarray) -> TransitionMatrix
```

**Computes**:
- Count matrix: How many transitions from each state to each state
- Probability matrix: P(label_{t+1} = j | label_t = i)
- Stationary distribution: Long-run proportion of each label
- Persistence deviation: How much P(same) deviates from stationary

**Example Output**:
```
From\To   Down    Stable    Up
Down      0.972   0.028    0.000
Stable    0.021   0.958    0.021
Up        0.000   0.028    0.972
```

**Critical Insight**: The 97% diagonal values mean a "predict same as last" baseline achieves ~96% accuracy. The model must learn to predict **transitions** to add value.

#### 4. Time Regime Analysis

```python
def compute_regime_stats(aligned_features, labels) -> List[RegimeStats]
```

Analyzes label distribution by market session:

| Regime | Time |
|--------|------|
| 0 (Open) | 9:30-9:45 |
| 1 (Early) | 9:45-10:30 |
| 2 (Midday) | 10:30-15:30 |
| 3 (Close) | 15:30-16:00 |
| 4 (Closed) | After hours |

#### 5. Signal-Label Correlations

```python
def compute_signal_label_correlations(aligned_features, labels) -> List[SignalCorrelation]
```

**Uses Bonferroni-corrected significance threshold**: α = 0.05 / n_signals

---

## 6. Module: signal_stats

**File**: `signal_stats.py` (425 lines)  
**Purpose**: Distribution statistics and stationarity tests.

### Key Analyses

#### 1. Distribution Statistics

```python
def compute_distribution_stats(features, signal_indices=None) -> pd.DataFrame
```

**Metrics per signal**:
- Mean, std, min, max, median
- Skewness (3rd moment)
- Kurtosis (4th moment, excess)
- Outlier percentage (|z| > 3)
- Normality test p-value (D'Agostino-Pearson)

#### 2. Augmented Dickey-Fuller (ADF) Stationarity Test

```python
def compute_stationarity_test(signal, max_samples=100000) -> Tuple[float, float, Dict, bool]
```

**Formula**:
```
ΔX_t = α + βt + γX_{t-1} + Σδ_i ΔX_{t-i} + ε_t

H0: γ = 0 (unit root, non-stationary)
H1: γ < 0 (stationary)
```

**Interpretation**:
| p-value | Meaning |
|---------|---------|
| < 0.05 | Stationary (reject null) |
| < 0.10 | Marginally stationary |
| ≥ 0.10 | Non-stationary (consider differencing) |

#### 3. Rolling Statistics

```python
def compute_rolling_stats(signal, window_size=10000, n_windows=10) -> Tuple[...]
```

Detects non-stationarity through:
- Mean drift (change from first to last window)
- Std drift
- Mean range (max - min across windows)

**Stability Thresholds**:
- Mean stable: `mean_range < 0.5 * overall_std`
- Std stable: `|std_drift| < 0.3 * overall_std`

---

## 7. Module: signal_correlations

**File**: `signal_correlations.py` (481 lines)  
**Purpose**: Correlation matrix, redundancy detection, PCA, and VIF.

### Key Analyses

#### 1. Correlation Matrix

```python
def compute_signal_correlation_matrix(features, signal_indices=None) -> Tuple[np.ndarray, List[str]]
```

#### 2. Redundant Pairs

```python
def find_redundant_pairs(corr_matrix, signal_names, threshold=0.5) -> List[Dict]
```

Identifies signal pairs with |correlation| > threshold.

#### 3. PCA (Principal Component Analysis)

```python
def compute_pca_analysis(features, signal_indices=None, n_components=None) -> PCAResult
```

**Formula**:
```
X = USV^T
Explained variance = λ_i / Σλ
```

**Returns**:
- Explained variance ratio per component
- Cumulative variance
- Components needed for 90%/95% variance
- Dominant signal per component

#### 4. VIF (Variance Inflation Factor)

```python
def compute_vif(features, signal_indices=None) -> List[VIFResult]
```

**Formula**:
```
VIF = 1 / (1 - R²)

where R² is from regressing signal i on all other signals
```

**Interpretation**:
| VIF | Meaning |
|-----|---------|
| = 1 | No correlation with others |
| > 5 | Moderate multicollinearity (concerning) |
| > 10 | Severe multicollinearity (problematic) |

#### 5. Signal Clustering

```python
def cluster_signals(corr_matrix, signal_names, signal_indices, threshold=0.5) -> List[SignalCluster]
```

Groups signals based on correlation threshold using simple agglomerative clustering.

---

## 8. Module: predictive_power

**File**: `predictive_power.py` (297 lines)  
**Purpose**: Measure which signals predict labels.

### Key Analyses

#### 1. Signal Metrics

```python
def compute_signal_metrics(signal, labels, expected_sign='?') -> Dict
```

**Metrics**:
- Pearson/Spearman correlation
- AUC (Up vs Not-Up)
- AUC (Down vs Not-Down)
- Mutual Information (nats and bits)
- Conditional means (mean signal given each label)

#### 2. Binned Probability Analysis

```python
def compute_binned_probabilities(signal, labels, n_bins=10) -> pd.DataFrame
```

Bins signal into quantiles and computes:
- P(Up | signal in bin i)
- P(Down | signal in bin i)
- P(Stable | signal in bin i)

**Why This Matters**: Reveals non-linear relationships missed by correlation.

---

## 9. Module: temporal_dynamics

**File**: `temporal_dynamics.py` (751 lines)  
**Purpose**: Time-series properties critical for sequence model design.

### Key Analyses

#### 1. Signal Autocorrelation

```python
def compute_signal_autocorrelations(features, signal_indices=None, max_lag=100) -> List[SignalAutocorrelation]
```

**Returns per signal**:
- ACF values at each lag
- Half-life (lag where ACF drops below 0.5)
- Decay rate (exponential fit: ACF ≈ exp(-λk))

**Interpretation**:
| Half-life | Meaning |
|-----------|---------|
| ≤ 5 | Fast decay: Short memory |
| 6-20 | Moderate persistence |
| 21-50 | Strong persistence |
| > 50 | Very persistent: Long-term trends |

#### 2. Lead-Lag Relationships

```python
def compute_lead_lag_relations(features, signal_indices=None, max_lag=20) -> List[LeadLagRelation]
```

**Formula**:
```
CCF(k) = Corr(X_t, Y_{t+k})
k > 0 means X leads Y
```

**Identifies**:
- Which signal leads which
- Optimal lag for maximum correlation
- Significance (|correlation| > 0.2)

#### 3. Predictive Decay

```python
def compute_all_predictive_decays(features, labels, signal_indices=None, lags=None) -> List[PredictiveDecay]
```

**Answers**: "How quickly does signal information become stale?"

**Computes**: Correlation between signal[t] and label[t+horizon] for various horizons.

#### 4. Level vs Change Analysis

```python
def compute_all_level_vs_change(features, labels, signal_indices=None) -> List[LevelVsChangeAnalysis]
```

**Compares**:
- Predictive power of signal level (end of window)
- Predictive power of signal change (end - start of window)

**Recommendation**: "level", "change", or "both"

#### 5. Half-Life Definitions (Important!)

Two different half-life metrics are used:

| Metric | Definition | Context |
|--------|------------|---------|
| **ACF Half-life** | First lag where ACF < 0.5 | Signal autocorrelation |
| **Predictive Half-life** | First lag where \|corr\| < peak/2 | Signal-label correlation decay |

**ACF Half-life**: Standard time series definition. Measures how long until the signal loses half its "memory" (self-correlation).

**Predictive Half-life**: Relative to peak. Measures how quickly predictive power decays. A signal with high peak correlation but fast decay is only useful for very short-term prediction.

#### 6. Sequence Model Justification

The module automatically determines if a sequence model is justified based on:
- Average ACF half-life > 10
- Persistent prediction (predictive half-life > 10)
- Significant lead-lag relationships

---

## 10. Module: generalization

**File**: `generalization.py` (550 lines)  
**Purpose**: Test robustness across trading days.

### Key Analyses

#### 1. Day-to-Day Statistics

```python
def compute_day_statistics(days: List[Dict]) -> List[DayStatistics]
```

Per-day metrics:
- Sample count
- Label count
- Label distribution (Up%, Down%, Stable%)

#### 2. Signal Day Stats

```python
def compute_signal_day_stats(days, signal_indices=None) -> List[SignalDayStats]
```

Per-signal, per-day:
- Mean, std
- Correlation with labels

**Aggregates**:
- Mean of correlations across days
- Std of correlations across days
- Stability score: |mean(r)| / std(r)

**Stability Interpretation**:
| Score | Meaning |
|-------|---------|
| > 2.0 | Very stable |
| 1.0-2.0 | Moderate |
| < 1.0 | Unstable (use with caution) |

#### 3. Walk-Forward Validation

```python
def walk_forward_validation(days, signal_indices=None, min_train_days=3) -> List[WalkForwardResult]
```

For each test day:
1. Train on all previous days
2. Test on that day
3. Measure:
   - Signal correlations on test day
   - Simple threshold-based prediction accuracy

**Simple Predictor**:
- Predict Up if OFI > mean + 0.5σ
- Predict Down if OFI < mean - 0.5σ
- Else predict Stable

---

## 11. Output Files Reference

### Generated JSON Files

| File | Module | Contents |
|------|--------|----------|
| `nvda_data_overview.json` | data_overview | DatasetSummary |
| `nvda_label_analysis.json` | label_analysis | LabelAnalysisSummary |
| `signal_analysis_results.json` | signal_stats | Distribution stats |
| `nvda_temporal_dynamics.json` | temporal_dynamics | TemporalDynamicsSummary |
| `nvda_generalization.json` | generalization | GeneralizationSummary |
| `nvda_complete_analysis.json` | streaming | All analyses combined |

### Streaming Output Files

| File | Contents |
|------|----------|
| `nvda_streaming_overview.json` | Overview (streaming) |
| `nvda_streaming_labels.json` | Label analysis (streaming) |
| `nvda_streaming_signals.json` | Signal stats (streaming) |
| `nvda_streaming_days.json` | Per-day breakdown |

---

## 12. Interpretation Guide

### How to Read the Results

#### Label Autocorrelation (ACF)

```
Lag-1 ACF: 0.703
```

**Meaning**: 70% correlation with previous label. Strong clustering.

**Action**: Sequence models (LSTM/Transformer) will benefit from this temporal structure.

#### Transition Matrix

```
Down → Down: 97.2%
Down → Up: 0.0%
```

**Meaning**: Once in a downtrend, 97% chance to stay. Direct reversals nearly impossible.

**Action**: Focus model on detecting **transitions** (Down↔Stable, Stable↔Up), not steady states.

#### Signal-Label Correlation

```
true_ofi: r = +0.007, p < 0.001
```

**Meaning**: Weak but statistically significant positive correlation. Higher OFI → slightly more likely to go Up.

**Action**: Correlations < 0.01 are expected in HFT. Alpha is in subtle patterns, not linear relationships. Use as features but don't expect miracles.

#### Signal Stability Score

```
true_ofi: stability = 0.32
signed_mp_delta_bps: stability = 0.02
```

**Meaning**: true_ofi is 16× more stable across days than signed_mp_delta_bps.

**Action**: Prioritize stable signals for production models. Unstable signals may overfit.

#### Walk-Forward Accuracy

```
Average: 33%
```

**Meaning**: Simple threshold prediction slightly better than random (33% for 3 classes).

**Action**: More sophisticated models needed. This is the baseline to beat.

### Red Flags to Watch For

| Issue | Where to Find | Action |
|-------|--------------|--------|
| NaN/Inf in data | `data_quality.is_clean` | Investigate Rust pipeline |
| VIF > 10 | `vif_results` | Remove redundant signals |
| Non-stationary signals | `stationarity_results` | Consider differencing |
| Sign-flipping correlations | `signal_day_stats.correlations` | Use with caution |
| Low stability score | `generalization.overall_stability_score` | Need more data or adaptation |

---

## Appendix: Runner Scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_data_overview.py` | Generate DatasetSummary |
| `scripts/run_label_analysis.py` | Generate LabelAnalysisSummary |
| `scripts/run_signal_analysis.py` | Distribution + stationarity |
| `scripts/run_temporal_dynamics.py` | Temporal analysis |
| `scripts/run_generalization.py` | Walk-forward validation |
| `scripts/run_streaming_analysis.py` | Memory-efficient (1 day at a time) |
| `scripts/run_complete_streaming_analysis.py` | All analyses, streaming mode |

### Example Usage

```bash
# Full streaming analysis (recommended for large datasets)
cd lob-model-trainer
.venv/bin/python scripts/run_complete_streaming_analysis.py \
    --data-dir ../data/exports/nvda_98feat_full \
    --symbol NVDA

# Individual analysis
.venv/bin/python scripts/run_label_analysis.py \
    --data-dir ../data/exports/nvda_98feat_full \
    --symbol NVDA \
    --split train
```

