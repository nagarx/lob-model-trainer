"""
Label analysis utilities.

Provides comprehensive analysis of label characteristics:
- Distribution and class balance
- Autocorrelation (clustering/momentum detection)
- Transition probabilities (Markov analysis)
- Regime-specific label behavior
- Signal-label correlations

All functions are designed to be reusable across different datasets/symbols.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from scipy.stats import chi2_contingency

from ..constants import (
    FEATURE_COUNT,
    FeatureIndex,
    LABEL_DOWN, LABEL_STABLE, LABEL_UP, LABEL_NAMES,
)

# Time regime names for display
REGIME_NAMES = {
    0: 'Open (9:30-9:45)',
    1: 'Early (9:45-10:30)',
    2: 'Midday (10:30-15:30)',
    3: 'Close (15:30-16:00)',
    4: 'Closed',
}


@dataclass
class LabelDistribution:
    """Label distribution statistics."""
    total: int
    down_count: int
    stable_count: int
    up_count: int
    down_pct: float
    stable_pct: float
    up_pct: float
    imbalance_ratio: float
    majority_class: str
    minority_class: str
    
    @property
    def is_balanced(self) -> bool:
        """Labels are balanced if max/min ratio < 1.5."""
        return self.imbalance_ratio < 1.5


@dataclass
class AutocorrelationResult:
    """Autocorrelation analysis results."""
    lags: List[int]
    acf_values: List[float]
    confidence_interval: float
    lag_1_acf: float
    lag_5_acf: float
    lag_10_acf: float
    interpretation: str


@dataclass 
class TransitionMatrix:
    """Markov transition matrix analysis."""
    labels: List[int]  # Ordered labels (e.g., [-1, 0, 1])
    counts: List[List[int]]  # Count matrix
    probabilities: List[List[float]]  # Probability matrix
    stationary_probs: List[float]  # Stationary distribution
    persistence_deviation: Dict[str, float]  # Deviation from stationary per class


@dataclass
class RegimeStats:
    """Label statistics for a single time regime."""
    regime: int
    name: str
    n_samples: int
    up_pct: float
    down_pct: float
    stable_pct: float
    ofi_correlation: float


@dataclass
class SignalCorrelation:
    """Correlation between a signal and labels."""
    signal_name: str
    signal_index: int
    correlation: float
    p_value: float
    is_significant: bool


@dataclass
class LabelAnalysisSummary:
    """Comprehensive label analysis results."""
    distribution: LabelDistribution
    autocorrelation: AutocorrelationResult
    transition_matrix: TransitionMatrix
    regime_stats: List[RegimeStats]
    signal_correlations: List[SignalCorrelation]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            'distribution': asdict(self.distribution),
            'autocorrelation': asdict(self.autocorrelation),
            'transition_matrix': asdict(self.transition_matrix),
            'regime_stats': [asdict(r) for r in self.regime_stats],
            'signal_correlations': [asdict(s) for s in self.signal_correlations],
        }


def compute_label_distribution(labels: np.ndarray) -> LabelDistribution:
    """
    Compute label distribution statistics.
    
    Args:
        labels: Label array with values in {-1, 0, 1}
    
    Returns:
        LabelDistribution with counts, percentages, and balance metrics
    """
    total = len(labels)
    
    down_count = int((labels == LABEL_DOWN).sum())
    stable_count = int((labels == LABEL_STABLE).sum())
    up_count = int((labels == LABEL_UP).sum())
    
    down_pct = 100 * down_count / total if total > 0 else 0
    stable_pct = 100 * stable_count / total if total > 0 else 0
    up_pct = 100 * up_count / total if total > 0 else 0
    
    counts = [(down_count, LABEL_DOWN), (stable_count, LABEL_STABLE), (up_count, LABEL_UP)]
    max_count, max_label = max(counts, key=lambda x: x[0])
    min_count, min_label = min(counts, key=lambda x: x[0])
    
    imbalance_ratio = max_count / max(min_count, 1)
    
    return LabelDistribution(
        total=total,
        down_count=down_count,
        stable_count=stable_count,
        up_count=up_count,
        down_pct=down_pct,
        stable_pct=stable_pct,
        up_pct=up_pct,
        imbalance_ratio=imbalance_ratio,
        majority_class=LABEL_NAMES.get(max_label, str(max_label)),
        minority_class=LABEL_NAMES.get(min_label, str(min_label)),
    )


def compute_autocorrelation(
    labels: np.ndarray, 
    max_lag: int = 100,
) -> AutocorrelationResult:
    """
    Compute autocorrelation function for labels.
    
    ACF(k) = Corr(label_t, label_{t+k})
    
    Args:
        labels: Label array
        max_lag: Maximum lag to compute (default: 100)
    
    Returns:
        AutocorrelationResult with ACF values and interpretation
    """
    n = len(labels)
    labels_float = labels.astype(float)
    mean = labels_float.mean()
    var = labels_float.var()
    
    # 95% confidence interval for white noise: ±1.96/sqrt(n)
    ci = 1.96 / np.sqrt(n)
    
    if var == 0:
        # Constant labels
        acf = np.ones(max_lag + 1)
    else:
        acf = np.zeros(max_lag + 1)
        acf[0] = 1.0  # Correlation with self
        
        for lag in range(1, min(max_lag + 1, n)):
            cov = np.mean((labels_float[:-lag] - mean) * (labels_float[lag:] - mean))
            acf[lag] = cov / var
    
    # Interpretation
    lag_1 = acf[1] if len(acf) > 1 else 0
    if lag_1 > 0.1:
        interpretation = "Strong positive autocorrelation: Labels cluster (trends persist)"
    elif lag_1 > ci:
        interpretation = "Weak positive autocorrelation: Some label persistence"
    elif lag_1 < -ci:
        interpretation = "Negative autocorrelation: Mean-reversion in labels"
    else:
        interpretation = "No significant autocorrelation: Labels appear random"
    
    return AutocorrelationResult(
        lags=list(range(len(acf))),
        acf_values=acf.tolist(),
        confidence_interval=float(ci),
        lag_1_acf=float(acf[1]) if len(acf) > 1 else 0.0,
        lag_5_acf=float(acf[5]) if len(acf) > 5 else 0.0,
        lag_10_acf=float(acf[10]) if len(acf) > 10 else 0.0,
        interpretation=interpretation,
    )


def compute_transition_matrix(labels: np.ndarray) -> TransitionMatrix:
    """
    Compute Markov transition matrix for labels.
    
    P[i, j] = P(label_{t+1} = j | label_t = i)
    
    Args:
        labels: Label array
    
    Returns:
        TransitionMatrix with counts, probabilities, and analysis
    """
    label_order = sorted(np.unique(labels).tolist())
    n_labels = len(label_order)
    label_to_idx = {lbl: i for i, lbl in enumerate(label_order)}
    
    # Count transitions
    counts = np.zeros((n_labels, n_labels), dtype=int)
    for i in range(len(labels) - 1):
        from_idx = label_to_idx[labels[i]]
        to_idx = label_to_idx[labels[i + 1]]
        counts[from_idx, to_idx] += 1
    
    # Normalize to probabilities
    row_sums = counts.sum(axis=1, keepdims=True)
    probs = np.divide(counts, row_sums, where=row_sums > 0, out=np.zeros_like(counts, dtype=float))
    
    # Stationary distribution (proportion of each label)
    total = len(labels)
    stationary_probs = [(labels == lbl).sum() / total for lbl in label_order]
    
    # Persistence deviation: how much P(same) deviates from stationary
    persistence_deviation = {}
    for i, lbl in enumerate(label_order):
        name = LABEL_NAMES.get(lbl, str(lbl))
        persist_prob = probs[i, i]
        stationary = stationary_probs[i]
        deviation = persist_prob - stationary
        persistence_deviation[name] = float(deviation)
    
    return TransitionMatrix(
        labels=label_order,
        counts=counts.tolist(),
        probabilities=probs.tolist(),
        stationary_probs=stationary_probs,
        persistence_deviation=persistence_deviation,
    )


def compute_regime_stats(
    aligned_features: np.ndarray,
    labels: np.ndarray,
    min_samples: int = 100,
) -> List[RegimeStats]:
    """
    Compute label statistics for each time regime.
    
    Args:
        aligned_features: Features aligned with labels (n_labels, n_features)
        labels: Label array
        min_samples: Minimum samples for a regime to be included
    
    Returns:
        List of RegimeStats for each regime with sufficient samples
    """
    time_regimes = aligned_features[:, FeatureIndex.TIME_REGIME]
    true_ofi = aligned_features[:, FeatureIndex.TRUE_OFI]
    
    stats_list = []
    for regime in sorted(np.unique(time_regimes)):
        regime_int = int(regime)
        mask = time_regimes == regime
        regime_labels = labels[mask]
        
        if len(regime_labels) < min_samples:
            continue
        
        up_pct = 100 * (regime_labels == LABEL_UP).mean()
        down_pct = 100 * (regime_labels == LABEL_DOWN).mean()
        stable_pct = 100 * (regime_labels == LABEL_STABLE).mean()
        
        # OFI-label correlation in this regime
        regime_ofi = true_ofi[mask]
        ofi_corr = np.corrcoef(regime_ofi, regime_labels)[0, 1]
        
        stats_list.append(RegimeStats(
            regime=regime_int,
            name=REGIME_NAMES.get(regime_int, f"Regime {regime_int}"),
            n_samples=len(regime_labels),
            up_pct=float(up_pct),
            down_pct=float(down_pct),
            stable_pct=float(stable_pct),
            ofi_correlation=float(ofi_corr) if np.isfinite(ofi_corr) else 0.0,
        ))
    
    return stats_list


def compute_signal_label_correlations(
    aligned_features: np.ndarray,
    labels: np.ndarray,
    signal_indices: Optional[Dict[int, str]] = None,
) -> List[SignalCorrelation]:
    """
    Compute correlations between signals and labels.
    
    Args:
        aligned_features: Features aligned with labels
        labels: Label array
        signal_indices: Optional dict mapping index -> name
            If None, uses default signal indices (84-91)
    
    Returns:
        List of SignalCorrelation sorted by absolute correlation
    """
    if signal_indices is None:
        signal_indices = {
            84: 'true_ofi',
            85: 'depth_norm_ofi',
            86: 'executed_pressure',
            87: 'signed_mp_delta_bps',
            88: 'trade_asymmetry',
            89: 'cancel_asymmetry',
            90: 'fragility_score',
            91: 'depth_asymmetry',
        }
    
    from scipy.stats import pearsonr
    
    correlations = []
    n = len(labels)
    
    for idx, name in signal_indices.items():
        signal = aligned_features[:, idx]
        
        # Skip if signal is constant
        if np.std(signal) == 0:
            correlations.append(SignalCorrelation(
                signal_name=name,
                signal_index=idx,
                correlation=0.0,
                p_value=1.0,
                is_significant=False,
            ))
            continue
        
        corr, p_value = pearsonr(signal, labels)
        
        # Significance threshold (Bonferroni-corrected)
        alpha = 0.05 / len(signal_indices)
        is_significant = p_value < alpha
        
        correlations.append(SignalCorrelation(
            signal_name=name,
            signal_index=idx,
            correlation=float(corr),
            p_value=float(p_value),
            is_significant=is_significant,
        ))
    
    # Sort by absolute correlation (descending)
    correlations.sort(key=lambda x: abs(x.correlation), reverse=True)
    
    return correlations


def run_label_analysis(
    features: np.ndarray,
    labels: np.ndarray,
    window_size: int = 100,
    stride: int = 10,
) -> LabelAnalysisSummary:
    """
    Run comprehensive label analysis.
    
    Args:
        features: Full feature array (sample-level)
        labels: Label array (sequence-level)
        window_size: Samples per sequence window
        stride: Samples between sequence starts
    
    Returns:
        LabelAnalysisSummary with all analysis results
    """
    from .data_loading import align_features_with_labels
    
    # Align features with labels
    aligned_features = align_features_with_labels(features, len(labels), window_size, stride)
    
    # Run all analyses
    distribution = compute_label_distribution(labels)
    autocorrelation = compute_autocorrelation(labels)
    transition_matrix = compute_transition_matrix(labels)
    regime_stats = compute_regime_stats(aligned_features, labels)
    signal_correlations = compute_signal_label_correlations(aligned_features, labels)
    
    return LabelAnalysisSummary(
        distribution=distribution,
        autocorrelation=autocorrelation,
        transition_matrix=transition_matrix,
        regime_stats=regime_stats,
        signal_correlations=signal_correlations,
    )


def print_label_analysis(summary: LabelAnalysisSummary) -> None:
    """
    Print formatted label analysis to console.
    
    Args:
        summary: LabelAnalysisSummary to print
    """
    print("=" * 70)
    print("LABEL ANALYSIS SUMMARY")
    print("=" * 70)
    
    # Distribution
    print("\n1. LABEL DISTRIBUTION")
    print("-" * 50)
    d = summary.distribution
    print(f"Total labels: {d.total:,}")
    print(f"Down:   {d.down_count:6,} ({d.down_pct:5.2f}%)")
    print(f"Stable: {d.stable_count:6,} ({d.stable_pct:5.2f}%)")
    print(f"Up:     {d.up_count:6,} ({d.up_pct:5.2f}%)")
    print(f"Imbalance ratio: {d.imbalance_ratio:.2f}")
    print(f"Majority class: {d.majority_class}")
    print(f"Minority class: {d.minority_class}")
    status = "✅ Balanced (ratio < 1.5)" if d.is_balanced else "⚠️ Imbalanced (consider weighting)"
    print(f"Status: {status}")
    
    # Autocorrelation
    print("\n2. AUTOCORRELATION")
    print("-" * 50)
    a = summary.autocorrelation
    print(f"Lag-1 ACF:  {a.lag_1_acf:+.4f}")
    print(f"Lag-5 ACF:  {a.lag_5_acf:+.4f}")
    print(f"Lag-10 ACF: {a.lag_10_acf:+.4f}")
    print(f"95% CI (white noise): ±{a.confidence_interval:.4f}")
    print(f"Interpretation: {a.interpretation}")
    
    # Transition matrix
    print("\n3. TRANSITION MATRIX")
    print("-" * 50)
    t = summary.transition_matrix
    labels = t.labels
    probs = t.probabilities
    
    # Header
    print(f"{'From \\ To':>10}", end='')
    for lbl in labels:
        print(f"{LABEL_NAMES.get(lbl, str(lbl)):>10}", end='')
    print()
    
    # Rows
    for i, from_lbl in enumerate(labels):
        print(f"{LABEL_NAMES.get(from_lbl, str(from_lbl)):>10}", end='')
        for j in range(len(labels)):
            print(f"{probs[i][j]:>10.3f}", end='')
        print()
    
    print("\nPersistence deviation from stationary:")
    for name, dev in t.persistence_deviation.items():
        direction = "persists" if dev > 0.05 else "reverts" if dev < -0.05 else "neutral"
        print(f"  {name}: {dev:+.3f} ({direction})")
    
    # Regime stats
    print("\n4. TIME REGIME ANALYSIS")
    print("-" * 50)
    if summary.regime_stats:
        print(f"{'Regime':<25} {'N':>8} {'Up%':>7} {'Down%':>7} {'Stable%':>8} {'OFI r':>8}")
        for r in summary.regime_stats:
            print(f"{r.name:<25} {r.n_samples:>8,} {r.up_pct:>6.1f}% {r.down_pct:>6.1f}% {r.stable_pct:>7.1f}% {r.ofi_correlation:>+8.4f}")
    else:
        print("No regime data available")
    
    # Signal correlations
    print("\n5. SIGNAL-LABEL CORRELATIONS")
    print("-" * 50)
    print(f"{'Signal':<25} {'Corr':>10} {'p-value':>12} {'Significant':>12}")
    for s in summary.signal_correlations:
        sig = "***" if s.is_significant else ""
        print(f"{s.signal_name:<25} {s.correlation:>+10.4f} {s.p_value:>12.2e} {sig:>12}")
    
    # Top predictors
    print("\nTop predictors:")
    for i, s in enumerate(summary.signal_correlations[:3], 1):
        direction = "bullish" if s.correlation > 0 else "bearish"
        print(f"  {i}. {s.signal_name}: r = {s.correlation:+.4f} ({direction})")
    
    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    recommendations = []
    
    if summary.distribution.is_balanced:
        recommendations.append("✅ Use standard cross-entropy loss (balanced classes)")
    else:
        recommendations.append("⚠️ Use class-weighted loss or oversampling")
    
    if summary.autocorrelation.lag_1_acf > 0.05:
        recommendations.append("✅ Sequence models (LSTM/Transformer) likely beneficial")
    else:
        recommendations.append("⚠️ Point prediction models may suffice")
    
    top_signal = summary.signal_correlations[0] if summary.signal_correlations else None
    if top_signal and abs(top_signal.correlation) > 0.05:
        recommendations.append(f"✅ Focus on {top_signal.signal_name} as primary feature")
    else:
        recommendations.append("⚠️ Need feature engineering or more signals")
    
    for rec in recommendations:
        print(f"  {rec}")
    
    print("\n" + "=" * 70)
    print("✅ LABEL ANALYSIS COMPLETE")
    print("=" * 70)

