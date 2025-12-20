"""
Temporal dynamics analysis for signals.

Analyzes time-series properties critical for sequence model design:
- Signal autocorrelation (persistence/half-life)
- Cross-correlation (lead-lag relationships between signals)
- Predictive decay (how quickly signal-label correlation fades with lag)
- Signal level vs change analysis

These analyses inform:
- Optimal lookback window for sequence models
- Whether signals have temporal structure worth modeling
- Lead-lag relationships for feature engineering

References:
- Cont et al. (2014): Order flow persistence
- Sirignano & Cont (2019): Deep learning for LOB
"""

import numpy as np
from scipy.stats import pearsonr
from scipy.signal import correlate
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
import warnings

from ..constants import FeatureIndex, LABEL_DOWN, LABEL_STABLE, LABEL_UP
from .data_loading import get_signal_info, CORE_SIGNAL_INDICES


@dataclass
class SignalAutocorrelation:
    """Autocorrelation analysis for a single signal."""
    signal_name: str
    signal_index: int
    acf_values: List[float]  # ACF at each lag
    lags: List[int]
    half_life: int  # Lag at which ACF drops below 0.5
    decay_rate: float  # Exponential decay rate (higher = faster decay)
    persistence_interpretation: str


@dataclass
class LeadLagRelation:
    """Lead-lag relationship between two signals."""
    leader: str
    follower: str
    leader_index: int
    follower_index: int
    optimal_lag: int  # Positive = leader leads by this many samples
    max_correlation: float
    is_significant: bool


@dataclass
class PredictiveDecay:
    """How signal-label correlation decays with lag."""
    signal_name: str
    signal_index: int
    lags: List[int]
    correlations: List[float]
    half_life: int  # Lag at which correlation halves
    optimal_lag: int  # Lag with maximum |correlation|
    max_correlation: float


@dataclass
class LevelVsChangeAnalysis:
    """Compares predictive power of signal level vs change."""
    signal_name: str
    signal_index: int
    level_correlation: float
    change_correlation: float  # Correlation of signal change with label
    level_auc: float
    change_auc: float
    recommendation: str  # "level", "change", or "both"


@dataclass
class TemporalDynamicsSummary:
    """Complete temporal dynamics analysis results."""
    autocorrelations: List[SignalAutocorrelation]
    lead_lag_relations: List[LeadLagRelation]
    predictive_decays: List[PredictiveDecay]
    level_vs_change: List[LevelVsChangeAnalysis]
    optimal_lookback: int
    sequence_model_justified: bool
    justification: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            'autocorrelations': [asdict(a) for a in self.autocorrelations],
            'lead_lag_relations': [asdict(l) for l in self.lead_lag_relations],
            'predictive_decays': [asdict(p) for p in self.predictive_decays],
            'level_vs_change': [asdict(l) for l in self.level_vs_change],
            'optimal_lookback': self.optimal_lookback,
            'sequence_model_justified': self.sequence_model_justified,
            'justification': self.justification,
        }


def compute_autocorrelation(
    signal: np.ndarray,
    max_lag: int = 100,
) -> Tuple[np.ndarray, int, float]:
    """
    Compute autocorrelation function for a signal.
    
    ACF(k) = Corr(X_t, X_{t+k})
    
    Args:
        signal: 1D signal array
        max_lag: Maximum lag to compute
    
    Returns:
        acf: Array of ACF values from lag 0 to max_lag
        half_life: Lag at which ACF drops below 0.5
        decay_rate: Estimated exponential decay rate
    
    Formula:
        ACF(k) = Cov(X_t, X_{t+k}) / Var(X)
    """
    n = len(signal)
    if n < max_lag + 1:
        max_lag = n - 1
    
    signal = signal.astype(np.float64)
    mean = np.mean(signal)
    var = np.var(signal)
    
    if var < 1e-10:
        # Constant signal
        return np.ones(max_lag + 1), max_lag, 0.0
    
    # Compute ACF using FFT for efficiency
    # Normalize signal
    normalized = signal - mean
    
    acf = np.zeros(max_lag + 1)
    acf[0] = 1.0  # Correlation with self
    
    for lag in range(1, max_lag + 1):
        if n - lag > 0:
            cov = np.mean(normalized[:-lag] * normalized[lag:])
            acf[lag] = cov / var
    
    # Find half-life (first lag where ACF < 0.5)
    half_life = max_lag
    for i, val in enumerate(acf):
        if val < 0.5:
            half_life = i
            break
    
    # Estimate exponential decay rate
    # ACF(k) ≈ exp(-λk) → λ ≈ -ln(ACF(k))/k
    if acf[1] > 0 and acf[1] < 1:
        decay_rate = -np.log(acf[1])
    else:
        decay_rate = 0.0
    
    return acf, half_life, decay_rate


def compute_signal_autocorrelations(
    features: np.ndarray,
    signal_indices: List[int] = None,
    max_lag: int = 100,
) -> List[SignalAutocorrelation]:
    """
    Compute autocorrelation analysis for all signals.
    
    Args:
        features: (N, D) feature array (sample-level, NOT aligned)
        signal_indices: Which signals to analyze
        max_lag: Maximum lag for ACF
    
    Returns:
        List of SignalAutocorrelation for each signal
    """
    if signal_indices is None:
        signal_indices = CORE_SIGNAL_INDICES
    
    signal_info = get_signal_info()
    results = []
    
    for idx in signal_indices:
        info = signal_info.get(idx, {'name': f'signal_{idx}'})
        signal = features[:, idx]
        
        acf, half_life, decay_rate = compute_autocorrelation(signal, max_lag)
        
        # Interpretation
        if half_life <= 5:
            interpretation = "Fast decay: Signal changes rapidly, short memory"
        elif half_life <= 20:
            interpretation = "Moderate persistence: Some temporal structure"
        elif half_life <= 50:
            interpretation = "Strong persistence: Clear temporal patterns"
        else:
            interpretation = "Very persistent: Long-term trends dominate"
        
        results.append(SignalAutocorrelation(
            signal_name=info.get('name', f'signal_{idx}'),
            signal_index=idx,
            acf_values=acf.tolist(),
            lags=list(range(len(acf))),
            half_life=int(half_life),
            decay_rate=float(decay_rate),
            persistence_interpretation=interpretation,
        ))
    
    return results


def compute_cross_correlation(
    signal1: np.ndarray,
    signal2: np.ndarray,
    max_lag: int = 50,
) -> Tuple[int, float]:
    """
    Compute cross-correlation to find lead-lag relationship.
    
    Args:
        signal1: First signal
        signal2: Second signal
        max_lag: Maximum lag to consider
    
    Returns:
        optimal_lag: Lag at maximum correlation (positive = signal1 leads)
        max_corr: Maximum correlation value
    
    Formula:
        CCF(k) = Corr(X_t, Y_{t+k})
        k > 0 means X leads Y
    """
    n = min(len(signal1), len(signal2))
    signal1 = signal1[:n].astype(np.float64)
    signal2 = signal2[:n].astype(np.float64)
    
    # Normalize
    s1 = (signal1 - signal1.mean()) / (signal1.std() + 1e-10)
    s2 = (signal2 - signal2.mean()) / (signal2.std() + 1e-10)
    
    # Compute cross-correlation for different lags
    correlations = []
    lags = range(-max_lag, max_lag + 1)
    
    for lag in lags:
        if lag > 0:
            corr = np.corrcoef(s1[:-lag], s2[lag:])[0, 1]
        elif lag < 0:
            corr = np.corrcoef(s1[-lag:], s2[:lag])[0, 1]
        else:
            corr = np.corrcoef(s1, s2)[0, 1]
        
        if np.isfinite(corr):
            correlations.append(corr)
        else:
            correlations.append(0.0)
    
    correlations = np.array(correlations)
    
    # Find lag with maximum absolute correlation
    best_idx = np.argmax(np.abs(correlations))
    optimal_lag = list(lags)[best_idx]
    max_corr = correlations[best_idx]
    
    return optimal_lag, float(max_corr)


def compute_lead_lag_relations(
    features: np.ndarray,
    signal_indices: List[int] = None,
    max_lag: int = 20,
    min_correlation: float = 0.1,
) -> List[LeadLagRelation]:
    """
    Find lead-lag relationships between all pairs of signals.
    
    Args:
        features: (N, D) feature array (sample-level)
        signal_indices: Which signals to analyze
        max_lag: Maximum lag to consider
        min_correlation: Minimum |correlation| to report
    
    Returns:
        List of significant LeadLagRelation
    """
    if signal_indices is None:
        signal_indices = CORE_SIGNAL_INDICES
    
    signal_info = get_signal_info()
    results = []
    
    n_signals = len(signal_indices)
    for i in range(n_signals):
        for j in range(i + 1, n_signals):
            idx1, idx2 = signal_indices[i], signal_indices[j]
            signal1 = features[:, idx1]
            signal2 = features[:, idx2]
            
            optimal_lag, max_corr = compute_cross_correlation(
                signal1, signal2, max_lag
            )
            
            if abs(max_corr) < min_correlation:
                continue
            
            name1 = signal_info.get(idx1, {}).get('name', f'signal_{idx1}')
            name2 = signal_info.get(idx2, {}).get('name', f'signal_{idx2}')
            
            # Determine leader/follower
            if optimal_lag > 0:
                leader, follower = name1, name2
                leader_idx, follower_idx = idx1, idx2
            else:
                leader, follower = name2, name1
                leader_idx, follower_idx = idx2, idx1
                optimal_lag = -optimal_lag
            
            results.append(LeadLagRelation(
                leader=leader,
                follower=follower,
                leader_index=leader_idx,
                follower_index=follower_idx,
                optimal_lag=int(optimal_lag),
                max_correlation=float(max_corr),
                is_significant=abs(max_corr) > 0.2,
            ))
    
    # Sort by correlation strength
    results.sort(key=lambda x: abs(x.max_correlation), reverse=True)
    
    return results


def compute_predictive_decay(
    signal: np.ndarray,
    labels: np.ndarray,
    lags: List[int] = None,
) -> Tuple[List[float], int, int, float]:
    """
    Compute how signal-label correlation decays with lag.
    
    This answers: "How quickly does signal information become stale?"
    
    Args:
        signal: (N,) signal values (sample-level)
        labels: (M,) label values (may be shorter due to alignment)
        lags: List of lags to test
    
    Returns:
        correlations: Correlation at each lag
        half_life: Lag at which correlation halves
        optimal_lag: Lag with maximum |correlation|
        max_corr: Maximum correlation
    """
    if lags is None:
        lags = [0, 1, 2, 5, 10, 20, 50, 100]
    
    n_signal = len(signal)
    n_labels = len(labels)
    
    correlations = []
    
    for lag in lags:
        # For each lag, correlate lagged signal with labels
        # Note: Need to handle alignment between sample-level signal and sequence-level labels
        # Assuming labels are spaced by stride=10 samples
        stride = 10
        window_size = 100
        
        # Align: label[i] corresponds to signal at (i*stride + window_size - 1)
        # With lag, we use signal at (i*stride + window_size - 1 - lag)
        
        valid_labels = []
        valid_signals = []
        
        for i in range(n_labels):
            signal_idx = i * stride + window_size - 1 - lag
            if 0 <= signal_idx < n_signal:
                valid_labels.append(labels[i])
                valid_signals.append(signal[signal_idx])
        
        if len(valid_labels) < 100:
            correlations.append(0.0)
            continue
        
        valid_labels = np.array(valid_labels)
        valid_signals = np.array(valid_signals)
        
        corr, _ = pearsonr(valid_signals, valid_labels)
        correlations.append(float(corr) if np.isfinite(corr) else 0.0)
    
    correlations = np.array(correlations)
    
    # Find optimal lag (max |correlation|)
    if len(correlations) > 0:
        best_idx = np.argmax(np.abs(correlations))
        optimal_lag = lags[best_idx]
        max_corr = correlations[best_idx]
        
        # Find half-life (first lag where |correlation| drops below half of max)
        half_threshold = abs(max_corr) / 2
        half_life = lags[-1]
        for i, (lag, corr) in enumerate(zip(lags, correlations)):
            if abs(corr) < half_threshold:
                half_life = lag
                break
    else:
        optimal_lag, max_corr, half_life = 0, 0.0, 0
    
    return correlations.tolist(), half_life, optimal_lag, max_corr


def compute_all_predictive_decays(
    features: np.ndarray,
    labels: np.ndarray,
    signal_indices: List[int] = None,
    lags: List[int] = None,
) -> List[PredictiveDecay]:
    """
    Compute predictive decay for all signals.
    
    Args:
        features: (N, D) feature array (sample-level)
        labels: (M,) label array
        signal_indices: Which signals to analyze
        lags: Lags to test
    
    Returns:
        List of PredictiveDecay for each signal
    """
    if signal_indices is None:
        signal_indices = CORE_SIGNAL_INDICES
    
    if lags is None:
        lags = [0, 1, 2, 5, 10, 20, 50, 100]
    
    signal_info = get_signal_info()
    results = []
    
    for idx in signal_indices:
        info = signal_info.get(idx, {'name': f'signal_{idx}'})
        signal = features[:, idx]
        
        corrs, half_life, optimal_lag, max_corr = compute_predictive_decay(
            signal, labels, lags
        )
        
        results.append(PredictiveDecay(
            signal_name=info.get('name', f'signal_{idx}'),
            signal_index=idx,
            lags=lags,
            correlations=corrs,
            half_life=half_life,
            optimal_lag=optimal_lag,
            max_correlation=max_corr,
        ))
    
    return results


def compute_level_vs_change(
    features: np.ndarray,
    labels: np.ndarray,
    signal_index: int,
    window_size: int = 100,
    stride: int = 10,
) -> LevelVsChangeAnalysis:
    """
    Compare predictive power of signal level vs signal change.
    
    Args:
        features: (N, D) feature array (sample-level)
        labels: (M,) label array
        signal_index: Which signal to analyze
        window_size: Samples per sequence window
        stride: Samples between sequences
    
    Returns:
        LevelVsChangeAnalysis comparing level vs change
    """
    from sklearn.metrics import roc_auc_score
    
    signal_info = get_signal_info()
    info = signal_info.get(signal_index, {'name': f'signal_{signal_index}'})
    signal = features[:, signal_index]
    
    n_labels = len(labels)
    
    # Extract aligned signal levels (at end of each window)
    levels = []
    changes = []
    valid_labels = []
    
    for i in range(n_labels):
        end_idx = i * stride + window_size - 1
        start_idx = i * stride
        
        if end_idx >= len(signal) or start_idx < 0:
            continue
        
        # Level at end of window
        level = signal[end_idx]
        
        # Change over the window
        change = signal[end_idx] - signal[start_idx]
        
        levels.append(level)
        changes.append(change)
        valid_labels.append(labels[i])
    
    levels = np.array(levels)
    changes = np.array(changes)
    valid_labels = np.array(valid_labels)
    
    if len(valid_labels) < 100:
        return LevelVsChangeAnalysis(
            signal_name=info.get('name', f'signal_{signal_index}'),
            signal_index=signal_index,
            level_correlation=0.0,
            change_correlation=0.0,
            level_auc=0.5,
            change_auc=0.5,
            recommendation="insufficient_data",
        )
    
    # Compute correlations
    level_corr, _ = pearsonr(levels, valid_labels)
    change_corr, _ = pearsonr(changes, valid_labels)
    
    # Compute AUC for Up prediction
    y_up = (valid_labels == LABEL_UP).astype(int)
    
    if 0 < y_up.sum() < len(y_up):
        level_auc = roc_auc_score(y_up, levels)
        change_auc = roc_auc_score(y_up, changes)
    else:
        level_auc = 0.5
        change_auc = 0.5
    
    # Determine recommendation
    level_score = abs(level_corr) + (level_auc - 0.5)
    change_score = abs(change_corr) + (change_auc - 0.5)
    
    if level_score > change_score * 1.2:
        recommendation = "level"
    elif change_score > level_score * 1.2:
        recommendation = "change"
    else:
        recommendation = "both"
    
    return LevelVsChangeAnalysis(
        signal_name=info.get('name', f'signal_{signal_index}'),
        signal_index=signal_index,
        level_correlation=float(level_corr) if np.isfinite(level_corr) else 0.0,
        change_correlation=float(change_corr) if np.isfinite(change_corr) else 0.0,
        level_auc=float(level_auc),
        change_auc=float(change_auc),
        recommendation=recommendation,
    )


def compute_all_level_vs_change(
    features: np.ndarray,
    labels: np.ndarray,
    signal_indices: List[int] = None,
    window_size: int = 100,
    stride: int = 10,
) -> List[LevelVsChangeAnalysis]:
    """
    Compare level vs change for all signals.
    """
    if signal_indices is None:
        signal_indices = CORE_SIGNAL_INDICES
    
    return [
        compute_level_vs_change(features, labels, idx, window_size, stride)
        for idx in signal_indices
    ]


def run_temporal_dynamics_analysis(
    features: np.ndarray,
    labels: np.ndarray,
    signal_indices: List[int] = None,
    max_acf_lag: int = 100,
    max_leadlag_lag: int = 20,
    window_size: int = 100,
    stride: int = 10,
) -> TemporalDynamicsSummary:
    """
    Run complete temporal dynamics analysis.
    
    Args:
        features: (N, D) feature array (sample-level)
        labels: (M,) label array
        signal_indices: Which signals to analyze
        max_acf_lag: Maximum lag for autocorrelation
        max_leadlag_lag: Maximum lag for lead-lag analysis
        window_size: Samples per sequence window
        stride: Samples between sequences
    
    Returns:
        TemporalDynamicsSummary with all results
    """
    if signal_indices is None:
        signal_indices = CORE_SIGNAL_INDICES
    
    # 1. Signal autocorrelations
    autocorrelations = compute_signal_autocorrelations(
        features, signal_indices, max_acf_lag
    )
    
    # 2. Lead-lag relationships
    lead_lag_relations = compute_lead_lag_relations(
        features, signal_indices, max_leadlag_lag
    )
    
    # 3. Predictive decay
    predictive_decays = compute_all_predictive_decays(
        features, labels, signal_indices
    )
    
    # 4. Level vs change
    level_vs_change = compute_all_level_vs_change(
        features, labels, signal_indices, window_size, stride
    )
    
    # 5. Determine optimal lookback and sequence model justification
    # Average half-life across signals
    avg_half_life = np.mean([a.half_life for a in autocorrelations])
    max_half_life = max(a.half_life for a in autocorrelations)
    
    # Check if any signal has persistent predictive power
    has_persistent_prediction = any(
        p.half_life > 10 for p in predictive_decays
    )
    
    # Check for significant lead-lag
    has_lead_lag = any(r.is_significant for r in lead_lag_relations)
    
    # Determine optimal lookback (2x average half-life, but at least 50)
    optimal_lookback = max(50, min(int(avg_half_life * 2), max_acf_lag))
    
    # Sequence model justification
    sequence_justified = (
        avg_half_life > 10 or 
        has_persistent_prediction or 
        has_lead_lag
    )
    
    if sequence_justified:
        justification = (
            f"Sequence model JUSTIFIED: "
            f"avg_half_life={avg_half_life:.1f}, "
            f"persistent_prediction={has_persistent_prediction}, "
            f"lead_lag_found={has_lead_lag}"
        )
    else:
        justification = (
            f"Sequence model may NOT be needed: "
            f"avg_half_life={avg_half_life:.1f} (low), "
            f"no persistent prediction, no significant lead-lag"
        )
    
    return TemporalDynamicsSummary(
        autocorrelations=autocorrelations,
        lead_lag_relations=lead_lag_relations,
        predictive_decays=predictive_decays,
        level_vs_change=level_vs_change,
        optimal_lookback=optimal_lookback,
        sequence_model_justified=sequence_justified,
        justification=justification,
    )


def print_temporal_dynamics(summary: TemporalDynamicsSummary) -> None:
    """Print formatted temporal dynamics analysis."""
    print("=" * 80)
    print("TEMPORAL DYNAMICS ANALYSIS")
    print("=" * 80)
    
    # 1. Autocorrelation
    print("\n1. SIGNAL AUTOCORRELATION (Persistence)")
    print("-" * 60)
    print(f"{'Signal':<25} {'Half-life':>10} {'Decay Rate':>12} Interpretation")
    for a in sorted(summary.autocorrelations, key=lambda x: -x.half_life):
        print(f"{a.signal_name:<25} {a.half_life:>10} {a.decay_rate:>12.4f} {a.persistence_interpretation}")
    
    # 2. Lead-Lag
    print("\n2. LEAD-LAG RELATIONSHIPS")
    print("-" * 60)
    if summary.lead_lag_relations:
        print(f"{'Leader':<20} {'Follower':<20} {'Lag':>6} {'Corr':>8} {'Sig':>5}")
        for r in summary.lead_lag_relations[:10]:  # Top 10
            sig = "***" if r.is_significant else ""
            print(f"{r.leader:<20} {r.follower:<20} {r.optimal_lag:>6} {r.max_correlation:>+8.3f} {sig:>5}")
    else:
        print("  No significant lead-lag relationships found")
    
    # 3. Predictive Decay
    print("\n3. PREDICTIVE DECAY (Signal-Label Correlation vs Lag)")
    print("-" * 60)
    print(f"{'Signal':<25} {'Optimal Lag':>11} {'Max Corr':>10} {'Half-life':>10}")
    for p in sorted(summary.predictive_decays, key=lambda x: -abs(x.max_correlation)):
        print(f"{p.signal_name:<25} {p.optimal_lag:>11} {p.max_correlation:>+10.4f} {p.half_life:>10}")
    
    # 4. Level vs Change
    print("\n4. LEVEL VS CHANGE ANALYSIS")
    print("-" * 60)
    print(f"{'Signal':<25} {'Level r':>10} {'Change r':>10} {'Level AUC':>10} {'Change AUC':>11} Recommendation")
    for l in summary.level_vs_change:
        print(f"{l.signal_name:<25} {l.level_correlation:>+10.4f} {l.change_correlation:>+10.4f} "
              f"{l.level_auc:>10.4f} {l.change_auc:>11.4f} {l.recommendation}")
    
    # 5. Summary
    print("\n" + "=" * 80)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 80)
    
    print(f"\n  Optimal lookback window: {summary.optimal_lookback} samples")
    print(f"  Sequence model justified: {'✅ YES' if summary.sequence_model_justified else '❌ NO'}")
    print(f"  Justification: {summary.justification}")
    
    # Key insights
    print("\n  KEY INSIGHTS:")
    
    # Most persistent signal
    most_persistent = max(summary.autocorrelations, key=lambda x: x.half_life)
    print(f"    • Most persistent signal: {most_persistent.signal_name} (half-life={most_persistent.half_life})")
    
    # Best predictor at lag 0
    best_predictor = max(summary.predictive_decays, key=lambda x: abs(x.correlations[0]) if x.correlations else 0)
    lag0_corr = best_predictor.correlations[0] if best_predictor.correlations else 0
    print(f"    • Best immediate predictor: {best_predictor.signal_name} (r={lag0_corr:+.4f} at lag 0)")
    
    # Any signal where change beats level?
    change_winners = [l for l in summary.level_vs_change if l.recommendation == "change"]
    if change_winners:
        print(f"    • Signals where CHANGE beats level: {', '.join(l.signal_name for l in change_winners)}")
    else:
        print(f"    • Level information dominates for all signals (change not more predictive)")
    
    print("\n" + "=" * 80)
    print("✅ TEMPORAL DYNAMICS ANALYSIS COMPLETE")
    print("=" * 80)

