"""
Intraday Seasonality Analysis: Regime-Stratified Signal Analysis.

This module analyzes how signal-label relationships vary across different market sessions.

Background (Cont et al. 2014 §3.3):
    "Price impact is five times higher at the market open compared to the market close."
    
    The paper demonstrates that the price impact coefficient β varies significantly
    by time of day, with highest values at market open and lowest at close.

Time Regime Encoding (from feature index 93):
    0 = OPEN (9:30-9:45 ET): High volatility, wide spreads, information release
    1 = EARLY (9:45-10:30 ET): Settling period, still elevated activity
    2 = MIDDAY (10:30-15:30 ET): Most stable, lowest impact
    3 = CLOSE (15:30-16:00 ET): Position squaring, increased activity
    4 = CLOSED (After hours): Low liquidity, should be filtered

Key Questions Answered:
    1. Do signal-label correlations differ by regime?
    2. Is label variance (volatility) regime-dependent?
    3. Should models treat different regimes differently?
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np
from scipy import stats
import warnings

from ..constants import FeatureIndex

# Regime definitions matching Rust implementation
REGIME_NAMES = {
    0: "OPEN",
    1: "EARLY", 
    2: "MIDDAY",
    3: "CLOSE",
    4: "CLOSED",
}

# Core signals to analyze (continuous signals only, exclude categorical)
CORE_SIGNAL_INDICES = {
    'true_ofi': FeatureIndex.TRUE_OFI,
    'depth_norm_ofi': FeatureIndex.DEPTH_NORM_OFI,
    'executed_pressure': FeatureIndex.EXECUTED_PRESSURE,
    'signed_mp_delta_bps': FeatureIndex.SIGNED_MP_DELTA_BPS,
    'trade_asymmetry': FeatureIndex.TRADE_ASYMMETRY,
    'cancel_asymmetry': FeatureIndex.CANCEL_ASYMMETRY,
    'fragility_score': FeatureIndex.FRAGILITY_SCORE,
    'depth_asymmetry': FeatureIndex.DEPTH_ASYMMETRY,
}

# Expected sign for each signal (> 0 = buy pressure → expect UP label)
# From 01-SIGNAL-HIERARCHY.md
EXPECTED_SIGNS = {
    'true_ofi': '+',           # Positive OFI → expect Up
    'depth_norm_ofi': '+',     # Same as OFI
    'executed_pressure': '+',  # Net buying → expect Up
    'signed_mp_delta_bps': '+', # Microprice above mid → expect Up
    'trade_asymmetry': '+',    # More buy trades → expect Up
    'cancel_asymmetry': '+',   # Ask cancels → expect Up (liquidity withdrawal)
    'fragility_score': '?',    # Higher = more fragile (direction unclear)
    'depth_asymmetry': '+',    # More bid depth → expect Up (support)
}


@dataclass
class RegimeStats:
    """Statistics for a single regime."""
    regime: int
    regime_name: str
    n_samples: int
    label_mean: float
    label_std: float
    label_distribution: Dict[int, float]  # {-1: pct, 0: pct, 1: pct}


@dataclass
class SignalRegimeCorrelation:
    """Signal-label correlation for a specific regime."""
    signal_name: str
    regime: int
    regime_name: str
    n_samples: int
    correlation: float
    p_value: float
    is_significant: bool  # p < 0.01
    expected_sign: str
    sign_consistent: bool  # Does correlation sign match expected?


@dataclass
class SignalSeasonality:
    """
    Seasonality analysis for a single signal across all regimes.
    
    Attributes:
        signal_name: Name of the signal
        regime_correlations: Correlation in each regime
        correlation_range: max - min correlation across regimes
        most_predictive_regime: Regime with highest |correlation|
        least_predictive_regime: Regime with lowest |correlation|
        is_regime_dependent: True if correlation_range > threshold
        regime_impact_ratio: max|corr| / min|corr| (how much regimes differ)
    """
    signal_name: str
    regime_correlations: Dict[int, float]  # regime -> correlation
    correlation_range: float
    most_predictive_regime: int
    least_predictive_regime: int
    is_regime_dependent: bool
    regime_impact_ratio: float


@dataclass
class IntradaySeasonalitySummary:
    """
    Complete intraday seasonality analysis.
    
    Attributes:
        regime_stats: Basic statistics per regime
        signal_regime_correlations: All signal-regime correlations
        signal_seasonality: Seasonality summary per signal
        overall_regime_importance: Ranking of regimes by predictive power
        recommendations: Data-driven recommendations
    """
    regime_stats: List[RegimeStats]
    signal_regime_correlations: List[SignalRegimeCorrelation]
    signal_seasonality: List[SignalSeasonality]
    overall_regime_importance: Dict[int, float]  # regime -> avg |correlation|
    recommendations: List[str]


def compute_regime_stats(
    labels: np.ndarray,
    time_regimes: np.ndarray,
) -> List[RegimeStats]:
    """
    Compute label statistics per time regime.
    
    Args:
        labels: Array of labels (-1, 0, 1)
        time_regimes: Array of regime values (0-4)
    
    Returns:
        List of RegimeStats, one per unique regime
    
    Raises:
        ValueError: If arrays have different lengths or are empty
    """
    if len(labels) != len(time_regimes):
        raise ValueError(
            f"Length mismatch: labels={len(labels)}, time_regimes={len(time_regimes)}"
        )
    
    if len(labels) == 0:
        raise ValueError("Empty arrays provided")
    
    results = []
    unique_regimes = sorted(np.unique(time_regimes[np.isfinite(time_regimes)]).astype(int))
    
    for regime in unique_regimes:
        if regime not in REGIME_NAMES:
            warnings.warn(f"Unknown regime value: {regime}")
            continue
            
        mask = time_regimes == regime
        regime_labels = labels[mask]
        
        if len(regime_labels) == 0:
            continue
        
        # Label distribution
        label_dist = {}
        for label_val in [-1, 0, 1]:
            count = np.sum(regime_labels == label_val)
            label_dist[label_val] = float(count / len(regime_labels) * 100)
        
        results.append(RegimeStats(
            regime=regime,
            regime_name=REGIME_NAMES[regime],
            n_samples=int(np.sum(mask)),
            label_mean=float(np.mean(regime_labels)),
            label_std=float(np.std(regime_labels)),
            label_distribution=label_dist,
        ))
    
    return results


def compute_signal_regime_correlation(
    signal: np.ndarray,
    labels: np.ndarray,
    signal_name: str,
    regime: int,
) -> SignalRegimeCorrelation:
    """
    Compute correlation between a signal and labels for a specific regime.
    
    Formula: Pearson correlation r = cov(signal, labels) / (std(signal) * std(labels))
    
    Significance test: Two-tailed t-test with H0: r = 0
        t = r * sqrt(n-2) / sqrt(1-r²)
        
    Args:
        signal: Signal values (already filtered to regime)
        labels: Label values (already filtered to regime)
        signal_name: Name of the signal for reporting
        regime: Regime value for reporting
    
    Returns:
        SignalRegimeCorrelation with correlation, p-value, and significance
    
    Note:
        Returns correlation=0.0 if insufficient valid samples or constant arrays.
    """
    # Filter out NaN/Inf
    valid_mask = np.isfinite(signal) & np.isfinite(labels)
    signal_clean = signal[valid_mask]
    labels_clean = labels[valid_mask]
    
    n_samples = len(signal_clean)
    
    # Need at least 3 samples for meaningful correlation
    if n_samples < 3:
        return SignalRegimeCorrelation(
            signal_name=signal_name,
            regime=regime,
            regime_name=REGIME_NAMES.get(regime, f"UNKNOWN_{regime}"),
            n_samples=n_samples,
            correlation=0.0,
            p_value=1.0,
            is_significant=False,
            expected_sign=EXPECTED_SIGNS.get(signal_name, '?'),
            sign_consistent=False,
        )
    
    # Check for constant arrays
    if np.std(signal_clean) < 1e-10 or np.std(labels_clean) < 1e-10:
        return SignalRegimeCorrelation(
            signal_name=signal_name,
            regime=regime,
            regime_name=REGIME_NAMES.get(regime, f"UNKNOWN_{regime}"),
            n_samples=n_samples,
            correlation=0.0,
            p_value=1.0,
            is_significant=False,
            expected_sign=EXPECTED_SIGNS.get(signal_name, '?'),
            sign_consistent=False,
        )
    
    # Compute Pearson correlation
    try:
        corr, p_value = stats.pearsonr(signal_clean, labels_clean)
    except Exception:
        corr, p_value = 0.0, 1.0
    
    # Handle NaN correlation
    if not np.isfinite(corr):
        corr = 0.0
        p_value = 1.0
    
    # Check sign consistency
    expected_sign = EXPECTED_SIGNS.get(signal_name, '?')
    if expected_sign == '+':
        sign_consistent = corr > 0
    elif expected_sign == '-':
        sign_consistent = corr < 0
    else:
        sign_consistent = True  # Unknown expected sign, always consistent
    
    return SignalRegimeCorrelation(
        signal_name=signal_name,
        regime=regime,
        regime_name=REGIME_NAMES.get(regime, f"UNKNOWN_{regime}"),
        n_samples=n_samples,
        correlation=float(corr),
        p_value=float(p_value),
        is_significant=p_value < 0.01,
        expected_sign=expected_sign,
        sign_consistent=sign_consistent,
    )


def compute_all_regime_correlations(
    features: np.ndarray,
    labels: np.ndarray,
    signal_indices: Optional[Dict[str, int]] = None,
) -> List[SignalRegimeCorrelation]:
    """
    Compute signal-label correlations for all signals across all regimes.
    
    Args:
        features: (N, D) feature array with features aligned to labels
        labels: (N,) label array
        signal_indices: Dict mapping signal names to feature indices
    
    Returns:
        List of SignalRegimeCorrelation for each (signal, regime) pair
    """
    if signal_indices is None:
        signal_indices = CORE_SIGNAL_INDICES
    
    if len(features) != len(labels):
        raise ValueError(
            f"Length mismatch: features={len(features)}, labels={len(labels)}"
        )
    
    # Extract time_regime
    time_regime_idx = FeatureIndex.TIME_REGIME
    time_regimes = features[:, time_regime_idx]
    
    results = []
    unique_regimes = sorted(np.unique(time_regimes[np.isfinite(time_regimes)]).astype(int))
    
    for signal_name, signal_idx in signal_indices.items():
        signal = features[:, signal_idx]
        
        for regime in unique_regimes:
            if regime not in REGIME_NAMES:
                continue
                
            mask = time_regimes == regime
            
            corr_result = compute_signal_regime_correlation(
                signal=signal[mask],
                labels=labels[mask],
                signal_name=signal_name,
                regime=regime,
            )
            results.append(corr_result)
    
    return results


def compute_signal_seasonality(
    regime_correlations: List[SignalRegimeCorrelation],
    regime_dependence_threshold: float = 0.01,
) -> List[SignalSeasonality]:
    """
    Analyze seasonality patterns for each signal.
    
    A signal is considered "regime-dependent" if the difference between
    max and min correlation across regimes exceeds the threshold.
    
    Args:
        regime_correlations: All signal-regime correlations
        regime_dependence_threshold: Threshold for correlation_range to be
            considered "regime-dependent"
    
    Returns:
        List of SignalSeasonality, one per signal
    """
    # Group by signal name
    by_signal: Dict[str, List[SignalRegimeCorrelation]] = {}
    for rc in regime_correlations:
        if rc.signal_name not in by_signal:
            by_signal[rc.signal_name] = []
        by_signal[rc.signal_name].append(rc)
    
    results = []
    for signal_name, corrs in by_signal.items():
        # Build regime -> correlation mapping
        regime_corrs = {rc.regime: rc.correlation for rc in corrs}
        
        if len(regime_corrs) == 0:
            continue
        
        # Compute statistics
        abs_corrs = {r: abs(c) for r, c in regime_corrs.items()}
        
        max_regime = max(abs_corrs, key=abs_corrs.get)
        min_regime = min(abs_corrs, key=abs_corrs.get)
        
        max_abs = abs_corrs[max_regime]
        min_abs = abs_corrs[min_regime]
        
        correlation_range = max_abs - min_abs
        
        # Impact ratio: how much more predictive is best regime vs worst?
        if min_abs > 1e-10:
            regime_impact_ratio = max_abs / min_abs
        else:
            regime_impact_ratio = float('inf') if max_abs > 1e-10 else 1.0
        
        results.append(SignalSeasonality(
            signal_name=signal_name,
            regime_correlations=regime_corrs,
            correlation_range=correlation_range,
            most_predictive_regime=max_regime,
            least_predictive_regime=min_regime,
            is_regime_dependent=correlation_range > regime_dependence_threshold,
            regime_impact_ratio=regime_impact_ratio,
        ))
    
    return results


def compute_regime_importance(
    regime_correlations: List[SignalRegimeCorrelation],
) -> Dict[int, float]:
    """
    Compute overall importance of each regime for prediction.
    
    Importance = average |correlation| across all signals.
    
    Args:
        regime_correlations: All signal-regime correlations
    
    Returns:
        Dict mapping regime -> average absolute correlation
    """
    by_regime: Dict[int, List[float]] = {}
    
    for rc in regime_correlations:
        if rc.regime not in by_regime:
            by_regime[rc.regime] = []
        by_regime[rc.regime].append(abs(rc.correlation))
    
    return {
        regime: float(np.mean(corrs)) if corrs else 0.0
        for regime, corrs in by_regime.items()
    }


def generate_recommendations(
    regime_stats: List[RegimeStats],
    signal_seasonality: List[SignalSeasonality],
    regime_importance: Dict[int, float],
) -> List[str]:
    """
    Generate data-driven recommendations based on analysis.
    
    Args:
        regime_stats: Statistics per regime
        signal_seasonality: Seasonality analysis per signal
        regime_importance: Importance ranking per regime
    
    Returns:
        List of actionable recommendations
    """
    recommendations = []
    
    # 1. Regime importance ranking
    sorted_regimes = sorted(regime_importance.items(), key=lambda x: x[1], reverse=True)
    if sorted_regimes:
        best_regime = sorted_regimes[0][0]
        worst_regime = sorted_regimes[-1][0]
        best_name = REGIME_NAMES.get(best_regime, f"UNKNOWN_{best_regime}")
        worst_name = REGIME_NAMES.get(worst_regime, f"UNKNOWN_{worst_regime}")
        
        ratio = regime_importance[best_regime] / (regime_importance[worst_regime] + 1e-10)
        
        if ratio > 2.0:
            recommendations.append(
                f"REGIME MATTERS: {best_name} has {ratio:.1f}× higher signal correlation than {worst_name}. "
                f"Consider regime-specific models or weighting."
            )
        else:
            recommendations.append(
                f"REGIME STABLE: Signal correlations are similar across regimes (ratio {ratio:.1f}×). "
                f"A single model may suffice."
            )
    
    # 2. Regime-dependent signals
    regime_dependent = [s for s in signal_seasonality if s.is_regime_dependent]
    if regime_dependent:
        names = [s.signal_name for s in regime_dependent[:3]]
        recommendations.append(
            f"REGIME-DEPENDENT SIGNALS: {', '.join(names)} show significant variation across regimes. "
            f"Consider regime-specific feature weighting."
        )
    
    # 3. Most predictive regime
    if regime_stats:
        # Check if any regime has much higher variance
        variances = {rs.regime: rs.label_std for rs in regime_stats}
        if variances:
            max_var_regime = max(variances, key=variances.get)
            min_var_regime = min(variances, key=variances.get)
            var_ratio = variances[max_var_regime] / (variances[min_var_regime] + 1e-10)
            
            if var_ratio > 1.5:
                max_name = REGIME_NAMES.get(max_var_regime, f"UNKNOWN_{max_var_regime}")
                recommendations.append(
                    f"VOLATILITY VARIES: {max_name} has {var_ratio:.1f}× higher label variance. "
                    f"This regime offers more movement but also more noise."
                )
    
    # 4. CLOSED regime filtering
    closed_stats = [rs for rs in regime_stats if rs.regime == 4]
    if closed_stats and closed_stats[0].n_samples > 0:
        recommendations.append(
            f"FILTER RECOMMENDATION: Found {closed_stats[0].n_samples} samples in CLOSED regime (4). "
            f"Consider filtering these for training (low liquidity, unreliable signals)."
        )
    
    return recommendations


def run_intraday_seasonality_analysis(
    features: np.ndarray,
    labels: np.ndarray,
    signal_indices: Optional[Dict[str, int]] = None,
    regime_dependence_threshold: float = 0.01,
) -> IntradaySeasonalitySummary:
    """
    Run complete intraday seasonality analysis.
    
    This analysis answers:
    1. Do signal-label correlations differ by time of day?
    2. Which regimes are most predictive?
    3. Should models treat different regimes differently?
    
    Args:
        features: (N, D) feature array, aligned with labels
        labels: (N,) label array
        signal_indices: Optional dict of signal names to indices
        regime_dependence_threshold: Threshold for declaring regime-dependence
    
    Returns:
        IntradaySeasonalitySummary with all results
    
    Reference:
        Cont, R., Kukanov, A., & Stoikov, S. (2014). "The price impact of order book events."
        Journal of Financial Econometrics, 12(1), 47-88.
        
        §3.3: "Price impact is five times higher at the market open compared to the market close."
    """
    if signal_indices is None:
        signal_indices = CORE_SIGNAL_INDICES
    
    # Extract time_regime
    time_regime_idx = FeatureIndex.TIME_REGIME
    time_regimes = features[:, time_regime_idx]
    
    # 1. Compute regime statistics
    regime_stats = compute_regime_stats(labels, time_regimes)
    
    # 2. Compute all signal-regime correlations
    signal_regime_correlations = compute_all_regime_correlations(
        features, labels, signal_indices
    )
    
    # 3. Compute signal seasonality
    signal_seasonality = compute_signal_seasonality(
        signal_regime_correlations, regime_dependence_threshold
    )
    
    # 4. Compute regime importance
    regime_importance = compute_regime_importance(signal_regime_correlations)
    
    # 5. Generate recommendations
    recommendations = generate_recommendations(
        regime_stats, signal_seasonality, regime_importance
    )
    
    return IntradaySeasonalitySummary(
        regime_stats=regime_stats,
        signal_regime_correlations=signal_regime_correlations,
        signal_seasonality=signal_seasonality,
        overall_regime_importance=regime_importance,
        recommendations=recommendations,
    )


def to_dict(summary: IntradaySeasonalitySummary) -> Dict[str, Any]:
    """
    Convert IntradaySeasonalitySummary to JSON-serializable dict.
    
    Note: Explicitly converts numpy types to Python natives for JSON serialization.
    """
    return {
        'regime_stats': [
            {
                'regime': int(rs.regime),
                'regime_name': rs.regime_name,
                'n_samples': int(rs.n_samples),
                'label_mean': float(rs.label_mean),
                'label_std': float(rs.label_std),
                'label_distribution': {
                    str(k): float(v) for k, v in rs.label_distribution.items()
                },
            }
            for rs in summary.regime_stats
        ],
        'signal_regime_correlations': [
            {
                'signal_name': src.signal_name,
                'regime': int(src.regime),
                'regime_name': src.regime_name,
                'n_samples': int(src.n_samples),
                'correlation': float(src.correlation),
                'p_value': float(src.p_value),
                'is_significant': bool(src.is_significant),
                'expected_sign': src.expected_sign,
                'sign_consistent': bool(src.sign_consistent),
            }
            for src in summary.signal_regime_correlations
        ],
        'signal_seasonality': [
            {
                'signal_name': ss.signal_name,
                'regime_correlations': {
                    str(k): float(v) for k, v in ss.regime_correlations.items()
                },
                'correlation_range': float(ss.correlation_range),
                'most_predictive_regime': int(ss.most_predictive_regime),
                'most_predictive_regime_name': REGIME_NAMES.get(
                    ss.most_predictive_regime, 'UNKNOWN'
                ),
                'least_predictive_regime': int(ss.least_predictive_regime),
                'least_predictive_regime_name': REGIME_NAMES.get(
                    ss.least_predictive_regime, 'UNKNOWN'
                ),
                'is_regime_dependent': bool(ss.is_regime_dependent),
                'regime_impact_ratio': float(ss.regime_impact_ratio) if np.isfinite(ss.regime_impact_ratio) else None,
            }
            for ss in summary.signal_seasonality
        ],
        'overall_regime_importance': {
            str(k): float(v) for k, v in summary.overall_regime_importance.items()
        },
        'regime_importance_ranking': [
            {
                'regime': int(r),
                'regime_name': REGIME_NAMES.get(r, 'UNKNOWN'),
                'avg_abs_correlation': float(c),
            }
            for r, c in sorted(
                summary.overall_regime_importance.items(),
                key=lambda x: x[1],
                reverse=True,
            )
        ],
        'recommendations': summary.recommendations,
    }

