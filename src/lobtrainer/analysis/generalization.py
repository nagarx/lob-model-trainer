"""
Generalization and robustness analysis.

Tests whether findings generalize across:
- Different trading days (day-to-day variance)
- Time periods (walk-forward validation)
- Market conditions

Critical for avoiding overfitting to specific market conditions.

Key Analyses:
- Day-to-day signal statistics variance
- Day-to-day predictive power variance
- Walk-forward stability tests
- Cross-validation on temporal splits
"""

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, accuracy_score
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
from pathlib import Path
import warnings

from ..constants import FeatureIndex, LABEL_DOWN, LABEL_STABLE, LABEL_UP
from .data_loading import get_signal_info, CORE_SIGNAL_INDICES, WINDOW_SIZE, STRIDE


@dataclass
class DayStatistics:
    """Statistics for a single trading day."""
    date: str
    n_samples: int
    n_labels: int
    label_up_pct: float
    label_down_pct: float
    label_stable_pct: float


@dataclass
class SignalDayStats:
    """Per-day statistics for a single signal."""
    signal_name: str
    signal_index: int
    dates: List[str]
    means: List[float]
    stds: List[float]
    correlations: List[float]  # Correlation with labels per day
    mean_of_means: float
    std_of_means: float
    mean_of_correlations: float
    std_of_correlations: float
    is_stable: bool  # Low variance across days
    stability_score: float  # Higher = more stable


@dataclass
class WalkForwardResult:
    """Results from a single walk-forward fold."""
    train_days: List[str]
    test_day: str
    train_n: int
    test_n: int
    signal_correlations: Dict[str, float]  # Signal name -> test correlation
    prediction_accuracy: float  # Simple threshold-based prediction
    label_distribution: Dict[str, float]  # Up/Down/Stable percentages


@dataclass
class GeneralizationSummary:
    """Complete generalization analysis results."""
    day_statistics: List[DayStatistics]
    signal_day_stats: List[SignalDayStats]
    walk_forward_results: List[WalkForwardResult]
    overall_stability_score: float
    most_stable_signals: List[str]
    least_stable_signals: List[str]
    walk_forward_avg_accuracy: float
    generalization_assessment: str
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            'day_statistics': [asdict(d) for d in self.day_statistics],
            'signal_day_stats': [asdict(s) for s in self.signal_day_stats],
            'walk_forward_results': [asdict(w) for w in self.walk_forward_results],
            'overall_stability_score': self.overall_stability_score,
            'most_stable_signals': self.most_stable_signals,
            'least_stable_signals': self.least_stable_signals,
            'walk_forward_avg_accuracy': self.walk_forward_avg_accuracy,
            'generalization_assessment': self.generalization_assessment,
            'recommendations': self.recommendations,
        }


def load_day_data(data_dir: Path, split: str = 'train') -> List[Dict]:
    """
    Load data for each day separately.
    
    Args:
        data_dir: Path to dataset root
        split: 'train', 'val', or 'test'
    
    Returns:
        List of dicts with {date, features, labels}
    """
    split_dir = Path(data_dir) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    
    days = []
    for feat_file in sorted(split_dir.glob('*_features.npy')):
        date = feat_file.stem.replace('_features', '')
        label_file = split_dir / f"{date}_labels.npy"
        
        if not label_file.exists():
            continue
        
        features = np.load(feat_file)
        labels = np.load(label_file)
        
        days.append({
            'date': date,
            'features': features,
            'labels': labels,
        })
    
    return days


def compute_day_statistics(days: List[Dict]) -> List[DayStatistics]:
    """
    Compute basic statistics for each day.
    
    Args:
        days: List of {date, features, labels} dicts
    
    Returns:
        List of DayStatistics
    """
    results = []
    
    for day in days:
        labels = day['labels']
        n_labels = len(labels)
        
        up_pct = 100 * (labels == LABEL_UP).mean()
        down_pct = 100 * (labels == LABEL_DOWN).mean()
        stable_pct = 100 * (labels == LABEL_STABLE).mean()
        
        results.append(DayStatistics(
            date=day['date'],
            n_samples=day['features'].shape[0],
            n_labels=n_labels,
            label_up_pct=float(up_pct),
            label_down_pct=float(down_pct),
            label_stable_pct=float(stable_pct),
        ))
    
    return results


def align_features_for_day(
    features: np.ndarray,
    n_labels: int,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
) -> np.ndarray:
    """Align features with labels for a single day."""
    aligned = np.zeros((n_labels, features.shape[1]))
    
    for i in range(n_labels):
        feat_idx = min(i * stride + window_size - 1, features.shape[0] - 1)
        aligned[i] = features[feat_idx]
    
    return aligned


def compute_signal_day_stats(
    days: List[Dict],
    signal_indices: List[int] = None,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
) -> List[SignalDayStats]:
    """
    Compute per-day statistics for each signal.
    
    Args:
        days: List of {date, features, labels} dicts
        signal_indices: Which signals to analyze
        window_size: Samples per sequence window
        stride: Samples between sequences
    
    Returns:
        List of SignalDayStats
    """
    if signal_indices is None:
        signal_indices = CORE_SIGNAL_INDICES
    
    signal_info = get_signal_info()
    results = []
    
    for idx in signal_indices:
        info = signal_info.get(idx, {'name': f'signal_{idx}'})
        signal_name = info.get('name', f'signal_{idx}')
        
        dates = []
        means = []
        stds = []
        correlations = []
        
        for day in days:
            date = day['date']
            features = day['features']
            labels = day['labels']
            
            # Align features with labels
            aligned = align_features_for_day(features, len(labels), window_size, stride)
            signal = aligned[:, idx]
            
            # Compute statistics
            dates.append(date)
            means.append(float(np.mean(signal)))
            stds.append(float(np.std(signal)))
            
            # Correlation with labels
            if len(labels) > 10 and np.std(signal) > 1e-10:
                corr, _ = pearsonr(signal, labels)
                correlations.append(float(corr) if np.isfinite(corr) else 0.0)
            else:
                correlations.append(0.0)
        
        # Compute aggregate statistics
        mean_of_means = float(np.mean(means))
        std_of_means = float(np.std(means))
        mean_of_correlations = float(np.mean(correlations))
        std_of_correlations = float(np.std(correlations))
        
        # Stability score: inverse of coefficient of variation
        # Higher = more stable
        if abs(mean_of_correlations) > 1e-10:
            stability_score = abs(mean_of_correlations) / (std_of_correlations + 1e-10)
        else:
            stability_score = 0.0
        
        # Is stable if correlation sign is consistent and variance is low
        signs = [1 if c > 0 else -1 if c < 0 else 0 for c in correlations]
        sign_consistency = len(set(signs)) <= 2 and 0 not in signs[:len(signs)//2 + 1]
        is_stable = stability_score > 1.0 and sign_consistency
        
        results.append(SignalDayStats(
            signal_name=signal_name,
            signal_index=idx,
            dates=dates,
            means=means,
            stds=stds,
            correlations=correlations,
            mean_of_means=mean_of_means,
            std_of_means=std_of_means,
            mean_of_correlations=mean_of_correlations,
            std_of_correlations=std_of_correlations,
            is_stable=is_stable,
            stability_score=float(stability_score),
        ))
    
    return results


def walk_forward_validation(
    days: List[Dict],
    signal_indices: List[int] = None,
    min_train_days: int = 3,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
) -> List[WalkForwardResult]:
    """
    Perform walk-forward validation.
    
    For each test day, train on all previous days and test on that day.
    
    Args:
        days: List of {date, features, labels} dicts (must be sorted by date)
        signal_indices: Which signals to analyze
        min_train_days: Minimum days required for training
        window_size: Samples per sequence window
        stride: Samples between sequences
    
    Returns:
        List of WalkForwardResult for each fold
    """
    if signal_indices is None:
        signal_indices = CORE_SIGNAL_INDICES
    
    signal_info = get_signal_info()
    results = []
    
    n_days = len(days)
    
    for test_idx in range(min_train_days, n_days):
        train_days_data = days[:test_idx]
        test_day_data = days[test_idx]
        
        # Combine training data
        train_features_list = []
        train_labels_list = []
        
        for day in train_days_data:
            aligned = align_features_for_day(day['features'], len(day['labels']), window_size, stride)
            train_features_list.append(aligned)
            train_labels_list.append(day['labels'])
        
        train_features = np.vstack(train_features_list)
        train_labels = np.concatenate(train_labels_list)
        
        # Test data
        test_aligned = align_features_for_day(
            test_day_data['features'],
            len(test_day_data['labels']),
            window_size, stride
        )
        test_labels = test_day_data['labels']
        
        # Compute signal correlations on test day
        signal_correlations = {}
        for idx in signal_indices:
            info = signal_info.get(idx, {'name': f'signal_{idx}'})
            signal_name = info.get('name', f'signal_{idx}')
            
            test_signal = test_aligned[:, idx]
            if len(test_labels) > 10 and np.std(test_signal) > 1e-10:
                corr, _ = pearsonr(test_signal, test_labels)
                signal_correlations[signal_name] = float(corr) if np.isfinite(corr) else 0.0
            else:
                signal_correlations[signal_name] = 0.0
        
        # Simple threshold-based prediction using best signal (true_ofi)
        # This is a baseline to measure generalization
        ofi_idx = FeatureIndex.TRUE_OFI
        train_ofi = train_features[:, ofi_idx]
        test_ofi = test_aligned[:, ofi_idx]
        
        # Compute threshold from training data
        # Predict Up if OFI > mean + 0.5*std, Down if OFI < mean - 0.5*std
        train_mean = train_ofi.mean()
        train_std = train_ofi.std()
        
        predictions = np.zeros(len(test_labels), dtype=int)
        predictions[test_ofi > train_mean + 0.5 * train_std] = LABEL_UP
        predictions[test_ofi < train_mean - 0.5 * train_std] = LABEL_DOWN
        
        accuracy = float(accuracy_score(test_labels, predictions))
        
        # Label distribution
        label_dist = {
            'up': float(100 * (test_labels == LABEL_UP).mean()),
            'down': float(100 * (test_labels == LABEL_DOWN).mean()),
            'stable': float(100 * (test_labels == LABEL_STABLE).mean()),
        }
        
        results.append(WalkForwardResult(
            train_days=[d['date'] for d in train_days_data],
            test_day=test_day_data['date'],
            train_n=len(train_labels),
            test_n=len(test_labels),
            signal_correlations=signal_correlations,
            prediction_accuracy=accuracy,
            label_distribution=label_dist,
        ))
    
    return results


def run_generalization_analysis(
    data_dir: Path,
    split: str = 'train',
    signal_indices: List[int] = None,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
) -> GeneralizationSummary:
    """
    Run complete generalization analysis.
    
    Args:
        data_dir: Path to dataset root
        split: Which split to analyze
        signal_indices: Which signals to analyze
        window_size: Samples per sequence window
        stride: Samples between sequences
    
    Returns:
        GeneralizationSummary with all results
    """
    if signal_indices is None:
        signal_indices = CORE_SIGNAL_INDICES
    
    # Load all days
    days = load_day_data(data_dir, split)
    
    if len(days) < 3:
        raise ValueError(f"Need at least 3 days for generalization analysis, got {len(days)}")
    
    # 1. Day statistics
    day_statistics = compute_day_statistics(days)
    
    # 2. Signal day-to-day stats
    signal_day_stats = compute_signal_day_stats(
        days, signal_indices, window_size, stride
    )
    
    # 3. Walk-forward validation
    walk_forward_results = walk_forward_validation(
        days, signal_indices, min_train_days=3, window_size=window_size, stride=stride
    )
    
    # 4. Compute summary metrics
    # Overall stability score (average across signals)
    overall_stability_score = float(np.mean([s.stability_score for s in signal_day_stats]))
    
    # Most and least stable signals
    sorted_signals = sorted(signal_day_stats, key=lambda x: x.stability_score, reverse=True)
    most_stable_signals = [s.signal_name for s in sorted_signals[:3]]
    least_stable_signals = [s.signal_name for s in sorted_signals[-3:]]
    
    # Walk-forward average accuracy
    if walk_forward_results:
        walk_forward_avg_accuracy = float(np.mean([w.prediction_accuracy for w in walk_forward_results]))
    else:
        walk_forward_avg_accuracy = 0.0
    
    # Generalization assessment
    if overall_stability_score > 2.0 and walk_forward_avg_accuracy > 0.35:
        generalization_assessment = "GOOD: Signals are stable and predictive across days"
    elif overall_stability_score > 1.0 or walk_forward_avg_accuracy > 0.35:
        generalization_assessment = "MODERATE: Some stability, but expect variance across days"
    else:
        generalization_assessment = "POOR: High day-to-day variance, findings may not generalize"
    
    # Recommendations
    recommendations = []
    
    if overall_stability_score < 1.5:
        recommendations.append("Consider day-specific normalization or model adaptation")
    
    unstable = [s.signal_name for s in signal_day_stats if not s.is_stable]
    if unstable:
        recommendations.append(f"Unstable signals (use with caution): {', '.join(unstable[:3])}")
    
    if walk_forward_avg_accuracy < 0.35:
        recommendations.append("Walk-forward accuracy is low; simple threshold prediction insufficient")
    
    sign_flippers = [s.signal_name for s in signal_day_stats 
                     if any(c > 0 for c in s.correlations) and any(c < 0 for c in s.correlations)]
    if sign_flippers:
        recommendations.append(f"Sign-flipping signals (correlation changes sign): {', '.join(sign_flippers[:3])}")
    
    if len(days) < 10:
        recommendations.append(f"Only {len(days)} days available; consider more data for robust estimates")
    
    return GeneralizationSummary(
        day_statistics=day_statistics,
        signal_day_stats=signal_day_stats,
        walk_forward_results=walk_forward_results,
        overall_stability_score=overall_stability_score,
        most_stable_signals=most_stable_signals,
        least_stable_signals=least_stable_signals,
        walk_forward_avg_accuracy=walk_forward_avg_accuracy,
        generalization_assessment=generalization_assessment,
        recommendations=recommendations,
    )


def print_generalization_analysis(summary: GeneralizationSummary) -> None:
    """Print formatted generalization analysis."""
    print("=" * 80)
    print("GENERALIZATION & ROBUSTNESS ANALYSIS")
    print("=" * 80)
    
    # 1. Day statistics overview
    print("\n1. DAY-TO-DAY OVERVIEW")
    print("-" * 60)
    print(f"{'Date':<12} {'Samples':>10} {'Labels':>8} {'Up%':>7} {'Down%':>7} {'Stable%':>8}")
    for d in summary.day_statistics:
        print(f"{d.date:<12} {d.n_samples:>10,} {d.n_labels:>8,} "
              f"{d.label_up_pct:>6.1f}% {d.label_down_pct:>6.1f}% {d.label_stable_pct:>7.1f}%")
    
    # 2. Signal stability
    print("\n2. SIGNAL STABILITY ACROSS DAYS")
    print("-" * 60)
    print(f"{'Signal':<25} {'Mean(r)':>10} {'Std(r)':>10} {'Stability':>10} {'Stable':>8}")
    for s in sorted(summary.signal_day_stats, key=lambda x: -x.stability_score):
        stable = "✅" if s.is_stable else "❌"
        print(f"{s.signal_name:<25} {s.mean_of_correlations:>+10.4f} {s.std_of_correlations:>10.4f} "
              f"{s.stability_score:>10.2f} {stable:>8}")
    
    # 3. Walk-forward results
    print("\n3. WALK-FORWARD VALIDATION")
    print("-" * 60)
    if summary.walk_forward_results:
        print(f"{'Test Day':<12} {'Train N':>10} {'Test N':>8} {'Accuracy':>10} {'OFI r':>10}")
        for w in summary.walk_forward_results:
            ofi_r = w.signal_correlations.get('true_ofi', 0.0)
            print(f"{w.test_day:<12} {w.train_n:>10,} {w.test_n:>8,} "
                  f"{w.prediction_accuracy:>9.1%} {ofi_r:>+10.4f}")
        
        print(f"\n  Average walk-forward accuracy: {summary.walk_forward_avg_accuracy:.1%}")
    else:
        print("  Insufficient days for walk-forward validation")
    
    # 4. Per-signal day-by-day correlations
    print("\n4. SIGNAL-LABEL CORRELATION BY DAY (Top 3 Signals)")
    print("-" * 60)
    
    top_signals = sorted(summary.signal_day_stats, key=lambda x: -x.stability_score)[:3]
    dates = summary.signal_day_stats[0].dates if summary.signal_day_stats else []
    
    # Header
    print(f"{'Signal':<20}", end='')
    for date in dates:
        print(f"{date[-4:]:>8}", end='')  # Last 4 chars of date
    print()
    
    # Rows
    for s in top_signals:
        print(f"{s.signal_name:<20}", end='')
        for corr in s.correlations:
            print(f"{corr:>+8.3f}", end='')
        print()
    
    # 5. Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print(f"\n  Overall stability score: {summary.overall_stability_score:.2f}")
    print(f"  Generalization assessment: {summary.generalization_assessment}")
    print(f"\n  Most stable signals: {', '.join(summary.most_stable_signals)}")
    print(f"  Least stable signals: {', '.join(summary.least_stable_signals)}")
    
    # Recommendations
    print("\n  RECOMMENDATIONS:")
    for rec in summary.recommendations:
        print(f"    • {rec}")
    
    print("\n" + "=" * 80)
    print("✅ GENERALIZATION ANALYSIS COMPLETE")
    print("=" * 80)

