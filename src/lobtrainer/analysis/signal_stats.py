"""
Signal distribution statistics.

Analyzes each signal's statistical properties:
- Mean, std, min, max
- Skewness, kurtosis
- Outlier frequency
- Normality tests
"""

import numpy as np
from scipy.stats import skew, kurtosis, normaltest
from typing import Dict, List
import pandas as pd

from .data_loading import get_signal_info, CORE_SIGNAL_INDICES


def compute_distribution_stats(
    features: np.ndarray,
    signal_indices: List[int] = None,
) -> pd.DataFrame:
    """
    Compute distribution statistics for each signal.
    
    Args:
        features: (N, 98) feature array
        signal_indices: Which signal indices to analyze (default: core signals)
    
    Returns:
        DataFrame with columns:
            - index, name, mean, std, min, max, median
            - skewness, kurtosis, pct_outliers, p_normal
    """
    if signal_indices is None:
        signal_indices = CORE_SIGNAL_INDICES
    
    signal_info = get_signal_info()
    results = []
    
    for idx in signal_indices:
        info = signal_info.get(idx, {'name': f'signal_{idx}'})
        signal = features[:, idx]
        
        # Basic stats
        mean = float(np.mean(signal))
        std = float(np.std(signal))
        min_val = float(np.min(signal))
        max_val = float(np.max(signal))
        median = float(np.median(signal))
        
        # Higher moments
        signal_skew = float(skew(signal))
        signal_kurt = float(kurtosis(signal))  # Excess kurtosis
        
        # Outlier frequency (|z| > 3)
        if std > 1e-10:
            z_scores = np.abs((signal - mean) / std)
            pct_outliers = 100.0 * float(np.mean(z_scores > 3))
        else:
            pct_outliers = 0.0
        
        # Normality test (subsample for speed)
        subsample = signal[::100][:5000]
        if len(subsample) >= 20:
            try:
                _, p_normal = normaltest(subsample)
                p_normal = float(p_normal)
            except Exception:
                p_normal = np.nan
        else:
            p_normal = np.nan
        
        results.append({
            'index': idx,
            'name': info.get('name', f'signal_{idx}'),
            'mean': mean,
            'std': std,
            'min': min_val,
            'max': max_val,
            'median': median,
            'skewness': signal_skew,
            'kurtosis': signal_kurt,
            'pct_outliers': pct_outliers,
            'p_normal': p_normal,
        })
    
    return pd.DataFrame(results)


def print_distribution_summary(df_stats: pd.DataFrame) -> None:
    """
    Print formatted distribution statistics.
    
    Args:
        df_stats: DataFrame from compute_distribution_stats
    """
    print("=" * 100)
    print("SIGNAL DISTRIBUTION STATISTICS")
    print("=" * 100)
    
    # Display key columns
    display_cols = ['name', 'mean', 'std', 'skewness', 'kurtosis', 'pct_outliers']
    df_display = df_stats[display_cols].copy()
    
    # Format for readability
    for col in ['mean', 'std', 'skewness', 'kurtosis']:
        df_display[col] = df_display[col].apply(lambda x: f"{x:+.4f}")
    df_display['pct_outliers'] = df_display['pct_outliers'].apply(lambda x: f"{x:.2f}%")
    
    print(df_display.to_string(index=False))
    
    # Key insights
    print("\n" + "=" * 100)
    print("KEY INSIGHTS")
    print("=" * 100)
    
    # Highly skewed signals
    skewed = df_stats[df_stats['skewness'].abs() > 1.0]
    if len(skewed) > 0:
        print("\nHighly Skewed Signals (|skewness| > 1.0):")
        for _, row in skewed.iterrows():
            direction = "right" if row['skewness'] > 0 else "left"
            print(f"  • {row['name']}: skewness = {row['skewness']:+.2f} ({direction}-tailed)")
    
    # Heavy-tailed signals
    heavy_tailed = df_stats[df_stats['kurtosis'] > 3.0]
    if len(heavy_tailed) > 0:
        print("\nHeavy-Tailed Signals (excess kurtosis > 3.0):")
        for _, row in heavy_tailed.iterrows():
            print(f"  • {row['name']}: kurtosis = {row['kurtosis']:.2f} (normal = 0)")
    
    # High outlier signals
    high_outliers = df_stats[df_stats['pct_outliers'] > 1.0]
    if len(high_outliers) > 0:
        print("\nHigh Outlier Frequency (> 1% outside 3σ):")
        for _, row in high_outliers.iterrows():
            print(f"  • {row['name']}: {row['pct_outliers']:.2f}% outliers")
    
    # Non-normal signals
    non_normal = df_stats[df_stats['p_normal'] < 0.01]
    if len(non_normal) > 0:
        print("\nNon-Normal Signals (p < 0.01):")
        print(f"  • All {len(non_normal)} tested signals are non-normal (expected after Z-scoring)")
    
    print("\n" + "=" * 100)

