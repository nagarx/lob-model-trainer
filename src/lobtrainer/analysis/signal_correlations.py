"""
Signal correlation and redundancy analysis.

Analyzes relationships between signals:
- Correlation matrix
- Redundant pairs (|r| > threshold)
- For future: PCA, VIF
"""

import numpy as np
from typing import Dict, List, Tuple
import pandas as pd

from .data_loading import get_signal_info, CORE_SIGNAL_INDICES


def compute_signal_correlation_matrix(
    features: np.ndarray,
    signal_indices: List[int] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute correlation matrix between signals.
    
    Args:
        features: (N, 98) feature array
        signal_indices: Which signal indices to analyze (default: core signals)
    
    Returns:
        corr_matrix: (n_signals, n_signals) correlation matrix
        signal_names: List of signal names in same order
    """
    if signal_indices is None:
        signal_indices = CORE_SIGNAL_INDICES
    
    signal_info = get_signal_info()
    signal_names = [signal_info[idx]['name'] for idx in signal_indices]
    
    # Extract signal columns
    signal_matrix = features[:, signal_indices]
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(signal_matrix.T)
    
    return corr_matrix, signal_names


def find_redundant_pairs(
    corr_matrix: np.ndarray,
    signal_names: List[str],
    threshold: float = 0.5,
) -> List[Dict]:
    """
    Find pairs of signals with correlation above threshold.
    
    Args:
        corr_matrix: Correlation matrix
        signal_names: Names corresponding to matrix rows/cols
        threshold: Correlation threshold (absolute value)
    
    Returns:
        List of dicts with {signal_1, signal_2, correlation}
    """
    n = len(signal_names)
    pairs = []
    
    for i in range(n):
        for j in range(i + 1, n):
            r = corr_matrix[i, j]
            if abs(r) > threshold:
                pairs.append({
                    'signal_1': signal_names[i],
                    'signal_2': signal_names[j],
                    'correlation': float(r),
                })
    
    # Sort by absolute correlation (descending)
    pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
    
    return pairs


def print_correlation_summary(
    corr_matrix: np.ndarray,
    signal_names: List[str],
    redundant_pairs: List[Dict] = None,
) -> None:
    """
    Print formatted correlation analysis.
    
    Args:
        corr_matrix: Correlation matrix
        signal_names: Signal names
        redundant_pairs: Pre-computed redundant pairs (optional)
    """
    print("=" * 80)
    print("SIGNAL CORRELATION ANALYSIS")
    print("=" * 80)
    
    # Print correlation matrix
    print("\nCorrelation Matrix:")
    print("-" * 80)
    
    # Header
    print(f"{'':>20s}", end='')
    for name in signal_names:
        print(f"{name[:8]:>9s}", end='')
    print()
    
    # Rows
    for i, name in enumerate(signal_names):
        print(f"{name:>20s}", end='')
        for j in range(len(signal_names)):
            if j >= i:
                r = corr_matrix[i, j]
                print(f"{r:>+9.2f}", end='')
            else:
                print(f"{'':>9s}", end='')
        print()
    
    # Redundant pairs
    if redundant_pairs is None:
        redundant_pairs = find_redundant_pairs(corr_matrix, signal_names)
    
    print("\n" + "=" * 80)
    print("REDUNDANT SIGNAL PAIRS (|r| > 0.5)")
    print("=" * 80)
    
    if redundant_pairs:
        for pair in redundant_pairs:
            print(f"  • {pair['signal_1']} ↔ {pair['signal_2']}: r = {pair['correlation']:+.3f}")
    else:
        print("  No highly correlated pairs found")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    if redundant_pairs:
        print("\n  To reduce multicollinearity, consider removing:")
        seen = set()
        for pair in redundant_pairs:
            # Suggest removing the signal that appears in more pairs
            if pair['signal_2'] not in seen:
                print(f"    - {pair['signal_2']} (redundant with {pair['signal_1']})")
                seen.add(pair['signal_2'])
    else:
        print("\n  All signals provide independent information")
    
    print("\n" + "=" * 80)

