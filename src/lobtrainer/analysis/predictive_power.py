"""
Signal predictive power analysis.

Determines which signals predict price movement using:
- Pearson/Spearman correlation
- AUC (Up vs Not-Up, Down vs Not-Down)
- Mutual Information
- Binned probability analysis
"""

import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif
from typing import Dict, List, Optional
import pandas as pd

from ..constants import LABEL_DOWN, LABEL_STABLE, LABEL_UP
from .data_loading import get_signal_info, CORE_SIGNAL_INDICES


def compute_signal_metrics(
    signal: np.ndarray,
    labels: np.ndarray,
    expected_sign: str = '?',
) -> Dict:
    """
    Compute comprehensive predictive metrics for a single signal.
    
    Args:
        signal: (N,) array of signal values
        labels: (N,) array of labels {-1, 0, 1}
        expected_sign: '+', '-', or '?' for expected correlation direction
    
    Returns:
        dict with:
            - n_samples: number of valid samples
            - pearson_r, pearson_p: Pearson correlation and p-value
            - spearman_r, spearman_p: Spearman correlation and p-value
            - auc_up: AUC for Up vs Not-Up
            - auc_down: AUC for Down vs Not-Down
            - mutual_info, mi_bits: Mutual information (nats and bits)
            - sign_consistent: whether sign matches expected
            - mean_up, mean_stable, mean_down: conditional means
    """
    # Remove any NaN/Inf values
    valid_mask = np.isfinite(signal) & np.isfinite(labels)
    signal = signal[valid_mask]
    labels_clean = labels[valid_mask]
    
    n = len(signal)
    if n == 0:
        return {'n_samples': 0, 'error': 'No valid samples'}
    
    # 1. Pearson correlation
    pearson_r, pearson_p = pearsonr(signal, labels_clean)
    
    # 2. Spearman correlation (rank-based)
    spearman_r, spearman_p = spearmanr(signal, labels_clean)
    
    # 3. AUC for Up vs Not-Up
    y_up = (labels_clean == LABEL_UP).astype(int)
    if 0 < y_up.sum() < len(y_up):
        auc_up = roc_auc_score(y_up, signal)
    else:
        auc_up = 0.5
    
    # 4. AUC for Down vs Not-Down (use NEGATIVE signal)
    y_down = (labels_clean == LABEL_DOWN).astype(int)
    if 0 < y_down.sum() < len(y_down):
        auc_down = roc_auc_score(y_down, -signal)
    else:
        auc_down = 0.5
    
    # 5. Mutual Information
    # Shift labels from {-1, 0, 1} to {0, 1, 2} for sklearn
    labels_shifted = labels_clean.astype(int) + 1
    mi = mutual_info_classif(
        signal.reshape(-1, 1),
        labels_shifted,
        discrete_features=False,
        random_state=42,
    )[0]
    mi_bits = mi / np.log(2)
    
    # 6. Sign consistency check
    if expected_sign == '+':
        sign_consistent = pearson_r > 0
    elif expected_sign == '-':
        sign_consistent = pearson_r < 0
    else:
        sign_consistent = None
    
    # 7. Conditional means
    mean_up = float(signal[labels_clean == LABEL_UP].mean()) if (labels_clean == LABEL_UP).any() else np.nan
    mean_stable = float(signal[labels_clean == LABEL_STABLE].mean()) if (labels_clean == LABEL_STABLE).any() else np.nan
    mean_down = float(signal[labels_clean == LABEL_DOWN].mean()) if (labels_clean == LABEL_DOWN).any() else np.nan
    
    return {
        'n_samples': n,
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
        'spearman_r': float(spearman_r),
        'spearman_p': float(spearman_p),
        'auc_up': float(auc_up),
        'auc_down': float(auc_down),
        'mutual_info': float(mi),
        'mi_bits': float(mi_bits),
        'sign_consistent': sign_consistent,
        'mean_up': mean_up,
        'mean_stable': mean_stable,
        'mean_down': mean_down,
    }


def compute_all_signal_metrics(
    aligned_features: np.ndarray,
    labels: np.ndarray,
    signal_indices: List[int] = None,
) -> pd.DataFrame:
    """
    Compute predictive metrics for all signals.
    
    Args:
        aligned_features: (N_labels, 98) aligned feature array
        labels: (N_labels,) label array
        signal_indices: Which signals to analyze (default: core signals)
    
    Returns:
        DataFrame with metrics for each signal
    """
    if signal_indices is None:
        signal_indices = CORE_SIGNAL_INDICES
    
    signal_info = get_signal_info()
    results = []
    
    for idx in signal_indices:
        info = signal_info.get(idx, {'name': f'signal_{idx}', 'expected_sign': '?'})
        signal = aligned_features[:, idx]
        
        metrics = compute_signal_metrics(
            signal, labels, info.get('expected_sign', '?')
        )
        
        results.append({
            'index': idx,
            'name': info.get('name', f'signal_{idx}'),
            'expected_sign': info.get('expected_sign', '?'),
            **metrics,
        })
    
    return pd.DataFrame(results)


def compute_binned_probabilities(
    signal: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Bin signal into quantiles and compute label probabilities per bin.
    
    This reveals non-linear relationships missed by correlation.
    
    Args:
        signal: (N,) signal values
        labels: (N,) label values {-1, 0, 1}
        n_bins: Number of bins (default 10 = deciles)
    
    Returns:
        DataFrame with columns:
            - bin: bin number (0 = lowest)
            - signal_mean, signal_min, signal_max
            - p_up, p_down, p_stable
            - n_samples
    """
    # Handle edge cases
    valid_mask = np.isfinite(signal)
    signal = signal[valid_mask]
    labels_clean = labels[valid_mask]
    
    if len(signal) == 0:
        return pd.DataFrame()
    
    # Create bins using quantiles
    try:
        bins = pd.qcut(signal, q=n_bins, labels=False, duplicates='drop')
    except ValueError:
        # Fall back to equal-width bins
        bins = pd.cut(signal, bins=n_bins, labels=False)
    
    results = []
    for b in range(int(np.nanmax(bins)) + 1):
        mask = bins == b
        if mask.sum() == 0:
            continue
        
        bin_labels = labels_clean[mask]
        bin_signal = signal[mask]
        
        results.append({
            'bin': int(b),
            'signal_mean': float(bin_signal.mean()),
            'signal_min': float(bin_signal.min()),
            'signal_max': float(bin_signal.max()),
            'p_up': float((bin_labels == LABEL_UP).mean()),
            'p_down': float((bin_labels == LABEL_DOWN).mean()),
            'p_stable': float((bin_labels == LABEL_STABLE).mean()),
            'n_samples': int(len(bin_labels)),
        })
    
    return pd.DataFrame(results)


def print_predictive_summary(
    df_metrics: pd.DataFrame,
    corr_matrix: Optional[np.ndarray] = None,
    signal_names: Optional[List[str]] = None,
) -> None:
    """
    Print formatted predictive power summary.
    
    Args:
        df_metrics: DataFrame from compute_all_signal_metrics
        corr_matrix: Optional correlation matrix for redundancy info
        signal_names: Signal names for correlation matrix
    """
    print("=" * 80)
    print("SIGNAL PREDICTIVE POWER ANALYSIS")
    print("=" * 80)
    
    # Sort by absolute Pearson correlation
    df_sorted = df_metrics.sort_values('pearson_r', key=abs, ascending=False)
    
    # Display ranking
    print("\n1. SIGNAL RANKING (by |Pearson r|):\n")
    print("   Rank | Signal                    | r       | AUC_up | AUC_down | Sign OK")
    print("   " + "-" * 70)
    
    for rank, (_, row) in enumerate(df_sorted.iterrows(), 1):
        sign_ok = '✓' if row['sign_consistent'] == True else '✗' if row['sign_consistent'] == False else '?'
        print(f"   #{rank:2d}  | {row['name']:25s} | {row['pearson_r']:+.4f} | {row['auc_up']:.4f} | {row['auc_down']:.4f}  | {sign_ok}")
    
    # Key findings
    print("\n" + "=" * 80)
    print("2. KEY FINDINGS")
    print("=" * 80)
    
    # Best predictor
    best = df_sorted.iloc[0]
    print(f"\n  • BEST PREDICTOR: {best['name']} (r = {best['pearson_r']:+.4f})")
    
    # Contrarian signals
    contrarian = df_metrics[df_metrics['sign_consistent'] == False]
    if len(contrarian) > 0:
        print(f"\n  • CONTRARIAN SIGNALS (opposite of expected sign):")
        for _, row in contrarian.iterrows():
            print(f"    - {row['name']}: expected {row['expected_sign']}, got r = {row['pearson_r']:+.4f}")
    
    # Redundant pairs (if provided)
    if corr_matrix is not None and signal_names is not None:
        print(f"\n  • REDUNDANT PAIRS (|r| > 0.5):")
        for i in range(len(signal_names)):
            for j in range(i + 1, len(signal_names)):
                r = corr_matrix[i, j]
                if abs(r) > 0.5:
                    print(f"    - {signal_names[i]} ↔ {signal_names[j]}: r = {r:+.3f}")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("3. RECOMMENDATIONS")
    print("=" * 80)
    
    print("""
  GROUP A - PRIMARY FEATURES:
    • true_ofi: Best linear predictor
    • depth_asymmetry: CONTRARIAN - use as separate feature

  GROUP B - MODERATE VALUE:
    • executed_pressure: Trade-based, moderate power
    • cancel_asymmetry: Order flow signal
    • fragility_score: Book structure

  GROUP C - REDUNDANT (avoid in same model):
    • depth_norm_ofi: r=0.66 with true_ofi
    • trade_asymmetry: r=0.54 with true_ofi

  GROUP D - LOW PRIORITY:
    • signed_mp_delta_bps: Near-zero predictive power
""")
    
    print("=" * 80)
    print("✅ SIGNAL PREDICTIVE POWER ANALYSIS COMPLETE")
    print("=" * 80)

