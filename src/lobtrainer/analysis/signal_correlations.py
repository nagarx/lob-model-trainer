"""
Signal correlation and redundancy analysis.

Analyzes relationships between signals:
- Correlation matrix
- Redundant pairs (|r| > threshold)
- PCA (Principal Component Analysis) for dimensionality reduction
- VIF (Variance Inflation Factor) for multicollinearity detection
- Hierarchical clustering for signal grouping
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import pandas as pd
import warnings

from .data_loading import get_signal_info, CORE_SIGNAL_INDICES


@dataclass
class PCAResult:
    """PCA analysis result."""
    n_components: int
    explained_variance_ratio: List[float]
    cumulative_variance: List[float]
    n_components_95: int  # Components needed for 95% variance
    n_components_90: int  # Components needed for 90% variance
    component_loadings: List[List[float]]  # (n_components, n_signals)
    signal_names: List[str]
    dominant_signal_per_component: List[str]


@dataclass
class VIFResult:
    """Variance Inflation Factor result for a single signal."""
    signal_name: str
    signal_index: int
    vif: float
    is_problematic: bool  # VIF > 5
    is_severe: bool  # VIF > 10


@dataclass
class SignalCluster:
    """A cluster of related signals."""
    cluster_id: int
    signals: List[str]
    signal_indices: List[int]
    mean_within_correlation: float


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


def compute_pca_analysis(
    features: np.ndarray,
    signal_indices: List[int] = None,
    n_components: Optional[int] = None,
) -> PCAResult:
    """
    Perform PCA on signals to identify orthogonal factors.
    
    Args:
        features: (N, D) feature array
        signal_indices: Which signals to include
        n_components: Number of components (default: all)
    
    Returns:
        PCAResult with variance explained and loadings
    
    Formula:
        X = USV^T
        Explained variance = λ_i / Σλ
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    if signal_indices is None:
        signal_indices = CORE_SIGNAL_INDICES
    
    signal_info = get_signal_info()
    signal_names = [signal_info[idx]['name'] for idx in signal_indices]
    
    # Extract and standardize signals
    signal_matrix = features[:, signal_indices]
    
    # Handle any remaining NaN/Inf
    valid_mask = np.isfinite(signal_matrix).all(axis=1)
    signal_matrix = signal_matrix[valid_mask]
    
    if len(signal_matrix) < 100:
        raise ValueError("Insufficient valid samples for PCA")
    
    # Standardize (PCA is sensitive to scale)
    scaler = StandardScaler()
    signal_scaled = scaler.fit_transform(signal_matrix)
    
    # Fit PCA
    if n_components is None:
        n_components = len(signal_indices)
    
    pca = PCA(n_components=n_components)
    pca.fit(signal_scaled)
    
    # Extract results
    explained_variance = pca.explained_variance_ratio_.tolist()
    cumulative_variance = np.cumsum(explained_variance).tolist()
    
    # Components needed for thresholds
    n_components_95 = int(np.searchsorted(cumulative_variance, 0.95) + 1)
    n_components_90 = int(np.searchsorted(cumulative_variance, 0.90) + 1)
    
    # Component loadings (correlations between components and original signals)
    loadings = pca.components_.tolist()
    
    # Dominant signal per component
    dominant_signals = []
    for component in pca.components_:
        max_idx = int(np.argmax(np.abs(component)))
        dominant_signals.append(signal_names[max_idx])
    
    return PCAResult(
        n_components=n_components,
        explained_variance_ratio=explained_variance,
        cumulative_variance=cumulative_variance,
        n_components_95=n_components_95,
        n_components_90=n_components_90,
        component_loadings=loadings,
        signal_names=signal_names,
        dominant_signal_per_component=dominant_signals,
    )


def compute_vif(
    features: np.ndarray,
    signal_indices: List[int] = None,
) -> List[VIFResult]:
    """
    Compute Variance Inflation Factor for each signal.
    
    VIF measures how much variance of a coefficient is inflated
    due to multicollinearity with other predictors.
    
    VIF = 1 / (1 - R²) where R² is from regressing signal on all others
    
    Interpretation:
        VIF = 1: No correlation with others
        VIF > 5: Moderate multicollinearity (concerning)
        VIF > 10: Severe multicollinearity (problematic)
    
    Args:
        features: (N, D) feature array
        signal_indices: Which signals to analyze
    
    Returns:
        List of VIFResult for each signal
    """
    if signal_indices is None:
        signal_indices = CORE_SIGNAL_INDICES
    
    signal_info = get_signal_info()
    
    # Extract signals
    signal_matrix = features[:, signal_indices]
    
    # Handle any NaN/Inf
    valid_mask = np.isfinite(signal_matrix).all(axis=1)
    signal_matrix = signal_matrix[valid_mask]
    
    if len(signal_matrix) < 100:
        raise ValueError("Insufficient valid samples for VIF computation")
    
    results = []
    n_signals = len(signal_indices)
    
    for i, idx in enumerate(signal_indices):
        info = signal_info.get(idx, {'name': f'signal_{idx}'})
        signal_name = info.get('name', f'signal_{idx}')
        
        # Regress signal i on all other signals
        y = signal_matrix[:, i]
        X = np.delete(signal_matrix, i, axis=1)
        
        # Add constant term
        X_with_const = np.column_stack([np.ones(len(y)), X])
        
        try:
            # OLS: β = (X'X)^-1 X'y
            beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
            y_pred = X_with_const @ beta
            
            # Compute R²
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            
            if ss_tot > 1e-10:
                r_squared = 1 - ss_res / ss_tot
            else:
                r_squared = 0.0
            
            # VIF = 1 / (1 - R²)
            if r_squared < 0.9999:
                vif = 1 / (1 - r_squared)
            else:
                vif = float('inf')
        except Exception:
            vif = float('nan')
        
        results.append(VIFResult(
            signal_name=signal_name,
            signal_index=idx,
            vif=float(vif) if np.isfinite(vif) else 100.0,
            is_problematic=vif > 5,
            is_severe=vif > 10,
        ))
    
    return results


def cluster_signals(
    corr_matrix: np.ndarray,
    signal_names: List[str],
    signal_indices: List[int],
    threshold: float = 0.5,
) -> List[SignalCluster]:
    """
    Cluster signals based on correlation.
    
    Uses simple agglomerative clustering based on correlation threshold.
    
    Args:
        corr_matrix: Correlation matrix
        signal_names: Signal names
        signal_indices: Signal indices
        threshold: Correlation threshold for grouping
    
    Returns:
        List of SignalCluster
    """
    n = len(signal_names)
    assigned = [False] * n
    clusters = []
    cluster_id = 0
    
    for i in range(n):
        if assigned[i]:
            continue
        
        # Start new cluster
        cluster_signals = [signal_names[i]]
        cluster_indices = [signal_indices[i]]
        assigned[i] = True
        
        # Find all signals correlated above threshold
        for j in range(i + 1, n):
            if not assigned[j] and abs(corr_matrix[i, j]) > threshold:
                cluster_signals.append(signal_names[j])
                cluster_indices.append(signal_indices[j])
                assigned[j] = True
        
        # Compute mean within-cluster correlation
        if len(cluster_signals) > 1:
            cluster_corrs = []
            for ci in range(len(cluster_signals)):
                for cj in range(ci + 1, len(cluster_signals)):
                    orig_i = signal_names.index(cluster_signals[ci])
                    orig_j = signal_names.index(cluster_signals[cj])
                    cluster_corrs.append(abs(corr_matrix[orig_i, orig_j]))
            mean_corr = float(np.mean(cluster_corrs))
        else:
            mean_corr = 1.0
        
        clusters.append(SignalCluster(
            cluster_id=cluster_id,
            signals=cluster_signals,
            signal_indices=cluster_indices,
            mean_within_correlation=mean_corr,
        ))
        cluster_id += 1
    
    return clusters


def print_advanced_correlation_summary(
    pca_result: Optional[PCAResult] = None,
    vif_results: Optional[List[VIFResult]] = None,
    clusters: Optional[List[SignalCluster]] = None,
) -> None:
    """Print advanced correlation analysis (PCA, VIF, clustering)."""
    
    if pca_result:
        print("\n" + "=" * 80)
        print("PCA ANALYSIS (Dimensionality Reduction)")
        print("=" * 80)
        
        print("\nVariance Explained by Component:")
        for i, (var, cum_var) in enumerate(zip(
            pca_result.explained_variance_ratio[:5],
            pca_result.cumulative_variance[:5]
        )):
            print(f"  PC{i+1}: {var*100:5.1f}% (cumulative: {cum_var*100:5.1f}%)")
        
        print(f"\nComponents for 90% variance: {pca_result.n_components_90}")
        print(f"Components for 95% variance: {pca_result.n_components_95}")
        
        print("\nDominant Signal per Component:")
        for i, signal in enumerate(pca_result.dominant_signal_per_component[:5]):
            var = pca_result.explained_variance_ratio[i] * 100
            print(f"  PC{i+1}: {signal} ({var:.1f}%)")
    
    if vif_results:
        print("\n" + "=" * 80)
        print("VIF ANALYSIS (Multicollinearity)")
        print("=" * 80)
        
        print(f"\n{'Signal':<25} {'VIF':>10} {'Status':>15}")
        print("-" * 50)
        
        for v in sorted(vif_results, key=lambda x: -x.vif):
            if v.is_severe:
                status = "❌ Severe"
            elif v.is_problematic:
                status = "⚠️ Moderate"
            else:
                status = "✅ OK"
            print(f"{v.signal_name:<25} {v.vif:>10.2f} {status:>15}")
        
        severe = [v.signal_name for v in vif_results if v.is_severe]
        if severe:
            print(f"\n  ⚠️ Severely multicollinear signals (VIF > 10): {', '.join(severe)}")
            print("     Consider removing these to avoid model instability.")
    
    if clusters:
        print("\n" + "=" * 80)
        print("SIGNAL CLUSTERING")
        print("=" * 80)
        
        print(f"\nFound {len(clusters)} clusters:")
        for c in clusters:
            if len(c.signals) > 1:
                print(f"  Cluster {c.cluster_id}: {', '.join(c.signals)} (avg r={c.mean_within_correlation:.2f})")
            else:
                print(f"  Cluster {c.cluster_id}: {c.signals[0]} (independent)")
    
    print("\n" + "=" * 80)

