#!/usr/bin/env python3
"""
Run comprehensive signal analysis (Phase 2A).

This script combines:
- Signal distribution analysis (02_signal_distributions)
- Signal correlation analysis (03_signal_correlations)  
- Signal predictive power analysis (04_signal_predictive_power)
- Stationarity tests (ADF, rolling stats)
- PCA and VIF multicollinearity analysis

Usage:
    python scripts/run_signal_analysis.py [--data-dir PATH] [--output-dir PATH]
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np

from lobtrainer.analysis import (
    load_split,
    align_features_with_labels,
    # Distribution
    compute_distribution_stats,
    print_distribution_summary,
    # Stationarity
    compute_all_stationarity_tests,
    compute_all_rolling_stats,
    print_stationarity_summary,
    # Correlation
    compute_signal_correlation_matrix,
    find_redundant_pairs,
    print_correlation_summary,
    compute_pca_analysis,
    compute_vif,
    cluster_signals,
    print_advanced_correlation_summary,
    # Predictive power
    compute_all_signal_metrics,
    print_predictive_summary,
    CORE_SIGNAL_INDICES,
)


def main():
    parser = argparse.ArgumentParser(description='Run comprehensive signal analysis')
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path(__file__).parent.parent.parent / 'data' / 'exports' / 'nvda_98feat',
        help='Path to dataset directory',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(__file__).parent.parent / 'docs',
        help='Output directory for results',
    )
    parser.add_argument(
        '--skip-distributions',
        action='store_true',
        help='Skip distribution analysis',
    )
    parser.add_argument(
        '--skip-correlations',
        action='store_true',
        help='Skip correlation analysis',
    )
    parser.add_argument(
        '--skip-predictive',
        action='store_true',
        help='Skip predictive power analysis',
    )
    parser.add_argument(
        '--skip-stationarity',
        action='store_true',
        help='Skip stationarity analysis',
    )
    parser.add_argument(
        '--skip-pca',
        action='store_true',
        help='Skip PCA/VIF analysis',
    )
    parser.add_argument(
        '--symbol',
        type=str,
        default='NVDA',
        help='Symbol name for labeling',
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("PHASE 2A: COMPREHENSIVE SIGNAL ANALYSIS")
    print("=" * 80)
    print(f"\nSymbol: {args.symbol}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Load training data
    print("\n[1/7] Loading training data...")
    train_data = load_split(args.data_dir, 'train')
    features = train_data['features']
    labels = train_data['labels']
    
    print(f"  Features: {features.shape}")
    print(f"  Labels: {labels.shape}")
    print(f"  Days: {train_data['n_days']}")
    
    # Align features with labels
    print("\n[2/7] Aligning features with labels...")
    aligned_features = align_features_with_labels(features, len(labels))
    print(f"  Aligned features: {aligned_features.shape}")
    
    results = {
        'symbol': args.symbol,
        'data_dir': str(args.data_dir),
        'n_samples': int(features.shape[0]),
        'n_labels': len(labels),
        'n_days': train_data['n_days'],
    }
    
    # 1. Distribution Analysis
    if not args.skip_distributions:
        print("\n[3/7] Computing distribution statistics...")
        df_dist = compute_distribution_stats(features)
        print_distribution_summary(df_dist)
        
        # Save results
        df_dist.to_csv(args.output_dir / 'signal_distribution_stats.csv', index=False)
        results['distributions'] = df_dist.to_dict('records')
        print(f"\n  Saved: {args.output_dir / 'signal_distribution_stats.csv'}")
    else:
        print("\n[3/7] Skipping distribution analysis")
    
    # 2. Stationarity Analysis
    if not args.skip_stationarity:
        print("\n[4/7] Computing stationarity tests...")
        stationarity_results = compute_all_stationarity_tests(features)
        rolling_results = compute_all_rolling_stats(features)
        print_stationarity_summary(stationarity_results, rolling_results)
        
        results['stationarity'] = [
            {
                'signal': s.signal_name,
                'adf_statistic': s.adf_statistic,
                'p_value': s.p_value,
                'is_stationary': s.is_stationary,
            }
            for s in stationarity_results
        ]
        results['rolling_stability'] = [
            {
                'signal': r.signal_name,
                'mean_drift': r.mean_drift,
                'is_mean_stable': r.is_mean_stable,
                'is_std_stable': r.is_std_stable,
            }
            for r in rolling_results
        ]
    else:
        print("\n[4/7] Skipping stationarity analysis")
    
    # 3. Correlation Analysis
    if not args.skip_correlations:
        print("\n[5/7] Computing signal correlations...")
        corr_matrix, signal_names = compute_signal_correlation_matrix(aligned_features)
        redundant_pairs = find_redundant_pairs(corr_matrix, signal_names)
        print_correlation_summary(corr_matrix, signal_names, redundant_pairs)
        
        # Save results
        results['correlation_matrix'] = corr_matrix.tolist()
        results['signal_names'] = signal_names
        results['redundant_pairs'] = redundant_pairs
        print(f"\n  Correlation analysis complete")
    else:
        print("\n[5/7] Skipping correlation analysis")
        corr_matrix, signal_names = None, None
    
    # 4. PCA and VIF Analysis
    if not args.skip_pca:
        print("\n[6/7] Computing PCA and VIF...")
        try:
            pca_result = compute_pca_analysis(aligned_features)
            vif_results = compute_vif(aligned_features)
            
            if corr_matrix is not None and signal_names is not None:
                clusters = cluster_signals(corr_matrix, signal_names, CORE_SIGNAL_INDICES)
            else:
                clusters = None
            
            print_advanced_correlation_summary(pca_result, vif_results, clusters)
            
            results['pca'] = {
                'n_components_90': pca_result.n_components_90,
                'n_components_95': pca_result.n_components_95,
                'explained_variance': pca_result.explained_variance_ratio[:5],
                'dominant_signals': pca_result.dominant_signal_per_component[:5],
            }
            results['vif'] = [
                {
                    'signal': v.signal_name,
                    'vif': v.vif,
                    'is_problematic': v.is_problematic,
                    'is_severe': v.is_severe,
                }
                for v in vif_results
            ]
        except Exception as e:
            print(f"  ⚠️ PCA/VIF analysis failed: {e}")
    else:
        print("\n[6/7] Skipping PCA/VIF analysis")
    
    # 5. Predictive Power Analysis
    if not args.skip_predictive:
        print("\n[7/7] Computing predictive power metrics...")
        df_metrics = compute_all_signal_metrics(aligned_features, labels)
        print_predictive_summary(df_metrics, corr_matrix, signal_names)
        
        # Save results
        df_metrics.to_csv(args.output_dir / 'signal_predictive_metrics.csv', index=False)
        results['predictive_metrics'] = df_metrics.to_dict('records')
        print(f"\n  Saved: {args.output_dir / 'signal_predictive_metrics.csv'}")
    else:
        print("\n[7/7] Skipping predictive power analysis")
    
    # Save comprehensive results as JSON
    output_file = args.output_dir / 'signal_analysis_results.json'
    
    # Clean up results for JSON serialization
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bool):
            return obj
        elif obj is None or isinstance(obj, (str, int, float)):
            return obj
        else:
            return str(obj)
    
    with open(output_file, 'w') as f:
        json.dump(clean_for_json(results), f, indent=2)
    
    print(f"\n  Saved comprehensive results: {output_file}")
    
    print("\n" + "=" * 80)
    print("✅ PHASE 2A SIGNAL ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()

