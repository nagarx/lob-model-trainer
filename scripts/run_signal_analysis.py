#!/usr/bin/env python3
"""
Run comprehensive signal analysis (Phase 2A).

This script combines:
- Signal distribution analysis (02_signal_distributions)
- Signal correlation analysis (03_signal_correlations)  
- Signal predictive power analysis (04_signal_predictive_power)

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
    compute_distribution_stats,
    print_distribution_summary,
    compute_signal_correlation_matrix,
    find_redundant_pairs,
    print_correlation_summary,
    compute_all_signal_metrics,
    print_predictive_summary,
)
from lobtrainer.analysis.data_loading import CORE_SIGNAL_INDICES, get_signal_info


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
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("PHASE 2A: COMPREHENSIVE SIGNAL ANALYSIS")
    print("=" * 80)
    print(f"\nData directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Load training data
    print("\n[1/5] Loading training data...")
    train_data = load_split(args.data_dir, 'train')
    features = train_data['features']
    labels = train_data['labels']
    
    print(f"  Features: {features.shape}")
    print(f"  Labels: {labels.shape}")
    print(f"  Days: {train_data['n_days']}")
    
    # Align features with labels
    print("\n[2/5] Aligning features with labels...")
    aligned_features = align_features_with_labels(features, len(labels))
    print(f"  Aligned features: {aligned_features.shape}")
    
    results = {}
    
    # 1. Distribution Analysis
    if not args.skip_distributions:
        print("\n[3/5] Computing distribution statistics...")
        df_dist = compute_distribution_stats(features)
        print_distribution_summary(df_dist)
        
        # Save results
        df_dist.to_csv(args.output_dir / 'signal_distribution_stats.csv', index=False)
        results['distributions'] = df_dist.to_dict('records')
        print(f"\n  Saved: {args.output_dir / 'signal_distribution_stats.csv'}")
    else:
        print("\n[3/5] Skipping distribution analysis")
    
    # 2. Correlation Analysis
    if not args.skip_correlations:
        print("\n[4/5] Computing signal correlations...")
        corr_matrix, signal_names = compute_signal_correlation_matrix(aligned_features)
        redundant_pairs = find_redundant_pairs(corr_matrix, signal_names)
        print_correlation_summary(corr_matrix, signal_names, redundant_pairs)
        
        # Save results
        results['correlation_matrix'] = corr_matrix.tolist()
        results['signal_names'] = signal_names
        results['redundant_pairs'] = redundant_pairs
        print(f"\n  Correlation analysis complete")
    else:
        print("\n[4/5] Skipping correlation analysis")
        corr_matrix, signal_names = None, None
    
    # 3. Predictive Power Analysis
    if not args.skip_predictive:
        print("\n[5/5] Computing predictive power metrics...")
        df_metrics = compute_all_signal_metrics(aligned_features, labels)
        print_predictive_summary(df_metrics, corr_matrix, signal_names)
        
        # Save results
        df_metrics.to_csv(args.output_dir / 'signal_predictive_metrics.csv', index=False)
        results['predictive_metrics'] = df_metrics.to_dict('records')
        print(f"\n  Saved: {args.output_dir / 'signal_predictive_metrics.csv'}")
    else:
        print("\n[5/5] Skipping predictive power analysis")
    
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

