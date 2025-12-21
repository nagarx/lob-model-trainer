#!/usr/bin/env python3
"""
Phase 0: Empirical Validation of Alignment Bugs

This script quantifies the ACTUAL impact of the identified bugs:
1. Day-boundary alignment drift
2. Safety gate rates (book_valid, mbo_ready, invalidity_delta)
3. time_regime actual values
4. Correlation difference between correct and incorrect alignment

Run this BEFORE any fixes to document the magnitude of issues.

Usage:
    python scripts/validate_alignment_bugs.py \
        --data-dir ../data/exports/nvda_98feat_full \
        --symbol NVDA
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lobtrainer.constants import FeatureIndex

# Constants from Rust pipeline (MUST match exactly)
WINDOW_SIZE = 100
STRIDE = 10

# Feature indices for safety gates
BOOK_VALID_IDX = 92
TIME_REGIME_IDX = 93
MBO_READY_IDX = 94
INVALIDITY_DELTA_IDX = 96

# Signal indices for correlation test
TRUE_OFI_IDX = 84
DEPTH_NORM_OFI_IDX = 85


def load_day_data(feat_file: Path) -> Tuple[np.ndarray, np.ndarray, str]:
    """Load a single day's features and labels."""
    date = feat_file.stem.replace('_features', '')
    label_file = feat_file.parent / f"{date}_labels.npy"
    
    features = np.load(feat_file)
    labels = np.load(label_file)
    
    return features, labels, date


def align_correct(features: np.ndarray, n_labels: int) -> np.ndarray:
    """
    CORRECT alignment: For label[i], use feature at index (i * stride + window_size - 1).
    
    Formula: feat_idx = i * stride + window_size - 1
    This is the LAST feature in the window for label i.
    """
    aligned = np.zeros((n_labels, features.shape[1]), dtype=features.dtype)
    for i in range(n_labels):
        feat_idx = min(i * STRIDE + WINDOW_SIZE - 1, features.shape[0] - 1)
        aligned[i] = features[feat_idx]
    return aligned


def compute_alignment_drift(
    data_dir: Path,
    split: str = 'train',
) -> Dict[str, Any]:
    """
    Measure the drift between GLOBAL alignment formula and PER-DAY alignment.
    
    The bug: When days are concatenated, using global label index i gives wrong feature.
    
    Returns:
        Dict with drift statistics per day
    """
    split_dir = data_dir / split
    feature_files = sorted(split_dir.glob('*_features.npy'))
    
    results = {
        'days': [],
        'cumulative_offset': [],
        'expected_vs_computed': [],
        'error_in_labels': [],
    }
    
    # Track cumulative samples and labels
    cumulative_samples = 0
    cumulative_labels = 0
    
    for day_idx, feat_file in enumerate(feature_files):
        features, labels, date = load_day_data(feat_file)
        n_samples = features.shape[0]
        n_labels = len(labels)
        
        # Day 1 has no drift
        if day_idx == 0:
            cumulative_samples = n_samples
            cumulative_labels = n_labels
            results['days'].append(date)
            results['cumulative_offset'].append(0)
            results['expected_vs_computed'].append((0, 0))
            results['error_in_labels'].append(0)
            continue
        
        # For first label of this day:
        # - Expected feature index (per-day): WINDOW_SIZE - 1 = 99
        # - Global label index: cumulative_labels
        # - Computed feature index (global formula): cumulative_labels * STRIDE + WINDOW_SIZE - 1
        # - Expected feature index (global): cumulative_samples + WINDOW_SIZE - 1
        
        global_label_idx = cumulative_labels
        computed_feat_idx = global_label_idx * STRIDE + WINDOW_SIZE - 1
        expected_feat_idx = cumulative_samples + WINDOW_SIZE - 1
        
        offset = computed_feat_idx - expected_feat_idx
        error_in_labels = abs(offset) / STRIDE
        
        results['days'].append(date)
        results['cumulative_offset'].append(int(offset))
        results['expected_vs_computed'].append((int(expected_feat_idx), int(computed_feat_idx)))
        results['error_in_labels'].append(float(error_in_labels))
        
        # Update cumulative counts
        cumulative_samples += n_samples
        cumulative_labels += n_labels
    
    # Summary statistics
    offsets = [x for x in results['cumulative_offset'] if x != 0]
    if offsets:
        results['summary'] = {
            'max_offset_samples': max(offsets),
            'max_offset_labels': max(offsets) / STRIDE,
            'max_offset_day': results['days'][results['cumulative_offset'].index(max(offsets))],
            'mean_offset': np.mean(offsets),
            'drift_per_day': offsets[-1] / len(offsets) if offsets else 0,
        }
    else:
        results['summary'] = {'max_offset_samples': 0}
    
    return results


def compute_safety_gate_stats(
    data_dir: Path,
    split: str = 'train',
) -> Dict[str, Any]:
    """
    Measure rates of invalid samples using CORRECT (per-day) alignment.
    
    Checks:
    - book_valid == 0 (invalid book state)
    - mbo_ready == 0 (warmup not complete)
    - invalidity_delta > 0 (feed problems)
    """
    split_dir = data_dir / split
    feature_files = sorted(split_dir.glob('*_features.npy'))
    
    total_labels = 0
    book_invalid_count = 0
    mbo_warmup_count = 0
    feed_problem_count = 0
    any_invalid_count = 0
    
    per_day_stats = []
    
    for feat_file in feature_files:
        features, labels, date = load_day_data(feat_file)
        n_labels = len(labels)
        
        # CORRECT alignment (per-day)
        aligned = align_correct(features, n_labels)
        
        # Check each safety gate
        book_invalid = aligned[:, BOOK_VALID_IDX] != 1.0
        mbo_warmup = aligned[:, MBO_READY_IDX] != 1.0
        feed_problem = aligned[:, INVALIDITY_DELTA_IDX] > 0
        any_invalid = book_invalid | mbo_warmup | feed_problem
        
        day_stats = {
            'date': date,
            'n_labels': n_labels,
            'book_invalid_pct': 100 * book_invalid.sum() / n_labels,
            'mbo_warmup_pct': 100 * mbo_warmup.sum() / n_labels,
            'feed_problem_pct': 100 * feed_problem.sum() / n_labels,
            'any_invalid_pct': 100 * any_invalid.sum() / n_labels,
        }
        per_day_stats.append(day_stats)
        
        total_labels += n_labels
        book_invalid_count += book_invalid.sum()
        mbo_warmup_count += mbo_warmup.sum()
        feed_problem_count += feed_problem.sum()
        any_invalid_count += any_invalid.sum()
    
    return {
        'total_labels': total_labels,
        'book_invalid_pct': 100 * book_invalid_count / total_labels,
        'mbo_warmup_pct': 100 * mbo_warmup_count / total_labels,
        'feed_problem_pct': 100 * feed_problem_count / total_labels,
        'any_invalid_pct': 100 * any_invalid_count / total_labels,
        'per_day': per_day_stats,
    }


def compute_time_regime_values(
    data_dir: Path,
    split: str = 'train',
) -> Dict[str, Any]:
    """
    Check actual time_regime values in exports.
    
    Expected from docs: 0-3 (OPEN, EARLY, MIDDAY, CLOSE)
    Question: Does 4 (CLOSED/after-hours) ever appear?
    """
    split_dir = data_dir / split
    feature_files = sorted(split_dir.glob('*_features.npy'))
    
    all_values = set()
    value_counts = {}
    
    for feat_file in feature_files:
        features, labels, date = load_day_data(feat_file)
        aligned = align_correct(features, len(labels))
        
        time_regimes = aligned[:, TIME_REGIME_IDX]
        unique_vals = np.unique(time_regimes)
        
        for val in unique_vals:
            val_float = float(val)
            all_values.add(val_float)
            count = (time_regimes == val).sum()
            value_counts[val_float] = value_counts.get(val_float, 0) + count
    
    total = sum(value_counts.values())
    value_pcts = {k: 100 * v / total for k, v in value_counts.items()}
    
    return {
        'unique_values': sorted(list(all_values)),
        'value_counts': value_counts,
        'value_percentages': value_pcts,
        'contains_4': 4.0 in all_values,
        'expected_0_to_3_only': all_values.issubset({0.0, 1.0, 2.0, 3.0}),
    }


def compute_correlation_comparison(
    data_dir: Path,
    split: str = 'train',
    max_days: int = 30,
) -> Dict[str, Any]:
    """
    Compare signal-label correlation with CORRECT vs INCORRECT alignment.
    
    This quantifies how much the 99-sample offset affects correlations.
    """
    split_dir = data_dir / split
    feature_files = sorted(split_dir.glob('*_features.npy'))[:max_days]
    
    # Collect data with both alignment methods
    correct_ofi = []
    correct_labels = []
    
    incorrect_ofi = []
    incorrect_labels = []
    
    for feat_file in feature_files:
        features, labels, date = load_day_data(feat_file)
        n_labels = len(labels)
        
        # CORRECT: Use proper alignment
        aligned = align_correct(features, n_labels)
        correct_ofi.extend(aligned[:, TRUE_OFI_IDX].tolist())
        correct_labels.extend(labels.tolist())
        
        # INCORRECT: Use crude step-based subsampling (simulating the bug)
        step = max(1, features.shape[0] // n_labels)
        subsampled = features[::step][:n_labels]
        incorrect_ofi.extend(subsampled[:, TRUE_OFI_IDX].tolist())
        incorrect_labels.extend(labels.tolist())
    
    # Compute correlations
    correct_ofi = np.array(correct_ofi)
    correct_labels = np.array(correct_labels)
    incorrect_ofi = np.array(incorrect_ofi)
    incorrect_labels = np.array(incorrect_labels)
    
    # Handle NaN/Inf
    correct_mask = np.isfinite(correct_ofi) & np.isfinite(correct_labels)
    incorrect_mask = np.isfinite(incorrect_ofi) & np.isfinite(incorrect_labels)
    
    if correct_mask.sum() > 100 and incorrect_mask.sum() > 100:
        correct_corr = np.corrcoef(correct_ofi[correct_mask], correct_labels[correct_mask])[0, 1]
        incorrect_corr = np.corrcoef(incorrect_ofi[incorrect_mask], incorrect_labels[incorrect_mask])[0, 1]
    else:
        correct_corr = 0.0
        incorrect_corr = 0.0
    
    return {
        'signal': 'true_ofi',
        'correct_alignment_correlation': float(correct_corr),
        'incorrect_alignment_correlation': float(incorrect_corr),
        'difference': float(correct_corr - incorrect_corr),
        'ratio': float(correct_corr / incorrect_corr) if incorrect_corr != 0 else float('inf'),
        'n_samples_correct': int(correct_mask.sum()),
        'n_samples_incorrect': int(incorrect_mask.sum()),
        'days_analyzed': len(feature_files),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Validate alignment bugs and quantify their impact'
    )
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--symbol', type=str, default='NVDA')
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--split', type=str, default='train')
    
    args = parser.parse_args()
    
    if not args.data_dir.exists():
        print(f"❌ Error: Data directory not found: {args.data_dir}")
        sys.exit(1)
    
    if args.output_dir is None:
        args.output_dir = Path(__file__).parent.parent / 'docs'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = datetime.now()
    
    print("=" * 80)
    print("PHASE 0: EMPIRICAL VALIDATION OF ALIGNMENT BUGS")
    print("=" * 80)
    print(f"Symbol: {args.symbol}")
    print(f"Data: {args.data_dir}")
    print(f"Split: {args.split}")
    print(f"Started: {start_time.isoformat()}")
    
    results = {
        'symbol': args.symbol,
        'data_dir': str(args.data_dir),
        'split': args.split,
        'timestamp': start_time.isoformat(),
    }
    
    # 1. Alignment Drift Analysis
    print("\n" + "-" * 40)
    print("[1/4] Measuring Alignment Drift...")
    print("-" * 40)
    
    drift_results = compute_alignment_drift(args.data_dir, args.split)
    results['alignment_drift'] = drift_results
    
    if drift_results['summary'].get('max_offset_samples', 0) > 0:
        print(f"  ⚠️  CRITICAL: Maximum offset = {drift_results['summary']['max_offset_samples']} samples")
        print(f"      = {drift_results['summary']['max_offset_labels']:.1f} labels worth of drift")
        print(f"      Occurs at day: {drift_results['summary']['max_offset_day']}")
        print(f"      Drift per day: ~{drift_results['summary']['drift_per_day']:.1f} samples")
    else:
        print("  ✅ No alignment drift detected (single day?)")
    
    # 2. Safety Gate Analysis
    print("\n" + "-" * 40)
    print("[2/4] Measuring Safety Gate Rates...")
    print("-" * 40)
    
    gate_results = compute_safety_gate_stats(args.data_dir, args.split)
    results['safety_gates'] = {k: v for k, v in gate_results.items() if k != 'per_day'}
    
    print(f"  Total labels analyzed: {gate_results['total_labels']:,}")
    print(f"  book_valid == 0:       {gate_results['book_invalid_pct']:.4f}%")
    print(f"  mbo_ready == 0:        {gate_results['mbo_warmup_pct']:.4f}%")
    print(f"  invalidity_delta > 0:  {gate_results['feed_problem_pct']:.4f}%")
    print(f"  ANY invalid:           {gate_results['any_invalid_pct']:.4f}%")
    
    if gate_results['any_invalid_pct'] > 0.1:
        print(f"  ⚠️  {gate_results['any_invalid_pct']:.2f}% of labels have invalid data")
    else:
        print(f"  ✅ Very low invalid rate ({gate_results['any_invalid_pct']:.4f}%)")
    
    # 3. Time Regime Analysis
    print("\n" + "-" * 40)
    print("[3/4] Checking time_regime Values...")
    print("-" * 40)
    
    regime_results = compute_time_regime_values(args.data_dir, args.split)
    results['time_regime'] = regime_results
    
    print(f"  Unique values found: {regime_results['unique_values']}")
    print(f"  Contains 4 (after-hours): {regime_results['contains_4']}")
    print(f"  Only 0-3 (expected): {regime_results['expected_0_to_3_only']}")
    
    if regime_results['contains_4']:
        print(f"  ⚠️  time_regime=4 found! Docs say 0-3 only")
    elif not regime_results['expected_0_to_3_only']:
        print(f"  ⚠️  Unexpected time_regime values!")
    else:
        print(f"  ✅ time_regime values are 0-3 as expected")
    
    for val, pct in sorted(regime_results['value_percentages'].items()):
        print(f"      Regime {int(val)}: {pct:.1f}%")
    
    # 4. Correlation Comparison
    print("\n" + "-" * 40)
    print("[4/4] Comparing Correct vs Incorrect Alignment Correlation...")
    print("-" * 40)
    
    corr_results = compute_correlation_comparison(args.data_dir, args.split)
    results['correlation_comparison'] = corr_results
    
    print(f"  Signal: {corr_results['signal']}")
    print(f"  Days analyzed: {corr_results['days_analyzed']}")
    print(f"  CORRECT alignment correlation:   {corr_results['correct_alignment_correlation']:+.6f}")
    print(f"  INCORRECT alignment correlation: {corr_results['incorrect_alignment_correlation']:+.6f}")
    print(f"  Difference: {corr_results['difference']:+.6f}")
    
    if abs(corr_results['difference']) > 0.001:
        print(f"  ⚠️  Alignment matters! Correlation differs by {abs(corr_results['difference']):.4f}")
    else:
        print(f"  ✅ Alignment has minimal effect on this signal's correlation")
    
    # Save results
    output_path = args.output_dir / f'{args.symbol.lower()}_alignment_validation.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Final Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    critical_issues = []
    if drift_results['summary'].get('max_offset_samples', 0) > 0:
        critical_issues.append(f"Alignment drift: {drift_results['summary']['max_offset_samples']} samples")
    if gate_results['any_invalid_pct'] > 1.0:
        critical_issues.append(f"Invalid samples: {gate_results['any_invalid_pct']:.1f}%")
    if not regime_results['expected_0_to_3_only']:
        critical_issues.append("Unexpected time_regime values")
    if abs(corr_results['difference']) > 0.005:
        critical_issues.append(f"Correlation affected by {abs(corr_results['difference']):.4f}")
    
    if critical_issues:
        print("⚠️  CRITICAL ISSUES CONFIRMED:")
        for issue in critical_issues:
            print(f"    - {issue}")
        print("\n  → Proceed with fixes as planned")
    else:
        print("✅ No critical issues detected")
    
    print(f"\nDuration: {duration:.1f} seconds")
    print(f"Output: {output_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()

