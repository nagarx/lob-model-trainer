#!/usr/bin/env python3
"""
Complete memory-efficient streaming analysis for large datasets.

Runs ALL analysis types with streaming/sampling to stay under 1GB memory:
1. Data Overview (streaming)
2. Label Analysis (streaming) 
3. Signal Statistics (streaming)
4. Signal-Label Correlations (sampled)
5. Temporal Dynamics (sampled)
6. Generalization / Walk-Forward (per-day)

Usage:
    python scripts/run_complete_streaming_analysis.py \
        --data-dir ../data/exports/nvda_98feat_full \
        --symbol NVDA
"""

import argparse
import json
import sys
import gc
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lobtrainer.analysis.streaming import (
    iter_days,
    compute_streaming_overview,
    compute_streaming_label_analysis,
    RunningStats,
)
from lobtrainer.constants import FeatureIndex


# Signal configuration - indices from FeatureIndex
SIGNAL_INDICES = {
    'true_ofi': FeatureIndex.TRUE_OFI,
    'depth_norm_ofi': FeatureIndex.DEPTH_NORM_OFI,
    'executed_pressure': FeatureIndex.EXECUTED_PRESSURE,
    'signed_mp_delta_bps': FeatureIndex.SIGNED_MP_DELTA_BPS,
    'trade_asymmetry': FeatureIndex.TRADE_ASYMMETRY,
    'cancel_asymmetry': FeatureIndex.CANCEL_ASYMMETRY,
    'fragility_score': FeatureIndex.FRAGILITY_SCORE,
    'depth_asymmetry': FeatureIndex.DEPTH_ASYMMETRY,
}


def compute_signal_label_correlations(
    data_dir: Path,
    split: str = 'train',
    sample_rate: int = 100,  # Sample every Nth point
    max_samples: int = 500000,
) -> Dict[str, Any]:
    """
    Compute signal-label correlations with memory-efficient sampling.
    
    Uses reservoir sampling to get representative samples without loading all data.
    """
    print("  Computing signal-label correlations (sampled)...")
    
    # Collect samples
    signal_samples = {name: [] for name in SIGNAL_INDICES}
    label_samples = []
    
    total_samples = 0
    for day in iter_days(data_dir, split, dtype=np.float32):
        # Sample from this day
        step = max(1, day.n_samples // (max_samples // 165))  # Aim for even distribution
        
        for name, idx in SIGNAL_INDICES.items():
            signal_samples[name].extend(day.features[::step, idx].tolist())
        
        # Subsample labels to align (labels are fewer than features)
        label_step = max(1, day.n_labels // (len(day.features[::step]) // 10))
        label_samples.extend(day.labels[::label_step].tolist())
        
        total_samples += len(day.features[::step])
        
        if total_samples > max_samples:
            break
    
    # Align lengths
    min_len = min(len(label_samples), min(len(v) for v in signal_samples.values()))
    label_arr = np.array(label_samples[:min_len])
    
    results = {}
    for name, samples in signal_samples.items():
        signal_arr = np.array(samples[:min_len])
        
        # Handle NaN/Inf
        mask = np.isfinite(signal_arr) & np.isfinite(label_arr)
        if mask.sum() < 100:
            results[name] = {'correlation': 0.0, 'n_samples': 0, 'is_significant': False}
            continue
        
        corr = np.corrcoef(signal_arr[mask], label_arr[mask])[0, 1]
        
        # Significance test (approximate)
        n = mask.sum()
        t_stat = corr * np.sqrt((n - 2) / (1 - corr**2 + 1e-10))
        # For n > 100, t > 2.58 is p < 0.01
        is_significant = abs(t_stat) > 2.58
        
        results[name] = {
            'correlation': float(corr) if np.isfinite(corr) else 0.0,
            'n_samples': int(n),
            'is_significant': is_significant,
            't_statistic': float(t_stat) if np.isfinite(t_stat) else 0.0,
        }
    
    # Sort by absolute correlation
    sorted_results = dict(sorted(
        results.items(), 
        key=lambda x: abs(x[1]['correlation']), 
        reverse=True
    ))
    
    gc.collect()
    return {'signal_correlations': sorted_results, 'total_samples_used': min_len}


def compute_signal_autocorrelations(
    data_dir: Path,
    split: str = 'train',
    max_lag: int = 50,
    sample_days: int = 20,
) -> Dict[str, Any]:
    """
    Compute signal autocorrelation using a sample of days.
    """
    print("  Computing signal autocorrelations (sampled days)...")
    
    results = {}
    
    # Get list of dates
    dates = []
    for day in iter_days(data_dir, split, dtype=np.float32):
        dates.append(day.date)
    
    # Sample evenly across time
    if len(dates) > sample_days:
        step = len(dates) // sample_days
        sample_dates = set(dates[::step])
    else:
        sample_dates = set(dates)
    
    # Collect signal data from sampled days
    signal_data = {name: [] for name in SIGNAL_INDICES}
    
    for day in iter_days(data_dir, split, dtype=np.float32):
        if day.date not in sample_dates:
            continue
        
        for name, idx in SIGNAL_INDICES.items():
            # Subsample within day for memory
            signal_data[name].extend(day.features[::10, idx].tolist())
    
    # Compute ACF for each signal
    for name, data in signal_data.items():
        arr = np.array(data, dtype=np.float32)
        
        if len(arr) < max_lag * 2:
            results[name] = {'acf': [], 'half_life': None}
            continue
        
        # Compute ACF
        mean = arr.mean()
        var = arr.var()
        
        if var < 1e-10:
            results[name] = {'acf': [1.0] * min(max_lag, len(arr)), 'half_life': None}
            continue
        
        acf = []
        for lag in range(min(max_lag, len(arr) // 2)):
            if lag == 0:
                acf.append(1.0)
            else:
                cov = np.mean((arr[:-lag] - mean) * (arr[lag:] - mean))
                acf.append(float(cov / var))
        
        # Compute half-life (where ACF drops to 0.5 * lag-1)
        half_life = None
        if len(acf) > 1 and acf[1] > 0:
            threshold = acf[1] * 0.5
            for i, a in enumerate(acf[1:], 1):
                if a < threshold:
                    half_life = i
                    break
        
        results[name] = {
            'acf_lag_1': float(acf[1]) if len(acf) > 1 else 0.0,
            'acf_lag_5': float(acf[5]) if len(acf) > 5 else 0.0,
            'acf_lag_10': float(acf[10]) if len(acf) > 10 else 0.0,
            'half_life': half_life,
            'acf_first_20': [float(a) for a in acf[:20]],
        }
    
    gc.collect()
    return {'signal_autocorrelations': results}


def compute_predictive_decay(
    data_dir: Path,
    split: str = 'train',
    horizons: List[int] = [1, 2, 5, 10, 20, 50],
    sample_days: int = 30,
) -> Dict[str, Any]:
    """
    Compute how signal-label correlation decays with prediction horizon.
    """
    print("  Computing predictive decay (sampled)...")
    
    # Get sample of days
    dates = []
    for day in iter_days(data_dir, split, dtype=np.float32):
        dates.append(day.date)
    
    if len(dates) > sample_days:
        step = len(dates) // sample_days
        sample_dates = set(dates[::step])
    else:
        sample_dates = set(dates)
    
    # Collect aligned signal and label data
    all_signals = {name: [] for name in SIGNAL_INDICES}
    all_labels = []
    
    for day in iter_days(data_dir, split, dtype=np.float32):
        if day.date not in sample_dates:
            continue
        
        # Subsample features to roughly align with labels
        step = max(1, day.n_samples // day.n_labels)
        for name, idx in SIGNAL_INDICES.items():
            all_signals[name].extend(day.features[::step, idx][:day.n_labels].tolist())
        all_labels.extend(day.labels.tolist())
    
    # Compute correlations at different horizons
    results = {}
    labels = np.array(all_labels)
    
    for name, signal_list in all_signals.items():
        signal = np.array(signal_list[:len(labels)])
        
        decay_results = {}
        for horizon in horizons:
            if horizon >= len(signal) - 1:
                continue
            
            # Correlation between signal[t] and label[t+horizon]
            sig = signal[:-horizon]
            lab = labels[horizon:]
            
            mask = np.isfinite(sig)
            if mask.sum() < 100:
                continue
            
            corr = np.corrcoef(sig[mask], lab[mask])[0, 1]
            decay_results[f'horizon_{horizon}'] = float(corr) if np.isfinite(corr) else 0.0
        
        results[name] = decay_results
    
    gc.collect()
    return {'predictive_decay': results}


def compute_walk_forward_validation(
    data_dir: Path,
    split: str = 'train',
    min_train_days: int = 20,
) -> Dict[str, Any]:
    """
    Walk-forward validation: train on N days, test on day N+1.
    """
    print("  Computing walk-forward validation...")
    
    # Collect per-day statistics
    day_stats = []
    
    for day in iter_days(data_dir, split, dtype=np.float32):
        # Compute signal-label correlations for this day
        day_corrs = {}
        
        # Align features with labels
        step = max(1, day.n_samples // day.n_labels)
        aligned_features = day.features[::step][:day.n_labels]
        
        for name, idx in SIGNAL_INDICES.items():
            signal = aligned_features[:, idx]
            mask = np.isfinite(signal)
            if mask.sum() > 50:
                corr = np.corrcoef(signal[mask], day.labels[mask])[0, 1]
                day_corrs[name] = float(corr) if np.isfinite(corr) else 0.0
            else:
                day_corrs[name] = 0.0
        
        day_stats.append({
            'date': day.date,
            'n_labels': day.n_labels,
            'up_pct': float((day.labels == 1).mean() * 100),
            'down_pct': float((day.labels == -1).mean() * 100),
            'stable_pct': float((day.labels == 0).mean() * 100),
            'signal_correlations': day_corrs,
        })
    
    # Compute walk-forward metrics
    walk_forward_results = []
    
    for i in range(min_train_days, len(day_stats)):
        train_days = day_stats[:i]
        test_day = day_stats[i]
        
        # Average correlations from training days
        train_corrs = {name: [] for name in SIGNAL_INDICES}
        for d in train_days:
            for name, corr in d['signal_correlations'].items():
                train_corrs[name].append(corr)
        
        avg_train_corrs = {name: float(np.mean(corrs)) for name, corrs in train_corrs.items()}
        
        walk_forward_results.append({
            'test_date': test_day['date'],
            'train_days': i,
            'test_label_dist': {
                'up': test_day['up_pct'],
                'down': test_day['down_pct'],
                'stable': test_day['stable_pct'],
            },
            'train_avg_correlations': avg_train_corrs,
            'test_correlations': test_day['signal_correlations'],
        })
    
    # Compute stability metrics
    stability = {}
    for name in SIGNAL_INDICES:
        corrs = [d['signal_correlations'][name] for d in day_stats]
        stability[name] = {
            'mean': float(np.mean(corrs)),
            'std': float(np.std(corrs)),
            'stability_ratio': float(abs(np.mean(corrs)) / (np.std(corrs) + 1e-10)),
        }
    
    gc.collect()
    return {
        'day_stats': day_stats,
        'walk_forward': walk_forward_results[:20],  # First 20 for brevity
        'signal_stability': stability,
        'n_days': len(day_stats),
    }


def main():
    parser = argparse.ArgumentParser(description='Complete streaming analysis')
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--symbol', type=str, default='NVDA')
    parser.add_argument('--output-dir', type=Path, default=None)
    
    args = parser.parse_args()
    
    if not args.data_dir.exists():
        print(f"❌ Error: Data directory not found: {args.data_dir}")
        sys.exit(1)
    
    if args.output_dir is None:
        args.output_dir = Path(__file__).parent.parent / 'docs'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = datetime.now()
    
    print("=" * 80)
    print("COMPLETE STREAMING ANALYSIS")
    print("=" * 80)
    print(f"Symbol: {args.symbol}")
    print(f"Data: {args.data_dir}")
    print(f"Started: {start_time.isoformat()}")
    
    all_results = {
        'symbol': args.symbol,
        'data_dir': str(args.data_dir),
        'analysis_timestamp': start_time.isoformat(),
    }
    
    # 1. Basic Overview (already have from streaming)
    print("\n[1/6] Data Overview...")
    try:
        overview = compute_streaming_overview(args.data_dir, symbol=args.symbol)
        all_results['overview'] = {
            'total_days': overview['total_days'],
            'total_samples': overview['total_samples'],
            'total_labels': overview['total_labels'],
            'date_range': overview['date_range'],
            'data_quality': overview['data_quality'],
            'label_distribution': overview['label_distribution'],
        }
        print(f"  ✅ {overview['total_days']} days, {overview['total_samples']:,} samples")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        traceback.print_exc()
    
    # 2. Label Analysis
    print("\n[2/6] Label Analysis...")
    try:
        labels = compute_streaming_label_analysis(args.data_dir, split='train')
        all_results['labels'] = {
            'distribution': labels['distribution'],
            'autocorrelation': labels['autocorrelation'],
            'transition_matrix': labels['transition_matrix'],
        }
        print(f"  ✅ Lag-1 ACF: {labels['autocorrelation']['lag_1']:.4f}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        traceback.print_exc()
    
    # 3. Signal-Label Correlations
    print("\n[3/6] Signal-Label Correlations...")
    try:
        corr_results = compute_signal_label_correlations(args.data_dir, split='train')
        all_results['signal_correlations'] = corr_results
        
        # Print top signals
        print("  Top predictive signals:")
        for name, stats in list(corr_results['signal_correlations'].items())[:3]:
            sig = "***" if stats['is_significant'] else ""
            print(f"    {name}: r = {stats['correlation']:+.4f} {sig}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        traceback.print_exc()
    
    # 4. Signal Autocorrelations
    print("\n[4/6] Signal Autocorrelations...")
    try:
        acf_results = compute_signal_autocorrelations(args.data_dir, split='train')
        all_results['signal_autocorrelations'] = acf_results
        
        print("  Signal persistence:")
        for name, stats in acf_results['signal_autocorrelations'].items():
            hl = stats.get('half_life', 'N/A')
            print(f"    {name}: lag-1 ACF = {stats['acf_lag_1']:.3f}, half-life = {hl}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        traceback.print_exc()
    
    # 5. Predictive Decay
    print("\n[5/6] Predictive Decay...")
    try:
        decay_results = compute_predictive_decay(args.data_dir, split='train')
        all_results['predictive_decay'] = decay_results
        
        print("  Decay for true_ofi:")
        if 'true_ofi' in decay_results['predictive_decay']:
            for h, c in decay_results['predictive_decay']['true_ofi'].items():
                print(f"    {h}: r = {c:+.4f}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        traceback.print_exc()
    
    # 6. Walk-Forward Validation
    print("\n[6/6] Walk-Forward Validation...")
    try:
        wf_results = compute_walk_forward_validation(args.data_dir, split='train')
        all_results['generalization'] = {
            'n_days': wf_results['n_days'],
            'signal_stability': wf_results['signal_stability'],
            'walk_forward_sample': wf_results['walk_forward'][:5],
        }
        
        print("  Signal stability (|mean|/std):")
        sorted_stability = sorted(
            wf_results['signal_stability'].items(),
            key=lambda x: x[1]['stability_ratio'],
            reverse=True
        )
        for name, stats in sorted_stability[:3]:
            print(f"    {name}: ratio = {stats['stability_ratio']:.2f}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        traceback.print_exc()
    
    # Save results
    output_path = args.output_dir / f'{args.symbol.lower()}_complete_analysis.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 80)
    print("✅ COMPLETE ANALYSIS FINISHED")
    print("=" * 80)
    print(f"Duration: {duration:.1f} seconds")
    print(f"Output: {output_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()

