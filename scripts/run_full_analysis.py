#!/usr/bin/env python3
"""
Run comprehensive analysis on the full NVDA 98-feature dataset.
Writes all output to files for reliable capture.
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from io import StringIO

# Setup paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / 'data' / 'exports' / 'nvda_98feat_full'
OUTPUT_DIR = Path(__file__).parent.parent / 'docs'
OUTPUT_DIR.mkdir(exist_ok=True)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Capture all output
output_buffer = StringIO()

def log(msg=""):
    """Print and capture output."""
    print(msg)
    output_buffer.write(msg + "\n")

def save_output():
    """Save captured output to file."""
    with open(OUTPUT_DIR / 'nvda_full_analysis_output.txt', 'w') as f:
        f.write(output_buffer.getvalue())

def clean_for_json(obj):
    """Clean numpy types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def run_data_overview():
    """Run data overview analysis."""
    log("\n" + "=" * 80)
    log("1. DATA OVERVIEW ANALYSIS")
    log("=" * 80)
    
    from lobtrainer.analysis import generate_dataset_summary, print_data_overview
    
    summary = generate_dataset_summary(DATA_DIR, symbol='NVDA')
    
    log(f"\nSymbol: {summary.symbol}")
    log(f"Date range: {summary.date_range[0]} to {summary.date_range[1]}")
    log(f"Total days: {summary.total_days}")
    log(f"Total samples: {summary.total_samples:,}")
    log(f"Total labels: {summary.total_labels:,}")
    log(f"Feature count: {summary.feature_count}")
    
    # Data quality
    dq = summary.data_quality
    if dq.is_clean:
        log(f"Data quality: ✅ Clean (no NaN/Inf)")
    else:
        log(f"Data quality: ❌ Issues detected")
        log(f"  NaN count: {dq.nan_count}")
        log(f"  Inf count: {dq.inf_count}")
    
    # Label distribution
    ld = summary.label_distribution
    log(f"\nLabel Distribution:")
    log(f"  Down:   {ld.down_count:,} ({ld.down_pct:.1f}%)")
    log(f"  Stable: {ld.stable_count:,} ({ld.stable_pct:.1f}%)")
    log(f"  Up:     {ld.up_count:,} ({ld.up_pct:.1f}%)")
    log(f"  Balance: {'✅ Balanced' if ld.is_balanced else '⚠️ Imbalanced'} (ratio {ld.imbalance_ratio:.2f})")
    
    # Save to JSON
    with open(OUTPUT_DIR / 'nvda_full_data_overview.json', 'w') as f:
        json.dump(clean_for_json(summary.to_dict()), f, indent=2)
    
    return summary


def run_label_analysis():
    """Run comprehensive label analysis."""
    log("\n" + "=" * 80)
    log("2. LABEL ANALYSIS")
    log("=" * 80)
    
    from lobtrainer.analysis import load_split, run_label_analysis, print_label_analysis
    
    # Load train data
    data = load_split(DATA_DIR, 'train')
    features = data['features']
    labels = data['labels']
    dates = data['dates']
    
    log(f"\nLoaded train data:")
    log(f"  Features shape: {features.shape}")
    log(f"  Labels shape: {labels.shape}")
    log(f"  Date range: {dates[0]} to {dates[-1]}")
    
    # Run analysis
    log("\nRunning label analysis...")
    summary = run_label_analysis(features=features, labels=labels, window_size=100, stride=10)
    
    # Distribution
    d = summary.distribution
    log(f"\nLabel Distribution:")
    log(f"  Down:   {d.down_count:,} ({d.down_pct:.1f}%)")
    log(f"  Stable: {d.stable_count:,} ({d.stable_pct:.1f}%)")
    log(f"  Up:     {d.up_count:,} ({d.up_pct:.1f}%)")
    
    # Autocorrelation
    a = summary.autocorrelation
    log(f"\nAutocorrelation:")
    log(f"  Lag-1 ACF: {a.lag_1_acf:+.4f}")
    log(f"  Lag-5 ACF: {a.lag_5_acf:+.4f}")
    log(f"  Lag-10 ACF: {a.lag_10_acf:+.4f}")
    log(f"  Interpretation: {a.interpretation}")
    
    # Transition matrix
    t = summary.transition_matrix
    log(f"\nTransition Matrix (P(To | From)):")
    labels_names = ['Down', 'Stable', 'Up']
    log("          " + "  ".join(f"{name:>7}" for name in labels_names))
    for i, from_label in enumerate(labels_names):
        row = "  ".join(f"{t.probabilities[i][j]:7.3f}" for j in range(3))
        log(f"  {from_label:>7}: {row}")
    
    # Diagonal persistence
    persist = [t.probabilities[i][i] for i in range(3)]
    log(f"\nPersistence (staying in same state):")
    log(f"  Down: {persist[0]:.1%}")
    log(f"  Stable: {persist[1]:.1%}")
    log(f"  Up: {persist[2]:.1%}")
    
    # Top signal correlations
    if summary.signal_correlations:
        log(f"\nTop Signal-Label Correlations:")
        for i, sig in enumerate(summary.signal_correlations[:5]):
            signif = "✅" if sig.is_significant else "❌"
            log(f"  {i+1}. {sig.signal_name}: r = {sig.correlation:+.4f} {signif}")
    
    # Save to JSON
    output_data = {
        'symbol': 'NVDA',
        'split': 'train',
        'date_range': [dates[0], dates[-1]],
        'n_samples': int(features.shape[0]),
        'n_labels': len(labels),
        **summary.to_dict(),
    }
    with open(OUTPUT_DIR / 'nvda_full_label_analysis.json', 'w') as f:
        json.dump(clean_for_json(output_data), f, indent=2)
    
    return summary


def run_signal_analysis():
    """Run signal statistics and correlation analysis."""
    log("\n" + "=" * 80)
    log("3. SIGNAL ANALYSIS")
    log("=" * 80)
    
    from lobtrainer.analysis import load_split
    from lobtrainer.constants import (
        SIGNAL_NAMES, TRUE_OFI, DEPTH_NORM_OFI, EXECUTED_PRESSURE,
        SIGNED_MP_DELTA_BPS, TRADE_ASYMMETRY, CANCEL_ASYMMETRY,
        FRAGILITY_SCORE, DEPTH_ASYMMETRY, TIME_REGIME
    )
    
    # Load train data
    data = load_split(DATA_DIR, 'train')
    features = data['features']
    labels = data['labels']
    
    log(f"\nLoaded {features.shape[0]:,} samples with {features.shape[1]} features")
    
    # Key signal indices
    signal_indices = {
        'true_ofi': TRUE_OFI,
        'depth_norm_ofi': DEPTH_NORM_OFI,
        'executed_pressure': EXECUTED_PRESSURE,
        'signed_mp_delta_bps': SIGNED_MP_DELTA_BPS,
        'trade_asymmetry': TRADE_ASYMMETRY,
        'cancel_asymmetry': CANCEL_ASYMMETRY,
        'fragility_score': FRAGILITY_SCORE,
        'depth_asymmetry': DEPTH_ASYMMETRY,
    }
    
    log(f"\nSignal Statistics (Train Set):")
    log("-" * 80)
    log(f"{'Signal':<25} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12} {'Label Corr':>12}")
    log("-" * 80)
    
    signal_stats = {}
    for name, idx in signal_indices.items():
        col = features[:, idx]
        finite_mask = np.isfinite(col)
        col_clean = col[finite_mask]
        
        # Align with labels (subsample to match labels)
        step = len(col) // len(labels) if len(labels) > 0 else 1
        label_aligned_col = col[::step][:len(labels)]
        
        # Calculate correlation with labels
        label_mask = np.isfinite(label_aligned_col)
        if label_mask.sum() > 100:
            corr = np.corrcoef(label_aligned_col[label_mask], labels[label_mask])[0, 1]
        else:
            corr = np.nan
        
        stats = {
            'mean': float(np.mean(col_clean)),
            'std': float(np.std(col_clean)),
            'min': float(np.min(col_clean)),
            'max': float(np.max(col_clean)),
            'label_corr': float(corr) if np.isfinite(corr) else 0.0,
        }
        signal_stats[name] = stats
        
        log(f"{name:<25} {stats['mean']:>12.4f} {stats['std']:>12.4f} {stats['min']:>12.4f} {stats['max']:>12.4f} {stats['label_corr']:>+12.4f}")
    
    log("-" * 80)
    
    # Signal correlations matrix (top signals)
    log(f"\nSignal Cross-Correlations:")
    signal_data = np.column_stack([features[:, idx] for idx in signal_indices.values()])
    signal_names = list(signal_indices.keys())
    
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(signal_data.T)
    
    log("\n" + " " * 20 + "  ".join(f"{n[:8]:>8}" for n in signal_names))
    for i, name in enumerate(signal_names):
        row = "  ".join(f"{corr_matrix[i, j]:>8.3f}" for j in range(len(signal_names)))
        log(f"{name:<20} {row}")
    
    # Identify highly correlated pairs
    log(f"\nHighly Correlated Signal Pairs (|r| > 0.5):")
    for i in range(len(signal_names)):
        for j in range(i + 1, len(signal_names)):
            r = corr_matrix[i, j]
            if abs(r) > 0.5:
                log(f"  {signal_names[i]} ↔ {signal_names[j]}: r = {r:+.3f}")
    
    # Save results
    with open(OUTPUT_DIR / 'nvda_full_signal_analysis.json', 'w') as f:
        json.dump({
            'signal_stats': signal_stats,
            'correlation_matrix': corr_matrix.tolist(),
            'signal_names': signal_names,
        }, f, indent=2)
    
    return signal_stats


def run_temporal_analysis():
    """Run temporal dynamics analysis."""
    log("\n" + "=" * 80)
    log("4. TEMPORAL DYNAMICS ANALYSIS")
    log("=" * 80)
    
    from lobtrainer.analysis import load_split
    from lobtrainer.constants import TRUE_OFI, DEPTH_NORM_OFI, EXECUTED_PRESSURE
    
    # Load data
    data = load_split(DATA_DIR, 'train')
    features = data['features']
    labels = data['labels']
    
    log(f"\nAnalyzing signal persistence and predictive decay...")
    
    # Analyze OFI autocorrelation
    ofi = features[:, TRUE_OFI]
    
    # Calculate autocorrelation at different lags
    lags = [1, 5, 10, 20, 50, 100]
    acf_values = []
    
    log(f"\nOFI Autocorrelation:")
    for lag in lags:
        if lag < len(ofi):
            acf = np.corrcoef(ofi[:-lag], ofi[lag:])[0, 1]
            acf_values.append(acf)
            log(f"  Lag-{lag}: {acf:+.4f}")
    
    # Calculate half-life (where ACF drops to 0.5 of lag-1)
    lag1_acf = acf_values[0]
    half_life_threshold = lag1_acf * 0.5
    half_life = None
    for i, (lag, acf) in enumerate(zip(lags, acf_values)):
        if acf < half_life_threshold:
            # Interpolate
            if i > 0:
                prev_lag, prev_acf = lags[i-1], acf_values[i-1]
                half_life = prev_lag + (lag - prev_lag) * (prev_acf - half_life_threshold) / (prev_acf - acf)
            else:
                half_life = lag
            break
    
    if half_life:
        log(f"\n  Half-life: ~{half_life:.1f} samples")
        log(f"  Interpretation: OFI signal decays to 50% strength in ~{half_life:.0f} samples")
    
    # Predictive decay: correlation between signal at time t and label at time t+horizon
    horizons = [1, 5, 10, 20, 50, 100]
    
    log(f"\nPredictive Decay (OFI → Future Labels):")
    log("-" * 50)
    
    # Subsample OFI to align with labels
    step = len(ofi) // len(labels) if len(labels) > 0 else 1
    ofi_aligned = ofi[::step][:len(labels)]
    
    for horizon in horizons:
        if horizon < len(labels):
            # Correlation between OFI now and label at horizon
            corr = np.corrcoef(ofi_aligned[:-horizon], labels[horizon:])[0, 1]
            log(f"  Horizon-{horizon}: r = {corr:+.4f}")
    
    # Time regime analysis
    log(f"\nTime Regime Distribution:")
    from lobtrainer.constants import TIME_REGIME
    time_regime = features[:, TIME_REGIME]
    
    # Count regime occurrences (after potential normalization, check unique values)
    unique_regimes = np.unique(time_regime[np.isfinite(time_regime)])
    log(f"  Unique regime values: {len(unique_regimes)}")
    
    # Save results
    results = {
        'ofi_autocorrelation': dict(zip([f'lag_{l}' for l in lags], [float(a) for a in acf_values])),
        'half_life_samples': float(half_life) if half_life else None,
    }
    
    with open(OUTPUT_DIR / 'nvda_full_temporal_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def run_generalization_check():
    """Run cross-day variance analysis."""
    log("\n" + "=" * 80)
    log("5. GENERALIZATION / CROSS-DAY ANALYSIS")
    log("=" * 80)
    
    from lobtrainer.constants import TRUE_OFI, DEPTH_NORM_OFI
    
    # Load per-day statistics
    train_dir = DATA_DIR / 'train'
    
    day_stats = []
    for feat_file in sorted(train_dir.glob("*_features.npy")):
        date = feat_file.stem.replace("_features", "")
        features = np.load(feat_file)
        label_file = train_dir / f"{date}_labels.npy"
        labels = np.load(label_file) if label_file.exists() else None
        
        ofi = features[:, TRUE_OFI]
        
        stats = {
            'date': date,
            'n_samples': len(features),
            'ofi_mean': float(np.mean(ofi)),
            'ofi_std': float(np.std(ofi)),
        }
        
        if labels is not None:
            stats['up_pct'] = float((labels == 1).mean() * 100)
            stats['down_pct'] = float((labels == -1).mean() * 100)
            stats['stable_pct'] = float((labels == 0).mean() * 100)
        
        day_stats.append(stats)
    
    log(f"\nAnalyzed {len(day_stats)} training days")
    
    # Calculate variance across days
    ofi_means = [s['ofi_mean'] for s in day_stats]
    ofi_stds = [s['ofi_std'] for s in day_stats]
    up_pcts = [s['up_pct'] for s in day_stats]
    down_pcts = [s['down_pct'] for s in day_stats]
    
    log(f"\nCross-Day Statistics:")
    log(f"  OFI Mean:  μ = {np.mean(ofi_means):.4f}, σ = {np.std(ofi_means):.4f}")
    log(f"  OFI Std:   μ = {np.mean(ofi_stds):.4f}, σ = {np.std(ofi_stds):.4f}")
    log(f"  Up %:      μ = {np.mean(up_pcts):.1f}%, σ = {np.std(up_pcts):.1f}%")
    log(f"  Down %:    μ = {np.mean(down_pcts):.1f}%, σ = {np.std(down_pcts):.1f}%")
    
    # Identify outlier days
    ofi_mean_zscore = (np.array(ofi_means) - np.mean(ofi_means)) / np.std(ofi_means)
    outlier_days = [(day_stats[i]['date'], z) for i, z in enumerate(ofi_mean_zscore) if abs(z) > 2]
    
    if outlier_days:
        log(f"\nOutlier Days (|z| > 2 for OFI mean):")
        for date, z in sorted(outlier_days, key=lambda x: abs(x[1]), reverse=True)[:5]:
            log(f"  {date}: z = {z:+.2f}")
    else:
        log(f"\n✅ No significant outlier days detected")
    
    # Train-Val-Test comparison
    log(f"\nSplit Comparison:")
    for split in ['train', 'val', 'test']:
        split_dir = DATA_DIR / split
        all_labels = []
        for f in split_dir.glob("*_labels.npy"):
            all_labels.extend(np.load(f))
        all_labels = np.array(all_labels)
        
        up = (all_labels == 1).mean() * 100
        down = (all_labels == -1).mean() * 100
        stable = (all_labels == 0).mean() * 100
        
        log(f"  {split:>5}: Up={up:.1f}%, Down={down:.1f}%, Stable={stable:.1f}%")
    
    # Save results
    with open(OUTPUT_DIR / 'nvda_full_generalization.json', 'w') as f:
        json.dump({
            'n_days': len(day_stats),
            'cross_day_ofi_mean_std': float(np.std(ofi_means)),
            'cross_day_up_pct_std': float(np.std(up_pcts)),
            'day_stats': day_stats,
        }, f, indent=2)
    
    return day_stats


def main():
    """Run all analyses."""
    start_time = datetime.now()
    
    log("=" * 80)
    log("COMPREHENSIVE ANALYSIS: NVDA 98-Feature Full Dataset")
    log(f"Dataset: {DATA_DIR}")
    log(f"Started: {start_time.isoformat()}")
    log("=" * 80)
    
    try:
        # 1. Data Overview
        run_data_overview()
        
        # 2. Label Analysis
        run_label_analysis()
        
        # 3. Signal Analysis
        run_signal_analysis()
        
        # 4. Temporal Dynamics
        run_temporal_analysis()
        
        # 5. Generalization Check
        run_generalization_check()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        log("\n" + "=" * 80)
        log("✅ ALL ANALYSES COMPLETE")
        log(f"Duration: {duration:.1f} seconds")
        log(f"Output directory: {OUTPUT_DIR}")
        log("=" * 80)
        
    except Exception as e:
        log(f"\n❌ ERROR: {e}")
        import traceback
        log(traceback.format_exc())
    
    finally:
        # Always save output
        save_output()
        print(f"\nOutput saved to: {OUTPUT_DIR / 'nvda_full_analysis_output.txt'}")


if __name__ == '__main__':
    main()

