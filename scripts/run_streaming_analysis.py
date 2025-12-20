#!/usr/bin/env python3
"""
Memory-efficient streaming analysis for large datasets.

Uses streaming/incremental algorithms to analyze datasets of any size
with constant memory usage (~200-500MB regardless of data size).

Key Features:
- Processes one day at a time (never loads full dataset)
- Uses Welford's algorithm for online mean/variance
- Memory-mapped file access where possible
- float32 by default (50% memory reduction vs float64)

Usage:
    python scripts/run_streaming_analysis.py --data-dir ../data/exports/nvda_98feat_full --symbol NVDA
"""

import argparse
import json
import sys
import gc
import tracemalloc
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lobtrainer.analysis.streaming import (
    compute_streaming_overview,
    compute_streaming_label_analysis,
    compute_streaming_signal_stats,
    estimate_memory_usage,
    get_memory_efficient_config,
    iter_days,
)


def format_bytes(n: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def main():
    parser = argparse.ArgumentParser(
        description='Memory-efficient streaming analysis for large datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        required=True,
        help='Path to dataset directory',
    )
    parser.add_argument(
        '--symbol',
        type=str,
        default='UNKNOWN',
        help='Symbol name for labeling',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory (default: data-dir/../docs or lob-model-trainer/docs)',
    )
    parser.add_argument(
        '--skip-overview',
        action='store_true',
        help='Skip data overview analysis',
    )
    parser.add_argument(
        '--skip-labels',
        action='store_true',
        help='Skip label analysis',
    )
    parser.add_argument(
        '--skip-signals',
        action='store_true',
        help='Skip signal statistics',
    )
    parser.add_argument(
        '--trace-memory',
        action='store_true',
        help='Enable memory tracing (slower but shows peak usage)',
    )
    
    args = parser.parse_args()
    
    # Validate data directory
    if not args.data_dir.exists():
        print(f"❌ Error: Data directory not found: {args.data_dir}")
        sys.exit(1)
    
    # Determine output directory
    if args.output_dir is None:
        # Try to find lob-model-trainer/docs
        args.output_dir = Path(__file__).parent.parent / 'docs'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Start timing and optional memory tracing
    start_time = datetime.now()
    if args.trace_memory:
        tracemalloc.start()
    
    print("=" * 80)
    print("MEMORY-EFFICIENT STREAMING ANALYSIS")
    print("=" * 80)
    print(f"\nSymbol: {args.symbol}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Started: {start_time.isoformat()}")
    
    # Show memory estimates
    print("\n[0/4] Estimating memory requirements...")
    try:
        mem_estimates = estimate_memory_usage(args.data_dir)
        print(f"\nDataset size estimates:")
        for split, est in mem_estimates.items():
            if split == 'total':
                print(f"  TOTAL: {est['samples']:,} samples = {est['gb']:.2f} GB (if loaded at once)")
            else:
                print(f"  {split}: {est['samples']:,} samples = {est['mb']:.1f} MB")
        
        config = get_memory_efficient_config()
        print(f"\nUsing memory-efficient config:")
        print(f"  dtype: {config['dtype']} (vs float64)")
        print(f"  max_days_in_memory: {config['max_days_in_memory']}")
        print(f"  Target peak memory: < 500 MB")
    except Exception as e:
        print(f"  Warning: Could not estimate memory: {e}")
    
    results = {}
    
    # 1. Data Overview
    if not args.skip_overview:
        print("\n[1/4] Computing data overview (streaming)...")
        try:
            overview = compute_streaming_overview(args.data_dir, symbol=args.symbol)
            results['overview'] = overview
            
            # Print summary
            print(f"\n  ✅ Overview complete:")
            print(f"     Total days: {overview['total_days']}")
            print(f"     Total samples: {overview['total_samples']:,}")
            print(f"     Date range: {overview['date_range'][0]} to {overview['date_range'][1]}")
            print(f"     Data quality: {'✅ Clean' if overview['data_quality']['is_clean'] else '❌ Issues'}")
            
            # Save
            output_path = args.output_dir / f'{args.symbol.lower()}_streaming_overview.json'
            with open(output_path, 'w') as f:
                json.dump(overview, f, indent=2)
            print(f"     Saved to: {output_path}")
            
            gc.collect()
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            traceback.print_exc()
    else:
        print("\n[1/4] Skipping data overview")
    
    # 2. Label Analysis
    if not args.skip_labels:
        print("\n[2/4] Computing label analysis (streaming)...")
        try:
            label_analysis = compute_streaming_label_analysis(args.data_dir, split='train')
            results['labels'] = label_analysis
            
            # Print summary
            dist = label_analysis['distribution']
            print(f"\n  ✅ Label analysis complete:")
            print(f"     Train days: {label_analysis['n_days']}")
            print(f"     Distribution: Down={dist['down_pct']:.1f}%, Stable={dist['stable_pct']:.1f}%, Up={dist['up_pct']:.1f}%")
            print(f"     Lag-1 ACF: {label_analysis['autocorrelation']['lag_1']:.4f}")
            
            # Transition matrix
            tm = label_analysis['transition_matrix']['probabilities']
            print(f"     Persistence: Down→Down={tm[0][0]:.1%}, Stable→Stable={tm[1][1]:.1%}, Up→Up={tm[2][2]:.1%}")
            
            # Save
            output_path = args.output_dir / f'{args.symbol.lower()}_streaming_labels.json'
            with open(output_path, 'w') as f:
                json.dump(label_analysis, f, indent=2)
            print(f"     Saved to: {output_path}")
            
            gc.collect()
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            traceback.print_exc()
    else:
        print("\n[2/4] Skipping label analysis")
    
    # 3. Signal Statistics
    if not args.skip_signals:
        print("\n[3/4] Computing signal statistics (streaming)...")
        try:
            signal_stats = compute_streaming_signal_stats(args.data_dir, split='train')
            results['signals'] = signal_stats
            
            # Print summary
            print(f"\n  ✅ Signal statistics complete:")
            print(f"     Signals analyzed: {len(signal_stats)}")
            print(f"\n     {'Signal':<25} {'Mean':>10} {'Std':>10} {'N samples':>12}")
            print("     " + "-" * 60)
            for name, stats in signal_stats.items():
                print(f"     {name:<25} {stats['mean']:>+10.4f} {stats['std']:>10.4f} {stats['n']:>12,}")
            
            # Save
            output_path = args.output_dir / f'{args.symbol.lower()}_streaming_signals.json'
            with open(output_path, 'w') as f:
                json.dump(signal_stats, f, indent=2)
            print(f"\n     Saved to: {output_path}")
            
            gc.collect()
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            traceback.print_exc()
    else:
        print("\n[3/4] Skipping signal statistics")
    
    # 4. Per-day analysis summary
    print("\n[4/4] Generating per-day summary...")
    try:
        day_summaries = []
        for split in ['train', 'val', 'test']:
            split_dir = args.data_dir / split
            if not split_dir.exists():
                continue
            
            for day in iter_days(args.data_dir, split):
                day_summaries.append({
                    'split': split,
                    'date': day.date,
                    'n_samples': day.n_samples,
                    'n_labels': day.n_labels,
                    'memory_mb': day.memory_bytes / (1024 * 1024),
                })
        
        results['days'] = day_summaries
        print(f"\n  ✅ Processed {len(day_summaries)} days")
        
        # Save
        output_path = args.output_dir / f'{args.symbol.lower()}_streaming_days.json'
        with open(output_path, 'w') as f:
            json.dump(day_summaries, f, indent=2)
        print(f"     Saved to: {output_path}")
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        traceback.print_exc()
    
    # Final summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nDuration: {duration:.1f} seconds")
    
    if args.trace_memory:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"\nMemory usage:")
        print(f"  Current: {format_bytes(current)}")
        print(f"  Peak:    {format_bytes(peak)}")
        
        if peak < 500 * 1024 * 1024:
            print(f"  ✅ Peak memory under 500 MB target")
        else:
            print(f"  ⚠️ Peak memory exceeded 500 MB target")
    
    print(f"\nOutput files saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    import traceback
    main()

