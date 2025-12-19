#!/usr/bin/env python3
"""
Run comprehensive label analysis.

This script analyzes label characteristics:
- Distribution and class balance
- Autocorrelation (momentum/clustering detection)
- Transition probabilities (Markov analysis)
- Regime-specific label behavior
- Signal-label correlations

Designed to be reusable across different symbols/datasets.

Usage:
    python scripts/run_label_analysis.py [--data-dir PATH] [--symbol SYMBOL] [--output-dir PATH]

Examples:
    # Analyze NVDA labels
    python scripts/run_label_analysis.py --data-dir ../data/exports/nvda_98feat --symbol NVDA
    
    # Analyze with custom parameters
    python scripts/run_label_analysis.py --data-dir ../data/exports/aapl_98feat --symbol AAPL --max-lag 200
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
    run_label_analysis,
    print_label_analysis,
    WINDOW_SIZE,
    STRIDE,
)


def main():
    parser = argparse.ArgumentParser(
        description='Run comprehensive label analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze NVDA labels
    python scripts/run_label_analysis.py --data-dir ../data/exports/nvda_98feat --symbol NVDA
    
    # Analyze AAPL with extended autocorrelation
    python scripts/run_label_analysis.py --data-dir ../data/exports/aapl_98feat --symbol AAPL --max-lag 200
        """,
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path(__file__).parent.parent.parent / 'data' / 'exports' / 'nvda_98feat',
        help='Path to dataset directory (default: ../data/exports/nvda_98feat)',
    )
    parser.add_argument(
        '--symbol',
        type=str,
        default='NVDA',
        help='Symbol name for labeling (default: NVDA)',
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'val', 'test'],
        help='Data split to analyze (default: train)',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(__file__).parent.parent / 'docs',
        help='Output directory for results (default: docs/)',
    )
    parser.add_argument(
        '--output-json',
        type=str,
        default=None,
        help='Output JSON filename (default: {symbol}_label_analysis.json)',
    )
    parser.add_argument(
        '--window-size',
        type=int,
        default=WINDOW_SIZE,
        help=f'Samples per sequence window (default: {WINDOW_SIZE})',
    )
    parser.add_argument(
        '--stride',
        type=int,
        default=STRIDE,
        help=f'Samples between sequence starts (default: {STRIDE})',
    )
    parser.add_argument(
        '--save-figures',
        action='store_true',
        help='Save visualization figures',
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress detailed console output',
    )
    
    args = parser.parse_args()
    
    # Validate data directory
    if not args.data_dir.exists():
        print(f"❌ Error: Data directory not found: {args.data_dir}")
        sys.exit(1)
    
    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine output filename
    if args.output_json is None:
        output_filename = f"{args.symbol.lower()}_label_analysis.json"
    else:
        output_filename = args.output_json
    
    output_path = args.output_dir / output_filename
    
    print("=" * 80)
    print("COMPREHENSIVE LABEL ANALYSIS")
    print("=" * 80)
    print(f"\nSymbol: {args.symbol}")
    print(f"Data directory: {args.data_dir}")
    print(f"Split: {args.split}")
    print(f"Window size: {args.window_size}")
    print(f"Stride: {args.stride}")
    print(f"Output: {output_path}")
    
    # Load data
    print(f"\n[1/3] Loading {args.split} data...")
    try:
        data = load_split(args.data_dir, args.split)
        features = data['features']
        labels = data['labels']
        n_days = data['n_days']
        dates = data['dates']
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)
    
    print(f"  Features: {features.shape}")
    print(f"  Labels: {labels.shape}")
    print(f"  Days: {n_days}")
    print(f"  Date range: {dates[0]} to {dates[-1]}")
    
    # Run analysis
    print("\n[2/3] Running label analysis...")
    try:
        summary = run_label_analysis(
            features=features,
            labels=labels,
            window_size=args.window_size,
            stride=args.stride,
        )
    except Exception as e:
        print(f"❌ Error running analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Print results
    if not args.quiet:
        print("\n[3/3] Analysis Results:\n")
        print_label_analysis(summary)
    else:
        print("\n[3/3] Analysis complete (quiet mode)")
    
    # Prepare output data
    output_data = {
        'symbol': args.symbol,
        'data_dir': str(args.data_dir),
        'split': args.split,
        'date_range': [dates[0], dates[-1]],
        'n_days': n_days,
        'n_samples': int(features.shape[0]),
        'n_labels': len(labels),
        'window_size': args.window_size,
        'stride': args.stride,
        **summary.to_dict(),
    }
    
    # Helper to clean numpy types for JSON serialization
    def clean_for_json(obj):
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
    
    output_data = clean_for_json(output_data)
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")
    
    # Save figures if requested
    if args.save_figures:
        save_label_figures(summary, args.output_dir, args.symbol)
    
    # Print quick summary
    print("\n" + "-" * 40)
    print("KEY FINDINGS")
    print("-" * 40)
    
    d = summary.distribution
    print(f"\n1. Label Distribution:")
    print(f"   Down: {d.down_pct:.1f}%, Stable: {d.stable_pct:.1f}%, Up: {d.up_pct:.1f}%")
    print(f"   Balance: {'✅ Balanced' if d.is_balanced else '⚠️ Imbalanced'} (ratio {d.imbalance_ratio:.2f})")
    
    a = summary.autocorrelation
    print(f"\n2. Autocorrelation:")
    print(f"   Lag-1 ACF: {a.lag_1_acf:+.4f}")
    print(f"   {a.interpretation}")
    
    if summary.signal_correlations:
        top_signal = summary.signal_correlations[0]
        print(f"\n3. Top Predictor:")
        print(f"   {top_signal.signal_name}: r = {top_signal.correlation:+.4f}")
        if top_signal.is_significant:
            print(f"   ✅ Statistically significant (p < {top_signal.p_value:.2e})")
    
    if summary.regime_stats:
        best_regime = max(summary.regime_stats, key=lambda r: abs(r.ofi_correlation))
        print(f"\n4. Best Regime for OFI:")
        print(f"   {best_regime.name}: r = {best_regime.ofi_correlation:+.4f}")
    
    print("\n" + "=" * 80)
    print("✅ LABEL ANALYSIS COMPLETE")
    print("=" * 80)


def save_label_figures(summary, output_dir: Path, symbol: str):
    """Save visualization figures for label analysis."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("⚠️ matplotlib/seaborn not available, skipping figures")
        return
    
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Label distribution bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    d = summary.distribution
    labels_names = ['Down', 'Stable', 'Up']
    counts = [d.down_count, d.stable_count, d.up_count]
    colors = ['#e74c3c', '#95a5a6', '#27ae60']
    
    bars = ax.bar(labels_names, counts, color=colors, edgecolor='black')
    ax.set_xlabel('Label')
    ax.set_ylabel('Count')
    ax.set_title(f'{symbol} Label Distribution')
    
    for bar, cnt in zip(bars, counts):
        pct = 100 * cnt / d.total
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                f'{cnt:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    fig.savefig(figures_dir / f'{symbol.lower()}_label_distribution.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # 2. Autocorrelation plot
    a = summary.autocorrelation
    fig, ax = plt.subplots(figsize=(14, 5))
    
    lags = a.lags[:min(100, len(a.lags))]
    acf = a.acf_values[:min(100, len(a.acf_values))]
    
    ax.bar(lags, acf, color='steelblue', alpha=0.7, width=0.8)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=a.confidence_interval, color='red', linestyle='--', alpha=0.7, label=f'95% CI (±{a.confidence_interval:.4f})')
    ax.axhline(y=-a.confidence_interval, color='red', linestyle='--', alpha=0.7)
    
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title(f'{symbol} Label Autocorrelation Function (ACF)')
    ax.legend()
    
    plt.tight_layout()
    fig.savefig(figures_dir / f'{symbol.lower()}_label_autocorrelation.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # 3. Transition matrix heatmap
    t = summary.transition_matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    
    from lobtrainer.constants import LABEL_NAMES
    label_names_ordered = [LABEL_NAMES.get(lbl, str(lbl)) for lbl in t.labels]
    probs_array = np.array(t.probabilities)
    
    sns.heatmap(probs_array, annot=True, fmt='.3f', cmap='Blues', ax=ax,
                xticklabels=label_names_ordered, yticklabels=label_names_ordered,
                vmin=0, vmax=1)
    ax.set_xlabel('To')
    ax.set_ylabel('From')
    ax.set_title(f'{symbol} Label Transition Probabilities P(To | From)')
    
    plt.tight_layout()
    fig.savefig(figures_dir / f'{symbol.lower()}_transition_matrix.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  📊 Figures saved to: {figures_dir}")


if __name__ == '__main__':
    main()

