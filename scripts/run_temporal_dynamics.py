#!/usr/bin/env python3
"""
Run temporal dynamics analysis.

Analyzes time-series properties critical for sequence model design:
- Signal autocorrelation (persistence/half-life)
- Lead-lag relationships between signals
- Predictive decay (how signal-label correlation fades with lag)
- Signal level vs change comparison

Answers key questions:
- Are sequence models (LSTM/Transformer) justified?
- What's the optimal lookback window?
- Do signals have temporal structure worth modeling?

Usage:
    python scripts/run_temporal_dynamics.py [--data-dir PATH] [--symbol SYMBOL]

Examples:
    # Analyze NVDA temporal dynamics
    python scripts/run_temporal_dynamics.py --data-dir ../data/exports/nvda_98feat --symbol NVDA
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
    run_temporal_dynamics_analysis,
    print_temporal_dynamics,
    WINDOW_SIZE,
    STRIDE,
)


def main():
    parser = argparse.ArgumentParser(
        description='Run temporal dynamics analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze NVDA temporal dynamics
    python scripts/run_temporal_dynamics.py --data-dir ../data/exports/nvda_98feat --symbol NVDA
    
    # Custom lookback analysis
    python scripts/run_temporal_dynamics.py --data-dir ../data/exports/nvda_98feat --max-acf-lag 200
        """,
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path(__file__).parent.parent.parent / 'data' / 'exports' / 'nvda_98feat',
        help='Path to dataset directory',
    )
    parser.add_argument(
        '--symbol',
        type=str,
        default='NVDA',
        help='Symbol name for labeling',
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'val', 'test'],
        help='Data split to analyze',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(__file__).parent.parent / 'docs',
        help='Output directory for results',
    )
    parser.add_argument(
        '--max-acf-lag',
        type=int,
        default=100,
        help='Maximum lag for autocorrelation analysis',
    )
    parser.add_argument(
        '--max-leadlag-lag',
        type=int,
        default=20,
        help='Maximum lag for lead-lag analysis',
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
    
    output_filename = f"{args.symbol.lower()}_temporal_dynamics.json"
    output_path = args.output_dir / output_filename
    
    print("=" * 80)
    print("TEMPORAL DYNAMICS ANALYSIS")
    print("=" * 80)
    print(f"\nSymbol: {args.symbol}")
    print(f"Data directory: {args.data_dir}")
    print(f"Split: {args.split}")
    print(f"Max ACF lag: {args.max_acf_lag}")
    print(f"Max lead-lag: {args.max_leadlag_lag}")
    print(f"Output: {output_path}")
    
    # Load data
    print(f"\n[1/3] Loading {args.split} data...")
    try:
        data = load_split(args.data_dir, args.split)
        features = data['features']
        labels = data['labels']
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)
    
    print(f"  Features: {features.shape}")
    print(f"  Labels: {labels.shape}")
    
    # Run analysis
    print("\n[2/3] Running temporal dynamics analysis...")
    try:
        summary = run_temporal_dynamics_analysis(
            features=features,
            labels=labels,
            max_acf_lag=args.max_acf_lag,
            max_leadlag_lag=args.max_leadlag_lag,
            window_size=WINDOW_SIZE,
            stride=STRIDE,
        )
    except Exception as e:
        print(f"❌ Error running analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Print results
    if not args.quiet:
        print("\n[3/3] Analysis Results:\n")
        print_temporal_dynamics(summary)
    else:
        print("\n[3/3] Analysis complete (quiet mode)")
    
    # Prepare output data
    output_data = {
        'symbol': args.symbol,
        'data_dir': str(args.data_dir),
        'split': args.split,
        'n_samples': int(features.shape[0]),
        'n_labels': len(labels),
        'max_acf_lag': args.max_acf_lag,
        'max_leadlag_lag': args.max_leadlag_lag,
        **summary.to_dict(),
    }
    
    # Clean numpy types for JSON serialization
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
        save_temporal_figures(summary, args.output_dir, args.symbol)
    
    # Print key findings
    print("\n" + "-" * 40)
    print("KEY FINDINGS")
    print("-" * 40)
    
    print(f"\n1. Sequence Model Justified: {'✅ YES' if summary.sequence_model_justified else '❌ NO'}")
    print(f"   {summary.justification}")
    
    print(f"\n2. Optimal Lookback Window: {summary.optimal_lookback} samples")
    
    most_persistent = max(summary.autocorrelations, key=lambda x: x.half_life)
    print(f"\n3. Most Persistent Signal: {most_persistent.signal_name}")
    print(f"   Half-life: {most_persistent.half_life} samples")
    
    if summary.lead_lag_relations:
        top_leadlag = summary.lead_lag_relations[0]
        print(f"\n4. Strongest Lead-Lag: {top_leadlag.leader} → {top_leadlag.follower}")
        print(f"   Lag: {top_leadlag.optimal_lag} samples, r={top_leadlag.max_correlation:+.3f}")
    
    print("\n" + "=" * 80)
    print("✅ TEMPORAL DYNAMICS ANALYSIS COMPLETE")
    print("=" * 80)


def save_temporal_figures(summary, output_dir: Path, symbol: str):
    """Save visualization figures for temporal dynamics."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️ matplotlib not available, skipping figures")
        return
    
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Autocorrelation plot for top signals
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    top_signals = sorted(summary.autocorrelations, key=lambda x: -x.half_life)[:4]
    
    for ax, s in zip(axes, top_signals):
        lags = s.lags[:50]
        acf = s.acf_values[:50]
        ax.bar(lags, acf, color='steelblue', alpha=0.7, width=0.8)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Half-life threshold')
        ax.axvline(x=s.half_life, color='green', linestyle='--', alpha=0.7, label=f'Half-life={s.half_life}')
        ax.set_xlabel('Lag')
        ax.set_ylabel('Autocorrelation')
        ax.set_title(f'{s.signal_name}')
        ax.legend(loc='upper right', fontsize=8)
    
    plt.suptitle(f'{symbol} Signal Autocorrelation', fontsize=14)
    plt.tight_layout()
    fig.savefig(figures_dir / f'{symbol.lower()}_signal_autocorrelation.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # 2. Predictive decay plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for p in summary.predictive_decays:
        ax.plot(p.lags, p.correlations, marker='o', label=p.signal_name)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Lag (samples before label)')
    ax.set_ylabel('Signal-Label Correlation')
    ax.set_title(f'{symbol} Predictive Decay: How Signal-Label Correlation Fades with Lag')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(figures_dir / f'{symbol.lower()}_predictive_decay.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  📊 Figures saved to: {figures_dir}")


if __name__ == '__main__':
    main()

