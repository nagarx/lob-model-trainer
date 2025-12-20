#!/usr/bin/env python3
"""
Run generalization and robustness analysis.

Tests whether findings generalize across:
- Different trading days (day-to-day variance)
- Time periods (walk-forward validation)

Critical for avoiding overfitting to specific market conditions.

Usage:
    python scripts/run_generalization.py [--data-dir PATH] [--symbol SYMBOL]

Examples:
    # Analyze NVDA generalization
    python scripts/run_generalization.py --data-dir ../data/exports/nvda_98feat --symbol NVDA
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np

from lobtrainer.analysis import (
    run_generalization_analysis,
    print_generalization_analysis,
    WINDOW_SIZE,
    STRIDE,
)


def main():
    parser = argparse.ArgumentParser(
        description='Run generalization and robustness analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze NVDA generalization
    python scripts/run_generalization.py --data-dir ../data/exports/nvda_98feat --symbol NVDA
    
    # Analyze with custom min train days
    python scripts/run_generalization.py --data-dir ../data/exports/nvda_98feat --min-train-days 5
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
    
    output_filename = f"{args.symbol.lower()}_generalization.json"
    output_path = args.output_dir / output_filename
    
    print("=" * 80)
    print("GENERALIZATION & ROBUSTNESS ANALYSIS")
    print("=" * 80)
    print(f"\nSymbol: {args.symbol}")
    print(f"Data directory: {args.data_dir}")
    print(f"Split: {args.split}")
    print(f"Output: {output_path}")
    
    # Run analysis
    print(f"\n[1/2] Running generalization analysis...")
    try:
        summary = run_generalization_analysis(
            data_dir=args.data_dir,
            split=args.split,
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
        print("\n[2/2] Analysis Results:\n")
        print_generalization_analysis(summary)
    else:
        print("\n[2/2] Analysis complete (quiet mode)")
    
    # Prepare output data
    output_data = {
        'symbol': args.symbol,
        'data_dir': str(args.data_dir),
        'split': args.split,
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
        save_generalization_figures(summary, args.output_dir, args.symbol)
    
    # Print key findings
    print("\n" + "-" * 40)
    print("KEY FINDINGS")
    print("-" * 40)
    
    print(f"\n1. Overall Stability Score: {summary.overall_stability_score:.2f}")
    print(f"   Assessment: {summary.generalization_assessment}")
    
    print(f"\n2. Most Stable Signals: {', '.join(summary.most_stable_signals)}")
    print(f"   Least Stable Signals: {', '.join(summary.least_stable_signals)}")
    
    print(f"\n3. Walk-Forward Accuracy: {summary.walk_forward_avg_accuracy:.1%}")
    print(f"   (Baseline would be ~33% for random 3-class)")
    
    print(f"\n4. Days Analyzed: {len(summary.day_statistics)}")
    
    print("\n5. Recommendations:")
    for rec in summary.recommendations[:3]:
        print(f"   • {rec}")
    
    print("\n" + "=" * 80)
    print("✅ GENERALIZATION ANALYSIS COMPLETE")
    print("=" * 80)


def save_generalization_figures(summary, output_dir: Path, symbol: str):
    """Save visualization figures for generalization analysis."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("⚠️ matplotlib/seaborn not available, skipping figures")
        return
    
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Day-by-day signal correlation heatmap
    if summary.signal_day_stats:
        n_signals = len(summary.signal_day_stats)
        n_days = len(summary.signal_day_stats[0].dates)
        
        # Create correlation matrix (signals x days)
        corr_matrix = np.zeros((n_signals, n_days))
        signal_names = []
        dates = summary.signal_day_stats[0].dates
        
        for i, s in enumerate(summary.signal_day_stats):
            corr_matrix[i, :] = s.correlations
            signal_names.append(s.signal_name)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            center=0,
            ax=ax,
            xticklabels=[d[-4:] for d in dates],
            yticklabels=signal_names,
            vmin=-0.1,
            vmax=0.1,
        )
        ax.set_xlabel('Date')
        ax.set_ylabel('Signal')
        ax.set_title(f'{symbol} Signal-Label Correlation by Day')
        
        plt.tight_layout()
        fig.savefig(figures_dir / f'{symbol.lower()}_day_correlations.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    # 2. Walk-forward accuracy plot
    if summary.walk_forward_results:
        fig, ax = plt.subplots(figsize=(12, 5))
        
        dates = [w.test_day for w in summary.walk_forward_results]
        accuracies = [w.prediction_accuracy for w in summary.walk_forward_results]
        
        ax.bar(range(len(dates)), accuracies, color='steelblue', alpha=0.7)
        ax.axhline(y=summary.walk_forward_avg_accuracy, color='red', linestyle='--', 
                   label=f'Average: {summary.walk_forward_avg_accuracy:.1%}')
        ax.axhline(y=0.333, color='gray', linestyle=':', alpha=0.7, label='Random baseline (33%)')
        
        ax.set_xticks(range(len(dates)))
        ax.set_xticklabels([d[-4:] for d in dates], rotation=45, ha='right')
        ax.set_xlabel('Test Day')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{symbol} Walk-Forward Prediction Accuracy')
        ax.legend()
        ax.set_ylim(0, 0.6)
        
        plt.tight_layout()
        fig.savefig(figures_dir / f'{symbol.lower()}_walk_forward.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    # 3. Label distribution by day
    if summary.day_statistics:
        fig, ax = plt.subplots(figsize=(12, 5))
        
        dates = [d.date for d in summary.day_statistics]
        up_pcts = [d.label_up_pct for d in summary.day_statistics]
        down_pcts = [d.label_down_pct for d in summary.day_statistics]
        stable_pcts = [d.label_stable_pct for d in summary.day_statistics]
        
        x = np.arange(len(dates))
        width = 0.25
        
        ax.bar(x - width, down_pcts, width, label='Down', color='#e74c3c')
        ax.bar(x, stable_pcts, width, label='Stable', color='#95a5a6')
        ax.bar(x + width, up_pcts, width, label='Up', color='#27ae60')
        
        ax.set_xticks(x)
        ax.set_xticklabels([d[-4:] for d in dates], rotation=45, ha='right')
        ax.set_xlabel('Date')
        ax.set_ylabel('Percentage')
        ax.set_title(f'{symbol} Label Distribution by Day')
        ax.legend()
        
        plt.tight_layout()
        fig.savefig(figures_dir / f'{symbol.lower()}_label_distribution_by_day.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    print(f"  📊 Figures saved to: {figures_dir}")


if __name__ == '__main__':
    main()

