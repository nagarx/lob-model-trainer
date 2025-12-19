#!/usr/bin/env python3
"""
Run data overview and validation analysis.

This script performs comprehensive data validation and quality checks:
- File structure validation
- Shape and dtype verification
- NaN/Inf detection
- Label distribution analysis
- Categorical feature validation
- Signal statistics computation

Designed to be reusable across different symbols/datasets.

Usage:
    python scripts/run_data_overview.py [--data-dir PATH] [--symbol SYMBOL] [--output-dir PATH]

Examples:
    # Analyze NVDA dataset
    python scripts/run_data_overview.py --data-dir ../data/exports/nvda_98feat --symbol NVDA
    
    # Analyze a different symbol
    python scripts/run_data_overview.py --data-dir ../data/exports/aapl_98feat --symbol AAPL
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lobtrainer.analysis import (
    generate_dataset_summary,
    print_data_overview,
)


def main():
    parser = argparse.ArgumentParser(
        description='Run data overview and validation analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze NVDA dataset
    python scripts/run_data_overview.py --data-dir ../data/exports/nvda_98feat --symbol NVDA
    
    # Analyze AAPL dataset with custom output
    python scripts/run_data_overview.py --data-dir ../data/exports/aapl_98feat --symbol AAPL --output-dir docs/aapl
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
        '--output-dir',
        type=Path,
        default=Path(__file__).parent.parent / 'docs',
        help='Output directory for results (default: docs/)',
    )
    parser.add_argument(
        '--output-json',
        type=str,
        default=None,
        help='Output JSON filename (default: {symbol}_data_overview.json)',
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
        output_filename = f"{args.symbol.lower()}_data_overview.json"
    else:
        output_filename = args.output_json
    
    output_path = args.output_dir / output_filename
    
    print("=" * 80)
    print("DATA OVERVIEW & VALIDATION ANALYSIS")
    print("=" * 80)
    print(f"\nSymbol: {args.symbol}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output: {output_path}")
    
    # Generate summary
    print("\n[1/2] Generating dataset summary...")
    try:
        summary = generate_dataset_summary(args.data_dir, symbol=args.symbol)
    except Exception as e:
        print(f"❌ Error generating summary: {e}")
        sys.exit(1)
    
    # Print summary
    if not args.quiet:
        print("\n[2/2] Dataset Overview:\n")
        print_data_overview(summary)
    else:
        print("\n[2/2] Summary generated (quiet mode)")
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(summary.to_dict(), f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")
    
    # Print quick summary
    print("\n" + "-" * 40)
    print("QUICK SUMMARY")
    print("-" * 40)
    print(f"  Symbol: {summary.symbol}")
    print(f"  Date range: {summary.date_range[0]} to {summary.date_range[1]}")
    print(f"  Total days: {summary.total_days}")
    print(f"  Total samples: {summary.total_samples:,}")
    print(f"  Total labels: {summary.total_labels:,}")
    print(f"  Feature count: {summary.feature_count}")
    
    # Data quality status
    dq = summary.data_quality
    if dq.is_clean:
        print(f"  Data quality: ✅ Clean (no NaN/Inf)")
    else:
        print(f"  Data quality: ❌ Issues detected")
        if dq.nan_count > 0:
            print(f"    - {dq.nan_count} NaN values")
        if dq.inf_count > 0:
            print(f"    - {dq.inf_count} Inf values")
    
    # Label balance status
    ld = summary.label_distribution
    if ld.is_balanced:
        print(f"  Labels: ✅ Balanced (ratio {ld.imbalance_ratio:.2f})")
    else:
        print(f"  Labels: ⚠️ Imbalanced (ratio {ld.imbalance_ratio:.2f})")
    
    print("\n" + "=" * 80)
    print("✅ DATA OVERVIEW COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()

