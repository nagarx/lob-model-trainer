#!/usr/bin/env python3
"""
Pre-compute normalization statistics for a dataset.

This script computes normalization stats using a memory-efficient streaming
algorithm and caches them to JSON files. Subsequent training runs will load
the cached stats instantly, avoiding the need to load all data into memory.

Usage:
    # Compute hybrid stats (recommended for 98-feature datasets)
    python scripts/precompute_norm_stats.py --data-dir data/exports/nvda_98feat
    
    # Compute global Z-score stats (for 40-feature TLOB datasets)
    python scripts/precompute_norm_stats.py --data-dir data/exports/nvda_40feat --mode global
    
    # Force recompute (e.g., after re-exporting data)
    python scripts/precompute_norm_stats.py --data-dir data/exports/nvda_98feat --force

Benefits:
    - Memory efficient: Never loads more than one day's data at a time
    - Fast training: Cached stats load instantly on subsequent runs
    - Reproducible: Same stats used across all training runs
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lobtrainer.data.normalization import (
    compute_hybrid_stats_streaming,
    compute_and_save_normalization_stats,
    HybridNormalizationStats,
    GlobalNormalizationStats,
)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute normalization statistics for LOB training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-dir", "-d",
        type=str,
        required=True,
        help="Path to dataset directory (containing train/val/test splits)",
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["hybrid", "global"],
        default="hybrid",
        help="Normalization mode: 'hybrid' for 98-feature, 'global' for 40-feature (default: hybrid)",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force recompute even if cached stats exist",
    )
    parser.add_argument(
        "--num-features", "-n",
        type=int,
        default=None,
        help="Number of features (auto-detected if not specified)",
    )
    
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)
    
    if not (data_dir / "train").exists():
        print(f"ERROR: No 'train' split found in {data_dir}")
        sys.exit(1)
    
    if args.mode == "hybrid":
        cache_path = data_dir / "hybrid_normalization_stats.json"
        
        if cache_path.exists() and not args.force:
            print(f"Cached stats already exist at {cache_path}")
            print("Use --force to recompute.")
            
            # Show existing stats
            stats = HybridNormalizationStats.load(cache_path)
            print(f"\nExisting stats:")
            print(f"  LOB prices: mean={stats.lob_stats.mean_prices:.4f}, std={stats.lob_stats.std_prices:.4f}")
            print(f"  LOB sizes:  mean={stats.lob_stats.mean_sizes:.2f}, std={stats.lob_stats.std_sizes:.2f}")
            print(f"  Total features: {stats.num_features}")
            print(f"  Excluded indices: {stats.exclude_indices}")
            return
        
        print(f"Computing hybrid normalization stats for {data_dir}...")
        print("This uses streaming (memory-efficient) - may take a few minutes.\n")
        
        stats = compute_hybrid_stats_streaming(
            data_dir,
            num_features=args.num_features,
        )
        
        print(f"\nStats saved to {cache_path}")
        print(f"\nSummary:")
        print(f"  LOB prices: mean={stats.lob_stats.mean_prices:.4f}, std={stats.lob_stats.std_prices:.4f}")
        print(f"  LOB sizes:  mean={stats.lob_stats.mean_sizes:.2f}, std={stats.lob_stats.std_sizes:.2f}")
        print(f"  Total features: {stats.num_features}")
        
    else:  # global
        cache_path = data_dir / "normalization_stats.json"
        
        if cache_path.exists() and not args.force:
            print(f"Cached stats already exist at {cache_path}")
            print("Use --force to recompute.")
            
            stats = GlobalNormalizationStats.load(cache_path)
            print(f"\nExisting stats:")
            print(f"  Prices: mean={stats.mean_prices:.4f}, std={stats.std_prices:.4f}")
            print(f"  Sizes:  mean={stats.mean_sizes:.2f}, std={stats.std_sizes:.2f}")
            return
        
        print(f"Computing global Z-score normalization stats for {data_dir}...")
        
        stats = compute_and_save_normalization_stats(
            data_dir,
            num_features=args.num_features,
        )
        
        print(f"\nStats saved to {cache_path}")
        print(f"\nSummary:")
        print(f"  Prices: mean={stats.mean_prices:.4f}, std={stats.std_prices:.4f}")
        print(f"  Sizes:  mean={stats.mean_sizes:.2f}, std={stats.std_sizes:.2f}")
    
    print("\nDone! Training will now load these cached stats instantly.")


if __name__ == "__main__":
    main()
