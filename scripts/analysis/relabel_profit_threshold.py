#!/usr/bin/env python3
"""
Relabel existing exports with profit-threshold-aligned labels.

Uses existing sequence data (mid_price at feature index 40) to compute
new labels where directional predictions correspond to moves above
the option trading breakeven cost.

The existing TLOB dynamic threshold at H60 is ~5 bps (close to ATM call
breakeven of 4.7 bps). This script creates labels with explicit fixed
thresholds for cleaner cost-aligned experiments.

Approach:
  For each sample i, compute the smoothed return at horizon H:
    price_future = mean(prices[i+1 : i+H/stride+1])
    price_past = mean(prices[i-smooth+1 : i+1])
    return = price_future / price_past - 1
  If return > threshold: UP (1)
  If return < -threshold: DOWN (-1)
  Otherwise: STABLE (0)

Usage:
    python scripts/relabel_profit_threshold.py \\
        --data-dir ../data/exports/nvda_xnas_128feat \\
        --output-dir ../data/exports/nvda_xnas_128feat_profit8bps \\
        --threshold 0.0008 --horizon 60 --smoothing 10
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

MID_PRICE_INDEX = 40
STRIDE = 10


def compute_profit_labels(
    prices: np.ndarray,
    horizon_events: int,
    threshold: float,
    smoothing_window: int,
    n_sequences_day: int,
) -> np.ndarray:
    """
    Compute profit-threshold labels from price series.

    Args:
        prices: Mid-prices for each sample (shape: [N]).
        horizon_events: Number of events into the future (e.g. 60).
        threshold: Absolute return threshold (e.g. 0.0008 = 8 bps).
        smoothing_window: Number of past samples for smoothing.
        n_sequences_day: Number of sequences in this day (for boundary).

    Returns:
        Labels array (shape: [N]), values: -1 (Down), 0 (Stable), 1 (Up).
    """
    n = len(prices)
    labels = np.zeros(n, dtype=np.int8)
    k_forward = horizon_events // STRIDE

    for i in range(n):
        if i + k_forward >= n_sequences_day:
            labels[i] = 0
            continue

        future_end = min(i + k_forward + 1, n_sequences_day)
        future_start = i + 1
        if future_start >= future_end:
            labels[i] = 0
            continue

        past_start = max(i - smoothing_window + 1, 0)
        price_future = prices[future_start:future_end].mean()
        price_past = prices[past_start:i + 1].mean()

        if price_past < 1e-8:
            labels[i] = 0
            continue

        ret = price_future / price_past - 1.0

        if ret > threshold:
            labels[i] = 1
        elif ret < -threshold:
            labels[i] = -1

    return labels


def process_split(
    data_dir: Path,
    output_dir: Path,
    split: str,
    horizon_events: int,
    threshold: float,
    smoothing_window: int,
):
    """Process one data split (train/val/test)."""
    split_in = data_dir / split
    split_out = output_dir / split
    split_out.mkdir(parents=True, exist_ok=True)

    meta_files = sorted(split_in.glob("*_metadata.json"))
    if not meta_files:
        print(f"  No metadata files in {split_in}")
        return

    total_samples = 0
    total_up = 0
    total_down = 0
    total_stable = 0

    for meta_file in meta_files:
        with open(meta_file) as f:
            meta = json.load(f)
        day = meta["day"]
        n_seq = meta["n_sequences"]

        seq_file = split_in / f"{day}_sequences.npy"
        if not seq_file.exists():
            continue

        seqs = np.load(seq_file)
        prices = seqs[:, -1, MID_PRICE_INDEX].astype(np.float64)

        labels = compute_profit_labels(
            prices, horizon_events, threshold, smoothing_window, n_seq,
        )

        np.save(split_out / f"{day}_labels.npy", labels.reshape(-1, 1))

        shutil.copy2(seq_file, split_out / f"{day}_sequences.npy")

        new_meta = meta.copy()
        new_meta["labeling"] = {
            "strategy": "profit_threshold",
            "horizon_events": horizon_events,
            "threshold": threshold,
            "threshold_bps": threshold * 10000,
            "smoothing_window": smoothing_window,
            "label_encoding": {"-1": "Down", "0": "Stable", "1": "Up"},
            "horizons": [horizon_events],
            "num_horizons": 1,
        }
        n_up = int((labels == 1).sum())
        n_down = int((labels == -1).sum())
        n_stable = int((labels == 0).sum())
        new_meta["label_distribution"] = {"Down": n_down, "Stable": n_stable, "Up": n_up}

        with open(split_out / f"{day}_metadata.json", "w") as f:
            json.dump(new_meta, f, indent=2)

        total_samples += n_seq
        total_up += n_up
        total_down += n_down
        total_stable += n_stable

    if total_samples > 0:
        print(f"  {split}: {total_samples:,} samples | "
              f"Down {100*total_down/total_samples:.1f}% | "
              f"Stable {100*total_stable/total_samples:.1f}% | "
              f"Up {100*total_up/total_samples:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Relabel with profit-threshold labels")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.0008,
                        help="Return threshold (0.0008 = 8 bps)")
    parser.add_argument("--horizon", type=int, default=60,
                        help="Horizon in events (60 ≈ 1 minute)")
    parser.add_argument("--smoothing", type=int, default=10)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("  PROFIT-THRESHOLD RELABELING")
    print("=" * 60)
    print(f"  Source: {data_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Threshold: {args.threshold} ({args.threshold*10000:.0f} bps)")
    print(f"  Horizon: {args.horizon} events")
    print(f"  Smoothing: {args.smoothing}")
    print()

    for split in args.splits:
        process_split(data_dir, output_dir, split, args.horizon, args.threshold, args.smoothing)

    print(f"\n  Done. Output at: {output_dir}")


if __name__ == "__main__":
    main()
