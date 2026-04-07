#!/usr/bin/env python3
"""
Move Magnitude Analysis at Readability Gates.

THE profitability question: How much does the underlying price actually
move in the predicted direction at readability-gated windows?

For OPRA-calibrated option costs of ~$1.40/contract round-trip, we need
the underlying to move at least 2.15 bps ($0.028 on $130 stock) at
delta=0.50 to break even.

This script reads the exported signals (50,724 samples with prices,
predictions, agreement, confirmation, spreads) and computes:

1. Price move distribution for k=1..30 (10-300 events ahead, stride=10)
2. Directional move (in predicted direction) in bps
3. Win rate at various cost thresholds
4. Breakeven horizon analysis

Usage:
    python scripts/analyze_move_magnitudes.py \\
        --signals outputs/experiments/nvda_hmhp_40feat_h10/signals/test/
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_day_boundaries(data_dir: str, split: str) -> list:
    """Load day boundaries from raw export metadata."""
    split_dir = Path(data_dir) / split
    days = []
    for meta_file in sorted(split_dir.glob("*_metadata.json")):
        with open(meta_file) as f:
            meta = json.load(f)
        days.append({"day": meta["day"], "n": meta["n_sequences"]})
    return days


def compute_move_stats(
    prices: np.ndarray,
    predictions: np.ndarray,
    agreement: np.ndarray,
    confirmation: np.ndarray,
    spreads: np.ndarray,
    day_boundaries: list,
    max_k: int = 30,
    min_agreement: float = 1.0,
    min_confidence: float = 0.65,
    max_spread_bps: float = 1.05,
) -> dict:
    """Compute directional move statistics at each horizon k."""

    n = len(prices)

    gate = (
        (agreement >= min_agreement) &
        (confirmation > min_confidence) &
        (spreads <= max_spread_bps) &
        ((predictions == 0) | (predictions == 2))
    )

    gated_indices = np.where(gate)[0]
    position_side = np.where(predictions == 2, 1, -1)

    cumulative = 0
    day_starts = []
    day_ends = []
    for d in day_boundaries:
        day_starts.append(cumulative)
        day_ends.append(cumulative + d["n"] - 1)
        cumulative += d["n"]

    def get_day_end(idx):
        for start, end in zip(day_starts, day_ends):
            if start <= idx <= end:
                return end
        return n - 1

    results = {}

    for k in range(1, max_k + 1):
        horizon_events = k * 10
        valid_moves = []
        raw_moves_bps = []
        entry_prices = []

        for i in gated_indices:
            target = i + k
            day_end = get_day_end(i)
            if target > day_end:
                continue

            entry_price = prices[i]
            future_price = prices[target]

            if entry_price <= 0 or not np.isfinite(entry_price) or not np.isfinite(future_price):
                continue

            raw_move_bps = (future_price - entry_price) / entry_price * 10000.0
            directional_move_bps = raw_move_bps * position_side[i]

            valid_moves.append(directional_move_bps)
            raw_moves_bps.append(raw_move_bps)
            entry_prices.append(entry_price)

        if not valid_moves:
            continue

        moves = np.array(valid_moves)
        raw = np.array(raw_moves_bps)
        prices_arr = np.array(entry_prices)

        avg_price = float(prices_arr.mean())
        breakeven_bps = 1.40 / (0.50 * avg_price / 10000.0 * 100)

        results[horizon_events] = {
            "k": k,
            "horizon_events": horizon_events,
            "n_samples": len(moves),
            "directional_move_bps": {
                "mean": float(moves.mean()),
                "median": float(np.median(moves)),
                "std": float(moves.std()),
                "p10": float(np.percentile(moves, 10)),
                "p25": float(np.percentile(moves, 25)),
                "p75": float(np.percentile(moves, 75)),
                "p90": float(np.percentile(moves, 90)),
                "min": float(moves.min()),
                "max": float(moves.max()),
            },
            "raw_move_bps": {
                "mean": float(raw.mean()),
                "std": float(raw.std()),
                "abs_mean": float(np.abs(raw).mean()),
                "abs_median": float(np.median(np.abs(raw))),
            },
            "win_rates": {
                "move_positive": float((moves > 0).mean()),
                "move_gt_1bps": float((moves > 1.0).mean()),
                "move_gt_2bps": float((moves > 2.0).mean()),
                "move_gt_2.15bps": float((moves > 2.15).mean()),
                "move_gt_3bps": float((moves > 3.0).mean()),
                "move_gt_5bps": float((moves > 5.0).mean()),
                "move_gt_10bps": float((moves > 10.0).mean()),
            },
            "breakeven_bps": round(breakeven_bps, 2),
            "avg_entry_price": round(avg_price, 2),
            "option_pnl_per_contract": {
                "gross_pnl_mean": round(0.50 * moves.mean() * avg_price / 10000.0 * 100, 4),
                "cost_round_trip": 1.40,
                "net_pnl_mean": round(0.50 * moves.mean() * avg_price / 10000.0 * 100 - 1.40, 4),
            },
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Move Magnitude Analysis at Readability Gates")
    parser.add_argument("--signals", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="../data/exports/nvda_xnas_128feat")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max-k", type=int, default=30)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    sig_dir = Path(args.signals)

    print("=" * 70)
    print("  MOVE MAGNITUDE ANALYSIS AT READABILITY GATES")
    print("=" * 70)

    prices = np.load(sig_dir / "prices.npy")
    predictions = np.load(sig_dir / "predictions.npy")
    agreement = np.load(sig_dir / "agreement_ratio.npy")
    confirmation = np.load(sig_dir / "confirmation_score.npy")
    spreads = np.load(sig_dir / "spreads.npy")

    day_boundaries = load_day_boundaries(args.data_dir, args.split)

    print(f"  Samples: {len(prices):,}")
    print(f"  Days: {len(day_boundaries)}")
    print(f"  Price range: ${prices.min():.2f} - ${prices.max():.2f}")

    gate = (agreement >= 1.0) & (confirmation > 0.65) & (spreads <= 1.05) & ((predictions == 0) | (predictions == 2))
    print(f"  Gate pass: {gate.sum():,} ({100*gate.mean():.1f}%)")

    print(f"\n  Computing moves for k=1..{args.max_k} (10-{args.max_k*10} events)...")

    results = compute_move_stats(
        prices, predictions, agreement, confirmation, spreads,
        day_boundaries, max_k=args.max_k,
    )

    print(f"\n  {'Horizon':>8} {'N':>7} {'DirMove':>8} {'Median':>8} {'P75':>8} "
          f"{'Win>0':>7} {'Win>2.15':>8} {'NetPnL':>8} {'BE bps':>7}")
    print(f"  {'-'*8} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*8} {'-'*8} {'-'*7}")

    breakeven_horizon = None
    for h_events in sorted(results.keys()):
        r = results[h_events]
        dm = r["directional_move_bps"]
        wr = r["win_rates"]
        pnl = r["option_pnl_per_contract"]

        is_profitable = pnl["net_pnl_mean"] > 0
        marker = " <<<" if is_profitable and breakeven_horizon is None else ""
        if is_profitable and breakeven_horizon is None:
            breakeven_horizon = h_events

        print(f"  {h_events:>7}e {r['n_samples']:>7,} {dm['mean']:>+7.2f} {dm['median']:>+7.2f} "
              f"{dm['p75']:>+7.2f} {wr['move_positive']:>6.1%} {wr['move_gt_2.15bps']:>7.1%} "
              f"${pnl['net_pnl_mean']:>+7.2f} {r['breakeven_bps']:>6.1f}{marker}")

    print(f"\n{'='*70}")
    print(f"  PROFITABILITY ASSESSMENT (OPRA-calibrated options)")
    print(f"{'='*70}")

    if breakeven_horizon:
        print(f"  BREAKEVEN HORIZON: {breakeven_horizon} events (~{breakeven_horizon/10:.0f} seconds)")
        r = results[breakeven_horizon]
        print(f"  At breakeven: mean dir move = {r['directional_move_bps']['mean']:+.2f} bps")
        print(f"  Net option P&L per contract: ${r['option_pnl_per_contract']['net_pnl_mean']:+.4f}")
        print(f"  Win rate (move > 2.15 bps): {r['win_rates']['move_gt_2.15bps']:.1%}")
    else:
        print(f"  NO PROFITABLE HORIZON FOUND in k=1..{args.max_k}")
        print(f"  The directional edge does not overcome option round-trip cost ($1.40)")
        if 10 in results:
            r10 = results[10]
            print(f"  At H10: mean dir move = {r10['directional_move_bps']['mean']:+.2f} bps")
            print(f"  Breakeven needs: {r10['breakeven_bps']:.2f} bps")

    avg_price = np.mean([r["avg_entry_price"] for r in results.values()])
    print(f"\n  Cost model:")
    print(f"    Avg underlying price: ${avg_price:.2f}")
    print(f"    Option spread (OPRA): $0.03 median (call), $0.02 (put)")
    print(f"    IBKR commission: $0.70/contract")
    print(f"    Round-trip cost: ~$1.40/contract")
    print(f"    Breakeven underlying move: {1.40/(0.50*avg_price/10000*100):.2f} bps")
    print(f"    Delta: 0.50 (ATM)")

    output_path = Path(args.output) if args.output else sig_dir / "move_magnitude_analysis.json"
    output_data = {
        "analysis": "move_magnitude_at_readability_gates",
        "cost_model": {
            "option_spread_call": 0.03,
            "option_spread_put": 0.02,
            "ibkr_commission": 0.70,
            "round_trip_cost": 1.40,
            "delta": 0.50,
            "breakeven_bps": round(1.40 / (0.50 * avg_price / 10000 * 100), 2),
        },
        "breakeven_horizon_events": breakeven_horizon,
        "horizons": {str(k): v for k, v in results.items()},
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
