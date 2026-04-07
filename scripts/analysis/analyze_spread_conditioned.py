#!/usr/bin/env python3
"""
Spread-Conditioned Readability Analysis

Combines HMHP model predictions with raw spread data to answer:
"When the model says 'readable' AND the spread is 1 tick,
 what is the directional accuracy?"

This implements the readability-first trading framework:
  Pre-filter:  spread = 1 tick (profiler: OFI r=0.546 at 1-tick vs 0.365 at 2-tick)
  Gate:        agreement_ratio = 1.0 (all horizons agree = readable)
  Confidence:  confirmation_score > 0.65 (decoder confidence high)
  Signal:      predicted direction (Up/Down)

The script loads the trained model, runs inference, and simultaneously
reads raw (pre-normalization) spread values from the export data.

Usage:
    python scripts/analyze_spread_conditioned.py \\
        --experiment outputs/experiments/nvda_hmhp_128feat_xnas_h10 \\
        --splits val test
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'lob-models' / 'src'))

from lobtrainer.config.schema import ExperimentConfig
from lobtrainer.training.trainer import Trainer

try:
    from hft_contracts import SIGNAL_SPREAD_FEATURE_INDEX
    SPREAD_BPS_INDEX = SIGNAL_SPREAD_FEATURE_INDEX
except ImportError:
    SPREAD_BPS_INDEX = 42
TICK_SIZE = 0.01
CLASS_NAMES = {0: "Down", 1: "Stable", 2: "Up"}
DIRECTIONAL_CLASSES = {0, 2}


@torch.no_grad()
def collect_with_spread(
    trainer: Trainer,
    loader: torch.utils.data.DataLoader,
    horizons: List[int],
    data_dir: str,
    split: str,
) -> Dict[str, np.ndarray]:
    """
    Run inference and collect per-sample: predictions, confidence, AND raw spread.

    Raw spread is loaded directly from the .npy files (pre-normalization).
    The model operates on normalized+selected features via the DataLoader.
    We align them by iterating both in the same order.
    """
    model = trainer.model
    model.eval()
    device = trainer.device
    first_h = horizons[0]

    all_final_preds = []
    all_labels = []
    all_agreement = []
    all_confirmation = []
    all_spread_bps = []

    split_dir = Path(data_dir) / split
    day_files = sorted(split_dir.glob("*_sequences.npy"))

    raw_spreads_per_day = []
    for day_file in day_files:
        seqs = np.load(day_file, mmap_mode='r')
        spread_bps = seqs[:, -1, SPREAD_BPS_INDEX]
        raw_spreads_per_day.append(spread_bps)

    raw_spreads_flat = np.concatenate(raw_spreads_per_day)

    sample_idx = 0
    for features, label_dict in loader:
        batch_size = features.size(0)
        features = features.to(device)
        output = model(features)

        final_preds = output.logits.argmax(dim=1).cpu().numpy()
        all_final_preds.append(final_preds)
        all_labels.append(label_dict[first_h].numpy())
        all_agreement.append(output.agreement.squeeze(-1).cpu().numpy())
        all_confirmation.append(output.confidence.squeeze(-1).cpu().numpy())

        sample_idx += batch_size

    return {
        "final_preds": np.concatenate(all_final_preds),
        "labels": np.concatenate(all_labels),
        "agreement": np.concatenate(all_agreement),
        "confirmation": np.concatenate(all_confirmation),
        "spread_bps": raw_spreads_flat[:sample_idx],
    }


def analyze_split(
    data: Dict[str, np.ndarray],
    split_name: str,
    tick_size_bps: float,
) -> Dict:
    """Compute accuracy at various spread + confidence tiers."""

    preds = data["final_preds"]
    labels = data["labels"]
    agreement = data["agreement"]
    confirmation = data["confirmation"]
    spread_bps = data["spread_bps"]
    n = len(preds)

    print(f"\n{'='*70}")
    print(f"  {split_name} split: {n:,} samples")
    print(f"{'='*70}")

    spread_pcts = np.percentile(spread_bps, [10, 25, 50, 75, 90, 95, 99])
    print(f"\n  Spread (bps) percentiles:")
    print(f"    p10={spread_pcts[0]:.2f}  p25={spread_pcts[1]:.2f}  "
          f"p50={spread_pcts[2]:.2f}  p75={spread_pcts[3]:.2f}  "
          f"p90={spread_pcts[4]:.2f}  p95={spread_pcts[5]:.2f}  "
          f"p99={spread_pcts[6]:.2f}")

    one_tick_bps = tick_size_bps
    spread_1tick = spread_bps <= one_tick_bps * 1.05
    spread_2tick = (spread_bps > one_tick_bps * 1.05) & (spread_bps <= one_tick_bps * 2.05)
    spread_wide = spread_bps > one_tick_bps * 2.05

    print(f"\n  Spread distribution:")
    print(f"    1-tick (<=~{one_tick_bps:.1f} bps): {spread_1tick.sum():>8,} ({100*spread_1tick.mean():.1f}%)")
    print(f"    2-tick:                       {spread_2tick.sum():>8,} ({100*spread_2tick.mean():.1f}%)")
    print(f"    Wide (>2 tick):               {spread_wide.sum():>8,} ({100*spread_wide.mean():.1f}%)")

    def compute_metrics(mask, name):
        if mask.sum() == 0:
            return {"filter": name, "n": 0, "rate": 0, "acc": 0, "dir_acc": 0, "dir_n": 0}
        fp = preds[mask]
        lb = labels[mask]
        acc = (fp == lb).mean()
        dir_mask = np.isin(fp, list(DIRECTIONAL_CLASSES)) & np.isin(lb, list(DIRECTIONAL_CLASSES))
        dir_n = dir_mask.sum()
        dir_acc = (fp[dir_mask] == lb[dir_mask]).mean() if dir_n > 0 else 0.0
        return {
            "filter": name,
            "n": int(mask.sum()),
            "rate": float(mask.sum() / n),
            "acc": float(acc),
            "dir_acc": float(dir_acc),
            "dir_n": int(dir_n),
        }

    agree_1 = agreement == 1.0
    high_conf = (agreement == 1.0) & (confirmation > 0.65)
    directional_pred = np.isin(preds, list(DIRECTIONAL_CLASSES))

    filters = [
        ("all_samples",                            np.ones(n, dtype=bool)),
        ("spread=1tick",                           spread_1tick),
        ("spread=2tick",                           spread_2tick),
        ("spread=wide",                            spread_wide),
        ("agree=1.0",                              agree_1),
        ("agree=1.0 + spread=1tick",               agree_1 & spread_1tick),
        ("agree=1.0 + spread=2tick",               agree_1 & spread_2tick),
        ("high_conf",                              high_conf),
        ("high_conf + spread=1tick",               high_conf & spread_1tick),
        ("high_conf + spread=2tick",               high_conf & spread_2tick),
        ("FULL_READABILITY: high_conf+1tick+dir",  high_conf & spread_1tick & directional_pred),
        ("disagree (<1.0)",                        ~agree_1),
        ("disagree + spread=1tick",                ~agree_1 & spread_1tick),
    ]

    results = []
    print(f"\n  {'Filter':<45} {'N':>8} {'Rate':>7} {'Acc':>7} {'DirAcc':>7} {'DirN':>7}")
    print(f"  {'-'*45} {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

    for name, mask in filters:
        m = compute_metrics(mask, name)
        results.append(m)
        print(f"  {name:<45} {m['n']:>8,} {m['rate']:>6.1%} "
              f"{m['acc']:>6.2%} {m['dir_acc']:>6.2%} {m['dir_n']:>7,}")

    return {
        "split": split_name,
        "total_samples": n,
        "tick_size_bps": tick_size_bps,
        "spread_percentiles": {
            f"p{p}": round(float(v), 4)
            for p, v in zip([10, 25, 50, 75, 90, 95, 99], spread_pcts)
        },
        "spread_distribution": {
            "1tick_pct": round(float(spread_1tick.mean()), 4),
            "2tick_pct": round(float(spread_2tick.mean()), 4),
            "wide_pct": round(float(spread_wide.mean()), 4),
        },
        "conditioned_metrics": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Spread-Conditioned Readability Analysis")
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    exp_dir = Path(args.experiment)
    checkpoint_path = exp_dir / "checkpoints" / "best.pt"
    config_path = exp_dir / "config.yaml"

    print("="*70)
    print("  SPREAD-CONDITIONED READABILITY ANALYSIS")
    print("="*70)
    print(f"  Experiment: {exp_dir}")

    config = ExperimentConfig.from_yaml(str(config_path))
    horizons = config.model.hmhp_horizons
    data_dir = config.data.data_dir
    tick_size = 0.01

    symbol_meta_path = Path(data_dir) / "dataset_manifest.json"
    if symbol_meta_path.exists():
        manifest = json.load(open(symbol_meta_path))
        tick_size = manifest.get("tick_size", 0.01)

    tick_size_bps = (tick_size / 100.0) * 10000.0

    print(f"  Tick size: ${tick_size} = {tick_size_bps:.1f} bps (at ~$100 stock)")
    print(f"  Data dir: {data_dir}")
    print(f"  Horizons: {horizons}")

    trainer = Trainer(config)
    trainer.setup()
    trainer.load_checkpoint(checkpoint_path, load_optimizer=False)
    trainer.model.eval()

    all_results = {
        "experiment": str(exp_dir),
        "tick_size": tick_size,
        "tick_size_bps": tick_size_bps,
        "splits": {},
    }

    for split in args.splits:
        loader = getattr(trainer, f"_{split}_loader", None)
        if loader is None:
            print(f"\n  WARNING: No loader for split '{split}', skipping")
            continue
        data = collect_with_spread(trainer, loader, horizons, data_dir, split)
        split_results = analyze_split(data, split, tick_size_bps)
        all_results["splits"][split] = split_results

    output_path = Path(args.output) if args.output else exp_dir / "spread_analysis.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
