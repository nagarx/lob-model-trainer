#!/usr/bin/env python3
"""
HMHP Confidence-Conditioned Analysis

Loads a trained HMHP checkpoint, runs inference on val/test splits,
and computes accuracy conditioned on agreement_ratio and confirmation_score
thresholds. This quantifies the "high conviction" trade performance
for the 0DTE ATM options strategy.

Metrics computed at each threshold:
  - Overall accuracy (3-class)
  - Directional accuracy (Up/Down only, excluding Stable)
  - Signal rate (fraction of samples passing the filter)
  - Per-class precision/recall for the filtered subset
  - Per-horizon accuracy within the filtered subset

Usage:
    python scripts/analyze_hmhp_confidence.py \\
        --experiment outputs/experiments/nvda_hmhp_128feat_xnas_h10 \\
        --splits val test
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'lob-models' / 'src'))

from lobtrainer.config.schema import ExperimentConfig
from lobtrainer.training.trainer import Trainer


CLASS_NAMES = {0: "Down", 1: "Stable", 2: "Up"}
DIRECTIONAL_CLASSES = {0, 2}  # Down and Up (exclude Stable=1)


@dataclass
class ConditionedMetrics:
    """Metrics for a subset of samples passing a confidence filter."""
    filter_name: str
    total_samples: int
    filtered_samples: int
    filter_rate: float

    accuracy: float
    directional_accuracy: float
    directional_count: int

    per_class_correct: Dict[int, int] = field(default_factory=dict)
    per_class_total: Dict[int, int] = field(default_factory=dict)
    per_class_predicted: Dict[int, int] = field(default_factory=dict)

    per_horizon_accuracy: Dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        result = {
            "filter": self.filter_name,
            "total_samples": self.total_samples,
            "filtered_samples": self.filtered_samples,
            "filter_rate": round(self.filter_rate, 4),
            "accuracy": round(self.accuracy, 4),
            "directional_accuracy": round(self.directional_accuracy, 4),
            "directional_count": self.directional_count,
        }
        for c in sorted(self.per_class_total.keys()):
            name = CLASS_NAMES.get(c, f"class_{c}").lower()
            total = self.per_class_total[c]
            correct = self.per_class_correct.get(c, 0)
            predicted = self.per_class_predicted.get(c, 0)
            precision = correct / predicted if predicted > 0 else 0.0
            recall = correct / total if total > 0 else 0.0
            result[f"{name}_precision"] = round(precision, 4)
            result[f"{name}_recall"] = round(recall, 4)
            result[f"{name}_support"] = total
            result[f"{name}_predicted"] = predicted
        for h, acc in sorted(self.per_horizon_accuracy.items()):
            result[f"h{h}_accuracy"] = round(acc, 4)
        return result


def compute_conditioned_metrics(
    final_preds: np.ndarray,
    labels: np.ndarray,
    agreement: np.ndarray,
    confirmation: np.ndarray,
    horizon_preds: Dict[int, np.ndarray],
    horizon_labels: Dict[int, np.ndarray],
    filter_name: str,
    mask: np.ndarray,
) -> ConditionedMetrics:
    """Compute metrics for a filtered subset defined by mask."""
    total = len(final_preds)
    filtered = mask.sum()

    if filtered == 0:
        return ConditionedMetrics(
            filter_name=filter_name,
            total_samples=total,
            filtered_samples=0,
            filter_rate=0.0,
            accuracy=0.0,
            directional_accuracy=0.0,
            directional_count=0,
        )

    fp = final_preds[mask]
    lb = labels[mask]

    accuracy = (fp == lb).mean()

    directional_mask = np.isin(fp, list(DIRECTIONAL_CLASSES)) & np.isin(lb, list(DIRECTIONAL_CLASSES))
    directional_count = directional_mask.sum()
    if directional_count > 0:
        directional_accuracy = (fp[directional_mask] == lb[directional_mask]).mean()
    else:
        directional_accuracy = 0.0

    per_class_correct = {}
    per_class_total = {}
    per_class_predicted = {}
    for c in range(3):
        true_c = lb == c
        pred_c = fp == c
        per_class_total[c] = int(true_c.sum())
        per_class_predicted[c] = int(pred_c.sum())
        per_class_correct[c] = int((true_c & pred_c).sum())

    per_horizon_accuracy = {}
    for h in sorted(horizon_preds.keys()):
        hp = horizon_preds[h][mask]
        hl = horizon_labels[h][mask]
        per_horizon_accuracy[h] = float((hp == hl).mean())

    return ConditionedMetrics(
        filter_name=filter_name,
        total_samples=total,
        filtered_samples=int(filtered),
        filter_rate=float(filtered / total),
        accuracy=float(accuracy),
        directional_accuracy=float(directional_accuracy),
        directional_count=int(directional_count),
        per_class_correct=per_class_correct,
        per_class_total=per_class_total,
        per_class_predicted=per_class_predicted,
        per_horizon_accuracy=per_horizon_accuracy,
    )


@torch.no_grad()
def collect_inference_data(
    trainer: Trainer,
    loader: torch.utils.data.DataLoader,
    horizons: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Run inference and collect per-sample metrics.

    Returns:
        final_preds: [N] int array of ensemble predictions
        labels: [N] int array of ground truth (H10)
        agreement: [N] float array of agreement_ratio
        confirmation: [N] float array of confirmation_score
        horizon_preds: {horizon: [N] int array}
        horizon_labels: {horizon: [N] int array}
    """
    model = trainer.model
    model.eval()
    device = trainer.device
    first_h = horizons[0]

    all_final_preds = []
    all_labels = []
    all_agreement = []
    all_confirmation = []
    all_horizon_preds = {h: [] for h in horizons}
    all_horizon_labels = {h: [] for h in horizons}

    for features, label_dict in loader:
        features = features.to(device)
        output = model(features)

        final_preds = output.logits.argmax(dim=1).cpu().numpy()
        all_final_preds.append(final_preds)
        all_labels.append(label_dict[first_h].numpy())
        all_agreement.append(output.agreement.squeeze(-1).cpu().numpy())
        all_confirmation.append(output.confidence.squeeze(-1).cpu().numpy())

        for h in horizons:
            hp = output.horizon_logits[h].argmax(dim=1).cpu().numpy()
            all_horizon_preds[h].append(hp)
            all_horizon_labels[h].append(label_dict[h].numpy())

    return (
        np.concatenate(all_final_preds),
        np.concatenate(all_labels),
        np.concatenate(all_agreement),
        np.concatenate(all_confirmation),
        {h: np.concatenate(all_horizon_preds[h]) for h in horizons},
        {h: np.concatenate(all_horizon_labels[h]) for h in horizons},
    )


def analyze_split(
    trainer: Trainer,
    loader: torch.utils.data.DataLoader,
    horizons: List[int],
    split_name: str,
) -> Dict:
    """Run confidence-conditioned analysis on one data split."""

    print(f"\n{'='*70}")
    print(f"  Analyzing {split_name} split")
    print(f"{'='*70}")

    final_preds, labels, agreement, confirmation, h_preds, h_labels = \
        collect_inference_data(trainer, loader, horizons)

    n = len(final_preds)
    print(f"  Total samples: {n:,}")

    # --- Distribution summaries ---
    unique_agree, agree_counts = np.unique(agreement, return_counts=True)
    print(f"\n  Agreement ratio distribution:")
    for val, cnt in zip(unique_agree, agree_counts):
        print(f"    {val:.4f}: {cnt:>8,} ({100*cnt/n:.1f}%)")

    confirm_pcts = np.percentile(confirmation, [10, 25, 50, 75, 90, 95, 99])
    print(f"\n  Confirmation score percentiles:")
    print(f"    p10={confirm_pcts[0]:.4f}  p25={confirm_pcts[1]:.4f}  "
          f"p50={confirm_pcts[2]:.4f}  p75={confirm_pcts[3]:.4f}  "
          f"p90={confirm_pcts[4]:.4f}  p95={confirm_pcts[5]:.4f}  "
          f"p99={confirm_pcts[6]:.4f}")

    # --- Define filters ---
    filters = [
        ("all_samples",                   np.ones(n, dtype=bool)),
        ("agreement=1.0",                 agreement == 1.0),
        ("agreement>=0.667",              agreement >= 0.667),
        ("agreement<1.0",                 agreement < 1.0),
        ("confirm>0.50",                  confirmation > 0.50),
        ("confirm>0.55",                  confirmation > 0.55),
        ("confirm>0.60",                  confirmation > 0.60),
        ("confirm>0.65",                  confirmation > 0.65),
        ("confirm>0.70",                  confirmation > 0.70),
        ("confirm>0.75",                  confirmation > 0.75),
        ("agree=1.0 AND confirm>0.60",    (agreement == 1.0) & (confirmation > 0.60)),
        ("agree=1.0 AND confirm>0.65",    (agreement == 1.0) & (confirmation > 0.65)),
        ("agree=1.0 AND confirm>0.70",    (agreement == 1.0) & (confirmation > 0.70)),
        ("agree=1.0 AND confirm>0.75",    (agreement == 1.0) & (confirmation > 0.75)),
        ("directional AND agree=1.0",     (agreement == 1.0) & np.isin(final_preds, list(DIRECTIONAL_CLASSES))),
        ("directional AND agree=1.0 AND confirm>0.70",
         (agreement == 1.0) & (confirmation > 0.70) & np.isin(final_preds, list(DIRECTIONAL_CLASSES))),
    ]

    results = []
    print(f"\n  {'Filter':<45} {'N':>8} {'Rate':>7} {'Acc':>7} {'DirAcc':>7} {'H10':>7} {'H60':>7} {'H300':>7}")
    print(f"  {'-'*45} {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

    for name, mask in filters:
        m = compute_conditioned_metrics(
            final_preds, labels, agreement, confirmation,
            h_preds, h_labels, name, mask,
        )
        results.append(m.to_dict())

        h10_acc = m.per_horizon_accuracy.get(horizons[0], 0.0)
        h60_acc = m.per_horizon_accuracy.get(horizons[1], 0.0) if len(horizons) > 1 else 0.0
        h300_acc = m.per_horizon_accuracy.get(horizons[2], 0.0) if len(horizons) > 2 else 0.0

        print(f"  {name:<45} {m.filtered_samples:>8,} {m.filter_rate:>6.1%} "
              f"{m.accuracy:>6.2%} {m.directional_accuracy:>6.2%} "
              f"{h10_acc:>6.2%} {h60_acc:>6.2%} {h300_acc:>6.2%}")

    return {
        "split": split_name,
        "total_samples": n,
        "agreement_distribution": {
            f"{v:.4f}": int(c) for v, c in zip(unique_agree, agree_counts)
        },
        "confirmation_percentiles": {
            f"p{p}": round(float(v), 6)
            for p, v in zip([10, 25, 50, 75, 90, 95, 99], confirm_pcts)
        },
        "conditioned_metrics": results,
    }


def main():
    parser = argparse.ArgumentParser(description="HMHP Confidence-Conditioned Analysis")
    parser.add_argument("--experiment", type=str, required=True,
                        help="Path to experiment output directory")
    parser.add_argument("--splits", nargs="+", default=["val", "test"],
                        help="Data splits to analyze")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: <experiment>/confidence_analysis.json)")
    args = parser.parse_args()

    exp_dir = Path(args.experiment)
    checkpoint_path = exp_dir / "checkpoints" / "best.pt"
    config_path = exp_dir / "config.yaml"

    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        sys.exit(1)

    print("="*70)
    print("  HMHP CONFIDENCE-CONDITIONED ANALYSIS")
    print("="*70)
    print(f"  Experiment: {exp_dir}")
    print(f"  Checkpoint: {checkpoint_path.name}")

    config = ExperimentConfig.from_yaml(str(config_path))
    horizons = config.model.hmhp_horizons

    print(f"  Horizons: {horizons}")
    print(f"  Feature preset: {config.data.feature_preset}")
    print(f"  Splits: {args.splits}")

    trainer = Trainer(config)
    trainer.setup()
    trainer.load_checkpoint(checkpoint_path, load_optimizer=False)
    trainer.model.eval()

    all_results = {
        "experiment": str(exp_dir),
        "checkpoint": checkpoint_path.name,
        "horizons": horizons,
        "splits": {},
    }

    for split in args.splits:
        loader = getattr(trainer, f"_{split}_loader", None)
        if loader is None:
            print(f"\n  WARNING: No loader for split '{split}', skipping")
            continue
        split_results = analyze_split(trainer, loader, horizons, split)
        all_results["splits"][split] = split_results

    output_path = Path(args.output) if args.output else exp_dir / "confidence_analysis.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
