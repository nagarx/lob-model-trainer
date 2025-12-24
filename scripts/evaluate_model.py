#!/usr/bin/env python3
"""
Comprehensive model evaluation script.

Evaluates a trained model with:
- Per-class precision, recall, F1
- Confusion matrix
- Comparison against naive baselines
- Trading-specific metrics

Usage:
    python scripts/evaluate_model.py --checkpoint outputs/baseline_lstm_h10_run1/checkpoints/best.pt
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

# Add parent directory to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lobtrainer import set_seed
from lobtrainer.config import load_config, ExperimentConfig
from lobtrainer.data import load_split_data, LOBSequenceDataset
from lobtrainer.models import create_model
from lobtrainer.training.metrics import (
    compute_classification_report,
    compute_trading_metrics,
    compute_transition_accuracy,
)
from lobtrainer.training.evaluation import evaluate_naive_baseline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Data split to evaluate on (default: test)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save evaluation results (JSON)",
    )
    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device):
    """Load model and config from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config
    config_dict = checkpoint.get('config')
    if config_dict is None:
        raise ValueError("Checkpoint does not contain config")
    
    config = ExperimentConfig.from_dict(config_dict)
    
    # Create model
    model = create_model(config.model)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    
    return model, config


@torch.no_grad()
def get_predictions(model, dataset, device, batch_size=64):
    """Get predictions for entire dataset."""
    from torch.utils.data import DataLoader
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_preds = []
    all_labels = []
    
    for features, labels in loader:
        features = features.to(device)
        outputs = model(features)
        preds = outputs.argmax(dim=1).cpu().numpy()
        
        all_preds.append(preds)
        all_labels.append(labels.numpy())
    
    return np.concatenate(all_preds), np.concatenate(all_labels)


def print_confusion_matrix(cm, labels):
    """Pretty print confusion matrix."""
    label_names = ['Down', 'Stable', 'Up']
    
    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("            " + "".join(f"{name:>8}" for name in label_names))
    print("          +" + "-" * 24)
    
    for i, true_name in enumerate(label_names):
        row = f"True {true_name:>6}|"
        for j in range(len(labels)):
            row += f"{cm[i,j]:>8}"
        print(row)


def main():
    args = parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    model, config = load_model_from_checkpoint(checkpoint_path, device)
    set_seed(config.train.seed)
    
    # Load data
    data_dir = config.data.data_dir
    logger.info(f"Loading {args.split} data from {data_dir}")
    
    days = load_split_data(data_dir, args.split)
    dataset = LOBSequenceDataset(days, horizon_idx=0)
    
    logger.info(f"Dataset: {len(dataset)} samples")
    
    # Get predictions
    logger.info("Running inference...")
    y_pred, y_true = get_predictions(model, dataset, device)
    
    # Compute metrics
    # Note: Labels are in {0, 1, 2} after dataset shift
    metrics = compute_classification_report(y_true, y_pred, labels=[0, 1, 2])
    trading = compute_trading_metrics(y_true, y_pred)
    transitions = compute_transition_accuracy(y_true, y_pred)
    
    # Print results
    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS: {args.split.upper()}")
    print("=" * 60)
    
    print(f"\nModel: {model.name if hasattr(model, 'name') else 'Unknown'}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Samples: {len(y_true):,}")
    
    print("\n" + "-" * 60)
    print("CLASSIFICATION METRICS")
    print("-" * 60)
    
    print(f"\nOverall Accuracy: {metrics.accuracy:.4f} ({metrics.accuracy*100:.2f}%)")
    print(f"Macro F1: {metrics.macro_f1:.4f}")
    print(f"Weighted F1: {metrics.weighted_f1:.4f}")
    
    print("\nPer-Class Metrics:")
    print(f"{'Class':>8} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 50)
    
    label_names = {0: 'Down', 1: 'Stable', 2: 'Up'}
    for pc in metrics.per_class:
        name = label_names.get(pc.label, str(pc.label))
        print(f"{name:>8} {pc.precision:>10.4f} {pc.recall:>10.4f} {pc.f1:>10.4f} {pc.support:>10,}")
    
    print_confusion_matrix(metrics.confusion_matrix, [0, 1, 2])
    
    print("\n" + "-" * 60)
    print("TRADING METRICS")
    print("-" * 60)
    print(f"Directional Accuracy: {trading['directional_accuracy']:.4f}")
    print(f"Up Precision: {trading['up_precision']:.4f}")
    print(f"Down Precision: {trading['down_precision']:.4f}")
    print(f"Signal Rate: {trading['signal_rate']:.4f} ({trading['signal_rate']*100:.2f}%)")
    
    print("\n" + "-" * 60)
    print("TRANSITION ANALYSIS")
    print("-" * 60)
    print(f"Overall Accuracy: {transitions['overall_accuracy']:.4f}")
    print(f"Transition Accuracy: {transitions['transition_accuracy']:.4f}")
    print(f"Stable Accuracy: {transitions['stable_accuracy']:.4f}")
    print(f"Transition Rate: {transitions['transition_rate']:.4f}")
    
    # Baseline comparison
    print("\n" + "-" * 60)
    print("BASELINE COMPARISON")
    print("-" * 60)
    
    # Naive baselines need original {-1, 0, 1} labels
    y_true_orig = y_true - 1  # Shift back to {-1, 0, 1}
    baseline_metrics = evaluate_naive_baseline(y_true_orig, args.split)
    
    print(f"\nClass Prior Baseline: {baseline_metrics['class_prior'].accuracy:.4f}")
    print(f"Previous Label Baseline: {baseline_metrics['previous_label'].accuracy:.4f}")
    print(f"Our Model: {metrics.accuracy:.4f}")
    
    improvement_prior = (metrics.accuracy - baseline_metrics['class_prior'].accuracy) * 100
    improvement_prev = (metrics.accuracy - baseline_metrics['previous_label'].accuracy) * 100
    
    print(f"\nImprovement over Class Prior: {improvement_prior:+.2f}pp")
    print(f"Improvement over Previous Label: {improvement_prev:+.2f}pp")
    
    print("\n" + "=" * 60)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        results = {
            'split': args.split,
            'checkpoint': str(checkpoint_path),
            'n_samples': len(y_true),
            'classification': metrics.to_dict(),
            'trading': trading,
            'transitions': transitions,
            'baselines': {
                'class_prior': baseline_metrics['class_prior'].accuracy,
                'previous_label': baseline_metrics['previous_label'].accuracy,
            },
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()

