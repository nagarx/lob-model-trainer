#!/usr/bin/env python3
"""
Baseline evaluation script for LOB price prediction.

This script establishes the performance floor by evaluating:
1. NaiveClassPrior: Always predict most common class (~37%)
2. NaivePreviousLabel: Predict previous label (~76% due to high ACF)
3. LogisticRegression: Linear baseline with all features

Key insight:
- Label lag-1 ACF = 0.92 (high persistence due to smoothing)
- Any model must significantly beat the previous-label baseline (~76%)
  to demonstrate real predictive value

Usage:
    python scripts/run_baseline_evaluation.py --data-dir ../data/exports/nvda_98feat_full

Output:
    Saves baseline_report.json with full comparison metrics.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lobtrainer.data.dataset import load_numpy_data
from lobtrainer.models.baselines import (
    NaiveClassPrior,
    NaivePreviousLabel,
    LogisticBaseline,
    LogisticBaselineConfig,
)
from lobtrainer.training.metrics import (
    compute_classification_report,
    compute_trading_metrics,
    compute_transition_accuracy,
)
from lobtrainer.training.evaluation import (
    evaluate_naive_baseline,
    create_baseline_report,
)
from lobtrainer.constants import FeatureIndex

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Feature sets for experiments
# =============================================================================

# Tier 1 signals (high correlation + stability from analysis)
TIER1_SIGNALS = [
    FeatureIndex.DEPTH_NORM_OFI,   # 85
    FeatureIndex.TRUE_OFI,         # 84
    FeatureIndex.TRADE_ASYMMETRY,  # 88
    FeatureIndex.EXECUTED_PRESSURE,# 86
]

# All signals
ALL_SIGNALS = list(range(84, 98))  # Indices 84-97

# Safety-gated features (include safety gates)
SAFE_FEATURES = TIER1_SIGNALS + [
    FeatureIndex.BOOK_VALID,       # 92
    FeatureIndex.MBO_READY,        # 94
]


def run_baseline_evaluation(
    data_dir: Path,
    output_dir: Path,
    feature_sets: Dict[str, List[int]],
) -> Dict:
    """
    Run comprehensive baseline evaluation.
    
    Args:
        data_dir: Path to exported data
        output_dir: Directory for output files
        feature_sets: Dict mapping name -> list of feature indices
    
    Returns:
        Full evaluation results
    """
    logger.info("=" * 70)
    logger.info("BASELINE EVALUATION")
    logger.info("=" * 70)
    logger.info(f"Data: {data_dir}")
    logger.info(f"Output: {output_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "data_dir": str(data_dir),
        "splits": {},
    }
    
    # Evaluate on each split
    for split in ["train", "val", "test"]:
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Split: {split}")
        logger.info(f"{'=' * 50}")
        
        try:
            X, y = load_numpy_data(data_dir, split, flat=True)
        except FileNotFoundError:
            logger.warning(f"Split '{split}' not found, skipping")
            continue
        
        logger.info(f"Loaded {len(y)} samples, {X.shape[1]} features")
        
        split_results = {
            "n_samples": len(y),
            "n_features": X.shape[1],
            "baselines": {},
            "logistic": {},
        }
        
        # =====================================================================
        # 1. Naive Baselines
        # =====================================================================
        logger.info("\n[1] Naive Baselines")
        logger.info("-" * 40)
        
        baseline_metrics = evaluate_naive_baseline(y, split)
        
        for name, metrics in baseline_metrics.items():
            split_results["baselines"][name] = {
                "accuracy": metrics.accuracy,
                "macro_f1": metrics.macro_f1,
                "weighted_f1": metrics.weighted_f1,
            }
            logger.info(f"  {name}: accuracy={metrics.accuracy:.4f}, F1={metrics.macro_f1:.4f}")
        
        # =====================================================================
        # 2. Logistic Regression with different feature sets
        # =====================================================================
        logger.info("\n[2] Logistic Regression")
        logger.info("-" * 40)
        
        for feat_name, feat_indices in feature_sets.items():
            logger.info(f"\n  Feature set: {feat_name} ({len(feat_indices)} features)")
            
            # Select features
            X_subset = X[:, feat_indices]
            
            # Train/eval split handling
            if split == "train":
                # For train, we use CV or holdout
                # Here we just train and eval on same data (for baseline)
                X_train, y_train = X_subset, y
                X_eval, y_eval = X_subset, y
            else:
                # For val/test, load train data for fitting
                X_train_full, y_train = load_numpy_data(data_dir, "train", flat=True)
                X_train = X_train_full[:, feat_indices]
                X_eval, y_eval = X_subset, y
            
            # Fit logistic regression
            model = LogisticBaseline(LogisticBaselineConfig(
                C=1.0,
                max_iter=1000,
                class_weight="balanced",
                normalize=True,
            ))
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_eval)
            metrics = compute_classification_report(y_eval, y_pred)
            
            # Create baseline report
            report = create_baseline_report(model, X_eval, y_eval, split)
            
            split_results["logistic"][feat_name] = {
                "n_features": len(feat_indices),
                "accuracy": metrics.accuracy,
                "macro_f1": metrics.macro_f1,
                "weighted_f1": metrics.weighted_f1,
                "beats_class_prior": report.beats_class_prior,
                "beats_previous_label": report.beats_previous_label,
                "improvement_over_prior_pp": report.improvement_over_prior,
                "improvement_over_previous_pp": report.improvement_over_previous,
                "per_class": {pc.name: pc.f1 for pc in metrics.per_class},
            }
            
            status = "✅" if report.beats_previous_label else "❌"
            logger.info(f"    Accuracy: {metrics.accuracy:.4f} {status}")
            logger.info(f"    vs Prior: {report.improvement_over_prior:+.2f}pp")
            logger.info(f"    vs Prev:  {report.improvement_over_previous:+.2f}pp")
        
        # =====================================================================
        # 3. Additional analysis: Transition accuracy
        # =====================================================================
        logger.info("\n[3] Transition Analysis (using best logistic)")
        logger.info("-" * 40)
        
        # Use Tier1 features
        X_tier1 = X[:, TIER1_SIGNALS]
        X_train_full, y_train = load_numpy_data(data_dir, "train", flat=True)
        X_train_tier1 = X_train_full[:, TIER1_SIGNALS]
        
        model = LogisticBaseline()
        model.fit(X_train_tier1, y_train)
        y_pred = model.predict(X_tier1)
        
        transition_metrics = compute_transition_accuracy(y, y_pred)
        split_results["transition_analysis"] = transition_metrics
        
        logger.info(f"  Overall accuracy:    {transition_metrics['overall_accuracy']:.4f}")
        logger.info(f"  Transition accuracy: {transition_metrics['transition_accuracy']:.4f}")
        logger.info(f"  Stable accuracy:     {transition_metrics['stable_accuracy']:.4f}")
        logger.info(f"  Transition rate:     {transition_metrics['transition_rate']:.1%}")
        
        results["splits"][split] = split_results
    
    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    
    # Get test results (primary evaluation)
    if "test" in results["splits"]:
        test = results["splits"]["test"]
        
        logger.info("\nTest Set Results:")
        logger.info(f"  Samples: {test['n_samples']}")
        logger.info(f"\nBaselines:")
        logger.info(f"  Class Prior:    {test['baselines']['class_prior']['accuracy']:.1%}")
        logger.info(f"  Previous Label: {test['baselines']['previous_label']['accuracy']:.1%}")
        
        logger.info(f"\nLogistic Regression:")
        for name, res in test["logistic"].items():
            status = "✅" if res["beats_previous_label"] else "❌"
            logger.info(f"  {name}: {res['accuracy']:.1%} {status}")
    
    # Save results
    output_path = output_dir / "baseline_report.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline evaluation for LOB price prediction"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("../data/exports/nvda_11month_complete"),
        help="Path to exported data directory (default: nvda_11month_complete)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../data/baseline_results"),
        help="Output directory for results",
    )
    
    args = parser.parse_args()
    
    # Define feature sets to evaluate
    feature_sets = {
        "tier1_signals": TIER1_SIGNALS,
        "all_signals": ALL_SIGNALS,
        "all_features": list(range(98)),  # All 98 features
    }
    
    results = run_baseline_evaluation(
        args.data_dir,
        args.output_dir,
        feature_sets,
    )
    
    # Final verdict
    if "test" in results["splits"]:
        test = results["splits"]["test"]
        best_beats = any(
            res.get("beats_previous_label", False)
            for res in test["logistic"].values()
        )
        
        print("\n" + "=" * 70)
        if best_beats:
            print("✅ At least one model beats the previous-label baseline!")
            print("   This suggests real predictive signal in the features.")
        else:
            print("❌ No model beats the previous-label baseline.")
            print("   The features may not have enough predictive power,")
            print("   or the horizon=200 may be too long for these signals.")
        print("=" * 70)


if __name__ == "__main__":
    main()

