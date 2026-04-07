#!/usr/bin/env python3
"""
Feature Importance Analysis Script for Trained Models.

Extracts and visualizes feature importance from trained models that expose
`feature_importance` or `class_coefficients` properties (e.g., LogisticLOB).

Design Principles (RULE.md):
- Works with any model that exposes feature_importance property
- Produces structured JSON output for programmatic analysis
- Maps feature indices to human-readable names
- Supports comparison across different model checkpoints

Usage:
    # Analyze best checkpoint
    python scripts/analyze_feature_importance.py \\
        --checkpoint outputs/logistic_baseline_h10/checkpoints/best.pt \\
        --config outputs/logistic_baseline_h10/config.yaml
    
    # Compare multiple checkpoints
    python scripts/analyze_feature_importance.py \\
        --checkpoint outputs/logistic_baseline_h10/checkpoints/best.pt \\
                     outputs/logistic_signals_h10/checkpoints/best.pt \\
        --config outputs/logistic_baseline_h10/config.yaml
    
    # Output to specific file
    python scripts/analyze_feature_importance.py \\
        --checkpoint outputs/logistic_baseline_h10/checkpoints/best.pt \\
        --config outputs/logistic_baseline_h10/config.yaml \\
        --output analysis_output/feature_importance.json

Output:
    - Console summary of top features per class
    - JSON report with full coefficient matrix and importance scores
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lobtrainer.config import load_config, ExperimentConfig
from lobtrainer.models import create_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Feature Name Mapping (Centralized - same as diagnose_feature_correlations.py)
# =============================================================================

SIGNAL_NAMES = {
    84: "TRUE_OFI",
    85: "DEPTH_NORM_OFI",
    86: "EXECUTED_PRESSURE",
    87: "SIGNED_MP_DELTA_BPS",
    88: "TRADE_ASYMMETRY",
    89: "CANCEL_ASYMMETRY",
    90: "FRAGILITY_SCORE",
    91: "DEPTH_ASYMMETRY",
}

DERIVED_NAMES = {
    40: "MID_PRICE",
    41: "SPREAD",
    42: "SPREAD_BPS",
    43: "TOTAL_BID_VOLUME",
    44: "TOTAL_ASK_VOLUME",
    45: "VOLUME_IMBALANCE",
    46: "WEIGHTED_MID_PRICE",
    47: "PRICE_IMPACT",
}

CLASS_NAMES = {
    0: "Down",
    1: "Stable",
    2: "Up",
}


def get_feature_name(idx: int) -> str:
    """Get human-readable feature name from index."""
    if idx in SIGNAL_NAMES:
        return SIGNAL_NAMES[idx]
    if idx in DERIVED_NAMES:
        return DERIVED_NAMES[idx]
    if 0 <= idx < 10:
        return f"ASK_PRICE_{idx}"
    if 10 <= idx < 20:
        return f"ASK_SIZE_{idx - 10}"
    if 20 <= idx < 30:
        return f"BID_PRICE_{idx - 20}"
    if 30 <= idx < 40:
        return f"BID_SIZE_{idx - 30}"
    if 48 <= idx < 84:
        return f"MBO_{idx - 48}"
    if idx == 92:
        return "BOOK_VALID"
    if idx == 93:
        return "TIME_REGIME"
    if idx == 94:
        return "MBO_READY"
    if idx == 95:
        return "DT_SECONDS"
    if idx == 96:
        return "INVALIDITY_DELTA"
    if idx == 97:
        return "SCHEMA_VERSION"
    return f"FEATURE_{idx}"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class FeatureWeight:
    """Weight information for a single feature."""
    feature_idx: int
    feature_name: str
    importance: float  # Aggregate importance (L1 norm across classes)
    coefficients: Dict[str, float]  # Per-class coefficients
    direction: str  # "bullish", "bearish", "neutral"


@dataclass
class ModelImportanceReport:
    """Feature importance report for a single model."""
    checkpoint_path: str
    model_name: str
    n_features: int
    n_classes: int
    feature_indices_used: List[int]  # For models with feature selection
    
    # Top features by importance
    top_features: List[FeatureWeight]
    
    # Per-class analysis
    bullish_predictors: List[Tuple[str, float]]  # (name, weight for Up class)
    bearish_predictors: List[Tuple[str, float]]  # (name, weight for Down class)
    
    # Raw coefficients for further analysis
    coefficient_matrix: List[List[float]]  # [n_classes, n_features]


# =============================================================================
# Analysis Functions
# =============================================================================

def load_model_from_checkpoint(
    checkpoint_path: Path,
    config: ExperimentConfig,
) -> torch.nn.Module:
    """Load model from checkpoint file."""
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Create model using the single-argument factory function
    # The create_model function extracts all parameters from config.model
    model = create_model(config.model)
    
    # Load state dict
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def analyze_model_importance(
    model: torch.nn.Module,
    checkpoint_path: str,
    feature_indices: Optional[List[int]] = None,
) -> ModelImportanceReport:
    """
    Analyze feature importance from a trained model.
    
    Args:
        model: Trained model with feature_importance property
        checkpoint_path: Path to checkpoint (for reporting)
        feature_indices: Feature indices used by model (for feature selection)
    
    Returns:
        ModelImportanceReport with detailed analysis
    """
    model_name = getattr(model, "name", model.__class__.__name__)
    logger.info(f"Analyzing model: {model_name}")
    
    # Check if model has feature importance
    if not hasattr(model, "feature_importance"):
        raise ValueError(
            f"Model {model_name} does not expose feature_importance property. "
            "This script only works with models that provide interpretable weights."
        )
    
    # Get feature importance and coefficients
    importance = model.feature_importance.detach().cpu().numpy()  # [n_features]
    
    if hasattr(model, "class_coefficients"):
        coefficients = model.class_coefficients.detach().cpu().numpy()  # [n_classes, n_features]
    else:
        coefficients = None
    
    n_features = len(importance)
    n_classes = coefficients.shape[0] if coefficients is not None else 3
    
    # Determine actual feature indices
    if feature_indices is None:
        feature_indices = list(range(n_features))
    
    # Build feature weight list
    feature_weights = []
    for i, (idx, imp) in enumerate(zip(feature_indices, importance)):
        name = get_feature_name(idx)
        
        if coefficients is not None:
            coef_dict = {
                CLASS_NAMES[c]: float(coefficients[c, i])
                for c in range(n_classes)
            }
            
            # Determine direction based on Up vs Down coefficient
            up_coef = coef_dict.get("Up", 0)
            down_coef = coef_dict.get("Down", 0)
            
            if up_coef > down_coef + 0.01:
                direction = "bullish"
            elif down_coef > up_coef + 0.01:
                direction = "bearish"
            else:
                direction = "neutral"
        else:
            coef_dict = {}
            direction = "unknown"
        
        feature_weights.append(FeatureWeight(
            feature_idx=idx,
            feature_name=name,
            importance=float(imp),
            coefficients=coef_dict,
            direction=direction,
        ))
    
    # Sort by importance
    feature_weights.sort(key=lambda x: x.importance, reverse=True)
    
    # Top bullish and bearish predictors
    if coefficients is not None:
        up_idx = 2  # Up class index
        down_idx = 0  # Down class index
        
        bullish = [
            (get_feature_name(feature_indices[i]), float(coefficients[up_idx, i]))
            for i in range(n_features)
        ]
        bullish.sort(key=lambda x: x[1], reverse=True)
        
        bearish = [
            (get_feature_name(feature_indices[i]), float(coefficients[down_idx, i]))
            for i in range(n_features)
        ]
        bearish.sort(key=lambda x: x[1], reverse=True)
    else:
        bullish = []
        bearish = []
    
    return ModelImportanceReport(
        checkpoint_path=str(checkpoint_path),
        model_name=model_name,
        n_features=n_features,
        n_classes=n_classes,
        feature_indices_used=feature_indices,
        top_features=feature_weights[:20],  # Top 20
        bullish_predictors=bullish[:10],
        bearish_predictors=bearish[:10],
        coefficient_matrix=coefficients.tolist() if coefficients is not None else [],
    )


def print_report(report: ModelImportanceReport) -> None:
    """Print human-readable report to console."""
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE ANALYSIS REPORT")
    print("=" * 70)
    print(f"\nCheckpoint: {report.checkpoint_path}")
    print(f"Model: {report.model_name}")
    print(f"Features used: {report.n_features}")
    print(f"Classes: {report.n_classes}")
    
    if report.feature_indices_used != list(range(report.n_features)):
        print(f"Feature indices: {report.feature_indices_used}")
    
    print(f"\n{'─' * 70}")
    print("TOP FEATURES BY IMPORTANCE (L1 norm across classes)")
    print(f"{'─' * 70}")
    
    for i, fw in enumerate(report.top_features[:15]):
        bar = "█" * int(fw.importance * 100)
        direction_icon = "📈" if fw.direction == "bullish" else ("📉" if fw.direction == "bearish" else "➖")
        print(f"{i+1:2d}. {fw.feature_name:25s} {fw.importance:.4f} {direction_icon} {bar}")
    
    if report.bullish_predictors:
        print(f"\n{'─' * 70}")
        print("TOP BULLISH PREDICTORS (highest weight for UP class)")
        print(f"{'─' * 70}")
        for name, weight in report.bullish_predictors[:10]:
            bar = "█" * int(abs(weight) * 50) if weight > 0 else ""
            sign = "+" if weight > 0 else ""
            print(f"  {name:25s} {sign}{weight:.4f} {bar}")
    
    if report.bearish_predictors:
        print(f"\n{'─' * 70}")
        print("TOP BEARISH PREDICTORS (highest weight for DOWN class)")
        print(f"{'─' * 70}")
        for name, weight in report.bearish_predictors[:10]:
            bar = "█" * int(abs(weight) * 50) if weight > 0 else ""
            sign = "+" if weight > 0 else ""
            print(f"  {name:25s} {sign}{weight:.4f} {bar}")
    
    # Interpretation
    print(f"\n{'=' * 70}")
    print("INTERPRETATION")
    print("=" * 70)
    
    if report.top_features:
        top = report.top_features[0]
        print(f"Most important feature: {top.feature_name} (importance={top.importance:.4f})")
        
        # Check if signals are in top features
        signal_in_top5 = [f for f in report.top_features[:5] if f.feature_idx in SIGNAL_NAMES]
        if signal_in_top5:
            print(f"Signals in top 5: {[f.feature_name for f in signal_in_top5]}")
        else:
            print("⚠️  No trading signals in top 5 features")


def save_report(report: ModelImportanceReport, output_path: Path) -> None:
    """Save report to JSON file."""
    # Convert dataclasses to dicts
    def to_dict(obj):
        if hasattr(obj, "__dataclass_fields__"):
            return {k: to_dict(v) for k, v in asdict(obj).items()}
        elif isinstance(obj, list):
            return [to_dict(v) for v in obj]
        elif isinstance(obj, tuple):
            return list(obj)
        else:
            return obj
    
    report_dict = to_dict(report)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report_dict, f, indent=2)
    
    logger.info(f"Report saved to: {output_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze feature importance from trained model checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", "-p",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to model checkpoint file(s)",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to experiment config YAML file (used to recreate model architecture)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSON file path",
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Analyze each checkpoint
    all_reports = []
    for checkpoint_path in args.checkpoint:
        checkpoint_path = Path(checkpoint_path)
        
        # Load model
        model = load_model_from_checkpoint(checkpoint_path, config)
        
        # Get feature indices if model uses feature selection
        feature_indices = None
        if hasattr(model, "_feature_indices") and model._feature_indices is not None:
            feature_indices = model._feature_indices.tolist()
        elif config.model.logistic_feature_indices:
            feature_indices = config.model.logistic_feature_indices
        
        # Analyze
        report = analyze_model_importance(model, str(checkpoint_path), feature_indices)
        all_reports.append(report)
        
        # Print to console
        print_report(report)
    
    # Save to file
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(config.output_dir) / "feature_importance.json"
    
    # Save combined report
    if len(all_reports) == 1:
        save_report(all_reports[0], output_path)
    else:
        # For multiple checkpoints, save as array
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump([asdict(r) for r in all_reports], f, indent=2)
        logger.info(f"Combined report saved to: {output_path}")
    
    print(f"\n✅ Analysis complete. Report saved to: {output_path}")


if __name__ == "__main__":
    main()
