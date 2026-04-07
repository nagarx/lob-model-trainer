#!/usr/bin/env python3
"""
Feature-Label Correlation Diagnostic Script.

Computes Pearson correlations between NORMALIZED features and labels
using the EXACT same normalization pipeline as training. This verifies
that predictive signal exists in the actual data fed to models.

Design Principles (RULE.md):
- Reusable across any dataset/config combination
- Uses same normalization as training (no data leakage)
- Reports per-horizon correlations for all feature groups
- Outputs structured JSON for programmatic analysis

Usage:
    # Analyze specific config
    python scripts/diagnose_feature_correlations.py \\
        --config configs/experiments/nvda_logistic_baseline_h10.yaml
    
    # Analyze with custom feature groups
    python scripts/diagnose_feature_correlations.py \\
        --config configs/experiments/nvda_logistic_baseline_h10.yaml \\
        --feature-groups signals derived lob
    
    # Output to specific file
    python scripts/diagnose_feature_correlations.py \\
        --config configs/experiments/nvda_logistic_baseline_h10.yaml \\
        --output analysis_output/correlations.json

Output:
    - Console summary of top predictors per horizon
    - JSON report with all correlations
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import numpy as np
from scipy import stats

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lobtrainer.config import load_config, ExperimentConfig
from lobtrainer.data import load_split_data
from lobtrainer.data.normalization import (
    HybridNormalizer,
    GlobalZScoreNormalizer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Feature Group Definitions (Centralized)
# =============================================================================

FEATURE_GROUPS = {
    "lob_prices": {
        "name": "LOB Prices",
        "indices": list(range(0, 10)) + list(range(20, 30)),  # Ask + Bid prices
        "description": "Raw LOB price levels (20 features)",
    },
    "lob_sizes": {
        "name": "LOB Sizes", 
        "indices": list(range(10, 20)) + list(range(30, 40)),  # Ask + Bid sizes
        "description": "Raw LOB size levels (20 features)",
    },
    "derived": {
        "name": "Derived Features",
        "indices": list(range(40, 48)),
        "description": "Mid-price, spread, volume imbalance, etc. (8 features)",
    },
    "mbo": {
        "name": "MBO Features",
        "indices": list(range(48, 84)),
        "description": "Market-by-Order features (36 features)",
    },
    "signals": {
        "name": "Trading Signals",
        "indices": [84, 85, 86, 87, 88, 89, 90, 91],
        "description": "Core trading signals - OFI, asymmetry, etc. (8 features)",
    },
    "control": {
        "name": "Control Features",
        "indices": [92, 93, 94, 95, 96, 97],
        "description": "Safety gates, time regime, schema version (6 features)",
    },
}

# Individual signal feature names for detailed reporting
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

# Derived feature names
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


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class FeatureCorrelation:
    """Correlation result for a single feature."""
    feature_idx: int
    feature_name: str
    pearson_r: float
    p_value: float
    abs_r: float
    sign: str  # "positive", "negative", "neutral"


@dataclass
class HorizonCorrelations:
    """Correlation results for one horizon."""
    horizon_idx: int
    horizon_samples: int  # e.g., 10, 20, 50, 100
    n_samples: int
    correlations: List[FeatureCorrelation]
    top_positive: List[Tuple[str, float]]
    top_negative: List[Tuple[str, float]]


@dataclass
class CorrelationReport:
    """Complete correlation diagnostic report."""
    config_name: str
    data_dir: str
    n_training_days: int
    n_samples_analyzed: int
    normalization_strategy: str
    horizons: List[HorizonCorrelations]
    summary: Dict[str, any]


# =============================================================================
# Core Analysis Functions
# =============================================================================

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


def compute_feature_correlations(
    features: np.ndarray,
    labels: np.ndarray,
    feature_indices: List[int],
) -> List[FeatureCorrelation]:
    """
    Compute Pearson correlation between each feature and labels.
    
    Args:
        features: Normalized features [n_samples, n_features]
        labels: Integer labels [n_samples]
        feature_indices: Which feature indices to analyze
    
    Returns:
        List of FeatureCorrelation results sorted by |r|
    """
    results = []
    
    # Convert labels to numeric for correlation
    labels_numeric = labels.astype(np.float64)
    
    for idx in feature_indices:
        if idx >= features.shape[1]:
            continue
            
        feature_values = features[:, idx]
        
        # Skip if feature is constant (no variance)
        if np.std(feature_values) < 1e-10:
            results.append(FeatureCorrelation(
                feature_idx=idx,
                feature_name=get_feature_name(idx),
                pearson_r=0.0,
                p_value=1.0,
                abs_r=0.0,
                sign="neutral",
            ))
            continue
        
        # Compute Pearson correlation
        r, p = stats.pearsonr(feature_values, labels_numeric)
        
        if np.isnan(r):
            r = 0.0
            p = 1.0
        
        sign = "positive" if r > 0.01 else ("negative" if r < -0.01 else "neutral")
        
        results.append(FeatureCorrelation(
            feature_idx=idx,
            feature_name=get_feature_name(idx),
            pearson_r=float(r),
            p_value=float(p),
            abs_r=float(abs(r)),
            sign=sign,
        ))
    
    # Sort by absolute correlation (strongest first)
    results.sort(key=lambda x: x.abs_r, reverse=True)
    return results


def analyze_horizon(
    train_days: List,
    normalizer,
    horizon_idx: int,
    feature_indices: List[int],
    max_samples: int = 100000,
) -> HorizonCorrelations:
    """
    Analyze feature-label correlations for a specific horizon.
    
    Args:
        train_days: List of DayData objects
        normalizer: Fitted normalizer (HybridNormalizer or GlobalZScoreNormalizer)
        horizon_idx: Which horizon to analyze
        feature_indices: Which features to analyze
        max_samples: Maximum samples to use (for speed)
    
    Returns:
        HorizonCorrelations with all results
    """
    logger.info(f"  Analyzing horizon index {horizon_idx}...")
    
    # Collect samples
    all_features = []
    all_labels = []
    
    for day in train_days:
        if day.sequences is None:
            continue
        
        # Get last timestep features (matching 'last' pooling)
        features = day.sequences[:, -1, :]  # [n_seq, n_features]
        
        # Get labels for this horizon
        if day.is_multi_horizon:
            labels = day.labels[:, horizon_idx]
        else:
            labels = day.labels
        
        all_features.append(features)
        all_labels.append(labels)
    
    # Concatenate
    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    # Subsample if too large
    if len(features) > max_samples:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(features), max_samples, replace=False)
        features = features[indices]
        labels = labels[indices]
    
    logger.info(f"    Samples: {len(features)}")
    
    # Apply normalization (same as training)
    # HybridNormalizer and GlobalZScoreNormalizer use __call__ interface
    features_normalized = normalizer(features)
    
    # Compute correlations
    correlations = compute_feature_correlations(
        features_normalized, labels, feature_indices
    )
    
    # Get top positive and negative
    top_positive = [
        (c.feature_name, c.pearson_r)
        for c in correlations if c.sign == "positive"
    ][:5]
    
    top_negative = [
        (c.feature_name, c.pearson_r)
        for c in correlations if c.sign == "negative"
    ][:5]
    
    # Determine horizon samples from manifest if available
    horizon_samples = [10, 20, 50, 100][horizon_idx] if horizon_idx < 4 else horizon_idx
    
    return HorizonCorrelations(
        horizon_idx=horizon_idx,
        horizon_samples=horizon_samples,
        n_samples=len(features),
        correlations=correlations,
        top_positive=top_positive,
        top_negative=top_negative,
    )


# =============================================================================
# Main Analysis
# =============================================================================

def run_correlation_diagnostic(
    config: ExperimentConfig,
    feature_groups: Optional[List[str]] = None,
    max_samples: int = 100000,
) -> CorrelationReport:
    """
    Run complete correlation diagnostic.
    
    Args:
        config: Experiment configuration
        feature_groups: Which feature groups to analyze (default: all)
        max_samples: Maximum samples per horizon
    
    Returns:
        CorrelationReport with all results
    """
    logger.info(f"Starting correlation diagnostic for: {config.name}")
    logger.info(f"Data directory: {config.data.data_dir}")
    
    # Determine feature indices to analyze
    if feature_groups is None:
        feature_groups = ["signals", "derived", "mbo"]  # Skip LOB and control by default
    
    feature_indices = []
    for group_name in feature_groups:
        if group_name in FEATURE_GROUPS:
            feature_indices.extend(FEATURE_GROUPS[group_name]["indices"])
    feature_indices = sorted(set(feature_indices))
    
    logger.info(f"Analyzing {len(feature_indices)} features from groups: {feature_groups}")
    
    # Load training data
    data_path = Path(config.data.data_dir)
    logger.info("Loading training data...")
    train_days = load_split_data(data_path, "train")
    logger.info(f"Loaded {len(train_days)} training days")
    
    # Create and fit normalizer (same as training)
    logger.info("Fitting normalizer on training data...")
    norm_strategy = config.data.normalization.strategy.value
    exclude_features = tuple(config.data.normalization.exclude_features)
    
    if norm_strategy == "hybrid":
        normalizer = HybridNormalizer.from_train_data(
            train_days,
            layout="grouped",
            num_features=config.data.feature_count,
            exclude_indices=exclude_features,
            eps=config.data.normalization.eps,
            clip_value=config.data.normalization.clip_value,
        )
    elif norm_strategy == "global_zscore":
        normalizer = GlobalZScoreNormalizer.from_train_data(
            train_days,
            layout="grouped",
            num_features=config.data.feature_count,
            eps=config.data.normalization.eps,
            clip_value=config.data.normalization.clip_value,
        )
    else:
        raise ValueError(f"Unsupported normalization strategy: {norm_strategy}")
    
    # Determine number of horizons
    sample_day = train_days[0]
    n_horizons = sample_day.num_horizons if sample_day.is_multi_horizon else 1
    logger.info(f"Dataset has {n_horizons} horizons")
    
    # Analyze each horizon
    horizon_results = []
    for h_idx in range(n_horizons):
        result = analyze_horizon(
            train_days, normalizer, h_idx, feature_indices, max_samples
        )
        horizon_results.append(result)
    
    # Compute summary
    total_samples = sum(h.n_samples for h in horizon_results)
    
    # Find best predictor across all horizons
    best_overall = None
    best_r = 0.0
    for h in horizon_results:
        for c in h.correlations:
            if c.abs_r > best_r:
                best_r = c.abs_r
                best_overall = (c.feature_name, c.pearson_r, h.horizon_idx)
    
    summary = {
        "best_predictor": best_overall[0] if best_overall else None,
        "best_correlation": best_overall[1] if best_overall else None,
        "best_horizon": best_overall[2] if best_overall else None,
        "signal_correlations_h0": {
            c.feature_name: c.pearson_r
            for c in horizon_results[0].correlations
            if c.feature_idx in SIGNAL_NAMES
        } if horizon_results else {},
    }
    
    return CorrelationReport(
        config_name=config.name,
        data_dir=str(config.data.data_dir),
        n_training_days=len(train_days),
        n_samples_analyzed=total_samples,
        normalization_strategy=norm_strategy,
        horizons=horizon_results,
        summary=summary,
    )


def print_report(report: CorrelationReport) -> None:
    """Print human-readable report to console."""
    print("\n" + "=" * 70)
    print("FEATURE-LABEL CORRELATION DIAGNOSTIC REPORT")
    print("=" * 70)
    print(f"\nConfig: {report.config_name}")
    print(f"Data: {report.data_dir}")
    print(f"Training days: {report.n_training_days}")
    print(f"Normalization: {report.normalization_strategy}")
    print(f"Total samples: {report.n_samples_analyzed:,}")
    
    for h in report.horizons:
        print(f"\n{'─' * 70}")
        print(f"HORIZON {h.horizon_idx} (H={h.horizon_samples} samples)")
        print(f"{'─' * 70}")
        print(f"Samples analyzed: {h.n_samples:,}")
        
        print("\n📈 Top POSITIVE correlations:")
        for name, r in h.top_positive[:5]:
            bar = "█" * int(abs(r) * 50)
            print(f"  {name:25s} r = {r:+.4f} {bar}")
        
        print("\n📉 Top NEGATIVE correlations:")
        for name, r in h.top_negative[:5]:
            bar = "█" * int(abs(r) * 50)
            print(f"  {name:25s} r = {r:+.4f} {bar}")
        
        # Signal-specific summary
        signal_corrs = [c for c in h.correlations if c.feature_idx in SIGNAL_NAMES]
        if signal_corrs:
            print("\n🎯 Trading Signals Summary:")
            for c in signal_corrs:
                significance = "***" if c.p_value < 0.001 else ("**" if c.p_value < 0.01 else ("*" if c.p_value < 0.05 else ""))
                print(f"  {c.feature_name:25s} r = {c.pearson_r:+.4f} {significance}")
    
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    if report.summary["best_predictor"]:
        print(f"Best predictor: {report.summary['best_predictor']} "
              f"(r={report.summary['best_correlation']:+.4f} at H{report.summary['best_horizon']})")
    
    # Interpretation
    best_r = abs(report.summary.get("best_correlation", 0) or 0)
    if best_r < 0.1:
        print("\n⚠️  WARNING: Very weak correlations (<0.1). Features may have minimal predictive power.")
    elif best_r < 0.2:
        print("\n⚡ Moderate correlations (0.1-0.2). Some signal present but weak.")
    else:
        print("\n✅ Good correlations (>0.2). Features have meaningful predictive power.")


def save_report(report: CorrelationReport, output_path: Path) -> None:
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
        description="Diagnose feature-label correlations using normalized training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to experiment config YAML file",
    )
    parser.add_argument(
        "--feature-groups", "-g",
        type=str,
        nargs="+",
        default=["signals", "derived"],
        choices=list(FEATURE_GROUPS.keys()),
        help="Feature groups to analyze (default: signals derived)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSON file path (default: outputs/<config_name>/correlation_diagnostic.json)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100000,
        help="Maximum samples per horizon (default: 100000)",
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Run diagnostic
    report = run_correlation_diagnostic(
        config,
        feature_groups=args.feature_groups,
        max_samples=args.max_samples,
    )
    
    # Print to console
    print_report(report)
    
    # Save to file
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(config.output_dir) / "correlation_diagnostic.json"
    
    save_report(report, output_path)
    
    print(f"\n✅ Diagnostic complete. Report saved to: {output_path}")


if __name__ == "__main__":
    main()
