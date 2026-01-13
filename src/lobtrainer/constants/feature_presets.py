"""
Feature presets for experiment configuration.

Provides named feature subsets for easy experiment setup:
- LOB-only features (DeepLOB benchmark)
- Full 98-feature set
- Signal-focused subsets
- Custom combinations

Design principles (RULE.md):
- Named presets avoid magic indices in configs
- Presets are immutable (use tuples)
- Clear documentation for each preset's purpose
- Easy to extend with new presets

Usage:
    >>> from lobtrainer.constants.feature_presets import (
    ...     get_feature_preset,
    ...     PRESET_LOB_ONLY,
    ...     PRESET_FULL,
    ... )
    >>> 
    >>> # Get feature indices for a preset
    >>> lob_indices = get_feature_preset("lob_only")
    >>> full_indices = get_feature_preset("full")
    >>> 
    >>> # Use in config
    >>> config.model.feature_indices = get_feature_preset("signals_core")
"""

from typing import Dict, List, Tuple, Optional
from lobtrainer.constants.feature_index import FeatureIndex


# =============================================================================
# Feature Group Definitions (from feature_index.py)
# =============================================================================

# Raw LOB features (40 total): prices and sizes for 10 levels
RAW_LOB_INDICES: Tuple[int, ...] = tuple(range(0, 40))
"""
Raw LOB features: Ask prices (0-9), Ask sizes (10-19), 
Bid prices (20-29), Bid sizes (30-39).
"""

# Derived features (8 total): computed from raw LOB
DERIVED_INDICES: Tuple[int, ...] = tuple(range(40, 48))
"""
Derived features: mid_price, spread, spread_bps, total_bid_volume, 
total_ask_volume, volume_imbalance, weighted_mid_price, price_impact.
"""

# MBO features (36 total): order flow and queue dynamics
MBO_INDICES: Tuple[int, ...] = tuple(range(48, 84))
"""
MBO features: Order flow, size distribution, queue dynamics, 
institutional behavior metrics.
"""

# Trading signals (14 total): aggregated trading indicators
SIGNAL_INDICES: Tuple[int, ...] = tuple(range(84, 98))
"""
Trading signals: TRUE_OFI, DEPTH_NORM_OFI, EXECUTED_PRESSURE, 
SIGNED_MP_DELTA_BPS, TRADE_ASYMMETRY, CANCEL_ASYMMETRY, 
FRAGILITY_SCORE, DEPTH_ASYMMETRY, BOOK_VALID, TIME_REGIME, 
MBO_READY, DT_SECONDS, INVALIDITY_DELTA, SCHEMA_VERSION.
"""


# =============================================================================
# Named Presets
# =============================================================================

# Preset: LOB-only (DeepLOB benchmark)
PRESET_LOB_ONLY: Tuple[int, ...] = RAW_LOB_INDICES
"""
LOB-only features (40 features).

Use case: DeepLOB benchmark mode, comparing with original paper.
Reference: Zhang et al. (2019), "DeepLOB"
"""

# Preset: LOB + Derived (48 features)
PRESET_LOB_DERIVED: Tuple[int, ...] = RAW_LOB_INDICES + DERIVED_INDICES
"""
LOB + derived features (48 features).

Use case: Enhanced LOB-only models with spread, microprice, etc.
"""

# Preset: Full 98 features
PRESET_FULL: Tuple[int, ...] = tuple(range(98))
"""
All 98 features.

Use case: TLOB experiments, full feature experiments.
"""

# Preset: Core trading signals (8 most predictive)
PRESET_SIGNALS_CORE: Tuple[int, ...] = (
    FeatureIndex.TRUE_OFI,           # 84
    FeatureIndex.DEPTH_NORM_OFI,     # 85
    FeatureIndex.EXECUTED_PRESSURE,  # 86
    FeatureIndex.SIGNED_MP_DELTA_BPS,# 87
    FeatureIndex.TRADE_ASYMMETRY,    # 88
    FeatureIndex.CANCEL_ASYMMETRY,   # 89
    FeatureIndex.FRAGILITY_SCORE,    # 90
    FeatureIndex.DEPTH_ASYMMETRY,    # 91
)
"""
Core trading signals (8 features).

These are the most predictive signals based on analysis:
- TRUE_OFI: Best predictor (~0.29 correlation with labels)
- DEPTH_NORM_OFI: Normalized version
- EXECUTED_PRESSURE: Trade execution imbalance
- SIGNED_MP_DELTA_BPS: Mid-price momentum
- TRADE_ASYMMETRY: Buy vs sell trade volume
- CANCEL_ASYMMETRY: Cancel pressure imbalance
- FRAGILITY_SCORE: Book stability measure
- DEPTH_ASYMMETRY: Bid/ask depth imbalance

Use case: Signal-only experiments, feature importance studies.
"""

# Preset: Signals + validity flags (10 features)
PRESET_SIGNALS_FULL: Tuple[int, ...] = SIGNAL_INDICES
"""
All trading signals including validity flags (14 features).

Use case: Full signal experiments with quality checks.
"""

# Preset: LOB + Signals (no MBO)
PRESET_LOB_SIGNALS: Tuple[int, ...] = RAW_LOB_INDICES + SIGNAL_INDICES
"""
LOB + trading signals, no MBO features (54 features).

Use case: When MBO data quality is uncertain.
"""

# Preset: No validity flags (excludes BOOK_VALID, MBO_READY, INVALIDITY_DELTA, SCHEMA_VERSION)
PRESET_NO_META: Tuple[int, ...] = tuple(range(0, 92))
"""
All features except metadata/validity flags (92 features).

Excludes: BOOK_VALID (92), TIME_REGIME (93), MBO_READY (94), 
DT_SECONDS (95), INVALIDITY_DELTA (96), SCHEMA_VERSION (97).

Use case: When you want predictive features only.
"""

# Preset: DeepLOB extended (LOB + key MBO features)
PRESET_DEEPLOB_EXTENDED: Tuple[int, ...] = (
    # Raw LOB (40)
    *RAW_LOB_INDICES,
    # Key derived features (4)
    FeatureIndex.MID_PRICE,
    FeatureIndex.SPREAD,
    FeatureIndex.VOLUME_IMBALANCE,
    FeatureIndex.WEIGHTED_MID_PRICE,
    # Core signals (8)
    *PRESET_SIGNALS_CORE,
)
"""
DeepLOB extended mode (52 features).

LOB + selected derived features + core signals.
Use case: Extended DeepLOB experiments.
"""


# =============================================================================
# Preset Registry
# =============================================================================

FEATURE_PRESETS: Dict[str, Tuple[int, ...]] = {
    "lob_only": PRESET_LOB_ONLY,
    "lob_derived": PRESET_LOB_DERIVED,
    "full": PRESET_FULL,
    "signals_core": PRESET_SIGNALS_CORE,
    "signals_full": PRESET_SIGNALS_FULL,
    "lob_signals": PRESET_LOB_SIGNALS,
    "no_meta": PRESET_NO_META,
    "deeplob_extended": PRESET_DEEPLOB_EXTENDED,
}
"""
Registry of all named presets.

Keys are lowercase identifiers used in configs.
"""


def get_feature_preset(name: str) -> Tuple[int, ...]:
    """
    Get feature indices for a named preset.
    
    Args:
        name: Preset name (case-insensitive).
    
    Returns:
        Tuple of feature indices.
    
    Raises:
        ValueError: If preset name is not recognized.
    
    Example:
        >>> indices = get_feature_preset("lob_only")
        >>> len(indices)
        40
    """
    name_lower = name.lower()
    if name_lower not in FEATURE_PRESETS:
        available = ", ".join(sorted(FEATURE_PRESETS.keys()))
        raise ValueError(
            f"Unknown feature preset: '{name}'. "
            f"Available presets: {available}"
        )
    return FEATURE_PRESETS[name_lower]


def list_presets() -> Dict[str, int]:
    """
    List all available presets with feature counts.
    
    Returns:
        Dict mapping preset name to feature count.
    """
    return {name: len(indices) for name, indices in FEATURE_PRESETS.items()}


def describe_preset(name: str) -> str:
    """
    Get description for a preset.
    
    Args:
        name: Preset name.
    
    Returns:
        Human-readable description.
    """
    descriptions = {
        "lob_only": "Raw LOB features (40): 10 levels × 4 (ask/bid price/size)",
        "lob_derived": "LOB + derived (48): LOB + spread, microprice, etc.",
        "full": "All features (98): Complete feature set",
        "signals_core": "Core signals (8): Most predictive trading signals",
        "signals_full": "All signals (14): Trading signals with validity flags",
        "lob_signals": "LOB + signals (54): Raw LOB + trading signals",
        "no_meta": "No metadata (92): All except validity/schema flags",
        "deeplob_extended": "DeepLOB extended (52): LOB + selected derived + core signals",
    }
    return descriptions.get(name.lower(), "No description available")


def get_preset_summary() -> str:
    """
    Get summary of all presets.
    
    Returns:
        Formatted summary string.
    """
    lines = ["Feature Presets:", ""]
    
    for name, indices in sorted(FEATURE_PRESETS.items()):
        desc = describe_preset(name)
        lines.append(f"  {name}: {len(indices)} features")
        lines.append(f"    {desc}")
        lines.append("")
    
    return "\n".join(lines)
