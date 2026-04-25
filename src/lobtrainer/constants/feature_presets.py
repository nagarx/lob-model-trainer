"""
Feature presets for experiment configuration.

Provides named feature subsets for easy experiment setup:
- LOB-only features (DeepLOB benchmark)
- Full 98-feature set
- Signal-focused subsets
- Custom combinations

**DEPRECATION NOTICE (Phase 4 Batch 4c, 2026-04-15)**: This module is
scheduled for removal in favor of the ``contracts/feature_sets/`` content-
addressed registry. See ``FEATURE_PRESET_DEPRECATION_SCHEDULE`` below for
the timeline; the ``DeprecationWarning`` emitted by
``DataConfig.__post_init__`` cites these dates.

Migration path:
    1. Run ``hft-ops evaluate --config <evaluator.yaml> --criteria <criteria.yaml>
       --save-feature-set <name>_v1 --applies-to-assets NVDA --applies-to-horizons <h>``
       to produce a registry entry.
    2. Replace ``data.feature_preset: <name>`` with ``data.feature_set: <name>_v1``
       in your trainer YAML.
    3. Delete the old preset entry once no configs reference it.

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
    >>> # Use in config — direct field mutation crashes under Pydantic
    >>> # frozen=True (Phase A.5.3i, 2026-04-24 KEYSTONE). Two patterns:
    >>> #
    >>> # (a) Re-fire validators on user-data path (RECOMMENDED for
    >>> #     external callers — catches invalid feature_indices):
    >>> config = ExperimentConfig.model_validate({
    ...     **config.model_dump(),
    ...     "model": {**config.model.model_dump(),
    ...               "feature_indices": get_feature_preset("signals_core")},
    ... })
    >>> #
    >>> # (b) Skip validators (FASTER; safe INSIDE validators only —
    >>> #     state is already validated):
    >>> new_model = config.model.model_copy(
    ...     update={"feature_indices": get_feature_preset("signals_core")}
    ... )
    >>> config = config.model_copy(update={"model": new_model})
"""

# Phase 4 Batch 4c (2026-04-15): single source of truth for the
# feature_preset deprecation schedule. The DataConfig.__post_init__
# DeprecationWarning references these dates. Keeping them in one place
# prevents drift when the schedule slips (change here, ripples to the
# warning message automatically).
FEATURE_PRESET_DEPRECATION_SCHEDULE = {
    "announced":          "2026-04-15",  # DeprecationWarning emitted
    "escalate_to_pending": "2026-06-15",  # PendingDeprecationWarning
    "hard_error_date":    "2026-08-15",  # ImportError on lookup
    "final_delete_date":  "2026-10-15",  # module removal
}
"""Feature preset deprecation schedule (Phase 4 Batch 4c).

Dates are ISO 8601 strings so they match the ledger provenance convention.
See PIPELINE_ARCHITECTURE.md for the full deprecation policy."""

from typing import Dict, List, Tuple, Optional
from lobtrainer.constants.feature_index import FeatureIndex, ExperimentalFeatureIndex


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


# Preset: Short-Term 40 (H10/H20 Optimized)
PRESET_SHORT_TERM_40: Tuple[int, ...] = (
    # =========================================================================
    # L1-L2 Order Book (8 features) - Immediate Liquidity
    # =========================================================================
    FeatureIndex.ASK_PRICE_L0,  # 0
    FeatureIndex.ASK_PRICE_L1,  # 1
    FeatureIndex.ASK_SIZE_L0,   # 10
    FeatureIndex.ASK_SIZE_L1,   # 11
    FeatureIndex.BID_PRICE_L0,  # 20
    FeatureIndex.BID_PRICE_L1,  # 21
    FeatureIndex.BID_SIZE_L0,   # 30
    FeatureIndex.BID_SIZE_L1,   # 31
    
    # =========================================================================
    # Derived Features (7 features) - Book State Summaries
    # =========================================================================
    FeatureIndex.MID_PRICE,         # 40: Reference price
    FeatureIndex.SPREAD,            # 41: Urgency indicator
    FeatureIndex.SPREAD_BPS,        # 42: Normalized spread
    FeatureIndex.TOTAL_BID_VOLUME,  # 43: corr -0.08 @ H10
    FeatureIndex.TOTAL_ASK_VOLUME,  # 44: corr +0.04 @ H10
    FeatureIndex.VOLUME_IMBALANCE,  # 45: corr -0.25 @ H10 *** HIGH ***
    FeatureIndex.WEIGHTED_MID_PRICE,  # 46: Microprice (Stoikov)
    # Note: Excluding PRICE_IMPACT (47) - unsigned, not useful for direction
    
    # =========================================================================
    # Order Flow Dynamics (9 features) - Flow Velocity and Direction
    # =========================================================================
    FeatureIndex.ADD_RATE_BID,    # 48: corr +0.07 @ H10
    FeatureIndex.ADD_RATE_ASK,    # 49: corr -0.07 @ H10
    FeatureIndex.CANCEL_RATE_BID,  # 50
    FeatureIndex.CANCEL_RATE_ASK,  # 51
    FeatureIndex.TRADE_RATE_BID,  # 52: corr -0.14 @ H10 *** MODERATE ***
    FeatureIndex.TRADE_RATE_ASK,  # 53: corr +0.13 @ H10 *** MODERATE ***
    FeatureIndex.NET_ORDER_FLOW,  # 54: corr +0.18 @ H10 *** MODERATE ***
    FeatureIndex.NET_CANCEL_FLOW,  # 55: corr -0.07 @ H10
    FeatureIndex.NET_TRADE_FLOW,  # 56: corr +0.26 @ H10 *** HIGH ***
    
    # =========================================================================
    # Flow Indicators (3 features) - Conviction and Regime
    # =========================================================================
    FeatureIndex.AGGRESSIVE_ORDER_RATIO,  # 57: Conviction indicator
    FeatureIndex.ORDER_FLOW_VOLATILITY,   # 58: Regime indicator
    FeatureIndex.FLOW_REGIME_INDICATOR,   # 59: Acceleration indicator
    
    # =========================================================================
    # Primary Trading Signals (8 features) - Core Predictors
    # =========================================================================
    FeatureIndex.TRUE_OFI,          # 84: corr +0.24 @ H10 *** HIGH ***
    FeatureIndex.DEPTH_NORM_OFI,    # 85: corr +0.29 @ H10 *** BEST ***
    FeatureIndex.EXECUTED_PRESSURE,  # 86: corr +0.20 @ H10 *** HIGH ***
    FeatureIndex.SIGNED_MP_DELTA_BPS,  # 87: Independent signal cluster
    FeatureIndex.TRADE_ASYMMETRY,   # 88: corr +0.26 @ H10 *** HIGH ***
    FeatureIndex.CANCEL_ASYMMETRY,  # 89: corr -0.07 @ H10
    FeatureIndex.FRAGILITY_SCORE,   # 90: Book vulnerability
    FeatureIndex.DEPTH_ASYMMETRY,   # 91: corr +0.06 @ H10
    
    # =========================================================================
    # Safety Gates (3 features) - Data Validity
    # =========================================================================
    FeatureIndex.BOOK_VALID,   # 92: Required - skip if 0
    FeatureIndex.TIME_REGIME,  # 93: Categorical - exclude from normalization
    FeatureIndex.MBO_READY,    # 94: Required - MBO features valid
    
    # =========================================================================
    # Institutional Patience (2 features) - Short-Term Predictive
    # =========================================================================
    # Only experimental features with demonstrated correlation at H10
    ExperimentalFeatureIndex.FILL_PATIENCE_BID,  # 104: corr -0.06 @ H10
    ExperimentalFeatureIndex.FILL_PATIENCE_ASK,  # 105: corr +0.04 @ H10
)
"""
Short-term optimized features (40 features).

Evidence-based selection for H10/H20 prediction horizons from correlation 
analysis: lob-dataset-analyzer/analysis/nvda_116_correlation/horizon_correlations.json

Selection criteria:
- Top 8 trading signals (corr 0.06-0.29 @ H10)
- Order flow dynamics (corr 0.03-0.18 @ H10)
- L1-L2 book microstructure for immediate liquidity
- Key derived features including volume_imbalance (corr -0.25 @ H10)
- Safety gates for data validity
- Institutional patience signals (corr 0.04-0.06 @ H10)

Use case: Short-term prediction experiments (H10, H20 horizons).
Source: Evidence-based feature selection from NVDA 116-feature analysis.
"""

# Validate SHORT_TERM count
assert len(PRESET_SHORT_TERM_40) == 40, (
    f"PRESET_SHORT_TERM_40 should have 40 features, got {len(PRESET_SHORT_TERM_40)}"
)


# Preset: Full 116 features (with experimental, no MLOFI)
PRESET_FULL_116: Tuple[int, ...] = tuple(range(116))
"""
All 116 features (standard + experimental without MLOFI).

Use case: Experiments with experimental groups:
- Standard 98 features
- Institutional V2 (98-105)
- Volatility (106-111)
- Seasonality (112-115)
"""

# Preset: Full 128 features (all experimental including MLOFI)
PRESET_FULL_128: Tuple[int, ...] = tuple(range(128))
"""
All 128 features (standard + all experimental including MLOFI).

Includes everything in PRESET_FULL_116 plus:
- Multi-Level OFI (116-127): total_mlofi, weighted_mlofi, ofi_level_1-10
  Reference: Kolm, Turiel & Westray (2023), R²_OOS ~0.60 vs ~0.10 for L1 OFI
"""

# Preset: Analysis-ready 128 features (dead features excluded)
# Based on FEATURE_SIGNAL_ANALYSIS_REPORT.md Section 1:
# Dead features (zero variance on XNAS): avg_queue_position(68),
# queue_size_ahead(69), modification_score(76), iceberg_proxy(77),
# book_valid(92), mbo_ready(94), invalidity_delta(96),
# schema_version(97), mod_before_cancel(102)
_DEAD_FEATURES = frozenset({68, 69, 76, 77, 92, 94, 96, 97, 102})
PRESET_ANALYSIS_READY_128: Tuple[int, ...] = tuple(
    i for i in range(128) if i not in _DEAD_FEATURES
)
"""
119 features: all 128 minus 9 dead/constant features.

Use case: Primary training preset for 128-feature exports.
Excludes features with zero variance identified by the feature analyzer.
"""


# =============================================================================
# Preset Registry
# =============================================================================

FEATURE_PRESETS: Dict[str, Tuple[int, ...]] = {
    "lob_only": PRESET_LOB_ONLY,
    "lob_derived": PRESET_LOB_DERIVED,
    "full": PRESET_FULL,
    "full_98": PRESET_FULL,  # Alias for clarity
    "full_116": PRESET_FULL_116,
    "full_128": PRESET_FULL_128,
    "analysis_ready_128": PRESET_ANALYSIS_READY_128,
    "signals_core": PRESET_SIGNALS_CORE,
    "signals_full": PRESET_SIGNALS_FULL,
    "lob_signals": PRESET_LOB_SIGNALS,
    "no_meta": PRESET_NO_META,
    "deeplob_extended": PRESET_DEEPLOB_EXTENDED,
    "short_term_40": PRESET_SHORT_TERM_40,
}
"""
Registry of all named presets.

Keys are lowercase identifiers used in configs.
Common presets:
- "short_term_40": Evidence-based 40 features for H10/H20 prediction
- "full_116": All 116 features (standard + experimental)
- "full" or "full_98": Standard 98 features
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
        "full": "All features (98): Complete standard feature set",
        "full_98": "All features (98): Complete standard feature set",
        "full_116": "All features (116): Standard + experimental (institutional, vol, seasonality)",
        "signals_core": "Core signals (8): Most predictive trading signals",
        "signals_full": "All signals (14): Trading signals with validity flags",
        "lob_signals": "LOB + signals (54): Raw LOB + trading signals",
        "no_meta": "No metadata (92): All except validity/schema flags",
        "deeplob_extended": "DeepLOB extended (52): LOB + selected derived + core signals",
        "short_term_40": "Short-term optimized (40): Evidence-based for H10/H20 prediction",
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
