"""
Constants module for LOB Model Trainer (Schema v2.2).

All feature index constants are sourced from the hft-contracts package
(single source of truth). This module re-exports them for backward
compatibility so existing ``from lobtrainer.constants import ...`` imports
continue to work unchanged.

Source of truth: contracts/pipeline_contract.toml
Regenerate with: python contracts/generate_python_contract.py

Sign Convention (RULE.md §9):
    - All directional signals: > 0 = BULLISH, < 0 = BEARISH
    - Exception: PRICE_IMPACT (47) is unsigned

Version History:
    - v2.0: Initial signal layer
    - v2.1: Fixed net_trade_flow (56) and net_cancel_flow (55) sign convention
    - v2.2: Fixed MBO Core feature names (78-82), added experimental features
"""

# Feature presets for experiment configuration
from lobtrainer.constants.feature_presets import (
    PRESET_LOB_ONLY,
    PRESET_LOB_DERIVED,
    PRESET_FULL,
    PRESET_FULL_116,
    PRESET_SHORT_TERM_40,
    PRESET_SIGNALS_CORE,
    PRESET_SIGNALS_FULL,
    PRESET_LOB_SIGNALS,
    PRESET_NO_META,
    PRESET_DEEPLOB_EXTENDED,
    FEATURE_PRESETS,
    get_feature_preset,
    list_presets,
    describe_preset,
    get_preset_summary,
)

from lobtrainer.constants.feature_index import (
    FeatureIndex,
    SignalIndex,
    FEATURE_COUNT,
    STANDARD_FEATURE_COUNT,
    EXTENDED_FEATURE_COUNT,
    LOB_FEATURE_COUNT,
    DERIVED_FEATURE_COUNT,
    MBO_FEATURE_COUNT,
    SIGNAL_FEATURE_COUNT,
    EXPERIMENTAL_FEATURE_COUNT,
    SCHEMA_VERSION,
    # Label encoding (original: {-1, 0, 1})
    LABEL_DOWN,
    LABEL_STABLE,
    LABEL_UP,
    NUM_CLASSES,
    LABEL_NAMES,
    # Label encoding (shifted for PyTorch: {0, 1, 2})
    SHIFTED_LABEL_DOWN,
    SHIFTED_LABEL_STABLE,
    SHIFTED_LABEL_UP,
    SHIFTED_LABEL_NAMES,
    get_label_name,
    # Slices for feature groups
    LOB_BID_PRICES,
    LOB_ASK_PRICES,
    LOB_BID_SIZES,
    LOB_ASK_SIZES,
    LOB_ALL,
    DERIVED_ALL,
    MBO_ALL,
    SIGNALS_ALL,
    # Experimental feature slices (Schema v2.2)
    EXPERIMENTAL_ALL,
    EXPERIMENTAL_INSTITUTIONAL_V2,
    EXPERIMENTAL_VOLATILITY,
    EXPERIMENTAL_SEASONALITY,
    # Feature groups
    SAFETY_GATES,
    PRIMARY_SIGNALS,
    ASYMMETRY_SIGNALS,
    # Sign convention notes
    UNSIGNED_FEATURES,
    # Feature index validation
    validate_feature_indices,
    # Feature layout indices (for normalization)
    GROUPED_PRICE_INDICES,
    GROUPED_SIZE_INDICES,
    LOBSTER_PRICE_INDICES,
    LOBSTER_SIZE_INDICES,
    get_price_size_indices,
    detect_layout_from_metadata,
)

__all__ = [
    "FeatureIndex",
    "SignalIndex",
    "FEATURE_COUNT",
    "STANDARD_FEATURE_COUNT",
    "EXTENDED_FEATURE_COUNT",
    "LOB_FEATURE_COUNT",
    "DERIVED_FEATURE_COUNT",
    "MBO_FEATURE_COUNT",
    "SIGNAL_FEATURE_COUNT",
    "EXPERIMENTAL_FEATURE_COUNT",
    "SCHEMA_VERSION",
    # Label encoding (original: {-1, 0, 1})
    "LABEL_DOWN",
    "LABEL_STABLE",
    "LABEL_UP",
    "NUM_CLASSES",
    "LABEL_NAMES",
    # Label encoding (shifted for PyTorch: {0, 1, 2})
    "SHIFTED_LABEL_DOWN",
    "SHIFTED_LABEL_STABLE",
    "SHIFTED_LABEL_UP",
    "SHIFTED_LABEL_NAMES",
    "get_label_name",
    # Feature slices
    "LOB_BID_PRICES",
    "LOB_ASK_PRICES",
    "LOB_BID_SIZES",
    "LOB_ASK_SIZES",
    "LOB_ALL",
    "DERIVED_ALL",
    "MBO_ALL",
    "SIGNALS_ALL",
    # Experimental feature slices (Schema v2.2)
    "EXPERIMENTAL_ALL",
    "EXPERIMENTAL_INSTITUTIONAL_V2",
    "EXPERIMENTAL_VOLATILITY",
    "EXPERIMENTAL_SEASONALITY",
    # Feature groups
    "SAFETY_GATES",
    "PRIMARY_SIGNALS",
    "ASYMMETRY_SIGNALS",
    "UNSIGNED_FEATURES",
    # Feature index validation
    "validate_feature_indices",
    # Feature layout indices (for normalization)
    "GROUPED_PRICE_INDICES",
    "GROUPED_SIZE_INDICES",
    "LOBSTER_PRICE_INDICES",
    "LOBSTER_SIZE_INDICES",
    "get_price_size_indices",
    "detect_layout_from_metadata",
    # Feature presets (NEW v0.4)
    "PRESET_LOB_ONLY",
    "PRESET_LOB_DERIVED",
    "PRESET_FULL",
    "PRESET_FULL_116",
    "PRESET_SHORT_TERM_40",
    "PRESET_SIGNALS_CORE",
    "PRESET_SIGNALS_FULL",
    "PRESET_LOB_SIGNALS",
    "PRESET_NO_META",
    "PRESET_DEEPLOB_EXTENDED",
    "FEATURE_PRESETS",
    "get_feature_preset",
    "list_presets",
    "describe_preset",
    "get_preset_summary",
]

