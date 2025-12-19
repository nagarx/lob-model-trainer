"""
Constants module for LOB Model Trainer.

Provides exact feature index mappings that match the Rust pipeline export contract.
"""

from lobtrainer.constants.feature_index import (
    FeatureIndex,
    SignalIndex,
    FEATURE_COUNT,
    LOB_FEATURE_COUNT,
    DERIVED_FEATURE_COUNT,
    MBO_FEATURE_COUNT,
    SIGNAL_FEATURE_COUNT,
    SCHEMA_VERSION,
    # Label encoding
    LABEL_DOWN,
    LABEL_STABLE,
    LABEL_UP,
    NUM_CLASSES,
    LABEL_NAMES,
    # Slices for feature groups
    LOB_BID_PRICES,
    LOB_ASK_PRICES,
    LOB_BID_SIZES,
    LOB_ASK_SIZES,
    LOB_ALL,
    DERIVED_ALL,
    MBO_ALL,
    SIGNALS_ALL,
    # Feature groups
    SAFETY_GATES,
    PRIMARY_SIGNALS,
    ASYMMETRY_SIGNALS,
    # Sign convention warnings
    OPPOSITE_SIGN_FEATURES,
    UNSIGNED_FEATURES,
    # Helper functions
    get_corrected_net_trade_flow,
)

__all__ = [
    "FeatureIndex",
    "SignalIndex",
    "FEATURE_COUNT",
    "LOB_FEATURE_COUNT",
    "DERIVED_FEATURE_COUNT",
    "MBO_FEATURE_COUNT",
    "SIGNAL_FEATURE_COUNT",
    "SCHEMA_VERSION",
    # Label encoding
    "LABEL_DOWN",
    "LABEL_STABLE",
    "LABEL_UP",
    "NUM_CLASSES",
    "LABEL_NAMES",
    # Feature slices
    "LOB_BID_PRICES",
    "LOB_ASK_PRICES",
    "LOB_BID_SIZES",
    "LOB_ASK_SIZES",
    "LOB_ALL",
    "DERIVED_ALL",
    "MBO_ALL",
    "SIGNALS_ALL",
    "SAFETY_GATES",
    "PRIMARY_SIGNALS",
    "ASYMMETRY_SIGNALS",
    "OPPOSITE_SIGN_FEATURES",
    "UNSIGNED_FEATURES",
    "get_corrected_net_trade_flow",
]

