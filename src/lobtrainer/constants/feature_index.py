"""
Feature index contract for the LOB Model Trainer.

This module re-exports all shared contract constants from the hft_contracts
package (the single source of truth) and adds trainer-specific helpers for
normalization layout detection.

Source of truth: contracts/pipeline_contract.toml
Regenerate with: python contracts/generate_python_contract.py

Schema Version History:
    - v2.0: Initial signal layer
    - v2.1: Fixed net_trade_flow (56) and net_cancel_flow (55) sign convention
    - v2.2: Fixed MBO Core feature names (78-82) to match Rust implementation;
            added experimental features (institutional_v2, volatility, seasonality)
"""

from typing import Final, Tuple

# =========================================================================
# Re-export everything from the shared contract package.
# Consumers importing from `lobtrainer.constants.feature_index` continue
# to work identically — this module is a backward-compatible shim.
# =========================================================================

from hft_contracts import (
    # Schema
    SCHEMA_VERSION,
    SCHEMA_VERSION_FLOAT,
    # Feature counts
    LOB_FEATURE_COUNT,
    DERIVED_FEATURE_COUNT,
    MBO_FEATURE_COUNT,
    SIGNAL_FEATURE_COUNT,
    EXPERIMENTAL_FEATURE_COUNT,
    FEATURE_COUNT,
    STANDARD_FEATURE_COUNT,
    EXTENDED_FEATURE_COUNT,
    FULL_FEATURE_COUNT,
    FEATURE_COUNT_WITH_EXPERIMENTAL,
    # Enums
    FeatureIndex,
    ExperimentalFeatureIndex,
    SignalIndex,
    # Slices
    LOB_ASK_PRICES,
    LOB_ASK_SIZES,
    LOB_BID_PRICES,
    LOB_BID_SIZES,
    LOB_ALL,
    DERIVED_ALL,
    MBO_ALL,
    SIGNALS_ALL,
    EXPERIMENTAL_ALL,
    EXPERIMENTAL_INSTITUTIONAL_V2,
    EXPERIMENTAL_VOLATILITY,
    EXPERIMENTAL_SEASONALITY,
    # Layout index tuples
    GROUPED_PRICE_INDICES,
    GROUPED_SIZE_INDICES,
    LOBSTER_PRICE_INDICES,
    LOBSTER_SIZE_INDICES,
    # Classification sets
    CATEGORICAL_INDICES,
    UNSIGNED_FEATURES,
    SAFETY_GATES,
    PRIMARY_SIGNALS,
    ASYMMETRY_SIGNALS,
    SIGNAL_NAMES,
    EXPERIMENTAL_FEATURE_NAMES,
    # Labels
    LABEL_DOWN,
    LABEL_STABLE,
    LABEL_UP,
    NUM_CLASSES,
    LABEL_NAMES,
    SHIFTED_LABEL_DOWN,
    SHIFTED_LABEL_STABLE,
    SHIFTED_LABEL_UP,
    SHIFTED_LABEL_NAMES,
    get_label_name,
    # Validation
    validate_feature_indices,
)


# =========================================================================
# Trainer-Specific Helpers (layout detection for normalization)
# =========================================================================


def get_price_size_indices(
    layout: str,
    num_levels: int = 10,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """
    Get price and size column indices for a given feature layout.

    This is the authoritative function for determining which columns contain
    prices vs sizes based on the feature layout. Used by normalization code.

    Args:
        layout: Feature layout identifier. Supported values:
            - "grouped": Our Rust pipeline format [ask_p, ask_s, bid_p, bid_s]
            - "lobster", "fi2010", "interleaved": LOBSTER per-level format
            - Auto-detection from metadata string containing layout info
        num_levels: Number of LOB levels (default 10)

    Returns:
        Tuple of (price_indices, size_indices) as tuples of ints.

    Raises:
        ValueError: If layout is not recognized.

    Data Contract:
        - GROUPED: [ask_prices(L), ask_sizes(L), bid_prices(L), bid_sizes(L)]
        - LOBSTER: [ask_p_L0, ask_s_L0, bid_p_L0, bid_s_L0, ..., per level]
    """
    layout_lower = layout.lower().strip()

    if layout_lower in ("grouped", "extended"):
        return GROUPED_PRICE_INDICES, GROUPED_SIZE_INDICES

    if layout_lower in ("lobster", "fi2010", "interleaved"):
        return LOBSTER_PRICE_INDICES, LOBSTER_SIZE_INDICES

    if "ask_prices" in layout_lower and "ask_sizes" in layout_lower:
        return GROUPED_PRICE_INDICES, GROUPED_SIZE_INDICES

    if "sell" in layout_lower or "buy" in layout_lower:
        return LOBSTER_PRICE_INDICES, LOBSTER_SIZE_INDICES

    raise ValueError(
        f"Unknown feature layout: '{layout}'. "
        f"Supported: 'grouped', 'lobster', 'fi2010', 'interleaved', "
        f"or metadata string like 'raw_ask_prices_10_ask_sizes_10_...'"
    )


def detect_layout_from_metadata(metadata: dict) -> str:
    """
    Detect feature layout from export metadata.

    Examines the metadata dict (from *_metadata.json) to determine
    the feature layout used during export.

    Args:
        metadata: Metadata dict from export, typically containing:
            - 'normalization.feature_layout': Layout string from Rust exporter
            - 'n_features': Number of features

    Returns:
        Layout name: "grouped" or "lobster"

    Raises:
        ValueError: If layout cannot be determined from metadata.
    """
    if "normalization" in metadata and isinstance(metadata["normalization"], dict):
        layout_str = metadata["normalization"].get("feature_layout", "")
        if layout_str:
            if "ask_prices" in layout_str:
                return "grouped"
            if "sell" in layout_str.lower():
                return "lobster"

    if "feature_layout" in metadata:
        layout_str = metadata["feature_layout"]
        if "ask_prices" in layout_str:
            return "grouped"
        if "sell" in layout_str.lower():
            return "lobster"

    n_features = metadata.get("n_features", 0)
    if n_features in (40, 98, 116):
        return "grouped"

    raise ValueError(
        f"Cannot detect feature layout from metadata. "
        f"Expected 'normalization.feature_layout' or 'feature_layout' field."
    )
