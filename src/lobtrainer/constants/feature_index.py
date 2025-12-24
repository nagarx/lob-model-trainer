"""
Authoritative Feature Index Map (v2.1 — 98 Features).

This module defines the EXACT feature indices matching the Rust pipeline export.
This is a DATA CONTRACT - any changes must be synchronized with:
    - feature-extractor-MBO-LOB/src/features/mod.rs
    - plan/03-FEATURE-INDEX-MAP-v2.md

Source: plan/03-FEATURE-INDEX-MAP-v2.md

Feature Layout Overview:
    | Range   | Count | Category        |
    |---------|-------|-----------------|
    | 0-39    | 40    | Raw LOB         |
    | 40-47   | 8     | Derived         |
    | 48-83   | 36    | MBO             |
    | 84-97   | 14    | Trading Signals |
    | Total   | 98    |                 |

Sign Convention (RULE.md §9):
    - All directional signals follow: > 0 = BULLISH, < 0 = BEARISH
    - Exception: PRICE_IMPACT (47) is unsigned - cannot determine direction

Schema Version History:
    - v2.0: Initial signal layer implementation
    - v2.1: Fixed sign convention for net_trade_flow (56) and net_cancel_flow (55)
"""

from enum import IntEnum
from typing import Final

# =============================================================================
# Schema Version
# =============================================================================

SCHEMA_VERSION: Final[float] = 2.1
"""
Schema version for feature export format. Must match Rust pipeline.

Version History:
    - 2.0: Initial signal layer
    - 2.1: Fixed net_trade_flow (56) and net_cancel_flow (55) sign convention
"""

# =============================================================================
# Label Encoding
# =============================================================================

LABEL_DOWN: Final[int] = -1
"""Price moved down (bearish)."""

LABEL_STABLE: Final[int] = 0
"""Price stayed within threshold (neutral)."""

LABEL_UP: Final[int] = 1
"""Price moved up (bullish)."""

NUM_CLASSES: Final[int] = 3
"""Number of label classes for classification."""

LABEL_NAMES: Final[dict] = {
    LABEL_DOWN: "Down",
    LABEL_STABLE: "Stable",
    LABEL_UP: "Up",
}
"""Human-readable label names (original encoding: {-1, 0, 1})."""

# =============================================================================
# Shifted Label Encoding (for PyTorch CrossEntropyLoss)
# =============================================================================
# PyTorch CrossEntropyLoss requires labels in {0, 1, ..., num_classes-1}.
# We shift original labels {-1, 0, 1} to {0, 1, 2} during training.
#
# Mapping: original + 1 = shifted
#   -1 (Down)   -> 0
#    0 (Stable) -> 1
#   +1 (Up)     -> 2

SHIFTED_LABEL_DOWN: Final[int] = 0
"""Down label after shift (original: -1)."""

SHIFTED_LABEL_STABLE: Final[int] = 1
"""Stable label after shift (original: 0)."""

SHIFTED_LABEL_UP: Final[int] = 2
"""Up label after shift (original: +1)."""

SHIFTED_LABEL_NAMES: Final[dict] = {
    SHIFTED_LABEL_DOWN: "Down",
    SHIFTED_LABEL_STABLE: "Stable",
    SHIFTED_LABEL_UP: "Up",
}
"""Human-readable label names (shifted encoding: {0, 1, 2})."""


def get_label_name(label: int, shifted: bool = False) -> str:
    """
    Get human-readable name for a label value.
    
    Args:
        label: Label value (-1/0/1 for original, 0/1/2 for shifted)
        shifted: True if using shifted encoding (PyTorch), False for original
    
    Returns:
        Label name: "Down", "Stable", or "Up"
    
    Example:
        >>> get_label_name(-1, shifted=False)  # "Down"
        >>> get_label_name(0, shifted=True)    # "Down"
    """
    mapping = SHIFTED_LABEL_NAMES if shifted else LABEL_NAMES
    return mapping.get(label, str(label))

# =============================================================================
# Feature Counts
# =============================================================================

LOB_FEATURE_COUNT: Final[int] = 40
"""Raw LOB features: 10 levels × 4 values (bid_price, ask_price, bid_size, ask_size)."""

DERIVED_FEATURE_COUNT: Final[int] = 8
"""Derived features: mid_price, spread, spread_bps, volumes, microprice, etc."""

MBO_FEATURE_COUNT: Final[int] = 36
"""MBO features: order flow rates, queue stats, institutional detection, etc."""

SIGNAL_FEATURE_COUNT: Final[int] = 14
"""Trading signals: OFI, asymmetry, regime, safety gates, etc."""

FEATURE_COUNT: Final[int] = (
    LOB_FEATURE_COUNT + DERIVED_FEATURE_COUNT + MBO_FEATURE_COUNT + SIGNAL_FEATURE_COUNT
)
"""Total feature count: 98."""

assert FEATURE_COUNT == 98, f"Feature count mismatch: expected 98, got {FEATURE_COUNT}"


# =============================================================================
# Feature Index Enum
# =============================================================================


class FeatureIndex(IntEnum):
    """
    Complete feature index mapping for 98-feature export (Schema v2.1).
    
    Usage:
        >>> features[:, FeatureIndex.TRUE_OFI]  # Access OFI signal
        >>> features[:, FeatureIndex.MID_PRICE]  # Access mid price
    
    Sign Conventions (RULE.md §9):
        - All directional signals: > 0 = BULLISH, < 0 = BEARISH
        - Exception: PRICE_IMPACT (47) is unsigned - do not use for direction
    
    Note: As of v2.1, net_trade_flow (56) and net_cancel_flow (55) follow
          the standard sign convention. No negation required.
    """
    
    # =========================================================================
    # Raw LOB Features (40) — Indices 0-39
    # =========================================================================
    # Layout: [bid_prices(10), ask_prices(10), bid_sizes(10), ask_sizes(10)]
    
    # Bid prices (levels 0-9)
    BID_PRICE_L0 = 0
    BID_PRICE_L1 = 1
    BID_PRICE_L2 = 2
    BID_PRICE_L3 = 3
    BID_PRICE_L4 = 4
    BID_PRICE_L5 = 5
    BID_PRICE_L6 = 6
    BID_PRICE_L7 = 7
    BID_PRICE_L8 = 8
    BID_PRICE_L9 = 9
    
    # Ask prices (levels 0-9)
    ASK_PRICE_L0 = 10
    ASK_PRICE_L1 = 11
    ASK_PRICE_L2 = 12
    ASK_PRICE_L3 = 13
    ASK_PRICE_L4 = 14
    ASK_PRICE_L5 = 15
    ASK_PRICE_L6 = 16
    ASK_PRICE_L7 = 17
    ASK_PRICE_L8 = 18
    ASK_PRICE_L9 = 19
    
    # Bid sizes (levels 0-9)
    BID_SIZE_L0 = 20
    BID_SIZE_L1 = 21
    BID_SIZE_L2 = 22
    BID_SIZE_L3 = 23
    BID_SIZE_L4 = 24
    BID_SIZE_L5 = 25
    BID_SIZE_L6 = 26
    BID_SIZE_L7 = 27
    BID_SIZE_L8 = 28
    BID_SIZE_L9 = 29
    
    # Ask sizes (levels 0-9)
    ASK_SIZE_L0 = 30
    ASK_SIZE_L1 = 31
    ASK_SIZE_L2 = 32
    ASK_SIZE_L3 = 33
    ASK_SIZE_L4 = 34
    ASK_SIZE_L5 = 35
    ASK_SIZE_L6 = 36
    ASK_SIZE_L7 = 37
    ASK_SIZE_L8 = 38
    ASK_SIZE_L9 = 39
    
    # =========================================================================
    # Derived Features (8) — Indices 40-47
    # =========================================================================
    
    MID_PRICE = 40
    """Mid price: (best_bid + best_ask) / 2. USE for microprice calculations."""
    
    SPREAD = 41
    """Spread in dollars: best_ask - best_bid."""
    
    SPREAD_BPS = 42
    """Spread in basis points: spread / mid_price × 10000. USE for spread filter."""
    
    TOTAL_BID_VOLUME = 43
    """Total bid volume across all levels: Σ bid_sizes[0..L]."""
    
    TOTAL_ASK_VOLUME = 44
    """Total ask volume across all levels: Σ ask_sizes[0..L]."""
    
    VOLUME_IMBALANCE = 45
    """Volume imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol). Range: [-1, 1]."""
    
    WEIGHTED_MID_PRICE = 46
    """Microprice (Stoikov): (bid×ask_vol + ask×bid_vol) / (bid_vol + ask_vol)."""
    
    PRICE_IMPACT = 47
    """⚠️ UNSIGNED: |mid - microprice|. Cannot determine direction - do not use for signals."""
    
    # =========================================================================
    # MBO Features (36) — Indices 48-83
    # =========================================================================
    
    # Order flow rates (6)
    ADD_RATE_BID = 48
    ADD_RATE_ASK = 49
    CANCEL_RATE_BID = 50
    """USE for cancel_asymmetry calculation."""
    CANCEL_RATE_ASK = 51
    """USE for cancel_asymmetry calculation."""
    TRADE_RATE_BID = 52
    """Trade rate at bid = SELL-initiated trades."""
    TRADE_RATE_ASK = 53
    """Trade rate at ask = BUY-initiated trades."""
    
    # Net flows (3)
    NET_ORDER_FLOW = 54
    """
    Net order flow: (add_bid - add_ask) / total.
    > 0 = more bid adds (buy orders) = BULLISH.
    """
    
    NET_CANCEL_FLOW = 55
    """
    Net cancel flow: (cancel_ask - cancel_bid) / total.
    > 0 = more ask cancels (sellers pulling) = BULLISH.
    Fixed in v2.1 to follow standard sign convention.
    """
    
    NET_TRADE_FLOW = 56
    """
    Net trade flow: (trade_ask - trade_bid) / total.
    In MBO: trade_ask = buyer hitting ask = BUY-initiated.
    > 0 = more BUY-initiated trades = BULLISH.
    Fixed in v2.1 to follow standard sign convention.
    """
    
    # Conviction indicators (3)
    AGGRESSIVE_ORDER_RATIO = 57
    """Ratio of market orders to total: trades / (adds + trades). Range: [0, 1]. USE for conviction."""
    
    ORDER_FLOW_VOLATILITY = 58
    """Std dev of net flow over sub-windows. USE for regime gate."""
    
    FLOW_REGIME_INDICATOR = 59
    """Fast flow / slow flow ratio. > 1 = accelerating. USE for regime gate."""
    
    # Size distribution (8)
    SIZE_MEAN_BID = 60
    SIZE_MEAN_ASK = 61
    SIZE_STD_BID = 62
    SIZE_STD_ASK = 63
    SIZE_MAX_BID = 64
    SIZE_MAX_ASK = 65
    SIZE_SKEWNESS = 66
    SIZE_CONCENTRATION = 67
    
    # Queue depth (6)
    QUEUE_SIZE_BID = 68
    QUEUE_SIZE_ASK = 69
    AVERAGE_QUEUE_POSITION = 70
    LEVEL_CONCENTRATION = 71
    """HHI of volume across levels. USE for fragility_score."""
    DEPTH_TICKS_BID = 72
    """Volume-weighted average depth in ticks (bid side)."""
    DEPTH_TICKS_ASK = 73
    """Volume-weighted average depth in ticks (ask side)."""
    
    # Institutional detection (4)
    LARGE_ORDER_COUNT_BID = 74
    LARGE_ORDER_COUNT_ASK = 75
    INSTITUTIONAL_RATIO_BID = 76
    INSTITUTIONAL_RATIO_ASK = 77
    
    # Core MBO metrics (6)
    ORDER_LIFETIME_MEAN = 78
    ORDER_LIFETIME_STD = 79
    CANCEL_RATIO = 80
    MODIFY_RATIO = 81
    QUEUE_SIZE_AHEAD = 82
    ORDER_COUNT_ACTIVE = 83
    
    # =========================================================================
    # Trading Signals (14) — Indices 84-97
    # =========================================================================
    # See: plan/01-SIGNAL-HIERARCHY.md for priority and usage
    
    TRUE_OFI = 84
    """
    Cont et al. (2014) Order Flow Imbalance: Σ e_n over sampling interval.
    CRITICAL signal. > 0 = BUY pressure, < 0 = SELL pressure.
    Source: "The price impact of order book events", §2.1
    """
    
    DEPTH_NORM_OFI = 85
    """
    Depth-normalized OFI: true_ofi / avg_depth.
    Accounts for β ∝ 1/AD per Cont et al. §3.2.
    > 0 = BUY pressure, < 0 = SELL pressure.
    """
    
    EXECUTED_PRESSURE = 86
    """
    Trade confirmation: trade_rate_ask - trade_rate_bid.
    > 0 = net buying (BUY-initiated > SELL-initiated).
    Should align with TRUE_OFI sign in coherent markets.
    """
    
    SIGNED_MP_DELTA_BPS = 87
    """
    Microprice deviation from mid in basis points: (microprice - mid) / mid × 10000.
    > 0 = microprice above mid = BUY pressure.
    Stoikov (2018): "The Micro-Price: A High Frequency Estimator of Future Prices"
    """
    
    TRADE_ASYMMETRY = 88
    """
    Normalized executed pressure: (trade_ask - trade_bid) / (trade_ask + trade_bid).
    Range: [-1, 1]. > 0 = BUY pressure.
    """
    
    CANCEL_ASYMMETRY = 89
    """
    Cancel imbalance: (cancel_ask - cancel_bid) / (cancel_ask + cancel_bid).
    Range: [-1, 1]. > 0 = more ask cancels = bullish (sellers pulling quotes).
    """
    
    FRAGILITY_SCORE = 90
    """
    Book fragility: level_concentration / ln(avg_depth).
    Higher = more fragile book, susceptible to large moves.
    """
    
    DEPTH_ASYMMETRY = 91
    """
    Depth imbalance: (depth_bid - depth_ask) / (depth_bid + depth_ask).
    Range: [-1, 1]. > 0 = more bid depth = bullish (stronger support).
    """
    
    BOOK_VALID = 92
    """
    Safety gate: 1.0 if book is valid (bid < ask, both exist), 0.0 otherwise.
    MUST check before using any other signals. Skip sample if == 0.
    """
    
    TIME_REGIME = 93
    """
    Market session encoding (ET timezone):
        0 = OPEN (9:30-9:45): High volatility, wide spreads
        1 = EARLY (9:45-10:30): Settling period
        2 = MIDDAY (10:30-15:30): Most stable
        3 = CLOSE (15:30-16:00): Position squaring
        4 = CLOSED (after hours): Pre-market/after-hours, ~6% of samples
    Cont et al. (2014) §3.3: "Price impact is 5× higher at open vs close."
    """
    
    MBO_READY = 94
    """
    Warmup gate: 1.0 if MBO features are warmed up (>= 100 state changes), 0.0 otherwise.
    MUST check before using indices 48-91. Features may be NaN during warmup.
    """
    
    DT_SECONDS = 95
    """
    Sample duration in seconds (wall-clock time since last sample).
    Use for time-based logic instead of sample counts.
    """
    
    INVALIDITY_DELTA = 96
    """
    Feed quality: Count of crossed/locked events since last sample.
    0 = clean feed. > 0 = feed had problems (even if book_valid == 1).
    Critical under UseLastValid policy where book_valid stays 1.
    """
    
    SCHEMA_VERSION_FEATURE = 97
    """Fixed value = 2.1 (after sign convention fix). For forward compatibility checking."""


# =============================================================================
# Signal-Specific Index Class (Convenience)
# =============================================================================


class SignalIndex(IntEnum):
    """
    Convenience enum for just the 14 trading signals (indices 84-97).
    
    Usage:
        >>> signals = features[:, SignalIndex.TRUE_OFI:SignalIndex.SCHEMA_VERSION + 1]
    """
    
    TRUE_OFI = 84
    DEPTH_NORM_OFI = 85
    EXECUTED_PRESSURE = 86
    SIGNED_MP_DELTA_BPS = 87
    TRADE_ASYMMETRY = 88
    CANCEL_ASYMMETRY = 89
    FRAGILITY_SCORE = 90
    DEPTH_ASYMMETRY = 91
    BOOK_VALID = 92
    TIME_REGIME = 93
    MBO_READY = 94
    DT_SECONDS = 95
    INVALIDITY_DELTA = 96
    SCHEMA_VERSION = 97


# =============================================================================
# Feature Groups (for selective loading)
# =============================================================================

# LOB level slices
LOB_BID_PRICES = slice(0, 10)
LOB_ASK_PRICES = slice(10, 20)
LOB_BID_SIZES = slice(20, 30)
LOB_ASK_SIZES = slice(30, 40)
LOB_ALL = slice(0, 40)

# Category slices
DERIVED_ALL = slice(40, 48)
MBO_ALL = slice(48, 84)
SIGNALS_ALL = slice(84, 98)

# Commonly used feature groups
SAFETY_GATES = (FeatureIndex.BOOK_VALID, FeatureIndex.MBO_READY, FeatureIndex.INVALIDITY_DELTA)
"""Features that must be checked before using other signals."""

PRIMARY_SIGNALS = (FeatureIndex.TRUE_OFI, FeatureIndex.EXECUTED_PRESSURE, FeatureIndex.BOOK_VALID)
"""Most important signals per research (Cont et al. 2014)."""

ASYMMETRY_SIGNALS = (
    FeatureIndex.TRADE_ASYMMETRY,
    FeatureIndex.CANCEL_ASYMMETRY,
    FeatureIndex.DEPTH_ASYMMETRY,
)
"""Normalized asymmetry signals, all in range [-1, 1]."""


# =============================================================================
# Sign Convention Notes
# =============================================================================

UNSIGNED_FEATURES = frozenset({FeatureIndex.PRICE_IMPACT})
"""Unsigned features. Cannot be used for directional signals."""

# NOTE: As of v2.1, all directional features follow standard convention:
#       > 0 = BULLISH, < 0 = BEARISH
#       The previous OPPOSITE_SIGN_FEATURES set has been removed as the
#       Rust pipeline now correctly implements sign conventions for:
#       - net_trade_flow (56)
#       - net_cancel_flow (55)

