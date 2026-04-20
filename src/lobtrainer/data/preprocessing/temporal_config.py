"""Domain-layer factory for ``hft_metrics.TemporalFeatureConfig``.

Phase IV (2026-04-20). Maps ``hft_contracts`` enum values to
``hft_metrics.TemporalFeatureConfig`` fields, binding domain knowledge
(feature-index semantics, data-source taxonomy) to the pure statistics
primitive. This is the correct home for the binding:

  - ``hft_metrics`` is a pure statistics leaf per its CLAUDE.md (no domain
    knowledge; no hft_contracts imports).
  - ``lob_model_trainer.data.preprocessing`` hosts preprocessing helpers that
    are allowed to combine domain knowledge with primitives.

Factory methods:
  - ``for_mbo_lob()``: MBO-LOB pipeline (F=98/148). Uses FeatureIndex enum values.
  - ``for_basic_pipeline(**overrides)``: BASIC off-exchange pipeline (F=34).
    Uses OffExchangeFeatureIndex; disables regime (no TIME_REGIME/DT_SECONDS
    features exist in BASIC). Allows per-caller override of signal/context/
    cross_pairs since the BASIC feature layout is newer and research conventions
    are still evolving.

Usage:
    from lobtrainer.data.preprocessing import for_mbo_lob, for_basic_pipeline
    from hft_metrics import engineer_features

    config = for_mbo_lob()            # MBO defaults
    features = engineer_features(sequences, config)

    # BASIC — override signal_indices as needed
    config = for_basic_pipeline(signal_indices=[5, 12, 18])
    features = engineer_features(basic_sequences, config)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from hft_contracts import FeatureIndex
from hft_metrics import TemporalFeatureConfig


def for_mbo_lob() -> TemporalFeatureConfig:
    """Produce a ``TemporalFeatureConfig`` with MBO-LOB defaults (F=98/148).

    Bit-exact equivalent to ``TemporalFeatureConfig()`` default constructor —
    the factory exists for explicitness + uniformity with ``for_basic_pipeline``.
    Using this factory signals "MBO data source" at the call site, improving
    readability and grep-ability when multiple pipelines coexist in a codebase.
    """
    return TemporalFeatureConfig(
        signal_indices=[
            int(FeatureIndex.DEPTH_NORM_OFI),       # 85
            int(FeatureIndex.TRUE_OFI),             # 84
            int(FeatureIndex.EXECUTED_PRESSURE),    # 86
            int(FeatureIndex.NET_TRADE_FLOW),       # 56
            int(FeatureIndex.VOLUME_IMBALANCE),     # 45
        ],
        top_k_signals=3,
        rolling_windows=[5, 10, 20],
        cross_pairs=[
            (int(FeatureIndex.DEPTH_NORM_OFI), int(FeatureIndex.TRUE_OFI)),
            (int(FeatureIndex.DEPTH_NORM_OFI), int(FeatureIndex.EXECUTED_PRESSURE)),
            (int(FeatureIndex.DEPTH_NORM_OFI), int(FeatureIndex.VOLUME_IMBALANCE)),
            (int(FeatureIndex.TRUE_OFI), int(FeatureIndex.NET_TRADE_FLOW)),
            (int(FeatureIndex.TRUE_OFI), int(FeatureIndex.EXECUTED_PRESSURE)),
            (int(FeatureIndex.EXECUTED_PRESSURE), int(FeatureIndex.VOLUME_IMBALANCE)),
        ],
        context_indices=[
            int(FeatureIndex.SPREAD_BPS),           # 42
            int(FeatureIndex.VOLUME_IMBALANCE),     # 45
            int(FeatureIndex.NET_ORDER_FLOW),       # 54
            int(FeatureIndex.MID_PRICE),            # 40
            int(FeatureIndex.DT_SECONDS),           # 95
        ],
        time_regime_idx=int(FeatureIndex.TIME_REGIME),
        dt_seconds_idx=int(FeatureIndex.DT_SECONDS),
        include_regime=True,
        include_volatility=True,
        vol_windows=[10, 5],
        mid_price_idx=int(FeatureIndex.MID_PRICE),
        include_momentum=True,
        momentum_short=5,
        momentum_long=20,
    )


def for_basic_pipeline(
    *,
    signal_indices: Optional[List[int]] = None,
    context_indices: Optional[List[int]] = None,
    cross_pairs: Optional[List[Tuple[int, int]]] = None,
    mid_price_idx: Optional[int] = None,
    top_k_signals: int = 1,
    rolling_windows: Optional[List[int]] = None,
    vol_windows: Optional[List[int]] = None,
    momentum_short: int = 5,
    momentum_long: int = 20,
) -> TemporalFeatureConfig:
    """Produce a ``TemporalFeatureConfig`` for the BASIC off-exchange pipeline (F=34).

    Closes FRESH-1 (validation report 2026-04-20): legacy
    ``TemporalFeatureConfig()`` default had ``time_regime_idx=93`` +
    ``dt_seconds_idx=95`` — BASIC data (F=34) crashed at ``IndexError`` when
    engineer_temporal_features accessed those indices. This factory sets both
    to None and forces ``include_regime=False``.

    ``signal_indices`` is **REQUIRED** (no silent default). Phase II hardening
    SB-4 (2026-04-20): the prior ``signal_indices=[0]`` + ``mid_price_idx=0`` +
    ``context_indices=[1, 0]`` bare defaults produced self-correlated garbage
    features (rolling-window-of-mid-price vs mid-price ≈ 1). Per hft-rules §5
    "fail fast with a precise error — never silently degrade." Research
    workflows that don't yet have canonical BASIC signal indices must supply
    placeholder picks explicitly — e.g.
    ``for_basic_pipeline(signal_indices=[5, 12, 18], mid_price_idx=0)`` — so
    the intent is visible at the call site. The BASIC feature layout (see
    ``hft_contracts.OffExchangeFeatureIndex``) enumerates the valid range.

    Args:
        signal_indices: BASIC signal feature positions. **REQUIRED** — no
            default. Typical picks for 34-feature BASIC layout live in
            ``hft_contracts.OffExchangeFeatureIndex``; use semantic enum
            values, not literal integers, to make intent readable.
        context_indices: BASIC context feature positions. Default ``[]`` —
            no context emission unless explicitly supplied.
        cross_pairs: Signal cross-products. Default ``[]`` — BASIC pair synergies
            not yet empirically validated; opt-in only.
        mid_price_idx: Position of mid-price in BASIC layout. Default ``None``
            — caller must supply when ``include_volatility=True`` (always on
            here). If unsupplied, raises ``ValueError`` with actionable message.
        top_k_signals: How many of ``signal_indices`` to rolling-window.
        rolling_windows: Windows for rolling mean/slope/ROC. Default ``[5, 10, 20]``.
        vol_windows: Windows for realized volatility. Default ``[10, 5]``.
        momentum_short / momentum_long: Short/long mean-diff windows.

    Returns:
        Config with ``include_regime=False`` + ``time_regime_idx=dt_seconds_idx=None``
        — safe on F=34 data.

    Raises:
        ValueError: If ``signal_indices`` is not supplied (fail-fast per
            hft-rules §5 — no silent garbage-feature path).
        ValueError: If ``mid_price_idx`` is not supplied (required for the
            realized-vol feature; caller MUST declare the mid-price location
            in the BASIC layout).
    """
    # Lazy import — OffExchangeFeatureIndex may not exist in minimal hft-contracts installs
    from hft_contracts import OffExchangeFeatureIndex  # noqa: F401

    # SB-4 fail-fast: signal_indices must be explicit (per hft-rules §5).
    # Previous silent default `[0]` + mid_price_idx=0 produced rolling-window
    # features of mid-price against mid-price itself (self-correlation ~ 1).
    if signal_indices is None:
        raise ValueError(
            "for_basic_pipeline: signal_indices is required. BASIC feature "
            "conventions are still evolving; supply explicit picks (e.g., "
            "`signal_indices=[5, 12, 18]` or semantic values from "
            "`hft_contracts.OffExchangeFeatureIndex`). Prior silent default [0] "
            "was removed in Phase II hardening (SB-4) because it produced "
            "self-correlated garbage features on BASIC data."
        )
    if mid_price_idx is None:
        raise ValueError(
            "for_basic_pipeline: mid_price_idx is required (used for realized "
            "volatility feature). Supply the mid-price column position in your "
            "BASIC feature layout (typically 0 — verify against your export's "
            "metadata or hft_contracts.OffExchangeFeatureIndex.MID_PRICE)."
        )

    if context_indices is None:
        # Empty default — no context features unless caller explicitly opts in.
        # Fail-safe: no empirical commitment that hasn't been validated.
        context_indices = []
    if cross_pairs is None:
        cross_pairs = []
    if rolling_windows is None:
        rolling_windows = [5, 10, 20]
    if vol_windows is None:
        vol_windows = [10, 5]

    return TemporalFeatureConfig(
        signal_indices=signal_indices,
        top_k_signals=top_k_signals,
        rolling_windows=rolling_windows,
        cross_pairs=cross_pairs,
        context_indices=context_indices,
        time_regime_idx=None,         # FRESH-1 fix: BASIC has no TIME_REGIME
        dt_seconds_idx=None,          # FRESH-1 fix: BASIC has no DT_SECONDS
        include_regime=False,         # forced off
        include_volatility=True,
        vol_windows=vol_windows,
        mid_price_idx=mid_price_idx,
        include_momentum=True,
        momentum_short=momentum_short,
        momentum_long=momentum_long,
    )
