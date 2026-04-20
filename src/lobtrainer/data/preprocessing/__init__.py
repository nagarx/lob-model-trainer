"""Data preprocessing — domain-layer composition on top of ``hft_metrics``.

Phase IV (2026-04-20). This subpackage hosts preprocessing helpers that
compose hft_metrics primitives with ``hft_contracts`` domain knowledge
(FeatureIndex constants, data-source taxonomy). Domain imports live HERE,
not in hft_metrics — hft_metrics is a pure statistics leaf per its CLAUDE.md.

Modules:
  - ``temporal_config``: ``for_mbo_lob()`` / ``for_basic_pipeline()`` factories
    for ``hft_metrics.TemporalFeatureConfig`` that consume
    ``hft_contracts.FeatureIndex`` / ``OffExchangeFeatureIndex`` enums.
"""

from lobtrainer.data.preprocessing.temporal_config import (
    for_mbo_lob,
    for_basic_pipeline,
)

__all__ = ["for_mbo_lob", "for_basic_pipeline"]
