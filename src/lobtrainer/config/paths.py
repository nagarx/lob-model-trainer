"""Canonical path resolvers for config navigation.

Retires the ``config.labels`` vs ``config.data.labels`` canonical-path-drift
bug class that manifested as Phase II producer-path bugs (see Phase A
bug ledger, 2026-04-23). Every consumer of ``LabelsConfig`` across the
trainer codebase MUST route through :func:`resolve_labels_config` — never
direct attribute access on ``ExperimentConfig``.

The canonical location of ``LabelsConfig`` on a live config is
``ExperimentConfig.data.labels``. Historically, a handful of call sites
in ``lob-model-trainer/src/lobtrainer/export/exporter.py`` and
``lob-model-trainer/src/lobtrainer/training/importance/callback.py`` read
``config.labels`` directly — an attribute that does not exist on
``ExperimentConfig``. The resulting ``AttributeError`` was silently
swallowed by broad ``except Exception`` catches, leaving the producer
unable to emit ``CompatibilityContract`` blocks and ``signal_metadata.json``
files without ``compatibility_fingerprint``.

This helper centralizes the canonical lookup + provides a small
compatibility escape hatch for subprocess / test callers that pass
``DataConfig`` directly (instead of the outer ``ExperimentConfig``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lobtrainer.config.schema import LabelsConfig


def resolve_labels_config(config: Any) -> "LabelsConfig":
    """Return the :class:`LabelsConfig` attached to ``config``.

    Resolution strategy:

    1. ``config.data.labels`` — the canonical location on
       :class:`~lobtrainer.config.schema.ExperimentConfig`.
    2. ``config.labels`` — tolerated fallback when ``config`` is itself a
       :class:`~lobtrainer.config.schema.DataConfig` (subprocess / test
       convenience). Pre-T9 configs also landed here; new code paths
       should always pass an ``ExperimentConfig``.

    This helper NEVER falls through silently to ``None``. If neither
    path yields a ``LabelsConfig``-like object, :class:`AttributeError`
    is raised with a diagnostic message that points the caller at the
    canonical layout. Failing loud at the helper boundary is the
    explicit design — it replaces the broad ``except Exception`` /
    silent-``None`` anti-pattern that masked the Phase II producer-path
    bug cluster.

    Parameters
    ----------
    config
        An :class:`ExperimentConfig` (canonical) or :class:`DataConfig`
        (compatibility). Any other type raises :class:`AttributeError`.

    Returns
    -------
    LabelsConfig
        The attached label configuration.

    Raises
    ------
    AttributeError
        When neither ``config.data.labels`` nor ``config.labels`` exists
        or is non-``None``. The exception message names the ``type(config)``
        to aid debugging.
    """
    data = getattr(config, "data", None)
    if data is not None:
        labels = getattr(data, "labels", None)
        if labels is not None:
            return labels
    # Compatibility fallback: caller passed DataConfig / pre-T9 config directly.
    labels = getattr(config, "labels", None)
    if labels is not None:
        return labels
    raise AttributeError(
        f"resolve_labels_config: neither "
        f"{type(config).__name__}.data.labels nor "
        f"{type(config).__name__}.labels is accessible (both are absent or "
        f"None). Canonical location is `ExperimentConfig.data.labels` "
        f"(schema.py:561). When constructing a config for testing, set "
        f"`config.data.labels = LabelsConfig(...)` explicitly."
    )


__all__ = ["resolve_labels_config"]
