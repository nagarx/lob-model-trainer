"""Phase 2 P2.A — bootstrap CI on test metrics from stored signal arrays.

Reads ``predicted_returns.npy`` + ``regression_labels.npy`` +
``signal_metadata.json`` from a trainer experiment's ``signals/test/``
directory; computes block-bootstrap confidence intervals on 7 standard
test metrics (test_ic / test_directional_accuracy / test_r2 / test_pearson /
test_mae / test_rmse / test_profitable_accuracy); persists as
``TestMetricsCIArtifact`` to ``test_metrics_ci_v1.json`` alongside the
experiment's existing ``test_metrics.json``.

Architecture:
  - Per hft-rules §0 reuse-first: ``compute_ci`` delegates to
    ``hft_metrics.block_bootstrap_ci`` SSoT; metric statistic_fns wrap
    the 7 ``hft_metrics`` regression/IC primitives.
  - Per hft-rules §8 fail-loud: ``assert_finite_pair`` validates inputs
    BEFORE bootstrap loop (block_bootstrap_ci itself does NOT validate
    per Round 1 Agent 1 finding).
  - Per hft-rules §11 provenance-before-comparison: ``ci_low <= point
    <= ci_high`` invariant enforced at ``TestMetricsCIArtifact``
    construction time.
  - Per #PY-67 mitigation: operates on stored signal arrays — does NOT
    reload checkpoints (avoids stale R9-R14 ``compatibility_fingerprint``).
  - Per Agent 3 architectural critique (Round 1, 2026-05-07):
    decomposed API (``TestMetricsCIConfig`` + ``compute_ci`` +
    ``from_signal_dir`` + ``compute_test_metrics_ci_for_experiment``)
    mirrors ``hft_evaluator.experiments.offexchange_gate.run`` template.

Phase Y composability: artifact integrates with future
``experiment_provenance_hash`` via ``record.artifacts[].sha256``
projection (routed through hft-ops ``_POST_STAGE_ARTIFACT_PATTERNS``).

References:
  - Politis & Romano (1994). The Stationary Bootstrap. JASA 89:1303-1313
    [block_length auto-derive ``ceil(n^(1/3))``]
  - Künsch, H. R. (1989). The jackknife and the bootstrap for general
    stationary observations. Annals of Statistics 17:1217-1241
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, Optional, Tuple

import numpy as np

from hft_contracts.test_metrics_ci_artifact import (
    TEST_METRICS_CI_SCHEMA_VERSION,
    MetricCIBound,
    TestMetricsCIArtifact,
)
from hft_metrics import block_bootstrap_ci
from hft_metrics import ic as ic_module
from hft_metrics._sanitize import assert_finite_pair
from hft_metrics.regression import (
    directional_accuracy,
    mean_absolute_error,
    profitable_accuracy,
    r_squared,
    root_mean_squared_error,
)


LOGGER = logging.getLogger(__name__)


# Registry of OUTPUT metric name → (statistic_fn, INPUT metric name lookup
# in signal_metadata.metrics).
#
# - statistic_fn signature: ``(y_true, y_pred) -> float``. block_bootstrap_ci
#   passes the resampled (x, y) pair positionally with our convention
#   x=labels, y=predictions per Round 1 Agent 1 verification.
# - spearman_ic and pearson_r return ``(rho, p_value)`` tuples — wrap to
#   extract the first element (the correlation/IC value). Per Round 1
#   Agent 1: these were NOT migrated to fail-loud in #PY-63 (only the 5
#   regression metrics were); they still silently sanitize via
#   sanitize_pair internally. Acceptable here because input arrays are
#   pre-validated by ``assert_finite_pair`` upstream.
# - INPUT metric name (e.g., "ic", "r2") matches signal_metadata.metrics
#   key convention (no "test_" prefix at producer side); OUTPUT name
#   (e.g., "test_ic") matches Plan v4 §4.1 schema for the CI artifact.
_METRIC_REGISTRY: Dict[str, Tuple[Callable[[np.ndarray, np.ndarray], float], str]] = {
    "test_ic": (lambda yt, yp: ic_module.spearman_ic(yt, yp)[0], "ic"),
    "test_pearson": (lambda yt, yp: ic_module.pearson_r(yt, yp)[0], "pearson"),
    "test_r2": (r_squared, "r2"),
    "test_mae": (mean_absolute_error, "mae"),
    "test_rmse": (root_mean_squared_error, "rmse"),
    "test_directional_accuracy": (directional_accuracy, "directional_accuracy"),
    "test_profitable_accuracy": (profitable_accuracy, "profitable_accuracy"),
}

# Default per Plan v4 §4.1 — all 7 metrics in canonical order.
DEFAULT_METRIC_NAMES: Tuple[str, ...] = tuple(_METRIC_REGISTRY.keys())


@dataclass(frozen=True)
class TestMetricsCIConfig:
    """Configuration for bootstrap CI computation.

    Frozen — produced once by orchestrator; consumers read it.

    Defaults are Plan v4 §4.1 recommended values:
      - ``n_bootstraps=10000`` (vs hft_metrics primitive default 1000;
        Plan v4 v1→v2 correction)
      - ``block_length=None`` (auto-derive ``ceil(n^(1/3))`` per
        Politis-Romano 1994; Plan v4 v3→v4 correction — DO NOT hardcode 18)
      - ``ci=0.95`` (NOT alpha=0.05 — Plan v4 v1→v2 wrong-API correction)
      - ``seed=42`` (deterministic)
      - ``primary_horizon_idx=0`` (slice for multi-horizon HMHP-R)

    Per hft-rules §5 fail-fast: ``__post_init__`` rejects degenerate
    parameter combinations.
    """

    # Pytest-discovery suppression: class name starts with "Test" but it is
    # a domain config dataclass, not a pytest test class.
    __test__: ClassVar[bool] = False

    n_bootstraps: int = 10_000
    block_length: Optional[int] = None
    ci: float = 0.95
    seed: int = 42
    primary_horizon_idx: int = 0
    require_paired_n: bool = True
    metric_names: Tuple[str, ...] = field(default_factory=lambda: DEFAULT_METRIC_NAMES)

    def __post_init__(self) -> None:
        if self.n_bootstraps < 100:
            raise ValueError(
                f"TestMetricsCIConfig: n_bootstraps={self.n_bootstraps} < 100 — "
                f"too few for stable CI per Politis-Romano sample-size analysis"
            )
        if self.block_length is not None and self.block_length < 2:
            raise ValueError(
                f"TestMetricsCIConfig: block_length={self.block_length} < 2 — "
                f"degenerate; block resampling requires block_length >= 2 to "
                f"preserve autocorrelation (block_length=1 is element-wise iid)"
            )
        if not 0.0 < self.ci < 1.0:
            raise ValueError(
                f"TestMetricsCIConfig: ci={self.ci} not in (0.0, 1.0). "
                f"Common values: 0.90, 0.95, 0.99 (NOT alpha=0.05 — wrong-API)"
            )
        if self.primary_horizon_idx < 0:
            raise ValueError(
                f"TestMetricsCIConfig: primary_horizon_idx="
                f"{self.primary_horizon_idx} < 0"
            )
        if not self.metric_names:
            raise ValueError(
                "TestMetricsCIConfig: metric_names must not be empty"
            )
        unknown = set(self.metric_names) - set(_METRIC_REGISTRY.keys())
        if unknown:
            raise ValueError(
                f"TestMetricsCIConfig: unknown metric_names: {sorted(unknown)}. "
                f"Registered metrics: {sorted(_METRIC_REGISTRY.keys())}"
            )


def _slice_to_primary_horizon(
    arr: np.ndarray,
    primary_horizon_idx: int,
    array_role: str,
) -> np.ndarray:
    """Slice an array to its primary horizon if multi-horizon.

    TLOB regression: shape (N,) → identity (no slicing).
    HMHP-R: shape (N, H) → slice ``[:, primary_horizon_idx]``.
    Other shapes: fail-loud per hft-rules §8.

    Args:
      arr: Array to slice.
      primary_horizon_idx: Horizon index for slicing (0-based).
      array_role: Diagnostic name (e.g., "predicted_returns", "labels").

    Returns:
      1-D array.

    Raises:
      ValueError: If shape is neither (N,) nor (N, H).
      IndexError: If primary_horizon_idx is out-of-bounds for (N, H).
    """
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        if primary_horizon_idx >= arr.shape[1]:
            raise IndexError(
                f"{array_role}: primary_horizon_idx={primary_horizon_idx} >= "
                f"shape[1]={arr.shape[1]} — out-of-bounds for multi-horizon"
            )
        return arr[:, primary_horizon_idx]
    raise ValueError(
        f"{array_role}: unexpected shape {arr.shape}. Expected (N,) for "
        f"single-horizon regression or (N, H) for multi-horizon (HMHP-R)."
    )


def _resolve_block_length(
    n_samples: int,
    config_block_length: Optional[int],
) -> Tuple[int, str]:
    """Resolve effective block_length + human-readable source string.

    Mirrors ``hft_metrics.block_bootstrap_ci`` internal auto-derive logic so
    the artifact correctly records the value that was actually used.

    DOCUMENTED-INTENTIONAL-DUPLICATE (Round 1 §7 LOW): the formula
    ``ceil(n^(1/3))`` also lives at ``hft_metrics.bootstrap.py:180``.
    Re-derived here so the artifact's ``block_length_source`` field
    captures the value rather than reading-back-from-primitive (which
    would require the primitive to expose its resolved block_length —
    multi-call-site change deferred). Future formula change requires
    synchronized edit at both sites.

    Round 1 §1 HIGH degenerate-block guard (mid-impl adversarial finding):
    if the resolved block_length is ``>= n_samples``, the bootstrap
    collapses to a single-block resample (variance → 0; CI degenerates
    to ``ci_low == ci_high == point``). Per hft-rules §8 fail-loud + §5
    fail-fast, raise instead of producing a misleading degenerate
    artifact.

    Args:
      n_samples: Number of paired (label, prediction) samples.
      config_block_length: User-provided override, or None for auto-derive.

    Returns:
      (effective_block_length, block_length_source).

    Raises:
      ValueError: If effective block_length >= n_samples (degenerate).
    """
    if config_block_length is not None:
        effective = int(config_block_length)
        source = f"explicit override (config.block_length={config_block_length})"
    else:
        # Politis & Romano (1994): block_length = ceil(n^(1/3)).
        effective = int(math.ceil(n_samples ** (1.0 / 3.0)))
        source = (
            f"auto-derive ceil(n^(1/3))={effective} per Politis-Romano (1994) "
            f"for n_samples={n_samples}"
        )

    if effective >= n_samples:
        raise ValueError(
            f"_resolve_block_length: effective_block_length={effective} "
            f">= n_samples={n_samples}. Block resampling produces degenerate "
            f"single-block bootstrap (variance collapses; CI bounds equal "
            f"point). Either: (a) provide more samples (need at least "
            f"block_length+1), OR (b) explicitly override block_length to "
            f"a value < n_samples (config.block_length=...)."
        )

    return effective, source


def compute_ci(
    predicted: np.ndarray,
    labels: np.ndarray,
    config: TestMetricsCIConfig,
    metadata_overlay: Optional[Dict[str, Any]] = None,
) -> TestMetricsCIArtifact:
    """Compute bootstrap CI on test metrics from in-memory arrays.

    Per hft-rules §0 reuse-first: delegates to ``hft_metrics.block_bootstrap_ci``.
    Per hft-rules §8 fail-loud: validates inputs via ``assert_finite_pair``
    BEFORE bootstrap loop (block_bootstrap_ci itself does NOT validate).
    Per hft-rules §11: ``ci_low <= point <= ci_high`` enforced at artifact
    construction.

    Args:
      predicted: Predictions array. Shape ``(N,)`` for single-horizon (TLOB)
        or ``(N, H)`` for multi-horizon (HMHP-R) — sliced to
        ``config.primary_horizon_idx``.
      labels: Labels array. Same shape constraint as ``predicted``.
      config: ``TestMetricsCIConfig`` controlling bootstrap parameters.
      metadata_overlay: Optional dict of fields populating the artifact's
        traceability columns (``compatibility_fingerprint``,
        ``model_config_hash``, ``normalization_stats_sha256``,
        ``signal_export_output_dir``, ``experiment_id``, ``fingerprint``,
        ``model_type``, ``method_caveats``). Caller responsibility.

    Returns:
      ``TestMetricsCIArtifact`` (frozen dataclass).

    Raises:
      ValueError: If inputs contain NaN/Inf or shape mismatch.
      IndexError: If ``primary_horizon_idx`` out-of-bounds.
      RuntimeError: If ``block_bootstrap_ci`` fails for any metric.
    """
    pred_1d = _slice_to_primary_horizon(predicted, config.primary_horizon_idx, "predicted")
    labels_1d = _slice_to_primary_horizon(labels, config.primary_horizon_idx, "labels")

    if config.require_paired_n and pred_1d.shape != labels_1d.shape:
        raise ValueError(
            f"compute_ci: predicted.shape={pred_1d.shape} != "
            f"labels.shape={labels_1d.shape} after horizon slicing — "
            f"unpaired arrays violate bootstrap pairing assumption"
        )

    n_samples = int(pred_1d.size)

    # Fail-loud per §8: NaN/Inf in input arrays is a caller-invariant violation.
    # #PY-63 producer-side fail-loud guarantees signal exports are clean — but
    # defense-in-depth here for direct-construction (Jupyter / tests).
    assert_finite_pair(
        labels_1d, pred_1d,
        name=(
            f"compute_ci.input(N={n_samples}, "
            f"primary_horizon_idx={config.primary_horizon_idx})"
        ),
    )

    effective_block_length, block_length_source = _resolve_block_length(
        n_samples, config.block_length
    )

    metric_bounds: Dict[str, MetricCIBound] = {}
    for metric_name in config.metric_names:
        statistic_fn, _input_key = _METRIC_REGISTRY[metric_name]
        try:
            estimate, ci_low, ci_high = block_bootstrap_ci(
                statistic_fn=statistic_fn,
                x=labels_1d, y=pred_1d,
                n_bootstraps=config.n_bootstraps,
                block_length=config.block_length,  # None → primitive auto-derives
                ci=config.ci,
                seed=config.seed,
            )
        except Exception as exc:
            raise RuntimeError(
                f"compute_ci: block_bootstrap_ci failed for metric "
                f"{metric_name!r} on n_samples={n_samples}: {exc}"
            ) from exc

        # Numerical fail-loud guard (Round 1 §1 HIGH mid-impl adversarial
        # finding): if ``not (ci_low <= estimate <= ci_high)``, the
        # ``hft_metrics.block_bootstrap_ci`` per-iteration NaN replacement
        # (bootstrap.py:199) + nanpercentile fallback produced a degenerate
        # result. Per hft-rules §8 ("never silently drop, clamp, or 'fix'
        # data"): RAISE rather than clamp. The previous clamping behavior
        # silently rewrote ``point`` to a tampered value indistinguishable
        # from the original bootstrap output — breaking provenance (§9).
        # ``MetricCIBound.__post_init__`` would also catch this as a
        # secondary line of defense.
        if not (ci_low <= estimate <= ci_high):
            raise RuntimeError(
                f"compute_ci: metric {metric_name!r} produced degenerate "
                f"bootstrap result (estimate={estimate} outside "
                f"[ci_low={ci_low}, ci_high={ci_high}]). Likely causes: "
                f"(a) constant resample blocks producing undefined "
                f"statistic, (b) numerical instability in the metric "
                f"function, (c) upstream bug in hft_metrics.block_bootstrap_ci. "
                f"Re-run with a different seed (config.seed=...) or "
                f"investigate. Per hft-rules §8 silent clamping is forbidden."
            )

        # Construct MetricCIBound — its __post_init__ enforces
        # finiteness + invariant + n_samples>0 (Round 1 §2 HIGH fix).
        metric_bounds[metric_name] = MetricCIBound(
            point=float(estimate),
            ci_low=float(ci_low),
            ci_high=float(ci_high),
            n_samples=n_samples,
        )

    overlay = metadata_overlay or {}
    return TestMetricsCIArtifact(
        schema_version=TEST_METRICS_CI_SCHEMA_VERSION,
        method="block_bootstrap",
        block_length=effective_block_length,
        block_length_source=block_length_source,
        n_bootstraps=config.n_bootstraps,
        ci=config.ci,
        seed=config.seed,
        n_test_samples=n_samples,
        metrics=metric_bounds,
        compatibility_fingerprint=overlay.get("compatibility_fingerprint"),
        model_config_hash=overlay.get("model_config_hash"),
        normalization_stats_sha256=overlay.get("normalization_stats_sha256"),
        signal_export_output_dir=overlay.get("signal_export_output_dir"),
        experiment_id=overlay.get("experiment_id", ""),
        fingerprint=overlay.get("fingerprint", ""),
        model_type=overlay.get("model_type", ""),
        timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        method_caveats=tuple(overlay.get("method_caveats", ())),
    )


def from_signal_dir(
    signals_dir: Path,
    config: TestMetricsCIConfig,
) -> TestMetricsCIArtifact:
    """Load signal arrays + metadata from signal export dir, compute CI.

    Per hft-rules §8 fail-loud:
      - Missing ``predicted_returns.npy`` → ``FileNotFoundError``
      - Missing ``regression_labels.npy`` → ``FileNotFoundError``
      - Missing ``signal_metadata.json`` → ``FileNotFoundError``
      - Malformed metadata JSON → ``json.JSONDecodeError``

    The metadata overlay is built from ``signal_metadata.json`` top-level
    fields. Per Round 1 Agent 2 verification, these fields exist on
    PyTorch-path signal exports (R9 + cousins); sklearn-path exports
    (TemporalRidge / TemporalGradBoost) lack them pre-Phase-Q.6.5 — they
    will populate as None in the resulting artifact, which is allowed.

    Args:
      signals_dir: Path to ``signals/test/`` directory.
      config: ``TestMetricsCIConfig``.

    Returns:
      ``TestMetricsCIArtifact`` with metadata overlay populated.
    """
    signals_dir = Path(signals_dir)
    pred_path = signals_dir / "predicted_returns.npy"
    labels_path = signals_dir / "regression_labels.npy"
    metadata_path = signals_dir / "signal_metadata.json"
    for path in (pred_path, labels_path, metadata_path):
        if not path.exists():
            raise FileNotFoundError(
                f"from_signal_dir: missing required file {path}. Expected "
                f"predicted_returns.npy + regression_labels.npy + "
                f"signal_metadata.json in {signals_dir}."
            )

    predicted = np.load(pred_path)
    labels = np.load(labels_path)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Build metadata overlay from signal_metadata.json top-level keys.
    overlay: Dict[str, Any] = {
        "compatibility_fingerprint": metadata.get("compatibility_fingerprint"),
        "model_config_hash": metadata.get("model_config_hash"),
        "normalization_stats_sha256": metadata.get("normalization_stats_sha256"),
        # Phase V.1 L1.2: signal_export_output_dir is run-time-captured;
        # fall back to the actual signals_dir if metadata didn't capture it.
        "signal_export_output_dir": (
            metadata.get("signal_export_output_dir") or str(signals_dir.resolve())
        ),
        "experiment_id": metadata.get("experiment_id", ""),
        "fingerprint": metadata.get("fingerprint", ""),
        "model_type": metadata.get("model_type", ""),
    }

    # If metadata's primary_horizon_idx differs from config, log warning.
    metadata_horizon_idx = metadata.get("horizon_idx")
    if (
        metadata_horizon_idx is not None
        and metadata_horizon_idx != config.primary_horizon_idx
    ):
        LOGGER.warning(
            "from_signal_dir: metadata horizon_idx=%s differs from "
            "config.primary_horizon_idx=%s. Using config value (caller intent); "
            "verify alignment with signal_metadata to avoid silent drift.",
            metadata_horizon_idx, config.primary_horizon_idx,
        )

    return compute_ci(predicted, labels, config, metadata_overlay=overlay)


def compute_test_metrics_ci_for_experiment(
    experiment_dir: Path,
    output_path: Optional[Path] = None,
    config: Optional[TestMetricsCIConfig] = None,
    skip_if_exists: bool = True,
) -> TestMetricsCIArtifact:
    """Orchestration entry: read experiment dir, compute CI, optionally save.

    Cache-hit semantics:
      - If ``skip_if_exists=True`` AND ``output_path`` exists AND parses as
        a valid ``TestMetricsCIArtifact``, returns the existing artifact
        without recompute (idempotency).
      - On invalid existing artifact (corrupt JSON, schema-version drift),
        logs a warning and recomputes.

    Args:
      experiment_dir: Path to ``outputs/experiments/<exp_name>/`` directory.
      output_path: Where to save the artifact JSON. If None, defaults to
        ``experiment_dir/test_metrics_ci_v1.json``.
      config: ``TestMetricsCIConfig``. If None, uses dataclass defaults.
      skip_if_exists: If True (default), skip recompute on cache-hit.

    Returns:
      ``TestMetricsCIArtifact``.
    """
    experiment_dir = Path(experiment_dir)
    if config is None:
        config = TestMetricsCIConfig()
    if output_path is None:
        output_path = experiment_dir / "test_metrics_ci_v1.json"

    if skip_if_exists and output_path.exists():
        try:
            existing = TestMetricsCIArtifact.load(output_path)
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            LOGGER.warning(
                "compute_test_metrics_ci_for_experiment: existing artifact at "
                "%s is invalid (%s); recomputing.", output_path, exc,
            )
        else:
            # Round 1 §1 MEDIUM cache-hit config-drift detection: silent
            # stale-artifact return is §8 violation if config differs.
            # Re-derive effective_block_length to compare against existing.
            #
            # NOTE: we re-derive here rather than calling _resolve_block_length
            # because that would require knowing n_samples — which we'd only
            # know by loading the signals. Compare config.block_length AND
            # the resolved value lazily via the existing artifact's recorded
            # block_length (which encodes the resolution outcome).
            config_metric_set = set(config.metric_names)
            existing_metric_set = set(existing.metrics.keys())
            drift_reasons: list[str] = []
            if existing.n_bootstraps != config.n_bootstraps:
                drift_reasons.append(
                    f"n_bootstraps {existing.n_bootstraps}→{config.n_bootstraps}"
                )
            if existing.ci != config.ci:
                drift_reasons.append(f"ci {existing.ci}→{config.ci}")
            if existing.seed != config.seed:
                drift_reasons.append(f"seed {existing.seed}→{config.seed}")
            if config_metric_set != existing_metric_set:
                only_config = sorted(config_metric_set - existing_metric_set)
                only_existing = sorted(existing_metric_set - config_metric_set)
                drift_reasons.append(
                    f"metric_names diff (+{only_config}, -{only_existing})"
                )
            # block_length: if config.block_length is explicit, must match;
            # if None (auto-derive), accept existing value (auto-derive is
            # deterministic given n_samples, which is consistent on re-run).
            if (
                config.block_length is not None
                and existing.block_length != config.block_length
            ):
                drift_reasons.append(
                    f"block_length {existing.block_length}→{config.block_length}"
                )
            if drift_reasons:
                LOGGER.warning(
                    "compute_test_metrics_ci_for_experiment: existing artifact "
                    "at %s has config drift (%s); recomputing.",
                    output_path, ", ".join(drift_reasons),
                )
            else:
                LOGGER.info(
                    "compute_test_metrics_ci_for_experiment: cache-hit for %s "
                    "(content_hash=%s, %d metrics)",
                    experiment_dir.name, existing.content_hash()[:16],
                    len(existing.metrics),
                )
                return existing

    signals_dir = experiment_dir / "signals" / "test"
    artifact = from_signal_dir(signals_dir, config)
    artifact.save(output_path)
    LOGGER.info(
        "compute_test_metrics_ci_for_experiment: saved %s "
        "(content_hash=%s, n_test_samples=%d)",
        output_path.name, artifact.content_hash()[:16], artifact.n_test_samples,
    )
    return artifact
