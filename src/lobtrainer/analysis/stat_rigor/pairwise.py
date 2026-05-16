"""Phase 2 P2.C — K-way pairwise-comparison via paired moving-block bootstrap.

Loads K experiment signal-export dirs, verifies all K share
``compatibility_fingerprint`` + paired-data SHA-256 invariants, runs
``hft_metrics.pairwise.pairwise_paired_bootstrap_compare`` for K*(K-1)/2
pairs with BH FDR correction, persists as ``PairwiseCompareArtifact`` to
``outputs/comparisons/<sorted_exp_ids>/pairwise_compare_v1.json``.

Architecture:
  - Per hft-rules §0 reuse-first: ``compare_k_way`` delegates to
    ``hft_metrics.pairwise.pairwise_paired_bootstrap_compare`` SSoT;
    statistic_fns wrap the same hft_metrics IC/regression primitives
    used by P2.A.
  - Per hft-rules §8 fail-loud: validates inputs at boundary
    (paired-labels SHA-256 + finite + shared compat_fp + max_drop_frac
    threshold) BEFORE bootstrap loop.
  - Per #PY-67 mitigation: operates on STORED SIGNAL ARRAYS — does NOT
    reload checkpoints.
  - Per Round 1+2 architectural critique (2026-05-07):
    + K-arbitrary canonical entry (NOT K=2 only)
    + Effect size in artifact (statistic_a, statistic_b, delta + CI)
    + Strict ``compat_fp`` invariant across all K
    + Phase Y composability via parallel-indexed parent_* tuples
  - Helper duplication note: this library reimplements paired-data
    helpers (``_PairedSignals``, ``_assert_paired_labels``,
    ``_drop_nonfinite_paired``) that have analogues in
    ``hft_ops.ledger.statistical_compare`` for the sweep-id
    orchestration use-case. Documented-intentional-duplicate per
    dependency-graph: lob-model-trainer imports from hft_metrics (leaf)
    and hft_contracts but NOT from hft-ops (downstream). Mirror of the
    ``block_index_permutations`` precedent in
    ``lobtrainer.training.importance.permutation``.

References:
  - Künsch, H. R. (1989). The jackknife and the bootstrap for general
    stationary observations. Annals of Statistics 17:1217-1241.
  - Politis & Romano (1994). The Stationary Bootstrap. JASA 89:1303-1313.
  - Efron & Tibshirani (1993). An Introduction to the Bootstrap. Ch 15
    eq 15.22.
  - Benjamini & Hochberg (1995). Controlling the false discovery rate.
    JRSS B 57:289-300.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, List, Optional, Sequence, Tuple

import numpy as np

from hft_contracts.pairwise_compare_artifact import (
    PAIRWISE_COMPARE_SCHEMA_VERSION,
    PairwiseCompareArtifact,
    PairwiseResultRecord,
)
from hft_metrics import ic as ic_module
from hft_metrics.pairwise import pairwise_paired_bootstrap_compare
from hft_metrics.regression import (
    directional_accuracy,
    mean_absolute_error,
    profitable_accuracy,
    r_squared,
    root_mean_squared_error,
)


LOGGER = logging.getLogger(__name__)


# Registry of metric_name → statistic_fn(y_true, y_pred) -> float.
# Mirrors P2.A's _METRIC_REGISTRY but P2.C operates on a SINGLE metric
# at a time (the "comparison axis"). Default is spearman_ic per Plan v4.
# The hft_metrics.pairwise primitive accepts statistic_fn(x, y) with
# convention x=labels (paired ground-truth), y=predictions.
_METRIC_REGISTRY: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "spearman_ic": lambda yt, yp: float(ic_module.spearman_ic(yt, yp)[0]),
    "pearson_r": lambda yt, yp: float(ic_module.pearson_r(yt, yp)[0]),
    "r_squared": r_squared,
    "mean_absolute_error": mean_absolute_error,
    "root_mean_squared_error": root_mean_squared_error,
    "directional_accuracy": directional_accuracy,
    "profitable_accuracy": profitable_accuracy,
}

DEFAULT_METRIC_NAME = "spearman_ic"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairwiseCompareConfig:
    """Configuration for K-way pairwise-comparison.

    Frozen — produced once by orchestrator; consumers read it.

    Defaults are Plan v4 §4.3 + Round 1 P2.A precedent:
      - ``n_bootstraps=10000`` (matches P2.A; Plan v4 v1→v2 correction)
      - ``block_length=None`` (auto-derive ``ceil(n^(1/3))`` per
        Politis-Romano 1994)
      - ``alpha=0.05`` (NOT ci=0.95 — pairwise primitive uses alpha)
      - ``seed=42`` (deterministic)
      - ``metric_name="spearman_ic"`` (default per Plan v4 §4.3)
      - ``primary_horizon_idx=0`` (HMHP-R slicing; matches P2.A)
      - ``max_drop_frac=0.05`` (max paired NaN-row drop fraction;
        mirrors hft_ops compare_sweep_statistical default)

    Per hft-rules §5 fail-fast: ``__post_init__`` rejects degenerate
    parameter combinations.
    """

    # Pytest-discovery suppression
    __test__: ClassVar[bool] = False

    n_bootstraps: int = 10_000
    block_length: Optional[int] = None
    alpha: float = 0.05
    seed: int = 42
    metric_name: str = DEFAULT_METRIC_NAME
    primary_horizon_idx: int = 0
    max_drop_frac: float = 0.05

    def __post_init__(self) -> None:
        if self.n_bootstraps < 100:
            raise ValueError(
                f"PairwiseCompareConfig: n_bootstraps={self.n_bootstraps} "
                f"< 100 — too few for stable bootstrap distribution"
            )
        if self.block_length is not None and self.block_length < 2:
            raise ValueError(
                f"PairwiseCompareConfig: block_length={self.block_length} "
                f"< 2 — degenerate; block resampling requires >= 2"
            )
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(
                f"PairwiseCompareConfig: alpha={self.alpha} not in (0, 1). "
                f"Common: 0.01, 0.05, 0.10. Note: this library uses alpha "
                f"(NOT ci=0.95) to match hft_metrics.pairwise primitive."
            )
        if self.metric_name not in _METRIC_REGISTRY:
            raise ValueError(
                f"PairwiseCompareConfig: metric_name={self.metric_name!r} "
                f"not in registry {sorted(_METRIC_REGISTRY.keys())}"
            )
        if self.primary_horizon_idx < 0:
            raise ValueError(
                f"PairwiseCompareConfig: primary_horizon_idx="
                f"{self.primary_horizon_idx} < 0"
            )
        if not 0.0 <= self.max_drop_frac < 1.0:
            raise ValueError(
                f"PairwiseCompareConfig: max_drop_frac={self.max_drop_frac} "
                f"not in [0.0, 1.0)"
            )


# ---------------------------------------------------------------------------
# Internal helpers (intentional duplicate of hft-ops compare_sweep_statistical
# helpers — different consumer scope; lobtrainer cannot import from hft-ops
# per dep-direction; documented per §0 reuse-first audit)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PairedSignals:
    """Per-treatment loaded signals + metadata for paired comparison.

    Frozen. Produced by ``_load_paired_signals_from_dir``; consumed by
    ``compare_k_way``.

    Mirrors hft-ops ``_RecordSignals`` semantics but anchored to
    signal-export DIR paths (NOT ledger record entries) — different
    consumer scope per dep-direction.

    Fields:
      label: Human-readable treatment label.
      experiment_id: Source experiment_id (from signal_metadata).
      compatibility_fingerprint: 64-hex SHA-256 (or empty string for
        sklearn pre-Phase-Q.6.5).
      model_config_hash: Optional 64-hex SHA-256 (None for pre-Phase-X.1).
      signal_dir: Source directory path (for diagnostic/provenance).
      predicted_returns_full: Loaded predicted_returns.npy (raw shape).
      regression_labels_full: Loaded regression_labels.npy (raw shape).
      regression_labels_sha256: SHA-256 of regression_labels (post horizon-slice
        — used to verify all K treatments share paired-data).
    """

    label: str
    experiment_id: str
    compatibility_fingerprint: str
    model_config_hash: Optional[str]
    signal_dir: Path
    predicted_returns_full: np.ndarray
    regression_labels_full: np.ndarray
    regression_labels_sha256: str


def _slice_to_primary_horizon(
    arr: np.ndarray,
    primary_horizon_idx: int,
    array_role: str,
) -> np.ndarray:
    """Slice array to primary horizon if multi-horizon.

    TLOB (N,) → identity. HMHP-R (N, H) → slice [:, primary_horizon_idx].

    Mirrors P2.A's _slice_to_primary_horizon (intentional duplicate per
    §0 reuse-first audit — different orchestration; same primitive).
    """
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        if primary_horizon_idx >= arr.shape[1]:
            raise IndexError(
                f"{array_role}: primary_horizon_idx={primary_horizon_idx} "
                f">= shape[1]={arr.shape[1]} — out-of-bounds"
            )
        return arr[:, primary_horizon_idx]
    raise ValueError(
        f"{array_role}: unexpected shape {arr.shape}. Expected (N,) or (N, H)."
    )


def _load_paired_signals_from_dir(
    signal_dir: Path,
    label: str,
    primary_horizon_idx: int,
) -> _PairedSignals:
    """Load signal arrays + metadata from a single signal-export dir.

    Per hft-rules §8 fail-loud: missing files / malformed metadata raise.
    """
    signal_dir = Path(signal_dir)
    pred_path = signal_dir / "predicted_returns.npy"
    labels_path = signal_dir / "regression_labels.npy"
    metadata_path = signal_dir / "signal_metadata.json"
    for p in (pred_path, labels_path, metadata_path):
        if not p.exists():
            raise FileNotFoundError(
                f"_load_paired_signals_from_dir(label={label!r}): "
                f"missing required file {p}. Expected predicted_returns.npy "
                f"+ regression_labels.npy + signal_metadata.json."
            )

    predicted_full = np.load(pred_path, allow_pickle=False)
    labels_full = np.load(labels_path, allow_pickle=False)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Slice horizon EARLY — labels SHA-256 is computed post-slice so all K
    # treatments are checked on the SAME slice axis.
    labels_1d = _slice_to_primary_horizon(
        labels_full, primary_horizon_idx, f"{label}.labels"
    )
    labels_sha256 = hashlib.sha256(labels_1d.tobytes()).hexdigest()

    compat_fp = metadata.get("compatibility_fingerprint") or ""
    model_cfg_hash = metadata.get("model_config_hash")  # may be None

    return _PairedSignals(
        label=label,
        experiment_id=metadata.get("experiment_id", "") or label,
        compatibility_fingerprint=compat_fp,
        model_config_hash=model_cfg_hash,
        signal_dir=signal_dir,
        predicted_returns_full=predicted_full,
        regression_labels_full=labels_full,
        regression_labels_sha256=labels_sha256,
    )


def _assert_paired_labels(loaded: List[_PairedSignals]) -> None:
    """Verify all K treatments share paired-data invariants per §8.

    Checks:
      - All K must have non-empty compat_fp
      - All K must share the SAME compat_fp
      - All K must share the SAME labels SHA-256 (post horizon-slice)

    Mirrors hft_ops compare_sweep_statistical._assert_paired_labels.
    """
    if len(loaded) < 2:
        raise ValueError(
            f"_assert_paired_labels: K={len(loaded)} < 2; pairwise comparison "
            f"requires at least 2 treatments"
        )

    # All K must have non-empty compat_fp
    for sig in loaded:
        if not sig.compatibility_fingerprint:
            raise ValueError(
                f"_assert_paired_labels: treatment {sig.label!r} (dir="
                f"{sig.signal_dir}) lacks compatibility_fingerprint. "
                f"Cannot verify paired-data invariant. This is the "
                f"#PY-68 risk realized for sklearn pre-Phase-Q.6.5 "
                f"experiments — re-export via canonical path or skip "
                f"this experiment."
            )

    # All K must share compat_fp
    compat_fps = {sig.compatibility_fingerprint for sig in loaded}
    if len(compat_fps) > 1:
        raise ValueError(
            f"_assert_paired_labels: K={len(loaded)} treatments do NOT "
            f"share compatibility_fingerprint (got {len(compat_fps)} "
            f"unique values: "
            f"{sorted({sig.label: sig.compatibility_fingerprint[:16] + '...' for sig in loaded}.items())}"
            f"). Pairwise comparison requires shared paired-data."
        )

    # All K must share labels SHA-256
    labels_shas = {sig.regression_labels_sha256 for sig in loaded}
    if len(labels_shas) > 1:
        raise ValueError(
            f"_assert_paired_labels: K={len(loaded)} treatments do NOT "
            f"share regression_labels.npy SHA-256 (paired-data integrity "
            f"violation). Got {len(labels_shas)} unique label SHAs. "
            f"Treatments must consume identical labels for paired bootstrap "
            f"to be valid. Investigate signal-export pipeline."
        )


def _drop_nonfinite_paired(
    x: np.ndarray,
    Y: np.ndarray,
    max_drop_frac: float,
) -> Tuple[np.ndarray, np.ndarray, int, int, float]:
    """Drop rows where x or any column of Y is non-finite. Per-row paired drop.

    Returns:
      (x_clean, Y_clean, n_dropped, n_raw, drop_fraction)

    Raises:
      ValueError: If drop_fraction > max_drop_frac (per §8 fail-loud
        threshold; mirrors hft_ops compare_sweep_statistical pattern).

    Mirrors hft_ops compare_sweep_statistical's NaN-row drop logic
    (intentional duplicate per §0 reuse-first audit).
    """
    n_raw = len(x)
    if Y.shape[0] != n_raw:
        raise ValueError(
            f"_drop_nonfinite_paired: x.shape={x.shape} != Y.shape[0]="
            f"{Y.shape[0]}; arrays must be paired (same N)"
        )
    mask = np.isfinite(x) & np.isfinite(Y).all(axis=1)
    n_clean = int(mask.sum())
    n_dropped = n_raw - n_clean
    drop_frac = n_dropped / n_raw if n_raw > 0 else 0.0

    if drop_frac > max_drop_frac:
        raise ValueError(
            f"_drop_nonfinite_paired: paired NaN-row drop fraction "
            f"{drop_frac:.4f} exceeds max_drop_frac={max_drop_frac}. "
            f"Investigate upstream producer (signal exporter or trainer "
            f"prediction pipeline) — this many NaN/Inf rows in paired "
            f"data is suspicious. n_dropped={n_dropped}/n_raw={n_raw}."
        )

    return x[mask], Y[mask, :], n_dropped, n_raw, drop_frac


def _resolve_block_length_for_pairwise(
    n_samples: int,
    config_block_length: Optional[int],
) -> Tuple[int, str]:
    """Resolve effective block_length + source string for pairwise.

    Mirrors P2.A's _resolve_block_length (intentional duplicate per §0
    reuse-first audit — different orchestration). Includes the same
    block_length>=n_samples degenerate guard from Round 1 P2.A.
    """
    if config_block_length is not None:
        effective = int(config_block_length)
        source = f"explicit override (config.block_length={config_block_length})"
    else:
        effective = int(math.ceil(n_samples ** (1.0 / 3.0)))
        source = (
            f"auto-derive ceil(n^(1/3))={effective} per Politis-Romano (1994) "
            f"for n_samples={n_samples}"
        )

    if effective >= n_samples:
        raise ValueError(
            f"_resolve_block_length_for_pairwise: effective_block_length="
            f"{effective} >= n_samples={n_samples}. Block resampling "
            f"produces degenerate single-block bootstrap. Provide more "
            f"samples or override block_length to a value < n_samples."
        )
    return effective, source


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compare_k_way(
    treatments: Sequence[Tuple[str, np.ndarray, np.ndarray]],
    config: PairwiseCompareConfig,
    parent_metadata: Optional[List[Dict[str, Any]]] = None,
) -> PairwiseCompareArtifact:
    """K-way pairwise comparison via paired moving-block bootstrap.

    Args:
      treatments: List of (label, predicted_returns, labels) triples.
        K = len(treatments) must be >= 2. All ``labels`` arrays must
        be byte-identical (paired-data invariant — verify upstream).
        ``predicted_returns`` arrays must all have shape (N,) post-slice.
      config: PairwiseCompareConfig.
      parent_metadata: Optional list of K dicts with keys
        ``compatibility_fingerprint``, ``model_config_hash``,
        ``experiment_id`` for Phase Y composability projection.

    Returns:
      PairwiseCompareArtifact with K*(K-1)/2 PairwiseResultRecord entries.

    Raises:
      ValueError: If K < 2, paired-data invariant violated, drop fraction
        exceeds threshold, or block_length degenerate.
    """
    if len(treatments) < 2:
        raise ValueError(
            f"compare_k_way: K={len(treatments)} < 2; pairwise comparison "
            f"requires at least 2 treatments"
        )

    K = len(treatments)
    labels_list = [labels for (_, _, labels) in treatments]
    pred_list = [preds for (_, preds, _) in treatments]
    treatment_labels = tuple(label for (label, _, _) in treatments)

    # Verify all K share labels (paired-data invariant)
    labels_first = labels_list[0]
    for k, labels_k in enumerate(labels_list[1:], start=1):
        if not np.array_equal(labels_first, labels_k):
            raise ValueError(
                f"compare_k_way: treatment {k} ({treatment_labels[k]!r}) "
                f"labels differ from treatment 0 ({treatment_labels[0]!r}). "
                f"Paired bootstrap requires byte-identical labels across "
                f"all K treatments."
            )
    x = labels_first
    n_raw = len(x)

    # Stack predictions → Y[N, K]
    Y = np.column_stack(pred_list).astype(np.float64)

    # Drop NaN rows paired
    x_clean, Y_clean, n_dropped, n_samples_raw, drop_frac = _drop_nonfinite_paired(
        x, Y, config.max_drop_frac
    )
    n_samples_paired = n_samples_raw - n_dropped

    # Resolve block_length post-drop (uses cleaned size for auto-derive)
    effective_block_length, block_length_source = _resolve_block_length_for_pairwise(
        n_samples_paired, config.block_length
    )

    # Dispatch statistic_fn from registry
    statistic_fn = _METRIC_REGISTRY[config.metric_name]

    # Run pairwise primitive (returns List[PairwiseResult])
    primitive_results = pairwise_paired_bootstrap_compare(
        x=x_clean.astype(np.float64),
        Y=Y_clean,
        statistic_fn=statistic_fn,
        n_bootstraps=config.n_bootstraps,
        block_length=config.block_length,  # None → primitive auto-derives
        alpha=config.alpha,
        seed=config.seed,
    )

    # Convert primitive PairwiseResults → artifact PairwiseResultRecords
    pair_records = tuple(
        PairwiseResultRecord.from_hft_metrics_result(r, treatment_labels)
        for r in primitive_results
    )

    # Build parent_* tuples
    if parent_metadata is None:
        parent_metadata = [{} for _ in range(K)]
    parent_experiment_ids = tuple(
        meta.get("experiment_id", treatment_labels[i])
        for i, meta in enumerate(parent_metadata)
    )
    parent_compat_fps = tuple(
        meta.get("compatibility_fingerprint", "") for meta in parent_metadata
    )
    parent_model_cfg_hashes = tuple(
        meta.get("model_config_hash") for meta in parent_metadata  # Optional
    )

    # Compute paired_compat_fp + paired_labels_sha256 invariants
    unique_compat_fps = {fp for fp in parent_compat_fps if fp}
    if len(unique_compat_fps) != 1:
        raise ValueError(
            f"compare_k_way: K={K} treatments do NOT share compatibility_"
            f"fingerprint (or some are empty). Got {len(unique_compat_fps)} "
            f"unique non-empty values. Pairwise comparison requires shared "
            f"paired-data per §8."
        )
    paired_compat_fp = unique_compat_fps.pop()
    paired_labels_sha = hashlib.sha256(x_clean.tobytes()).hexdigest()

    return PairwiseCompareArtifact(
        schema_version=PAIRWISE_COMPARE_SCHEMA_VERSION,
        method="paired_block_bootstrap",
        metric_name=config.metric_name,
        block_length=effective_block_length,
        block_length_source=block_length_source,
        n_bootstraps=config.n_bootstraps,
        alpha=config.alpha,
        seed=config.seed,
        n_treatments=K,
        n_samples_paired=n_samples_paired,
        n_samples_raw=n_samples_raw,
        n_dropped_nonfinite=n_dropped,
        drop_fraction=drop_frac,
        primary_horizon_idx=config.primary_horizon_idx,
        parent_experiment_ids=parent_experiment_ids,
        parent_compatibility_fingerprints=parent_compat_fps,
        parent_model_config_hashes=parent_model_cfg_hashes,
        paired_compat_fingerprint=paired_compat_fp,
        paired_labels_sha256=paired_labels_sha,
        pairs=pair_records,
        treatment_labels=treatment_labels,
        timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def compare_pair(
    label_a: str, predicted_a: np.ndarray, labels_a: np.ndarray,
    label_b: str, predicted_b: np.ndarray, labels_b: np.ndarray,
    config: PairwiseCompareConfig,
    parent_metadata: Optional[List[Dict[str, Any]]] = None,
) -> PairwiseCompareArtifact:
    """K=2 sugar over compare_k_way.

    Per Round 1 architectural critique: K=2 is special case of K-way;
    BH FDR at K=2 is degenerate (q-value ≡ raw p-value).
    """
    return compare_k_way(
        treatments=[
            (label_a, predicted_a, labels_a),
            (label_b, predicted_b, labels_b),
        ],
        config=config,
        parent_metadata=parent_metadata,
    )


def from_signal_dirs(
    signal_dirs: Sequence[Tuple[str, Path]],
    config: PairwiseCompareConfig,
) -> PairwiseCompareArtifact:
    """Load K signal-export dirs + verify paired invariants + run K-way.

    Args:
      signal_dirs: List of (label, signal_dir_path) tuples. K >= 2.
      config: PairwiseCompareConfig.

    Returns:
      PairwiseCompareArtifact.

    Raises:
      FileNotFoundError: Any required file (predicted_returns.npy /
        regression_labels.npy / signal_metadata.json) missing.
      ValueError: Paired-data invariant violation (compat_fp not shared,
        labels SHA-256 not shared, drop_fraction exceeds threshold).
    """
    if len(signal_dirs) < 2:
        raise ValueError(
            f"from_signal_dirs: K={len(signal_dirs)} < 2; pairwise "
            f"comparison requires at least 2 signal-dir treatments"
        )

    # Load all K signal dirs
    loaded = [
        _load_paired_signals_from_dir(signal_dir, label, config.primary_horizon_idx)
        for (label, signal_dir) in signal_dirs
    ]

    # Verify paired-data invariants BEFORE the bootstrap loop
    _assert_paired_labels(loaded)

    # Build treatments list (slice predictions to primary horizon)
    treatments: List[Tuple[str, np.ndarray, np.ndarray]] = []
    parent_metadata: List[Dict[str, Any]] = []
    for sig in loaded:
        pred_1d = _slice_to_primary_horizon(
            sig.predicted_returns_full,
            config.primary_horizon_idx,
            f"{sig.label}.predicted_returns",
        )
        labels_1d = _slice_to_primary_horizon(
            sig.regression_labels_full,
            config.primary_horizon_idx,
            f"{sig.label}.regression_labels",
        )
        treatments.append((sig.label, pred_1d, labels_1d))
        parent_metadata.append({
            "experiment_id": sig.experiment_id,
            "compatibility_fingerprint": sig.compatibility_fingerprint,
            "model_config_hash": sig.model_config_hash,
        })

    return compare_k_way(treatments, config, parent_metadata=parent_metadata)


def compute_pairwise_compare_for_experiments(
    signal_dirs: Sequence[Tuple[str, Path]],
    output_dir: Optional[Path] = None,
    config: Optional[PairwiseCompareConfig] = None,
    skip_if_exists: bool = True,
) -> PairwiseCompareArtifact:
    """Orchestration entry: load K signal dirs, compute K-way comparison, save.

    Output path: ``output_dir/pairwise_compare_v1.json``. If ``output_dir``
    is None, defaults to ``outputs/comparisons/<sorted_labels_joined>/``.

    Cache-hit: if output file exists AND parses + matches config, skip.
    """
    if config is None:
        config = PairwiseCompareConfig()
    if output_dir is None:
        # Symmetric default per Round 1 architectural critique #6
        sorted_labels = sorted(label for (label, _) in signal_dirs)
        output_dir = (
            Path("outputs/comparisons") / "_vs_".join(sorted_labels)
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "pairwise_compare_v1.json"

    if skip_if_exists and output_path.exists():
        try:
            existing = PairwiseCompareArtifact.load(output_path)
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            LOGGER.warning(
                "compute_pairwise_compare_for_experiments: existing artifact "
                "at %s is invalid (%s); recomputing.", output_path, exc,
            )
        else:
            # Cache-drift detection (mirror P2.A pattern)
            drift_reasons: List[str] = []
            if existing.n_bootstraps != config.n_bootstraps:
                drift_reasons.append(
                    f"n_bootstraps {existing.n_bootstraps}→{config.n_bootstraps}"
                )
            if existing.alpha != config.alpha:
                drift_reasons.append(f"alpha {existing.alpha}→{config.alpha}")
            if existing.seed != config.seed:
                drift_reasons.append(f"seed {existing.seed}→{config.seed}")
            if existing.metric_name != config.metric_name:
                drift_reasons.append(
                    f"metric_name {existing.metric_name}→{config.metric_name}"
                )
            if (
                config.block_length is not None
                and existing.block_length != config.block_length
            ):
                drift_reasons.append(
                    f"block_length {existing.block_length}→{config.block_length}"
                )
            if drift_reasons:
                LOGGER.warning(
                    "compute_pairwise_compare_for_experiments: existing "
                    "artifact at %s has config drift (%s); recomputing.",
                    output_path, ", ".join(drift_reasons),
                )
            else:
                LOGGER.info(
                    "compute_pairwise_compare_for_experiments: cache-hit for "
                    "K=%d (content_hash=%s)",
                    existing.n_treatments, existing.content_hash()[:16],
                )
                return existing

    artifact = from_signal_dirs(signal_dirs, config)
    artifact.save(output_path)
    LOGGER.info(
        "compute_pairwise_compare_for_experiments: saved K=%d comparison to "
        "%s (content_hash=%s, n_pairs=%d)",
        artifact.n_treatments, output_path,
        artifact.content_hash()[:16], len(artifact.pairs),
    )
    return artifact
