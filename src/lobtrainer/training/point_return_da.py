"""Point-return-DA tripwire scalars (Phase 3c — the E8 honesty-rail producer).

Computes three test-split scalars that the hft-ops ``post_training_gate``
consumes (``_check_e8_tripwire``) to catch the E8 trap (FINDING-001 /
FINDING-008): a model trained on the ``smoothed_return`` label can score a high
IC on that label yet have directional accuracy below a coin on the TRADEABLE
point-to-point return.

The three scalars (returned WITHOUT the ``test_`` prefix — BOTH producer paths,
``SimpleModelTrainer.save`` and ``scripts/train.py::_dump_test_metrics``, prepend
``test_`` themselves, yielding the final ``test_point_return_*`` keys):

    point_return_da   — directional accuracy of sign(pred) vs sign(point_return)
                        on the primary horizon, over non-zero-point-return samples.
    point_return_n    — count of those non-zero-point-return samples.
    point_return_rho1 — lag-1 autocorrelation of the per-sample sign-agreement
                        indicator. Overlapping point returns are autocorrelated,
                        so the gate deflates the effective N by this (a hard
                        DA<=0.50 cut would fire on noise — see FINDING-001).

The point return is RE-DERIVED from the split's ``forward_prices`` via
``LabelFactory.point_return`` — deliberately NOT the ``smoothed_return`` label
the model trained on (that orthogonality IS the E8 trap).

OBSERVATION TIER: callers MUST wrap this in try/except and never let a diagnostic
kill a training run. It returns ``None`` on expected skips (classification-only
export, no/un-exported forward_prices, out-of-range primary horizon, degenerate
split) and RAISES ``ValueError`` only on a real alignment bug (re-derived
point-return rows != prediction count) — which the caller logs and swallows.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

from hft_contracts.label_factory import ForwardPriceContract, LabelFactory
from hft_metrics.acf import autocorrelation
from lobtrainer.data.horizons_resolver import resolve_horizons_from_export

logger = logging.getLogger(__name__)

# The final gate-consumed keys (after each producer path prepends ``test_``).
POINT_RETURN_DA_KEYS = ("point_return_da", "point_return_n", "point_return_rho1")


def compute_point_return_da_scalars(
    data_dir: Union[str, Path],
    y_pred: np.ndarray,
    *,
    primary_horizon_idx: Optional[int],
    split: str = "test",
) -> Optional[Dict[str, float]]:
    """Compute the E8 point-return-DA tripwire scalars for one split.

    Args:
        data_dir: Corpus root (parent of train/val/test).
        y_pred: Model predictions for ``split``, in on-disk sorted-day,
            within-day-sequential order (the unshuffled loader order). Shape [N].
        primary_horizon_idx: Index into the export's horizon list for the traded
            horizon. None -> 0 (first horizon).
        split: Split name (default "test").

    Returns:
        ``{"point_return_da", "point_return_n", "point_return_rho1"}`` (unprefixed),
        or ``None`` on an expected skip (see the module docstring).

    Raises:
        ValueError: if the re-derived point-return row count does not match
            ``len(y_pred)`` — a structural alignment bug (hft-rules §9). The
            caller logs and swallows (observation tier).
    """
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if y_pred.size == 0:
        return None

    split_dir = Path(data_dir) / split

    # Resolve the traded horizon VALUE from the export (SSoT). FileNotFoundError
    # => classification-only / corrupt export => expected skip.
    try:
        horizons = resolve_horizons_from_export(data_dir, split)
    except FileNotFoundError:
        logger.debug("E8 tripwire: no *_horizons.json in %s; skipping", split_dir)
        return None

    idx = 0 if primary_horizon_idx is None else int(primary_horizon_idx)
    if not (0 <= idx < len(horizons)):
        logger.debug(
            "E8 tripwire: primary_horizon_idx=%s out of range for horizons=%s; skipping",
            primary_horizon_idx, horizons,
        )
        return None
    horizon_value = horizons[idx]

    # Re-derive point returns per day in sorted-day order (matches the unshuffled
    # loader's prediction order; forward_prices is 1:1 with sequences on disk).
    meta_files = sorted(split_dir.glob("*_metadata.json"))
    if not meta_files:
        return None

    point_chunks = []
    for meta_file in meta_files:
        with open(meta_file) as f:
            metadata = json.load(f)
        try:
            contract = ForwardPriceContract.from_metadata(metadata)
        except KeyError:
            logger.debug(
                "E8 tripwire: forward_prices not exported for %s; skipping",
                meta_file.name,
            )
            return None
        day_stem = meta_file.name[: -len("_metadata.json")]
        fp_path = split_dir / f"{day_stem}_forward_prices.npy"
        if not fp_path.exists():
            logger.debug("E8 tripwire: missing %s; skipping", fp_path.name)
            return None
        fp = np.load(fp_path, allow_pickle=False)  # #PY-291 security lock
        try:
            point_day = LabelFactory.point_return(
                fp,
                horizon=horizon_value,
                smoothing_window=contract.smoothing_window_offset,
            )
        except ValueError:
            # Horizon exceeds the export's max — config/data inconsistency; skip.
            logger.debug(
                "E8 tripwire: horizon %s invalid for %s; skipping",
                horizon_value, meta_file.name,
            )
            return None
        point_chunks.append(np.asarray(point_day, dtype=np.float64).ravel())

    point_true = np.concatenate(point_chunks)

    # Alignment guard — fail-loud (the caller logs + swallows). A mismatch means
    # the re-derived rows and the predictions are NOT the same samples.
    if point_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"E8 tripwire alignment mismatch in {split_dir}: re-derived "
            f"{point_true.shape[0]} point-return rows but got {y_pred.shape[0]} "
            f"predictions. forward_prices/sequences row counts or day ordering "
            f"drifted (hft-rules §9 provenance-before-comparison)."
        )

    # Non-zero-point-return mask (the literal spec — intentionally differs from
    # hft_metrics.regression.directional_accuracy's dual-zero (yt!=0 & yp!=0) mask).
    mask = point_true != 0.0
    n = int(mask.sum())
    if n == 0:
        return None

    indicator = (np.sign(y_pred[mask]) == np.sign(point_true[mask])).astype(np.float64)
    da = float(indicator.mean())

    # Lag-1 autocorrelation of the hit-indicator (the overlap diagnostic). The
    # SSoT returns (acf_values, half_life, decay_rate); rho1 = acf_values[1].
    # Degenerate (constant / too-short) -> 0.0 (the gate then treats it as iid).
    rho1 = 0.0
    if indicator.size > 1:
        acf_values, _, _ = autocorrelation(indicator, max_lag=1)
        if len(acf_values) > 1 and np.isfinite(acf_values[1]):
            rho1 = float(acf_values[1])

    return {
        "point_return_da": da,
        "point_return_n": float(n),
        "point_return_rho1": rho1,
    }
