#!/usr/bin/env python3
# PRODUCTION INFRA — not an experiment
"""K-way pairwise comparison of trained experiments via paired moving-block bootstrap.

Phase 2 P2.C — Cyclelet B (2026-05-07). Thin CLI wrapper around
``lobtrainer.analysis.stat_rigor.pairwise.compute_pairwise_compare_for_experiments``.

Per hft-rules §4: this script is reserved for production infra (matches
the pattern of ``compute_test_metrics_ci.py``, ``train.py``,
``export_signals.py``). Reusable logic lives in the library; this is
a thin argparse + formatting layer.

Usage:
    # Compare K=3 TLOB family experiments (all share compat_fp 67c8ff36...):
    python scripts/compare_experiments_pairwise.py \\
        --experiment R9_TLOB_no_CVML:nvda_first_pytorch_v3p0 \\
        --experiment R10_TLOB_CVML:nvda_first_pytorch_v3p0_cvml \\
        --experiment R11_TLOB_GMADL_CVML:nvda_first_pytorch_v3p0_gmadl_cvml \\
        --metric spearman_ic

    # Custom output dir + n_bootstraps:
    python scripts/compare_experiments_pairwise.py \\
        --experiment A:exp_a --experiment B:exp_b \\
        --output-dir outputs/comparisons/custom \\
        --n-bootstraps 50000

Output: writes ``<output_dir>/pairwise_compare_v1.json``; prints summary
table to stdout.

Per hft-rules §8 fail-loud: missing files, malformed metadata, NaN inputs
above max_drop_frac, mismatched compat_fp/labels all raise — script exits
non-zero with diagnostic.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

from lobtrainer.analysis.stat_rigor.pairwise import (
    PairwiseCompareConfig,
    compute_pairwise_compare_for_experiments,
)


LOGGER = logging.getLogger("compare_experiments_pairwise")


def _parse_experiment_arg(arg: str, outputs_root: Path) -> Tuple[str, Path]:
    """Parse '--experiment LABEL:NAME' or '--experiment LABEL:PATH' format.

    If NAME contains '/', treat as explicit path. Otherwise resolve against
    ``<outputs_root>/<NAME>/signals/test/``.
    """
    if ":" not in arg:
        raise ValueError(
            f"--experiment value {arg!r} missing 'LABEL:' prefix. "
            f"Use 'LABEL:experiment_name' (e.g., 'R9:nvda_first_pytorch_v3p0')."
        )
    label, target = arg.split(":", 1)
    if not label or not target:
        raise ValueError(
            f"--experiment value {arg!r}: both LABEL and target must be non-empty"
        )
    if "/" in target or target.startswith("./") or target.startswith("../"):
        signal_dir = Path(target)
    else:
        signal_dir = outputs_root / target / "signals" / "test"
    return label, signal_dir


def _format_summary(artifact, K: int) -> str:
    """Render a formatted text table for a K-way comparison."""
    lines = [
        f"\n=== K={K} Pairwise Comparison ===",
        f"  metric: {artifact.metric_name}",
        f"  paired_compat_fp: {artifact.paired_compat_fingerprint[:16]}...",
        f"  paired_labels_sha: {artifact.paired_labels_sha256[:16]}...",
        f"  n_samples: {artifact.n_samples_paired}/{artifact.n_samples_raw} "
        f"(dropped {artifact.n_dropped_nonfinite}, "
        f"frac={artifact.drop_fraction:.4f})",
        f"  block_length: {artifact.block_length} ({artifact.block_length_source})",
        f"  n_bootstraps: {artifact.n_bootstraps}, alpha: {artifact.alpha}, "
        f"seed: {artifact.seed}",
        f"  content_hash: {artifact.content_hash()[:16]}...",
        "",
        f"  {'pair':<55} {'stat_a':>10} {'stat_b':>10} {'delta':>10} "
        f"{'CI_low':>10} {'CI_high':>10} {'p_raw':>8} {'p_BH':>8} {'sig?':>5}",
        "  " + "-" * 132,
    ]
    for p in artifact.pairs:
        sig = "YES" if p.p_value_bh < artifact.alpha else "no"
        pair_label = f"{p.treatment_a_label} vs {p.treatment_b_label}"
        lines.append(
            f"  {pair_label:<55} {p.statistic_a:>10.4f} {p.statistic_b:>10.4f} "
            f"{p.delta:>10.4f} {p.delta_ci_low:>10.4f} {p.delta_ci_high:>10.4f} "
            f"{p.p_value_raw:>8.4f} {p.p_value_bh:>8.4f} {sig:>5}"
        )
    return "\n".join(lines)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--experiment",
        action="append",
        default=[],
        required=True,
        metavar="LABEL:NAME",
        help=(
            "Experiment in 'LABEL:NAME' form. LABEL is human-readable "
            "(e.g., 'R9_TLOB_no_CVML'). NAME is resolved as "
            "<outputs-root>/NAME/signals/test/ (or explicit path if NAME "
            "contains '/'). Repeat at least 2 times for K>=2 comparison."
        ),
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("outputs/experiments"),
        help="Root dir for --experiment NAME resolution (default: outputs/experiments)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory for pairwise_compare_v1.json. Default: "
            "outputs/comparisons/<sorted_labels_joined>/"
        ),
    )
    parser.add_argument(
        "--metric",
        default="spearman_ic",
        choices=[
            "spearman_ic", "pearson_r", "r_squared", "mean_absolute_error",
            "root_mean_squared_error", "directional_accuracy", "profitable_accuracy",
        ],
        help="Comparison metric (default: spearman_ic)",
    )
    parser.add_argument(
        "--n-bootstraps", type=int, default=10_000,
        help="Bootstrap iterations (default: 10000 per Plan v4 §4.3)",
    )
    parser.add_argument(
        "--block-length", type=int, default=None,
        help="Block length (default: auto-derive ceil(n^(1/3)) per Politis-Romano)",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="Significance level in (0, 1) (default: 0.05)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Bootstrap RNG seed (default: 42)",
    )
    parser.add_argument(
        "--primary-horizon-idx", type=int, default=0,
        help="Horizon index for multi-horizon HMHP-R slicing (default: 0)",
    )
    parser.add_argument(
        "--max-drop-frac", type=float, default=0.05,
        help="Max paired NaN-row drop fraction; above this raises (default: 0.05)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force recompute even if artifact exists (skip cache)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log verbosity (default: INFO)",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if len(args.experiment) < 2:
        LOGGER.error(
            "Need at least 2 --experiment LABEL:NAME flags for pairwise "
            "comparison; got %d", len(args.experiment),
        )
        return 1

    signal_dirs = [
        _parse_experiment_arg(arg, args.outputs_root) for arg in args.experiment
    ]
    K = len(signal_dirs)

    config = PairwiseCompareConfig(
        n_bootstraps=args.n_bootstraps,
        block_length=args.block_length,
        alpha=args.alpha,
        seed=args.seed,
        metric_name=args.metric,
        primary_horizon_idx=args.primary_horizon_idx,
        max_drop_frac=args.max_drop_frac,
    )
    LOGGER.info("Computing K=%d pairwise comparison; metric=%s, n_bootstraps=%d",
                K, args.metric, args.n_bootstraps)

    try:
        artifact = compute_pairwise_compare_for_experiments(
            signal_dirs=signal_dirs,
            output_dir=args.output_dir,
            config=config,
            skip_if_exists=not args.force,
        )
    except Exception as exc:
        LOGGER.error("Pairwise comparison failed: %s", exc,
                     exc_info=args.log_level == "DEBUG")
        return 1

    print(_format_summary(artifact, K))
    LOGGER.info("OK: K=%d comparison saved with %d pairs", K, len(artifact.pairs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
