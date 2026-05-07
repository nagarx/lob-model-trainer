#!/usr/bin/env python3
# PRODUCTION INFRA — not an experiment
"""Compute bootstrap CI on test metrics for one or more trained experiments.

Phase 2 P2.A — Cyclelet B (2026-05-07). Thin CLI wrapper around
``lobtrainer.analysis.stat_rigor.ci.compute_test_metrics_ci_for_experiment``
per Plan v4 §5 library-first / §4 modularity.

Per hft-rules §4: this script is reserved for production infra (matches
the pattern of ``train.py``, ``export_signals.py``,
``precompute_norm_stats.py``). Reusable logic lives in the library
(``lobtrainer.analysis.stat_rigor.ci``) — this script is a thin argparse
+ formatting layer.

Usage:
    # Single experiment from default outputs/experiments/<name>/ tree
    python scripts/compute_test_metrics_ci.py --experiment nvda_first_pytorch_v3p0

    # Multiple experiments via repeated --experiment flag
    python scripts/compute_test_metrics_ci.py \\
        --experiment nvda_first_pytorch_v3p0 \\
        --experiment nvda_first_pytorch_v3p0_cvml \\
        --experiment nvda_first_pytorch_v3p0_gmadl_cvml

    # Custom experiment dir + n_bootstraps
    python scripts/compute_test_metrics_ci.py \\
        --experiment-dir outputs/experiments/nvda_first_hmhp_r_v3p0 \\
        --n-bootstraps 10000 --primary-horizon-idx 0

    # Force recompute (skip cache)
    python scripts/compute_test_metrics_ci.py \\
        --experiment R9 --force

Output: writes ``<experiment_dir>/test_metrics_ci_v1.json`` per experiment;
prints summary table to stdout.

Per hft-rules §8 fail-loud: missing files, malformed metadata, NaN inputs
all raise — script exits non-zero with diagnostic.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

from lobtrainer.analysis.stat_rigor.ci import (
    DEFAULT_METRIC_NAMES,
    TestMetricsCIConfig,
    compute_test_metrics_ci_for_experiment,
)


LOGGER = logging.getLogger("compute_test_metrics_ci")


def _resolve_experiment_dirs(args: argparse.Namespace) -> List[Path]:
    """Resolve --experiment NAME (positional path under outputs/experiments/)
    + --experiment-dir EXPLICIT_PATH into a list of experiment_dir paths."""
    experiment_dirs: List[Path] = []
    if args.experiment:
        for name in args.experiment:
            experiment_dirs.append(args.outputs_root / name)
    if args.experiment_dir:
        for explicit_path in args.experiment_dir:
            experiment_dirs.append(Path(explicit_path))
    if not experiment_dirs:
        raise ValueError(
            "Must specify at least one --experiment NAME or "
            "--experiment-dir PATH"
        )
    return experiment_dirs


def _format_summary(artifact, experiment_label: str) -> str:
    """Render a formatted text table for one artifact."""
    lines = [
        f"\n=== {experiment_label} ===",
        f"  experiment_id: {artifact.experiment_id or '(empty)'}",
        f"  model_type: {artifact.model_type or '(unset)'}",
        f"  compat_fingerprint: "
        f"{artifact.compatibility_fingerprint[:16] + '...' if artifact.compatibility_fingerprint else '(none)'}",
        f"  n_test_samples: {artifact.n_test_samples}",
        f"  block_length: {artifact.block_length} ({artifact.block_length_source})",
        f"  n_bootstraps: {artifact.n_bootstraps}, ci: {artifact.ci}, seed: {artifact.seed}",
        f"  content_hash: {artifact.content_hash()[:16]}...",
        "",
        f"  {'metric':<32}  {'point':>12}  {'ci_low':>12}  {'ci_high':>12}  {'width':>10}",
        "  " + "-" * 86,
    ]
    for name, bound in sorted(artifact.metrics.items()):
        width = bound.ci_high - bound.ci_low
        lines.append(
            f"  {name:<32}  {bound.point:>12.6f}  {bound.ci_low:>12.6f}  "
            f"{bound.ci_high:>12.6f}  {width:>10.6f}"
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
        metavar="NAME",
        help=(
            "Experiment name (resolved as <outputs-root>/<NAME>/). "
            "Repeat to process multiple experiments."
        ),
    )
    parser.add_argument(
        "--experiment-dir",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "Explicit experiment directory path. Repeat to process multiple."
        ),
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("outputs/experiments"),
        help=(
            "Root dir for --experiment NAME resolution. "
            "Default: outputs/experiments"
        ),
    )
    parser.add_argument(
        "--n-bootstraps",
        type=int,
        default=10_000,
        help="Number of bootstrap replicates (default: 10000 per Plan v4 §4.1)",
    )
    parser.add_argument(
        "--block-length",
        type=int,
        default=None,
        help=(
            "Block length for moving-block bootstrap. Default: None "
            "(auto-derive ceil(n^(1/3)) per Politis-Romano 1994)"
        ),
    )
    parser.add_argument(
        "--ci",
        type=float,
        default=0.95,
        help="Confidence level in (0.0, 1.0). Default: 0.95",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Bootstrap RNG seed (default: 42)",
    )
    parser.add_argument(
        "--primary-horizon-idx",
        type=int,
        default=0,
        help=(
            "Horizon index for multi-horizon (HMHP-R) slicing. Default: 0 "
            "(matches signal_metadata convention)"
        ),
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        choices=list(DEFAULT_METRIC_NAMES),
        help=(
            "Subset of metrics to compute. Default: all 7 standard metrics "
            f"({', '.join(DEFAULT_METRIC_NAMES)})"
        ),
    )
    parser.add_argument(
        "--output-name",
        default="test_metrics_ci_v1.json",
        help="Output filename within experiment_dir (default: test_metrics_ci_v1.json)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recompute even if artifact exists (skip cache hit)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log verbosity (default: INFO)",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    experiment_dirs = _resolve_experiment_dirs(args)

    metric_names = tuple(args.metrics) if args.metrics else DEFAULT_METRIC_NAMES
    config = TestMetricsCIConfig(
        n_bootstraps=args.n_bootstraps,
        block_length=args.block_length,
        ci=args.ci,
        seed=args.seed,
        primary_horizon_idx=args.primary_horizon_idx,
        metric_names=metric_names,
    )

    LOGGER.info(
        "Computing bootstrap CI for %d experiment(s) — config: %s",
        len(experiment_dirs), config,
    )

    skip_if_exists = not args.force
    failures = 0
    for experiment_dir in experiment_dirs:
        if not experiment_dir.exists():
            LOGGER.error("Experiment dir does not exist: %s", experiment_dir)
            failures += 1
            continue
        try:
            output_path = experiment_dir / args.output_name
            artifact = compute_test_metrics_ci_for_experiment(
                experiment_dir=experiment_dir,
                output_path=output_path,
                config=config,
                skip_if_exists=skip_if_exists,
            )
            print(_format_summary(artifact, experiment_dir.name))
        except Exception as exc:
            LOGGER.error(
                "Failed to compute CI for %s: %s", experiment_dir.name, exc,
                exc_info=args.log_level == "DEBUG",
            )
            failures += 1

    if failures:
        LOGGER.error("FAILED: %d of %d experiments", failures, len(experiment_dirs))
        return 1
    LOGGER.info(
        "OK: %d of %d experiments processed successfully",
        len(experiment_dirs) - failures, len(experiment_dirs),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
