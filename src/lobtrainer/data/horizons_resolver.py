"""
Phase X.3 Empirical Trust — Horizons resolver (Phase C.1, 2026-05-05).

Auto-derives ``LabelsConfig.horizons`` from the export's ``*_horizons.json``
files at config-resolve time, eliminating the silent fallback in
``compatibility.py:233-236`` that read ``model.hmhp_horizons`` (which
defaulted to classification ``[10, 20, 50, 100, 200]``) when
``data.labels.horizons`` was empty.

Per hft-rules §1 ("layout as contract — single source of truth"): the
export's ``*_horizons.json`` is the AUTHORITATIVE source for what
horizons exist in the corpus. The trainer-side ``labels.horizons``
should match — empty means "trainer config didn't override; resolve
from export," NOT "fall back to a hardcoded classification default
that may not match the actual data."

ROOT CAUSE FIXED: B5 horizon drift — pre-Phase-C.1, all 5 of 6 v3p0
TLOB-family stages (R9-R12) reported ``compatibility.horizons =
[10, 20, 50, 100, 200]`` in signal_metadata, but the actual data was
``[10, 60, 300]``. This meant Phase II ``compatibility_fingerprint``
drifted from the data — two trainings on same data corpus + horizons
[10,60,300] (regression) vs [10,20,50,100,200] (classification) would
produce IDENTICAL fingerprint (both fall back to classification
defaults), breaking the cross-experiment composability invariant.

DESIGN:
1. Resolver reads first day's ``*_horizons.json`` in the requested split.
2. Defensive: validates consistency across all days in the split (per
   hft-rules §8 "never silently drop, clamp, or 'fix' data" — drift
   between days is an upstream data corruption issue, not auto-recoverable).
3. Returns ``Tuple[int, ...]`` matching ``LabelsConfig.horizons`` type.

CALLERS:
- ``Trainer.setup()`` calls before ``build_compatibility_contract``.
- ``SimpleModelTrainer.setup()`` calls before signal metadata emit.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Tuple, Union

logger = logging.getLogger(__name__)


def resolve_horizons_from_export(
    data_dir: Union[str, Path], split: str = "train"
) -> Tuple[int, ...]:
    """Read horizons from the export's ``*_horizons.json`` files.

    Args:
        data_dir: Path to corpus root (parent of ``train/``/``val/``/``test/``).
        split: Split to read from. Defaults to ``"train"`` since labels are
            generated identically across splits (the *_horizons.json files
            are duplicated per-day in each split — same content).

    Returns:
        Tuple of horizon values in the order they appear in
        ``*_horizons.json``. Type ``Tuple[int, ...]`` matches
        ``LabelsConfig.horizons``.

    Raises:
        FileNotFoundError: If the requested split has no ``*_horizons.json``
            files. Indicates either a non-regression corpus (classification-
            only event-based exports may not emit horizons.json) OR a
            corrupt export.
        ValueError: If horizons differ across days in the same split (data
            corruption — fail-loud per hft-rules §8).
        ValueError: If a ``*_horizons.json`` file is malformed (missing
            ``horizons`` key OR non-list value OR non-integer entries).

    Phase X.3 Empirical Trust (2026-05-05) — Phase C.1.
    """
    split_dir = Path(data_dir) / split
    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"Cannot resolve horizons: split directory does not exist: {split_dir}"
        )

    horizons_files = sorted(split_dir.glob("*_horizons.json"))
    if not horizons_files:
        raise FileNotFoundError(
            f"Cannot resolve horizons: no *_horizons.json files in {split_dir}. "
            f"Either the export is classification-only (no regression labels) "
            f"OR the export is corrupt. Set `data.labels.horizons` explicitly "
            f"in the YAML config to bypass auto-resolution."
        )

    # Read first day
    with open(horizons_files[0]) as f:
        first_payload = json.load(f)

    if "horizons" not in first_payload:
        raise ValueError(
            f"Malformed {horizons_files[0].name}: missing 'horizons' key. "
            f"Keys present: {sorted(first_payload.keys())}"
        )

    first_horizons = first_payload["horizons"]
    if not isinstance(first_horizons, list):
        raise ValueError(
            f"Malformed {horizons_files[0].name}: 'horizons' must be a list, "
            f"got {type(first_horizons).__name__}"
        )

    if not all(isinstance(h, int) for h in first_horizons):
        raise ValueError(
            f"Malformed {horizons_files[0].name}: 'horizons' must contain integers, "
            f"got {first_horizons!r}"
        )

    if len(first_horizons) == 0:
        raise ValueError(
            f"Malformed {horizons_files[0].name}: 'horizons' list is empty. "
            f"At least one horizon required."
        )

    # Defensive cross-day consistency check — drift is upstream corruption
    # per hft-rules §8 ("never silently drop, clamp, or 'fix' data")
    n_days_checked = 1
    for hf in horizons_files[1:]:
        with open(hf) as f:
            other_payload = json.load(f)
        other_horizons = other_payload.get("horizons")
        if other_horizons != first_horizons:
            raise ValueError(
                f"Horizons drift in {split} split: {hf.name} has "
                f"{other_horizons!r}, but first-day {horizons_files[0].name} "
                f"has {first_horizons!r}. Upstream data corruption — "
                f"re-export the corpus."
            )
        n_days_checked += 1

    logger.debug(
        f"Resolved horizons from {data_dir}/{split}/: {first_horizons} "
        f"(verified consistent across {n_days_checked} day(s))"
    )

    return tuple(first_horizons)


__all__ = ["resolve_horizons_from_export"]
