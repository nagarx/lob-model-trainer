"""Multi-source day bundle and fusion (T12).

Loads data from multiple sources, aligns by first-N heuristic, and fuses
into standard DayData for seamless integration with the existing Trainer.

The alignment heuristic: both MBO and BASIC exports use the same 60-second
time grid anchored at 09:30 ET. With stride=1, the first sequences from
both sources correspond to the same wall-clock time. The N difference
(MBO < BASIC) is at the END due to different label-truncation horizons.
Empirically validated: Pearson r = 0.930 at offset=0 for base mid-prices
on 2025-02-03.

Reference: plan/EXPERIMENTATION_FIRST_ARCHITECTURE.md §18
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from lobtrainer.data.sources import DataSource, SourceDay, normalize_date

logger = logging.getLogger(__name__)


@dataclass
class DayBundle:
    """Multi-source day container.

    Holds one SourceDay per data source for the same calendar date,
    plus shared label/weight arrays from the primary source.

    Use ``to_fused_day_data()`` to convert to a standard DayData
    with concatenated features for the existing Trainer pipeline.
    """

    date: str
    primary_source: str
    sources: Dict[str, SourceDay]
    labels: np.ndarray
    regression_labels: Optional[np.ndarray] = None
    forward_prices: Optional[np.ndarray] = None
    sample_weights: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None

    @property
    def n_sequences(self) -> int:
        """Aligned sequence count (same across all sources)."""
        primary = self.sources.get(self.primary_source)
        return primary.n_sequences if primary is not None else 0

    def to_fused_day_data(
        self,
        source_order: Optional[List[str]] = None,
        feature_indices: Optional[Dict[str, List[int]]] = None,
    ) -> "DayData":
        """Fuse features across sources into a standard DayData.

        Concatenates sequences along the feature axis in ``source_order``.
        The result is a standard DayData with [N, T, F_total] sequences
        that the existing Trainer/LOBSequenceDataset can consume unchanged.

        Args:
            source_order: Sources to include in concat order.
                Default: [primary, ...auxiliaries sorted by name].
            feature_indices: Per-source feature index selection.
                {source_name: [idx0, idx1, ...]}. None = all features.

        Returns:
            DayData with concatenated features.
        """
        from lobtrainer.data.dataset import DayData

        if source_order is None:
            source_order = [self.primary_source] + sorted(
                k for k in self.sources if k != self.primary_source
            )

        # Validate matching window sizes (T dimension) before concatenation
        window_sizes = {
            name: self.sources[name].window_size
            for name in source_order
            if name in self.sources
        }
        unique_ws = set(window_sizes.values())
        if len(unique_ws) > 1:
            raise ValueError(
                f"Cannot fuse sources with different window sizes: "
                f"{window_sizes}. All sources must have the same T dimension."
            )

        pieces = []
        for src_name in source_order:
            if src_name not in self.sources:
                raise KeyError(
                    f"Source '{src_name}' not in bundle. "
                    f"Available: {list(self.sources.keys())}"
                )
            src = self.sources[src_name]
            seq = src.sequences
            if feature_indices is not None and src_name in feature_indices:
                idx = feature_indices[src_name]
                seq = seq[:, :, idx]
            pieces.append(seq)

        fused_sequences = np.concatenate(pieces, axis=-1)
        fused_features = fused_sequences[:, -1, :]

        return DayData(
            date=self.date,
            features=fused_features,
            labels=self.labels,
            sequences=fused_sequences,
            regression_labels=self.regression_labels,
            forward_prices=self.forward_prices,
            sample_weights=self.sample_weights,
            metadata=self.metadata,
            is_aligned=True,
        )


def _align_sources(
    primary: SourceDay,
    *auxiliaries: SourceDay,
) -> Tuple[SourceDay, ...]:
    """Align multiple sources by taking first min(N) sequences.

    Both sources must use the same time grid (60s bins from 09:30 ET,
    stride=1). The first sequences are temporally aligned; the N
    difference is at the tail end due to different label-truncation
    horizons.

    Returns:
        Tuple of aligned SourceDay objects (primary first, then auxiliaries)
        with equal N.
    """
    n_values = [primary.n_sequences] + [a.n_sequences for a in auxiliaries]
    n_aligned = min(n_values)

    if n_aligned == 0:
        raise ValueError(
            f"Cannot align sources: at least one has 0 sequences "
            f"(primary={primary.n_sequences}, "
            f"auxiliaries={[a.n_sequences for a in auxiliaries]})"
        )

    results = [_trim_source_day(primary, n_aligned)]
    for aux in auxiliaries:
        results.append(_trim_source_day(aux, n_aligned))

    if any(n != n_aligned for n in n_values):
        logger.info(
            "Aligned %d sources to N=%d (original: %s)",
            len(n_values), n_aligned,
            {s.name: s.n_sequences for s in [primary] + list(auxiliaries)},
        )

    return tuple(results)


def _trim_source_day(source: SourceDay, n: int) -> SourceDay:
    """Trim a SourceDay to first n sequences."""
    if source.n_sequences == n:
        return source
    return SourceDay(
        name=source.name,
        date=source.date,
        sequences=source.sequences[:n],
        features=source.features[:n],
        metadata=source.metadata,
        n_features=source.n_features,
        window_size=source.window_size,
    )


def _load_source_day(
    source: DataSource,
    split_dir: Path,
    date_str: str,
) -> Optional[SourceDay]:
    """Load one SourceDay from disk for a given date.

    Handles date format normalization (YYYY-MM-DD vs YYYYMMDD).
    Returns None if the day's files are not found.
    """
    norm_date = normalize_date(date_str)

    # Try both date formats for filename lookup
    seq_path = None
    for fmt in [date_str, norm_date,
                f"{norm_date[:4]}-{norm_date[4:6]}-{norm_date[6:8]}"]:
        candidate = split_dir / f"{fmt}_sequences.npy"
        if candidate.exists():
            seq_path = candidate
            break

    if seq_path is None:
        return None

    file_date = seq_path.stem.replace("_sequences", "")
    sequences = np.load(seq_path).astype(np.float64)
    features = sequences[:, -1, :]

    # Load metadata
    meta_path = split_dir / f"{file_date}_metadata.json"
    metadata = None
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)

    return SourceDay(
        name=source.name,
        date=norm_date,
        sequences=sequences,
        features=features,
        metadata=metadata,
    )


def _discover_dates_in_split(split_dir: Path) -> List[str]:
    """Discover all dates available in a split directory.

    Returns normalized YYYYMMDD date strings, sorted.
    """
    dates = set()
    for f in split_dir.glob("*_sequences.npy"):
        raw_date = f.stem.replace("_sequences", "")
        dates.add(normalize_date(raw_date))
    return sorted(dates)


def load_split_bundles(
    sources: List[DataSource],
    split: str,
    labels_config: Optional[Any] = None,
    source_order: Optional[List[str]] = None,
    feature_indices: Optional[Dict[str, List[int]]] = None,
) -> List["DayData"]:
    """Load multi-source data and fuse into List[DayData].

    This is the multi-source equivalent of load_split_data(). It:
    1. Discovers dates available in EACH source
    2. Finds the intersection (common dates across ALL sources)
    3. For each common date, loads all sources and aligns
    4. Computes labels from primary source's forward_prices (T9)
    5. Computes sample weights (T10)
    6. Fuses into standard DayData via to_fused_day_data

    Args:
        sources: List of DataSource configs. Exactly one must be primary.
        split: "train", "val", or "test".
        labels_config: Optional LabelsConfig for forward-prices label
            computation and sample weights.
        source_order: Sources to include in fusion (concat order).
        feature_indices: Per-source feature index selection.

    Returns:
        List[DayData] with fused features, sorted by date.
    """
    from lobtrainer.data.dataset import load_day_data

    # Validate: exactly one primary
    primaries = [s for s in sources if s.role == "primary"]
    if len(primaries) != 1:
        raise ValueError(
            f"Exactly one source must have role='primary', "
            f"got {len(primaries)}: {[s.name for s in primaries]}"
        )
    primary_source = primaries[0]
    auxiliaries = [s for s in sources if s.role == "auxiliary"]

    # 1. Discover dates per source
    source_dates: Dict[str, set] = {}
    for src in sources:
        split_dir = Path(src.data_dir) / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Split directory not found: {split_dir} "
                f"(source={src.name})"
            )
        dates = set(_discover_dates_in_split(split_dir))
        source_dates[src.name] = dates
        logger.info(
            "Source '%s' split '%s': %d dates", src.name, split, len(dates)
        )

    # 2. Common dates (intersection)
    common_dates = sorted(
        set.intersection(*source_dates.values())
    )
    if not common_dates:
        counts = {k: len(v) for k, v in source_dates.items()}
        raise ValueError(
            f"No common dates across sources for split '{split}'. "
            f"Per-source counts: {counts}"
        )
    logger.info(
        "Common dates for split '%s': %d (from %s to %s)",
        split, len(common_dates), common_dates[0], common_dates[-1],
    )

    # Read export stride from primary manifest (for sample weights)
    manifest_path = Path(primary_source.data_dir) / "dataset_manifest.json"
    export_stride = 1
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        export_stride = manifest.get("stride", 1)

    # 3. Load and fuse each common date
    fused_days = []
    for date in common_dates:
        # Load primary source via existing load_day_data (T9 labels + T10 weights)
        pri_split_dir = Path(primary_source.data_dir) / split
        pri_date_raw = _find_date_filename(pri_split_dir, date)
        if pri_date_raw is None:
            logger.warning("Primary source missing date %s, skipping", date)
            continue

        primary_day_data = load_day_data(
            pri_split_dir / f"{pri_date_raw}_sequences.npy",
            pri_split_dir / f"{pri_date_raw}_labels.npy",
            metadata_path=(
                pri_split_dir / f"{pri_date_raw}_metadata.json"
                if (pri_split_dir / f"{pri_date_raw}_metadata.json").exists()
                else None
            ),
            regression_labels_path=(
                pri_split_dir / f"{pri_date_raw}_regression_labels.npy"
                if (pri_split_dir / f"{pri_date_raw}_regression_labels.npy").exists()
                else None
            ),
            forward_prices_path=(
                pri_split_dir / f"{pri_date_raw}_forward_prices.npy"
                if (pri_split_dir / f"{pri_date_raw}_forward_prices.npy").exists()
                else None
            ),
            labels_config=labels_config,
            export_stride=export_stride,
            validate=False,
        )

        # Build primary SourceDay
        primary_sd = SourceDay(
            name=primary_source.name,
            date=date,
            sequences=primary_day_data.sequences,
            features=primary_day_data.features,
            metadata=primary_day_data.metadata,
        )

        # Load auxiliary sources
        aux_sds = []
        for aux_src in auxiliaries:
            aux_split_dir = Path(aux_src.data_dir) / split
            aux_sd = _load_source_day(aux_src, aux_split_dir, date)
            if aux_sd is None:
                logger.warning(
                    "Auxiliary source '%s' missing date %s, skipping",
                    aux_src.name, date,
                )
                break
            aux_sds.append(aux_sd)
        else:
            # All auxiliaries loaded — align
            aligned = _align_sources(primary_sd, *aux_sds)
            aligned_primary = aligned[0]
            aligned_auxs = aligned[1:]

            # Build DayBundle
            source_dict = {aligned_primary.name: aligned_primary}
            for a in aligned_auxs:
                source_dict[a.name] = a

            n_aligned = aligned_primary.n_sequences
            bundle = DayBundle(
                date=date,
                primary_source=primary_source.name,
                sources=source_dict,
                labels=primary_day_data.labels[:n_aligned],
                regression_labels=(
                    primary_day_data.regression_labels[:n_aligned]
                    if primary_day_data.regression_labels is not None
                    else None
                ),
                forward_prices=(
                    primary_day_data.forward_prices[:n_aligned]
                    if primary_day_data.forward_prices is not None
                    else None
                ),
                sample_weights=(
                    primary_day_data.sample_weights[:n_aligned]
                    if primary_day_data.sample_weights is not None
                    else None
                ),
                metadata=primary_day_data.metadata,
            )

            # Fuse to DayData
            fused = bundle.to_fused_day_data(
                source_order=source_order,
                feature_indices=feature_indices,
            )
            fused_days.append(fused)
            continue

        # If we broke out of the aux loop (missing aux), skip this date
        logger.warning("Skipping date %s due to missing auxiliary source", date)

    logger.info(
        "Loaded %d fused days from %d common dates "
        "(sources: %s)",
        len(fused_days), len(common_dates),
        [s.name for s in sources],
    )
    return fused_days


def _find_date_filename(split_dir: Path, normalized_date: str) -> Optional[str]:
    """Find the actual filename date format for a normalized date.

    Returns the raw date string used in filenames, or None if not found.
    """
    # Try compact YYYYMMDD
    if (split_dir / f"{normalized_date}_sequences.npy").exists():
        return normalized_date
    # Try hyphenated YYYY-MM-DD
    hyphenated = (
        f"{normalized_date[:4]}-{normalized_date[4:6]}-{normalized_date[6:8]}"
    )
    if (split_dir / f"{hyphenated}_sequences.npy").exists():
        return hyphenated
    return None
