"""
PyTorch Dataset implementations for LOB feature data.

Data Contract (Rust pipeline Schema v2.2):
    - NEW aligned format (*_sequences.npy): [N_seq, 100, 98] - 3D sequences, 1:1 with labels
    - LEGACY flat format (*_features.npy): [N_samples, 98] - requires manual alignment
    - Labels:
        - Single-horizon: [N_seq] int8 arrays - (-1=Down, 0=Stable, 1=Up)
        - Multi-horizon: [N_seq, n_horizons] int8 arrays - one label per horizon

Files are organized as:
    data_dir/
    ├── train/
    │   ├── 20250203_sequences.npy  # NEW: [N_seq, 100, 98]
    │   ├── 20250203_labels.npy     # [N_seq] or [N_seq, n_horizons]
    │   ├── 20250203_metadata.json
    │   └── ...
    ├── val/
    └── test/

Design principles (RULE.md):
- Memory-efficient: Option to stream from disk or cache in memory
- Deterministic: Same sequence of data for reproducibility
- Validated: Check data integrity on load
- Format-agnostic: Auto-detect and handle both export formats
- Horizon-flexible: Support single and multi-horizon labels
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import logging

import numpy as np
import torch
from torch.utils.data import Dataset

from lobtrainer.constants import FEATURE_COUNT, FeatureIndex

from hft_contracts import (
    FEATURE_COUNT as CONTRACT_FEATURE_COUNT,
    FULL_FEATURE_COUNT,
    get_contract,
)
from hft_contracts.validation import (
    ContractError,
    validate_export_contract,
    validate_normalization_not_applied,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Contract Validation at Boundary
# =============================================================================


def _validate_day_metadata(metadata: Optional[Dict], date: str) -> None:
    """
    Validate export metadata for a single day at the load boundary.

    Called once per split on the first day's metadata. Checks schema_version,
    feature count, and normalization boundary contract.

    Args:
        metadata: Loaded metadata dict (or None for legacy data).
        date: Date string for error messages.

    Raises:
        ContractError: If metadata fails validation.
    """
    if metadata is None:
        return

    if "schema_version" not in metadata:
        logger.warning(
            "Metadata for %s has no schema_version field. "
            "Re-export with latest feature extractor to enable contract validation.",
            date,
        )
        return

    warnings = validate_export_contract(metadata, strict_completeness=False)
    for w in warnings:
        logger.warning("Contract warning (%s): %s", date, w)


# =============================================================================
# Label Encoding Utilities
# =============================================================================


def _determine_label_shift_from_metadata(
    labeling_strategy: Optional[str],
    days: List["DayData"],
) -> bool:
    """
    Determine whether labels need to be shifted from {-1, 0, 1} to {0, 1, 2}.

    Resolution order:
        1. Explicit ``labeling_strategy`` argument -- consult the pipeline contract.
        2. ``label_strategy`` field in metadata -- consult the pipeline contract.
        3. Legacy heuristics for backward compatibility.
        4. Default: assume shift is needed.

    Args:
        labeling_strategy: Strategy passed in constructor (may be None)
        days: DayData list (to check metadata if strategy not specified)

    Returns:
        True if labels need +1 shift, False if already 0-indexed
    """
    strategy_name = labeling_strategy

    if strategy_name is None and days and days[0].metadata:
        meta = days[0].metadata
        strategy_name = meta.get("label_strategy")
        if strategy_name is None:
            labeling = meta.get("labeling", {})
            if isinstance(labeling, dict):
                strategy_name = labeling.get("strategy")

    if strategy_name is not None:
        strategy_key = (
            strategy_name.lower()
            if isinstance(strategy_name, str)
            else strategy_name.value
        )
        try:
            contract = get_contract(strategy_key)
            needs_shift = getattr(contract, 'shift_for_crossentropy', False)
            logger.info(
                "%s labeling (contract): shift_for_crossentropy=%s",
                strategy_key,
                needs_shift,
            )
            return needs_shift
        except ValueError:
            logger.warning(
                "Unknown labeling strategy '%s'; falling back to heuristic detection",
                strategy_key,
            )

    if days and days[0].metadata:
        meta = days[0].metadata
        encoding = meta.get("label_encoding", {})
        if isinstance(encoding, dict):
            note = encoding.get("note", "")
            if "class indices 0, 1, 2" in note.lower():
                logger.info(
                    "Auto-detected 0-indexed labels from label_encoding metadata"
                )
                return False

    logger.info(
        "No labeling strategy resolved; assuming TLOB/Opportunity format "
        "with labels {-1, 0, 1}, shifting to {0, 1, 2}"
    )
    return True


# =============================================================================
# Data Loading Functions
# =============================================================================


def _detect_export_format(split_dir: Path) -> str:
    """
    Detect whether directory contains new aligned format or legacy format.
    
    Returns:
        'aligned' for *_sequences.npy (3D), 'legacy' for *_features.npy (2D)
    """
    seq_files = list(split_dir.glob('*_sequences.npy'))
    feat_files = list(split_dir.glob('*_features.npy'))
    
    if seq_files:
        return 'aligned'
    elif feat_files:
        return 'legacy'
    else:
        raise FileNotFoundError(f"No data files found in {split_dir}")


@dataclass
class DayData:
    """
    Data from a single trading day.
    
    For aligned format:
        - features: [N_seq, 98] - last timestep of each sequence (for flat models)
        - sequences: [N_seq, 100, 98] - full 3D sequences (for sequence models)
        - labels: [N_seq] or [N_seq, n_horizons] - 1:1 aligned with features/sequences
    
    For legacy format:
        - features: [N_samples, 98] - flat samples (NOT aligned with labels)
        - sequences: None
        - labels: [N_labels] - requires manual alignment
    
    Multi-horizon labels:
        - When labels.ndim == 2, each column is a different prediction horizon
        - horizons metadata indicates the actual horizon values
        - Use get_labels(horizon_idx) to select a specific horizon
    
    Lazy Loading Mode (memory-efficient):
        - Set lazy=True to defer loading until data is accessed
        - Data loaded via memory-mapping (mmap) for minimal RAM usage
        - Ideal for large datasets that don't fit in memory
    """
    date: str
    features: np.ndarray  # [N, 98] - flat features
    labels: np.ndarray    # [N_seq], [N_labels], or [N_seq, n_horizons]
    sequences: Optional[np.ndarray] = None  # [N_seq, 100, 98] - only for aligned format
    regression_labels: Optional[np.ndarray] = None  # [N_seq, n_horizons] float64 bps - from feature extractor
    forward_prices: Optional[np.ndarray] = None  # [N, k + max_H + 1] float64 USD (T9)
    sample_weights: Optional[np.ndarray] = None  # [N] float64 mean=1.0 (T10)
    metadata: Optional[Dict] = None
    is_aligned: bool = False  # True if features are 1:1 with labels
    
    # Lazy loading support
    _sequences_path: Optional[Path] = field(default=None, repr=False)
    _features_path: Optional[Path] = field(default=None, repr=False)
    _labels_path: Optional[Path] = field(default=None, repr=False)
    _lazy: bool = field(default=False, repr=False)
    _mmap_mode: Optional[str] = field(default=None, repr=False)  # 'r' for read-only mmap
    
    @property
    def num_samples(self) -> int:
        """Number of flat feature samples."""
        return self.features.shape[0]
    
    @property
    def num_sequences(self) -> int:
        """Number of labeled sequences."""
        return self.labels.shape[0]
    
    @property
    def window_size(self) -> int:
        """Window size (100 for aligned, 0 for legacy)."""
        if self.sequences is not None:
            return self.sequences.shape[1]
        return 0
    
    @property
    def is_multi_horizon(self) -> bool:
        """Check if labels have multiple horizons."""
        return self.labels.ndim == 2
    
    @property
    def num_horizons(self) -> int:
        """Number of prediction horizons (1 for single-horizon, n for multi-horizon)."""
        if self.is_multi_horizon:
            return self.labels.shape[1]
        return 1
    
    @property
    def horizons(self) -> Optional[List[int]]:
        """Get horizon values from metadata (if available).
        
        Checks multiple locations for compatibility:
        - Top-level 'horizons' (new format)
        - 'labeling.horizons' (Triple Barrier format)
        - 'label_config.horizons' (legacy format)
        """
        if self.metadata:
            # New format: horizons at top level
            if 'horizons' in self.metadata:
                return self.metadata['horizons']
            # Triple Barrier format: nested in 'labeling'
            if 'labeling' in self.metadata:
                labeling = self.metadata['labeling']
                if 'horizons' in labeling:
                    return labeling['horizons']
            # Legacy format: nested in label_config
            if 'label_config' in self.metadata:
                label_config = self.metadata['label_config']
                if 'horizons' in label_config:
                    return label_config['horizons']
        return None
    
    def get_labels(self, horizon_idx: Optional[int] = None) -> np.ndarray:
        """
        Get labels for a specific horizon.
        
        Args:
            horizon_idx: Index of horizon to select (0-based).
                         None returns all labels (1D for single-horizon, 2D for multi).
        
        Returns:
            Label array of shape [N_seq] for single horizon or selected horizon,
            or [N_seq, n_horizons] for all multi-horizon labels.
        """
        if horizon_idx is None:
            return self.labels
        
        if not self.is_multi_horizon:
            if horizon_idx != 0:
                raise ValueError(
                    f"Single-horizon data only has horizon_idx=0, got {horizon_idx}"
                )
            return self.labels
        
        if horizon_idx < 0 or horizon_idx >= self.num_horizons:
            raise ValueError(
                f"horizon_idx {horizon_idx} out of range [0, {self.num_horizons})"
            )
        
        return self.labels[:, horizon_idx]
    
    def validate(self, expected_feature_count: Optional[int] = None) -> None:
        """
        Validate data integrity.
        
        Args:
            expected_feature_count: Expected number of features. If None, skips
                feature count validation (allows any valid feature count like 40 or 98).
        
        Raises:
            ValueError: If data is invalid
        """
        if self.features.ndim != 2:
            raise ValueError(f"Expected 2D features, got shape {self.features.shape}")
        
        # Only check feature count if explicitly specified
        if expected_feature_count is not None:
            if self.features.shape[1] != expected_feature_count:
                raise ValueError(
                    f"Expected {expected_feature_count} features, got {self.features.shape[1]}"
                )
        # Labels can be 1D [N_seq] or 2D [N_seq, n_horizons]
        if self.labels.ndim not in (1, 2):
            raise ValueError(f"Expected 1D or 2D labels, got shape {self.labels.shape}")
        
        # For aligned format, check 1:1 alignment
        if self.is_aligned and len(self.features) != self.labels.shape[0]:
            raise ValueError(
                f"Aligned format requires features[0] == labels[0], got "
                f"{len(self.features)} features and {self.labels.shape[0]} labels"
            )
        
        # Check for NaN/Inf in features
        # For 98-feature datasets: use BOOK_VALID mask to exclude invalid samples
        # For 40-feature datasets (LOB only): check all samples
        num_features = self.features.shape[1]
        
        if num_features >= FeatureIndex.BOOK_VALID + 1:
            # Full 98-feature set: use BOOK_VALID mask
            book_valid = self.features[:, FeatureIndex.BOOK_VALID]
            valid_mask = book_valid > 0.5
            if valid_mask.any():
                valid_features = self.features[valid_mask]
            else:
                valid_features = self.features  # fallback to all
        else:
            # 40-feature LOB-only: check all samples (no BOOK_VALID flag)
            valid_features = self.features
        
        if not np.isfinite(valid_features).all():
            nan_count = np.isnan(valid_features).sum()
            inf_count = np.isinf(valid_features).sum()
            raise ValueError(
                f"Found {nan_count} NaN and {inf_count} Inf values in features"
            )


# =============================================================================
# Forward-Prices Label Computation (T9)
# =============================================================================


def _compute_labels_from_forward_prices(
    forward_prices: np.ndarray,
    metadata: Dict,
    labels_config: "LabelsConfig",
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Compute labels from {day}_forward_prices.npy via LabelFactory.

    Reads smoothing_window EXCLUSIVELY from ForwardPriceContract (enforces
    Bug B2 fix — user config cannot override k).

    Args:
        forward_prices: [N, k + max_H + 1] float64 USD array.
        metadata: Parsed {day}_metadata.json dict.
        labels_config: LabelsConfig driving return_type / horizons / task.

    Returns:
        (classification_labels, regression_labels):
            classification_labels: [N, n_horizons] int8 in canonical {-1, 0, +1}
                when task is classification, else None. (Downstream
                ``_labels_need_shift`` handles the +1 shift to {0, 1, 2}
                uniformly with the legacy path.)
            regression_labels: [N, n_horizons] float64 bps (always populated).

    Raises:
        ValueError: Horizon bounds exceeded or non-finite values produced.
        hft_contracts.validation.ContractError: Missing horizons in metadata.
        KeyError: forward_prices not exported in metadata.
    """
    from hft_contracts import ForwardPriceContract, LabelFactory
    from hft_contracts.validation import ContractError

    contract = ForwardPriceContract.from_metadata(metadata)
    contract.validate_shape(forward_prices)

    # CRITICAL: smoothing_window ALWAYS from contract, NEVER from user config.
    k = contract.smoothing_window_offset

    # Resolve exported horizons — mirror DayData.horizons fallback for consistency.
    exported_horizons = None
    if metadata is not None:
        exported_horizons = metadata.get("horizons")
        if exported_horizons is None:
            labeling = metadata.get("labeling", {})
            if isinstance(labeling, dict):
                exported_horizons = labeling.get("horizons")
        if exported_horizons is None:
            label_config_section = metadata.get("label_config", {})
            if isinstance(label_config_section, dict):
                exported_horizons = label_config_section.get("horizons")

    if exported_horizons is None:
        raise ContractError(
            "metadata is missing horizons field (checked: "
            "metadata['horizons'], metadata['labeling']['horizons'], "
            "metadata['label_config']['horizons']). Cannot compute "
            "labels from forward_prices without horizon specification."
        )

    # User horizons: empty list → all exported. Else user's subset/superset.
    if not labels_config.horizons:
        horizons = list(exported_horizons)
    else:
        horizons = list(labels_config.horizons)
        for h in horizons:
            if h not in exported_horizons:
                logger.info(
                    "Requested horizon %d not in exported %s; computing via "
                    "LabelFactory (valid if h + k < n_columns=%d)",
                    h, list(exported_horizons), contract.n_columns,
                )

    # Compute regression labels. Pre-T9 validation guards against out-of-bounds.
    regression = LabelFactory.multi_horizon(
        forward_prices, horizons, k, labels_config.return_type
    )

    # Defense in depth — Pre-T9 should catch this, but we defend anyway.
    if not np.isfinite(regression).all():
        raise ValueError(
            f"LabelFactory.multi_horizon produced non-finite values. "
            f"horizons={horizons}, k={k}, "
            f"max_horizon={contract.max_horizon}. "
            f"Verify forward_prices contains no NaN/Inf."
        )

    # Resolve task.
    task = labels_config.task
    if task == "auto":
        label_strategy_raw = metadata.get("label_strategy")
        if label_strategy_raw is None:
            labeling = metadata.get("labeling", {})
            if isinstance(labeling, dict):
                label_strategy_raw = labeling.get("strategy")

        if label_strategy_raw is not None:
            from hft_contracts import is_regression_strategy
            strategy_key = (
                label_strategy_raw.lower()
                if isinstance(label_strategy_raw, str)
                else label_strategy_raw.value
            )
            try:
                task = (
                    "regression"
                    if is_regression_strategy(strategy_key)
                    else "classification"
                )
            except (ValueError, KeyError):
                task = "regression"
        else:
            # Safest default when forward_prices source is active.
            task = "regression"

    # Compute classification labels when requested. Return canonical {-1, 0, +1}.
    # Downstream _labels_need_shift handles the CE shift uniformly.
    classification = None
    if task == "classification":
        classification = LabelFactory.classify(
            regression, labels_config.threshold_bps
        )

    return classification, regression


def load_day_data(
    data_file: Union[str, Path],
    labels_path: Union[str, Path],
    metadata_path: Optional[Union[str, Path]] = None,
    regression_labels_path: Optional[Union[str, Path]] = None,
    forward_prices_path: Optional[Union[str, Path]] = None,
    labels_config: Optional["LabelsConfig"] = None,
    export_stride: int = 1,
    validate: bool = True,
    lazy: bool = False,
    mmap_mode: Optional[str] = None,
) -> DayData:
    """
    Load data for a single trading day.

    Auto-detects format:
    - *_sequences.npy: 3D aligned format [N_seq, 100, 98]
    - *_features.npy: 2D legacy format [N_samples, 98]

    Label source resolution (T9):
        1. If labels_config.source == "forward_prices": REQUIRE
           forward_prices.npy + metadata forward_prices.exported == True.
           Compute labels via LabelFactory.multi_horizon.
        2. If labels_config.source == "auto" AND forward_prices.npy exists AND
           metadata declares exported: compute via LabelFactory.
        3. Else: load precomputed labels (unchanged legacy path).

    Args:
        data_file: Path to data .npy file (sequences or features)
        labels_path: Path to labels .npy file (classification labels).
            May not exist for pure regression or forward_prices exports.
        metadata_path: Optional path to metadata .json file
        regression_labels_path: Optional path to regression labels .npy file
            (float64 bps). When provided, loads precomputed regression targets.
        forward_prices_path: Optional path to forward_prices .npy file (T9).
        labels_config: Optional LabelsConfig driving label source resolution (T9).
        validate: Whether to validate data integrity
        lazy: If True, use memory-mapped arrays (load on-demand)
        mmap_mode: Memory-map mode ('r' for read-only, 'r+' for read/write).
                   If lazy=True and mmap_mode=None, defaults to 'r'.

    Returns:
        DayData instance

    Raises:
        FileNotFoundError: If no label source found (labels, regression, or forward_prices)
        ValueError: If data validation fails
    """
    data_file = Path(data_file)
    labels_path = Path(labels_path)
    
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    has_classification_labels = labels_path.exists()
    has_regression_labels = (
        regression_labels_path is not None and Path(regression_labels_path).exists()
    )
    has_forward_prices = (
        forward_prices_path is not None and Path(forward_prices_path).exists()
    )

    if not has_classification_labels and not has_regression_labels and not has_forward_prices:
        raise FileNotFoundError(
            f"No label source found for {data_file.stem}: "
            f"neither {labels_path}, regression_labels, "
            f"nor forward_prices exist"
        )
    
    # Detect format from filename
    is_aligned = '_sequences' in data_file.stem
    
    # Extract date from filename
    date = data_file.stem.replace('_sequences', '').replace('_features', '')
    
    # Determine mmap mode
    effective_mmap = mmap_mode if mmap_mode else ('r' if lazy else None)
    
    # Load data (potentially memory-mapped)
    raw_data = np.load(data_file, mmap_mode=effective_mmap)

    if has_classification_labels:
        labels = np.load(labels_path, mmap_mode=effective_mmap)
        if not lazy:
            labels = labels.astype(np.int64)
    else:
        labels = np.empty(0, dtype=np.int64)

    regression_labels = None
    if has_regression_labels:
        regression_labels = np.load(
            Path(regression_labels_path), mmap_mode=effective_mmap
        )
        if not lazy:
            regression_labels = regression_labels.astype(np.float64)
        if labels.size == 0:
            labels = np.zeros(regression_labels.shape, dtype=np.int64)
    
    # Handle 3D sequences vs 2D flat
    if is_aligned and len(raw_data.shape) == 3:
        # NEW format: [N_seq, 100, 98]
        if lazy:
            sequences = raw_data  # Keep as mmap
            # For lazy mode, defer features extraction
            features = None
        else:
            sequences = raw_data.astype(np.float64)
            features = sequences[:, -1, :]  # Extract last timestep for flat models
    else:
        # LEGACY format: [N_samples, 98]
        sequences = None
        if lazy:
            features = raw_data
        else:
            features = raw_data.astype(np.float64)
    
    # Load metadata if provided
    metadata = None
    if metadata_path is not None:
        metadata_path = Path(metadata_path)
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)

    # T9: Forward-prices label computation.
    # Determine label source and optionally compute labels from forward_prices.
    from lobtrainer.config.schema import LabelsConfig as _LabelsConfig
    if labels_config is None:
        labels_config = _LabelsConfig()

    forward_prices_data = None
    use_forward_prices = False

    meta_declares_fp = (
        metadata is not None
        and isinstance(metadata.get("forward_prices"), dict)
        and metadata["forward_prices"].get("exported", False)
    )

    if labels_config.source == "forward_prices":
        if not has_forward_prices:
            raise FileNotFoundError(
                f"labels.source='forward_prices' but "
                f"{forward_prices_path} does not exist. "
                f"Ensure the export config sets export_forward_prices=true."
            )
        if not meta_declares_fp:
            raise ValueError(
                f"labels.source='forward_prices' but metadata does not "
                f"declare forward_prices.exported=True for {data_file.stem}."
            )
        use_forward_prices = True
    elif labels_config.source == "auto" and has_forward_prices and meta_declares_fp:
        use_forward_prices = True

    if use_forward_prices:
        forward_prices_data = np.load(
            forward_prices_path, mmap_mode=effective_mmap
        )
        if not lazy:
            forward_prices_data = forward_prices_data.astype(np.float64)

        # Alignment invariant: forward_prices rows must equal sequence rows
        if sequences is not None and forward_prices_data.shape[0] != sequences.shape[0]:
            raise ValueError(
                f"forward_prices has {forward_prices_data.shape[0]} rows but "
                f"sequences has {sequences.shape[0]} rows for "
                f"{data_file.stem}. The export contract requires 1:1 alignment."
            )

        classification, regression_from_fp = _compute_labels_from_forward_prices(
            forward_prices_data, metadata, labels_config
        )

        # Override labels with computed values.
        if classification is not None:
            labels = classification.astype(np.int64)
        else:
            labels = np.zeros(regression_from_fp.shape, dtype=np.int64)
        regression_labels = regression_from_fp

    # T10: compute sample weights if requested
    computed_sample_weights = None
    if (
        labels_config is not None
        and labels_config.sample_weights != "none"
        and not lazy
    ):
        from lobtrainer.data.sample_weights import compute_sample_weights_for_day
        n_weight_samples = labels.shape[0] if labels.ndim >= 1 and labels.shape[0] > 0 else 0
        if n_weight_samples > 0:
            computed_sample_weights = compute_sample_weights_for_day(
                n_samples=n_weight_samples,
                metadata=metadata,
                labels_config=labels_config,
                stride=export_stride,
            )

    # For lazy aligned format, we need to create features from sequences on access
    if lazy and is_aligned and features is None:
        day_data = DayData(
            date=date,
            features=np.empty((0, 0)),
            labels=labels,
            sequences=sequences,
            regression_labels=regression_labels,
            forward_prices=forward_prices_data,
            sample_weights=computed_sample_weights,
            metadata=metadata,
            is_aligned=is_aligned,
            _sequences_path=data_file,
            _labels_path=labels_path,
            _lazy=True,
            _mmap_mode=effective_mmap,
        )
        day_data.features = sequences[:, -1, :] if sequences is not None else np.empty((0, 0))
    else:
        day_data = DayData(
            date=date,
            features=features if features is not None else np.empty((0, 0)),
            labels=labels,
            sequences=sequences,
            regression_labels=regression_labels,
            forward_prices=forward_prices_data,
            sample_weights=computed_sample_weights,
            metadata=metadata,
            is_aligned=is_aligned,
            _sequences_path=data_file if lazy else None,
            _labels_path=labels_path if lazy else None,
            _lazy=lazy,
            _mmap_mode=effective_mmap,
        )
    
    if validate and not lazy:
        day_data.validate()
    
    return day_data


def load_day_metadata_only(
    split_dir: Path,
    date: str,
) -> Tuple[Optional[Dict], int, int]:
    """
    Load only metadata and shapes for a day (no data loaded into RAM).
    
    This is useful for computing normalization stats without loading all data.
    
    Args:
        split_dir: Directory containing the day's files
        date: Date string (e.g., "20250203")
    
    Returns:
        Tuple of (metadata, num_sequences, num_features)
    """
    metadata_path = split_dir / f"{date}_metadata.json"
    sequences_path = split_dir / f"{date}_sequences.npy"
    features_path = split_dir / f"{date}_features.npy"
    
    # Load metadata
    metadata = None
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    
    # Get shape without loading data
    if sequences_path.exists():
        # Use numpy's header reading to get shape without loading
        with open(sequences_path, 'rb') as f:
            version = np.lib.format.read_magic(f)
            if version[0] == 1:
                shape, _, _ = np.lib.format.read_array_header_1_0(f)
            else:
                shape, _, _ = np.lib.format.read_array_header_2_0(f)
        return metadata, shape[0], shape[-1]
    elif features_path.exists():
        with open(features_path, 'rb') as f:
            version = np.lib.format.read_magic(f)
            if version[0] == 1:
                shape, _, _ = np.lib.format.read_array_header_1_0(f)
            else:
                shape, _, _ = np.lib.format.read_array_header_2_0(f)
        return metadata, shape[0], shape[-1]
    
    return metadata, 0, 0


def load_split_data(
    data_dir: Union[str, Path],
    split: str = "train",
    labels_config: Optional["LabelsConfig"] = None,
    validate: bool = True,
    lazy: bool = False,
    mmap_mode: Optional[str] = None,
) -> List[DayData]:
    """
    Load all data for a split (train/val/test).

    Auto-detects format (aligned 3D or legacy 2D).

    Args:
        data_dir: Root data directory
        split: Split name ("train", "val", or "test")
        labels_config: Optional LabelsConfig for label source resolution (T9).
        validate: Whether to validate data integrity
        lazy: If True, use memory-mapped arrays for minimal RAM usage.
              Essential for large datasets that don't fit in memory.
        mmap_mode: Memory-map mode ('r' for read-only). Only used if lazy=True.

    Returns:
        List of DayData instances, sorted by date

    Raises:
        FileNotFoundError: If split directory doesn't exist
    """
    data_dir = Path(data_dir)
    split_dir = data_dir / split

    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    # T10: read export stride from manifest for sample weight computation
    export_stride = 1  # safe default (time-based exports)
    manifest_path = data_dir / "dataset_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as _mf:
            _manifest = json.load(_mf)
        export_stride = _manifest.get("stride", 1)

    # Detect format
    export_format = _detect_export_format(split_dir)

    if export_format == 'aligned':
        data_files = sorted(split_dir.glob("*_sequences.npy"))
        suffix = '_sequences'
    else:
        data_files = sorted(split_dir.glob("*_features.npy"))
        suffix = '_features'

    if not data_files:
        raise FileNotFoundError(f"No data files found in {split_dir}")

    days = []
    contract_validated = False
    expected_feature_count: Optional[int] = None

    for data_file in data_files:
        date = data_file.stem.replace(suffix, '')
        label_path = split_dir / f"{date}_labels.npy"
        regression_label_path = split_dir / f"{date}_regression_labels.npy"
        forward_prices_path = split_dir / f"{date}_forward_prices.npy"
        metadata_path = split_dir / f"{date}_metadata.json"

        has_class = label_path.exists()
        has_reg = regression_label_path.exists()
        has_fp = forward_prices_path.exists()
        if not has_class and not has_reg and not has_fp:
            raise FileNotFoundError(
                f"Missing labels for {date}: no labels.npy, "
                f"regression_labels.npy, or forward_prices.npy in {split_dir}"
            )

        day_data = load_day_data(
            data_file,
            label_path,
            metadata_path if metadata_path.exists() else None,
            regression_labels_path=regression_label_path if has_reg else None,
            forward_prices_path=forward_prices_path if has_fp else None,
            labels_config=labels_config,
            export_stride=export_stride,
            validate=False,
            lazy=lazy,
            mmap_mode=mmap_mode,
        )

        if not contract_validated and day_data.metadata is not None:
            _validate_day_metadata(day_data.metadata, date)
            n_feat = day_data.metadata.get("n_features")
            if n_feat is not None:
                expected_feature_count = int(n_feat)
            contract_validated = True

        if validate and not lazy:
            day_data.validate(expected_feature_count=expected_feature_count)

        days.append(day_data)
    
    mode_str = " (lazy/mmap)" if lazy else ""
    logger.info(f"Loaded {len(days)} days from {split} ({export_format} format){mode_str}")
    return days


# =============================================================================
# PyTorch Datasets
# =============================================================================


class LOBFlatDataset(Dataset):
    """
    PyTorch Dataset for flat LOB features (no sequence dimension).
    
    Use this for models that don't need temporal context (e.g., Logistic, XGBoost, MLP).
    
    For aligned format: Uses last timestep of each sequence.
    For legacy format: Uses raw flat features.
    
    Supports both single-horizon and multi-horizon labels:
        - Single-horizon: Returns scalar label per sample
        - Multi-horizon: Specify horizon_idx to select one horizon, or get all
    
    Args:
        days: List of DayData instances
        transform: Optional transform to apply to features
        feature_indices: Optional list of feature indices to select
        horizon_idx: For multi-horizon labels, which horizon to use (0-based).
                     None returns all horizons (for multi-output models).
        labeling_strategy: Labeling strategy used in the export. Important for
                          correct label encoding:
                          - 'tlob', 'opportunity': Labels are {-1, 0, 1}, shift to {0, 1, 2}
                          - 'triple_barrier': Labels are already {0, 1, 2}, no shift needed
                          If None, auto-detects from metadata or assumes {-1, 0, 1}.
    
    Example:
        >>> days = load_split_data("data/exports/nvda_98feat_full", "train")
        >>> dataset = LOBFlatDataset(days)
        >>> features, label = dataset[0]
        >>> features.shape  # (98,) or (n_selected,)
        
        >>> # Multi-horizon: select horizon index 2
        >>> dataset = LOBFlatDataset(days, horizon_idx=2)
        >>> features, label = dataset[0]
        >>> label.shape  # () - scalar
        
        >>> # Multi-horizon: get all horizons
        >>> dataset = LOBFlatDataset(days, horizon_idx=None)
        >>> features, label = dataset[0]
        >>> label.shape  # (n_horizons,) if multi-horizon
    """
    
    def __init__(
        self,
        days: List[DayData],
        transform: Optional[callable] = None,
        feature_indices: Optional[List[int]] = None,
        horizon_idx: Optional[int] = 0,  # Default: first horizon (backward compatible)
        labeling_strategy: Optional[str] = None,
    ):
        self.days = days
        self.transform = transform
        self.feature_indices = feature_indices
        self.horizon_idx = horizon_idx
        
        # Determine label encoding behavior based on labeling strategy
        # (RULE.md §1: Explicit contracts - label encoding is a critical contract)
        self._labeling_strategy = labeling_strategy
        self._labels_need_shift = _determine_label_shift_from_metadata(labeling_strategy, days)
        
        # Build index mapping: global_idx -> (day_idx, local_idx)
        self._index_map: List[Tuple[int, int]] = []
        for day_idx, day in enumerate(days):
            for local_idx in range(day.num_sequences):
                self._index_map.append((day_idx, local_idx))
        
        # Verify we have aligned data
        self._is_aligned = all(d.is_aligned for d in days)
        if not self._is_aligned:
            logger.warning(
                "Dataset contains legacy (non-aligned) data. "
                "Features may not be correctly aligned with labels."
            )
        
        # Detect multi-horizon
        self._is_multi_horizon = any(d.is_multi_horizon for d in days) if days else False
        if self._is_multi_horizon:
            self._num_horizons = days[0].num_horizons
            logger.info(f"Multi-horizon labels detected: {self._num_horizons} horizons")
            if horizon_idx is not None:
                logger.info(f"Using horizon index {horizon_idx}")
        else:
            self._num_horizons = 1
    
    def __len__(self) -> int:
        return len(self._index_map)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        day_idx, local_idx = self._index_map[idx]
        day = self.days[day_idx]
        
        # Get features (already extracted from sequences for aligned format)
        features = day.features[local_idx].copy()
        
        # Get labels - handle multi-horizon
        if self._is_multi_horizon:
            if self.horizon_idx is not None:
                label = day.get_labels(self.horizon_idx)[local_idx]
            else:
                label = day.labels[local_idx]  # All horizons: [n_horizons]
        else:
            label = day.labels[local_idx]
        
        # IMPORTANT: Order matters for feature selection and normalization!
        # 1. Apply transform (normalization) FIRST on all features
        # 2. Then select features AFTER normalization
        if self.transform is not None:
            features = self.transform(features)
        
        # Select specific features AFTER normalization
        if self.feature_indices is not None:
            features = features[self.feature_indices]
        
        # Handle label tensor dtype
        # Label encoding depends on labeling strategy:
        # - TLOB/Opportunity: {-1, 0, 1} → shift +1 → {0, 1, 2}
        # - Triple Barrier: {0, 1, 2} → no shift (already 0-indexed)
        if isinstance(label, np.ndarray) and label.ndim > 0:
            # Multi-horizon: [n_horizons]
            label_array = label.astype(np.int64)
            if self._labels_need_shift:
                label_array = label_array + 1
            label_tensor = torch.from_numpy(label_array)
        else:
            # Single-horizon: scalar
            label_val = int(label)
            if self._labels_need_shift:
                label_val = label_val + 1
            label_tensor = torch.tensor(label_val, dtype=torch.long)
        
        return (
            torch.from_numpy(features).float(),
            label_tensor,
        )
    
    @property
    def num_features(self) -> int:
        if self.feature_indices is not None:
            return len(self.feature_indices)
        return self.days[0].features.shape[1] if self.days else 0
    
    @property
    def num_classes(self) -> int:
        return 3  # Down, Stable, Up
    
    @property
    def is_aligned(self) -> bool:
        return self._is_aligned
    
    @property
    def is_multi_horizon(self) -> bool:
        """Check if dataset has multi-horizon labels."""
        return self._is_multi_horizon
    
    @property
    def num_horizons(self) -> int:
        """Number of prediction horizons."""
        return self._num_horizons
    
    @property
    def horizons(self) -> Optional[List[int]]:
        """Get horizon values from metadata (if available)."""
        if self.days and self.days[0].horizons:
            return self.days[0].horizons
        return None


class LOBSequenceDataset(Dataset):
    """
    PyTorch Dataset for LOB feature sequences with temporal dimension.
    
    Use this for sequence models (LSTM, Transformer, etc.).
    
    REQUIRES aligned format (*_sequences.npy). For legacy format,
    use LOBFlatDataset or manually window the data.
    
    Supports both single-horizon and multi-horizon labels:
        - Single-horizon: Returns scalar label per sample
        - Multi-horizon: Specify horizon_idx to select one horizon, or get all
    
    Supports label transformation for Two-Stage training:
        - label_transform: Transform 3-class → 2-class for binary signal detection
    
    HMHP Multi-Horizon Mode:
        - Set return_labels_as_dict=True to return labels as {horizon_value: label}
        - Required for HMHP model which expects labels in this format
    
    Args:
        days: List of DayData instances (must be aligned format)
        transform: Optional transform to apply to sequences
        feature_indices: Optional list of feature indices to select
        horizon_idx: For multi-horizon labels, which horizon to use (0-based).
                     None returns all horizons (for multi-output models).
        label_transform: Optional transform to apply to labels (e.g., BinaryLabelTransform).
                        Applied AFTER horizon selection.
        num_classes: Number of output classes (2 for binary, 3 for multiclass).
                    Used for property access; automatically inferred if label_transform provided.
        return_labels_as_dict: If True and horizon_idx is None, returns labels as
                               {horizon_value: label} dict instead of tensor. Required for HMHP.
    
    Example:
        >>> days = load_split_data("data/exports/nvda_98feat_full", "train")
        >>> dataset = LOBSequenceDataset(days)
        >>> sequence, label = dataset[0]
        >>> sequence.shape  # (100, 98) or (100, n_selected)
        
        >>> # Multi-horizon: select horizon index 2
        >>> dataset = LOBSequenceDataset(days, horizon_idx=2)
        >>> sequence, label = dataset[0]
        >>> label.shape  # () - scalar
        
        >>> # HMHP mode: return labels as dict
        >>> dataset = LOBSequenceDataset(
        ...     days,
        ...     horizon_idx=None,  # Return ALL horizons
        ...     return_labels_as_dict=True
        ... )
        >>> sequence, labels = dataset[0]
        >>> labels  # {10: tensor(1), 20: tensor(2), 50: tensor(0), ...}
        
        >>> # Binary signal detection (Two-Stage training)
        >>> from lobtrainer.data import BinaryLabelTransform
        >>> dataset = LOBSequenceDataset(
        ...     days,
        ...     horizon_idx=2,
        ...     label_transform=BinaryLabelTransform(),
        ...     num_classes=2
        ... )
        >>> sequence, label = dataset[0]
        >>> label  # 0 (NoSignal) or 1 (Signal)
    """
    
    def __init__(
        self,
        days: List[DayData],
        transform: Optional[callable] = None,
        feature_indices: Optional[List[int]] = None,
        horizon_idx: Optional[int] = 0,  # Default: first horizon (backward compatible)
        label_transform: Optional[callable] = None,
        num_classes: int = 3,
        return_labels_as_dict: bool = False,
        labeling_strategy: Optional[str] = None,
        return_regression_targets: bool = False,
        use_precomputed_regression: bool = False,
        mid_price_feature_index: int = 40,
        stride: int = 10,
        return_sample_weights: bool = False,
    ):
        """
        Initialize LOB Sequence Dataset.

        Args:
            days: List of DayData instances (must be aligned format)
            transform: Optional transform to apply to sequences
            feature_indices: Optional list of feature indices to select
            horizon_idx: For multi-horizon labels, which horizon to use (0-based).
                         None returns all horizons (for multi-output models).
            label_transform: Optional transform to apply to labels.
            num_classes: Number of output classes (2 for binary, 3 for multiclass).
            return_labels_as_dict: If True and horizon_idx is None, returns labels as
                                   {horizon_value: label} dict. Required for HMHP.
            labeling_strategy: Labeling strategy used in the export. Important for
                              correct label encoding:
                              - 'tlob', 'opportunity': Labels are {-1, 0, 1}, shift to {0, 1, 2}
                              - 'triple_barrier': Labels are already {0, 1, 2}, no shift needed
                              If None, auto-detects from metadata or assumes {-1, 0, 1}.
            return_regression_targets: If True and return_labels_as_dict is True,
                returns precomputed regression targets from regression_labels.npy.
                Requires use_precomputed_regression=True (on-the-fly computation
                is not supported due to formula mismatch with the feature extractor).
            mid_price_feature_index: Index of mid_price in RAW features (before
                selection). Default 40 from pipeline_contract.toml.
            stride: Sequence stride in events. Used to convert horizon events to
                sequence index offsets. Default 10.
            return_sample_weights: If True and day.sample_weights is not None,
                return a 3-tuple (sequence, label, weight) instead of 2-tuple.
                Enable only for the TRAIN split (T10).
        """
        # Verify all days have sequences
        for day in days:
            if day.sequences is None:
                raise ValueError(
                    f"Day {day.date} has no sequences. "
                    "LOBSequenceDataset requires aligned format (*_sequences.npy)."
                )
        
        self.days = days
        self.transform = transform
        self.feature_indices = feature_indices
        self.horizon_idx = horizon_idx
        self.label_transform = label_transform
        self._num_classes = num_classes
        self.return_labels_as_dict = return_labels_as_dict
        self.return_regression_targets = return_regression_targets
        self.use_precomputed_regression = use_precomputed_regression
        self._mid_price_idx = mid_price_feature_index
        self._stride = stride
        self.return_sample_weights = return_sample_weights
        if return_sample_weights:
            for day in days:
                if day.sample_weights is None:
                    raise ValueError(
                        f"return_sample_weights=True but day {day.date} has no "
                        f"sample_weights. Ensure all days have weights computed "
                        f"(set sample_weights='concurrent_overlap' in LabelsConfig) "
                        f"or disable return_sample_weights."
                    )

        if use_precomputed_regression:
            for day in days:
                if day.regression_labels is None:
                    raise ValueError(
                        f"Day {day.date} has no regression_labels but "
                        "use_precomputed_regression=True. Ensure regression_labels.npy "
                        "was loaded via load_split_data()."
                    )
        
        # Determine label encoding behavior based on labeling strategy
        # (RULE.md §1: Explicit contracts - label encoding is a critical contract)
        self._labeling_strategy = labeling_strategy
        self._labels_need_shift = _determine_label_shift_from_metadata(labeling_strategy, days)
        
        # Build index mapping: global_idx -> (day_idx, local_idx)
        self._index_map: List[Tuple[int, int]] = []
        for day_idx, day in enumerate(days):
            for local_idx in range(day.num_sequences):
                self._index_map.append((day_idx, local_idx))
        
        # Pre-compute per-day sequence counts for boundary checks
        self._day_sizes = [day.num_sequences for day in days]
        
        # Detect multi-horizon
        self._is_multi_horizon = any(d.is_multi_horizon for d in days) if days else False
        if self._is_multi_horizon:
            self._num_horizons = days[0].num_horizons
            logger.info(f"Multi-horizon labels detected: {self._num_horizons} horizons")
            if horizon_idx is not None:
                logger.info(f"Using horizon index {horizon_idx}")
        else:
            self._num_horizons = 1
        
        # Validate HMHP mode requirements
        if return_labels_as_dict:
            if not self._is_multi_horizon:
                raise ValueError(
                    "return_labels_as_dict=True requires multi-horizon labels"
                )
            if horizon_idx is not None:
                raise ValueError(
                    "return_labels_as_dict=True requires horizon_idx=None "
                    "(to return all horizons)"
                )
            logger.info("HMHP mode: labels will be returned as {horizon_value: label} dict")
        
        if return_regression_targets:
            if not return_labels_as_dict and not use_precomputed_regression:
                raise ValueError(
                    "Regression targets require either return_labels_as_dict=True (HMHP mode) "
                    "or use_precomputed_regression=True (precomputed labels)."
                )
            logger.info(
                f"Regression targets enabled: mid_price_idx={mid_price_feature_index}, "
                f"stride={stride}"
            )
        
        # Log label transform if provided
        if label_transform is not None:
            logger.info(f"Using label transform: {label_transform}")
    
    def __len__(self) -> int:
        return len(self._index_map)
    
    def __getitem__(self, idx: int):
        day_idx, local_idx = self._index_map[idx]
        day = self.days[day_idx]

        # T10: precompute sample weight tensor (None if disabled)
        _weight = None
        if self.return_sample_weights and day.sample_weights is not None:
            _weight = torch.tensor(
                day.sample_weights[local_idx], dtype=torch.float32
            )

        # Get full sequence
        sequence = day.sequences[local_idx].copy()
        
        # Get labels - handle multi-horizon
        if self._is_multi_horizon:
            if self.horizon_idx is not None:
                label = day.get_labels(self.horizon_idx)[local_idx]
            else:
                label = day.labels[local_idx]  # All horizons: [n_horizons]
        else:
            label = day.labels[local_idx]
        
        # IMPORTANT: Order matters for feature selection and normalization!
        # 1. Apply transform (normalization) FIRST on all features
        # 2. Then select features AFTER normalization
        # This ensures normalization stats are computed on original indices.
        if self.transform is not None:
            sequence = self.transform(sequence)
        
        # Select specific features AFTER normalization
        if self.feature_indices is not None:
            sequence = sequence[:, self.feature_indices]
        
        # Non-HMHP regression: return regression target as simple tensor
        if self.return_regression_targets and self.use_precomputed_regression and not self.return_labels_as_dict:
            reg_row = day.regression_labels[local_idx]
            if self.horizon_idx is not None and reg_row.ndim > 0:
                reg_target = float(reg_row[self.horizon_idx])
            elif reg_row.ndim > 0:
                reg_target = float(reg_row[0])
            else:
                reg_target = float(reg_row)
            result = (
                torch.from_numpy(sequence).float(),
                torch.tensor(reg_target, dtype=torch.float32),
            )
            return (*result, _weight) if _weight is not None else result
        
        # HMHP dict mode: return {horizon_value: label} dict
        if self.return_labels_as_dict:
            horizons = day.horizons
            if horizons is None:
                raise ValueError(
                    "return_labels_as_dict=True requires horizons metadata. "
                    "Check that *_metadata.json files contain 'horizons' field."
                )
            
            # label is [n_horizons] array
            # Apply shift only if labels need it (TLOB/Opportunity format)
            # (RULE.md §1: Explicit label encoding contract)
            label_dict = {}
            for i, h in enumerate(horizons):
                raw_label = int(label[i])
                if self._labels_need_shift:
                    raw_label = raw_label + 1  # {-1,0,1} -> {0,1,2}
                label_dict[h] = torch.tensor(raw_label, dtype=torch.long)
            
            # Regression targets from precomputed regression_labels.npy (preferred)
            if self.return_regression_targets and self.use_precomputed_regression:
                reg_dict = {}
                reg_row = day.regression_labels[local_idx]
                for i, h in enumerate(horizons):
                    reg_dict[h] = torch.tensor(float(reg_row[i]), dtype=torch.float32)
                result = (
                    torch.from_numpy(sequence).float(),
                    label_dict,
                    reg_dict,
                )
                return (*result, _weight) if _weight is not None else result

            # Regression targets without precomputed labels: hard error.
            # On-the-fly computation used a different formula (simple point return)
            # than the feature extractor (TLOB-smoothed return), producing silently
            # wrong targets. Require precomputed regression_labels.npy instead.
            if self.return_regression_targets:
                raise ValueError(
                    "Regression targets require precomputed regression_labels.npy "
                    "from the feature extractor (use_precomputed_regression=True). "
                    "On-the-fly computation is not supported because it uses a "
                    "different formula (point return) than the extractor (smoothed "
                    "return). Re-export data with strategy='regression' in the "
                    "feature extractor config."
                )
            
            result = (
                torch.from_numpy(sequence).float(),
                label_dict,
            )
            return (*result, _weight) if _weight is not None else result

        # Handle label tensor dtype (non-dict mode)
        # Label encoding depends on labeling strategy:
        # - TLOB/Opportunity: {-1, 0, 1} → shift +1 → {0, 1, 2}
        # - Triple Barrier: {0, 1, 2} → no shift (already 0-indexed)
        # (RULE.md §1: Explicit contracts - label encoding is critical)
        if isinstance(label, np.ndarray) and label.ndim > 0:
            # Multi-horizon: [n_horizons]
            label_value = label.astype(np.int64)
            if self._labels_need_shift:
                label_value = label_value + 1  # {-1,0,1} -> {0,1,2}
        else:
            # Single-horizon: scalar
            label_value = int(label)
            if self._labels_need_shift:
                label_value = label_value + 1  # {-1,0,1} -> {0,1,2}
        
        # Apply label transform if provided (e.g., for binary signal detection)
        # This converts 3-class {0=Down, 1=Stable, 2=Up} to 2-class {0=NoSignal, 1=Signal}
        if self.label_transform is not None:
            if isinstance(label_value, np.ndarray):
                label_value = self.label_transform.transform_array(label_value)
            else:
                label_value = self.label_transform(label_value)
        
        # Convert to tensor
        if isinstance(label_value, np.ndarray):
            label_tensor = torch.from_numpy(label_value.astype(np.int64))
        else:
            label_tensor = torch.tensor(label_value, dtype=torch.long)
        
        result = (
            torch.from_numpy(sequence).float(),
            label_tensor,
        )
        return (*result, _weight) if _weight is not None else result

    @property
    def num_features(self) -> int:
        if self.feature_indices is not None:
            return len(self.feature_indices)
        return self.days[0].sequences.shape[2] if self.days else 0
    
    @property
    def num_classes(self) -> int:
        """Number of output classes (2 for binary, 3 for multiclass)."""
        return self._num_classes
    
    @property
    def sequence_length(self) -> int:
        return self.days[0].sequences.shape[1] if self.days else 0
    
    @property
    def sequence_shape(self) -> Tuple[int, int]:
        """Shape of each sequence: (sequence_length, num_features)."""
        return (self.sequence_length, self.num_features)
    
    @property
    def is_multi_horizon(self) -> bool:
        """Check if dataset has multi-horizon labels."""
        return self._is_multi_horizon
    
    @property
    def num_horizons(self) -> int:
        """Number of prediction horizons."""
        return self._num_horizons
    
    @property
    def horizons(self) -> Optional[List[int]]:
        """Get horizon values from metadata (if available)."""
        if self.days and self.days[0].horizons:
            return self.days[0].horizons
        return None


# =============================================================================
# Dataset Factory
# =============================================================================


def _hmhp_collate_fn(batch):
    """
    Custom collate function for HMHP multi-horizon dict labels.

    Handles variable-length tuples:
        - 2-tuple: (sequence, labels)
        - 3-tuple: (sequence, labels, regression_targets) OR (sequence, labels, weight)
        - 4-tuple: (sequence, labels, regression_targets, weight)

    The last element is a weight (scalar tensor) if it's a torch.Tensor with ndim==0.
    Dict elements are regression targets.
    """
    n_items = len(batch[0])
    sequences = torch.stack([item[0] for item in batch])

    horizons = batch[0][1].keys()
    labels = {h: torch.stack([item[1][h] for item in batch]) for h in horizons}

    # Detect optional regression targets (dict) and sample weights (scalar tensor)
    has_reg = n_items >= 3 and isinstance(batch[0][2], dict)
    # Weight is the last element if it's a scalar tensor
    last = batch[0][-1] if n_items >= 3 else None
    has_weight = (
        last is not None
        and isinstance(last, torch.Tensor)
        and last.ndim == 0
    )

    result = [sequences, labels]
    if has_reg:
        reg_horizons = batch[0][2].keys()
        regression_targets = {
            h: torch.stack([item[2][h] for item in batch]) for h in reg_horizons
        }
        result.append(regression_targets)
    if has_weight:
        weight_idx = n_items - 1
        weights = torch.stack([item[weight_idx] for item in batch])
        result.append(weights)

    return tuple(result)


def create_dataloaders(
    data_dir: Union[str, Path],
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    transform: Optional[callable] = None,
    feature_indices: Optional[List[int]] = None,
    use_sequences: bool = True,
    horizon_idx: Optional[int] = 0,
    return_labels_as_dict: bool = False,
    labeling_strategy: Optional[str] = None,
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create DataLoaders for train/val/test splits.
    
    Args:
        data_dir: Root data directory
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        transform: Optional feature transform
        feature_indices: Optional feature selection
        use_sequences: If True, return LOBSequenceDataset; if False, return LOBFlatDataset
        horizon_idx: For multi-horizon labels, which horizon to use (0-based).
                     None returns all horizons (for multi-output models).
        return_labels_as_dict: If True and horizon_idx is None, return labels as
                               {horizon_value: labels [B]} dict. Required for HMHP.
        labeling_strategy: Labeling strategy ('tlob', 'opportunity', 'triple_barrier').
                          If None, auto-detects from metadata.
    
    Returns:
        Dict with 'train', 'val', 'test' DataLoaders
    """
    from torch.utils.data import DataLoader
    
    loaders = {}
    
    # Use custom collate function for HMHP dict labels
    collate_fn = _hmhp_collate_fn if return_labels_as_dict else None
    
    for split in ["train", "val", "test"]:
        try:
            days = load_split_data(data_dir, split, validate=True)
        except FileNotFoundError:
            logger.info(f"Split '{split}' not found, skipping")
            continue
        
        # Choose dataset type
        if use_sequences:
            try:
                dataset = LOBSequenceDataset(
                    days,
                    transform=transform,
                    feature_indices=feature_indices,
                    horizon_idx=horizon_idx,
                    return_labels_as_dict=return_labels_as_dict,
                    labeling_strategy=labeling_strategy,
                )
            except ValueError:
                logger.warning(
                    f"Split '{split}' has no sequences, falling back to flat dataset"
                )
                dataset = LOBFlatDataset(
                    days,
                    transform=transform,
                    feature_indices=feature_indices,
                    horizon_idx=horizon_idx,
                    labeling_strategy=labeling_strategy,
                )
        else:
            dataset = LOBFlatDataset(
                days,
                transform=transform,
                feature_indices=feature_indices,
                horizon_idx=horizon_idx,
                labeling_strategy=labeling_strategy,
            )
        
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(split == "train"),
            collate_fn=collate_fn,
        )
    
    return loaders


# =============================================================================
# NumPy-based data loading (for sklearn models)
# =============================================================================


def load_numpy_data(
    data_dir: Union[str, Path],
    split: str = "train",
    feature_indices: Optional[List[int]] = None,
    flat: bool = True,
    horizon_idx: Optional[int] = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load data as NumPy arrays (for sklearn/XGBoost).
    
    Args:
        data_dir: Root data directory
        split: Split name
        feature_indices: Optional feature selection
        flat: If True, return 2D features; if False, return 3D sequences
        horizon_idx: For multi-horizon labels, which horizon to use (0-based).
                     None returns all horizons as 2D array [N, n_horizons].
    
    Returns:
        Tuple of (features, labels) arrays
        - features: [N, 98] if flat, [N, 100, 98] if sequences
        - labels: [N] if single horizon or horizon_idx specified, [N, n_horizons] otherwise
    """
    days = load_split_data(data_dir, split, validate=True)
    
    features_list = []
    labels_list = []
    
    for day in days:
        if flat:
            features_list.append(day.features)
        else:
            if day.sequences is None:
                raise ValueError(f"Day {day.date} has no sequences")
            features_list.append(day.sequences)
        
        # Handle horizon selection
        if horizon_idx is not None:
            labels_list.append(day.get_labels(horizon_idx))
        else:
            labels_list.append(day.labels)
    
    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    
    # Select features
    if feature_indices is not None:
        if flat:
            features = features[:, feature_indices]
        else:
            features = features[:, :, feature_indices]
    
    return features, labels


def get_dataset_info(data_dir: Union[str, Path]) -> Dict:
    """
    Get information about a dataset without loading all data.
    
    Returns:
        Dict with keys:
        - num_days: Number of trading days
        - is_multi_horizon: Whether labels have multiple horizons
        - num_horizons: Number of prediction horizons
        - horizons: List of horizon values (if available in metadata)
        - feature_count: Number of features
        - export_format: 'aligned' or 'legacy'
    """
    data_dir = Path(data_dir)
    
    # Try to find any split
    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        if split_dir.exists():
            break
    else:
        raise FileNotFoundError(f"No splits found in {data_dir}")
    
    export_format = _detect_export_format(split_dir)
    
    # Load first day to get info
    if export_format == 'aligned':
        data_files = sorted(split_dir.glob("*_sequences.npy"))
        suffix = '_sequences'
    else:
        data_files = sorted(split_dir.glob("*_features.npy"))
        suffix = '_features'
    
    if not data_files:
        raise FileNotFoundError(f"No data files in {split_dir}")
    
    first_file = data_files[0]
    date = first_file.stem.replace(suffix, '')
    label_path = split_dir / f"{date}_labels.npy"
    metadata_path = split_dir / f"{date}_metadata.json"
    
    # Load labels to check shape
    labels = np.load(label_path)
    
    # Load metadata if available
    metadata = None
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    
    # Count total days
    total_days = 0
    for s in ["train", "val", "test"]:
        s_dir = data_dir / s
        if s_dir.exists():
            if export_format == 'aligned':
                total_days += len(list(s_dir.glob("*_sequences.npy")))
            else:
                total_days += len(list(s_dir.glob("*_features.npy")))
    
    # Get feature count
    if export_format == 'aligned':
        data = np.load(first_file)
        feature_count = data.shape[2] if len(data.shape) == 3 else data.shape[1]
    else:
        data = np.load(first_file)
        feature_count = data.shape[1]
    
    # Multi-horizon info
    is_multi_horizon = labels.ndim == 2
    num_horizons = labels.shape[1] if is_multi_horizon else 1
    
    horizons = None
    if metadata:
        # New format: horizons at top level
        if 'horizons' in metadata:
            horizons = metadata['horizons']
        # Legacy format: nested in label_config
        elif 'label_config' in metadata:
            horizons = metadata['label_config'].get('horizons')
    
    return {
        'num_days': total_days,
        'is_multi_horizon': is_multi_horizon,
        'num_horizons': num_horizons,
        'horizons': horizons,
        'feature_count': feature_count,
        'export_format': export_format,
    }
