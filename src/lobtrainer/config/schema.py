"""
Configuration schema for LOB Model Trainer.

All configuration is done via type-safe dataclasses that can be serialized
to YAML/JSON for experiment tracking and reproducibility.

Design principles (per RULE.md):
- All thresholds, behaviors, and hyperparameters via configuration
- Sensible defaults with full override capability
- Configs are serializable for experiment tracking
- Document valid ranges and constraints for each parameter

Usage:
    >>> config = ExperimentConfig.from_yaml("configs/baseline.yaml")
    >>> config.data.feature_count  # 98
    >>> config.train.learning_rate  # 1e-4
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional, List, Tuple, Any, ClassVar, Dict, TYPE_CHECKING
import json
import math  # Phase A.5.3a.1 (2026-04-24): isfinite() guard against NaN/Inf float inputs.
import yaml

# Phase A.5.3a (2026-04-24): Pydantic v2 migration starts with LabelsConfig.
# Subsequent A.5.3b-i migrate the remaining 8 config classes; A.5.3i retires
# dacite entirely. Imported at module level because BaseModel base-class
# declaration must be available at class-definition time (not lazy).
# Phase A.5.3a.1 (2026-04-24 post-audit): + field_validator for list→tuple
# coercion on `horizons` (preserves YAML ergonomics under strict=True).
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Phase 8C-α Stage C.1 trainer wire-in (2026-04-20 post-audit round-2):
# ImportanceConfig cannot be imported at schema.py module-load time —
# doing so triggers lobtrainer.training.__init__.py → trainer.py →
# lobtrainer.config (circular). Use TYPE_CHECKING guard for static
# type-checkers; do runtime conversion via __post_init__ on
# ExperimentConfig (see `_coerce_importance` below). dacite loaders
# pass through the raw dict; we convert to ImportanceConfig after
# the dataclass is constructed.
if TYPE_CHECKING:
    from lobtrainer.training.importance.config import ImportanceConfig  # noqa: F401


# =============================================================================
# Enums for configuration options
# =============================================================================


class NormalizationStrategy(str, Enum):
    """Normalization strategy for features."""
    
    NONE = "none"
    """No normalization applied."""
    
    ZSCORE = "zscore"
    """Per-sample Z-score: (x - mean) / std."""
    
    ZSCORE_PER_DAY = "zscore_per_day"
    """Per-day Z-score: normalize within each trading day."""
    
    GLOBAL_ZSCORE = "global_zscore"
    """
    Global Z-score matching official TLOB repository.
    
    Computes ONE mean/std for ALL prices and ONE mean/std for ALL sizes
    from the ENTIRE training set, then applies to train/val/test.
    
    This matches TLOB/utils/utils_data.py::z_score_orderbook():
        mean_prices = ALL_PRICES.stack().mean()  (one value for all 20 price cols)
        std_prices  = ALL_PRICES.stack().std()   (one value for all 20 price cols)
        mean_sizes  = ALL_SIZES.stack().mean()   (one value for all 20 size cols)
        std_sizes   = ALL_SIZES.stack().std()    (one value for all 20 size cols)
    
    Reference: TLOB GitHub repository preprocessing
    """
    
    MINMAX = "minmax"
    """Min-max scaling to [0, 1]."""
    
    ROBUST = "robust"
    """Robust scaling using median and IQR."""
    
    HYBRID = "hybrid"
    """
    Hybrid normalization for 98-feature datasets.
    
    Applies different strategies to different feature groups:
    - Raw LOB (0-39): Global Z-score (prices pooled, sizes pooled)
    - Derived/MBO/Signals (40-91, 95): Per-feature Z-score
    - Categorical/Binary (92-94, 96-97): No normalization
    
    Use this for full 98-feature datasets where features have heterogeneous scales.
    """


class LabelEncoding(str, Enum):
    """Label encoding scheme."""
    
    CATEGORICAL = "categorical"
    """Integer labels: 0=Down, 1=Stable, 2=Up."""
    
    BINARY_UP = "binary_up"
    """Binary: 1 if Up, 0 otherwise."""
    
    BINARY_DOWN = "binary_down"
    """Binary: 1 if Down, 0 otherwise."""
    
    BINARY_SIGNAL = "binary_signal"
    """Binary: 1 if Up OR Down (any signal), 0 if Stable (no opportunity)."""


class LabelingStrategy(str, Enum):
    """
    Labeling strategy used in the exported data.
    
    Each strategy has different semantic meanings for its class indices,
    which affects how metrics should be computed and interpreted.
    """
    
    OPPORTUNITY = "opportunity"
    """
    Opportunity detection: peak return within horizon.
    Classes: 0=BigDown, 1=NoOpportunity, 2=BigUp
    Goal: Detect when a significant move will occur.
    """
    
    TRIPLE_BARRIER = "triple_barrier"
    """
    Triple Barrier: which barrier is hit first.
    Classes: 0=StopLoss, 1=Timeout, 2=ProfitTarget
    Goal: Model entry → hold → exit trading logic.
    Timeout means "don't trade" - no clear outcome expected.
    
    Reference: López de Prado (2018), "Advances in Financial Machine Learning"
    """
    
    TLOB = "tlob"
    """
    TLOB/DeepLOB: smoothed endpoint return.
    Classes: 0=Down, 1=Stable, 2=Up
    Goal: Predict price direction at horizon.
    
    Reference: Zhang et al. (2019), "DeepLOB"
    """

    REGRESSION = "regression"
    """
    Regression: continuous forward returns in basis points.
    No discrete classes. Output is float64 bps at each horizon.
    Goal: Predict return magnitude, not just direction.
    """


def _coerce_importance(config: Any) -> None:
    """Phase 8C-α Stage C.1 trainer wire-in helper.

    Coerces ``config.importance`` from dict (YAML-load default) or
    None → ImportanceConfig instance, OR leaves a pre-constructed
    ImportanceConfig intact. Deferred import breaks the schema.py ↔
    training.importance.config circular dependency at module load
    time. ImportanceConfig's own ``__post_init__`` validates field
    ranges (n_permutations >= 1, etc.), so we just trigger
    construction.

    Mutates ``config`` in place (ExperimentConfig is NOT frozen).
    """
    imp = getattr(config, "importance", None)
    if imp is None:
        return  # already None — disabled, nothing to do
    if isinstance(imp, dict):
        from lobtrainer.training.importance.config import ImportanceConfig
        config.importance = ImportanceConfig(**imp)
        return
    # Round-3 post-audit Agent-4 H2 fix: real isinstance check instead
    # of duck-typing. We're inside __post_init__ — the training package
    # has finished importing its parents before this runs (called via
    # dataclass machinery, not at module-import time) — so we can
    # safely import and do a proper isinstance check. Duck-typing
    # (hasattr 'enabled' + 'method') accepts arbitrary objects
    # including partially-corrupt deserializations; strict check
    # prevents silent garbage-in.
    from lobtrainer.training.importance.config import ImportanceConfig
    if isinstance(imp, ImportanceConfig):
        return
    raise TypeError(
        f"ExperimentConfig.importance must be None, a dict, or an "
        f"ImportanceConfig instance; got {type(imp).__name__}."
    )


@dataclass
class SourceConfig:
    """Single data source specification (T12).

    Used in DataConfig.sources for multi-source fusion.

    Args:
        name: Unique identifier (e.g., "mbo", "basic").
        data_dir: Path to export directory containing train/val/test/.
        role: "primary" (labels + features) or "auxiliary" (features only).
            Exactly one source must be primary.
    """

    name: str = "mbo"
    data_dir: str = ""
    role: str = "primary"
    feature_count: int = 0
    """Feature count for this source. 0 = auto-detect at load time.
    When all sources specify feature_count > 0, model.input_size can
    be auto-derived as the sum (T13)."""

    def __post_init__(self) -> None:
        if self.role not in ("primary", "auxiliary"):
            raise ValueError(
                f"SourceConfig.role must be 'primary' or 'auxiliary', "
                f"got {self.role!r}"
            )
        if self.feature_count < 0:
            raise ValueError(
                f"SourceConfig.feature_count must be >= 0, "
                f"got {self.feature_count}"
            )


class LabelsConfig(BaseModel):
    """Unified label specification for training.

    Replaces the overlap between DataConfig.labeling_strategy and
    DataConfig.horizon_idx with a single source of truth. Backward
    compatible via auto-derivation in DataConfig.__post_init__; legacy
    fields remain supported with DeprecationWarning.

    **Phase A.5.3a (2026-04-24)**: migrated from ``@dataclass`` to Pydantic
    v2 ``BaseModel`` with ``frozen=True, extra="forbid"``. Retires four bug
    classes at the TYPE layer:

        1. Canonical-path-drift (``config.labels`` vs ``config.data.labels``)
           — Pydantic rejects unknown attribute access natively.
        2. Silent mutation — ``config.data.labels.task = "..."`` raises
           ``ValidationError`` (frozen=True).
        3. Extra-field acceptance — typos like ``horizen_idx`` rejected at
           ``model_validate`` time (extra="forbid").
        4. Silent-None field access — Pydantic's ``__init__`` validates
           required fields at construction, not at first read.

    See ``hft-contracts/tests/fixtures/pre_pydantic_label_strategy_hash.json``
    for the byte-identity lock across the dataclass→BaseModel migration —
    ``compute_label_strategy_hash(LabelsConfig())`` must produce the same
    SHA-256 hash post-migration as the frozen dataclass fixture (verified
    by ``test_label_strategy_hash_real_pydantic_parity`` in this repo's
    ``tests/test_config.py`` + the mock-Pydantic parity test in
    hft-contracts itself).

    DESIGN INVARIANT: smoothing_window is NEVER a field on this dataclass.
    It is read exclusively from ForwardPriceContract.smoothing_window_offset,
    which is baked into the export metadata at extraction time.

    Rationale (Bug B2, 2026-04-12): passing k=3 when Rust exported k=5
    produces 34.8 bps silent label error (25x Deep ITM breakeven).

    Fields:
        source: Label computation source.
            - "auto" (default): If forward_prices.npy exists and metadata
              declares exported=True, compute via LabelFactory. Else load
              precomputed labels from disk.
            - "forward_prices": Require forward_prices.npy; compute via
              LabelFactory. Raises if missing.
            - "precomputed": Load {day}_labels.npy / {day}_regression_labels.npy.

        return_type: Which LabelFactory return function to use.
            Only used when source is forward_prices or auto (with fp present).

        task: Whether to produce regression (continuous bps) or classification
            ({-1, 0, +1}) labels.
            - "auto" (default): Detect from metadata label_strategy field.

        threshold_bps: Classification threshold in bps. Only used when
            task="classification".

        horizons: Event counts to materialize (e.g. [10, 60, 300]).
            Empty list (default) = use all exported horizons from metadata.

        primary_horizon_idx: Index INTO the resolved horizons list.
            int >= 0 = single-horizon mode. None = all-horizons / HMHP mode.
    """

    # Pydantic v2 configuration — retires 4 bug classes at the TYPE layer.
    # See class docstring for the full rationale.
    #
    # ``frozen=True``: field assignment raises ValidationError. Closes bug
    #   class #2 (silent mutation) for the most common idiom (``cfg.source = X``).
    # ``extra="forbid"``: unknown kwargs raise ValidationError at construction.
    #   Closes bug class #3 (extra-field acceptance / typo propagation).
    # ``strict=True`` (Phase A.5.3a.1): NO implicit type coercion. Rejects
    #   string-to-int for ``horizons: Tuple[int, ...]`` (e.g. ``horizons=["10"]``
    #   — empirically confirmed would coerce under lax mode), bool-to-int
    #   for ``primary_horizon_idx: Optional[int]`` (e.g. ``primary_horizon_idx=True``
    #   would coerce to 1). Without this flag, Pydantic v2 lax mode silently
    #   coerces — violating hft-rules §5 fail-fast + introducing NEW bug
    #   classes not intended to exist.
    #   NOTE: strict does NOT reject NaN/Inf floats (IEEE 754 considers them
    #   valid floats); the explicit ``math.isfinite`` check in
    #   ``_validate_all`` closes that separate gap.
    #
    # Note: ``validate_assignment=True`` is NOT set — empirically confirmed
    # it does NOT fix ``model_copy(update={...})`` bypass (Pydantic v2's
    # ``model_copy`` uses ``model_construct`` which ALWAYS skips validation).
    # The fix for that bug class is the custom ``model_copy`` override
    # below — closes the validator-bypass vector structurally.
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
    )

    source: str = "auto"
    return_type: str = "smoothed_return"
    task: str = "auto"
    threshold_bps: float = 8.0
    # Phase A.5.3a.1 (2026-04-24): type changed from ``List[int]`` to
    # ``Tuple[int, ...]`` for true immutability. With ``frozen=True``,
    # Pydantic blocks ``cfg.horizons = [...]`` assignment but does NOT
    # block ``cfg.horizons.append(99)`` (mutable-container bypass,
    # empirically confirmed). Tuple is immutable — ``.append`` raises
    # AttributeError. YAML input is a list; the ``@field_validator(mode="before")``
    # below coerces list→tuple BEFORE strict type-check runs. Byte-identity
    # preserved: ``sanitize_for_hash`` canonicalizes tuples → lists before
    # hashing, so hash output is unchanged.
    horizons: Tuple[int, ...] = Field(default_factory=tuple)
    primary_horizon_idx: Optional[int] = 0
    sample_weights: str = "none"
    """Sample weighting method to correct for non-IID overlapping labels.
    - "none" (default): no weighting, standard uniform loss.
    - "concurrent_overlap": de Prado (2018) AFML §4.5.1 — weight inversely
      proportional to label concurrency. Corrects for overlapping horizons.
    """

    # v3-A (Phase A.5 Scope D v4) — ``ClassVar[frozenset[str]]`` annotation
    # is LOAD-BEARING under Pydantic v2. Without it, Pydantic treats these
    # four constants as model FIELDS and leaks them into ``model_dump()``,
    # which would break the byte-identity of every stored ``compatibility_
    # fingerprint`` across the dataclass→BaseModel migration. Regression
    # locked by ``test_labels_config_class_constants_not_in_model_dump``.
    _VALID_SOURCES: ClassVar[frozenset[str]] = frozenset(
        {"auto", "precomputed", "forward_prices"}
    )
    _VALID_RETURN_TYPES: ClassVar[frozenset[str]] = frozenset(
        {"smoothed_return", "point_return", "mean_return", "peak_return"}
    )
    _VALID_TASKS: ClassVar[frozenset[str]] = frozenset(
        {"auto", "regression", "classification"}
    )
    _VALID_SAMPLE_WEIGHTS: ClassVar[frozenset[str]] = frozenset(
        {"none", "concurrent_overlap"}
    )

    @field_validator("horizons", mode="before")
    @classmethod
    def _coerce_horizons_input_to_tuple(cls, v: Any) -> Any:
        """Phase A.5.3a.1 (2026-04-24): accept list input from YAML/dacite
        while enforcing strict ``Tuple[int, ...]`` at validation time.

        YAML loaders emit ``horizons: [10, 60]`` as a Python list. Under
        ``strict=True``, Pydantic rejects list→tuple coercion at the type
        check. This pre-validator (``mode="before"``) fires BEFORE strict
        type-check, converting list→tuple cleanly. Strings inside the list
        still get rejected by the downstream ``Tuple[int, ...]`` check
        (the strict-mode gate is applied AFTER this coercion per Pydantic
        v2 validator ordering).
        """
        if isinstance(v, list):
            return tuple(v)
        return v

    def model_copy(self, *, update: Optional[Dict[str, Any]] = None, deep: bool = False) -> "LabelsConfig":
        """Override Pydantic v2 ``model_copy`` to re-run validators on update.

        Phase A.5.3a.1 (2026-04-24): default Pydantic v2 ``model_copy(update=...)``
        uses ``model_construct`` internally which SKIPS ALL validation.
        Empirically confirmed: ``cfg.model_copy(update={"source": "bogus"})``
        produces an invalid LabelsConfig with no ValidationError. This
        re-opens bug class #3 (extra-field acceptance / invalid-value
        acceptance) through the common Pydantic idiom.

        This override forces re-validation via ``model_validate`` when
        update is provided — closes the validator-bypass vector at the
        structural level (callers don't need to remember to use
        ``model_validate({**dump(), **overrides})``).

        Trade-off: slight perf cost on ``model_copy(update=...)`` (full
        validation vs fast construct). Acceptable for a config class
        that is rarely copy-mutated in hot paths.

        Args:
            update: Dict of field overrides. If non-empty, triggers full
                re-validation. If None or empty, falls through to fast path.
            deep: Deep copy flag, passed through to super().
        """
        if update:
            return self.__class__.model_validate({**self.model_dump(), **update})
        return super().model_copy(deep=deep)

    @model_validator(mode="after")
    def _validate_all(self) -> "LabelsConfig":
        """Pydantic equivalent of the legacy ``__post_init__``.

        ``mode="after"`` runs on the fully-constructed model, mirroring
        dataclass ``__post_init__`` semantics. Raises ``ValueError``
        (automatically wrapped in ``pydantic.ValidationError`` by the
        framework) on any invariant violation.

        Returning ``self`` is the Pydantic convention — allows chainable
        post-validation copies via ``model_copy(update=...)`` where
        needed. We never mutate here (frozen=True would raise anyway).
        """
        if self.source not in self._VALID_SOURCES:
            raise ValueError(
                f"LabelsConfig.source must be one of "
                f"{sorted(self._VALID_SOURCES)}, got {self.source!r}"
            )
        if self.return_type not in self._VALID_RETURN_TYPES:
            raise ValueError(
                f"LabelsConfig.return_type must be one of "
                f"{sorted(self._VALID_RETURN_TYPES)}, got {self.return_type!r}"
            )
        if self.task not in self._VALID_TASKS:
            raise ValueError(
                f"LabelsConfig.task must be one of "
                f"{sorted(self._VALID_TASKS)}, got {self.task!r}"
            )
        if not math.isfinite(self.threshold_bps):
            # Phase A.5.3a.1 (2026-04-24): strict=True rejects str→float
            # coercion but accepts NaN/Inf as valid floats per IEEE 754.
            # Explicit isfinite check closes the silent-NaN gap — a NaN
            # threshold would poison every classification filter downstream.
            raise ValueError(
                f"LabelsConfig.threshold_bps must be a finite float "
                f"(not NaN/Inf), got {self.threshold_bps!r}"
            )
        if self.threshold_bps < 0:
            raise ValueError(
                f"LabelsConfig.threshold_bps must be >= 0, "
                f"got {self.threshold_bps}"
            )
        if self.primary_horizon_idx is not None and self.primary_horizon_idx < 0:
            raise ValueError(
                f"LabelsConfig.primary_horizon_idx must be >= 0 or None, "
                f"got {self.primary_horizon_idx}"
            )
        if any(h < 1 for h in self.horizons):
            raise ValueError(
                f"LabelsConfig.horizons must all be >= 1 "
                f"(horizon 0 is degenerate), got {self.horizons}"
            )
        if len(self.horizons) != len(set(self.horizons)):
            raise ValueError(
                f"LabelsConfig.horizons contains duplicates: {self.horizons}"
            )
        if self.sample_weights not in self._VALID_SAMPLE_WEIGHTS:
            raise ValueError(
                f"LabelsConfig.sample_weights must be one of "
                f"{sorted(self._VALID_SAMPLE_WEIGHTS)}, "
                f"got {self.sample_weights!r}"
            )
        # Note: horizons tuple immutability is now enforced at the TYPE
        # layer via ``horizons: Tuple[int, ...]`` field declaration +
        # ``_coerce_horizons_input_to_tuple`` @field_validator(mode="before").
        # The previous post-validation object.__setattr__ is obsolete.
        return self


class TaskType(str, Enum):
    """Task type for training."""
    
    MULTICLASS = "multiclass"
    """Standard 3-class classification: Down/Stable/Up."""
    
    BINARY_SIGNAL = "binary_signal"
    """Binary signal detection: Signal (Up or Down) vs NoSignal (Stable)."""
    
    REGRESSION = "regression"
    """Continuous return prediction: predict bps forward return."""


class LossType(str, Enum):
    """Loss function type."""
    
    CROSS_ENTROPY = "cross_entropy"
    """Standard cross-entropy loss."""
    
    FOCAL = "focal"
    """Focal loss for class imbalance."""
    
    WEIGHTED_CE = "weighted_ce"
    """Cross-entropy with inverse-frequency class weights."""
    
    MSE = "mse"
    """Mean squared error (for regression)."""
    
    HUBER = "huber"
    """Huber loss (robust regression, less sensitive to outliers)."""
    
    HETEROSCEDASTIC = "heteroscedastic"
    """Heteroscedastic regression: jointly predicts mean and variance."""

    GMADL = "gmadl"
    """Generalized Mean Absolute Directional Loss (Michankov et al. 2024)."""


class ModelType(str, Enum):
    """Model architecture type."""
    
    LOGISTIC = "logistic"
    """Logistic regression baseline."""
    
    XGBOOST = "xgboost"
    """XGBoost classifier. NOT in ModelRegistry — use scripts/analysis/train_xgboost_baseline.py directly."""

    LSTM = "lstm"
    """LSTM sequence model."""

    GRU = "gru"
    """GRU sequence model."""

    TRANSFORMER = "transformer"
    """NOT IMPLEMENTED — reserved for future use."""
    
    DEEPLOB = "deeplob"
    """DeepLOB (CNN + LSTM)."""
    
    TLOB = "tlob"
    """TLOB (Transformer LOB with dual attention). Berti & Kasneci 2025."""
    
    HMHP = "hmhp"
    """Hierarchical Multi-Horizon Predictor (classification)."""
    
    HMHP_REGRESSION = "hmhp_regression"
    """Hierarchical Multi-Horizon Regressor — pure regression variant.
    Predicts continuous bps returns instead of class probabilities.
    Same SharedEncoder and cascading architecture as HMHP.
    
    Original HMHP docstring:
    Hierarchical Multi-Horizon Predictor.
    
    Cascading multi-horizon model where shorter-horizon predictions inform
    and confirm longer-horizon predictions. Outputs for all horizons simultaneously.
    
    Key features:
    - Shared encoder (TLOB-based or linear)
    - Per-horizon decoders with state passing
    - Confirmation module for cross-horizon agreement
    - Multi-task loss with per-horizon weights
    
    Reference: docs/HMHP_ARCHITECTURE.md
    """

    TEMPORAL_RIDGE = "temporal_ridge"
    """Ridge regression on hand-crafted temporal features.
    Non-PyTorch (sklearn). Achieves IC=0.616 (91% of TLOB).
    Uses lobmodels.models.simple.TemporalRidge.
    """

    TEMPORAL_GRADBOOST = "temporal_gradboost"
    """GradientBoosting on hand-crafted temporal features.
    Non-PyTorch (sklearn). Achieves R²=0.397 (85.6% of TLOB).
    Uses lobmodels.models.simple.TemporalGradBoost.
    """


# =============================================================================
# Data Configuration
# =============================================================================


@dataclass
class SequenceConfig:
    """
    Configuration for sequence construction from flat features.
    
    Sequences are built by sliding a window over flat feature vectors.
    Labels are aligned to the END of each window.
    
    Example:
        window_size=100, stride=10:
        - Window 0: features[0:100], label for sample 99
        - Window 1: features[10:110], label for sample 109
        - ...
    """
    
    window_size: int = 100
    """Number of samples per sequence. Must match Rust export config."""
    
    stride: int = 10
    """Step size between consecutive sequences. Smaller = more overlap."""
    
    def __post_init__(self) -> None:
        if self.window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {self.window_size}")
        if self.stride < 1:
            raise ValueError(f"stride must be >= 1, got {self.stride}")
        if self.stride > self.window_size:
            raise ValueError(
                f"stride ({self.stride}) should not exceed window_size ({self.window_size})"
            )


@dataclass
class NormalizationConfig:
    """
    Configuration for feature normalization.
    
    Normalization is applied per-feature (column-wise) using statistics
    computed from training data only to avoid data leakage.
    """
    
    strategy: NormalizationStrategy = NormalizationStrategy.ZSCORE_PER_DAY
    """Normalization strategy. Default: per-day Z-score."""
    
    eps: float = 1e-8
    """Small constant to prevent division by zero."""
    
    clip_value: Optional[float] = 10.0
    """
    Clip normalized values to [-clip_value, clip_value] to handle outliers.
    Set to None to disable clipping.
    """
    
    exclude_features: List[int] = field(default_factory=list)
    """
    Feature indices to exclude from normalization (e.g., categorical features).
    Default: [93] (time_regime is categorical).
    """
    
    def __post_init__(self) -> None:
        if self.eps <= 0:
            raise ValueError(f"eps must be > 0, got {self.eps}")
        if self.clip_value is not None and self.clip_value <= 0:
            raise ValueError(f"clip_value must be > 0, got {self.clip_value}")
        # Note: exclude_features should be explicitly set in the config
        # for datasets with >93 features (to exclude TIME_REGIME).
        # For 40-feature datasets (LOB only), no categorical features exist.
        # The default empty list is intentional to avoid out-of-bounds errors.


@dataclass
class DataConfig:
    """
    Configuration for data loading and preprocessing.
    
    Data is loaded from NumPy arrays exported by the Rust pipeline.
    """
    
    data_dir: str = "../data/exports/nvda_11month_complete"
    """
    Path to exported data directory (relative to lob-model-trainer or absolute).
    Directory should contain train/, val/, test/ subdirectories.
    
    Current datasets:
    - nvda_11month_complete: 234 days, TLOB multi-horizon labels
    - nvda_11month_triple_barrier: 234 days, Triple Barrier labels
    """
    
    feature_count: int = 98
    """
    Number of features per sample. Must match Rust export AND model.input_size.
    
    Supported configurations:
    - 40: Raw LOB only (official TLOB repo, DeepLOB benchmark)
          Features: 10 levels × 4 (ask_price, ask_size, bid_price, bid_size)
    - 48: LOB + derived features (FI-2010 style)
    - 76: LOB + derived + MBO features
    - 98: Full feature set (LOB + derived + MBO + signals)
    - 116: Full + experimental (98 + institutional_v2 + volatility + seasonality)
    
    IMPORTANT: Changing this requires also changing model.input_size.
    """
    
    sequence: SequenceConfig = field(default_factory=SequenceConfig)
    """Sequence construction configuration."""
    
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    """Feature normalization configuration."""
    
    label_encoding: LabelEncoding = LabelEncoding.CATEGORICAL
    """Label encoding scheme."""
    
    labeling_strategy: LabelingStrategy = LabelingStrategy.TLOB
    """
    Labeling strategy used in the exported data.
    
    This determines the semantic meaning of class indices and affects
    how metrics are computed and interpreted:
    
    - TLOB: 0=Down, 1=Stable, 2=Up (smoothed endpoint return)
    - TRIPLE_BARRIER: 0=StopLoss, 1=Timeout, 2=ProfitTarget
    - OPPORTUNITY: 0=BigDown, 1=NoOpportunity, 2=BigUp
    
    Must match the strategy used when exporting data from Rust.
    Default: TLOB (most common for DeepLOB/TLOB experiments).
    """
    
    num_classes: int = 3
    """Number of output classes. Default: 3 (Down, Stable, Up)."""
    
    cache_in_memory: bool = True
    """Whether to cache all data in memory. Faster but uses more RAM."""

    # T9: unified label configuration. None = derive from legacy fields below.
    labels: Optional[LabelsConfig] = None
    """
    Unified label specification (T9). When provided, takes precedence over
    legacy labeling_strategy and horizon_idx fields.

    Set to None (default) to auto-derive from legacy fields for backward
    compatibility. See LabelsConfig docstring for full field documentation.
    """

    horizon_idx: Optional[int] = 0
    """
    Which prediction horizon to use for multi-horizon datasets.
    
    Index into the horizons array from the export config. For example:
        horizons = [10, 20, 50, 100, 200] in the Rust export config:
        - horizon_idx=0 → 10 samples ahead (~1 second for 10ms sampling)
        - horizon_idx=1 → 20 samples ahead (~2 seconds)
        - horizon_idx=2 → 50 samples ahead (~5 seconds)
        - horizon_idx=3 → 100 samples ahead (~10 seconds)
        - horizon_idx=4 → 200 samples ahead (~20 seconds)
    
    Set to None to return all horizons (required for HMHP multi-horizon training).
    When None, the dataset returns a dict of labels: {horizon_value: label, ...}
    """
    
    feature_preset: Optional[str] = None
    """
    Named feature preset for feature selection (optional).
    
    If specified, only the selected features are passed to the model.
    Requires model.input_size to match the preset's feature count.
    
    Available presets:
    - "short_term_40": 40 features optimized for H10/H20 prediction
    - "full_116": All 116 features (standard + experimental)
    - "full" or "full_98": Standard 98 features
    - "lob_only": Raw LOB features (40)
    - "lob_derived": LOB + derived (48)
    - "signals_core": Core trading signals (8)
    
    Note: Either feature_preset OR feature_indices can be set, not both.
    """
    
    feature_indices: Optional[List[int]] = None
    """
    Custom feature indices for selection (optional).

    If specified, only these feature indices are passed to the model.
    Requires model.input_size to match len(feature_indices).

    Example: [84, 85, 86, 87, 88] to select only signal features.

    Note: At most ONE of {feature_set, feature_indices, feature_preset}
    may be set (mutual exclusion enforced in __post_init__ — Phase 4
    Batch 4c). ``feature_set_per_horizon`` was removed in 4c hardening
    and returns in 4d alongside HMHP ``feature_attention`` activation.
    """

    feature_set: Optional[str] = None
    """
    Named FeatureSet reference (Phase 4 Batch 4c, 2026-04-15).

    If specified, the trainer loads ``contracts/feature_sets/<name>.json``
    at dataloader construction time, verifies its content hash, and
    populates the internal ``_feature_indices_resolved`` cache with the
    resolved indices. ``feature_set`` REPLACES both ``feature_preset``
    and ``feature_indices`` for the same role — preset mapping is
    handled inside the FeatureSet registry (see ``contracts/feature_sets/SCHEMA.md``).

    Example: ``feature_set: momentum_hft_v1``.

    Mutual exclusion: at most one of {feature_set, feature_indices,
    feature_preset} may be set. To override a base that defines a
    different selection field, set the others to ``null`` in the child
    YAML.
    """

    # Phase 4 Batch 4c hardening (2026-04-15): the previously-reserved
    # ``feature_set_per_horizon: Optional[Dict[int, str]]`` field was removed
    # per adversarial design audit finding D2. It will return in Phase 4
    # Batch 4d alongside the HMHP ``feature_attention`` activation in
    # ``lob-models/.../hmhp.py``. Shipping a reserved field that only
    # raises ``NotImplementedError`` on direct use added documentation,
    # test, and mutual-exclusion surface area for zero current value;
    # 4d will add it back as a real feature, not a placeholder.

    feature_sets_dir: Optional[str] = None
    """
    Explicit override for the FeatureSet registry directory (Phase 4 Batch 4c).

    When ``None`` (default), the trainer auto-detects the registry by
    walking up from ``data_dir`` looking for
    ``contracts/pipeline_contract.toml``. Set this to a path string for:

    - Test isolation (point tests at a temp registry).
    - Multi-registry workflows (team-shared registry vs. local experiments).
    - Running the trainer from a CWD outside the monorepo.

    Resolved to an absolute path by
    ``lobtrainer.data.feature_set_resolver.find_feature_sets_dir``
    fallback when unset; used directly when set.
    """

    sources: Optional[List[SourceConfig]] = None
    """Multi-source configuration (T12). When provided, data is loaded from
    multiple sources and fused at load time. Exactly one source must have
    role='primary'. The existing data_dir field is used as the single source
    when sources is None (backward compat).

    Example YAML:
        data:
          sources:
            - name: mbo
              data_dir: "../data/exports/e5_timebased_60s"
              role: primary
            - name: basic
              data_dir: "../data/exports/basic_nvda_60s"
              role: auxiliary
    """

    # -- Private runtime caches (NOT serialized, NOT user-facing) ------------
    # Phase 4 Batch 4c: populated by the FeatureSet resolver at dataloader
    # construction time. Consumers read these INSTEAD of resolving from
    # `feature_set` every call. Non-init + non-repr so `to_yaml()` /
    # dataclasses.asdict() do not leak them into the on-disk YAML (the
    # source of truth stays `feature_set: <name>`, preserving round-trip
    # fidelity — see Phase 4 R3 "runtime cache, not YAML mutation").

    _feature_indices_resolved: Optional[List[int]] = field(
        default=None, init=False, repr=False, compare=False
    )
    """Resolved `feature_set` → list[int] cache. Populated by the trainer's
    feature_set_resolver at dataloader construction. Do NOT read directly
    from user code — use the public field(s) that drive the semantics."""

    _feature_set_ref_resolved: Optional[Tuple[str, str]] = field(
        default=None, init=False, repr=False, compare=False
    )
    """Resolved (name, content_hash) pair for the active FeatureSet.
    Propagated into signal_metadata.json and ExperimentRecord by
    downstream stages so backtest artifacts self-identify the selection
    provenance. None when no feature_set was used."""

    def __post_init__(self) -> None:
        # Validate feature_count is reasonable
        # Supported configurations:
        # - 40: Raw LOB only (official TLOB repo, DeepLOB benchmark)
        # - 48: LOB + derived (FI-2010 style)
        # - 76: LOB + derived + MBO
        # - 98: Full feature set (extended)
        # - 116: Full + experimental features (no MLOFI)
        # - 128: Full + all experimental including MLOFI (Kolm 2023)
        VALID_FEATURE_COUNTS = {40, 48, 76, 98, 116, 128}
        
        if self.feature_count < 1:
            raise ValueError(
                f"feature_count must be >= 1, got {self.feature_count}"
            )
        if self.feature_count > 200:
            raise ValueError(
                f"feature_count must be <= 200, got {self.feature_count}"
            )
        if self.feature_count not in VALID_FEATURE_COUNTS:
            import warnings
            warnings.warn(
                f"feature_count={self.feature_count} is not a standard configuration. "
                f"Expected one of: {sorted(VALID_FEATURE_COUNTS)}. "
                "Proceeding anyway, but ensure your export matches this count."
            )
        
        # horizon_idx validation: None is valid for multi-horizon mode
        if self.horizon_idx is not None and self.horizon_idx < 0:
            raise ValueError(
                f"horizon_idx must be >= 0 or None, got {self.horizon_idx}"
            )
        
        # Feature selection: mutual exclusion across the three user-facing
        # fields (Phase 4 Batch 4c, 2026-04-15). At most ONE of
        # {feature_set, feature_indices, feature_preset} may be set.
        # Raises immediately if the user (or a _base: merge) produces
        # more than one — explicit is better than a silent priority-based
        # precedence.
        _selection_fields = [
            ("feature_set", self.feature_set),
            ("feature_indices", self.feature_indices),
            ("feature_preset", self.feature_preset),
        ]
        _active_selection = [n for n, v in _selection_fields if v is not None]
        if len(_active_selection) > 1:
            raise ValueError(
                "At most one feature-selection field may be set on DataConfig, "
                f"got {len(_active_selection)}: {_active_selection}. "
                "To override a base that sets a different selection field, "
                "explicitly set the others to `null` in the child YAML "
                "(e.g., `feature_preset: null` alongside `feature_set: my_v1`)."
            )

        # Validate `feature_set` shape (Phase 4 Batch 4c). The resolver
        # (feature_set_resolver.resolve_feature_set) does the full
        # integrity + contract-compat check at dataloader construction
        # time; here we just catch obviously-wrong values at parse time.
        if self.feature_set is not None:
            if not isinstance(self.feature_set, str) or not self.feature_set.strip():
                raise ValueError(
                    f"feature_set must be a non-empty string, got "
                    f"{self.feature_set!r}"
                )
            if "/" in self.feature_set or "\\" in self.feature_set:
                raise ValueError(
                    f"feature_set must not contain path separators "
                    f"(reserved for directory-traversal safety in the "
                    f"registry resolver). Got: {self.feature_set!r}"
                )

        # Validate preset exists (import here to avoid circular imports)
        # AND emit DeprecationWarning — feature_preset is scheduled for
        # removal (Phase 4 4-month 3-step deprecation; see
        # FEATURE_PRESET_DEPRECATION_SCHEDULE in feature_presets.py for
        # the authoritative timeline).
        if self.feature_preset is not None:
            from lobtrainer.constants import FEATURE_PRESETS
            from lobtrainer.constants.feature_presets import (
                FEATURE_PRESET_DEPRECATION_SCHEDULE,
            )
            preset_lower = self.feature_preset.lower()
            if preset_lower not in FEATURE_PRESETS:
                available = sorted(FEATURE_PRESETS.keys())
                raise ValueError(
                    f"Unknown feature_preset: '{self.feature_preset}'. "
                    f"Available presets: {available}"
                )
            import warnings
            _sched = FEATURE_PRESET_DEPRECATION_SCHEDULE
            warnings.warn(
                f"DataConfig.feature_preset='{self.feature_preset}' is "
                f"DEPRECATED since {_sched['announced']} (Phase 4 Batch 4c). "
                f"Migrate to a FeatureSet registry entry:\n"
                f"  1. Run `hft-ops evaluate --config <evaluator.yaml> "
                f"--criteria <criteria.yaml> --save-feature-set "
                f"<name>_v1 --applies-to-assets NVDA --applies-to-horizons <h>`.\n"
                f"  2. Update this config: `data.feature_set: <name>_v1` "
                f"(and remove `feature_preset`).\n"
                f"See `contracts/feature_sets/SCHEMA.md` for details. This "
                f"warning escalates to PendingDeprecationWarning on "
                f"{_sched['escalate_to_pending']} and becomes an ImportError "
                f"on {_sched['hard_error_date']}.",
                DeprecationWarning,
                stacklevel=3,
            )

        # Validate feature_indices if provided
        if self.feature_indices is not None:
            if len(self.feature_indices) == 0:
                raise ValueError("feature_indices cannot be empty")
            if len(self.feature_indices) != len(set(self.feature_indices)):
                raise ValueError("feature_indices contains duplicate values")
            if any(idx < 0 for idx in self.feature_indices):
                raise ValueError("feature_indices contains negative values")
            # Note: Upper bound checked at runtime when we know source_feature_count

        # T12: validate sources configuration (if provided)
        if self.sources is not None:
            primaries = [s for s in self.sources if s.role == "primary"]
            if len(primaries) != 1:
                raise ValueError(
                    f"Exactly one source must have role='primary', "
                    f"got {len(primaries)}: {[s.name for s in primaries]}"
                )
            names = [s.name for s in self.sources]
            if len(names) != len(set(names)):
                raise ValueError(
                    f"Duplicate source names in data.sources: {names}"
                )

        # T9: derive labels from legacy fields if user did not provide new config.
        # DataConfig.__post_init__ runs before ExperimentConfig.__post_init__,
        # guaranteeing self.labels is non-None when deprecation warnings fire.
        if self.labels is None:
            if self.labeling_strategy == LabelingStrategy.REGRESSION:
                derived_task = "regression"
            else:
                derived_task = "classification"
            self.labels = LabelsConfig(
                source="auto",
                task=derived_task,
                primary_horizon_idx=self.horizon_idx,
            )


# =============================================================================
# Model Configuration
# =============================================================================


class DeepLOBMode(str, Enum):
    """DeepLOB operational mode."""
    
    BENCHMARK = "benchmark"
    """Original paper architecture: 40 LOB features only."""
    
    EXTENDED = "extended"
    """Extended architecture: All 98 features."""


@dataclass
class ModelConfig:
    """
    Configuration for model architecture.

    New format (Phase 3): 7 named fields + opaque params dict.
    Old format (pre-Phase 3): 37 flat fields with model-prefixed names.

    The __post_init__ auto-migrates old flat format to new format, so existing
    YAML configs and test code continue to work without modification.

    New YAML format::

        model:
          name: tlob
          input_size: 98
          num_classes: 3
          params:
            hidden_dim: 64
            num_layers: 4
            dropout: 0.2

    Old YAML format (auto-migrated)::

        model:
          model_type: tlob
          input_size: 98
          num_classes: 3
          tlob_hidden_dim: 64
          tlob_num_layers: 4
          dropout: 0.2
    """

    # =========================================================================
    # Core fields (accessed by trainer/strategies outside create_model)
    # =========================================================================

    model_type: ModelType = ModelType.LSTM
    """Model architecture type. Use 'name' for new configs; model_type is
    preserved for backward compatibility with existing YAML files."""

    input_size: int = 98
    """Input feature dimension. MUST match data.feature_count."""

    num_classes: int = 3
    """Number of output classes."""

    params: dict = field(default_factory=dict)
    """Architecture-specific parameters passed through to the model's config class.
    Keys map directly to the model's config dataclass fields (e.g., TLOBConfig).
    Injected automatically at creation time: num_features, num_classes, sequence_length, task_type."""

    # HMHP-specific (needed by strategies)
    hmhp_horizons: Optional[List[int]] = None
    """Prediction horizons for HMHP models. None for non-HMHP."""

    hmhp_use_regression: bool = False
    """Whether HMHP classification uses dual regression heads."""

    # DeepLOB-specific (needed by data pipeline)
    deeplob_mode: str = "benchmark"
    """DeepLOB mode: 'benchmark' (40 LOB features) or 'extended' (all features)."""

    # =========================================================================
    # Legacy fields (auto-migrated to params in __post_init__)
    # These exist ONLY for backward compatibility with existing YAML configs.
    # New configs should use the 'params' dict directly.
    # =========================================================================

    # Shared legacy fields
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    task_type: str = "classification"
    regression_loss_type: str = "huber"
    regression_loss_delta: float = 10.0

    # LSTM/GRU legacy
    lstm_bidirectional: bool = False
    lstm_attention: bool = False

    # Transformer legacy
    transformer_num_heads: int = 4
    transformer_dim_feedforward: int = 256

    # DeepLOB legacy
    deeplob_conv_filters: int = 32
    deeplob_inception_filters: int = 64
    deeplob_lstm_hidden: int = 64
    deeplob_num_levels: int = 10

    # TLOB legacy
    tlob_hidden_dim: int = 64
    tlob_num_layers: int = 4
    tlob_num_heads: int = 1
    tlob_mlp_expansion: float = 4.0
    tlob_use_sinusoidal_pe: bool = True
    tlob_use_bin: bool = True
    tlob_dataset_type: str = "nvda"
    tlob_use_cvml: bool = False
    tlob_cvml_out_channels: int = 0
    gmadl_a: float = 10.0
    gmadl_b: float = 1.5

    # Logistic legacy
    logistic_pooling: str = "last"
    logistic_feature_indices: Optional[List[int]] = None

    # HMHP legacy (architecture params — migrate to params dict)
    hmhp_cascade_mode: str = "full"
    hmhp_state_fusion: str = "gate"
    hmhp_cascade_connections: Optional[List[Tuple[int, int]]] = None
    hmhp_encoder_type: str = "tlob"
    hmhp_encoder_hidden_dim: int = 64
    hmhp_num_encoder_layers: int = 2
    hmhp_decoder_hidden_dim: int = 32
    hmhp_state_dim: int = 32
    hmhp_use_confirmation: bool = True
    hmhp_regression_loss_type: str = "huber"
    hmhp_optimal_features_by_horizon: Optional[dict] = None
    hmhp_loss_weights: Optional[dict] = None

    def __post_init__(self) -> None:
        # Set default horizons
        if self.hmhp_horizons is None:
            self.hmhp_horizons = [10, 20, 50, 100, 200]

        # Validate core fields that stay on ModelConfig
        # T13: allow input_size=0 as sentinel for auto-derivation
        # (resolved in ExperimentConfig.__post_init__)
        if self.input_size < 0:
            raise ValueError(
                f"input_size must be >= 0 (0=auto-derive), "
                f"got {self.input_size}"
            )
        if self.num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {self.num_classes}")
        if self.dropout < 0 or self.dropout > 1:
            raise ValueError(f"dropout must be in [0, 1], got {self.dropout}")

        # Auto-migrate legacy flat fields into params dict if params is empty.
        # This ensures old YAML configs (with flat tlob_hidden_dim etc.) work
        # seamlessly — the legacy fields populate params for create_model().
        if not self.params:
            self.params = self._build_params_from_legacy()

    @property
    def name(self) -> str:
        """Registry key for this model (derived from model_type)."""
        _MODEL_TYPE_TO_NAME = {
            "lstm": "lstm",
            "gru": "gru",
            "logistic": "logistic_lob",
            "deeplob": "deeplob",
            "tlob": "tlob",
            "hmhp": "hmhp",
            "hmhp_regression": "hmhp_regressor",
            "mlplob": "mlplob",
            "temporal_ridge": "temporal_ridge",
            "temporal_gradboost": "temporal_gradboost",
        }
        mt = self.model_type.value if isinstance(self.model_type, Enum) else str(self.model_type)
        return _MODEL_TYPE_TO_NAME.get(mt, mt)

    def _build_params_from_legacy(self) -> dict:
        """Extract architecture params from legacy flat fields.

        Maps model_type-prefixed fields to the params dict that will be
        passed to the model's config class via ModelRegistry.
        """
        mt = self.model_type.value if isinstance(self.model_type, Enum) else str(self.model_type)

        if mt in ("lstm", "gru"):
            return {
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "num_classes": self.num_classes,
                "dropout": self.dropout,
                "bidirectional": self.lstm_bidirectional,
                "attention": self.lstm_attention if mt == "lstm" else False,
            }

        elif mt == "logistic":
            return {
                "num_features": self.input_size,
                "num_classes": self.num_classes,
                "pooling": self.logistic_pooling,
                "dropout": self.dropout,
                "feature_indices": self.logistic_feature_indices,
            }

        elif mt == "deeplob":
            return {
                "mode": self.deeplob_mode if isinstance(self.deeplob_mode, str) else self.deeplob_mode.value,
                "feature_layout": "grouped",
                "num_levels": self.deeplob_num_levels,
                "num_classes": self.num_classes,
                "conv_filters": self.deeplob_conv_filters,
                "inception_filters": self.deeplob_inception_filters,
                "lstm_hidden": self.deeplob_lstm_hidden,
                "lstm_layers": 1,
                "dropout": self.dropout,
                "task_type": self.task_type,
                "regression_loss_type": self.regression_loss_type,
                "regression_loss_delta": self.regression_loss_delta,
            }

        elif mt == "tlob":
            return {
                "num_features": self.input_size,
                "num_classes": self.num_classes,
                "hidden_dim": self.tlob_hidden_dim,
                "num_layers": self.tlob_num_layers,
                "num_heads": self.tlob_num_heads,
                "mlp_expansion": self.tlob_mlp_expansion,
                "use_sinusoidal_pe": self.tlob_use_sinusoidal_pe,
                "use_bin": self.tlob_use_bin,
                "dropout": self.dropout,
                "dataset_type": self.tlob_dataset_type,
                "task_type": self.task_type,
                "regression_loss_type": self.regression_loss_type,
                "regression_loss_delta": self.regression_loss_delta,
                "use_cvml": self.tlob_use_cvml,
                "cvml_out_channels": self.tlob_cvml_out_channels,
                "gmadl_a": self.gmadl_a,
                "gmadl_b": self.gmadl_b,
            }

        elif mt in ("hmhp", "hmhp_regression"):
            # Note: num_classes is passed to create_hmhp() as explicit kwarg
            # but create_hmhp_regressor() hardcodes it. We include it here
            # and let _create_hmhp_model handle the per-factory logic.
            p = {
                "num_features": self.input_size,
                "horizons": self.hmhp_horizons,
                "encoder_type": self.hmhp_encoder_type,
                "hidden_dim": self.hmhp_encoder_hidden_dim,
                "num_encoder_layers": self.hmhp_num_encoder_layers,
                "decoder_hidden_dim": self.hmhp_decoder_hidden_dim,
                "state_dim": self.hmhp_state_dim,
                "state_fusion": self.hmhp_state_fusion,
                "cascade_mode": self.hmhp_cascade_mode,
                "dropout": self.dropout,
            }
            if mt == "hmhp":
                p["cascade_connections"] = self.hmhp_cascade_connections
                p["optimal_features_by_horizon"] = self.hmhp_optimal_features_by_horizon
                p["use_confirmation"] = self.hmhp_use_confirmation
                p["use_regression"] = self.hmhp_use_regression
                if self.hmhp_loss_weights is not None:
                    p["loss_weights"] = self.hmhp_loss_weights
            if mt == "hmhp_regression":
                # create_hmhp_regressor() kwarg is 'loss_type', not 'default_loss_type'
                p["loss_type"] = self.hmhp_regression_loss_type
            return p

        else:
            # Unknown model type — return shared fields as params
            return {
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "num_classes": self.num_classes,
                "dropout": self.dropout,
            }


# =============================================================================
# Training Configuration
# =============================================================================


@dataclass
class TrainConfig:
    """
    Configuration for training hyperparameters.
    """
    
    batch_size: int = 64
    """Batch size for training."""
    
    learning_rate: float = 1e-4
    """Initial learning rate."""
    
    weight_decay: float = 1e-5
    """L2 regularization weight."""
    
    epochs: int = 100
    """Maximum number of training epochs."""
    
    early_stopping_patience: int = 10
    """Stop training if validation loss doesn't improve for this many epochs."""
    
    gradient_clip_norm: Optional[float] = 1.0
    """Gradient clipping norm. Set to None to disable."""
    
    optimizer: str = "adamw"
    """Optimizer: 'adamw', 'adam', or 'sgd'."""

    scheduler: str = "cosine"
    """Learning rate scheduler: 'cosine', 'step', 'plateau', or 'none'."""
    
    scheduler_step_size: int = 30
    """Step size for StepLR scheduler."""
    
    scheduler_gamma: float = 0.1
    """Multiplicative factor for StepLR scheduler."""
    
    num_workers: int = 4
    """Number of data loader workers."""
    
    pin_memory: bool = True
    """Pin memory for faster GPU transfer."""
    
    seed: int = 42
    """Random seed for reproducibility."""
    
    mixed_precision: bool = False
    """Use automatic mixed precision (AMP) for faster training."""
    
    use_class_weights: bool = True
    """
    Apply inverse-frequency class weighting to CrossEntropyLoss.
    
    Useful for imbalanced datasets where some classes are much more frequent.
    Weights are computed as: weight[c] = total / (n_classes × count[c])
    
    Set to False if using a balanced dataset (e.g., quantile-based thresholds).
    """
    
    # =========================================================================
    # Task and Loss Configuration (Stage 1/2 Training)
    # =========================================================================
    
    task_type: TaskType = TaskType.MULTICLASS
    """
    Classification task type:
    - multiclass: Standard 3-class (Down/Stable/Up)
    - binary_signal: Binary signal detection (Signal vs NoSignal)
    
    Stage 1 training uses binary_signal to detect trading opportunities.
    Stage 2 uses multiclass (or binary Up/Down) to predict direction.
    """
    
    loss_type: LossType = LossType.WEIGHTED_CE
    """
    Loss function type:
    - cross_entropy: Standard unweighted CE
    - weighted_ce: CE with inverse-frequency class weights
    - focal: Focal loss for handling class imbalance
    
    For imbalanced data (like 71% Stable in nvda_bigmove), use focal or weighted_ce.
    """
    
    focal_gamma: float = 2.0
    """
    Focal loss gamma parameter. Higher = more focus on hard examples.
    
    Common values:
    - gamma=0: Equivalent to cross-entropy
    - gamma=2: Default, strong focus on hard examples (recommended)
    - gamma=5: Very aggressive focusing
    
    Only used when loss_type='focal'.
    """
    
    focal_alpha: Optional[float] = None
    """
    Focal loss alpha parameter for class balancing.
    
    For binary: alpha = weight for positive class (Signal).
                alpha > 0.5 gives more importance to Signal class.
    For multi-class: Should be a list, but we use inverse-frequency weights instead.
    
    If None, uses inverse-frequency weighting computed from training data.
    Only used when loss_type='focal'.
    """
    
    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.focal_gamma < 0:
            raise ValueError(f"focal_gamma must be >= 0, got {self.focal_gamma}")
        if self.focal_alpha is not None and (self.focal_alpha < 0 or self.focal_alpha > 1):
            raise ValueError(f"focal_alpha must be in [0, 1], got {self.focal_alpha}")

        # Cross-validate loss_type against task_type to prevent silent misconfiguration.
        # Classification strategies only support CE/Focal/WeightedCE.
        # Regression loss is controlled by model.regression_loss_type, not train.loss_type.
        _CLASSIFICATION_LOSSES = {LossType.CROSS_ENTROPY, LossType.WEIGHTED_CE, LossType.FOCAL}
        _REGRESSION_LOSSES = {LossType.MSE, LossType.HUBER, LossType.HETEROSCEDASTIC, LossType.GMADL}

        if self.task_type == TaskType.REGRESSION and self.loss_type in _CLASSIFICATION_LOSSES:
            raise ValueError(
                f"loss_type='{self.loss_type.value}' is a classification loss but "
                f"task_type='{self.task_type.value}'. For regression, use one of: "
                f"{sorted(lt.value for lt in _REGRESSION_LOSSES)}."
            )
        if self.task_type != TaskType.REGRESSION and self.loss_type in _REGRESSION_LOSSES:
            raise ValueError(
                f"loss_type='{self.loss_type.value}' is a regression loss but "
                f"task_type='{self.task_type.value}'. For classification, use one of: "
                f"{sorted(lt.value for lt in _CLASSIFICATION_LOSSES)}. "
                f"For regression loss control, set model.regression_loss_type instead."
            )


# =============================================================================
# Experiment Configuration (Top-Level)
# =============================================================================


@dataclass
class CVConfig:
    """Cross-validation configuration (T11).

    Optional — only used when running purged K-fold CV via CVTrainer.
    When cv is None on ExperimentConfig, no CV is performed.

    Reference: de Prado (2018) AFML Chapter 7.
    """

    n_splits: int = 5
    """Number of temporal folds (K). Must be >= 2."""

    embargo_days: int = 1
    """Days after each val block excluded from training. Prevents feature
    autocorrelation leakage across temporal boundaries."""

    def __post_init__(self) -> None:
        if self.n_splits < 2:
            raise ValueError(
                f"CVConfig.n_splits must be >= 2, got {self.n_splits}"
            )
        if self.embargo_days < 0:
            raise ValueError(
                f"CVConfig.embargo_days must be >= 0, got {self.embargo_days}"
            )


@dataclass
class ExperimentConfig:
    """
    Top-level configuration combining all sub-configs.

    This is the main configuration object used throughout the codebase.

    Usage:
        >>> config = ExperimentConfig.from_yaml("configs/baseline.yaml")
        >>> trainer = Trainer(config)
        >>> trainer.train()
    """

    name: str = "default"
    """Experiment name for tracking."""

    description: str = ""
    """Experiment description."""

    data: DataConfig = field(default_factory=DataConfig)
    """Data loading configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    """Model architecture configuration."""

    train: TrainConfig = field(default_factory=TrainConfig)
    """Training hyperparameters."""

    cv: Optional[CVConfig] = None
    """Cross-validation configuration (T11). None = no CV."""

    # Logical type: Optional["ImportanceConfig"]. Kept as Optional[Any]
    # at runtime to avoid the schema.py ↔ training.importance.config
    # circular import chain (see TYPE_CHECKING guard at module top).
    # Actual type coercion (dict → ImportanceConfig) happens in
    # ``__post_init__`` below via ``_coerce_importance``.
    importance: Optional[Any] = None
    """Phase 8C-α Stage C.1 trainer wire-in (2026-04-20): post-training
    feature-permutation-importance configuration. When set AND
    ``importance.enabled=True``, the trainer auto-registers
    ``PermutationImportanceCallback`` which runs on_train_end to compute
    per-feature importance + write ``outputs/<name>/feature_importance_v1.json``
    for hft-ops Stage C.3 ledger routing. None = disabled (default,
    zero compute overhead). OBSERVATION — NOT a treatment — so
    fingerprint-excluded via `hft_ops.ledger.dedup._extract_fingerprint_fields`
    (enabling importance does NOT change what gets trained; only adds
    post-hoc analysis). Locked by
    `test_dedup.py::test_importance_field_excluded_from_fingerprint`.
    See `lobtrainer.training.importance.config.ImportanceConfig` for
    field-level documentation + `lobtrainer.training.importance.callback`
    for the invocation mechanism."""

    output_dir: str = "outputs"
    """Directory for checkpoints and logs."""

    log_level: str = "INFO"
    """Logging level: DEBUG, INFO, WARNING, ERROR."""

    tags: List[str] = field(default_factory=list)
    """Tags for experiment tracking."""
    
    def __post_init__(self) -> None:
        """Validate cross-config consistency."""
        # T13: Resolve expected feature count and auto-derive model.input_size
        import logging as _log
        _t13_logger = _log.getLogger(__name__)

        # Phase 8C-α Stage C.1 trainer wire-in (post-audit round-2):
        # coerce importance from dict → ImportanceConfig. Deferred import
        # breaks the schema.py ↔ training.importance.config circular
        # dependency. See `_coerce_importance` below for the conversion
        # + validation delegation to ImportanceConfig.__post_init__.
        _coerce_importance(self)

        # Phase 6 6A.1 (2026-04-17, revised after validation audit):
        # `data.feature_set` is resolved AT TRAINER RUNTIME (inside
        # `_create_dataloaders` via the FeatureSet registry resolver) because
        # __post_init__ lacks the filesystem / registry-path context the
        # resolver needs. Consequence: T13 auto-derivation (`input_size=0`
        # → fill from resolved feature count) CANNOT work for `feature_set`
        # in this code generation — the model is constructed at `setup()`
        # L717 BEFORE the resolver runs at L725.
        #
        # Fix: require the user to set `model.input_size` EXPLICITLY when
        # using `data.feature_set`. Fails fast + clearly at config-load.
        # A proper architectural fix (reorder `setup()` to run resolver
        # before model construction, or make model creation lazy) is
        # Phase 7 scope (lobtrainer-core split — architect Concern #2).
        if self.data.feature_set is not None:
            if self.model.input_size == 0:
                raise ValueError(
                    "`data.feature_set` requires `model.input_size` to be set "
                    "explicitly (non-zero). Auto-derivation is unavailable "
                    "because FeatureSet resolution needs registry-path context "
                    "unavailable at config-load time. Set `model.input_size` "
                    "to the expected feature count of the resolved FeatureSet "
                    "(inspect contracts/feature_sets/<name>.json → "
                    "len(feature_indices)), or use `data.feature_preset` / "
                    "`data.feature_indices` which support auto-derivation. "
                    "(Proper resolver-time auto-derivation is deferred to "
                    "Phase 7 lobtrainer-core split.)"
                )
            # input_size is set explicitly — skip T13. trainer.py::_create_dataloaders
            # L413-418 verifies it matches the resolved feature count at runtime.
        else:
            if self.data.sources is not None:
                # Multi-source: sum per-source feature_counts (if all > 0)
                source_counts = [
                    s.feature_count for s in self.data.sources
                    if s.feature_count > 0
                ]
                if len(source_counts) == len(self.data.sources):
                    resolved_input_size = sum(source_counts)
                else:
                    resolved_input_size = self.data.feature_count
            elif self.data.feature_preset is not None:
                from lobtrainer.constants import get_feature_preset
                preset_indices = get_feature_preset(self.data.feature_preset)
                resolved_input_size = len(preset_indices)
            elif self.data.feature_indices is not None:
                resolved_input_size = len(self.data.feature_indices)
            else:
                resolved_input_size = self.data.feature_count

            if self.model.input_size == 0:
                # Auto-derive (T13)
                self.model.input_size = resolved_input_size
                # Update params dict (built during ModelConfig.__post_init__)
                for _key in ("num_features", "input_size"):
                    if _key in self.model.params:
                        self.model.params[_key] = resolved_input_size
                _t13_logger.info(
                    "Auto-derived model.input_size = %d", resolved_input_size
                )
            elif self.model.input_size != resolved_input_size:
                raise ValueError(
                    f"model.input_size ({self.model.input_size}) != resolved "
                    f"feature count ({resolved_input_size}). "
                    f"Set model.input_size: 0 to auto-derive."
                    )
        
        # Validate exclude_features are within bounds of SOURCE data
        for idx in self.data.normalization.exclude_features:
            if idx >= self.data.feature_count:
                raise ValueError(
                    f"normalization.exclude_features contains index {idx}, "
                    f"but feature_count is only {self.data.feature_count}. "
                    f"Remove or adjust exclude_features for your dataset."
                )

        # T9: deprecation warnings for legacy label fields.
        # Only fire when legacy fields have non-default values.
        import warnings as _warnings
        if self.data.labeling_strategy != LabelingStrategy.TLOB:
            _warnings.warn(
                f"DataConfig.labeling_strategy is deprecated. "
                f"Use data.labels.task instead. "
                f"Current effective value: "
                f"data.labels.task={self.data.labels.task!r}.",
                DeprecationWarning,
                stacklevel=2,
            )
        if self.data.horizon_idx != 0:
            _warnings.warn(
                f"DataConfig.horizon_idx is deprecated. "
                f"Use data.labels.primary_horizon_idx instead. "
                f"Current effective value: "
                f"data.labels.primary_horizon_idx="
                f"{self.data.labels.primary_horizon_idx!r}.",
                DeprecationWarning,
                stacklevel=2,
            )

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization.

        Phase 4 Batch 4c (2026-04-15): keys whose name begins with an
        underscore are filtered OUT of every dataclass and every nested
        dict in the output. This keeps resolver-populated runtime caches
        (``DataConfig._feature_indices_resolved``, ``_feature_set_ref_resolved``)
        out of serialized YAML/JSON so the on-disk config round-trip is
        preserved: user writes ``feature_set: momentum_v1``, reads back
        the same. Without this filter, ``asdict()`` would leak the cache
        into ``to_yaml()`` output and corrupt the source-of-truth
        (Phase 4 R3 invariant).

        Why both the dataclass and dict branches filter: ``asdict(obj)``
        recursively converts nested dataclasses to plain dicts in one
        pass, so by the time ``_convert`` recurses into a nested value,
        it looks like a plain dict (not a dataclass). Filtering at the
        dataclass boundary alone would not reach fields on nested
        dataclasses; filtering at the dict branch catches them at any
        depth. No currently-known config uses ``_``-prefixed dict keys
        legitimately, so this convention is safe.
        """
        def _convert(obj: Any) -> Any:
            if isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, BaseModel):
                # Phase A.5.3a (2026-04-24): Pydantic v2 BaseModel branch
                # dispatched BEFORE ``__dataclass_fields__``. Critical during
                # the A.5.3a-f staged migration when a BaseModel child (e.g.
                # LabelsConfig) is held inside a still-dataclass parent (e.g.
                # DataConfig); ``dataclasses.asdict`` does NOT recurse into
                # BaseModel fields (it returns the BaseModel instance as-is),
                # which would silently leak the instance into to_yaml() /
                # to_json() output. ``model_dump(exclude_none=False)``
                # recursively serializes nested BaseModels + returns a plain
                # dict with canonical field names, matching the output shape
                # of asdict() on the equivalent dataclass.
                return {
                    k: _convert(v)
                    for k, v in obj.model_dump(exclude_none=False).items()
                    if not (isinstance(k, str) and k.startswith("_"))
                }
            elif hasattr(obj, "__dataclass_fields__"):
                return {
                    k: _convert(v)
                    for k, v in asdict(obj).items()
                    if not k.startswith("_")
                }
            elif isinstance(obj, (list, tuple)):
                # Phase A.5.3a.1 (2026-04-24): tuple handling matches
                # ``sanitize_for_hash`` canonical-form convention (tuples
                # serialize as JSON arrays / YAML sequences — no
                # ``!!python/tuple`` tag leaks into on-disk configs).
                # Post-A.5.3a LabelsConfig.horizons is ``Tuple[int, ...]``.
                return [_convert(item) for item in obj]
            elif isinstance(obj, dict):
                return {
                    k: _convert(v)
                    for k, v in obj.items()
                    if not (isinstance(k, str) and k.startswith("_"))
                }
            else:
                return obj
        return _convert(self)
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    def to_json(self, path: str) -> None:
        """Save configuration to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentConfig":
        """
        Create config from dictionary.

        Uses dacite with type casting for robust parsing of YAML/JSON input.
        Handles common parsing issues like scientific notation strings.

        Note: _base inheritance is resolved in from_yaml(), not here.
        If a _base key leaks into a from_dict() call, it means inheritance
        was not resolved — raise an error directing the caller to from_yaml().
        """
        from dacite import from_dict, Config as DaciteConfig

        if "_base" in data:
            raise ValueError(
                "_base key found in config dict. Config inheritance is only "
                "supported when loading from YAML files via from_yaml(). "
                "Remove _base from the dict or use from_yaml() instead."
            )

        # Preprocess data to handle YAML parsing quirks
        data = _normalize_config_types(data)

        # Phase A.5.3a (2026-04-24): Pydantic-migrated classes are NOT
        # dataclasses, so dacite's dataclass-centric constructor path won't
        # recognize them. Register ``type_hooks`` that route the raw dict
        # through ``BaseModel.model_validate`` — which runs Pydantic's
        # validators + enforces ``extra="forbid"`` + returns a BaseModel
        # instance that dacite can place into the parent dataclass field
        # slot. At A.5.3a, only ``LabelsConfig`` is migrated; subsequent
        # commits A.5.3b-i will extend this table. A.5.3i retires dacite
        # entirely and uses ``ExperimentConfig.model_validate(data)``
        # directly.
        #
        # Hook is defensive: runs ONLY when the value is a dict (not an
        # already-constructed BaseModel). This supports callers that
        # construct nested Pydantic models programmatically before passing
        # to ExperimentConfig(...) — a pattern in test code.
        _type_hooks: Dict[type, Any] = {
            LabelsConfig: lambda d: (
                LabelsConfig.model_validate(d) if isinstance(d, dict) else d
            ),
        }

        return from_dict(
            data_class=cls,
            data=data,
            config=DaciteConfig(
                cast=[Enum, Path, float, int, bool],
                type_hooks=_type_hooks,
            ),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """Load configuration from YAML file.

        Supports config inheritance via the ``_base`` key, in two forms:

            _base: "../bases/regression.yaml"          # v1 single-base

            _base:                                       # v2 multi-base
              - "../bases/models/tlob_compact.yaml"
              - "../bases/datasets/nvda_e5_60s.yaml"
              - "../bases/labels/regression_huber.yaml"
              - "../bases/train/regression_default.yaml"

        Each base is loaded first (recursively resolving its own ``_base``),
        then merged left-to-right (each successive base overrides the previous),
        then this config's values are deep-merged on top of the accumulator.
        Chained inheritance is supported with cycle detection and a depth
        cap of 10. Paths are resolved relative to the directory of the file
        containing the ``_base`` key.

        Semantics:
            - Dicts: recursively merged
            - Lists: replaced entirely (not appended)
            - Scalars / None: override value wins
            - Multi-base order: later entries in the list override earlier;
              this config's own keys override everything

        Raises:
            ValueError: If the target YAML is a partial base (declares
                ``_partial: true`` at the top level). Partial bases are
                standalone-invalid and must be composed with peer bases via
                multi-base ``_base: [...]``. Catches the common mistake of
                accidentally loading ``bases/models/tlob_compact.yaml``
                directly instead of going through an experiment config.
        """
        from pathlib import Path as _Path
        from lobtrainer.config.merge import is_partial_base, resolve_inheritance

        config_path = _Path(path).resolve()

        # Fail-fast on partial bases so the error points at the root cause,
        # not at a dacite missing-required-field error deep in the validator.
        if is_partial_base(config_path):
            raise ValueError(
                f"Partial base config cannot be loaded standalone: {config_path}\n"
                f"This file declares `_partial: true` — it only becomes valid "
                f"when composed with peer bases via multi-base inheritance.\n"
                f"See configs/bases/README.md for the composition rule."
            )

        with open(config_path) as f:
            # Phase 6 6A.5: empty YAML file → yaml.safe_load returns None;
            # resolve_inheritance(None, ...) would crash. Defensive fallback
            # mirrors the pattern at merge.py:169.
            # Phase 6 post-validation hardening (2026-04-18): also reject
            # valid-but-non-dict YAML payloads (top-level list / string /
            # int / bool). The prior `or {}` silently passed those through,
            # producing an unhelpful deep-merge crash. Now raise ValueError
            # with a precise message pointing at the root cause.
            raw = yaml.safe_load(f)
        if raw is None:
            data = {}
        elif isinstance(raw, dict):
            data = raw
        else:
            raise ValueError(
                f"{config_path}: top-level YAML payload must be a mapping "
                f"(key/value dict) or empty, got {type(raw).__name__}. "
                f"Example valid shapes: `{{}}`, `name: foo` + nested keys."
            )
        data = resolve_inheritance(data, config_path)
        return cls.from_dict(data)
    
    @classmethod
    def from_json(cls, path: str) -> "ExperimentConfig":
        """Load configuration from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)


# =============================================================================
# Config Normalization Utilities
# =============================================================================


def _normalize_config_types(data: dict) -> dict:
    """
    Recursively normalize config data to handle YAML parsing quirks.
    
    YAML parsers can sometimes parse scientific notation (1e-8) as strings
    instead of floats. This function detects and converts such values.
    
    This is a long-term fix that handles edge cases in YAML parsing across
    different Python versions and YAML library versions.
    
    Args:
        data: Dictionary from YAML/JSON parsing
        
    Returns:
        Normalized dictionary with proper types
    """
    import re
    
    # Pattern to match scientific notation strings like "1e-8", "1E-8", "1.5e-10"
    SCIENTIFIC_NOTATION_PATTERN = re.compile(r'^-?[0-9]+\.?[0-9]*[eE][+-]?[0-9]+$')
    
    def _normalize_value(value):
        """Normalize a single value."""
        if isinstance(value, dict):
            return {k: _normalize_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [_normalize_value(v) for v in value]
        elif isinstance(value, str):
            # Check if string looks like scientific notation
            if SCIENTIFIC_NOTATION_PATTERN.match(value):
                try:
                    return float(value)
                except ValueError:
                    return value
            # Check if string is "true"/"false" (YAML boolean edge cases)
            if value.lower() == 'true':
                return True
            if value.lower() == 'false':
                return False
            # Check if string is a plain integer
            if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                try:
                    return int(value)
                except ValueError:
                    return value
            return value
        else:
            return value
    
    return {k: _normalize_value(v) for k, v in data.items()}


# =============================================================================
# Convenience Functions
# =============================================================================


def load_config(path: str) -> ExperimentConfig:
    """
    Load configuration from file (auto-detect format).
    
    Args:
        path: Path to configuration file (.yaml, .yml, or .json)
    
    Returns:
        ExperimentConfig instance
    
    Raises:
        ValueError: If file format is not supported
    """
    path_lower = path.lower()
    if path_lower.endswith((".yaml", ".yml")):
        return ExperimentConfig.from_yaml(path)
    elif path_lower.endswith(".json"):
        return ExperimentConfig.from_json(path)
    else:
        raise ValueError(f"Unsupported config format: {path}. Use .yaml, .yml, or .json")


def save_config(config: ExperimentConfig, path: str) -> None:
    """
    Save configuration to file (auto-detect format).
    
    Args:
        config: ExperimentConfig instance
        path: Output path (.yaml, .yml, or .json)
    
    Raises:
        ValueError: If file format is not supported
    """
    path_lower = path.lower()
    if path_lower.endswith((".yaml", ".yml")):
        config.to_yaml(path)
    elif path_lower.endswith(".json"):
        config.to_json(path)
    else:
        raise ValueError(f"Unsupported config format: {path}. Use .yaml, .yml, or .json")

