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
# Phase A.5.3b (2026-04-24): + SafeBaseModel shared base — packages the 4
# hardening patterns (frozen + extra_forbid + strict + model_copy override)
# so each subsequent migration inherits "fail-fast by default" rather than
# re-deriving. See lobtrainer.config.base module docstring for the full
# rationale + 4-bug empirical trace.
# Phase A.5.3g (2026-04-24): + PrivateAttr for DataConfig's resolver caches
# (_feature_indices_resolved + _feature_set_ref_resolved). PrivateAttr is
# explicitly designed to be mutable even under frozen=True — correct escape
# hatch for post-construction state written by an external resolver.
from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator

from lobtrainer.config.base import SafeBaseModel

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


# Phase A.5.3i (2026-04-24 KEYSTONE): the ``_coerce_importance(config)``
# helper that previously mutated ``config.importance`` in ExperimentConfig's
# ``__post_init__`` has been REPLACED by a ``@field_validator("importance",
# mode="before")`` on ExperimentConfig itself (see the class definition
# below). The validator path avoids the post-construction mutation entirely
# (incompatible with ``frozen=True``) and fires at field-coercion time,
# BEFORE the strict type check. Functionality preserved; see
# ``ExperimentConfig._coerce_importance_field`` for the new home.


class SourceConfig(SafeBaseModel):
    """Single data source specification (T12).

    Used in DataConfig.sources for multi-source fusion.

    **Phase A.5.3d (2026-04-24)**: migrated to SafeBaseModel. No Enum fields,
    no mutable containers — simplest migration in the cycle after SequenceConfig.
    ``role`` uses the LabelsConfig-style ``ClassVar[frozenset[str]]`` pattern
    for discoverable allowed values + v3-A ClassVar-leak regression coverage.

    Args:
        name: Unique identifier (e.g., "mbo", "basic").
        data_dir: Path to export directory containing train/val/test/.
        role: "primary" (labels + features) or "auxiliary" (features only).
            Exactly one source must be primary.
        feature_count: Feature count for this source. 0 = auto-detect at
            load time. When all sources specify feature_count > 0,
            model.input_size can be auto-derived as the sum (T13).
    """

    name: str = "mbo"
    data_dir: str = ""
    role: str = "primary"
    feature_count: int = 0

    # Phase A.5.3d (2026-04-24): ClassVar[frozenset[str]] annotation is
    # LOAD-BEARING under Pydantic v2 strict=True. Without it, Pydantic would
    # treat the constant as a model FIELD and leak it into model_dump(),
    # polluting YAML round-trips. See v3-A plan amendment + LabelsConfig
    # precedent (_VALID_SOURCES etc.).
    _VALID_ROLES: ClassVar[frozenset[str]] = frozenset({"primary", "auxiliary"})

    @model_validator(mode="after")
    def _validate_all(self) -> "SourceConfig":
        if self.role not in self._VALID_ROLES:
            raise ValueError(
                f"SourceConfig.role must be one of "
                f"{sorted(self._VALID_ROLES)}, got {self.role!r}"
            )
        if self.feature_count < 0:
            raise ValueError(
                f"SourceConfig.feature_count must be >= 0, "
                f"got {self.feature_count}"
            )
        return self


class LabelsConfig(SafeBaseModel):
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

    # Phase A.5.3b (2026-04-24): ``model_config`` + ``model_copy`` override
    # now inherited from ``SafeBaseModel``. DO NOT re-declare either here —
    # Pydantic v2 REPLACES (not merges) parent config when subclass declares
    # its own, which would silently strip hardening. See
    # ``lobtrainer.config.base.SafeBaseModel`` for the full 4-pattern rationale
    # + the A.5.3a.1 empirical bug-class trace.

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

    # Phase A.5.3b (2026-04-24): inline ``model_copy`` override moved to
    # SafeBaseModel. LabelsConfig inherits the re-validation semantics
    # automatically. Regression tests (TestLabelsConfigPydanticHardening
    # at tests/test_config.py) continue to exercise identical behavior via
    # the inherited method — tests are NOT rewritten.

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

    # -----------------------------------------------------------------
    # Phase A.5.4 (2026-04-24) — horizon-slicing SSoT primitive.
    #
    # The method + its parametric helper close 3 bugs in the
    # exporter/callback slicing paths (plan v4 bug ledger #2 + #5):
    #
    # Bug #2 — No bounds check before ``preds[:, primary_idx]`` slicing.
    #   Python's negative-indexing silently picks last-N (instead of
    #   raising). A config with ``primary_horizon_idx=-1`` would slice
    #   the LAST horizon without any diagnostic, producing silently-wrong
    #   calibration/metric values.
    #
    # Bug #5 — Silent fallback ``primary_horizon = None`` when the idx
    #   is >= len(horizons). No diagnostic fired; the metadata's
    #   ``primary_horizon`` field was just null, making post-hoc
    #   investigation opaque.
    #
    # The method returns a VALIDATED integer in [0, n_horizons);
    # raises ``ValueError`` with an actionable diagnostic on any
    # negative OR out-of-bounds index. Caller gets ONE canonical
    # validate-slice-report surface instead of 4+ open-coded sites.
    # -----------------------------------------------------------------

    def validate_primary_horizon_idx_for(self, n_horizons: int) -> int:
        """Validate + return the canonical primary_horizon_idx for slicing.

        Applies Phase A.5 bounds-check discipline: negative indices are
        rejected (Python negative-indexing semantics would silently
        select last-N column — silent-wrong-result hazard per
        hft-rules §8). Out-of-bounds indices raise with a diagnostic
        citing ``n_horizons`` and the offending value.

        Args:
            n_horizons: Number of horizons available to slice (typically
                ``preds.shape[-1]`` or ``len(labels_cfg.horizons)``).

        Returns:
            Non-negative integer in ``[0, n_horizons)``. Falls back to
            ``0`` when ``self.primary_horizon_idx is None`` (the "first
            horizon is primary" convention — pre-Phase-A.5 behavior).

        Raises:
            ValueError: if ``n_horizons < 1`` (can't slice zero horizons).
            ValueError: if ``primary_horizon_idx`` is negative.
            ValueError: if ``primary_horizon_idx >= n_horizons`` (out
                of bounds).

        Usage::

            labels_cfg = resolve_labels_config(config)
            primary_idx = labels_cfg.validate_primary_horizon_idx_for(
                preds.shape[-1]
            )
            preds_1d = preds[:, primary_idx]  # safe, validated

        Phase B extensibility: future horizon fields (e.g.,
        ``secondary_horizon_idx``) get a one-line wrapper around
        ``_validate_horizon_idx_for`` — no N-method explosion.
        """
        return self._validate_horizon_idx_for(
            field_name="primary_horizon_idx",
            idx_value=self.primary_horizon_idx,
            n_horizons=n_horizons,
        )

    @staticmethod
    def _validate_horizon_idx_for(
        field_name: str,
        idx_value: Optional[int],
        n_horizons: int,
    ) -> int:
        """Parametric bounds-check helper for horizon-index fields.

        Phase B extension point: accepts ``field_name`` so future
        horizon fields (secondary / tertiary / cascade) can reuse the
        same validation via one-line wrappers:

            def validate_secondary_horizon_idx_for(self, n_horizons):
                return self._validate_horizon_idx_for(
                    "secondary_horizon_idx",
                    self.secondary_horizon_idx,
                    n_horizons,
                )

        Args:
            field_name: Name of the field being validated (embedded in
                error messages for operator diagnostics).
            idx_value: The integer index to validate (or None for
                auto-default-to-0).
            n_horizons: Array size (must be >= 1).

        Returns:
            Validated non-negative int in [0, n_horizons).

        Raises:
            ValueError: on negative idx / out-of-bounds idx / n_horizons < 1.
        """
        if n_horizons < 1:
            raise ValueError(
                f"LabelsConfig.{field_name} validation requires n_horizons >= 1, "
                f"got n_horizons={n_horizons!r}. Cannot slice zero horizons."
            )
        effective = idx_value if idx_value is not None else 0
        if effective < 0:
            raise ValueError(
                f"LabelsConfig.{field_name}={effective!r} is negative. "
                f"Python negative indexing would silently select column "
                f"{n_horizons + effective} (last-N from end) — silent-wrong-result "
                f"hazard per hft-rules §8. Use a non-negative index in "
                f"[0, {n_horizons})."
            )
        if effective >= n_horizons:
            raise ValueError(
                f"LabelsConfig.{field_name}={effective!r} >= "
                f"n_horizons={n_horizons!r}. Cannot slice — index out of bounds. "
                f"Available indices: [0, {n_horizons})."
            )
        return effective


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


class SequenceConfig(SafeBaseModel):
    """
    Configuration for sequence construction from flat features.

    Sequences are built by sliding a window over flat feature vectors.
    Labels are aligned to the END of each window.

    Example:
        window_size=100, stride=10:
        - Window 0: features[0:100], label for sample 99
        - Window 1: features[10:110], label for sample 109
        - ...

    **Phase A.5.3b (2026-04-24)**: migrated from ``@dataclass`` to Pydantic
    v2 ``BaseModel`` via shared ``SafeBaseModel`` base. Inherits
    ``frozen=True, extra="forbid", strict=True`` + ``model_copy`` re-validation
    from ``SafeBaseModel``. See ``lobtrainer.config.base`` module docstring
    for the full 4-bug-class retirement rationale (packaged at the base
    class so subclasses stay minimal).

    Bug classes retired at the TYPE layer via SafeBaseModel inheritance:

        1. Silent post-construction mutation (``cfg.window_size = 1000`` raises)
        2. Typo propagation (``SequenceConfig(stide=10)`` raises — note typo)
        3. String-to-int coercion (``SequenceConfig(window_size="100")`` raises)
        4. ``model_copy(update={...})`` validator bypass (inherited override
           re-validates on update)

    Cross-field invariant (``stride ≤ window_size``) validated via
    ``@model_validator(mode="after")`` since it needs access to both
    fields post-construction — ``@field_validator`` on a single field
    cannot reference siblings.
    """

    window_size: int = 100
    """Number of samples per sequence. Must match Rust export config."""

    stride: int = 10
    """Step size between consecutive sequences. Smaller = more overlap."""

    @model_validator(mode="after")
    def _validate_all(self) -> "SequenceConfig":
        """Pydantic equivalent of the legacy ``__post_init__``.

        Raises ``ValueError`` (automatically wrapped as
        ``pydantic.ValidationError`` by the framework) on any invariant
        violation. Returns ``self`` per Pydantic convention (allows
        chainable post-validation copies where needed).
        """
        if self.window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {self.window_size}")
        if self.stride < 1:
            raise ValueError(f"stride must be >= 1, got {self.stride}")
        if self.stride > self.window_size:
            raise ValueError(
                f"stride ({self.stride}) should not exceed "
                f"window_size ({self.window_size})"
            )
        return self


class NormalizationConfig(SafeBaseModel):
    """
    Configuration for feature normalization.

    Normalization is applied per-feature (column-wise) using statistics
    computed from training data only to avoid data leakage.

    **Phase A.5.3c (2026-04-24)**: migrated from ``@dataclass`` to Pydantic
    v2 ``BaseModel`` via shared ``SafeBaseModel`` base (inherits
    ``frozen=True, extra="forbid", strict=True`` + ``model_copy``
    re-validation).

    **First class in the Phase A.5 cycle with an Enum field** (``strategy:
    NormalizationStrategy``). Under ``strict=True``, Pydantic rejects
    string→Enum coercion (YAML loads ``strategy: zscore_per_day`` as a
    plain string). The ``@field_validator(mode="before")`` below fires
    BEFORE the strict type check and converts string → Enum, preserving
    YAML ergonomics. This pattern establishes the precedent for every
    subsequent migration with Enum fields (A.5.3e TrainConfig.task_type/
    loss_type, A.5.3g DataConfig.labeling_strategy, A.5.3h
    ModelConfig.model_type/deeplob_mode).

    **Container immutability**: ``exclude_features`` changed from ``List[int]``
    to ``Tuple[int, ...]`` for true immutability (closes the ``.append``
    bypass of ``frozen=True``). Consumer at ``ExperimentConfig.__post_init__``
    line 1631 iterates the field — tuple iteration works identically, no
    consumer changes needed.
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

    exclude_features: Tuple[int, ...] = Field(default_factory=tuple)
    """
    Feature indices to exclude from normalization (e.g., categorical features).
    Default: empty tuple (explicit opt-in per dataset; 40-feature LOB-only
    datasets have no categorical features; 98/148-feature datasets should
    set ``[93]`` for TIME_REGIME exclusion).

    Phase A.5.3c: Tuple type enforces immutability — ``.append`` bypasses
    frozen=True on List but tuple is truly immutable.
    """

    @field_validator("strategy", mode="before")
    @classmethod
    def _coerce_strategy_string(cls, v: Any) -> Any:
        """Accept YAML string input under strict=True.

        YAML parses ``strategy: zscore_per_day`` as a plain Python string.
        Strict Pydantic v2 rejects string→Enum coercion at the type check.
        This pre-validator (mode="before") fires BEFORE strict type-check
        and converts via ``NormalizationStrategy(v)`` — raises a clean
        ValueError on invalid value (e.g. ``'invalid_strat' is not a valid
        NormalizationStrategy``) which Pydantic wraps as ValidationError.

        If v is already a NormalizationStrategy instance (programmatic
        construction from test code), passthrough — strict check accepts.
        Other types (int, None, etc.) passthrough — strict check rejects
        with its standard error message.
        """
        if isinstance(v, str):
            return NormalizationStrategy(v)
        return v

    @field_validator("exclude_features", mode="before")
    @classmethod
    def _coerce_exclude_features_to_tuple(cls, v: Any) -> Any:
        """Accept YAML list input under strict ``Tuple[int, ...]``.

        Same list→tuple coercion pattern as LabelsConfig.horizons (A.5.3a.1).
        YAML emits ``exclude_features: [93]`` as Python list; strict mode
        rejects list→tuple coercion. Pre-validator converts list→tuple
        BEFORE type check; downstream strict Tuple[int, ...] enforces int
        items (rejects string-in-list silent coercion, the ship-blocker
        #2 from A.5.3a.1 post-audit).
        """
        if isinstance(v, list):
            return tuple(v)
        return v

    @model_validator(mode="after")
    def _validate_all(self) -> "NormalizationConfig":
        """Pydantic equivalent of the legacy ``__post_init__``.

        Phase A.5.3c adds ``math.isfinite`` checks on the two float fields
        (``eps``, ``clip_value``) per the A.5.3a.1 hardening pattern —
        strict mode does NOT reject NaN/Inf floats (IEEE 754 considers
        them valid), so explicit finite checks close the silent-NaN gap.
        A NaN eps would silently reproduce div-by-zero; a NaN clip_value
        would break clip arithmetic downstream.
        """
        if not math.isfinite(self.eps):
            raise ValueError(
                f"NormalizationConfig.eps must be finite (not NaN/Inf), "
                f"got {self.eps!r}"
            )
        if self.eps <= 0:
            raise ValueError(f"eps must be > 0, got {self.eps}")
        if self.clip_value is not None:
            if not math.isfinite(self.clip_value):
                raise ValueError(
                    f"NormalizationConfig.clip_value must be finite "
                    f"(not NaN/Inf), got {self.clip_value!r}"
                )
            if self.clip_value <= 0:
                raise ValueError(f"clip_value must be > 0, got {self.clip_value}")
        # Note: exclude_features should be explicitly set in the config
        # for datasets with >93 features (to exclude TIME_REGIME).
        # For 40-feature datasets (LOB only), no categorical features exist.
        # The default empty tuple is intentional to avoid out-of-bounds errors.
        return self


class DataConfig(SafeBaseModel):
    """
    Configuration for data loading and preprocessing.

    Data is loaded from NumPy arrays exported by the Rust pipeline.

    **Phase A.5.3g (2026-04-24)**: migrated from ``@dataclass`` to Pydantic
    v2 ``BaseModel`` via shared ``SafeBaseModel`` base — the most complex
    leaf migration in the Phase A.5 cycle. Composite class holding 4
    already-migrated SafeBaseModel children (LabelsConfig, SequenceConfig,
    NormalizationConfig, SourceConfig) plus 2 ``PrivateAttr`` resolver
    caches.

    Key design decisions (from pre-planning agents):

    1. ``_feature_indices_resolved`` + ``_feature_set_ref_resolved`` use
       Pydantic ``PrivateAttr()`` — explicitly designed to be mutable
       even under ``frozen=True``. The resolver at trainer.py:416-419
       continues to write via direct assignment; NO ``object.__setattr__``
       needed. Pydantic excludes PrivateAttr from ``model_dump()`` by
       default, preserving the Phase 4 R3 invariant (caches must NOT leak
       into YAML round-trip).

    2. ``feature_indices: List[int]`` → ``Tuple[int, ...]`` + pre-validator
       for list→tuple coercion (A.5.3a.1 pattern applied to the 3rd
       migrated container field).

    3. ``label_encoding: LabelEncoding`` + ``labeling_strategy: LabelingStrategy``
       get ``@field_validator(mode="before")`` for string→Enum coercion
       (NormalizationConfig.strategy pattern / A.5.3c precedent).

    4. T9 labels auto-derivation: when ``labels is None`` the validator
       constructs a LabelsConfig from legacy fields. Under frozen=True,
       public field assignment raises — uses ``object.__setattr__`` which
       is the Pydantic v2-sanctioned escape hatch for in-validator
       self-mutation (per the Pydantic docs + plan v4 lines 3725, 3740).
       NOT ``model_copy(update=...)`` which would recursively re-trigger
       the same validator.

    5. T12 sources validation: exactly-one-primary + unique-names check
       preserved from @dataclass __post_init__ body — operates on the
       already-migrated SourceConfig BaseModels.

    6. DeprecationWarning for legacy ``feature_preset``: fires inside
       ``@model_validator(mode="after")``. Under ``model_copy(update={"feature_preset": "x"})``
       via SafeBaseModel's override, the warning RE-FIRES — this is
       correct (user created a new config with deprecated field set).

    Retires same 4 bug classes as other SafeBaseModel subclasses. Adds
    the nested-BaseModel pattern that A.5.3h/i will reuse.
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
    
    sequence: SequenceConfig = Field(default_factory=SequenceConfig)
    """Sequence construction configuration."""

    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
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
    
    feature_indices: Optional[Tuple[int, ...]] = None
    """
    Custom feature indices for selection (optional).

    If specified, only these feature indices are passed to the model.
    Requires model.input_size to match len(feature_indices).

    Example: [84, 85, 86, 87, 88] to select only signal features (YAML
    list input coerced to tuple via @field_validator(mode="before")).

    Phase A.5.3g (2026-04-24): type changed from ``Optional[List[int]]``
    to ``Optional[Tuple[int, ...]]`` for true immutability (A.5.3a.1
    pattern — tuple has no .append, closes the frozen-bypass). YAML
    loaders emit list; the coercer below converts before strict type
    check fires. ``sanitize_for_hash`` canonicalizes tuples to lists
    before hashing so fingerprint byte-identity is preserved.

    Consumer tolerance (verified): trainer.py:416 already wraps in
    ``list(...)`` when reading; other reads iterate or len-check which
    works identically on tuple.

    Note: At most ONE of {feature_set, feature_indices, feature_preset}
    may be set (mutual exclusion enforced in _validate_all — Phase 4
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

    sources: Optional[List[SourceConfig]] = Field(default=None)
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

    # -- Private runtime caches (Phase A.5.3g: PrivateAttr pattern) ----------
    # Phase 4 Batch 4c: populated by the FeatureSet resolver at dataloader
    # construction time. Consumers read these INSTEAD of resolving from
    # `feature_set` every call.
    #
    # Phase A.5.3g (2026-04-24): migrated from dataclass
    # ``field(default=None, init=False, repr=False, compare=False)`` to
    # Pydantic ``PrivateAttr(default=None)``. PrivateAttr is EXPLICITLY
    # designed to be mutable even under ``frozen=True`` — Pydantic
    # guarantees it is NOT included in ``model_dump()`` (preserving the
    # Phase 4 R3 "cache MUST NOT leak into YAML round-trip" invariant),
    # NOT included in ``__eq__`` comparisons, NOT included in ``__init__``
    # signature (underscore-prefix naming convention), but IS mutable
    # via direct attribute assignment from external code.
    #
    # This is the correct architectural escape hatch for "runtime state
    # populated post-construction by an external resolver" — the pattern
    # was absent from the pre-Pydantic @dataclass version but enforced
    # indirectly via field metadata (init=False, repr=False, compare=False).

    _feature_indices_resolved: Optional[List[int]] = PrivateAttr(default=None)
    """Resolved ``feature_set`` → list[int] cache. Populated by the trainer's
    feature_set_resolver at dataloader construction. Do NOT read directly
    from user code — use the public field(s) that drive the semantics.

    Phase A.5.3g: list kept (NOT converted to tuple) because the resolver
    writes the result via ``cfg_data._feature_indices_resolved = list(...)``
    at trainer.py:416; consumer expects List[int]."""

    _feature_set_ref_resolved: Optional[Tuple[str, str]] = PrivateAttr(default=None)
    """Resolved (name, content_hash) pair for the active FeatureSet.
    Propagated into signal_metadata.json and ExperimentRecord by
    downstream stages so backtest artifacts self-identify the selection
    provenance. None when no feature_set was used."""

    @field_validator("label_encoding", mode="before")
    @classmethod
    def _coerce_label_encoding_string(cls, v: Any) -> Any:
        """Accept YAML string input under strict=True (A.5.3c pattern)."""
        if isinstance(v, str):
            return LabelEncoding(v)
        return v

    @field_validator("labeling_strategy", mode="before")
    @classmethod
    def _coerce_labeling_strategy_string(cls, v: Any) -> Any:
        """Accept YAML string input under strict=True (A.5.3c pattern)."""
        if isinstance(v, str):
            return LabelingStrategy(v)
        return v

    @field_validator("feature_indices", mode="before")
    @classmethod
    def _coerce_feature_indices_to_tuple(cls, v: Any) -> Any:
        """Accept YAML list input under strict Tuple[int, ...] (A.5.3a.1 pattern)."""
        if isinstance(v, list):
            return tuple(v)
        return v

    @model_validator(mode="after")
    def _validate_all(self) -> "DataConfig":
        """Pydantic equivalent of the legacy ``__post_init__``.

        Preserves every invariant from the dataclass version:
        - feature_count range + standard-value advisory
        - horizon_idx >= 0 or None
        - 3-field mutual exclusion on {feature_set, feature_indices, feature_preset}
        - feature_set shape (non-empty str, no path separators)
        - feature_preset existence check + DeprecationWarning emission
        - feature_indices: non-empty + unique + non-negative
        - T12 sources: exactly-one-primary + unique-names
        - T9 labels derivation from legacy fields (uses object.__setattr__
          because direct field assignment raises under frozen=True)
        """
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

        # Validate `feature_set` shape (Phase 4 Batch 4c).
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

        # Validate preset exists AND emit DeprecationWarning (Phase 4 Batch 4c).
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

        # Validate feature_indices if provided (now Tuple[int, ...] post-migration)
        if self.feature_indices is not None:
            if len(self.feature_indices) == 0:
                raise ValueError("feature_indices cannot be empty")
            if len(self.feature_indices) != len(set(self.feature_indices)):
                raise ValueError("feature_indices contains duplicate values")
            if any(idx < 0 for idx in self.feature_indices):
                raise ValueError("feature_indices contains negative values")

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
        # Phase A.5.3g (2026-04-24): under frozen=True, direct ``self.labels = ...``
        # assignment raises. ``object.__setattr__`` is the Pydantic v2-sanctioned
        # escape hatch for in-validator self-mutation (documented in Pydantic
        # docs + plan v4). DO NOT use ``self.model_copy(update={"labels": ...})``
        # here — that would recursively re-trigger _validate_all.
        if self.labels is None:
            if self.labeling_strategy == LabelingStrategy.REGRESSION:
                derived_task = "regression"
            else:
                derived_task = "classification"
            derived = LabelsConfig(
                source="auto",
                task=derived_task,
                primary_horizon_idx=self.horizon_idx,
            )
            object.__setattr__(self, "labels", derived)

        return self


# =============================================================================
# Model Configuration
# =============================================================================


class DeepLOBMode(str, Enum):
    """DeepLOB operational mode."""
    
    BENCHMARK = "benchmark"
    """Original paper architecture: 40 LOB features only."""
    
    EXTENDED = "extended"
    """Extended architecture: All 98 features."""


class ModelConfig(SafeBaseModel):
    """
    Configuration for model architecture.

    New format (Phase 3): 7 named fields + opaque params dict.
    Old format (pre-Phase 3): 37 flat fields with model-prefixed names.

    The post-init validator auto-migrates old flat format to new format, so
    existing YAML configs and test code continue to work without modification.

    **Phase A.5.3h (2026-04-24)**: migrated from ``@dataclass`` to Pydantic
    v2 ``BaseModel`` via shared ``SafeBaseModel`` base — final leaf before
    the ExperimentConfig root keystone (A.5.3i). The most field-heavy class
    in the cycle (~40 fields), with 4 distinct hardening pattern intersections:

    1. ``ModelType`` Enum string→instance coercer (A.5.3c pattern).
    2. ``hmhp_horizons: List[int]`` → ``Tuple[int, ...]`` with default
       ``(10, 20, 50, 100, 200)`` + ``None``-→-default + list-→-tuple
       coercer (A.5.3a.1 immutability pattern + auto-default for legacy
       None input).
    3. ``logistic_feature_indices: Optional[List[int]]`` → ``Optional[Tuple[int, ...]]``
       + list-→-tuple coercer (A.5.3a.1 pattern).
    4. ``hmhp_cascade_connections: Optional[List[Tuple[int, int]]]`` →
       ``Optional[Tuple[Tuple[int, int], ...]]`` + NESTED list-of-lists
       → tuple-of-tuples coercer (YAML yields lists, strict rejects).

    **CRITICAL self-mutation pattern**: ``_build_params_from_legacy()``
    populates ``self.params`` when the user supplied no explicit params
    dict. Under frozen=True, direct field assignment raises; uses
    ``object.__setattr__(self, "params", ...)`` — the Pydantic v2-sanctioned
    escape hatch for in-validator cross-field self-mutation (same pattern
    as DataConfig's T9 labels auto-derivation, A.5.3g).

    ``params: dict = Field(default_factory=dict)`` — the inner dict's
    CONTENTS remain mutable even under frozen=True because Python does
    not auto-freeze mutable-container CONTENTS (only the field-slot
    assignment is blocked). This matches the legacy @dataclass behavior
    and is load-bearing for T13 auto-derivation at
    ExperimentConfig._validate_all (:1887-1896) which updates
    ``params["num_features"]`` / ``params["input_size"]`` in-place.

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

    Retires same 4 bug classes as other SafeBaseModel subclasses. Sets
    up ExperimentConfig (A.5.3i) to be the final @dataclass → BaseModel
    cut in the cycle.
    """

    # =========================================================================
    # Core fields (accessed by trainer/strategies outside create_model)
    # =========================================================================

    model_type: ModelType = ModelType.LSTM
    """Model architecture type. Use 'name' for new configs; model_type is
    preserved for backward compatibility with existing YAML files.

    Phase A.5.3h: YAML operators pass strings (``model_type: "tlob"``);
    under strict=True, a @field_validator(mode="before") coerces string
    → ModelType instance before the strict type check fires.
    """

    input_size: int = 98
    """Input feature dimension. MUST match data.feature_count."""

    num_classes: int = 3
    """Number of output classes."""

    params: Dict[str, Any] = Field(default_factory=dict)
    """Architecture-specific parameters passed through to the model's config class.
    Keys map directly to the model's config dataclass fields (e.g., TLOBConfig).
    Injected automatically at creation time: num_features, num_classes, sequence_length, task_type.

    Phase A.5.3h: ``dict`` → ``Dict[str, Any]`` for explicit type annotation.
    The dict's contents remain mutable post-construction (load-bearing for
    T13 auto-derivation at ExperimentConfig._validate_all); only the field-
    slot ASSIGNMENT is blocked by frozen=True.
    """

    # HMHP-specific (needed by strategies)
    hmhp_horizons: Tuple[int, ...] = (10, 20, 50, 100, 200)
    """Prediction horizons for HMHP models. Default to the canonical
    5-horizon set; non-HMHP models still receive the default (legacy
    contract: post-construction hmhp_horizons is never None).

    Phase A.5.3h: ``Optional[List[int]] = None`` (auto-set in __post_init__)
    → ``Tuple[int, ...]`` with explicit default tuple. @field_validator(mode="before")
    converts None (explicit YAML null) → default tuple AND list → tuple.
    True immutability (A.5.3a.1 pattern) closes the container-mutation
    bypass.
    """

    hmhp_use_regression: bool = False
    """Whether HMHP classification uses dual regression heads."""

    # DeepLOB-specific (needed by data pipeline)
    deeplob_mode: str = "benchmark"
    """DeepLOB mode: 'benchmark' (40 LOB features) or 'extended' (all features)."""

    # =========================================================================
    # Legacy fields (auto-migrated to params in _validate_all)
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
    logistic_feature_indices: Optional[Tuple[int, ...]] = None
    """Optional subset of feature indices for LogisticLOB.

    Phase A.5.3h: ``Optional[List[int]]`` → ``Optional[Tuple[int, ...]]``
    for container immutability. @field_validator(mode="before") coerces
    list input → tuple (YAML-compat).
    """

    # HMHP legacy (architecture params — migrate to params dict)
    hmhp_cascade_mode: str = "full"
    hmhp_state_fusion: str = "gate"
    hmhp_cascade_connections: Optional[Tuple[Tuple[int, int], ...]] = None
    """Optional cascade-graph edges (from, to) for HMHP horizon dependencies.

    Phase A.5.3h: ``Optional[List[Tuple[int, int]]]`` → ``Optional[Tuple[Tuple[int, int], ...]]``
    with NESTED list-of-lists → tuple-of-tuples coercer (YAML yields
    lists at BOTH levels: ``[[0, 1], [1, 2]]``).
    """
    hmhp_encoder_type: str = "tlob"
    hmhp_encoder_hidden_dim: int = 64
    hmhp_num_encoder_layers: int = 2
    hmhp_decoder_hidden_dim: int = 32
    hmhp_state_dim: int = 32
    hmhp_use_confirmation: bool = True
    hmhp_regression_loss_type: str = "huber"
    hmhp_optimal_features_by_horizon: Optional[Dict[Any, Any]] = None
    """Optional per-horizon feature subsets. YAML input is Dict[int, List[int]]
    or Dict[int, List[str]]; typed as Dict[Any, Any] to avoid strict-mode
    issues on heterogeneous value types."""
    hmhp_loss_weights: Optional[Dict[str, float]] = None
    """Optional per-loss-component weights (e.g., ``{"ce": 1.0, "consistency": 0.1}``)."""

    # --- Enum + container coercers (mode="before" bridges under strict=True) ---

    @field_validator("model_type", mode="before")
    @classmethod
    def _coerce_model_type_string(cls, v: Any) -> Any:
        """Accept YAML string input (e.g. ``model_type: "tlob"``) under strict=True."""
        if isinstance(v, str):
            return ModelType(v)
        return v

    @field_validator("hmhp_horizons", mode="before")
    @classmethod
    def _coerce_hmhp_horizons(cls, v: Any) -> Any:
        """Legacy contract: None → default tuple; list → tuple (A.5.3a.1)."""
        if v is None:
            return (10, 20, 50, 100, 200)
        if isinstance(v, list):
            return tuple(v)
        return v

    @field_validator("logistic_feature_indices", mode="before")
    @classmethod
    def _coerce_logistic_feature_indices(cls, v: Any) -> Any:
        """list → tuple coercer for YAML-compat (A.5.3a.1)."""
        if isinstance(v, list):
            return tuple(v)
        return v

    @field_validator("hmhp_cascade_connections", mode="before")
    @classmethod
    def _coerce_hmhp_cascade_connections(cls, v: Any) -> Any:
        """NESTED list-of-lists → tuple-of-tuples. YAML yields
        ``[[0, 1], [1, 2]]`` (outer list, inner lists); strict mode
        rejects both without explicit coercion."""
        if isinstance(v, list):
            return tuple(tuple(pair) if isinstance(pair, list) else pair for pair in v)
        return v

    @model_validator(mode="after")
    def _validate_all(self) -> "ModelConfig":
        """Pydantic equivalent of the legacy ``__post_init__``.

        Preserves every invariant from the dataclass version:
        - input_size >= 0 (T13 auto-derive sentinel)
        - num_classes >= 2
        - dropout in [0, 1]
        - auto-migration of legacy flat fields to ``params`` dict if empty

        Phase A.5.3h: ``self.params = ...`` assignment raises under
        frozen=True. Uses ``object.__setattr__(self, "params", ...)`` —
        same pattern as DataConfig's T9 labels auto-derivation.
        """
        # Validate core fields that stay on ModelConfig
        # T13: allow input_size=0 as sentinel for auto-derivation
        # (resolved in ExperimentConfig._validate_all)
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
        #
        # Phase A.5.3h: under frozen=True, ``self.params = ...`` raises.
        # ``object.__setattr__`` is the Pydantic v2-sanctioned escape hatch
        # for in-validator cross-field self-mutation (plan v4 line 3725).
        if not self.params:
            object.__setattr__(self, "params", self._build_params_from_legacy())

        return self

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
            # Phase A.5.3h (2026-04-24): logistic_feature_indices is now
            # Tuple[int, ...]; convert to list for lob-models LogisticLOBConfig
            # which typed-declares ``feature_indices: Optional[List[int]]``.
            return {
                "num_features": self.input_size,
                "num_classes": self.num_classes,
                "pooling": self.logistic_pooling,
                "dropout": self.dropout,
                "feature_indices": (
                    list(self.logistic_feature_indices)
                    if self.logistic_feature_indices is not None
                    else None
                ),
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
            #
            # Phase A.5.3h (2026-04-24): trainer's self.hmhp_horizons is now
            # Tuple[int, ...] + self.hmhp_cascade_connections is
            # Tuple[Tuple[int, int], ...]. But lob-models' HMHPConfig declares
            # ``horizons: List[int]`` + uses ``sorted(self.horizons)``-based
            # validation (sorted returns list — tuple vs list inequality is
            # ALWAYS True, fires a spurious "horizons must be in ascending
            # order" at dataclass __post_init__). Convert to list here to
            # honor the lob-models typed contract + preserve pre-migration
            # fingerprint byte-identity (compute_fingerprint canonicalizes
            # tuples to lists via sanitize_for_hash — equivalent hash).
            p = {
                "num_features": self.input_size,
                "horizons": list(self.hmhp_horizons),
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
                # cascade_connections: list-of-tuples in lob-models HMHPConfig
                # (not nested list) — honor that typed contract.
                p["cascade_connections"] = (
                    list(self.hmhp_cascade_connections)
                    if self.hmhp_cascade_connections is not None
                    else None
                )
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


class TrainConfig(SafeBaseModel):
    """
    Configuration for training hyperparameters.

    **Phase A.5.3e (2026-04-24)**: migrated to SafeBaseModel. Contains 2
    Enum fields (``task_type``, ``loss_type``) requiring string→Enum
    coercion under strict=True (same pattern as NormalizationConfig.strategy),
    plus 2 module-local frozenset constants (``_CLASSIFICATION_LOSSES``,
    ``_REGRESSION_LOSSES``) promoted to class-level ``ClassVar[frozenset]``
    for cross-field validation in ``_validate_all``. Cross-field invariant:
    classification task_type incompatible with regression loss_type (and
    vice versa) — enforced identically to legacy @dataclass behavior.

    Retires same 4 bug classes as other SafeBaseModel subclasses.
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
    
    # Phase A.5.3e (2026-04-24): module-local frozensets promoted to
    # class-level ClassVar[frozenset[LossType]] for discoverability +
    # cross-field validation access. Unannotated class-level frozensets
    # under Pydantic v2 would leak into model_dump() per v3-A discipline.
    _CLASSIFICATION_LOSSES: ClassVar[frozenset] = frozenset({
        LossType.CROSS_ENTROPY, LossType.WEIGHTED_CE, LossType.FOCAL,
    })
    _REGRESSION_LOSSES: ClassVar[frozenset] = frozenset({
        LossType.MSE, LossType.HUBER, LossType.HETEROSCEDASTIC, LossType.GMADL,
    })

    @field_validator("task_type", mode="before")
    @classmethod
    def _coerce_task_type_string(cls, v: Any) -> Any:
        """Accept YAML string input (e.g. 'regression') under strict=True.

        Same pattern as NormalizationConfig.strategy (A.5.3c). Strict mode
        rejects string→Enum coercion; pre-validator converts before type check.
        """
        if isinstance(v, str):
            return TaskType(v)
        return v

    @field_validator("loss_type", mode="before")
    @classmethod
    def _coerce_loss_type_string(cls, v: Any) -> Any:
        """Accept YAML string input (e.g. 'cross_entropy') under strict=True."""
        if isinstance(v, str):
            return LossType(v)
        return v

    @model_validator(mode="after")
    def _validate_all(self) -> "TrainConfig":
        """Pydantic equivalent of legacy ``__post_init__``.

        Added math.isfinite checks on float fields per A.5.3a.1 pattern
        (strict=True doesn't reject NaN — IEEE 754 valid — so explicit
        isfinite gates catch it). Cross-field task_type ↔ loss_type
        compatibility check uses the ClassVar frozensets.
        """
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if not math.isfinite(self.learning_rate):
            raise ValueError(
                f"learning_rate must be finite (not NaN/Inf), "
                f"got {self.learning_rate!r}"
            )
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if not math.isfinite(self.focal_gamma):
            raise ValueError(
                f"focal_gamma must be finite (not NaN/Inf), "
                f"got {self.focal_gamma!r}"
            )
        if self.focal_gamma < 0:
            raise ValueError(f"focal_gamma must be >= 0, got {self.focal_gamma}")
        if self.focal_alpha is not None:
            if not math.isfinite(self.focal_alpha):
                raise ValueError(
                    f"focal_alpha must be finite (not NaN/Inf), "
                    f"got {self.focal_alpha!r}"
                )
            if self.focal_alpha < 0 or self.focal_alpha > 1:
                raise ValueError(f"focal_alpha must be in [0, 1], got {self.focal_alpha}")

        # Cross-field task_type ↔ loss_type compatibility (uses ClassVar
        # frozensets promoted from module-local scope).
        if self.task_type == TaskType.REGRESSION and self.loss_type in self._CLASSIFICATION_LOSSES:
            raise ValueError(
                f"loss_type='{self.loss_type.value}' is a classification loss but "
                f"task_type='{self.task_type.value}'. For regression, use one of: "
                f"{sorted(lt.value for lt in self._REGRESSION_LOSSES)}."
            )
        if self.task_type != TaskType.REGRESSION and self.loss_type in self._REGRESSION_LOSSES:
            raise ValueError(
                f"loss_type='{self.loss_type.value}' is a regression loss but "
                f"task_type='{self.task_type.value}'. For classification, use one of: "
                f"{sorted(lt.value for lt in self._CLASSIFICATION_LOSSES)}. "
                f"For regression loss control, set model.regression_loss_type instead."
            )
        return self


# =============================================================================
# Experiment Configuration (Top-Level)
# =============================================================================


class CVConfig(SafeBaseModel):
    """Cross-validation configuration (T11).

    Optional — only used when running purged K-fold CV via CVTrainer.
    When cv is None on ExperimentConfig, no CV is performed.

    Reference: de Prado (2018) AFML Chapter 7.

    **Phase A.5.3f (2026-04-24)**: migrated to SafeBaseModel. Simplest
    migration alongside SequenceConfig — 2 scalar int fields, no Enum,
    no mutable containers, no class-level constants. Validators port
    1:1 to ``@model_validator(mode="after")``.
    """

    n_splits: int = 5
    """Number of temporal folds (K). Must be >= 2."""

    embargo_days: int = 1
    """Days after each val block excluded from training. Prevents feature
    autocorrelation leakage across temporal boundaries."""

    @model_validator(mode="after")
    def _validate_all(self) -> "CVConfig":
        if self.n_splits < 2:
            raise ValueError(
                f"CVConfig.n_splits must be >= 2, got {self.n_splits}"
            )
        if self.embargo_days < 0:
            raise ValueError(
                f"CVConfig.embargo_days must be >= 0, got {self.embargo_days}"
            )
        return self


# =============================================================================
# Phase A.5.3b (2026-04-24): placeholder for the module-level registry of
# Pydantic-migrated config classes. Originally built the
# ``_PYDANTIC_TYPE_HOOKS`` dacite-bridge table so dacite could route dict
# → BaseModel construction through ``model_validate``.
#
# **Phase A.5.3i (2026-04-24 KEYSTONE)**: dacite retired + registry
# RELOCATED to AFTER ``ExperimentConfig`` definition so ``ExperimentConfig``
# can be included in the list (forward-reference impossible at this point
# in module load order). See registry definition later in this file.
#
# ``_PYDANTIC_TYPE_HOOKS`` retired (no consumer after from_dict rewrite).
# =============================================================================


class ExperimentConfig(SafeBaseModel):
    """
    Top-level configuration combining all sub-configs.

    This is the main configuration object used throughout the codebase.

    **Phase A.5.3i (2026-04-24)**: migrated from ``@dataclass`` to Pydantic
    v2 ``BaseModel`` via shared ``SafeBaseModel`` base — the KEYSTONE
    commit closing the entire Phase A.5 Scope D cycle. Every config class
    in the hierarchy is now a frozen Pydantic model; dacite retired from
    the dependency tree.

    Retires 4 bug classes empirically proven at LabelsConfig (A.5.3a):

    1. **Canonical-path-drift** — ``config.labels`` vs ``config.data.labels``
       type-rejected at ``model_validate`` time (no such attribute on
       ExperimentConfig; Pydantic ``extra='forbid'`` enforces).
    2. **Silent mutation** — ``config.output_dir = X`` raises
       ValidationError under ``frozen=True``; callers use
       ``config.model_copy(update={"output_dir": X})`` which re-fires
       validators.
    3. **Extra-field acceptance** — typo ``names: "foo"`` (for ``name``)
       rejected at ``model_validate`` time by ``extra='forbid'``.
    4. **Silent-None field access** — typed fields discoverable via
       ``.model_fields``; validators catch missing-required + type
       mismatch at construction.

    Subsystem coordination:

    - ``dacite`` dependency retired (previously bridged dict→dataclass
      for BaseModel children via ``_PYDANTIC_TYPE_HOOKS`` table). Pydantic
      v2 handles nested BaseModel construction natively via
      ``cls.model_validate(data)``.
    - ``_coerce_importance`` helper (Phase 8C-α) inlined as a
      ``@field_validator(mode="before")`` on the ``importance`` field —
      no post-construction mutation needed.
    - T13 auto-derivation of ``model.input_size`` uses
      ``object.__setattr__(self, "model", ...)`` inside
      ``@model_validator(mode="after")`` — the Pydantic v2-sanctioned
      escape hatch for in-validator cross-field self-mutation under
      ``frozen=True``.
    - ``from_dict`` body rewritten to ``cls.model_validate(data)``; public
      ``from_yaml``/``to_yaml``/``to_dict``/``to_json`` API preserved
      (backward compatibility — zero call-site breakage across 37+ known
      consumer sites).

    CLI override pattern (all 3 production sites):

        # Before A.5.3i (mutation):
        config.output_dir = args.output_dir

        # After A.5.3i (model_copy re-validates):
        config = config.model_copy(update={"output_dir": args.output_dir})

    Usage:
        >>> config = ExperimentConfig.from_yaml("configs/baseline.yaml")
        >>> trainer = Trainer(config)
        >>> trainer.train()
    """

    name: str = "default"
    """Experiment name for tracking."""

    description: str = ""
    """Experiment description."""

    data: DataConfig = Field(default_factory=DataConfig)
    """Data loading configuration."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    """Model architecture configuration."""

    train: TrainConfig = Field(default_factory=TrainConfig)
    """Training hyperparameters."""

    cv: Optional[CVConfig] = None
    """Cross-validation configuration (T11). None = no CV."""

    # Logical type: Optional["ImportanceConfig"]. Kept as Optional[Any]
    # at runtime to avoid the schema.py ↔ training.importance.config
    # circular import chain (see TYPE_CHECKING guard at module top).
    # Phase A.5.3i (2026-04-24): coercion moved from __post_init__
    # helper (`_coerce_importance`) into ``@field_validator(mode="before")``
    # below — fires before strict type check, no post-construction
    # mutation needed.
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

    tags: List[str] = Field(default_factory=list)
    """Tags for experiment tracking.

    Phase A.5.3i: kept as ``List[str]`` (not Tuple) — tags are user
    observational metadata, mutated throughout experiment lifecycle.
    Under frozen=True the LIST REFERENCE cannot be reassigned
    (``config.tags = [...]`` raises), but list-content mutation
    (``config.tags.append("x")``) remains permitted (Python does not
    freeze mutable-container contents). Matches legacy semantics.
    """

    @field_validator("importance", mode="before")
    @classmethod
    def _coerce_importance_field(cls, v: Any) -> Any:
        """Coerce YAML dict → ImportanceConfig at field-validation time.

        Phase A.5.3i (2026-04-24): replaces the module-level
        ``_coerce_importance(self)`` helper that previously mutated
        ``config.importance`` in ``__post_init__``. Under frozen=True,
        post-construction field assignment raises; this validator runs
        BEFORE the field type check (Optional[Any] accepts anything,
        but we want typed semantics).

        Accepts:
          - ``None`` (explicit disabled — default) → None
          - ``dict`` (YAML-load default when ``importance:`` is specified)
            → ImportanceConfig(**dict) (ImportanceConfig __post_init__
            validates ranges + raises on bad values)
          - ``ImportanceConfig`` instance (test code / programmatic
            construction) → passed through

        Deferred import breaks the schema.py ↔ training.importance.config
        circular dependency — safe because this validator fires at
        field-coercion time (after training package has finished loading
        for any caller that constructed ExperimentConfig).

        Round-3 post-audit Agent-4 H2 fix preserved: real ``isinstance``
        check (not duck-typing) — strict check prevents silent
        garbage-in from partially-corrupt deserializations.
        """
        if v is None:
            return None
        from lobtrainer.training.importance.config import ImportanceConfig
        if isinstance(v, dict):
            return ImportanceConfig(**v)
        if isinstance(v, ImportanceConfig):
            return v
        # Phase A.5.3i: ``raise ValueError`` (not TypeError) so Pydantic v2
        # wraps as ``ValidationError`` — consistent with every other
        # @field_validator in this module (unified failure mode at all
        # config boundaries; callers catch ValidationError uniformly).
        raise ValueError(
            f"ExperimentConfig.importance must be None, a dict, or an "
            f"ImportanceConfig instance; got {type(v).__name__}."
        )

    @model_validator(mode="after")
    def _validate_all(self) -> "ExperimentConfig":
        """Pydantic equivalent of the legacy ``__post_init__``.

        Preserves every invariant from the dataclass version:

        - Phase 6 6A.1 `data.feature_set` + `model.input_size=0` fail-fast
          (resolver-time auto-derivation deferred to Phase 7 lobtrainer-core
          split)
        - T13 auto-derive `model.input_size=0` → resolved_input_size (from
          sources / feature_preset / feature_indices / feature_count, in
          priority order). Under frozen=True, uses
          ``object.__setattr__(self, "model", new_model)`` — the Pydantic
          v2-sanctioned escape hatch for in-validator self-mutation.
        - normalization.exclude_features bounds check
        - T9 deprecation warnings for legacy labeling_strategy / horizon_idx

        Returns self (Pydantic convention).
        """
        # T13: Resolve expected feature count and auto-derive model.input_size
        import logging as _log
        _t13_logger = _log.getLogger(__name__)

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
                # Auto-derive (T13).
                #
                # Phase A.5.3i (2026-04-24 KEYSTONE): BOTH ExperimentConfig
                # AND ModelConfig are now frozen Pydantic BaseModels.
                # Two-layer mutation pattern:
                #
                #   (a) INNER: ModelConfig.model_copy(update=...) re-fires
                #       the ModelConfig validator on the coherent updated
                #       state (input_size + params["num_features"]).
                #
                #   (b) OUTER: ``object.__setattr__(self, "model", ...)``
                #       is the Pydantic v2-sanctioned escape hatch for
                #       in-validator cross-field self-mutation under
                #       frozen=True. Same pattern as DataConfig's T9
                #       labels auto-derivation (A.5.3g) + ModelConfig's
                #       params self-mutation (A.5.3h). Direct
                #       ``self.model = ...`` would raise ValidationError.
                #       ``self.model_copy(update={"model": ...})`` would
                #       recursively re-trigger this validator → infinite
                #       loop. object.__setattr__ bypasses both paths.
                _new_params = dict(self.model.params)
                for _key in ("num_features", "input_size"):
                    if _key in _new_params:
                        _new_params[_key] = resolved_input_size
                _new_model = self.model.model_copy(update={
                    "input_size": resolved_input_size,
                    "params": _new_params,
                })
                object.__setattr__(self, "model", _new_model)
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

        # Phase A.5.3i (2026-04-24): Pydantic @model_validator(mode="after")
        # requires explicit return of self (or a new instance). Legacy
        # @dataclass __post_init__ returned None implicitly; Pydantic is
        # stricter to support "validator-returns-new-instance" use cases.
        return self

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

        Phase A.5.3i (2026-04-24 KEYSTONE): ``dacite`` retired. Pydantic
        v2 ``model_validate`` handles nested BaseModel construction
        natively — recursively routes each nested field dict through the
        corresponding BaseModel's construction pipeline (including every
        ``@field_validator`` / ``@model_validator`` for Enum coercion,
        list→tuple conversion, cross-field invariants, etc.).

        Preserves all pre-A.5.3i behavior:

        - ``_base`` inheritance rejection (inheritance belongs in
          ``from_yaml``, not ``from_dict`` — resolved BEFORE the dict
          reaches here).
        - ``_normalize_config_types`` YAML-quirk preprocessing (scientific
          notation strings → float; "true"/"false" strings → bool;
          integer strings → int). Applied BEFORE ``model_validate`` so
          strict mode sees pre-normalized values.

        Gains from Pydantic-native construction:

        - Every sub-config's validators fire (dacite's ``type_hooks`` table
          only routed construction; it did not re-run validators on
          already-constructed BaseModels).
        - Unknown-field rejection via ``extra='forbid'`` — typo
          ``feature_cont: 98`` (for ``feature_count``) raises
          ValidationError with the exact field path (previously silently
          dropped by dacite).
        - Type rejection under strict mode — string-to-int, bool-to-int,
          list-to-tuple all rejected without explicit coercers (subclasses
          that want YAML-compat provide ``@field_validator(mode="before")``
          bridges).
        """
        if "_base" in data:
            raise ValueError(
                "_base key found in config dict. Config inheritance is only "
                "supported when loading from YAML files via from_yaml(). "
                "Remove _base from the dict or use from_yaml() instead."
            )

        # Preprocess data to handle YAML parsing quirks
        data = _normalize_config_types(data)

        return cls.model_validate(data)

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


# =============================================================================
# Migrated-class registry (Phase A.5.3i relocation)
# =============================================================================
#
# Placed AFTER ``ExperimentConfig`` (forward-reference impossible earlier
# in module load order — the registry would reference an undefined name).
# The registry is retained post-dacite-retirement as:
#
#   - **Parametrized regression-test target** — ``TestPydanticHardeningCoverageGaps``
#     in test_config.py iterates this list for pickle/deepcopy round-trip
#     sweeps. Every migrated class gets uniform cross-class coverage.
#   - **Audit / discovery aid** — ``grep _PYDANTIC_CONFIG_CLASSES`` is a
#     single-query "which classes use SafeBaseModel?" — future contributors
#     are reminded via this grep that the pattern is convention.
#
# ``_PYDANTIC_TYPE_HOOKS`` (ex-dacite bridge table) retired in A.5.3i —
# ``ExperimentConfig.from_dict`` now delegates to ``cls.model_validate(data)``
# which handles nested BaseModel construction natively.
# =============================================================================

_PYDANTIC_CONFIG_CLASSES: List[type] = [
    LabelsConfig,         # A.5.3a (commit 1507b87)
    SequenceConfig,       # A.5.3b (commit f32288f)
    NormalizationConfig,  # A.5.3c (commit 52516e5 — first Enum-field class)
    SourceConfig,         # A.5.3d (commit f54a838)
    TrainConfig,          # A.5.3e (commit 7c91170 — 2 Enum fields + cross-field)
    CVConfig,             # A.5.3f (commit 26f6f2a)
    DataConfig,           # A.5.3g (commit dd23333 — composite + PrivateAttr + in-validator derivation)
    ModelConfig,          # A.5.3h (commit dd2bf20 — last leaf; Enum + nested tuple + params self-mutation)
    ExperimentConfig,     # A.5.3i (this commit — KEYSTONE; dacite retired; closes A.5 Scope D)
]
