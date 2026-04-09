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
from typing import Optional, List, Tuple, Any
import json
import yaml


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
    """XGBoost classifier."""
    
    LSTM = "lstm"
    """LSTM sequence model."""
    
    GRU = "gru"
    """GRU sequence model."""
    
    TRANSFORMER = "transformer"
    """Transformer encoder."""
    
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
    
    Note: Either feature_preset OR feature_indices can be set, not both.
    """
    
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
        
        # Feature selection validation: only one of preset or indices can be set
        if self.feature_preset is not None and self.feature_indices is not None:
            raise ValueError(
                "Cannot specify both 'feature_preset' and 'feature_indices'. "
                "Use one or the other for feature selection."
            )
        
        # Validate preset exists (import here to avoid circular imports)
        if self.feature_preset is not None:
            from lobtrainer.constants import FEATURE_PRESETS
            preset_lower = self.feature_preset.lower()
            if preset_lower not in FEATURE_PRESETS:
                available = sorted(FEATURE_PRESETS.keys())
                raise ValueError(
                    f"Unknown feature_preset: '{self.feature_preset}'. "
                    f"Available presets: {available}"
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
        if self.input_size < 1:
            raise ValueError(f"input_size must be >= 1, got {self.input_size}")
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


# =============================================================================
# Experiment Configuration (Top-Level)
# =============================================================================


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
    
    output_dir: str = "outputs"
    """Directory for checkpoints and logs."""
    
    log_level: str = "INFO"
    """Logging level: DEBUG, INFO, WARNING, ERROR."""
    
    tags: List[str] = field(default_factory=list)
    """Tags for experiment tracking."""
    
    def __post_init__(self) -> None:
        """Validate cross-config consistency."""
        # Determine expected model input size based on feature selection config
        expected_input_size = self.data.feature_count
        
        if self.data.feature_preset is not None:
            # Feature preset specified - model input must match preset size
            from lobtrainer.constants import get_feature_preset
            preset_indices = get_feature_preset(self.data.feature_preset)
            expected_input_size = len(preset_indices)
            
        elif self.data.feature_indices is not None:
            # Custom feature indices - model input must match indices count
            expected_input_size = len(self.data.feature_indices)
        
        # CRITICAL: model.input_size must match expected feature count
        # This prevents dimension mismatch errors at runtime
        if self.model.input_size != expected_input_size:
            if self.data.feature_preset is not None:
                raise ValueError(
                    f"Config mismatch: model.input_size ({self.model.input_size}) "
                    f"must equal feature_preset '{self.data.feature_preset}' size ({expected_input_size}). "
                    f"Set model.input_size: {expected_input_size}"
                )
            elif self.data.feature_indices is not None:
                raise ValueError(
                    f"Config mismatch: model.input_size ({self.model.input_size}) "
                    f"must equal len(feature_indices) ({expected_input_size}). "
                    f"Set model.input_size: {expected_input_size}"
                )
            else:
                raise ValueError(
                    f"Config mismatch: data.feature_count ({self.data.feature_count}) "
                    f"must equal model.input_size ({self.model.input_size}). "
                    f"Ensure both are set to match your exported data."
                )
        
        # Validate exclude_features are within bounds of SOURCE data
        for idx in self.data.normalization.exclude_features:
            if idx >= self.data.feature_count:
                raise ValueError(
                    f"normalization.exclude_features contains index {idx}, "
                    f"but feature_count is only {self.data.feature_count}. "
                    f"Remove or adjust exclude_features for your dataset."
                )
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        def _convert(obj: Any) -> Any:
            if isinstance(obj, Enum):
                return obj.value
            elif hasattr(obj, "__dataclass_fields__"):
                return {k: _convert(v) for k, v in asdict(obj).items()}
            elif isinstance(obj, list):
                return [_convert(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
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
        """
        from dacite import from_dict, Config as DaciteConfig
        
        # Preprocess data to handle YAML parsing quirks
        data = _normalize_config_types(data)
        
        return from_dict(
            data_class=cls,
            data=data,
            config=DaciteConfig(cast=[Enum, Path, float, int, bool]),
        )
    
    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
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

