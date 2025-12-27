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
    
    MINMAX = "minmax"
    """Min-max scaling to [0, 1]."""
    
    ROBUST = "robust"
    """Robust scaling using median and IQR."""


class LabelEncoding(str, Enum):
    """Label encoding scheme."""
    
    CATEGORICAL = "categorical"
    """Integer labels: 0=Down, 1=Stable, 2=Up."""
    
    BINARY_UP = "binary_up"
    """Binary: 1 if Up, 0 otherwise."""
    
    BINARY_DOWN = "binary_down"
    """Binary: 1 if Down, 0 otherwise."""


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
        # Default: exclude time_regime (categorical)
        if not self.exclude_features:
            from lobtrainer.constants import FeatureIndex
            self.exclude_features = [FeatureIndex.TIME_REGIME]


@dataclass
class DataConfig:
    """
    Configuration for data loading and preprocessing.
    
    Data is loaded from NumPy arrays exported by the Rust pipeline.
    """
    
    data_dir: str = "../data/exports/nvda_98feat"
    """
    Path to exported data directory (relative to lob-model-trainer or absolute).
    Directory should contain train/, val/, test/ subdirectories.
    """
    
    feature_count: int = 98
    """Number of features per sample. Must match Rust export."""
    
    sequence: SequenceConfig = field(default_factory=SequenceConfig)
    """Sequence construction configuration."""
    
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    """Feature normalization configuration."""
    
    label_encoding: LabelEncoding = LabelEncoding.CATEGORICAL
    """Label encoding scheme."""
    
    num_classes: int = 3
    """Number of output classes. Default: 3 (Down, Stable, Up)."""
    
    cache_in_memory: bool = True
    """Whether to cache all data in memory. Faster but uses more RAM."""
    
    horizon_idx: int = 0
    """
    Which prediction horizon to use for multi-horizon datasets.
    
    Index into the horizons array from the export config. For example:
        horizons = [10, 20, 50, 100, 200] in the Rust export config:
        - horizon_idx=0 → 10 samples ahead (~1 second for 10ms sampling)
        - horizon_idx=1 → 20 samples ahead (~2 seconds)
        - horizon_idx=2 → 50 samples ahead (~5 seconds)
        - horizon_idx=3 → 100 samples ahead (~10 seconds)
        - horizon_idx=4 → 200 samples ahead (~20 seconds)
    
    Set to None to return all horizons (advanced: multi-task learning).
    """
    
    def __post_init__(self) -> None:
        if self.feature_count != 98:
            raise ValueError(
                f"feature_count must be 98 for current schema, got {self.feature_count}"
            )
        if self.horizon_idx < 0:
            raise ValueError(
                f"horizon_idx must be >= 0, got {self.horizon_idx}"
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
    
    Architecture-specific parameters are in nested dataclasses or dicts.
    """
    
    model_type: ModelType = ModelType.LSTM
    """Model architecture type."""
    
    input_size: int = 98
    """Input feature dimension. Must match DataConfig.feature_count."""
    
    hidden_size: int = 64
    """Hidden layer size for sequence models."""
    
    num_layers: int = 2
    """Number of layers for sequence models."""
    
    dropout: float = 0.2
    """Dropout probability. Range: [0, 1]."""
    
    num_classes: int = 3
    """Number of output classes."""
    
    # Architecture-specific parameters
    lstm_bidirectional: bool = False
    """Use bidirectional LSTM/GRU. Doubles hidden dimension for classifier."""
    
    lstm_attention: bool = False
    """Use self-attention over LSTM sequence outputs before classification."""
    
    transformer_num_heads: int = 4
    """Number of attention heads for Transformer."""
    
    transformer_dim_feedforward: int = 256
    """Feedforward dimension for Transformer."""
    
    # DeepLOB-specific parameters (Zhang et al. 2019)
    deeplob_mode: DeepLOBMode = DeepLOBMode.BENCHMARK
    """
    DeepLOB operational mode:
    - benchmark: Original paper architecture, uses first 40 LOB features
    - extended: Adapted for all 98 features (experimental)
    """
    
    deeplob_conv_filters: int = 32
    """Number of filters in DeepLOB convolutional blocks. Paper default: 32."""
    
    deeplob_inception_filters: int = 64
    """Number of filters per Inception branch. Paper default: 64."""
    
    deeplob_lstm_hidden: int = 64
    """LSTM hidden size in DeepLOB. Paper default: 64."""
    
    deeplob_num_levels: int = 10
    """Number of LOB levels to use. Paper default: 10."""
    
    def __post_init__(self) -> None:
        if self.dropout < 0 or self.dropout > 1:
            raise ValueError(f"dropout must be in [0, 1], got {self.dropout}")
        if self.hidden_size < 1:
            raise ValueError(f"hidden_size must be >= 1, got {self.hidden_size}")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")
        if self.deeplob_conv_filters < 1:
            raise ValueError(f"deeplob_conv_filters must be >= 1, got {self.deeplob_conv_filters}")
        if self.deeplob_inception_filters < 1:
            raise ValueError(f"deeplob_inception_filters must be >= 1, got {self.deeplob_inception_filters}")
        if self.deeplob_lstm_hidden < 1:
            raise ValueError(f"deeplob_lstm_hidden must be >= 1, got {self.deeplob_lstm_hidden}")
        if self.deeplob_num_levels < 1:
            raise ValueError(f"deeplob_num_levels must be >= 1, got {self.deeplob_num_levels}")


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
    
    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")


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
        """Create config from dictionary."""
        from dacite import from_dict, Config as DaciteConfig
        return from_dict(
            data_class=cls,
            data=data,
            config=DaciteConfig(cast=[Enum, Path]),
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

