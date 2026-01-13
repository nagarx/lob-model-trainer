# LOB-Model-Trainer: Codebase Technical Reference

> **Version**: 0.4.0  
> **Last Updated**: January 13, 2026  
> **Purpose**: Complete technical reference for LLMs and developers to understand, modify, and extend the codebase.
>
> **Scope**: This library focuses solely on **model training**. For dataset analysis, use `lob-dataset-analyzer`.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Module Architecture](#2-module-architecture)
3. [Core Data Flow](#3-core-data-flow)
4. [Constants and Feature Indices](#4-constants-and-feature-indices)
5. [Configuration System](#5-configuration-system)
6. [Data Loading Pipeline](#6-data-loading-pipeline)
7. [PyTorch Dataset Classes](#7-pytorch-dataset-classes)
8. [Data Transforms](#8-data-transforms)
9. [Model Implementations](#9-model-implementations)
10. [Training Infrastructure](#10-training-infrastructure)
11. [Callback System](#11-callback-system)
12. [Metrics and Evaluation](#12-metrics-and-evaluation)
13. [Baseline Models](#13-baseline-models)
14. [Monitoring and Diagnostics](#14-monitoring-and-diagnostics)
15. [Experiment Tracking](#15-experiment-tracking)
16. [Reproducibility Utilities](#16-reproducibility-utilities)
17. [Scripts and CLI](#17-scripts-and-cli)
18. [Configuration Reference](#18-configuration-reference)
19. [Testing Patterns](#19-testing-patterns)
20. [Known Limitations](#20-known-limitations)

---

## 1. Project Overview

### Purpose

Python library for training and evaluating ML models on LOB (Limit Order Book) data for price movement prediction. Designed for HFT research with emphasis on:

- **Modularity**: Clean separation between data, models, and training
- **Reproducibility**: Explicit seed management and configuration-driven experiments
- **Flexibility**: Multi-horizon labels, multiple model architectures
- **Monitoring**: Gradient tracking, training diagnostics, experiment comparison

### Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| **LSTM/GRU Models** | ✅ Complete | Sequence models with optional attention |
| **DeepLOB** | ✅ Complete | Via `lob-models` package integration |
| **TLOB** | ✅ Complete | Via `lob-models` package integration |
| **Strategy-Aware Metrics** | ✅ Complete | MetricsCalculator for TLOB/Triple Barrier/Opportunity |
| **Focal Loss** | ✅ Complete | For class imbalance handling |
| **Multi-Horizon Labels** | ✅ Complete | Support for multiple prediction horizons |
| **Experiment Tracking** | ✅ Complete | ExperimentRegistry, comparison tables |
| **Monitoring Callbacks** | ✅ Complete | Gradient, LR, diagnostics tracking |
| **Tests** | ✅ Complete | 15 test modules |

### Core Dependencies

```toml
torch = ">=2.0"           # Deep learning framework
numpy = ">=1.24"          # Numerical operations
pandas = ">=2.0"          # Data manipulation
scikit-learn = ">=1.3"    # Classical ML, metrics
scipy = ">=1.10"          # Statistical tests
pyyaml = ">=6.0"          # Configuration files
dacite = ">=1.8"          # Dataclass from dict
tqdm = ">=4.65"           # Progress bars
```

---

## 2. Module Architecture

```
src/lobtrainer/
├── __init__.py                    # Public API exports (v0.4.0)
│
├── constants/
│   ├── __init__.py                # Module exports
│   ├── feature_index.py           # FeatureIndex, SignalIndex (98 features)
│   └── feature_presets.py         # Named feature subsets (8 presets)
│
├── config/
│   ├── __init__.py                # Module exports
│   └── schema.py                  # ExperimentConfig, DataConfig, ModelConfig, TrainConfig
│
├── data/
│   ├── __init__.py                # Module exports
│   ├── dataset.py                 # DayData, LOBFlatDataset, LOBSequenceDataset
│   └── transforms.py              # ZScoreNormalizer, BinaryLabelTransform
│
├── models/
│   ├── __init__.py                # create_model factory, LOBMODELS_AVAILABLE
│   ├── lstm.py                    # LSTMClassifier, GRUClassifier, LSTMConfig
│   └── baselines.py               # NaiveClassPrior, NaivePreviousLabel, LogisticBaseline
│
├── training/
│   ├── __init__.py                # Module exports (30+ symbols)
│   ├── trainer.py                 # Trainer, TrainingState, create_trainer
│   ├── callbacks.py               # EarlyStopping, ModelCheckpoint, MetricLogger
│   ├── metrics.py                 # MetricsCalculator, ClassificationMetrics
│   ├── loss.py                    # FocalLoss, BinaryFocalLoss, create_focal_loss
│   ├── evaluation.py              # BaselineReport, evaluate_model, full_evaluation
│   └── monitoring.py              # GradientMonitor, TrainingDiagnostics, LRTracker
│
├── experiments/
│   ├── __init__.py                # Module exports
│   ├── result.py                  # ExperimentResult, ExperimentMetrics
│   └── registry.py                # ExperimentRegistry, create_comparison_table
│
└── utils/
    ├── __init__.py                # Module exports
    └── reproducibility.py         # set_seed, SeedManager, worker_init_fn

scripts/
├── train.py                       # Training CLI
├── evaluate_model.py              # Model evaluation CLI
├── run_baseline_evaluation.py     # Baseline comparison
└── validate_export.py             # Dataset validation

configs/
├── README_configs.md              # Complete config reference
├── experiments/                   # Active experiment configs (3)
│   ├── nvda_tlob_h10_v1.yaml
│   ├── nvda_tlob_h100_v1.yaml
│   └── nvda_tlob_triple_barrier_11mo_v1.yaml
└── archive/                       # Reference configs (6)
    ├── baseline_lstm.yaml
    ├── deeplob_benchmark.yaml
    ├── lstm_attn_bidir_h20.yaml
    ├── nvda_bigmove_opportunity_v1.yaml
    ├── nvda_tlob_bigmove_v1.yaml
    └── nvda_tlob_binary_signal_v1.yaml

tests/                             # 14 test modules
├── test_baselines.py
├── test_config.py
├── test_deeplob_integration.py
├── test_evaluation.py
├── test_experiments.py
├── test_feature_index.py
├── test_feature_presets.py
├── test_integration.py
├── test_loss.py
├── test_monitoring.py
├── test_strategy_metrics.py
├── test_tlob_integration.py
├── test_trainer.py
└── test_transforms.py
```

---

## 3. Core Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       TRAINING PIPELINE DATA FLOW                            │
└─────────────────────────────────────────────────────────────────────────────┘

    Raw NumPy Files           DayData Objects           PyTorch Tensors
         │                          │                          │
         ▼                          ▼                          ▼
┌─────────────────┐     ┌─────────────────────┐     ┌──────────────────┐
│ {date}_sequences│────▶│     DayData         │────▶│ LOBSequenceDataset│
│ {date}_labels   │     │                     │     │                  │
│ (per day)       │     │ .sequences [N,T,F]  │     │ .__getitem__()   │
│                 │     │ .labels [N] or [N,H]│     │  → (seq, label)  │
└─────────────────┘     │ .get_labels(h_idx)  │     └──────────────────┘
                        └─────────────────────┘              │
                                                             ▼
                                                    ┌──────────────────┐
                                                    │   DataLoader     │
                                                    │   (batches)      │
                                                    └──────────────────┘
                                                             │
                                                             ▼
                                                    ┌──────────────────┐
                                                    │     Trainer      │
                                                    │                  │
                                                    │ model(features)  │
                                                    │ → predictions    │
                                                    │                  │
                                                    │ loss.backward()  │
                                                    │ optimizer.step() │
                                                    └──────────────────┘
```

### Label Encoding Flow

```
Original Labels: {-1, 0, 1}    →    Dataset Shift: {0, 1, 2}    →    Metrics Display
    (Down, Stable, Up)                 (for CrossEntropyLoss)        (Down, Stable, Up)
                                  
IMPORTANT: Labels are shifted by +1 in __getitem__ for PyTorch compatibility.
```

---

## 4. Constants and Feature Indices

### FeatureIndex (src/lobtrainer/constants/feature_index.py)

The feature vector has exactly **98 features**:

```python
class FeatureIndex:
    """Zero-based indices into the 98-feature vector."""
    
    # Raw LOB (40 features: indices 0-39)
    ASK_PRICE_0 = 0      # Best ask price
    ASK_PRICE_9 = 9      # Level 10 ask price
    ASK_SIZE_0 = 10      # Best ask size
    ASK_SIZE_9 = 19      # Level 10 ask size
    BID_PRICE_0 = 20     # Best bid price
    BID_PRICE_9 = 29     # Level 10 bid price
    BID_SIZE_0 = 30      # Best bid size
    BID_SIZE_9 = 39      # Level 10 bid size
    
    # Derived Features (8 features: indices 40-47)
    MID_PRICE = 40
    SPREAD = 41
    SPREAD_BPS = 42
    TOTAL_BID_VOLUME = 43
    TOTAL_ASK_VOLUME = 44
    VOLUME_IMBALANCE = 45
    WEIGHTED_MID_PRICE = 46
    PRICE_IMPACT = 47
    
    # MBO Features (36 features: indices 48-83)
    # ... order flow, size distribution, queue metrics
    
    # Trading Signals (14 features: indices 84-97)
    TRUE_OFI = 84
    DEPTH_NORM_OFI = 85
    EXECUTED_PRESSURE = 86
    SIGNED_MP_DELTA_BPS = 87
    TRADE_ASYMMETRY = 88
    CANCEL_ASYMMETRY = 89
    FRAGILITY_SCORE = 90
    DEPTH_ASYMMETRY = 91
    BOOK_VALID = 92          # Safety gate
    TIME_REGIME = 93         # Categorical {0-4}
    MBO_READY = 94           # Safety gate
    DT_SECONDS = 95
    INVALIDITY_DELTA = 96    # Safety gate
    SCHEMA_VERSION = 97
```

### Label Encoding

```python
# Original labels (from Rust pipeline)
LABEL_DOWN: Final[int] = -1
LABEL_STABLE: Final[int] = 0
LABEL_UP: Final[int] = 1
NUM_CLASSES: Final[int] = 3

# Shifted labels (for PyTorch CrossEntropyLoss)
SHIFTED_LABEL_DOWN: Final[int] = 0    # Was -1
SHIFTED_LABEL_STABLE: Final[int] = 1  # Was 0
SHIFTED_LABEL_UP: Final[int] = 2      # Was 1
```

### Feature Presets (src/lobtrainer/constants/feature_presets.py)

Named feature subsets for easy configuration:

```python
FEATURE_PRESETS = {
    "lob_only": list(range(0, 40)),         # 40 raw LOB features
    "lob_derived": list(range(0, 48)),      # LOB + derived (48)
    "full": list(range(0, 98)),             # All 98 features
    "signals_core": [84, 85, 86, 87, 88, 89, 90, 91],  # 8 core signals
    "signals_full": list(range(84, 98)),    # 14 signal features
    "lob_signals": list(range(0, 40)) + list(range(84, 92)),  # LOB + core signals
    "no_meta": list(range(0, 92)),          # Exclude meta (92-97)
    "deeplob_extended": list(range(0, 48)), # For extended DeepLOB mode
}

# Usage
from lobtrainer.constants import get_feature_preset, list_presets, describe_preset

indices = get_feature_preset("signals_core")  # [84, 85, 86, 87, 88, 89, 90, 91]
print(list_presets())  # ['lob_only', 'lob_derived', 'full', ...]
describe_preset("signals_core")  # Prints description and indices
```

---

## 5. Configuration System

### ExperimentConfig (src/lobtrainer/config/schema.py)

Root configuration object:

```python
@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    name: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    
    output_dir: str = "outputs"
    log_level: str = "INFO"
```

### DataConfig

```python
@dataclass
class DataConfig:
    data_dir: str = "../data/exports/nvda_11month_complete"
    feature_count: int = 98
    horizon_idx: Optional[int] = 0
    
    sequence: SequenceConfig = field(default_factory=SequenceConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    
    labeling_strategy: LabelingStrategy = LabelingStrategy.TLOB
    num_classes: int = 3
    cache_in_memory: bool = True
```

### ModelConfig

```python
class ModelType(str, Enum):
    LOGISTIC = "logistic"
    XGBOOST = "xgboost"      # Not implemented
    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"  # Not implemented
    DEEPLOB = "deeplob"
    TLOB = "tlob"

@dataclass
class ModelConfig:
    model_type: ModelType = ModelType.LSTM
    input_size: int = 98
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    num_classes: int = 3
    
    # LSTM-specific
    lstm_bidirectional: bool = False
    lstm_attention: bool = False
    
    # DeepLOB-specific
    deeplob_mode: DeepLOBMode = DeepLOBMode.BENCHMARK
    deeplob_conv_filters: int = 32
    deeplob_inception_filters: int = 64
    deeplob_lstm_hidden: int = 64
    deeplob_num_levels: int = 10
    
    # TLOB-specific
    tlob_hidden_dim: int = 64
    tlob_num_layers: int = 4
    tlob_num_heads: int = 1
    tlob_mlp_expansion: float = 4.0
    tlob_use_sinusoidal_pe: bool = True
    tlob_use_bin: bool = True
    tlob_dataset_type: str = "nvda"
```

### TrainConfig

```python
class LossType(str, Enum):
    CROSS_ENTROPY = "cross_entropy"
    FOCAL = "focal"
    WEIGHTED_CE = "weighted_ce"

class TaskType(str, Enum):
    MULTICLASS = "multiclass"
    BINARY_SIGNAL = "binary_signal"

@dataclass
class TrainConfig:
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 50
    early_stopping_patience: int = 10
    gradient_clip_norm: Optional[float] = 1.0
    
    scheduler: str = "cosine"
    num_workers: int = 4
    pin_memory: bool = True
    seed: int = 42
    
    # Loss configuration
    loss_type: LossType = LossType.WEIGHTED_CE
    use_class_weights: bool = True
    task_type: TaskType = TaskType.MULTICLASS
    
    # Focal loss parameters
    focal_gamma: float = 2.0
    focal_alpha: Optional[float] = None
```

### LabelingStrategy

```python
class LabelingStrategy(str, Enum):
    TLOB = "tlob"
    """Classes: 0=Down, 1=Stable, 2=Up (default)"""
    
    TRIPLE_BARRIER = "triple_barrier"
    """Classes: 0=StopLoss, 1=Timeout, 2=ProfitTarget"""
    
    OPPORTUNITY = "opportunity"
    """Classes: 0=BigDown, 1=NoOpportunity, 2=BigUp"""
```

---

## 6. Data Loading Pipeline

### Directory Structure Expected

```
data/exports/nvda_11month_complete/
├── train/
│   ├── 2025-02-03_sequences.npy    # [N_seq, 100, 98] float32
│   ├── 2025-02-03_labels.npy       # [N_seq, 4] int8 (multi-horizon)
│   └── ...
├── val/
│   └── ...
├── test/
│   └── ...
└── dataset_manifest.json
```

### DayData Class

```python
@dataclass
class DayData:
    """Container for one day's data."""
    date: str                              # e.g., "2025-02-03"
    features: np.ndarray                   # [N, 98] - flat features
    labels: np.ndarray                     # [N_seq] or [N_seq, n_horizons]
    sequences: Optional[np.ndarray] = None # [N_seq, 100, 98]
    metadata: Optional[Dict] = None
    is_aligned: bool = False
    
    @property
    def num_sequences(self) -> int:
        return self.labels.shape[0]
    
    @property
    def is_multi_horizon(self) -> bool:
        return self.labels.ndim == 2
    
    @property
    def num_horizons(self) -> int:
        return self.labels.shape[1] if self.is_multi_horizon else 1
    
    def get_labels(self, horizon_idx: Optional[int] = 0) -> np.ndarray:
        """Get labels for specific horizon or all horizons."""
        if horizon_idx is None:
            return self.labels
        if not self.is_multi_horizon:
            return self.labels
        return self.labels[:, horizon_idx]
```

### Loading Functions

```python
from lobtrainer.data import load_split_data, load_day_data, create_dataloaders

# Load all days in a split
train_days: List[DayData] = load_split_data(Path("data/exports/nvda_11month_complete"), "train")

# Create dataloaders
loaders = create_dataloaders(
    data_dir="data/exports/nvda_11month_complete",
    batch_size=64,
    horizon_idx=0,
)
```

---

## 7. PyTorch Dataset Classes

### LOBSequenceDataset

For sequence models (LSTM, Transformer, TLOB):

```python
class LOBSequenceDataset(Dataset):
    """
    Each item is (sequence, label) where:
    - sequence: [seq_len, n_features] tensor
    - label: scalar tensor (shifted to {0, 1, 2})
    """
    
    def __init__(
        self,
        days: List[DayData],
        horizon_idx: Optional[int] = 0,
        feature_indices: Optional[List[int]] = None,
        transform: Optional[Callable] = None,
    ): ...
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get sequence and label
        sequence = day.sequences[local_idx].copy()
        label = day.get_labels(self.horizon_idx)[local_idx]
        
        # CRITICAL: Shift labels from {-1, 0, 1} to {0, 1, 2}
        label = label + 1
        
        return (
            torch.from_numpy(sequence).float(),
            torch.tensor(label, dtype=torch.long),
        )
```

### LOBFlatDataset

For non-sequence models (XGBoost, MLP, Logistic):

```python
class LOBFlatDataset(Dataset):
    """
    Each item is (features, label) where features is flattened.
    """
    
    def __init__(
        self,
        days: List[DayData],
        flatten_mode: str = "last",  # "last", "flatten", "mean"
        horizon_idx: Optional[int] = 0,
        ...
    ): ...
```

---

## 8. Data Transforms

### ZScoreNormalizer

```python
class ZScoreNormalizer:
    """Z-score normalization with feature exclusion."""
    
    def __init__(
        self,
        eps: float = 1e-8,
        clip_value: float = 10.0,
        exclude_features: Optional[List[int]] = None,  # e.g., [93] for TIME_REGIME
    ): ...
    
    def fit(self, data: np.ndarray) -> 'ZScoreNormalizer': ...
    def transform(self, data: np.ndarray) -> np.ndarray: ...
    def fit_transform(self, data: np.ndarray) -> np.ndarray: ...
```

### BinaryLabelTransform

```python
class BinaryLabelTransform:
    """Convert multi-class to binary (signal vs no-signal)."""
    
    def __init__(self, positive_classes: List[int] = [0, 2]):
        """Classes 0 (Down) and 2 (Up) become 1 (Signal)."""
        ...
```

---

## 9. Model Implementations

### LSTMClassifier (src/lobtrainer/models/lstm.py)

```python
class LSTMClassifier(nn.Module):
    """
    LSTM-based classifier with optional attention and bidirectional.
    
    Architecture:
        Input [B, T, F] → LSTM layers → (optional) Attention → FC → Output [B, C]
    """
    
    def __init__(
        self,
        input_size: int = 98,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.2,
        bidirectional: bool = False,
        attention: bool = False,
    ): ...
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: x: [batch, seq_len, features]
        Returns: logits: [batch, num_classes]
        """
        ...
```

### Model Factory

```python
from lobtrainer.models import create_model, LOBMODELS_AVAILABLE
from lobtrainer.config import ModelConfig, ModelType

# LSTM
model = create_model(ModelConfig(model_type=ModelType.LSTM))

# DeepLOB (requires lob-models package)
model = create_model(ModelConfig(
    model_type=ModelType.DEEPLOB,
    deeplob_mode=DeepLOBMode.BENCHMARK,
))

# TLOB (requires lob-models package)
model = create_model(ModelConfig(
    model_type=ModelType.TLOB,
    tlob_hidden_dim=64,
    tlob_num_layers=4,
))
```

---

## 10. Training Infrastructure

### Trainer Class (src/lobtrainer/training/trainer.py)

```python
class Trainer:
    """Main training class for LOB models."""
    
    def __init__(
        self,
        config: ExperimentConfig,
        callbacks: Optional[List[Callback]] = None,
        device: Optional[torch.device] = None,
    ): ...
    
    def train(self) -> Dict[str, Any]:
        """
        Run full training loop.
        
        Returns:
            Dict with best_val_metric, best_epoch, total_epochs, history
        """
        ...
    
    def evaluate(self, split: str = "test") -> ClassificationMetrics:
        """Evaluate model on a data split."""
        ...
```

### TrainingState

```python
@dataclass
class TrainingState:
    """Mutable training state."""
    current_epoch: int = 0
    global_step: int = 0
    best_val_metric: float = float('inf')
    best_epoch: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)
```

### Loss Functions (src/lobtrainer/training/loss.py)

```python
class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Reference: Lin et al. (2017), "Focal Loss for Dense Object Detection"
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ): ...

class BinaryFocalLoss(nn.Module):
    """Focal Loss for binary classification."""
    ...

def create_focal_loss(
    num_classes: int,
    gamma: float = 2.0,
    class_counts: Optional[torch.Tensor] = None,
) -> nn.Module:
    """Factory function for focal loss."""
    ...
```

---

## 11. Callback System

### Base Callback (src/lobtrainer/training/callbacks.py)

```python
class Callback:
    """Base class for training callbacks."""
    
    def on_train_start(self) -> None: ...
    def on_train_end(self) -> None: ...
    def on_epoch_start(self, epoch: int) -> None: ...
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None: ...
    def on_batch_start(self, batch_idx: int) -> None: ...
    def on_batch_end(self, batch_idx: int, metrics: Dict[str, float]) -> None: ...
    def on_validation_end(self, epoch: int, metrics: Dict[str, float]) -> None: ...
```

### Built-in Callbacks

```python
class EarlyStopping(Callback):
    """Stop training when metric stops improving."""
    def __init__(
        self,
        patience: int = 10,
        metric: str = "val_loss",
        mode: str = "min",
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
    ): ...

class ModelCheckpoint(Callback):
    """Save model checkpoints during training."""
    def __init__(
        self,
        save_dir: Union[str, Path],
        metric: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
        max_checkpoints: int = 5,
    ): ...

class MetricLogger(Callback):
    """Log metrics to file and console."""
    ...

class ProgressCallback(Callback):
    """Display training progress with tqdm."""
    ...
```

---

## 12. Metrics and Evaluation

### MetricsCalculator (src/lobtrainer/training/metrics.py)

Strategy-aware metrics calculator:

```python
class MetricsCalculator:
    """
    Computes metrics that match labeling strategy semantics.
    
    Example:
        >>> calc = MetricsCalculator("triple_barrier")
        >>> metrics = calc.compute(preds, labels)
        >>> print(metrics.strategy_metrics["predicted_trade_win_rate"])
    """
    
    def __init__(self, strategy: str = "tlob", num_classes: int = 3): ...
    
    def compute(
        self,
        predictions: Tensor,
        labels: Tensor,
        loss: Optional[float] = None,
    ) -> ClassificationMetrics: ...
```

### ClassificationMetrics

```python
@dataclass
class ClassificationMetrics:
    """Complete classification evaluation results."""
    accuracy: float
    loss: float
    
    per_class_precision: Dict[int, float]
    per_class_recall: Dict[int, float]
    per_class_f1: Dict[int, float]
    per_class_count: Dict[int, int]
    
    macro_precision: float
    macro_recall: float
    macro_f1: float
    
    confusion_matrix: np.ndarray
    
    # Strategy-specific metrics
    strategy_metrics: Dict[str, float]
    """
    Examples:
    - triple_barrier: predicted_trade_win_rate, decisive_prediction_rate
    - opportunity: directional_accuracy, opportunity_prediction_rate
    - tlob: directional_accuracy, signal_rate
    """
    
    class_names: List[str]
    
    def summary(self) -> str: ...
    def to_dict(self) -> Dict[str, float]: ...
```

### Strategy-Specific Metrics

| Strategy | Key Metrics | Description |
|----------|-------------|-------------|
| **Triple Barrier** | `predicted_trade_win_rate` | When predicting trade, actual win rate |
| | `decisive_prediction_rate` | How often we predict StopLoss or ProfitTarget |
| **Opportunity** | `directional_accuracy` | When predicting direction, correctness |
| | `opportunity_prediction_rate` | How often we predict BigUp or BigDown |
| **TLOB** | `directional_accuracy` | Accuracy on non-Stable predictions |
| | `signal_rate` | Fraction of Up/Down predictions |

### Evaluation Framework (src/lobtrainer/training/evaluation.py)

```python
@dataclass
class BaselineReport:
    """Comprehensive report comparing model against baselines."""
    
    model_name: str
    split: str
    n_samples: int
    
    model_metrics: ClassificationMetrics
    class_prior_metrics: ClassificationMetrics
    previous_label_metrics: ClassificationMetrics
    
    beats_class_prior: bool
    beats_previous_label: bool
    improvement_over_prior: float
    improvement_over_previous: float

def evaluate_model(model, X, y, name=None) -> ClassificationMetrics: ...
def evaluate_naive_baseline(y_true, split_name="test") -> Dict[str, ClassificationMetrics]: ...
def create_baseline_report(model, X, y, split="test") -> BaselineReport: ...
def full_evaluation(model, X, y, split="test") -> Dict: ...
```

---

## 13. Baseline Models

### Available Baselines (src/lobtrainer/models/baselines.py)

```python
class NaiveClassPrior(BaseModel):
    """Always predicts the most common class."""
    
    def fit(self, X, y) -> 'NaiveClassPrior': ...
    def predict(self, X) -> np.ndarray: ...

class NaivePreviousLabel(BaseModel):
    """
    Persistence baseline: predict the previous label.
    Exploits label autocorrelation.
    """
    
    def predict(self, X) -> np.ndarray: ...

class LogisticBaseline(BaseModel):
    """Logistic regression baseline using sklearn."""
    
    def __init__(
        self,
        config: LogisticBaselineConfig = None,
        flatten_mode: str = "last",
        feature_indices: Optional[List[int]] = None,
    ): ...
```

---

## 14. Monitoring and Diagnostics

### GradientMonitor (src/lobtrainer/training/monitoring.py)

```python
@dataclass
class GradientStats:
    total_norm: float
    max_norm: float
    min_norm: float
    mean_norm: float
    num_zero_grads: int
    num_nan_grads: int
    num_inf_grads: int
    layer_norms: Dict[str, float]

class GradientMonitor(Callback):
    """
    Monitor gradient statistics to detect training issues.
    
    Detects:
    - Vanishing gradients (norms < 1e-7)
    - Exploding gradients (norms > 1000)
    - NaN gradients
    """
    
    def __init__(self, log_every: int = 100): ...
    def get_history(self) -> List[GradientStats]: ...
```

### TrainingDiagnostics

```python
@dataclass
class HealthCheckResult:
    is_healthy: bool
    warnings: List[str]
    recommendations: List[str]

class TrainingDiagnostics(Callback):
    """
    Aggregate training diagnostics.
    
    Tracks:
    - Loss trends
    - Gradient health
    - Learning rate
    - Time per epoch
    """
    
    def get_summary(self) -> Dict: ...
    def check_health(self) -> HealthCheckResult: ...
```

### PerClassMetricsTracker

```python
class PerClassMetricsTracker(Callback):
    """
    Track per-class metrics (precision, recall, F1) per epoch.
    
    Useful for detecting:
    - Model ignoring minority class
    - Precision/recall trade-offs
    """
    ...
```

### Convenience Factory

```python
def create_standard_monitoring(
    log_every: int = 100,
    include_gradients: bool = True,
    include_lr: bool = True,
    include_diagnostics: bool = True,
    include_per_class: bool = False,
) -> List[Callback]:
    """Create standard monitoring callbacks."""
    ...
```

---

## 15. Experiment Tracking

### ExperimentResult (src/lobtrainer/experiments/result.py)

```python
@dataclass
class ExperimentMetrics:
    accuracy: float = 0.0
    loss: float = 0.0
    macro_f1: float = 0.0
    directional_accuracy: float = 0.0
    signal_rate: float = 0.0
    per_class_f1: Dict[str, float] = field(default_factory=dict)
    strategy_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class ExperimentResult:
    experiment_id: str
    name: str
    config: Dict[str, Any]
    
    train_metrics: Optional[ExperimentMetrics] = None
    val_metrics: Optional[ExperimentMetrics] = None
    test_metrics: Optional[ExperimentMetrics] = None
    
    timestamp: str = ""
    duration_seconds: float = 0.0
    checkpoint_path: Optional[str] = None
    
    def save(self, path: Path) -> None: ...
    
    @classmethod
    def load(cls, path: Path) -> 'ExperimentResult': ...
```

### ExperimentRegistry (src/lobtrainer/experiments/registry.py)

```python
class ExperimentRegistry:
    """Central tracker for all experiments."""
    
    def __init__(self, base_dir: Union[str, Path], create_if_missing: bool = True): ...
    
    def register(self, result: ExperimentResult) -> str: ...
    def get(self, experiment_id: str) -> Optional[ExperimentResult]: ...
    def list_all(self) -> List[ExperimentResult]: ...
    def filter(
        self,
        tags: Optional[List[str]] = None,
        model_type: Optional[str] = None,
        min_accuracy: Optional[float] = None,
    ) -> List[ExperimentResult]: ...

def create_comparison_table(
    registry: ExperimentRegistry,
    metric_keys: List[str] = None,
    sort_by: str = "macro_f1",
) -> str:
    """Create markdown comparison table of experiments."""
    ...
```

---

## 16. Reproducibility Utilities

### set_seed (src/lobtrainer/utils/reproducibility.py)

```python
def set_seed(seed: int = 42, deterministic_cudnn: bool = True) -> None:
    """
    Set random seeds for all RNGs.
    
    Sets:
    - Python random module
    - NumPy random
    - PyTorch CPU
    - PyTorch CUDA (all GPUs)
    - CuDNN determinism (if enabled)
    """
    ...

@dataclass
class SeedManager:
    """Context manager for reproducible code blocks."""
    seed: int
    restore_state: bool = False
    
    def __enter__(self) -> 'SeedManager': ...
    def __exit__(self, *args) -> None: ...

def create_worker_init_fn(base_seed: int):
    """Create worker_init_fn for DataLoader with deterministic seeding."""
    ...
```

---

## 17. Scripts and CLI

### train.py

```bash
# Basic training
python scripts/train.py --config configs/experiments/nvda_tlob_h10_v1.yaml

# With overrides
python scripts/train.py --config configs/experiments/nvda_tlob_h10_v1.yaml \
    --epochs 50 \
    --batch-size 128

# Resume from checkpoint
python scripts/train.py --config configs/experiments/nvda_tlob_h10_v1.yaml \
    --resume outputs/checkpoints/best.pt

# Evaluation only
python scripts/train.py --config configs/experiments/nvda_tlob_h10_v1.yaml \
    --eval-only --resume outputs/checkpoints/best.pt
```

### Other Scripts

| Script | Purpose |
|--------|---------|
| `evaluate_model.py` | Evaluate trained model checkpoint |
| `run_baseline_evaluation.py` | Compare against naive baselines |
| `validate_export.py` | Validate exported dataset integrity |

---

## 18. Configuration Reference

See `configs/README_configs.md` for complete configuration reference including:

- Active experiment configs (3)
- Archived reference configs (6)
- Horizon index mapping
- Model type options
- Loss function selection
- Configuration template

### Current Datasets

| Dataset | Days | Labels | Horizons |
|---------|------|--------|----------|
| `nvda_11month_complete` | **234** | TLOB | [10, 20, 50, 100] |
| `nvda_11month_triple_barrier` | **234** | Triple Barrier | [50, 100, 200] |

---

## 19. Testing Patterns

### Test Structure

```python
# tests/test_trainer.py
class TestTrainingState:
    def test_initial_state(self):
        state = TrainingState()
        assert state.current_epoch == 0
        assert state.best_val_metric == float('inf')

class TestTrainer:
    @pytest.fixture
    def mock_config(self, tmp_path):
        return ExperimentConfig(...)
    
    def test_trainer_creation(self, mock_config):
        trainer = Trainer(mock_config)
        assert trainer.config.name == "test"
```

### Test Modules (14)

| Test File | Coverage |
|-----------|----------|
| `test_baselines.py` | NaiveClassPrior, NaivePreviousLabel, LogisticBaseline |
| `test_config.py` | Configuration loading and validation |
| `test_deeplob_integration.py` | DeepLOB model creation |
| `test_evaluation.py` | BaselineReport, evaluate_model |
| `test_experiments.py` | ExperimentResult, ExperimentRegistry |
| `test_feature_index.py` | Feature indices and constants |
| `test_feature_presets.py` | Feature preset functions |
| `test_integration.py` | End-to-end training |
| `test_loss.py` | FocalLoss, BinaryFocalLoss |
| `test_monitoring.py` | GradientMonitor, TrainingDiagnostics |
| `test_strategy_metrics.py` | MetricsCalculator |
| `test_tlob_integration.py` | TLOB model creation |
| `test_trainer.py` | Trainer class |
| `test_transforms.py` | ZScoreNormalizer, transforms |

---

## 20. Known Limitations

### Label Shift (+1)

Labels are shifted from `{-1, 0, 1}` to `{0, 1, 2}` in `__getitem__`:

```python
label = label + 1  # For PyTorch CrossEntropyLoss
```

### Single-Horizon Training

While data supports multi-horizon labels, the trainer trains on a single horizon:

```yaml
horizon_idx: 0  # Use first horizon
```

### num_workers Default

DataLoader workers default to 4. Set to 0 if experiencing multiprocessing issues.

### External Model Dependency

DeepLOB and TLOB require the `lob-models` package:

```bash
pip install -e ../lob-models
```

Check availability:

```python
from lobtrainer.models import LOBMODELS_AVAILABLE
print(LOBMODELS_AVAILABLE)  # True if installed
```

### TIME_REGIME Exclusion

Index 93 (TIME_REGIME) should be excluded from normalization - it's categorical `{0, 1, 2, 3, 4}`.

---

## Quick Reference

### Imports

```python
# Core
from lobtrainer import Trainer, create_trainer, set_seed
from lobtrainer.config import load_config, ExperimentConfig, ModelType

# Models
from lobtrainer.models import create_model, LSTMClassifier, LogisticBaseline

# Data
from lobtrainer.data import LOBSequenceDataset, create_dataloaders

# Metrics
from lobtrainer.training import MetricsCalculator, ClassificationMetrics

# Experiments
from lobtrainer.experiments import ExperimentResult, ExperimentRegistry

# Constants
from lobtrainer.constants import FeatureIndex, get_feature_preset
```

### Key Defaults

| Parameter | Default |
|-----------|---------|
| Sequence length | 100 |
| Stride | 10 |
| Batch size | 64 |
| Hidden size | 64 |
| Learning rate | 1e-4 |
| Epochs | 50 |
| Patience | 10 |
| Seed | 42 |

---

*Last updated: January 13, 2026*
*Version: 0.4.0*
