# LOB-Model-Trainer: Codebase Technical Reference

> **Purpose**: This document provides complete technical details for LLMs and developers to understand, modify, and extend the codebase without prior context.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Module Architecture](#2-module-architecture)
3. [Core Data Flow](#3-core-data-flow)
4. [Constants and Feature Indices](#4-constants-and-feature-indices)
5. [Configuration System](#5-configuration-system)
6. [Data Loading Pipeline](#6-data-loading-pipeline)
7. [PyTorch Dataset Classes](#7-pytorch-dataset-classes)
8. [Data Transforms and Normalization](#8-data-transforms-and-normalization)
9. [Model Implementations](#9-model-implementations)
10. [Training Infrastructure](#10-training-infrastructure)
11. [Callback System](#11-callback-system)
12. [Evaluation Metrics](#12-evaluation-metrics)
13. [Baseline Models](#13-baseline-models)
14. [Streaming Analysis Module](#14-streaming-analysis-module)
15. [Statistical Analysis Suite](#15-statistical-analysis-suite)
16. [Reproducibility Utilities](#16-reproducibility-utilities)
17. [Scripts and CLI](#17-scripts-and-cli)
18. [Configuration Examples](#18-configuration-examples)
19. [Testing Patterns](#19-testing-patterns)
20. [Integration with Preprocessing Libraries](#20-integration-with-preprocessing-libraries)
21. [Known Limitations and Design Decisions](#21-known-limitations-and-design-decisions)

---

## 1. Project Overview

### Purpose

Python library for training and evaluating machine learning models on LOB (Limit Order Book) data for price movement prediction. Designed for HFT research with emphasis on:

- **Modularity**: Clean separation between data loading, models, training, and analysis
- **Scalability**: Memory-efficient streaming analysis for large datasets
- **Reproducibility**: Explicit seed management and configuration-driven experiments
- **Flexibility**: Multi-horizon label support, multiple model architectures

### Core Dependencies

```toml
[dependencies]
torch = ">=2.0"              # Deep learning framework
numpy = ">=1.24"             # Numerical operations
pandas = ">=2.0"             # Data manipulation
scikit-learn = ">=1.3"       # Classical ML, metrics
xgboost = ">=2.0"            # Gradient boosting baselines
scipy = ">=1.10"             # Statistical tests
statsmodels = ">=0.14"       # ADF stationarity tests
pyyaml = ">=6.0"             # Configuration files
dacite = ">=1.8"             # Dataclass from dict
tqdm = ">=4.65"              # Progress bars
```

### Research Paper Alignment

Implements methodologies from:
- **DeepLOB** (Zhang et al., 2019): CNN-LSTM architecture for LOB
- **HLOB** (Briola et al., 2024): Hierarchical LOB representations
- **TLOB**: Transformer-based LOB prediction
- **FI-2010**: Standard benchmark dataset methodology

---

## 2. Module Architecture

```
src/lobtrainer/
├── __init__.py                    # Public API exports, version
├── constants/
│   ├── __init__.py                # Module exports
│   └── feature_index.py           # Feature indices, label encoding constants
│
├── config/
│   ├── __init__.py                # Module exports
│   └── schema.py                  # ExperimentConfig, DataConfig, ModelConfig, etc.
│
├── data/
│   ├── __init__.py                # Module exports
│   ├── dataset.py                 # DayData, LOBFlatDataset, LOBSequenceDataset
│   └── transforms.py              # Normalization, augmentation, feature selection
│
├── models/
│   ├── __init__.py                # Module exports, create_model factory
│   ├── lstm.py                    # LSTMClassifier, GRUClassifier, LSTMConfig
│   └── baselines.py               # NaiveClassPrior, LogisticRegressionBaseline
│
├── training/
│   ├── __init__.py                # Module exports
│   ├── trainer.py                 # Trainer class, training loop, evaluation
│   ├── callbacks.py               # EarlyStopping, ModelCheckpoint, MetricLogger
│   ├── metrics.py                 # ClassificationMetrics, compute_classification_report
│   └── evaluation.py              # BaselineReport, evaluate_model, create_baseline_report
│
├── analysis/
│   ├── __init__.py                # Module exports
│   ├── streaming.py               # DayData, AlignedDayData, iter_days, iter_days_aligned
│   ├── data_loading.py            # load_split, load_split_aligned, align_features_with_labels
│   ├── data_overview.py           # Dataset overview statistics
│   ├── label_analysis.py          # Label distribution, autocorrelation
│   ├── signal_stats.py            # Stationarity tests, rolling statistics
│   ├── signal_correlations.py     # PCA, VIF, correlation matrices
│   ├── predictive_power.py        # Signal-label correlations, predictive decay
│   ├── temporal_dynamics.py       # Temporal autocorrelation analysis
│   ├── generalization.py          # Walk-forward validation
│   └── intraday_seasonality.py    # Regime-stratified analysis
│
├── utils/
│   ├── __init__.py                # Module exports
│   └── reproducibility.py         # set_seed, SeedManager, worker_init_fn
│
scripts/
├── train.py                       # Training CLI entry point
├── run_complete_streaming_analysis.py  # Full dataset analysis
└── evaluate_model.py              # Model evaluation

tests/
├── test_trainer.py                # Trainer unit tests
├── test_config.py                 # Configuration tests
├── test_transforms.py             # Transform tests
├── test_alignment.py              # Feature-label alignment tests
├── test_feature_index.py          # Feature index tests
└── ...                            # Additional test modules
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
                                                    │                  │
                                                    │ → (batch_seq,    │
                                                    │    batch_labels) │
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
Original Labels: {-1, 0, 1}    →    Dataset Shift: {0, 1, 2}    →    Metrics: Use shifted names
    (Down, Stable, Up)                 (for CrossEntropyLoss)           (Down, Stable, Up)
                                  
IMPORTANT: Labels are shifted by +1 in __getitem__ for PyTorch compatibility.
           Metrics use LABEL_SHIFTED_NAMES for correct display.
```

---

## 4. Constants and Feature Indices

### Feature Index Mapping (src/lobtrainer/constants/feature_index.py)

The feature vector has exactly **98 features** when all feature groups are enabled:

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
    # ... (order flow, size distribution, queue metrics, institutional)
    
    # Trading Signals (14 features: indices 84-97)
    TRUE_OFI = 84
    DEPTH_NORM_OFI = 85
    EXECUTED_PRESSURE = 86
    SIGNED_MP_DELTA_BPS = 87
    TRADE_ASYMMETRY = 88
    CANCEL_ASYMMETRY = 89
    FRAGILITY_SCORE = 90
    DEPTH_ASYMMETRY = 91
    BOOK_VALID = 92
    TIME_REGIME = 93
    MBO_READY = 94
    DT_SECONDS = 95
    INVALIDITY_DELTA = 96
    SCHEMA_VERSION = 97
```

### Label Encoding Constants

```python
# Original labels (from Rust pipeline)
LABEL_DOWN: Final[int] = -1     # Price moved down
LABEL_STABLE: Final[int] = 0    # Price within threshold
LABEL_UP: Final[int] = 1        # Price moved up
NUM_CLASSES: Final[int] = 3

LABEL_NAMES: Final[dict] = {
    LABEL_DOWN: "Down",
    LABEL_STABLE: "Stable",
    LABEL_UP: "Up",
}

# Shifted labels (for PyTorch CrossEntropyLoss)
SHIFTED_LABEL_DOWN: Final[int] = 0    # Was -1
SHIFTED_LABEL_STABLE: Final[int] = 1  # Was 0
SHIFTED_LABEL_UP: Final[int] = 2      # Was 1

SHIFTED_LABEL_NAMES: Final[dict] = {
    SHIFTED_LABEL_DOWN: "Down",
    SHIFTED_LABEL_STABLE: "Stable",
    SHIFTED_LABEL_UP: "Up",
}
```

### Feature Groups

```python
FEATURE_COUNT: Final[int] = 98  # Total features

# Group boundaries
RAW_LOB_FEATURES = list(range(0, 40))      # 40 features
DERIVED_FEATURES = list(range(40, 48))     # 8 features
MBO_FEATURES = list(range(48, 84))         # 36 features
SIGNAL_FEATURES = list(range(84, 98))      # 14 features

# Core signals for analysis
CORE_SIGNAL_INDICES = [84, 85, 86, 87, 88, 89, 90, 91]
```

---

## 5. Configuration System

### ExperimentConfig (src/lobtrainer/config/schema.py)

Root configuration object for experiments:

```python
@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    name: str                          # Experiment name
    description: str = ""              # Human-readable description
    tags: List[str] = field(default_factory=list)  # For filtering
    
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    
    output_dir: str = "outputs"
    log_level: str = "INFO"
    
    def to_dict(self) -> Dict[str, Any]: ...
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ExperimentConfig': ...
```

### DataConfig

```python
@dataclass
class DataConfig:
    """Data loading configuration."""
    data_dir: str                              # Path to exported dataset
    feature_count: int = 98                    # Expected features per timestep
    horizon_idx: Optional[int] = 0             # Which horizon for multi-horizon labels
    
    sequence: SequenceConfig = field(default_factory=SequenceConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    
    label_encoding: str = "categorical"        # "categorical" or "regression"
    num_classes: int = 3                       # For classification
    cache_in_memory: bool = True               # Cache loaded data
```

### ModelConfig

```python
@dataclass
class ModelConfig:
    """Model architecture configuration."""
    model_type: ModelType = ModelType.LSTM     # "lstm", "gru", "transformer"
    
    input_size: int = 98                       # Features per timestep
    hidden_size: int = 64                      # Hidden layer dimension
    num_layers: int = 2                        # Number of recurrent layers
    dropout: float = 0.2                       # Dropout rate
    num_classes: int = 3                       # Output classes
    
    # LSTM-specific
    lstm_bidirectional: bool = False           # Use bidirectional LSTM
    lstm_attention: bool = False               # Add attention mechanism
```

### TrainConfig

```python
@dataclass
class TrainConfig:
    """Training hyperparameters."""
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 100
    early_stopping_patience: int = 10
    gradient_clip_norm: Optional[float] = 1.0
    
    scheduler: str = "cosine"                  # "none", "cosine", "step", "plateau"
    num_workers: int = 0                       # DataLoader workers
    pin_memory: bool = False
    seed: int = 42
    
    use_class_weights: bool = True             # Weight loss by class frequency
```

### Loading Configuration

```python
from lobtrainer.config import load_config, save_config

# Load from YAML
config = load_config("configs/baseline_lstm.yaml")

# Save to YAML
save_config(config, "configs/experiment_v2.yaml")
```

---

## 6. Data Loading Pipeline

### Directory Structure Expected

```
data/exports/nvda_balanced/
├── train/
│   ├── 2025-02-03_sequences.npy    # [N_seq, 100, 98] float32
│   ├── 2025-02-03_labels.npy       # [N_seq, 5] int8 (multi-horizon)
│   ├── 2025-02-04_sequences.npy
│   ├── 2025-02-04_labels.npy
│   └── ...
├── val/
│   └── ...
├── test/
│   └── ...
└── dataset_manifest.json            # Metadata (horizons, feature count, etc.)
```

### Multi-Horizon Label Support

Labels can be single-horizon `(N,)` or multi-horizon `(N, H)`:

```python
# Single-horizon: labels.npy shape (17800,)
# Multi-horizon: labels.npy shape (17800, 5) for horizons [10, 20, 50, 100, 200]

# Accessing specific horizon
day.get_labels(0)      # First horizon (H=10), returns (N,)
day.get_labels(1)      # Second horizon (H=20), returns (N,)
day.get_labels(None)   # All horizons, returns (N, 5)
```

### DayData Class (src/lobtrainer/data/dataset.py)

Container for one day's data loaded from NumPy files:

```python
@dataclass
class DayData:
    """Container for a single day's data."""
    date: str                          # e.g., "2025-02-03"
    sequences: np.ndarray              # [N, 100, 98] sequences
    labels: np.ndarray                 # [N] or [N, H] labels
    metadata: Dict[str, Any]           # From metadata JSON
    
    @property
    def num_sequences(self) -> int:
        return len(self.sequences)
    
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

### Loading Functions (src/lobtrainer/analysis/data_loading.py)

```python
from lobtrainer.analysis.data_loading import load_split, load_split_aligned

# Load raw features and labels
data = load_split(Path("data/exports/nvda_balanced"), "train", horizon_idx=0)
# Returns: {'features': (N, 98), 'labels': (N,), 'n_days': int, 'dates': list}

# Load with correct alignment (for analysis)
data = load_split_aligned(Path("data/exports/nvda_balanced"), "train", horizon_idx=0)
# Returns: {'features': (N, 98), 'labels': (N,), 'day_boundaries': list, ...}
```

---

## 7. PyTorch Dataset Classes

### LOBSequenceDataset (src/lobtrainer/data/dataset.py)

Primary dataset for sequence models (LSTM, Transformer):

```python
class LOBSequenceDataset(Dataset):
    """
    PyTorch Dataset for LOB sequences.
    
    Each item is a tuple (sequence, label) where:
    - sequence: [seq_len, n_features] tensor
    - label: scalar tensor (shifted to {0, 1, 2})
    """
    
    def __init__(
        self,
        days: List[DayData],
        horizon_idx: Optional[int] = 0,
        feature_indices: Optional[List[int]] = None,
        transform: Optional[Callable] = None,
    ):
        self.days = days
        self.horizon_idx = horizon_idx
        self.feature_indices = feature_indices
        self.transform = transform
        
        # Build index map: global_idx -> (day_idx, local_idx)
        self._index_map = self._build_index_map()
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        day_idx, local_idx = self._index_map[idx]
        day = self.days[day_idx]
        
        # Get sequence [seq_len, n_features]
        sequence = day.sequences[local_idx].copy()
        
        # Get label (handle multi-horizon)
        if self._is_multi_horizon:
            label = day.get_labels(self.horizon_idx)[local_idx]
        else:
            label = day.labels[local_idx]
        
        # CRITICAL: Shift labels from {-1, 0, 1} to {0, 1, 2}
        label = label + 1
        
        # Feature selection
        if self.feature_indices is not None:
            sequence = sequence[:, self.feature_indices]
        
        # Apply transforms
        if self.transform is not None:
            sequence = self.transform(sequence)
        
        return (
            torch.from_numpy(sequence).float(),
            torch.tensor(label, dtype=torch.long),
        )
```

### LOBFlatDataset

For non-sequence models (XGBoost, MLP):

```python
class LOBFlatDataset(Dataset):
    """
    PyTorch Dataset for flat feature vectors.
    
    Each item is (features, label) where features is flattened:
    [seq_len × n_features] or just last timestep [n_features].
    """
    
    def __init__(
        self,
        days: List[DayData],
        flatten_mode: str = "last",  # "last", "flatten", "mean"
        horizon_idx: Optional[int] = 0,
        ...
    ): ...
```

### Creating DataLoaders

```python
from lobtrainer.data import create_dataloaders

loaders = create_dataloaders(
    data_dir="data/exports/nvda_balanced",
    batch_size=128,
    num_workers=0,
    use_sequences=True,  # LOBSequenceDataset vs LOBFlatDataset
    horizon_idx=0,       # Which horizon to use
)

train_loader = loaders['train']
val_loader = loaders['val']
test_loader = loaders['test']
```

---

## 8. Data Transforms and Normalization

### NormalizationStrategy (src/lobtrainer/data/transforms.py)

```python
class NormalizationStrategy(Enum):
    """Normalization strategies for LOB data."""
    NONE = "none"                        # No normalization
    ZSCORE_PER_FEATURE = "zscore"        # Independent z-score per feature
    ZSCORE_PER_DAY = "zscore_per_day"    # Z-score computed per day
    MINMAX = "minmax"                    # Min-max to [0, 1]
    ROBUST = "robust"                    # Median/IQR based (outlier robust)
```

### ZScoreNormalizer

```python
class ZScoreNormalizer:
    """
    Z-score normalization with optional feature exclusion.
    
    Args:
        eps: Small constant for numerical stability
        clip_value: Clip normalized values to [-clip_value, clip_value]
        exclude_features: Feature indices to skip (e.g., categorical)
    """
    
    def __init__(
        self,
        eps: float = 1e-8,
        clip_value: float = 10.0,
        exclude_features: Optional[List[int]] = None,
    ): ...
    
    def fit(self, data: np.ndarray) -> 'ZScoreNormalizer':
        """Compute mean and std from data [N, T, F] or [N, F]."""
        ...
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply normalization."""
        normalized = (data - self.mean) / (self.std + self.eps)
        if self.clip_value:
            normalized = np.clip(normalized, -self.clip_value, self.clip_value)
        return normalized
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform in one call."""
        return self.fit(data).transform(data)
```

### FeatureSelector

```python
class FeatureSelector:
    """Select subset of features by index or name."""
    
    def __init__(
        self,
        indices: Optional[List[int]] = None,
        names: Optional[List[str]] = None,
    ): ...
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Select features from last dimension."""
        return data[..., self.indices]
```

### Composing Transforms

```python
from lobtrainer.data.transforms import Compose, ZScoreNormalizer, FeatureSelector

transform = Compose([
    FeatureSelector(indices=list(range(84, 92))),  # Core signals only
    ZScoreNormalizer(clip_value=5.0),
])

dataset = LOBSequenceDataset(days, transform=transform)
```

---

## 9. Model Implementations

### LSTMClassifier (src/lobtrainer/models/lstm.py)

```python
class LSTMClassifier(nn.Module):
    """
    LSTM-based classifier for LOB sequences.
    
    Architecture:
        Input [B, T, F] → LSTM layers → (optional) Attention → FC → Output [B, C]
    
    Args:
        input_size: Features per timestep (98)
        hidden_size: LSTM hidden dimension (64)
        num_layers: Number of LSTM layers (2)
        num_classes: Output classes (3)
        dropout: Dropout rate (0.2)
        bidirectional: Use bidirectional LSTM
        attention: Add self-attention over sequence
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
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        
        if attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=lstm_output_size,
                num_heads=4,
                dropout=dropout,
                batch_first=True,
            )
        else:
            self.attention = None
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_size, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, features]
        
        Returns:
            logits: [batch, num_classes]
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: [B, T, H*D]
        
        if self.attention is not None:
            # Self-attention over sequence
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            # Use mean of attended sequence
            out = attn_out.mean(dim=1)
        else:
            # Use final hidden state
            if self.bidirectional:
                # Concatenate forward and backward final hidden states
                out = torch.cat([h_n[-2], h_n[-1]], dim=1)
            else:
                out = h_n[-1]
        
        out = self.dropout(out)
        logits = self.fc(out)
        
        return logits
```

### GRUClassifier

Similar architecture using GRU instead of LSTM:

```python
class GRUClassifier(nn.Module):
    """GRU-based classifier with optional attention."""
    ...
```

### Model Factory

```python
from lobtrainer.models import create_model

# From config
model = create_model(config.model)

# Direct creation
model = create_model(ModelConfig(
    model_type=ModelType.LSTM,
    input_size=98,
    hidden_size=64,
    num_layers=2,
    lstm_bidirectional=True,
    lstm_attention=True,
))
```

---

## 10. Training Infrastructure

### Trainer Class (src/lobtrainer/training/trainer.py)

Central class for training and evaluation:

```python
class Trainer:
    """
    Main training class for LOB models.
    
    Handles:
    - Model creation and initialization
    - Data loader setup
    - Training loop with validation
    - Learning rate scheduling
    - Gradient clipping
    - Checkpointing
    - Evaluation metrics
    
    Example:
        config = load_config("configs/lstm.yaml")
        trainer = Trainer(config, callbacks=[...])
        trainer.train()
        metrics = trainer.evaluate("test")
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        callbacks: Optional[List[Callback]] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config
        self.callbacks = CallbackList(callbacks or [])
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Lazy initialization
        self._model = None
        self._optimizer = None
        self._scheduler = None
        self._criterion = None
        self._train_loader = None
        self._val_loader = None
        self._test_loader = None
        
        # Training state
        self.state = TrainingState()
    
    def train(self) -> Dict[str, Any]:
        """
        Run full training loop.
        
        Returns:
            Dict with best_val_metric, best_epoch, total_epochs, history
        """
        if self._train_loader is None:
            self.setup()
        
        self.callbacks.on_train_start()
        
        for epoch in range(self.config.train.epochs):
            self.state.current_epoch = epoch
            self.callbacks.on_epoch_start(epoch)
            
            # Training phase
            train_metrics = self._train_epoch()
            
            # Validation phase
            val_metrics = {}
            if self._val_loader is not None:
                val_metrics = self._validate()
                self.callbacks.on_validation_end(epoch, val_metrics)
            
            # Update best metric
            val_loss = val_metrics.get('val_loss', float('inf'))
            if val_loss < self.state.best_val_metric:
                self.state.best_val_metric = val_loss
                self.state.best_epoch = epoch
            
            self.callbacks.on_epoch_end(epoch, {**train_metrics, **val_metrics})
            
            # Early stopping check
            if self.callbacks.should_stop:
                break
        
        self.callbacks.on_train_end()
        return {...}
    
    def evaluate(self, split: str = "test") -> ClassificationMetrics:
        """
        Evaluate model on a data split.
        
        Returns:
            ClassificationMetrics with accuracy, per-class F1, etc.
        """
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
    training_started: bool = False
    training_completed: bool = False
```

### Class Weighting for Imbalanced Data

The Trainer automatically computes class weights when `use_class_weights=True`:

```python
def _create_criterion(self) -> nn.Module:
    if not self.config.train.use_class_weights:
        return nn.CrossEntropyLoss()
    
    # Count class frequencies
    class_counts = torch.zeros(3)
    for _, labels in self._train_loader:
        for c in range(3):
            class_counts[c] += (labels == c).sum().item()
    
    # Inverse frequency weighting
    total = class_counts.sum()
    weights = total / (3.0 * class_counts.clamp(min=1))
    
    return nn.CrossEntropyLoss(weight=weights.to(self.device))
```

### Factory Function

```python
from lobtrainer import create_trainer

# From config file
trainer = create_trainer("configs/baseline_lstm.yaml")

# From config object
trainer = create_trainer(config, callbacks=[
    EarlyStopping(patience=10),
    ModelCheckpoint(save_dir="checkpoints/"),
])
```

---

## 11. Callback System

### Callback Base Class (src/lobtrainer/training/callbacks.py)

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

### EarlyStopping

```python
class EarlyStopping(Callback):
    """
    Stop training when metric stops improving.
    
    Args:
        patience: Epochs to wait before stopping
        metric: Metric to monitor (default: "val_loss")
        mode: "min" or "max"
        min_delta: Minimum change to qualify as improvement
        restore_best_weights: Restore model to best epoch on stop
    """
    
    def __init__(
        self,
        patience: int = 10,
        metric: str = "val_loss",
        mode: str = "min",
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
    ): ...
```

### ModelCheckpoint

```python
class ModelCheckpoint(Callback):
    """
    Save model checkpoints during training.
    
    Args:
        save_dir: Directory for checkpoints
        metric: Metric to monitor
        mode: "min" or "max"
        save_best_only: Only save when metric improves
        max_checkpoints: Maximum checkpoints to keep
    """
    
    def __init__(
        self,
        save_dir: Union[str, Path],
        metric: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
        max_checkpoints: int = 5,
    ): ...
```

### MetricLogger

```python
class MetricLogger(Callback):
    """
    Log metrics to file and optionally console.
    
    Args:
        log_every_n_batches: Log batch metrics every N batches
        log_to_file: Save metrics to JSON file
        log_file: Path to log file
    """
    
    def __init__(
        self,
        log_every_n_batches: Optional[int] = None,
        log_to_file: bool = True,
        log_file: Optional[Union[str, Path]] = None,
    ): ...
```

### CallbackList

```python
class CallbackList:
    """Container for multiple callbacks."""
    
    def __init__(self, callbacks: List[Callback] = None): ...
    
    @property
    def should_stop(self) -> bool:
        """Check if any callback requests early stopping."""
        return any(
            getattr(cb, 'should_stop', False)
            for cb in self.callbacks
        )
```

---

## 12. Evaluation Metrics

### ClassificationMetrics (src/lobtrainer/training/metrics.py)

```python
@dataclass
class PerClassMetrics:
    """Metrics for a single class."""
    label: int
    name: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass
class ClassificationMetrics:
    """Complete classification evaluation results."""
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    per_class: List[PerClassMetrics]
    confusion_matrix: np.ndarray
    
    def summary(self) -> str:
        """Format metrics as human-readable string."""
        ...
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        ...
```

### compute_classification_report

```python
def compute_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[int]] = None,
    shifted: bool = False,
) -> ClassificationMetrics:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: Label values (default: inferred from y_true)
        shifted: If True, use SHIFTED_LABEL_NAMES for display
    
    Returns:
        ClassificationMetrics with all computed metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support,
        confusion_matrix as sklearn_confusion_matrix,
    )
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    
    # Per-class metrics
    per_class = []
    for i, label in enumerate(labels):
        name = SHIFTED_LABEL_NAMES.get(label, str(label)) if shifted else LABEL_NAMES.get(label, str(label))
        per_class.append(PerClassMetrics(
            label=label,
            name=name,
            precision=float(precision[i]),
            recall=float(recall[i]),
            f1=float(f1[i]),
            support=int(support[i]),
        ))
    
    return ClassificationMetrics(
        accuracy=accuracy,
        macro_precision=float(precision.mean()),
        macro_recall=float(recall.mean()),
        macro_f1=float(f1.mean()),
        weighted_f1=float(np.average(f1, weights=support)),
        per_class=per_class,
        confusion_matrix=sklearn_confusion_matrix(y_true, y_pred, labels=labels),
    )
```

### Trading Metrics

```python
def compute_trading_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute trading-specific metrics.
    
    Returns:
        directional_accuracy: Correct prediction of Up/Down
        stable_accuracy: Correct prediction of Stable
        up_recall: True positive rate for Up
        down_recall: True positive rate for Down
    """
    ...
```

### Evaluation Framework (src/lobtrainer/training/evaluation.py)

The evaluation module provides structured comparison against baselines:

```python
from lobtrainer.training import (
    evaluate_model,
    evaluate_naive_baseline,
    create_baseline_report,
    full_evaluation,
    BaselineReport,
)
```

#### BaselineReport

```python
@dataclass
class BaselineReport:
    """
    Comprehensive report comparing model against baselines.
    
    Answers: "Is the model actually learning, or just exploiting
    label autocorrelation?"
    """
    
    model_name: str
    split: str
    n_samples: int
    
    # Metrics
    model_metrics: ClassificationMetrics
    class_prior_metrics: ClassificationMetrics
    previous_label_metrics: ClassificationMetrics
    
    # Derived (computed in __post_init__)
    beats_class_prior: bool           # model > always-predict-majority
    beats_previous_label: bool        # model > predict-previous-label
    improvement_over_prior: float     # percentage points
    improvement_over_previous: float  # percentage points
    
    def summary(self) -> str:
        """Human-readable summary."""
    
    def to_dict(self) -> Dict:
        """JSON-serializable dictionary."""
    
    def save(self, path: Path) -> None:
        """Save report to JSON file."""
```

#### Key Functions

```python
def evaluate_model(
    model: BaseModel,
    X: np.ndarray,
    y: np.ndarray,
    name: Optional[str] = None,
) -> ClassificationMetrics:
    """Evaluate any model with predict() method."""

def evaluate_naive_baseline(
    y_true: np.ndarray,
    split_name: str = "test",
) -> Dict[str, ClassificationMetrics]:
    """
    Evaluate naive baselines on labels.
    
    Returns:
        'class_prior': Always predict most common class
        'previous_label': Predict y[i-1] for y[i]
    """

def create_baseline_report(
    model: BaseModel,
    X: np.ndarray,
    y: np.ndarray,
    split: str = "test",
) -> BaselineReport:
    """Create comprehensive baseline comparison report."""

def full_evaluation(
    model: BaseModel,
    X: np.ndarray,
    y: np.ndarray,
    split: str = "test",
) -> Dict:
    """
    Full evaluation with all metrics:
    - Classification metrics
    - Trading metrics
    - Transition analysis
    - Baseline comparison
    """
```

---

## 13. Baseline Models

### NaiveClassPrior (src/lobtrainer/models/baselines.py)

```python
class NaiveClassPrior(BaseModel):
    """
    Baseline that always predicts the most common class.
    
    Establishes a floor for model performance.
    """
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'NaiveClassPrior':
        """Learn majority class from labels."""
        unique, counts = np.unique(y, return_counts=True)
        self.majority_class = unique[counts.argmax()]
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict majority class for all samples."""
        return np.full(len(X), self.majority_class)
```

### NaivePreviousLabel

```python
class NaivePreviousLabel(BaseModel):
    """
    Persistence baseline: predict the previous label.
    
    Exploits label autocorrelation. A model that cannot beat
    this baseline is not learning temporal patterns.
    
    Key insight: With label autocorrelation of ~0.44, this
    baseline achieves ~44% accuracy on balanced data.
    """
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict y[i] = y[i-1].
        
        Note: Requires labels to be passed in temporal order.
        First prediction defaults to 0 (Stable).
        """
        # Shift labels by 1
        return np.concatenate([[0], X[:-1].flatten()])
```

### LogisticRegressionBaseline

```python
class LogisticRegressionBaseline(BaseModel):
    """
    Logistic regression baseline on flattened features.
    
    Uses last timestep or mean across sequence.
    """
    
    def __init__(
        self,
        flatten_mode: str = "last",  # "last", "mean", "flatten"
        max_iter: int = 1000,
        class_weight: str = "balanced",
    ): ...
```

> **Note**: `BaselineReport` is documented in [Section 12: Evaluation Metrics](#12-evaluation-metrics) under the Evaluation Framework subsection.

---

## 14. Streaming Analysis Module

### Memory-Efficient Design (src/lobtrainer/analysis/streaming.py)

The streaming module processes large datasets day-by-day to stay under memory limits:

```python
# Memory budget: < 4GB for any dataset size
# One day of data: ~100K samples × 98 features × 4 bytes = ~40MB (float32)
```

### DayData (Streaming)

```python
@dataclass
class DayData:
    """Container for a single day's data (streaming version)."""
    date: str
    features: np.ndarray      # (N, 98) - loaded on demand
    labels: np.ndarray        # (M,) or (M, n_horizons)
    n_samples: int
    n_labels: int
    is_multi_horizon: bool = False
    num_horizons: int = 1
    
    def get_labels(self, horizon_idx: Optional[int] = 0) -> np.ndarray:
        """Get labels for a specific horizon."""
        if horizon_idx is None:
            return self.labels
        if not self.is_multi_horizon:
            return self.labels
        return self.labels[:, horizon_idx]
```

### AlignedDayData

```python
@dataclass
class AlignedDayData:
    """
    Container for ALIGNED data where features[i] corresponds to labels[i].
    
    Use this for signal-label correlation analysis.
    """
    date: str
    features: np.ndarray      # (N_labels, 98) - aligned with labels
    labels: np.ndarray        # (N_labels,) or (N_labels, n_horizons)
    n_pairs: int              # Number of aligned feature-label pairs
    is_multi_horizon: bool = False
    num_horizons: int = 1
    
    def get_labels(self, horizon_idx: Optional[int] = 0) -> np.ndarray: ...
```

### Streaming Iterators

```python
def iter_days(
    data_dir: Path,
    split: str,
    dtype: np.dtype = np.float32,
    mmap_mode: Optional[str] = None,
) -> Generator[DayData, None, None]:
    """
    Iterate over days, yielding one day at a time.
    
    Memory freed after each yield - never loads all data into memory.
    
    Example:
        for day in iter_days(data_dir, 'train'):
            labels = day.get_labels(0)  # First horizon
            process(day.features, labels)
    """
    ...


def iter_days_aligned(
    data_dir: Path,
    split: str,
    window_size: int = 100,
    stride: int = 10,
    dtype: np.dtype = np.float32,
) -> Generator[AlignedDayData, None, None]:
    """
    Iterate over days with CORRECT alignment.
    
    Critical for signal-label correlation analysis.
    
    Alignment formula:
        For label[i], the corresponding feature is at:
        feat_idx = i * stride + window_size - 1
    """
    ...
```

### Incremental Statistics (Welford's Algorithm)

```python
@dataclass
class RunningStats:
    """
    Welford's online algorithm for mean and variance.
    
    Numerically stable, single-pass, constant memory.
    """
    n: int = 0
    mean: float = 0.0
    M2: float = 0.0
    min_val: float = float('inf')
    max_val: float = float('-inf')
    
    def update(self, x: float) -> None:
        """Update with a single value."""
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
        self.min_val = min(self.min_val, x)
        self.max_val = max(self.max_val, x)
    
    @property
    def variance(self) -> float:
        return self.M2 / self.n if self.n > 0 else 0.0
    
    @property
    def std(self) -> float:
        return np.sqrt(self.variance)
```

---

## 15. Statistical Analysis Suite

### Complete Analysis Pipeline (scripts/run_complete_streaming_analysis.py)

Runs 9 analysis modules:

1. **Data Overview**: Sample counts, date ranges, data quality
2. **Label Analysis**: Distribution, autocorrelation, transition matrix
3. **Signal-Label Correlations**: Which features predict labels
4. **Signal Autocorrelations**: Feature persistence over time
5. **Predictive Decay**: How signal strength decays with horizon
6. **Walk-Forward Validation**: Out-of-sample signal stability
7. **Stationarity Tests**: ADF tests for feature stationarity
8. **PCA & VIF**: Dimensionality and multicollinearity
9. **Intraday Seasonality**: Regime-stratified analysis

### Running Analysis

```bash
cd lob-model-trainer
source .venv/bin/activate

python scripts/run_complete_streaming_analysis.py \
    --data-dir ../data/exports/nvda_balanced \
    --symbol NVDA
```

### Output Format

```json
{
  "symbol": "NVDA",
  "overview": {
    "total_days": 163,
    "total_samples": 17800000,
    "label_distribution": {
      "down_pct": 41.1,
      "stable_pct": 17.8,
      "up_pct": 41.1
    }
  },
  "signal_correlations": {
    "depth_norm_ofi": {
      "correlation": 0.2944,
      "is_significant": true
    },
    ...
  },
  "predictive_decay": {
    "depth_norm_ofi": {
      "horizon_0": 0.2944,
      "horizon_10": 0.1785,
      "horizon_50": 0.0609
    }
  },
  ...
}
```

### Key Analysis Modules

#### Stationarity Test (src/lobtrainer/analysis/signal_stats.py)

```python
def compute_stationarity_test(
    signal: np.ndarray,
    max_samples: int = 100000,
) -> Tuple[float, float, Dict[str, float], bool]:
    """
    Augmented Dickey-Fuller test for stationarity.
    
    Returns:
        adf_stat: ADF test statistic
        p_value: p-value
        critical_values: Dict at 1%, 5%, 10%
        is_stationary: True if p < 0.05
    """
    from statsmodels.tsa.stattools import adfuller
    result = adfuller(signal, autolag='AIC')
    return result[0], result[1], result[4], result[1] < 0.05
```

#### PCA Analysis (src/lobtrainer/analysis/signal_correlations.py)

```python
def compute_pca_analysis(
    features: np.ndarray,
    signal_indices: List[int] = None,
    n_components: Optional[int] = None,
) -> PCAResult:
    """
    PCA on signals to identify orthogonal factors.
    
    Returns:
        PCAResult with explained_variance_ratio, n_components_95, etc.
    """
    ...
```

#### VIF (Variance Inflation Factor)

```python
def compute_vif(
    features: np.ndarray,
    signal_indices: List[int] = None,
) -> List[VIFResult]:
    """
    Compute VIF for multicollinearity detection.
    
    VIF > 5: Moderate multicollinearity
    VIF > 10: Severe multicollinearity
    """
    ...
```

#### Intraday Seasonality (src/lobtrainer/analysis/intraday_seasonality.py)

```python
def run_intraday_seasonality_analysis(
    features: np.ndarray,
    labels: np.ndarray,
    signal_indices: Optional[Dict[str, int]] = None,
) -> IntradaySeasonalitySummary:
    """
    Analyze signal-label correlations by market regime.
    
    Regimes (from TIME_REGIME feature):
        0 = OPEN (9:30-9:45 ET): Highest volatility
        1 = EARLY (9:45-10:30 ET): Settling period
        2 = MIDDAY (10:30-15:30 ET): Most stable
        3 = CLOSE (15:30-16:00 ET): Position squaring
        4 = CLOSED (After hours): Filter these
    """
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
    
    Example:
        >>> from lobtrainer import set_seed
        >>> set_seed(42)  # Call at start of training
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    if deterministic_cudnn and torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

### SeedManager Context

```python
@dataclass
class SeedManager:
    """
    Context manager for reproducible code blocks.
    
    Example:
        >>> with SeedManager(42, restore_state=True):
        ...     x = torch.rand(3)  # Always same values
        >>> # After exiting, RNG state is restored
    """
    seed: int
    restore_state: bool = False
    
    def __enter__(self) -> 'SeedManager':
        if self.restore_state:
            self._saved_state = get_seed_state()
        set_seed(self.seed)
        return self
    
    def __exit__(self, *args) -> None:
        if self.restore_state:
            set_seed_state(self._saved_state)
```

### Worker Init Function

```python
def create_worker_init_fn(base_seed: int):
    """
    Create worker_init_fn for DataLoader.
    
    Each worker gets a unique but deterministic seed.
    
    Example:
        loader = DataLoader(
            dataset,
            num_workers=4,
            worker_init_fn=create_worker_init_fn(42),
        )
    """
    def init_fn(worker_id: int) -> None:
        worker_seed = base_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    return init_fn
```

---

## 17. Scripts and CLI

### Training Script (scripts/train.py)

```bash
# Basic training
python scripts/train.py --config configs/baseline_lstm.yaml

# With overrides
python scripts/train.py --config configs/baseline_lstm.yaml \
    --epochs 50 \
    --batch-size 256 \
    --learning-rate 0.0005 \
    --output-dir outputs/experiment1

# Resume from checkpoint
python scripts/train.py --config configs/baseline_lstm.yaml \
    --resume outputs/checkpoints/best.pt

# Evaluation only
python scripts/train.py --config configs/baseline_lstm.yaml \
    --eval-only --resume outputs/checkpoints/best.pt
```

### Complete Analysis Script

```bash
python scripts/run_complete_streaming_analysis.py \
    --data-dir ../data/exports/nvda_balanced \
    --symbol NVDA \
    --output-dir docs/
```

---

## 18. Configuration Examples

### Baseline LSTM (configs/baseline_lstm.yaml)

```yaml
name: baseline_lstm
description: |
  Baseline LSTM model for LOB price prediction.
  
tags:
  - baseline
  - lstm
  - nvda

data:
  data_dir: "../data/exports/nvda_balanced"
  feature_count: 98
  horizon_idx: 0
  
  sequence:
    window_size: 100
    stride: 10
  
  normalization:
    strategy: zscore_per_day
    eps: 1.0e-8
    clip_value: 10.0
    exclude_features:
      - 93  # TIME_REGIME (categorical)
  
  label_encoding: categorical
  num_classes: 3
  cache_in_memory: true

model:
  model_type: lstm
  input_size: 98
  hidden_size: 64
  num_layers: 2
  dropout: 0.2
  num_classes: 3
  lstm_bidirectional: false
  lstm_attention: false

train:
  batch_size: 128
  learning_rate: 1.0e-3
  weight_decay: 1.0e-5
  epochs: 100
  early_stopping_patience: 10
  gradient_clip_norm: 1.0
  scheduler: cosine
  num_workers: 0
  pin_memory: false
  seed: 42
  use_class_weights: true

output_dir: outputs/baseline_lstm
log_level: INFO
```

### LSTM with Attention (configs/lstm_attn_bidir_h20.yaml)

```yaml
name: lstm_attn_bidir_h20
description: |
  LSTM with attention, bidirectional, horizon 20.

data:
  data_dir: "../data/exports/nvda_balanced"
  horizon_idx: 1  # Second horizon (H=20)
  ...

model:
  model_type: lstm
  hidden_size: 64
  num_layers: 2
  lstm_bidirectional: true
  lstm_attention: true

train:
  epochs: 20
  early_stopping_patience: 5
  use_class_weights: true
```

---

## 19. Testing Patterns

### Unit Test Structure

```python
# tests/test_trainer.py

import pytest
import torch
import numpy as np
from lobtrainer.training import Trainer, TrainingState
from lobtrainer.config import ExperimentConfig

class TestTrainingState:
    def test_initial_state(self):
        state = TrainingState()
        assert state.current_epoch == 0
        assert state.best_val_metric == float('inf')
        assert not state.training_started

class TestTrainer:
    @pytest.fixture
    def mock_config(self, tmp_path):
        """Create minimal config for testing."""
        return ExperimentConfig(
            name="test",
            data=DataConfig(data_dir=str(tmp_path)),
            model=ModelConfig(input_size=10, hidden_size=8),
            train=TrainingConfig(epochs=2, batch_size=4),
        )
    
    def test_trainer_creation(self, mock_config):
        trainer = Trainer(mock_config)
        assert trainer.config.name == "test"
        assert trainer.device is not None
```

### Integration Test Pattern

```python
def test_full_training_pipeline(tmp_path, sample_data):
    """Test complete training flow with real data."""
    # Setup
    config = create_test_config(tmp_path, sample_data)
    trainer = create_trainer(config)
    
    # Train
    result = trainer.train()
    
    # Verify
    assert result['total_epochs'] > 0
    assert 'best_val_metric' in result
    assert Path(config.output_dir / 'checkpoints').exists()
    
    # Evaluate
    metrics = trainer.evaluate('test')
    assert metrics.accuracy >= 0.33  # Better than random
```

### Testing Reproducibility

```python
def test_reproducibility():
    """Verify same seed produces same results."""
    from lobtrainer import set_seed
    
    results = []
    for _ in range(2):
        set_seed(42)
        model = LSTMClassifier(input_size=10)
        x = torch.randn(4, 5, 10)
        out = model(x)
        results.append(out.detach().numpy())
    
    np.testing.assert_array_almost_equal(results[0], results[1])
```

---

## 20. Integration with Preprocessing Libraries

### Data Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    COMPLETE HFT DATA PIPELINE                                │
└─────────────────────────────────────────────────────────────────────────────┘

    Raw MBO Data            LOB State              Features + Labels
         │                     │                          │
         ▼                     ▼                          ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────┐
│ MBO-LOB         │───▶│ Feature         │───▶│ LOB-Model-Trainer   │
│ Reconstructor   │    │ Extractor       │    │                     │
│ (Rust)          │    │ (Rust)          │    │ (Python)            │
│                 │    │                 │    │                     │
│ .dbn.zst files  │    │ export_dataset  │    │ Train/evaluate      │
│ → LobState      │    │ → NumPy files   │    │ ML models           │
└─────────────────┘    └─────────────────┘    └─────────────────────┘
```

### Expected Input Format

The trainer expects data exported by `feature-extractor-MBO-LOB`:

```
data/exports/nvda_balanced/
├── train/
│   ├── {date}_sequences.npy   # [N_seq, window_size, n_features]
│   ├── {date}_labels.npy      # [N_seq] or [N_seq, n_horizons]
│   └── ...
├── val/
├── test/
└── dataset_manifest.json
```

### Metadata Schema

```json
{
  "schema_version": "2.2",
  "symbol": "NVDA",
  "date_range": ["2025-02-03", "2025-09-29"],
  "feature_count": 98,
  "horizons": [10, 20, 50, 100, 200],
  "num_horizons": 5,
  "label_config": {
    "threshold_strategy": {
      "type": "quantile",
      "target_proportion": 0.33
    }
  },
  "splits": {
    "train": {"days": 114, "sequences": 12460000},
    "val": {"days": 24, "sequences": 2670000},
    "test": {"days": 25, "sequences": 2670000}
  }
}
```

---

## 21. Known Limitations and Design Decisions

### Label Shift (+1)

Labels are shifted from `{-1, 0, 1}` to `{0, 1, 2}` in the dataset `__getitem__`:

```python
# In LOBSequenceDataset.__getitem__
label = label + 1  # {-1, 0, 1} → {0, 1, 2}
```

**Reason**: PyTorch's `CrossEntropyLoss` requires labels in `[0, num_classes)`.

**Impact**: Metrics must use `SHIFTED_LABEL_NAMES` for correct class names.

### Single-Horizon Training

While data supports multi-horizon labels `(N, H)`, the trainer currently trains on a single horizon:

```python
# In config
horizon_idx: 0  # Use first horizon (H=10)
```

**Future**: Multi-task learning across horizons.

### num_workers=0 Default

DataLoader workers are disabled by default:

```python
num_workers: int = 0
```

**Reason**: Avoids multiprocessing complexity and pickling issues. Still fast for typical dataset sizes.

### Memory-Efficient Streaming

Analysis uses streaming to handle large datasets:

```python
# Instead of loading all data:
all_data = load_all()  # ❌ Out of memory

# Use streaming:
for day in iter_days(data_dir, 'train'):  # ✅ ~40MB per day
    process(day)
```

**Trade-off**: Some analyses (like full dataset PCA) use sampling.

### Feature Index Hardcoding

Feature indices are hardcoded for 98-feature schema:

```python
TRUE_OFI = 84
DEPTH_NORM_OFI = 85
```

**Constraint**: Requires matching feature extractor configuration.

### Time Regime Exclusion

Time regime (index 93) is excluded from normalization:

```yaml
exclude_features:
  - 93  # TIME_REGIME (categorical)
```

**Reason**: It's categorical `{0, 1, 2, 3, 4}`, not continuous.

---

## Quick Reference

### Imports

```python
# Core training
from lobtrainer import Trainer, create_trainer, set_seed
from lobtrainer.config import load_config, ExperimentConfig

# Models
from lobtrainer.models import LSTMClassifier, GRUClassifier, create_model

# Data
from lobtrainer.data import LOBSequenceDataset, create_dataloaders

# Metrics
from lobtrainer.training import compute_classification_report

# Analysis
from lobtrainer.analysis.streaming import iter_days, iter_days_aligned

# Constants
from lobtrainer.constants import (
    FeatureIndex, FEATURE_COUNT,
    LABEL_DOWN, LABEL_STABLE, LABEL_UP,
    SHIFTED_LABEL_NAMES,
)
```

### Feature Counts

| Configuration | Count | Description |
|---------------|-------|-------------|
| Raw LOB | 40 | 10 levels × 4 (prices + sizes) |
| + Derived | 48 | + 8 derived features |
| + MBO | 84 | + 36 MBO features |
| + Signals | 98 | + 14 trading signals |

### Key Defaults

| Parameter | Default | Notes |
|-----------|---------|-------|
| Sequence length | 100 | Timesteps per sample |
| Stride | 10 | Between sequences |
| Batch size | 128 | Training batch |
| Hidden size | 64 | LSTM hidden dimension |
| Learning rate | 1e-3 | AdamW optimizer |
| Epochs | 100 | With early stopping |
| Patience | 10 | Early stopping patience |
| Seed | 42 | Reproducibility |

### Label Encoding

| Original | Shifted | Meaning |
|----------|---------|---------|
| -1 | 0 | Down (price decreased) |
| 0 | 1 | Stable (within threshold) |
| 1 | 2 | Up (price increased) |

---

*Last updated: December 24, 2025*
*Version: 0.2.0*

