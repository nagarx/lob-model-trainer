"""
Experiment result container for structured experiment tracking.

ExperimentResult encapsulates all outputs from a training run:
- Configuration used
- Final metrics (train, val, test)
- Training history
- Model checkpoint path
- Timing and resource usage

Design principles (RULE.md):
- Immutable after creation
- Self-contained (all info to reproduce)
- Serializable to JSON for persistence
- Comparable across experiments

Reference:
- López de Prado (2018), Ch. 11: "Backtesting on Synthetic Data"
  Emphasizes tracking all experiment metadata for reproducibility.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import hashlib


@dataclass
class ExperimentMetrics:
    """
    Metrics from a single evaluation (train, val, or test).
    
    Contains both generic classification metrics and strategy-specific ones.
    """
    
    # Core metrics
    accuracy: float = 0.0
    """Overall accuracy."""
    
    loss: float = 0.0
    """Average loss value."""
    
    macro_f1: float = 0.0
    """Macro-averaged F1 score."""
    
    macro_precision: float = 0.0
    """Macro-averaged precision."""
    
    macro_recall: float = 0.0
    """Macro-averaged recall."""
    
    # Per-class metrics (indexed by class name)
    per_class_precision: Dict[str, float] = field(default_factory=dict)
    per_class_recall: Dict[str, float] = field(default_factory=dict)
    per_class_f1: Dict[str, float] = field(default_factory=dict)
    
    # Strategy-specific metrics
    directional_accuracy: float = 0.0
    """Accuracy on directional predictions (non-stable)."""
    
    signal_rate: float = 0.0
    """Fraction of directional predictions."""
    
    # Triple Barrier specific
    predicted_trade_win_rate: float = 0.0
    """Win rate of predicted trades (Triple Barrier)."""
    
    decisive_prediction_rate: float = 0.0
    """Rate of non-timeout predictions (Triple Barrier)."""
    
    # Additional metrics
    extra_metrics: Dict[str, float] = field(default_factory=dict)
    """Any additional metrics not covered above."""
    
    @classmethod
    def from_classification_metrics(cls, metrics) -> "ExperimentMetrics":
        """Create from ClassificationMetrics object."""
        result = cls(
            accuracy=metrics.accuracy,
            loss=metrics.loss,
            macro_f1=metrics.macro_f1,
            macro_precision=metrics.macro_precision,
            macro_recall=metrics.macro_recall,
        )
        
        # Per-class metrics
        for class_id, precision in metrics.per_class_precision.items():
            class_name = metrics.class_names[class_id] if class_id < len(metrics.class_names) else f"class_{class_id}"
            result.per_class_precision[class_name] = precision
        
        for class_id, recall in metrics.per_class_recall.items():
            class_name = metrics.class_names[class_id] if class_id < len(metrics.class_names) else f"class_{class_id}"
            result.per_class_recall[class_name] = recall
        
        for class_id, f1 in metrics.per_class_f1.items():
            class_name = metrics.class_names[class_id] if class_id < len(metrics.class_names) else f"class_{class_id}"
            result.per_class_f1[class_name] = f1
        
        # Strategy-specific
        strategy_metrics = metrics.strategy_metrics
        result.directional_accuracy = strategy_metrics.get('directional_accuracy', 0.0)
        result.signal_rate = strategy_metrics.get('signal_rate', 0.0)
        result.predicted_trade_win_rate = strategy_metrics.get('predicted_trade_win_rate', 0.0)
        result.decisive_prediction_rate = strategy_metrics.get('decisive_prediction_rate', 0.0)
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'accuracy': self.accuracy,
            'loss': self.loss,
            'macro_f1': self.macro_f1,
            'macro_precision': self.macro_precision,
            'macro_recall': self.macro_recall,
            'per_class_precision': self.per_class_precision,
            'per_class_recall': self.per_class_recall,
            'per_class_f1': self.per_class_f1,
            'directional_accuracy': self.directional_accuracy,
            'signal_rate': self.signal_rate,
            'predicted_trade_win_rate': self.predicted_trade_win_rate,
            'decisive_prediction_rate': self.decisive_prediction_rate,
            'extra_metrics': self.extra_metrics,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentMetrics":
        """Create from dictionary."""
        return cls(
            accuracy=data.get('accuracy', 0.0),
            loss=data.get('loss', 0.0),
            macro_f1=data.get('macro_f1', 0.0),
            macro_precision=data.get('macro_precision', 0.0),
            macro_recall=data.get('macro_recall', 0.0),
            per_class_precision=data.get('per_class_precision', {}),
            per_class_recall=data.get('per_class_recall', {}),
            per_class_f1=data.get('per_class_f1', {}),
            directional_accuracy=data.get('directional_accuracy', 0.0),
            signal_rate=data.get('signal_rate', 0.0),
            predicted_trade_win_rate=data.get('predicted_trade_win_rate', 0.0),
            decisive_prediction_rate=data.get('decisive_prediction_rate', 0.0),
            extra_metrics=data.get('extra_metrics', {}),
        )


@dataclass
class ExperimentResult:
    """
    Complete result from a training experiment.
    
    Captures everything needed to:
    - Reproduce the experiment
    - Compare with other experiments
    - Analyze training dynamics
    - Load the trained model
    
    Attributes:
        experiment_id: Unique identifier (auto-generated if not provided).
        name: Human-readable experiment name.
        description: Detailed description.
        config: Full experiment configuration (serialized).
        train_metrics: Metrics from training set (final epoch).
        val_metrics: Metrics from validation set (best epoch).
        test_metrics: Metrics from test set (using best model).
        training_history: Per-epoch metrics.
        best_epoch: Epoch with best validation metric.
        total_epochs: Total epochs trained (may differ due to early stopping).
        training_time_seconds: Total training wall-clock time.
        checkpoint_path: Path to saved model checkpoint.
        created_at: Timestamp when result was created.
        tags: List of tags for filtering/grouping.
    
    Example:
        >>> # After training
        >>> result = ExperimentResult.from_trainer(trainer, test_metrics)
        >>> result.save("experiments/my_experiment.json")
        >>> 
        >>> # Later
        >>> loaded = ExperimentResult.load("experiments/my_experiment.json")
    """
    
    # Identity
    experiment_id: str = ""
    """Unique identifier for this experiment."""
    
    name: str = ""
    """Human-readable experiment name (from config)."""
    
    description: str = ""
    """Experiment description."""
    
    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    """Full experiment configuration."""
    
    # Metrics
    train_metrics: Optional[ExperimentMetrics] = None
    """Metrics from training set (final epoch)."""
    
    val_metrics: Optional[ExperimentMetrics] = None
    """Metrics from validation set (best epoch)."""
    
    test_metrics: Optional[ExperimentMetrics] = None
    """Metrics from test set (using best model)."""
    
    # Training dynamics
    training_history: List[Dict[str, float]] = field(default_factory=list)
    """Per-epoch metrics history."""
    
    best_epoch: int = 0
    """Epoch with best validation metric."""
    
    total_epochs: int = 0
    """Total epochs trained."""
    
    training_time_seconds: float = 0.0
    """Total training wall-clock time."""
    
    # Artifacts
    checkpoint_path: Optional[str] = None
    """Path to saved model checkpoint."""
    
    output_dir: str = ""
    """Directory containing all outputs."""
    
    # Metadata
    created_at: str = ""
    """ISO timestamp when result was created."""
    
    tags: List[str] = field(default_factory=list)
    """Tags for filtering/grouping."""
    
    # Model info
    model_type: str = ""
    """Model architecture type."""
    
    model_params: int = 0
    """Total model parameters."""
    
    # Data info
    labeling_strategy: str = ""
    """Labeling strategy used."""
    
    num_train_samples: int = 0
    """Number of training samples."""
    
    num_val_samples: int = 0
    """Number of validation samples."""
    
    num_test_samples: int = 0
    """Number of test samples."""
    
    def __post_init__(self):
        """Generate experiment ID if not provided."""
        if not self.experiment_id:
            self.experiment_id = self._generate_id()
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
    
    def _generate_id(self) -> str:
        """Generate unique experiment ID from config hash + timestamp."""
        config_str = json.dumps(self.config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.name}_{timestamp}_{config_hash}"
    
    @classmethod
    def from_trainer(
        cls,
        trainer,
        test_metrics=None,
        training_time_seconds: float = 0.0,
    ) -> "ExperimentResult":
        """
        Create ExperimentResult from a trained Trainer instance.
        
        Args:
            trainer: Trained Trainer instance.
            test_metrics: Optional ClassificationMetrics from test evaluation.
            training_time_seconds: Total training time.
        
        Returns:
            ExperimentResult with all captured data.
        """
        config = trainer.config
        
        # Extract model info
        model_params = sum(p.numel() for p in trainer.model.parameters())
        
        # Build result
        result = cls(
            name=config.name,
            description=config.description,
            config=config.to_dict(),
            best_epoch=trainer.state.best_epoch,
            total_epochs=trainer.state.current_epoch + 1,
            training_time_seconds=training_time_seconds,
            training_history=trainer.state.history,
            output_dir=str(trainer.output_dir),
            tags=config.tags,
            model_type=config.model.model_type.value,
            model_params=model_params,
            labeling_strategy=config.data.labeling_strategy.value,
        )
        
        # Checkpoint path
        best_checkpoint = trainer.output_dir / 'checkpoints' / 'best.pt'
        if best_checkpoint.exists():
            result.checkpoint_path = str(best_checkpoint)
        
        # Test metrics
        if test_metrics is not None:
            result.test_metrics = ExperimentMetrics.from_classification_metrics(test_metrics)
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'experiment_id': self.experiment_id,
            'name': self.name,
            'description': self.description,
            'config': self.config,
            'train_metrics': self.train_metrics.to_dict() if self.train_metrics else None,
            'val_metrics': self.val_metrics.to_dict() if self.val_metrics else None,
            'test_metrics': self.test_metrics.to_dict() if self.test_metrics else None,
            'training_history': self.training_history,
            'best_epoch': self.best_epoch,
            'total_epochs': self.total_epochs,
            'training_time_seconds': self.training_time_seconds,
            'checkpoint_path': self.checkpoint_path,
            'output_dir': self.output_dir,
            'created_at': self.created_at,
            'tags': self.tags,
            'model_type': self.model_type,
            'model_params': self.model_params,
            'labeling_strategy': self.labeling_strategy,
            'num_train_samples': self.num_train_samples,
            'num_val_samples': self.num_val_samples,
            'num_test_samples': self.num_test_samples,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentResult":
        """Create from dictionary."""
        return cls(
            experiment_id=data.get('experiment_id', ''),
            name=data.get('name', ''),
            description=data.get('description', ''),
            config=data.get('config', {}),
            train_metrics=ExperimentMetrics.from_dict(data['train_metrics']) if data.get('train_metrics') else None,
            val_metrics=ExperimentMetrics.from_dict(data['val_metrics']) if data.get('val_metrics') else None,
            test_metrics=ExperimentMetrics.from_dict(data['test_metrics']) if data.get('test_metrics') else None,
            training_history=data.get('training_history', []),
            best_epoch=data.get('best_epoch', 0),
            total_epochs=data.get('total_epochs', 0),
            training_time_seconds=data.get('training_time_seconds', 0.0),
            checkpoint_path=data.get('checkpoint_path'),
            output_dir=data.get('output_dir', ''),
            created_at=data.get('created_at', ''),
            tags=data.get('tags', []),
            model_type=data.get('model_type', ''),
            model_params=data.get('model_params', 0),
            labeling_strategy=data.get('labeling_strategy', ''),
            num_train_samples=data.get('num_train_samples', 0),
            num_val_samples=data.get('num_val_samples', 0),
            num_test_samples=data.get('num_test_samples', 0),
        )
    
    def save(self, path: Union[str, Path]) -> None:
        """Save result to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "ExperimentResult":
        """Load result from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Experiment: {self.name}",
            f"ID: {self.experiment_id}",
            f"Model: {self.model_type} ({self.model_params:,} params)",
            f"Strategy: {self.labeling_strategy}",
            f"",
            f"Training: {self.total_epochs} epochs ({self.training_time_seconds:.1f}s)",
            f"Best epoch: {self.best_epoch}",
        ]
        
        if self.test_metrics:
            lines.extend([
                "",
                "Test Metrics:",
                f"  Accuracy: {self.test_metrics.accuracy:.4f}",
                f"  Macro F1: {self.test_metrics.macro_f1:.4f}",
                f"  Directional Accuracy: {self.test_metrics.directional_accuracy:.4f}",
                f"  Signal Rate: {self.test_metrics.signal_rate:.4f}",
            ])
            
            if self.labeling_strategy == 'triple_barrier':
                lines.extend([
                    f"  Predicted Trade Win Rate: {self.test_metrics.predicted_trade_win_rate:.4f}",
                    f"  Decisive Prediction Rate: {self.test_metrics.decisive_prediction_rate:.4f}",
                ])
        
        return "\n".join(lines)
