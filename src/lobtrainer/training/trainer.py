"""
Core training infrastructure for LOB models.

The Trainer class provides a complete training loop with:
- Configuration-driven setup
- Reproducibility via seed management
- Extensibility via callbacks
- Early stopping and checkpointing
- Validation and metrics tracking

Design principles (RULE.md):
- Configuration-driven: All parameters via ExperimentConfig
- Deterministic: Same seed produces identical results
- Modular: Callbacks for extensibility without modifying core
- Validated: Explicit checks for data integrity and config consistency

Usage:
    >>> from lobtrainer.training import Trainer
    >>> from lobtrainer.config import load_config
    >>> 
    >>> config = load_config("configs/baseline_lstm.yaml")
    >>> trainer = Trainer(config)
    >>> trainer.train()
    >>> metrics = trainer.evaluate("test")
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    StepLR,
)
from torch.utils.data import DataLoader

from lobtrainer.config import ExperimentConfig, ModelType, TaskType, LossType
from lobtrainer.data import (
    LOBSequenceDataset,
    LOBFlatDataset,
    load_split_data,
    create_dataloaders,
)
from lobtrainer.training.callbacks import (
    Callback,
    CallbackList,
    EarlyStopping,
    ModelCheckpoint,
    MetricLogger,
)
from lobtrainer.training.metrics import (
    compute_accuracy,
    ClassificationMetrics,
)
from lobtrainer.utils.reproducibility import set_seed, create_worker_init_fn

logger = logging.getLogger(__name__)


# =============================================================================
# Training State
# =============================================================================


@dataclass
class TrainingState:
    """
    Mutable state during training.
    
    Separated from Trainer for clarity and potential checkpointing.
    """
    
    current_epoch: int = 0
    global_step: int = 0
    best_val_metric: float = float('inf')
    best_epoch: int = 0
    training_started: bool = False
    training_completed: bool = False
    history: List[Dict[str, float]] = field(default_factory=list)


# =============================================================================
# Core Trainer Class
# =============================================================================


class Trainer:
    """
    Trainer for LOB prediction models.
    
    Provides a complete training pipeline with:
    - Data loading and preprocessing
    - Model training with gradient clipping
    - Validation and metric computation
    - Early stopping and checkpointing (via callbacks)
    - Learning rate scheduling
    
    Args:
        config: ExperimentConfig with all settings
        model: Optional pre-built model (if None, created from config)
        callbacks: Optional list of callbacks
        device: Device to train on (auto-detected if None)
    
    Example:
        >>> config = ExperimentConfig.from_yaml("configs/lstm.yaml")
        >>> trainer = Trainer(config)
        >>> trainer.train()
        >>> test_metrics = trainer.evaluate("test")
        >>> print(test_metrics.summary())
    
    Design notes:
        - Model is created lazily (on first train() or evaluate())
        - Data loaders are created lazily for memory efficiency
        - Callbacks are called at appropriate lifecycle points
        - All random operations use the configured seed
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        model: Optional[nn.Module] = None,
        callbacks: Optional[List[Callback]] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config
        self._model = model
        
        # Setup callbacks
        self.callbacks = CallbackList(callbacks or [])
        self.callbacks.set_trainer(self)
        
        # Auto-detect device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        # Initialize state
        self.state = TrainingState()
        
        # Lazy initialization
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._scheduler: Optional[Any] = None
        self._train_loader: Optional[DataLoader] = None
        self._val_loader: Optional[DataLoader] = None
        self._test_loader: Optional[DataLoader] = None
        self._criterion: Optional[nn.Module] = None
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"Trainer initialized: device={self.device}, "
            f"output_dir={self.output_dir}"
        )
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def model(self) -> nn.Module:
        """Get model (creates if not exists)."""
        if self._model is None:
            self._model = self._create_model()
        return self._model
    
    @model.setter
    def model(self, model: nn.Module) -> None:
        """Set model."""
        self._model = model
    
    @property
    def optimizer(self) -> torch.optim.Optimizer:
        """Get optimizer (creates if not exists)."""
        if self._optimizer is None:
            self._optimizer = self._create_optimizer()
        return self._optimizer
    
    @property
    def scheduler(self) -> Optional[Any]:
        """Get learning rate scheduler."""
        return self._scheduler
    
    @property
    def criterion(self) -> nn.Module:
        """Get loss function."""
        if self._criterion is None:
            self._criterion = self._create_criterion()
        return self._criterion
    
    # =========================================================================
    # Setup Methods
    # =========================================================================
    
    def _create_model(self) -> nn.Module:
        """
        Create model from configuration.
        
        Override this method to add custom model types.
        """
        from lobtrainer.models import create_model
        
        model = create_model(self.config.model)
        model = model.to(self.device)
        
        # Log model info
        num_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            f"Created model: {self.config.model.model_type.value}, "
            f"{num_params:,} params ({trainable_params:,} trainable)"
        )
        
        return model
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer from configuration."""
        params = self.model.parameters()
        cfg = self.config.train
        
        # Use AdamW by default (better weight decay handling)
        optimizer = AdamW(
            params,
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        
        logger.debug(f"Created optimizer: AdamW(lr={cfg.learning_rate}, wd={cfg.weight_decay})")
        return optimizer
    
    def _create_scheduler(self, optimizer: torch.optim.Optimizer) -> Optional[Any]:
        """Create learning rate scheduler from configuration."""
        cfg = self.config.train
        
        if cfg.scheduler == 'none':
            return None
        elif cfg.scheduler == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)
        elif cfg.scheduler == 'step':
            scheduler = StepLR(
                optimizer,
                step_size=cfg.scheduler_step_size,
                gamma=cfg.scheduler_gamma,
            )
        elif cfg.scheduler == 'plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                patience=5,
                factor=0.5,
            )
        else:
            logger.warning(f"Unknown scheduler '{cfg.scheduler}', using none")
            return None
        
        logger.debug(f"Created scheduler: {cfg.scheduler}")
        return scheduler
    
    def _create_criterion(self) -> nn.Module:
        """
        Create loss function based on configuration.
        
        Supports:
        - cross_entropy: Standard unweighted CE
        - weighted_ce: CE with inverse-frequency class weights
        - focal: Focal loss for handling class imbalance
        
        For binary_signal task, uses 2 classes instead of 3.
        """
        from lobtrainer.training.loss import FocalLoss, BinaryFocalLoss
        
        cfg = self.config.train
        loss_type = getattr(cfg, 'loss_type', LossType.WEIGHTED_CE)
        task_type = getattr(cfg, 'task_type', TaskType.MULTICLASS)
        
        # Determine number of classes based on task type
        if task_type == TaskType.BINARY_SIGNAL:
            num_classes = 2
            class_names = ["NoSignal", "Signal"]
        else:
            num_classes = 3
            class_names = ["Down", "Stable", "Up"]
        
        # For unweighted cross-entropy, return immediately
        if loss_type == LossType.CROSS_ENTROPY:
            logger.info(f"Using unweighted CrossEntropyLoss ({num_classes} classes)")
            return nn.CrossEntropyLoss()
        
        # Compute class weights from training data for weighted losses
        class_counts = None
        try:
            if self._train_loader is not None:
                class_counts = torch.zeros(num_classes)
                for _, labels in self._train_loader:
                    for c in range(num_classes):
                        class_counts[c] += (labels == c).sum().item()
                
                total = class_counts.sum()
                if total > 0:
                    # Log class distribution
                    for i, name in enumerate(class_names):
                        pct = class_counts[i] / total * 100
                        logger.info(f"  Class {name}: {int(class_counts[i]):,} ({pct:.1f}%)")
        except Exception as e:
            logger.warning(f"Could not compute class counts: {e}")
            class_counts = None
        
        # Create appropriate loss function
        if loss_type == LossType.FOCAL:
            gamma = getattr(cfg, 'focal_gamma', 2.0)
            
            # For binary OR multi-class with num_classes output logits, use FocalLoss
            # (not BinaryFocalLoss which expects single logit for sigmoid)
            # Note: Our model outputs [batch, num_classes] logits for both binary and multiclass
            weights = None
            if class_counts is not None:
                total = class_counts.sum()
                if total > 0:
                    # Inverse frequency weighting
                    weights = total / (num_classes * class_counts.clamp(min=1))
                    weights = weights.to(self.device)
            
            logger.info(f"Using FocalLoss(gamma={gamma}, alpha={weights})")
            return FocalLoss(gamma=gamma, alpha=weights)
        
        else:  # LossType.WEIGHTED_CE
            # Compute class weights from training data
            if class_counts is not None:
                total = class_counts.sum()
                if total > 0:
                    # Inverse frequency weighting
                    weights = total / (float(num_classes) * class_counts.clamp(min=1))
                    weights = weights.to(self.device)
                    
                    weight_str = ", ".join(
                        f"{name}={weights[i]:.2f}" for i, name in enumerate(class_names)
                    )
                    logger.info(f"Using class weights: {weight_str}")
                    return nn.CrossEntropyLoss(weight=weights)
        
        # Fallback to unweighted loss
        logger.info("Using unweighted CrossEntropyLoss (fallback)")
        return nn.CrossEntropyLoss()
    
    def _create_dataloaders(self) -> Dict[str, DataLoader]:
        """Create data loaders for all splits with proper task configuration."""
        cfg_data = self.config.data
        cfg_train = self.config.train
        cfg_model = self.config.model
        
        # Note: num_workers > 0 requires picklable worker_init_fn
        # For simplicity, we use num_workers=0 which is still fast for our dataset size
        # and avoids multiprocessing complexity
        num_workers = 0 if cfg_train.num_workers > 0 else 0
        
        # Load using the existing infrastructure
        # horizon_idx selects which prediction horizon to use (0=10 steps, 1=20, etc.)
        horizon_idx = getattr(cfg_data, 'horizon_idx', 0)
        
        # Feature selection for DeepLOB benchmark mode
        # DeepLOB in benchmark mode uses only the first 40 LOB features
        feature_indices = None
        from lobtrainer.config import ModelType, DeepLOBMode
        if cfg_model.model_type == ModelType.DEEPLOB:
            deeplob_mode = getattr(cfg_model, 'deeplob_mode', DeepLOBMode.BENCHMARK)
            if deeplob_mode == DeepLOBMode.BENCHMARK:
                # First 40 features are LOB: bid_prices(10), ask_prices(10), 
                # bid_sizes(10), ask_sizes(10) in GROUPED layout
                from lobtrainer.constants import LOB_FEATURE_COUNT
                feature_indices = list(range(LOB_FEATURE_COUNT))  # [0, 1, ..., 39]
                logger.info(
                    f"DeepLOB benchmark mode: selecting first {LOB_FEATURE_COUNT} LOB features"
                )
        
        # Determine label transform based on task type
        label_transform = None
        num_classes = 3
        task_type = getattr(cfg_train, 'task_type', TaskType.MULTICLASS)
        
        if task_type == TaskType.BINARY_SIGNAL:
            from lobtrainer.data import BinaryLabelTransform
            label_transform = BinaryLabelTransform()  # Down,Up → Signal; Stable → NoSignal
            num_classes = 2
            logger.info("Binary signal detection mode: converting 3-class → 2-class labels")
        
        # Create dataloaders with proper configuration
        loaders = self._create_dataloaders_with_transform(
            data_dir=cfg_data.data_dir,
            batch_size=cfg_train.batch_size,
            num_workers=num_workers,
            pin_memory=cfg_train.pin_memory and torch.cuda.is_available(),
            horizon_idx=horizon_idx,
            feature_indices=feature_indices,
            label_transform=label_transform,
            num_classes=num_classes,
        )
        
        return loaders
    
    def _create_dataloaders_with_transform(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        horizon_idx: int,
        feature_indices: Optional[List[int]],
        label_transform: Optional[callable],
        num_classes: int,
    ) -> Dict[str, DataLoader]:
        """Create dataloaders with custom label transform support."""
        loaders = {}
        
        for split in ["train", "val", "test"]:
            try:
                days = load_split_data(data_dir, split, validate=True)
            except FileNotFoundError:
                logger.info(f"Split '{split}' not found, skipping")
                continue
            
            # Create dataset with label transform
            dataset = LOBSequenceDataset(
                days,
                transform=None,  # No feature transform for now
                feature_indices=feature_indices,
                horizon_idx=horizon_idx,
                label_transform=label_transform,
                num_classes=num_classes,
            )
            
            loaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == "train"),
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=(split == "train"),
            )
        
        return loaders
    
    def setup(self) -> None:
        """
        Full setup: model, optimizer, scheduler, data loaders.
        
        Call this explicitly if you want to setup before training,
        otherwise it's called automatically by train().
        """
        # Set seed for reproducibility
        set_seed(self.config.train.seed)
        
        # Create components (lazy properties)
        _ = self.model
        _ = self.optimizer
        self._scheduler = self._create_scheduler(self.optimizer)
        
        # Create data loaders BEFORE criterion (criterion needs class counts)
        loaders = self._create_dataloaders()
        self._train_loader = loaders.get('train')
        self._val_loader = loaders.get('val')
        self._test_loader = loaders.get('test')
        
        if self._train_loader is None:
            raise ValueError("No training data found")
        
        # Create criterion AFTER data loaders (needs class counts for weights)
        _ = self.criterion
        
        logger.info(
            f"Setup complete: train={len(self._train_loader)} batches, "
            f"val={len(self._val_loader) if self._val_loader else 0} batches"
        )
    
    # =========================================================================
    # Training Loop
    # =========================================================================
    
    def train(self) -> Dict[str, Any]:
        """
        Run full training loop.
        
        Returns:
            Dict with final metrics and training history
        
        Raises:
            ValueError: If no training data available
        """
        # Setup if not already done
        if self._train_loader is None:
            self.setup()
        
        cfg = self.config.train
        self.state.training_started = True
        
        # Notify callbacks
        self.callbacks.on_train_start()
        
        logger.info(f"Starting training for {cfg.epochs} epochs")
        start_time = time.time()
        
        try:
            for epoch in range(cfg.epochs):
                self.state.current_epoch = epoch
                self.callbacks.on_epoch_start(epoch)
                
                # Training phase
                train_metrics = self._train_epoch()
                
                # Validation phase
                val_metrics = {}
                if self._val_loader is not None:
                    val_metrics = self._validate()
                    self.callbacks.on_validation_end(epoch, val_metrics)
                
                # Combine metrics
                epoch_metrics = {**train_metrics, **val_metrics}
                self.state.history.append({
                    'epoch': epoch,
                    **epoch_metrics,
                })
                
                # Update best metric
                val_loss = val_metrics.get('val_loss', float('inf'))
                if val_loss < self.state.best_val_metric:
                    self.state.best_val_metric = val_loss
                    self.state.best_epoch = epoch
                
                # Notify callbacks
                self.callbacks.on_epoch_end(epoch, epoch_metrics)
                
                # Learning rate scheduling
                if self._scheduler is not None:
                    if isinstance(self._scheduler, ReduceLROnPlateau):
                        self._scheduler.step(val_loss)
                    else:
                        self._scheduler.step()
                
                # Check early stopping
                if self.callbacks.should_stop:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        finally:
            self.state.training_completed = True
            self.callbacks.on_train_end()
        
        total_time = time.time() - start_time
        logger.info(
            f"Training completed in {total_time:.1f}s. "
            f"Best val_loss={self.state.best_val_metric:.6f} at epoch {self.state.best_epoch}"
        )
        
        return {
            'best_val_metric': self.state.best_val_metric,
            'best_epoch': self.state.best_epoch,
            'total_epochs': self.state.current_epoch + 1,
            'total_time_seconds': total_time,
            'history': self.state.history,
        }
    
    def _train_epoch(self) -> Dict[str, float]:
        """
        Run one training epoch.
        
        Returns:
            Dict with training metrics (train_loss, train_accuracy)
        """
        self.model.train()
        cfg = self.config.train
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, (features, labels) in enumerate(self._train_loader):
            self.callbacks.on_batch_start(batch_idx)
            
            # Move to device
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if cfg.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    cfg.gradient_clip_norm,
                )
            
            self.optimizer.step()
            self.state.global_step += 1
            
            # Track metrics
            total_loss += loss.item() * features.size(0)
            predictions = outputs.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += features.size(0)
            
            # Batch callback
            batch_metrics = {'loss': loss.item()}
            self.callbacks.on_batch_end(batch_idx, batch_metrics)
        
        # Compute epoch metrics
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        
        return {
            'train_loss': avg_loss,
            'train_accuracy': accuracy,
        }
    
    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """
        Run validation with strategy-aware metrics.
        
        Returns:
            Dict with validation metrics. The specific metrics depend on the
            labeling_strategy configured in the experiment:
            
            For TRIPLE_BARRIER:
            - val_loss, val_accuracy, val_macro_f1
            - stoploss_precision, timeout_precision, profittarget_precision
            - profit_target_precision, stop_loss_precision (trading-specific)
            - decisive_prediction_rate, true_win_rate, predicted_trade_win_rate
            
            For OPPORTUNITY:
            - val_loss, val_accuracy, val_macro_f1
            - bigdown_precision, noopportunity_precision, bigup_precision
            - directional_accuracy, opportunity_prediction_rate
            
            For TLOB:
            - val_loss, val_accuracy, val_macro_f1
            - down_precision, stable_precision, up_precision
            - directional_accuracy, signal_rate
        """
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        
        # Collect all predictions for metrics
        all_preds = []
        all_labels = []
        
        for features, labels in self._val_loader:
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            
            total_loss += loss.item() * features.size(0)
            predictions = outputs.argmax(dim=1)
            total_samples += features.size(0)
            
            # Collect for metrics
            all_preds.append(predictions.cpu())
            all_labels.append(labels.cpu())
        
        avg_loss = total_loss / total_samples
        
        # Concatenate all predictions and labels
        y_pred = torch.cat(all_preds)
        y_true = torch.cat(all_labels)
        
        # Get labeling strategy from config (default to opportunity for backward compat)
        strategy = getattr(self.config.data, 'labeling_strategy', 'opportunity')
        if hasattr(strategy, 'value'):
            strategy = strategy.value
        
        num_classes = self.config.data.num_classes
        
        # Compute strategy-aware metrics
        metrics_calculator = MetricsCalculator(strategy, num_classes)
        metrics = metrics_calculator.compute(y_pred, y_true, loss=avg_loss)
        
        # Build result dict with val_ prefix
        result = {
            'val_loss': avg_loss,
            'val_accuracy': metrics.accuracy,
            'val_macro_f1': metrics.macro_f1,
        }
        
        # Add per-class precision with class names (lowercase)
        for class_id, precision in metrics.per_class_precision.items():
            if class_id < len(metrics.class_names):
                name = metrics.class_names[class_id].lower()
                result[f'val_{name}_precision'] = precision
        
        # Add strategy-specific metrics with val_ prefix
        for key, value in metrics.strategy_metrics.items():
            result[f'val_{key}'] = value
        
        # Backward compatibility: add directional metrics for TLOB and OPPORTUNITY
        if strategy in ('tlob', 'opportunity'):
            if 'directional_accuracy' in metrics.strategy_metrics:
                result['val_directional_accuracy'] = metrics.strategy_metrics['directional_accuracy']
            if 'signal_rate' in metrics.strategy_metrics:
                result['val_signal_rate'] = metrics.strategy_metrics['signal_rate']
        
        return result
    
    # =========================================================================
    # Evaluation
    # =========================================================================
    
    @torch.no_grad()
    def evaluate(
        self,
        split: str = 'test',
        loader: Optional[DataLoader] = None,
    ) -> ClassificationMetrics:
        """
        Evaluate model on a data split.
        
        Args:
            split: Data split to evaluate ('train', 'val', 'test')
            loader: Optional custom DataLoader (uses split if None)
        
        Returns:
            ClassificationMetrics with full evaluation
        """
        # Get loader
        if loader is None:
            if split == 'train':
                loader = self._train_loader
            elif split == 'val':
                loader = self._val_loader
            elif split == 'test':
                loader = self._test_loader
            else:
                raise ValueError(f"Unknown split: {split}")
        
        if loader is None:
            # Try to load
            if self._train_loader is None:
                self.setup()
            loader = getattr(self, f'_{split}_loader')
        
        if loader is None:
            raise ValueError(f"No data available for split: {split}")
        
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        
        for features, labels in loader:
            features = features.to(self.device)
            outputs = self.model(features)
            predictions = outputs.argmax(dim=1).cpu().numpy()
            
            all_predictions.append(predictions)
            all_labels.append(labels.numpy())
        
        y_pred = np.concatenate(all_predictions)
        y_true = np.concatenate(all_labels)
        
        # Use MetricsCalculator for strategy-aware metrics
        from lobtrainer.training.metrics import MetricsCalculator
        metrics_calculator = MetricsCalculator(
            labeling_strategy=self.config.data.labeling_strategy,
            num_classes=self.config.model.num_classes,
        )
        metrics = metrics_calculator.compute(y_pred, y_true)
        
        # Log key metrics
        logger.info(
            f"Evaluation [{split}]: accuracy={metrics.accuracy:.4f}, "
            f"macro_f1={metrics.macro_f1:.4f}"
        )
        
        # Log strategy-specific metrics
        if metrics.strategy_metrics:
            strategy_str = ", ".join(
                f"{k}={v:.4f}" for k, v in list(metrics.strategy_metrics.items())[:5]
            )
            logger.info(f"  Strategy metrics: {strategy_str}")
        
        return metrics
    
    def predict(
        self,
        features: Union[np.ndarray, torch.Tensor],
        return_proba: bool = False,
    ) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            features: Feature array [batch, seq_len, features] or [batch, features]
            return_proba: If True, return class probabilities
        
        Returns:
            Predictions [batch] or probabilities [batch, num_classes]
        """
        self.model.eval()
        
        # Convert to tensor if needed
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float()
        
        features = features.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(features)
            
            if return_proba:
                proba = torch.softmax(outputs, dim=1)
                return proba.cpu().numpy()
            else:
                predictions = outputs.argmax(dim=1)
                return predictions.cpu().numpy()
    
    # =========================================================================
    # Checkpoint Management
    # =========================================================================
    
    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            'epoch': self.state.current_epoch,
            'global_step': self.state.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict(),
            'state': {
                'best_val_metric': self.state.best_val_metric,
                'best_epoch': self.state.best_epoch,
            },
        }
        
        if self._scheduler is not None:
            checkpoint['scheduler_state_dict'] = self._scheduler.state_dict()
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: Union[str, Path], load_optimizer: bool = True) -> None:
        """
        Load model from checkpoint.
        
        Args:
            path: Path to checkpoint
            load_optimizer: Whether to load optimizer state
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self._scheduler is not None:
            self._scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore state
        self.state.current_epoch = checkpoint.get('epoch', 0)
        self.state.global_step = checkpoint.get('global_step', 0)
        
        if 'state' in checkpoint:
            self.state.best_val_metric = checkpoint['state'].get('best_val_metric', float('inf'))
            self.state.best_epoch = checkpoint['state'].get('best_epoch', 0)
        
        logger.info(f"Loaded checkpoint from {path} (epoch {self.state.current_epoch})")


# =============================================================================
# Factory Function
# =============================================================================


def create_trainer(
    config: Union[str, Path, ExperimentConfig],
    **kwargs,
) -> Trainer:
    """
    Create a Trainer from config file or object.
    
    Args:
        config: Path to config file or ExperimentConfig object
        **kwargs: Additional arguments passed to Trainer
    
    Returns:
        Configured Trainer instance
    
    Example:
        >>> trainer = create_trainer("configs/lstm.yaml")
        >>> trainer.train()
    """
    from lobtrainer.config import load_config
    
    if isinstance(config, (str, Path)):
        config = load_config(str(config))
    
    # Setup default callbacks if none provided
    if 'callbacks' not in kwargs:
        output_dir = Path(config.output_dir)
        callbacks = [
            EarlyStopping(
                patience=config.train.early_stopping_patience,
                metric='val_loss',
                mode='min',
                restore_best_weights=True,
            ),
            ModelCheckpoint(
                save_dir=output_dir / 'checkpoints',
                metric='val_loss',
                mode='min',
                save_best_only=True,
            ),
            MetricLogger(
                log_to_file=True,
                log_file=output_dir / 'training_history.json',
            ),
        ]
        kwargs['callbacks'] = callbacks
    
    return Trainer(config, **kwargs)

