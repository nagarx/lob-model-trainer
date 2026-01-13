"""
Enhanced monitoring callbacks for training diagnostics.

Provides real-time insights into training health and helps catch issues early:
- Gradient norms: Detect vanishing/exploding gradients
- Learning rate: Track schedule changes
- Training health: Alert on suspicious patterns
- Per-class metrics: Track class-specific performance

Design principles (RULE.md):
- All thresholds configurable
- Metrics tracked are well-defined with clear semantics
- Non-invasive: Monitoring should not affect training behavior
- Structured output for post-hoc analysis

Usage:
    >>> from lobtrainer.training.monitoring import (
    ...     GradientMonitor,
    ...     TrainingDiagnostics,
    ...     LearningRateTracker,
    ... )
    >>> 
    >>> callbacks = [
    ...     GradientMonitor(log_every_n_batches=100),
    ...     TrainingDiagnostics(alert_on_nan=True),
    ...     LearningRateTracker(),
    ... ]
    >>> trainer = Trainer(config, callbacks=callbacks)
"""

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json

import numpy as np
import torch
import torch.nn as nn

from lobtrainer.training.callbacks import Callback

logger = logging.getLogger(__name__)


# =============================================================================
# Gradient Monitoring
# =============================================================================


@dataclass
class GradientStats:
    """Statistics for a single gradient snapshot."""
    
    total_norm: float
    """Total gradient norm across all parameters (L2)."""
    
    max_norm: float
    """Maximum gradient norm of any single parameter."""
    
    min_norm: float
    """Minimum gradient norm of any single parameter."""
    
    mean_norm: float
    """Mean gradient norm across parameters."""
    
    num_zero_grads: int
    """Number of parameters with zero gradients (potential issue)."""
    
    num_nan_grads: int
    """Number of parameters with NaN gradients (critical issue)."""
    
    num_inf_grads: int
    """Number of parameters with Inf gradients (critical issue)."""
    
    layer_norms: Dict[str, float] = field(default_factory=dict)
    """Gradient norms per named layer (for debugging)."""
    
    @property
    def is_healthy(self) -> bool:
        """Check if gradients are healthy (no NaN/Inf, reasonable norms)."""
        return (
            self.num_nan_grads == 0 and
            self.num_inf_grads == 0 and
            0 < self.total_norm < 1000  # Reasonable range
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'total_norm': self.total_norm,
            'max_norm': self.max_norm,
            'min_norm': self.min_norm,
            'mean_norm': self.mean_norm,
            'num_zero_grads': self.num_zero_grads,
            'num_nan_grads': self.num_nan_grads,
            'num_inf_grads': self.num_inf_grads,
            'is_healthy': self.is_healthy,
        }


class GradientMonitor(Callback):
    """
    Monitor gradient statistics during training.
    
    Helps detect:
    - Vanishing gradients (very small norms)
    - Exploding gradients (very large norms)
    - NaN/Inf gradients (critical failures)
    - Dead neurons (zero gradients)
    
    Args:
        log_every_n_batches: Log gradient stats every N batches. None = epoch end only.
        warn_threshold_low: Warn if total gradient norm below this. Default: 1e-7.
        warn_threshold_high: Warn if total gradient norm above this. Default: 100.
        track_per_layer: Track gradients per named layer (more detailed but slower).
        save_history: Save gradient history to file at end of training.
    
    Example:
        >>> monitor = GradientMonitor(
        ...     log_every_n_batches=100,
        ...     warn_threshold_low=1e-7,
        ...     warn_threshold_high=100,
        ... )
    
    Interpretation:
        - total_norm << 1e-4: Possible vanishing gradients (learning stalled)
        - total_norm >> 10: Possible exploding gradients (need gradient clipping)
        - num_nan_grads > 0: Critical! Training is corrupted
    """
    
    def __init__(
        self,
        log_every_n_batches: Optional[int] = None,
        warn_threshold_low: float = 1e-7,
        warn_threshold_high: float = 100.0,
        track_per_layer: bool = False,
        save_history: bool = True,
    ):
        super().__init__()
        
        self.log_every_n_batches = log_every_n_batches
        self.warn_threshold_low = warn_threshold_low
        self.warn_threshold_high = warn_threshold_high
        self.track_per_layer = track_per_layer
        self.save_history = save_history
        
        self._history: List[Dict[str, Any]] = []
        self._current_epoch = 0
        self._batch_stats: List[GradientStats] = []
    
    def compute_gradient_stats(self, model: nn.Module) -> GradientStats:
        """Compute gradient statistics for a model."""
        total_norm = 0.0
        norms = []
        layer_norms = {}
        num_zero = 0
        num_nan = 0
        num_inf = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach()
                param_norm = grad.norm(2).item()
                
                # Track per-layer norms
                if self.track_per_layer:
                    layer_norms[name] = param_norm
                
                # Check for issues
                if torch.isnan(grad).any():
                    num_nan += 1
                elif torch.isinf(grad).any():
                    num_inf += 1
                elif param_norm == 0:
                    num_zero += 1
                else:
                    norms.append(param_norm)
                    total_norm += param_norm ** 2
        
        total_norm = total_norm ** 0.5
        
        return GradientStats(
            total_norm=total_norm,
            max_norm=max(norms) if norms else 0.0,
            min_norm=min(norms) if norms else 0.0,
            mean_norm=np.mean(norms) if norms else 0.0,
            num_zero_grads=num_zero,
            num_nan_grads=num_nan,
            num_inf_grads=num_inf,
            layer_norms=layer_norms,
        )
    
    def on_epoch_start(self, epoch: int) -> None:
        """Reset batch stats at epoch start."""
        self._current_epoch = epoch
        self._batch_stats = []
    
    def on_batch_end(self, batch_idx: int, logs: Dict[str, float]) -> None:
        """Compute and optionally log gradient stats."""
        if self.trainer is None or self.trainer._model is None:
            return
        
        stats = self.compute_gradient_stats(self.trainer.model)
        self._batch_stats.append(stats)
        
        # Check for critical issues
        if stats.num_nan_grads > 0:
            logger.error(
                f"CRITICAL: NaN gradients detected at batch {batch_idx}! "
                f"Training may be corrupted."
            )
        if stats.num_inf_grads > 0:
            logger.error(
                f"CRITICAL: Inf gradients detected at batch {batch_idx}! "
                f"Consider gradient clipping."
            )
        
        # Warn on suspicious values
        if stats.total_norm < self.warn_threshold_low:
            logger.warning(
                f"Batch {batch_idx}: Very small gradient norm ({stats.total_norm:.2e}). "
                f"Possible vanishing gradients."
            )
        elif stats.total_norm > self.warn_threshold_high:
            logger.warning(
                f"Batch {batch_idx}: Large gradient norm ({stats.total_norm:.2f}). "
                f"Consider gradient clipping."
            )
        
        # Periodic logging
        if self.log_every_n_batches is not None:
            if (batch_idx + 1) % self.log_every_n_batches == 0:
                logger.info(
                    f"Batch {batch_idx + 1} gradients: "
                    f"total_norm={stats.total_norm:.4f}, "
                    f"max={stats.max_norm:.4f}, "
                    f"zeros={stats.num_zero_grads}"
                )
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, float]) -> None:
        """Summarize epoch gradient statistics."""
        if not self._batch_stats:
            return
        
        # Aggregate stats
        total_norms = [s.total_norm for s in self._batch_stats]
        nan_counts = sum(s.num_nan_grads for s in self._batch_stats)
        inf_counts = sum(s.num_inf_grads for s in self._batch_stats)
        
        epoch_summary = {
            'epoch': epoch,
            'grad_norm_mean': float(np.mean(total_norms)),
            'grad_norm_std': float(np.std(total_norms)),
            'grad_norm_max': float(max(total_norms)),
            'grad_norm_min': float(min(total_norms)),
            'nan_count': nan_counts,
            'inf_count': inf_counts,
        }
        
        self._history.append(epoch_summary)
        
        logger.info(
            f"Epoch {epoch + 1} gradient summary: "
            f"mean_norm={epoch_summary['grad_norm_mean']:.4f} "
            f"(±{epoch_summary['grad_norm_std']:.4f}), "
            f"range=[{epoch_summary['grad_norm_min']:.4f}, {epoch_summary['grad_norm_max']:.4f}]"
        )
    
    def on_train_end(self) -> None:
        """Save gradient history if configured."""
        if self.save_history and self.trainer is not None:
            output_dir = Path(self.trainer.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            history_path = output_dir / 'gradient_history.json'
            with open(history_path, 'w') as f:
                json.dump(self._history, f, indent=2)
            
            logger.info(f"GradientMonitor: saved history to {history_path}")
    
    @property
    def history(self) -> List[Dict[str, Any]]:
        """Gradient history for analysis."""
        return self._history


# =============================================================================
# Learning Rate Tracking
# =============================================================================


class LearningRateTracker(Callback):
    """
    Track learning rate changes during training.
    
    Useful for:
    - Visualizing learning rate schedules
    - Debugging scheduler behavior
    - Correlating LR changes with metric changes
    
    Args:
        save_history: Save LR history to file at end of training.
    
    Example:
        >>> tracker = LearningRateTracker()
        >>> # After training:
        >>> tracker.history  # [(epoch, lr), ...]
    """
    
    def __init__(self, save_history: bool = True):
        super().__init__()
        self.save_history = save_history
        self._history: List[Dict[str, Any]] = []
    
    def on_epoch_start(self, epoch: int) -> None:
        """Record learning rate at epoch start."""
        if self.trainer is None or self.trainer._optimizer is None:
            return
        
        # Get current LR (handle multiple param groups)
        lrs = [pg['lr'] for pg in self.trainer.optimizer.param_groups]
        
        self._history.append({
            'epoch': epoch,
            'learning_rates': lrs,
            'lr': lrs[0],  # Primary LR for convenience
        })
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, float]) -> None:
        """Log LR at epoch end (after potential scheduler step)."""
        if self.trainer is None or self.trainer._optimizer is None:
            return
        
        lr = self.trainer.optimizer.param_groups[0]['lr']
        logger.debug(f"Epoch {epoch + 1} learning rate: {lr:.2e}")
    
    def on_train_end(self) -> None:
        """Save LR history if configured."""
        if self.save_history and self.trainer is not None:
            output_dir = Path(self.trainer.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            history_path = output_dir / 'learning_rate_history.json'
            with open(history_path, 'w') as f:
                json.dump(self._history, f, indent=2)
            
            logger.info(f"LearningRateTracker: saved history to {history_path}")
    
    @property
    def history(self) -> List[Dict[str, Any]]:
        """LR history for analysis."""
        return self._history


# =============================================================================
# Training Diagnostics
# =============================================================================


@dataclass
class HealthCheckResult:
    """Result of a training health check."""
    
    is_healthy: bool
    """Overall health status."""
    
    issues: List[str] = field(default_factory=list)
    """List of detected issues."""
    
    warnings: List[str] = field(default_factory=list)
    """List of warnings (not critical)."""
    
    metrics: Dict[str, Any] = field(default_factory=dict)
    """Relevant metrics for debugging."""


class TrainingDiagnostics(Callback):
    """
    Comprehensive training health monitoring.
    
    Performs automatic health checks during training:
    - NaN/Inf detection in loss
    - Accuracy stagnation detection
    - Loss divergence detection
    - Class imbalance warnings
    
    Args:
        alert_on_nan: Raise error on NaN loss. Default: True.
        stagnation_patience: Epochs without improvement before warning. Default: 5.
        divergence_threshold: Loss increase ratio to trigger warning. Default: 2.0.
        check_class_balance: Warn on severe class imbalance. Default: True.
    
    Example:
        >>> diagnostics = TrainingDiagnostics(
        ...     alert_on_nan=True,
        ...     stagnation_patience=5,
        ... )
    """
    
    def __init__(
        self,
        alert_on_nan: bool = True,
        stagnation_patience: int = 5,
        divergence_threshold: float = 2.0,
        check_class_balance: bool = True,
    ):
        super().__init__()
        
        self.alert_on_nan = alert_on_nan
        self.stagnation_patience = stagnation_patience
        self.divergence_threshold = divergence_threshold
        self.check_class_balance = check_class_balance
        
        self._loss_history: List[float] = []
        self._val_loss_history: List[float] = []
        self._accuracy_history: List[float] = []
        self._best_accuracy = 0.0
        self._epochs_without_improvement = 0
        self._initial_loss: Optional[float] = None
        self._health_history: List[HealthCheckResult] = []
    
    def on_train_start(self) -> None:
        """Reset state at training start."""
        self._loss_history = []
        self._val_loss_history = []
        self._accuracy_history = []
        self._best_accuracy = 0.0
        self._epochs_without_improvement = 0
        self._initial_loss = None
        self._health_history = []
        
        logger.info("TrainingDiagnostics: monitoring enabled")
    
    def on_batch_end(self, batch_idx: int, logs: Dict[str, float]) -> None:
        """Check batch-level health."""
        loss = logs.get('loss', logs.get('train_loss'))
        
        if loss is not None:
            # Check for NaN
            if np.isnan(loss):
                msg = f"NaN loss detected at batch {batch_idx}!"
                logger.error(msg)
                if self.alert_on_nan:
                    raise ValueError(msg)
            
            # Check for Inf
            if np.isinf(loss):
                msg = f"Inf loss detected at batch {batch_idx}!"
                logger.error(msg)
                if self.alert_on_nan:
                    raise ValueError(msg)
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, float]) -> None:
        """Perform comprehensive health check at epoch end."""
        issues = []
        warnings_list = []
        
        # Track losses
        train_loss = logs.get('train_loss', logs.get('loss'))
        val_loss = logs.get('val_loss')
        accuracy = logs.get('val_accuracy', logs.get('accuracy'))
        
        if train_loss is not None:
            self._loss_history.append(train_loss)
            if self._initial_loss is None:
                self._initial_loss = train_loss
        
        if val_loss is not None:
            self._val_loss_history.append(val_loss)
        
        if accuracy is not None:
            self._accuracy_history.append(accuracy)
            
            # Check for improvement
            if accuracy > self._best_accuracy:
                self._best_accuracy = accuracy
                self._epochs_without_improvement = 0
            else:
                self._epochs_without_improvement += 1
        
        # === Health Checks ===
        
        # 1. Loss divergence check
        if (
            self._initial_loss is not None and
            train_loss is not None and
            train_loss > self._initial_loss * self.divergence_threshold
        ):
            warnings_list.append(
                f"Loss divergence: current ({train_loss:.4f}) > "
                f"{self.divergence_threshold}x initial ({self._initial_loss:.4f})"
            )
        
        # 2. Stagnation check
        if self._epochs_without_improvement >= self.stagnation_patience:
            warnings_list.append(
                f"Accuracy stagnation: no improvement for "
                f"{self._epochs_without_improvement} epochs "
                f"(best: {self._best_accuracy:.4f})"
            )
        
        # 3. Overfitting check (train loss decreasing, val loss increasing)
        if len(self._loss_history) >= 3 and len(self._val_loss_history) >= 3:
            recent_train = self._loss_history[-3:]
            recent_val = self._val_loss_history[-3:]
            
            train_decreasing = all(
                recent_train[i] >= recent_train[i+1] 
                for i in range(len(recent_train)-1)
            )
            val_increasing = all(
                recent_val[i] <= recent_val[i+1] 
                for i in range(len(recent_val)-1)
            )
            
            if train_decreasing and val_increasing:
                warnings_list.append(
                    "Potential overfitting: train loss decreasing while val loss increasing"
                )
        
        # 4. Very low accuracy check
        if accuracy is not None and accuracy < 0.35 and epoch >= 3:
            warnings_list.append(
                f"Low accuracy ({accuracy:.3f}): model may not be learning"
            )
        
        # Create health result
        is_healthy = len(issues) == 0
        result = HealthCheckResult(
            is_healthy=is_healthy,
            issues=issues,
            warnings=warnings_list,
            metrics={
                'train_loss': train_loss,
                'val_loss': val_loss,
                'accuracy': accuracy,
                'epochs_without_improvement': self._epochs_without_improvement,
            }
        )
        
        self._health_history.append(result)
        
        # Log warnings
        for warning in warnings_list:
            logger.warning(f"Epoch {epoch + 1}: {warning}")
        
        # Log issues (critical)
        for issue in issues:
            logger.error(f"Epoch {epoch + 1}: {issue}")
    
    def on_train_end(self) -> None:
        """Generate final health report."""
        if self.trainer is None:
            return
        
        # Summary
        total_warnings = sum(len(h.warnings) for h in self._health_history)
        total_issues = sum(len(h.issues) for h in self._health_history)
        
        logger.info(
            f"TrainingDiagnostics summary: "
            f"{total_issues} issues, {total_warnings} warnings across "
            f"{len(self._health_history)} epochs"
        )
        
        # Save report
        output_dir = Path(self.trainer.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report = {
            'total_epochs': len(self._health_history),
            'total_issues': total_issues,
            'total_warnings': total_warnings,
            'final_best_accuracy': self._best_accuracy,
            'loss_history': self._loss_history,
            'val_loss_history': self._val_loss_history,
            'accuracy_history': self._accuracy_history,
        }
        
        report_path = output_dir / 'training_diagnostics.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"TrainingDiagnostics: saved report to {report_path}")
    
    @property
    def health_history(self) -> List[HealthCheckResult]:
        """Health check history."""
        return self._health_history


# =============================================================================
# Per-Class Metrics Tracker
# =============================================================================


class PerClassMetricsTracker(Callback):
    """
    Track per-class metrics during training for detailed analysis.
    
    Useful for:
    - Detecting class-specific learning issues
    - Monitoring precision/recall balance
    - Understanding model behavior on minority classes
    
    Args:
        class_names: Optional list of class names for logging.
        save_history: Save per-class history to file.
    
    Example:
        >>> tracker = PerClassMetricsTracker(
        ...     class_names=['Down', 'Stable', 'Up']
        ... )
    """
    
    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        save_history: bool = True,
    ):
        super().__init__()
        self.class_names = class_names
        self.save_history = save_history
        self._history: List[Dict[str, Any]] = []
    
    def on_validation_end(self, epoch: int, logs: Dict[str, float]) -> None:
        """Extract per-class metrics from validation logs."""
        epoch_metrics = {'epoch': epoch}
        
        # Extract per-class metrics from logs
        # Format: {class_name}_precision, {class_name}_recall, etc.
        for key, value in logs.items():
            if any(metric in key for metric in ['precision', 'recall', 'f1']):
                epoch_metrics[key] = value
        
        if epoch_metrics:
            self._history.append(epoch_metrics)
    
    def on_train_end(self) -> None:
        """Save per-class history."""
        if self.save_history and self.trainer is not None and self._history:
            output_dir = Path(self.trainer.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            history_path = output_dir / 'per_class_metrics_history.json'
            with open(history_path, 'w') as f:
                json.dump(self._history, f, indent=2)
            
            logger.info(f"PerClassMetricsTracker: saved history to {history_path}")
    
    @property
    def history(self) -> List[Dict[str, Any]]:
        """Per-class metrics history."""
        return self._history


# =============================================================================
# Convenience Function
# =============================================================================


def create_standard_monitoring(
    gradient_log_every: int = 100,
    include_diagnostics: bool = True,
    include_lr_tracker: bool = True,
    include_per_class: bool = True,
) -> List[Callback]:
    """
    Create a standard set of monitoring callbacks.
    
    Args:
        gradient_log_every: Log gradients every N batches.
        include_diagnostics: Include TrainingDiagnostics.
        include_lr_tracker: Include LearningRateTracker.
        include_per_class: Include PerClassMetricsTracker.
    
    Returns:
        List of monitoring callbacks.
    
    Example:
        >>> callbacks = create_standard_monitoring()
        >>> trainer = Trainer(config, callbacks=callbacks)
    """
    callbacks = [
        GradientMonitor(log_every_n_batches=gradient_log_every),
    ]
    
    if include_diagnostics:
        callbacks.append(TrainingDiagnostics())
    
    if include_lr_tracker:
        callbacks.append(LearningRateTracker())
    
    if include_per_class:
        callbacks.append(PerClassMetricsTracker())
    
    return callbacks
