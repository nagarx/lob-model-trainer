"""
Callback system for training hooks.

Callbacks provide a modular way to add functionality to the training loop
without modifying the core Trainer class. This follows the open-closed principle:
the trainer is open for extension but closed for modification.

Callback lifecycle:
    on_train_start()     - Called once at the start of training
    on_epoch_start()     - Called at the start of each epoch
    on_batch_start()     - Called before each training batch
    on_batch_end()       - Called after each training batch
    on_epoch_end()       - Called at the end of each epoch
    on_validation_end()  - Called after validation
    on_train_end()       - Called once at the end of training

Design principles (RULE.md):
- Callbacks are optional and composable
- Each callback has a single responsibility
- Callbacks should not modify training behavior unexpectedly
- State is explicit and managed by the callback

Usage:
    >>> callbacks = [
    ...     EarlyStopping(patience=10, metric='val_loss'),
    ...     ModelCheckpoint(save_dir='checkpoints/', save_best_only=True),
    ... ]
    >>> trainer = Trainer(config, callbacks=callbacks)
"""

import json
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import shutil

import torch

# Phase X.3 Empirical Trust (2026-05-05): MonitorMetricUndefined raised by
# EarlyStopping + ModelCheckpoint when the monitored metric is non-finite,
# replacing the pre-X.3 silent `nan < best = False` "no improvement" stall.
# Local import at use-site below to avoid potential circular imports during
# module load (callbacks is imported very early in trainer init).

logger = logging.getLogger(__name__)


# =============================================================================
# Base Callback Interface
# =============================================================================


class Callback(ABC):
    """
    Base class for all callbacks.
    
    Subclasses should override the methods they need. Default implementations
    are no-ops to allow selective overriding.
    
    Callbacks have access to the trainer via the `trainer` attribute which
    is set by the trainer before any hooks are called.
    """
    
    def __init__(self):
        self.trainer = None  # Set by Trainer
    
    def on_train_start(self) -> None:
        """Called at the start of training (before first epoch)."""
        pass
    
    def on_train_end(self) -> None:
        """Called at the end of training (after last epoch or early stop)."""
        pass
    
    def on_epoch_start(self, epoch: int) -> None:
        """Called at the start of each epoch."""
        pass
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, float]) -> None:
        """
        Called at the end of each epoch.
        
        Args:
            epoch: Current epoch number (0-indexed)
            logs: Dict with metrics (e.g., 'train_loss', 'val_loss', 'val_accuracy')
        """
        pass
    
    def on_batch_start(self, batch_idx: int) -> None:
        """Called before processing each training batch."""
        pass
    
    def on_batch_end(self, batch_idx: int, logs: Dict[str, float]) -> None:
        """
        Called after processing each training batch.
        
        Args:
            batch_idx: Current batch index within the epoch
            logs: Dict with batch metrics (e.g., 'loss')
        """
        pass
    
    def on_validation_end(self, epoch: int, logs: Dict[str, float]) -> None:
        """
        Called after validation is complete.

        Args:
            epoch: Current epoch number
            logs: Dict with validation metrics
        """
        pass

    def state_dict(self) -> Dict[str, Any]:
        """Return JSON-serializable callback state for checkpoint embedding.

        Phase DESIGN-1 G-1 (2026-05-10): sister-site to RNG state in
        checkpoint dict. Default returns empty dict — stateless callbacks
        (e.g., ProgressCallback) need not override. Subclasses with
        resume-relevant state (EarlyStopping wait_count + best_value,
        ModelCheckpoint _best_value, MetricLogger _history) override to
        return a JSON-serializable dict that ``load_state_dict`` consumes
        on resume.

        REUSES ``hft_contracts.atomic_io.atomic_write_json`` discipline:
        returned dict MUST be canonical-JSON compatible (no tuples, no
        ndarray, no Tensor). For numpy/torch state, convert via
        ``.tolist()`` or skip entirely.

        Returns:
            Dict with arbitrary serializable callback state. Default
            ``{}`` (empty) — back-compat with pre-G-1 callbacks; on
            resume, ``load_state_dict({})`` is a no-op.
        """
        return {}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore callback state from checkpoint.

        Phase DESIGN-1 G-1 (2026-05-10): inverse of ``state_dict``.
        Default no-op — stateless callbacks need not override. Pre-G-1
        checkpoints lack ``callback_state`` key, in which case the
        Trainer skips this call entirely (back-compat preserved).

        Subclasses MUST tolerate an empty / partial dict: when the
        checkpoint was produced by a different version of this callback,
        unknown keys should be ignored and missing keys should fall
        back to current state. Per hft-rules §8 — never silently drop,
        but tolerate forward-compat omissions.

        Args:
            state: Dict produced by a prior ``state_dict()`` call.
                MUST tolerate empty dict (pre-G-1 checkpoint resumed
                via post-G-1 trainer with state_dict-aware callback).
        """
        pass

    @property
    def should_stop(self) -> bool:
        """
        Whether training should stop.

        Override this property to implement early stopping logic.
        The trainer checks this after each epoch.
        """
        return False


class CallbackList:
    """
    Container for managing multiple callbacks.
    
    Dispatches callback events to all contained callbacks in order.
    """
    
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []
    
    def set_trainer(self, trainer) -> None:
        """Set trainer reference for all callbacks."""
        for callback in self.callbacks:
            callback.trainer = trainer
    
    def on_train_start(self) -> None:
        for callback in self.callbacks:
            callback.on_train_start()
    
    def on_train_end(self) -> None:
        for callback in self.callbacks:
            callback.on_train_end()
    
    def on_epoch_start(self, epoch: int) -> None:
        for callback in self.callbacks:
            callback.on_epoch_start(epoch)
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, float]) -> None:
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)
    
    def on_batch_start(self, batch_idx: int) -> None:
        for callback in self.callbacks:
            callback.on_batch_start(batch_idx)
    
    def on_batch_end(self, batch_idx: int, logs: Dict[str, float]) -> None:
        for callback in self.callbacks:
            callback.on_batch_end(batch_idx, logs)
    
    def on_validation_end(self, epoch: int, logs: Dict[str, float]) -> None:
        for callback in self.callbacks:
            callback.on_validation_end(epoch, logs)
    
    @property
    def should_stop(self) -> bool:
        """Returns True if any callback signals to stop."""
        return any(callback.should_stop for callback in self.callbacks)
    
    def append(self, callback: Callback) -> None:
        self.callbacks.append(callback)


# =============================================================================
# Early Stopping Callback
# =============================================================================


@dataclass
class EarlyStoppingState:
    """Internal state for EarlyStopping callback."""
    best_value: float = float('inf')
    best_epoch: int = 0
    wait_count: int = 0
    stopped: bool = False


class EarlyStopping(Callback):
    """
    Stop training when a monitored metric stops improving.
    
    This prevents overfitting by stopping training when the validation
    metric hasn't improved for `patience` consecutive epochs.
    
    Args:
        patience: Number of epochs with no improvement before stopping.
        metric: Metric to monitor (default: 'val_loss').
        mode: 'min' for loss (lower is better), 'max' for accuracy (higher is better).
        min_delta: Minimum change to qualify as an improvement.
        restore_best_weights: If True, restore model to best epoch's weights when stopped.
    
    Example:
        >>> early_stop = EarlyStopping(
        ...     patience=10,
        ...     metric='val_loss',
        ...     mode='min',
        ...     restore_best_weights=True,
        ... )
    
    Design note (RULE.md §4):
        All parameters are configurable, no hardcoded values.
    """
    
    def __init__(
        self,
        patience: int = 10,
        metric: str = 'val_loss',
        mode: str = 'min',
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
    ):
        super().__init__()
        
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        if mode not in ('min', 'max'):
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")
        if min_delta < 0:
            raise ValueError(f"min_delta must be >= 0, got {min_delta}")
        
        self.patience = patience
        self.metric = metric
        self.mode = mode
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self._state = EarlyStoppingState()
        self._best_weights: Optional[Dict[str, Any]] = None
        
        # Set comparison function based on mode.
        #
        # Phase X.3 Empirical Trust (2026-05-05): replace lambda with a
        # closure that fail-loud-raises ``MonitorMetricUndefined`` on
        # non-finite ``new``. Pre-X.3 ``lambda new, best: new < best`` with
        # ``new=NaN`` returned False (NumPy/IEEE NaN comparison semantics) →
        # silently treated NaN as "no improvement" → patience counter
        # incremented but never reset → silent stall. Caught by 2026-05-05
        # multi-agent audit. Default args (`_md`, `_m`) capture by value at
        # closure creation to lock the behavior to the exact construction-
        # time min_delta + metric (defensive, avoids late-binding surprises).
        _metric = self.metric

        if mode == 'min':
            self._state.best_value = float('inf')

            def _is_better(new, best, _md=min_delta, _m=_metric):
                if not math.isfinite(new):
                    from lobtrainer.training.exceptions import MonitorMetricUndefined
                    raise MonitorMetricUndefined(metric=_m, value=new)
                return new < best - _md

            self._is_better = _is_better
        else:
            self._state.best_value = float('-inf')

            def _is_better(new, best, _md=min_delta, _m=_metric):
                if not math.isfinite(new):
                    from lobtrainer.training.exceptions import MonitorMetricUndefined
                    raise MonitorMetricUndefined(metric=_m, value=new)
                return new > best + _md

            self._is_better = _is_better
    
    def on_train_start(self) -> None:
        """Reset state at start of training.

        Phase X.1.K minimum-viable (2026-05-04): when the trainer is
        resuming from a checkpoint (``trainer._resumed_from_checkpoint=True``),
        SKIP the reset so the patience counter / best_value / best_weights
        carry over the resume boundary. The flag is consumed (reset to
        False) by Trainer.train()'s finally block, so a subsequent
        train() call without re-load resets normally.
        """
        if (
            self.trainer is not None
            and getattr(self.trainer, "_resumed_from_checkpoint", False)
        ):
            # Resume mode — preserve state from the prior checkpoint
            return

        if self.mode == 'min':
            self._state.best_value = float('inf')
        else:
            self._state.best_value = float('-inf')
        self._state.best_epoch = 0
        self._state.wait_count = 0
        self._state.stopped = False
        self._best_weights = None

    def on_epoch_end(self, epoch: int, logs: Dict[str, float]) -> None:
        """Check if metric improved."""
        if self.metric not in logs:
            logger.warning(
                f"EarlyStopping: metric '{self.metric}' not found in logs. "
                f"Available metrics: {list(logs.keys())}"
            )
            return
        
        current_value = logs[self.metric]
        
        if self._is_better(current_value, self._state.best_value):
            # Improvement found
            self._state.best_value = current_value
            self._state.best_epoch = epoch
            self._state.wait_count = 0
            
            # Save best weights
            if self.restore_best_weights and self.trainer is not None:
                self._best_weights = {
                    k: v.cpu().clone() for k, v in self.trainer.model.state_dict().items()
                }
            
            logger.debug(
                f"EarlyStopping: {self.metric} improved to {current_value:.6f}"
            )
        else:
            # No improvement
            self._state.wait_count += 1
            logger.debug(
                f"EarlyStopping: {self.metric}={current_value:.6f}, "
                f"no improvement for {self._state.wait_count}/{self.patience} epochs"
            )
            
            if self._state.wait_count >= self.patience:
                self._state.stopped = True
                logger.info(
                    f"EarlyStopping: stopped at epoch {epoch}. "
                    f"Best {self.metric}={self._state.best_value:.6f} at epoch {self._state.best_epoch}"
                )
    
    def on_train_end(self) -> None:
        """Restore best weights if applicable.

        Phase DESIGN-1 G-1 mid-impl post-fix (2026-05-10) — CRITIQUE 3
        closure: cross-process resume excludes ``_best_weights`` from
        ``state_dict()`` by design (100MB-1GB bloat avoidance per Agent
        Z). On post-resume train_end, ``_best_weights`` is ``None`` →
        the existing branch silently does nothing → returned model =
        last-epoch weights instead of operator-expected best-epoch
        weights. Per hft-rules §8 ("never silently drop") + §11 ("docs
        must reflect behavior"), surface the silent-degrade with a
        WARNING when ``restore_best_weights=True`` was configured by
        the operator but cannot be honored due to G-1's design choice.
        """
        if (
            self.restore_best_weights
            and self._best_weights is None
            and self.trainer is not None
            and getattr(self.trainer, "_resumed_from_checkpoint", False)
        ):
            logger.warning(
                "EarlyStopping: restore_best_weights=True was configured but "
                "_best_weights is None on a resumed run. Phase DESIGN-1 G-1 "
                "deliberately excludes _best_weights from the checkpoint "
                "callback_state dict (avoids 100MB-1GB bloat). Returned "
                "model = LAST-epoch weights, NOT best-epoch weights. To "
                "preserve best-weights restoration across resume, save the "
                "best checkpoint via ModelCheckpoint(save_best_only=True) "
                "and load THAT checkpoint instead of the last-epoch one."
            )
            return
        if self.restore_best_weights and self._best_weights is not None and self.trainer is not None:
            logger.info(
                f"EarlyStopping: restoring best weights from epoch {self._state.best_epoch}"
            )
            self.trainer.model.load_state_dict(self._best_weights)

    def state_dict(self) -> Dict[str, Any]:
        """Phase DESIGN-1 G-1 (2026-05-10): serialize patience tracking.

        Returns 4 fields covering wait/best/stopped semantics. Deliberately
        EXCLUDES ``_best_weights`` (model state_dict snapshot — typically
        100 MB to 1 GB for TLOB/HMHP) to avoid checkpoint bloat. The
        ``_best_weights`` snapshot can be reconstructed from the
        checkpoint's own ``model_state_dict`` at resume time IF the
        operator wants ``restore_best_weights=True`` post-resume. This
        decision documented as a known limitation per Agent Z.
        """
        return {
            "best_value": float(self._state.best_value),
            "best_epoch": int(self._state.best_epoch),
            "wait_count": int(self._state.wait_count),
            "stopped": bool(self._state.stopped),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Phase DESIGN-1 G-1 (2026-05-10): restore patience tracking.

        Tolerates empty / partial dict (forward-compat). Missing keys
        keep their current __init__ defaults.
        """
        if not state:
            return
        if "best_value" in state:
            self._state.best_value = float(state["best_value"])
        if "best_epoch" in state:
            self._state.best_epoch = int(state["best_epoch"])
        if "wait_count" in state:
            self._state.wait_count = int(state["wait_count"])
        if "stopped" in state:
            self._state.stopped = bool(state["stopped"])

    @property
    def should_stop(self) -> bool:
        return self._state.stopped
    
    @property
    def best_value(self) -> float:
        """Best metric value observed."""
        return self._state.best_value
    
    @property
    def best_epoch(self) -> int:
        """Epoch with best metric value."""
        return self._state.best_epoch


# =============================================================================
# Model Checkpoint Callback
# =============================================================================


class ModelCheckpoint(Callback):
    """
    Save model checkpoints during training.
    
    Can save:
    - Best model only (based on monitored metric)
    - Every epoch
    - Every N epochs
    
    Args:
        save_dir: Directory to save checkpoints.
        metric: Metric to monitor for 'best' saving.
        mode: 'min' or 'max' for metric comparison.
        save_best_only: If True, only save when metric improves.
        save_every_n_epochs: Save every N epochs (in addition to best).
        max_checkpoints: Maximum number of checkpoints to keep (None = keep all).
        filename_template: Template for checkpoint filename.
    
    Example:
        >>> checkpoint = ModelCheckpoint(
        ...     save_dir='checkpoints/',
        ...     metric='val_loss',
        ...     save_best_only=True,
        ... )
    
    Checkpoint contents:
        - model_state_dict: Model weights
        - optimizer_state_dict: Optimizer state (for resuming)
        - epoch: Epoch number
        - metrics: Dict with metrics at save time
        - config: Experiment configuration (for reproducibility)
    """
    
    def __init__(
        self,
        save_dir: Union[str, Path],
        metric: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True,
        save_every_n_epochs: Optional[int] = None,
        max_checkpoints: Optional[int] = 5,
        filename_template: str = 'checkpoint_epoch{epoch:03d}_{metric:.4f}.pt',
    ):
        super().__init__()
        
        self.save_dir = Path(save_dir)
        self.metric = metric
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_every_n_epochs = save_every_n_epochs
        self.max_checkpoints = max_checkpoints
        self.filename_template = filename_template
        
        # State
        #
        # Phase X.3 Empirical Trust (2026-05-05): same NaN-loud pattern as
        # EarlyStopping above. Without this guard, NaN val_loss would be
        # silently treated as "no improvement" → best.pt never saved on
        # NaN epochs → silent training stall + degraded checkpoint kept.
        _metric = self.metric

        if mode == 'min':
            self._best_value = float('inf')

            def _is_better(new, best, _m=_metric):
                if not math.isfinite(new):
                    from lobtrainer.training.exceptions import MonitorMetricUndefined
                    raise MonitorMetricUndefined(metric=_m, value=new)
                return new < best

            self._is_better = _is_better
        else:
            self._best_value = float('-inf')

            def _is_better(new, best, _m=_metric):
                if not math.isfinite(new):
                    from lobtrainer.training.exceptions import MonitorMetricUndefined
                    raise MonitorMetricUndefined(metric=_m, value=new)
                return new > best

            self._is_better = _is_better
        
        self._saved_checkpoints: List[Path] = []
        self._best_checkpoint_path: Optional[Path] = None
    
    def on_train_start(self) -> None:
        """Create save directory.

        Phase X.1.K minimum-viable (2026-05-04): when resuming, SKIP the
        ``_best_value`` reset so the next epoch's metric is compared
        against the prior session's best (not against fresh inf).
        """
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if (
            self.trainer is not None
            and getattr(self.trainer, "_resumed_from_checkpoint", False)
        ):
            # Resume mode — preserve _best_value snapshot from prior session
            return

        # Reset state (fresh training)
        if self.mode == 'min':
            self._best_value = float('inf')
        else:
            self._best_value = float('-inf')
        self._saved_checkpoints = []
        self._best_checkpoint_path = None
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, float]) -> None:
        """Save checkpoint if conditions are met."""
        # Phase X.3 Empirical Trust (2026-05-05): mirror EarlyStopping's
        # missing-metric handling at callbacks.py:274-279. Pre-X.3,
        # ``logs.get(self.metric, 0.0)`` silently returned 0.0 — for
        # mode='min' this trivially beat ``best_value=inf`` → spurious
        # "best" checkpoint saved with metric_value=0.0 in filename when
        # the metric was actually missing from logs (e.g., classification
        # config not emitting val_loss). Per hft-rules §5/§8.
        if self.metric not in logs:
            logger.warning(
                f"ModelCheckpoint: metric '{self.metric}' not found in logs. "
                f"Available metrics: {list(logs.keys())}. Skipping checkpoint save."
            )
            return

        current_value = logs[self.metric]

        should_save = False

        # Check if this is an improvement (raises MonitorMetricUndefined on NaN)
        is_improvement = self._is_better(current_value, self._best_value)
        if is_improvement:
            self._best_value = current_value
            should_save = True
        
        # Check periodic saving
        if self.save_every_n_epochs is not None:
            if (epoch + 1) % self.save_every_n_epochs == 0:
                should_save = True
        
        # If save_best_only, only save on improvements
        if self.save_best_only and not is_improvement:
            should_save = False
        
        if should_save:
            self._save_checkpoint(epoch, logs, is_best=is_improvement)
    
    def _save_checkpoint(
        self,
        epoch: int,
        logs: Dict[str, float],
        is_best: bool = False,
    ) -> None:
        """Save a checkpoint."""
        if self.trainer is None:
            logger.warning("ModelCheckpoint: trainer not set, skipping save")
            return

        # Phase X.3 Empirical Trust (2026-05-05): direct metric access — by
        # the time _save_checkpoint is called, the on_epoch_end gate at
        # callbacks.py:431 has already verified self.metric is in logs.
        # If it isn't (programming error), fail-loud via KeyError surfaces
        # the bug immediately. Pre-X.3 used ``logs.get(self.metric, 0.0)``
        # which silently filename-templated checkpoints with metric=0.0000
        # (misleading filename when metric was actually missing).
        metric_value = logs[self.metric]
        
        # Create filename
        filename = self.filename_template.format(
            epoch=epoch,
            metric=metric_value,
        )
        filepath = self.save_dir / filename
        
        # Phase X.1 v2 (2026-05-04): build canonical checkpoint via the
        # shared Trainer._build_checkpoint_dict helper. Eliminates 3-writer
        # divergence — Trainer.save_checkpoint, this callback, and any
        # future writer all emit IDENTICAL keys. Includes
        # 'compatibility' / 'compatibility_fingerprint' / 'model_config_hash'.
        if hasattr(self.trainer, '_build_checkpoint_dict'):
            checkpoint = self.trainer._build_checkpoint_dict(
                epoch_override=epoch,
                metrics_override=logs,
            )
        else:
            # Fallback for legacy trainers (e.g., third-party Trainer subclasses
            # that haven't picked up X.1 v2). Same shape as pre-X.1 callback dict.
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.trainer.model.state_dict(),
                'metrics': logs,
            }
            if hasattr(self.trainer, 'optimizer') and self.trainer.optimizer is not None:
                checkpoint['optimizer_state_dict'] = self.trainer.optimizer.state_dict()
            if hasattr(self.trainer, 'config') and self.trainer.config is not None:
                checkpoint['config'] = self.trainer.config.to_dict()

        # #PY-73 atomic write — ModelCheckpoint writes per-epoch + best.pt
        # in the hot loop. Pre-migration: bare torch.save + shutil.copy
        # both non-atomic; SIGKILL mid-write corrupts the LARGE (100MB-1GB)
        # checkpoint file. Migrated 2026-05-11 (hft-contracts v2.7.0).
        from hft_contracts.atomic_io import atomic_copy, atomic_write_torch
        atomic_write_torch(filepath, checkpoint)
        self._saved_checkpoints.append(filepath)

        logger.info(f"ModelCheckpoint: saved {filepath}")

        # Update best checkpoint path
        if is_best:
            # Also save as 'best.pt' (atomic copy — eliminates partial-write
            # window of bare shutil.copy on SIGKILL mid-duplication).
            best_path = self.save_dir / 'best.pt'
            atomic_copy(filepath, best_path)
            self._best_checkpoint_path = best_path
            logger.info(f"ModelCheckpoint: updated best.pt ({self.metric}={metric_value:.6f})")
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints if exceeding max_checkpoints."""
        if self.max_checkpoints is None:
            return
        
        # Keep best.pt separate, only manage numbered checkpoints
        numbered_checkpoints = [
            p for p in self._saved_checkpoints 
            if p.name != 'best.pt'
        ]
        
        while len(numbered_checkpoints) > self.max_checkpoints:
            oldest = numbered_checkpoints.pop(0)
            if oldest.exists():
                oldest.unlink()
                logger.debug(f"ModelCheckpoint: removed old checkpoint {oldest}")
            self._saved_checkpoints.remove(oldest)
    
    @property
    def best_checkpoint_path(self) -> Optional[Path]:
        """Path to the best checkpoint."""
        return self._best_checkpoint_path

    def state_dict(self) -> Dict[str, Any]:
        """Phase DESIGN-1 G-1 (2026-05-10): serialize best-tracking state.

        Returns ``_best_value`` (float) + ``_best_checkpoint_path`` (str
        or None for JSON serializability — Path → str). Deliberately
        EXCLUDES ``_saved_checkpoints`` list (recreated naturally as new
        checkpoints save during the resumed run).
        """
        return {
            "best_value": float(self._best_value),
            "best_checkpoint_path": (
                str(self._best_checkpoint_path)
                if self._best_checkpoint_path is not None
                else None
            ),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Phase DESIGN-1 G-1 (2026-05-10): restore best-tracking state."""
        if not state:
            return
        if "best_value" in state:
            self._best_value = float(state["best_value"])
        if "best_checkpoint_path" in state:
            ckpt_path = state["best_checkpoint_path"]
            self._best_checkpoint_path = (
                Path(ckpt_path) if ckpt_path is not None else None
            )


# =============================================================================
# Logging Callbacks
# =============================================================================


class MetricLogger(Callback):
    """
    Log training metrics to console and/or file.
    
    Args:
        log_every_n_batches: Log batch metrics every N batches (None = only epoch end)
        log_to_file: If True, also write metrics to JSON file
        log_file: Path for JSON log file
    
    Example:
        >>> logger = MetricLogger(log_every_n_batches=100)
    """
    
    def __init__(
        self,
        log_every_n_batches: Optional[int] = None,
        log_to_file: bool = True,
        log_file: Optional[Union[str, Path]] = None,
    ):
        super().__init__()
        
        self.log_every_n_batches = log_every_n_batches
        self.log_to_file = log_to_file
        self.log_file = Path(log_file) if log_file else None
        
        self._history: List[Dict[str, Any]] = []
    
    def on_train_start(self) -> None:
        """Clear history at start of training.

        Phase X.1.K minimum-viable (2026-05-04): when resuming, preserve
        the in-memory history from the prior session — useful for
        continuation runs that want to plot full training curves across
        the resume boundary.
        """
        if (
            self.trainer is not None
            and getattr(self.trainer, "_resumed_from_checkpoint", False)
        ):
            # Resume mode — preserve _history list
            return
        self._history = []
    
    def on_batch_end(self, batch_idx: int, logs: Dict[str, float]) -> None:
        """Log batch metrics periodically."""
        if self.log_every_n_batches is not None:
            if (batch_idx + 1) % self.log_every_n_batches == 0:
                metrics_str = ', '.join(f'{k}={v:.4f}' for k, v in logs.items())
                logger.info(f"Batch {batch_idx + 1}: {metrics_str}")
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, float]) -> None:
        """Log epoch metrics."""
        # Format metrics
        metrics_str = ', '.join(f'{k}={v:.4f}' for k, v in sorted(logs.items()))
        logger.info(f"Epoch {epoch + 1}: {metrics_str}")
        
        # Store history
        self._history.append({
            'epoch': epoch,
            **logs,
        })
    
    def on_train_end(self) -> None:
        """Save history to file."""
        if self.log_to_file and self.log_file is not None:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, 'w') as f:
                json.dump(self._history, f, indent=2)
            logger.info(f"MetricLogger: saved history to {self.log_file}")
    
    @property
    def history(self) -> List[Dict[str, Any]]:
        """Training history as list of dicts."""
        return self._history

    def state_dict(self) -> Dict[str, Any]:
        """Phase DESIGN-1 G-1 (2026-05-10): serialize epoch-history list.

        Useful for cross-process resume of plotting / monitoring tools
        that want to render full training curves spanning the resume
        boundary. ``_history`` is List[Dict] with JSON-native value types
        (epoch int + metric floats) so direct round-trip is safe.
        """
        return {
            "history": list(self._history),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Phase DESIGN-1 G-1 (2026-05-10): restore epoch-history list."""
        if not state:
            return
        if "history" in state:
            self._history = list(state["history"])


class ProgressCallback(Callback):
    """
    Display training progress bar.
    
    Requires tqdm to be installed. Falls back to simple logging if not available.
    
    Args:
        show_epoch_progress: Show progress bar for epochs
        show_batch_progress: Show progress bar for batches within epoch
    """
    
    def __init__(
        self,
        show_epoch_progress: bool = True,
        show_batch_progress: bool = False,
    ):
        super().__init__()
        
        self.show_epoch_progress = show_epoch_progress
        self.show_batch_progress = show_batch_progress
        
        self._epoch_pbar = None
        self._batch_pbar = None
        
        # Check for tqdm
        try:
            from tqdm import tqdm
            self._tqdm = tqdm
        except ImportError:
            self._tqdm = None
            logger.warning("ProgressCallback: tqdm not installed, using simple logging")
    
    def on_train_start(self) -> None:
        """Initialize epoch progress bar."""
        if self._tqdm is None or not self.show_epoch_progress:
            return
        
        if self.trainer is not None:
            total_epochs = self.trainer.config.train.epochs
            self._epoch_pbar = self._tqdm(
                total=total_epochs,
                desc="Training",
                unit="epoch",
            )
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, float]) -> None:
        """Update epoch progress bar."""
        if self._epoch_pbar is not None:
            # Update postfix with metrics
            postfix = {k: f'{v:.4f}' for k, v in logs.items() if 'loss' in k or 'acc' in k}
            self._epoch_pbar.set_postfix(postfix)
            self._epoch_pbar.update(1)
    
    def on_train_end(self) -> None:
        """Close progress bars."""
        if self._epoch_pbar is not None:
            self._epoch_pbar.close()

