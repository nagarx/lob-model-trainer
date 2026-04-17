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
    >>> config = load_config("configs/experiments/nvda_tlob_h10_v1.yaml")
    >>> trainer = Trainer(config)
    >>> trainer.train()
    >>> metrics = trainer.evaluate("test")
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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

from lobtrainer.config import ExperimentConfig, NormalizationStrategy, TaskType
from lobtrainer.data import (
    LOBSequenceDataset,
    load_split_data,
    GlobalZScoreNormalizer,
    HybridNormalizer,
    create_feature_selector,
)
from lobtrainer.training.callbacks import (
    Callback,
    CallbackList,
    EarlyStopping,
    ModelCheckpoint,
    MetricLogger,
)
from lobtrainer.training.metrics import ClassificationMetrics
from lobtrainer.training.strategy import TrainingStrategy, create_strategy
from lobtrainer.utils.reproducibility import set_seed

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
        preloaded_days: Optional[Dict[str, List["DayData"]]] = None,
    ):
        self.config = config
        self._model = model
        self._preloaded_days = preloaded_days  # T11: bypass disk loading for CV folds
        
        # Setup callbacks
        self.callbacks = CallbackList(callbacks or [])
        self.callbacks.set_trainer(self)
        
        # Auto-detect device (CUDA > MPS > CPU)
        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
        self.device = device
        
        # Initialize state
        self.state = TrainingState()
        
        # Lazy initialization
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._scheduler: Optional[Any] = None
        self._train_loader: Optional[DataLoader] = None
        self._val_loader: Optional[DataLoader] = None
        self._test_loader: Optional[DataLoader] = None
        self._strategy: Optional[TrainingStrategy] = None
        
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
    def model_initialized(self) -> bool:
        """True if model has been created (without triggering lazy creation)."""
        return self._model is not None

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        """Get optimizer (creates if not exists)."""
        if self._optimizer is None:
            self._optimizer = self._create_optimizer()
        return self._optimizer

    @property
    def optimizer_initialized(self) -> bool:
        """True if optimizer has been created (without triggering lazy creation)."""
        return self._optimizer is not None

    @property
    def scheduler(self) -> Optional[Any]:
        """Get learning rate scheduler."""
        return self._scheduler

    @property
    def strategy(self) -> Optional[TrainingStrategy]:
        """Get the training strategy (created during setup)."""
        return self._strategy

    def get_loader(self, split: str) -> Optional[DataLoader]:
        """Get the data loader for a split.

        Args:
            split: 'train', 'val', or 'test'.

        Returns:
            DataLoader or None if split not available.
        """
        return getattr(self, f"_{split}_loader", None)
    
    # =========================================================================
    # Setup Methods
    # =========================================================================
    
    def _create_model(self) -> nn.Module:
        """
        Create model from configuration.
        
        Override this method to add custom model types.
        """
        from lobtrainer.models import create_model
        
        seq_len = getattr(self.config.data.sequence, 'window_size', 100)
        model = create_model(self.config.model, sequence_length=seq_len)
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
        """Create optimizer from configuration.

        Supports: 'adamw' (default), 'adam', 'sgd'.
        Configured via config.train.optimizer string field.
        """
        params = self.model.parameters()
        cfg = self.config.train

        if cfg.optimizer == "adamw":
            optimizer = AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        elif cfg.optimizer == "adam":
            optimizer = Adam(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        elif cfg.optimizer == "sgd":
            optimizer = SGD(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay, momentum=0.9)
        else:
            raise ValueError(
                f"Unknown optimizer '{cfg.optimizer}'. Options: 'adamw', 'adam', 'sgd'"
            )

        logger.debug(f"Created optimizer: {cfg.optimizer}(lr={cfg.learning_rate}, wd={cfg.weight_decay})")
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
    
    def _create_dataloaders(self) -> Dict[str, DataLoader]:
        """Create data loaders for all splits with proper task configuration."""
        cfg_data = self.config.data
        cfg_train = self.config.train
        cfg_model = self.config.model
        
        num_workers = cfg_train.num_workers
        # Only force num_workers=0 when using dict-label collation (HMHP)
        if self._strategy is not None and self._strategy.requires_dict_labels:
            if num_workers > 0:
                logger.warning(
                    f"num_workers={num_workers} requested but HMHP dict-label collation "
                    f"is incompatible with multiprocessing. Falling back to num_workers=0."
                )
                num_workers = 0

        # Horizon selection: strategy determines if we return all or one.
        # T9: TrainingStrategy.horizon_idx reads from labels.primary_horizon_idx.
        if self._strategy is not None:
            horizon_idx = self._strategy.horizon_idx
        else:
            horizon_idx = cfg_data.labels.primary_horizon_idx

        # Feature selection configuration
        # Priority order (post-Phase-4-Batch-4c): mutual-exclusion enforced
        # in DataConfig.__post_init__; at most one of the four fields set.
        # The code below turns whichever is set into an `effective_indices`
        # list that the downstream FeatureSelector path uses uniformly.
        feature_indices = None
        feature_selector = None

        from lobtrainer.config import ModelType, DeepLOBMode

        feature_set_name = getattr(cfg_data, 'feature_set', None)
        feature_preset = getattr(cfg_data, 'feature_preset', None)
        config_feature_indices = getattr(cfg_data, 'feature_indices', None)

        # Phase 4 Batch 4c.2: resolve `feature_set` → indices BEFORE the
        # existing preset/indices path. The resolver populates the
        # DataConfig private cache so downstream stages (signal export,
        # ledger record) can retrieve the `FeatureSetRef` without
        # re-resolving. After this block, `config_feature_indices` acts
        # as the single effective-indices conduit — the existing path
        # below handles validation + FeatureSelector construction.
        if feature_set_name is not None:
            if cfg_data.sources is not None:
                raise ValueError(
                    "feature_set is not supported with multi-source mode "
                    "(data.sources). Use per-source feature_indices on "
                    "each SourceConfig instead. (T12 + Phase 4 boundary)"
                )
            from lobtrainer.data.feature_set_resolver import (
                find_feature_sets_dir,
                resolve_feature_set,
            )
            # Phase 4 Batch 4c hardening: (1) explicit .resolve() on
            # data_dir so find_feature_sets_dir's implicit CWD anchoring
            # is never load-bearing; (2) honor user-override via
            # cfg_data.feature_sets_dir; (3) pass expected_contract_version
            # from hft_contracts SSoT so contract-version drift between
            # producer and consumer is caught deterministically rather
            # than depending on source_feature_count as an indirect guard.
            if cfg_data.feature_sets_dir is not None:
                registry_dir = Path(cfg_data.feature_sets_dir).resolve()
            else:
                registry_dir = find_feature_sets_dir(
                    Path(cfg_data.data_dir).resolve()
                )
            from hft_contracts import SCHEMA_VERSION as _CURRENT_CONTRACT_VERSION
            resolved = resolve_feature_set(
                name=feature_set_name,
                registry_dir=registry_dir,
                expected_source_feature_count=cfg_data.feature_count,
                expected_contract_version=_CURRENT_CONTRACT_VERSION,
            )
            # Populate the DataConfig private cache. Propagates to
            # signal_metadata.json + ExperimentRecord in Batch 4c.4.
            cfg_data._feature_indices_resolved = list(resolved.feature_indices)
            cfg_data._feature_set_ref_resolved = (
                resolved.name, resolved.content_hash,
            )
            # Thread through the existing indices-driven flow.
            config_feature_indices = list(resolved.feature_indices)
            logger.info(
                f"FeatureSet resolved: '{resolved.name}' → "
                f"{len(resolved.feature_indices)} / "
                f"{resolved.source_feature_count} features "
                f"(contract_version={resolved.contract_version}, "
                f"content_hash={resolved.content_hash[:16]}...)"
            )

        # T12: forbid single-source feature selection in multi-source mode
        if cfg_data.sources is not None and (feature_preset is not None or config_feature_indices is not None):
            raise ValueError(
                "feature_preset and feature_indices are not supported with "
                "multi-source mode (data.sources). Use DayBundle.to_fused_day_data "
                "with per-source feature_indices instead."
            )

        if feature_preset is not None or config_feature_indices is not None:
            feature_selector = create_feature_selector(
                preset=feature_preset,
                indices=config_feature_indices,
                source_feature_count=cfg_data.feature_count,
            )
            if feature_selector is not None:
                feature_indices = list(feature_selector.indices)
                logger.info(
                    f"Feature selection from config: {feature_selector.name} "
                    f"({feature_selector.output_size} features from {cfg_data.feature_count})"
                )

                if cfg_model.input_size != feature_selector.output_size:
                    raise ValueError(
                        f"Model input_size ({cfg_model.input_size}) does not match "
                        f"selected feature count ({feature_selector.output_size}). "
                        f"Set model.input_size: {feature_selector.output_size} in config."
                    )

        elif cfg_model.model_type == ModelType.DEEPLOB:
            deeplob_mode = getattr(cfg_model, 'deeplob_mode', DeepLOBMode.BENCHMARK)
            if deeplob_mode == DeepLOBMode.BENCHMARK:
                from lobtrainer.constants import LOB_FEATURE_COUNT
                feature_indices = list(range(LOB_FEATURE_COUNT))
                logger.info(
                    f"DeepLOB benchmark mode: selecting first {LOB_FEATURE_COUNT} LOB features"
                )

        # Label transform for binary signal detection (task_type driven, not strategy)
        label_transform = None
        num_classes = 3
        task_type = getattr(cfg_train, 'task_type', TaskType.MULTICLASS)

        if task_type == TaskType.BINARY_SIGNAL:
            from lobtrainer.data import BinaryLabelTransform
            label_transform = BinaryLabelTransform()
            num_classes = 2
            logger.info("Binary signal detection mode: converting 3-class -> 2-class labels")

        # Strategy-driven data format properties
        return_labels_as_dict = (
            self._strategy.requires_dict_labels if self._strategy is not None else False
        )
        return_regression_targets = (
            self._strategy.requires_regression_targets if self._strategy is not None else False
        )

        if return_labels_as_dict:
            logger.info(
                f"HMHP mode: returning all horizons as dict labels "
                f"(horizons={cfg_model.hmhp_horizons})"
            )

        # Labeling strategy: derive from LabelsConfig.task for label-shift dispatch.
        # T9: labels_config.task replaces legacy cfg_data.labeling_strategy.
        labels_task = cfg_data.labels.task
        if labels_task == "regression":
            labeling_strategy = "regression"
        elif labels_task == "classification":
            labeling_strategy = getattr(cfg_data, 'labeling_strategy', None)
            if hasattr(labeling_strategy, 'value'):
                labeling_strategy = labeling_strategy.value
        else:
            # task="auto": let _determine_label_shift_from_metadata resolve
            labeling_strategy = getattr(cfg_data, 'labeling_strategy', None)
            if hasattr(labeling_strategy, 'value'):
                labeling_strategy = labeling_strategy.value

        loaders = self._create_dataloaders_with_transform(
            data_dir=cfg_data.data_dir,
            batch_size=cfg_train.batch_size,
            num_workers=num_workers,
            pin_memory=cfg_train.pin_memory and torch.cuda.is_available(),
            horizon_idx=horizon_idx,
            feature_indices=feature_indices,
            label_transform=label_transform,
            num_classes=num_classes,
            return_labels_as_dict=return_labels_as_dict,
            labeling_strategy=labeling_strategy,
            return_regression_targets=return_regression_targets,
        )

        return loaders
    
    def _create_dataloaders_with_transform(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        horizon_idx: Optional[int],
        feature_indices: Optional[List[int]],
        label_transform: Optional[callable],
        num_classes: int,
        return_labels_as_dict: bool = False,
        labeling_strategy: Optional[str] = None,
        return_regression_targets: bool = False,
    ) -> Dict[str, DataLoader]:
        """
        Create dataloaders with custom label transform support.
        
        Memory-efficient approach:
        1. Create/load cached normalization stats FIRST (without loading all data)
        2. Then load data with lazy/mmap mode for memory efficiency
        3. Apply cached normalization during training
        
        This allows training on datasets that don't fit in RAM.
        """
        loaders = {}
        all_days = {}
        
        # Create feature transform based on normalization strategy
        # IMPORTANT: Do this BEFORE loading all data to avoid OOM
        feature_transform = None
        norm_strategy = self.config.data.normalization.strategy

        # T11: warn about normalization approximation in CV mode
        if (
            self._preloaded_days is not None
            and norm_strategy != NormalizationStrategy.NONE
        ):
            logger.warning(
                "CV mode: using cached normalization stats from %s. "
                "For exact per-fold normalization, set "
                "normalization.strategy: none",
                data_dir,
            )

        # T12: multi-source mode requires normalization: none
        # (stats from single-source dir are wrong for fused feature tensor)
        if (
            self.config.data.sources is not None
            and norm_strategy != NormalizationStrategy.NONE
        ):
            raise ValueError(
                f"Multi-source mode (data.sources) requires "
                f"normalization.strategy='none', got '{norm_strategy.value}'. "
                f"Per-source normalization is not yet supported for fused "
                f"tensors. Set data.normalization.strategy: none in your config."
            )

        if norm_strategy == NormalizationStrategy.GLOBAL_ZSCORE:
            # Global Z-score matching TLOB repository
            num_features = self.config.data.feature_count
            stats_path = Path(data_dir) / "normalization_stats.json"
            
            if stats_path.exists():
                # Fast path: load cached stats
                from lobtrainer.data.normalization import GlobalNormalizationStats
                logger.info(f"Loading cached normalization stats from {stats_path}")
                stats = GlobalNormalizationStats.load(stats_path)
                feature_transform = GlobalZScoreNormalizer(
                    stats, 
                    eps=self.config.data.normalization.eps
                )
            else:
                # Slow path: compute stats (uses streaming internally)
                logger.info(f"Computing GlobalZScoreNormalizer stats for {num_features} features...")
                # Load training data with lazy loading for stats computation
                train_days_for_stats = load_split_data(
                    data_dir, "train",
                    labels_config=self.config.data.labels,
                    validate=False, lazy=True,
                )
                feature_transform = GlobalZScoreNormalizer.from_train_data(
                    train_days_for_stats,
                    num_features=num_features,
                    eps=self.config.data.normalization.eps,
                )
                feature_transform.stats.save(stats_path)
                logger.info(f"Saved normalization stats to {stats_path}")
                del train_days_for_stats  # Free memory
        
        elif norm_strategy == NormalizationStrategy.HYBRID:
            # Hybrid normalization for 98-feature datasets
            num_features = self.config.data.feature_count
            stats_path = Path(data_dir) / "hybrid_normalization_stats.json"
            
            if stats_path.exists():
                # Fast path: load cached stats (instant, no data loading)
                from lobtrainer.data.normalization import HybridNormalizationStats
                logger.info(f"Loading cached hybrid normalization stats from {stats_path}")
                stats = HybridNormalizationStats.load(stats_path)
                feature_transform = HybridNormalizer(
                    stats,
                    eps=self.config.data.normalization.eps,
                    clip_value=self.config.data.normalization.clip_value,
                )
            else:
                # Slow path: compute stats using memory-efficient streaming
                logger.info(f"Computing HybridNormalizer stats for {num_features} features...")
                logger.info("  (This may take a few minutes on first run, cached for future runs)")
                from lobtrainer.data.normalization import compute_hybrid_stats_streaming
                stats = compute_hybrid_stats_streaming(
                    data_dir,
                    num_features=num_features,
                    eps=self.config.data.normalization.eps,
                    clip_value=self.config.data.normalization.clip_value,
                )
                feature_transform = HybridNormalizer(
                    stats,
                    eps=self.config.data.normalization.eps,
                    clip_value=self.config.data.normalization.clip_value,
                )
                logger.info(f"Cached normalization stats at {stats_path}")
        
        # NOW load all splits (after normalization is ready)
        # T11: preloaded_days bypass — for CV, days are pre-split by CVTrainer.
        if self._preloaded_days is not None:
            all_days = dict(self._preloaded_days)
            logger.info(
                "Using preloaded days (CV mode): %s",
                {k: len(v) for k, v in all_days.items()},
            )
        elif self.config.data.sources is not None:
            # T12: Multi-source fusion — load from multiple sources
            from lobtrainer.data.bundle import load_split_bundles
            from lobtrainer.data.sources import DataSource

            t12_sources = [
                DataSource(name=s.name, data_dir=s.data_dir, role=s.role)
                for s in self.config.data.sources
            ]
            logger.info(
                "Multi-source mode: %s",
                [(s.name, s.role) for s in t12_sources],
            )
            for split in ["train", "val", "test"]:
                try:
                    all_days[split] = load_split_bundles(
                        t12_sources, split,
                        labels_config=self.config.data.labels,
                    )
                except (FileNotFoundError, ValueError) as e:
                    logger.info(f"Split '{split}' not available: {e}")
                    continue
        else:
            # Standard single-source path: load from disk for each split
            logger.info("Loading training data...")
            for split in ["train", "val", "test"]:
                try:
                    all_days[split] = load_split_data(
                        data_dir, split,
                        labels_config=self.config.data.labels,
                        validate=True, lazy=False,
                    )
                except FileNotFoundError:
                    logger.info(f"Split '{split}' not found, skipping")
                    continue
        
        if "train" not in all_days:
            raise ValueError("No training data found")
        
        # Import collate function for HMHP dict labels
        from lobtrainer.data.dataset import _hmhp_collate_fn
        
        # Detect precomputed regression labels
        use_precomputed = return_regression_targets and any(
            day.regression_labels is not None
            for days_list in all_days.values()
            for day in days_list
        )
        if use_precomputed:
            logger.info(
                "Using precomputed regression labels from regression_labels.npy"
            )

        # T10: check if sample weights should be returned (train split only)
        _has_weights = any(
            day.sample_weights is not None
            for days_list in all_days.values()
            for day in days_list
        )

        # Create datasets with transform
        for split, days in all_days.items():
            dataset = LOBSequenceDataset(
                days,
                transform=feature_transform,
                feature_indices=feature_indices,
                horizon_idx=horizon_idx,
                label_transform=label_transform,
                num_classes=num_classes,
                return_labels_as_dict=return_labels_as_dict,
                labeling_strategy=labeling_strategy,
                return_regression_targets=return_regression_targets,
                use_precomputed_regression=use_precomputed,
                return_sample_weights=(_has_weights and split == "train"),
            )
            
            # Use custom collate function for HMHP dict labels
            collate_fn = _hmhp_collate_fn if return_labels_as_dict else None
            
            loaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == "train"),
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=(split == "train"),
                collate_fn=collate_fn,
            )
        
        return loaders
    
    def setup(self) -> None:
        """
        Full setup: model, optimizer, scheduler, strategy, data loaders.

        Call this explicitly if you want to setup before training,
        otherwise it's called automatically by train().
        """
        # Set seed for reproducibility
        set_seed(self.config.train.seed)

        # Create components (lazy properties)
        _ = self.model
        _ = self.optimizer
        self._scheduler = self._create_scheduler(self.optimizer)

        # Create strategy BEFORE data loaders (strategy properties drive data loading)
        self._strategy = create_strategy(self.config, self.device)

        # Create data loaders (strategy properties determine label format)
        loaders = self._create_dataloaders()
        self._train_loader = loaders.get('train')
        self._val_loader = loaders.get('val')
        self._test_loader = loaders.get('test')

        if self._train_loader is None:
            raise ValueError("No training data found")

        # Strategy one-time initialization (e.g., criterion creation for classification)
        self._strategy.initialize(self._train_loader, self.model)

        logger.info(
            f"Setup complete: strategy={type(self._strategy).__name__}, "
            f"train={len(self._train_loader)} batches, "
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

        Delegates batch processing to the strategy while managing the
        optimization loop (zero_grad, backward, clip, step).

        Returns:
            Dict with training metrics (train_loss, and strategy-specific metrics).
        """
        self.model.train()
        cfg = self.config.train

        results = []
        total_samples = 0

        for batch_idx, batch_data in enumerate(self._train_loader):
            self.callbacks.on_batch_start(batch_idx)

            self.optimizer.zero_grad()
            result = self._strategy.process_batch(self.model, batch_data)
            result.loss.backward()

            if cfg.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    cfg.gradient_clip_norm,
                )

            self.optimizer.step()
            self.state.global_step += 1

            results.append(result)
            total_samples += result.batch_size

            self.callbacks.on_batch_end(batch_idx, {'loss': result.loss.item()})

        if total_samples == 0:
            return {'train_loss': 0.0}

        return self._strategy.aggregate_epoch_metrics(results, total_samples)
    
    def _validate(self) -> Dict[str, float]:
        """
        Run validation via strategy.

        Delegates to self._strategy.validate(). Kept as a method for
        backward compatibility with tests that call _validate() directly.

        Returns:
            Dict with 'val_loss' and strategy-specific validation metrics.
        """
        return self._strategy.validate(self.model, self._val_loader)

    # =========================================================================
    # Evaluation
    # =========================================================================
    
    @torch.no_grad()
    def evaluate(
        self,
        split: str = 'test',
        loader: Optional[DataLoader] = None,
    ) -> Union[ClassificationMetrics, Dict[str, Any]]:
        """
        Evaluate model on a data split via strategy.

        Args:
            split: Data split to evaluate ('train', 'val', 'test')
            loader: Optional custom DataLoader (uses split if None)

        Returns:
            ClassificationMetrics for classification strategies, or Dict for
            regression strategies with per-horizon regression metrics.
        """
        if loader is None:
            loader = self.get_loader(split)

        if loader is None:
            if self._train_loader is None:
                self.setup()
            loader = self.get_loader(split)

        if loader is None:
            raise ValueError(f"No data available for split: {split}")

        return self._strategy.evaluate(self.model, loader, split)

    def predict(
        self,
        features: Union[np.ndarray, torch.Tensor],
        return_proba: bool = False,
    ) -> np.ndarray:
        """
        Make predictions on new data via strategy.

        Args:
            features: Feature array [batch, seq_len, features] or [batch, features]
            return_proba: If True, return class probabilities (classification only)

        Returns:
            Predictions [batch] or probabilities [batch, num_classes]
        """
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float()

        features = features.to(self.device)

        with torch.no_grad():
            return self._strategy.predict(self.model, features, return_proba)
    
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

