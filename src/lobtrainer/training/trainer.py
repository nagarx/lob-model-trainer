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
from lobtrainer.training.simple_trainer import SimpleModelTrainer
from lobtrainer.training.strategy import TrainingStrategy, create_strategy
from lobtrainer.utils.reproducibility import set_seed

logger = logging.getLogger(__name__)


# =============================================================================
# Phase 1 N7 helper — normalization stats path derivation
# =============================================================================


def _derive_normalization_stats_path(config: ExperimentConfig) -> Optional[Path]:
    """Derive the on-disk path of the normalization stats file from the active config.

    Phase 1 N7 (#PY-10, 2026-05-06): consolidates the strategy → filename
    mapping that ``Trainer.setup()`` open-codes at trainer.py:619-682
    (GLOBAL_ZSCORE → ``normalization_stats.json``; HYBRID → ``hybrid_normalization_stats.json``).
    Returns ``None`` when no stats file is produced — ``NormalizationStrategy.NONE``,
    multi-source mode (``data_dir`` is None), or future strategies that don't
    cache stats to disk.

    Used by:
      * ``_build_checkpoint_dict``: hash the file (if present) into the
        checkpoint so resume-time validation can detect data-stats drift
        (sibling to Phase X.1 v2 F-13 closure of CONFIG drift).
      * ``load_checkpoint``: rebuild the path to validate against the embedded SHA.

    Args:
        config: Active ``ExperimentConfig`` (NOT the checkpoint's saved config).

    Returns:
        Path to the strategy-specific stats file (whether or not it exists),
        or ``None`` when the active strategy doesn't write a stats file.
    """
    data_dir = config.data.data_dir
    if data_dir is None:
        return None  # multi-source mode (config.data.sources is set instead)

    norm_strategy = config.data.normalization.strategy
    if norm_strategy == NormalizationStrategy.GLOBAL_ZSCORE:
        return Path(data_dir) / "normalization_stats.json"
    elif norm_strategy == NormalizationStrategy.HYBRID:
        return Path(data_dir) / "hybrid_normalization_stats.json"
    else:
        return None  # NormalizationStrategy.NONE or future unsupported strategies


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
        callbacks_list = list(callbacks or [])

        # Phase 8C-α Integration Close-Out Round-3 post-audit Agent-4 H1
        # fix: auto-register PermutationImportanceCallback here in
        # `__init__` (NOT in `create_trainer`) so the scripts/train.py +
        # hft-ops subprocess path — which EXPLICITLY passes callbacks=
        # and bypasses `create_trainer`'s default-list builder — still
        # gets the callback wired. Also covers CVTrainer which calls
        # `Trainer(fold_config, ...)` directly. Precedence rule:
        # user-supplied PermutationImportanceCallback wins (duck-typed
        # class-name check to avoid pulling torch in the config path).
        if getattr(config, "importance", None) is not None:
            user_has_importance_cb = any(
                type(cb).__name__ == "PermutationImportanceCallback"
                for cb in callbacks_list
            )
            if not user_has_importance_cb:
                try:
                    from lobtrainer.training.importance.callback import (
                        PermutationImportanceCallback,
                    )
                    callbacks_list.append(
                        PermutationImportanceCallback(config.importance)
                    )
                except Exception:
                    # Defensive: missing torch / circular import at
                    # callback-module load would otherwise break
                    # unrelated trainer construction. Log and continue
                    # without the callback (observation-tier failure
                    # should not kill training per §8).
                    import logging as _log
                    _log.getLogger(__name__).warning(
                        "PermutationImportanceCallback auto-registration "
                        "failed; training will proceed without importance "
                        "audit. Check lobtrainer.training.importance.callback "
                        "import + torch availability."
                    )

        # Phase X.3 Empirical Trust (2026-05-05): auto-register
        # TrainingDiagnostics(alert_on_nan=True) as a post-hoc audit safety
        # net for non-finite loss. The PRIMARY guard is the direct check at
        # _train_epoch:902 (PRE-backward); this callback check runs AFTER
        # optimizer.step() (post-hoc) but provides defense-in-depth if the
        # direct guard is bypassed by Trainer subclasses overriding
        # _train_epoch. Same exception type (TrainingDivergedError) +
        # uniform error contract across both guard sites — see
        # lobtrainer.training.exceptions module docstring for the dual-
        # guard rationale. Same registration pattern as
        # PermutationImportanceCallback above (duck-typed user-already-has-it
        # check; narrow except clause).
        user_has_diagnostics_cb = any(
            type(cb).__name__ == "TrainingDiagnostics"
            for cb in callbacks_list
        )
        if not user_has_diagnostics_cb:
            try:
                from lobtrainer.training.monitoring import TrainingDiagnostics
                callbacks_list.append(TrainingDiagnostics(alert_on_nan=True))
            except (ImportError, OSError):
                # Defensive: monitoring import is normally fine (torch is
                # already loaded by trainer); fall back to direct guard
                # only. Narrow except per hft-rules §5 (avoid masking
                # unexpected errors).
                import logging as _log
                _log.getLogger(__name__).warning(
                    "TrainingDiagnostics auto-registration failed; "
                    "training will proceed with the direct loss-finiteness "
                    "guard at _train_epoch only (no post-hoc audit fallback)."
                )

        self.callbacks = CallbackList(callbacks_list)
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

        # Phase X.1.K minimum-viable: resume signal flag. Set to True at the end
        # of load_checkpoint(); reset to False at the end of train(). Consumed
        # by callbacks' on_train_start to skip resetting in-process state
        # (EarlyStopping patience, ModelCheckpoint best_value snapshot).
        self._resumed_from_checkpoint: bool = False
        
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

        # Phase X.3 Empirical Trust (2026-05-05) — Phase C.1: auto-derive
        # ``data.labels.horizons`` from the export's ``*_horizons.json``
        # files when the YAML doesn't override. Pre-Phase-C.1, the empty
        # default fell back to ``model.hmhp_horizons = (10,20,50,100,200)``
        # in ``compatibility.py:233`` — classification defaults that did
        # NOT match the regression corpus's actual horizons [10,60,300].
        # This caused B5 horizon drift in 5 of 6 v3p0 stages' compat_fp.
        # Per hft-rules §1 ("layout as contract — single source of truth"):
        # the export IS authoritative; trainer auto-derives if not overridden.
        if not self.config.data.labels.horizons:
            from lobtrainer.data.horizons_resolver import resolve_horizons_from_export

            try:
                actual_horizons = resolve_horizons_from_export(
                    self.config.data.data_dir, split="train"
                )
                # LabelsConfig is frozen; use Pydantic model_copy to
                # construct a new instance with horizons populated.
                new_labels = self.config.data.labels.model_copy(
                    update={"horizons": actual_horizons}
                )
                new_data = self.config.data.model_copy(
                    update={"labels": new_labels}
                )
                self.config = self.config.model_copy(
                    update={"data": new_data}
                )
                logger.info(
                    f"Auto-resolved data.labels.horizons={list(actual_horizons)} "
                    f"from {self.config.data.data_dir}/train/*_horizons.json "
                    f"(Phase X.3 / Phase C.1 truth-pinning — was previously "
                    f"falling back to model.hmhp_horizons classification defaults)."
                )
            except FileNotFoundError as exc:
                # Classification-only exports (no horizons.json) — silent
                # pass-through; CompatibilityContract construction will
                # fail-loud at fp time if horizons still empty.
                logger.debug(
                    f"Horizons auto-resolution skipped: {exc}. "
                    f"If running classification, set data.labels.horizons "
                    f"explicitly OR ensure CompatibilityContract.horizons "
                    f"is provided by some other path."
                )

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
        # Phase X.1 v2 post-validation fix (Agent 1 Q6 2026-05-04):
        # Wrap entire body in try/finally so the `_resumed_from_checkpoint`
        # flag-reset is unconditional, even when setup() / config-access
        # raises BEFORE on_train_start. Pre-fix, a setup() failure would
        # leak the True flag into the next train() call.
        start_time = time.time()
        try:
            # Setup if not already done
            if self._train_loader is None:
                self.setup()

            cfg = self.config.train
            self.state.training_started = True

            # Notify callbacks
            self.callbacks.on_train_start()

            # Phase 1 N2 forensic-bug closure (#PY-10, 2026-05-06):
            # When --resume loads a checkpoint, load_checkpoint sets
            # self.state.current_epoch to the checkpoint's epoch index.
            # Pre-fix this loop started at 0 unconditionally, wiping resume
            # progress + causing duplicate per-epoch state writes. Now starts
            # at the resumed epoch index (default 0 for fresh training).
            start_epoch = self.state.current_epoch
            remaining_epochs = cfg.epochs - start_epoch
            if remaining_epochs <= 0:
                logger.info(
                    f"Resume already at or past target ({start_epoch}/{cfg.epochs} "
                    f"epochs); nothing to do"
                )
            else:
                logger.info(
                    f"Starting training: epochs {start_epoch}..{cfg.epochs - 1} "
                    f"({remaining_epochs} remaining of {cfg.epochs} total)"
                )
            for epoch in range(start_epoch, cfg.epochs):
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
            # Phase X.1.K: consume the resumed-from-checkpoint flag. A subsequent
            # train() call (without re-load) resets callback state normally.
            self._resumed_from_checkpoint = False
        
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

            # Phase X.3 Empirical Trust (2026-05-05): fail-loud guard on
            # non-finite loss BEFORE backward propagates NaN/Inf into model
            # parameters. PRIMARY defense — direct + pre-backward placement
            # ensures gradient + param state stays clean. Stage 4 GMADL
            # near-collapse (pred_std=7.7e-5) was caught only because IC was
            # visibly near 0; future loss-explosions / sigmoid-saturation /
            # log-of-zero / exp-overflow scenarios would have produced NaN
            # parameters + later NaN val_loss + EarlyStopping silent stall
            # without this guard. Per hft-rules §2 ("zero tolerance for
            # precision errors") + §8 ("never silently drop, clamp, or 'fix'
            # data"). Caught by 2026-05-05 multi-agent audit (Cluster V).
            #
            # Companion post-hoc audit: TrainingDiagnostics.on_batch_end
            # raises the SAME exception type (TrainingDivergedError) for
            # defense-in-depth — see lobtrainer.training.exceptions module
            # docstring for the dual-guard rationale.
            if not torch.isfinite(result.loss).all():
                from lobtrainer.training.exceptions import TrainingDivergedError
                raise TrainingDivergedError(
                    epoch=self.state.current_epoch,
                    batch=batch_idx,
                    loss_value=float(result.loss.item()),
                    global_step=self.state.global_step,
                )

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
    
    def _build_checkpoint_dict(
        self,
        *,
        epoch_override: Optional[int] = None,
        metrics_override: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Build the canonical checkpoint dict.

        Phase X.1 v2 (2026-05-04): centralizes the checkpoint contract so
        Trainer.save_checkpoint AND ModelCheckpoint._save_checkpoint emit
        IDENTICAL keys. Eliminates the pre-X.1 3-writer divergence (per
        Agent 1 sanity-check Q9 finding).

        Includes 3 Phase X.1 v2 contract keys:
          * ``compatibility``: full CompatibilityContract dict (data-side
            architecture: feature_count, window_size, horizons,
            primary_horizon_idx, normalization_strategy, label_strategy_hash,
            feature_layout, data_source, calibration_method, contract_version,
            schema_version).
          * ``compatibility_fingerprint``: 64-hex SHA-256 over the
            CompatibilityContract canonical form.
          * ``model_config_hash``: 64-hex SHA-256 over filtered model.params
            (loss-tuning keys excluded per ``_LOSS_TUNING_KEYS`` denylist).

        Phase 1 N7 (#PY-10, 2026-05-06): adds 4th contract key for the
        data-stats plane (sibling to Phase X.1 v2's CONFIG-drift closure):
          * ``normalization_stats_sha256``: 64-hex SHA-256 over the
            active normalization stats JSON file (or ``None`` when no
            stats file is produced — NormalizationStrategy.NONE,
            multi-source mode, or stats not yet computed). Hashed via
            ``hft_contracts.provenance.hash_file`` SSoT. ``load_checkpoint``
            validates against the active config to detect data re-extraction
            silently changing per-day stats; pre-N7 checkpoints lacking
            this key silently skip the check (back-compat preserved).

        Args:
            epoch_override: Used by ModelCheckpoint callback to record per-epoch
                'epoch' separate from trainer's current_epoch state.
            metrics_override: Used by ModelCheckpoint callback to record
                per-epoch metrics dict from on_epoch_end logs.

        Returns:
            Checkpoint dict (NOT yet saved). Caller adds 'scheduler_state_dict'
            if scheduler exists, then torch.save's the result.
        """
        from lobtrainer.training.compatibility import (
            build_compatibility_contract,
            compute_model_config_hash,
        )

        compat = build_compatibility_contract(self.config)
        model_cfg_hash = compute_model_config_hash(self.config.model)

        # Phase 1 N7 (#PY-10, 2026-05-06): hash the active normalization stats
        # file (if produced by the active strategy) and embed in the checkpoint
        # so load_checkpoint can detect data-stats drift on resume. SIBLING
        # closure to Phase X.1 v2 F-13 (CONFIG drift) — N7 closes the data-stats
        # surface (re-extracting the dataset with different per-day stats
        # silently changes inference behavior). REUSES
        # hft_contracts.provenance.hash_file SSoT per #PY-41 (do NOT inline
        # raw hashlib.sha256 — convention locked by Phase V.1.5 consolidation).
        from hft_contracts.provenance import hash_file
        norm_stats_path = _derive_normalization_stats_path(self.config)
        norm_stats_sha: Optional[str] = None
        if norm_stats_path is not None and norm_stats_path.exists():
            norm_stats_sha = hash_file(norm_stats_path, missing_ok=False)
        # else: graceful — no stats file (NONE strategy, multi-source mode, or
        # stats file not yet computed). load_checkpoint treats this None as
        # "no binding to validate" (consistent with the X.1 v2 missing-key
        # back-compat pattern: pre-N7 checkpoints lack the key entirely; this
        # silently skips validation rather than raising).

        # Phase X.1 v2 post-validation fix (Agent 1 Q1 + Q1b 2026-05-04):
        # - `self.optimizer` is a lazy property (line 237) that CONSTRUCTS an
        #   optimizer on demand. Using it in `is not None` would always trigger
        #   construction (which requires `_create_optimizer` to succeed; this
        #   side-effect can fail in test bypass paths). Use `self._optimizer`
        #   (the underlying private attr) directly for the None-check.
        # - When `_optimizer is None`, write None into the dict (NOT call
        #   `.state_dict()` on None which would AttributeError). The mirror
        #   guard in `load_checkpoint` skips `load_state_dict(None)`.
        return {
            'epoch': epoch_override if epoch_override is not None else self.state.current_epoch,
            'global_step': self.state.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': (
                self._optimizer.state_dict() if self._optimizer is not None else None
            ),
            'config': self.config.to_dict(),
            'state': {
                'best_val_metric': self.state.best_val_metric,
                'best_epoch': self.state.best_epoch,
            },
            'metrics': metrics_override or {},  # callback supplies per-epoch metrics
            # Phase X.1 v2: 3 NEW keys (additive, back-compat preserved — pre-X.1
            # checkpoints simply lack them; load_checkpoint emits
            # CheckpointMissingFingerprintWarning in that case).
            'compatibility': compat.to_canonical_dict() if compat is not None else None,
            'compatibility_fingerprint': compat.fingerprint() if compat is not None else None,
            'model_config_hash': model_cfg_hash,
            # Phase 1 N7 (#PY-10, 2026-05-06): NEW key — additive, back-compat
            # preserved. Pre-N7 checkpoints lack this key; load_checkpoint
            # treats absence as "no binding" (silent skip). Post-N7 checkpoints
            # carry None when no stats file is produced (NONE strategy etc.)
            # OR a 64-hex SHA-256 when a stats file exists.
            'normalization_stats_sha256': norm_stats_sha,
        }

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """
        Save model checkpoint.

        Phase X.1 v2 (2026-05-04): embeds CompatibilityContract +
        compatibility_fingerprint + model_config_hash via
        ``_build_checkpoint_dict`` shared helper. Pre-X.1 v2 callers see
        the 3 new keys as additive — back-compat preserved.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = self._build_checkpoint_dict()
        if self._scheduler is not None:
            checkpoint['scheduler_state_dict'] = self._scheduler.state_dict()

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(
        self,
        path: Union[str, Path],
        load_optimizer: bool = True,
        strict_config: bool = False,
    ) -> None:
        """
        Load model from checkpoint.

        Phase X.1 v2 (2026-05-04): validates the checkpoint's
        ``compatibility_fingerprint`` + ``model_config_hash`` against the
        active config. Default ``strict_config=False`` emits
        ``CheckpointConfigMismatchWarning`` on mismatch (warn-only mode);
        ``strict_config=True`` promotes to ``CheckpointConfigMismatchError``.
        Pre-X.1 checkpoints lacking the contract fields emit
        ``CheckpointMissingFingerprintWarning`` and bypass validation
        entirely (operators must re-train or opt-in via strict).

        Phase X.4 will flip the default to ``strict_config=True`` once all
        in-flight checkpoints have fingerprints — gated by:
          (a) zero CheckpointMissingFingerprintWarning across CI runs for 2 weeks
          (b) PHASE_P_BACKLOG.md F-12 retrain item closed
          (c) audit script reports 100% coverage

        Args:
            path: Path to checkpoint
            load_optimizer: Whether to load optimizer state
            strict_config: When True, raise on fingerprint mismatch instead
                of warn. Default False (warn-only) per Phase X.4 promotion plan.
        """
        import warnings
        from lobtrainer.training.compatibility import (
            build_compatibility_contract,
            compute_model_config_hash,
            CheckpointConfigMismatchError,
            CheckpointConfigMismatchWarning,
            CheckpointMissingFingerprintWarning,
        )

        checkpoint = torch.load(path, map_location=self.device)

        # Phase X.1 v2: 3-way contract validation BEFORE model_state_dict load.
        # Reads checkpoint['compatibility_fingerprint'] + ['model_config_hash'].
        # Pre-X.1 checkpoints lack both keys → CheckpointMissingFingerprintWarning.
        ckpt_compat_dict = checkpoint.get('compatibility')
        ckpt_compat_fingerprint = checkpoint.get('compatibility_fingerprint')
        ckpt_model_cfg_hash = checkpoint.get('model_config_hash')

        if ckpt_compat_fingerprint is None and ckpt_model_cfg_hash is None:
            warnings.warn(
                f"Checkpoint at {path} lacks Phase X.1 v2 contract fields "
                f"('compatibility_fingerprint', 'model_config_hash'). "
                f"Pre-X.1 artifact — cannot validate against active config. "
                f"Re-train or opt-in via strict_config=True to raise.",
                CheckpointMissingFingerprintWarning,
                stacklevel=2,
            )
        else:
            active_compat = build_compatibility_contract(self.config)
            active_model_cfg_hash = compute_model_config_hash(self.config.model)

            # Compatibility-contract mismatch (data-side architecture)
            if (
                ckpt_compat_fingerprint is not None
                and active_compat is not None
                and ckpt_compat_fingerprint != active_compat.fingerprint()
            ):
                # Reconstruct ckpt_compat for actionable diff()
                from hft_contracts.compatibility import CompatibilityContract
                try:
                    # Phase X.1 v2 sanity-check fix: CompatibilityContract has no
                    # from_dict classmethod — reconstruct via direct **dict expansion.
                    # Frozen @dataclass __post_init__ runs validation + horizons coercion.
                    ckpt_compat = CompatibilityContract(**(ckpt_compat_dict or {}))
                    diff = active_compat.diff(ckpt_compat)
                    diff_msg = (
                        f"Checkpoint compatibility mismatch at {path}.\n"
                        f"  Differing fields (active vs checkpoint): {diff}\n"
                        f"  Likely cause: feature_count, window_size, horizons, "
                        f"primary_horizon_idx, normalization_strategy, label_strategy, "
                        f"or feature_layout differs between checkpoint and active config."
                    )
                except Exception as exc:
                    # Reconstruction failed (corrupt dict shape) — fall back to opaque hash msg
                    diff_msg = (
                        f"Checkpoint compatibility_fingerprint mismatch at {path}.\n"
                        f"  Checkpoint: {ckpt_compat_fingerprint[:16]}...\n"
                        f"  Active:     {active_compat.fingerprint()[:16]}...\n"
                        f"  (could not reconstruct CompatibilityContract for diff: {exc})"
                    )
                if strict_config:
                    raise CheckpointConfigMismatchError(diff_msg)
                else:
                    warnings.warn(
                        diff_msg, CheckpointConfigMismatchWarning, stacklevel=2,
                    )

            # Model architecture hash mismatch (model-side)
            if (
                ckpt_model_cfg_hash is not None
                and ckpt_model_cfg_hash != active_model_cfg_hash
            ):
                msg = (
                    f"Checkpoint model_config_hash mismatch at {path}.\n"
                    f"  Checkpoint: {ckpt_model_cfg_hash[:16]}...\n"
                    f"  Active:     {active_model_cfg_hash[:16]}...\n"
                    f"  Likely cause: model_type, hidden_dim, num_layers, num_heads, "
                    f"dropout, hmhp_pool_mode, attention, use_bin, num_classes, "
                    f"or other architectural keys differ.\n"
                    f"  Loss-tuning keys (gmadl_a/b, regression_loss_*, loss_weights) "
                    f"are excluded from this hash."
                )
                if strict_config:
                    raise CheckpointConfigMismatchError(msg)
                else:
                    warnings.warn(
                        msg, CheckpointConfigMismatchWarning, stacklevel=2,
                    )

        # Phase 1 N7 (#PY-10, 2026-05-06): validate normalization stats SHA —
        # closes the data-stats drift hazard. SIBLING to Phase X.1 v2 F-13
        # closure (CONFIG drift). Pre-N7 checkpoints lack this key → silent
        # skip (consistent with the X.1 v2 missing-key back-compat pattern;
        # CheckpointMissingFingerprintWarning above already covers fully
        # pre-X.1 artifacts). Defense-in-depth: if normalization strategy
        # changed, compatibility_fingerprint already mismatches at the check
        # above — N7 here adds DATA-side detection that compat_fingerprint
        # can't see (same strategy + same config but different file contents).
        ckpt_norm_sha = checkpoint.get('normalization_stats_sha256')
        if ckpt_norm_sha is not None:  # post-N7 checkpoint with binding
            from hft_contracts.provenance import hash_file
            active_stats_path = _derive_normalization_stats_path(self.config)
            n7_msg: Optional[str] = None
            if active_stats_path is None:
                # Active config produces no stats file but checkpoint expected
                # one. Strategy divergence — SHOULD already have been caught
                # by compat_fingerprint above, but guard explicitly here per
                # defense-in-depth (compat_fingerprint may be None/missing for
                # pre-X.1-but-post-N7 hypothetical artifacts).
                n7_msg = (
                    f"Checkpoint at {path} embeds normalization_stats_sha256="
                    f"{ckpt_norm_sha[:16]}... but the active config produces no "
                    f"stats file (strategy={self.config.data.normalization.strategy}). "
                    f"Likely cause: normalization strategy changed between train and resume."
                )
            elif not active_stats_path.exists():
                n7_msg = (
                    f"Checkpoint at {path} expects normalization stats file at "
                    f"{active_stats_path} (sha256={ckpt_norm_sha[:16]}...) but the "
                    f"file does not exist. Re-extract the dataset (or restore the "
                    f"missing stats file) before resuming."
                )
            else:
                active_norm_sha = hash_file(active_stats_path, missing_ok=False)
                if active_norm_sha != ckpt_norm_sha:
                    n7_msg = (
                        f"Normalization stats SHA mismatch at {path}.\n"
                        f"  Checkpoint: {ckpt_norm_sha[:16]}...\n"
                        f"  Active:     {active_norm_sha[:16]}...\n"
                        f"  Active stats file: {active_stats_path}\n"
                        f"  Likely cause: data was re-extracted with different "
                        f"normalization (per-day stats changed). Either re-train OR "
                        f"pin the checkpoint's data."
                    )
            if n7_msg is not None:
                if strict_config:
                    raise CheckpointConfigMismatchError(n7_msg)
                else:
                    warnings.warn(
                        n7_msg, CheckpointConfigMismatchWarning, stacklevel=2,
                    )

        # Existing model + optimizer + scheduler load (preserved verbatim).
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Phase X.1 v2 post-validation fix (Agent 1 Q1): guard against
        # `optimizer_state_dict: None` from a checkpoint built when
        # `_optimizer` was None at save time. PyTorch's load_state_dict(None)
        # raises TypeError. Three conditions: (a) caller wants optimizer load,
        # (b) checkpoint actually has a non-None optimizer state, (c) target
        # trainer has an optimizer to load into.
        if (
            load_optimizer
            and checkpoint.get('optimizer_state_dict') is not None
            and self._optimizer is not None
        ):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint and self._scheduler is not None:
            self._scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Restore state
        self.state.current_epoch = checkpoint.get('epoch', 0)
        self.state.global_step = checkpoint.get('global_step', 0)

        if 'state' in checkpoint:
            self.state.best_val_metric = checkpoint['state'].get('best_val_metric', float('inf'))
            self.state.best_epoch = checkpoint['state'].get('best_epoch', 0)

        # Phase X.1.K minimum-viable: signal that this is a resume so callbacks
        # can preserve state across the load_checkpoint → train() boundary.
        # The flag is reset back to False at the end of train() (consumed once).
        self._resumed_from_checkpoint = True

        logger.info(f"Loaded checkpoint from {path} (epoch {self.state.current_epoch})")

    def export_signals(
        self,
        split: str = "test",
        *,
        output_dir: Optional[Path] = None,
        calibration: str = "none",
    ) -> Path:
        """Export predicted signals + ``signal_metadata.json`` for a split.

        Phase Q.6.5.B (2026-05-04 night): satisfies the ``BaseTrainer``
        Protocol method by delegating to the existing
        ``lobtrainer.export.exporter.SignalExporter`` wrapper. Closes the
        Q1 asymmetry (sklearn ``SimpleModelTrainer`` had a direct method;
        PyTorch ``Trainer`` required the operator to manually instantiate
        ``SignalExporter`` AND call ``setup()`` + ``load_checkpoint()`` in
        the right order).

        After this method exists, ``scripts/export_signals.py`` can
        collapse to a thin ``create_trainer + setup + load_checkpoint +
        export_signals`` wrapper (Phase Q.6.5.B Part 2 closes F-16 by
        replacing the direct ``Trainer(config)`` instantiation that broke
        sklearn dispatch through the canonical script).

        Args:
            split: Data split — ``"val"`` or ``"test"``. Training split is
                refused by ``SignalExporter`` (DataLoader uses
                ``drop_last=True``; alignment mismatch with raw features).
            output_dir: Override default output directory. ``None`` uses
                ``<self.config.output_dir>/signals/<split>/``.
            calibration: Calibration strategy. ``"none"`` (default) emits
                raw predictions. ``"variance_match"`` rescales to match
                per-horizon label std (regression-only — emits WARN +
                no-op when the inference's ``signal_type`` is
                ``"classification"``, see exporter.py:646-662).

        Returns:
            Output directory path (``Path``). Use this to drive downstream
            backtest invocations.

        Raises:
            ValueError: For invalid split (``"train"``) or unknown
                calibration strategy. Both come from ``SignalExporter``.
            RuntimeError: If the trainer's DataLoader for ``split`` is
                ``None`` (caller did not invoke ``setup()`` first).
        """
        # Lazy-import to avoid circular module dependency.
        # exporter.py imports Trainer for type hints; trainer.py importing
        # exporter at module-load time would close the cycle.
        from lobtrainer.export.exporter import SignalExporter

        exporter = SignalExporter(self, calibration=calibration)
        result = exporter.export(split=split, output_dir=output_dir)
        return result.output_dir


# =============================================================================
# Factory Function
# =============================================================================


def create_trainer(
    config: Union[str, Path, ExperimentConfig],
    **kwargs,
):
    """
    Create a trainer from a config file or object, dispatched on the
    model's registered ``framework`` field.

    Phase Q.5 (2026-05-04): closes the dispatch fault line where
    sklearn-registered models (``framework="sklearn"``) silently fell
    through to the PyTorch ``Trainer`` and failed at
    ``model.parameters()``. Now ``ModelRegistry.get(name).framework``
    is inspected and the appropriate trainer is returned. Both
    ``Trainer`` (PyTorch) and ``SimpleModelTrainer`` (sklearn) satisfy
    the ``BaseTrainer`` Protocol so callers (``scripts/train.py``,
    ``hft-ops``) can use a uniform interface.

    Args:
        config: Path to config file or ExperimentConfig object.
        **kwargs: Additional arguments passed to the concrete trainer.
            For sklearn dispatch, a ``callbacks`` kwarg is dropped
            with an INFO log (sklearn trainer doesn't run callbacks).

    Returns:
        Concrete trainer satisfying ``BaseTrainer``: ``Trainer`` for
        PyTorch frameworks, ``SimpleModelTrainer`` for sklearn.

    Raises:
        ValueError: If the registered framework is neither ``"pytorch"``
            nor ``"sklearn"``.

    Example:
        >>> trainer = create_trainer("configs/lstm.yaml")  # → Trainer
        >>> trainer = create_trainer("configs/temporal_ridge.yaml")  # → SimpleModelTrainer
        >>> trainer.train()
    """
    from lobtrainer.config import load_config

    if isinstance(config, (str, Path)):
        config = load_config(str(config))

    # Resolve the registered framework. Default "pytorch" for unknown
    # models so the existing fallback (Trainer raises a descriptive
    # error at model construction) is preserved.
    framework = "pytorch"
    try:
        from lobmodels import ModelRegistry  # type: ignore
        model_name = config.model.name  # property → registry key
        framework = ModelRegistry.get(model_name).framework
    except (KeyError, ImportError, AttributeError):
        # Unknown model name OR registry not importable OR
        # config.model.name unresolvable — fall through to PyTorch.
        # This preserves prior behavior for any unusual config that
        # bypassed the registry.
        framework = "pytorch"

    if framework == "sklearn":
        # Sklearn trainers run a one-shot fit; PyTorch callbacks
        # (early-stopping, checkpoint, metric-logger, progress) don't
        # apply. Drop them with a single INFO log so the caller sees
        # what was discarded.
        if "callbacks" in kwargs:
            dropped = kwargs.pop("callbacks")
            n = len(dropped) if dropped is not None else 0
            logger.info(
                "Dropping %d callback(s) for sklearn trainer "
                "(framework=%s, model=%s)",
                n, framework, getattr(config.model, "name", "?"),
            )
        return SimpleModelTrainer.from_config(config, **kwargs)

    if framework == "pytorch":
        # Existing default-callbacks block — preserved verbatim for
        # PyTorch back-compat.
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
            # Phase 8C-α Integration Close-Out Round-3 post-audit note:
            # PermutationImportanceCallback auto-registration moved to
            # `Trainer.__init__` (see callback-registration block there).
            # Moving it earlier ensures `scripts/train.py` path (which
            # explicitly passes `callbacks=...` kwarg and bypasses this
            # block entirely) still gets the callback wired, and covers
            # CVTrainer which calls Trainer(fold_config, ...) directly.
            kwargs['callbacks'] = callbacks
        return Trainer(config, **kwargs)

    raise ValueError(
        f"Unknown framework '{framework}' for model "
        f"'{getattr(config.model, 'name', '?')}'. Expected one of "
        f"'pytorch', 'sklearn'. Update the @register decorator on the "
        f"model class to specify a supported framework."
    )

