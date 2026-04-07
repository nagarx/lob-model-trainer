"""
SimpleModelTrainer: training loop for non-PyTorch (sklearn-style) models.

Handles the full pipeline for simple models:
    1. Load sequences and labels from exported .npy files
    2. Engineer temporal features via TemporalFeatureConfig
    3. Train model via fit()
    4. Evaluate on val/test splits
    5. Export predictions for backtester
    6. Save model checkpoint

Output format matches the PyTorch trainer so the backtester works unchanged.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from lobmodels.features.temporal import TemporalFeatureConfig, engineer_temporal_features
from lobmodels.models.simple import (
    BaseSimpleModel,
    TemporalRidge,
    TemporalRidgeConfig,
    TemporalGradBoost,
    TemporalGradBoostConfig,
)

logger = logging.getLogger(__name__)

try:
    from hft_contracts import SIGNAL_PRICE_FEATURE_INDEX as MID_PRICE_IDX
    from hft_contracts import SIGNAL_SPREAD_FEATURE_INDEX as SPREAD_BPS_IDX
except ImportError:
    MID_PRICE_IDX = 40
    SPREAD_BPS_IDX = 42


def _load_split(data_dir: Path, split: str, horizon_idx: int = 0, max_days: int = None):
    """Load sequences and regression labels for one split."""
    split_dir = data_dir / split
    meta_files = sorted(split_dir.glob("*_metadata.json"))
    if max_days:
        meta_files = meta_files[:max_days]

    all_seqs, all_labels, all_spreads, all_prices = [], [], [], []
    for mf in meta_files:
        with open(mf) as f:
            m = json.load(f)
        day = m["day"]
        seq = np.load(split_dir / f"{day}_sequences.npy", mmap_mode="r")
        reg = np.load(split_dir / f"{day}_regression_labels.npy")
        all_seqs.append(seq)
        all_labels.append(reg[:, horizon_idx])
        all_spreads.append(seq[:, -1, SPREAD_BPS_IDX].astype(np.float64))
        all_prices.append(seq[:, -1, MID_PRICE_IDX].astype(np.float64))

    return (
        np.concatenate(all_seqs, axis=0),
        np.concatenate(all_labels, axis=0),
        np.concatenate(all_spreads, axis=0),
        np.concatenate(all_prices, axis=0),
    )


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics matching the PyTorch trainer.

    Delegates to hft-metrics via the regression_metrics adapter (Rule 0: no duplication).
    """
    from lobtrainer.training.regression_metrics import compute_all_regression_metrics

    return compute_all_regression_metrics(y_true, y_pred)


class SimpleModelTrainer:
    """Training pipeline for non-PyTorch models.

    Mirrors the PyTorch Trainer's output format so the backtester,
    signal export, and experiment tracking work unchanged.

    Args:
        config: Parsed experiment configuration dict or object with
                data_dir, model_type, horizon_idx, output_dir fields.
    """

    def __init__(
        self,
        data_dir: str,
        model_type: str,
        model_config: dict,
        feature_config: dict = None,
        horizon_idx: int = 0,
        output_dir: str = "outputs/experiments/simple_model",
    ):
        self.data_dir = Path(data_dir)
        self.model_type = model_type
        self.model_config = model_config
        self.feature_config_dict = feature_config or {}
        self.horizon_idx = horizon_idx
        self.output_dir = Path(output_dir)

        self.model: Optional[BaseSimpleModel] = None
        self.feat_config: Optional[TemporalFeatureConfig] = None
        self.train_metrics: Dict[str, float] = {}
        self.val_metrics: Dict[str, float] = {}
        self.test_metrics: Dict[str, float] = {}

    def setup(self):
        """Load data, create model, engineer features."""
        logger.info("Loading data...")
        self._seq_train, self._y_train, self._spreads_train, self._prices_train = \
            _load_split(self.data_dir, "train", self.horizon_idx)
        self._seq_val, self._y_val, _, _ = \
            _load_split(self.data_dir, "val", self.horizon_idx)
        self._seq_test, self._y_test, self._spreads_test, self._prices_test = \
            _load_split(self.data_dir, "test", self.horizon_idx)

        logger.info(f"Train: {len(self._y_train):,}, Val: {len(self._y_val):,}, Test: {len(self._y_test):,}")

        self.feat_config = TemporalFeatureConfig(**self.feature_config_dict) \
            if self.feature_config_dict else TemporalFeatureConfig()

        logger.info(f"Engineering {self.feat_config.num_features} temporal features...")
        self._X_train = engineer_temporal_features(self._seq_train, self.feat_config)
        self._X_val = engineer_temporal_features(self._seq_val, self.feat_config)
        self._X_test = engineer_temporal_features(self._seq_test, self.feat_config)

        if self.model_type == "temporal_ridge":
            cfg = TemporalRidgeConfig(
                alpha=self.model_config.get("alpha", 1.0),
                feature_config=self.feat_config,
            )
            self.model = TemporalRidge(cfg)
        elif self.model_type == "temporal_gradboost":
            cfg = TemporalGradBoostConfig(
                n_estimators=self.model_config.get("n_estimators", 200),
                max_depth=self.model_config.get("max_depth", 5),
                learning_rate=self.model_config.get("learning_rate", 0.05),
                subsample=self.model_config.get("subsample", 0.8),
                min_samples_leaf=self.model_config.get("min_samples_leaf", 50),
                loss_type=self.model_config.get("loss_type", "huber"),
                huber_delta=self.model_config.get("huber_delta", 0.9),
                max_train_samples=self.model_config.get("max_train_samples", 50000),
                feature_config=self.feat_config,
            )
            self.model = TemporalGradBoost(cfg)
        else:
            raise ValueError(f"Unknown simple model type: {self.model_type}")

        logger.info(f"Model: {self.model.name}")

    def train(self) -> Dict[str, float]:
        """Fit the model and compute train + val metrics."""
        t0 = time.time()
        self.model.fit(self._X_train, self._y_train)
        fit_time = time.time() - t0

        y_pred_train = self.model.predict(self._X_train)
        self.train_metrics = _compute_metrics(self._y_train, y_pred_train)
        self.train_metrics["fit_time_seconds"] = round(fit_time, 2)

        y_pred_val = self.model.predict(self._X_val)
        self.val_metrics = _compute_metrics(self._y_val, y_pred_val)

        logger.info(
            f"Train: R²={self.train_metrics['r2']:.4f}, IC={self.train_metrics['ic']:.4f} | "
            f"Val: R²={self.val_metrics['r2']:.4f}, IC={self.val_metrics['ic']:.4f} | "
            f"Fit: {fit_time:.1f}s"
        )
        return self.val_metrics

    def evaluate(self) -> Dict[str, float]:
        """Evaluate on the test split."""
        y_pred = self.model.predict(self._X_test)
        self.test_metrics = _compute_metrics(self._y_test, y_pred)
        return self.test_metrics

    def export_signals(self, split: str = "test") -> Path:
        """Export predictions in the backtester-compatible format."""
        if split == "test":
            X, y, spreads, prices = self._X_test, self._y_test, self._spreads_test, self._prices_test
        else:
            raise ValueError(f"Only 'test' split export supported, got '{split}'")

        y_pred = self.model.predict(X)

        signal_dir = self.output_dir / "signals" / split
        signal_dir.mkdir(parents=True, exist_ok=True)

        np.save(signal_dir / "predicted_returns.npy", y_pred.astype(np.float64))
        np.save(signal_dir / "regression_labels.npy", y.astype(np.float64))
        np.save(signal_dir / "spreads.npy", spreads)
        np.save(signal_dir / "prices.npy", prices)

        metadata = {
            "model_type": self.model_type,
            "model_name": self.model.name,
            "parameters": self.model.num_parameters,
            "total_samples": int(len(y_pred)),
            "split": split,
            "horizon_idx": self.horizon_idx,
            "feature_config": self.feat_config.to_dict(),
            "metrics": {k: round(v, 6) for k, v in self.test_metrics.items()},
        }
        with open(signal_dir / "signal_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Exported {len(y_pred):,} signals to {signal_dir}")
        return signal_dir

    def save(self):
        """Save model checkpoint, metrics, and config."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir = self.output_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)

        self.model.save(ckpt_dir / "best.pkl")

        if self.test_metrics:
            with open(self.output_dir / "test_metrics.json", "w") as f:
                json.dump({f"test_{k}": v for k, v in self.test_metrics.items()}, f, indent=2)

        history = {
            "model_type": self.model_type,
            "model_name": self.model.name,
            "parameters": self.model.num_parameters,
            "train_metrics": self.train_metrics,
            "val_metrics": self.val_metrics,
            "test_metrics": self.test_metrics,
            "feature_config": self.feat_config.to_dict(),
        }
        with open(self.output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        logger.info(f"Saved to {self.output_dir}")
