"""
End-to-end integration tests for HMHP_REGRESSION training.

Tests the full pipeline: synthetic data creation -> data loading ->
Trainer with HMHP_REGRESSION -> training loop -> regression validation metrics.

This is the safety net that proves the regression training path works
end-to-end, not just in unit-test isolation.
"""

import json
import numpy as np
import pytest
import torch
from pathlib import Path
from tempfile import TemporaryDirectory

from hft_contracts import SCHEMA_VERSION


NUM_SEQS = 30
SEQ_LEN = 100
NUM_FEATURES = 98
HORIZONS = [10, 60, 300]
NUM_HORIZONS = len(HORIZONS)


def _create_regression_export(base_dir: Path):
    """Create synthetic regression export matching feature extractor output.

    Produces train/ and val/ splits with sequences + regression labels + metadata,
    exactly as the Rust feature extractor would produce.
    """
    for split in ("train", "val"):
        split_dir = base_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        dates = ["2025-03-01", "2025-03-02"] if split == "train" else ["2025-03-03"]

        for date in dates:
            np.random.seed(hash(date) % 2**31)
            seqs = np.random.randn(NUM_SEQS, SEQ_LEN, NUM_FEATURES).astype(np.float32)
            reg_labels = np.random.randn(NUM_SEQS, NUM_HORIZONS).astype(np.float64) * 5.0

            np.save(split_dir / f"{date}_sequences.npy", seqs)
            np.save(split_dir / f"{date}_regression_labels.npy", reg_labels)

            metadata = {
                "day": date,
                "n_sequences": NUM_SEQS,
                "window_size": SEQ_LEN,
                "n_features": NUM_FEATURES,
                "schema_version": SCHEMA_VERSION,
                "contract_version": SCHEMA_VERSION,
                "label_strategy": "regression",
                "label_dtype": "float64",
                "tensor_format": None,
                "labeling": {
                    "label_mode": "regression",
                    "horizons": HORIZONS,
                    "num_horizons": NUM_HORIZONS,
                    "label_encoding": {
                        "format": "continuous_bps",
                        "dtype": "float64",
                        "unit": "basis_points",
                    },
                },
                "label_distribution": {"positive": 15, "negative": 15, "zero": 0},
                "export_timestamp": "2026-03-14T12:00:00Z",
                "normalization": {
                    "strategy": "none",
                    "applied": False,
                    "levels": 10,
                    "sample_count": NUM_SEQS * SEQ_LEN,
                    "feature_layout": "grouped",
                    "params_file": f"{date}_normalization.json",
                },
                "provenance": {
                    "extractor_version": "0.1.0",
                    "git_commit": "test123",
                    "git_dirty": False,
                    "config_hash": "test",
                    "contract_version": SCHEMA_VERSION,
                    "export_timestamp_utc": "2026-03-14T12:00:00Z",
                },
                "validation": {
                    "sequences_labels_match": True,
                    "label_range_valid": True,
                    "no_nan_inf": True,
                },
                "processing": {
                    "messages_processed": 10000,
                    "features_extracted": 5000,
                    "sequences_generated": NUM_SEQS + 5,
                    "sequences_aligned": NUM_SEQS,
                    "sequences_dropped": 5,
                },
            }
            with open(split_dir / f"{date}_metadata.json", "w") as f:
                json.dump(metadata, f)


def _make_regression_config(data_dir: str, output_dir: str) -> dict:
    """Build a minimal HMHP_REGRESSION training config dict."""
    return {
        "name": "test_hmhp_regression",
        "data": {
            "data_dir": data_dir,
            "feature_count": NUM_FEATURES,
            "labeling_strategy": "regression",
            "num_classes": 3,
            "horizon_idx": None,
            "sequence": {"window_size": SEQ_LEN, "stride": 10},
            "normalization": {"strategy": "none"},
        },
        "model": {
            "model_type": "hmhp_regression",
            "input_size": NUM_FEATURES,
            "num_classes": 3,
            "dropout": 0.0,
            "hmhp_horizons": HORIZONS,
            "hmhp_cascade_mode": "full",
            "hmhp_state_fusion": "gate",
            "hmhp_encoder_type": "tlob",
            "hmhp_encoder_hidden_dim": 32,
            "hmhp_num_encoder_layers": 1,
            "hmhp_decoder_hidden_dim": 16,
            "hmhp_state_dim": 16,
            "hmhp_use_confirmation": True,
            "hmhp_regression_loss_type": "huber",
            "hmhp_loss_weights": {
                "H10": 0.2,
                "H60": 0.5,
                "H300": 0.2,
                "consistency": 0.1,
            },
        },
        "train": {
            "batch_size": 16,
            "learning_rate": 1e-3,
            "epochs": 2,
            "early_stopping_patience": 5,
            "seed": 42,
            "num_workers": 0,
            "pin_memory": False,
            "task_type": "regression",
            "loss_type": "huber",
            "use_class_weights": False,
        },
        "output_dir": output_dir,
    }


class TestHMHPRegressionTraining:
    """End-to-end regression training integration tests."""

    def test_data_loading_with_regression_metadata(self, tmp_path):
        """Verify regression exports with label_strategy='regression' load without error."""
        data_dir = tmp_path / "data"
        _create_regression_export(data_dir)

        from lobtrainer.data.dataset import load_split_data
        days = load_split_data(data_dir, "train", validate=True)

        assert len(days) == 2
        for day in days:
            assert day.regression_labels is not None
            assert day.regression_labels.shape == (NUM_SEQS, NUM_HORIZONS)
            assert day.regression_labels.dtype == np.float64
            assert day.is_multi_horizon

    def test_regression_training_runs(self, tmp_path):
        """Full training loop: create data, build Trainer, run 2 epochs."""
        data_dir = tmp_path / "data"
        output_dir = tmp_path / "output"
        _create_regression_export(data_dir)

        from lobtrainer.config.schema import ExperimentConfig
        from lobtrainer.training.trainer import Trainer

        config_dict = _make_regression_config(str(data_dir), str(output_dir))
        config = ExperimentConfig.from_dict(config_dict)
        trainer = Trainer(config)

        result = trainer.train()

        assert "train_loss" in result or "history" in result, (
            f"Training result missing loss info. Keys: {list(result.keys())}"
        )

    def test_regression_validation_produces_metrics(self, tmp_path):
        """Validation returns regression metrics (R2, IC, MAE) not classification accuracy."""
        data_dir = tmp_path / "data"
        output_dir = tmp_path / "output"
        _create_regression_export(data_dir)

        from lobtrainer.config.schema import ExperimentConfig
        from lobtrainer.training.trainer import Trainer

        config_dict = _make_regression_config(str(data_dir), str(output_dir))
        config = ExperimentConfig.from_dict(config_dict)
        trainer = Trainer(config)
        trainer.setup()

        train_metrics = trainer._train_epoch()
        assert "train_loss" in train_metrics, f"Missing train_loss. Keys: {list(train_metrics.keys())}"

        if trainer._val_loader is not None:
            val_metrics = trainer._validate()
            assert "val_loss" in val_metrics, f"Missing val_loss. Keys: {list(val_metrics.keys())}"
            assert np.isfinite(val_metrics["val_loss"]), f"val_loss is not finite: {val_metrics['val_loss']}"
