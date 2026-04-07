"""
End-to-end integration tests for standard model regression training.

Tests that DeepLOB can be trained in regression mode through the full
trainer pipeline, validating the unified dual-task design.
"""

import json
import numpy as np
import pytest
import torch
from pathlib import Path

from hft_contracts import SCHEMA_VERSION


NUM_SEQS = 30
SEQ_LEN = 100
NUM_FEATURES = 98
MODEL_INPUT = 40
NUM_HORIZONS = 3
HORIZONS = [10, 60, 300]


def _create_regression_export(base_dir: Path):
    """Create synthetic regression export for DeepLOB benchmark (40 features)."""
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
                "label_strategy": "tlob",
                "label_dtype": "float64",
                "tensor_format": None,
                "labeling": {
                    "label_mode": "regression",
                    "horizons": HORIZONS,
                    "num_horizons": NUM_HORIZONS,
                    "label_encoding": {
                        "format": "continuous_bps",
                        "dtype": "float64",
                    },
                },
                "label_distribution": {"Down": 10, "Stable": 10, "Up": 10},
                "label_encoding": {
                    "format": "signed_int8",
                    "values": {"-1": "Down", "0": "Stable", "1": "Up"},
                },
                "export_timestamp": "2026-03-15T12:00:00Z",
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
                    "export_timestamp_utc": "2026-03-15T12:00:00Z",
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

            # Also create dummy classification labels (required by load_split_data)
            cls_labels = np.random.choice([-1, 0, 1], size=(NUM_SEQS, NUM_HORIZONS)).astype(np.int8)
            np.save(split_dir / f"{date}_labels.npy", cls_labels)


def _make_deeplob_regression_config(data_dir: str, output_dir: str) -> dict:
    return {
        "name": "test_deeplob_regression",
        "data": {
            "data_dir": data_dir,
            "feature_count": NUM_FEATURES,
            "labeling_strategy": "tlob",
            "num_classes": 3,
            "horizon_idx": 1,
            "feature_preset": "lob_only",
            "sequence": {"window_size": SEQ_LEN, "stride": 10},
            "normalization": {"strategy": "none"},
        },
        "model": {
            "model_type": "deeplob",
            "input_size": MODEL_INPUT,
            "num_classes": 3,
            "task_type": "regression",
            "dropout": 0.0,
            "deeplob_mode": "benchmark",
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


class TestDeepLOBRegressionTraining:

    def test_deeplob_regression_training_runs(self, tmp_path):
        data_dir = tmp_path / "data"
        output_dir = tmp_path / "output"
        _create_regression_export(data_dir)

        from lobtrainer.config.schema import ExperimentConfig
        from lobtrainer.training.trainer import Trainer

        config_dict = _make_deeplob_regression_config(str(data_dir), str(output_dir))
        config = ExperimentConfig.from_dict(config_dict)
        trainer = Trainer(config)
        result = trainer.train()

        assert "train_loss" in result or "history" in result

    def test_deeplob_regression_validation_produces_metrics(self, tmp_path):
        data_dir = tmp_path / "data"
        output_dir = tmp_path / "output"
        _create_regression_export(data_dir)

        from lobtrainer.config.schema import ExperimentConfig
        from lobtrainer.training.trainer import Trainer

        config_dict = _make_deeplob_regression_config(str(data_dir), str(output_dir))
        config = ExperimentConfig.from_dict(config_dict)
        trainer = Trainer(config)
        trainer.setup()

        train_metrics = trainer._train_epoch()
        assert "train_loss" in train_metrics
        assert "train_accuracy" not in train_metrics, "Regression should not compute accuracy"

        if trainer._val_loader is not None:
            val_metrics = trainer._validate()
            assert "val_loss" in val_metrics
            assert np.isfinite(val_metrics["val_loss"])

    def test_deeplob_regression_predict(self, tmp_path):
        data_dir = tmp_path / "data"
        output_dir = tmp_path / "output"
        _create_regression_export(data_dir)

        from lobtrainer.config.schema import ExperimentConfig
        from lobtrainer.training.trainer import Trainer

        config_dict = _make_deeplob_regression_config(str(data_dir), str(output_dir))
        config = ExperimentConfig.from_dict(config_dict)
        trainer = Trainer(config)
        trainer.setup()

        x = np.random.randn(4, SEQ_LEN, MODEL_INPUT).astype(np.float32)
        preds = trainer.predict(x)
        assert preds.ndim == 1, f"Regression predictions should be 1D, got {preds.ndim}D"
        assert preds.shape[0] == 4
        assert np.all(np.isfinite(preds))
