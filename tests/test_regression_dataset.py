"""
Tests for regression label loading in the dataset module.

Validates:
- DayData.regression_labels field
- load_day_data with regression_labels.npy
- load_split_data with pure regression (no _labels.npy)
- LOBSequenceDataset with use_precomputed_regression=True
"""

import json
import numpy as np
import pytest
import tempfile
from pathlib import Path

from lobtrainer.data.dataset import (
    DayData,
    load_day_data,
    load_split_data,
    LOBSequenceDataset,
)


NUM_SEQS = 50
SEQ_LEN = 100
NUM_FEATURES = 98
NUM_HORIZONS = 3
HORIZONS = [10, 60, 300]


def _create_synthetic_export(
    tmpdir: Path,
    split: str = "train",
    dates: list = None,
    include_class_labels: bool = True,
    include_reg_labels: bool = True,
):
    """Create a synthetic export directory mimicking feature extractor output."""
    if dates is None:
        dates = ["2025-03-01", "2025-03-02"]

    split_dir = tmpdir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    for date in dates:
        seqs = np.random.randn(NUM_SEQS, SEQ_LEN, NUM_FEATURES).astype(np.float32)
        np.save(split_dir / f"{date}_sequences.npy", seqs)

        if include_class_labels:
            labels = np.random.choice([-1, 0, 1], size=(NUM_SEQS, NUM_HORIZONS)).astype(np.int8)
            np.save(split_dir / f"{date}_labels.npy", labels)

        if include_reg_labels:
            reg_labels = np.random.randn(NUM_SEQS, NUM_HORIZONS).astype(np.float64)
            np.save(split_dir / f"{date}_regression_labels.npy", reg_labels)

        metadata = {
            "n_features": NUM_FEATURES,
            "n_sequences": NUM_SEQS,
            "schema_version": "2.2",
            "label_strategy": "tlob",
            "labeling": {
                "horizons": HORIZONS,
                "label_mode": "regression",
            },
        }
        with open(split_dir / f"{date}_metadata.json", "w") as f:
            json.dump(metadata, f)


class TestLoadDayDataRegression:
    def test_load_with_both_labels(self, tmp_path):
        _create_synthetic_export(tmp_path, include_class_labels=True, include_reg_labels=True)
        split_dir = tmp_path / "train"
        day = load_day_data(
            split_dir / "2025-03-01_sequences.npy",
            split_dir / "2025-03-01_labels.npy",
            regression_labels_path=split_dir / "2025-03-01_regression_labels.npy",
        )
        assert day.labels is not None
        assert day.regression_labels is not None
        assert day.regression_labels.shape == (NUM_SEQS, NUM_HORIZONS)
        assert day.regression_labels.dtype == np.float64

    def test_load_pure_regression_no_class_labels(self, tmp_path):
        _create_synthetic_export(tmp_path, include_class_labels=False, include_reg_labels=True)
        split_dir = tmp_path / "train"
        day = load_day_data(
            split_dir / "2025-03-01_sequences.npy",
            split_dir / "2025-03-01_labels.npy",
            regression_labels_path=split_dir / "2025-03-01_regression_labels.npy",
        )
        assert day.regression_labels is not None
        assert day.regression_labels.shape == (NUM_SEQS, NUM_HORIZONS)
        assert day.labels.shape[0] == NUM_SEQS

    def test_no_labels_at_all_raises(self, tmp_path):
        _create_synthetic_export(tmp_path, include_class_labels=False, include_reg_labels=False)
        split_dir = tmp_path / "train"
        with pytest.raises(FileNotFoundError, match="No labels found"):
            load_day_data(
                split_dir / "2025-03-01_sequences.npy",
                split_dir / "2025-03-01_labels.npy",
            )


class TestLoadSplitDataRegression:
    def test_load_split_with_regression(self, tmp_path):
        _create_synthetic_export(tmp_path, include_class_labels=True, include_reg_labels=True)
        days = load_split_data(tmp_path, "train", validate=False)
        assert len(days) == 2
        for day in days:
            assert day.regression_labels is not None
            assert day.regression_labels.shape == (NUM_SEQS, NUM_HORIZONS)

    def test_load_split_pure_regression(self, tmp_path):
        _create_synthetic_export(tmp_path, include_class_labels=False, include_reg_labels=True)
        days = load_split_data(tmp_path, "train", validate=False)
        assert len(days) == 2
        for day in days:
            assert day.regression_labels is not None


class TestLOBSequenceDatasetPrecomputedRegression:
    def _make_dataset(self, tmp_path, precomputed=True):
        _create_synthetic_export(tmp_path, include_class_labels=True, include_reg_labels=True)
        days = load_split_data(tmp_path, "train", validate=False)
        return LOBSequenceDataset(
            days,
            horizon_idx=None,
            return_labels_as_dict=True,
            return_regression_targets=True,
            use_precomputed_regression=precomputed,
            labeling_strategy="regression",
        )

    def test_returns_three_tuple(self, tmp_path):
        ds = self._make_dataset(tmp_path)
        item = ds[0]
        assert len(item) == 3, f"Expected 3-tuple, got {len(item)}-tuple"
        seq, labels, reg_targets = item
        assert seq.dim() == 2
        assert isinstance(labels, dict)
        assert isinstance(reg_targets, dict)

    def test_regression_targets_from_file(self, tmp_path):
        ds = self._make_dataset(tmp_path)
        _, _, reg_targets = ds[0]
        assert len(reg_targets) == NUM_HORIZONS

    def test_missing_regression_labels_raises(self, tmp_path):
        _create_synthetic_export(tmp_path, include_class_labels=True, include_reg_labels=False)
        days = load_split_data(tmp_path, "train", validate=False)
        with pytest.raises(ValueError, match="no regression_labels"):
            LOBSequenceDataset(
                days,
                horizon_idx=None,
                return_labels_as_dict=True,
                return_regression_targets=True,
                use_precomputed_regression=True,
            )


class TestPureRegressionDictMode:
    """Test pure regression mode (no _labels.npy) with dict label returns.

    This covers the label shape bug: placeholder labels must be 2D [N, num_horizons]
    so that is_multi_horizon detection works and label_dict building doesn't crash.
    """

    def test_pure_regression_dict_mode_works(self, tmp_path):
        _create_synthetic_export(tmp_path, include_class_labels=False, include_reg_labels=True)
        days = load_split_data(tmp_path, "train", validate=False)

        for day in days:
            assert day.labels.ndim == 2, (
                f"Placeholder labels should be 2D, got shape {day.labels.shape}"
            )
            assert day.labels.shape == day.regression_labels.shape, (
                f"Placeholder labels shape {day.labels.shape} != "
                f"regression_labels shape {day.regression_labels.shape}"
            )
            assert day.is_multi_horizon, "Pure regression with multi-horizon should be detected"

        ds = LOBSequenceDataset(
            days,
            horizon_idx=None,
            return_labels_as_dict=True,
            return_regression_targets=True,
            use_precomputed_regression=True,
            labeling_strategy="regression",
        )

        seq, label_dict, reg_dict = ds[0]
        assert isinstance(label_dict, dict)
        assert isinstance(reg_dict, dict)
        assert len(label_dict) == NUM_HORIZONS
        assert len(reg_dict) == NUM_HORIZONS

    def test_pure_regression_placeholder_labels_are_zero(self, tmp_path):
        _create_synthetic_export(tmp_path, include_class_labels=False, include_reg_labels=True)
        days = load_split_data(tmp_path, "train", validate=False)

        for day in days:
            assert (day.labels == 0).all(), "Placeholder labels should all be zero"
