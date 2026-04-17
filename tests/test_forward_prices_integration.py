"""Tests for the forward-prices bridge (Phase T9).

Covers:
    - LabelsConfig dataclass validation
    - DataConfig auto-derivation from legacy fields
    - ExperimentConfig deprecation warnings
    - (T9-D) _compute_labels_from_forward_prices function
    - (T9-E) Forward prices loading integration
    - (T9-E) Dacite compatibility for Optional[LabelsConfig]

Reference: plan/EXPERIMENTATION_FIRST_ARCHITECTURE.md §15
"""

import warnings
from dataclasses import dataclass

import numpy as np
import pytest

from lobtrainer.config.schema import (
    DataConfig,
    ExperimentConfig,
    LabelingStrategy,
    LabelsConfig,
)


# =============================================================================
# TestLabelsConfig — dataclass validation (7 tests)
# =============================================================================


class TestLabelsConfig:
    """Validate LabelsConfig field constraints."""

    def test_default_values(self):
        """All defaults match the contract in pipeline_contract.toml."""
        lc = LabelsConfig()
        assert lc.source == "auto"
        assert lc.return_type == "smoothed_return"
        assert lc.task == "auto"
        assert lc.threshold_bps == 8.0
        assert lc.horizons == []
        assert lc.primary_horizon_idx == 0

    def test_validate_source_rejects_invalid(self):
        with pytest.raises(ValueError, match="LabelsConfig.source"):
            LabelsConfig(source="invalid")

    def test_validate_return_type_rejects_invalid(self):
        with pytest.raises(ValueError, match="LabelsConfig.return_type"):
            LabelsConfig(return_type="quadratic_return")

    def test_validate_task_rejects_invalid(self):
        with pytest.raises(ValueError, match="LabelsConfig.task"):
            LabelsConfig(task="reinforcement")

    def test_validate_threshold_bps_rejects_negative(self):
        with pytest.raises(ValueError, match="threshold_bps must be >= 0"):
            LabelsConfig(threshold_bps=-1.0)

    def test_validate_primary_horizon_idx_rejects_negative(self):
        with pytest.raises(ValueError, match="primary_horizon_idx must be >= 0"):
            LabelsConfig(primary_horizon_idx=-1)

    def test_validate_horizons_rejects_duplicates(self):
        with pytest.raises(ValueError, match="duplicates"):
            LabelsConfig(horizons=[10, 60, 10])


# =============================================================================
# TestDataConfigAutoDerive — legacy → LabelsConfig derivation (5 tests)
# =============================================================================


class TestDataConfigAutoDerive:
    """Verify that DataConfig.__post_init__ derives labels from legacy fields."""

    def test_default_derives_classification_task_and_idx_0(self):
        """Default DataConfig() → labels.task='classification', idx=0."""
        dc = DataConfig()
        assert dc.labels is not None
        assert dc.labels.task == "classification"
        assert dc.labels.primary_horizon_idx == 0
        assert dc.labels.source == "auto"

    def test_regression_strategy_derives_regression_task(self):
        dc = DataConfig(labeling_strategy=LabelingStrategy.REGRESSION)
        assert dc.labels.task == "regression"

    def test_horizon_idx_2_derives_primary_horizon_idx_2(self):
        dc = DataConfig(horizon_idx=2)
        assert dc.labels.primary_horizon_idx == 2

    def test_horizon_idx_none_derives_primary_horizon_idx_none(self):
        """horizon_idx=None (HMHP mode) propagates to primary_horizon_idx=None."""
        dc = DataConfig(horizon_idx=None)
        assert dc.labels.primary_horizon_idx is None

    def test_explicit_labels_skips_derivation(self):
        """When labels is explicitly provided, derivation is skipped."""
        explicit = LabelsConfig(
            source="forward_prices",
            task="regression",
            horizons=[10, 60],
        )
        dc = DataConfig(labels=explicit)
        assert dc.labels.source == "forward_prices"
        assert dc.labels.task == "regression"
        assert dc.labels.horizons == [10, 60]


# =============================================================================
# TestDeprecationWarnings — ExperimentConfig fires on non-default legacy (4 tests)
# =============================================================================


class TestDeprecationWarnings:
    """Verify deprecation warnings fire correctly."""

    def test_non_default_labeling_strategy_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ExperimentConfig(
                data=DataConfig(labeling_strategy=LabelingStrategy.REGRESSION)
            )
            dep = [x for x in w if issubclass(x.category, DeprecationWarning)]
            msgs = [str(d.message) for d in dep]
            assert any("labeling_strategy is deprecated" in m for m in msgs)

    def test_non_default_horizon_idx_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ExperimentConfig(data=DataConfig(horizon_idx=2))
            dep = [x for x in w if issubclass(x.category, DeprecationWarning)]
            msgs = [str(d.message) for d in dep]
            assert any("horizon_idx is deprecated" in m for m in msgs)

    def test_default_legacy_values_no_warning(self):
        """Default legacy values (TLOB, horizon_idx=0) emit no DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ExperimentConfig()
            dep = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep) == 0, f"Got unexpected deprecation: {[str(d.message) for d in dep]}"

    def test_explicit_labels_with_default_legacy_no_warning(self):
        """Explicit labels + default legacy = no DeprecationWarning."""
        explicit = LabelsConfig(task="regression")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ExperimentConfig(data=DataConfig(labels=explicit))
            dep = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep) == 0


# =============================================================================
# Helpers for T9-D / T9-E tests
# =============================================================================


def _make_synthetic_forward_prices(
    n_seq: int = 50,
    k: int = 5,
    max_h: int = 300,
    seed: int = 42,
) -> tuple:
    """Create a synthetic forward_prices array + matching metadata.

    Returns:
        (forward_prices, metadata) where:
            forward_prices: [n_seq, k + max_h + 1] float64
            metadata: dict with forward_prices and horizons sections
    """
    rng = np.random.default_rng(seed)
    n_cols = k + max_h + 1
    # Random walk prices around 100
    base = 100.0
    increments = rng.normal(0, 0.01, size=(n_seq, n_cols))
    forward_prices = base + np.cumsum(increments, axis=1)

    metadata = {
        "forward_prices": {
            "exported": True,
            "smoothing_window_offset": k,
            "max_horizon": max_h,
            "n_columns": n_cols,
            "units": "USD",
            "column_layout": f"col_0=t-{k}, col_{k}=t, col_{k + max_h}=t+{max_h}",
        },
        "horizons": [10, 60, 300],
        "label_strategy": "regression",
    }
    return forward_prices, metadata


# =============================================================================
# TestLabelComputation — _compute_labels_from_forward_prices (6 tests)
# =============================================================================


class TestLabelComputation:
    """Test the label computation bridge function."""

    def test_compute_regression_returns_2d_shape(self):
        """Regression labels have shape [N, n_horizons]."""
        from lobtrainer.data.dataset import _compute_labels_from_forward_prices

        fp, meta = _make_synthetic_forward_prices()
        lc = LabelsConfig(task="regression")
        cls_labels, reg_labels = _compute_labels_from_forward_prices(fp, meta, lc)
        assert cls_labels is None, "regression task should not produce classification labels"
        assert reg_labels.shape == (50, 3)  # 3 horizons [10, 60, 300]
        assert reg_labels.dtype == np.float64

    def test_compute_classification_returns_canonical_labels(self):
        """Classification labels are int8 in {-1, 0, +1} (canonical form)."""
        from lobtrainer.data.dataset import _compute_labels_from_forward_prices

        fp, meta = _make_synthetic_forward_prices()
        lc = LabelsConfig(task="classification", threshold_bps=5.0)
        cls_labels, reg_labels = _compute_labels_from_forward_prices(fp, meta, lc)
        assert cls_labels is not None
        assert cls_labels.dtype == np.int8
        assert set(np.unique(cls_labels)).issubset({-1, 0, 1})
        # Regression labels are always produced
        assert reg_labels.shape == (50, 3)

    def test_k_read_from_contract_not_user_config(self):
        """smoothing_window comes from ForwardPriceContract, not LabelsConfig.

        LabelsConfig has NO smoothing_window field (Bug B2 prevention).
        Verify that the function reads k from metadata.
        """
        from lobtrainer.data.dataset import _compute_labels_from_forward_prices

        fp, meta = _make_synthetic_forward_prices(k=5)
        lc = LabelsConfig(task="regression")
        # Verify there's no smoothing_window on LabelsConfig
        assert not hasattr(lc, "smoothing_window")
        # Function should use k=5 from metadata, not crash
        _, reg = _compute_labels_from_forward_prices(fp, meta, lc)
        assert reg.shape[0] == fp.shape[0]

    def test_horizon_subset_computes_only_requested(self):
        """Custom horizons list materializes only those horizons."""
        from lobtrainer.data.dataset import _compute_labels_from_forward_prices

        fp, meta = _make_synthetic_forward_prices()
        lc = LabelsConfig(task="regression", horizons=[10])
        _, reg = _compute_labels_from_forward_prices(fp, meta, lc)
        assert reg.shape == (50, 1)  # Only 1 horizon materialized

    def test_auto_task_detects_regression_from_metadata(self):
        """task='auto' + metadata label_strategy='regression' → regression."""
        from lobtrainer.data.dataset import _compute_labels_from_forward_prices

        fp, meta = _make_synthetic_forward_prices()
        meta["label_strategy"] = "regression"
        lc = LabelsConfig(task="auto")
        cls_labels, _ = _compute_labels_from_forward_prices(fp, meta, lc)
        assert cls_labels is None  # regression task

    def test_auto_task_detects_classification_from_metadata(self):
        """task='auto' + metadata label_strategy='tlob' → classification."""
        from lobtrainer.data.dataset import _compute_labels_from_forward_prices

        fp, meta = _make_synthetic_forward_prices()
        meta["label_strategy"] = "tlob"
        lc = LabelsConfig(task="auto", threshold_bps=5.0)
        cls_labels, _ = _compute_labels_from_forward_prices(fp, meta, lc)
        assert cls_labels is not None  # classification task
        assert cls_labels.dtype == np.int8


# =============================================================================
# Helpers for disk-based fixture tests
# =============================================================================


def _create_synthetic_export_with_fp(
    root: "Path",
    n_seq: int = 10,
    window_size: int = 20,
    n_features: int = 98,
    k: int = 5,
    max_h: int = 300,
    include_precomputed_labels: bool = True,
    include_forward_prices: bool = True,
    fp_exported_in_metadata: bool = True,
    seed: int = 42,
) -> None:
    """Create a minimal synthetic export directory with forward_prices support.

    Creates ``root/train/20250101_*`` files.
    """
    from pathlib import Path

    train_dir = Path(root) / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    date = "20250101"
    n_cols = k + max_h + 1

    # Sequences
    seq = rng.standard_normal((n_seq, window_size, n_features)).astype(np.float32)
    np.save(train_dir / f"{date}_sequences.npy", seq)

    # Precomputed labels (classification)
    if include_precomputed_labels:
        labels = rng.choice([-1, 0, 1], size=n_seq).astype(np.int8)
        np.save(train_dir / f"{date}_labels.npy", labels)

    # Forward prices
    if include_forward_prices:
        fp = 100.0 + np.cumsum(rng.normal(0, 0.01, (n_seq, n_cols)), axis=1)
        np.save(train_dir / f"{date}_forward_prices.npy", fp)

    # Metadata
    meta = {
        "day": date,
        "n_sequences": n_seq,
        "window_size": window_size,
        "n_features": n_features,
        "schema_version": "2.2",
        "contract_version": "2.2",
        "label_strategy": "regression",
        "label_encoding": {"note": "smoothed forward returns in bps"},
        "normalization": {"strategy": "none", "applied": False, "params_file": ""},
        "provenance": {
            "extractor_version": "test",
            "git_commit": "abc123",
            "git_dirty": False,
            "config_hash": "test",
            "contract_version": "2.2",
            "export_timestamp_utc": "2025-01-01T00:00:00Z",
        },
        "export_timestamp": "2025-01-01T00:00:00Z",
        "horizons": [10, 60, 300],
    }

    if include_forward_prices and fp_exported_in_metadata:
        meta["forward_prices"] = {
            "exported": True,
            "smoothing_window_offset": k,
            "max_horizon": max_h,
            "n_columns": n_cols,
            "units": "USD",
            "column_layout": f"col_0=t-{k}, col_{k}=t, col_{k + max_h}=t+{max_h}",
        }
    else:
        meta["forward_prices"] = {"exported": False}

    import json

    with open(train_dir / f"{date}_metadata.json", "w") as f:
        json.dump(meta, f)


# =============================================================================
# TestForwardPricesLoading — disk-based integration tests (6 tests)
# =============================================================================


class TestForwardPricesLoading:
    """Test load_day_data and load_split_data with forward_prices on disk."""

    def test_auto_source_uses_fp_when_available(self, tmp_path):
        """source=auto + forward_prices.npy present → uses fp path."""
        from lobtrainer.data.dataset import load_day_data

        _create_synthetic_export_with_fp(tmp_path)
        d = tmp_path / "train"
        day = load_day_data(
            d / "20250101_sequences.npy",
            d / "20250101_labels.npy",
            d / "20250101_metadata.json",
            forward_prices_path=d / "20250101_forward_prices.npy",
            labels_config=LabelsConfig(source="auto", task="regression"),
            validate=False,
        )
        assert day.forward_prices is not None
        assert day.regression_labels is not None
        assert day.regression_labels.shape == (10, 3)

    def test_auto_source_falls_back_when_fp_absent(self, tmp_path):
        """source=auto + no forward_prices.npy → loads precomputed labels."""
        from lobtrainer.data.dataset import load_day_data

        _create_synthetic_export_with_fp(tmp_path, include_forward_prices=False)
        d = tmp_path / "train"
        day = load_day_data(
            d / "20250101_sequences.npy",
            d / "20250101_labels.npy",
            d / "20250101_metadata.json",
            labels_config=LabelsConfig(source="auto"),
            validate=False,
        )
        assert day.forward_prices is None
        assert day.labels is not None

    def test_explicit_fp_source_raises_when_file_missing(self, tmp_path):
        """source=forward_prices + no fp file → FileNotFoundError."""
        from lobtrainer.data.dataset import load_day_data

        _create_synthetic_export_with_fp(tmp_path, include_forward_prices=False)
        d = tmp_path / "train"
        with pytest.raises(FileNotFoundError, match="forward_prices"):
            load_day_data(
                d / "20250101_sequences.npy",
                d / "20250101_labels.npy",
                d / "20250101_metadata.json",
                forward_prices_path=d / "20250101_forward_prices.npy",
                labels_config=LabelsConfig(source="forward_prices", task="regression"),
                validate=False,
            )

    def test_explicit_fp_source_raises_when_metadata_exported_false(self, tmp_path):
        """source=forward_prices + metadata says exported=False → ValueError."""
        from lobtrainer.data.dataset import load_day_data

        _create_synthetic_export_with_fp(
            tmp_path, include_forward_prices=True, fp_exported_in_metadata=False
        )
        d = tmp_path / "train"
        with pytest.raises(ValueError, match="forward_prices.exported=True"):
            load_day_data(
                d / "20250101_sequences.npy",
                d / "20250101_labels.npy",
                d / "20250101_metadata.json",
                forward_prices_path=d / "20250101_forward_prices.npy",
                labels_config=LabelsConfig(source="forward_prices", task="regression"),
                validate=False,
            )

    def test_explicit_precomputed_ignores_fp(self, tmp_path):
        """source=precomputed → forward_prices ignored even if present."""
        from lobtrainer.data.dataset import load_day_data

        _create_synthetic_export_with_fp(tmp_path)
        d = tmp_path / "train"
        day = load_day_data(
            d / "20250101_sequences.npy",
            d / "20250101_labels.npy",
            d / "20250101_metadata.json",
            forward_prices_path=d / "20250101_forward_prices.npy",
            labels_config=LabelsConfig(source="precomputed"),
            validate=False,
        )
        assert day.forward_prices is None

    def test_fp_alignment_mismatch_raises(self, tmp_path):
        """Misaligned forward_prices (wrong row count) → ValueError."""
        from lobtrainer.data.dataset import load_day_data

        _create_synthetic_export_with_fp(tmp_path, n_seq=10)
        d = tmp_path / "train"
        # Overwrite with wrong-sized forward_prices
        bad_fp = np.random.randn(15, 306)  # 15 rows, not 10
        np.save(d / "20250101_forward_prices.npy", bad_fp)
        with pytest.raises(ValueError, match="alignment"):
            load_day_data(
                d / "20250101_sequences.npy",
                d / "20250101_labels.npy",
                d / "20250101_metadata.json",
                forward_prices_path=d / "20250101_forward_prices.npy",
                labels_config=LabelsConfig(source="forward_prices", task="regression"),
                validate=False,
            )


# =============================================================================
# TestLoadDayDataForwardPricesField — DayData.forward_prices population (2 tests)
# =============================================================================


class TestLoadDayDataForwardPricesField:
    """Verify DayData.forward_prices is correctly populated or None."""

    def test_forward_prices_populated_on_fp_load(self, tmp_path):
        from lobtrainer.data.dataset import load_day_data

        _create_synthetic_export_with_fp(tmp_path)
        d = tmp_path / "train"
        day = load_day_data(
            d / "20250101_sequences.npy",
            d / "20250101_labels.npy",
            d / "20250101_metadata.json",
            forward_prices_path=d / "20250101_forward_prices.npy",
            labels_config=LabelsConfig(source="forward_prices", task="regression"),
            validate=False,
        )
        assert day.forward_prices is not None
        assert day.forward_prices.shape[0] == 10
        assert day.forward_prices.dtype == np.float64

    def test_forward_prices_none_on_precomputed_load(self, tmp_path):
        from lobtrainer.data.dataset import load_day_data

        _create_synthetic_export_with_fp(tmp_path, include_forward_prices=False)
        d = tmp_path / "train"
        day = load_day_data(
            d / "20250101_sequences.npy",
            d / "20250101_labels.npy",
            d / "20250101_metadata.json",
            labels_config=LabelsConfig(source="precomputed"),
            validate=False,
        )
        assert day.forward_prices is None


# =============================================================================
# TestDaciteCompatibility — first Optional[nested_dataclass] in schema (3 tests)
# =============================================================================


class TestDaciteCompatibility:
    """Verify dacite handles Optional[LabelsConfig] correctly."""

    def test_from_dict_missing_labels_key_gives_none_then_derived(self):
        """Dict with no 'labels' key → DataConfig.labels derived from legacy."""
        config = ExperimentConfig.from_dict({
            "name": "test",
            "data": {"feature_count": 98},
            "model": {"input_size": 98},
        })
        assert config.data.labels is not None
        assert config.data.labels.task == "classification"  # derived from TLOB default
        assert config.data.labels.primary_horizon_idx == 0

    def test_from_dict_labels_dict_constructs_labelsconfig(self):
        """Dict with labels: {...} → LabelsConfig constructed by dacite."""
        config = ExperimentConfig.from_dict({
            "name": "test",
            "data": {
                "feature_count": 98,
                "labels": {
                    "source": "forward_prices",
                    "task": "regression",
                    "horizons": [10, 60],
                },
            },
            "model": {"input_size": 98},
        })
        assert config.data.labels is not None
        assert config.data.labels.source == "forward_prices"
        assert config.data.labels.task == "regression"
        assert config.data.labels.horizons == [10, 60]

    def test_from_dict_labels_null_gives_none_then_derived(self):
        """Dict with labels: None → DataConfig.labels derived from legacy."""
        config = ExperimentConfig.from_dict({
            "name": "test",
            "data": {"feature_count": 98, "labels": None},
            "model": {"input_size": 98},
        })
        assert config.data.labels is not None  # derived
        assert config.data.labels.source == "auto"
