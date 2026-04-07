"""
Unit tests for the signal export module.

Tests RawFeatureExtractor, SignalExporter inference paths, metadata builder,
and backtester contract compliance.

Schema v2.2.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from lobtrainer.export.raw_features import RawFeatureExtractor, RawFeatures
from lobtrainer.export.metadata import build_signal_metadata

# Contract constants
from hft_contracts import SIGNAL_SPREAD_FEATURE_INDEX, SIGNAL_PRICE_FEATURE_INDEX


# =============================================================================
# Helper to write synthetic export files
# =============================================================================


def _write_test_sequences(split_dir: Path, n_days: int = 2, n_seqs: int = 10,
                          seq_len: int = 20, n_features: int = 98):
    """Write synthetic sequence files for testing RawFeatureExtractor."""
    split_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    total = 0
    for i in range(n_days):
        date = f"2025-03-{i+1:02d}"
        seqs = rng.standard_normal((n_seqs, seq_len, n_features)).astype(np.float32)
        # Set known values at spread/price indices in last timestep
        seqs[:, -1, SIGNAL_SPREAD_FEATURE_INDEX] = 5.0 + i  # spread = 5, 6, ...
        seqs[:, -1, SIGNAL_PRICE_FEATURE_INDEX] = 130.0 + i  # price = 130, 131, ...
        np.save(split_dir / f"{date}_sequences.npy", seqs)
        total += n_seqs
    return total


# =============================================================================
# RawFeatureExtractor Tests
# =============================================================================


class TestRawFeatureExtractor:
    """Test raw spread/price extraction from disk."""

    def test_extracts_correct_values(self, tmp_path):
        """Spread and price values match what was written at contract indices."""
        split_dir = tmp_path / "test"
        _write_test_sequences(split_dir, n_days=2, n_seqs=10)

        extractor = RawFeatureExtractor(tmp_path, "test")
        raw = extractor.extract()

        assert raw.n_samples == 20
        # Day 1: spread=5.0, price=130.0 (10 samples)
        np.testing.assert_allclose(raw.spreads[:10], 5.0, atol=1e-6)
        np.testing.assert_allclose(raw.prices[:10], 130.0, atol=1e-6)
        # Day 2: spread=6.0, price=131.0 (10 samples)
        np.testing.assert_allclose(raw.spreads[10:], 6.0, atol=1e-6)
        np.testing.assert_allclose(raw.prices[10:], 131.0, atol=1e-6)

    def test_uses_contract_indices(self):
        """Verify contract constants are the expected values."""
        assert SIGNAL_SPREAD_FEATURE_INDEX == 42
        assert SIGNAL_PRICE_FEATURE_INDEX == 40

    def test_missing_split_raises(self, tmp_path):
        """FileNotFoundError if split directory doesn't exist."""
        with pytest.raises(FileNotFoundError, match="not found"):
            RawFeatureExtractor(tmp_path, "test")

    def test_no_sequence_files_raises(self, tmp_path):
        """FileNotFoundError if no *_sequences.npy files found."""
        (tmp_path / "test").mkdir()
        extractor = RawFeatureExtractor(tmp_path, "test")
        with pytest.raises(FileNotFoundError, match="No.*sequences"):
            extractor.extract()

    def test_sorted_glob_ordering(self, tmp_path):
        """Files are processed in sorted order (matching DataLoader)."""
        split_dir = tmp_path / "test"
        split_dir.mkdir(parents=True)

        # Write in reverse date order to verify sorting
        for date in ["2025-03-03", "2025-03-01", "2025-03-02"]:
            seqs = np.ones((5, 10, 98), dtype=np.float32)
            day_num = int(date[-2:])
            seqs[:, -1, SIGNAL_PRICE_FEATURE_INDEX] = float(day_num)
            np.save(split_dir / f"{date}_sequences.npy", seqs)

        raw = RawFeatureExtractor(tmp_path, "test").extract()
        # Should be sorted: day 01 (5 samples), day 02, day 03
        assert raw.prices[0] == pytest.approx(1.0)  # 2025-03-01
        assert raw.prices[5] == pytest.approx(2.0)  # 2025-03-02
        assert raw.prices[10] == pytest.approx(3.0)  # 2025-03-03

    def test_output_dtype_float64(self, tmp_path):
        """Output arrays are float64 regardless of input dtype."""
        split_dir = tmp_path / "test"
        _write_test_sequences(split_dir, n_days=1, n_seqs=5)
        raw = RawFeatureExtractor(tmp_path, "test").extract()
        assert raw.spreads.dtype == np.float64
        assert raw.prices.dtype == np.float64


# =============================================================================
# Metadata Builder Tests
# =============================================================================


class TestBuildSignalMetadata:
    """Test the metadata builder function."""

    def test_core_fields_always_present(self):
        """Core fields are always in metadata."""
        meta = build_signal_metadata(
            model_type="tlob", model_name="TLOB(2L,64h)",
            parameters=93000, signal_type="regression",
            split="test", total_samples=1000,
            checkpoint="/path/to/best.pt",
        )
        assert meta["model_type"] == "tlob"
        assert meta["model_name"] == "TLOB(2L,64h)"
        assert meta["parameters"] == 93000
        assert meta["signal_type"] == "regression"
        assert meta["split"] == "test"
        assert meta["total_samples"] == 1000
        assert "exported_at" in meta

    def test_optional_fields_omitted_when_none(self):
        """Optional fields are not present when not provided."""
        meta = build_signal_metadata(
            model_type="tlob", model_name="test",
            parameters=100, signal_type="classification",
            split="test", total_samples=10,
            checkpoint="/test.pt",
        )
        assert "metrics" not in meta
        assert "calibration" not in meta
        assert "predictions_distribution" not in meta

    def test_classification_fields_included(self):
        """Classification-specific fields are included when provided."""
        meta = build_signal_metadata(
            model_type="hmhp", model_name="HMHP",
            parameters=171000, signal_type="classification",
            split="test", total_samples=500,
            checkpoint="/test.pt",
            predictions_distribution={"Down": 100, "Stable": 300, "Up": 100},
            directional_rate=0.4,
        )
        assert meta["predictions_distribution"]["Stable"] == 300
        assert meta["directional_rate"] == 0.4

    def test_regression_fields_included(self):
        """Regression-specific fields are included when provided."""
        meta = build_signal_metadata(
            model_type="tlob", model_name="TLOB-R",
            parameters=93000, signal_type="regression",
            split="test", total_samples=1000,
            checkpoint="/test.pt",
            metrics={"r2": 0.464, "ic": 0.677},
            calibration={"scale_factor": 3.73},
        )
        assert meta["metrics"]["r2"] == 0.464
        assert meta["calibration"]["scale_factor"] == 3.73

    def test_metadata_json_serializable(self):
        """Metadata can be serialized to JSON without errors."""
        meta = build_signal_metadata(
            model_type="tlob", model_name="test",
            parameters=100, signal_type="regression",
            split="test", total_samples=10,
            checkpoint="/test.pt",
            spread_stats={"mean": 5.0, "median": 4.5, "p90": 8.0},
            prediction_stats={"mean": 0.1, "std": 5.2},
        )
        # Should not raise
        json_str = json.dumps(meta)
        assert isinstance(json_str, str)


# =============================================================================
# File Contract Tests (validate output format)
# =============================================================================


class TestSignalFileContract:
    """Test that exported files match the backtester's signal contract."""

    def test_classification_output_files(self, tmp_path):
        """Verify classification signal file shapes and dtypes."""
        # Simulate what SignalExporter._write_files produces
        n = 100
        np.save(tmp_path / "prices.npy", np.random.rand(n).astype(np.float64) + 100)
        np.save(tmp_path / "spreads.npy", np.random.rand(n).astype(np.float64) * 10)
        np.save(tmp_path / "predictions.npy", np.random.randint(0, 3, n).astype(np.int32))
        np.save(tmp_path / "labels.npy", np.random.randint(0, 3, n).astype(np.int32))

        # Verify contract
        prices = np.load(tmp_path / "prices.npy")
        preds = np.load(tmp_path / "predictions.npy")
        assert prices.shape == (n,)
        assert prices.dtype == np.float64
        assert preds.dtype == np.int32
        assert set(np.unique(preds)).issubset({0, 1, 2})
        assert prices.shape[0] == preds.shape[0]

    def test_regression_output_files(self, tmp_path):
        """Verify regression signal file shapes and dtypes."""
        n = 100
        np.save(tmp_path / "prices.npy", np.random.rand(n).astype(np.float64) + 100)
        np.save(tmp_path / "predicted_returns.npy", np.random.randn(n).astype(np.float64) * 5)

        prices = np.load(tmp_path / "prices.npy")
        returns = np.load(tmp_path / "predicted_returns.npy")
        assert prices.shape == returns.shape
        assert returns.dtype == np.float64
        assert np.isfinite(returns).all()

    def test_multi_horizon_regression_shape(self, tmp_path):
        """Multi-horizon regression has shape [N, H]."""
        n, h = 100, 3
        np.save(tmp_path / "predicted_returns.npy",
                np.random.randn(n, h).astype(np.float64))
        returns = np.load(tmp_path / "predicted_returns.npy")
        assert returns.shape == (n, h)
        assert returns.dtype == np.float64

    def test_alignment_all_arrays_same_n(self, tmp_path):
        """All signal arrays must have identical first dimension."""
        n = 50
        np.save(tmp_path / "prices.npy", np.ones(n, dtype=np.float64))
        np.save(tmp_path / "predictions.npy", np.zeros(n, dtype=np.int32))
        np.save(tmp_path / "spreads.npy", np.ones(n, dtype=np.float64))
        np.save(tmp_path / "labels.npy", np.ones(n, dtype=np.int32))

        files = ["prices.npy", "predictions.npy", "spreads.npy", "labels.npy"]
        shapes = [np.load(tmp_path / f).shape[0] for f in files]
        assert len(set(shapes)) == 1, f"Shape mismatch: {dict(zip(files, shapes))}"


# =============================================================================
# ExportResult Tests
# =============================================================================


class TestExportResult:
    """Test ExportResult dataclass."""

    def test_export_result_creation(self, tmp_path):
        """ExportResult can be created with all fields."""
        from lobtrainer.export.exporter import ExportResult
        result = ExportResult(
            output_dir=tmp_path,
            n_samples=100,
            signal_type="classification",
            files_written=["prices.npy", "predictions.npy"],
            metadata={"model_type": "tlob"},
        )
        assert result.n_samples == 100
        assert result.signal_type == "classification"
        assert len(result.files_written) == 2


# =============================================================================
# SignalExporter validation tests
# =============================================================================


class TestSignalExporterValidation:
    """Test SignalExporter input validation."""

    def test_refuses_train_split(self):
        """Training split is refused due to drop_last alignment issue."""
        from lobtrainer.export.exporter import SignalExporter

        class MockTrainer:
            strategy = None
            model = None
            device = "cpu"
            config = None
            def get_loader(self, split): return None

        exporter = SignalExporter(MockTrainer())
        with pytest.raises(ValueError, match="Cannot export training split"):
            exporter.export(split="train")

    def test_refuses_missing_loader(self):
        """ValueError if no DataLoader for the requested split."""
        from lobtrainer.export.exporter import SignalExporter

        class MockTrainer:
            strategy = None
            model = None
            device = "cpu"
            config = type("C", (), {"output_dir": "/tmp"})()
            def get_loader(self, split): return None

        exporter = SignalExporter(MockTrainer())
        with pytest.raises(ValueError, match="No DataLoader"):
            exporter.export(split="test")
