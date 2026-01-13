"""
Tests for experiment tracking module.

Tests ExperimentResult, ExperimentMetrics, and ExperimentRegistry:
- Serialization/deserialization roundtrip
- Registry storage and retrieval
- Filtering and comparison
- Edge cases (empty registry, missing fields)

RULE.md compliance:
- Data contracts for experiment results
- Deterministic ID generation
- No data loss during serialization
"""

import json
import pytest
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, patch

from lobtrainer.experiments.result import ExperimentResult, ExperimentMetrics
from lobtrainer.experiments.registry import (
    ExperimentRegistry,
    create_comparison_table,
    filter_experiments,
    rank_experiments,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_metrics():
    """Create sample ExperimentMetrics."""
    return ExperimentMetrics(
        accuracy=0.75,
        loss=0.5,
        macro_f1=0.72,
        macro_precision=0.70,
        macro_recall=0.74,
        per_class_precision={"Down": 0.65, "Stable": 0.80, "Up": 0.65},
        per_class_recall={"Down": 0.60, "Stable": 0.85, "Up": 0.77},
        per_class_f1={"Down": 0.62, "Stable": 0.82, "Up": 0.70},
        directional_accuracy=0.68,
        signal_rate=0.30,
        predicted_trade_win_rate=0.55,
        decisive_prediction_rate=0.40,
    )


@pytest.fixture
def sample_result(sample_metrics):
    """Create sample ExperimentResult."""
    return ExperimentResult(
        name="test_experiment",
        description="A test experiment",
        config={"model": {"type": "lstm"}, "data": {"horizon": 10}},
        test_metrics=sample_metrics,
        training_history=[
            {"epoch": 0, "train_loss": 1.0, "val_loss": 1.1},
            {"epoch": 1, "train_loss": 0.8, "val_loss": 0.9},
        ],
        best_epoch=1,
        total_epochs=2,
        training_time_seconds=120.5,
        tags=["lstm", "test"],
        model_type="lstm",
        model_params=10000,
        labeling_strategy="tlob",
        num_train_samples=5000,
        num_val_samples=1000,
        num_test_samples=1000,
    )


@pytest.fixture
def temp_registry_dir():
    """Create temporary directory for registry."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# ExperimentMetrics Tests
# =============================================================================


class TestExperimentMetrics:
    """Test ExperimentMetrics dataclass."""
    
    def test_default_init(self):
        """Default initialization should work."""
        metrics = ExperimentMetrics()
        assert metrics.accuracy == 0.0
        assert metrics.loss == 0.0
        assert metrics.macro_f1 == 0.0
    
    def test_to_dict(self, sample_metrics):
        """to_dict should return all fields."""
        d = sample_metrics.to_dict()
        
        assert d["accuracy"] == 0.75
        assert d["loss"] == 0.5
        assert d["macro_f1"] == 0.72
        assert "per_class_precision" in d
        assert d["directional_accuracy"] == 0.68
    
    def test_from_dict(self, sample_metrics):
        """from_dict should reconstruct metrics."""
        d = sample_metrics.to_dict()
        reconstructed = ExperimentMetrics.from_dict(d)
        
        assert reconstructed.accuracy == sample_metrics.accuracy
        assert reconstructed.loss == sample_metrics.loss
        assert reconstructed.macro_f1 == sample_metrics.macro_f1
    
    def test_from_dict_with_missing_fields(self):
        """from_dict should handle missing fields gracefully."""
        partial_data = {
            "accuracy": 0.8,
            "loss": 0.3,
        }
        
        metrics = ExperimentMetrics.from_dict(partial_data)
        
        assert metrics.accuracy == 0.8
        assert metrics.macro_f1 == 0.0  # Default
    
    def test_serialization_roundtrip(self, sample_metrics):
        """JSON serialization should preserve all data."""
        d = sample_metrics.to_dict()
        json_str = json.dumps(d)
        loaded = json.loads(json_str)
        reconstructed = ExperimentMetrics.from_dict(loaded)
        
        assert reconstructed.accuracy == sample_metrics.accuracy
        assert reconstructed.per_class_precision == sample_metrics.per_class_precision


# =============================================================================
# ExperimentResult Tests
# =============================================================================


class TestExperimentResultInit:
    """Test ExperimentResult initialization."""
    
    def test_auto_generates_id(self):
        """Should auto-generate experiment_id if not provided."""
        result = ExperimentResult(name="test")
        
        assert result.experiment_id != "", "Should generate ID"
        assert "test" in result.experiment_id, "ID should contain name"
    
    def test_auto_generates_timestamp(self):
        """Should auto-generate created_at if not provided."""
        result = ExperimentResult(name="test")
        
        assert result.created_at != "", "Should generate timestamp"
        # Should be valid ISO format
        datetime.fromisoformat(result.created_at)
    
    def test_preserves_provided_id(self):
        """Should preserve provided experiment_id."""
        result = ExperimentResult(
            name="test",
            experiment_id="custom_id_123",
        )
        
        assert result.experiment_id == "custom_id_123"


class TestExperimentResultSerialization:
    """Test ExperimentResult serialization."""
    
    def test_to_dict(self, sample_result):
        """to_dict should include all fields."""
        d = sample_result.to_dict()
        
        assert d["name"] == "test_experiment"
        assert d["model_type"] == "lstm"
        assert d["best_epoch"] == 1
        assert "test_metrics" in d
        assert d["test_metrics"]["accuracy"] == 0.75
    
    def test_from_dict(self, sample_result):
        """from_dict should reconstruct result."""
        d = sample_result.to_dict()
        reconstructed = ExperimentResult.from_dict(d)
        
        assert reconstructed.name == sample_result.name
        assert reconstructed.model_type == sample_result.model_type
        assert reconstructed.test_metrics.accuracy == sample_result.test_metrics.accuracy
    
    def test_from_dict_without_metrics(self):
        """from_dict should handle None metrics."""
        data = {
            "name": "test",
            "config": {},
            "test_metrics": None,
            "train_metrics": None,
            "val_metrics": None,
        }
        
        result = ExperimentResult.from_dict(data)
        
        assert result.test_metrics is None


class TestExperimentResultFile:
    """Test file save/load operations."""
    
    def test_save_and_load(self, sample_result, temp_registry_dir):
        """Should save and load correctly."""
        path = temp_registry_dir / "experiment.json"
        
        sample_result.save(path)
        loaded = ExperimentResult.load(path)
        
        assert loaded.name == sample_result.name
        assert loaded.model_type == sample_result.model_type
        assert loaded.best_epoch == sample_result.best_epoch
    
    def test_save_creates_parent_dirs(self, sample_result, temp_registry_dir):
        """save should create parent directories."""
        path = temp_registry_dir / "subdir" / "deep" / "experiment.json"
        
        sample_result.save(path)
        
        assert path.exists()
    
    def test_saved_file_is_valid_json(self, sample_result, temp_registry_dir):
        """Saved file should be valid JSON."""
        path = temp_registry_dir / "experiment.json"
        
        sample_result.save(path)
        
        with open(path) as f:
            data = json.load(f)
        
        assert "name" in data
        assert data["name"] == "test_experiment"


class TestExperimentResultSummary:
    """Test summary generation."""
    
    def test_summary_includes_key_info(self, sample_result):
        """Summary should include key experiment info."""
        summary = sample_result.summary()
        
        assert "test_experiment" in summary
        assert "lstm" in summary
        assert "10,000" in summary  # model params (comma formatted)
    
    def test_summary_includes_metrics(self, sample_result):
        """Summary should include test metrics."""
        summary = sample_result.summary()
        
        assert "Accuracy" in summary or "accuracy" in summary.lower()
        assert "0.75" in summary or "75" in summary


# =============================================================================
# ExperimentRegistry Tests
# =============================================================================


class TestExperimentRegistryInit:
    """Test ExperimentRegistry initialization."""
    
    def test_creates_directory(self, temp_registry_dir):
        """Should create directory if missing."""
        new_dir = temp_registry_dir / "new_registry"
        
        registry = ExperimentRegistry(new_dir)
        
        assert new_dir.exists()
    
    def test_loads_existing_index(self, temp_registry_dir):
        """Should load existing index on init."""
        # Create registry and add experiment
        registry1 = ExperimentRegistry(temp_registry_dir)
        result = ExperimentResult(name="test", model_type="lstm")
        registry1.register(result)
        
        # Create new registry instance pointing to same dir
        registry2 = ExperimentRegistry(temp_registry_dir)
        
        assert registry2.count() == 1


class TestExperimentRegistryRegister:
    """Test experiment registration."""
    
    def test_register_saves_result(self, sample_result, temp_registry_dir):
        """register should save experiment to file."""
        registry = ExperimentRegistry(temp_registry_dir)
        
        exp_id = registry.register(sample_result)
        
        # File should exist
        result_path = temp_registry_dir / f"{exp_id}.json"
        assert result_path.exists()
    
    def test_register_updates_index(self, sample_result, temp_registry_dir):
        """register should update index."""
        registry = ExperimentRegistry(temp_registry_dir)
        
        registry.register(sample_result)
        
        assert registry.count() == 1
    
    def test_register_returns_id(self, sample_result, temp_registry_dir):
        """register should return experiment ID."""
        registry = ExperimentRegistry(temp_registry_dir)
        
        exp_id = registry.register(sample_result)
        
        assert exp_id == sample_result.experiment_id


class TestExperimentRegistryGet:
    """Test experiment retrieval."""
    
    def test_get_existing(self, sample_result, temp_registry_dir):
        """get should return registered experiment."""
        registry = ExperimentRegistry(temp_registry_dir)
        exp_id = registry.register(sample_result)
        
        retrieved = registry.get(exp_id)
        
        assert retrieved is not None
        assert retrieved.name == sample_result.name
    
    def test_get_nonexistent(self, temp_registry_dir):
        """get should return None for unknown ID."""
        registry = ExperimentRegistry(temp_registry_dir)
        
        result = registry.get("nonexistent_id")
        
        assert result is None


class TestExperimentRegistryList:
    """Test listing experiments."""
    
    def test_list_all(self, temp_registry_dir):
        """list_all should return all experiments."""
        registry = ExperimentRegistry(temp_registry_dir)
        
        # Add multiple experiments
        for i in range(3):
            result = ExperimentResult(name=f"exp_{i}", model_type="lstm")
            registry.register(result)
        
        all_exps = registry.list_all()
        
        assert len(all_exps) == 3
    
    def test_list_ids(self, temp_registry_dir):
        """list_ids should return all experiment IDs."""
        registry = ExperimentRegistry(temp_registry_dir)
        
        ids = []
        for i in range(3):
            result = ExperimentResult(name=f"exp_{i}", model_type="lstm")
            ids.append(registry.register(result))
        
        listed_ids = registry.list_ids()
        
        assert set(listed_ids) == set(ids)


class TestExperimentRegistryFilter:
    """Test filtering experiments."""
    
    def test_filter_by_model_type(self, temp_registry_dir):
        """Should filter by model type."""
        registry = ExperimentRegistry(temp_registry_dir)
        
        # Add different model types (unique names to avoid ID collision)
        for i, model_type in enumerate(["lstm", "lstm", "deeplob", "tlob"]):
            metrics = ExperimentMetrics(accuracy=0.7, macro_f1=0.6)
            result = ExperimentResult(
                name=f"exp_{model_type}_{i}",  # Unique name
                model_type=model_type,
                test_metrics=metrics,
            )
            registry.register(result)
        
        lstm_exps = registry.filter(model_type="lstm")
        
        assert len(lstm_exps) == 2
    
    def test_filter_by_tags(self, temp_registry_dir):
        """Should filter by tags."""
        registry = ExperimentRegistry(temp_registry_dir)
        
        # Add experiments with different tags (unique names with version to avoid collision)
        for i, tags in enumerate([["baseline"], ["experiment", "v1"], ["experiment", "v2"]]):
            metrics = ExperimentMetrics(accuracy=0.7, macro_f1=0.6)
            result = ExperimentResult(
                name=f"exp_{tags[0]}_{i}",  # Unique name
                tags=tags,
                test_metrics=metrics,
            )
            registry.register(result)
        
        exp_exps = registry.filter(tags=["experiment"])
        
        assert len(exp_exps) == 2
    
    def test_filter_by_min_accuracy(self, temp_registry_dir):
        """Should filter by minimum accuracy."""
        registry = ExperimentRegistry(temp_registry_dir)
        
        for acc in [0.5, 0.6, 0.7, 0.8]:
            metrics = ExperimentMetrics(accuracy=acc, macro_f1=0.6)
            result = ExperimentResult(
                name=f"exp_{acc}",
                test_metrics=metrics,
            )
            registry.register(result)
        
        high_acc = registry.filter(min_accuracy=0.65)
        
        assert len(high_acc) == 2


class TestExperimentRegistrySummary:
    """Test registry summary."""
    
    def test_summary_includes_count(self, sample_result, temp_registry_dir):
        """Summary should include experiment count."""
        registry = ExperimentRegistry(temp_registry_dir)
        registry.register(sample_result)
        
        summary = registry.summary()
        
        assert "1" in summary or "Total experiments: 1" in summary


# =============================================================================
# Comparison Utilities Tests
# =============================================================================


class TestCreateComparisonTable:
    """Test comparison table generation."""
    
    def test_empty_list(self):
        """Should handle empty list."""
        result = create_comparison_table([])
        
        assert "No experiments" in result
    
    def test_with_results(self, temp_registry_dir):
        """Should create table with multiple results."""
        results = []
        for i in range(3):
            metrics = ExperimentMetrics(
                accuracy=0.5 + i * 0.1,
                macro_f1=0.4 + i * 0.1,
                directional_accuracy=0.3 + i * 0.1,
                signal_rate=0.2 + i * 0.1,
            )
            result = ExperimentResult(
                name=f"exp_{i}",
                model_type="lstm",
                labeling_strategy="tlob",
                total_epochs=10,
                test_metrics=metrics,
            )
            results.append(result)
        
        table = create_comparison_table(results)
        
        # Should contain headers
        assert "Name" in table or "name" in table.lower()
        assert "Model" in table or "model" in table.lower()
    
    def test_custom_metrics(self, sample_result):
        """Should use custom metric keys."""
        table = create_comparison_table(
            [sample_result],
            metric_keys=["accuracy", "macro_f1"],
        )
        
        assert "Accuracy" in table or "accuracy" in table.lower()


class TestFilterExperiments:
    """Test filter_experiments function."""
    
    def test_custom_predicate(self):
        """Should filter by custom predicate."""
        results = []
        for i in range(5):
            metrics = ExperimentMetrics(accuracy=0.1 * i)
            result = ExperimentResult(name=f"exp_{i}", test_metrics=metrics)
            results.append(result)
        
        # Filter to accuracy > 0.2
        filtered = filter_experiments(
            results,
            lambda r: r.test_metrics.accuracy > 0.2,
        )
        
        assert len(filtered) == 2  # 0.3, 0.4


class TestRankExperiments:
    """Test rank_experiments function."""
    
    def test_ranking_by_macro_f1(self):
        """Should rank by specified metric."""
        results = []
        for f1 in [0.5, 0.8, 0.3, 0.9, 0.6]:
            metrics = ExperimentMetrics(macro_f1=f1)
            result = ExperimentResult(name=f"exp_{f1}", test_metrics=metrics)
            results.append(result)
        
        ranked = rank_experiments(results, metric="macro_f1")
        
        # Should be in descending order
        f1_values = [r.test_metrics.macro_f1 for r in ranked]
        assert f1_values == sorted(f1_values, reverse=True)
    
    def test_top_k(self):
        """Should return top K results."""
        results = []
        for f1 in [0.5, 0.8, 0.3, 0.9, 0.6]:
            metrics = ExperimentMetrics(macro_f1=f1)
            result = ExperimentResult(name=f"exp_{f1}", test_metrics=metrics)
            results.append(result)
        
        top2 = rank_experiments(results, metric="macro_f1", top_k=2)
        
        assert len(top2) == 2
        assert top2[0].test_metrics.macro_f1 == 0.9
        assert top2[1].test_metrics.macro_f1 == 0.8


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_result_without_test_metrics(self, temp_registry_dir):
        """Should handle result without test metrics."""
        registry = ExperimentRegistry(temp_registry_dir)
        
        result = ExperimentResult(
            name="no_metrics",
            model_type="lstm",
            test_metrics=None,
        )
        
        # Should not raise
        exp_id = registry.register(result)
        retrieved = registry.get(exp_id)
        
        assert retrieved.test_metrics is None
    
    def test_empty_training_history(self):
        """Should handle empty training history."""
        result = ExperimentResult(
            name="empty_history",
            training_history=[],
        )
        
        assert len(result.training_history) == 0
        
        # Should serialize correctly
        d = result.to_dict()
        assert d["training_history"] == []
    
    def test_special_characters_in_name(self, temp_registry_dir):
        """Should handle special characters in name."""
        registry = ExperimentRegistry(temp_registry_dir)
        
        result = ExperimentResult(
            name="test with spaces & symbols!",
            model_type="lstm",
        )
        
        exp_id = registry.register(result)
        retrieved = registry.get(exp_id)
        
        assert retrieved.name == "test with spaces & symbols!"
