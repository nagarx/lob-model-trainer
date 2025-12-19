"""
Tests for configuration schema.
"""

import tempfile
from pathlib import Path
import pytest

from lobtrainer.config import (
    DataConfig,
    SequenceConfig,
    NormalizationConfig,
    ModelConfig,
    TrainConfig,
    ExperimentConfig,
    load_config,
    save_config,
)
from lobtrainer.config.schema import NormalizationStrategy, ModelType, LabelEncoding


class TestSequenceConfig:
    """Test SequenceConfig validation."""
    
    def test_default_values(self):
        """Default values should be valid."""
        config = SequenceConfig()
        assert config.window_size == 100
        assert config.stride == 10
    
    def test_invalid_window_size(self):
        """Window size must be >= 1."""
        with pytest.raises(ValueError, match="window_size must be >= 1"):
            SequenceConfig(window_size=0)
    
    def test_invalid_stride(self):
        """Stride must be >= 1."""
        with pytest.raises(ValueError, match="stride must be >= 1"):
            SequenceConfig(stride=0)
    
    def test_stride_exceeds_window(self):
        """Stride should not exceed window size."""
        with pytest.raises(ValueError, match="stride.*should not exceed window_size"):
            SequenceConfig(window_size=50, stride=100)


class TestNormalizationConfig:
    """Test NormalizationConfig."""
    
    def test_default_strategy(self):
        """Default is per-day Z-score."""
        config = NormalizationConfig()
        assert config.strategy == NormalizationStrategy.ZSCORE_PER_DAY
    
    def test_default_excludes_time_regime(self):
        """Time regime (93) should be excluded by default."""
        config = NormalizationConfig()
        assert 93 in config.exclude_features  # TIME_REGIME index
    
    def test_invalid_eps(self):
        """Eps must be > 0."""
        with pytest.raises(ValueError, match="eps must be > 0"):
            NormalizationConfig(eps=0)


class TestModelConfig:
    """Test ModelConfig."""
    
    def test_default_values(self):
        """Default values should be valid."""
        config = ModelConfig()
        assert config.model_type == ModelType.LSTM
        assert config.hidden_size == 64
        assert config.num_layers == 2
    
    def test_invalid_dropout(self):
        """Dropout must be in [0, 1]."""
        with pytest.raises(ValueError, match="dropout must be in"):
            ModelConfig(dropout=1.5)
        with pytest.raises(ValueError, match="dropout must be in"):
            ModelConfig(dropout=-0.1)


class TestTrainConfig:
    """Test TrainConfig."""
    
    def test_default_seed(self):
        """Default seed is 42."""
        config = TrainConfig()
        assert config.seed == 42
    
    def test_invalid_batch_size(self):
        """Batch size must be >= 1."""
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            TrainConfig(batch_size=0)
    
    def test_invalid_learning_rate(self):
        """Learning rate must be > 0."""
        with pytest.raises(ValueError, match="learning_rate must be > 0"):
            TrainConfig(learning_rate=0)


class TestExperimentConfig:
    """Test ExperimentConfig serialization."""
    
    def test_default_config(self):
        """Default config should be valid."""
        config = ExperimentConfig()
        assert config.data.feature_count == 98
        assert config.model.input_size == 98
    
    def test_to_dict(self):
        """Config should serialize to dict."""
        config = ExperimentConfig(name="test")
        data = config.to_dict()
        assert data["name"] == "test"
        assert data["data"]["feature_count"] == 98
        assert data["model"]["model_type"] == "lstm"  # Enum to string
    
    def test_yaml_round_trip(self):
        """Config should survive YAML serialization."""
        config = ExperimentConfig(
            name="test_experiment",
            description="Test description",
            tags=["test", "baseline"],
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            config.to_yaml(str(path))
            loaded = ExperimentConfig.from_yaml(str(path))
        
        assert loaded.name == config.name
        assert loaded.description == config.description
        assert loaded.tags == config.tags
        assert loaded.data.feature_count == config.data.feature_count
    
    def test_json_round_trip(self):
        """Config should survive JSON serialization."""
        config = ExperimentConfig(name="json_test")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            config.to_json(str(path))
            loaded = ExperimentConfig.from_json(str(path))
        
        assert loaded.name == config.name


class TestLoadSaveConfig:
    """Test convenience load/save functions."""
    
    def test_load_yaml(self):
        """load_config detects YAML format."""
        config = ExperimentConfig(name="yaml_test")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            save_config(config, str(path))
            loaded = load_config(str(path))
        
        assert loaded.name == "yaml_test"
    
    def test_load_json(self):
        """load_config detects JSON format."""
        config = ExperimentConfig(name="json_test")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            save_config(config, str(path))
            loaded = load_config(str(path))
        
        assert loaded.name == "json_test"
    
    def test_unsupported_format(self):
        """Unsupported format raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported config format"):
            load_config("config.txt")

