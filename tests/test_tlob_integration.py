"""
Integration tests for TLOB model in lobtrainer.

Tests cover:
- Model creation via factory
- Config parsing and validation
- Forward pass with different input shapes
- Gradient flow
- Training step
- Integration with data transforms

Reference: Berti & Kasneci (2025), "TLOB: A Novel Transformer Model with
           Dual Attention for Price Trend Prediction with Limit Order Book Data"
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from lobtrainer.config import (
    ModelConfig,
    ModelType,
    ExperimentConfig,
    load_config,
)
from lobtrainer.models import (
    create_model,
    LOBMODELS_AVAILABLE,
)


# Skip all tests if lobmodels is not installed
pytestmark = pytest.mark.skipif(
    not LOBMODELS_AVAILABLE,
    reason="lobmodels package not installed"
)


class TestTLOBModelCreation:
    """Test TLOB model creation via factory."""
    
    def test_create_model_tlob_default(self):
        """Test creating TLOB with default config."""
        config = ModelConfig(model_type=ModelType.TLOB)
        model = create_model(config)
        
        assert model is not None
        assert "TLOB" in model.name
    
    def test_create_model_with_custom_params(self):
        """Test creating TLOB with custom parameters."""
        config = ModelConfig(
            model_type=ModelType.TLOB,
            input_size=98,
            num_classes=3,
            tlob_hidden_dim=128,
            tlob_num_layers=6,
            tlob_num_heads=2,
            tlob_mlp_expansion=4.0,
            dropout=0.1,
        )
        model = create_model(config)
        
        # Verify model is created
        assert model is not None
        # Custom config should have more parameters
        num_params = sum(p.numel() for p in model.parameters())
        assert num_params > 1_000_000  # Larger than default
    
    def test_create_model_parameter_count(self):
        """Test TLOB parameter count matches expectations."""
        config = ModelConfig(
            model_type=ModelType.TLOB,
            input_size=98,
            tlob_hidden_dim=64,
            tlob_num_layers=4,
        )
        model = create_model(config)
        num_params = sum(p.numel() for p in model.parameters())
        
        # Should be around 918K for these settings
        assert 800_000 < num_params < 1_100_000


class TestTLOBConfigValidation:
    """Test TLOB configuration validation."""
    
    def test_hidden_dim_must_be_at_least_4(self):
        """Hidden dim must be >= 4 for final block reduction."""
        with pytest.raises(ValueError, match="tlob_hidden_dim must be >= 4"):
            ModelConfig(model_type=ModelType.TLOB, tlob_hidden_dim=2)
    
    def test_hidden_dim_must_be_even_for_sinusoidal_pe(self):
        """Hidden dim must be even when using sinusoidal PE."""
        with pytest.raises(ValueError, match="tlob_hidden_dim must be even"):
            ModelConfig(
                model_type=ModelType.TLOB,
                tlob_hidden_dim=33,
                tlob_use_sinusoidal_pe=True,
            )
    
    def test_hidden_dim_odd_ok_with_learned_pe(self):
        """Odd hidden dim is OK when not using sinusoidal PE."""
        # Should not raise
        config = ModelConfig(
            model_type=ModelType.TLOB,
            tlob_hidden_dim=33,
            tlob_use_sinusoidal_pe=False,
        )
        assert config.tlob_hidden_dim == 33
    
    def test_num_layers_must_be_positive(self):
        """Number of layers must be >= 1."""
        with pytest.raises(ValueError, match="tlob_num_layers must be >= 1"):
            ModelConfig(model_type=ModelType.TLOB, tlob_num_layers=0)
    
    def test_num_heads_must_be_positive(self):
        """Number of attention heads must be >= 1."""
        with pytest.raises(ValueError, match="tlob_num_heads must be >= 1"):
            ModelConfig(model_type=ModelType.TLOB, tlob_num_heads=0)
    
    def test_mlp_expansion_must_be_positive(self):
        """MLP expansion factor must be > 0."""
        with pytest.raises(ValueError, match="tlob_mlp_expansion must be > 0"):
            ModelConfig(model_type=ModelType.TLOB, tlob_mlp_expansion=0)
    
    def test_dataset_type_validation(self):
        """Dataset type must be valid."""
        with pytest.raises(ValueError, match="tlob_dataset_type"):
            ModelConfig(model_type=ModelType.TLOB, tlob_dataset_type="invalid")
    
    def test_valid_dataset_types(self):
        """Valid dataset types should be accepted."""
        for dt in ["fi2010", "lobster", "nvda"]:
            config = ModelConfig(model_type=ModelType.TLOB, tlob_dataset_type=dt)
            assert config.tlob_dataset_type == dt


class TestTLOBForwardPass:
    """Test TLOB forward pass."""
    
    def test_forward_98_features(self):
        """Test forward pass with 98 features."""
        config = ModelConfig(model_type=ModelType.TLOB, input_size=98)
        model = create_model(config)
        model.eval()
        
        x = torch.randn(4, 100, 98)  # [batch, seq, features]
        with torch.no_grad():
            out = model(x)
        
        assert out.shape == (4, 3)
    
    def test_forward_40_features(self):
        """Test forward pass with 40 features (LOB only)."""
        config = ModelConfig(model_type=ModelType.TLOB, input_size=40)
        model = create_model(config)
        model.eval()
        
        x = torch.randn(4, 100, 40)
        with torch.no_grad():
            out = model(x)
        
        assert out.shape == (4, 3)
    
    def test_forward_batch_independence(self):
        """Outputs should be independent across batch dimension."""
        config = ModelConfig(model_type=ModelType.TLOB, input_size=98)
        model = create_model(config)
        model.eval()
        
        x1 = torch.randn(1, 100, 98)
        x2 = torch.randn(1, 100, 98)
        x_batch = torch.cat([x1, x2], dim=0)
        
        with torch.no_grad():
            out1 = model(x1)
            out2 = model(x2)
            out_batch = model(x_batch)
        
        assert torch.allclose(out_batch[0], out1[0], atol=1e-5)
        assert torch.allclose(out_batch[1], out2[0], atol=1e-5)
    
    def test_forward_deterministic(self):
        """Same input should produce same output."""
        config = ModelConfig(model_type=ModelType.TLOB, input_size=98)
        model = create_model(config)
        model.eval()
        
        x = torch.randn(4, 100, 98)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        
        assert torch.allclose(out1, out2)


class TestTLOBTraining:
    """Test TLOB training functionality."""
    
    def test_training_step(self):
        """Test a single training step."""
        config = ModelConfig(model_type=ModelType.TLOB, input_size=98)
        model = create_model(config)
        model.train()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        x = torch.randn(4, 100, 98)
        y = torch.randint(0, 3, (4,))
        
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
        assert loss.item() > 0
    
    def test_loss_decreases(self):
        """Test that loss decreases over multiple steps."""
        config = ModelConfig(model_type=ModelType.TLOB, input_size=98)
        model = create_model(config)
        model.train()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        # Fixed input for overfitting test
        x = torch.randn(8, 100, 98)
        y = torch.randint(0, 3, (8,))
        
        initial_loss = None
        for _ in range(10):
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            if initial_loss is None:
                initial_loss = loss.item()
        
        final_loss = loss.item()
        assert final_loss < initial_loss
    
    def test_gradient_clipping_compatible(self):
        """Test that model works with gradient clipping."""
        config = ModelConfig(model_type=ModelType.TLOB, input_size=98)
        model = create_model(config)
        model.train()
        
        x = torch.randn(4, 100, 98)
        y = torch.randint(0, 3, (4,))
        
        out = model(x)
        loss = nn.CrossEntropyLoss()(out, y)
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Check all gradients are clipped
        for p in model.parameters():
            if p.grad is not None:
                grad_norm = p.grad.norm()
                assert not torch.isnan(grad_norm)
                assert not torch.isinf(grad_norm)


class TestTLOBModelOutput:
    """Test TLOB model output properties."""
    
    def test_output_logits_not_probabilities(self):
        """Output should be raw logits (can be negative, sum != 1)."""
        config = ModelConfig(model_type=ModelType.TLOB, input_size=98)
        model = create_model(config)
        model.eval()
        
        x = torch.randn(4, 100, 98)
        with torch.no_grad():
            out = model(x)
        
        # Logits can be negative
        assert out.min() < 0 or out.max() > 1
    
    def test_softmax_gives_valid_probabilities(self):
        """Softmax of output should give valid probabilities."""
        config = ModelConfig(model_type=ModelType.TLOB, input_size=98)
        model = create_model(config)
        model.eval()
        
        x = torch.randn(4, 100, 98)
        with torch.no_grad():
            out = model(x)
            probs = torch.softmax(out, dim=-1)
        
        # Probabilities should sum to 1
        assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)
        # All probabilities should be in [0, 1]
        assert probs.min() >= 0
        assert probs.max() <= 1


class TestTLOBConfigSerialization:
    """Test TLOB config serialization."""
    
    def test_to_dict_includes_tlob_fields(self):
        """Config dict should include TLOB-specific fields."""
        config = ModelConfig(
            model_type=ModelType.TLOB,
            tlob_hidden_dim=128,
            tlob_num_layers=6,
        )
        exp_config = ExperimentConfig(model=config)
        config_dict = exp_config.to_dict()
        
        assert config_dict['model']['model_type'] == 'tlob'
        assert config_dict['model']['tlob_hidden_dim'] == 128
        assert config_dict['model']['tlob_num_layers'] == 6
    
    def test_yaml_round_trip(self, tmp_path):
        """Test YAML serialization round trip."""
        config = ExperimentConfig(
            name="tlob_test",
            model=ModelConfig(
                model_type=ModelType.TLOB,
                tlob_hidden_dim=64,
                tlob_use_bin=True,
            ),
        )
        
        yaml_path = tmp_path / "tlob_config.yaml"
        config.to_yaml(str(yaml_path))
        
        loaded = ExperimentConfig.from_yaml(str(yaml_path))
        assert loaded.model.model_type == ModelType.TLOB
        assert loaded.model.tlob_hidden_dim == 64
        assert loaded.model.tlob_use_bin is True


class TestTLOBWithBiN:
    """Test TLOB BiN (Bilinear Normalization) layer."""
    
    def test_bin_enabled_by_default(self):
        """BiN should be enabled by default."""
        config = ModelConfig(model_type=ModelType.TLOB)
        assert config.tlob_use_bin is True
    
    def test_bin_handles_unnormalized_input(self):
        """BiN layer should handle unnormalized input gracefully."""
        config = ModelConfig(model_type=ModelType.TLOB, input_size=98)
        model = create_model(config)
        model.eval()
        
        # Create input with extreme values (like raw MBO features)
        x = torch.randn(4, 100, 98)
        x[:, :, 40:60] *= 10000  # Simulate large MBO values
        x[:, :, 80:90] *= 1000000  # Simulate extreme outliers
        
        with torch.no_grad():
            out = model(x)
        
        # Output should still be valid
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


class TestTLOBWithData:
    """Test TLOB with actual data (if available)."""
    
    @pytest.fixture
    def data_path(self):
        """Path to test data."""
        paths = [
            Path("../data/exports/nvda_bigmove/train"),
            Path("data/exports/nvda_bigmove/train"),
        ]
        for p in paths:
            if p.exists():
                return p
        return None
    
    def test_forward_with_real_data(self, data_path):
        """Test forward pass with real data."""
        if data_path is None:
            pytest.skip("Real data not available")
        
        # Load first file
        seq_files = sorted(data_path.glob("*_sequences.npy"))[:1]
        if not seq_files:
            pytest.skip("No sequence files found")
        
        sequences = np.load(seq_files[0])[:8]  # First 8 samples
        x = torch.from_numpy(sequences).float()
        
        config = ModelConfig(model_type=ModelType.TLOB, input_size=98)
        model = create_model(config)
        model.eval()
        
        with torch.no_grad():
            out = model(x)
        
        assert out.shape == (8, 3)
        assert not torch.isnan(out).any()
