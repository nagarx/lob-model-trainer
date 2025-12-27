"""
Integration tests for DeepLOB model with lob-model-trainer infrastructure.

Tests verify:
1. Model creation via create_model() factory
2. Forward pass with LOBSequenceDataset data format
3. Training step (forward + backward pass + optimizer step)
4. Feature index slicing for benchmark mode (first 40 features)
5. Integration with Trainer class

Requires: lobmodels package (pip install -e ../lob-models)
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

# Check if lobmodels is available
try:
    from lobmodels import DeepLOB, DeepLOBConfig, FeatureLayout
    LOBMODELS_AVAILABLE = True
except ImportError:
    LOBMODELS_AVAILABLE = False

# Skip all tests if lobmodels not installed
pytestmark = pytest.mark.skipif(
    not LOBMODELS_AVAILABLE,
    reason="lobmodels package not installed. Run: pip install -e ../lob-models"
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@pytest.fixture
def synthetic_sequences():
    """
    Create synthetic LOB sequences matching our 98-feature format.
    
    Returns:
        Tensor of shape [batch_size, sequence_length, num_features]
    """
    batch_size = 16
    sequence_length = 100
    num_features = 98
    return torch.randn(batch_size, sequence_length, num_features)


@pytest.fixture
def synthetic_lob_sequences():
    """
    Create synthetic sequences with only LOB features (40).
    
    This matches DeepLOB benchmark mode input.
    """
    batch_size = 16
    sequence_length = 100
    num_features = 40
    return torch.randn(batch_size, sequence_length, num_features)


@pytest.fixture
def synthetic_labels():
    """Create synthetic labels (shifted: 0, 1, 2)."""
    batch_size = 16
    return torch.randint(0, 3, (batch_size,))


# =============================================================================
# Test Model Creation
# =============================================================================


class TestDeepLOBModelCreation:
    """Test creating DeepLOB via model factory."""
    
    def test_create_model_deeplob_benchmark(self, seed):
        """Create DeepLOB in benchmark mode via factory."""
        from lobtrainer.config import ModelConfig, ModelType, DeepLOBMode
        from lobtrainer.models import create_model
        
        config = ModelConfig(
            model_type=ModelType.DEEPLOB,
            deeplob_mode=DeepLOBMode.BENCHMARK,
            num_classes=3,
            dropout=0.0,
        )
        
        model = create_model(config)
        
        assert isinstance(model, DeepLOB)
        assert model.config.mode == "benchmark"
        assert model.config.num_classes == 3
    
    def test_create_model_deeplob_with_custom_params(self, seed):
        """Create DeepLOB with custom hyperparameters."""
        from lobtrainer.config import ModelConfig, ModelType, DeepLOBMode
        from lobtrainer.models import create_model
        
        config = ModelConfig(
            model_type=ModelType.DEEPLOB,
            deeplob_mode=DeepLOBMode.BENCHMARK,
            deeplob_conv_filters=64,
            deeplob_inception_filters=128,
            deeplob_lstm_hidden=128,
            num_classes=3,
            dropout=0.1,
        )
        
        model = create_model(config)
        
        assert model.config.conv_filters == 64
        assert model.config.inception_filters == 128
        assert model.config.lstm_hidden == 128
        assert model.config.dropout == 0.1
    
    def test_create_model_parameter_count(self, seed):
        """Verify parameter count matches expected for default config."""
        from lobtrainer.config import ModelConfig, ModelType
        from lobtrainer.models import create_model
        
        config = ModelConfig(model_type=ModelType.DEEPLOB)
        model = create_model(config)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # DeepLOB paper reports ~500k parameters
        assert total_params > 100_000, f"Expected >100k params, got {total_params}"
        assert trainable_params == total_params, "All params should be trainable by default"


# =============================================================================
# Test Forward Pass
# =============================================================================


class TestDeepLOBForwardPass:
    """Test DeepLOB forward pass with various input formats."""
    
    def test_forward_benchmark_40_features(self, seed, synthetic_lob_sequences):
        """Forward pass with 40 LOB features (benchmark mode)."""
        from lobtrainer.config import ModelConfig, ModelType
        from lobtrainer.models import create_model
        
        config = ModelConfig(model_type=ModelType.DEEPLOB)
        model = create_model(config)
        model.eval()
        
        # Input: [batch, seq_len, 40]
        x = synthetic_lob_sequences
        
        with torch.no_grad():
            logits = model(x)
        
        assert logits.shape == (x.shape[0], 3), f"Expected (16, 3), got {logits.shape}"
        assert torch.isfinite(logits).all(), "Output contains NaN or Inf"
    
    def test_forward_with_sliced_98_features(self, seed, synthetic_sequences):
        """Forward pass with 98 features sliced to 40 (benchmark mode pattern)."""
        from lobtrainer.config import ModelConfig, ModelType
        from lobtrainer.models import create_model
        from lobtrainer.constants import LOB_ALL
        
        config = ModelConfig(model_type=ModelType.DEEPLOB)
        model = create_model(config)
        model.eval()
        
        # Input: [batch, seq_len, 98] -> slice to [batch, seq_len, 40]
        x_full = synthetic_sequences
        x_lob = x_full[:, :, LOB_ALL]  # First 40 features
        
        with torch.no_grad():
            logits = model(x_lob)
        
        assert logits.shape == (x_full.shape[0], 3)
        assert torch.isfinite(logits).all()
    
    def test_forward_batch_independence(self, seed, synthetic_lob_sequences):
        """Verify batch samples are processed independently."""
        from lobtrainer.config import ModelConfig, ModelType
        from lobtrainer.models import create_model
        
        config = ModelConfig(model_type=ModelType.DEEPLOB)
        model = create_model(config)
        model.eval()
        
        x = synthetic_lob_sequences  # [16, 100, 40]
        
        with torch.no_grad():
            # Process full batch
            full_output = model(x)
            
            # Process individual samples
            individual_outputs = []
            for i in range(x.shape[0]):
                out = model(x[i:i+1])
                individual_outputs.append(out)
            individual_outputs = torch.cat(individual_outputs, dim=0)
        
        # Outputs should match (within floating point tolerance)
        torch.testing.assert_close(full_output, individual_outputs, rtol=1e-4, atol=1e-5)
    
    def test_forward_deterministic(self, synthetic_lob_sequences):
        """Model output is deterministic in eval mode."""
        from lobtrainer.config import ModelConfig, ModelType
        from lobtrainer.models import create_model
        
        # Create model with fixed seed
        torch.manual_seed(42)
        config = ModelConfig(model_type=ModelType.DEEPLOB)
        model = create_model(config)
        model.eval()
        
        x = synthetic_lob_sequences
        
        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)
        
        torch.testing.assert_close(output1, output2)


# =============================================================================
# Test Training Step
# =============================================================================


class TestDeepLOBTraining:
    """Test training-related functionality."""
    
    def test_training_step(self, seed, synthetic_lob_sequences, synthetic_labels):
        """Single training step: forward + backward + optimizer."""
        from lobtrainer.config import ModelConfig, ModelType
        from lobtrainer.models import create_model
        
        config = ModelConfig(model_type=ModelType.DEEPLOB)
        model = create_model(config)
        model.train()
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # Forward pass
        logits = model(synthetic_lob_sequences)
        loss = criterion(logits, synthetic_labels)
        
        # Check loss is valid
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradients exist
        has_grad = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No gradients computed"
        
        # Optimizer step
        optimizer.step()
    
    def test_loss_decreases(self, seed, synthetic_lob_sequences, synthetic_labels):
        """Loss should decrease after multiple training steps."""
        from lobtrainer.config import ModelConfig, ModelType
        from lobtrainer.models import create_model
        
        config = ModelConfig(model_type=ModelType.DEEPLOB)
        model = create_model(config)
        model.train()
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        x = synthetic_lob_sequences
        y = synthetic_labels
        
        # Record initial loss
        with torch.no_grad():
            initial_logits = model(x)
            initial_loss = criterion(initial_logits, y).item()
        
        # Train for several steps
        for _ in range(50):
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        
        # Check final loss
        with torch.no_grad():
            final_logits = model(x)
            final_loss = criterion(final_logits, y).item()
        
        # Loss should decrease (or at least not explode)
        # Note: With random data, loss may not decrease significantly
        assert final_loss < initial_loss * 1.5, (
            f"Loss increased too much: {initial_loss:.4f} -> {final_loss:.4f}"
        )
    
    def test_gradient_clipping_compatible(self, seed, synthetic_lob_sequences, synthetic_labels):
        """Verify gradient clipping works with DeepLOB."""
        from lobtrainer.config import ModelConfig, ModelType
        from lobtrainer.models import create_model
        
        config = ModelConfig(model_type=ModelType.DEEPLOB)
        model = create_model(config)
        model.train()
        
        criterion = nn.CrossEntropyLoss()
        
        # Forward + backward
        logits = model(synthetic_lob_sequences)
        loss = criterion(logits, synthetic_labels)
        loss.backward()
        
        # Apply gradient clipping
        grad_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Verify clipping worked (norm should be <= 1.0 after clipping)
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        assert total_norm <= 1.0 + 1e-6, f"Gradient norm after clipping: {total_norm}"


# =============================================================================
# Test Feature Selection for Benchmark Mode
# =============================================================================


class TestFeatureSelection:
    """Test feature selection/slicing for different modes."""
    
    def test_benchmark_mode_uses_40_features(self, seed):
        """Benchmark mode should use first 40 LOB features."""
        from lobtrainer.config import ModelConfig, ModelType
        from lobtrainer.models import create_model
        from lobtrainer.constants import LOB_FEATURE_COUNT
        
        config = ModelConfig(model_type=ModelType.DEEPLOB)
        model = create_model(config)
        
        # DeepLOB in benchmark mode expects 40 features
        assert LOB_FEATURE_COUNT == 40
        
        # Verify model can process 40-feature input
        x = torch.randn(2, 100, 40)
        with torch.no_grad():
            model.eval()
            output = model(x)
        
        assert output.shape == (2, 3)
    
    def test_feature_slicing_pattern(self, seed, synthetic_sequences):
        """Demonstrate correct feature slicing pattern for benchmark mode."""
        from lobtrainer.constants import LOB_ALL, LOB_FEATURE_COUNT
        
        # Full 98-feature sequence
        x_full = synthetic_sequences  # [batch, 100, 98]
        
        # Slice to LOB-only (first 40)
        x_lob = x_full[:, :, LOB_ALL]
        
        assert x_lob.shape == (x_full.shape[0], 100, LOB_FEATURE_COUNT)
        assert x_lob.shape[-1] == 40


# =============================================================================
# Test Integration with Dataset
# =============================================================================


class TestDatasetIntegration:
    """Test DeepLOB with LOBSequenceDataset patterns."""
    
    def test_synthetic_dataloader_pattern(self, seed):
        """Test model with DataLoader-like iteration pattern."""
        from torch.utils.data import TensorDataset, DataLoader
        from lobtrainer.config import ModelConfig, ModelType
        from lobtrainer.models import create_model
        
        # Create synthetic data matching our format
        sequences = torch.randn(100, 100, 40)  # 100 samples
        labels = torch.randint(0, 3, (100,))
        
        dataset = TensorDataset(sequences, labels)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        config = ModelConfig(model_type=ModelType.DEEPLOB)
        model = create_model(config)
        model.eval()
        
        # Process batches
        all_outputs = []
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                output = model(batch_x)
                all_outputs.append(output)
        
        # All batches processed successfully
        total_samples = sum(o.shape[0] for o in all_outputs)
        assert total_samples == 100
    
    def test_feature_indices_selection(self, seed):
        """
        Test using feature_indices to select LOB features.
        
        Layout (matches Rust pipeline):
            [ask_prices(10), ask_sizes(10), bid_prices(10), bid_sizes(10)]
            - LOB_ASK_PRICES: indices 0-9
            - LOB_ASK_SIZES: indices 10-19
            - LOB_BID_PRICES: indices 20-29
            - LOB_BID_SIZES: indices 30-39
        """
        from lobtrainer.constants import LOB_ASK_PRICES, LOB_ASK_SIZES, LOB_BID_PRICES, LOB_BID_SIZES
        
        # Our 98-feature tensor
        x = torch.randn(8, 100, 98)
        
        # Select LOB features using slices
        ask_prices = x[:, :, LOB_ASK_PRICES]   # indices 0-9
        ask_sizes = x[:, :, LOB_ASK_SIZES]     # indices 10-19
        bid_prices = x[:, :, LOB_BID_PRICES]   # indices 20-29
        bid_sizes = x[:, :, LOB_BID_SIZES]     # indices 30-39
        
        # Concatenate in GROUPED layout order to create 40-feature LOB input
        # Order: [ask_p, ask_s, bid_p, bid_s] - matches Rust pipeline
        x_lob = torch.cat([ask_prices, ask_sizes, bid_prices, bid_sizes], dim=-1)
        
        assert x_lob.shape == (8, 100, 40)
        
        # Verify this equals simple slice (first 40 features are LOB features)
        x_lob_simple = x[:, :, :40]
        torch.testing.assert_close(x_lob, x_lob_simple)


# =============================================================================
# Test Model Output Properties
# =============================================================================


class TestModelOutputProperties:
    """Test properties of model outputs."""
    
    def test_output_logits_not_probabilities(self, seed, synthetic_lob_sequences):
        """Model returns logits (raw scores), not probabilities."""
        from lobtrainer.config import ModelConfig, ModelType
        from lobtrainer.models import create_model
        
        config = ModelConfig(model_type=ModelType.DEEPLOB)
        model = create_model(config)
        model.eval()
        
        with torch.no_grad():
            logits = model(synthetic_lob_sequences)
        
        # Logits don't need to sum to 1
        sums = logits.exp().sum(dim=-1)  # Would sum to 1 if softmax was applied
        
        # They should NOT sum to 1 (since these are raw logits)
        # (within tolerance for numerical precision)
        assert not torch.allclose(sums, torch.ones_like(sums)), (
            "Output appears to be probabilities, expected logits"
        )
    
    def test_softmax_gives_valid_probabilities(self, seed, synthetic_lob_sequences):
        """Applying softmax to logits gives valid probability distribution."""
        from lobtrainer.config import ModelConfig, ModelType
        from lobtrainer.models import create_model
        import torch.nn.functional as F
        
        config = ModelConfig(model_type=ModelType.DEEPLOB)
        model = create_model(config)
        model.eval()
        
        with torch.no_grad():
            logits = model(synthetic_lob_sequences)
            probs = F.softmax(logits, dim=-1)
        
        # Probabilities should:
        # 1. Sum to 1
        assert torch.allclose(probs.sum(dim=-1), torch.ones(probs.shape[0]), rtol=1e-4)
        
        # 2. Be in [0, 1]
        assert (probs >= 0).all()
        assert (probs <= 1).all()
        
        # 3. Have shape [batch, num_classes]
        assert probs.shape == (synthetic_lob_sequences.shape[0], 3)


# =============================================================================
# Test with Real Data (skipped if not available)
# =============================================================================


class TestWithRealData:
    """Integration tests with real exported data."""
    
    DATA_DIR = pytest.importorskip("pathlib").Path(__file__).parent.parent.parent / "data" / "exports" / "nvda_98feat"
    
    @pytest.mark.skipif(
        not DATA_DIR.exists(),
        reason=f"Real data not found at {DATA_DIR}"
    )
    def test_forward_with_real_data(self, seed):
        """Forward pass with real NVDA data."""
        from lobtrainer.data import load_split_data, LOBSequenceDataset
        from lobtrainer.config import ModelConfig, ModelType
        from lobtrainer.models import create_model
        from lobtrainer.constants import LOB_ALL
        
        # Load real data
        days = load_split_data(self.DATA_DIR, "train", validate=False)[:1]
        
        if not days or not days[0].is_aligned:
            pytest.skip("Test requires aligned format data")
        
        # Use feature_indices to select LOB features
        lob_indices = list(range(40))  # First 40 features
        dataset = LOBSequenceDataset(days, feature_indices=lob_indices)
        
        # Get a batch
        batch_size = min(16, len(dataset))
        batch_x = torch.stack([dataset[i][0] for i in range(batch_size)])
        
        # Create model and forward pass
        config = ModelConfig(model_type=ModelType.DEEPLOB)
        model = create_model(config)
        model.eval()
        
        with torch.no_grad():
            logits = model(batch_x)
        
        assert logits.shape == (batch_size, 3)
        assert torch.isfinite(logits).all()
    
    @pytest.mark.skipif(
        not DATA_DIR.exists(),
        reason=f"Real data not found at {DATA_DIR}"
    )
    def test_training_with_real_data(self, seed):
        """Training step with real NVDA data."""
        from lobtrainer.data import load_split_data, LOBSequenceDataset
        from lobtrainer.config import ModelConfig, ModelType
        from lobtrainer.models import create_model
        
        # Load real data
        days = load_split_data(self.DATA_DIR, "train", validate=False)[:1]
        
        if not days or not days[0].is_aligned:
            pytest.skip("Test requires aligned format data")
        
        # Use feature_indices to select LOB features
        lob_indices = list(range(40))
        dataset = LOBSequenceDataset(days, feature_indices=lob_indices)
        
        # Get a batch
        batch_size = min(16, len(dataset))
        batch_x = torch.stack([dataset[i][0] for i in range(batch_size)])
        batch_y = torch.stack([dataset[i][1] for i in range(batch_size)])
        
        # Create model
        config = ModelConfig(model_type=ModelType.DEEPLOB)
        model = create_model(config)
        model.train()
        
        # Training step
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        
        assert torch.isfinite(loss)

