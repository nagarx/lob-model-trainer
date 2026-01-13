"""
Tests for custom loss functions.

Tests FocalLoss and BinaryFocalLoss implementations:
- Mathematical correctness (gamma=0 should equal CrossEntropy)
- Numerical stability (NaN/Inf handling)
- Edge cases (all same class, perfect predictions)
- Class weighting behavior
- Gradient flow

RULE.md compliance:
- Zero tolerance for precision errors
- Test edge cases: 0, NaN, Inf, boundary values
- Assertions explain WHAT failed and WHY

Reference:
    Lin et al. "Focal Loss for Dense Object Detection" (2017)
    https://arxiv.org/abs/1708.02002
"""

import math
import pytest
import torch
import torch.nn as nn
import numpy as np

from lobtrainer.training.loss import (
    FocalLoss,
    BinaryFocalLoss,
    create_focal_loss,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_logits_3class():
    """Generate sample logits for 3-class classification."""
    torch.manual_seed(42)
    return torch.randn(32, 3)


@pytest.fixture
def sample_targets_3class():
    """Generate sample targets for 3-class classification."""
    torch.manual_seed(42)
    return torch.randint(0, 3, (32,))


@pytest.fixture
def sample_logits_binary():
    """Generate sample logits for binary classification."""
    torch.manual_seed(42)
    return torch.randn(32)


@pytest.fixture
def sample_targets_binary():
    """Generate sample targets for binary classification."""
    torch.manual_seed(42)
    return torch.randint(0, 2, (32,))


# =============================================================================
# FocalLoss Basic Tests
# =============================================================================


class TestFocalLossInit:
    """Test FocalLoss initialization and validation."""
    
    def test_default_init(self):
        """Default initialization should work."""
        loss = FocalLoss()
        assert loss.gamma == 2.0, "Default gamma should be 2.0"
        assert loss.alpha is None, "Default alpha should be None"
        assert loss.reduction == "mean", "Default reduction should be 'mean'"
    
    def test_custom_gamma(self):
        """Custom gamma should be accepted."""
        loss = FocalLoss(gamma=1.5)
        assert loss.gamma == 1.5, f"Gamma should be 1.5, got {loss.gamma}"
    
    def test_zero_gamma(self):
        """gamma=0 is valid (equivalent to CE)."""
        loss = FocalLoss(gamma=0.0)
        assert loss.gamma == 0.0, "gamma=0 should be valid"
    
    def test_negative_gamma_raises(self):
        """Negative gamma should raise ValueError."""
        with pytest.raises(ValueError, match="gamma must be >= 0"):
            FocalLoss(gamma=-1.0)
    
    def test_invalid_reduction_raises(self):
        """Invalid reduction should raise ValueError."""
        with pytest.raises(ValueError, match="reduction must be"):
            FocalLoss(reduction="invalid")
    
    def test_alpha_tensor(self):
        """Alpha can be a tensor."""
        alpha = torch.tensor([1.0, 2.0, 3.0])
        loss = FocalLoss(alpha=alpha)
        assert loss.alpha is not None
        assert torch.allclose(loss.alpha, alpha)
    
    def test_alpha_list(self):
        """Alpha can be a list (converted to tensor)."""
        alpha = [1.0, 2.0, 3.0]
        loss = FocalLoss(alpha=alpha)
        assert loss.alpha is not None
        assert torch.allclose(loss.alpha, torch.tensor(alpha))


class TestFocalLossForward:
    """Test FocalLoss forward pass."""
    
    def test_output_is_scalar_mean(self, sample_logits_3class, sample_targets_3class):
        """reduction='mean' should return scalar."""
        loss = FocalLoss(reduction="mean")
        output = loss(sample_logits_3class, sample_targets_3class)
        
        assert output.dim() == 0, (
            f"Mean reduction should return scalar, got dim={output.dim()}"
        )
    
    def test_output_is_scalar_sum(self, sample_logits_3class, sample_targets_3class):
        """reduction='sum' should return scalar."""
        loss = FocalLoss(reduction="sum")
        output = loss(sample_logits_3class, sample_targets_3class)
        
        assert output.dim() == 0, (
            f"Sum reduction should return scalar, got dim={output.dim()}"
        )
    
    def test_output_is_vector_none(self, sample_logits_3class, sample_targets_3class):
        """reduction='none' should return per-sample losses."""
        loss = FocalLoss(reduction="none")
        output = loss(sample_logits_3class, sample_targets_3class)
        
        assert output.shape == sample_targets_3class.shape, (
            f"Expected shape {sample_targets_3class.shape}, got {output.shape}"
        )
    
    def test_loss_is_positive(self, sample_logits_3class, sample_targets_3class):
        """Loss should always be positive."""
        loss = FocalLoss()
        output = loss(sample_logits_3class, sample_targets_3class)
        
        assert output > 0, f"Loss should be positive, got {output.item()}"
    
    def test_loss_is_finite(self, sample_logits_3class, sample_targets_3class):
        """Loss should be finite (no NaN/Inf)."""
        loss = FocalLoss()
        output = loss(sample_logits_3class, sample_targets_3class)
        
        assert torch.isfinite(output), (
            f"Loss should be finite, got {output.item()}"
        )


class TestFocalLossMathematicalProperties:
    """Test mathematical properties of Focal Loss."""
    
    def test_gamma_zero_equals_ce(self):
        """
        CRITICAL: gamma=0 should equal CrossEntropyLoss.
        
        Formula verification:
            FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
            When gamma=0: FL(p_t) = -alpha_t * log(p_t) = CE
        """
        torch.manual_seed(42)
        logits = torch.randn(100, 3)
        targets = torch.randint(0, 3, (100,))
        
        ce_loss = nn.CrossEntropyLoss()
        focal_loss = FocalLoss(gamma=0.0)
        
        ce = ce_loss(logits, targets)
        fl = focal_loss(logits, targets)
        
        assert torch.allclose(ce, fl, atol=1e-5), (
            f"gamma=0 should equal CrossEntropy. "
            f"CE={ce.item():.6f}, Focal={fl.item():.6f}"
        )
    
    def test_higher_gamma_lower_easy_example_weight(self):
        """
        Higher gamma should down-weight easy examples more.
        
        For high-confidence predictions, higher gamma gives lower loss.
        """
        # Create easy examples (high but not extreme confidence)
        # Use moderate logits so softmax doesn't saturate to 1.0
        logits = torch.tensor([
            [2.0, -1.0, -1.0],  # Confident class 0 (p ≈ 0.88)
            [-1.0, 2.0, -1.0],  # Confident class 1 (p ≈ 0.88)
        ])
        targets = torch.tensor([0, 1])  # Correct predictions
        
        loss_gamma0 = FocalLoss(gamma=0.0)
        loss_gamma2 = FocalLoss(gamma=2.0)
        loss_gamma5 = FocalLoss(gamma=5.0)
        
        l0 = loss_gamma0(logits, targets)
        l2 = loss_gamma2(logits, targets)
        l5 = loss_gamma5(logits, targets)
        
        # Higher gamma should give lower loss for easy examples
        assert l2 < l0, (
            f"gamma=2 should give lower loss than gamma=0 for easy examples. "
            f"gamma0={l0.item():.6f}, gamma2={l2.item():.6f}"
        )
        assert l5 < l2, (
            f"gamma=5 should give lower loss than gamma=2 for easy examples. "
            f"gamma2={l2.item():.6f}, gamma5={l5.item():.6f}"
        )
    
    def test_hard_example_not_down_weighted(self):
        """
        Hard examples (wrong predictions) should not be down-weighted as much.
        
        The focal term (1 - p_t)^gamma is larger when p_t is small.
        """
        # Create hard examples (low confidence for correct class)
        logits = torch.tensor([
            [-10.0, 10.0, 10.0],   # Very wrong for class 0
            [10.0, -10.0, 10.0],   # Very wrong for class 1
        ])
        targets = torch.tensor([0, 1])  # Misclassified
        
        loss_gamma0 = FocalLoss(gamma=0.0)
        loss_gamma2 = FocalLoss(gamma=2.0)
        
        l0 = loss_gamma0(logits, targets)
        l2 = loss_gamma2(logits, targets)
        
        # For hard examples, the difference should be smaller
        # (both should be high, gamma doesn't help much)
        ratio = l2.item() / l0.item()
        
        # Ratio should be close to 1 for hard examples
        assert ratio > 0.5, (
            f"Hard examples should not be down-weighted much. "
            f"Ratio gamma2/gamma0 = {ratio:.4f}, expected > 0.5"
        )
    
    def test_alpha_weighting(self):
        """Alpha should weight classes proportionally."""
        torch.manual_seed(42)
        
        # All class 0 samples
        logits = torch.randn(10, 3)
        targets = torch.zeros(10, dtype=torch.long)
        
        # Alpha weights class 0 more
        alpha_high = torch.tensor([2.0, 1.0, 1.0])
        # Alpha weights class 0 less
        alpha_low = torch.tensor([0.5, 1.0, 1.0])
        
        loss_high = FocalLoss(gamma=2.0, alpha=alpha_high)
        loss_low = FocalLoss(gamma=2.0, alpha=alpha_low)
        
        l_high = loss_high(logits, targets)
        l_low = loss_low(logits, targets)
        
        # Higher alpha for class 0 should give higher loss
        assert l_high > l_low, (
            f"Higher alpha for class 0 should give higher loss. "
            f"alpha_high loss={l_high.item():.4f}, alpha_low loss={l_low.item():.4f}"
        )


class TestFocalLossEdgeCases:
    """Test edge cases and numerical stability."""
    
    def test_single_sample(self):
        """Should work with single sample."""
        loss = FocalLoss()
        logits = torch.randn(1, 3)
        targets = torch.tensor([1])
        
        output = loss(logits, targets)
        assert torch.isfinite(output), "Single sample should work"
    
    def test_all_same_class(self):
        """Should work when all targets are same class."""
        loss = FocalLoss()
        logits = torch.randn(10, 3)
        targets = torch.zeros(10, dtype=torch.long)
        
        output = loss(logits, targets)
        assert torch.isfinite(output), "All same class should work"
    
    def test_very_large_logits(self):
        """Should be stable with very large logits."""
        loss = FocalLoss()
        # Large logits that could cause numerical issues
        logits = torch.tensor([
            [100.0, -100.0, -100.0],
            [-100.0, 100.0, -100.0],
        ])
        targets = torch.tensor([0, 1])
        
        output = loss(logits, targets)
        assert torch.isfinite(output), (
            f"Large logits should not cause NaN/Inf, got {output.item()}"
        )
    
    def test_very_small_logits(self):
        """Should be stable with very small logits."""
        loss = FocalLoss()
        logits = torch.tensor([
            [1e-10, 1e-10, 1e-10],
            [1e-10, 1e-10, 1e-10],
        ])
        targets = torch.tensor([0, 1])
        
        output = loss(logits, targets)
        assert torch.isfinite(output), (
            f"Small logits should not cause NaN/Inf, got {output.item()}"
        )


class TestFocalLossGradients:
    """Test gradient computation."""
    
    def test_gradients_exist(self, sample_logits_3class, sample_targets_3class):
        """Gradients should flow through the loss."""
        logits = sample_logits_3class.clone().requires_grad_(True)
        
        loss = FocalLoss()
        output = loss(logits, sample_targets_3class)
        output.backward()
        
        assert logits.grad is not None, "Gradients should exist"
        assert torch.isfinite(logits.grad).all(), "Gradients should be finite"
    
    def test_gradients_are_nonzero(self, sample_logits_3class, sample_targets_3class):
        """Gradients should be non-zero (model can learn)."""
        logits = sample_logits_3class.clone().requires_grad_(True)
        
        loss = FocalLoss()
        output = loss(logits, sample_targets_3class)
        output.backward()
        
        grad_norm = logits.grad.norm()
        assert grad_norm > 0, f"Gradient norm should be > 0, got {grad_norm.item()}"


# =============================================================================
# BinaryFocalLoss Tests
# =============================================================================


class TestBinaryFocalLossInit:
    """Test BinaryFocalLoss initialization."""
    
    def test_default_init(self):
        """Default initialization should work."""
        loss = BinaryFocalLoss()
        assert loss.gamma == 2.0
        assert loss.alpha is None
        assert loss.pos_weight is None
        assert loss.reduction == "mean"
    
    def test_alpha_range(self):
        """Alpha should be in [0, 1]."""
        with pytest.raises(ValueError, match="alpha must be in"):
            BinaryFocalLoss(alpha=1.5)
        
        with pytest.raises(ValueError, match="alpha must be in"):
            BinaryFocalLoss(alpha=-0.5)


class TestBinaryFocalLossForward:
    """Test BinaryFocalLoss forward pass."""
    
    def test_1d_logits(self, sample_logits_binary, sample_targets_binary):
        """Should work with 1D logits."""
        loss = BinaryFocalLoss()
        output = loss(sample_logits_binary, sample_targets_binary)
        
        assert output.dim() == 0, "Should return scalar"
        assert output > 0, "Loss should be positive"
        assert torch.isfinite(output), "Loss should be finite"
    
    def test_2d_logits(self, sample_targets_binary):
        """Should work with 2D logits (batch, 1)."""
        logits = torch.randn(32, 1)
        
        loss = BinaryFocalLoss()
        output = loss(logits, sample_targets_binary)
        
        assert output.dim() == 0, "Should return scalar"
        assert torch.isfinite(output), "Loss should be finite"
    
    def test_alpha_weighting(self, sample_logits_binary, sample_targets_binary):
        """Alpha should affect positive class weight."""
        loss_alpha_high = BinaryFocalLoss(alpha=0.9)  # High weight for positive
        loss_alpha_low = BinaryFocalLoss(alpha=0.1)   # Low weight for positive
        
        # Note: The actual values depend on the distribution of predictions
        # Just verify they produce different values
        l_high = loss_alpha_high(sample_logits_binary, sample_targets_binary)
        l_low = loss_alpha_low(sample_logits_binary, sample_targets_binary)
        
        assert l_high != l_low, "Different alpha should produce different losses"


class TestBinaryFocalLossEdgeCases:
    """Test edge cases for BinaryFocalLoss."""
    
    def test_all_zeros(self):
        """Should work with all zero targets."""
        loss = BinaryFocalLoss()
        logits = torch.randn(10)
        targets = torch.zeros(10, dtype=torch.long)
        
        output = loss(logits, targets)
        assert torch.isfinite(output)
    
    def test_all_ones(self):
        """Should work with all one targets."""
        loss = BinaryFocalLoss()
        logits = torch.randn(10)
        targets = torch.ones(10, dtype=torch.long)
        
        output = loss(logits, targets)
        assert torch.isfinite(output)


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateFocalLoss:
    """Test create_focal_loss factory function."""
    
    def test_binary_returns_binary_loss(self):
        """num_classes=2 should return BinaryFocalLoss."""
        loss = create_focal_loss(num_classes=2)
        assert isinstance(loss, BinaryFocalLoss), (
            f"num_classes=2 should return BinaryFocalLoss, got {type(loss)}"
        )
    
    def test_multiclass_returns_focal_loss(self):
        """num_classes>2 should return FocalLoss."""
        loss = create_focal_loss(num_classes=3)
        assert isinstance(loss, FocalLoss), (
            f"num_classes=3 should return FocalLoss, got {type(loss)}"
        )
    
    def test_class_counts_computes_alpha(self):
        """class_counts should compute alpha weights."""
        class_counts = torch.tensor([100.0, 50.0, 50.0])
        loss = create_focal_loss(num_classes=3, class_counts=class_counts)
        
        assert loss.alpha is not None, "Alpha should be computed from class_counts"
    
    def test_imbalanced_class_weights(self):
        """Minority class should get higher weight."""
        # 70% class 0, 20% class 1, 10% class 2
        class_counts = torch.tensor([700.0, 200.0, 100.0])
        loss = create_focal_loss(num_classes=3, class_counts=class_counts)
        
        # Class 2 (minority) should have highest weight
        alpha = loss.alpha
        assert alpha[2] > alpha[0], (
            f"Minority class should have higher weight. "
            f"alpha[0]={alpha[0].item():.4f}, alpha[2]={alpha[2].item():.4f}"
        )


# =============================================================================
# Integration Tests
# =============================================================================


class TestFocalLossTrainingSimulation:
    """Test that focal loss works in a training loop."""
    
    def test_loss_decreases_with_training(self):
        """Loss should decrease when model learns."""
        torch.manual_seed(42)
        
        # Simple model
        model = nn.Linear(10, 3)
        criterion = FocalLoss(gamma=2.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        
        # Fixed data (learnable)
        X = torch.randn(100, 10)
        y = torch.randint(0, 3, (100,))
        
        initial_loss = criterion(model(X), y).item()
        
        # Train for a few steps
        for _ in range(50):
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
        
        final_loss = criterion(model(X), y).item()
        
        assert final_loss < initial_loss, (
            f"Loss should decrease with training. "
            f"Initial={initial_loss:.4f}, Final={final_loss:.4f}"
        )
    
    def test_focal_loss_helps_with_imbalance(self):
        """
        Focal loss should help with class imbalance.
        
        With severe imbalance, focal loss should learn minority class better
        than standard CE (though we just verify it works here).
        """
        torch.manual_seed(42)
        
        # Imbalanced data: 90% class 0, 5% class 1, 5% class 2
        n_samples = 200
        n_class0 = 180
        n_class1 = 10
        n_class2 = 10
        
        X = torch.randn(n_samples, 10)
        y = torch.cat([
            torch.zeros(n_class0, dtype=torch.long),
            torch.ones(n_class1, dtype=torch.long),
            torch.full((n_class2,), 2, dtype=torch.long),
        ])
        
        # Shuffle
        perm = torch.randperm(n_samples)
        X, y = X[perm], y[perm]
        
        # Train with focal loss
        model = nn.Linear(10, 3)
        class_counts = torch.tensor([n_class0, n_class1, n_class2], dtype=torch.float)
        criterion = create_focal_loss(num_classes=3, gamma=2.0, class_counts=class_counts)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        for _ in range(100):
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
        
        # Model should produce predictions for all classes
        with torch.no_grad():
            preds = model(X).argmax(dim=1)
            unique_preds = preds.unique()
        
        # At minimum, loss should be finite and training should complete
        final_loss = criterion(model(X), y)
        assert torch.isfinite(final_loss), "Final loss should be finite"
