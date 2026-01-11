"""
Custom loss functions for LOB models.

Includes:
- FocalLoss: Handles class imbalance by down-weighting easy examples
- BinaryFocalLoss: Focal loss for binary classification

Design principles (RULE.md):
- Modular: Each loss is self-contained
- Configurable: All hyperparameters exposed
- Numerically stable: Proper clamping and log handling
- Tested: Each loss has unit tests

References:
- Lin et al. "Focal Loss for Dense Object Detection" (2017)
  https://arxiv.org/abs/1708.02002
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification with class imbalance.
    
    Focal Loss down-weights easy examples (high-confidence predictions) and
    focuses training on hard examples. This is especially useful when one class
    (e.g., Stable/NoOpportunity) dominates the dataset.
    
    Formula:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Where:
        - p_t: Probability of correct class
        - gamma: Focusing parameter (higher = more focus on hard examples)
        - alpha_t: Class weight for balancing (optional)
    
    Args:
        gamma: Focusing parameter. 
               gamma=0 is equivalent to CrossEntropyLoss.
               gamma=2 is commonly used (TLOB paper suggestion).
        alpha: Optional class weights, tensor of shape (num_classes,).
               If None, no class weighting is applied.
        reduction: 'mean', 'sum', or 'none'
    
    Example:
        >>> criterion = FocalLoss(gamma=2.0)
        >>> loss = criterion(logits, targets)  # logits: [B, C], targets: [B]
    
    Research reference:
        Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
        Focal loss for dense object detection. ICCV.
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction}")
        
        self.gamma = gamma
        self.reduction = reduction
        
        # Register alpha as buffer (not a parameter, but saved with state_dict)
        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Logits of shape (batch_size, num_classes)
            targets: Class indices of shape (batch_size,)
        
        Returns:
            Loss value (scalar if reduction != 'none', else (batch_size,))
        """
        # Compute cross-entropy without reduction
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        
        # Get probabilities for target class
        # p_t = softmax(inputs)[i, targets[i]]
        pt = torch.exp(-ce_loss)  # Equivalent to p_t
        
        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply class weights if provided
        if self.alpha is not None:
            # alpha_t = alpha[targets]
            alpha_t = self.alpha.to(inputs.device).gather(0, targets)
            focal_weight = alpha_t * focal_weight
        
        # Compute focal loss
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
    
    def __repr__(self) -> str:
        return f"FocalLoss(gamma={self.gamma}, alpha={self.alpha}, reduction='{self.reduction}')"


class BinaryFocalLoss(nn.Module):
    """
    Focal Loss for binary classification.
    
    Specialized version of Focal Loss for 2-class problems (Signal vs NoSignal).
    More numerically efficient than using FocalLoss with 2 classes.
    
    Formula:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Where p_t = p if y=1, else (1-p).
    
    Args:
        gamma: Focusing parameter (default: 2.0)
        alpha: Weight for positive class (Signal). 
               alpha > 0.5 gives more weight to positive class.
               If None, no weighting is applied.
        pos_weight: Alternative to alpha - weight for positive examples.
                    Used when positive class is rare.
        reduction: 'mean', 'sum', or 'none'
    
    Example:
        >>> # For detecting rare signals (29% positive, 71% negative)
        >>> criterion = BinaryFocalLoss(gamma=2.0, alpha=0.75)
        >>> loss = criterion(logits, targets)  # logits: [B, 1] or [B], targets: [B]
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[float] = None,
        pos_weight: Optional[float] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        if alpha is not None and (alpha < 0 or alpha > 1):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        
        self.gamma = gamma
        self.alpha = alpha
        self.pos_weight = pos_weight
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute binary focal loss.
        
        Args:
            inputs: Logits of shape (batch_size,) or (batch_size, 1)
            targets: Binary labels of shape (batch_size,), values in {0, 1}
        
        Returns:
            Loss value
        """
        # Flatten if needed
        if inputs.dim() == 2 and inputs.size(1) == 1:
            inputs = inputs.squeeze(1)
        
        # Convert to probabilities
        p = torch.sigmoid(inputs)
        
        # p_t = p if y=1, else 1-p
        targets_float = targets.float()
        pt = p * targets_float + (1 - p) * (1 - targets_float)
        
        # Compute focal weight
        focal_weight = (1 - pt) ** self.gamma
        
        # Compute binary cross-entropy
        # BCE = -[y*log(p) + (1-y)*log(1-p)]
        eps = 1e-7
        bce = -targets_float * torch.log(p + eps) - (1 - targets_float) * torch.log(1 - p + eps)
        
        # Apply alpha weighting if provided
        if self.alpha is not None:
            alpha_t = self.alpha * targets_float + (1 - self.alpha) * (1 - targets_float)
            focal_weight = alpha_t * focal_weight
        
        # Apply pos_weight if provided (alternative to alpha)
        if self.pos_weight is not None:
            pos_weight_t = self.pos_weight * targets_float + (1 - targets_float)
            focal_weight = pos_weight_t * focal_weight
        
        # Compute focal loss
        focal_loss = focal_weight * bce
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
    
    def __repr__(self) -> str:
        return (
            f"BinaryFocalLoss(gamma={self.gamma}, alpha={self.alpha}, "
            f"pos_weight={self.pos_weight}, reduction='{self.reduction}')"
        )


def create_focal_loss(
    num_classes: int,
    gamma: float = 2.0,
    alpha: Optional[torch.Tensor] = None,
    class_counts: Optional[torch.Tensor] = None,
) -> nn.Module:
    """
    Factory function to create appropriate focal loss.
    
    Args:
        num_classes: Number of classes (2 for binary, 3 for tri-class)
        gamma: Focusing parameter
        alpha: Class weights (if None and class_counts provided, compute from counts)
        class_counts: Counts per class (used to compute alpha if alpha is None)
    
    Returns:
        Appropriate loss module (BinaryFocalLoss or FocalLoss)
    
    Example:
        >>> # For binary signal detection with known class imbalance
        >>> class_counts = torch.tensor([71000, 29000])  # 71% NoSignal, 29% Signal
        >>> loss = create_focal_loss(num_classes=2, gamma=2.0, class_counts=class_counts)
    """
    if class_counts is not None and alpha is None:
        # Compute inverse frequency weights
        total = class_counts.sum()
        alpha = total / (len(class_counts) * class_counts.clamp(min=1))
        alpha = alpha / alpha.sum() * len(class_counts)  # Normalize to sum to num_classes
    
    if num_classes == 2:
        # Binary focal loss
        if alpha is not None:
            # alpha for BinaryFocalLoss is weight for positive class
            alpha_pos = alpha[1] / (alpha[0] + alpha[1])
            return BinaryFocalLoss(gamma=gamma, alpha=float(alpha_pos))
        return BinaryFocalLoss(gamma=gamma)
    else:
        return FocalLoss(gamma=gamma, alpha=alpha)


# =============================================================================
# Unit Tests (run with pytest)
# =============================================================================

if __name__ == "__main__":
    import numpy as np
    
    # Test FocalLoss
    print("Testing FocalLoss...")
    criterion = FocalLoss(gamma=2.0)
    logits = torch.randn(32, 3)
    targets = torch.randint(0, 3, (32,))
    loss = criterion(logits, targets)
    print(f"  FocalLoss output: {loss.item():.4f}")
    assert loss.dim() == 0, "Loss should be scalar"
    assert loss > 0, "Loss should be positive"
    
    # Test with gamma=0 (should be close to CE)
    ce_loss_fn = nn.CrossEntropyLoss()
    focal_gamma0 = FocalLoss(gamma=0.0)
    
    logits = torch.randn(100, 3)
    targets = torch.randint(0, 3, (100,))
    
    ce = ce_loss_fn(logits, targets)
    fl = focal_gamma0(logits, targets)
    print(f"  CE Loss: {ce.item():.4f}")
    print(f"  Focal(gamma=0): {fl.item():.4f}")
    assert abs(ce.item() - fl.item()) < 0.01, "gamma=0 should be ~CE"
    
    # Test BinaryFocalLoss
    print("\nTesting BinaryFocalLoss...")
    criterion = BinaryFocalLoss(gamma=2.0, alpha=0.75)
    logits = torch.randn(32)
    targets = torch.randint(0, 2, (32,))
    loss = criterion(logits, targets)
    print(f"  BinaryFocalLoss output: {loss.item():.4f}")
    assert loss.dim() == 0
    assert loss > 0
    
    # Test factory function
    print("\nTesting create_focal_loss...")
    class_counts = torch.tensor([71000.0, 29000.0])
    loss_fn = create_focal_loss(num_classes=2, gamma=2.0, class_counts=class_counts)
    print(f"  Created: {loss_fn}")
    
    print("\nAll tests passed!")
