"""
Reproducibility utilities for deterministic experiments.

Ensures that experiments with the same seed produce identical results across:
- Python random module
- NumPy random number generators
- PyTorch (CPU and CUDA)
- CuDNN (if available)

Design principles (RULE.md §6):
- Same inputs MUST produce identical outputs across runs
- Explicit random seeds for any stochastic components
- Document and enforce floating-point comparison tolerances

Reference:
    PyTorch Reproducibility: https://pytorch.org/docs/stable/notes/randomness.html

Usage:
    >>> from lobtrainer.utils import set_seed
    >>> set_seed(42)  # Call at start of training
"""

import random
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


# =============================================================================
# Core Seed Management
# =============================================================================


def set_seed(seed: int = 42, deterministic_cudnn: bool = True) -> None:
    """
    Set random seeds for all random number generators.
    
    This ensures reproducibility across:
    - Python's random module
    - NumPy's random module
    - PyTorch (CPU operations)
    - PyTorch CUDA (if available)
    - CuDNN (if available and deterministic_cudnn=True)
    
    Args:
        seed: Random seed (default: 42)
        deterministic_cudnn: If True, configure CuDNN for deterministic behavior.
                            This may impact performance but ensures reproducibility.
                            (RULE.md §6: Determinism is prioritized over speed)
    
    Example:
        >>> set_seed(42)
        >>> torch.rand(3)  # Always produces same values
    
    Notes:
        - Call this at the start of your training script BEFORE any model/data creation
        - Some operations may still be non-deterministic even with these settings
          (e.g., scatter/gather operations on GPU)
        - For full determinism, set CUBLAS_WORKSPACE_CONFIG environment variable
    
    Reference:
        https://pytorch.org/docs/stable/notes/randomness.html
    """
    # Validate seed
    if not isinstance(seed, int):
        raise TypeError(f"seed must be an int, got {type(seed)}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch CPU
    torch.manual_seed(seed)
    
    # PyTorch CUDA (all GPUs)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # CuDNN determinism
    if deterministic_cudnn and torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set CUBLAS_WORKSPACE_CONFIG for PyTorch >= 1.8
        # This is needed for some CUDA operations to be deterministic
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    
    logger.debug(
        f"Set seed={seed}, deterministic_cudnn={deterministic_cudnn}, "
        f"cuda_available={torch.cuda.is_available()}"
    )


def get_seed_state() -> Dict[str, Any]:
    """
    Get current state of all random number generators.
    
    Useful for:
    - Debugging reproducibility issues
    - Saving/restoring RNG state for checkpointing
    
    Returns:
        Dict with states for: python, numpy, torch, torch_cuda (if available)
    
    Example:
        >>> state = get_seed_state()
        >>> # ... do some operations ...
        >>> # state can be used to restore RNG state
    """
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    
    if torch.cuda.is_available():
        state['torch_cuda'] = torch.cuda.get_rng_state_all()
    
    return state


def set_seed_state(state: Dict[str, Any]) -> None:
    """
    Restore random number generators to a saved state.
    
    Args:
        state: State dict from get_seed_state()
    
    Example:
        >>> state = get_seed_state()
        >>> # ... do some operations ...
        >>> set_seed_state(state)  # Restore to saved state
    """
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    
    if 'torch_cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['torch_cuda'])


# =============================================================================
# Seed Manager Context
# =============================================================================


@dataclass
class SeedManager:
    """
    Context manager for reproducible code blocks.
    
    Provides a clean way to ensure a code block uses a specific seed
    and optionally restores the original state afterward.
    
    Args:
        seed: Random seed to use within the context
        restore_state: If True, restore original RNG state after exiting
                      (useful for testing, not recommended for training)
    
    Example:
        >>> # Reproducible block that doesn't affect global state
        >>> with SeedManager(42, restore_state=True):
        ...     x = torch.rand(3)  # Always same values
        >>> # After exiting, RNG state is restored
        
        >>> # For training (no state restoration)
        >>> with SeedManager(config.train.seed):
        ...     train_model()
    
    Design note (RULE.md §6):
        - restore_state=True is useful for testing to avoid cross-test contamination
        - restore_state=False (default) is correct for training since we want
          the seed to affect all subsequent operations
    """
    
    seed: int
    restore_state: bool = False
    deterministic_cudnn: bool = True
    
    def __post_init__(self):
        self._saved_state: Optional[Dict[str, Any]] = None
    
    def __enter__(self) -> 'SeedManager':
        if self.restore_state:
            self._saved_state = get_seed_state()
        
        set_seed(self.seed, self.deterministic_cudnn)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.restore_state and self._saved_state is not None:
            set_seed_state(self._saved_state)
        return None  # Don't suppress exceptions


# =============================================================================
# Worker Initialization for DataLoader
# =============================================================================


def worker_init_fn(worker_id: int, base_seed: Optional[int] = None) -> None:
    """
    Initialize random seeds for DataLoader workers.
    
    Each worker should have a different seed to avoid duplicate data augmentations,
    but the seeds should be deterministic for reproducibility.
    
    Args:
        worker_id: Worker ID (0, 1, ..., num_workers-1)
        base_seed: Base seed. If None, uses a fixed default (42).
    
    Usage:
        >>> from functools import partial
        >>> loader = DataLoader(
        ...     dataset,
        ...     num_workers=4,
        ...     worker_init_fn=partial(worker_init_fn, base_seed=42),
        ... )
    
    Reference:
        https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    if base_seed is None:
        base_seed = 42
    
    # Each worker gets a unique but deterministic seed
    worker_seed = base_seed + worker_id
    
    # Set seeds for this worker
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def create_worker_init_fn(base_seed: int):
    """
    Create a worker_init_fn with a fixed base seed.
    
    This is a convenience function to create a worker_init_fn
    with the seed baked in (avoids functools.partial).
    
    Args:
        base_seed: Base seed for worker initialization
    
    Returns:
        A worker_init_fn suitable for DataLoader
    
    Example:
        >>> loader = DataLoader(
        ...     dataset,
        ...     num_workers=4,
        ...     worker_init_fn=create_worker_init_fn(42),
        ... )
    """
    def init_fn(worker_id: int) -> None:
        worker_init_fn(worker_id, base_seed)
    return init_fn

