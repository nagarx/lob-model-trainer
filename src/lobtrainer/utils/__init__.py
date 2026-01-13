"""
Utility modules for LOB Model Trainer.

Provides:
- reproducibility: Deterministic seed management for reproducible experiments
  - set_seed: Set all random seeds (Python, NumPy, PyTorch)
  - get_seed_state/set_seed_state: Save/restore RNG state
  - SeedManager: Context manager for reproducible blocks
  - worker_init_fn: DataLoader worker seeding

Design principles (RULE.md §6):
- Same inputs MUST produce identical outputs across runs
- Explicit random seeds for any stochastic components
- Document and enforce floating-point comparison tolerances
"""

from lobtrainer.utils.reproducibility import (
    set_seed,
    get_seed_state,
    set_seed_state,
    SeedManager,
    worker_init_fn,
    create_worker_init_fn,
)

__all__ = [
    # Core seed management
    "set_seed",
    "get_seed_state",
    "set_seed_state",
    "SeedManager",
    # DataLoader worker seeding
    "worker_init_fn",
    "create_worker_init_fn",
]

