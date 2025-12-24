"""
Utility modules for LOB Model Trainer.

Provides:
- reproducibility: Deterministic seed management for reproducible experiments
- logging: Structured logging configuration

Design principles (RULE.md §6):
- Same inputs MUST produce identical outputs across runs
- Explicit random seeds for any stochastic components
- Document and enforce floating-point comparison tolerances
"""

from lobtrainer.utils.reproducibility import (
    set_seed,
    get_seed_state,
    SeedManager,
)

__all__ = [
    "set_seed",
    "get_seed_state",
    "SeedManager",
]

