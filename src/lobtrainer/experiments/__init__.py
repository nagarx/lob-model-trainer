"""
Experiment tracking and comparison framework.

Provides structured management of ML experiments:
- ExperimentResult: Standardized container for experiment outcomes
- ExperimentRegistry: Central tracker for all experiments
- Comparison utilities: Side-by-side metric comparison

Design principles (RULE.md):
- Results are immutable once recorded
- All configs and metrics are persisted
- Reproducibility via seed and config storage
- Easy comparison across experiments

Usage:
    >>> from lobtrainer.experiments import (
    ...     ExperimentResult,
    ...     ExperimentRegistry,
    ...     create_comparison_table,
    ... )
    >>> 
    >>> # After training
    >>> result = ExperimentResult.from_trainer(trainer)
    >>> registry = ExperimentRegistry("experiments/")
    >>> registry.register(result)
    >>> 
    >>> # Compare experiments
    >>> table = create_comparison_table(registry, metric_keys=['macro_f1', 'directional_accuracy'])
"""

from lobtrainer.experiments.result import (
    ExperimentResult,
    ExperimentMetrics,
)

from lobtrainer.experiments.registry import (
    ExperimentRegistry,
    create_comparison_table,
    filter_experiments,
)

__all__ = [
    # Core
    "ExperimentResult",
    "ExperimentMetrics",
    # Registry
    "ExperimentRegistry",
    # Utilities
    "create_comparison_table",
    "filter_experiments",
]
