"""
Phase 2A: Exploratory Data Analysis

Reusable analysis functions for understanding signals, labels, and their relationships.

Modules:
- data_loading: Load and align features with labels
- signal_stats: Distribution statistics (skewness, kurtosis, normality)
- signal_correlations: Correlation matrix, PCA, redundancy
- predictive_power: Signal-label relationships (AUC, MI, correlations)
"""

from .data_loading import (
    load_split,
    load_all_splits,
    align_features_with_labels,
    WINDOW_SIZE,
    STRIDE,
)

from .signal_stats import (
    compute_distribution_stats,
    print_distribution_summary,
)

from .signal_correlations import (
    compute_signal_correlation_matrix,
    find_redundant_pairs,
    print_correlation_summary,
)

from .predictive_power import (
    compute_signal_metrics,
    compute_all_signal_metrics,
    compute_binned_probabilities,
    print_predictive_summary,
)

__all__ = [
    # Data loading
    'load_split',
    'load_all_splits', 
    'align_features_with_labels',
    'WINDOW_SIZE',
    'STRIDE',
    # Signal stats
    'compute_distribution_stats',
    'print_distribution_summary',
    # Correlations
    'compute_signal_correlation_matrix',
    'find_redundant_pairs',
    'print_correlation_summary',
    # Predictive power
    'compute_signal_metrics',
    'compute_all_signal_metrics',
    'compute_binned_probabilities',
    'print_predictive_summary',
]

