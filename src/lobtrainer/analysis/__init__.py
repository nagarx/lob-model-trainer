"""
Phase 2A: Exploratory Data Analysis

Reusable analysis functions for understanding signals, labels, and their relationships.

Modules:
- data_loading: Load and align features with labels
- data_overview: Data validation, quality checks, dataset summaries
- label_analysis: Label distribution, autocorrelation, transitions
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
    CORE_SIGNAL_INDICES,
    get_signal_info,
)

from .data_overview import (
    validate_file_structure,
    compute_shape_validation,
    compute_data_quality,
    compute_label_distribution as compute_label_dist_overview,
    compute_all_categorical_validations,
    compute_signal_statistics,
    generate_dataset_summary,
    print_data_overview,
    DatasetSummary,
    DataQuality,
    FileInventory,
    ShapeValidation,
    CategoricalValidation,
    SignalStatistics,
)

from .label_analysis import (
    compute_label_distribution,
    compute_autocorrelation,
    compute_transition_matrix,
    compute_regime_stats,
    compute_signal_label_correlations,
    run_label_analysis,
    print_label_analysis,
    LabelAnalysisSummary,
    LabelDistribution,
    AutocorrelationResult,
    TransitionMatrix,
    RegimeStats,
    SignalCorrelation,
    REGIME_NAMES,
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
    'CORE_SIGNAL_INDICES',
    'get_signal_info',
    # Data overview
    'validate_file_structure',
    'compute_shape_validation',
    'compute_data_quality',
    'compute_label_dist_overview',
    'compute_all_categorical_validations',
    'compute_signal_statistics',
    'generate_dataset_summary',
    'print_data_overview',
    'DatasetSummary',
    'DataQuality',
    'FileInventory',
    'ShapeValidation',
    'CategoricalValidation',
    'SignalStatistics',
    # Label analysis
    'compute_label_distribution',
    'compute_autocorrelation',
    'compute_transition_matrix',
    'compute_regime_stats',
    'compute_signal_label_correlations',
    'run_label_analysis',
    'print_label_analysis',
    'LabelAnalysisSummary',
    'LabelDistribution',
    'AutocorrelationResult',
    'TransitionMatrix',
    'RegimeStats',
    'SignalCorrelation',
    'REGIME_NAMES',
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

