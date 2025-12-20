"""
Phase 2A: Exploratory Data Analysis

Reusable analysis functions for understanding signals, labels, and their relationships.

Modules:
- data_loading: Load and align features with labels
- data_overview: Data validation, quality checks, dataset summaries
- label_analysis: Label distribution, autocorrelation, transitions
- signal_stats: Distribution statistics (skewness, kurtosis, normality, stationarity)
- signal_correlations: Correlation matrix, PCA, VIF, redundancy
- predictive_power: Signal-label relationships (AUC, MI, correlations)
- temporal_dynamics: Signal autocorrelation, lead-lag, predictive decay
- generalization: Day-to-day variance, walk-forward validation
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
    compute_stationarity_test,
    compute_all_stationarity_tests,
    compute_rolling_stats,
    compute_all_rolling_stats,
    print_stationarity_summary,
    StationarityResult,
    RollingStatsResult,
)

from .signal_correlations import (
    compute_signal_correlation_matrix,
    find_redundant_pairs,
    print_correlation_summary,
    compute_pca_analysis,
    compute_vif,
    cluster_signals,
    print_advanced_correlation_summary,
    PCAResult,
    VIFResult,
    SignalCluster,
)

from .predictive_power import (
    compute_signal_metrics,
    compute_all_signal_metrics,
    compute_binned_probabilities,
    print_predictive_summary,
)

from .temporal_dynamics import (
    compute_signal_autocorrelations,
    compute_lead_lag_relations,
    compute_all_predictive_decays,
    compute_all_level_vs_change,
    run_temporal_dynamics_analysis,
    print_temporal_dynamics,
    SignalAutocorrelation,
    LeadLagRelation,
    PredictiveDecay,
    LevelVsChangeAnalysis,
    TemporalDynamicsSummary,
)

from .generalization import (
    load_day_data,
    compute_day_statistics,
    compute_signal_day_stats,
    walk_forward_validation,
    run_generalization_analysis,
    print_generalization_analysis,
    DayStatistics,
    SignalDayStats,
    WalkForwardResult,
    GeneralizationSummary,
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
    # Signal stats (including stationarity)
    'compute_distribution_stats',
    'print_distribution_summary',
    'compute_stationarity_test',
    'compute_all_stationarity_tests',
    'compute_rolling_stats',
    'compute_all_rolling_stats',
    'print_stationarity_summary',
    'StationarityResult',
    'RollingStatsResult',
    # Correlations (including PCA, VIF)
    'compute_signal_correlation_matrix',
    'find_redundant_pairs',
    'print_correlation_summary',
    'compute_pca_analysis',
    'compute_vif',
    'cluster_signals',
    'print_advanced_correlation_summary',
    'PCAResult',
    'VIFResult',
    'SignalCluster',
    # Predictive power
    'compute_signal_metrics',
    'compute_all_signal_metrics',
    'compute_binned_probabilities',
    'print_predictive_summary',
    # Temporal dynamics
    'compute_signal_autocorrelations',
    'compute_lead_lag_relations',
    'compute_all_predictive_decays',
    'compute_all_level_vs_change',
    'run_temporal_dynamics_analysis',
    'print_temporal_dynamics',
    'SignalAutocorrelation',
    'LeadLagRelation',
    'PredictiveDecay',
    'LevelVsChangeAnalysis',
    'TemporalDynamicsSummary',
    # Generalization
    'load_day_data',
    'compute_day_statistics',
    'compute_signal_day_stats',
    'walk_forward_validation',
    'run_generalization_analysis',
    'print_generalization_analysis',
    'DayStatistics',
    'SignalDayStats',
    'WalkForwardResult',
    'GeneralizationSummary',
]

