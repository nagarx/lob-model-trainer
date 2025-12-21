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
- streaming: Memory-efficient streaming analysis (for large datasets)

Memory Efficiency:
- For large datasets (>100 days), use the streaming module
- Streaming functions use O(1) memory regardless of dataset size
- Example: compute_streaming_overview() instead of generate_dataset_summary()
"""

from .data_loading import (
    load_split,
    load_split_aligned,  # Use this for multi-day signal-label analysis
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

# Intraday seasonality analysis (regime-stratified correlations)
from .intraday_seasonality import (
    compute_regime_stats as compute_regime_stats_intraday,
    compute_signal_regime_correlation,
    compute_all_regime_correlations,
    compute_signal_seasonality,
    compute_regime_importance,
    run_intraday_seasonality_analysis,
    to_dict as intraday_to_dict,
    RegimeStats as IntradayRegimeStats,
    SignalRegimeCorrelation,
    SignalSeasonality,
    IntradaySeasonalitySummary,
    REGIME_NAMES as INTRADAY_REGIME_NAMES,
    CORE_SIGNAL_INDICES as INTRADAY_SIGNAL_INDICES,
    EXPECTED_SIGNS,
)

# Memory-efficient streaming analysis (for large datasets)
from .streaming import (
    iter_days,
    iter_days_aligned,  # Use this for signal-label correlation (correct alignment)
    count_days,
    get_dates,
    align_features_for_day,
    compute_streaming_overview,
    compute_streaming_label_analysis,
    compute_streaming_signal_stats,
    estimate_memory_usage,
    get_memory_efficient_config,
    DayData,
    AlignedDayData,
    RunningStats,
    StreamingColumnStats,
    StreamingLabelCounter,
    StreamingDataQuality,
)

__all__ = [
    # Data loading
    'load_split',
    'load_split_aligned',  # PREFERRED for multi-day signal-label analysis
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
    # Streaming (memory-efficient for large datasets)
    'iter_days',
    'iter_days_aligned',  # PREFERRED for signal-label analysis
    'count_days',
    'get_dates',
    'align_features_for_day',
    'compute_streaming_overview',
    'compute_streaming_label_analysis',
    'compute_streaming_signal_stats',
    'estimate_memory_usage',
    'get_memory_efficient_config',
    'DayData',
    'AlignedDayData',
    'RunningStats',
    'StreamingColumnStats',
    'StreamingLabelCounter',
    'StreamingDataQuality',
    # Intraday seasonality (regime-stratified analysis)
    'compute_regime_stats_intraday',
    'compute_signal_regime_correlation',
    'compute_all_regime_correlations',
    'compute_signal_seasonality',
    'compute_regime_importance',
    'run_intraday_seasonality_analysis',
    'intraday_to_dict',
    'IntradayRegimeStats',
    'SignalRegimeCorrelation',
    'SignalSeasonality',
    'IntradaySeasonalitySummary',
    'INTRADAY_REGIME_NAMES',
    'INTRADAY_SIGNAL_INDICES',
    'EXPECTED_SIGNS',
]

