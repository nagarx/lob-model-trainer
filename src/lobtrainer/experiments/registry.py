"""
Experiment registry for tracking and comparing experiments.

The ExperimentRegistry provides:
- Centralized storage of experiment results
- Querying by tags, date, model type
- Comparison tables for metrics
- Export to various formats

Design principles (RULE.md):
- Single source of truth for experiments
- Append-only (experiments are not deleted, only marked)
- Efficient querying for large experiment sets
- Portable (JSON-based storage)

Usage:
    >>> registry = ExperimentRegistry("experiments/")
    >>> registry.register(result)
    >>> 
    >>> # Query experiments
    >>> tlob_experiments = registry.filter(model_type="tlob")
    >>> 
    >>> # Compare
    >>> table = create_comparison_table(registry, ['macro_f1', 'directional_accuracy'])
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from lobtrainer.experiments.result import ExperimentResult

logger = logging.getLogger(__name__)


class ExperimentRegistry:
    """
    Registry for tracking multiple experiments.
    
    Stores experiment results in a directory with metadata index for fast queries.
    
    Args:
        base_dir: Directory to store experiment results.
        create_if_missing: Create directory if it doesn't exist.
    
    Directory structure:
        base_dir/
        ├── index.json           # Metadata index for fast queries
        ├── {experiment_id}.json # Individual experiment results
        └── ...
    
    Example:
        >>> registry = ExperimentRegistry("experiments/")
        >>> 
        >>> # Register new experiment
        >>> registry.register(result)
        >>> 
        >>> # List all experiments
        >>> for exp in registry.list_all():
        ...     print(f"{exp.name}: F1={exp.test_metrics.macro_f1:.4f}")
        >>> 
        >>> # Filter by tag
        >>> tlob_exps = registry.filter(tags=['tlob'])
    """
    
    def __init__(
        self,
        base_dir: Union[str, Path],
        create_if_missing: bool = True,
    ):
        self.base_dir = Path(base_dir)
        
        if create_if_missing:
            self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self._index_path = self.base_dir / 'index.json'
        self._index: Dict[str, Dict[str, Any]] = {}
        
        # Load existing index
        if self._index_path.exists():
            self._load_index()
    
    def _load_index(self) -> None:
        """Load index from disk."""
        try:
            with open(self._index_path) as f:
                self._index = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load index: {e}. Starting fresh.")
            self._index = {}
    
    def _save_index(self) -> None:
        """Save index to disk."""
        with open(self._index_path, 'w') as f:
            json.dump(self._index, f, indent=2)
    
    def register(self, result: ExperimentResult) -> str:
        """
        Register an experiment result.
        
        Args:
            result: ExperimentResult to register.
        
        Returns:
            Experiment ID (for reference).
        """
        # Save full result to file
        result_path = self.base_dir / f"{result.experiment_id}.json"
        result.save(result_path)
        
        # Add to index (metadata only for fast queries)
        self._index[result.experiment_id] = {
            'name': result.name,
            'model_type': result.model_type,
            'labeling_strategy': result.labeling_strategy,
            'tags': result.tags,
            'created_at': result.created_at,
            'best_epoch': result.best_epoch,
            'total_epochs': result.total_epochs,
            # Key metrics for quick comparison
            'test_accuracy': result.test_metrics.accuracy if result.test_metrics else None,
            'test_macro_f1': result.test_metrics.macro_f1 if result.test_metrics else None,
            'test_directional_accuracy': result.test_metrics.directional_accuracy if result.test_metrics else None,
            'test_signal_rate': result.test_metrics.signal_rate if result.test_metrics else None,
            'test_predicted_trade_win_rate': result.test_metrics.predicted_trade_win_rate if result.test_metrics else None,
            'model_params': result.model_params,
            'training_time_seconds': result.training_time_seconds,
        }
        
        self._save_index()
        
        logger.info(f"Registered experiment: {result.experiment_id}")
        return result.experiment_id
    
    def get(self, experiment_id: str) -> Optional[ExperimentResult]:
        """
        Get full experiment result by ID.
        
        Args:
            experiment_id: Experiment ID.
        
        Returns:
            ExperimentResult or None if not found.
        """
        if experiment_id not in self._index:
            return None
        
        result_path = self.base_dir / f"{experiment_id}.json"
        if not result_path.exists():
            logger.warning(f"Experiment {experiment_id} in index but file missing")
            return None
        
        return ExperimentResult.load(result_path)
    
    def list_all(self) -> List[ExperimentResult]:
        """List all registered experiments (loads full results)."""
        results = []
        for exp_id in self._index:
            result = self.get(exp_id)
            if result:
                results.append(result)
        return results
    
    def list_ids(self) -> List[str]:
        """List all experiment IDs."""
        return list(self._index.keys())
    
    def filter(
        self,
        model_type: Optional[str] = None,
        labeling_strategy: Optional[str] = None,
        tags: Optional[List[str]] = None,
        min_accuracy: Optional[float] = None,
        min_f1: Optional[float] = None,
        created_after: Optional[str] = None,
        created_before: Optional[str] = None,
    ) -> List[ExperimentResult]:
        """
        Filter experiments by criteria.
        
        Args:
            model_type: Filter by model type (e.g., 'tlob', 'lstm').
            labeling_strategy: Filter by labeling strategy.
            tags: Filter by any matching tag.
            min_accuracy: Minimum test accuracy.
            min_f1: Minimum test macro F1.
            created_after: ISO timestamp lower bound.
            created_before: ISO timestamp upper bound.
        
        Returns:
            List of matching ExperimentResult objects.
        """
        matching = []
        
        for exp_id, meta in self._index.items():
            # Model type filter
            if model_type and meta.get('model_type') != model_type:
                continue
            
            # Labeling strategy filter
            if labeling_strategy and meta.get('labeling_strategy') != labeling_strategy:
                continue
            
            # Tags filter (any match)
            if tags:
                exp_tags = set(meta.get('tags', []))
                if not any(t in exp_tags for t in tags):
                    continue
            
            # Accuracy filter
            if min_accuracy is not None:
                acc = meta.get('test_accuracy')
                if acc is None or acc < min_accuracy:
                    continue
            
            # F1 filter
            if min_f1 is not None:
                f1 = meta.get('test_macro_f1')
                if f1 is None or f1 < min_f1:
                    continue
            
            # Date filters
            created_at = meta.get('created_at', '')
            if created_after and created_at < created_after:
                continue
            if created_before and created_at > created_before:
                continue
            
            # Load full result
            result = self.get(exp_id)
            if result:
                matching.append(result)
        
        return matching
    
    def count(self) -> int:
        """Return number of registered experiments."""
        return len(self._index)
    
    def summary(self) -> str:
        """Generate registry summary."""
        lines = [
            f"ExperimentRegistry: {self.base_dir}",
            f"Total experiments: {self.count()}",
        ]
        
        # Group by model type
        by_model = {}
        for meta in self._index.values():
            model = meta.get('model_type', 'unknown')
            by_model[model] = by_model.get(model, 0) + 1
        
        if by_model:
            lines.append("\nBy model type:")
            for model, count in sorted(by_model.items()):
                lines.append(f"  {model}: {count}")
        
        # Group by labeling strategy
        by_strategy = {}
        for meta in self._index.values():
            strategy = meta.get('labeling_strategy', 'unknown')
            by_strategy[strategy] = by_strategy.get(strategy, 0) + 1
        
        if by_strategy:
            lines.append("\nBy labeling strategy:")
            for strategy, count in sorted(by_strategy.items()):
                lines.append(f"  {strategy}: {count}")
        
        return "\n".join(lines)


# =============================================================================
# Comparison Utilities
# =============================================================================


def create_comparison_table(
    registry_or_results: Union[ExperimentRegistry, List[ExperimentResult]],
    metric_keys: Optional[List[str]] = None,
    sort_by: Optional[str] = None,
    ascending: bool = False,
) -> str:
    """
    Create a comparison table of experiments.
    
    Args:
        registry_or_results: ExperimentRegistry or list of ExperimentResult.
        metric_keys: Metrics to include (default: common ones).
        sort_by: Metric key to sort by.
        ascending: Sort order.
    
    Returns:
        Formatted table string.
    
    Example:
        >>> table = create_comparison_table(registry, ['macro_f1', 'directional_accuracy'])
        >>> print(table)
        
        | Name                | Model | Strategy | Macro F1 | Dir. Acc |
        |---------------------|-------|----------|----------|----------|
        | TLOB_NVDA_h10_v1    | tlob  | tlob     | 0.4523   | 0.5812   |
        | LSTM_NVDA_h10_v1    | lstm  | tlob     | 0.4102   | 0.5234   |
    """
    # Get results
    if isinstance(registry_or_results, ExperimentRegistry):
        results = registry_or_results.list_all()
    else:
        results = registry_or_results
    
    if not results:
        return "No experiments to compare."
    
    # Default metrics
    if metric_keys is None:
        metric_keys = ['macro_f1', 'directional_accuracy', 'signal_rate']
    
    # Build table data
    rows = []
    for result in results:
        row = {
            'name': result.name[:25],  # Truncate long names
            'model': result.model_type,
            'strategy': result.labeling_strategy,
            'epochs': result.total_epochs,
        }
        
        # Add metrics
        if result.test_metrics:
            for key in metric_keys:
                value = getattr(result.test_metrics, key, None)
                if value is None:
                    value = result.test_metrics.extra_metrics.get(key, 'N/A')
                row[key] = value
        else:
            for key in metric_keys:
                row[key] = 'N/A'
        
        rows.append(row)
    
    # Sort if requested
    if sort_by and sort_by in metric_keys:
        rows.sort(
            key=lambda r: r.get(sort_by, 0) if isinstance(r.get(sort_by), (int, float)) else 0,
            reverse=not ascending,
        )
    
    # Format table
    headers = ['Name', 'Model', 'Strategy', 'Epochs'] + [k.replace('_', ' ').title() for k in metric_keys]
    col_widths = [max(len(h), 10) for h in headers]
    
    # Adjust widths based on content
    for row in rows:
        for i, key in enumerate(['name', 'model', 'strategy', 'epochs'] + metric_keys):
            value = row.get(key, '')
            if isinstance(value, float):
                value = f"{value:.4f}"
            col_widths[i] = max(col_widths[i], len(str(value)))
    
    # Build output
    lines = []
    
    # Header
    header_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    separator = "|-" + "-|-".join("-" * w for w in col_widths) + "-|"
    
    lines.append(header_line)
    lines.append(separator)
    
    # Data rows
    for row in rows:
        values = []
        for key in ['name', 'model', 'strategy', 'epochs'] + metric_keys:
            value = row.get(key, '')
            if isinstance(value, float):
                value = f"{value:.4f}"
            values.append(str(value))
        
        row_line = "| " + " | ".join(v.ljust(col_widths[i]) for i, v in enumerate(values)) + " |"
        lines.append(row_line)
    
    return "\n".join(lines)


def filter_experiments(
    results: List[ExperimentResult],
    predicate: Callable[[ExperimentResult], bool],
) -> List[ExperimentResult]:
    """
    Filter experiments by custom predicate.
    
    Args:
        results: List of ExperimentResult.
        predicate: Function that returns True for experiments to keep.
    
    Returns:
        Filtered list.
    
    Example:
        >>> # Keep only experiments with F1 > 0.4
        >>> filtered = filter_experiments(results, lambda r: r.test_metrics.macro_f1 > 0.4)
    """
    return [r for r in results if predicate(r)]


def rank_experiments(
    results: List[ExperimentResult],
    metric: str = 'macro_f1',
    top_k: Optional[int] = None,
) -> List[ExperimentResult]:
    """
    Rank experiments by a metric.
    
    Args:
        results: List of ExperimentResult.
        metric: Metric to rank by.
        top_k: Return only top K (None = all).
    
    Returns:
        Sorted list (best first).
    """
    def get_metric(r: ExperimentResult) -> float:
        if r.test_metrics is None:
            return 0.0
        value = getattr(r.test_metrics, metric, None)
        if value is None:
            value = r.test_metrics.extra_metrics.get(metric, 0.0)
        return value if isinstance(value, (int, float)) else 0.0
    
    ranked = sorted(results, key=get_metric, reverse=True)
    
    if top_k is not None:
        ranked = ranked[:top_k]
    
    return ranked
