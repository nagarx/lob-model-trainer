"""
Feature Selection for LOB Model Training.

Provides immutable, validated feature selection that works with the data pipeline.
Feature selection is applied AFTER normalization to ensure normalization statistics
are computed correctly on original feature indices.

Design Principles (RULE.md):
- Single Source of Truth: Uses presets from constants.feature_presets
- Immutable: FeatureSelector is frozen dataclass - cannot be modified after creation
- Validated: All indices validated at construction time
- Explicit Contract: Clear input/output shape transformation

Data Flow:
    Raw Data [T, F_src]
        → Normalize (using stats computed on all F_src features)
        → Select [T, F_sel] (FeatureSelector.select())
        → Model Input

Usage:
    >>> from lobtrainer.data import FeatureSelector
    >>> 
    >>> # From preset
    >>> selector = FeatureSelector.from_preset("short_term_40", source_feature_count=116)
    >>> 
    >>> # From custom indices
    >>> selector = FeatureSelector.from_indices([0, 1, 2, 84, 85], source_feature_count=116)
    >>> 
    >>> # Apply selection
    >>> selected = selector.select(normalized_data)  # [T, 116] → [T, 5]
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union
import logging

import numpy as np

from lobtrainer.constants import (
    get_feature_preset,
    validate_feature_indices,
    FEATURE_COUNT,
    EXTENDED_FEATURE_COUNT,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureSelector:
    """
    Immutable feature selector for LOB data.
    
    Selects a subset of features from input data arrays. The selector is
    created with a specific source feature count and validated indices,
    ensuring consistent behavior throughout training.
    
    Contract:
    - Input: ndarray with shape [..., source_feature_count]
    - Output: ndarray with shape [..., len(indices)]
    - Order: output[..., i] = input[..., indices[i]]
    
    Attributes:
        indices: Tuple of feature indices to select (sorted internally)
        name: Human-readable name for logging and debugging
        source_feature_count: Expected number of features in source data
    
    Example:
        >>> selector = FeatureSelector.from_preset("short_term_40", 116)
        >>> selector.output_size
        40
        >>> 
        >>> data = np.random.randn(100, 116)  # [T, F]
        >>> selected = selector.select(data)
        >>> selected.shape
        (100, 40)
    """
    
    indices: Tuple[int, ...]
    name: str
    source_feature_count: int
    
    def __post_init__(self) -> None:
        """Validate indices at construction time."""
        validate_feature_indices(
            self.indices,
            self.source_feature_count,
            name=self.name,
        )
    
    @classmethod
    def from_preset(
        cls,
        preset: str,
        source_feature_count: int,
    ) -> "FeatureSelector":
        """
        Create selector from a named preset.
        
        Args:
            preset: Preset name (e.g., "short_term_40", "full_116", "lob_only")
            source_feature_count: Number of features in source data
        
        Returns:
            FeatureSelector configured for the preset
        
        Raises:
            ValueError: If preset is not recognized or indices exceed source_feature_count
        
        Example:
            >>> selector = FeatureSelector.from_preset("short_term_40", 116)
            >>> len(selector.indices)
            40
        """
        indices = get_feature_preset(preset)
        return cls(
            indices=indices,
            name=preset,
            source_feature_count=source_feature_count,
        )
    
    @classmethod
    def from_indices(
        cls,
        indices: Sequence[int],
        source_feature_count: int,
        name: str = "custom",
    ) -> "FeatureSelector":
        """
        Create selector from explicit index list.
        
        Args:
            indices: Sequence of feature indices to select
            source_feature_count: Number of features in source data
            name: Name for logging/debugging
        
        Returns:
            FeatureSelector configured with given indices
        
        Raises:
            ValueError: If any index is invalid
        
        Example:
            >>> selector = FeatureSelector.from_indices([84, 85, 86], 98, "ofi_only")
            >>> selector.output_size
            3
        """
        return cls(
            indices=tuple(indices),
            name=name,
            source_feature_count=source_feature_count,
        )
    
    @classmethod
    def all_features(
        cls,
        source_feature_count: int,
    ) -> "FeatureSelector":
        """
        Create selector that keeps all features (no-op selection).
        
        Args:
            source_feature_count: Number of features in source data
        
        Returns:
            FeatureSelector that returns input unchanged
        
        Example:
            >>> selector = FeatureSelector.all_features(116)
            >>> selector.output_size
            116
        """
        return cls(
            indices=tuple(range(source_feature_count)),
            name=f"full_{source_feature_count}",
            source_feature_count=source_feature_count,
        )
    
    @property
    def output_size(self) -> int:
        """Number of features after selection."""
        return len(self.indices)
    
    @property
    def is_identity(self) -> bool:
        """True if selector keeps all features in original order."""
        return (
            len(self.indices) == self.source_feature_count
            and self.indices == tuple(range(self.source_feature_count))
        )
    
    def select(self, data: np.ndarray) -> np.ndarray:
        """
        Apply feature selection to data array.
        
        The selection is applied to the LAST dimension of the array,
        supporting both 2D [T, F] and 3D [N, T, F] inputs.
        
        Args:
            data: Input array with shape [..., source_feature_count]
        
        Returns:
            Selected array with shape [..., output_size]
        
        Raises:
            ValueError: If data's last dimension doesn't match source_feature_count
        
        Example:
            >>> selector = FeatureSelector.from_indices([0, 10, 20], 40, "l1_prices")
            >>> data = np.random.randn(100, 40)
            >>> selected = selector.select(data)
            >>> selected.shape
            (100, 3)
        """
        if data.shape[-1] != self.source_feature_count:
            raise ValueError(
                f"Data has {data.shape[-1]} features but selector '{self.name}' "
                f"expects {self.source_feature_count}"
            )
        
        # Fast path: identity selector returns input unchanged
        if self.is_identity:
            return data
        
        # Apply selection on last dimension
        # Using tuple indexing is faster than list for numpy
        return data[..., self.indices]
    
    def get_index_mapping(self) -> dict:
        """
        Get mapping from output index to original index.
        
        Returns:
            Dict mapping output_idx → original_idx
        
        Example:
            >>> selector = FeatureSelector.from_indices([84, 85, 86], 98, "ofi")
            >>> selector.get_index_mapping()
            {0: 84, 1: 85, 2: 86}
        """
        return {i: orig for i, orig in enumerate(self.indices)}
    
    def __repr__(self) -> str:
        return (
            f"FeatureSelector(name='{self.name}', "
            f"output_size={self.output_size}, "
            f"source_feature_count={self.source_feature_count})"
        )


def create_feature_selector(
    preset: Optional[str] = None,
    indices: Optional[Sequence[int]] = None,
    source_feature_count: int = EXTENDED_FEATURE_COUNT,
) -> Optional[FeatureSelector]:
    """
    Factory function to create FeatureSelector from config options.
    
    Exactly one of `preset` or `indices` should be provided.
    If neither is provided, returns None (meaning "use all features").
    
    Args:
        preset: Named preset (e.g., "short_term_40")
        indices: Custom list of feature indices
        source_feature_count: Number of features in source data
    
    Returns:
        FeatureSelector or None if no selection specified
    
    Raises:
        ValueError: If both preset and indices are provided
    
    Example:
        >>> selector = create_feature_selector(preset="short_term_40", source_feature_count=116)
        >>> selector.output_size
        40
        
        >>> selector = create_feature_selector()  # No selection
        >>> selector is None
        True
    """
    if preset and indices:
        raise ValueError(
            "Specify either 'preset' or 'indices' for feature selection, not both"
        )
    
    if preset:
        selector = FeatureSelector.from_preset(preset, source_feature_count)
        logger.info(
            f"Created FeatureSelector from preset '{preset}': "
            f"{selector.output_size} features from {source_feature_count}"
        )
        return selector
    
    if indices:
        selector = FeatureSelector.from_indices(
            indices, source_feature_count, name="custom"
        )
        logger.info(
            f"Created FeatureSelector from custom indices: "
            f"{selector.output_size} features from {source_feature_count}"
        )
        return selector
    
    # No selection specified - return None (use all features)
    return None
