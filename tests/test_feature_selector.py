"""
Tests for FeatureSelector and create_feature_selector factory.

Covers: construction (preset/indices/all_features), validation (4 error paths),
selection (1D/2D/3D), identity fast path, factory function, properties.

Schema v2.2.  Reference: hft-rules.md §6 (Testing Philosophy).
"""

import numpy as np
import pytest

from lobtrainer.data.feature_selector import (
    FeatureSelector,
    create_feature_selector,
)


# =============================================================================
# FeatureSelector construction
# =============================================================================


class TestFeatureSelectorConstruction:
    """Test FeatureSelector creation via different paths."""

    def test_from_preset_lob_only(self):
        """from_preset('lob_only', 98) creates selector with 40 indices."""
        sel = FeatureSelector.from_preset("lob_only", 98)
        assert sel.output_size == 40
        assert sel.name == "lob_only"
        assert sel.source_feature_count == 98

    def test_from_preset_full_98(self):
        """from_preset('full_98', 98) creates identity selector."""
        sel = FeatureSelector.from_preset("full_98", 98)
        assert sel.output_size == 98
        assert sel.is_identity

    def test_from_preset_unknown_raises(self):
        """Unknown preset raises ValueError."""
        with pytest.raises(ValueError, match="Unknown feature preset"):
            FeatureSelector.from_preset("nonexistent_preset", 98)

    def test_from_preset_exceeds_source_raises(self):
        """Preset with indices > source_feature_count raises ValueError."""
        with pytest.raises(ValueError):
            FeatureSelector.from_preset("full_116", 98)  # 116 has indices > 97

    def test_from_indices_valid(self):
        """from_indices with valid indices works."""
        sel = FeatureSelector.from_indices([0, 10, 20, 84, 85], 98, "custom_5")
        assert sel.output_size == 5
        assert sel.indices == (0, 10, 20, 84, 85)

    def test_all_features(self):
        """all_features creates identity selector."""
        sel = FeatureSelector.all_features(98)
        assert sel.output_size == 98
        assert sel.is_identity
        assert sel.indices == tuple(range(98))


# =============================================================================
# Validation (4 error paths from validate_feature_indices)
# =============================================================================


class TestFeatureSelectorValidation:
    """Test all 4 validation error paths."""

    def test_empty_indices_raises(self):
        """Empty indices raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            FeatureSelector.from_indices([], 98)

    def test_duplicate_indices_raises(self):
        """Duplicate indices raises ValueError."""
        with pytest.raises(ValueError, match="duplicates"):
            FeatureSelector.from_indices([0, 1, 1, 2], 98)

    def test_negative_index_raises(self):
        """Negative index raises ValueError."""
        with pytest.raises(ValueError, match="negative"):
            FeatureSelector.from_indices([-1, 0, 1], 98)

    def test_out_of_bounds_raises(self):
        """Index >= source_feature_count raises ValueError."""
        with pytest.raises(ValueError, match="98"):
            FeatureSelector.from_indices([0, 1, 98], 98)


# =============================================================================
# Selection
# =============================================================================


class TestFeatureSelectorSelect:
    """Test select() on different input shapes."""

    def test_select_2d(self):
        """select() on [T, F] extracts correct columns."""
        sel = FeatureSelector.from_indices([0, 5, 10], 20, "test")
        data = np.arange(100).reshape(5, 20)  # [5, 20]
        result = sel.select(data)
        assert result.shape == (5, 3)
        np.testing.assert_array_equal(result[:, 0], data[:, 0])
        np.testing.assert_array_equal(result[:, 1], data[:, 5])
        np.testing.assert_array_equal(result[:, 2], data[:, 10])

    def test_select_3d(self):
        """select() on [N, T, F] extracts last dimension."""
        sel = FeatureSelector.from_indices([0, 1], 5, "test")
        data = np.arange(30).reshape(2, 3, 5)  # [2, 3, 5]
        result = sel.select(data)
        assert result.shape == (2, 3, 2)
        np.testing.assert_array_equal(result[..., 0], data[..., 0])
        np.testing.assert_array_equal(result[..., 1], data[..., 1])

    def test_select_1d(self):
        """select() on [F] works (undocumented but valid)."""
        sel = FeatureSelector.from_indices([0, 5], 10, "test")
        data = np.arange(10)
        result = sel.select(data)
        assert result.shape == (2,)
        np.testing.assert_array_equal(result, [0, 5])

    def test_select_wrong_features_raises(self):
        """Data with wrong feature count raises ValueError."""
        sel = FeatureSelector.from_indices([0, 1], 10, "test")
        with pytest.raises(ValueError, match="expects 10"):
            sel.select(np.zeros((5, 20)))

    def test_select_identity_returns_input(self):
        """Identity selector returns the same array object (fast path)."""
        sel = FeatureSelector.all_features(10)
        data = np.zeros((5, 10))
        result = sel.select(data)
        assert result is data  # Same object, not a copy


# =============================================================================
# Properties and utilities
# =============================================================================


class TestFeatureSelectorProperties:
    """Test properties and utility methods."""

    def test_output_size(self):
        """output_size matches len(indices)."""
        sel = FeatureSelector.from_indices([0, 10, 20], 40, "test")
        assert sel.output_size == 3

    def test_is_identity_true(self):
        """is_identity=True when all features selected in order."""
        sel = FeatureSelector.all_features(40)
        assert sel.is_identity is True

    def test_is_identity_false_subset(self):
        """is_identity=False when subset selected."""
        sel = FeatureSelector.from_indices([0, 1, 2], 40, "test")
        assert sel.is_identity is False

    def test_get_index_mapping(self):
        """get_index_mapping returns {output_idx: original_idx}."""
        sel = FeatureSelector.from_indices([84, 85, 86], 98, "ofi")
        mapping = sel.get_index_mapping()
        assert mapping == {0: 84, 1: 85, 2: 86}

    def test_repr(self):
        """repr includes name and sizes."""
        sel = FeatureSelector.from_indices([0, 1], 98, "test_sel")
        r = repr(sel)
        assert "test_sel" in r
        assert "2" in r  # output_size
        assert "98" in r  # source_feature_count

    def test_frozen(self):
        """FeatureSelector is frozen (immutable)."""
        sel = FeatureSelector.from_indices([0, 1], 98, "test")
        with pytest.raises(AttributeError):
            sel.indices = (0, 1, 2)


# =============================================================================
# Factory function
# =============================================================================


class TestCreateFeatureSelector:
    """Test create_feature_selector factory."""

    def test_from_preset(self):
        """Factory with preset returns FeatureSelector."""
        sel = create_feature_selector(preset="lob_only", source_feature_count=98)
        assert sel is not None
        assert sel.output_size == 40

    def test_from_indices(self):
        """Factory with indices returns FeatureSelector."""
        sel = create_feature_selector(indices=[0, 1, 2], source_feature_count=98)
        assert sel is not None
        assert sel.output_size == 3

    def test_both_raises(self):
        """Specifying both preset and indices raises ValueError."""
        with pytest.raises(ValueError, match="not both"):
            create_feature_selector(
                preset="lob_only", indices=[0, 1], source_feature_count=98
            )

    def test_neither_returns_none(self):
        """No arguments returns None."""
        sel = create_feature_selector()
        assert sel is None
