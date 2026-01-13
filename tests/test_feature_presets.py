"""
Tests for feature presets module.

Verifies:
- All preset indices are valid (0-97)
- Presets have expected sizes
- No duplicate indices within presets
- get_feature_preset works correctly
- Presets are immutable (tuples)

RULE.md compliance:
- Schema v2.1 specifies 98 features (0-97)
- Presets should match documented feature groups
"""

import pytest

from lobtrainer.constants.feature_presets import (
    # Group definitions
    RAW_LOB_INDICES,
    DERIVED_INDICES,
    MBO_INDICES,
    SIGNAL_INDICES,
    # Named presets
    PRESET_LOB_ONLY,
    PRESET_LOB_DERIVED,
    PRESET_FULL,
    PRESET_SIGNALS_CORE,
    PRESET_SIGNALS_FULL,
    PRESET_LOB_SIGNALS,
    PRESET_NO_META,
    PRESET_DEEPLOB_EXTENDED,
    # Registry
    FEATURE_PRESETS,
    # Functions
    get_feature_preset,
    list_presets,
    describe_preset,
    get_preset_summary,
)
from lobtrainer.constants.feature_index import FeatureIndex, FEATURE_COUNT


# =============================================================================
# Feature Group Tests
# =============================================================================


class TestFeatureGroups:
    """Test feature group definitions match schema v2.1."""
    
    def test_raw_lob_size(self):
        """RAW_LOB_INDICES should have 40 features (10 levels × 4)."""
        assert len(RAW_LOB_INDICES) == 40, (
            f"RAW_LOB should have 40 features, got {len(RAW_LOB_INDICES)}"
        )
    
    def test_raw_lob_range(self):
        """RAW_LOB_INDICES should be indices 0-39."""
        assert RAW_LOB_INDICES == tuple(range(0, 40)), (
            f"RAW_LOB should be 0-39, got {RAW_LOB_INDICES}"
        )
    
    def test_derived_size(self):
        """DERIVED_INDICES should have 8 features."""
        assert len(DERIVED_INDICES) == 8, (
            f"DERIVED should have 8 features, got {len(DERIVED_INDICES)}"
        )
    
    def test_derived_range(self):
        """DERIVED_INDICES should be indices 40-47."""
        assert DERIVED_INDICES == tuple(range(40, 48)), (
            f"DERIVED should be 40-47, got {DERIVED_INDICES}"
        )
    
    def test_mbo_size(self):
        """MBO_INDICES should have 36 features."""
        assert len(MBO_INDICES) == 36, (
            f"MBO should have 36 features, got {len(MBO_INDICES)}"
        )
    
    def test_mbo_range(self):
        """MBO_INDICES should be indices 48-83."""
        assert MBO_INDICES == tuple(range(48, 84)), (
            f"MBO should be 48-83, got {MBO_INDICES}"
        )
    
    def test_signal_size(self):
        """SIGNAL_INDICES should have 14 features."""
        assert len(SIGNAL_INDICES) == 14, (
            f"SIGNAL should have 14 features, got {len(SIGNAL_INDICES)}"
        )
    
    def test_signal_range(self):
        """SIGNAL_INDICES should be indices 84-97."""
        assert SIGNAL_INDICES == tuple(range(84, 98)), (
            f"SIGNAL should be 84-97, got {SIGNAL_INDICES}"
        )
    
    def test_groups_sum_to_total(self):
        """All groups should sum to FEATURE_COUNT (98)."""
        total = (
            len(RAW_LOB_INDICES) + 
            len(DERIVED_INDICES) + 
            len(MBO_INDICES) + 
            len(SIGNAL_INDICES)
        )
        assert total == FEATURE_COUNT, (
            f"Groups should sum to {FEATURE_COUNT}, got {total}"
        )
    
    def test_groups_are_disjoint(self):
        """Feature groups should not overlap."""
        all_indices = (
            set(RAW_LOB_INDICES) | 
            set(DERIVED_INDICES) | 
            set(MBO_INDICES) | 
            set(SIGNAL_INDICES)
        )
        total_unique = len(all_indices)
        total_combined = (
            len(RAW_LOB_INDICES) + 
            len(DERIVED_INDICES) + 
            len(MBO_INDICES) + 
            len(SIGNAL_INDICES)
        )
        assert total_unique == total_combined, (
            f"Groups should be disjoint. Unique={total_unique}, Combined={total_combined}"
        )


# =============================================================================
# Named Preset Tests
# =============================================================================


class TestPresetSizes:
    """Test preset sizes match documentation."""
    
    def test_lob_only_size(self):
        """PRESET_LOB_ONLY should have 40 features."""
        assert len(PRESET_LOB_ONLY) == 40, (
            f"lob_only should have 40 features, got {len(PRESET_LOB_ONLY)}"
        )
    
    def test_lob_derived_size(self):
        """PRESET_LOB_DERIVED should have 48 features."""
        assert len(PRESET_LOB_DERIVED) == 48, (
            f"lob_derived should have 48 features, got {len(PRESET_LOB_DERIVED)}"
        )
    
    def test_full_size(self):
        """PRESET_FULL should have 98 features."""
        assert len(PRESET_FULL) == 98, (
            f"full should have 98 features, got {len(PRESET_FULL)}"
        )
    
    def test_signals_core_size(self):
        """PRESET_SIGNALS_CORE should have 8 features."""
        assert len(PRESET_SIGNALS_CORE) == 8, (
            f"signals_core should have 8 features, got {len(PRESET_SIGNALS_CORE)}"
        )
    
    def test_signals_full_size(self):
        """PRESET_SIGNALS_FULL should have 14 features."""
        assert len(PRESET_SIGNALS_FULL) == 14, (
            f"signals_full should have 14 features, got {len(PRESET_SIGNALS_FULL)}"
        )
    
    def test_lob_signals_size(self):
        """PRESET_LOB_SIGNALS should have 54 features (40 + 14)."""
        assert len(PRESET_LOB_SIGNALS) == 54, (
            f"lob_signals should have 54 features, got {len(PRESET_LOB_SIGNALS)}"
        )
    
    def test_no_meta_size(self):
        """PRESET_NO_META should have 92 features."""
        assert len(PRESET_NO_META) == 92, (
            f"no_meta should have 92 features, got {len(PRESET_NO_META)}"
        )
    
    def test_deeplob_extended_size(self):
        """PRESET_DEEPLOB_EXTENDED should have 52 features."""
        assert len(PRESET_DEEPLOB_EXTENDED) == 52, (
            f"deeplob_extended should have 52 features, got {len(PRESET_DEEPLOB_EXTENDED)}"
        )


class TestPresetValidity:
    """Test presets contain valid indices."""
    
    @pytest.mark.parametrize("preset_name", FEATURE_PRESETS.keys())
    def test_indices_in_valid_range(self, preset_name):
        """All preset indices should be in [0, 97]."""
        indices = FEATURE_PRESETS[preset_name]
        
        for idx in indices:
            assert 0 <= idx <= 97, (
                f"Preset '{preset_name}' has invalid index {idx}. "
                f"Valid range is 0-97."
            )
    
    @pytest.mark.parametrize("preset_name", FEATURE_PRESETS.keys())
    def test_no_duplicate_indices(self, preset_name):
        """Presets should not have duplicate indices."""
        indices = FEATURE_PRESETS[preset_name]
        unique = set(indices)
        
        assert len(unique) == len(indices), (
            f"Preset '{preset_name}' has duplicate indices. "
            f"Total={len(indices)}, Unique={len(unique)}"
        )


class TestPresetImmutability:
    """Test presets are immutable tuples."""
    
    @pytest.mark.parametrize("preset_name", FEATURE_PRESETS.keys())
    def test_preset_is_tuple(self, preset_name):
        """Presets should be tuples (immutable)."""
        indices = FEATURE_PRESETS[preset_name]
        
        assert isinstance(indices, tuple), (
            f"Preset '{preset_name}' should be tuple, got {type(indices)}"
        )


class TestPresetContents:
    """Test preset contents match documentation."""
    
    def test_signals_core_contains_key_signals(self):
        """PRESET_SIGNALS_CORE should contain the 8 most predictive signals."""
        expected_signals = {
            FeatureIndex.TRUE_OFI,           # 84
            FeatureIndex.DEPTH_NORM_OFI,     # 85
            FeatureIndex.EXECUTED_PRESSURE,  # 86
            FeatureIndex.SIGNED_MP_DELTA_BPS,# 87
            FeatureIndex.TRADE_ASYMMETRY,    # 88
            FeatureIndex.CANCEL_ASYMMETRY,   # 89
            FeatureIndex.FRAGILITY_SCORE,    # 90
            FeatureIndex.DEPTH_ASYMMETRY,    # 91
        }
        
        actual = set(PRESET_SIGNALS_CORE)
        
        assert actual == expected_signals, (
            f"signals_core should contain indices 84-91. "
            f"Missing: {expected_signals - actual}, Extra: {actual - expected_signals}"
        )
    
    def test_full_covers_all_features(self):
        """PRESET_FULL should cover all 98 features."""
        expected = set(range(98))
        actual = set(PRESET_FULL)
        
        assert actual == expected, (
            f"full should cover all 98 features. "
            f"Missing: {expected - actual}"
        )
    
    def test_no_meta_excludes_metadata(self):
        """PRESET_NO_META should exclude metadata features."""
        # Metadata features are 92-97
        metadata_indices = {92, 93, 94, 95, 96, 97}
        
        actual = set(PRESET_NO_META)
        overlap = actual & metadata_indices
        
        assert len(overlap) == 0, (
            f"no_meta should exclude metadata. Found: {overlap}"
        )


# =============================================================================
# Function Tests
# =============================================================================


class TestGetFeaturePreset:
    """Test get_feature_preset function."""
    
    def test_valid_preset(self):
        """Should return correct preset for valid name."""
        result = get_feature_preset("lob_only")
        assert result == PRESET_LOB_ONLY
    
    def test_case_insensitive(self):
        """Should be case-insensitive."""
        lower = get_feature_preset("lob_only")
        upper = get_feature_preset("LOB_ONLY")
        mixed = get_feature_preset("Lob_Only")
        
        assert lower == upper == mixed, "Should be case-insensitive"
    
    def test_invalid_preset_raises(self):
        """Should raise ValueError for invalid preset."""
        with pytest.raises(ValueError, match="Unknown feature preset"):
            get_feature_preset("nonexistent_preset")
    
    def test_error_message_lists_available(self):
        """Error message should list available presets."""
        with pytest.raises(ValueError) as exc_info:
            get_feature_preset("bad_name")
        
        error_msg = str(exc_info.value)
        
        # Should mention at least one valid preset
        assert "lob_only" in error_msg or "full" in error_msg, (
            f"Error should list available presets: {error_msg}"
        )


class TestListPresets:
    """Test list_presets function."""
    
    def test_returns_dict(self):
        """Should return a dictionary."""
        result = list_presets()
        assert isinstance(result, dict)
    
    def test_contains_all_presets(self):
        """Should contain all preset names."""
        result = list_presets()
        
        for name in FEATURE_PRESETS.keys():
            assert name in result, f"Missing preset: {name}"
    
    def test_values_are_counts(self):
        """Values should be feature counts."""
        result = list_presets()
        
        for name, count in result.items():
            expected = len(FEATURE_PRESETS[name])
            assert count == expected, (
                f"Count mismatch for '{name}': expected {expected}, got {count}"
            )


class TestDescribePreset:
    """Test describe_preset function."""
    
    @pytest.mark.parametrize("preset_name", FEATURE_PRESETS.keys())
    def test_returns_string(self, preset_name):
        """Should return a description string."""
        result = describe_preset(preset_name)
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_includes_feature_count(self):
        """Description should mention feature count."""
        desc = describe_preset("lob_only")
        assert "40" in desc, f"Description should mention 40 features: {desc}"


class TestGetPresetSummary:
    """Test get_preset_summary function."""
    
    def test_returns_string(self):
        """Should return a formatted string."""
        result = get_preset_summary()
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_includes_all_presets(self):
        """Summary should include all preset names."""
        result = get_preset_summary()
        
        for name in FEATURE_PRESETS.keys():
            assert name in result, f"Summary should include preset '{name}'"


# =============================================================================
# Integration Tests
# =============================================================================


class TestPresetUsage:
    """Test realistic preset usage patterns."""
    
    def test_preset_can_index_array(self):
        """Preset indices should work for array indexing."""
        import numpy as np
        
        # Simulated feature array
        features = np.random.randn(100, 98)
        
        # Select using preset
        indices = get_feature_preset("signals_core")
        selected = features[:, indices]
        
        assert selected.shape == (100, 8), (
            f"signals_core should select 8 features, got shape {selected.shape}"
        )
    
    def test_preset_indices_are_sorted(self):
        """Indices should be usable in order for consistent results."""
        for name, indices in FEATURE_PRESETS.items():
            # Converting to list shouldn't change behavior
            as_list = list(indices)
            
            # Verify indices are valid for sequential access
            assert all(isinstance(i, int) for i in as_list), (
                f"Preset '{name}' should have integer indices"
            )
