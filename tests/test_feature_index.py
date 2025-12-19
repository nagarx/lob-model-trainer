"""
Tests for feature index constants.

Validates that feature indices match the Rust pipeline export contract.
"""

import pytest
from lobtrainer.constants import (
    FeatureIndex,
    SignalIndex,
    FEATURE_COUNT,
    LOB_FEATURE_COUNT,
    DERIVED_FEATURE_COUNT,
    MBO_FEATURE_COUNT,
    SIGNAL_FEATURE_COUNT,
    SCHEMA_VERSION,
    LOB_ALL,
    DERIVED_ALL,
    MBO_ALL,
    SIGNALS_ALL,
    OPPOSITE_SIGN_FEATURES,
    UNSIGNED_FEATURES,
)


class TestFeatureCounts:
    """Test feature count constants match documentation."""
    
    def test_total_feature_count(self):
        """Total should be exactly 98."""
        assert FEATURE_COUNT == 98
    
    def test_feature_count_breakdown(self):
        """Sum of categories should equal total."""
        expected = LOB_FEATURE_COUNT + DERIVED_FEATURE_COUNT + MBO_FEATURE_COUNT + SIGNAL_FEATURE_COUNT
        assert expected == FEATURE_COUNT, (
            f"Feature count mismatch: {LOB_FEATURE_COUNT} + {DERIVED_FEATURE_COUNT} + "
            f"{MBO_FEATURE_COUNT} + {SIGNAL_FEATURE_COUNT} = {expected} != {FEATURE_COUNT}"
        )
    
    def test_lob_feature_count(self):
        """LOB features: 10 levels × 4 values = 40."""
        assert LOB_FEATURE_COUNT == 40
    
    def test_derived_feature_count(self):
        """Derived features: 8."""
        assert DERIVED_FEATURE_COUNT == 8
    
    def test_mbo_feature_count(self):
        """MBO features: 36."""
        assert MBO_FEATURE_COUNT == 36
    
    def test_signal_feature_count(self):
        """Trading signals: 14."""
        assert SIGNAL_FEATURE_COUNT == 14
    
    def test_schema_version(self):
        """Schema version should be 2."""
        assert SCHEMA_VERSION == 2


class TestFeatureIndexRanges:
    """Test that feature index ranges are correct."""
    
    def test_lob_range(self):
        """LOB features should be 0-39."""
        assert LOB_ALL == slice(0, 40)
        assert FeatureIndex.BID_PRICE_L0 == 0
        assert FeatureIndex.ASK_SIZE_L9 == 39
    
    def test_derived_range(self):
        """Derived features should be 40-47."""
        assert DERIVED_ALL == slice(40, 48)
        assert FeatureIndex.MID_PRICE == 40
        assert FeatureIndex.PRICE_IMPACT == 47
    
    def test_mbo_range(self):
        """MBO features should be 48-83."""
        assert MBO_ALL == slice(48, 84)
        assert FeatureIndex.ADD_RATE_BID == 48
        assert FeatureIndex.ORDER_COUNT_ACTIVE == 83
    
    def test_signal_range(self):
        """Signal features should be 84-97."""
        assert SIGNALS_ALL == slice(84, 98)
        assert FeatureIndex.TRUE_OFI == 84
        assert FeatureIndex.SCHEMA_VERSION_FEATURE == 97


class TestCriticalIndices:
    """Test critical feature indices match plan documentation."""
    
    def test_ofi_indices(self):
        """OFI signals at expected positions."""
        assert FeatureIndex.TRUE_OFI == 84
        assert FeatureIndex.DEPTH_NORM_OFI == 85
        assert FeatureIndex.EXECUTED_PRESSURE == 86
    
    def test_safety_gates(self):
        """Safety gate indices."""
        assert FeatureIndex.BOOK_VALID == 92
        assert FeatureIndex.MBO_READY == 94
        assert FeatureIndex.INVALIDITY_DELTA == 96
    
    def test_time_regime(self):
        """Time regime is categorical at index 93."""
        assert FeatureIndex.TIME_REGIME == 93
    
    def test_microprice_indices(self):
        """Microprice features at expected positions."""
        assert FeatureIndex.WEIGHTED_MID_PRICE == 46
        assert FeatureIndex.SIGNED_MP_DELTA_BPS == 87
    
    def test_asymmetry_indices(self):
        """Asymmetry signals at expected positions."""
        assert FeatureIndex.TRADE_ASYMMETRY == 88
        assert FeatureIndex.CANCEL_ASYMMETRY == 89
        assert FeatureIndex.DEPTH_ASYMMETRY == 91


class TestSignalIndex:
    """Test SignalIndex convenience enum."""
    
    def test_signal_index_matches_feature_index(self):
        """SignalIndex values should match FeatureIndex."""
        assert SignalIndex.TRUE_OFI == FeatureIndex.TRUE_OFI
        assert SignalIndex.BOOK_VALID == FeatureIndex.BOOK_VALID
        assert SignalIndex.SCHEMA_VERSION == FeatureIndex.SCHEMA_VERSION_FEATURE
    
    def test_signal_index_count(self):
        """SignalIndex should have 14 members."""
        assert len(SignalIndex) == 14


class TestSignConventions:
    """Test sign convention warnings."""
    
    def test_opposite_sign_features(self):
        """NET_TRADE_FLOW has opposite sign convention."""
        assert FeatureIndex.NET_TRADE_FLOW in OPPOSITE_SIGN_FEATURES
        assert len(OPPOSITE_SIGN_FEATURES) == 1
    
    def test_unsigned_features(self):
        """PRICE_IMPACT is unsigned."""
        assert FeatureIndex.PRICE_IMPACT in UNSIGNED_FEATURES
        assert len(UNSIGNED_FEATURES) == 1


class TestLOBLevelIndices:
    """Test LOB level indexing."""
    
    def test_bid_price_levels(self):
        """Bid prices at indices 0-9."""
        for level in range(10):
            idx = getattr(FeatureIndex, f"BID_PRICE_L{level}")
            assert idx == level, f"BID_PRICE_L{level} should be {level}, got {idx}"
    
    def test_ask_price_levels(self):
        """Ask prices at indices 10-19."""
        for level in range(10):
            idx = getattr(FeatureIndex, f"ASK_PRICE_L{level}")
            assert idx == 10 + level, f"ASK_PRICE_L{level} should be {10+level}, got {idx}"
    
    def test_bid_size_levels(self):
        """Bid sizes at indices 20-29."""
        for level in range(10):
            idx = getattr(FeatureIndex, f"BID_SIZE_L{level}")
            assert idx == 20 + level, f"BID_SIZE_L{level} should be {20+level}, got {idx}"
    
    def test_ask_size_levels(self):
        """Ask sizes at indices 30-39."""
        for level in range(10):
            idx = getattr(FeatureIndex, f"ASK_SIZE_L{level}")
            assert idx == 30 + level, f"ASK_SIZE_L{level} should be {30+level}, got {idx}"

