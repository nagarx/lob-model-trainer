"""
Tests for _determine_label_shift_from_metadata — the 4-path label shift resolution.

Covers all resolution paths:
  1. Explicit labeling_strategy argument → contract lookup
  2. Metadata label_strategy / labeling.strategy → contract lookup
  3. Legacy heuristic (label_encoding note)
  4. Default fallback (assumes shift needed)

Schema v2.2.  Reference: hft-rules.md §6 (Testing Philosophy), §9 (ML Pipeline Integrity).
"""

import numpy as np
import pytest

from lobtrainer.data.dataset import DayData, _determine_label_shift_from_metadata


def _make_day_with_metadata(metadata=None) -> DayData:
    """Create minimal DayData with custom metadata."""
    return DayData(
        date="2025-01-01",
        features=np.zeros((5, 10)),
        labels=np.zeros(5, dtype=np.int8),
        metadata=metadata,
    )


# =============================================================================
# Path 1: Explicit strategy argument → contract lookup
# =============================================================================


class TestExplicitStrategy:
    """Test Path 1: explicit labeling_strategy argument."""

    def test_tlob_needs_shift(self):
        """TLOB labels {-1,0,1} need +1 shift."""
        assert _determine_label_shift_from_metadata("tlob", []) is True

    def test_trend_needs_shift(self):
        """Trend (alias for TLOB) needs shift."""
        assert _determine_label_shift_from_metadata("trend", []) is True

    def test_triple_barrier_no_shift(self):
        """Triple barrier labels {0,1,2} do NOT need shift."""
        assert _determine_label_shift_from_metadata("triple_barrier", []) is False

    def test_opportunity_needs_shift(self):
        """Opportunity labels {-1,0,1} need shift."""
        assert _determine_label_shift_from_metadata("opportunity", []) is True

    def test_regression_no_shift(self):
        """Regression labels are float, no shift.

        get_contract('regression') returns RegressionLabelContract which has
        no shift_for_crossentropy attribute. getattr(..., False) returns False.
        """
        assert _determine_label_shift_from_metadata("regression", []) is False

    def test_case_insensitive(self):
        """Strategy name is case-insensitive."""
        assert _determine_label_shift_from_metadata("TLOB", []) is True
        assert _determine_label_shift_from_metadata("Triple_Barrier", []) is False


# =============================================================================
# Path 2: Metadata lookup → contract lookup
# =============================================================================


class TestMetadataLookup:
    """Test Path 2: strategy from metadata."""

    def test_label_strategy_field(self):
        """Reads label_strategy from metadata."""
        day = _make_day_with_metadata({"label_strategy": "triple_barrier"})
        assert _determine_label_shift_from_metadata(None, [day]) is False

    def test_labeling_strategy_nested(self):
        """Reads labeling.strategy from nested metadata."""
        day = _make_day_with_metadata({
            "labeling": {"strategy": "tlob"}
        })
        assert _determine_label_shift_from_metadata(None, [day]) is True


# =============================================================================
# Path 3: Legacy heuristic
# =============================================================================


class TestLegacyHeuristic:
    """Test Path 3: label_encoding note heuristic."""

    def test_encoding_note_0_indexed(self):
        """Note containing 'class indices 0, 1, 2' means no shift."""
        day = _make_day_with_metadata({
            "label_encoding": {
                "note": "Class indices 0, 1, 2 for Down/Stable/Up"
            }
        })
        assert _determine_label_shift_from_metadata(None, [day]) is False

    def test_encoding_note_case_insensitive(self):
        """Heuristic is case-insensitive."""
        day = _make_day_with_metadata({
            "label_encoding": {
                "note": "CLASS INDICES 0, 1, 2"
            }
        })
        assert _determine_label_shift_from_metadata(None, [day]) is False


# =============================================================================
# Path 4: Default fallback
# =============================================================================


class TestDefaultFallback:
    """Test Path 4: default behavior."""

    def test_no_info_defaults_to_shift(self):
        """No strategy, no metadata → defaults to shift (True)."""
        day = _make_day_with_metadata({})
        assert _determine_label_shift_from_metadata(None, [day]) is True

    def test_empty_days_defaults_to_shift(self):
        """Empty days list with no strategy → defaults to shift."""
        assert _determine_label_shift_from_metadata(None, []) is True

    def test_unknown_strategy_falls_through(self):
        """Unknown strategy logs warning and falls through to default."""
        assert _determine_label_shift_from_metadata("unknown_strategy_xyz", []) is True


# =============================================================================
# Edge cases
# =============================================================================


class TestLabelShiftEdgeCases:
    """Test edge cases."""

    def test_explicit_overrides_metadata(self):
        """Explicit strategy argument takes precedence over metadata."""
        day = _make_day_with_metadata({"label_strategy": "triple_barrier"})
        # Explicit "tlob" overrides metadata's "triple_barrier"
        assert _determine_label_shift_from_metadata("tlob", [day]) is True

    def test_metadata_none(self):
        """Day with metadata=None falls through to default."""
        day = _make_day_with_metadata(None)
        assert _determine_label_shift_from_metadata(None, [day]) is True
