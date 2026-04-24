"""Phase A.5.4 (2026-04-24) regression locks for
``LabelsConfig.validate_primary_horizon_idx_for(n_horizons)``.

Closes plan v4 bugs #2 + #5 at the SSoT primitive layer:

- Bug #2: No bounds check before ``preds[:, primary_idx]`` slicing.
  Python's negative-indexing silently picks last-N (no raise). A config
  with ``primary_horizon_idx=-1`` would slice the LAST horizon without
  any diagnostic — silent-wrong-result hazard per hft-rules §8.

- Bug #5: Silent fallback ``primary_horizon = None`` when the idx is
  >= len(horizons) — no diagnostic fired; metadata had a null field
  making post-hoc investigation opaque.

The method is the canonical SSoT for horizon-index validation across
every downstream slicing site (exporter calibration, stats,
regression metrics, callback metric_fn). Failing-loud at this layer
means NONE of the 4+ call sites can accidentally bypass the gate.
"""
from __future__ import annotations

import pytest

from lobtrainer.config.schema import LabelsConfig


class TestValidatePrimaryHorizonIdxFor:
    """Phase A.5.4 regression locks for the SSoT horizon-idx validator."""

    # --- Defaulting (None → 0) ---------------------------------------------

    def test_none_primary_horizon_idx_defaults_to_zero(self):
        """Legacy convention preserved: ``primary_horizon_idx=None`` means
        "use the first horizon" (returns 0)."""
        lc = LabelsConfig(primary_horizon_idx=None)
        assert lc.validate_primary_horizon_idx_for(n_horizons=3) == 0

    # --- Valid cases --------------------------------------------------------

    def test_valid_int_returns_identity(self):
        """Valid idx within [0, n_horizons) returns unchanged."""
        lc = LabelsConfig(primary_horizon_idx=2)
        assert lc.validate_primary_horizon_idx_for(n_horizons=3) == 2

    def test_boundary_last_valid_idx(self):
        """Idx = n_horizons - 1 is the LAST valid index (inclusive
        upper-bound of [0, n_horizons))."""
        lc = LabelsConfig(primary_horizon_idx=4)
        assert lc.validate_primary_horizon_idx_for(n_horizons=5) == 4

    # --- Negative rejection (bug #2 fix) -----------------------------------

    def test_negative_idx_raises_with_actionable_message(self):
        """Negative idx rejected. Python's negative-indexing would
        silently select last-N column → silent-wrong-result. The error
        message must cite the dangerous silent-mapping."""
        # Construction blocks negative (Pydantic validator), so test via
        # the parametric helper directly.
        with pytest.raises(ValueError, match="Python negative indexing"):
            LabelsConfig._validate_horizon_idx_for(
                field_name="primary_horizon_idx",
                idx_value=-1,
                n_horizons=3,
            )

    def test_negative_idx_error_cites_field_name(self):
        """Error message embeds the field name (important for future
        horizon fields — secondary/tertiary/cascade — sharing the helper)."""
        with pytest.raises(ValueError, match="primary_horizon_idx"):
            LabelsConfig._validate_horizon_idx_for(
                field_name="primary_horizon_idx",
                idx_value=-2,
                n_horizons=5,
            )

    # --- Out-of-bounds rejection (bug #5 fix) ------------------------------

    def test_out_of_bounds_idx_raises_with_diagnostic(self):
        """Idx >= n_horizons raises. Error cites n_horizons AND the
        available index range for operator diagnostics."""
        lc = LabelsConfig(primary_horizon_idx=5)
        with pytest.raises(ValueError, match="out of bounds"):
            lc.validate_primary_horizon_idx_for(n_horizons=3)

    def test_out_of_bounds_error_cites_ranges(self):
        """Error message must state both the failing index AND the
        valid range [0, n_horizons) for actionable repair."""
        lc = LabelsConfig(primary_horizon_idx=10)
        with pytest.raises(ValueError, match=r"\[0, 5\)"):
            lc.validate_primary_horizon_idx_for(n_horizons=5)

    def test_idx_equal_to_n_horizons_raises(self):
        """Strict upper-bound: idx == n_horizons rejects (only [0, n)
        is valid, half-open interval)."""
        lc = LabelsConfig(primary_horizon_idx=3)
        with pytest.raises(ValueError, match="out of bounds"):
            lc.validate_primary_horizon_idx_for(n_horizons=3)

    # --- Zero-n_horizons (degenerate case) ---------------------------------

    def test_n_horizons_zero_raises(self):
        """Can't slice zero horizons — validator rejects at the
        argument-check layer before idx comparison even fires."""
        lc = LabelsConfig(primary_horizon_idx=0)
        with pytest.raises(ValueError, match="n_horizons >= 1"):
            lc.validate_primary_horizon_idx_for(n_horizons=0)

    def test_n_horizons_negative_raises(self):
        """Negative n_horizons is a programming error (not user data).
        Raises with same diagnostic as zero."""
        lc = LabelsConfig(primary_horizon_idx=0)
        with pytest.raises(ValueError, match="n_horizons >= 1"):
            lc.validate_primary_horizon_idx_for(n_horizons=-1)

    # --- Phase B extensibility — parametric helper direct tests -----------

    def test_parametric_helper_direct_call_with_custom_field_name(self):
        """Phase B: the static helper accepts any field_name. Locks the
        single-call path future horizon fields (e.g., secondary_horizon_idx)
        will reuse. Error messages embed the custom field_name so operator
        diagnostics target the correct field."""
        # Valid path
        result = LabelsConfig._validate_horizon_idx_for(
            field_name="secondary_horizon_idx",
            idx_value=1,
            n_horizons=3,
        )
        assert result == 1

        # Negative
        with pytest.raises(ValueError, match="secondary_horizon_idx"):
            LabelsConfig._validate_horizon_idx_for(
                field_name="secondary_horizon_idx",
                idx_value=-1,
                n_horizons=3,
            )

    def test_parametric_helper_none_idx_with_custom_field(self):
        """Custom field_name + None idx still defaults to 0 (consistent
        semantics across all horizon-idx fields)."""
        result = LabelsConfig._validate_horizon_idx_for(
            field_name="future_horizon_idx",
            idx_value=None,
            n_horizons=5,
        )
        assert result == 0
