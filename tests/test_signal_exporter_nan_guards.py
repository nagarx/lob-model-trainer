"""#PY-63 Adversarial Item 3 — exporter.py NaN guard regression tests.

Per saved feedback memory `feedback_final_adversarial_validation_round.md`
+ 2026-05-07 Round 2 Adversarial Item 3 (MEDIUM): lock the four
producer-side ``assert_finite_array`` guards in
``src/lobtrainer/export/exporter.py`` so future refactors cannot silently
drop them.

These are SOURCE-SNAPSHOT regression tests (matching the
``hft-ops/tests/test_bugfixes_phase2b.py::TestBackfillCorruptMetricsFileFailLoud``
convention used for F3 closure). Functional NaN-injection via full
SignalExporter integration is deferred — it requires a non-trivial
trainer/dataloader fixture; the SSoT helper itself is already covered
by 17 unit tests at
``hft-contracts/tests/test_validation_assert_finite.py``.

The four sites locked here:
  1. ``_infer_regression.predicted_returns`` (after np.concatenate, single-horizon path)
  2. ``_infer_hmhp_regression.predicted_returns`` (after np.column_stack, multi-horizon path)
  3. ``_infer_hmhp_regression.agreement_ratio`` (NEW SITE this cycle — discovered by
     adversarial Plan agent during the mid-impl round; protects cross-horizon
     decoder confidence from silently degrading ReadabilityHybridStrategy gate)
  4. ``_apply_calibration.calibrated`` (before ``result["calibrated"] = ...``
     assignment — prevents corrupt calibration outputs from reaching backtester)

If any of these fail, do NOT delete the test — investigate whether the
guard was intentionally removed (rare) or refactored to a different
location (more likely; update the test). Removing the guard without
updating this test is a §8 violation.
"""
from __future__ import annotations

from pathlib import Path


EXPORTER_PATH = (
    Path(__file__).resolve().parents[1]
    / "src" / "lobtrainer" / "export" / "exporter.py"
)


class TestExporterNaNGuards:
    """#PY-63 source-snapshot lock for the 4 producer-side fail-loud guards."""

    def test_exporter_source_at_expected_path(self) -> None:
        """Sanity check: exporter.py is where we expect."""
        assert EXPORTER_PATH.is_file(), (
            f"Missing source file: {EXPORTER_PATH}. "
            f"Has the trainer been refactored?"
        )

    def test_assert_finite_array_imports_present(self) -> None:
        """All 4 guard sites import assert_finite_array. The import is
        function-local (matches the existing exporter convention of
        keeping hft_contracts imports inside the methods that use them
        to avoid module-level circular-import risk).
        """
        source = EXPORTER_PATH.read_text()
        import_count = source.count(
            "from hft_contracts.validation import assert_finite_array"
        )
        assert import_count >= 4, (
            f"Expected ≥4 function-local imports of assert_finite_array, "
            f"found {import_count}. #PY-63 producer fail-loud requires "
            f"the helper at 4 sites (predicted_returns single-horizon + "
            f"multi-horizon + agreement_ratio NEW + calibrated)."
        )

    def test_site_1_infer_regression_predicted_returns_guard(self) -> None:
        """Site #1: _infer_regression has fail-loud on predicted_returns
        after np.concatenate (single-horizon regression path)."""
        source = EXPORTER_PATH.read_text()
        assert "def _infer_regression" in source, (
            "Function _infer_regression renamed or removed."
        )
        assert (
            'name="SignalExporter._infer_regression.predicted_returns"'
            in source
        ), (
            "Site #1 guard missing or name field drifted. "
            "#PY-63 closure required this guard to prevent corrupt "
            "single-horizon predictions from reaching downstream."
        )

    def test_site_2_infer_hmhp_regression_predicted_returns_guard(self) -> None:
        """Site #2: _infer_hmhp_regression has fail-loud on
        predicted_returns after np.column_stack (multi-horizon path)."""
        source = EXPORTER_PATH.read_text()
        assert "def _infer_hmhp_regression" in source, (
            "Function _infer_hmhp_regression renamed or removed."
        )
        assert (
            'name="SignalExporter._infer_hmhp_regression.predicted_returns"'
            in source
        ), (
            "Site #2 guard missing or name field drifted. "
            "#PY-63 closure required this guard to prevent corrupt "
            "multi-horizon predictions from reaching downstream."
        )

    def test_site_3_agreement_ratio_NEW_guard(self) -> None:
        """Site #3 (NEW SITE this cycle): agreement_ratio fail-loud in
        _infer_hmhp_regression. Discovered by adversarial Plan agent
        during #PY-63 mid-impl round.

        agreement_ratio is HMHP-R cross-horizon decoder confidence; NaN
        here would silently degrade ReadabilityHybridStrategy gate
        computation in the backtester. This is the THIRD producer-side
        site in _infer_hmhp_regression (after predicted_returns + the
        agreement-block presence check) — must protect symmetrically.
        """
        source = EXPORTER_PATH.read_text()
        assert (
            'name="SignalExporter._infer_hmhp_regression.agreement_ratio"'
            in source
        ), (
            "Site #3 (NEW) agreement_ratio guard missing or drifted. "
            "Adversarial Plan agent in #PY-63 mid-impl round flagged "
            "this gap — silent NaN here would propagate to "
            "ReadabilityHybridStrategy gate. Check exporter.py:396 area."
        )

    def test_site_4_apply_calibration_calibrated_guard(self) -> None:
        """Site #4: _apply_calibration has fail-loud on calibrated array
        BEFORE result["calibrated"] = ... assignment (prevents corrupt
        calibration outputs from reaching backtester)."""
        source = EXPORTER_PATH.read_text()
        assert "def _apply_calibration" in source, (
            "Function _apply_calibration renamed or removed."
        )
        assert (
            'name="SignalExporter._apply_calibration.calibrated"'
            in source
        ), (
            "Site #4 calibration guard missing or name field drifted. "
            "#PY-63 closure required this guard to prevent corrupt "
            "calibrated arrays from reaching the backtester via "
            "result['calibrated']."
        )

    def test_all_four_guard_call_sites_present(self) -> None:
        """All 4 #PY-63 producer fail-loud call sites must coexist.
        Catches removal or rename collateral damage from refactors."""
        source = EXPORTER_PATH.read_text()
        guard_call_count = source.count("assert_finite_array(")
        assert guard_call_count >= 4, (
            f"Expected ≥4 assert_finite_array call sites in exporter.py, "
            f"found {guard_call_count}. #PY-63 closure required 4 producer "
            f"sites: predicted_returns (single + multi-horizon), "
            f"agreement_ratio (NEW), calibrated. ANY drop is §8 violation."
        )
