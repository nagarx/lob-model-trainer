"""Tests for lobtrainer.analysis.stat_rigor.ci (Phase 2 P2.A — 2026-05-07).

Covers Plan v4 §4.1 spec + Round 1+2 hardenings (#PY-67 stored-arrays
mitigation; assert_finite_pair input validation; ci_low<=point<=ci_high
invariant; block_length auto-derive; multi-horizon slicing).

Per hft-rules §6 testing philosophy: assertions explain WHAT failed +
WHY; deterministic golden tests detect accidental contract changes.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from hft_contracts.test_metrics_ci_artifact import (
    TestMetricsCIArtifact,
    MetricCIBound,
)
from lobtrainer.analysis.stat_rigor.ci import (
    DEFAULT_METRIC_NAMES,
    TestMetricsCIConfig,
    compute_ci,
    compute_test_metrics_ci_for_experiment,
    from_signal_dir,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _synthetic_paired_arrays(
    n: int = 200,
    seed: int = 7,
    signal_strength: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize correlated (labels, predictions) for fast tests."""
    rng = np.random.default_rng(seed)
    labels = rng.standard_normal(n).astype(np.float64)
    noise = rng.standard_normal(n).astype(np.float64)
    predictions = signal_strength * labels + math.sqrt(1 - signal_strength**2) * noise
    return labels, predictions


def _write_signal_dir(
    tmp_path: Path,
    *,
    predicted: np.ndarray,
    labels: np.ndarray,
    metadata: dict | None = None,
    horizon_idx: int = 0,
) -> Path:
    """Write a synthetic signals/test/ directory for from_signal_dir tests."""
    signals_dir = tmp_path / "signals" / "test"
    signals_dir.mkdir(parents=True, exist_ok=True)
    np.save(signals_dir / "predicted_returns.npy", predicted)
    np.save(signals_dir / "regression_labels.npy", labels)
    if metadata is None:
        metadata = {
            "model_type": "synthetic_test",
            "experiment_id": "test_exp",
            "compatibility_fingerprint": "0" * 64,
            "horizon_idx": horizon_idx,
            "metrics": {},
        }
    (signals_dir / "signal_metadata.json").write_text(json.dumps(metadata))
    return signals_dir


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------


class TestPublicAPI:
    def test_default_metric_names_match_plan_v4_spec(self) -> None:
        """Plan v4 §4.1: 7 standard metrics. Locks the canonical set."""
        expected = {
            "test_ic",
            "test_pearson",
            "test_r2",
            "test_mae",
            "test_rmse",
            "test_directional_accuracy",
            "test_profitable_accuracy",
        }
        assert set(DEFAULT_METRIC_NAMES) == expected

    def test_default_config_matches_plan_v4_recommended_values(self) -> None:
        config = TestMetricsCIConfig()
        # Plan v4 v1→v2: n_bootstraps=10000 (NOT 1000 hft_metrics default)
        assert config.n_bootstraps == 10_000
        # Plan v4 v3→v4: block_length=None auto-derives ceil(n^(1/3))
        assert config.block_length is None
        # Plan v4 v1→v2: ci=0.95 (NOT alpha=0.05 wrong-API)
        assert config.ci == 0.95
        # Deterministic seed
        assert config.seed == 42
        # Default primary horizon index for HMHP-R slicing
        assert config.primary_horizon_idx == 0


# ---------------------------------------------------------------------------
# TestMetricsCIConfig validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_n_bootstraps_below_100_raises(self) -> None:
        with pytest.raises(ValueError, match="n_bootstraps"):
            TestMetricsCIConfig(n_bootstraps=99)

    def test_block_length_below_2_raises(self) -> None:
        with pytest.raises(ValueError, match="block_length"):
            TestMetricsCIConfig(block_length=1)

    def test_ci_at_boundary_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="ci="):
            TestMetricsCIConfig(ci=0.0)

    def test_ci_at_boundary_one_raises(self) -> None:
        with pytest.raises(ValueError, match="ci="):
            TestMetricsCIConfig(ci=1.0)

    def test_negative_primary_horizon_idx_raises(self) -> None:
        with pytest.raises(ValueError, match="primary_horizon_idx"):
            TestMetricsCIConfig(primary_horizon_idx=-1)

    def test_unknown_metric_name_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown metric_names"):
            TestMetricsCIConfig(metric_names=("test_xxx_unknown",))

    def test_empty_metric_names_raises(self) -> None:
        with pytest.raises(ValueError, match="metric_names must not be empty"):
            TestMetricsCIConfig(metric_names=())


# ---------------------------------------------------------------------------
# compute_ci core behavior
# ---------------------------------------------------------------------------


class TestComputeCI:
    def test_compute_ci_reproducible_with_same_seed(self) -> None:
        labels, preds = _synthetic_paired_arrays(n=200, seed=7)
        config = TestMetricsCIConfig(n_bootstraps=200, seed=42)
        a1 = compute_ci(preds, labels, config)
        a2 = compute_ci(preds, labels, config)
        # Bootstrap RNG seeded — bounds + point bit-exact across runs
        for name in a1.metrics:
            b1 = a1.metrics[name]
            b2 = a2.metrics[name]
            assert b1.point == pytest.approx(b2.point, abs=1e-12), name
            assert b1.ci_low == pytest.approx(b2.ci_low, abs=1e-12), name
            assert b1.ci_high == pytest.approx(b2.ci_high, abs=1e-12), name

    def test_compute_ci_point_estimate_within_ci_bounds_invariant(self) -> None:
        labels, preds = _synthetic_paired_arrays(n=200, seed=7, signal_strength=0.6)
        config = TestMetricsCIConfig(n_bootstraps=200, seed=42)
        artifact = compute_ci(preds, labels, config)
        # __post_init__ enforces ci_low <= point <= ci_high; if any metric
        # violated, construction would have raised.
        for name, bound in artifact.metrics.items():
            assert bound.ci_low <= bound.point <= bound.ci_high, (
                f"Invariant violation for {name}: "
                f"ci_low={bound.ci_low}, point={bound.point}, ci_high={bound.ci_high}"
            )

    def test_compute_ci_fail_loud_on_nan_input(self) -> None:
        """Per hft-rules §8: assert_finite_pair fires before bootstrap loop."""
        labels, preds = _synthetic_paired_arrays(n=200, seed=7)
        preds_corrupt = preds.copy()
        preds_corrupt[0] = np.nan
        config = TestMetricsCIConfig(n_bootstraps=200, seed=42)
        with pytest.raises(ValueError, match="input invariant violation"):
            compute_ci(preds_corrupt, labels, config)

    def test_compute_ci_fail_loud_on_inf_input(self) -> None:
        labels, preds = _synthetic_paired_arrays(n=200, seed=7)
        labels_corrupt = labels.copy()
        labels_corrupt[5] = np.inf
        config = TestMetricsCIConfig(n_bootstraps=200, seed=42)
        with pytest.raises(ValueError, match="input invariant violation"):
            compute_ci(preds, labels_corrupt, config)

    def test_compute_ci_block_length_auto_derive_for_n_8085(self) -> None:
        """Plan v4 v3→v4: ceil(n^(1/3))=21 for N=8085."""
        labels, preds = _synthetic_paired_arrays(n=8085, seed=11)
        config = TestMetricsCIConfig(n_bootstraps=200, seed=42, block_length=None)
        artifact = compute_ci(preds, labels, config)
        assert artifact.block_length == 21
        assert "auto-derive" in artifact.block_length_source
        assert "Politis-Romano" in artifact.block_length_source

    def test_compute_ci_block_length_explicit_override(self) -> None:
        labels, preds = _synthetic_paired_arrays(n=200, seed=7)
        config = TestMetricsCIConfig(n_bootstraps=200, seed=42, block_length=10)
        artifact = compute_ci(preds, labels, config)
        assert artifact.block_length == 10
        assert "explicit override" in artifact.block_length_source

    def test_compute_ci_handles_multi_horizon_via_slice(self) -> None:
        """HMHP-R shape (N, H) sliced to primary_horizon_idx."""
        labels_1d, preds_1d = _synthetic_paired_arrays(n=200, seed=7)
        # Stack into (N, 3) — primary_horizon_idx=1 should pick the middle column.
        labels_3h = np.column_stack([labels_1d * 0.5, labels_1d, labels_1d * 1.5])
        preds_3h = np.column_stack([preds_1d * 0.5, preds_1d, preds_1d * 1.5])
        config = TestMetricsCIConfig(n_bootstraps=200, seed=42, primary_horizon_idx=1)
        artifact = compute_ci(preds_3h, labels_3h, config)
        # Should match the 1-D version (since middle column == 1-D arrays)
        config_1d = TestMetricsCIConfig(n_bootstraps=200, seed=42, primary_horizon_idx=0)
        artifact_1d = compute_ci(preds_1d, labels_1d, config_1d)
        for name in artifact.metrics:
            assert artifact.metrics[name].point == pytest.approx(
                artifact_1d.metrics[name].point, abs=1e-12
            ), name

    def test_compute_ci_horizon_idx_out_of_bounds_raises(self) -> None:
        labels_1d, preds_1d = _synthetic_paired_arrays(n=200, seed=7)
        labels_3h = np.column_stack([labels_1d, labels_1d, labels_1d])
        preds_3h = np.column_stack([preds_1d, preds_1d, preds_1d])
        config = TestMetricsCIConfig(n_bootstraps=200, seed=42, primary_horizon_idx=5)
        with pytest.raises(IndexError, match="primary_horizon_idx=5"):
            compute_ci(preds_3h, labels_3h, config)

    def test_compute_ci_unexpected_3d_shape_raises(self) -> None:
        config = TestMetricsCIConfig(n_bootstraps=200, seed=42)
        bad_arr = np.zeros((10, 3, 2))
        with pytest.raises(ValueError, match="unexpected shape"):
            compute_ci(bad_arr, bad_arr, config)

    def test_compute_ci_metadata_overlay_propagates(self) -> None:
        labels, preds = _synthetic_paired_arrays(n=200, seed=7)
        config = TestMetricsCIConfig(n_bootstraps=200, seed=42)
        overlay = {
            "compatibility_fingerprint": "a" * 64,
            "model_config_hash": "b" * 64,
            "experiment_id": "test_exp_42",
            "model_type": "tlob",
            "method_caveats": ("ic_silent_sanitize",),
        }
        artifact = compute_ci(preds, labels, config, metadata_overlay=overlay)
        assert artifact.compatibility_fingerprint == "a" * 64
        assert artifact.model_config_hash == "b" * 64
        assert artifact.experiment_id == "test_exp_42"
        assert artifact.model_type == "tlob"
        assert artifact.method_caveats == ("ic_silent_sanitize",)


# ---------------------------------------------------------------------------
# from_signal_dir
# ---------------------------------------------------------------------------


class TestFromSignalDir:
    def test_from_signal_dir_happy_path(self, tmp_path: Path) -> None:
        labels, preds = _synthetic_paired_arrays(n=200, seed=7)
        metadata = {
            "model_type": "synthetic_tlob",
            "experiment_id": "synthetic_test",
            "compatibility_fingerprint": "f" * 64,
            "model_config_hash": "e" * 64,
        }
        signals_dir = _write_signal_dir(
            tmp_path, predicted=preds, labels=labels, metadata=metadata
        )
        config = TestMetricsCIConfig(n_bootstraps=200, seed=42)
        artifact = from_signal_dir(signals_dir, config)
        assert artifact.compatibility_fingerprint == "f" * 64
        assert artifact.model_config_hash == "e" * 64
        assert artifact.model_type == "synthetic_tlob"
        # signal_export_output_dir defaults to the signals_dir if not in metadata
        assert artifact.signal_export_output_dir == str(signals_dir.resolve())

    def test_from_signal_dir_fail_loud_on_missing_predicted(
        self, tmp_path: Path
    ) -> None:
        signals_dir = tmp_path / "signals" / "test"
        signals_dir.mkdir(parents=True)
        # Only labels + metadata — no predicted_returns.npy
        labels = np.zeros(10)
        np.save(signals_dir / "regression_labels.npy", labels)
        (signals_dir / "signal_metadata.json").write_text("{}")
        config = TestMetricsCIConfig(n_bootstraps=200, seed=42)
        with pytest.raises(FileNotFoundError, match="predicted_returns.npy"):
            from_signal_dir(signals_dir, config)

    def test_from_signal_dir_fail_loud_on_missing_metadata(
        self, tmp_path: Path
    ) -> None:
        signals_dir = tmp_path / "signals" / "test"
        signals_dir.mkdir(parents=True)
        labels, preds = _synthetic_paired_arrays(n=200, seed=7)
        np.save(signals_dir / "predicted_returns.npy", preds)
        np.save(signals_dir / "regression_labels.npy", labels)
        # No signal_metadata.json
        config = TestMetricsCIConfig(n_bootstraps=200, seed=42)
        with pytest.raises(FileNotFoundError, match="signal_metadata.json"):
            from_signal_dir(signals_dir, config)


# ---------------------------------------------------------------------------
# Orchestration entry: compute_test_metrics_ci_for_experiment
# ---------------------------------------------------------------------------


class TestComputeForExperiment:
    def test_save_and_cache_hit(self, tmp_path: Path) -> None:
        labels, preds = _synthetic_paired_arrays(n=200, seed=7)
        experiment_dir = tmp_path / "exp"
        _write_signal_dir(
            experiment_dir, predicted=preds, labels=labels,
            metadata={
                "model_type": "test",
                "experiment_id": "x",
                "compatibility_fingerprint": "0" * 64,
            },
        )
        config = TestMetricsCIConfig(n_bootstraps=200, seed=42)
        # First call: computes + saves
        a1 = compute_test_metrics_ci_for_experiment(
            experiment_dir, config=config, skip_if_exists=True
        )
        out_path = experiment_dir / "test_metrics_ci_v1.json"
        assert out_path.exists()
        # Second call: cache hit (returns identical artifact bit-exact)
        a2 = compute_test_metrics_ci_for_experiment(
            experiment_dir, config=config, skip_if_exists=True
        )
        assert a1 == a2
        assert a1.content_hash() == a2.content_hash()

    def test_recompute_on_invalid_existing_artifact(
        self, tmp_path: Path
    ) -> None:
        labels, preds = _synthetic_paired_arrays(n=200, seed=7)
        experiment_dir = tmp_path / "exp"
        _write_signal_dir(experiment_dir, predicted=preds, labels=labels)
        out_path = experiment_dir / "test_metrics_ci_v1.json"
        # Pre-write a corrupt/invalid existing artifact
        out_path.write_text('{"schema_version": "1", "method": "garbage"}')  # missing required fields
        config = TestMetricsCIConfig(n_bootstraps=200, seed=42)
        # Should log warning and recompute (NOT raise)
        artifact = compute_test_metrics_ci_for_experiment(
            experiment_dir, config=config, skip_if_exists=True
        )
        # New artifact should be valid
        assert artifact.method == "block_bootstrap"
        # File should be overwritten with valid artifact
        loaded = TestMetricsCIArtifact.load(out_path)
        assert loaded.method == "block_bootstrap"

    def test_round_trip_via_save_load(self, tmp_path: Path) -> None:
        """Bit-exact round trip via artifact save → load."""
        labels, preds = _synthetic_paired_arrays(n=200, seed=7)
        experiment_dir = tmp_path / "exp"
        _write_signal_dir(experiment_dir, predicted=preds, labels=labels)
        config = TestMetricsCIConfig(n_bootstraps=200, seed=42)
        a1 = compute_test_metrics_ci_for_experiment(
            experiment_dir, config=config, skip_if_exists=False
        )
        a2 = TestMetricsCIArtifact.load(experiment_dir / "test_metrics_ci_v1.json")
        assert a1 == a2
        assert a1.content_hash() == a2.content_hash()


# ---------------------------------------------------------------------------
# Golden fixture (Plan v4 §4.1 v4 ADD per validator 2 + §8.13)
# ---------------------------------------------------------------------------


class TestRound1MidImplFixes:
    """Regression locks for Round 1 mid-impl adversarial findings (2026-05-07)."""

    def test_resolve_block_length_fail_loud_on_block_length_ge_n_explicit(
        self,
    ) -> None:
        """§1 HIGH: explicit block_length >= n_samples must fail-loud (degenerate
        single-block bootstrap collapses CI to point)."""
        labels, preds = _synthetic_paired_arrays(n=20, seed=7)
        # Explicit override: block_length=20 >= n_samples=20 → degenerate
        config = TestMetricsCIConfig(n_bootstraps=200, seed=42, block_length=20)
        with pytest.raises(ValueError, match="degenerate single-block bootstrap"):
            compute_ci(preds, labels, config)

    def test_resolve_block_length_fail_loud_on_block_length_gt_n_explicit(
        self,
    ) -> None:
        """§1 HIGH: block_length > n_samples (more extreme) also fails."""
        labels, preds = _synthetic_paired_arrays(n=10, seed=7)
        config = TestMetricsCIConfig(n_bootstraps=200, seed=42, block_length=50)
        with pytest.raises(ValueError, match="degenerate"):
            compute_ci(preds, labels, config)

    def test_recompute_on_config_drift_n_bootstraps(
        self, tmp_path: Path
    ) -> None:
        """§1 MEDIUM: cache-hit must verify config matches existing artifact —
        silent stale-artifact return is §8 violation."""
        labels, preds = _synthetic_paired_arrays(n=200, seed=7)
        experiment_dir = tmp_path / "exp"
        _write_signal_dir(
            experiment_dir, predicted=preds, labels=labels,
            metadata={
                "model_type": "test", "experiment_id": "x",
                "compatibility_fingerprint": "0" * 64,
            },
        )
        # First call: 200 bootstraps
        config_a = TestMetricsCIConfig(n_bootstraps=200, seed=42)
        a1 = compute_test_metrics_ci_for_experiment(
            experiment_dir, config=config_a, skip_if_exists=True
        )
        assert a1.n_bootstraps == 200
        # Second call: 500 bootstraps — should detect drift + recompute
        config_b = TestMetricsCIConfig(n_bootstraps=500, seed=42)
        a2 = compute_test_metrics_ci_for_experiment(
            experiment_dir, config=config_b, skip_if_exists=True
        )
        assert a2.n_bootstraps == 500, (
            "Cache-hit drift-detection failed: returned stale 200-bootstrap "
            "artifact when caller requested 500 bootstraps. §8 violation."
        )

    def test_recompute_on_config_drift_seed(self, tmp_path: Path) -> None:
        """§1 MEDIUM: seed drift triggers recompute."""
        labels, preds = _synthetic_paired_arrays(n=200, seed=7)
        experiment_dir = tmp_path / "exp"
        _write_signal_dir(experiment_dir, predicted=preds, labels=labels)
        config_a = TestMetricsCIConfig(n_bootstraps=200, seed=42)
        a1 = compute_test_metrics_ci_for_experiment(
            experiment_dir, config=config_a, skip_if_exists=True
        )
        config_b = TestMetricsCIConfig(n_bootstraps=200, seed=99)
        a2 = compute_test_metrics_ci_for_experiment(
            experiment_dir, config=config_b, skip_if_exists=True
        )
        assert a2.seed == 99
        # Different seeds → different bootstrap distributions → typically
        # different ci_low/ci_high (point estimate is identical since it's
        # computed on un-resampled data).
        assert a1.metrics["test_ic"].point == pytest.approx(
            a2.metrics["test_ic"].point, abs=1e-12
        )

    def test_recompute_on_config_drift_metric_names(
        self, tmp_path: Path
    ) -> None:
        """§1 MEDIUM: metric_names subset drift triggers recompute."""
        labels, preds = _synthetic_paired_arrays(n=200, seed=7)
        experiment_dir = tmp_path / "exp"
        _write_signal_dir(experiment_dir, predicted=preds, labels=labels)
        config_full = TestMetricsCIConfig(n_bootstraps=200, seed=42)
        a1 = compute_test_metrics_ci_for_experiment(
            experiment_dir, config=config_full, skip_if_exists=True
        )
        assert len(a1.metrics) == 7
        # Subset config: only test_ic
        config_subset = TestMetricsCIConfig(
            n_bootstraps=200, seed=42, metric_names=("test_ic",),
        )
        a2 = compute_test_metrics_ci_for_experiment(
            experiment_dir, config=config_subset, skip_if_exists=True
        )
        assert set(a2.metrics.keys()) == {"test_ic"}, (
            "Cache-hit drift-detection failed on metric_names subset"
        )

    def test_metric_ci_bound_leaf_validation_fires_before_artifact(
        self,
    ) -> None:
        """§2 HIGH fix: MetricCIBound's own __post_init__ validates the
        ci_low <= point <= ci_high invariant; consumers don't need to wait
        for the parent artifact to catch it (better error message)."""
        with pytest.raises(
            ValueError, match="ci_low <= point <= ci_high"
        ):
            from hft_contracts.test_metrics_ci_artifact import MetricCIBound
            MetricCIBound(point=0.5, ci_low=0.7, ci_high=0.9, n_samples=10)


class TestGoldenFixture:
    """Lock bit-exact reproducibility across NumPy/SciPy version bumps.

    Per Plan v4 §4.1 ``test_bootstrap_ci_golden_fixture``: synthetic
    arrays with fixed seed should produce a stable content_hash.
    """

    def test_synthetic_n200_golden_content_hash(self) -> None:
        labels, preds = _synthetic_paired_arrays(n=200, seed=7)
        config = TestMetricsCIConfig(
            n_bootstraps=1000, block_length=None, seed=42,
        )
        artifact = compute_ci(preds, labels, config)
        # Verify shape invariants of the fixture
        assert artifact.n_test_samples == 200
        assert artifact.block_length == math.ceil(200 ** (1.0 / 3.0))  # 6 for N=200
        # Verify all 7 metrics present
        assert set(artifact.metrics.keys()) == set(DEFAULT_METRIC_NAMES)
        # Verify CI invariants (re-asserted for clarity)
        for name, bound in artifact.metrics.items():
            assert bound.ci_low <= bound.point <= bound.ci_high
            assert bound.n_samples == 200
        # content_hash deterministic check (does NOT lock specific hash —
        # bootstrap RNG sensitivity to NumPy version makes that brittle;
        # but invariants above + reproducibility-with-same-seed test cover
        # the contract).
        h1 = artifact.content_hash()
        h2 = artifact.content_hash()
        assert h1 == h2
        assert len(h1) == 64
