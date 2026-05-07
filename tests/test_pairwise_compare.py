"""Tests for lobtrainer.analysis.stat_rigor.pairwise (Phase 2 P2.C — 2026-05-07).

Mirrors test_bootstrap_ci_orchestration.py structure (P2.A precedent).
Locks K-way pairwise-comparison invariants + Round 1+2 hardenings.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from hft_contracts.pairwise_compare_artifact import (
    PairwiseCompareArtifact,
    PairwiseResultRecord,
)
from lobtrainer.analysis.stat_rigor.pairwise import (
    DEFAULT_METRIC_NAME,
    PairwiseCompareConfig,
    compare_k_way,
    compare_pair,
    compute_pairwise_compare_for_experiments,
    from_signal_dirs,
)


def _synthetic_treatments(
    K: int = 3,
    n: int = 200,
    seed: int = 7,
    signal_strengths: tuple = (0.6, 0.5, 0.0),
) -> tuple:
    """Synthesize K treatments with shared labels but different signal strengths."""
    rng = np.random.default_rng(seed)
    labels = rng.standard_normal(n).astype(np.float64)
    # K treatments — each with different correlation to labels
    treatments = []
    for k in range(K):
        s = signal_strengths[k] if k < len(signal_strengths) else 0.0
        noise = rng.standard_normal(n).astype(np.float64)
        preds = s * labels + math.sqrt(1 - s**2) * noise
        treatments.append((f"T{k}", preds, labels))
    return treatments


def _write_signal_dirs(
    tmp_path: Path,
    treatments: list,
    *,
    compat_fps: list | None = None,
    model_cfg_hashes: list | None = None,
) -> list:
    """Write K synthetic signal dirs + return [(label, path), ...]."""
    K = len(treatments)
    if compat_fps is None:
        compat_fps = ["a" * 64] * K  # all share
    if model_cfg_hashes is None:
        model_cfg_hashes = [f"{chr(ord('b') + k) * 64}" for k in range(K)]
    signal_dirs = []
    for k, (label, preds, labs) in enumerate(treatments):
        d = tmp_path / label / "signals" / "test"
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "predicted_returns.npy", preds)
        np.save(d / "regression_labels.npy", labs)
        metadata = {
            "model_type": "synthetic",
            "experiment_id": label,
            "compatibility_fingerprint": compat_fps[k],
            "model_config_hash": model_cfg_hashes[k],
        }
        (d / "signal_metadata.json").write_text(json.dumps(metadata))
        signal_dirs.append((label, d))
    return signal_dirs


# ---------------------------------------------------------------------------
# Public API + Config
# ---------------------------------------------------------------------------


class TestPublicAPI:
    def test_default_metric_name(self) -> None:
        assert DEFAULT_METRIC_NAME == "spearman_ic"

    def test_default_config_values(self) -> None:
        config = PairwiseCompareConfig()
        assert config.n_bootstraps == 10_000
        assert config.block_length is None  # auto-derive
        assert config.alpha == 0.05  # NOT ci=0.95 (primitive convention)
        assert config.seed == 42
        assert config.metric_name == "spearman_ic"
        assert config.primary_horizon_idx == 0
        assert config.max_drop_frac == 0.05


class TestConfigValidation:
    def test_n_bootstraps_below_100_raises(self) -> None:
        with pytest.raises(ValueError, match="n_bootstraps"):
            PairwiseCompareConfig(n_bootstraps=99)

    def test_block_length_below_2_raises(self) -> None:
        with pytest.raises(ValueError, match="block_length"):
            PairwiseCompareConfig(block_length=1)

    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1])
    def test_alpha_out_of_range_raises(self, alpha: float) -> None:
        with pytest.raises(ValueError, match="alpha"):
            PairwiseCompareConfig(alpha=alpha)

    def test_unknown_metric_name_raises(self) -> None:
        with pytest.raises(ValueError, match="metric_name"):
            PairwiseCompareConfig(metric_name="nonexistent_metric")

    def test_negative_horizon_idx_raises(self) -> None:
        with pytest.raises(ValueError, match="primary_horizon_idx"):
            PairwiseCompareConfig(primary_horizon_idx=-1)

    def test_max_drop_frac_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="max_drop_frac"):
            PairwiseCompareConfig(max_drop_frac=1.5)


# ---------------------------------------------------------------------------
# compare_k_way + compare_pair
# ---------------------------------------------------------------------------


class TestCompareKWay:
    def test_k3_basic(self) -> None:
        treatments = _synthetic_treatments(K=3, n=200, seed=7)
        config = PairwiseCompareConfig(n_bootstraps=200, seed=42)
        # parent_metadata required for compat_fp invariant validation
        parent = [{"compatibility_fingerprint": "a" * 64, "experiment_id": f"T{k}"}
                  for k in range(3)]
        artifact = compare_k_way(treatments, config, parent_metadata=parent)
        assert artifact.n_treatments == 3
        assert len(artifact.pairs) == 3  # K*(K-1)/2 = 3
        assert artifact.metric_name == "spearman_ic"
        # Higher signal strength should produce higher statistic
        # signal_strengths = (0.6, 0.5, 0.0)
        # T0 best, T2 worst — pairs validate this ordering
        for p in artifact.pairs:
            if p.treatment_a_label == "T0" and p.treatment_b_label == "T2":
                assert p.delta > 0, "T0 (signal_strength=0.6) should beat T2 (=0.0)"
                assert p.statistic_a > p.statistic_b

    def test_compare_k_way_reproducible_with_same_seed(self) -> None:
        treatments = _synthetic_treatments(K=3, n=200, seed=7)
        config = PairwiseCompareConfig(n_bootstraps=200, seed=42)
        parent = [{"compatibility_fingerprint": "a" * 64, "experiment_id": f"T{k}"}
                  for k in range(3)]
        a1 = compare_k_way(treatments, config, parent_metadata=parent)
        a2 = compare_k_way(treatments, config, parent_metadata=parent)
        # Bootstrap RNG seeded — pairs bit-exact across runs (modulo timestamp)
        for p1, p2 in zip(a1.pairs, a2.pairs):
            assert p1.delta == pytest.approx(p2.delta, abs=1e-12)
            assert p1.delta_ci_low == pytest.approx(p2.delta_ci_low, abs=1e-12)
            assert p1.delta_ci_high == pytest.approx(p2.delta_ci_high, abs=1e-12)

    def test_compare_pair_k2_sugar(self) -> None:
        labels = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 40)
        rng = np.random.default_rng(7)
        preds_a = labels + rng.standard_normal(len(labels))
        preds_b = labels + 2 * rng.standard_normal(len(labels))
        config = PairwiseCompareConfig(n_bootstraps=200, seed=42)
        parent = [
            {"compatibility_fingerprint": "a" * 64, "experiment_id": "A"},
            {"compatibility_fingerprint": "a" * 64, "experiment_id": "B"},
        ]
        artifact = compare_pair("A", preds_a, labels, "B", preds_b, labels, config,
                                parent_metadata=parent)
        assert artifact.n_treatments == 2
        assert len(artifact.pairs) == 1


class TestPairedDataInvariants:
    def test_unpaired_labels_raises(self) -> None:
        """Two treatments with different labels arrays must raise."""
        rng = np.random.default_rng(7)
        labels_a = rng.standard_normal(200)
        labels_b = rng.standard_normal(200)  # Different labels — invalid!
        preds_a = rng.standard_normal(200)
        preds_b = rng.standard_normal(200)
        treatments = [("A", preds_a, labels_a), ("B", preds_b, labels_b)]
        config = PairwiseCompareConfig(n_bootstraps=200, seed=42)
        parent = [{"compatibility_fingerprint": "a" * 64, "experiment_id": l}
                  for l in ["A", "B"]]
        with pytest.raises(ValueError, match="byte-identical"):
            compare_k_way(treatments, config, parent_metadata=parent)

    def test_k_below_2_raises(self) -> None:
        labels = np.array([1.0, 2.0, 3.0])
        treatments = [("A", labels.copy(), labels)]
        config = PairwiseCompareConfig(n_bootstraps=200, seed=42)
        with pytest.raises(ValueError, match="< 2"):
            compare_k_way(treatments, config)

    def test_compat_fps_not_shared_raises(self) -> None:
        treatments = _synthetic_treatments(K=2, n=200, seed=7)
        config = PairwiseCompareConfig(n_bootstraps=200, seed=42)
        # 2 different compat_fps — invalid!
        parent = [
            {"compatibility_fingerprint": "a" * 64, "experiment_id": "T0"},
            {"compatibility_fingerprint": "b" * 64, "experiment_id": "T1"},
        ]
        with pytest.raises(ValueError, match="shared"):
            compare_k_way(treatments, config, parent_metadata=parent)


# ---------------------------------------------------------------------------
# from_signal_dirs
# ---------------------------------------------------------------------------


class TestFromSignalDirs:
    def test_k3_happy_path(self, tmp_path: Path) -> None:
        treatments = _synthetic_treatments(K=3, n=200, seed=7)
        signal_dirs = _write_signal_dirs(tmp_path, treatments)
        config = PairwiseCompareConfig(n_bootstraps=200, seed=42)
        artifact = from_signal_dirs(signal_dirs, config)
        assert artifact.n_treatments == 3
        assert artifact.paired_compat_fingerprint == "a" * 64

    def test_missing_signal_file_raises(self, tmp_path: Path) -> None:
        """Missing predicted_returns.npy / regression_labels.npy /
        signal_metadata.json must raise FileNotFoundError per §8."""
        signal_dirs = [
            ("A", tmp_path / "missing"),
            ("B", tmp_path / "also_missing"),
        ]
        with pytest.raises(FileNotFoundError, match="missing required file"):
            from_signal_dirs(
                signal_dirs,
                PairwiseCompareConfig(n_bootstraps=200, seed=42),
            )

    def test_compat_fp_drift_across_dirs_raises(self, tmp_path: Path) -> None:
        treatments = _synthetic_treatments(K=2, n=200, seed=7)
        # Different compat_fps — paired-data invariant violation
        signal_dirs = _write_signal_dirs(
            tmp_path, treatments,
            compat_fps=["a" * 64, "b" * 64],
        )
        config = PairwiseCompareConfig(n_bootstraps=200, seed=42)
        with pytest.raises(ValueError, match="compat"):
            from_signal_dirs(signal_dirs, config)

    def test_pre_phase_q65_sklearn_missing_compat_fp_raises(
        self, tmp_path: Path
    ) -> None:
        """#PY-68 risk: sklearn pre-Phase-Q.6.5 lacks compat_fp.

        Pairwise comparison requires it for paired-data verification.
        """
        treatments = _synthetic_treatments(K=2, n=200, seed=7)
        signal_dirs = _write_signal_dirs(
            tmp_path, treatments,
            compat_fps=["", "a" * 64],  # First is empty (sklearn pre-Q.6.5)
        )
        config = PairwiseCompareConfig(n_bootstraps=200, seed=42)
        with pytest.raises(ValueError, match="compatibility_fingerprint"):
            from_signal_dirs(signal_dirs, config)


# ---------------------------------------------------------------------------
# compute_pairwise_compare_for_experiments orchestration
# ---------------------------------------------------------------------------


class TestComputeForExperiments:
    def test_save_and_cache_hit(self, tmp_path: Path) -> None:
        treatments = _synthetic_treatments(K=3, n=200, seed=7)
        signal_dirs = _write_signal_dirs(tmp_path, treatments)
        config = PairwiseCompareConfig(n_bootstraps=200, seed=42)
        out_dir = tmp_path / "comparison_out"
        a1 = compute_pairwise_compare_for_experiments(
            signal_dirs, output_dir=out_dir, config=config, skip_if_exists=True
        )
        out_path = out_dir / "pairwise_compare_v1.json"
        assert out_path.exists()
        # Second call: cache hit
        a2 = compute_pairwise_compare_for_experiments(
            signal_dirs, output_dir=out_dir, config=config, skip_if_exists=True
        )
        assert a1 == a2

    def test_round_trip_via_save_load(self, tmp_path: Path) -> None:
        treatments = _synthetic_treatments(K=3, n=200, seed=7)
        signal_dirs = _write_signal_dirs(tmp_path, treatments)
        config = PairwiseCompareConfig(n_bootstraps=200, seed=42)
        out_dir = tmp_path / "out"
        a1 = compute_pairwise_compare_for_experiments(
            signal_dirs, output_dir=out_dir, config=config, skip_if_exists=False
        )
        a2 = PairwiseCompareArtifact.load(out_dir / "pairwise_compare_v1.json")
        assert a1 == a2
        assert a1.content_hash() == a2.content_hash()

    def test_recompute_on_n_bootstraps_drift(self, tmp_path: Path) -> None:
        treatments = _synthetic_treatments(K=2, n=200, seed=7)
        signal_dirs = _write_signal_dirs(tmp_path, treatments)
        out_dir = tmp_path / "out"
        config_a = PairwiseCompareConfig(n_bootstraps=200, seed=42)
        a1 = compute_pairwise_compare_for_experiments(
            signal_dirs, output_dir=out_dir, config=config_a, skip_if_exists=True
        )
        assert a1.n_bootstraps == 200
        # n_bootstraps drift
        config_b = PairwiseCompareConfig(n_bootstraps=500, seed=42)
        a2 = compute_pairwise_compare_for_experiments(
            signal_dirs, output_dir=out_dir, config=config_b, skip_if_exists=True
        )
        assert a2.n_bootstraps == 500


# ---------------------------------------------------------------------------
# Round 1+2 hardenings (block_length>=n guard, etc.)
# ---------------------------------------------------------------------------


class TestRound1Hardenings:
    def test_block_length_ge_n_explicit_raises(self) -> None:
        treatments = _synthetic_treatments(K=2, n=20, seed=7)
        # block_length=20 >= n_samples=20 → degenerate single-block bootstrap
        config = PairwiseCompareConfig(n_bootstraps=200, seed=42, block_length=20)
        parent = [{"compatibility_fingerprint": "a" * 64, "experiment_id": f"T{k}"}
                  for k in range(2)]
        with pytest.raises(ValueError, match="degenerate"):
            compare_k_way(treatments, config, parent_metadata=parent)

    def test_max_drop_frac_violation_raises(self, tmp_path: Path) -> None:
        """Per §8 fail-loud: too many NaN rows → raise."""
        # Synthesize 200 paired rows but inject 50 NaN into predictions of T0
        rng = np.random.default_rng(7)
        labels = rng.standard_normal(200).astype(np.float64)
        preds_a = rng.standard_normal(200).astype(np.float64)
        preds_a[:50] = np.nan  # 25% drop
        preds_b = rng.standard_normal(200).astype(np.float64)
        treatments = [("A", preds_a, labels), ("B", preds_b, labels)]
        config = PairwiseCompareConfig(
            n_bootstraps=200, seed=42, max_drop_frac=0.05  # 5% threshold
        )
        parent = [{"compatibility_fingerprint": "a" * 64, "experiment_id": l}
                  for l in ["A", "B"]]
        with pytest.raises(ValueError, match="max_drop_frac"):
            compare_k_way(treatments, config, parent_metadata=parent)
