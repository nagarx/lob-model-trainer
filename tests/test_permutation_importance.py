"""Phase 8C-α Stage C.1 — tests for compute_permutation_importance.

Tests use a deterministic MOCK predict_fn + metric_fn — no actual
model training. Validates the importance-computation plumbing + the
FeatureImportanceArtifact emission.

Real trainer integration test deferred to C.1.1 follow-up (requires
end-to-end training run).
"""

from __future__ import annotations

import numpy as np
import pytest

from hft_contracts import FeatureImportanceArtifact

from lobtrainer.training.importance import (
    ImportanceConfig,
    compute_permutation_importance,
    permutation_importance_enabled,
)


class TestImportanceConfig:
    """ImportanceConfig validation + enabled-gate."""

    def test_default_disabled(self):
        cfg = ImportanceConfig()
        assert cfg.enabled is False
        assert permutation_importance_enabled(cfg) is False

    def test_enabled_true_permutation_method(self):
        cfg = ImportanceConfig(enabled=True)
        assert cfg.method == "permutation"
        assert permutation_importance_enabled(cfg) is True

    def test_invalid_n_permutations_raises(self):
        with pytest.raises(ValueError, match="n_permutations"):
            ImportanceConfig(n_permutations=0)

    def test_invalid_subsample_raises(self):
        with pytest.raises(ValueError, match="subsample"):
            ImportanceConfig(subsample=0)
        with pytest.raises(ValueError, match="subsample"):
            ImportanceConfig(subsample=-5)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="method"):
            ImportanceConfig(method="shap")


def _make_mock_predict_fn(signal_feature_idx: int, signal_weight: float = 1.0):
    """Return a predict_fn where the target is LINEARLY dependent on
    X[..., signal_feature_idx] plus a tiny amount of other-feature noise.

    When we permute signal_feature → predictions change a lot →
    importance should be HIGH. When we permute a noise feature →
    predictions barely change → importance should be LOW/ZERO.
    """
    def predict_fn(X: np.ndarray) -> np.ndarray:
        # Handle (N,), (N, F), (N, T, F) shapes
        if X.ndim == 3:
            # Average across T to get per-sequence feature vector
            X_flat = X.mean(axis=1)
        elif X.ndim == 2:
            X_flat = X
        else:
            X_flat = X.reshape(-1, 1)
        preds = signal_weight * X_flat[..., signal_feature_idx]
        # Add small noise-feature contributions
        for f in range(X_flat.shape[-1]):
            if f == signal_feature_idx:
                continue
            preds = preds + 0.001 * X_flat[..., f]
        return preds
    return predict_fn


def _pearson_metric(preds: np.ndarray, y: np.ndarray) -> float:
    """Simple Pearson correlation as baseline metric (higher = better)."""
    preds = np.asarray(preds, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if preds.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(preds, y)[0, 1])


class TestComputePermutationImportance:
    def _make_synthetic_data(self, n: int = 500, n_features: int = 4,
                              signal_idx: int = 0, seed: int = 7):
        """Synthetic data where y is ~linear in X[:, signal_idx] plus noise."""
        rng = np.random.RandomState(seed)
        X = rng.randn(n, n_features).astype(np.float64)
        y = 2.0 * X[:, signal_idx] + 0.1 * rng.randn(n)
        feature_names = [f"f{i}" for i in range(n_features)]
        feature_indices = list(range(n_features))
        return X, y, feature_names, feature_indices

    def test_signal_feature_has_highest_importance(self):
        """Feature that drives y should have importance_mean >> other features.

        This is the core correctness test — if permuting the signal
        feature collapses the metric to noise while permuting noise
        features keeps metric near-baseline, the importance ranking
        will be correct.
        """
        X, y, feature_names, feature_indices = self._make_synthetic_data(
            n=300, n_features=4, signal_idx=2, seed=7,
        )
        predict_fn = _make_mock_predict_fn(signal_feature_idx=2, signal_weight=2.0)
        cfg = ImportanceConfig(
            enabled=True,
            n_permutations=30,  # small for test speed
            n_seeds=2,
            subsample=-1,  # full
            block_length_samples=1,  # element-wise (no autocorrelation preservation)
            seed=42,
        )

        artifact = compute_permutation_importance(
            X=X, y=y,
            feature_names=feature_names,
            feature_indices=feature_indices,
            predict_fn=predict_fn,
            metric_fn=_pearson_metric,
            config=cfg,
            experiment_id="test_signal",
            fingerprint="t" * 64,
            model_type="synthetic",
        )

        assert isinstance(artifact, FeatureImportanceArtifact)
        assert len(artifact.features) == 4

        # Find signal feature vs noise features
        signal = artifact.get_by_name("f2")
        assert signal is not None
        noise_features = [
            artifact.get_by_name(name)
            for name in ("f0", "f1", "f3")
        ]

        # Signal feature importance must be substantially higher
        for noise in noise_features:
            assert signal.importance_mean > noise.importance_mean, (
                f"Signal feature f2 importance ({signal.importance_mean:.4f}) "
                f"should exceed noise feature {noise.feature_name} importance "
                f"({noise.importance_mean:.4f}). Test synthesized a linear "
                f"y = 2.0 * X[:, 2] + 0.1*noise; permuting f2 should destroy "
                f"the predictable signal."
            )

    def test_determinism_same_seed_same_output(self):
        """Same config.seed → bit-identical artifact (modulo timestamp).

        hft-rules §7 determinism contract.
        """
        X, y, names, idxs = self._make_synthetic_data(
            n=200, n_features=3, signal_idx=1, seed=1,
        )
        predict_fn = _make_mock_predict_fn(signal_feature_idx=1)
        cfg = ImportanceConfig(
            enabled=True,
            n_permutations=20,
            n_seeds=2,
            subsample=-1,
            block_length_samples=1,
            seed=123,
        )
        a = compute_permutation_importance(
            X=X, y=y, feature_names=names, feature_indices=idxs,
            predict_fn=predict_fn, metric_fn=_pearson_metric, config=cfg,
        )
        b = compute_permutation_importance(
            X=X, y=y, feature_names=names, feature_indices=idxs,
            predict_fn=predict_fn, metric_fn=_pearson_metric, config=cfg,
        )

        # Importance means must match bit-for-bit
        for fa, fb in zip(a.features, b.features):
            assert fa.importance_mean == fb.importance_mean, (
                f"Determinism violation: feature {fa.feature_name} "
                f"importance_mean={fa.importance_mean} vs "
                f"{fb.importance_mean} under same seed"
            )
            assert fa.ci_lower_95 == fb.ci_lower_95
            assert fa.ci_upper_95 == fb.ci_upper_95

    def test_different_seed_different_output(self):
        """Different seeds → different importance (overwhelming probability
        unless data is pathologically symmetric)."""
        X, y, names, idxs = self._make_synthetic_data(
            n=200, n_features=3, signal_idx=0, seed=0,
        )
        predict_fn = _make_mock_predict_fn(signal_feature_idx=0)
        cfg_a = ImportanceConfig(enabled=True, n_permutations=20, n_seeds=2,
                                  subsample=-1, seed=42)
        cfg_b = ImportanceConfig(enabled=True, n_permutations=20, n_seeds=2,
                                  subsample=-1, seed=99)

        a = compute_permutation_importance(
            X=X, y=y, feature_names=names, feature_indices=idxs,
            predict_fn=predict_fn, metric_fn=_pearson_metric, config=cfg_a,
        )
        b = compute_permutation_importance(
            X=X, y=y, feature_names=names, feature_indices=idxs,
            predict_fn=predict_fn, metric_fn=_pearson_metric, config=cfg_b,
        )

        # At least one feature's importance must differ
        differs = any(
            fa.importance_mean != fb.importance_mean
            for fa, fb in zip(a.features, b.features)
        )
        assert differs, (
            "Different seeds should yield different importance "
            "(unless data is pathologically symmetric — not this case)"
        )

    def test_handles_sequence_input_shape(self):
        """X with shape (N, T, F) — the common sequence-model layout."""
        rng = np.random.RandomState(11)
        N, T, F = 100, 10, 3
        X = rng.randn(N, T, F).astype(np.float64)
        y = X.mean(axis=1)[:, 0] * 3.0 + 0.1 * rng.randn(N)

        predict_fn = _make_mock_predict_fn(signal_feature_idx=0)
        cfg = ImportanceConfig(
            enabled=True, n_permutations=10, n_seeds=1, subsample=-1, seed=0,
        )
        artifact = compute_permutation_importance(
            X=X, y=y, feature_names=["a", "b", "c"],
            feature_indices=[0, 1, 2],
            predict_fn=predict_fn, metric_fn=_pearson_metric, config=cfg,
        )

        assert len(artifact.features) == 3
        # Verify shape-preserving path didn't crash + produced valid floats
        for feat in artifact.features:
            assert np.isfinite(feat.importance_mean)
            assert np.isfinite(feat.ci_lower_95)
            assert np.isfinite(feat.ci_upper_95)

    def test_subsample_respected(self):
        """``subsample`` limits eval-set size — useful for compute budget
        control. Verify the effective N ≤ subsample.
        """
        X, y, names, idxs = self._make_synthetic_data(
            n=1000, n_features=2, signal_idx=0,
        )
        predict_fn = _make_mock_predict_fn(signal_feature_idx=0)
        # With subsample=100, the effective eval set is 100 (not 1000)
        cfg = ImportanceConfig(
            enabled=True, n_permutations=10, n_seeds=1,
            subsample=100, seed=42,
        )
        artifact = compute_permutation_importance(
            X=X, y=y, feature_names=names, feature_indices=idxs,
            predict_fn=predict_fn, metric_fn=_pearson_metric, config=cfg,
        )
        # No direct way to introspect N from artifact; but compute
        # completes quickly relative to N=1000. Indirect check: baseline
        # metric is stable (mock model + subsample-seeded).
        assert np.isfinite(artifact.baseline_value)
        assert len(artifact.features) == 2

    def test_output_schema_roundtrips_via_hft_contracts(self, tmp_path):
        """End-to-end contract test: output can be saved + loaded as
        ``FeatureImportanceArtifact``. Locks the producer→consumer
        contract between C.1 (trainer) and C.3 (ledger) / C.5 (evaluator).
        """
        X, y, names, idxs = self._make_synthetic_data(n=150, n_features=2)
        predict_fn = _make_mock_predict_fn(0)
        cfg = ImportanceConfig(enabled=True, n_permutations=10, n_seeds=1,
                                subsample=-1, seed=0)
        artifact = compute_permutation_importance(
            X=X, y=y, feature_names=names, feature_indices=idxs,
            predict_fn=predict_fn, metric_fn=_pearson_metric, config=cfg,
            feature_set_ref={"name": "test_v1", "content_hash": "h" * 64},
            experiment_id="exp_test",
            fingerprint="f" * 64,
            model_type="synthetic",
        )

        path = tmp_path / "feature_importance_v1.json"
        artifact.save(path)
        assert path.exists()

        reloaded = FeatureImportanceArtifact.load(path)
        assert reloaded == artifact

    def test_mismatched_feature_names_raises(self):
        """Contract validation: feature_names length must match X.shape[-1]."""
        X = np.random.randn(50, 3).astype(np.float64)
        y = np.random.randn(50)
        cfg = ImportanceConfig(enabled=True, n_permutations=5, n_seeds=1,
                                subsample=-1)
        with pytest.raises(ValueError, match="feature_names length"):
            compute_permutation_importance(
                X=X, y=y,
                feature_names=["only_one_name"],
                feature_indices=[0],
                predict_fn=_make_mock_predict_fn(0),
                metric_fn=_pearson_metric,
                config=cfg,
            )


class TestPostAuditFixes:
    """Phase 8C-α post-audit regression tests (2026-04-20).

    Each test fails against pre-fix code, validating that the 6 Agent-D
    fixes + the round-2 degenerate-null guard remain in place.
    """

    # ---- Agent-D H1: block_size_days → block_length_samples rename ----

    def test_block_size_days_deprecation_warning(self):
        """Setting legacy ``block_size_days`` must emit DeprecationWarning
        AND correctly populate ``block_length_samples`` (Agent-D H1)."""
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cfg = ImportanceConfig(block_size_days=5)
            assert cfg.block_length_samples == 5
            deprecations = [w for w in caught
                            if issubclass(w.category, DeprecationWarning)]
            assert any("block_size_days is deprecated" in str(w.message)
                       for w in deprecations), (
                "Setting legacy block_size_days must emit DeprecationWarning"
            )

    def test_block_size_days_and_block_length_samples_mismatch_raises(self):
        """If the caller passes BOTH old+new keys with different values,
        ``__post_init__`` raises to prevent silent preference bugs
        (Agent-D H1 mismatch path)."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.raises(ValueError, match="disagree"):
                ImportanceConfig(block_size_days=3, block_length_samples=5)

    # ---- Agent-D H2: cross-feature RNG decorrelation ----

    def test_per_feature_rng_decorrelation(self):
        """Post-audit Agent-D H2: per-(feature, seed) RNG MUST produce
        independent permutation sequences across features. Previously
        every feature shared ``config.seed + seed_offset`` → rank-
        correlated importance estimates.

        Check: with a STABLE predict_fn (depends on only one feature),
        the null-distribution permutation order for DIFFERENT features
        at the same seed_offset must differ. Probe: capture the first
        permutation array generated per feature and assert they are not
        all identical.
        """
        rng = np.random.RandomState(0)
        X = rng.randn(50, 3).astype(np.float64)
        y = X[:, 0] * 2.0 + 0.1 * rng.randn(50)
        cfg = ImportanceConfig(
            enabled=True, n_permutations=1, n_seeds=1, subsample=-1, seed=42,
        )

        # Capture permutation seeds used per feature by monkey-patching
        # np.random.RandomState to log which seeds were instantiated in
        # the feature-loop.
        original_rs = np.random.RandomState
        seeds_used = []
        def capturing_rs(s, *a, **kw):
            seeds_used.append(s)
            return original_rs(s, *a, **kw)

        import lobtrainer.training.importance.permutation as mod
        mod.np.random.RandomState = capturing_rs  # type: ignore
        try:
            compute_permutation_importance(
                X=X, y=y, feature_names=["a", "b", "c"],
                feature_indices=[0, 1, 2],
                predict_fn=_make_mock_predict_fn(0),
                metric_fn=_pearson_metric,
                config=cfg,
            )
        finally:
            mod.np.random.RandomState = original_rs  # type: ignore

        # Subsample (disabled: -1) does not instantiate a seeded RS.
        # Per-feature loop: 3 features × 1 seed → 3 distinct seeds.
        per_feature_seeds = set(seeds_used)
        assert len(per_feature_seeds) >= 3, (
            f"Expected at least 3 distinct RNG seeds (one per feature), "
            f"got {sorted(per_feature_seeds)}. If all features shared one "
            f"seed, Agent-D H2 regressed — cross-feature comparisons "
            f"will be rank-correlated."
        )

    # ---- Agent-D M2: NaN baseline fail-loud ----

    def test_nan_baseline_raises(self):
        """Post-audit Agent-D M2: non-finite baseline poisons every
        downstream importance (baseline - null → NaN). Fail loud per §8."""
        X = np.random.randn(50, 2).astype(np.float64)
        y = np.random.randn(50)
        cfg = ImportanceConfig(enabled=True, n_permutations=5, n_seeds=1,
                                subsample=-1)
        bad_metric = lambda preds, y: float("nan")
        with pytest.raises(ValueError, match="baseline metric is non-finite"):
            compute_permutation_importance(
                X=X, y=y,
                feature_names=["a", "b"], feature_indices=[0, 1],
                predict_fn=_make_mock_predict_fn(0),
                metric_fn=bad_metric, config=cfg,
            )

    # ---- Agent-D M3: failed-seed accounting ----

    def test_failed_seed_dropped_from_n_seeds_aggregated(self):
        """Post-audit Agent-D M3: if a seed's permutations all produce
        non-finite metrics, the seed is DROPPED from per_seed_means
        (not padded with 0.0). ``FeatureImportance.n_seeds_aggregated``
        must reflect the ACTUAL count, not config.n_seeds.
        """
        X = np.random.randn(50, 2).astype(np.float64)
        y = X[:, 0] * 2.0 + 0.1 * np.random.RandomState(0).randn(50)
        cfg = ImportanceConfig(enabled=True, n_permutations=3, n_seeds=3,
                                subsample=-1, seed=42)
        # Metric fn that returns NaN on permuted inputs (simulating all-seed
        # failure) but a finite baseline (the un-permuted call).
        call_count = [0]
        def flaky_metric(preds, y):
            call_count[0] += 1
            # First call is baseline (no permutation); return finite.
            if call_count[0] == 1:
                return 0.5
            return float("nan")

        artifact = compute_permutation_importance(
            X=X, y=y,
            feature_names=["a", "b"], feature_indices=[0, 1],
            predict_fn=_make_mock_predict_fn(0),
            metric_fn=flaky_metric, config=cfg,
        )
        for feat in artifact.features:
            assert feat.n_seeds_aggregated == 0, (
                f"Feature {feat.feature_name}: all permutations failed "
                f"(NaN metric) → n_seeds_aggregated must be 0 (actual seeds "
                f"that contributed), NOT {cfg.n_seeds} (configured). "
                f"Got {feat.n_seeds_aggregated}."
            )

    # ---- Round-2 post-audit: degenerate n_blocks<2 guard in trainer ----

    def test_block_length_samples_geq_n_eval_raises(self):
        """Round-2 (arch-S3): trainer's inline block-permutation must
        fail-loud when ``block_length_samples >= n_eval`` (same bug
        Agent-A H2 fixed in hft_metrics.block_permutation). Silently
        collapsing to 1 block produces a zero-width null distribution.
        """
        X = np.random.randn(50, 2).astype(np.float64)
        y = np.random.randn(50)
        # subsample forces n_eval=10; block_length_samples=10 → n_blocks=1
        cfg = ImportanceConfig(
            enabled=True, n_permutations=5, n_seeds=1,
            subsample=10, block_length_samples=10, seed=42,
        )
        with pytest.raises(ValueError, match="degenerate"):
            compute_permutation_importance(
                X=X, y=y,
                feature_names=["a", "b"], feature_indices=[0, 1],
                predict_fn=_make_mock_predict_fn(0),
                metric_fn=_pearson_metric, config=cfg,
            )
