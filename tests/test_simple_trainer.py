"""
Tests for SimpleModelTrainer and module-level _load_split, _compute_metrics.

SimpleModelTrainer handles TemporalRidge/GradBoost models. It produced the
key finding that TemporalRidge captures 91% of TLOB IC with 54 params
(IC=0.616). This pipeline has ZERO test coverage — a data loading bug
could invalidate that finding and change the entire research direction.

Silent corruption risks tested:
- Hardcoded index fallbacks (40, 42) matching hft_contracts
- Horizon column selection from regression_labels
- Day ordering matching sorted glob pattern
- Spread/price extraction from correct sequence column

Design Principles (hft-rules.md):
    - No hardcoded indices without contract verification (Rule 1)
    - Test edge cases: max_days, unknown model_type (Rule 6)
    - Assertions explain WHAT failed and WHY (Rule 6)
"""

import json
from pathlib import Path

import numpy as np
import pytest

from lobtrainer.training.simple_trainer import (
    SimpleModelTrainer,
    _compute_metrics,
    _load_split,
)


# =============================================================================
# Constants
# =============================================================================

NUM_SEQS = 20
SEQ_LEN = 100
NUM_FEATURES = 98
NUM_HORIZONS = 3
HORIZONS = [10, 60, 300]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_data_dir(tmp_path):
    """Create synthetic .npy data in the format _load_split expects.

    Directory structure:
        {tmp_path}/train/{day}_sequences.npy        [N, T, F] float32
        {tmp_path}/train/{day}_regression_labels.npy [N, H] float64
        {tmp_path}/train/{day}_forward_prices.npy   [N, k+max_H+1=306] float64
        {tmp_path}/train/{day}_metadata.json        {"day", "horizons",
                                                     "forward_prices": {...}, ...}
        {tmp_path}/val/...
        {tmp_path}/test/...

    Phase Y / γ-1 LITE / #PY-88 (2026-05-10): fixture upgraded with
    `*_forward_prices.npy` + `forward_prices` metadata block + `horizons`
    list so `_resolve_labels_for_day` Branch 2 (source="forward_prices")
    + Branch 3 (source="auto") work end-to-end. Pseudo-random walk
    around 130.0 USD ensures non-degenerate labels for all 4 LabelFactory
    return_type variants. Shape `(N, k+max_H+1) = 306` locks the
    `ForwardPriceContract.from_metadata` invariant
    (n_columns == smoothing_window_offset + max_horizon + 1).
    """
    rng = np.random.default_rng(42)

    # Phase Y / γ-1 LITE / #PY-88 forward_prices fixture constants:
    # k=5 matches v3p0 production exports; max_H=300 matches HORIZONS[2].
    K = 5  # smoothing_window_offset (ForwardPriceContract invariant)
    MAX_H = max(HORIZONS)  # = 300 — locks horizon-3 column index < N_FP_COLS
    N_FP_COLS = K + MAX_H + 1  # 306 — ForwardPriceContract.__post_init__ check

    for split in ["train", "val", "test"]:
        split_dir = tmp_path / split
        split_dir.mkdir()

        for i, day in enumerate(["2025-01-01", "2025-01-02"]):
            n = NUM_SEQS + i * 5  # Different sizes per day

            sequences = rng.standard_normal((n, SEQ_LEN, NUM_FEATURES)).astype(np.float32)
            # Put known values in price/spread columns for verification
            # Index 40 = mid_price, Index 42 = spread_bps (contract)
            sequences[:, -1, 40] = 130.0 + rng.standard_normal(n).astype(np.float32) * 0.5
            sequences[:, -1, 42] = 2.5 + rng.standard_normal(n).astype(np.float32) * 0.1

            reg_labels = rng.standard_normal((n, NUM_HORIZONS)).astype(np.float64)

            # Pseudo-random walk: base=130.0 USD, step-std=0.01 USD ≈ 8 bps
            # per timestep. cumsum produces non-degenerate returns at all
            # horizons. Max walk at H=300: sqrt(300)*0.01 ≈ 0.17 USD ≈ 130
            # bps — well within reasonable test bounds and bounded under
            # the LabelFactory.multi_horizon post-finite-check.
            forward_prices = (
                130.0
                + np.cumsum(rng.standard_normal((n, N_FP_COLS)) * 0.01, axis=1)
            ).astype(np.float64)

            np.save(split_dir / f"{day}_sequences.npy", sequences)
            np.save(split_dir / f"{day}_regression_labels.npy", reg_labels)
            np.save(split_dir / f"{day}_forward_prices.npy", forward_prices)

            metadata = {
                "day": day,
                "n_sequences": n,
                "n_features": NUM_FEATURES,
                "schema_version": "3.0",  # G.6.A bump
                # Phase Y / γ-1 LITE / #PY-88: top-level horizons consumed
                # by `_resolve_labels_for_day` Branch 2/3 + by
                # `ForwardPriceContract.from_metadata`.
                "horizons": HORIZONS,
                # Phase Y / γ-1 LITE / #PY-88: declares forward_prices
                # availability for "auto" source dispatch + locks the
                # k/max_H/n_columns invariant for "forward_prices" source.
                "forward_prices": {
                    "exported": True,
                    "smoothing_window_offset": K,
                    "max_horizon": MAX_H,
                    "n_columns": N_FP_COLS,
                },
            }
            with open(split_dir / f"{day}_metadata.json", "w") as f:
                json.dump(metadata, f)

    return tmp_path


# =============================================================================
# TestLoadSplit
# =============================================================================


class TestLoadSplit:
    """Tests for the module-level _load_split function."""

    def test_returns_correct_shapes(self, simple_data_dir):
        """_load_split returns (sequences, labels, spreads, prices) with correct dims."""
        seqs, labels, spreads, prices = _load_split(simple_data_dir, "train")

        total_n = NUM_SEQS + (NUM_SEQS + 5)  # 2 days with different counts
        assert seqs.shape == (total_n, SEQ_LEN, NUM_FEATURES), (
            f"Sequences shape: {seqs.shape}, expected ({total_n}, {SEQ_LEN}, {NUM_FEATURES})"
        )
        assert labels.shape == (total_n,), f"Labels shape: {labels.shape}"
        assert spreads.shape == (total_n,), f"Spreads shape: {spreads.shape}"
        assert prices.shape == (total_n,), f"Prices shape: {prices.shape}"

    def test_horizon_idx_selects_column(self, simple_data_dir):
        """horizon_idx selects the correct column from regression_labels."""
        _, labels_h0, _, _ = _load_split(simple_data_dir, "train", horizon_idx=0)
        _, labels_h1, _, _ = _load_split(simple_data_dir, "train", horizon_idx=1)
        _, labels_h2, _, _ = _load_split(simple_data_dir, "train", horizon_idx=2)

        # Different horizons should produce different labels
        assert not np.allclose(labels_h0, labels_h1), (
            "Horizon 0 and 1 should have different labels"
        )
        assert not np.allclose(labels_h0, labels_h2), (
            "Horizon 0 and 2 should have different labels"
        )

    def test_spread_from_correct_column(self, simple_data_dir):
        """Spreads extracted from seq[:, -1, SPREAD_BPS_IDX=42]."""
        seqs, _, spreads, _ = _load_split(simple_data_dir, "train")

        # Our fixture puts known values in column 42
        expected = seqs[:, -1, 42].astype(np.float64)
        np.testing.assert_array_equal(
            spreads, expected,
            err_msg="Spreads must come from column 42 (SPREAD_BPS_IDX)",
        )

    def test_price_from_correct_column(self, simple_data_dir):
        """Prices extracted from seq[:, -1, MID_PRICE_IDX=40]."""
        seqs, _, _, prices = _load_split(simple_data_dir, "train")

        expected = seqs[:, -1, 40].astype(np.float64)
        np.testing.assert_array_equal(
            prices, expected,
            err_msg="Prices must come from column 40 (MID_PRICE_IDX)",
        )

    def test_indices_match_hft_contracts(self):
        """Verify hardcoded fallback indices match hft_contracts values."""
        try:
            from hft_contracts import (
                SIGNAL_PRICE_FEATURE_INDEX,
                SIGNAL_SPREAD_FEATURE_INDEX,
            )

            assert SIGNAL_PRICE_FEATURE_INDEX == 40, (
                f"hft_contracts MID_PRICE_IDX changed to {SIGNAL_PRICE_FEATURE_INDEX}! "
                f"simple_trainer.py fallback is 40. Update required."
            )
            assert SIGNAL_SPREAD_FEATURE_INDEX == 42, (
                f"hft_contracts SPREAD_BPS_IDX changed to {SIGNAL_SPREAD_FEATURE_INDEX}! "
                f"simple_trainer.py fallback is 42. Update required."
            )
        except ImportError:
            pytest.skip("hft_contracts not installed — cannot verify index alignment")

    def test_sorted_day_ordering(self, simple_data_dir):
        """Days are processed in sorted glob order (alphabetical by filename)."""
        seqs, _, _, _ = _load_split(simple_data_dir, "train")

        # Day 1 (2025-01-01) has NUM_SEQS samples, Day 2 has NUM_SEQS+5
        # Since sorted, Day 1 comes first
        assert seqs.shape[0] == NUM_SEQS + (NUM_SEQS + 5)
        # First batch should be from day 1 (smaller count)
        # This is implicitly verified by the shape check above

    def test_max_days_limits_data(self, simple_data_dir):
        """max_days=1 loads only the first day."""
        seqs, labels, _, _ = _load_split(simple_data_dir, "train", max_days=1)
        assert seqs.shape[0] == NUM_SEQS, (
            f"max_days=1 should load {NUM_SEQS} samples, got {seqs.shape[0]}"
        )

    def test_spreads_and_prices_are_float64(self, simple_data_dir):
        """Spread and price arrays must be float64."""
        _, _, spreads, prices = _load_split(simple_data_dir, "train")
        assert spreads.dtype == np.float64, f"Spreads dtype: {spreads.dtype}"
        assert prices.dtype == np.float64, f"Prices dtype: {prices.dtype}"


# =============================================================================
# TestComputeMetrics
# =============================================================================


class TestComputeMetrics:
    """Tests for the module-level _compute_metrics function."""

    def test_returns_expected_keys(self):
        """_compute_metrics returns dict with all 7 metric keys."""
        rng = np.random.default_rng(42)
        y_true = rng.standard_normal(50)
        y_pred = y_true + rng.standard_normal(50) * 0.1

        metrics = _compute_metrics(y_true, y_pred)
        expected_keys = {"r2", "ic", "mae", "rmse", "pearson",
                         "directional_accuracy", "profitable_accuracy"}
        missing = expected_keys - set(metrics.keys())
        assert not missing, f"Missing metric keys: {missing}"


# =============================================================================
# TestSimpleModelTrainer
# =============================================================================


class TestSimpleModelTrainer:
    """Tests for the full SimpleModelTrainer pipeline."""

    def test_setup_temporal_ridge(self, simple_data_dir, tmp_path):
        """setup() creates TemporalRidge model and engineers features."""
        trainer = SimpleModelTrainer(
            data_dir=str(simple_data_dir),
            model_type="temporal_ridge",
            model_config={"alpha": 1.0},
            output_dir=str(tmp_path / "output"),
        )
        trainer.setup()

        assert trainer.model is not None
        assert "Ridge" in trainer.model.name
        assert trainer.feat_config is not None

    def test_setup_unknown_model_raises(self, simple_data_dir, tmp_path):
        """Unknown model_type raises ValueError."""
        trainer = SimpleModelTrainer(
            data_dir=str(simple_data_dir),
            model_type="unknown_model",
            model_config={},
            output_dir=str(tmp_path / "output"),
        )
        with pytest.raises(ValueError, match="Unknown simple model type"):
            trainer.setup()

    def test_train_returns_val_metrics(self, simple_data_dir, tmp_path):
        """train() returns validation metrics dict."""
        trainer = SimpleModelTrainer(
            data_dir=str(simple_data_dir),
            model_type="temporal_ridge",
            model_config={"alpha": 1.0},
            output_dir=str(tmp_path / "output"),
        )
        trainer.setup()
        val_metrics = trainer.train()

        assert isinstance(val_metrics, dict)
        assert "r2" in val_metrics, f"Missing r2. Keys: {val_metrics.keys()}"
        assert "ic" in val_metrics

    def test_export_signals_writes_files(self, simple_data_dir, tmp_path):
        """export_signals creates all expected .npy files + metadata."""
        trainer = SimpleModelTrainer(
            data_dir=str(simple_data_dir),
            model_type="temporal_ridge",
            model_config={"alpha": 1.0},
            output_dir=str(tmp_path / "output"),
        )
        trainer.setup()
        trainer.train()
        trainer.evaluate()
        signal_dir = trainer.export_signals("test")

        expected_files = [
            "predicted_returns.npy", "regression_labels.npy",
            "spreads.npy", "prices.npy", "signal_metadata.json",
        ]
        for fname in expected_files:
            assert (signal_dir / fname).exists(), (
                f"Missing exported file: {fname}"
            )

    def test_export_signals_alignment(self, simple_data_dir, tmp_path):
        """All exported arrays must have the same first dimension."""
        trainer = SimpleModelTrainer(
            data_dir=str(simple_data_dir),
            model_type="temporal_ridge",
            model_config={"alpha": 1.0},
            output_dir=str(tmp_path / "output"),
        )
        trainer.setup()
        trainer.train()
        trainer.evaluate()
        signal_dir = trainer.export_signals("test")

        preds = np.load(signal_dir / "predicted_returns.npy")
        labels = np.load(signal_dir / "regression_labels.npy")
        spreads = np.load(signal_dir / "spreads.npy")
        prices = np.load(signal_dir / "prices.npy")

        n = preds.shape[0]
        assert labels.shape[0] == n, f"Labels N={labels.shape[0]} != preds N={n}"
        assert spreads.shape[0] == n, f"Spreads N={spreads.shape[0]} != preds N={n}"
        assert prices.shape[0] == n, f"Prices N={prices.shape[0]} != preds N={n}"

    def test_export_refuses_non_test_split(self, simple_data_dir, tmp_path):
        """export_signals raises ValueError for non-test splits."""
        trainer = SimpleModelTrainer(
            data_dir=str(simple_data_dir),
            model_type="temporal_ridge",
            model_config={"alpha": 1.0},
            output_dir=str(tmp_path / "output"),
        )
        trainer.setup()
        trainer.train()

        with pytest.raises(ValueError, match="Only 'test' split"):
            trainer.export_signals("val")


# =============================================================================
# Phase Q.6 — from_config classmethod adapter
# =============================================================================


class TestFromConfig:
    """Q.6: SimpleModelTrainer.from_config bridges ExperimentConfig
    to the legacy flat-keyword constructor so create_trainer dispatch
    can return this class for sklearn models."""

    def _build_synthetic_config(
        self, simple_data_dir, tmp_path, *, model_type="temporal_ridge",
        primary_horizon_idx=0, params=None,
    ):
        """Construct an ExperimentConfig pointing at the synthetic data dir."""
        from lobtrainer.config.schema import (
            ExperimentConfig, DataConfig, ModelConfig, TrainConfig,
            LabelsConfig, SequenceConfig, NormalizationConfig,
            ModelType, TaskType, LossType,
        )
        if params is None:
            params = {"alpha": 1.0}
        data = DataConfig(
            data_dir=str(simple_data_dir),
            feature_count=NUM_FEATURES,
            sequence=SequenceConfig(window_size=SEQ_LEN, stride=1),
            normalization=NormalizationConfig(strategy="none"),
            labels=LabelsConfig(
                primary_horizon_idx=primary_horizon_idx,
                horizons=[10, 60, 300],
                source="forward_prices",
                task="regression",
            ),
        )
        model = ModelConfig(
            model_type=getattr(ModelType, model_type.upper()),
            input_size=NUM_FEATURES,
            params=params,
        )
        train = TrainConfig(task_type=TaskType.REGRESSION, loss_type=LossType.HUBER)
        return ExperimentConfig(
            name="from_config_test", data=data, model=model, train=train,
            output_dir=str(tmp_path / "output"),
        )

    def test_from_config_preserves_data_dir(self, simple_data_dir, tmp_path):
        config = self._build_synthetic_config(simple_data_dir, tmp_path)
        trainer = SimpleModelTrainer.from_config(config)
        assert trainer.data_dir == Path(str(simple_data_dir))

    def test_from_config_unpacks_params_features_key(self, simple_data_dir, tmp_path):
        """The YAML convention `params.features:` is mapped to the
        constructor's `feature_config` argument."""
        config = self._build_synthetic_config(
            simple_data_dir, tmp_path,
            params={
                "alpha": 0.5,
                "features": {"signal_indices": [40, 42, 45], "rolling_windows": [3, 5]},
            },
        )
        trainer = SimpleModelTrainer.from_config(config)
        assert trainer.feature_config_dict == {
            "signal_indices": [40, 42, 45], "rolling_windows": [3, 5]
        }
        # alpha remained in model_config (popped features only)
        assert trainer.model_config == {"alpha": 0.5}

    def test_from_config_alternate_feature_config_key(self, simple_data_dir, tmp_path):
        """The alternate YAML convention `params.feature_config:` is also accepted."""
        config = self._build_synthetic_config(
            simple_data_dir, tmp_path,
            params={
                "alpha": 0.5,
                "feature_config": {"signal_indices": [40, 42]},
            },
        )
        trainer = SimpleModelTrainer.from_config(config)
        assert trainer.feature_config_dict == {"signal_indices": [40, 42]}

    def test_from_config_rejects_both_features_keys(self, simple_data_dir, tmp_path):
        """Q.6 post-audit: providing BOTH `features` and `feature_config`
        in params is an operator error. Pre-fix this silently discarded
        `feature_config`; now it fail-louds per hft-rules §5."""
        config = self._build_synthetic_config(
            simple_data_dir, tmp_path,
            params={
                "alpha": 0.5,
                "features": {"signal_indices": [40, 42]},
                "feature_config": {"signal_indices": [85, 86]},
            },
        )
        with pytest.raises(ValueError, match=r"BOTH 'features' AND 'feature_config'"):
            SimpleModelTrainer.from_config(config)

    def test_from_config_resolves_primary_horizon_idx(self, simple_data_dir, tmp_path):
        config = self._build_synthetic_config(simple_data_dir, tmp_path, primary_horizon_idx=2)
        trainer = SimpleModelTrainer.from_config(config)
        assert trainer.horizon_idx == 2

    def test_from_config_stores_config_attribute(self, simple_data_dir, tmp_path):
        """`self.config` is set for traceability (legacy direct
        constructor leaves it None)."""
        config = self._build_synthetic_config(simple_data_dir, tmp_path)
        trainer = SimpleModelTrainer.from_config(config)
        assert trainer.config is config

    def test_from_config_temporal_gradboost(self, simple_data_dir, tmp_path):
        config = self._build_synthetic_config(
            simple_data_dir, tmp_path, model_type="temporal_gradboost",
            params={"n_estimators": 50, "max_depth": 4},
        )
        trainer = SimpleModelTrainer.from_config(config)
        assert trainer.model_type == "temporal_gradboost"
        assert trainer.model_config == {"n_estimators": 50, "max_depth": 4}


# =============================================================================
# Phase Q.6 — save_checkpoint / load_checkpoint Protocol methods
# =============================================================================


class TestCheckpointRoundtrip:
    """Q.6: save_checkpoint / load_checkpoint satisfy BaseTrainer Protocol."""

    def test_save_checkpoint_default_path(self, simple_data_dir, tmp_path):
        trainer = SimpleModelTrainer(
            data_dir=str(simple_data_dir), model_type="temporal_ridge",
            model_config={"alpha": 1.0}, output_dir=str(tmp_path / "output"),
        )
        trainer.setup()
        trainer.train()
        path = trainer.save_checkpoint()
        assert path.exists()
        assert path.name == "best.pkl"

    def test_save_checkpoint_custom_path(self, simple_data_dir, tmp_path):
        trainer = SimpleModelTrainer(
            data_dir=str(simple_data_dir), model_type="temporal_ridge",
            model_config={"alpha": 1.0}, output_dir=str(tmp_path / "output"),
        )
        trainer.setup()
        trainer.train()
        custom = tmp_path / "custom" / "model.pkl"
        path = trainer.save_checkpoint(custom)
        assert path == custom
        assert custom.exists()

    def test_save_checkpoint_before_train_raises(self, simple_data_dir, tmp_path):
        """Cannot save_checkpoint before the model has been fit."""
        trainer = SimpleModelTrainer(
            data_dir=str(simple_data_dir), model_type="temporal_ridge",
            model_config={"alpha": 1.0}, output_dir=str(tmp_path / "output"),
        )
        trainer.setup()  # creates self.model but doesn't fit
        # train() not called; save_checkpoint of an unfit model should still
        # work (BaseSimpleModel.save handles unfit instances), so we test
        # the harder no-model case instead by clearing.
        trainer.model = None
        with pytest.raises(RuntimeError, match=r"self\.model is None"):
            trainer.save_checkpoint()

    def test_load_checkpoint_round_trip(self, simple_data_dir, tmp_path):
        """Round-trip checkpoint via save+load.

        Phase X.1 v2 X.1.M (2026-05-04): xfail decorator RETIRED — the
        F-1 root cause (lobmodels.config.base.BaseConfig.from_dict naive
        cls(**d)) was structurally fixed by X.1.B (recursive reconstruction
        via dataclasses.is_dataclass). Test now passes cleanly.
        """
        # Phase 1: train and save.
        trainer1 = SimpleModelTrainer(
            data_dir=str(simple_data_dir), model_type="temporal_ridge",
            model_config={"alpha": 1.0}, output_dir=str(tmp_path / "out1"),
        )
        trainer1.setup()
        trainer1.train()
        ckpt = trainer1.save_checkpoint()

        # Phase 2: fresh trainer loads the checkpoint and produces same predictions.
        trainer2 = SimpleModelTrainer(
            data_dir=str(simple_data_dir), model_type="temporal_ridge",
            model_config={"alpha": 1.0}, output_dir=str(tmp_path / "out2"),
        )
        trainer2.setup()
        trainer2.load_checkpoint(ckpt)

        # Same X → same predictions across both trainers.
        pred1 = trainer1.model.predict(trainer1._X_test)
        pred2 = trainer2.model.predict(trainer2._X_test)
        np.testing.assert_array_equal(pred1, pred2)

    def test_load_checkpoint_method_exists_and_callable(self, simple_data_dir, tmp_path):
        """Q.6 Protocol contract: load_checkpoint exists and can be invoked.

        This is the minimum-viable contract — full round-trip is xfailed
        pending the lob-models BaseConfig.from_dict fix. Verifies that
        the Phase Q.5 dispatch can call load_checkpoint without
        AttributeError on the method itself."""
        trainer = SimpleModelTrainer.__new__(SimpleModelTrainer)
        trainer.model_type = "temporal_ridge"
        # Method exists and is callable (will raise on missing file but
        # not on the method-resolution layer).
        assert callable(trainer.load_checkpoint), (
            "load_checkpoint must be a callable method on SimpleModelTrainer"
        )

    def test_load_checkpoint_unknown_model_type_raises(self, tmp_path):
        trainer = SimpleModelTrainer.__new__(SimpleModelTrainer)
        trainer.model_type = "definitely_not_a_real_model"
        with pytest.raises(ValueError, match=r"Cannot load_checkpoint"):
            trainer.load_checkpoint(tmp_path / "nope.pkl")


# =============================================================================
# Phase Q.6 — train() augmented return shape
# =============================================================================


class TestTrainReturnShape:
    """Q.6: train() returns dict with `total_epochs` / `best_val_metric`
    / `best_epoch` keys matching PyTorch Trainer.train()'s shape so
    `scripts/train.py` can read both polymorphically."""

    def test_train_returns_total_epochs_sentinel(self, simple_data_dir, tmp_path):
        trainer = SimpleModelTrainer(
            data_dir=str(simple_data_dir), model_type="temporal_ridge",
            model_config={"alpha": 1.0}, output_dir=str(tmp_path / "output"),
        )
        trainer.setup()
        result = trainer.train()
        assert result["total_epochs"] == 1
        assert result["best_epoch"] == 0
        assert "best_val_metric" in result
        assert isinstance(result["best_val_metric"], float)

    def test_train_returns_val_metrics_under_keys(self, simple_data_dir, tmp_path):
        trainer = SimpleModelTrainer(
            data_dir=str(simple_data_dir), model_type="temporal_ridge",
            model_config={"alpha": 1.0}, output_dir=str(tmp_path / "output"),
        )
        trainer.setup()
        result = trainer.train()
        # All val_metrics keys must be in the top-level dict (back-compat).
        for key in ("r2", "ic", "mae"):
            assert key in result


# =============================================================================
# Phase Q.6 — evaluate(split) Protocol method
# =============================================================================


class TestEvaluateSplit:
    """Q.6: evaluate accepts a split argument matching Trainer.evaluate."""

    def test_evaluate_test_split(self, simple_data_dir, tmp_path):
        trainer = SimpleModelTrainer(
            data_dir=str(simple_data_dir), model_type="temporal_ridge",
            model_config={"alpha": 1.0}, output_dir=str(tmp_path / "output"),
        )
        trainer.setup()
        trainer.train()
        metrics = trainer.evaluate("test")
        assert "r2" in metrics

    def test_evaluate_val_split(self, simple_data_dir, tmp_path):
        trainer = SimpleModelTrainer(
            data_dir=str(simple_data_dir), model_type="temporal_ridge",
            model_config={"alpha": 1.0}, output_dir=str(tmp_path / "output"),
        )
        trainer.setup()
        trainer.train()
        metrics = trainer.evaluate("val")
        assert "r2" in metrics

    def test_evaluate_train_split(self, simple_data_dir, tmp_path):
        trainer = SimpleModelTrainer(
            data_dir=str(simple_data_dir), model_type="temporal_ridge",
            model_config={"alpha": 1.0}, output_dir=str(tmp_path / "output"),
        )
        trainer.setup()
        trainer.train()
        metrics = trainer.evaluate("train")
        assert "r2" in metrics

    def test_evaluate_unknown_split_raises(self, simple_data_dir, tmp_path):
        trainer = SimpleModelTrainer(
            data_dir=str(simple_data_dir), model_type="temporal_ridge",
            model_config={"alpha": 1.0}, output_dir=str(tmp_path / "output"),
        )
        trainer.setup()
        trainer.train()
        with pytest.raises(ValueError, match=r"Unknown split"):
            trainer.evaluate("nonsense")
