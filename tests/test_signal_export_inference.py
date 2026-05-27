"""
Tests for SignalExporter inference paths and file writing.

SignalExporter produces every .npy signal file consumed by the backtester.
All 16 experiments and 8 backtest rounds depend on this code being correct.
A silent bug here could have caused us to discard a profitable strategy.

The 4 inference methods extract different ModelOutput fields per strategy:
- Classification: output.logits → argmax → int32 predictions
- HMHP Classification: + output.agreement, output.confidence
- Regression: output.predictions → float64 predicted_returns
- HMHP Regression: output.horizon_predictions → [N, H] float64

Silent corruption risks tested:
- [B,1] vs [B] shape (squeeze(-1) failures)
- int32 vs int64 dtype (backtester contract)
- float32 truncation (should be float64)
- Horizon column ordering (must be sorted by key)

Design Principles (hft-rules.md):
    - Shape tests validate all tensor dimensions (Rule 6)
    - Dtype tests prevent silent truncation (Rule 2)
    - Contract tests: producer → consumer (Rule 6)
"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from unittest.mock import MagicMock

from lobtrainer.export.exporter import SignalExporter
from lobtrainer.training.strategies.classification import ClassificationStrategy
from lobtrainer.training.strategies.regression import RegressionStrategy
from lobtrainer.training.strategies.hmhp_classification import HMHPClassificationStrategy
from lobtrainer.training.strategies.hmhp_regression import HMHPRegressionStrategy
from lobtrainer.config import ModelType, TaskType, LossType

from lobmodels import (
    LogisticLOB, LogisticLOBConfig, SequencePooling,
    TLOB, TLOBConfig, create_hmhp,
)
from lobmodels.models.hmhp_regressor import create_hmhp_regressor


# =============================================================================
# Constants
# =============================================================================

NUM_FEATURES = 10
SEQ_LEN = 20
NUM_CLASSES = 3
BATCH_SIZE = 4
N_SAMPLES = 8
HORIZONS = [10, 20]
DEVICE = torch.device("cpu")


# =============================================================================
# Helpers
# =============================================================================


def _make_config(**kw):
    """MagicMock config for strategy construction."""
    config = MagicMock()
    config.model.model_type = kw.get("model_type", ModelType.TLOB)
    config.model.num_classes = kw.get("num_classes", NUM_CLASSES)
    config.model.hmhp_horizons = kw.get("hmhp_horizons", None)
    config.model.hmhp_use_regression = kw.get("hmhp_use_regression", False)
    config.train.task_type = kw.get("task_type", TaskType.MULTICLASS)
    config.train.loss_type = kw.get("loss_type", LossType.CROSS_ENTROPY)
    config.train.focal_gamma = 2.0
    config.data.labeling_strategy = "opportunity"
    config.data.num_classes = NUM_CLASSES
    config.data.horizon_idx = 0
    return config


def _make_exporter(model, strategy):
    """Create SignalExporter with mock trainer wrapping real model+strategy."""
    mock_trainer = MagicMock()
    mock_trainer.model = model
    mock_trainer.device = DEVICE
    mock_trainer.strategy = strategy
    mock_trainer.config.data.labels.validate_primary_horizon_idx_for.return_value = 0
    return SignalExporter(mock_trainer)


class DictLabelDataset(Dataset):
    """Dict-label dataset for HMHP tests."""
    def __init__(self, features, horizons, include_regression=False):
        self.features = features
        self.labels = {h: torch.randint(0, NUM_CLASSES, (len(features),)) for h in horizons}
        self.include_regression = include_regression
        if include_regression:
            self.reg_targets = {h: torch.randn(len(features)) for h in horizons}
        self.horizons = horizons

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        labs = {h: self.labels[h][idx] for h in self.horizons}
        if self.include_regression:
            regs = {h: self.reg_targets[h][idx] for h in self.horizons}
            return self.features[idx], labs, regs
        return self.features[idx], labs


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def features():
    torch.manual_seed(42)
    return torch.randn(N_SAMPLES, SEQ_LEN, NUM_FEATURES)


@pytest.fixture
def clf_model():
    return LogisticLOB(LogisticLOBConfig(
        num_features=NUM_FEATURES, sequence_length=SEQ_LEN,
        num_classes=NUM_CLASSES, pooling=SequencePooling.LAST,
    ))


@pytest.fixture
def reg_model():
    config = TLOBConfig(
        num_features=NUM_FEATURES, sequence_length=SEQ_LEN,
        num_classes=NUM_CLASSES, hidden_dim=8, num_heads=2,
        num_layers=1, use_bin=False, dropout=0.0,
    )
    config.task_type = "regression"
    return TLOB(config)


@pytest.fixture
def hmhp_model():
    return create_hmhp(
        num_features=NUM_FEATURES, num_classes=NUM_CLASSES,
        sequence_length=SEQ_LEN, horizons=HORIZONS, hidden_dim=16,
        num_encoder_layers=1, use_bin=False, dropout=0.0,
    )


@pytest.fixture
def hmhp_reg_model():
    return create_hmhp_regressor(
        num_features=NUM_FEATURES, sequence_length=SEQ_LEN,
        horizons=HORIZONS, hidden_dim=16, num_encoder_layers=1,
        use_bin=False, dropout=0.0,
    )


@pytest.fixture
def clf_loader(features):
    return DataLoader(
        TensorDataset(features, torch.randint(0, NUM_CLASSES, (N_SAMPLES,))),
        batch_size=BATCH_SIZE,
    )


@pytest.fixture
def reg_loader(features):
    return DataLoader(
        TensorDataset(features, torch.randn(N_SAMPLES)),
        batch_size=BATCH_SIZE,
    )


@pytest.fixture
def hmhp_loader(features):
    return DataLoader(
        DictLabelDataset(features, HORIZONS),
        batch_size=BATCH_SIZE,
    )


@pytest.fixture
def hmhp_reg_loader(features):
    return DataLoader(
        DictLabelDataset(features, HORIZONS, include_regression=True),
        batch_size=BATCH_SIZE,
    )


# =============================================================================
# TestInferClassification
# =============================================================================


class TestInferClassification:
    """Tests for _infer_classification output contract."""

    def test_predictions_shape_1d(self, clf_model, clf_loader):
        """Predictions must be [N] 1D, NOT [N,1]."""
        config = _make_config(loss_type=LossType.CROSS_ENTROPY)
        strategy = ClassificationStrategy(config, DEVICE)
        exporter = _make_exporter(clf_model, strategy)

        result = exporter._infer_classification(clf_model, DEVICE, clf_loader)
        assert result["predictions"].ndim == 1, (
            f"Predictions should be 1D, got shape {result['predictions'].shape}"
        )
        assert result["predictions"].shape == (N_SAMPLES,)

    def test_predictions_dtype_int32(self, clf_model, clf_loader):
        """Predictions must be int32 (backtester contract)."""
        config = _make_config(loss_type=LossType.CROSS_ENTROPY)
        strategy = ClassificationStrategy(config, DEVICE)
        exporter = _make_exporter(clf_model, strategy)

        result = exporter._infer_classification(clf_model, DEVICE, clf_loader)
        assert result["predictions"].dtype == np.int32, (
            f"Expected int32, got {result['predictions'].dtype}"
        )
        assert result["labels"].dtype == np.int32

    def test_n_samples_matches_data(self, clf_model, clf_loader):
        """n_samples must equal total samples across all batches."""
        config = _make_config(loss_type=LossType.CROSS_ENTROPY)
        strategy = ClassificationStrategy(config, DEVICE)
        exporter = _make_exporter(clf_model, strategy)

        result = exporter._infer_classification(clf_model, DEVICE, clf_loader)
        assert result["n_samples"] == N_SAMPLES

    def test_prediction_values_in_range(self, clf_model, clf_loader):
        """All predictions must be valid class indices {0, 1, 2}."""
        config = _make_config(loss_type=LossType.CROSS_ENTROPY)
        strategy = ClassificationStrategy(config, DEVICE)
        exporter = _make_exporter(clf_model, strategy)

        result = exporter._infer_classification(clf_model, DEVICE, clf_loader)
        assert set(result["predictions"]).issubset({0, 1, 2}), (
            f"Invalid class indices: {set(result['predictions'])}"
        )

    # =========================================================================
    # Phase R-17a (2026-05-14) — agreement_ratio + confirmation_score synthesis
    # =========================================================================

    def test_agreement_ratio_present_and_ones(self, clf_model, clf_loader):
        """Phase R-17a: agreement_ratio synthesized as all-ones for single-horizon.

        Semantic: single-horizon classification trivially "all horizons agree"
        because there is only one horizon. Differs from HMHP where
        agreement_ratio measures inter-horizon agreement (hmhp.py:467-468).
        """
        config = _make_config(loss_type=LossType.CROSS_ENTROPY)
        strategy = ClassificationStrategy(config, DEVICE)
        exporter = _make_exporter(clf_model, strategy)

        result = exporter._infer_classification(clf_model, DEVICE, clf_loader)
        assert "agreement_ratio" in result, (
            f"Phase R-17a adapter must emit agreement_ratio key "
            f"(got keys: {list(result.keys())})"
        )
        assert result["agreement_ratio"].dtype == np.float64, (
            f"Expected float64, got {result['agreement_ratio'].dtype}"
        )
        assert result["agreement_ratio"].shape == (N_SAMPLES,), (
            f"Expected shape ({N_SAMPLES},), got {result['agreement_ratio'].shape}"
        )
        assert np.all(result["agreement_ratio"] == 1.0), (
            f"agreement_ratio should be synthetic-constant 1.0 for single-horizon; "
            f"got unique values: {np.unique(result['agreement_ratio'])}"
        )

    def test_confirmation_score_present_softmax_max(self, clf_model, clf_loader):
        """Phase R-17a: confirmation_score = per-sample softmax-max, dtype float64."""
        config = _make_config(loss_type=LossType.CROSS_ENTROPY)
        strategy = ClassificationStrategy(config, DEVICE)
        exporter = _make_exporter(clf_model, strategy)

        result = exporter._infer_classification(clf_model, DEVICE, clf_loader)
        assert "confirmation_score" in result, (
            f"Phase R-17a adapter must emit confirmation_score key "
            f"(got keys: {list(result.keys())})"
        )
        assert result["confirmation_score"].dtype == np.float64
        assert result["confirmation_score"].shape == (N_SAMPLES,)
        # Softmax-max upper bound (one-hot extreme) is 1.0.
        assert np.all(result["confirmation_score"] <= 1.0 + 1e-6), (
            f"confirmation_score must be <= 1.0 (softmax-max upper bound); "
            f"got max={result['confirmation_score'].max()}"
        )

    def test_confirmation_score_lower_bound_uniform(self, clf_model, clf_loader):
        """Phase R-17a: confirmation_score >= 1/num_classes (uniform-softmax lower bound).

        Mathematically, softmax-max for C classes is in [1/C, 1.0]: minimum at
        uniform logits (e.g., zeros) gives 1/C; maximum at one-hot logits gives 1.0.
        For NUM_CLASSES=3, lower bound is ~0.333.
        """
        config = _make_config(loss_type=LossType.CROSS_ENTROPY)
        strategy = ClassificationStrategy(config, DEVICE)
        exporter = _make_exporter(clf_model, strategy)

        result = exporter._infer_classification(clf_model, DEVICE, clf_loader)
        lower_bound = 1.0 / NUM_CLASSES - 1e-6  # epsilon for float rounding
        assert np.all(result["confirmation_score"] >= lower_bound), (
            f"confirmation_score must be >= 1/num_classes={1.0/NUM_CLASSES:.4f} "
            f"(softmax-max lower bound); got min={result['confirmation_score'].min():.4f}"
        )

    def test_nan_guard_on_degenerate_logits(self, clf_loader):
        """Phase R-17a §8 fail-loud: NaN logits → ValueError (refuse corrupt output).

        Mirrors `_infer_regression` (L312-320) + `_infer_hmhp_regression`
        (L396-403) patterns. Without this guard, NaN confirmation_score would
        silently flow into the readability backtester where
        ``confirmation_score > min_confidence`` evaluates to ``False`` for
        NaN, silently dropping every signal.
        """
        from types import SimpleNamespace

        class _NaNLogitsModel(torch.nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.num_classes = num_classes

            def forward(self, features):
                B = features.shape[0]
                # All-NaN logits — softmax-max will be NaN, guard must raise.
                logits = torch.full((B, self.num_classes), float("nan"))
                return SimpleNamespace(logits=logits)

        config = _make_config(loss_type=LossType.CROSS_ENTROPY)
        strategy = ClassificationStrategy(config, DEVICE)
        nan_model = _NaNLogitsModel(NUM_CLASSES)
        exporter = _make_exporter(nan_model, strategy)

        with pytest.raises(ValueError, match=r"(?i)confirmation_score|non.?finite|NaN"):
            exporter._infer_classification(nan_model, DEVICE, clf_loader)

    def test_binary_signal_defensive_guard(self, features):
        """Phase R-17a: num_classes=1 (BINARY_SIGNAL) → confirmation_score = 1.0.

        For ``num_classes == 1`` (BINARY_SIGNAL task_type at
        strategies/classification.py:91), ``F.softmax(logits, dim=1)`` on shape
        ``[B, 1]`` is trivially 1.0 — but emitting that as
        "confidence" is misleading. Defensive guard emits 1.0 explicitly so
        the semantic is clear at consumer side.
        """
        from types import SimpleNamespace

        class _BinaryLogitsModel(torch.nn.Module):
            def forward(self, features):
                B = features.shape[0]
                # Single-logit output; shape [B, 1] simulates BINARY_SIGNAL.
                logits = torch.randn(B, 1)
                return SimpleNamespace(logits=logits)

        config = _make_config(loss_type=LossType.CROSS_ENTROPY, num_classes=1)
        strategy = ClassificationStrategy(config, DEVICE)
        binary_model = _BinaryLogitsModel()
        binary_loader = DataLoader(
            TensorDataset(features, torch.zeros(N_SAMPLES, dtype=torch.long)),
            batch_size=BATCH_SIZE,
        )
        exporter = _make_exporter(binary_model, strategy)

        result = exporter._infer_classification(binary_model, DEVICE, binary_loader)
        # Defensive guard emits 1.0 for binary-signal (not degenerate softmax).
        assert np.all(result["confirmation_score"] == 1.0), (
            f"Binary signal (num_classes=1) should emit confirmation_score=1.0 "
            f"via defensive guard; got {result['confirmation_score'][:5]}"
        )


# =============================================================================
# TestInferHMHPClassification
# =============================================================================


class TestInferHMHPClassification:
    """Tests for HMHP classification inference output contract.

    Uses _run_inference (which has @torch.no_grad) to match production path.
    """

    def _run(self, hmhp_model, hmhp_loader):
        config = _make_config(model_type=ModelType.HMHP, hmhp_horizons=HORIZONS)
        strategy = HMHPClassificationStrategy(config, DEVICE)
        exporter = _make_exporter(hmhp_model, strategy)
        return exporter._run_inference(hmhp_loader)

    def test_agreement_shape_1d(self, hmhp_model, hmhp_loader):
        """agreement_ratio must be [N] 1D after squeeze(-1)."""
        result = self._run(hmhp_model, hmhp_loader)
        assert result["agreement_ratio"].ndim == 1, (
            f"agreement_ratio should be 1D, got shape {result['agreement_ratio'].shape}"
        )
        assert result["agreement_ratio"].shape == (N_SAMPLES,)

    def test_confidence_shape_1d(self, hmhp_model, hmhp_loader):
        """confirmation_score must be [N] 1D after squeeze(-1)."""
        result = self._run(hmhp_model, hmhp_loader)
        assert result["confirmation_score"].ndim == 1, (
            f"confirmation_score should be 1D, got {result['confirmation_score'].shape}"
        )

    def test_dtype_float64(self, hmhp_model, hmhp_loader):
        """agreement_ratio and confirmation_score must be float64."""
        result = self._run(hmhp_model, hmhp_loader)
        assert result["agreement_ratio"].dtype == np.float64
        assert result["confirmation_score"].dtype == np.float64


# =============================================================================
# TestInferRegression
# =============================================================================


class TestInferRegression:
    """Tests for regression inference output contract.

    Uses _run_inference (which has @torch.no_grad) to match production path.
    """

    def _run(self, reg_model, reg_loader):
        config = _make_config(task_type=TaskType.REGRESSION)
        strategy = RegressionStrategy(config, DEVICE)
        exporter = _make_exporter(reg_model, strategy)
        return exporter._run_inference(reg_loader)

    def test_predictions_shape_1d(self, reg_model, reg_loader):
        """predicted_returns must be [N] 1D."""
        result = self._run(reg_model, reg_loader)
        assert result["predicted_returns"].ndim == 1, (
            f"predicted_returns should be 1D, got {result['predicted_returns'].shape}"
        )
        assert result["predicted_returns"].shape == (N_SAMPLES,)

    def test_predictions_dtype_float64(self, reg_model, reg_loader):
        """predicted_returns must be float64 (not float32 truncation)."""
        result = self._run(reg_model, reg_loader)
        assert result["predicted_returns"].dtype == np.float64, (
            f"Expected float64, got {result['predicted_returns'].dtype}"
        )

    def test_predictions_finite(self, reg_model, reg_loader):
        """No NaN or Inf in predicted_returns."""
        result = self._run(reg_model, reg_loader)
        assert np.isfinite(result["predicted_returns"]).all(), (
            f"predicted_returns contains non-finite values"
        )


# =============================================================================
# TestInferHMHPRegression
# =============================================================================


class TestInferHMHPRegression:
    """Tests for HMHP regression inference output contract.

    Uses _run_inference (which has @torch.no_grad) to match production path.
    """

    def _run(self, hmhp_reg_model, hmhp_reg_loader):
        config = _make_config(model_type=ModelType.HMHP_REGRESSION, hmhp_horizons=HORIZONS)
        strategy = HMHPRegressionStrategy(config, DEVICE)
        exporter = _make_exporter(hmhp_reg_model, strategy)
        return exporter._run_inference(hmhp_reg_loader)

    def test_multi_horizon_shape(self, hmhp_reg_model, hmhp_reg_loader):
        """predicted_returns shape must be [N, H] where H = num_horizons."""
        result = self._run(hmhp_reg_model, hmhp_reg_loader)
        assert result["predicted_returns"].ndim == 2, (
            f"HMHP-R predicted_returns should be 2D [N,H], got {result['predicted_returns'].shape}"
        )
        assert result["predicted_returns"].shape == (N_SAMPLES, len(HORIZONS))

    def test_horizon_columns_sorted(self, hmhp_reg_model, hmhp_reg_loader):
        """Horizon columns must be sorted by horizon key (ascending)."""
        result = self._run(hmhp_reg_model, hmhp_reg_loader)
        assert result["horizons"] == sorted(HORIZONS), (
            f"Horizons should be sorted, got {result['horizons']}"
        )

    def test_agreement_present(self, hmhp_reg_model, hmhp_reg_loader):
        """HMHP-R also exports agreement_ratio."""
        result = self._run(hmhp_reg_model, hmhp_reg_loader)
        assert "agreement_ratio" in result, (
            f"HMHP-R should export agreement_ratio. Keys: {result.keys()}"
        )
        assert result["agreement_ratio"].shape == (N_SAMPLES,)


# =============================================================================
# TestRunInferenceDispatch
# =============================================================================


class TestRunInferenceDispatch:
    """Tests for _run_inference strategy dispatch."""

    def test_dispatches_to_classification(self, clf_model, clf_loader):
        """Classification strategy → _infer_classification path.

        Phase R-17a (2026-05-14): also asserts adapter-synthesized
        ``agreement_ratio`` + ``confirmation_score`` keys propagate through
        the dispatch (integration check — closes Adv2 gap that the per-method
        tests cover keys-in-result but the dispatcher might fail to forward).
        """
        config = _make_config(loss_type=LossType.CROSS_ENTROPY)
        strategy = ClassificationStrategy(config, DEVICE)
        exporter = _make_exporter(clf_model, strategy)

        result = exporter._run_inference(clf_loader)
        assert result["signal_type"] == "classification"
        # Phase R-17a: verify Phase 1 adapter keys propagate through dispatch
        assert "agreement_ratio" in result, (
            "Phase R-17a adapter keys must propagate through _run_inference dispatch"
        )
        assert "confirmation_score" in result

    def test_dispatches_to_regression(self, reg_model, reg_loader):
        """Regression strategy → _infer_regression path."""
        config = _make_config(task_type=TaskType.REGRESSION)
        strategy = RegressionStrategy(config, DEVICE)
        exporter = _make_exporter(reg_model, strategy)

        result = exporter._run_inference(reg_loader)
        assert result["signal_type"] == "regression"

    def test_dispatches_to_hmhp_classification(self, hmhp_model, hmhp_loader):
        """HMHP clf strategy → _infer_hmhp_classification path."""
        config = _make_config(model_type=ModelType.HMHP, hmhp_horizons=HORIZONS)
        strategy = HMHPClassificationStrategy(config, DEVICE)
        exporter = _make_exporter(hmhp_model, strategy)

        result = exporter._run_inference(hmhp_loader)
        assert result["signal_type"] == "classification"
        assert "agreement_ratio" in result

    def test_dispatches_to_hmhp_regression(self, hmhp_reg_model, hmhp_reg_loader):
        """HMHP reg strategy → _infer_hmhp_regression path."""
        config = _make_config(model_type=ModelType.HMHP_REGRESSION, hmhp_horizons=HORIZONS)
        strategy = HMHPRegressionStrategy(config, DEVICE)
        exporter = _make_exporter(hmhp_reg_model, strategy)

        result = exporter._run_inference(hmhp_reg_loader)
        assert result["signal_type"] == "regression"
        assert result["predicted_returns"].ndim == 2
