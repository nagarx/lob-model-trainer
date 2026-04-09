"""
Tests for training strategies (Phase 5 — Critical Test Coverage).

Tests verify the 4 concrete strategies correctly consume ModelOutput,
compute losses, and produce metrics. These strategies process EVERY
training gradient — a bug here silently corrupts all experiments.

Key test: test_process_batch_uses_criterion_not_compute_loss
    Regression test for Phase 4 Fix 1 (loss ownership bug). Before the
    fix, ClassificationStrategy.process_batch() called model.compute_loss()
    (bare F.cross_entropy) instead of self._criterion (configured
    weighted CE / FocalLoss). Train and val used DIFFERENT loss functions.

Test Categories:
    1. Factory dispatch: create_strategy routes to correct class
    2. Classification: criterion creation, loss ownership, predict
    3. Regression: model.compute_loss delegation, metric keys
    4. HMHP Classification: dict labels, agreement, per-horizon metrics
    5. HMHP Regression: 3-tuple requirement, per-horizon regression metrics

Design Principles (hft-rules.md):
    - Tests document behavior and expose correctness (Rule 6)
    - No tautological tests — test contracts, not implementation
    - Assertions explain WHAT failed and WHY
    - Deterministic: torch.manual_seed(42) in all fixtures
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from unittest.mock import MagicMock

from lobtrainer.config import LossType, ModelType, TaskType
from lobtrainer.training.strategy import BatchResult, TrainingStrategy, create_strategy
from lobtrainer.training.strategies.classification import ClassificationStrategy
from lobtrainer.training.strategies.regression import RegressionStrategy
from lobtrainer.training.strategies.hmhp_classification import (
    HMHPClassificationStrategy,
)
from lobtrainer.training.strategies.hmhp_regression import HMHPRegressionStrategy

# Models from lobmodels (the correct source — NOT lobtrainer.models.lstm)
from lobmodels import (
    LogisticLOB,
    LogisticLOBConfig,
    SequencePooling,
    TLOB,
    TLOBConfig,
    create_hmhp,
)
from lobmodels.models.hmhp_regressor import create_hmhp_regressor
from lobmodels.registry.output import ModelOutput


# =============================================================================
# Test Constants
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


def _make_config(
    model_type=ModelType.TLOB,
    task_type=TaskType.MULTICLASS,
    loss_type=LossType.WEIGHTED_CE,
    num_classes=NUM_CLASSES,
    hmhp_horizons=None,
    hmhp_use_regression=False,
    labeling_strategy="opportunity",
    focal_gamma=2.0,
):
    """Build MagicMock config covering all strategy attribute accesses.

    Validated against live code (V2): every config.X.Y access across
    all 4 strategies + factory is covered by this mock.
    """
    config = MagicMock()
    config.model.model_type = model_type
    config.model.num_classes = num_classes
    config.model.hmhp_horizons = hmhp_horizons
    config.model.hmhp_use_regression = hmhp_use_regression
    config.train.task_type = task_type
    config.train.loss_type = loss_type
    config.train.focal_gamma = focal_gamma
    config.data.labeling_strategy = labeling_strategy
    config.data.num_classes = num_classes
    config.data.horizon_idx = 0
    return config


class DictLabelDataset(Dataset):
    """Minimal dataset yielding dict labels for HMHP strategies.

    PyTorch default collate handles dicts correctly: values are stacked
    per key, producing {horizon: [B]} tensors (validated in V3).
    """

    def __init__(self, features, horizons, num_classes=3, include_regression=False):
        self.features = features
        self.labels = {
            h: torch.randint(0, num_classes, (len(features),)) for h in horizons
        }
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
    """Deterministic feature tensor [N_SAMPLES, SEQ_LEN, NUM_FEATURES]."""
    torch.manual_seed(42)
    return torch.randn(N_SAMPLES, SEQ_LEN, NUM_FEATURES)


@pytest.fixture
def tiny_clf_model():
    """LogisticLOB — smallest viable classification model (~33 params)."""
    return LogisticLOB(
        LogisticLOBConfig(
            num_features=NUM_FEATURES,
            sequence_length=SEQ_LEN,
            num_classes=NUM_CLASSES,
            pooling=SequencePooling.LAST,
            dropout=0.0,
        )
    )


@pytest.fixture
def tiny_reg_model():
    """TLOB in regression mode — smallest viable regression model.

    TLOBConfig does not have task_type as a field; it's injected dynamically
    (getattr fallback pattern). This matches how the Trainer injects it.
    """
    config = TLOBConfig(
        num_features=NUM_FEATURES,
        sequence_length=SEQ_LEN,
        num_classes=NUM_CLASSES,
        hidden_dim=8,
        num_heads=2,
        num_layers=1,
        use_bin=False,
        dropout=0.0,
    )
    config.task_type = "regression"
    return TLOB(config)


@pytest.fixture
def hmhp_model():
    """HMHP classification model with 2 horizons (~11K params).

    num_heads is hardcoded to 1 in SharedEncoder (not a factory param).
    use_bin=False avoids numerical issues with small batch sizes.
    """
    return create_hmhp(
        num_features=NUM_FEATURES,
        num_classes=NUM_CLASSES,
        sequence_length=SEQ_LEN,
        horizons=HORIZONS,
        hidden_dim=16,
        num_encoder_layers=1,
        use_bin=False,
        dropout=0.0,
    )


@pytest.fixture
def hmhp_reg_model():
    """HMHP regression model with 2 horizons."""
    return create_hmhp_regressor(
        num_features=NUM_FEATURES,
        sequence_length=SEQ_LEN,
        horizons=HORIZONS,
        hidden_dim=16,
        num_encoder_layers=1,
        use_bin=False,
        dropout=0.0,
    )


@pytest.fixture
def clf_loader(features):
    """Classification DataLoader: (features [B,T,F], labels [B] int64)."""
    labels = torch.randint(0, NUM_CLASSES, (N_SAMPLES,))
    return DataLoader(TensorDataset(features, labels), batch_size=BATCH_SIZE)


@pytest.fixture
def reg_loader(features):
    """Regression DataLoader: (features [B,T,F], targets [B] float32)."""
    targets = torch.randn(N_SAMPLES)
    return DataLoader(TensorDataset(features, targets), batch_size=BATCH_SIZE)


@pytest.fixture
def hmhp_loader(features):
    """HMHP classification DataLoader: (features, {h: labels})."""
    ds = DictLabelDataset(features, HORIZONS, num_classes=NUM_CLASSES)
    return DataLoader(ds, batch_size=BATCH_SIZE)


@pytest.fixture
def hmhp_reg_loader(features):
    """HMHP regression DataLoader: (features, {h: labels}, {h: reg_targets})."""
    ds = DictLabelDataset(
        features, HORIZONS, num_classes=NUM_CLASSES, include_regression=True
    )
    return DataLoader(ds, batch_size=BATCH_SIZE)


# =============================================================================
# TestCreateStrategy — Factory Dispatch
# =============================================================================


class TestCreateStrategy:
    """Verify create_strategy routes to the correct strategy class."""

    def test_hmhp_model_type_returns_hmhp_clf_strategy(self):
        """ModelType.HMHP -> HMHPClassificationStrategy (model_type takes priority)."""
        config = _make_config(model_type=ModelType.HMHP, hmhp_horizons=HORIZONS)
        strategy = create_strategy(config, DEVICE)
        assert isinstance(strategy, HMHPClassificationStrategy), (
            f"Expected HMHPClassificationStrategy, got {type(strategy).__name__}"
        )

    def test_hmhp_regression_returns_hmhp_reg_strategy(self):
        """ModelType.HMHP_REGRESSION -> HMHPRegressionStrategy."""
        config = _make_config(
            model_type=ModelType.HMHP_REGRESSION, hmhp_horizons=HORIZONS
        )
        strategy = create_strategy(config, DEVICE)
        assert isinstance(strategy, HMHPRegressionStrategy), (
            f"Expected HMHPRegressionStrategy, got {type(strategy).__name__}"
        )

    def test_regression_task_returns_regression_strategy(self):
        """TaskType.REGRESSION -> RegressionStrategy (for non-HMHP models)."""
        config = _make_config(task_type=TaskType.REGRESSION)
        strategy = create_strategy(config, DEVICE)
        assert isinstance(strategy, RegressionStrategy), (
            f"Expected RegressionStrategy, got {type(strategy).__name__}"
        )

    def test_default_returns_classification_strategy(self):
        """Default (non-HMHP, non-regression) -> ClassificationStrategy."""
        config = _make_config()
        strategy = create_strategy(config, DEVICE)
        assert isinstance(strategy, ClassificationStrategy), (
            f"Expected ClassificationStrategy, got {type(strategy).__name__}"
        )


# =============================================================================
# TestClassificationStrategy
# =============================================================================


class TestClassificationStrategy:
    """Tests for single-horizon classification (TLOB, DeepLOB, etc.)."""

    def test_data_pipeline_properties(self):
        """Classification: no dict labels, no regression targets, horizon_idx=0."""
        config = _make_config()
        strategy = ClassificationStrategy(config, DEVICE)
        assert strategy.requires_dict_labels is False
        assert strategy.requires_regression_targets is False
        assert strategy.horizon_idx == 0

    def test_initialize_creates_criterion(self, tiny_clf_model, clf_loader):
        """initialize() must create the loss criterion."""
        config = _make_config(loss_type=LossType.CROSS_ENTROPY)
        strategy = ClassificationStrategy(config, DEVICE)
        assert strategy.criterion is None, "Criterion should be None before initialize"
        strategy.initialize(clf_loader, tiny_clf_model)
        assert strategy.criterion is not None, (
            "Criterion must be created after initialize"
        )

    def test_create_criterion_cross_entropy(self, tiny_clf_model, clf_loader):
        """LossType.CROSS_ENTROPY -> nn.CrossEntropyLoss (unweighted)."""
        config = _make_config(loss_type=LossType.CROSS_ENTROPY)
        strategy = ClassificationStrategy(config, DEVICE)
        strategy.initialize(clf_loader, tiny_clf_model)
        assert isinstance(strategy.criterion, nn.CrossEntropyLoss)

    def test_create_criterion_focal(self, tiny_clf_model, clf_loader):
        """LossType.FOCAL -> FocalLoss with configured gamma."""
        from lobtrainer.training.loss import FocalLoss

        config = _make_config(loss_type=LossType.FOCAL, focal_gamma=3.0)
        strategy = ClassificationStrategy(config, DEVICE)
        strategy.initialize(clf_loader, tiny_clf_model)
        assert isinstance(strategy.criterion, FocalLoss), (
            f"Expected FocalLoss, got {type(strategy.criterion).__name__}"
        )
        assert strategy.criterion.gamma == 3.0

    def test_create_criterion_weighted_ce(self, tiny_clf_model, clf_loader):
        """LossType.WEIGHTED_CE -> nn.CrossEntropyLoss with class weights."""
        config = _make_config(loss_type=LossType.WEIGHTED_CE)
        strategy = ClassificationStrategy(config, DEVICE)
        strategy.initialize(clf_loader, tiny_clf_model)
        assert isinstance(strategy.criterion, nn.CrossEntropyLoss)
        # Weighted CE should have weight attribute set
        assert strategy.criterion.weight is not None, (
            "Weighted CE must have class weights"
        )
        assert strategy.criterion.weight.shape == (NUM_CLASSES,)

    def test_process_batch_uses_criterion_not_compute_loss(
        self, tiny_clf_model, clf_loader
    ):
        """CRITICAL: process_batch must use self._criterion, NOT model.compute_loss().

        This is the regression test for Phase 4 Fix 1 (loss ownership bug).
        Before the fix, process_batch called model.compute_loss() which uses
        bare F.cross_entropy — ignoring configured weighted CE / FocalLoss.

        Proof: mock model.compute_loss with a bomb. If process_batch succeeds,
        it proves the criterion is used instead of compute_loss.
        """
        config = _make_config(loss_type=LossType.CROSS_ENTROPY)
        strategy = ClassificationStrategy(config, DEVICE)
        strategy.initialize(clf_loader, tiny_clf_model)

        # Plant a bomb on model.compute_loss
        original = tiny_clf_model.compute_loss
        tiny_clf_model.compute_loss = lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError(
                "BUG: process_batch called model.compute_loss() "
                "instead of self._criterion"
            )
        )

        try:
            batch = next(iter(clf_loader))
            result = strategy.process_batch(tiny_clf_model, batch)

            assert result.loss.requires_grad, (
                "Loss must be a live tensor for .backward()"
            )
            assert result.batch_size == BATCH_SIZE
        finally:
            tiny_clf_model.compute_loss = original

    def test_process_batch_correct_count(self, tiny_clf_model, clf_loader):
        """process_batch returns correct_count matching argmax predictions."""
        config = _make_config(loss_type=LossType.CROSS_ENTROPY)
        strategy = ClassificationStrategy(config, DEVICE)
        strategy.initialize(clf_loader, tiny_clf_model)

        batch = next(iter(clf_loader))
        features, labels = batch
        result = strategy.process_batch(tiny_clf_model, batch)

        # Manually compute expected correct count
        with torch.no_grad():
            output = tiny_clf_model(features)
            preds = output.logits.argmax(dim=1)
            expected_correct = (preds == labels).sum().item()

        assert result.metrics["correct_count"] == expected_correct, (
            f"Expected {expected_correct} correct, got {result.metrics['correct_count']}"
        )

    def test_aggregate_empty_returns_zero(self):
        """aggregate_epoch_metrics with total_samples=0 returns zeros."""
        config = _make_config()
        strategy = ClassificationStrategy(config, DEVICE)
        metrics = strategy.aggregate_epoch_metrics([], total_samples=0)
        assert metrics["train_loss"] == 0.0
        assert metrics["train_accuracy"] == 0.0

    def test_validate_returns_expected_keys(self, tiny_clf_model, clf_loader):
        """validate() returns val_loss, val_accuracy, val_macro_f1."""
        config = _make_config(loss_type=LossType.CROSS_ENTROPY)
        strategy = ClassificationStrategy(config, DEVICE)
        strategy.initialize(clf_loader, tiny_clf_model)

        metrics = strategy.validate(tiny_clf_model, clf_loader)
        assert "val_loss" in metrics, f"Missing val_loss. Keys: {metrics.keys()}"
        assert "val_accuracy" in metrics, f"Missing val_accuracy. Keys: {metrics.keys()}"
        assert "val_macro_f1" in metrics, f"Missing val_macro_f1. Keys: {metrics.keys()}"
        assert metrics["val_loss"] >= 0, f"val_loss should be non-negative, got {metrics['val_loss']}"

    def test_evaluate_returns_classification_metrics(
        self, tiny_clf_model, clf_loader
    ):
        """evaluate() returns ClassificationMetrics with accuracy and f1."""
        config = _make_config(loss_type=LossType.CROSS_ENTROPY)
        strategy = ClassificationStrategy(config, DEVICE)
        strategy.initialize(clf_loader, tiny_clf_model)

        metrics = strategy.evaluate(tiny_clf_model, clf_loader, "test")
        assert hasattr(metrics, "accuracy"), "Missing accuracy attribute"
        assert hasattr(metrics, "macro_f1"), "Missing macro_f1 attribute"
        assert 0.0 <= metrics.accuracy <= 1.0

    def test_predict_returns_class_labels(self, tiny_clf_model, features):
        """predict(return_proba=False) returns integer class labels [B]."""
        config = _make_config()
        strategy = ClassificationStrategy(config, DEVICE)

        preds = strategy.predict(tiny_clf_model, features[:BATCH_SIZE])
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (BATCH_SIZE,)
        assert preds.dtype in (np.int32, np.int64)

    def test_predict_proba_sums_to_one(self, tiny_clf_model, features):
        """predict(return_proba=True) returns softmax probabilities summing to 1."""
        config = _make_config()
        strategy = ClassificationStrategy(config, DEVICE)

        proba = strategy.predict(tiny_clf_model, features[:BATCH_SIZE], return_proba=True)
        assert proba.shape == (BATCH_SIZE, NUM_CLASSES)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(
            row_sums, 1.0, atol=1e-5,
            err_msg="Softmax probabilities must sum to 1.0",
        )


# =============================================================================
# TestRegressionStrategy
# =============================================================================


class TestRegressionStrategy:
    """Tests for single-horizon regression (TLOB-R, etc.)."""

    def test_data_pipeline_properties(self):
        """Regression: requires regression targets, no dict labels."""
        config = _make_config(task_type=TaskType.REGRESSION)
        strategy = RegressionStrategy(config, DEVICE)
        assert strategy.requires_regression_targets is True
        assert strategy.requires_dict_labels is False

    def test_process_batch_delegates_to_compute_loss(self, tiny_reg_model, reg_loader):
        """Regression strategy MUST delegate to model.compute_loss().

        Symmetry test: classification does NOT call compute_loss (uses criterion),
        but regression DOES (model owns its regression loss function).
        """
        config = _make_config(task_type=TaskType.REGRESSION)
        strategy = RegressionStrategy(config, DEVICE)

        # Track whether compute_loss was called
        call_tracker = {"called": False}
        original = tiny_reg_model.compute_loss

        def tracking_compute_loss(*args, **kwargs):
            call_tracker["called"] = True
            return original(*args, **kwargs)

        tiny_reg_model.compute_loss = tracking_compute_loss
        try:
            batch = next(iter(reg_loader))
            result = strategy.process_batch(tiny_reg_model, batch)
            assert call_tracker["called"], (
                "RegressionStrategy.process_batch must call model.compute_loss()"
            )
            assert result.loss.requires_grad
        finally:
            tiny_reg_model.compute_loss = original

    def test_process_batch_converts_labels_to_float(self, tiny_reg_model):
        """Regression labels are converted to float (even if int64 from loader)."""
        config = _make_config(task_type=TaskType.REGRESSION)
        strategy = RegressionStrategy(config, DEVICE)

        # Provide integer targets (simulating misconfigured loader)
        torch.manual_seed(42)
        features = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_FEATURES)
        targets = torch.randint(0, 10, (BATCH_SIZE,))  # int64
        batch = (features, targets)

        # Should not crash — strategy converts to float internally
        result = strategy.process_batch(tiny_reg_model, batch)
        assert result.loss.requires_grad

    def test_aggregate_no_accuracy(self):
        """Regression aggregate_epoch_metrics has no accuracy key."""
        config = _make_config(task_type=TaskType.REGRESSION)
        strategy = RegressionStrategy(config, DEVICE)

        results = [
            BatchResult(loss=torch.tensor(1.0), batch_size=4, metrics={"loss": 1.0}),
            BatchResult(loss=torch.tensor(2.0), batch_size=4, metrics={"loss": 2.0}),
        ]
        metrics = strategy.aggregate_epoch_metrics(results, total_samples=8)
        assert "train_loss" in metrics
        assert "train_accuracy" not in metrics, (
            "Regression should not report accuracy"
        )

    def test_validate_returns_regression_metric_keys(self, tiny_reg_model, reg_loader):
        """validate() returns val_loss, val_r2, val_ic, val_mae, val_rmse, val_da."""
        config = _make_config(task_type=TaskType.REGRESSION)
        strategy = RegressionStrategy(config, DEVICE)

        metrics = strategy.validate(tiny_reg_model, reg_loader)
        expected_keys = {"val_loss", "val_r2", "val_ic", "val_mae", "val_rmse",
                         "val_directional_accuracy"}
        missing = expected_keys - set(metrics.keys())
        assert not missing, f"Missing regression metric keys: {missing}"

    def test_validate_empty_loader_returns_inf(self, tiny_reg_model):
        """validate() with empty loader returns {'val_loss': inf}."""
        config = _make_config(task_type=TaskType.REGRESSION)
        strategy = RegressionStrategy(config, DEVICE)

        empty_loader = DataLoader(
            TensorDataset(
                torch.randn(0, SEQ_LEN, NUM_FEATURES), torch.randn(0)
            ),
            batch_size=1,
        )
        metrics = strategy.validate(tiny_reg_model, empty_loader)
        assert metrics["val_loss"] == float("inf")

    def test_evaluate_returns_dict(self, tiny_reg_model, reg_loader):
        """evaluate() returns dict (not ClassificationMetrics)."""
        config = _make_config(task_type=TaskType.REGRESSION)
        strategy = RegressionStrategy(config, DEVICE)

        result = strategy.evaluate(tiny_reg_model, reg_loader, "test")
        assert isinstance(result, dict), (
            f"Regression evaluate should return dict, got {type(result).__name__}"
        )
        assert "r2" in result

    def test_predict_returns_array(self, tiny_reg_model, features):
        """predict() returns numpy array of continuous predictions [B]."""
        config = _make_config(task_type=TaskType.REGRESSION)
        strategy = RegressionStrategy(config, DEVICE)

        preds = strategy.predict(tiny_reg_model, features[:BATCH_SIZE])
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (BATCH_SIZE,)
        assert preds.dtype == np.float32 or preds.dtype == np.float64


# =============================================================================
# TestHMHPClassificationStrategy
# =============================================================================


class TestHMHPClassificationStrategy:
    """Tests for multi-horizon classification (HMHP model)."""

    def test_data_pipeline_properties(self):
        """HMHP clf: dict labels, horizon_idx=None (return all horizons)."""
        config = _make_config(
            model_type=ModelType.HMHP, hmhp_horizons=HORIZONS
        )
        strategy = HMHPClassificationStrategy(config, DEVICE)
        assert strategy.requires_dict_labels is True
        assert strategy.horizon_idx is None
        assert strategy.horizons == HORIZONS

    def test_process_batch_2_tuple(self, hmhp_model, hmhp_loader):
        """process_batch handles 2-tuple (features, labels_dict)."""
        config = _make_config(
            model_type=ModelType.HMHP, hmhp_horizons=HORIZONS
        )
        strategy = HMHPClassificationStrategy(config, DEVICE)

        batch = next(iter(hmhp_loader))
        assert len(batch) == 2, f"Expected 2-tuple, got {len(batch)}-tuple"

        result = strategy.process_batch(hmhp_model, batch)
        assert result.loss.requires_grad
        assert result.batch_size == BATCH_SIZE

    def test_agreement_metric_captured(self, hmhp_model, hmhp_loader):
        """process_batch captures agreement_mean from ModelOutput.agreement."""
        config = _make_config(
            model_type=ModelType.HMHP, hmhp_horizons=HORIZONS
        )
        strategy = HMHPClassificationStrategy(config, DEVICE)

        batch = next(iter(hmhp_loader))
        result = strategy.process_batch(hmhp_model, batch)

        assert "agreement_mean" in result.metrics, (
            f"Missing agreement_mean. Metrics: {result.metrics.keys()}"
        )
        assert 0.0 <= result.metrics["agreement_mean"] <= 1.0, (
            f"Agreement should be in [0,1], got {result.metrics['agreement_mean']}"
        )

    def test_validate_per_horizon_metrics(self, hmhp_model, hmhp_loader):
        """validate() returns per-horizon loss/accuracy + agreement + confirmation."""
        config = _make_config(
            model_type=ModelType.HMHP, hmhp_horizons=HORIZONS
        )
        strategy = HMHPClassificationStrategy(config, DEVICE)

        metrics = strategy.validate(hmhp_model, hmhp_loader)

        expected = {
            "val_loss", "val_accuracy",
            "val_agreement_ratio", "val_confirmation_score",
            "val_h10_loss", "val_h10_accuracy",
            "val_h20_loss", "val_h20_accuracy",
        }
        missing = expected - set(metrics.keys())
        assert not missing, f"Missing HMHP validation keys: {missing}"

    def test_evaluate_returns_classification_metrics(
        self, hmhp_model, hmhp_loader
    ):
        """evaluate() returns ClassificationMetrics with HMHP-specific extras."""
        config = _make_config(
            model_type=ModelType.HMHP, hmhp_horizons=HORIZONS
        )
        strategy = HMHPClassificationStrategy(config, DEVICE)

        result = strategy.evaluate(hmhp_model, hmhp_loader, "test")
        assert hasattr(result, "accuracy")
        assert hasattr(result, "strategy_metrics")
        assert "agreement_ratio" in result.strategy_metrics
        assert "confirmation_score" in result.strategy_metrics

    def test_predict_returns_ensemble_predictions(self, hmhp_model, features):
        """predict() uses ensemble (final) logits, not per-horizon."""
        config = _make_config(
            model_type=ModelType.HMHP, hmhp_horizons=HORIZONS
        )
        strategy = HMHPClassificationStrategy(config, DEVICE)

        preds = strategy.predict(hmhp_model, features[:BATCH_SIZE])
        assert preds.shape == (BATCH_SIZE,)
        assert set(preds).issubset({0, 1, 2}), (
            f"Predictions should be class indices {{0,1,2}}, got {set(preds)}"
        )


# =============================================================================
# TestHMHPRegressionStrategy
# =============================================================================


class TestHMHPRegressionStrategy:
    """Tests for multi-horizon regression (HMHP-R model)."""

    def test_data_pipeline_properties(self):
        """HMHP-R: dict labels + regression targets required, horizon_idx=None."""
        config = _make_config(
            model_type=ModelType.HMHP_REGRESSION, hmhp_horizons=HORIZONS
        )
        strategy = HMHPRegressionStrategy(config, DEVICE)
        assert strategy.requires_dict_labels is True
        assert strategy.requires_regression_targets is True
        assert strategy.horizon_idx is None

    def test_process_batch_raises_on_2_tuple(self, hmhp_reg_model, hmhp_loader):
        """process_batch raises ValueError when regression targets are missing.

        Unlike other strategies, HMHP-R REQUIRES a 3-tuple batch.
        """
        config = _make_config(
            model_type=ModelType.HMHP_REGRESSION, hmhp_horizons=HORIZONS
        )
        strategy = HMHPRegressionStrategy(config, DEVICE)

        batch = next(iter(hmhp_loader))  # 2-tuple (no reg targets)
        with pytest.raises(ValueError, match="regression targets"):
            strategy.process_batch(hmhp_reg_model, batch)

    def test_process_batch_3_tuple(self, hmhp_reg_model, hmhp_reg_loader):
        """process_batch works with 3-tuple (features, labels_dict, reg_targets_dict)."""
        config = _make_config(
            model_type=ModelType.HMHP_REGRESSION, hmhp_horizons=HORIZONS
        )
        strategy = HMHPRegressionStrategy(config, DEVICE)

        batch = next(iter(hmhp_reg_loader))
        assert len(batch) == 3, f"Expected 3-tuple, got {len(batch)}-tuple"

        result = strategy.process_batch(hmhp_reg_model, batch)
        assert result.loss.requires_grad
        assert "H10_loss" in result.metrics or "loss" in result.metrics

    def test_validate_primary_horizon_metrics(
        self, hmhp_reg_model, hmhp_reg_loader
    ):
        """validate() surfaces primary horizon metrics without prefix (for early stopping)."""
        config = _make_config(
            model_type=ModelType.HMHP_REGRESSION, hmhp_horizons=HORIZONS
        )
        strategy = HMHPRegressionStrategy(config, DEVICE)

        metrics = strategy.validate(hmhp_reg_model, hmhp_reg_loader)

        # Primary horizon (H10) metrics at top level
        assert "val_loss" in metrics
        assert "val_r2" in metrics, f"Missing val_r2 (primary horizon). Keys: {sorted(metrics.keys())}"
        assert "val_ic" in metrics

        # Per-horizon metrics with prefix
        assert "val_h10_r2" in metrics
        assert "val_h20_r2" in metrics

    def test_evaluate_returns_dict_with_per_horizon(
        self, hmhp_reg_model, hmhp_reg_loader
    ):
        """evaluate() returns dict with both primary and per-horizon metrics."""
        config = _make_config(
            model_type=ModelType.HMHP_REGRESSION, hmhp_horizons=HORIZONS
        )
        strategy = HMHPRegressionStrategy(config, DEVICE)

        result = strategy.evaluate(hmhp_reg_model, hmhp_reg_loader, "test")
        assert isinstance(result, dict)
        assert "r2" in result, "Missing primary horizon r2"
        assert "h10_r2" in result, "Missing per-horizon h10_r2"
        assert "h20_r2" in result, "Missing per-horizon h20_r2"
