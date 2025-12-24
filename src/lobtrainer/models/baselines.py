"""
Baseline models for LOB price prediction.

These baselines establish the performance floor that any ML model must beat
to demonstrate real predictive value.

Key insight from analysis:
- Label lag-1 ACF = 0.92 (high persistence due to smoothing)
- Transition matrix diagonals: ~0.92 for UP/DOWN, ~0.76 for STABLE
- A naive "predict previous label" baseline achieves ~76% accuracy

Design principles (RULE.md):
- No magic numbers - all parameters are explicit
- Deterministic - same inputs produce same outputs
- Documented - explain what each baseline does and why
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import logging

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# =============================================================================
# Base Model Interface
# =============================================================================


class BaseModel(ABC):
    """
    Abstract base class for all models.
    
    All models must implement:
    - fit(X, y): Train the model
    - predict(X): Make predictions
    - predict_proba(X): Get class probabilities (if supported)
    """
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseModel":
        """
        Train the model.
        
        Args:
            X: Features array of shape (n_samples, n_features) or (n_samples, seq_len, n_features)
            y: Labels array of shape (n_samples,)
        
        Returns:
            self
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features array
        
        Returns:
            Predicted labels of shape (n_samples,)
        """
        pass
    
    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        Get class probabilities (if supported).
        
        Args:
            X: Features array
        
        Returns:
            Probabilities of shape (n_samples, n_classes), or None if not supported
        """
        return None
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Model name for logging and reporting."""
        pass


# =============================================================================
# Naive Baselines
# =============================================================================


class NaivePreviousLabel(BaseModel):
    """
    Baseline: Predict the previous label.
    
    This baseline exploits the high autocorrelation in labels (lag-1 ACF = 0.92).
    It predicts that each sample's label equals the previous sample's label.
    
    Expected accuracy: ~76% (based on transition matrix diagonal average)
    
    Use case:
    - Establishes the performance floor due to label persistence
    - Any model that doesn't significantly beat this is just exploiting autocorrelation
    
    Note:
    - For the first sample, uses the class prior (most common class)
    - Works on temporally ordered data only
    """
    
    def __init__(self):
        self._class_prior: Optional[int] = None
        self._fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "NaivePreviousLabel":
        """
        Fit by computing class prior for first sample.
        
        Args:
            X: Features (ignored, only shape used)
            y: Labels for computing class prior
        
        Returns:
            self
        """
        # Compute most common class for first sample
        unique, counts = np.unique(y, return_counts=True)
        self._class_prior = unique[np.argmax(counts)]
        self._fitted = True
        
        logger.info(f"NaivePreviousLabel: class_prior={self._class_prior}")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict: each sample's label equals the previous label.
        
        Args:
            X: Features of shape (n_samples, ...) - only shape used
        
        Returns:
            Predictions where pred[i] = y[i-1] (using class_prior for i=0)
        
        Note: This requires the TRUE LABELS to make predictions, which makes
              it a "cheating" baseline for benchmarking purposes only.
              In practice, use NaiveClassPrior for a proper baseline.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # This baseline is special - it needs the previous label
        # For proper evaluation, we need to shift labels
        n_samples = X.shape[0]
        return np.full(n_samples, self._class_prior, dtype=np.int64)
    
    def predict_with_history(self, X: np.ndarray, y_history: np.ndarray) -> np.ndarray:
        """
        Predict using actual label history.
        
        This is the "true" naive baseline that predicts prev_label.
        
        Args:
            X: Features (shape used only)
            y_history: True labels from previous timestep
        
        Returns:
            Predictions where pred[i] = y_history[i]
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Simply return the history (previous labels)
        return y_history.copy()
    
    @property
    def name(self) -> str:
        return "NaivePreviousLabel"


class NaiveClassPrior(BaseModel):
    """
    Baseline: Always predict the most common class.
    
    This is the simplest baseline - it ignores features entirely and
    always predicts the class with highest prior probability.
    
    Expected accuracy: ~37% (for balanced 3-class problem)
    
    Use case:
    - Establishes the absolute minimum (random guessing floor)
    - Any model must beat this to show any signal learning
    """
    
    def __init__(self):
        self._class_prior: Optional[int] = None
        self._class_probs: Optional[np.ndarray] = None
        self._fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "NaiveClassPrior":
        """
        Fit by computing class prior.
        
        Args:
            X: Features (ignored)
            y: Labels for computing class distribution
        
        Returns:
            self
        """
        unique, counts = np.unique(y, return_counts=True)
        self._class_prior = unique[np.argmax(counts)]
        self._class_probs = counts / counts.sum()
        self._classes = unique
        self._fitted = True
        
        logger.info(
            f"NaiveClassPrior: prior={self._class_prior}, "
            f"probs={dict(zip(unique, self._class_probs))}"
        )
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict: always return the most common class.
        
        Args:
            X: Features of shape (n_samples, ...)
        
        Returns:
            Array filled with class prior
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        n_samples = X.shape[0]
        return np.full(n_samples, self._class_prior, dtype=np.int64)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return class prior probabilities for all samples.
        
        Args:
            X: Features of shape (n_samples, ...)
        
        Returns:
            Array of shape (n_samples, n_classes) with class priors
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        n_samples = X.shape[0]
        return np.tile(self._class_probs, (n_samples, 1))
    
    @property
    def name(self) -> str:
        return "NaiveClassPrior"


# =============================================================================
# Statistical Baselines
# =============================================================================


@dataclass
class LogisticBaselineConfig:
    """Configuration for LogisticBaseline."""
    
    C: float = 1.0
    """Inverse regularization strength. Smaller = stronger regularization."""
    
    max_iter: int = 1000
    """Maximum iterations for solver convergence."""
    
    solver: str = "lbfgs"
    """Optimization solver: 'lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'."""
    
    class_weight: Optional[str] = "balanced"
    """Class weighting: None, 'balanced', or dict."""
    
    normalize: bool = True
    """Whether to standardize features before fitting."""
    
    random_state: int = 42
    """Random seed for reproducibility."""


class LogisticBaseline(BaseModel):
    """
    Logistic Regression baseline for LOB price prediction.
    
    This provides a linear baseline that:
    - Uses all features (or selected subset)
    - Applies standard scaling
    - Handles class imbalance
    
    Expected accuracy: Should beat class prior but may struggle to beat
    naive previous-label baseline due to high label autocorrelation.
    
    Use case:
    - Simple, interpretable baseline
    - Feature importance via coefficients
    - Fast training and inference
    """
    
    def __init__(self, config: Optional[LogisticBaselineConfig] = None):
        self.config = config or LogisticBaselineConfig()
        
        self._model = LogisticRegression(
            C=self.config.C,
            max_iter=self.config.max_iter,
            solver=self.config.solver,
            class_weight=self.config.class_weight,
            random_state=self.config.random_state,
            n_jobs=-1,
        )
        self._scaler = StandardScaler() if self.config.normalize else None
        self._fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticBaseline":
        """
        Fit logistic regression model.
        
        Args:
            X: Features of shape (n_samples, n_features)
            y: Labels of shape (n_samples,)
        
        Returns:
            self
        """
        # Ensure 2D features
        if X.ndim == 3:
            # For sequence data, use last timestep
            X = X[:, -1, :]
            logger.info(f"LogisticBaseline: Using last timestep, shape={X.shape}")
        
        # Handle NaN/Inf
        valid_mask = np.isfinite(X).all(axis=1)
        if not valid_mask.all():
            n_invalid = (~valid_mask).sum()
            logger.warning(f"Removing {n_invalid} samples with NaN/Inf")
            X = X[valid_mask]
            y = y[valid_mask]
        
        # Standardize
        if self._scaler is not None:
            X = self._scaler.fit_transform(X)
        
        # Fit model
        self._model.fit(X, y)
        self._fitted = True
        
        logger.info(
            f"LogisticBaseline: fitted on {len(y)} samples, "
            f"n_features={X.shape[1]}"
        )
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels.
        
        Args:
            X: Features of shape (n_samples, n_features)
        
        Returns:
            Predicted labels
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Ensure 2D
        if X.ndim == 3:
            X = X[:, -1, :]
        
        # Handle NaN/Inf (replace with 0)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Standardize
        if self._scaler is not None:
            X = self._scaler.transform(X)
        
        return self._model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get class probabilities.
        
        Args:
            X: Features of shape (n_samples, n_features)
        
        Returns:
            Probabilities of shape (n_samples, n_classes)
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Ensure 2D
        if X.ndim == 3:
            X = X[:, -1, :]
        
        # Handle NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Standardize
        if self._scaler is not None:
            X = self._scaler.transform(X)
        
        return self._model.predict_proba(X)
    
    @property
    def name(self) -> str:
        return "LogisticRegression"
    
    @property
    def coef_(self) -> np.ndarray:
        """Model coefficients (feature importances)."""
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        return self._model.coef_
    
    @property
    def feature_importance(self) -> np.ndarray:
        """
        Feature importance as mean absolute coefficient across classes.
        
        Returns:
            Array of shape (n_features,) with importance scores
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        return np.abs(self._model.coef_).mean(axis=0)

