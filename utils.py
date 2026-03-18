"""
Epidemiological Pulse – Model Utilities
=========================================
Shared utilities for model training, evaluation, persistence, and reporting.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------------------------------

def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(mean_absolute_error(y_true, y_pred))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1.0) -> float:
    """
    Mean Absolute Percentage Error with epsilon to avoid division by zero.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        epsilon: Small constant added to denominator.

    Returns:
        MAPE as a percentage (0–100).
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute multiple evaluation metrics.

    Args:
        y_true: Ground truth array.
        y_pred: Prediction array.
        metrics: List of metric names. Defaults to ['mae', 'rmse', 'mape'].

    Returns:
        Dict mapping metric name → value.
    """
    metrics = metrics or ["mae", "rmse", "mape"]
    results: Dict[str, float] = {}

    metric_fns = {
        "mae": compute_mae,
        "rmse": compute_rmse,
        "mape": compute_mape,
    }

    for m in metrics:
        if m in metric_fns:
            results[m] = round(metric_fns[m](y_true, y_pred), 4)
        else:
            logger.warning(f"Unknown metric: {m}")

    return results


# ---------------------------------------------------------------------------
# Model Persistence
# ---------------------------------------------------------------------------

class ModelPersistence:
    """Handles saving and loading of trained models."""

    @staticmethod
    def save(model: Any, path: str) -> str:
        """
        Save a model (sklearn, scaler, etc.) using joblib.

        Args:
            model: Model object to save.
            path: File path (with .joblib extension recommended).

        Returns:
            Absolute path where model was saved.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
        size_mb = os.path.getsize(path) / 1e6
        logger.info(f"Model saved: {path} ({size_mb:.2f} MB)")
        return os.path.abspath(path)

    @staticmethod
    def load(path: str) -> Any:
        """
        Load a model from disk.

        Args:
            path: File path of saved model.

        Returns:
            Loaded model object.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        model = joblib.load(path)
        logger.info(f"Model loaded: {path}")
        return model

    @staticmethod
    def save_metadata(metadata: Dict, path: str) -> None:
        """Save model metadata as JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Metadata saved: {path}")

    @staticmethod
    def load_metadata(path: str) -> Dict:
        """Load model metadata from JSON."""
        with open(path) as f:
            return json.load(f)


# ---------------------------------------------------------------------------
# Feature Scalers
# ---------------------------------------------------------------------------

class FeatureScaler:
    """Wraps StandardScaler / MinMaxScaler with fit_transform/transform."""

    def __init__(self, method: str = "standard") -> None:
        """
        Args:
            method: 'standard' (zero mean, unit variance) or 'minmax' ([0,1]).
        """
        if method == "standard":
            self.scaler = StandardScaler()
        elif method == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler method: {method}")
        self.method = method
        self._fitted = False

    def fit_transform(
        self, X: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """Fit scaler and transform training data."""
        result = self.scaler.fit_transform(X)
        self._fitted = True
        return result

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Transform new data using fitted scaler."""
        if not self._fitted:
            raise RuntimeError("Scaler has not been fitted yet. Call fit_transform first.")
        return self.scaler.transform(X)

    def inverse_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Inverse transform predictions back to original scale."""
        return self.scaler.inverse_transform(X)

    def save(self, path: str) -> None:
        ModelPersistence.save(self.scaler, path)

    def load(self, path: str) -> None:
        self.scaler = ModelPersistence.load(path)
        self._fitted = True


# ---------------------------------------------------------------------------
# Data Preparation for ML
# ---------------------------------------------------------------------------

def prepare_ml_arrays(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract feature matrix X and target vector y from a DataFrame.

    Args:
        df: Feature-engineered DataFrame.
        feature_cols: List of feature column names.
        target_col: Target column name.

    Returns:
        (X, y) as numpy arrays.
    """
    # Ensure all feature cols are present
    available_cols = [c for c in feature_cols if c in df.columns]
    missing = set(feature_cols) - set(available_cols)
    if missing:
        logger.warning(f"Missing feature columns: {missing}")

    X = df[available_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)

    # Replace NaNs with column means (should be clean by this stage)
    col_means = np.nanmean(X, axis=0)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    return X, y


def create_sequences(
    data: np.ndarray,
    lookback: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create (X, y) sequences for LSTM training.

    Args:
        data: 2D array of shape (n_timesteps, n_features).
              Last column is assumed to be the target.
        lookback: Number of past timesteps per sample.

    Returns:
        X: shape (n_samples, lookback, n_features - 1)
        y: shape (n_samples,)
    """
    X_seqs, y_seqs = [], []
    n_features = data.shape[1] - 1  # exclude target (last col)

    for i in range(lookback, len(data)):
        X_seqs.append(data[i - lookback:i, :n_features])
        y_seqs.append(data[i, -1])

    return np.array(X_seqs, dtype=np.float32), np.array(y_seqs, dtype=np.float32)


# ---------------------------------------------------------------------------
# Evaluation Report
# ---------------------------------------------------------------------------

class EvaluationReporter:
    """Aggregates evaluation results across multiple models/regions."""

    def __init__(self) -> None:
        self.results: List[Dict] = []

    def add(
        self,
        model_name: str,
        region: str,
        metrics: Dict[str, float],
        n_samples: int,
        extra: Optional[Dict] = None,
    ) -> None:
        """
        Add evaluation result.

        Args:
            model_name: Name of the model (e.g., 'prophet', 'lstm').
            region: ZIP code or region identifier.
            metrics: Dict of metric_name → value.
            n_samples: Number of test samples.
            extra: Optional additional metadata.
        """
        record = {
            "model": model_name,
            "region": region,
            "n_samples": n_samples,
            **metrics,
            **(extra or {}),
        }
        self.results.append(record)

    def summary(self) -> pd.DataFrame:
        """Return summary DataFrame of all results."""
        df = pd.DataFrame(self.results)
        if df.empty:
            return df
        numeric_cols = [c for c in df.columns if c not in ["model", "region"]]
        return df.groupby("model")[numeric_cols].mean().round(4).reset_index()

    def save(self, path: str) -> None:
        """Save full results to JSON."""
        report = {
            "per_region": self.results,
            "summary": self.summary().to_dict(orient="records"),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Evaluation report saved: {path}")


# ---------------------------------------------------------------------------
# Risk Level Mapping
# ---------------------------------------------------------------------------

def score_to_risk_level(
    score: float,
    low_threshold: float = 0.33,
    medium_threshold: float = 0.66,
) -> str:
    """
    Convert a [0, 1] risk score to a categorical risk level.

    Args:
        score: Risk score between 0 and 1.
        low_threshold: Upper bound for 'Low' risk.
        medium_threshold: Upper bound for 'Medium' risk.

    Returns:
        Risk level string: 'Low', 'Medium', or 'High'.
    """
    if score < low_threshold:
        return "Low"
    elif score < medium_threshold:
        return "Medium"
    else:
        return "High"


RISK_COLORS = {
    "Low": "#2ecc71",       # green
    "Medium": "#f39c12",    # orange
    "High": "#e74c3c",      # red
}


def get_risk_color(risk_level: str) -> str:
    """Return color hex for a risk level string."""
    return RISK_COLORS.get(risk_level, "#95a5a6")
