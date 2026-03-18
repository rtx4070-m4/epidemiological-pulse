"""
Epidemiological Pulse – ER Visit Forecasting
==============================================
Implements two forecasting models for ER visit prediction per region:

1. ProphetForecaster  – Facebook Prophet time-series model
2. LSTMForecaster     – PyTorch LSTM recurrent neural network

Both follow the same interface:
  .fit(train_df, zipcode)
  .predict(horizon_days) -> pd.DataFrame
  .evaluate(test_df) -> Dict[str, float]
"""

import logging
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.models.utils import (
    ModelPersistence,
    FeatureScaler,
    create_sequences,
    evaluate_predictions,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prophet Forecaster
# ---------------------------------------------------------------------------

class ProphetForecaster:
    """
    Per-region ER visit forecasting using Facebook Prophet.

    Prophet handles:
      - Trend changepoints
      - Weekly seasonality
      - Annual seasonality
      - External regressors (weather, AQI, pharmacy sales)
    """

    REGRESSORS = [
        "temperature_c",
        "aqi",
        "pharmacy_sales",
        "negative_sentiment",
    ]

    def __init__(self, config: Dict) -> None:
        """
        Args:
            config: Main params.yaml config dict.
        """
        self.config = config
        prophet_cfg = config.get("prophet", {})

        self.changepoint_prior_scale = prophet_cfg.get("changepoint_prior_scale", 0.05)
        self.seasonality_prior_scale = prophet_cfg.get("seasonality_prior_scale", 10.0)
        self.seasonality_mode = prophet_cfg.get("seasonality_mode", "multiplicative")
        self.yearly_seasonality = prophet_cfg.get("yearly_seasonality", True)
        self.weekly_seasonality = prophet_cfg.get("weekly_seasonality", True)
        self.daily_seasonality = prophet_cfg.get("daily_seasonality", False)
        self.interval_width = prophet_cfg.get("interval_width", 0.90)
        self.forecast_horizon = prophet_cfg.get("forecast_horizon_days", 30)
        self.model_save_dir = prophet_cfg.get("model_save_dir", "outputs/models/prophet")

        self.models: Dict[str, object] = {}   # zipcode → Prophet model
        self.fitted_regressors: Dict[str, List[str]] = {}  # zipcode → list of regressors used

    def _make_prophet_df(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Convert DataFrame to Prophet format (ds, y, regressors).

        Args:
            df: Time series DataFrame for one ZIP code.

        Returns:
            (prophet_df, regressor_cols)
        """
        prophet_df = pd.DataFrame()
        prophet_df["ds"] = pd.to_datetime(df["date"])
        prophet_df["y"] = df["er_visits"].values.astype(float)

        regressors_used = []
        for reg in self.REGRESSORS:
            if reg in df.columns and df[reg].notna().all():
                prophet_df[reg] = df[reg].values.astype(float)
                regressors_used.append(reg)

        return prophet_df, regressors_used

    def fit(self, train_df: pd.DataFrame, zipcode: str) -> None:
        """
        Fit Prophet model for a single region.

        Args:
            train_df: Training DataFrame (all regions or pre-filtered).
            zipcode: ZIP code to train for.
        """
        try:
            from prophet import Prophet
        except ImportError:
            logger.error("Prophet not installed. Run: pip install prophet")
            raise

        region_df = train_df[train_df["zipcode"].astype(str) == str(zipcode)].copy()
        region_df = region_df.sort_values("date").dropna(subset=["er_visits"])

        if len(region_df) < 30:
            logger.warning(
                f"[{zipcode}] Insufficient training data ({len(region_df)} rows). Skipping."
            )
            return

        prophet_df, regressors_used = self._make_prophet_df(region_df)

        model = Prophet(
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            interval_width=self.interval_width,
        )

        for reg in regressors_used:
            model.add_regressor(reg)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(prophet_df)

        self.models[str(zipcode)] = model
        self.fitted_regressors[str(zipcode)] = regressors_used
        logger.info(
            f"[{zipcode}] Prophet fitted: {len(region_df)} rows, "
            f"regressors={regressors_used}"
        )

    def fit_all(self, train_df: pd.DataFrame, zipcodes: Optional[List[str]] = None) -> None:
        """
        Fit Prophet models for all regions.

        Args:
            train_df: Full training DataFrame.
            zipcodes: Optional subset of ZIP codes.
        """
        if zipcodes is None:
            zipcodes = train_df["zipcode"].unique().tolist()

        logger.info(f"Fitting Prophet models for {len(zipcodes)} regions …")
        for zipcode in zipcodes:
            try:
                self.fit(train_df, zipcode)
            except Exception as exc:
                logger.error(f"[{zipcode}] Prophet fit failed: {exc}")

    def predict(
        self,
        zipcode: str,
        horizon_days: Optional[int] = None,
        future_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Generate forecast for a region.

        Args:
            zipcode: ZIP code to forecast.
            horizon_days: Number of future days to forecast.
            future_df: Optional DataFrame with future regressor values.

        Returns:
            Forecast DataFrame with columns: ds, yhat, yhat_lower, yhat_upper.

        Raises:
            KeyError: If the model for this ZIP hasn't been fitted.
        """
        if str(zipcode) not in self.models:
            raise KeyError(f"No fitted model for ZIP {zipcode}.")

        model = self.models[str(zipcode)]
        horizon = horizon_days or self.forecast_horizon
        regressors = self.fitted_regressors.get(str(zipcode), [])

        if future_df is not None:
            forecast_df = future_df.copy()
            forecast_df["ds"] = pd.to_datetime(forecast_df["date"])
        else:
            forecast_df = model.make_future_dataframe(periods=horizon, freq="D")
            # Fill regressors with last known value (simple persistence forecast)
            for reg in regressors:
                # Would ideally be filled with actual future weather/AQI forecasts
                forecast_df[reg] = 0.0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = model.predict(forecast_df)

        # Clip negative predictions
        forecast["yhat"] = forecast["yhat"].clip(lower=0)
        forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0)

        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    def evaluate(
        self,
        test_df: pd.DataFrame,
        zipcode: str,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.

        Args:
            test_df: Test DataFrame.
            zipcode: ZIP code.
            metrics: Metrics to compute.

        Returns:
            Dict of metric_name → value.
        """
        region_test = test_df[test_df["zipcode"].astype(str) == str(zipcode)].copy()
        region_test = region_test.sort_values("date").dropna(subset=["er_visits"])

        if region_test.empty:
            return {}

        future_df = pd.DataFrame({"date": region_test["date"]})
        for reg in self.fitted_regressors.get(str(zipcode), []):
            if reg in region_test.columns:
                future_df[reg] = region_test[reg].values

        forecast = self.predict(zipcode, future_df=future_df)
        y_true = region_test["er_visits"].values
        y_pred = forecast["yhat"].values[: len(y_true)]

        result = evaluate_predictions(y_true, y_pred, metrics)
        logger.info(f"[{zipcode}] Prophet eval: {result}")
        return result

    def save_all(self) -> None:
        """Save all fitted models to disk."""
        import pickle
        Path(self.model_save_dir).mkdir(parents=True, exist_ok=True)
        for zipcode, model in self.models.items():
            path = os.path.join(self.model_save_dir, f"prophet_{zipcode}.pkl")
            with open(path, "wb") as f:
                pickle.dump(model, f)
        logger.info(f"Saved {len(self.models)} Prophet models to {self.model_save_dir}")

    def load_all(self) -> None:
        """Load all Prophet models from disk."""
        import pickle
        if not os.path.exists(self.model_save_dir):
            raise FileNotFoundError(f"Prophet model dir not found: {self.model_save_dir}")
        for fname in os.listdir(self.model_save_dir):
            if fname.startswith("prophet_") and fname.endswith(".pkl"):
                zipcode = fname.replace("prophet_", "").replace(".pkl", "")
                path = os.path.join(self.model_save_dir, fname)
                with open(path, "rb") as f:
                    self.models[zipcode] = pickle.load(f)
        logger.info(f"Loaded {len(self.models)} Prophet models from {self.model_save_dir}")


# ---------------------------------------------------------------------------
# LSTM Forecaster
# ---------------------------------------------------------------------------

class LSTMForecaster:
    """
    Per-region ER visit forecasting using a PyTorch LSTM.

    Architecture:
      Input → LSTM (2 layers) → Dropout → Linear → Output

    Trains one shared model across all regions (with region embedding optional).
    """

    def __init__(self, config: Dict) -> None:
        """
        Args:
            config: Main params.yaml config dict.
        """
        self.config = config
        lstm_cfg = config.get("lstm", {})

        self.enabled = lstm_cfg.get("enabled", True)
        self.lookback = lstm_cfg.get("lookback_window", 14)
        self.hidden_units = lstm_cfg.get("hidden_units", 64)
        self.dropout_rate = lstm_cfg.get("dropout_rate", 0.2)
        self.epochs = lstm_cfg.get("epochs", 50)
        self.batch_size = lstm_cfg.get("batch_size", 32)
        self.lr = lstm_cfg.get("learning_rate", 0.001)
        self.test_split = lstm_cfg.get("test_split", 0.2)
        self.model_save_dir = lstm_cfg.get("model_save_dir", "outputs/models/lstm")

        self.model = None
        self.scaler = FeatureScaler("standard")
        self.feature_cols: List[str] = []
        self.n_features: int = 0

    def _build_model(self, n_features: int):
        """Build the PyTorch LSTM model."""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            logger.error("PyTorch not installed. Run: pip install torch")
            raise

        class LSTMModel(nn.Module):
            def __init__(
                self,
                n_feat: int,
                hidden: int,
                dropout: float,
                n_layers: int = 2,
            ) -> None:
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=n_feat,
                    hidden_size=hidden,
                    num_layers=n_layers,
                    batch_first=True,
                    dropout=dropout if n_layers > 1 else 0.0,
                )
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(hidden, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                out = self.dropout(out[:, -1, :])
                return self.fc(out).squeeze(-1)

        model = LSTMModel(
            n_feat=n_features,
            hidden=self.hidden_units,
            dropout=self.dropout_rate,
        )
        return model

    def _prepare_sequences(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "er_visits",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare LSTM input sequences.

        Args:
            df: Feature DataFrame sorted by (zipcode, date).
            feature_cols: Columns to use as input features.
            target_col: Target column name.

        Returns:
            (X_sequences, y_sequences) numpy arrays.
        """
        all_X, all_y = [], []

        for zipcode in df["zipcode"].unique():
            z_df = df[df["zipcode"] == zipcode].sort_values("date").dropna(
                subset=feature_cols + [target_col]
            )
            if len(z_df) < self.lookback + 1:
                continue

            X_raw = z_df[feature_cols].values
            y_raw = z_df[target_col].values.reshape(-1, 1)
            data = np.hstack([X_raw, y_raw])

            X_seq, y_seq = create_sequences(data, self.lookback)
            all_X.append(X_seq)
            all_y.append(y_seq)

        if not all_X:
            raise ValueError("No valid sequences could be created.")

        return np.concatenate(all_X), np.concatenate(all_y)

    def fit(
        self,
        train_df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "er_visits",
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train the LSTM model.

        Args:
            train_df: Training DataFrame.
            feature_cols: Feature column names.
            target_col: Target column name.
            verbose: Print training progress.

        Returns:
            Training history dict with 'train_loss' and 'val_loss'.
        """
        if not self.enabled:
            logger.info("LSTM disabled in config. Skipping training.")
            return {}

        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            logger.error("PyTorch not installed.")
            raise

        logger.info("Training LSTM model …")
        self.feature_cols = feature_cols

        # Scale features
        all_data = train_df[feature_cols + [target_col]].dropna()
        scaled = self.scaler.fit_transform(all_data.values)
        scaled_df = train_df[["zipcode", "date"]].copy()
        scaled_df = scaled_df.join(
            pd.DataFrame(scaled, columns=feature_cols + [target_col], index=all_data.index)
        )

        X, y = self._prepare_sequences(scaled_df.dropna(), feature_cols, target_col)
        self.n_features = X.shape[2]

        # Train/val split
        split = int(len(X) * (1 - self.test_split))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"  Training on: {device}")

        self.model = self._build_model(self.n_features).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), torch.FloatTensor(y_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        history = {"train_loss": [], "val_loss": []}

        for epoch in range(self.epochs):
            self.model.train()
            train_losses = []
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                preds = self.model(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())

            # Validation
            self.model.eval()
            with torch.no_grad():
                X_val_t = torch.FloatTensor(X_val).to(device)
                y_val_t = torch.FloatTensor(y_val).to(device)
                val_preds = self.model(X_val_t)
                val_loss = criterion(val_preds, y_val_t).item()

            train_loss = np.mean(train_losses)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if verbose and (epoch + 1) % 10 == 0:
                logger.info(
                    f"  Epoch [{epoch+1}/{self.epochs}] "
                    f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
                )

        logger.info(
            f"LSTM training complete: "
            f"final train_loss={history['train_loss'][-1]:.4f}, "
            f"val_loss={history['val_loss'][-1]:.4f}"
        )
        return history

    def predict_region(
        self,
        df: pd.DataFrame,
        zipcode: str,
        horizon_days: int = 30,
    ) -> pd.DataFrame:
        """
        Generate multi-step forecast for a region.

        Uses recursive (one-step-ahead) prediction strategy.

        Args:
            df: Historical DataFrame.
            zipcode: ZIP code to forecast.
            horizon_days: Number of future days.

        Returns:
            DataFrame with date, yhat columns.
        """
        if self.model is None:
            raise RuntimeError("LSTM model not fitted. Call fit() first.")

        try:
            import torch
        except ImportError:
            raise

        region_df = df[df["zipcode"].astype(str) == str(zipcode)].copy()
        region_df = region_df.sort_values("date").dropna(subset=self.feature_cols)

        if len(region_df) < self.lookback:
            raise ValueError(
                f"[{zipcode}] Not enough history ({len(region_df)}) "
                f"for lookback={self.lookback}"
            )

        # Scale the last `lookback` rows
        last_rows = region_df.tail(self.lookback)[self.feature_cols].values
        last_target = region_df.tail(self.lookback)["er_visits"].values.reshape(-1, 1)
        last_data = np.hstack([last_rows, last_target])
        scaled_last = self.scaler.transform(
            np.vstack([last_data, last_data])  # need full matrix for scaler
        )[:self.lookback]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()

        sequence = scaled_last[:, : self.n_features]  # features only
        predictions = []
        last_date = region_df["date"].max()

        for _ in range(horizon_days):
            X = torch.FloatTensor(sequence[-self.lookback:]).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_scaled = self.model(X).cpu().numpy()

            # Inverse scale the prediction (need full row for scaler)
            dummy_row = np.zeros((1, self.n_features + 1))
            dummy_row[0, -1] = pred_scaled[0]
            inv_row = self.scaler.scaler.inverse_transform(dummy_row)
            pred_raw = max(0.0, float(inv_row[0, -1]))
            predictions.append(pred_raw)

            # Append prediction as next step (recursive forecasting)
            next_feat = sequence[-1:].copy()
            next_feat[0, -1] = pred_scaled[0]
            sequence = np.vstack([sequence, next_feat])

        dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=horizon_days,
            freq="D",
        )
        return pd.DataFrame({"date": dates, "yhat": predictions, "zipcode": zipcode})

    def evaluate(
        self,
        test_df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "er_visits",
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate LSTM on test data.

        Args:
            test_df: Test DataFrame.
            feature_cols: Feature columns.
            target_col: Target column.
            metrics: Metric names.

        Returns:
            Dict of metric → value.
        """
        if self.model is None:
            return {}

        try:
            import torch
        except ImportError:
            raise

        all_X, all_y = self._prepare_sequences(test_df, feature_cols, target_col)
        if len(all_X) == 0:
            return {}

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()

        with torch.no_grad():
            X_t = torch.FloatTensor(all_X).to(device)
            preds = self.model(X_t).cpu().numpy()

        result = evaluate_predictions(all_y, preds, metrics)
        logger.info(f"LSTM eval: {result}")
        return result

    def save(self) -> None:
        """Save LSTM model weights and scaler."""
        if self.model is None:
            raise RuntimeError("No model to save.")
        try:
            import torch
            Path(self.model_save_dir).mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), os.path.join(self.model_save_dir, "lstm_weights.pt"))
            self.scaler.save(os.path.join(self.model_save_dir, "lstm_scaler.joblib"))
            logger.info(f"LSTM model saved to {self.model_save_dir}")
        except Exception as exc:
            logger.error(f"Failed to save LSTM: {exc}")

    def load(self) -> None:
        """Load LSTM model weights and scaler."""
        try:
            import torch
            weights_path = os.path.join(self.model_save_dir, "lstm_weights.pt")
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"LSTM weights not found: {weights_path}")
            if self.model is None:
                self.model = self._build_model(self.n_features)
            self.model.load_state_dict(torch.load(weights_path, map_location="cpu"))
            self.scaler.load(os.path.join(self.model_save_dir, "lstm_scaler.joblib"))
            logger.info(f"LSTM model loaded from {self.model_save_dir}")
        except Exception as exc:
            logger.error(f"Failed to load LSTM: {exc}")


# ---------------------------------------------------------------------------
# Forecasting Pipeline
# ---------------------------------------------------------------------------

class ForecastingPipeline:
    """Orchestrates Prophet + LSTM forecasting for all regions."""

    def __init__(self, config: Dict) -> None:
        self.config = config
        self.prophet = ProphetForecaster(config)
        self.lstm = LSTMForecaster(config)

    def run(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "er_visits",
        zipcodes: Optional[List[str]] = None,
    ) -> Dict:
        """
        Full forecasting pipeline: train + evaluate both models.

        Args:
            train_df: Training DataFrame.
            test_df: Test DataFrame.
            feature_cols: Feature column names for LSTM.
            target_col: Target column name.
            zipcodes: Subset of ZIP codes.

        Returns:
            Results dict with model forecasts and evaluation metrics.
        """
        if zipcodes is None:
            zipcodes = train_df["zipcode"].unique().tolist()

        results = {
            "prophet_forecasts": {},
            "lstm_forecasts": {},
            "prophet_metrics": {},
            "lstm_metrics": {},
        }

        # --- Prophet ---
        logger.info("Training Prophet models …")
        self.prophet.fit_all(train_df, zipcodes)

        for zipcode in zipcodes:
            if str(zipcode) not in self.prophet.models:
                continue
            try:
                forecast = self.prophet.predict(str(zipcode))
                results["prophet_forecasts"][str(zipcode)] = forecast
                metrics = self.prophet.evaluate(test_df, str(zipcode))
                results["prophet_metrics"][str(zipcode)] = metrics
            except Exception as exc:
                logger.error(f"[{zipcode}] Prophet predict/eval failed: {exc}")

        self.prophet.save_all()

        # --- LSTM ---
        if self.config.get("lstm", {}).get("enabled", True):
            logger.info("Training LSTM model …")
            try:
                history = self.lstm.fit(train_df, feature_cols, target_col)
                metrics = self.lstm.evaluate(test_df, feature_cols, target_col)
                results["lstm_metrics"]["global"] = metrics
                results["lstm_history"] = history
                self.lstm.save()
            except Exception as exc:
                logger.error(f"LSTM training/eval failed: {exc}")
        else:
            logger.info("LSTM disabled in config.")

        # Summary
        if results["prophet_metrics"]:
            avg_mae = np.mean([m.get("mae", 0) for m in results["prophet_metrics"].values()])
            avg_rmse = np.mean([m.get("rmse", 0) for m in results["prophet_metrics"].values()])
            logger.info(f"Prophet avg MAE={avg_mae:.2f}, avg RMSE={avg_rmse:.2f}")

        return results
