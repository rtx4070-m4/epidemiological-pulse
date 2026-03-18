"""
Epidemiological Pulse – Feature Engineering
============================================
Builds ML-ready features from the cleaned master dataset.

Features generated:
  - Temporal: day_of_week, month, is_weekend, season, week_of_year
  - Lag variables: pharmacy_sales_lag_N, er_visits_lag_N
  - Rolling statistics: rolling_mean_N, rolling_std_N, rolling_max_N
  - Rate-of-change: pct_change_N
  - Composite scores: disease_risk_score
  - Anomaly flags: z-score and STL decomposition
  - Interaction features: temp × aqi, sentiment × pharmacy
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.seasonal import STL

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Temporal Features
# ---------------------------------------------------------------------------

class TemporalFeatureBuilder:
    """Extracts calendar-based temporal features from date column."""

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal features.

        Args:
            df: DataFrame with 'date' column.

        Returns:
            DataFrame with temporal feature columns added.
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

        df["day_of_week"] = df["date"].dt.dayofweek       # 0=Mon, 6=Sun
        df["day_of_year"] = df["date"].dt.dayofyear
        df["month"] = df["date"].dt.month
        df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
        df["year"] = df["date"].dt.year
        df["quarter"] = df["date"].dt.quarter
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

        # Season: 1=Winter, 2=Spring, 3=Summer, 4=Fall
        df["season"] = df["month"].map(
            {
                12: 1, 1: 1, 2: 1,   # Winter
                3: 2, 4: 2, 5: 2,    # Spring
                6: 3, 7: 3, 8: 3,    # Summer
                9: 4, 10: 4, 11: 4,  # Fall
            }
        )

        # Cyclical encoding for day_of_week and month (prevents ordinal artifacts)
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        logger.debug("Temporal features built.")
        return df


# ---------------------------------------------------------------------------
# Lag Features
# ---------------------------------------------------------------------------

class LagFeatureBuilder:
    """Creates lagged versions of key time-series signals."""

    def __init__(self, lag_days: List[int], target_cols: Optional[List[str]] = None) -> None:
        """
        Args:
            lag_days: List of lag periods in days (e.g., [1, 3, 7, 14]).
            target_cols: Columns to lag. Defaults to main epidemiological signals.
        """
        self.lag_days = lag_days
        self.target_cols = target_cols or [
            "pharmacy_sales",
            "negative_sentiment",
            "aqi",
            "er_visits",
            "temperature_c",
        ]

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add lag features per ZIP code.

        Args:
            df: DataFrame sorted by (zipcode, date).

        Returns:
            DataFrame with lag feature columns.
        """
        df = df.copy().sort_values(["zipcode", "date"])
        cols_present = [c for c in self.target_cols if c in df.columns]

        for col in cols_present:
            for lag in self.lag_days:
                lag_col = f"{col}_lag_{lag}"
                df[lag_col] = (
                    df.groupby("zipcode")[col]
                    .shift(lag)
                )
                logger.debug(f"  Added lag feature: {lag_col}")

        logger.debug(f"Lag features built: {len(cols_present)} cols × {len(self.lag_days)} lags")
        return df


# ---------------------------------------------------------------------------
# Rolling Statistics Features
# ---------------------------------------------------------------------------

class RollingFeatureBuilder:
    """Computes rolling window statistics."""

    def __init__(
        self,
        windows: List[int],
        target_cols: Optional[List[str]] = None,
        stats_to_compute: Optional[List[str]] = None,
    ) -> None:
        """
        Args:
            windows: Rolling window sizes in days (e.g., [3, 7, 14, 30]).
            target_cols: Columns to apply rolling stats.
            stats_to_compute: Stats to compute. Options: mean, std, max, min, sum.
        """
        self.windows = windows
        self.target_cols = target_cols or [
            "pharmacy_sales",
            "negative_sentiment",
            "er_visits",
            "aqi",
            "temperature_c",
        ]
        self.stats = stats_to_compute or ["mean", "std", "max"]

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add rolling statistics features.

        Args:
            df: DataFrame sorted by (zipcode, date).

        Returns:
            DataFrame with rolling feature columns.
        """
        df = df.copy().sort_values(["zipcode", "date"])
        cols_present = [c for c in self.target_cols if c in df.columns]

        for col in cols_present:
            grouped = df.groupby("zipcode")[col]
            for window in self.windows:
                for stat in self.stats:
                    feat_col = f"{col}_roll_{window}_{stat}"
                    if stat == "mean":
                        df[feat_col] = grouped.transform(
                            lambda x, w=window: x.rolling(w, min_periods=1).mean()
                        )
                    elif stat == "std":
                        df[feat_col] = grouped.transform(
                            lambda x, w=window: x.rolling(w, min_periods=2).std().fillna(0)
                        )
                    elif stat == "max":
                        df[feat_col] = grouped.transform(
                            lambda x, w=window: x.rolling(w, min_periods=1).max()
                        )
                    elif stat == "min":
                        df[feat_col] = grouped.transform(
                            lambda x, w=window: x.rolling(w, min_periods=1).min()
                        )
                    elif stat == "sum":
                        df[feat_col] = grouped.transform(
                            lambda x, w=window: x.rolling(w, min_periods=1).sum()
                        )
                    logger.debug(f"  Added rolling feature: {feat_col}")

        logger.debug(f"Rolling features built: {len(cols_present)} cols × {len(self.windows)} windows")
        return df


# ---------------------------------------------------------------------------
# Rate-of-Change Features
# ---------------------------------------------------------------------------

class RateOfChangeFeatureBuilder:
    """Computes percentage change features."""

    def __init__(
        self,
        periods: List[int] = None,
        target_cols: Optional[List[str]] = None,
    ) -> None:
        self.periods = periods or [1, 7]
        self.target_cols = target_cols or [
            "pharmacy_sales",
            "er_visits",
            "negative_sentiment",
            "aqi",
        ]

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add percentage change features."""
        df = df.copy().sort_values(["zipcode", "date"])
        cols_present = [c for c in self.target_cols if c in df.columns]

        for col in cols_present:
            for period in self.periods:
                feat_col = f"{col}_pct_change_{period}"
                df[feat_col] = (
                    df.groupby("zipcode")[col]
                    .transform(lambda x, p=period: x.pct_change(periods=p).fillna(0))
                )
                # Clip extreme pct changes
                df[feat_col] = df[feat_col].clip(-5, 5)

        return df


# ---------------------------------------------------------------------------
# Anomaly Score Features
# ---------------------------------------------------------------------------

class AnomalyFeatureBuilder:
    """Detects anomalies using Z-score and STL decomposition."""

    def __init__(
        self,
        config: Dict,
        use_stl: bool = False,
    ) -> None:
        """
        Args:
            config: Main config dict.
            use_stl: Use STL decomposition (slower but more accurate).
        """
        self.config = config
        self.use_stl = use_stl
        anomaly_cfg = config.get("anomaly", {})
        self.zscore_window = anomaly_cfg.get("zscore_window", 30)
        self.zscore_threshold = anomaly_cfg.get("zscore_threshold", 2.5)
        self.stl_period = config.get("features", {}).get("stl_period", 7)
        self.cols_to_check = anomaly_cfg.get(
            "columns_to_check",
            ["pharmacy_sales", "negative_sentiment", "er_visits", "aqi"],
        )

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add anomaly score features.

        For each tracked column, adds:
          {col}_zscore: rolling Z-score
          {col}_anomaly: binary flag

        Args:
            df: DataFrame sorted by (zipcode, date).

        Returns:
            DataFrame with anomaly feature columns.
        """
        df = df.copy().sort_values(["zipcode", "date"])
        cols_present = [c for c in self.cols_to_check if c in df.columns]

        for col in cols_present:
            if self.use_stl:
                df = self._stl_anomaly(df, col)
            else:
                df = self._zscore_anomaly(df, col)

        return df

    def _zscore_anomaly(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Add rolling Z-score anomaly for a column."""
        window = self.zscore_window

        def rolling_zscore(x: pd.Series) -> pd.Series:
            roll_mean = x.rolling(window, min_periods=5).mean()
            roll_std = x.rolling(window, min_periods=5).std()
            z = (x - roll_mean) / (roll_std + 1e-8)
            return z

        df[f"{col}_zscore"] = (
            df.groupby("zipcode")[col]
            .transform(rolling_zscore)
            .fillna(0)
        )
        df[f"{col}_anomaly"] = (
            np.abs(df[f"{col}_zscore"]) > self.zscore_threshold
        ).astype(int)

        n_anomalies = df[f"{col}_anomaly"].sum()
        logger.debug(f"  {col}: {n_anomalies} anomalies detected (Z > {self.zscore_threshold})")
        return df

    def _stl_anomaly(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Add STL decomposition residual anomaly for a column."""
        period = self.stl_period

        def stl_residuals(x: pd.Series) -> pd.Series:
            if len(x) < period * 2:
                return pd.Series(np.zeros(len(x)), index=x.index)
            try:
                stl = STL(x, period=period, robust=True)
                result = stl.fit()
                return pd.Series(result.resid, index=x.index)
            except Exception:
                return pd.Series(np.zeros(len(x)), index=x.index)

        df[f"{col}_stl_resid"] = (
            df.groupby("zipcode")[col]
            .transform(stl_residuals)
            .fillna(0)
        )
        resid_std = df.groupby("zipcode")[f"{col}_stl_resid"].transform("std").fillna(1)
        df[f"{col}_zscore"] = df[f"{col}_stl_resid"] / (resid_std + 1e-8)
        df[f"{col}_anomaly"] = (
            np.abs(df[f"{col}_zscore"]) > self.zscore_threshold
        ).astype(int)

        return df


# ---------------------------------------------------------------------------
# Composite Risk Score
# ---------------------------------------------------------------------------

class RiskScoreBuilder:
    """
    Builds a composite disease risk score by combining multiple signals.

    Score = weighted combination of normalized:
      - pharmacy_sales (high = risk)
      - negative_sentiment (high = risk)
      - aqi (high = risk)
      - temperature extremity (hot/cold = risk)
      - anomaly flags
    """

    WEIGHTS = {
        "pharmacy_sales": 0.30,
        "negative_sentiment": 0.25,
        "aqi": 0.20,
        "temp_extreme": 0.15,
        "anomaly_count": 0.10,
    }

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute composite risk score [0, 1] per row.

        Args:
            df: Feature-engineered DataFrame.

        Returns:
            DataFrame with 'disease_risk_score' column added.
        """
        df = df.copy()
        components = {}

        # Normalize each component to [0, 1]
        if "pharmacy_sales" in df.columns:
            components["pharmacy_sales"] = self._minmax(df["pharmacy_sales"])

        if "negative_sentiment" in df.columns:
            components["negative_sentiment"] = self._minmax(df["negative_sentiment"])

        if "aqi" in df.columns:
            components["aqi"] = self._minmax(df["aqi"])

        if "temperature_c" in df.columns:
            temp_extreme = np.abs(df["temperature_c"] - 18)  # deviation from 18°C comfort
            components["temp_extreme"] = self._minmax(temp_extreme)

        # Count anomaly flags
        anomaly_cols = [c for c in df.columns if c.endswith("_anomaly")]
        if anomaly_cols:
            components["anomaly_count"] = self._minmax(df[anomaly_cols].sum(axis=1))

        if not components:
            logger.warning("No components available to compute risk score.")
            df["disease_risk_score"] = 0.0
            return df

        score = sum(
            self.WEIGHTS.get(key, 0.1) * val
            for key, val in components.items()
        )

        # Normalize total weight used
        total_weight = sum(self.WEIGHTS.get(k, 0.1) for k in components)
        score /= max(total_weight, 1e-8)

        df["disease_risk_score"] = score.clip(0, 1)
        logger.debug(
            f"Risk score built: mean={df['disease_risk_score'].mean():.3f}, "
            f"max={df['disease_risk_score'].max():.3f}"
        )
        return df

    @staticmethod
    def _minmax(series: pd.Series) -> pd.Series:
        """Min-max normalize a series to [0, 1]."""
        lo, hi = series.min(), series.max()
        if hi == lo:
            return pd.Series(np.zeros(len(series)), index=series.index)
        return (series - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Interaction Features
# ---------------------------------------------------------------------------

class InteractionFeatureBuilder:
    """Creates interaction terms between key features."""

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add interaction feature columns.

        Args:
            df: DataFrame with base features present.

        Returns:
            DataFrame with interaction features.
        """
        df = df.copy()

        if "temperature_c" in df.columns and "aqi" in df.columns:
            df["temp_x_aqi"] = (
                np.abs(df["temperature_c"] - 18) * df["aqi"] / 100
            )

        if "negative_sentiment" in df.columns and "pharmacy_sales" in df.columns:
            df["sentiment_x_pharmacy"] = (
                df["negative_sentiment"] * df["pharmacy_sales"] / df["pharmacy_sales"].max()
            )

        if "humidity_pct" in df.columns and "temperature_c" in df.columns:
            # Heat index approximation
            df["heat_index"] = (
                df["temperature_c"] + 0.33 * (df["humidity_pct"] / 100 * 6.105
                * np.exp(17.27 * df["temperature_c"] / (237.7 + df["temperature_c"]))) - 4
            )

        return df


# ---------------------------------------------------------------------------
# Full Feature Engineering Pipeline
# ---------------------------------------------------------------------------

class FeatureEngineeringPipeline:
    """Orchestrates the full feature engineering workflow."""

    def __init__(self, config: Dict) -> None:
        """
        Args:
            config: Main params.yaml config dict.
        """
        self.config = config
        feat_cfg = config.get("features", {})

        lag_days = feat_cfg.get("lag_days", [1, 3, 7, 14])
        rolling_windows = feat_cfg.get("rolling_windows", [3, 7, 14, 30])

        self.temporal = TemporalFeatureBuilder()
        self.lag = LagFeatureBuilder(lag_days=lag_days)
        self.rolling = RollingFeatureBuilder(windows=rolling_windows)
        self.roc = RateOfChangeFeatureBuilder()
        self.anomaly = AnomalyFeatureBuilder(config)
        self.risk = RiskScoreBuilder()
        self.interaction = InteractionFeatureBuilder()

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute full feature engineering pipeline.

        Args:
            df: Preprocessed master DataFrame.

        Returns:
            Feature-rich DataFrame.
        """
        logger.info("Running feature engineering pipeline …")
        n_original_cols = len(df.columns)

        df = self.temporal.build(df)
        logger.info(f"  Temporal features: {len(df.columns) - n_original_cols} added")

        n = len(df.columns)
        df = self.lag.build(df)
        logger.info(f"  Lag features: {len(df.columns) - n} added")

        n = len(df.columns)
        df = self.rolling.build(df)
        logger.info(f"  Rolling features: {len(df.columns) - n} added")

        n = len(df.columns)
        df = self.roc.build(df)
        logger.info(f"  Rate-of-change features: {len(df.columns) - n} added")

        n = len(df.columns)
        df = self.anomaly.build(df)
        logger.info(f"  Anomaly features: {len(df.columns) - n} added")

        df = self.interaction.build(df)
        df = self.risk.build(df)

        # Drop rows with NaN target
        target = self.config.get("features", {}).get("target_column", "er_visits")
        before = len(df)
        df = df.dropna(subset=[target]).reset_index(drop=True)
        if before > len(df):
            logger.info(f"  Dropped {before - len(df)} rows with missing target.")

        total_new = len(df.columns) - n_original_cols
        logger.info(
            f"Feature engineering complete: "
            f"{n_original_cols} → {len(df.columns)} columns "
            f"(+{total_new} new features)"
        )
        return df

    def get_feature_names(self, df: pd.DataFrame) -> Tuple[List[str], str]:
        """
        Return list of feature columns and target column name.

        Args:
            df: Feature-engineered DataFrame.

        Returns:
            (feature_cols, target_col)
        """
        target = self.config.get("features", {}).get("target_column", "er_visits")
        exclude = {
            "date", "zipcode", "city", "state", target,
            "outbreak_event", "aqi_category",
        }
        exclude.update([c for c in df.columns if c.endswith("_outlier")])

        feature_cols = [
            c for c in df.columns
            if c not in exclude
            and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]
        ]
        return feature_cols, target
