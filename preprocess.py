"""
Epidemiological Pulse – Data Preprocessing
===========================================
Handles:
  - Data merging across sources
  - Missing value imputation
  - Outlier detection and capping
  - Data type normalization
  - Train/validation/test splitting
  - Data quality reporting
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DataMerger
# ---------------------------------------------------------------------------

class DataMerger:
    """Merges multiple source DataFrames on (date, zipcode)."""

    def __init__(self, config: Dict) -> None:
        self.config = config

    def merge(self, data_sources: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merge all source DataFrames.

        Args:
            data_sources: Dict of {name: DataFrame} where each DataFrame
                          must contain 'date' and 'zipcode' columns.

        Returns:
            Merged DataFrame.
        """
        logger.info("Merging data sources …")
        merged: Optional[pd.DataFrame] = None

        for source_name, df in data_sources.items():
            if df is None or df.empty:
                logger.warning(f"Source '{source_name}' is empty; skipping.")
                continue

            df = df.copy()
            df["date"] = pd.to_datetime(df["date"])
            df["zipcode"] = df["zipcode"].astype(str)

            if merged is None:
                merged = df
            else:
                merged = merged.merge(df, on=["date", "zipcode"], how="outer")
                logger.debug(f"  After merging '{source_name}': {len(merged)} rows")

        if merged is None:
            raise ValueError("All data sources are empty. Cannot merge.")

        # Add region metadata
        regions_cfg = self.config["data"].get("regions", [])
        if regions_cfg:
            meta_df = pd.DataFrame(regions_cfg)
            meta_df["zipcode"] = meta_df["zipcode"].astype(str)
            merged = merged.merge(meta_df, on="zipcode", how="left")

        merged = merged.sort_values(["zipcode", "date"]).reset_index(drop=True)
        logger.info(f"Merged dataset: {len(merged):,} rows × {len(merged.columns)} columns")
        return merged


# ---------------------------------------------------------------------------
# DataCleaner
# ---------------------------------------------------------------------------

class DataCleaner:
    """Handles missing values, data types, and basic sanity checks."""

    NUMERIC_COLS = [
        "pharmacy_sales",
        "negative_sentiment",
        "post_volume",
        "symptom_mention_rate",
        "temperature_c",
        "humidity_pct",
        "precipitation_mm",
        "wind_speed_kmh",
        "aqi",
        "pm25_ugm3",
        "er_visits",
    ]

    def __init__(self, config: Dict) -> None:
        self.config = config

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full cleaning pipeline.

        Steps:
          1. Remove duplicate rows
          2. Fix data types
          3. Handle missing values
          4. Cap physical impossibilities

        Args:
            df: Raw merged DataFrame.

        Returns:
            Cleaned DataFrame.
        """
        logger.info("Cleaning data …")
        original_len = len(df)

        df = self._remove_duplicates(df)
        df = self._fix_types(df)
        df = self._impute_missing(df)
        df = self._cap_values(df)

        logger.info(
            f"Cleaning complete: {original_len:,} → {len(df):,} rows "
            f"({original_len - len(df)} removed)"
        )
        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove exact duplicate rows."""
        n_before = len(df)
        df = df.drop_duplicates(subset=["date", "zipcode"], keep="first")
        removed = n_before - len(df)
        if removed > 0:
            logger.warning(f"Removed {removed} duplicate (date, zipcode) rows.")
        return df

    def _fix_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure correct dtypes."""
        df["date"] = pd.to_datetime(df["date"])
        df["zipcode"] = df["zipcode"].astype(str)

        for col in self.NUMERIC_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def _impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values using forward/backward fill within each ZIP,
        then global median as fallback.
        """
        missing_before = df[self.NUMERIC_COLS].isnull().sum()
        total_missing = missing_before.sum()

        if total_missing > 0:
            logger.info(f"Imputing {total_missing} missing values …")

            numeric_cols = [c for c in self.NUMERIC_COLS if c in df.columns]

            # Forward/backward fill within each ZIP
            df = df.sort_values(["zipcode", "date"])
            df[numeric_cols] = (
                df.groupby("zipcode")[numeric_cols]
                .transform(lambda x: x.ffill().bfill())
            )

            # Global median fallback for any remaining NaNs
            for col in numeric_cols:
                if df[col].isnull().any():
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    logger.debug(f"  Filled {col} with global median={median_val:.3f}")

        return df

    def _cap_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cap physically impossible values."""
        caps = {
            "negative_sentiment": (0.0, 1.0),
            "symptom_mention_rate": (0.0, 1.0),
            "humidity_pct": (0.0, 100.0),
            "precipitation_mm": (0.0, 500.0),
            "aqi": (0.0, 500.0),
            "pm25_ugm3": (0.0, 1000.0),
            "er_visits": (0, None),
            "pharmacy_sales": (0.0, None),
            "post_volume": (0, None),
            "wind_speed_kmh": (0.0, 300.0),
        }
        for col, (lo, hi) in caps.items():
            if col in df.columns:
                df[col] = df[col].clip(lower=lo, upper=hi)
        return df


# ---------------------------------------------------------------------------
# OutlierHandler
# ---------------------------------------------------------------------------

class OutlierHandler:
    """Detects and handles outliers using IQR and Z-score methods."""

    def __init__(self, config: Dict) -> None:
        self.config = config
        self.anomaly_cfg = config.get("anomaly", {})
        self.zscore_threshold = self.anomaly_cfg.get("zscore_threshold", 3.0)

    def flag_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add boolean outlier flag columns for key metrics.

        For each column in anomaly.columns_to_check, adds:
          {col}_outlier: bool

        Args:
            df: Cleaned DataFrame.

        Returns:
            DataFrame with additional outlier flag columns.
        """
        cols_to_check = self.anomaly_cfg.get("columns_to_check", [])
        logger.info(f"Flagging outliers for: {cols_to_check}")

        for col in cols_to_check:
            if col not in df.columns:
                continue

            flag_col = f"{col}_outlier"
            # Z-score within each ZIP group
            df[flag_col] = (
                df.groupby("zipcode")[col]
                .transform(
                    lambda x: np.abs(stats.zscore(x, nan_policy="omit")) > self.zscore_threshold
                )
                .fillna(False)
                .astype(bool)
            )
            n_outliers = df[flag_col].sum()
            logger.debug(
                f"  {col}: {n_outliers} outliers flagged "
                f"(|z| > {self.zscore_threshold})"
            )

        return df

    def cap_outliers(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Winsorize (cap) outliers at 1st and 99th percentile.

        Args:
            df: DataFrame.
            columns: Columns to winsorize.

        Returns:
            DataFrame with capped values.
        """
        for col in columns:
            if col not in df.columns:
                continue
            lo = df[col].quantile(0.01)
            hi = df[col].quantile(0.99)
            df[col] = df[col].clip(lo, hi)
            logger.debug(f"  Winsorized {col}: [{lo:.2f}, {hi:.2f}]")
        return df


# ---------------------------------------------------------------------------
# DataSplitter
# ---------------------------------------------------------------------------

class DataSplitter:
    """Creates temporal train/validation/test splits."""

    def __init__(
        self,
        train_frac: float = 0.70,
        val_frac: float = 0.15,
    ) -> None:
        """
        Args:
            train_frac: Fraction of time range for training.
            val_frac: Fraction for validation (test gets remainder).
        """
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = 1.0 - train_frac - val_frac

    def split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Temporal split (no shuffle to respect time series ordering).

        Args:
            df: Preprocessed DataFrame sorted by date.

        Returns:
            (train_df, val_df, test_df)
        """
        dates = df["date"].sort_values().unique()
        n = len(dates)
        train_cutoff = dates[int(n * self.train_frac)]
        val_cutoff = dates[int(n * (self.train_frac + self.val_frac))]

        train = df[df["date"] < train_cutoff]
        val = df[(df["date"] >= train_cutoff) & (df["date"] < val_cutoff)]
        test = df[df["date"] >= val_cutoff]

        logger.info(
            f"Data split → train: {len(train):,} | "
            f"val: {len(val):,} | test: {len(test):,}"
        )
        logger.info(
            f"  Train dates: {train['date'].min().date()} → {train['date'].max().date()}"
        )
        logger.info(
            f"  Val dates:   {val['date'].min().date()} → {val['date'].max().date()}"
        )
        logger.info(
            f"  Test dates:  {test['date'].min().date()} → {test['date'].max().date()}"
        )
        return train, val, test


# ---------------------------------------------------------------------------
# DataQualityReport
# ---------------------------------------------------------------------------

class DataQualityReport:
    """Generates a data quality summary report."""

    def generate(self, df: pd.DataFrame) -> Dict:
        """
        Generate a comprehensive data quality report.

        Args:
            df: DataFrame to assess.

        Returns:
            Dict with quality metrics.
        """
        report = {
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "n_regions": df["zipcode"].nunique() if "zipcode" in df.columns else None,
            "date_range": {
                "start": str(df["date"].min().date()) if "date" in df.columns else None,
                "end": str(df["date"].max().date()) if "date" in df.columns else None,
            },
            "missing_values": {},
            "outlier_counts": {},
            "numeric_stats": {},
        }

        # Missing values
        for col in df.columns:
            n_missing = df[col].isnull().sum()
            if n_missing > 0:
                report["missing_values"][col] = {
                    "count": int(n_missing),
                    "pct": round(n_missing / len(df) * 100, 2),
                }

        # Outlier counts
        outlier_cols = [c for c in df.columns if c.endswith("_outlier")]
        for col in outlier_cols:
            report["outlier_counts"][col.replace("_outlier", "")] = int(df[col].sum())

        # Numeric stats
        numeric_df = df.select_dtypes(include=[np.number])
        for col in numeric_df.columns:
            if col.endswith("_outlier"):
                continue
            report["numeric_stats"][col] = {
                "mean": round(float(numeric_df[col].mean()), 4),
                "std": round(float(numeric_df[col].std()), 4),
                "min": round(float(numeric_df[col].min()), 4),
                "max": round(float(numeric_df[col].max()), 4),
                "p50": round(float(numeric_df[col].median()), 4),
            }

        logger.info("Data quality report generated.")
        return report


# ---------------------------------------------------------------------------
# Full Preprocessing Pipeline
# ---------------------------------------------------------------------------

class PreprocessingPipeline:
    """Orchestrates the full data preprocessing workflow."""

    def __init__(self, config: Dict) -> None:
        self.config = config
        self.merger = DataMerger(config)
        self.cleaner = DataCleaner(config)
        self.outlier_handler = OutlierHandler(config)
        self.splitter = DataSplitter()
        self.quality_reporter = DataQualityReport()

    def run(
        self,
        data_sources: Dict[str, pd.DataFrame],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
        """
        Execute full preprocessing pipeline.

        Steps:
          1. Merge all sources
          2. Clean data
          3. Flag outliers
          4. Split into train/val/test
          5. Generate quality report

        Args:
            data_sources: Dict of {source_name: DataFrame}.

        Returns:
            Tuple of (merged_df, train_df, val_df, test_df, quality_report)
        """
        logger.info("Running preprocessing pipeline …")

        # Step 1: Merge
        merged = self.merger.merge(data_sources)

        # Step 2: Clean
        cleaned = self.cleaner.clean(merged)

        # Step 3: Flag outliers
        cleaned = self.outlier_handler.flag_outliers(cleaned)

        # Step 4: Split
        train, val, test = self.splitter.split(cleaned)

        # Step 5: Quality report
        report = self.quality_reporter.generate(cleaned)

        logger.info("Preprocessing pipeline complete.")
        return cleaned, train, val, test, report
