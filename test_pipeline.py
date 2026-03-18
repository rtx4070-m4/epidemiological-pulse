"""
tests/test_pipeline.py
======================
Integration and unit tests for the Epidemiological Pulse pipeline.

Run with:
    pytest tests/test_pipeline.py -v
    pytest tests/test_pipeline.py --cov=src --cov-report=term-missing -v
"""

import sys
import json
import logging
import tempfile
from pathlib import Path
from datetime import date, timedelta
from typing import Dict
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
import yaml

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocess import (
    DataMerger,
    DataCleaner,
    OutlierHandler,
    DataSplitter,
    DataQualityReport,
    PreprocessingPipeline,
)
from src.features.build_features import (
    TemporalFeatureBuilder,
    LagFeatureBuilder,
    RollingFeatureBuilder,
    RateOfChangeFeatureBuilder,
    AnomalyFeatureBuilder,
    RiskScoreBuilder,
    FeatureEngineeringPipeline,
)

# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def config() -> dict:
    """Load the project config YAML."""
    config_path = Path(__file__).parent.parent / "config" / "params.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def _make_daily_df(
    zip_codes: list,
    days: int = 90,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a minimal daily DataFrame for a list of ZIP codes."""
    rng = np.random.default_rng(seed)
    dates = [date(2023, 1, 1) + timedelta(days=i) for i in range(days)]
    rows = []
    for z in zip_codes:
        for d in dates:
            rows.append({
                "date": pd.Timestamp(d),
                "zip_code": z,
                "pharmacy_sales": float(rng.integers(50, 200)),
                "sentiment_score": float(rng.uniform(-1, 1)),
                "temperature": float(rng.uniform(30, 90)),
                "aqi": float(rng.integers(20, 150)),
                "er_visits": float(rng.integers(5, 50)),
                "latitude": 40.0 + float(zip_codes.index(z)) * 0.5,
                "longitude": -74.0 - float(zip_codes.index(z)) * 0.5,
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def sample_zips() -> list:
    return ["10001", "10002", "10003", "90210", "60601"]


@pytest.fixture(scope="session")
def sample_df(sample_zips) -> pd.DataFrame:
    return _make_daily_df(sample_zips, days=120)


@pytest.fixture(scope="session")
def raw_dfs(sample_zips) -> Dict[str, pd.DataFrame]:
    """Simulate the raw dict that collectors.py produces."""
    base = _make_daily_df(sample_zips, days=120)
    pharmacy  = base[["date", "zip_code", "pharmacy_sales"]].copy()
    sentiment = base[["date", "zip_code", "sentiment_score"]].copy()
    weather   = base[["date", "zip_code", "temperature"]].copy()
    aqi_df    = base[["date", "zip_code", "aqi"]].copy()
    er        = base[["date", "zip_code", "er_visits", "latitude", "longitude"]].copy()
    return {
        "pharmacy":  pharmacy,
        "sentiment": sentiment,
        "weather":   weather,
        "aqi":       aqi_df,
        "er_visits": er,
    }


# ─── DataMerger ───────────────────────────────────────────────────────────────

class TestDataMerger:
    def test_merge_produces_expected_columns(self, raw_dfs):
        merger = DataMerger()
        merged = merger.merge(raw_dfs)
        for col in ["pharmacy_sales", "sentiment_score", "temperature", "aqi", "er_visits"]:
            assert col in merged.columns, f"Column '{col}' missing after merge"

    def test_merge_row_count(self, raw_dfs, sample_zips):
        merger = DataMerger()
        merged = merger.merge(raw_dfs)
        # Should have at most n_zips × n_days rows (inner join may drop some)
        assert len(merged) <= len(sample_zips) * 120

    def test_merge_no_duplicate_date_zip(self, raw_dfs):
        merger = DataMerger()
        merged = merger.merge(raw_dfs)
        dupes = merged.duplicated(subset=["date", "zip_code"]).sum()
        assert dupes == 0, f"Found {dupes} duplicate (date, zip_code) pairs after merge"


# ─── DataCleaner ──────────────────────────────────────────────────────────────

class TestDataCleaner:
    def test_removes_duplicates(self, sample_df):
        dirty = pd.concat([sample_df, sample_df.head(10)], ignore_index=True)
        cleaner = DataCleaner()
        clean = cleaner.clean(dirty)
        assert len(clean) == len(sample_df)

    def test_date_column_is_datetime(self, sample_df):
        cleaner = DataCleaner()
        clean = cleaner.clean(sample_df.copy())
        assert pd.api.types.is_datetime64_any_dtype(clean["date"])

    def test_fills_missing_values(self, sample_df):
        dirty = sample_df.copy()
        dirty.loc[dirty.index[:5], "pharmacy_sales"] = np.nan
        cleaner = DataCleaner()
        clean = cleaner.clean(dirty)
        assert clean["pharmacy_sales"].isnull().sum() == 0


# ─── OutlierHandler ───────────────────────────────────────────────────────────

class TestOutlierHandler:
    def test_flags_obvious_outlier(self, sample_df):
        df = sample_df.copy()
        df.loc[df.index[0], "pharmacy_sales"] = 999_999  # extreme outlier
        handler = OutlierHandler(zscore_threshold=3.0)
        flagged = handler.flag_outliers(df, "pharmacy_sales")
        assert flagged["pharmacy_sales_outlier"].sum() >= 1

    def test_winsorize_caps_values(self, sample_df):
        df = sample_df.copy()
        df.loc[df.index[0], "pharmacy_sales"] = 999_999
        handler = OutlierHandler()
        capped = handler.winsorize(df, "pharmacy_sales", lower_pct=0.01, upper_pct=0.99)
        assert capped["pharmacy_sales"].max() < 999_999


# ─── DataSplitter ─────────────────────────────────────────────────────────────

class TestDataSplitter:
    def test_split_proportions(self, sample_df):
        splitter = DataSplitter(train_frac=0.7, val_frac=0.15)
        train, val, test = splitter.split(sample_df)
        total = len(train) + len(val) + len(test)
        assert total == len(sample_df)
        assert len(train) / total == pytest.approx(0.7, abs=0.05)
        assert len(val) / total == pytest.approx(0.15, abs=0.05)

    def test_split_temporal_ordering(self, sample_df):
        splitter = DataSplitter()
        train, val, test = splitter.split(sample_df)
        assert train["date"].max() <= val["date"].min()
        assert val["date"].max() <= test["date"].min()

    def test_no_data_leakage(self, sample_df):
        splitter = DataSplitter()
        train, val, test = splitter.split(sample_df)
        train_dates = set(train["date"])
        val_dates   = set(val["date"])
        test_dates  = set(test["date"])
        assert len(train_dates & val_dates)  == 0
        assert len(val_dates & test_dates)   == 0
        assert len(train_dates & test_dates) == 0


# ─── PreprocessingPipeline ────────────────────────────────────────────────────

class TestPreprocessingPipeline:
    def test_run_returns_four_items(self, config, raw_dfs):
        pipeline = PreprocessingPipeline(config)
        result = pipeline.run(raw_dfs)
        assert len(result) == 4

    def test_merged_df_not_empty(self, config, raw_dfs):
        pipeline = PreprocessingPipeline(config)
        merged, *_ = pipeline.run(raw_dfs)
        assert len(merged) > 0

    def test_all_expected_columns_present(self, config, raw_dfs):
        pipeline = PreprocessingPipeline(config)
        merged, *_ = pipeline.run(raw_dfs)
        required = ["date", "zip_code", "pharmacy_sales", "er_visits"]
        for col in required:
            assert col in merged.columns


# ─── TemporalFeatureBuilder ───────────────────────────────────────────────────

class TestTemporalFeatureBuilder:
    def test_adds_day_of_week(self, sample_df):
        builder = TemporalFeatureBuilder()
        out = builder.transform(sample_df)
        assert "day_of_week" in out.columns

    def test_adds_cyclical_encoding(self, sample_df):
        builder = TemporalFeatureBuilder()
        out = builder.transform(sample_df)
        assert "dow_sin" in out.columns
        assert "dow_cos" in out.columns

    def test_cyclical_values_in_range(self, sample_df):
        builder = TemporalFeatureBuilder()
        out = builder.transform(sample_df)
        assert out["dow_sin"].between(-1.0, 1.0).all()
        assert out["dow_cos"].between(-1.0, 1.0).all()

    def test_adds_month(self, sample_df):
        builder = TemporalFeatureBuilder()
        out = builder.transform(sample_df)
        assert "month" in out.columns
        assert out["month"].between(1, 12).all()


# ─── LagFeatureBuilder ────────────────────────────────────────────────────────

class TestLagFeatureBuilder:
    def test_creates_lag_columns(self, config, sample_df):
        builder = LagFeatureBuilder(config)
        out = builder.transform(sample_df)
        lag_cols = [c for c in out.columns if "lag" in c]
        assert len(lag_cols) > 0, "No lag columns created"

    def test_lag_values_are_shifted(self, config, sample_df):
        builder = LagFeatureBuilder(config)
        out = builder.transform(sample_df.sort_values(["zip_code", "date"]))
        # For a given ZIP, lag_1 at row i should equal value at row i-1
        zip0 = out["zip_code"].unique()[0]
        sub = out[out["zip_code"] == zip0].sort_values("date").reset_index(drop=True)
        lag1_col = [c for c in sub.columns if "pharmacy" in c and "lag_1" in c]
        if lag1_col:
            assert sub.loc[1, lag1_col[0]] == pytest.approx(
                sub.loc[0, "pharmacy_sales"], abs=1e-6
            )


# ─── RollingFeatureBuilder ────────────────────────────────────────────────────

class TestRollingFeatureBuilder:
    def test_creates_rolling_mean(self, config, sample_df):
        builder = RollingFeatureBuilder(config)
        out = builder.transform(sample_df)
        roll_cols = [c for c in out.columns if "roll" in c and "mean" in c]
        assert len(roll_cols) > 0

    def test_rolling_values_not_all_nan(self, config, sample_df):
        builder = RollingFeatureBuilder(config)
        out = builder.transform(sample_df)
        roll_cols = [c for c in out.columns if "roll" in c and "mean" in c]
        for col in roll_cols:
            assert out[col].notna().any(), f"All NaN in {col}"


# ─── AnomalyFeatureBuilder ────────────────────────────────────────────────────

class TestAnomalyFeatureBuilder:
    def test_creates_zscore_column(self, config, sample_df):
        builder = AnomalyFeatureBuilder(config)
        out = builder.transform(sample_df)
        zscore_cols = [c for c in out.columns if "zscore" in c]
        assert len(zscore_cols) > 0

    def test_zscore_is_numeric(self, config, sample_df):
        builder = AnomalyFeatureBuilder(config)
        out = builder.transform(sample_df)
        zscore_cols = [c for c in out.columns if "zscore" in c]
        for col in zscore_cols:
            assert pd.api.types.is_numeric_dtype(out[col])


# ─── RiskScoreBuilder ─────────────────────────────────────────────────────────

class TestRiskScoreBuilder:
    def test_risk_score_in_zero_one(self, config, sample_df):
        builder = RiskScoreBuilder(config)
        out = builder.transform(sample_df)
        if "risk_score" in out.columns:
            assert out["risk_score"].between(0.0, 1.0).all()

    def test_risk_score_column_exists(self, config, sample_df):
        builder = RiskScoreBuilder(config)
        out = builder.transform(sample_df)
        assert "risk_score" in out.columns


# ─── FeatureEngineeringPipeline ───────────────────────────────────────────────

class TestFeatureEngineeringPipeline:
    def test_pipeline_increases_column_count(self, config, sample_df):
        pipeline = FeatureEngineeringPipeline(config)
        out = pipeline.run(sample_df)
        assert out.shape[1] > sample_df.shape[1]

    def test_pipeline_preserves_row_count(self, config, sample_df):
        pipeline = FeatureEngineeringPipeline(config)
        out = pipeline.run(sample_df)
        assert len(out) == len(sample_df)

    def test_pipeline_no_inf_values(self, config, sample_df):
        pipeline = FeatureEngineeringPipeline(config)
        out = pipeline.run(sample_df)
        numeric = out.select_dtypes(include=[np.number])
        assert not np.isinf(numeric.values).any(), "Inf values found after feature engineering"


# ─── DataQualityReport ────────────────────────────────────────────────────────

class TestDataQualityReport:
    def test_report_is_dict(self, sample_df):
        report = DataQualityReport()
        result = report.generate(sample_df)
        assert isinstance(result, dict)

    def test_report_has_required_keys(self, sample_df):
        report = DataQualityReport()
        result = report.generate(sample_df)
        for key in ["rows", "columns", "missing_pct"]:
            assert key in result, f"Key '{key}' missing from quality report"


# ─── Synthetic data generator (smoke test) ────────────────────────────────────

class TestSyntheticDataGenerator:
    def test_generate_creates_csvs(self, config, tmp_path):
        """Smoke-test the synthetic data generator CLI module."""
        import subprocess, sys
        result = subprocess.run(
            [
                sys.executable,
                "data/generate_synthetic_data.py",
                "--config", "config/params.yaml",
                "--output", str(tmp_path),
                "--days", "30",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        assert result.returncode == 0, f"Generator failed:\n{result.stderr}"
        generated = list(tmp_path.glob("*.csv"))
        assert len(generated) >= 5, f"Expected ≥5 CSVs, found {len(generated)}"
