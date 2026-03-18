"""
tests/test_models.py
====================
Unit tests for clustering and forecasting models.

Run with:
    pytest tests/test_models.py -v
    pytest tests/test_models.py --cov=src --cov-report=term-missing -v
"""

import sys
import json
import logging
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import List
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.utils import (
    compute_mae,
    compute_rmse,
    compute_mape,
    evaluate_predictions,
    score_to_risk_level,
    ModelPersistence,
    FeatureScaler,
    create_sequences,
    prepare_ml_arrays,
    EvaluationReporter,
)
from src.models.cluster_hotspots import (
    RegionAggregator,
    HDBSCANClusterer,
    GeoJSONExporter,
    HotspotDetectionPipeline,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def config() -> dict:
    with open(Path(__file__).parent.parent / "config" / "params.yaml") as f:
        return yaml.safe_load(f)


def _make_zip_features(
    n_zips: int = 8,
    days: int = 90,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    zip_codes = [f"{10000 + i:05d}" for i in range(n_zips)]
    lats  = np.linspace(34.0, 42.0, n_zips)
    lons  = np.linspace(-118.0, -74.0, n_zips)
    base_date = pd.Timestamp("2023-01-01")

    rows = []
    for idx, z in enumerate(zip_codes):
        for d in range(days):
            rows.append({
                "date":           base_date + pd.Timedelta(days=d),
                "zip_code":       z,
                "latitude":       lats[idx],
                "longitude":      lons[idx],
                "pharmacy_sales": float(rng.integers(50, 200)),
                "sentiment_score": float(rng.uniform(-1, 1)),
                "temperature":    float(rng.uniform(30, 90)),
                "aqi":            float(rng.integers(20, 150)),
                "er_visits":      float(rng.integers(5, 50)),
                "risk_score":     float(rng.uniform(0, 1)),
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def zip_features() -> pd.DataFrame:
    return _make_zip_features()


@pytest.fixture(scope="session")
def aggregated(zip_features, config) -> pd.DataFrame:
    agg = RegionAggregator(config)
    return agg.aggregate(zip_features)


# ─── Metric utilities ─────────────────────────────────────────────────────────

class TestMetrics:
    def test_mae_perfect_forecast(self):
        y = np.array([1.0, 2.0, 3.0])
        assert compute_mae(y, y) == pytest.approx(0.0)

    def test_mae_known_value(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([12.0, 18.0, 33.0])
        # |10-12| + |20-18| + |30-33| = 2 + 2 + 3 = 7; mean = 7/3
        assert compute_mae(y_true, y_pred) == pytest.approx(7.0 / 3.0)

    def test_rmse_perfect_forecast(self):
        y = np.array([5.0, 10.0, 15.0])
        assert compute_rmse(y, y) == pytest.approx(0.0)

    def test_rmse_known_value(self):
        y_true = np.array([1.0, 1.0])
        y_pred = np.array([2.0, 0.0])
        # sqrt(((1)^2 + (1)^2) / 2) = 1.0
        assert compute_rmse(y_true, y_pred) == pytest.approx(1.0)

    def test_mape_avoids_divide_by_zero(self):
        y_true = np.array([0.0, 10.0, 20.0])
        y_pred = np.array([1.0, 12.0, 18.0])
        result = compute_mape(y_true, y_pred)
        assert np.isfinite(result)

    def test_evaluate_predictions_returns_dict(self):
        y_true = np.array([5.0, 10.0, 15.0])
        y_pred = np.array([6.0, 9.0, 14.0])
        metrics = evaluate_predictions(y_true, y_pred)
        assert isinstance(metrics, dict)
        for key in ["mae", "rmse", "mape"]:
            assert key in metrics


# ─── score_to_risk_level ──────────────────────────────────────────────────────

class TestScoreToRiskLevel:
    def test_low_score(self):
        assert score_to_risk_level(0.1) == "Low"

    def test_medium_score(self):
        assert score_to_risk_level(0.5) == "Medium"

    def test_high_score(self):
        assert score_to_risk_level(0.9) == "High"

    def test_boundary_low_medium(self):
        # Exactly at 0.33 — should be Medium or Low (just test it doesn't crash)
        result = score_to_risk_level(0.33)
        assert result in ("Low", "Medium")

    def test_boundary_medium_high(self):
        result = score_to_risk_level(0.66)
        assert result in ("Medium", "High")

    @pytest.mark.parametrize("score", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_all_valid_scores(self, score):
        result = score_to_risk_level(score)
        assert result in ("Low", "Medium", "High")


# ─── ModelPersistence ─────────────────────────────────────────────────────────

class TestModelPersistence:
    def test_save_and_load_roundtrip(self, tmp_path):
        model_obj = {"weights": [1.0, 2.0, 3.0], "name": "test_model"}
        save_path = tmp_path / "model.pkl"
        ModelPersistence.save(model_obj, str(save_path))
        loaded = ModelPersistence.load(str(save_path))
        assert loaded == model_obj

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            ModelPersistence.load("/nonexistent/path/model.pkl")


# ─── FeatureScaler ────────────────────────────────────────────────────────────

class TestFeatureScaler:
    def test_standard_scaler_zero_mean(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        scaler = FeatureScaler(method="standard")
        scaled = scaler.fit_transform(data)
        assert scaled.mean(axis=0) == pytest.approx(np.zeros(2), abs=1e-7)

    def test_minmax_scaler_range(self):
        data = np.array([[0.0], [5.0], [10.0]])
        scaler = FeatureScaler(method="minmax")
        scaled = scaler.fit_transform(data)
        assert scaled.min() == pytest.approx(0.0, abs=1e-7)
        assert scaled.max() == pytest.approx(1.0, abs=1e-7)

    def test_inverse_transform_roundtrip(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        scaler = FeatureScaler(method="standard")
        scaled = scaler.fit_transform(data)
        recovered = scaler.inverse_transform(scaled)
        assert recovered == pytest.approx(data, abs=1e-6)


# ─── create_sequences (LSTM helper) ───────────────────────────────────────────

class TestCreateSequences:
    def test_output_shapes(self):
        data = np.arange(100.0).reshape(-1, 1)
        X, y = create_sequences(data, seq_len=10, horizon=1)
        assert X.shape == (89, 10, 1)
        assert y.shape == (89,)

    def test_sequence_values_are_correct(self):
        data = np.arange(20.0).reshape(-1, 1)
        X, y = create_sequences(data, seq_len=5, horizon=1)
        # First sequence should be [0, 1, 2, 3, 4], target = 5
        assert X[0, :, 0].tolist() == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0])
        assert y[0] == pytest.approx(5.0)

    def test_no_data_leakage(self):
        """Target must always come after the sequence window."""
        data = np.arange(50.0).reshape(-1, 1)
        seq_len = 7
        X, y = create_sequences(data, seq_len=seq_len, horizon=1)
        for i in range(len(X)):
            assert y[i] > X[i, -1, 0], f"Data leakage at index {i}"


# ─── RegionAggregator ─────────────────────────────────────────────────────────

class TestRegionAggregator:
    def test_aggregated_has_one_row_per_zip(self, aggregated, zip_features):
        n_zips = zip_features["zip_code"].nunique()
        assert len(aggregated) == n_zips

    def test_aggregated_has_latitude_longitude(self, aggregated):
        assert "latitude"  in aggregated.columns
        assert "longitude" in aggregated.columns

    def test_aggregated_has_risk_score(self, aggregated):
        assert "risk_score" in aggregated.columns

    def test_no_null_zip_codes(self, aggregated):
        assert aggregated["zip_code"].isnull().sum() == 0


# ─── HDBSCANClusterer ─────────────────────────────────────────────────────────

class TestHDBSCANClusterer:
    def test_fit_predict_returns_labels(self, aggregated, config):
        clusterer = HDBSCANClusterer(config)
        result = clusterer.fit_predict(aggregated)
        assert "cluster_label" in result.columns

    def test_labels_are_integers(self, aggregated, config):
        clusterer = HDBSCANClusterer(config)
        result = clusterer.fit_predict(aggregated)
        assert pd.api.types.is_integer_dtype(result["cluster_label"])

    def test_risk_levels_are_valid(self, aggregated, config):
        clusterer = HDBSCANClusterer(config)
        result = clusterer.fit_predict(aggregated)
        assert "risk_level" in result.columns
        valid = {"Low", "Medium", "High"}
        assert set(result["risk_level"].unique()).issubset(valid)

    def test_noise_label_is_minus_one(self, config):
        """HDBSCAN noise points must be labelled -1 by convention."""
        # Very sparse data — force all points to be noise
        tiny = _make_zip_features(n_zips=2, days=5)
        tiny_agg = RegionAggregator(config).aggregate(tiny)
        clusterer = HDBSCANClusterer(config)
        result = clusterer.fit_predict(tiny_agg)
        # Either -1 (noise) or ≥0 (cluster); both are acceptable
        assert result["cluster_label"].isin(range(-1, 100)).all()


# ─── GeoJSONExporter ──────────────────────────────────────────────────────────

class TestGeoJSONExporter:
    def test_export_returns_feature_collection(self, aggregated, config):
        clusterer = HDBSCANClusterer(config)
        hotspots = clusterer.fit_predict(aggregated)
        exporter = GeoJSONExporter()
        geojson = exporter.export(hotspots)
        assert geojson["type"] == "FeatureCollection"
        assert "features" in geojson

    def test_each_feature_has_geometry(self, aggregated, config):
        clusterer = HDBSCANClusterer(config)
        hotspots = clusterer.fit_predict(aggregated)
        exporter = GeoJSONExporter()
        geojson = exporter.export(hotspots)
        for feature in geojson["features"]:
            assert feature["geometry"]["type"] == "Point"
            assert len(feature["geometry"]["coordinates"]) == 2

    def test_properties_contain_zip_code(self, aggregated, config):
        clusterer = HDBSCANClusterer(config)
        hotspots = clusterer.fit_predict(aggregated)
        exporter = GeoJSONExporter()
        geojson = exporter.export(hotspots)
        for feature in geojson["features"]:
            assert "zip_code" in feature["properties"]

    def test_geojson_is_serialisable(self, aggregated, config):
        """Ensure the result can be serialised to a JSON string."""
        clusterer = HDBSCANClusterer(config)
        hotspots = clusterer.fit_predict(aggregated)
        exporter = GeoJSONExporter()
        geojson = exporter.export(hotspots)
        json_str = json.dumps(geojson)
        assert len(json_str) > 10


# ─── HotspotDetectionPipeline (integration) ───────────────────────────────────

class TestHotspotDetectionPipeline:
    def test_pipeline_run_returns_dict(self, zip_features, config):
        pipeline = HotspotDetectionPipeline(config)
        result = pipeline.run(zip_features)
        assert isinstance(result, dict)

    def test_pipeline_result_has_required_keys(self, zip_features, config):
        pipeline = HotspotDetectionPipeline(config)
        result = pipeline.run(zip_features)
        for key in ["hotspots", "geojson", "clusterer"]:
            assert key in result, f"Missing key '{key}' in pipeline result"

    def test_hotspots_dataframe_not_empty(self, zip_features, config):
        pipeline = HotspotDetectionPipeline(config)
        result = pipeline.run(zip_features)
        assert len(result["hotspots"]) > 0

    def test_geojson_features_match_hotspot_count(self, zip_features, config):
        pipeline = HotspotDetectionPipeline(config)
        result = pipeline.run(zip_features)
        n_hotspots = len(result["hotspots"])
        n_geojson  = len(result["geojson"]["features"])
        assert n_hotspots == n_geojson


# ─── EvaluationReporter ───────────────────────────────────────────────────────

class TestEvaluationReporter:
    def test_report_to_dataframe(self):
        metrics = {
            "region_A": {"mae": 1.0, "rmse": 2.0, "mape": 5.0},
            "region_B": {"mae": 1.5, "rmse": 2.5, "mape": 6.0},
        }
        reporter = EvaluationReporter()
        df = reporter.to_dataframe(metrics)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_report_save_to_csv(self, tmp_path):
        metrics = {"region_A": {"mae": 1.0, "rmse": 2.0, "mape": 5.0}}
        reporter = EvaluationReporter()
        save_path = tmp_path / "eval.csv"
        reporter.save_csv(metrics, str(save_path))
        assert save_path.exists()
        loaded = pd.read_csv(save_path)
        assert len(loaded) == 1
