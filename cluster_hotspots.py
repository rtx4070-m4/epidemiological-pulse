"""
Epidemiological Pulse – Hotspot Clustering
============================================
Detects disease outbreak hotspots using HDBSCAN clustering on:
  - Geospatial coordinates (lat/lon)
  - Epidemiological risk features
  - Temporal aggregation (weekly snapshots)

Produces:
  - Cluster assignments per region
  - Risk level labels
  - GeoJSON export for map visualization
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hdbscan
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.models.utils import ModelPersistence, score_to_risk_level, get_risk_color

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Region Feature Aggregator
# ---------------------------------------------------------------------------

class RegionAggregator:
    """Aggregates time-series data to per-region snapshot for clustering."""

    def __init__(self, config: Dict) -> None:
        self.config = config
        self.geo_weight = config.get("clustering", {}).get("geospatial_weight", 2.0)

    def aggregate(
        self,
        df: pd.DataFrame,
        window_days: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Aggregate feature DataFrame to one row per region.

        Aggregation: mean over recent `window_days` days.

        Args:
            df: Feature-engineered DataFrame with lat/lon.
            window_days: How many recent days to include. None = all.

        Returns:
            Region-level aggregated DataFrame.
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

        if window_days:
            cutoff = df["date"].max() - pd.Timedelta(days=window_days)
            df = df[df["date"] >= cutoff]

        # Features to aggregate
        agg_cols = {
            "pharmacy_sales": "mean",
            "negative_sentiment": "mean",
            "aqi": "mean",
            "temperature_c": "mean",
            "er_visits": "mean",
            "disease_risk_score": "mean",
            "pharmacy_sales_anomaly": "sum",
            "er_visits_anomaly": "sum",
            "negative_sentiment_anomaly": "sum",
            "lat": "first",
            "lon": "first",
            "city": "first",
            "state": "first",
        }

        # Only use cols that exist
        agg_cols = {k: v for k, v in agg_cols.items() if k in df.columns}

        region_df = (
            df.groupby("zipcode")
            .agg(agg_cols)
            .reset_index()
        )

        logger.info(
            f"Aggregated to {len(region_df)} regions "
            f"(window={window_days or 'all'} days)"
        )
        return region_df


# ---------------------------------------------------------------------------
# HDBSCAN Clusterer
# ---------------------------------------------------------------------------

class HDBSCANClusterer:
    """
    Runs HDBSCAN clustering on region-level feature vectors.

    Feature vector for each region:
      [geo_weighted_lat, geo_weighted_lon, normalized_risk_score,
       normalized_pharmacy_sales, normalized_negative_sentiment,
       normalized_aqi, normalized_er_visits]
    """

    CLUSTER_FEATURE_COLS = [
        "pharmacy_sales",
        "negative_sentiment",
        "aqi",
        "er_visits",
        "disease_risk_score",
    ]

    def __init__(self, config: Dict) -> None:
        """
        Args:
            config: Main params.yaml config dict.
        """
        self.config = config
        cluster_cfg = config.get("clustering", {})

        self.min_cluster_size = cluster_cfg.get("min_cluster_size", 3)
        self.min_samples = cluster_cfg.get("min_samples", 2)
        self.metric = cluster_cfg.get("metric", "euclidean")
        self.cluster_selection_epsilon = cluster_cfg.get("cluster_selection_epsilon", 0.5)
        self.geo_weight = cluster_cfg.get("geospatial_weight", 2.0)

        self.model: Optional[hdbscan.HDBSCAN] = None
        self.scaler = StandardScaler()

    def _build_feature_matrix(self, region_df: pd.DataFrame) -> np.ndarray:
        """
        Build the weighted feature matrix for clustering.

        Geospatial coordinates are upweighted by geo_weight to ensure
        geographic proximity is a strong clustering signal.

        Args:
            region_df: Region-level aggregated DataFrame.

        Returns:
            Feature matrix of shape (n_regions, n_features).
        """
        features = []

        # Geospatial (upweighted)
        if "lat" in region_df.columns and "lon" in region_df.columns:
            lat_norm = (region_df["lat"] - region_df["lat"].mean()) / (region_df["lat"].std() + 1e-8)
            lon_norm = (region_df["lon"] - region_df["lon"].mean()) / (region_df["lon"].std() + 1e-8)
            features.append(lat_norm.values * self.geo_weight)
            features.append(lon_norm.values * self.geo_weight)

        # Epidemiological features
        for col in self.CLUSTER_FEATURE_COLS:
            if col in region_df.columns:
                col_data = region_df[col].fillna(0).values
                col_norm = (col_data - col_data.mean()) / (col_data.std() + 1e-8)
                features.append(col_norm)

        if not features:
            raise ValueError("No features available to build clustering matrix.")

        return np.column_stack(features)

    def fit_predict(
        self,
        region_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, "hdbscan.HDBSCAN"]:
        """
        Fit HDBSCAN and assign cluster labels and risk levels.

        Args:
            region_df: Region-level aggregated DataFrame.

        Returns:
            (labeled_df, fitted_hdbscan_model)
        """
        logger.info(
            f"Running HDBSCAN clustering: "
            f"min_cluster_size={self.min_cluster_size}, "
            f"min_samples={self.min_samples}, "
            f"n_regions={len(region_df)}"
        )

        if len(region_df) < self.min_cluster_size:
            logger.warning(
                f"Not enough regions ({len(region_df)}) for clustering. "
                f"min_cluster_size={self.min_cluster_size}. Assigning all to noise."
            )
            region_df = region_df.copy()
            region_df["cluster_id"] = -1
            region_df["cluster_probability"] = 0.0
            region_df["risk_level"] = "Low"
            region_df["risk_color"] = get_risk_color("Low")
            return region_df, None

        X = self._build_feature_matrix(region_df)

        self.model = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            prediction_data=True,
        )

        labels = self.model.fit_predict(X)
        probabilities = self.model.probabilities_

        region_df = region_df.copy()
        region_df["cluster_id"] = labels
        region_df["cluster_probability"] = probabilities

        # Assign risk level based on risk score and cluster membership
        risk_cfg = self.config.get("clustering", {}).get("risk_thresholds", {})
        low_t = risk_cfg.get("low", 0.33)
        med_t = risk_cfg.get("medium", 0.66)

        # Noise points (-1) assigned Low risk; within cluster, use disease_risk_score
        risk_col = "disease_risk_score" if "disease_risk_score" in region_df.columns else None

        def compute_risk(row: pd.Series) -> str:
            if row["cluster_id"] == -1:
                return "Low"
            if risk_col:
                return score_to_risk_level(row[risk_col], low_t, med_t)
            return "Medium"

        region_df["risk_level"] = region_df.apply(compute_risk, axis=1)
        region_df["risk_color"] = region_df["risk_level"].map(get_risk_color)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        logger.info(
            f"Clustering result: {n_clusters} clusters, "
            f"{n_noise} noise points"
        )

        # Log cluster summary
        if n_clusters > 0:
            for cid in sorted(set(labels)):
                if cid == -1:
                    continue
                cluster_regions = region_df[region_df["cluster_id"] == cid]
                risk_levels = cluster_regions["risk_level"].value_counts()
                logger.info(
                    f"  Cluster {cid}: {len(cluster_regions)} regions, "
                    f"risk={risk_levels.to_dict()}"
                )

        return region_df, self.model

    def save_model(self, path: str) -> None:
        """Save fitted HDBSCAN model."""
        if self.model is None:
            raise RuntimeError("No model fitted yet. Call fit_predict first.")
        ModelPersistence.save(self.model, path)

    def load_model(self, path: str) -> None:
        """Load a previously fitted HDBSCAN model."""
        self.model = ModelPersistence.load(path)


# ---------------------------------------------------------------------------
# GeoJSON Exporter
# ---------------------------------------------------------------------------

class GeoJSONExporter:
    """Exports clustering results to GeoJSON format for map visualization."""

    def export(
        self,
        region_df: pd.DataFrame,
        output_path: str,
    ) -> Dict:
        """
        Convert region DataFrame to GeoJSON FeatureCollection.

        Args:
            region_df: Clustered region DataFrame with lat, lon, cluster_id, etc.
            output_path: Path to write .geojson file.

        Returns:
            GeoJSON dict.
        """
        features = []
        required_geo = {"lat", "lon"}
        if not required_geo.issubset(region_df.columns):
            raise ValueError(f"region_df must contain {required_geo} columns.")

        for _, row in region_df.iterrows():
            if pd.isna(row.get("lat")) or pd.isna(row.get("lon")):
                continue

            properties = {
                "zipcode": str(row.get("zipcode", "")),
                "city": str(row.get("city", "")),
                "state": str(row.get("state", "")),
                "cluster_id": int(row.get("cluster_id", -1)),
                "risk_level": str(row.get("risk_level", "Unknown")),
                "risk_color": str(row.get("risk_color", "#95a5a6")),
                "disease_risk_score": round(float(row.get("disease_risk_score", 0)), 4),
                "er_visits_mean": round(float(row.get("er_visits", 0)), 2),
                "pharmacy_sales_mean": round(float(row.get("pharmacy_sales", 0)), 2),
                "negative_sentiment_mean": round(float(row.get("negative_sentiment", 0)), 4),
                "aqi_mean": round(float(row.get("aqi", 0)), 1),
                "cluster_probability": round(float(row.get("cluster_probability", 0)), 4),
            }

            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(row["lon"]), float(row["lat"])],
                },
                "properties": properties,
            }
            features.append(feature)

        geojson = {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "n_regions": len(features),
                "n_clusters": len({f["properties"]["cluster_id"] for f in features} - {-1}),
                "risk_distribution": {
                    "High": sum(1 for f in features if f["properties"]["risk_level"] == "High"),
                    "Medium": sum(1 for f in features if f["properties"]["risk_level"] == "Medium"),
                    "Low": sum(1 for f in features if f["properties"]["risk_level"] == "Low"),
                },
            },
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(geojson, f, indent=2)

        logger.info(f"GeoJSON exported: {output_path} ({len(features)} regions)")
        return geojson


# ---------------------------------------------------------------------------
# Full Hotspot Detection Pipeline
# ---------------------------------------------------------------------------

class HotspotDetectionPipeline:
    """Orchestrates region aggregation, clustering, and GeoJSON export."""

    def __init__(self, config: Dict) -> None:
        self.config = config
        self.aggregator = RegionAggregator(config)
        self.clusterer = HDBSCANClusterer(config)
        self.exporter = GeoJSONExporter()

    def run(
        self,
        df: pd.DataFrame,
        window_days: int = 30,
        save_model: bool = True,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Full pipeline: aggregate → cluster → export GeoJSON.

        Args:
            df: Feature-engineered DataFrame.
            window_days: Aggregation window (recent N days).
            save_model: Save the fitted HDBSCAN model.

        Returns:
            (clustered_region_df, geojson_dict)
        """
        logger.info("=" * 50)
        logger.info("Running Hotspot Detection Pipeline …")
        logger.info("=" * 50)

        # Step 1: Aggregate to region level
        region_df = self.aggregator.aggregate(df, window_days=window_days)

        # Step 2: Cluster
        clustered_df, model = self.clusterer.fit_predict(region_df)

        # Step 3: Save model
        if save_model and model is not None:
            model_dir = self.config.get("clustering", {}).get(
                "model_save_dir", "outputs/models"
            )
            model_path = os.path.join(model_dir, "hdbscan_model.joblib")
            self.clusterer.save_model(model_path)

        # Step 4: Export GeoJSON
        geojson_path = self.config.get("clustering", {}).get(
            "geojson_output", "outputs/geojson/hotspots.geojson"
        )
        geojson = self.exporter.export(clustered_df, geojson_path)

        # Summary
        risk_dist = geojson["metadata"]["risk_distribution"]
        logger.info("Hotspot Detection Summary:")
        logger.info(f"  Regions analyzed: {len(clustered_df)}")
        logger.info(f"  Clusters found:   {geojson['metadata']['n_clusters']}")
        logger.info(f"  High risk:        {risk_dist['High']}")
        logger.info(f"  Medium risk:      {risk_dist['Medium']}")
        logger.info(f"  Low risk:         {risk_dist['Low']}")

        return clustered_df, geojson


import os  # noqa: E402 – needed for model_path construction above
