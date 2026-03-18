"""
Epidemiological Pulse – Master Pipeline
=========================================
Orchestrates the full end-to-end pipeline:

  1. Data Collection (or synthetic data loading)
  2. Preprocessing & Quality Checks
  3. Feature Engineering
  4. Hotspot Clustering (HDBSCAN)
  5. ER Visit Forecasting (Prophet + LSTM)
  6. Results Export

CLI Usage:
    # Full pipeline from synthetic data
    python src/pipeline.py --mode full --data-dir data/synthetic

    # Only run clustering
    python src/pipeline.py --mode cluster

    # Only run forecasting
    python src/pipeline.py --mode forecast

    # Run dashboard
    python src/pipeline.py --mode dashboard
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------

def setup_logging(config: Dict, log_dir: str = "logs") -> None:
    """Configure logging to both console and file."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_level = getattr(logging, config.get("project", {}).get("log_level", "INFO"), logging.INFO)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"pipeline_{timestamp}.log")

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file),
    ]

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )

    logging.getLogger("prophet").setLevel(logging.WARNING)
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config Loader
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> Dict:
    """Load and validate YAML configuration."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Config loaded: {config_path}")
    return cfg


def load_api_keys(api_keys_path: str) -> Optional[Dict]:
    """Load API keys if available."""
    if os.path.exists(api_keys_path):
        with open(api_keys_path) as f:
            keys = yaml.safe_load(f)
        logger.info(f"API keys loaded from {api_keys_path}")
        return keys
    logger.info("No API keys file found – running in synthetic data mode.")
    return None


# ---------------------------------------------------------------------------
# Pipeline Stages
# ---------------------------------------------------------------------------

class EpidemiologicalPipeline:
    """
    Full epidemiological surveillance pipeline.

    Stages:
      1. load_data        – Load or generate synthetic data
      2. preprocess       – Clean, merge, validate data
      3. engineer_features– Build ML features
      4. detect_hotspots  – HDBSCAN clustering
      5. forecast_er      – Prophet + LSTM forecasting
      6. export_results   – Save outputs
    """

    def __init__(
        self,
        config_path: str = "config/params.yaml",
        api_keys_path: str = "config/api_keys.yaml",
        data_dir: str = "data/synthetic",
        output_dir: str = "outputs",
    ) -> None:
        """
        Initialize pipeline.

        Args:
            config_path: Path to params.yaml.
            api_keys_path: Path to api_keys.yaml (optional).
            data_dir: Directory with data CSVs.
            output_dir: Directory for output artifacts.
        """
        self.config = load_config(config_path)
        self.api_keys = load_api_keys(api_keys_path)
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # State
        self.master_df: Optional[pd.DataFrame] = None
        self.feature_df: Optional[pd.DataFrame] = None
        self.train_df: Optional[pd.DataFrame] = None
        self.val_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.clustered_df: Optional[pd.DataFrame] = None
        self.geojson: Optional[Dict] = None
        self.forecast_results: Optional[Dict] = None
        self.quality_report: Optional[Dict] = None

        self._start_time = time.time()

        logger.info("=" * 70)
        logger.info(
            f"Epidemiological Pulse Pipeline initialized | "
            f"v{self.config['project']['version']}"
        )
        logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Stage 1: Load Data
    # ------------------------------------------------------------------
    def load_data(self, generate_synthetic: bool = False) -> pd.DataFrame:
        """
        Load data from CSV files or generate synthetic data.

        Args:
            generate_synthetic: If True, generate fresh synthetic data.

        Returns:
            Merged master DataFrame.
        """
        logger.info("[Stage 1] Loading data …")

        master_path = os.path.join(self.data_dir, "master_dataset.csv")

        if generate_synthetic or not os.path.exists(master_path):
            logger.info("Generating synthetic data …")
            from data.generate_synthetic_data import SyntheticDataGenerator
            gen = SyntheticDataGenerator(self.config, self.data_dir)
            self.master_df = gen.generate_master_dataset()
        else:
            logger.info(f"Loading master dataset from {master_path}")
            self.master_df = pd.read_csv(master_path, parse_dates=["date"])
            logger.info(f"  Loaded: {len(self.master_df):,} rows")

        return self.master_df

    def load_from_sources(self) -> pd.DataFrame:
        """
        Collect data from configured data sources (API or synthetic).

        Returns:
            Merged master DataFrame.
        """
        from src.data.collectors import CollectorFactory
        from src.data.preprocess import PreprocessingPipeline

        logger.info("[Stage 1] Collecting from data sources …")

        factory = CollectorFactory(
            config=self.config,
            synthetic_data_dir=self.data_dir,
        )

        cfg = self.config["data"]
        start_date = cfg["date_range"]["start"]
        end_date = cfg["date_range"]["end"]
        zipcodes = [r["zipcode"] for r in cfg["regions"]]

        data_sources = factory.collect_all(start_date, end_date, zipcodes)

        prep = PreprocessingPipeline(self.config)
        self.master_df, self.train_df, self.val_df, self.test_df, self.quality_report = (
            prep.run(data_sources)
        )

        return self.master_df

    # ------------------------------------------------------------------
    # Stage 2: Preprocess
    # ------------------------------------------------------------------
    def preprocess(self) -> pd.DataFrame:
        """
        Preprocess the master DataFrame.

        Returns:
            Cleaned DataFrame.
        """
        logger.info("[Stage 2] Preprocessing data …")

        from src.data.preprocess import PreprocessingPipeline

        if self.master_df is None:
            raise RuntimeError("No data loaded. Call load_data() first.")

        # Convert single DataFrame to source dict (already merged)
        data_sources = {"master": self.master_df}
        prep = PreprocessingPipeline(self.config)
        cleaned, self.train_df, self.val_df, self.test_df, self.quality_report = (
            prep.run(data_sources)
        )
        self.master_df = cleaned

        # Log quality report summary
        logger.info(
            f"  Data quality: {self.quality_report['n_rows']:,} rows, "
            f"{self.quality_report['n_cols']} cols, "
            f"{len(self.quality_report.get('missing_values', {}))} cols with missing values"
        )

        return self.master_df

    # ------------------------------------------------------------------
    # Stage 3: Feature Engineering
    # ------------------------------------------------------------------
    def engineer_features(self) -> pd.DataFrame:
        """
        Build machine learning features.

        Returns:
            Feature-engineered DataFrame.
        """
        logger.info("[Stage 3] Engineering features …")

        from src.features.build_features import FeatureEngineeringPipeline

        if self.master_df is None:
            raise RuntimeError("No data to engineer features from.")

        feat_pipe = FeatureEngineeringPipeline(self.config)
        self.feature_df = feat_pipe.run(self.master_df)

        # Re-split on feature_df (lag features consume first N rows per region)
        from src.data.preprocess import DataSplitter
        splitter = DataSplitter()
        self.train_df, self.val_df, self.test_df = splitter.split(self.feature_df)

        logger.info(
            f"  Feature-engineered: {len(self.feature_df.columns)} total columns, "
            f"train={len(self.train_df):,}, test={len(self.test_df):,}"
        )

        return self.feature_df

    # ------------------------------------------------------------------
    # Stage 4: Hotspot Clustering
    # ------------------------------------------------------------------
    def detect_hotspots(self, window_days: int = 30) -> pd.DataFrame:
        """
        Run HDBSCAN hotspot detection.

        Args:
            window_days: Aggregation window for recent data.

        Returns:
            Clustered region DataFrame.
        """
        logger.info("[Stage 4] Detecting hotspots …")

        from src.models.cluster_hotspots import HotspotDetectionPipeline

        df = self.feature_df if self.feature_df is not None else self.master_df
        if df is None:
            raise RuntimeError("No data available for clustering.")

        hotspot_pipe = HotspotDetectionPipeline(self.config)
        self.clustered_df, self.geojson = hotspot_pipe.run(
            df,
            window_days=window_days,
            save_model=True,
        )

        return self.clustered_df

    # ------------------------------------------------------------------
    # Stage 5: ER Forecasting
    # ------------------------------------------------------------------
    def forecast_er_visits(self) -> Dict:
        """
        Run Prophet + LSTM forecasting.

        Returns:
            Forecast results dict.
        """
        logger.info("[Stage 5] Forecasting ER visits …")

        from src.features.build_features import FeatureEngineeringPipeline
        from src.models.predict_er import ForecastingPipeline

        if self.train_df is None or self.test_df is None:
            raise RuntimeError("No train/test data available. Run preprocess() first.")

        # Get feature column names
        feat_pipe = FeatureEngineeringPipeline(self.config)
        df = self.feature_df if self.feature_df is not None else self.train_df
        feature_cols, target_col = feat_pipe.get_feature_names(df)

        # Limit feature set for LSTM (top features only to avoid memory issues)
        priority_features = [
            "pharmacy_sales", "negative_sentiment", "aqi", "temperature_c",
            "humidity_pct", "day_of_week", "month", "season", "is_weekend",
            "pharmacy_sales_roll_7_mean", "er_visits_lag_1", "er_visits_lag_7",
            "disease_risk_score",
        ]
        feature_cols = [f for f in priority_features if f in df.columns]

        zipcodes = self.config["data"]["date_range"]
        all_zips = self.train_df["zipcode"].unique().tolist()

        fc_pipe = ForecastingPipeline(self.config)
        self.forecast_results = fc_pipe.run(
            train_df=self.train_df,
            test_df=self.test_df,
            feature_cols=feature_cols,
            target_col=target_col,
            zipcodes=all_zips,
        )

        # Save forecast CSVs for dashboard
        fc_dir = self.config.get("prophet", {}).get("model_save_dir", "outputs/models/prophet")
        Path(fc_dir).mkdir(parents=True, exist_ok=True)

        for zipcode, forecast_df in self.forecast_results.get("prophet_forecasts", {}).items():
            out_path = os.path.join(fc_dir, f"forecast_{zipcode}.csv")
            forecast_df.to_csv(out_path, index=False)

        logger.info(
            f"  Forecasted for {len(self.forecast_results.get('prophet_forecasts', {}))} regions"
        )
        return self.forecast_results

    # ------------------------------------------------------------------
    # Stage 6: Export Results
    # ------------------------------------------------------------------
    def export_results(self) -> Dict:
        """
        Save all pipeline outputs to disk.

        Returns:
            Dict of output file paths.
        """
        logger.info("[Stage 6] Exporting results …")
        outputs = {}

        # Master dataset
        if self.feature_df is not None:
            path = str(self.output_dir / "feature_dataset.csv")
            self.feature_df.to_csv(path, index=False)
            outputs["feature_dataset"] = path
            logger.info(f"  Feature dataset → {path}")

        # Clustered regions
        if self.clustered_df is not None:
            path = str(self.output_dir / "clustered_regions.csv")
            self.clustered_df.to_csv(path, index=False)
            outputs["clustered_regions"] = path
            logger.info(f"  Clustered regions → {path}")

        # Quality report
        if self.quality_report is not None:
            path = str(self.output_dir / "reports" / "quality_report.json")
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(self.quality_report, f, indent=2, default=str)
            outputs["quality_report"] = path

        # Evaluation metrics
        if self.forecast_results is not None:
            eval_path = self.config.get("evaluation", {}).get(
                "report_output", "outputs/reports/evaluation_report.json"
            )
            Path(eval_path).parent.mkdir(parents=True, exist_ok=True)
            with open(eval_path, "w") as f:
                json.dump(
                    {
                        "prophet_metrics": self.forecast_results.get("prophet_metrics", {}),
                        "lstm_metrics": self.forecast_results.get("lstm_metrics", {}),
                    },
                    f,
                    indent=2,
                    default=str,
                )
            outputs["evaluation_report"] = eval_path
            logger.info(f"  Evaluation report → {eval_path}")

        # Pipeline summary
        elapsed = time.time() - self._start_time
        summary = {
            "pipeline_version": self.config["project"]["version"],
            "run_timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 2),
            "outputs": outputs,
        }
        summary_path = str(self.output_dir / "reports" / "pipeline_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Pipeline complete in {elapsed:.1f}s")
        return outputs

    # ------------------------------------------------------------------
    # Full Pipeline Runner
    # ------------------------------------------------------------------
    def run_full(
        self,
        generate_synthetic: bool = False,
        skip_lstm: bool = False,
    ) -> None:
        """
        Run the complete pipeline end-to-end.

        Args:
            generate_synthetic: Regenerate synthetic data.
            skip_lstm: Skip LSTM training (faster).
        """
        logger.info("Starting FULL PIPELINE …")

        if skip_lstm:
            self.config["lstm"]["enabled"] = False

        self.load_data(generate_synthetic=generate_synthetic)
        self.preprocess()
        self.engineer_features()
        self.detect_hotspots()
        self.forecast_er_visits()
        self.export_results()

        logger.info("✓ Full pipeline completed successfully.")


# ---------------------------------------------------------------------------
# CLI Interface
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Epidemiological Pulse – Disease Outbreak Detection Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        choices=["full", "collect", "preprocess", "features", "cluster", "forecast", "dashboard", "generate"],
        default="full",
        help="Pipeline execution mode.",
    )
    parser.add_argument(
        "--config",
        default="config/params.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--api-keys",
        default="config/api_keys.yaml",
        help="Path to API keys YAML file.",
    )
    parser.add_argument(
        "--data-dir",
        default="data/synthetic",
        help="Directory containing data CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for pipeline outputs.",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Regenerate synthetic data even if it exists.",
    )
    parser.add_argument(
        "--skip-lstm",
        action="store_true",
        help="Skip LSTM model training.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode (verbose logging).",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=30,
        help="Aggregation window for hotspot detection.",
    )

    return parser.parse_args()


def main() -> None:
    """Main CLI entrypoint."""
    args = parse_args()

    config = load_config(args.config)
    if args.debug:
        config["project"]["log_level"] = "DEBUG"

    setup_logging(config, log_dir=config.get("project", {}).get("log_dir", "logs"))

    logger.info(f"Mode: {args.mode}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Data dir: {args.data_dir}")

    # --- Generate only ---
    if args.mode == "generate":
        from data.generate_synthetic_data import SyntheticDataGenerator
        gen = SyntheticDataGenerator(config, args.data_dir)
        gen.generate_master_dataset()
        sys.exit(0)

    # --- Dashboard only ---
    if args.mode == "dashboard":
        from src.visualization.dashboard import run_dashboard
        run_dashboard(
            config_path=args.config,
            data_dir=args.data_dir,
            debug=args.debug,
        )
        sys.exit(0)

    # --- Pipeline modes ---
    pipe = EpidemiologicalPipeline(
        config_path=args.config,
        api_keys_path=args.api_keys,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )

    if args.mode == "full":
        pipe.run_full(
            generate_synthetic=args.generate,
            skip_lstm=args.skip_lstm,
        )

    elif args.mode == "collect":
        pipe.load_from_sources()

    elif args.mode == "preprocess":
        pipe.load_data(generate_synthetic=args.generate)
        pipe.preprocess()

    elif args.mode == "features":
        pipe.load_data(generate_synthetic=args.generate)
        pipe.preprocess()
        pipe.engineer_features()

    elif args.mode == "cluster":
        pipe.load_data(generate_synthetic=args.generate)
        pipe.preprocess()
        pipe.engineer_features()
        pipe.detect_hotspots(window_days=args.window_days)

    elif args.mode == "forecast":
        pipe.load_data(generate_synthetic=args.generate)
        pipe.preprocess()
        pipe.engineer_features()
        if args.skip_lstm:
            pipe.config["lstm"]["enabled"] = False
        pipe.forecast_er_visits()

    pipe.export_results()
    logger.info(f"✓ Mode '{args.mode}' completed.")


if __name__ == "__main__":
    main()
