#!/usr/bin/env python3
"""
Epidemiological Pulse – Synthetic Data Generator
=================================================
Generates realistic synthetic datasets for:
  - Pharmacy OTC sales
  - Social media sentiment
  - Weather (temperature, humidity, precipitation)
  - Air Quality Index (AQI)
  - Emergency Room visit counts

All datasets simulate:
  - Seasonal trends
  - Gaussian noise
  - Correlated outbreak spikes
  - Geographic variation

Usage:
    python data/generate_synthetic_data.py --config config/params.yaml --output data/synthetic
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("synthetic_generator")


# ---------------------------------------------------------------------------
# Helper Utilities
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def seasonal_signal(
    dates: pd.DatetimeIndex,
    amplitude: float,
    phase_shift_days: int = 0,
    period_days: int = 365,
) -> np.ndarray:
    """
    Generate a smooth annual seasonal sinusoidal signal.

    Args:
        dates: DatetimeIndex of the time series.
        amplitude: Peak amplitude of the seasonal component.
        phase_shift_days: Shift the peak by this many days.
        period_days: Period of the seasonal cycle in days.

    Returns:
        Numpy array of seasonal values.
    """
    day_of_year = dates.day_of_year.values
    return amplitude * np.cos(
        2 * np.pi * (day_of_year - phase_shift_days) / period_days
    )


def inject_outbreaks(
    series: np.ndarray,
    dates: pd.DatetimeIndex,
    probability: float,
    multiplier_range: Tuple[float, float],
    duration_range: Tuple[int, int] = (7, 21),
    rng: np.random.Generator = None,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Randomly inject outbreak spikes into a time series.

    Args:
        series: Base time series array.
        dates: Corresponding dates.
        probability: Daily probability of an outbreak starting.
        multiplier_range: (min, max) multiplier applied during outbreak.
        duration_range: (min, max) duration in days.
        rng: NumPy random generator for reproducibility.

    Returns:
        Modified series and list of outbreak event metadata dicts.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(series)
    outbreak_events = []
    i = 0

    while i < n:
        if rng.random() < probability:
            duration = int(rng.integers(*duration_range))
            multiplier = rng.uniform(*multiplier_range)
            end_i = min(i + duration, n)

            # Bell-curve shaped spike
            spike_len = end_i - i
            bell = np.exp(
                -0.5 * ((np.linspace(-3, 3, spike_len)) ** 2)
            )
            series[i:end_i] *= 1 + (multiplier - 1) * bell

            outbreak_events.append(
                {
                    "start_date": str(dates[i].date()),
                    "end_date": str(dates[end_i - 1].date()),
                    "duration_days": spike_len,
                    "multiplier": round(multiplier, 3),
                }
            )
            i += duration  # skip past the outbreak
        else:
            i += 1

    return series, outbreak_events


# ---------------------------------------------------------------------------
# Dataset Generators
# ---------------------------------------------------------------------------

class SyntheticDataGenerator:
    """Generates all synthetic epidemiological datasets."""

    def __init__(self, config: Dict, output_dir: str) -> None:
        """
        Initialize generator.

        Args:
            config: Parsed YAML configuration dict.
            output_dir: Directory to write CSV files.
        """
        self.cfg = config
        self.syn = config["synthetic"]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        seed = self.syn.get("seed", 42)
        self.rng = np.random.default_rng(seed)

        regions_cfg = config["data"].get("regions", [])
        self.regions = [r["zipcode"] for r in regions_cfg]
        self.region_meta = {r["zipcode"]: r for r in regions_cfg}

        start = config["data"]["date_range"]["start"]
        end = config["data"]["date_range"]["end"]
        self.dates = pd.date_range(start=start, end=end, freq="D")
        self.n_days = len(self.dates)

        logger.info(
            f"Generator initialized: {len(self.regions)} regions, "
            f"{self.n_days} days ({start} → {end})"
        )

    # ------------------------------------------------------------------
    # Pharmacy Sales
    # ------------------------------------------------------------------
    def generate_pharmacy_sales(self) -> pd.DataFrame:
        """
        Generate OTC pharmacy sales data simulating cold/flu medication.

        Features:
          - Winter peak (seasonal cosine)
          - Random outbreak spikes
          - Gaussian noise
          - Regional baseline variation
        """
        logger.info("Generating pharmacy sales data …")
        cfg = self.syn["pharmacy"]
        records = []

        for zipcode in tqdm(self.regions, desc="Pharmacy"):
            base = cfg["base_sales_mean"] + self.rng.normal(0, cfg["base_sales_std"] * 0.3)
            base = max(base, 50)

            # Winter peak → phase shift = 0 (peak Jan 1)
            seasonal = seasonal_signal(self.dates, cfg["seasonal_amplitude"], phase_shift_days=15)
            noise = self.rng.normal(0, cfg["noise_std"], self.n_days)
            sales = base + seasonal + noise
            sales = np.clip(sales, 10, None)

            # Outbreak injection
            sales, outbreaks = inject_outbreaks(
                sales,
                self.dates,
                probability=cfg["outbreak_probability"],
                multiplier_range=(
                    cfg["outbreak_multiplier_min"],
                    cfg["outbreak_multiplier_max"],
                ),
                rng=self.rng,
            )

            for i, date in enumerate(self.dates):
                records.append(
                    {
                        "date": date.date(),
                        "zipcode": zipcode,
                        "pharmacy_sales": round(max(sales[i], 0), 2),
                        "outbreak_event": any(
                            o["start_date"] <= str(date.date()) <= o["end_date"]
                            for o in outbreaks
                        ),
                    }
                )

        df = pd.DataFrame(records)
        out_path = self.output_dir / "pharmacy_sales.csv"
        df.to_csv(out_path, index=False)
        logger.info(f"  → Saved {len(df):,} rows to {out_path}")
        return df

    # ------------------------------------------------------------------
    # Social Media Sentiment
    # ------------------------------------------------------------------
    def generate_social_media_sentiment(self) -> pd.DataFrame:
        """
        Generate daily social media sentiment scores per region.

        Metrics:
          - negative_sentiment: fraction of health-related posts with negative tone
          - post_volume: total health-related post count
          - symptom_mention_rate: fraction mentioning respiratory symptoms
        """
        logger.info("Generating social media sentiment data …")
        cfg = self.syn["social_media"]
        records = []

        for zipcode in tqdm(self.regions, desc="Sentiment"):
            base_neg = cfg["base_sentiment_negative"] + self.rng.uniform(-0.02, 0.02)
            seasonal = seasonal_signal(self.dates, cfg["seasonal_amplitude"], phase_shift_days=15)

            neg_sent = base_neg + seasonal + self.rng.normal(0, cfg["noise_std"], self.n_days)
            neg_sent = np.clip(neg_sent, 0.01, 0.95)

            # Post volume: higher in outbreak periods
            base_volume = 500 + self.rng.integers(0, 300)
            volume = base_volume + seasonal * 100 + self.rng.normal(0, 50, self.n_days)

            # Symptom mention rate
            symptom_rate = neg_sent * 0.6 + self.rng.normal(0, 0.02, self.n_days)
            symptom_rate = np.clip(symptom_rate, 0.01, 0.99)

            # Outbreak injection on negative sentiment
            neg_sent, outbreaks = inject_outbreaks(
                neg_sent,
                self.dates,
                probability=0.025,
                multiplier_range=(1.5, 3.0),
                rng=self.rng,
            )

            # Sync volume with sentiment
            for idx in range(self.n_days):
                if neg_sent[idx] > base_neg + 0.1:
                    volume[idx] *= 1.3

            for i, date in enumerate(self.dates):
                records.append(
                    {
                        "date": date.date(),
                        "zipcode": zipcode,
                        "negative_sentiment": round(float(neg_sent[i]), 4),
                        "post_volume": int(max(volume[i], 10)),
                        "symptom_mention_rate": round(float(symptom_rate[i]), 4),
                    }
                )

        df = pd.DataFrame(records)
        out_path = self.output_dir / "social_media_sentiment.csv"
        df.to_csv(out_path, index=False)
        logger.info(f"  → Saved {len(df):,} rows to {out_path}")
        return df

    # ------------------------------------------------------------------
    # Weather
    # ------------------------------------------------------------------
    def generate_weather(self) -> pd.DataFrame:
        """
        Generate daily weather data with realistic seasonal patterns.

        Features:
          - Temperature: sinusoidal seasonal pattern
          - Humidity: inversely correlated with temperature in summer
          - Precipitation: Bernoulli + exponential magnitude
          - Wind speed: random with mild seasonal component
        """
        logger.info("Generating weather data …")
        cfg = self.syn["weather"]
        records = []

        for zipcode in tqdm(self.regions, desc="Weather"):
            meta = self.region_meta[zipcode]
            lat = meta.get("lat", 40.0)

            # Temperature depends on latitude
            lat_factor = 1 - (abs(lat) - 25) / 50  # lower at high latitudes
            temp_mean_summer = cfg["temp_mean_summer"] * lat_factor
            temp_mean_winter = cfg["temp_mean_winter"] - (abs(lat) - 35) * 0.5

            temp_amplitude = (temp_mean_summer - temp_mean_winter) / 2
            temp_midpoint = (temp_mean_summer + temp_mean_winter) / 2

            # Summer peak → phase shift = 182 (July 1)
            temp_seasonal = seasonal_signal(self.dates, temp_amplitude, phase_shift_days=182)
            temperature = temp_midpoint + temp_seasonal + self.rng.normal(0, 3, self.n_days)

            # Humidity
            humidity_seasonal = -seasonal_signal(self.dates, 10, phase_shift_days=182)
            humidity = cfg["humidity_mean"] + humidity_seasonal + self.rng.normal(0, cfg["humidity_std"] * 0.3, self.n_days)
            humidity = np.clip(humidity, 10, 100)

            # Precipitation
            precip_occur = self.rng.random(self.n_days) < cfg["precipitation_prob"]
            precip_amount = self.rng.exponential(scale=10, size=self.n_days) * precip_occur
            precip_amount = np.clip(precip_amount, 0, cfg["precipitation_max"])

            # Wind speed
            wind_speed = self.rng.gamma(shape=2, scale=4, size=self.n_days)

            for i, date in enumerate(self.dates):
                records.append(
                    {
                        "date": date.date(),
                        "zipcode": zipcode,
                        "temperature_c": round(float(temperature[i]), 2),
                        "humidity_pct": round(float(humidity[i]), 2),
                        "precipitation_mm": round(float(precip_amount[i]), 2),
                        "wind_speed_kmh": round(float(wind_speed[i]), 2),
                    }
                )

        df = pd.DataFrame(records)
        out_path = self.output_dir / "weather.csv"
        df.to_csv(out_path, index=False)
        logger.info(f"  → Saved {len(df):,} rows to {out_path}")
        return df

    # ------------------------------------------------------------------
    # Air Quality Index (AQI)
    # ------------------------------------------------------------------
    def generate_aqi(self) -> pd.DataFrame:
        """
        Generate daily AQI data.

        AQI categories: Good (0-50), Moderate (51-100), Unhealthy (101-150),
        Very Unhealthy (151-200), Hazardous (201+)
        """
        logger.info("Generating AQI data …")
        cfg = self.syn["aqi"]
        records = []

        aqi_categories = {
            (0, 50): "Good",
            (51, 100): "Moderate",
            (101, 150): "Unhealthy for Sensitive Groups",
            (151, 200): "Unhealthy",
            (201, 300): "Very Unhealthy",
            (301, 500): "Hazardous",
        }

        def aqi_to_category(val: float) -> str:
            for (lo, hi), label in aqi_categories.items():
                if lo <= val <= hi:
                    return label
            return "Hazardous"

        for zipcode in tqdm(self.regions, desc="AQI"):
            base_aqi = cfg["base_aqi"] + self.rng.normal(0, 10)
            base_aqi = max(base_aqi, 15)

            # Summer peak for ozone/particulate matter
            aqi_seasonal = seasonal_signal(self.dates, cfg["seasonal_amplitude"], phase_shift_days=182)
            aqi = base_aqi + aqi_seasonal + self.rng.normal(0, cfg["aqi_std"] * 0.5, self.n_days)
            aqi = np.clip(aqi, 5, None)

            # Random spikes (wildfires, pollution events)
            spike_mask = self.rng.random(self.n_days) < cfg["spike_probability"]
            spike_vals = self.rng.uniform(1.5, cfg["spike_multiplier"], self.n_days)
            aqi[spike_mask] *= spike_vals[spike_mask]
            aqi = np.clip(aqi, 5, 500)

            # PM2.5 correlates with AQI
            pm25 = aqi * 0.4 + self.rng.normal(0, 3, self.n_days)
            pm25 = np.clip(pm25, 0, None)

            for i, date in enumerate(self.dates):
                records.append(
                    {
                        "date": date.date(),
                        "zipcode": zipcode,
                        "aqi": round(float(aqi[i]), 1),
                        "pm25_ugm3": round(float(pm25[i]), 2),
                        "aqi_category": aqi_to_category(aqi[i]),
                    }
                )

        df = pd.DataFrame(records)
        out_path = self.output_dir / "aqi.csv"
        df.to_csv(out_path, index=False)
        logger.info(f"  → Saved {len(df):,} rows to {out_path}")
        return df

    # ------------------------------------------------------------------
    # ER Visits (Target Variable)
    # ------------------------------------------------------------------
    def generate_er_visits(
        self,
        pharmacy_df: pd.DataFrame,
        weather_df: pd.DataFrame,
        aqi_df: pd.DataFrame,
        sentiment_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate ER visit counts that are causally linked to other signals.

        ER visits depend on:
          - Seasonal baseline (winter peak)
          - Lagged pharmacy sales (3-day lag)
          - Temperature extremes
          - AQI levels
          - Social media negativity

        Args:
            pharmacy_df: Pharmacy sales DataFrame.
            weather_df: Weather DataFrame.
            aqi_df: AQI DataFrame.
            sentiment_df: Social media sentiment DataFrame.

        Returns:
            DataFrame with ER visit counts per region per day.
        """
        logger.info("Generating ER visits data …")
        cfg = self.syn["er_visits"]
        records = []

        for zipcode in tqdm(self.regions, desc="ER Visits"):
            base = cfg["base_visits_mean"] + self.rng.normal(0, cfg["base_visits_std"] * 0.2)
            base = max(base, 20)

            # Winter peak
            seasonal = seasonal_signal(self.dates, cfg["seasonal_amplitude"], phase_shift_days=15)

            # Pull signals for this region
            pharm = pharmacy_df[pharmacy_df["zipcode"] == zipcode].sort_values("date")["pharmacy_sales"].values
            weather_z = weather_df[weather_df["zipcode"] == zipcode].sort_values("date")
            aqi_z = aqi_df[aqi_df["zipcode"] == zipcode].sort_values("date")["aqi"].values
            sent_z = sentiment_df[sentiment_df["zipcode"] == zipcode].sort_values("date")["negative_sentiment"].values

            temp = weather_z["temperature_c"].values
            precip = weather_z["precipitation_mm"].values

            # Lagged pharmacy effect (3-day lag)
            pharm_lag = np.roll(pharm, 3)
            pharm_lag[:3] = pharm[:3]
            pharm_normalized = (pharm_lag - pharm_lag.mean()) / (pharm_lag.std() + 1e-8)

            # Temperature extreme effect (hot or cold stress)
            temp_effect = np.abs(temp - 18) * 0.4  # 18°C is "comfortable"

            # AQI effect
            aqi_normalized = np.clip((aqi_z - 50) / 50, 0, None)

            # Sentiment effect
            sent_normalized = (sent_z - sent_z.mean()) / (sent_z.std() + 1e-8)

            # Precipitation effect (people go out less → some stay home, but outdoor events)
            precip_effect = (precip > 5).astype(float) * 3

            er = (
                base
                + seasonal
                + cfg["pharmacy_lag_coef"] * 10 * pharm_normalized
                + cfg["weather_effect_coef"] * temp_effect
                + cfg["aqi_effect_coef"] * 15 * aqi_normalized
                + 8 * sent_normalized
                + precip_effect
                + self.rng.normal(0, cfg["noise_std"], self.n_days)
            )
            er = np.clip(er, 5, None)

            # Correlated outbreak spikes
            er, outbreaks = inject_outbreaks(
                er,
                self.dates,
                probability=0.02,
                multiplier_range=(1.8, 3.5),
                rng=self.rng,
            )

            for i, date in enumerate(self.dates):
                records.append(
                    {
                        "date": date.date(),
                        "zipcode": zipcode,
                        "er_visits": int(max(round(er[i]), 0)),
                    }
                )

        df = pd.DataFrame(records)
        out_path = self.output_dir / "er_visits.csv"
        df.to_csv(out_path, index=False)
        logger.info(f"  → Saved {len(df):,} rows to {out_path}")
        return df

    # ------------------------------------------------------------------
    # Master Dataset
    # ------------------------------------------------------------------
    def generate_master_dataset(self) -> pd.DataFrame:
        """
        Generate all datasets and merge into a single master CSV.

        Returns:
            Master DataFrame with all features joined on (date, zipcode).
        """
        logger.info("=" * 60)
        logger.info("Starting full synthetic data generation …")
        logger.info("=" * 60)

        pharmacy_df = self.generate_pharmacy_sales()
        sentiment_df = self.generate_social_media_sentiment()
        weather_df = self.generate_weather()
        aqi_df = self.generate_aqi()
        er_df = self.generate_er_visits(pharmacy_df, weather_df, aqi_df, sentiment_df)

        logger.info("Merging all datasets into master CSV …")

        master = pharmacy_df.merge(sentiment_df, on=["date", "zipcode"], how="inner")
        master = master.merge(weather_df, on=["date", "zipcode"], how="inner")
        master = master.merge(aqi_df, on=["date", "zipcode"], how="inner")
        master = master.merge(er_df, on=["date", "zipcode"], how="inner")

        # Add region metadata
        meta_df = pd.DataFrame(list(self.region_meta.values()))
        meta_df = meta_df.rename(columns={"zipcode": "zipcode"})
        master = master.merge(meta_df, on="zipcode", how="left")

        master["date"] = pd.to_datetime(master["date"])
        master = master.sort_values(["zipcode", "date"]).reset_index(drop=True)

        out_path = self.output_dir / "master_dataset.csv"
        master.to_csv(out_path, index=False)
        logger.info(f"Master dataset saved: {len(master):,} rows × {len(master.columns)} cols → {out_path}")

        self._print_summary(master)
        return master

    def _print_summary(self, df: pd.DataFrame) -> None:
        """Print a brief summary of the generated data."""
        logger.info("\n" + "=" * 60)
        logger.info("SYNTHETIC DATA SUMMARY")
        logger.info("=" * 60)
        logger.info(f"  Regions: {df['zipcode'].nunique()}")
        logger.info(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")
        logger.info(f"  Total records: {len(df):,}")
        logger.info(f"  Columns: {list(df.columns)}")
        logger.info("\n  Numeric statistics:")
        num_cols = ["pharmacy_sales", "negative_sentiment", "temperature_c", "aqi", "er_visits"]
        for col in num_cols:
            if col in df.columns:
                logger.info(
                    f"    {col}: mean={df[col].mean():.2f}, "
                    f"std={df[col].std():.2f}, "
                    f"max={df[col].max():.2f}"
                )
        logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic epidemiological datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/params.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/synthetic",
        help="Output directory for synthetic CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)
    generator = SyntheticDataGenerator(config=config, output_dir=args.output)
    generator.generate_master_dataset()
    logger.info("✓ Synthetic data generation complete.")


if __name__ == "__main__":
    main()
