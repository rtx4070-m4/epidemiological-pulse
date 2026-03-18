"""
Epidemiological Pulse – Data Collectors
========================================
Provides modular data collection classes for each data source:
  - PharmacyDataCollector  : OTC sales (file or API)
  - SocialMediaCollector   : Twitter/Reddit health sentiment
  - WeatherCollector       : OpenWeatherMap historical weather
  - AQICollector           : IQAir / AirVisual AQI data
  - ERVisitsCollector      : CDC Wonder / local health dept ER data

All collectors implement a common interface:
  .collect(start_date, end_date, zipcodes) -> pd.DataFrame

In production, replace synthetic loaders with real API calls.
When api_keys.yaml is absent, collectors fall back to synthetic data.
"""

import logging
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base Collector
# ---------------------------------------------------------------------------

class BaseCollector(ABC):
    """Abstract base class for all data collectors."""

    RETRY_ATTEMPTS: int = 3
    RETRY_DELAY_SEC: float = 2.0

    def __init__(
        self,
        config: Dict,
        api_keys: Optional[Dict] = None,
        synthetic_data_dir: Optional[str] = None,
    ) -> None:
        """
        Initialize collector.

        Args:
            config: Main params.yaml config dict.
            api_keys: Optional api_keys.yaml dict.
            synthetic_data_dir: Path to pre-generated synthetic CSVs.
        """
        self.config = config
        self.api_keys = api_keys or {}
        self.synthetic_dir = Path(synthetic_data_dir) if synthetic_data_dir else None
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def collect(
        self,
        start_date: str,
        end_date: str,
        zipcodes: List[str],
    ) -> pd.DataFrame:
        """
        Collect data for the given date range and regions.

        Args:
            start_date: ISO date string 'YYYY-MM-DD'.
            end_date: ISO date string 'YYYY-MM-DD'.
            zipcodes: List of ZIP code strings.

        Returns:
            Collected DataFrame with at minimum (date, zipcode) columns.
        """
        ...

    def _get(self, url: str, params: Dict = None, headers: Dict = None) -> Dict:
        """
        HTTP GET with retry logic.

        Args:
            url: Request URL.
            params: Query parameters.
            headers: Request headers.

        Returns:
            Parsed JSON response dict.

        Raises:
            requests.HTTPError: After all retries exhausted.
        """
        for attempt in range(1, self.RETRY_ATTEMPTS + 1):
            try:
                resp = requests.get(url, params=params, headers=headers, timeout=30)
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                self.logger.warning(
                    f"Attempt {attempt}/{self.RETRY_ATTEMPTS} failed for {url}: {exc}"
                )
                if attempt < self.RETRY_ATTEMPTS:
                    time.sleep(self.RETRY_DELAY_SEC * attempt)
                else:
                    raise

    def _load_synthetic(self, filename: str) -> Optional[pd.DataFrame]:
        """Load a synthetic CSV if available."""
        if self.synthetic_dir is None:
            return None
        path = self.synthetic_dir / filename
        if path.exists():
            self.logger.info(f"Loading synthetic data from {path}")
            df = pd.read_csv(path, parse_dates=["date"])
            return df
        return None

    @staticmethod
    def _filter_by_date_and_zip(
        df: pd.DataFrame,
        start_date: str,
        end_date: str,
        zipcodes: List[str],
    ) -> pd.DataFrame:
        """Filter DataFrame by date range and ZIP codes."""
        mask = (
            (df["date"] >= pd.to_datetime(start_date))
            & (df["date"] <= pd.to_datetime(end_date))
            & (df["zipcode"].astype(str).isin([str(z) for z in zipcodes]))
        )
        return df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Pharmacy Collector
# ---------------------------------------------------------------------------

class PharmacyDataCollector(BaseCollector):
    """
    Collects OTC pharmacy sales data.

    Production: Connect to pharmacy chain APIs (e.g., Walgreens Health API,
    CVS data partnership, or state health department feeds).
    Dev/Test: Falls back to synthetic pharmacy_sales.csv.
    """

    def collect(
        self,
        start_date: str,
        end_date: str,
        zipcodes: List[str],
    ) -> pd.DataFrame:
        """
        Collect pharmacy OTC sales data.

        Returns DataFrame with columns:
          date, zipcode, pharmacy_sales, outbreak_event
        """
        self.logger.info(
            f"Collecting pharmacy data: {start_date} → {end_date}, "
            f"{len(zipcodes)} ZIP codes"
        )

        # Try loading synthetic data first
        df = self._load_synthetic("pharmacy_sales.csv")
        if df is not None:
            return self._filter_by_date_and_zip(df, start_date, end_date, zipcodes)

        # Production API call (placeholder for real integration)
        self.logger.warning("No synthetic data found; pharmacy API not configured. Returning empty DataFrame.")
        return pd.DataFrame(columns=["date", "zipcode", "pharmacy_sales", "outbreak_event"])

    def collect_from_api(
        self,
        start_date: str,
        end_date: str,
        zipcodes: List[str],
    ) -> pd.DataFrame:
        """
        Placeholder for real pharmacy API integration.

        Replace this method body with actual HTTP calls to your
        pharmacy data provider (e.g., IQVIA, Symphony Health).
        """
        raise NotImplementedError(
            "Production pharmacy API not implemented. "
            "Set synthetic_data_dir to use synthetic data."
        )


# ---------------------------------------------------------------------------
# Social Media Collector
# ---------------------------------------------------------------------------

class SocialMediaCollector(BaseCollector):
    """
    Collects health-related social media sentiment.

    Production: Twitter API v2 (filtered stream + recent search) and
    Reddit API (pushshift or official) for health-related posts.
    """

    HEALTH_KEYWORDS = [
        "flu", "sick", "fever", "cough", "cold", "hospital", "ER",
        "COVID", "covid", "pandemic", "outbreak", "symptoms", "nausea",
        "vomiting", "infection", "quarantine", "positive test",
    ]

    def collect(
        self,
        start_date: str,
        end_date: str,
        zipcodes: List[str],
    ) -> pd.DataFrame:
        """
        Collect social media sentiment data.

        Returns DataFrame with columns:
          date, zipcode, negative_sentiment, post_volume, symptom_mention_rate
        """
        self.logger.info(
            f"Collecting social media sentiment: {start_date} → {end_date}"
        )

        df = self._load_synthetic("social_media_sentiment.csv")
        if df is not None:
            return self._filter_by_date_and_zip(df, start_date, end_date, zipcodes)

        # Try Twitter API if credentials available
        twitter_cfg = self.api_keys.get("twitter", {})
        if twitter_cfg.get("bearer_token"):
            try:
                return self._collect_twitter(start_date, end_date, zipcodes, twitter_cfg)
            except Exception as exc:
                self.logger.error(f"Twitter collection failed: {exc}")

        self.logger.warning("Social media API not configured. Returning empty DataFrame.")
        return pd.DataFrame(
            columns=["date", "zipcode", "negative_sentiment", "post_volume", "symptom_mention_rate"]
        )

    def _collect_twitter(
        self,
        start_date: str,
        end_date: str,
        zipcodes: List[str],
        twitter_cfg: Dict,
    ) -> pd.DataFrame:
        """
        Collect from Twitter API v2 Recent Search.

        Note: Free tier limited to 7 days lookback.
        Full archive requires Academic/Pro access.
        """
        bearer = twitter_cfg["bearer_token"]
        headers = {"Authorization": f"Bearer {bearer}"}
        query = " OR ".join(self.HEALTH_KEYWORDS[:5]) + " lang:en"
        url = "https://api.twitter.com/2/tweets/search/recent"

        records = []
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
        current_dt = start_dt

        while current_dt <= end_dt:
            try:
                params = {
                    "query": query,
                    "start_time": current_dt.isoformat() + "Z",
                    "end_time": (current_dt + timedelta(days=1)).isoformat() + "Z",
                    "max_results": 100,
                    "tweet.fields": "created_at,geo",
                }
                data = self._get(url, params=params, headers=headers)
                tweets = data.get("data", [])

                negative_count = sum(
                    1 for t in tweets
                    if any(
                        kw in t.get("text", "").lower()
                        for kw in ["sick", "fever", "hospital", "symptoms"]
                    )
                )
                total = max(len(tweets), 1)

                # Without geolocation (most tweets lack geo), assign to first zipcode
                for zipcode in zipcodes:
                    records.append(
                        {
                            "date": current_dt.date(),
                            "zipcode": zipcode,
                            "negative_sentiment": round(negative_count / total, 4),
                            "post_volume": total,
                            "symptom_mention_rate": round(negative_count / total * 0.7, 4),
                        }
                    )
            except Exception as exc:
                self.logger.warning(f"Twitter error for {current_dt.date()}: {exc}")

            current_dt += timedelta(days=1)
            time.sleep(0.5)  # rate limiting

        return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Weather Collector
# ---------------------------------------------------------------------------

class WeatherCollector(BaseCollector):
    """
    Collects historical and forecast weather data.

    Production: OpenWeatherMap Historical Weather API.
    """

    OWM_HIST_URL = "https://api.openweathermap.org/data/2.5/onecall/timemachine"
    OWM_FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"

    def collect(
        self,
        start_date: str,
        end_date: str,
        zipcodes: List[str],
    ) -> pd.DataFrame:
        """
        Collect weather data.

        Returns DataFrame with columns:
          date, zipcode, temperature_c, humidity_pct, precipitation_mm, wind_speed_kmh
        """
        self.logger.info(f"Collecting weather data: {start_date} → {end_date}")

        df = self._load_synthetic("weather.csv")
        if df is not None:
            return self._filter_by_date_and_zip(df, start_date, end_date, zipcodes)

        owm_key = self.api_keys.get("openweathermap", {}).get("api_key")
        if owm_key:
            try:
                return self._collect_owm(start_date, end_date, zipcodes, owm_key)
            except Exception as exc:
                self.logger.error(f"OpenWeatherMap collection failed: {exc}")

        self.logger.warning("Weather API not configured. Returning empty DataFrame.")
        return pd.DataFrame(
            columns=["date", "zipcode", "temperature_c", "humidity_pct",
                     "precipitation_mm", "wind_speed_kmh"]
        )

    def _collect_owm(
        self,
        start_date: str,
        end_date: str,
        zipcodes: List[str],
        api_key: str,
    ) -> pd.DataFrame:
        """Collect from OpenWeatherMap API."""
        regions = self.config["data"]["regions"]
        region_map = {r["zipcode"]: r for r in regions}
        records = []

        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)

        for zipcode in zipcodes:
            region = region_map.get(zipcode)
            if not region:
                continue

            lat, lon = region["lat"], region["lon"]
            current_dt = start_dt

            while current_dt <= end_dt:
                timestamp = int(current_dt.timestamp())
                try:
                    data = self._get(
                        self.OWM_HIST_URL,
                        params={
                            "lat": lat,
                            "lon": lon,
                            "dt": timestamp,
                            "appid": api_key,
                            "units": "metric",
                        },
                    )
                    daily = data.get("current", {})
                    records.append(
                        {
                            "date": current_dt.date(),
                            "zipcode": zipcode,
                            "temperature_c": daily.get("temp", None),
                            "humidity_pct": daily.get("humidity", None),
                            "precipitation_mm": daily.get("rain", {}).get("1h", 0) * 24,
                            "wind_speed_kmh": daily.get("wind_speed", None),
                        }
                    )
                except Exception as exc:
                    self.logger.warning(f"OWM error {zipcode} {current_dt.date()}: {exc}")

                current_dt += timedelta(days=1)
                time.sleep(1.0)

        return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# AQI Collector
# ---------------------------------------------------------------------------

class AQICollector(BaseCollector):
    """
    Collects Air Quality Index data.

    Production: IQAir / AirVisual API.
    """

    def collect(
        self,
        start_date: str,
        end_date: str,
        zipcodes: List[str],
    ) -> pd.DataFrame:
        """
        Collect AQI data.

        Returns DataFrame with columns:
          date, zipcode, aqi, pm25_ugm3, aqi_category
        """
        self.logger.info(f"Collecting AQI data: {start_date} → {end_date}")

        df = self._load_synthetic("aqi.csv")
        if df is not None:
            return self._filter_by_date_and_zip(df, start_date, end_date, zipcodes)

        iqair_key = self.api_keys.get("iqair", {}).get("api_key")
        if iqair_key:
            try:
                return self._collect_iqair(start_date, end_date, zipcodes, iqair_key)
            except Exception as exc:
                self.logger.error(f"IQAir collection failed: {exc}")

        self.logger.warning("AQI API not configured. Returning empty DataFrame.")
        return pd.DataFrame(columns=["date", "zipcode", "aqi", "pm25_ugm3", "aqi_category"])

    def _collect_iqair(
        self,
        start_date: str,
        end_date: str,
        zipcodes: List[str],
        api_key: str,
    ) -> pd.DataFrame:
        """Collect from IQAir API (nearest station)."""
        regions = self.config["data"]["regions"]
        region_map = {r["zipcode"]: r for r in regions}
        records = []
        base_url = self.api_keys.get("iqair", {}).get("base_url", "https://api.airvisual.com/v2")

        for zipcode in zipcodes:
            region = region_map.get(zipcode)
            if not region:
                continue
            try:
                data = self._get(
                    f"{base_url}/nearest_city",
                    params={
                        "lat": region["lat"],
                        "lon": region["lon"],
                        "key": api_key,
                    },
                )
                current = data.get("data", {}).get("current", {}).get("pollution", {})
                records.append(
                    {
                        "date": datetime.now().date(),
                        "zipcode": zipcode,
                        "aqi": current.get("aqius", None),
                        "pm25_ugm3": current.get("p2", {}).get("conc", None),
                        "aqi_category": "Unknown",
                    }
                )
            except Exception as exc:
                self.logger.warning(f"IQAir error {zipcode}: {exc}")
            time.sleep(1.0)

        return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# ER Visits Collector
# ---------------------------------------------------------------------------

class ERVisitsCollector(BaseCollector):
    """
    Collects emergency room visit counts from health authority APIs.

    Production: CDC Wonder API, state health department data APIs,
    or syndromic surveillance systems (e.g., ESSENCE, BioSense Platform).
    """

    def collect(
        self,
        start_date: str,
        end_date: str,
        zipcodes: List[str],
    ) -> pd.DataFrame:
        """
        Collect ER visit data.

        Returns DataFrame with columns:
          date, zipcode, er_visits
        """
        self.logger.info(f"Collecting ER visit data: {start_date} → {end_date}")

        df = self._load_synthetic("er_visits.csv")
        if df is not None:
            return self._filter_by_date_and_zip(df, start_date, end_date, zipcodes)

        cdc_cfg = self.api_keys.get("cdc", {})
        if cdc_cfg.get("er_visits_dataset_id"):
            try:
                return self._collect_cdc(start_date, end_date, zipcodes, cdc_cfg)
            except Exception as exc:
                self.logger.error(f"CDC collection failed: {exc}")

        self.logger.warning("ER visit data source not configured. Returning empty DataFrame.")
        return pd.DataFrame(columns=["date", "zipcode", "er_visits"])

    def _collect_cdc(
        self,
        start_date: str,
        end_date: str,
        zipcodes: List[str],
        cdc_cfg: Dict,
    ) -> pd.DataFrame:
        """Collect from CDC data portal using Socrata API."""
        base_url = cdc_cfg.get("base_url", "https://data.cdc.gov/resource")
        dataset_id = cdc_cfg["er_visits_dataset_id"]
        url = f"{base_url}/{dataset_id}.json"

        records = []
        for zipcode in zipcodes:
            try:
                params = {
                    "$where": (
                        f"zip_code='{zipcode}' "
                        f"AND date >= '{start_date}' "
                        f"AND date <= '{end_date}'"
                    ),
                    "$limit": 10000,
                }
                data = self._get(url, params=params)
                for row in data:
                    records.append(
                        {
                            "date": row.get("date", ""),
                            "zipcode": zipcode,
                            "er_visits": int(row.get("visit_count", 0)),
                        }
                    )
            except Exception as exc:
                self.logger.warning(f"CDC error {zipcode}: {exc}")
            time.sleep(0.3)

        return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Collector Factory
# ---------------------------------------------------------------------------

class CollectorFactory:
    """Factory to instantiate all collectors with shared config."""

    def __init__(
        self,
        config: Dict,
        api_keys_path: Optional[str] = None,
        synthetic_data_dir: Optional[str] = None,
    ) -> None:
        """
        Initialize factory.

        Args:
            config: Parsed params.yaml dict.
            api_keys_path: Path to api_keys.yaml (optional).
            synthetic_data_dir: Path to synthetic CSV directory.
        """
        self.config = config
        self.synthetic_data_dir = synthetic_data_dir

        self.api_keys: Dict = {}
        if api_keys_path and os.path.exists(api_keys_path):
            with open(api_keys_path) as f:
                self.api_keys = yaml.safe_load(f) or {}
            logger.info(f"Loaded API keys from {api_keys_path}")
        else:
            logger.info("No API keys file found; using synthetic data mode.")

    def get_pharmacy_collector(self) -> PharmacyDataCollector:
        return PharmacyDataCollector(self.config, self.api_keys, self.synthetic_data_dir)

    def get_social_media_collector(self) -> SocialMediaCollector:
        return SocialMediaCollector(self.config, self.api_keys, self.synthetic_data_dir)

    def get_weather_collector(self) -> WeatherCollector:
        return WeatherCollector(self.config, self.api_keys, self.synthetic_data_dir)

    def get_aqi_collector(self) -> AQICollector:
        return AQICollector(self.config, self.api_keys, self.synthetic_data_dir)

    def get_er_visits_collector(self) -> ERVisitsCollector:
        return ERVisitsCollector(self.config, self.api_keys, self.synthetic_data_dir)

    def collect_all(
        self,
        start_date: str,
        end_date: str,
        zipcodes: List[str],
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect all data sources.

        Args:
            start_date: Start date string.
            end_date: End date string.
            zipcodes: List of ZIP codes.

        Returns:
            Dict mapping source name → DataFrame.
        """
        logger.info("Collecting all data sources …")
        return {
            "pharmacy": self.get_pharmacy_collector().collect(start_date, end_date, zipcodes),
            "social_media": self.get_social_media_collector().collect(start_date, end_date, zipcodes),
            "weather": self.get_weather_collector().collect(start_date, end_date, zipcodes),
            "aqi": self.get_aqi_collector().collect(start_date, end_date, zipcodes),
            "er_visits": self.get_er_visits_collector().collect(start_date, end_date, zipcodes),
        }
