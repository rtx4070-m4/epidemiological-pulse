"""
Epidemiological Pulse – Geocoder
==================================
Provides ZIP code → (lat, lon, city, state) resolution.

Supports:
  - In-memory lookup from config
  - US Census Bureau Geocoder API (free, no key required)
  - Google Maps Geocoding API (optional, requires key)
  - Offline fallback using pre-built ZIP → lat/lon mapping
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Static Fallback ZIP→Lat/Lon (sampled major US ZIPs)
# ---------------------------------------------------------------------------

ZIPCODE_FALLBACK: Dict[str, Dict] = {
    "10001": {"lat": 40.7484, "lon": -73.9967, "city": "New York",      "state": "NY"},
    "90001": {"lat": 33.9731, "lon": -118.2479, "city": "Los Angeles",  "state": "CA"},
    "60601": {"lat": 41.8858, "lon": -87.6181, "city": "Chicago",       "state": "IL"},
    "77001": {"lat": 29.7543, "lon": -95.3677, "city": "Houston",       "state": "TX"},
    "85001": {"lat": 33.4484, "lon": -112.0740, "city": "Phoenix",      "state": "AZ"},
    "19101": {"lat": 39.9526, "lon": -75.1652, "city": "Philadelphia",  "state": "PA"},
    "78201": {"lat": 29.4241, "lon": -98.4936, "city": "San Antonio",   "state": "TX"},
    "75201": {"lat": 32.7767, "lon": -96.7970, "city": "Dallas",        "state": "TX"},
    "95101": {"lat": 37.3382, "lon": -121.8863, "city": "San Jose",     "state": "CA"},
    "78701": {"lat": 30.2672, "lon": -97.7431, "city": "Austin",        "state": "TX"},
    "32801": {"lat": 30.3322, "lon": -81.6557, "city": "Jacksonville",  "state": "FL"},
    "94102": {"lat": 37.7749, "lon": -122.4194, "city": "San Francisco","state": "CA"},
    "43201": {"lat": 39.9612, "lon": -82.9988, "city": "Columbus",      "state": "OH"},
    "28201": {"lat": 35.2271, "lon": -80.8431, "city": "Charlotte",     "state": "NC"},
    "46201": {"lat": 39.7684, "lon": -86.1581, "city": "Indianapolis",  "state": "IN"},
    "98101": {"lat": 47.6062, "lon": -122.3321, "city": "Seattle",      "state": "WA"},
    "80202": {"lat": 39.7392, "lon": -104.9903, "city": "Denver",       "state": "CO"},
    "20001": {"lat": 38.9072, "lon": -77.0369, "city": "Washington",    "state": "DC"},
    "02101": {"lat": 42.3601, "lon": -71.0589, "city": "Boston",        "state": "MA"},
    "37201": {"lat": 36.1627, "lon": -86.7816, "city": "Nashville",     "state": "TN"},
    "97201": {"lat": 45.5051, "lon": -122.6750, "city": "Portland",     "state": "OR"},
    "89101": {"lat": 36.1699, "lon": -115.1398, "city": "Las Vegas",    "state": "NV"},
    "55401": {"lat": 44.9778, "lon": -93.2650, "city": "Minneapolis",   "state": "MN"},
    "85701": {"lat": 32.2540, "lon": -110.9742, "city": "Tucson",       "state": "AZ"},
    "30301": {"lat": 33.7490, "lon": -84.3880, "city": "Atlanta",       "state": "GA"},
}


# ---------------------------------------------------------------------------
# Geocoder Base
# ---------------------------------------------------------------------------

class BaseGeocoder:
    """Abstract base for ZIP code geocoders."""

    def geocode(self, zipcode: str) -> Optional[Dict]:
        """
        Resolve a ZIP code to location information.

        Args:
            zipcode: 5-digit US ZIP code string.

        Returns:
            Dict with keys: lat, lon, city, state, or None on failure.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Config-based Geocoder (fastest, config-driven)
# ---------------------------------------------------------------------------

class ConfigGeocoder(BaseGeocoder):
    """Resolves from config regions list."""

    def __init__(self, config: Dict) -> None:
        regions = config.get("data", {}).get("regions", [])
        self._lookup: Dict[str, Dict] = {
            str(r["zipcode"]): {
                "lat": r["lat"],
                "lon": r["lon"],
                "city": r.get("city", ""),
                "state": r.get("state", ""),
            }
            for r in regions
        }

    def geocode(self, zipcode: str) -> Optional[Dict]:
        return self._lookup.get(str(zipcode))


# ---------------------------------------------------------------------------
# Fallback Static Geocoder
# ---------------------------------------------------------------------------

class StaticFallbackGeocoder(BaseGeocoder):
    """Uses built-in ZIP→lat/lon mapping."""

    def geocode(self, zipcode: str) -> Optional[Dict]:
        return ZIPCODE_FALLBACK.get(str(zipcode))


# ---------------------------------------------------------------------------
# Census Bureau Geocoder
# ---------------------------------------------------------------------------

class CensusGeocoder(BaseGeocoder):
    """
    Uses US Census Bureau Geocoding Service (free, no API key needed).
    https://geocoding.geo.census.gov/geocoder/
    """

    BASE_URL = "https://geocoding.geo.census.gov/geocoder/locations/address"
    RETRY_ATTEMPTS = 3
    RETRY_DELAY = 2.0

    def geocode(self, zipcode: str) -> Optional[Dict]:
        """
        Geocode a ZIP code by querying Census Bureau.

        Args:
            zipcode: 5-digit ZIP code.

        Returns:
            Location dict or None.
        """
        for attempt in range(self.RETRY_ATTEMPTS):
            try:
                params = {
                    "street": "",
                    "city": "",
                    "state": "",
                    "zip": zipcode,
                    "benchmark": "Public_AR_Current",
                    "format": "json",
                }
                resp = requests.get(self.BASE_URL, params=params, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                results = data.get("result", {}).get("addressMatches", [])
                if results:
                    coords = results[0]["coordinates"]
                    components = results[0].get("addressComponents", {})
                    return {
                        "lat": float(coords["y"]),
                        "lon": float(coords["x"]),
                        "city": components.get("city", ""),
                        "state": components.get("state", ""),
                    }
                return None
            except Exception as exc:
                logger.warning(
                    f"Census geocoder attempt {attempt + 1} failed for {zipcode}: {exc}"
                )
                if attempt < self.RETRY_ATTEMPTS - 1:
                    time.sleep(self.RETRY_DELAY)
        return None


# ---------------------------------------------------------------------------
# Google Maps Geocoder
# ---------------------------------------------------------------------------

class GoogleGeocoder(BaseGeocoder):
    """Google Maps Geocoding API (requires API key)."""

    BASE_URL = "https://maps.googleapis.com/maps/api/geocode/json"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def geocode(self, zipcode: str) -> Optional[Dict]:
        """
        Geocode a ZIP code using Google Maps API.

        Args:
            zipcode: ZIP code string.

        Returns:
            Location dict or None.
        """
        try:
            params = {
                "address": f"{zipcode}, USA",
                "key": self.api_key,
                "components": f"postal_code:{zipcode}|country:US",
            }
            resp = requests.get(self.BASE_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if data.get("status") == "OK":
                result = data["results"][0]
                loc = result["geometry"]["location"]
                city = ""
                state = ""
                for comp in result.get("address_components", []):
                    if "locality" in comp["types"]:
                        city = comp["long_name"]
                    if "administrative_area_level_1" in comp["types"]:
                        state = comp["short_name"]
                return {
                    "lat": loc["lat"],
                    "lon": loc["lng"],
                    "city": city,
                    "state": state,
                }
        except Exception as exc:
            logger.warning(f"Google geocoder failed for {zipcode}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Chain Geocoder (tries multiple providers)
# ---------------------------------------------------------------------------

class ChainGeocoder:
    """
    Tries multiple geocoders in order until one succeeds.

    Priority: Config → Static fallback → Census → Google (if key available)
    """

    def __init__(
        self,
        config: Dict,
        google_api_key: Optional[str] = None,
    ) -> None:
        """
        Initialize chain.

        Args:
            config: Main config dict.
            google_api_key: Optional Google Maps API key.
        """
        self._chain: List[BaseGeocoder] = [
            ConfigGeocoder(config),
            StaticFallbackGeocoder(),
        ]
        # Census can be slow; add as third-level fallback
        self._chain.append(CensusGeocoder())

        if google_api_key:
            self._chain.append(GoogleGeocoder(google_api_key))

        self._cache: Dict[str, Optional[Dict]] = {}

    def geocode(self, zipcode: str) -> Optional[Dict]:
        """
        Geocode with chain fallback and caching.

        Args:
            zipcode: ZIP code string.

        Returns:
            Location dict or None if all providers fail.
        """
        zipcode = str(zipcode)

        if zipcode in self._cache:
            return self._cache[zipcode]

        for provider in self._chain:
            result = provider.geocode(zipcode)
            if result:
                self._cache[zipcode] = result
                return result

        logger.warning(f"All geocoders failed for ZIP {zipcode}")
        self._cache[zipcode] = None
        return None

    def geocode_dataframe(
        self,
        df: pd.DataFrame,
        zipcode_col: str = "zipcode",
    ) -> pd.DataFrame:
        """
        Add lat/lon/city/state columns to a DataFrame.

        Args:
            df: DataFrame with a ZIP code column.
            zipcode_col: Name of the ZIP code column.

        Returns:
            DataFrame with lat, lon, city, state columns added.
        """
        logger.info(f"Geocoding {df[zipcode_col].nunique()} unique ZIP codes …")
        unique_zips = df[zipcode_col].unique()

        geo_records = []
        for zipcode in unique_zips:
            result = self.geocode(zipcode)
            if result:
                geo_records.append({zipcode_col: str(zipcode), **result})
            else:
                geo_records.append(
                    {zipcode_col: str(zipcode), "lat": None, "lon": None, "city": "", "state": ""}
                )

        geo_df = pd.DataFrame(geo_records)
        df = df.copy()
        df[zipcode_col] = df[zipcode_col].astype(str)

        # Drop existing columns to avoid conflicts
        for col in ["lat", "lon", "city", "state"]:
            if col in df.columns:
                df = df.drop(columns=[col])

        merged = df.merge(geo_df, on=zipcode_col, how="left")
        missing_geo = merged["lat"].isnull().sum()
        if missing_geo > 0:
            logger.warning(f"{missing_geo} rows have missing geocoordinates.")

        return merged

    def get_coordinates(
        self,
        zipcode: str,
    ) -> Optional[Tuple[float, float]]:
        """
        Convenience method returning just (lat, lon).

        Args:
            zipcode: ZIP code string.

        Returns:
            (lat, lon) tuple or None.
        """
        result = self.geocode(zipcode)
        if result:
            return result["lat"], result["lon"]
        return None
