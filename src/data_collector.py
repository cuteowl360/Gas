"""
data_collector.py – GasWatch
────────────────────────────
Handles:
  • WTI crude oil price fetching via yfinance (with fallback)
  • Realistic sample gas-price data generation for 15 US cities
  • Loading / caching data to data/prices.csv
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# City registry  (base price USD/gal regular unleaded ≈ early-2024 average)
# ---------------------------------------------------------------------------
CITIES = {
    "Los Angeles, CA":  {"base": 4.50, "lat": 34.05,  "lon": -118.24},
    "San Francisco, CA":{"base": 4.60, "lat": 37.77,  "lon": -122.42},
    "San Diego, CA":    {"base": 4.45, "lat": 32.72,  "lon": -117.16},
    "New York, NY":     {"base": 3.80, "lat": 40.71,  "lon":  -74.01},
    "Chicago, IL":      {"base": 3.70, "lat": 41.88,  "lon":  -87.63},
    "Houston, TX":      {"base": 3.00, "lat": 29.76,  "lon":  -95.37},
    "Dallas, TX":       {"base": 3.05, "lat": 32.78,  "lon":  -96.80},
    "Phoenix, AZ":      {"base": 3.40, "lat": 33.45,  "lon": -112.07},
    "Seattle, WA":      {"base": 4.20, "lat": 47.61,  "lon": -122.33},
    "Denver, CO":       {"base": 3.50, "lat": 39.74,  "lon": -104.99},
    "Miami, FL":        {"base": 3.60, "lat": 25.77,  "lon":  -80.19},
    "Atlanta, GA":      {"base": 3.20, "lat": 33.75,  "lon":  -84.39},
    "Portland, OR":     {"base": 4.00, "lat": 45.52,  "lon": -122.68},
    "Nashville, TN":    {"base": 3.25, "lat": 36.17,  "lon":  -86.78},
    "Minneapolis, MN":  {"base": 3.55, "lat": 44.98,  "lon":  -93.27},
}

# Resolved path relative to this file's package root
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(_ROOT, "data", "prices.csv")


class DataCollector:
    """Fetches and generates gas/crude price data."""

    def __init__(self):
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

    # ------------------------------------------------------------------
    # Crude oil
    # ------------------------------------------------------------------
    def get_crude_oil_price(self) -> float:
        """Return current WTI crude oil price ($/barrel).
        Uses yfinance; falls back to a sensible default on failure."""
        try:
            import yfinance as yf
            ticker = yf.Ticker("CL=F")
            hist = ticker.history(period="5d")
            if not hist.empty:
                return round(float(hist["Close"].iloc[-1]), 2)
        except Exception:
            pass
        return 75.50  # default fallback

    # ------------------------------------------------------------------
    # Sample data generation
    # ------------------------------------------------------------------
    def _generate_wti_series(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """Generate a plausible WTI daily close series (Jan 2022 – today)."""
        n = len(dates)
        prices = np.zeros(n)
        prices[0] = 78.5

        rng = np.random.default_rng(seed=42)

        for i in range(1, n):
            d = dates[i]
            doy_frac = d.day_of_year / 365.0

            # Macro regime
            if d < pd.Timestamp("2022-03-01"):
                trend = 0.30
            elif d < pd.Timestamp("2022-06-15"):
                trend = 0.10   # peak plateau
            elif d < pd.Timestamp("2023-01-01"):
                trend = -0.22
            elif d < pd.Timestamp("2024-01-01"):
                trend = -0.05
            elif d < pd.Timestamp("2025-06-01"):
                trend = 0.04
            else:
                trend = -0.08

            seasonal = 2.5 * np.sin(2 * np.pi * doy_frac - np.pi / 3) * 0.04
            noise = rng.normal(0, 0.75)
            prices[i] = np.clip(prices[i - 1] + trend + seasonal + noise, 52, 130)

        return np.round(prices, 2)

    def generate_sample_data(self) -> pd.DataFrame:
        """Generate a multi-city, multi-year daily gas price DataFrame."""
        start = pd.Timestamp("2022-01-01")
        end   = pd.Timestamp(datetime.now().date())
        dates = pd.date_range(start, end, freq="D")
        wti   = self._generate_wti_series(dates)

        records = []
        for city, info in CITIES.items():
            base = info["base"]
            rng  = np.random.default_rng(seed=abs(hash(city)) % (2 ** 31))
            city_noise = rng.normal(0, 0.008, len(dates))

            for i, (d, w) in enumerate(zip(dates, wti)):
                doy_frac    = d.day_of_year / 365.0
                seasonal    = 0.18 * np.sin(2 * np.pi * doy_frac - np.pi / 4)
                crude_push  = (w - 70) * 0.018
                year_drift  = (d.year - 2022) * 0.07
                noise       = city_noise[i] + rng.normal(0, 0.004)

                price = max(1.50, base + crude_push + seasonal + noise + year_drift)
                records.append({
                    "date":      d,
                    "city":      city,
                    "price":     round(price, 3),
                    "crude_oil": w,
                    "lat":       info["lat"],
                    "lon":       info["lon"],
                })

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Load / persist
    # ------------------------------------------------------------------
    def load_or_generate_data(self) -> pd.DataFrame:
        """Return cached CSV if up-to-date, otherwise regenerate."""
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH, parse_dates=["date"])
            latest = df["date"].max()
            today  = pd.Timestamp(datetime.now().date())
            if (today - latest).days <= 2:
                return df

        df = self.generate_sample_data()
        df.to_csv(DATA_PATH, index=False)
        return df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def get_city_list(self) -> list:
        return sorted(CITIES.keys())

    def get_city_coords(self) -> dict:
        return {city: (info["lat"], info["lon"]) for city, info in CITIES.items()}
