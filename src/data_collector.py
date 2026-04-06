"""
data_collector.py – GasWatch
Handles:
  - WTI crude oil price fetching via yfinance (with fallback)
  - Realistic sample gas-price data for US + Canadian cities
  - Loading / caching data to data/prices.csv
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# CAD/L <-> USD/gal constants
# base prices are stored internally as USD/gal for model consistency.
# Canadian cities have a cad_per_litre field for display.
# USD/gal = CAD/L x (3.785 L/gal) / (exchange_rate CAD/USD)
# ---------------------------------------------------------------------------
CAD_PER_USD   = 1.36          # approximate exchange rate
LITRES_PER_GAL = 3.785

def cad_litre_to_usd_gal(cad_l: float) -> float:
    return round(cad_l * LITRES_PER_GAL / CAD_PER_USD, 3)

def usd_gal_to_cad_litre(usd_g: float) -> float:
    return round(usd_g * CAD_PER_USD / LITRES_PER_GAL, 3)

# ---------------------------------------------------------------------------
# City registry
# country: "CA" or "US"
# state: province/state abbreviation
# cad_per_litre: only for Canadian cities (for display)
# ---------------------------------------------------------------------------
CITIES = {
    # ── Ontario (default focus) ──
    "Oakville, ON":       {"base": cad_litre_to_usd_gal(1.52), "lat": 43.45, "lon":  -79.68, "state": "ON", "country": "CA", "cad_per_litre": 1.52},
    "Toronto, ON":        {"base": cad_litre_to_usd_gal(1.55), "lat": 43.65, "lon":  -79.38, "state": "ON", "country": "CA", "cad_per_litre": 1.55},
    "Mississauga, ON":    {"base": cad_litre_to_usd_gal(1.52), "lat": 43.59, "lon":  -79.64, "state": "ON", "country": "CA", "cad_per_litre": 1.52},
    "Brampton, ON":       {"base": cad_litre_to_usd_gal(1.53), "lat": 43.73, "lon":  -79.76, "state": "ON", "country": "CA", "cad_per_litre": 1.53},
    "Hamilton, ON":       {"base": cad_litre_to_usd_gal(1.48), "lat": 43.26, "lon":  -79.87, "state": "ON", "country": "CA", "cad_per_litre": 1.48},
    "London, ON":         {"base": cad_litre_to_usd_gal(1.50), "lat": 42.98, "lon":  -81.24, "state": "ON", "country": "CA", "cad_per_litre": 1.50},
    "Kitchener, ON":      {"base": cad_litre_to_usd_gal(1.49), "lat": 43.45, "lon":  -80.49, "state": "ON", "country": "CA", "cad_per_litre": 1.49},
    "Ottawa, ON":         {"base": cad_litre_to_usd_gal(1.50), "lat": 45.42, "lon":  -75.70, "state": "ON", "country": "CA", "cad_per_litre": 1.50},

    # ── Other Canadian cities ──
    "Vancouver, BC":      {"base": cad_litre_to_usd_gal(1.75), "lat": 49.28, "lon": -123.12, "state": "BC", "country": "CA", "cad_per_litre": 1.75},
    "Calgary, AB":        {"base": cad_litre_to_usd_gal(1.40), "lat": 51.05, "lon": -114.07, "state": "AB", "country": "CA", "cad_per_litre": 1.40},
    "Edmonton, AB":       {"base": cad_litre_to_usd_gal(1.35), "lat": 53.55, "lon": -113.49, "state": "AB", "country": "CA", "cad_per_litre": 1.35},
    "Montreal, QC":       {"base": cad_litre_to_usd_gal(1.72), "lat": 45.50, "lon":  -73.57, "state": "QC", "country": "CA", "cad_per_litre": 1.72},
    "Winnipeg, MB":       {"base": cad_litre_to_usd_gal(1.48), "lat": 49.90, "lon":  -97.14, "state": "MB", "country": "CA", "cad_per_litre": 1.48},

    # ── USA: West Coast / Pacific ──
    "Los Angeles, CA":    {"base": 4.50, "lat": 34.05,  "lon": -118.24, "state": "CA", "country": "US"},
    "San Francisco, CA":  {"base": 4.60, "lat": 37.77,  "lon": -122.42, "state": "CA", "country": "US"},
    "San Diego, CA":      {"base": 4.45, "lat": 32.72,  "lon": -117.16, "state": "CA", "country": "US"},
    "Sacramento, CA":     {"base": 4.40, "lat": 38.58,  "lon": -121.49, "state": "CA", "country": "US"},
    "Seattle, WA":        {"base": 4.20, "lat": 47.61,  "lon": -122.33, "state": "WA", "country": "US"},
    "Portland, OR":       {"base": 4.00, "lat": 45.52,  "lon": -122.68, "state": "OR", "country": "US"},
    "Honolulu, HI":       {"base": 5.00, "lat": 21.31,  "lon": -157.86, "state": "HI", "country": "US"},
    "Anchorage, AK":      {"base": 4.30, "lat": 61.22,  "lon": -149.90, "state": "AK", "country": "US"},

    # ── USA: Mountain / Southwest ──
    "Phoenix, AZ":        {"base": 3.40, "lat": 33.45,  "lon": -112.07, "state": "AZ", "country": "US"},
    "Tucson, AZ":         {"base": 3.35, "lat": 32.22,  "lon": -110.97, "state": "AZ", "country": "US"},
    "Denver, CO":         {"base": 3.50, "lat": 39.74,  "lon": -104.99, "state": "CO", "country": "US"},
    "Las Vegas, NV":      {"base": 3.80, "lat": 36.17,  "lon": -115.14, "state": "NV", "country": "US"},
    "Salt Lake City, UT": {"base": 3.55, "lat": 40.76,  "lon": -111.89, "state": "UT", "country": "US"},
    "Albuquerque, NM":    {"base": 3.20, "lat": 35.08,  "lon": -106.65, "state": "NM", "country": "US"},
    "Boise, ID":          {"base": 3.70, "lat": 43.62,  "lon": -116.21, "state": "ID", "country": "US"},

    # ── USA: South / Texas ──
    "Houston, TX":        {"base": 3.00, "lat": 29.76,  "lon":  -95.37, "state": "TX", "country": "US"},
    "Dallas, TX":         {"base": 3.05, "lat": 32.78,  "lon":  -96.80, "state": "TX", "country": "US"},
    "San Antonio, TX":    {"base": 2.98, "lat": 29.42,  "lon":  -98.49, "state": "TX", "country": "US"},
    "Austin, TX":         {"base": 3.02, "lat": 30.27,  "lon":  -97.74, "state": "TX", "country": "US"},
    "New Orleans, LA":    {"base": 3.10, "lat": 29.95,  "lon":  -90.07, "state": "LA", "country": "US"},
    "Oklahoma City, OK":  {"base": 3.05, "lat": 35.47,  "lon":  -97.52, "state": "OK", "country": "US"},

    # ── USA: Southeast ──
    "Miami, FL":          {"base": 3.60, "lat": 25.77,  "lon":  -80.19, "state": "FL", "country": "US"},
    "Orlando, FL":        {"base": 3.55, "lat": 28.54,  "lon":  -81.38, "state": "FL", "country": "US"},
    "Atlanta, GA":        {"base": 3.20, "lat": 33.75,  "lon":  -84.39, "state": "GA", "country": "US"},
    "Nashville, TN":      {"base": 3.25, "lat": 36.17,  "lon":  -86.78, "state": "TN", "country": "US"},
    "Charlotte, NC":      {"base": 3.30, "lat": 35.23,  "lon":  -80.84, "state": "NC", "country": "US"},
    "Raleigh, NC":        {"base": 3.28, "lat": 35.78,  "lon":  -78.64, "state": "NC", "country": "US"},
    "Virginia Beach, VA": {"base": 3.35, "lat": 36.85,  "lon":  -75.98, "state": "VA", "country": "US"},

    # ── USA: Midwest ──
    "Chicago, IL":        {"base": 3.70, "lat": 41.88,  "lon":  -87.63, "state": "IL", "country": "US"},
    "Minneapolis, MN":    {"base": 3.55, "lat": 44.98,  "lon":  -93.27, "state": "MN", "country": "US"},
    "Detroit, MI":        {"base": 3.45, "lat": 42.33,  "lon":  -83.05, "state": "MI", "country": "US"},
    "Columbus, OH":       {"base": 3.40, "lat": 39.96,  "lon":  -82.99, "state": "OH", "country": "US"},
    "Indianapolis, IN":   {"base": 3.38, "lat": 39.77,  "lon":  -86.16, "state": "IN", "country": "US"},
    "Kansas City, MO":    {"base": 3.15, "lat": 39.10,  "lon":  -94.58, "state": "MO", "country": "US"},
    "St. Louis, MO":      {"base": 3.12, "lat": 38.63,  "lon":  -90.20, "state": "MO", "country": "US"},

    # ── USA: Northeast ──
    "New York, NY":       {"base": 3.80, "lat": 40.71,  "lon":  -74.01, "state": "NY", "country": "US"},
    "Philadelphia, PA":   {"base": 3.65, "lat": 39.95,  "lon":  -75.17, "state": "PA", "country": "US"},
    "Boston, MA":         {"base": 3.75, "lat": 42.36,  "lon":  -71.06, "state": "MA", "country": "US"},
    "Washington, DC":     {"base": 3.55, "lat": 38.91,  "lon":  -77.04, "state": "DC", "country": "US"},
    "Baltimore, MD":      {"base": 3.50, "lat": 39.29,  "lon":  -76.61, "state": "MD", "country": "US"},
    "Pittsburgh, PA":     {"base": 3.60, "lat": 40.44,  "lon":  -79.99, "state": "PA", "country": "US"},
}

# Gas type multipliers (relative to regular)
GAS_TYPES = {
    "Regular":  1.000,
    "Midgrade": 1.110,
    "Premium":  1.200,
    "Diesel":   1.045,
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
                    "state":     info["state"],
                    "country":   info.get("country", "US"),
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
