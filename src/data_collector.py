"""
data_collector.py – GasWatch
Handles:
  - Real Canadian gas prices from NRCan (Natural Resources Canada)
  - Real US gas prices from AAA state pages
  - WTI crude oil history from yfinance for US backfill
  - Loading / caching data to data/prices.csv  (12-hour TTL)
"""

import io
import os
import re
import ssl
import time
import urllib.request

import numpy as np
import openpyxl
import pandas as pd
from datetime import datetime, timedelta

# ── SSL / HTTP helpers ──────────────────────────────────────────────────────
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

def _get(url: str, timeout: int = 20) -> bytes:
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as r:
        return r.read()

# ── Unit conversion ─────────────────────────────────────────────────────────
CAD_PER_USD    = 1.36
LITRES_PER_GAL = 3.785

def cad_litre_to_usd_gal(cad_l: float) -> float:
    return round(cad_l * LITRES_PER_GAL / CAD_PER_USD, 3)

def usd_gal_to_cad_litre(usd_g: float) -> float:
    return round(usd_g * CAD_PER_USD / LITRES_PER_GAL, 3)

# ── NRCan location names for Canadian cities ─────────────────────────────────
# NRCan reports data for: Toronto, Ottawa, Vancouver, Calgary, Edmonton,
# Winnipeg, Montreal.  GTA proxy cities share Toronto's series ± small offset.
NRCAN_LOCATION = {
    "Toronto, ON":      "Toronto",
    "Oakville, ON":     "Toronto",
    "Mississauga, ON":  "Toronto",
    "Brampton, ON":     "Toronto",
    "Hamilton, ON":     "Toronto",
    "London, ON":       "Toronto",
    "Kitchener, ON":    "Toronto",
    "Ottawa, ON":       "Ottawa",
    "Vancouver, BC":    "Vancouver",
    "Calgary, AB":      "Calgary",
    "Edmonton, AB":     "Edmonton",
    "Winnipeg, MB":     "Winnipeg",
    "Montreal, QC":     "Montreal",
}

# Small CAD/L offset applied so GTA cities are not all identical to Toronto
GTA_OFFSETS = {
    "Oakville, ON":     -0.010,
    "Mississauga, ON":  -0.010,
    "Brampton, ON":      0.010,
    "Hamilton, ON":     -0.020,
    "London, ON":        0.025,
    "Kitchener, ON":     0.015,
}

# AAA state code for each US city's state
AAA_STATE = {
    "AL": "AL", "AK": "AK", "AZ": "AZ", "AR": "AR", "CA": "CA",
    "CO": "CO", "CT": "CT", "DC": "DC", "DE": "DE", "FL": "FL",
    "GA": "GA", "HI": "HI", "ID": "ID", "IL": "IL", "IN": "IN",
    "IA": "IA", "KS": "KS", "KY": "KY", "LA": "LA", "ME": "ME",
    "MD": "MD", "MA": "MA", "MI": "MI", "MN": "MN", "MS": "MS",
    "MO": "MO", "MT": "MT", "NE": "NE", "NV": "NV", "NH": "NH",
    "NJ": "NJ", "NM": "NM", "NY": "NY", "NC": "NC", "ND": "ND",
    "OH": "OH", "OK": "OK", "OR": "OR", "PA": "PA", "RI": "RI",
    "SC": "SC", "SD": "SD", "TN": "TN", "TX": "TX", "UT": "UT",
    "VT": "VT", "VA": "VA", "WA": "WA", "WV": "WV", "WI": "WI",
    "WY": "WY",
}

# ── City registry ────────────────────────────────────────────────────────────
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
    """Fetches real gas price data from NRCan (Canada) and AAA (US)."""

    def __init__(self):
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

    # ──────────────────────────────────────────────────────────────────
    # Crude oil
    # ──────────────────────────────────────────────────────────────────
    def get_crude_oil_price(self) -> float:
        """Current WTI crude oil price ($/barrel) from yfinance."""
        try:
            import yfinance as yf
            hist = yf.Ticker("CL=F").history(period="5d")
            if not hist.empty:
                return round(float(hist["Close"].iloc[-1]), 2)
        except Exception:
            pass
        return 110.0  # fallback (current range as of April 2026)

    def _get_wti_history(self) -> pd.DataFrame:
        """Return ~2-year daily WTI history as DataFrame with columns [date, wti]."""
        try:
            import yfinance as yf
            hist = yf.Ticker("CL=F").history(period="2y")
            if not hist.empty:
                df = hist[["Close"]].copy()
                df.index = pd.to_datetime(df.index).tz_localize(None)
                df = df.reset_index().rename(columns={"Date": "date", "Close": "wti"})
                df["date"] = pd.to_datetime(df["date"]).dt.normalize()
                return df
        except Exception:
            pass
        # Fallback: flat series at current estimate
        today = pd.Timestamp(datetime.now().date())
        dates = pd.date_range(today - timedelta(days=730), today, freq="D")
        return pd.DataFrame({"date": dates, "wti": 110.0})

    # ──────────────────────────────────────────────────────────────────
    # NRCan – Canadian real prices
    # ──────────────────────────────────────────────────────────────────
    def fetch_nrcan_history(self, location_name: str) -> pd.DataFrame:
        """Download NRCan AWP weekly XLS for *location_name* (e.g. 'Toronto').
        Returns DataFrame with columns: date, regular_cad_l, diesel_cad_l.
        Prices in CAD/L (converted from cents/L).
        """
        url = (
            "https://www2.nrcan.gc.ca/eneene/sources/pripri/prices_byfuel_e.cfm"
            f"?locationName={location_name}&downloadXLS=awp"
        )
        raw = _get(url, timeout=30)
        wb = openpyxl.load_workbook(io.BytesIO(raw), read_only=True, data_only=True)
        ws = wb.active

        rows = []
        header_found = False
        date_col = reg_col = diesel_col = None

        for row in ws.iter_rows(values_only=True):
            if not header_found:
                row_lower = [str(c).lower() if c is not None else "" for c in row]
                if any("date" in c for c in row_lower):
                    header_found = True
                    for idx, cell in enumerate(row_lower):
                        if "date" in cell:
                            date_col = idx
                        elif "regular" in cell:
                            reg_col = idx
                        elif "diesel" in cell:
                            diesel_col = idx
                continue

            if date_col is None or reg_col is None:
                continue
            date_val = row[date_col]
            reg_val  = row[reg_col]
            if date_val is None or reg_val is None:
                continue
            try:
                if isinstance(date_val, datetime):
                    dt = pd.Timestamp(date_val)
                else:
                    dt = pd.Timestamp(str(date_val))
                reg_price    = float(reg_val) / 100.0       # cents/L → CAD/L
                diesel_price = (
                    float(row[diesel_col]) / 100.0
                    if diesel_col is not None and row[diesel_col] is not None
                    else reg_price * 1.045
                )
                rows.append({"date": dt, "regular_cad_l": reg_price, "diesel_cad_l": diesel_price})
            except (ValueError, TypeError):
                continue

        wb.close()
        if not rows:
            raise ValueError(f"NRCan XLS parse returned 0 rows for {location_name}")
        return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    def fetch_nrcan_current(self, location_name: str) -> dict:
        """Scrape today's NRCan prices for *location_name*.
        Returns dict like {'Regular': 1.822, 'Mid': 2.029, 'Premium': 2.130, 'Diesel': 2.245}.
        All prices in CAD/L.
        """
        url = (
            "https://www2.nrcan.gc.ca/eneene/sources/pripri/prices_byfuel_e.cfm"
            f"?locationName={location_name}"
        )
        txt = _get(url, timeout=20).decode("utf-8", errors="replace")
        # Match table row ids like "Regular Gasoline" and the first <td> value after
        pat = (
            r'id="(Regular[^"]*|Mid[^"]*|Premium[^"]*|Diesel[^"]*)"'
            r'\s+scope="row".*?<td[^>]*>\s*([0-9]+\.?[0-9]*)\s*</td>'
        )
        matches = re.findall(pat, txt, re.DOTALL)
        result = {}
        for label, value in matches:
            price_cad_l = float(value) / 100.0  # cents/L → CAD/L
            if "Regular" in label:
                result["Regular"] = price_cad_l
            elif "Mid" in label:
                result["Mid"] = price_cad_l
            elif "Premium" in label:
                result["Premium"] = price_cad_l
            elif "Diesel" in label:
                result["Diesel"] = price_cad_l
        return result

    # ──────────────────────────────────────────────────────────────────
    # AAA – US real prices
    # ──────────────────────────────────────────────────────────────────
    def fetch_aaa_current(self, state_code: str) -> float:
        """Fetch current regular gas price (USD/gal) for *state_code* from AAA.
        Uses the blue 'state avg' price box, not the red national average.
        """
        url = f"https://gasprices.aaa.com/?state={state_code}"
        txt = _get(url, timeout=20).decode("utf-8", errors="replace")
        # State-specific price is in a <p class="price-text price-text--blue"> block
        # National average is in the red block — skip it
        m = re.search(r'price-text--blue[^$]{0,300}\$(\d+\.\d{3})', txt, re.DOTALL)
        if m:
            price = float(m.group(1))
            if 2.0 <= price <= 9.0:
                return price
        # Fallback: skip first plausible match (national avg), take second
        all_prices = re.findall(r'\$(\d\.\d{3})', txt)
        plausible = [float(p) for p in all_prices if 2.0 <= float(p) <= 9.0]
        if len(plausible) >= 2:
            return plausible[1]
        if plausible:
            return plausible[0]
        raise ValueError(f"No AAA price found for state {state_code}")

    # ──────────────────────────────────────────────────────────────────
    # Build real dataset
    # ──────────────────────────────────────────────────────────────────
    def _build_canadian_rows(self, wti_df: pd.DataFrame) -> list:
        """Build daily rows for all Canadian cities using NRCan weekly XLS data."""
        records = []
        today = pd.Timestamp(datetime.now().date())
        start = today - timedelta(days=730)

        # Cache NRCan history per location so GTA proxies reuse Toronto data
        nrcan_cache: dict = {}

        for city, info in CITIES.items():
            if info.get("country") != "CA":
                continue
            location = NRCAN_LOCATION.get(city)
            if location is None:
                continue

            # Download real weekly history (only once per NRCan location)
            if location not in nrcan_cache:
                try:
                    print(f"  NRCan: fetching {location}...")
                    nrcan_cache[location] = self.fetch_nrcan_history(location)
                except Exception as e:
                    print(f"  NRCan: failed {location}: {e}")
                    nrcan_cache[location] = None

            weekly = nrcan_cache[location]
            if weekly is None:
                continue

            offset_cad_l = GTA_OFFSETS.get(city, 0.0)

            # Interpolate weekly → daily
            all_dates = pd.date_range(
                max(weekly["date"].min(), start), today, freq="D"
            )
            interp = (
                weekly.set_index("date")["regular_cad_l"]
                .reindex(all_dates)
                .interpolate(method="time")
                .ffill()
                .bfill()
            )

            # Try to get today's price from current scraped value; patch last row
            try:
                current = self.fetch_nrcan_current(location)
                if "Regular" in current:
                    interp.iloc[-1] = current["Regular"]
            except Exception:
                pass

            for dt, cad_l in interp.items():
                cad_l_adj = round(float(cad_l) + offset_cad_l, 3)
                cad_l_adj = max(0.80, cad_l_adj)
                usd_gal   = cad_litre_to_usd_gal(cad_l_adj)
                # WTI on this day
                wti_row = wti_df[wti_df["date"] == dt]
                wti_val = float(wti_row["wti"].iloc[0]) if not wti_row.empty else 110.0
                records.append({
                    "date":      dt,
                    "city":      city,
                    "state":     info["state"],
                    "country":   "CA",
                    "price":     usd_gal,
                    "crude_oil": wti_val,
                    "lat":       info["lat"],
                    "lon":       info["lon"],
                })

        return records

    def _build_us_rows(self, wti_df: pd.DataFrame) -> list:
        """Build daily rows for all US cities using AAA current price + WTI backfill."""
        records = []
        today = pd.Timestamp(datetime.now().date())
        start = today - timedelta(days=730)
        dates = pd.date_range(start, today, freq="D")

        # Fetch current AAA price per state (cache to avoid duplicate requests)
        aaa_cache: dict = {}
        for city, info in CITIES.items():
            if info.get("country") != "US":
                continue
            state = info["state"]
            if state in aaa_cache:
                continue
            try:
                print(f"  AAA: fetching {state}...")
                aaa_cache[state] = self.fetch_aaa_current(state)
            except Exception as e:
                print(f"  AAA: failed {state}: {e}")
                aaa_cache[state] = info["base"]   # fallback to base estimate

        # Get WTI today's close for scaling
        wti_today = float(wti_df["wti"].iloc[-1]) if not wti_df.empty else 110.0

        for city, info in CITIES.items():
            if info.get("country") != "US":
                continue
            state = info["state"]
            current_price = aaa_cache.get(state, info["base"])

            # Scale WTI history to produce price series that ends at current_price
            # Simple linear scaling: price(t) ≈ current_price * wti(t) / wti_today
            rng = np.random.default_rng(seed=abs(hash(city)) % (2 ** 31))

            for dt in dates:
                wti_row = wti_df[wti_df["date"] == dt]
                wti_val = float(wti_row["wti"].iloc[0]) if not wti_row.empty else wti_today
                # Scale relative to today, add tiny city-specific noise (±0.5 cents)
                scaled = current_price * (wti_val / wti_today)
                noise  = rng.normal(0, 0.005)
                price  = round(max(1.50, scaled + noise), 3)
                records.append({
                    "date":      dt,
                    "city":      city,
                    "state":     state,
                    "country":   "US",
                    "price":     price,
                    "crude_oil": round(wti_val, 2),
                    "lat":       info["lat"],
                    "lon":       info["lon"],
                })

        return records

    def build_real_data(self) -> pd.DataFrame:
        """Fetch real data from NRCan + AAA and return a combined DataFrame."""
        print("Fetching WTI history...")
        wti_df = self._get_wti_history()
        # Ensure one row per date
        wti_df = wti_df.groupby("date")["wti"].last().reset_index()

        print("Building Canadian city rows (NRCan)...")
        ca_rows = self._build_canadian_rows(wti_df)

        print("Building US city rows (AAA + WTI)...")
        us_rows = self._build_us_rows(wti_df)

        all_rows = ca_rows + us_rows
        if not all_rows:
            raise ValueError("build_real_data: no rows assembled")

        df = pd.DataFrame(all_rows)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["city", "date"]).reset_index(drop=True)
        print(f"  Done: {len(df):,} rows for {df['city'].nunique()} cities")
        return df

    # ──────────────────────────────────────────────────────────────────
    # Fallback sample data (used only if real fetch fails completely)
    # ──────────────────────────────────────────────────────────────────
    def _generate_wti_series(self, dates: pd.DatetimeIndex) -> np.ndarray:
        n = len(dates)
        prices = np.zeros(n)
        prices[0] = 95.0
        rng = np.random.default_rng(seed=42)
        for i in range(1, n):
            prices[i] = np.clip(prices[i - 1] + rng.normal(0, 0.8), 70, 130)
        return np.round(prices, 2)

    def generate_sample_data(self) -> pd.DataFrame:
        """Fallback: generates plausible fake data if real sources unavailable."""
        start = pd.Timestamp("2024-01-01")
        end   = pd.Timestamp(datetime.now().date())
        dates = pd.date_range(start, end, freq="D")
        wti   = self._generate_wti_series(dates)

        records = []
        for city, info in CITIES.items():
            base = info["base"]
            rng  = np.random.default_rng(seed=abs(hash(city)) % (2 ** 31))
            for i, (d, w) in enumerate(zip(dates, wti)):
                doy = d.day_of_year / 365.0
                price = max(1.50, base + (w - 95) * 0.018 + 0.12 * np.sin(2 * np.pi * doy) + rng.normal(0, 0.01))
                records.append({
                    "date": d, "city": city, "state": info["state"],
                    "country": info.get("country", "US"),
                    "price": round(price, 3), "crude_oil": w,
                    "lat": info["lat"], "lon": info["lon"],
                })
        return pd.DataFrame(records)

    # ──────────────────────────────────────────────────────────────────
    # Load / persist
    # ──────────────────────────────────────────────────────────────────
    def load_or_generate_data(self) -> pd.DataFrame:
        """Return cached CSV if <12 hours old; otherwise fetch real data."""
        cache_ttl_hours = 12
        if os.path.exists(DATA_PATH):
            age_hours = (time.time() - os.path.getmtime(DATA_PATH)) / 3600
            if age_hours < cache_ttl_hours:
                df = pd.read_csv(DATA_PATH, parse_dates=["date"])
                if len(df) > 0 and "city" in df.columns:
                    return df

        # Try real data first
        try:
            df = self.build_real_data()
            df.to_csv(DATA_PATH, index=False)
            return df
        except Exception as e:
            print(f"Real data fetch failed ({e}), falling back to sample data")

        # Last resort: sample data
        df = self.generate_sample_data()
        df.to_csv(DATA_PATH, index=False)
        return df

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────
    def get_city_list(self) -> list:
        return sorted(CITIES.keys())

    def get_city_coords(self) -> dict:
        return {city: (info["lat"], info["lon"]) for city, info in CITIES.items()}
