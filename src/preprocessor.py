"""
preprocessor.py – GasWatch
───────────────────────────
Feature engineering: lag prices, rolling averages, cyclical date encoding,
city label encoding.
"""

import numpy as np
import pandas as pd

# Ordered city list used for deterministic label encoding.
# Must stay in sync with CITIES in data_collector.py.
CITY_LIST = sorted([
    # USA
    "Albuquerque, NM", "Anchorage, AK", "Atlanta, GA", "Austin, TX",
    "Baltimore, MD", "Boise, ID", "Boston, MA", "Charlotte, NC",
    "Chicago, IL", "Columbus, OH", "Dallas, TX", "Denver, CO",
    "Detroit, MI", "Honolulu, HI", "Houston, TX", "Indianapolis, IN",
    "Kansas City, MO", "Las Vegas, NV", "Los Angeles, CA", "Miami, FL",
    "Minneapolis, MN", "Nashville, TN", "New Orleans, LA", "New York, NY",
    "Oklahoma City, OK", "Orlando, FL", "Philadelphia, PA", "Phoenix, AZ",
    "Pittsburgh, PA", "Portland, OR", "Raleigh, NC", "Sacramento, CA",
    "Salt Lake City, UT", "San Antonio, TX", "San Diego, CA", "San Francisco, CA",
    "Seattle, WA", "St. Louis, MO", "Tucson, AZ", "Virginia Beach, VA",
    "Washington, DC",
    # Canada
    "Brampton, ON", "Calgary, AB", "Edmonton, AB", "Hamilton, ON",
    "Kitchener, ON", "London, ON", "Mississauga, ON", "Montreal, QC",
    "Oakville, ON", "Ottawa, ON", "Toronto, ON", "Vancouver, BC",
    "Winnipeg, MB",
])


def encode_city(city: str) -> int:
    try:
        return CITY_LIST.index(city)
    except ValueError:
        return 0


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all ML features in-place (returns copy)."""
    df = df.copy().sort_values(["city", "date"])

    grp = df.groupby("city")["price"]

    # --- Lag prices ---
    for lag in (1, 2, 7, 14, 30):
        df[f"lag_{lag}"] = grp.shift(lag)

    # --- Rolling averages (shift 1 so no leakage) ---
    df["rolling_7"]  = grp.transform(lambda x: x.shift(1).rolling(7,  min_periods=1).mean())
    df["rolling_30"] = grp.transform(lambda x: x.shift(1).rolling(30, min_periods=1).mean())

    # --- Date features ---
    df["day_of_week"]  = df["date"].dt.dayofweek
    df["month"]        = df["date"].dt.month
    df["day_of_year"]  = df["date"].dt.dayofyear

    # --- Cyclical encoding ---
    df["month_sin"] = np.sin(2 * np.pi * df["month"]       / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"]       / 12)
    df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # --- City encoding ---
    df["city_code"] = df["city"].apply(encode_city)

    # --- Momentum ---
    df["price_diff_1"] = grp.diff(1)
    df["price_diff_7"] = grp.diff(7)

    return df


# Ordered list used for X matrix construction (must stay stable)
FEATURE_COLS = [
    "lag_1", "lag_2", "lag_7", "lag_14", "lag_30",
    "rolling_7", "rolling_30",
    "crude_oil",
    "day_of_week", "month", "day_of_year",
    "month_sin", "month_cos", "dow_sin", "dow_cos",
    "city_code",
    "price_diff_1", "price_diff_7",
]
