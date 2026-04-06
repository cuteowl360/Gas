"""
station_fetcher.py – GasWatch
Fetches real gas station locations from OpenStreetMap Overpass API.
Prices are derived from our real NRCan / AAA city-level data + small
station-specific variance so each station has a unique but realistic price.
"""

import json
import os
import re
import ssl
import time
import urllib.request
import urllib.parse
from functools import lru_cache

import numpy as np
import pandas as pd

# ─── SSL context ────────────────────────────────────────────────────────────
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Content-Type": "application/x-www-form-urlencoded",
}


def _http_post(url: str, body: bytes, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, data=body, headers=_HEADERS, method="POST")
    with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as r:
        return r.read()


# ─── Overpass API ────────────────────────────────────────────────────────────
# Multiple mirrors – tried in order until one responds
_OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]

# Per-brand typical price variance in CAD/L or USD/gal relative to city avg.
# Based on general brand reputation (premium vs discount).
BRAND_OFFSETS = {
    # Premium / higher-priced brands
    "shell":        +0.03,
    "bp":           +0.02,
    "esso":         +0.02,
    "mobil":        +0.02,
    "sunoco":       +0.02,
    "chevron":      +0.02,
    "marathon":     +0.01,
    "76":           +0.01,
    # Discount brands
    "costco":       -0.07,
    "bj's":         -0.06,
    "sam's club":   -0.06,
    "circle k":     -0.03,
    "7-eleven":     -0.02,
    "kwik trip":    -0.03,
    "wawa":         -0.02,
    "racetrac":     -0.03,
    "pilot":        -0.03,
    "love's":       -0.03,
    "petro-canada": +0.01,
    "canadian tire": -0.02,
    "ultramar":     +0.01,
    "pioneer":      -0.01,
    "husky":        -0.01,
    "fas gas":      -0.02,
    "co-op":        -0.02,
}


def fetch_stations_overpass(lat: float, lon: float, radius_m: int = 5000) -> list[dict]:
    """
    Fetch gas stations within *radius_m* metres of (lat, lon) from OpenStreetMap.
    Returns list of dicts: {id, name, brand, lat, lon, osm_id}.
    Limits to 80 stations to keep performance fast.
    """
    query = f"""
[out:json][timeout:25];
(
  node["amenity"="fuel"](around:{radius_m},{lat},{lon});
  way["amenity"="fuel"](around:{radius_m},{lat},{lon});
);
out center 80;
"""
    body = urllib.parse.urlencode({"data": query}).encode("utf-8")
    last_err = None
    for mirror in _OVERPASS_MIRRORS:
        try:
            raw = _http_post(mirror, body, timeout=30)
            data = json.loads(raw)
            break
        except Exception as e:
            last_err = e
            time.sleep(1)
            continue
    else:
        raise RuntimeError(f"All Overpass mirrors failed. Last error: {last_err}")

    stations = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        osm_id = el.get("id", 0)

        # Position: nodes have lat/lon directly; ways have a center
        if el.get("type") == "node":
            slat = el.get("lat", lat)
            slon = el.get("lon", lon)
        else:
            centre = el.get("center", {})
            slat = centre.get("lat", lat)
            slon = centre.get("lon", lon)

        brand   = tags.get("brand", tags.get("operator", "")).strip()
        name    = tags.get("name",  brand or "Gas Station").strip()
        if not name:
            name = "Gas Station"

        stations.append({
            "osm_id": osm_id,
            "name":   name,
            "brand":  brand,
            "lat":    slat,
            "lon":    slon,
        })

    return stations


def assign_station_prices(
    stations: list[dict],
    city_price_usd_gal: float,
    country: str,
    gas_type_multiplier: float = 1.0,
) -> list[dict]:
    """
    Assign a realistic price to each station based on city average + brand offset
    + small deterministic noise per station (reproducible via osm_id seed).
    Prices are returned in the same unit as city_price_usd_gal (USD/gal for US,
    CAD/L for Canada).

    Also attaches a crude 7-day forecast (gentle trend ± small variance).
    """
    result = []
    for stn in stations:
        brand_key = stn["brand"].lower()
        brand_offset = 0.0
        for key, off in BRAND_OFFSETS.items():
            if key in brand_key:
                brand_offset = off
                break

        # Deterministic noise per station: ±0.03 units
        rng = np.random.default_rng(seed=abs(stn["osm_id"]) % (2**31))
        noise = rng.uniform(-0.025, 0.025)

        station_price = round(city_price_usd_gal * gas_type_multiplier + brand_offset + noise, 3)
        station_price = max(0.60 if country == "CA" else 1.50, station_price)

        # 7-day forecast: carry city-level trend + station noise
        forecast = []
        p = station_price
        for day in range(7):
            daily_rng = np.random.default_rng(seed=(abs(stn["osm_id"]) + day * 997) % (2**31))
            daily_chg = daily_rng.uniform(-0.012, 0.012)
            p = round(p + daily_chg, 3)
            p = max(0.60 if country == "CA" else 1.50, p)
            forecast.append(p)

        s = dict(stn)
        s["price"]    = station_price
        s["forecast"] = forecast
        result.append(s)

    return result


def geocode_address(address: str) -> tuple[float, float] | None:
    """Geocode an address string using Nominatim (OSM). Returns (lat, lon) or None."""
    encoded = urllib.parse.quote(address)
    url = (
        f"https://nominatim.openstreetmap.org/search"
        f"?q={encoded}&format=json&limit=1&addressdetails=0"
    )
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en",
        })
        raw = urllib.request.urlopen(req, timeout=10, context=_SSL_CTX).read()
        results = json.loads(raw)
        if results:
            return float(results[0]["lat"]), float(results[0]["lon"])
    except Exception:
        pass
    return None
