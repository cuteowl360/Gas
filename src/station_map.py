"""
station_map.py – GasWatch
Generates an embedded Leaflet.js HTML map with pre-seeded gas station data.
All station locations are generated deterministically from city coordinates
– no external Overpass / OpenStreetMap API calls needed at runtime.

Prices are anchored to live NRCan city averages with realistic brand offsets.
"""

import json
import math
import hashlib
import numpy as np

# Canadian gas station brands and their typical price offsets (CAD/L)
_BRANDS_CA = [
    "Petro-Canada", "Shell", "Esso", "Husky", "Ultramar",
    "Pioneer", "Canadian Tire", "Mobil", "Co-op", "Fas Gas",
    "Costco Gas", "Parkland", "Irving", "BonTerre", "Sunoco",
    "7-Eleven", "Circle K", "Race Trac", "Ultramar", "Esso",
]
_OFFSETS_CA = {
    "Petro-Canada": 0.006, "Shell": 0.011, "Esso": 0.008,
    "Husky": -0.005, "Ultramar": 0.003, "Pioneer": -0.014,
    "Canadian Tire": -0.010, "Mobil": 0.012, "Co-op": -0.008,
    "Fas Gas": -0.012, "Costco Gas": -0.022, "Parkland": 0.001,
    "Irving": 0.000, "BonTerre": -0.004, "Sunoco": 0.007,
    "7-Eleven": -0.003, "Circle K": -0.006, "Race Trac": -0.007,
}

# US brands
_BRANDS_US = [
    "Shell", "Chevron", "ExxonMobil", "BP", "Citgo", "Marathon",
    "76", "Sunoco", "Mobil", "Valero", "Pilot", "Love's",
    "Circle K", "Wawa", "QuikTrip", "Casey's", "RaceTrac", "Costco",
]
_OFFSETS_US = {
    "Shell": 0.030, "Chevron": 0.025, "ExxonMobil": 0.020, "BP": 0.018,
    "Citgo": 0.010, "Marathon": 0.008, "76": 0.005, "Sunoco": 0.012,
    "Mobil": 0.015, "Valero": -0.005, "Pilot": -0.020, "Love's": -0.018,
    "Circle K": -0.025, "Wawa": -0.015, "QuikTrip": -0.022, "Casey's": -0.018,
    "RaceTrac": -0.020, "Costco": -0.070,
}


def _generate_stations(city_prices: dict, country: str) -> list:
    """
    Generate realistic gas station locations for all cities.
    Uses deterministic seeding – same city always produces same station layout.
    Returns a flat list dicts with keys: id, lat, lon, name, brand, base_price, forecast, city
    """
    brands   = _BRANDS_CA if country == "CA" else _BRANDS_US
    offsets  = _OFFSETS_CA if country == "CA" else _OFFSETS_US
    stations = []
    sid      = 0

    for city_name, city_data in city_prices.items():
        if city_data.get("country", country) != country:
            continue
        base_price = city_data["price"]
        city_lat   = city_data["lat"]
        city_lon   = city_data["lon"]
        city_fc    = city_data.get("forecast", [base_price] * 7)

        seed = int(hashlib.md5(city_name.encode()).hexdigest()[:8], 16) % (2 ** 31)
        rng  = np.random.RandomState(seed)

        n = 22  # stations per city
        for i in range(n):
            angle   = rng.uniform(0, 2 * math.pi)
            dist_km = float(np.clip(rng.exponential(2.2), 0.25, 9.0))
            dlat    = (dist_km / 111.0) * math.cos(angle)
            cos_lat = math.cos(math.radians(city_lat)) or 1e-9
            dlon    = (dist_km / (111.0 * cos_lat)) * math.sin(angle)

            brand   = brands[i % len(brands)]
            off     = offsets.get(brand, 0.0)
            noise   = rng.uniform(-0.007, 0.007)
            price   = round(max(0.70, base_price + off + noise), 3)

            fc = []
            for city_fp in city_fc[:7]:
                fc_noise = rng.uniform(-0.003, 0.003)
                fc.append(round(max(0.70, city_fp + off + fc_noise), 3))

            stations.append({
                "id":    sid,
                "lat":   round(city_lat + dlat, 5),
                "lon":   round(city_lon + dlon, 5),
                "name":  brand,
                "brand": brand,
                "base_price": price,   # Regular price (display unit)
                "forecast": fc,
                "city":  city_name,
            })
            sid += 1

    return stations


def build_station_map_html(
    center_lat: float,
    center_lon: float,
    city_prices: dict,
    country: str = "CA",
    zoom: int = 13,
    height_px: int = 620,
) -> str:
    """
    Returns self-contained HTML with a Leaflet map.

    Features:
    - Pre-embedded station data (no Overpass / external API calls)
    - Colour-coded pins by price
    - Popup with station name, current price, 7-day forecast bar chart
    - My Location button (browser geolocation)
    - Radius slider (filters visible stations by distance)
    - Gas type multiplier selector
    - Auto-filters as you pan the map
    """
    all_stations = _generate_stations(city_prices, country)
    stations_json = json.dumps(all_stations)

    unit_label = "CAD/L" if country == "CA" else "USD/gal"

    if country == "CA":
        cheap_thresh = 1.65
        mid_thresh   = 1.85
    else:
        cheap_thresh = 3.50
        mid_thresh   = 4.20

    n_stations = len(all_stations)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  html, body {{ height:100%; background:#0d1e33; }}
  #map {{ height:{height_px}px; width:100%; }}
  #controls {{
    display:flex; align-items:center; gap:10px; padding:8px 12px;
    background:rgba(13,30,51,.96); border-bottom:1px solid #1e3a55; flex-wrap:wrap;
  }}
  #controls label {{ color:#8ab4cc; font-size:.82em; font-family:sans-serif; }}
  #radius-slider {{ width:120px; accent-color:#74b9ff; }}
  #locate-btn {{
    background:#1a3a5c; color:#74b9ff; border:1px solid #2d5a7c;
    border-radius:6px; padding:5px 12px; cursor:pointer; font-size:.82em;
    font-family:sans-serif; white-space:nowrap;
  }}
  #locate-btn:hover {{ background:#2d5a7c; }}
  #status-msg {{ color:#6a8fa8; font-size:.78em; font-family:sans-serif; }}
  #gas-type-sel {{
    background:#1a3a5c; color:#74b9ff; border:1px solid #2d5a7c;
    border-radius:6px; padding:4px 8px; font-size:.82em; cursor:pointer;
  }}
  .legend {{
    background:rgba(13,30,51,.9); color:#dde8f0; border:1px solid #1e3a55;
    border-radius:8px; padding:8px 12px; font-size:.78em;
    font-family:sans-serif; line-height:1.7;
  }}
  .legend-dot {{
    display:inline-block; width:12px; height:12px;
    border-radius:50%; margin-right:6px; vertical-align:middle;
  }}
  /* ── Search bar ── */
  #search-bar {{
    display:flex; align-items:center; gap:8px; padding:7px 12px;
    background:rgba(13,30,51,.98); border-bottom:1px solid #1e3a55;
  }}
  #search-input {{
    flex:1; background:#112233; color:#dde8f0;
    border:1px solid #2d5a7c; border-radius:8px;
    padding:7px 12px; font-size:.88em; font-family:sans-serif;
    outline:none;
  }}
  #search-input::placeholder {{ color:#4a6a7c; }}
  #search-input:focus {{ border-color:#74b9ff; background:#0d2a44; }}
  #search-btn {{
    background:#1e4a6c; color:#74b9ff; border:1px solid #2d5a7c;
    border-radius:8px; padding:7px 14px; cursor:pointer;
    font-size:.88em; font-family:sans-serif; white-space:nowrap;
  }}
  #search-btn:hover {{ background:#2d5a7c; }}
  #search-results {{
    position:absolute; top:80px; left:12px; right:12px; z-index:9999;
    background:#0d2a44; border:1px solid #2d5a7c; border-radius:8px;
    overflow:hidden; box-shadow:0 8px 24px rgba(0,0,0,.6);
    max-height:220px; overflow-y:auto;
  }}
  .search-result-item {{
    padding:9px 14px; color:#dde8f0; font-size:.84em;
    font-family:sans-serif; cursor:pointer; border-bottom:1px solid #1e3a55;
  }}
  .search-result-item:last-child {{ border-bottom:none; }}
  .search-result-item:hover {{ background:#1e4a6c; color:#74b9ff; }}
  .search-result-name {{ font-weight:600; }}
  .search-result-detail {{ color:#6a8fa8; font-size:.85em; margin-top:1px; }}
  .leaflet-popup-content-wrapper {{
    background:#0d2a44; border:1px solid #2d5a7c; border-radius:12px;
    color:#dde8f0; box-shadow:0 8px 24px rgba(0,0,0,.5);
  }}
  .leaflet-popup-tip {{ background:#0d2a44; }}
  .popup-name {{ font-size:1.0em; font-weight:700; color:#74b9ff; margin-bottom:4px; }}
  .popup-price {{ font-size:1.6em; font-weight:800; margin:4px 0; }}
  .popup-brand {{ font-size:.78em; color:#6a8fa8; margin-bottom:8px; }}
  .popup-city {{ font-size:.72em; color:#4a6a7c; margin-bottom:4px; }}
  .popup-fc-title {{ font-size:.76em; color:#8ab4cc; margin-top:8px; margin-bottom:3px; }}
  .fc-bar-wrap {{ display:flex; gap:3px; align-items:flex-end; height:36px; }}
  .fc-bar {{
    flex:1; background:#1e4a6c; border-radius:3px 3px 0 0;
    min-height:4px; position:relative; cursor:default;
  }}
  .fc-bar:hover::after {{
    content:attr(data-tip);
    position:absolute; bottom:100%; left:50%; transform:translateX(-50%);
    background:#0d1e33; color:#74b9ff; font-size:.72em; white-space:nowrap;
    padding:2px 5px; border-radius:4px; pointer-events:none;
  }}
  .popup-distance {{ font-size:.74em; color:#6a8fa8; margin-top:6px; }}
  .cheap {{ color:#51cf66; }} .mid {{ color:#ffd43b; }} .exp {{ color:#ff6b6b; }}
</style>
</head>
<body>
<div id="search-bar">
  <input id="search-input" type="text" placeholder="Search an address, road, or place..." autocomplete="off"/>
  <button id="search-btn">Search</button>
</div>
<div id="search-results" style="display:none"></div>
<div id="controls">
  <button id="locate-btn">📍 My Location</button>
  <label>Radius:&nbsp;<input type="range" id="radius-slider" min="500" max="10000" value="3000" step="500"/>
  <span id="radius-val">3.0 km</span></label>
  <select id="gas-type-sel">
    <option value="1.000">Regular</option>
    <option value="1.110">Midgrade</option>
    <option value="1.200">Premium</option>
    <option value="1.045">Diesel</option>
  </select>
  <span id="status-msg">⛽ {n_stations} stations loaded. Click a pin for details.</span>
</div>
<div id="map"></div>

<script>
// ── Pre-embedded station data (generated server-side, no API calls needed) ──
const ALL_STATIONS  = {stations_json};
const UNIT          = "{unit_label}";
const CHEAP_THRESH  = {cheap_thresh};
const MID_THRESH    = {mid_thresh};
const INIT_LAT      = {center_lat};
const INIT_LON      = {center_lon};

// ── Helpers ───────────────────────────────────────────────────────────────
function distanceKm(lat1, lon1, lat2, lon2) {{
  const R = 6371;
  const dLat = (lat2-lat1)*Math.PI/180;
  const dLon = (lon2-lon1)*Math.PI/180;
  const a = Math.sin(dLat/2)**2
          + Math.cos(lat1*Math.PI/180)*Math.cos(lat2*Math.PI/180)*Math.sin(dLon/2)**2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
}}

function priceColor(p) {{
  if (p <= CHEAP_THRESH) return '#51cf66';
  if (p <= MID_THRESH)   return '#ffd43b';
  return '#ff6b6b';
}}
function priceClass(p) {{
  if (p <= CHEAP_THRESH) return 'cheap';
  if (p <= MID_THRESH)   return 'mid';
  return 'exp';
}}

function makeIcon(price) {{
  const color = priceColor(price);
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="60" height="32">
    <rect rx="6" ry="6" width="60" height="26" fill="${{color}}" fill-opacity=".93"/>
    <text x="30" y="17" font-family="sans-serif" font-size="11" font-weight="700"
          text-anchor="middle" fill="#0d1e33">${{price.toFixed(3)}}</text>
    <polygon points="22,26 38,26 30,32" fill="${{color}}" fill-opacity=".93"/>
  </svg>`;
  return L.divIcon({{ html:svg, className:'', iconAnchor:[30,32], popupAnchor:[0,-34] }});
}}

function forecastBars(fc, basePrice) {{
  const vals = [basePrice, ...fc];
  const mn = Math.min(...vals)*0.995, mx = Math.max(...vals)*1.005;
  const rng = mx-mn || 0.01;
  const days = ['Tmrw','D2','D3','D4','D5','D6','D7'];
  return fc.map((p,i) => {{
    const h = Math.round(((p-mn)/rng)*32+4);
    return `<div class="fc-bar ${{priceClass(p)}}" style="height:${{h}}px"
                 data-tip="${{days[i]}}: ${{p.toFixed(3)}} ${{UNIT}}"></div>`;
  }}).join('');
}}

// ── Map init ──────────────────────────────────────────────────────────────
const map = L.map('map').setView([INIT_LAT, INIT_LON], {zoom});
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
  attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
  maxZoom: 19
}}).addTo(map);

// Legend
const legend = L.control({{position:'bottomright'}});
legend.onAdd = () => {{
  const d = L.DomUtil.create('div','legend');
  d.innerHTML = `<b>⛽ Price ({unit_label})</b><br>
    <span class="legend-dot" style="background:#51cf66"></span>≤ {cheap_thresh}<br>
    <span class="legend-dot" style="background:#ffd43b"></span>≤ {mid_thresh}<br>
    <span class="legend-dot" style="background:#ff6b6b"></span>&gt; {mid_thresh}`;
  return d;
}};
legend.addTo(map);

// ── Station layer ─────────────────────────────────────────────────────────
let stationLayer = L.layerGroup().addTo(map);
let userMarker   = null;
let userLatLon   = null;
let currentGtMult = 1.0;

function renderStations(centerLat, centerLon, radiusM) {{
  stationLayer.clearLayers();
  const status = document.getElementById('status-msg');
  let count = 0, cheapCount = 0;

  ALL_STATIONS.forEach(s => {{
    const dist = distanceKm(centerLat, centerLon, s.lat, s.lon);
    if (dist * 1000 > radiusM) return;

    const price = +(s.base_price * currentGtMult).toFixed(3);
    const fc    = s.forecast.map(p => +(p * currentGtMult).toFixed(3));
    const distTxt = userLatLon
      ? `<div class="popup-distance">📏 ${{distanceKm(userLatLon[0],userLatLon[1],s.lat,s.lon).toFixed(1)}} km away</div>`
      : `<div class="popup-distance">📍 ${{dist.toFixed(1)}} km from centre</div>`;

    const popup = `
      <div class="popup-name">⛽ ${{s.name}}</div>
      <div class="popup-brand">${{s.brand}}</div>
      <div class="popup-city">📍 ${{s.city}}</div>
      <div class="popup-price ${{priceClass(price)}}">${{price.toFixed(3)}} <span style="font-size:.5em;color:#8ab4cc">${{UNIT}}</span></div>
      <div class="popup-fc-title">📈 7-Day Forecast</div>
      <div class="fc-bar-wrap">${{forecastBars(fc, price)}}</div>
      <div style="display:flex;justify-content:space-between;font-size:.72em;color:#6a8fa8;margin-top:2px">
        <span>Tmrw: ${{fc[0].toFixed(3)}}</span><span>+7d: ${{fc[6].toFixed(3)}}</span>
      </div>
      ${{distTxt}}`;

    L.marker([s.lat, s.lon], {{icon: makeIcon(price)}})
      .bindPopup(popup, {{maxWidth:240, minWidth:200}})
      .addTo(stationLayer);

    count++;
    if (price <= CHEAP_THRESH) cheapCount++;
  }});

  status.textContent = count > 0
    ? `✅ ${{count}} stations · ${{cheapCount}} cheap · Zoom/pan to explore`
    : `ℹ️ No stations in this radius — try increasing the radius slider`;
}}

// ── Radius slider ─────────────────────────────────────────────────────────
const slider = document.getElementById('radius-slider');
const radLabel = document.getElementById('radius-val');
function updateRadius() {{
  radLabel.textContent = (slider.value/1000).toFixed(1) + ' km';
  const c = map.getCenter();
  renderStations(c.lat, c.lng, parseInt(slider.value));
}}
slider.addEventListener('input', updateRadius);

// ── Gas type selector ─────────────────────────────────────────────────────
document.getElementById('gas-type-sel').addEventListener('change', e => {{
  currentGtMult = parseFloat(e.target.value);
  const c = map.getCenter();
  renderStations(c.lat, c.lng, parseInt(slider.value));
}});

// ── Re-render on map pan/zoom ─────────────────────────────────────────────
map.on('moveend', () => {{
  const c = map.getCenter();
  renderStations(c.lat, c.lng, parseInt(slider.value));
}});

// ── My Location button ────────────────────────────────────────────────────
document.getElementById('locate-btn').addEventListener('click', () => {{
  const status = document.getElementById('status-msg');
  if (!navigator.geolocation) {{
    status.textContent = '⚠️ Geolocation not supported by your browser.';
    return;
  }}
  status.textContent = '📍 Getting your location…';
  navigator.geolocation.getCurrentPosition(pos => {{
    userLatLon = [pos.coords.latitude, pos.coords.longitude];
    map.setView(userLatLon, 14);
    if (userMarker) map.removeLayer(userMarker);
    userMarker = L.circleMarker(userLatLon, {{
      radius:10, color:'#74b9ff', fillColor:'#74b9ff', fillOpacity:0.7, weight:3
    }}).bindPopup('<b>📍 You are here</b>').addTo(map);
    renderStations(userLatLon[0], userLatLon[1], parseInt(slider.value));
  }}, () => {{
    status.textContent = '⚠️ Location access denied. Pan the map manually.';
  }});
}});

// ── Address search (Nominatim / OpenStreetMap) ────────────────────────────
let searchMarker = null;
let searchTimeout  = null;

function clearResults() {{
  const r = document.getElementById('search-results');
  r.style.display = 'none';
  r.innerHTML = '';
}}

function handleSelect(lat, lon, displayName) {{
  clearResults();
  document.getElementById('search-input').value = displayName;
  status.textContent = 'Stations near: ' + displayName.split(',')[0];
  const pos = [parseFloat(lat), parseFloat(lon)];
  map.setView(pos, 14);
  if (searchMarker) map.removeLayer(searchMarker);
  searchMarker = L.marker(pos, {{
    icon: L.divIcon({{
      html: '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="32" viewBox="0 0 24 32">'
          + '<ellipse cx="12" cy="30" rx="5" ry="2" fill="rgba(0,0,0,.25)"/>'
          + '<path d="M12 0C7.6 0 4 3.6 4 8c0 6 8 22 8 22s8-16 8-22c0-4.4-3.6-8-8-8z" fill="#e74c3c"/>'
          + '<circle cx="12" cy="8" r="3" fill="#fff"/>'
          + '</svg>',
      className: '', iconSize: [24,32], iconAnchor: [12,32]
    }})
  }}).bindPopup('<b>📍 ' + displayName.split(',')[0] + '</b>').addTo(map).openPopup();
  renderStations(pos[0], pos[1], parseInt(slider.value));
}}

function searchAddress() {{
  const q = document.getElementById('search-input').value.trim();
  if (!q) return;
  status.textContent = 'Searching…';
  const url = 'https://nominatim.openstreetmap.org/search?q='
            + encodeURIComponent(q)
            + '&format=json&limit=5&countrycodes=ca';
  fetch(url, {{headers: {{'Accept-Language': 'en', 'User-Agent': 'GasWatch/1.0 (gaswatch.app)'}} }})
    .then(r => r.json())
    .then(data => {{
      if (!data.length) {{
        status.textContent = 'No results found for: ' + q;
        clearResults();
        return;
      }}
      if (data.length === 1) {{ handleSelect(data[0].lat, data[0].lon, data[0].display_name); return; }}
      const box = document.getElementById('search-results');
      box.innerHTML = '';
      data.forEach(item => {{
        const div = document.createElement('div');
        div.className = 'search-result-item';
        const parts = item.display_name.split(', ');
        div.innerHTML = '<div class="search-result-name">' + parts[0] + '</div>'
                      + '<div class="search-result-detail">' + parts.slice(1,3).join(', ') + '</div>';
        div.addEventListener('click', () => handleSelect(item.lat, item.lon, item.display_name));
        box.appendChild(div);
      }});
      box.style.display = 'block';
      status.textContent = data.length + ' results found';
    }})
    .catch(() => {{ status.textContent = 'Search failed — check your internet connection.'; }});
}}

document.getElementById('search-btn').addEventListener('click', () => {{ clearResults(); searchAddress(); }});
document.getElementById('search-input').addEventListener('keydown', e => {{
  if (e.key === 'Enter') {{ clearResults(); searchAddress(); }}
  if (e.key === 'Escape') clearResults();
}});
document.getElementById('search-input').addEventListener('input', () => {{
  clearTimeout(searchTimeout);
  searchTimeout = setTimeout(searchAddress, 600);
}});
document.addEventListener('click', e => {{
  if (!e.target.closest('#search-bar') && !e.target.closest('#search-results')) clearResults();
}});

// ── Initial render ────────────────────────────────────────────────────────
renderStations(INIT_LAT, INIT_LON, parseInt(slider.value));
</script>
</body>
</html>"""

    return html
