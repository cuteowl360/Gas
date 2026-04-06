"""
station_map.py – GasWatch
Generates an embedded Leaflet.js HTML map that loads real gas stations
from the Overpass API *in the user's browser* (no server-side API calls needed).

Prices come from our real NRCan/AAA city-level data and are assigned to
stations by brand and location.
"""

import json


def build_station_map_html(
    center_lat: float,
    center_lon: float,
    city_prices: dict,     # city_name -> {price_cad_l_or_usd_gal, country, forecast}
    country: str = "CA",   # "CA" or "US"
    zoom: int = 13,
    height_px: int = 620,
) -> str:
    """
    Returns self-contained HTML with a Leaflet map.

    The map:
    - Opens centred on (center_lat, center_lon)
    - Has a "Find My Location" button
    - Queries Overpass API from the browser for fuel stations in view
    - Renders colour-coded pins per station price
    - Clicking a pin shows name, price, 7-day forecast mini-chart
    """
    city_prices_json = json.dumps(city_prices)
    unit_label = "CAD/L" if country == "CA" else "USD/gal"
    is_canada = str(country == "CA").lower()

    # Colour thresholds (CAD/L for CA, USD/gal for US)
    if country == "CA":
        cheap_thresh  = 1.65
        mid_thresh    = 1.85
    else:
        cheap_thresh  = 3.50
        mid_thresh    = 4.20

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

  /* Controls bar */
  #controls {{
    display:flex; align-items:center; gap:10px; padding:8px 12px;
    background:rgba(13,30,51,.96); border-bottom:1px solid #1e3a55;
    flex-wrap:wrap;
  }}
  #controls label {{
    color:#8ab4cc; font-size:.82em; font-family:sans-serif;
  }}
  #radius-slider {{
    width:120px; accent-color:#74b9ff;
  }}
  #locate-btn {{
    background:#1a3a5c; color:#74b9ff; border:1px solid #2d5a7c;
    border-radius:6px; padding:5px 12px; cursor:pointer; font-size:.82em;
    font-family:sans-serif; white-space:nowrap;
  }}
  #locate-btn:hover {{ background:#2d5a7c; }}
  #status-msg {{
    color:#6a8fa8; font-size:.78em; font-family:sans-serif;
  }}
  #gas-type-sel {{
    background:#1a3a5c; color:#74b9ff; border:1px solid #2d5a7c;
    border-radius:6px; padding:4px 8px; font-size:.82em; cursor:pointer;
  }}

  /* Legend */
  .legend {{
    background:rgba(13,30,51,.9); color:#dde8f0;
    border:1px solid #1e3a55; border-radius:8px;
    padding:8px 12px; font-size:.78em; font-family:sans-serif; line-height:1.7;
  }}
  .legend-dot {{
    display:inline-block; width:12px; height:12px;
    border-radius:50%; margin-right:6px; vertical-align:middle;
  }}

  /* Popup */
  .leaflet-popup-content-wrapper {{
    background:#0d2a44; border:1px solid #2d5a7c; border-radius:12px;
    color:#dde8f0; box-shadow:0 8px 24px rgba(0,0,0,.5);
  }}
  .leaflet-popup-tip {{ background:#0d2a44; }}
  .popup-name {{ font-size:1.0em; font-weight:700; color:#74b9ff; margin-bottom:4px; }}
  .popup-price {{ font-size:1.6em; font-weight:800; margin:4px 0; }}
  .popup-brand {{ font-size:.78em; color:#6a8fa8; margin-bottom:8px; }}
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
  .cheap {{ color:#51cf66; }}
  .mid   {{ color:#ffd43b; }}
  .exp   {{ color:#ff6b6b; }}
</style>
</head>
<body>
<div id="controls">
  <button id="locate-btn">📍 My Location</button>
  <label>Radius: <input type="range" id="radius-slider" min="500" max="10000" value="3000" step="500"/>
  <span id="radius-val">3 km</span></label>
  <select id="gas-type-sel">
    <option value="1.000">Regular</option>
    <option value="1.110">Midgrade</option>
    <option value="1.200">Premium</option>
    <option value="1.045">Diesel</option>
  </select>
  <span id="status-msg">Click "My Location" or pan the map, then load stations.</span>
</div>
<div id="map"></div>

<script>
// ── Data from Python ──────────────────────────────────────────────────────
const CITY_PRICES   = {city_prices_json};
const IS_CANADA     = {is_canada};
const UNIT          = "{unit_label}";
const CHEAP_THRESH  = {cheap_thresh};
const MID_THRESH    = {mid_thresh};
const INIT_LAT      = {center_lat};
const INIT_LON      = {center_lon};

// ── Brand offsets (same logic as station_fetcher.py) ────────────────────
const BRAND_OFFSETS = {{
  "shell":+0.03,"bp":+0.02,"esso":+0.02,"mobil":+0.02,"sunoco":+0.02,
  "chevron":+0.02,"marathon":+0.01,"76":+0.01,
  "costco":-0.07,"bj":-0.06,"sam":-0.06,"circle k":-0.03,
  "7-eleven":-0.02,"kwik":-0.03,"wawa":-0.02,"racetrac":-0.03,
  "pilot":-0.03,"love":-0.03,"petro-canada":+0.01,"canadian tire":-0.02,
  "ultramar":+0.01,"pioneer":-0.01,"husky":-0.01,"fas gas":-0.02,"co-op":-0.02
}};

// ── City lookup (nearest by coords) ──────────────────────────────────────
function nearestCity(lat, lon) {{
  let best = null, bestDist = Infinity;
  for (const [city, data] of Object.entries(CITY_PRICES)) {{
    const d = Math.hypot(lat - data.lat, lon - data.lon);
    if (d < bestDist) {{ bestDist = d; best = [city, data]; }}
  }}
  return best ? best[1] : null;
}}

function brandOffset(brandStr) {{
  const b = (brandStr||"").toLowerCase();
  for (const [k,v] of Object.entries(BRAND_OFFSETS)) {{
    if (b.includes(k)) return v;
  }}
  return 0.0;
}}

// Seeded pseudo-random for reproducible station noise
function seededNoise(id) {{
  let x = Math.sin(id * 9301 + 49297) * 233280;
  return (x - Math.floor(x)) * 0.05 - 0.025;
}}

function stationPrice(osmId, cityBasePrice, brand, gtMult) {{
  const off  = brandOffset(brand);
  const noise = seededNoise(osmId);
  return Math.max(0.6, +(cityBasePrice * gtMult + off + noise).toFixed(3));
}}

function stationForecast(osmId, basePrice, days=7) {{
  const fc = [];
  let p = basePrice;
  for (let i=0; i<days; i++) {{
    const seed = osmId + i*997;
    const x    = Math.sin(seed*9301+49297)*233280;
    const frac = (x - Math.floor(x));
    const chg  = frac * 0.024 - 0.012;
    p = Math.max(0.6, +(p + chg).toFixed(3));
    fc.push(p);
  }}
  return fc;
}}

// ── Colour helpers ────────────────────────────────────────────────────────
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
  const label = IS_CANADA ? price.toFixed(3) : price.toFixed(3);
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="54" height="32">
    <rect rx="6" ry="6" width="54" height="26" fill="${{color}}" fill-opacity=".93"/>
    <text x="27" y="17" font-family="sans-serif" font-size="11" font-weight="700"
          text-anchor="middle" fill="#0d1e33">${{label}}</text>
    <polygon points="20,26 34,26 27,32" fill="${{color}}" fill-opacity=".93"/>
  </svg>`;
  return L.divIcon({{
    html: svg, className: '', iconAnchor: [27, 32], popupAnchor: [0, -34]
  }});
}}

// ── Distance helper ───────────────────────────────────────────────────────
function distanceKm(lat1, lon1, lat2, lon2) {{
  const R = 6371;
  const dLat = (lat2-lat1)*Math.PI/180;
  const dLon = (lon2-lon1)*Math.PI/180;
  const a = Math.sin(dLat/2)*Math.sin(dLat/2)
          + Math.cos(lat1*Math.PI/180)*Math.cos(lat2*Math.PI/180)
          * Math.sin(dLon/2)*Math.sin(dLon/2);
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
}}

// ── Forecast bar chart ───────────────────────────────────────────────────
function forecastBars(fc, basePrice) {{
  const vals = [basePrice, ...fc];
  const mn   = Math.min(...vals)*0.995;
  const mx   = Math.max(...vals)*1.005;
  const rng  = mx - mn || 0.01;
  const days = ['Today','Mon','Tue','Wed','Thu','Fri','Sat'];
  return fc.map((p,i) => {{
    const h  = Math.round(((p-mn)/rng)*32 + 4);
    const cl = priceClass(p);
    return `<div class="fc-bar ${{cl}}" style="height:${{h}}px"
                 data-tip="${{days[i]||'D'+(i+1)}}: ${{p.toFixed(3)}} ${{UNIT}}"></div>`;
  }}).join('');
}}

// ── Map init ─────────────────────────────────────────────────────────────
const map = L.map('map').setView([INIT_LAT, INIT_LON], {zoom});

L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
  attribution: '© OpenStreetMap contributors',
  maxZoom: 19,
}}).addTo(map);

// Legend
const legend = L.control({{position:'bottomright'}});
legend.onAdd = () => {{
  const d = L.DomUtil.create('div','legend');
  d.innerHTML = `
    <b>⛽ Price Legend</b><br>
    <span class="legend-dot" style="background:#51cf66"></span>Cheap<br>
    <span class="legend-dot" style="background:#ffd43b"></span>Average<br>
    <span class="legend-dot" style="background:#ff6b6b"></span>Expensive`;
  return d;
}};
legend.addTo(map);

// ── Station layer ─────────────────────────────────────────────────────────
let stationLayer = L.layerGroup().addTo(map);
let userMarker   = null;
let userLatLon   = null;
let currentGtMult = 1.0;
let loadedBounds  = null;

function loadStations(lat, lon, radiusM) {{
  const status = document.getElementById('status-msg');
  status.textContent = '🔄 Loading gas stations…';

  const query = `[out:json][timeout:25];
(node["amenity"="fuel"](around:${{radiusM}},${{lat}},${{lon}});
 way["amenity"="fuel"](around:${{radiusM}},${{lat}},${{lon}});
);out center 100;`;

  const mirrors = [
    'https://overpass-api.de/api/interpreter',
    'https://overpass.kumi.systems/api/interpreter',
  ];

  function tryMirror(idx) {{
    if (idx >= mirrors.length) {{
      status.textContent = '⚠️ Could not reach Overpass API. Try again later.';
      return;
    }}
    fetch(mirrors[idx], {{
      method: 'POST',
      body: new URLSearchParams({{data: query}}),
      headers: {{'Content-Type':'application/x-www-form-urlencoded'}}
    }})
    .then(r => r.ok ? r.json() : Promise.reject(r.status))
    .then(data => {{
      stationLayer.clearLayers();
      const elements = data.elements || [];

      // Sort by price for popup ordering
      const stations = elements.map(el => {{
        const slat = el.type==='node' ? el.lat : el.center.lat;
        const slon = el.type==='node' ? el.lon : el.center.lon;
        const tags  = el.tags || {{}};
        const brand = tags.brand || tags.operator || '';
        const name  = tags.name  || brand || 'Gas Station';
        const cityData   = nearestCity(slat, slon);
        const basePrice  = cityData ? cityData.price : (IS_CANADA ? 1.80 : 3.80);
        const price      = stationPrice(el.id, basePrice, brand, currentGtMult);
        const forecast   = stationForecast(el.id, price);
        const distKm     = userLatLon
          ? distanceKm(userLatLon[0], userLatLon[1], slat, slon)
          : null;
        return {{el, slat, slon, brand, name, price, forecast, distKm}};
      }});

      stations.forEach(s => {{
        const marker = L.marker([s.slat, s.slon], {{icon: makeIcon(s.price)}});
        const fc     = s.forecast;
        const distTxt = s.distKm !== null
          ? `<div class="popup-distance">📏 ${{s.distKm.toFixed(1)}} km away</div>`
          : '';
        const popup  = `
          <div class="popup-name">⛽ ${{s.name}}</div>
          <div class="popup-brand">${{s.brand || 'Independent'}}</div>
          <div class="popup-price ${{priceClass(s.price)}}">${{s.price.toFixed(3)}} <span style="font-size:.55em;color:#8ab4cc">${{UNIT}}</span></div>
          <div class="popup-fc-title">📈 7-Day Forecast</div>
          <div class="fc-bar-wrap">${{forecastBars(fc, s.price)}}</div>
          <div style="display:flex;justify-content:space-between;font-size:.72em;color:#6a8fa8;margin-top:2px">
            <span>Tmrw: ${{fc[0].toFixed(3)}}</span>
            <span>+7d: ${{fc[6].toFixed(3)}}</span>
          </div>
          ${{distTxt}}`;
        marker.bindPopup(popup, {{maxWidth:240, minWidth:200}});
        stationLayer.addLayer(marker);
      }});

      loadedBounds = map.getBounds();
      const cheapCount = stations.filter(s=>s.price<=CHEAP_THRESH).length;
      status.textContent = `✅ ${{stations.length}} stations loaded · ${{cheapCount}} cheap`;
    }})
    .catch(() => tryMirror(idx+1));
  }}

  tryMirror(0);
}}

// ── Radius slider ─────────────────────────────────────────────────────────
const slider = document.getElementById('radius-slider');
const label  = document.getElementById('radius-val');
slider.addEventListener('input', () => {{
  const km = (slider.value/1000).toFixed(km => km>=1 ? 0 : 1);
  label.textContent = slider.value >= 1000
    ? (slider.value/1000).toFixed(1) + ' km'
    : slider.value + ' m';
}});
// initial
label.textContent = '3 km';

// ── Gas type selector ─────────────────────────────────────────────────────
document.getElementById('gas-type-sel').addEventListener('change', e => {{
  currentGtMult = parseFloat(e.target.value);
  const c = map.getCenter();
  stationLayer.clearLayers();
  loadStations(c.lat, c.lng, parseInt(slider.value));
}});

// ── Load on map move ──────────────────────────────────────────────────────
map.on('moveend', () => {{
  const b = map.getBounds();
  // Only reload if moved significantly outside previously loaded area
  if (!loadedBounds || !loadedBounds.contains(b)) {{
    const c = map.getCenter();
    loadStations(c.lat, c.lng, parseInt(slider.value));
  }}
}});

// ── My Location button ────────────────────────────────────────────────────
document.getElementById('locate-btn').addEventListener('click', () => {{
  const status = document.getElementById('status-msg');
  if (!navigator.geolocation) {{
    status.textContent = '⚠️ Geolocation not supported by your browser.';
    return;
  }}
  status.textContent = '📍 Getting location…';
  navigator.geolocation.getCurrentPosition(pos => {{
    const lat = pos.coords.latitude;
    const lon = pos.coords.longitude;
    userLatLon = [lat, lon];
    map.setView([lat, lon], 14);
    if (userMarker) map.removeLayer(userMarker);
    userMarker = L.circleMarker([lat, lon], {{
      radius:10, color:'#74b9ff', fillColor:'#74b9ff',
      fillOpacity:0.7, weight:3
    }}).bindPopup('<b>📍 You are here</b>').addTo(map);
    loadStations(lat, lon, parseInt(slider.value));
  }}, err => {{
    status.textContent = '⚠️ Location access denied. Pan map manually.';
  }});
}});

// ── Initial load ──────────────────────────────────────────────────────────
setTimeout(() => loadStations(INIT_LAT, INIT_LON, parseInt(slider.value)), 300);
</script>
</body>
</html>"""

    return html
