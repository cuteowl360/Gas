"""
app.py – GasWatch  v2
──────────────────────
Weather-app-style gas price predictor built with Streamlit.

Run:
    streamlit run app.py
"""

import os, sys
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_collector import DataCollector, GAS_TYPES
from src.predictor import GasPricePredictor
from src.station_map import build_station_map_html

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GasWatch – Gas Price Predictor",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp {
    background: linear-gradient(160deg,#0a1628 0%,#0d1e33 40%,#122640 100%);
    color:#dde8f0;
    font-family:'Segoe UI',system-ui,sans-serif;
}
.block-container{padding-top:.6rem !important;}

[data-testid="stTabs"] button{
    color:#6a8fa8 !important;font-size:.95em;
    font-weight:600;letter-spacing:.3px;padding:6px 18px;
}
[data-testid="stTabs"] button[aria-selected="true"]{
    color:#74b9ff !important;
    border-bottom:2px solid #74b9ff !important;
}

.app-header{text-align:center;padding:4px 0 0 0;}
.app-title{font-size:2.4em;font-weight:800;letter-spacing:-1px;
           color:#fff;text-shadow:0 2px 14px rgba(0,0,0,.7);}
.app-sub{font-size:.92em;color:#5d7d90;margin-top:-4px;}

.kpi-card{
    background:linear-gradient(135deg,rgba(255,255,255,.08),rgba(255,255,255,.04));
    border:1px solid rgba(255,255,255,.10);border-radius:18px;
    padding:20px 16px 16px;text-align:center;
    box-shadow:0 6px 24px rgba(0,0,0,.4);min-height:148px;
}
.kpi-label{font-size:.72em;text-transform:uppercase;letter-spacing:1.8px;
           color:#4a6a7c;margin-bottom:5px;}
.kpi-value{font-size:2.8em;font-weight:800;line-height:1.1;
           font-variant-numeric:tabular-nums;}
.kpi-sub{font-size:.85em;margin-top:5px;opacity:.85;}
.kpi-unit{font-size:.68em;color:#354d5c;margin-top:4px;
          text-transform:uppercase;letter-spacing:1px;}

.fc-card{
    background:rgba(255,255,255,.05);
    border:1px solid rgba(255,255,255,.09);
    border-radius:14px;padding:14px 6px 12px;text-align:center;
}
.fc-day{font-size:.72em;text-transform:uppercase;letter-spacing:1.5px;
        color:#4a6a7c;margin-bottom:4px;}
.fc-date{font-size:.65em;color:#344a57;margin-bottom:4px;}
.fc-price{font-size:1.45em;font-weight:700;}
.fc-chg{font-size:.75em;color:#5d7d90;margin-top:3px;}

.sh{font-size:1em;font-weight:700;color:#8ab4cc;
    padding:8px 0 4px 2px;
    border-bottom:1px solid rgba(255,255,255,.08);
    margin-bottom:10px;letter-spacing:.4px;}

.alert-red{background:rgba(255,107,107,.12);border:1px solid rgba(255,107,107,.35);
           border-radius:10px;padding:9px 14px;color:#ff9090;
           font-size:.9em;margin-bottom:8px;}
.alert-green{background:rgba(81,207,102,.10);border:1px solid rgba(81,207,102,.30);
             border-radius:10px;padding:9px 14px;color:#6dda7e;
             font-size:.9em;margin-bottom:8px;}

[data-testid="stSidebar"]{
    background:rgba(6,15,26,.97) !important;
    border-right:1px solid rgba(255,255,255,.06);
}

.info-pill{background:rgba(116,185,255,.08);border:1px solid rgba(116,185,255,.22);
           border-radius:8px;padding:8px 12px;color:#7aadcc;font-size:.8em;}

.calc-result{
    background:linear-gradient(135deg,rgba(116,185,255,.12),rgba(253,121,168,.08));
    border:1px solid rgba(116,185,255,.25);
    border-radius:16px;padding:22px;text-align:center;
}
.calc-big{font-size:2.8em;font-weight:800;color:#74b9ff;}

.c-up{color:#ff6b6b;} .c-down{color:#51cf66;} .c-flat{color:#ffd43b;} .c-blue{color:#74b9ff;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ First-run: generating data & training model…")
def _boot():
    c = DataCollector()
    data = c.load_or_generate_data()
    p = GasPricePredictor()
    if not p.load():
        p.train(data)
        p.save()
    return c, p, data

collector, predictor, df = _boot()

@st.cache_data(ttl=3600)
def _crude() -> float:
    return collector.get_crude_oil_price()

@st.cache_data(ttl=600, show_spinner=False)
def _all_forecasts(crude: float) -> pd.DataFrame:
    rows, groups = [], {c: g.sort_values("date") for c, g in df.groupby("city")}
    for city, cdf in groups.items():
        today = float(cdf["price"].iloc[-1])
        fc = predictor.forecast_days(city=city, city_history=cdf,
                                     crude_price=crude, start_date=datetime.now(), days=1)
        rows.append({"city": city, "state": cdf["state"].iloc[-1],
                     "today": today, "tomorrow": fc[0],
                     "lat": cdf["lat"].iloc[-1], "lon": cdf["lon"].iloc[-1]})
    return pd.DataFrame(rows)


@st.cache_data(ttl=600, show_spinner=False)
def _station_map_prices(crude: float) -> dict:
    """Build city_prices dict for the station map: city -> {price, lat, lon, country}."""
    from src.data_collector import CITIES, usd_gal_to_cad_litre
    groups = {c: g.sort_values("date") for c, g in df.groupby("city")}
    result = {}
    for city, cdf in groups.items():
        city_info = CITIES.get(city, {})
        country   = city_info.get("country", "US")
        today_usd = float(cdf["price"].iloc[-1])
        fc_usd    = predictor.forecast_days(city=city, city_history=cdf,
                                             crude_price=crude,
                                             start_date=datetime.now(), days=7)
        if country == "CA":
            price   = usd_gal_to_cad_litre(today_usd)
            fc_disp = [usd_gal_to_cad_litre(p) for p in fc_usd]
        else:
            price   = round(today_usd, 3)
            fc_disp = [round(p, 3) for p in fc_usd]
        result[city] = {
            "price":   price,
            "lat":     city_info.get("lat", float(cdf["lat"].iloc[-1])),
            "lon":     city_info.get("lon", float(cdf["lon"].iloc[-1])),
            "country": country,
            "forecast": fc_disp,
        }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⛽ GasWatch")
    st.markdown("*Gas Price Forecasts · ML-Powered*\n\n---")

    cities = sorted(df["city"].unique().tolist())
    def_idx = cities.index("Oakville, ON") if "Oakville, ON" in cities else 0
    selected = st.selectbox("📍 City", cities, index=def_idx)

    gas_type   = st.selectbox("⛽ Gas Type", list(GAS_TYPES.keys()), index=0)
    multiplier = GAS_TYPES[gas_type]

    _sb_canadian = (df[df["city"]==selected]["country"].iloc[-1] == "CA") if "country" in df.columns else False

    st.markdown("---")
    st.markdown("**✏️ Manual Price Override**")
    use_manual = st.checkbox("Enter today's price manually")
    manual_price = 0.0
    if use_manual:
        if _sb_canadian:
            _inp_cad = st.number_input("Today's price (CAD/L)", 0.50, 4.00, 1.52, 0.01, "%.2f")
            manual_price = _inp_cad * 3.785 / 1.36  # store internally as USD/gal
        else:
            manual_price = st.number_input("Today's price ($/gal)", 0.50, 9.99,
                                           3.50, 0.01, "%.2f")

    st.markdown("---")
    st.markdown("**⚙️ Settings**")
    fc_days   = st.slider("Forecast days", 1, 7, 5)
    hist_days = st.slider("History window (days)", 7, 365, 60)

    st.markdown("---")
    st.markdown("**🔔 Price Alert**")
    alert_on    = st.checkbox("Notify if price exceeds")
    if _sb_canadian:
        alert_limit = st.number_input("Threshold (CAD/L)", 0.50, 4.00,
                                      1.75, 0.05, "%.2f", disabled=not alert_on)
    else:
        alert_limit = st.number_input("Threshold ($/gal)", 1.0, 9.0,
                                      4.50, 0.10, "%.2f", disabled=not alert_on)

    st.markdown("---")
    st.markdown('<div class="info-pill">💡 Canadian prices from <b>NRCan</b>. '
                'US prices from <b>AAA</b>. Crude oil (<b>WTI</b>) via Yahoo Finance.'
                ' Data refreshes every 12 hours.</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="app-header">'
    '<div class="app-title">⛽ GasWatch</div>'
    '<div class="app-sub">Gas Price Prediction Engine · North America · ML-Powered</div>'
    '</div>', unsafe_allow_html=True)

hl, hr = st.columns([4, 1])
with hl:
    st.markdown(f"### 📍 {selected}  ·  *{gas_type}*")
with hr:
    st.markdown(
        f"<div style='text-align:right;color:#344a57;padding-top:10px'>"
        f"{datetime.now().strftime('%A, %b %d %Y')}</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Core data
# ─────────────────────────────────────────────────────────────────────────────
city_df  = df[df["city"] == selected].sort_values("date").copy()
crude    = _crude()
all_fc   = _all_forecasts(crude)

raw_today     = manual_price if (use_manual and manual_price > 0) else float(city_df["price"].iloc[-1])
raw_yesterday = float(city_df["price"].iloc[-2]) if len(city_df) >= 2 else raw_today
raw_week_ago  = float(city_df["price"].iloc[-8]) if len(city_df) >= 8 else raw_today

raw_forecast  = predictor.forecast_days(
    city=selected, city_history=city_df,
    crude_price=crude, start_date=datetime.now(), days=fc_days)

today_p     = raw_today     * multiplier
yesterday_p = raw_yesterday * multiplier
week_ago_p  = raw_week_ago  * multiplier
fc_prices   = [p * multiplier for p in raw_forecast]
tomorrow_p  = fc_prices[0]

daily_chg    = today_p   - yesterday_p
weekly_chg   = today_p   - week_ago_p
tomorrow_chg = tomorrow_p - today_p

# ── Display unit helpers (CAD/L for Canadian cities, USD/gal for US) ────────
_is_canadian    = (city_df["country"].iloc[-1] == "CA") if "country" in city_df.columns else False
_CAD_PER_USD    = 1.36
_LITRES_PER_GAL = 3.785
_UNIT           = "CAD/L" if _is_canadian else "$/gal"
_CURR           = "C$"   if _is_canadian else "$"

def _to_disp(usd_gal):
    """Convert USD/gal → display unit (CAD/L for CA, USD/gal for US)."""
    return usd_gal * _CAD_PER_USD / _LITRES_PER_GAL if _is_canadian else usd_gal

def _fmt(usd_gal, d=3):
    v = _to_disp(usd_gal)
    return f"{v:.{d}f}"

def _fmt_chg(usd_gal_delta, d=3):
    v = usd_gal_delta * _CAD_PER_USD / _LITRES_PER_GAL if _is_canadian else usd_gal_delta
    return f"{v:+.{d}f}"

# Display-unit prices (what the user actually sees)
today_disp     = _to_disp(today_p)
yesterday_disp = _to_disp(yesterday_p)
week_ago_disp  = _to_disp(week_ago_p)
tomorrow_disp  = _to_disp(tomorrow_p)
fc_prices_disp = [_to_disp(p) for p in fc_prices]

daily_chg_disp    = today_disp  - yesterday_disp
weekly_chg_disp   = today_disp  - week_ago_disp
tomorrow_chg_disp = tomorrow_disp - today_disp

def _col(d):   return "c-up" if d > 0.015 else ("c-down" if d < -0.015 else "c-flat")
def _arrow(d): return "▲"   if d > 0.015 else ("▼"     if d < -0.015 else "●")

all_fc["today_typed"]    = all_fc["today"]    * multiplier
all_fc["tomorrow_typed"] = all_fc["tomorrow"] * multiplier
all_fc["chg"]            = all_fc["tomorrow_typed"] - all_fc["today_typed"]
all_fc["trend"] = all_fc["chg"].apply(
    lambda x: "🔴 Rising" if x > 0.015 else ("🟢 Falling" if x < -0.015 else "🟡 Stable"))


# ─────────────────────────────────────────────────────────────────────────────
# Alerts
# ─────────────────────────────────────────────────────────────────────────────
if alert_on:
    _cmp_today  = today_disp
    _cmp_tmrw   = tomorrow_disp
    if _cmp_today >= alert_limit:
        st.markdown(
            f'<div class="alert-red">🚨 <b>Alert!</b> Current {gas_type} price '
            f'<b>{_fmt(today_p, 2)} {_UNIT}</b> exceeds <b>{alert_limit:.2f} {_UNIT}</b> in {selected}.</div>',
            unsafe_allow_html=True)
    elif _cmp_tmrw >= alert_limit:
        st.markdown(
            f'<div class="alert-red">⚠️ Tomorrow\'s predicted price '
            f'<b>{_fmt(tomorrow_p, 2)} {_UNIT}</b> may exceed <b>{alert_limit:.2f} {_UNIT}</b>.</div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="alert-green">✅ Price <b>{_fmt(today_p, 2)} {_UNIT}</b> is below your '
            f'<b>{alert_limit:.2f} {_UNIT}</b> threshold.</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# KPI cards
# ─────────────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)

# KPI averages — split US vs Canada
_us_fc = all_fc[all_fc["city"].map(lambda c: not c.endswith((" ON"," BC"," AB"," QC"," MB")))]
_ca_fc = all_fc[all_fc["city"].map(lambda c:     c.endswith((" ON"," BC"," AB"," QC"," MB")))]

if _is_canadian:
    _avg_val   = _to_disp(_ca_fc["today_typed"].mean()) if len(_ca_fc) else today_disp
    _avg_label = f"CA Avg · {gas_type}"
    _avg_n     = len(_ca_fc)
else:
    _avg_val   = _us_fc["today_typed"].mean() if len(_us_fc) else all_fc["today_typed"].mean()
    _avg_label = f"US Avg · {gas_type}"
    _avg_n     = len(_us_fc)

for col, label, val, sub, unit, cls in [
    (k1, "Today's Price",    f"{_fmt(today_p, 2)}",
     f"{_arrow(daily_chg_disp)} {_fmt_chg(daily_chg)} vs yesterday", _UNIT + " · " + gas_type, _col(daily_chg_disp)),
    (k2, "Tomorrow Forecast",f"{_fmt(tomorrow_p, 2)}",
     f"{_arrow(tomorrow_chg_disp)} {_fmt_chg(tomorrow_chg)} projected", "ML · Random Forest", _col(tomorrow_chg_disp)),
    (k3, "7-Day Change",     f"{_arrow(weekly_chg_disp)} {abs(weekly_chg_disp):.3f}",
     f"{_fmt_chg(weekly_chg)} vs last week", "Weekly trend", _col(weekly_chg_disp)),
    (k4, "WTI Crude Oil",    f"${crude:.2f}",
     "per barrel", "West Texas Intermediate", "c-blue"),
    (k5, "Avg Today",        f"{_avg_val:.2f}",
     f"across {_avg_n} cities", _avg_label, "c-flat"),
]:
    with col:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value {cls}">{val}</div>'
            f'<div class="kpi-sub">{sub}</div>'
            f'<div class="kpi-unit">{unit}</div>'
            f'</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_dash, tab_map, tab_compare, tab_analysis, tab_calc = st.tabs([
    "📊 Dashboard", "🗺️ Live Map", "🏙️ City Compare", "📈 Analysis", "⛽ Calculator"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab_dash:
    # Forecast strip
    st.markdown('<div class="sh">📅 Day-by-Day Forecast</div>', unsafe_allow_html=True)
    fc_cols = st.columns(fc_days)
    fc_labels = ["Tomorrow"] + [
        (datetime.now() + timedelta(days=i+1)).strftime("%A") for i in range(1, fc_days)]
    for idx, (col, lbl, price_usd, price_d) in enumerate(zip(fc_cols, fc_labels, fc_prices, fc_prices_disp)):
        chg_d    = price_d - today_disp
        date_str = (datetime.now() + timedelta(days=idx+1)).strftime("%b %d")
        with col:
            st.markdown(
                f'<div class="fc-card">'
                f'<div class="fc-day">{lbl}</div>'
                f'<div class="fc-date">{date_str}</div>'
                f'<div class="fc-price {_col(chg_d)}">{price_d:.2f}</div>'
                f'<div class="fc-chg">{_arrow(chg_d)} {chg_d:+.3f} {_UNIT}</div>'
                f'</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # History + forecast chart
    st.markdown('<div class="sh">📈 Price History & Forecast</div>', unsafe_allow_html=True)

    chart_df = city_df.tail(hist_days).copy()
    chart_df["price_disp"] = chart_df["price"].apply(lambda p: _to_disp(p * multiplier))
    fc_dates = [pd.Timestamp(datetime.now().date() + timedelta(days=i+1)) for i in range(fc_days)]
    _band_w  = 0.05 * (_CAD_PER_USD / _LITRES_PER_GAL if _is_canadian else 1)
    upper    = [p + _band_w + i*_band_w*0.18 for i, p in enumerate(fc_prices_disp)]
    lower    = [p - _band_w - i*_band_w*0.18 for i, p in enumerate(fc_prices_disp)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=chart_df["date"], y=chart_df["price_disp"],
        mode="lines", name="Historical",
        line=dict(color="#74b9ff", width=2.5),
        fill="tozeroy", fillcolor="rgba(116,185,255,.06)",
        hovertemplate=f"<b>%{{x|%b %d %Y}}</b><br>%{{y:.3f}} {_UNIT}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=fc_dates + fc_dates[::-1], y=upper + lower[::-1],
        fill="toself", fillcolor="rgba(253,121,168,.09)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Confidence Band", hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=[chart_df["date"].iloc[-1]] + fc_dates,
        y=[float(chart_df["price_disp"].iloc[-1])] + fc_prices_disp,
        mode="lines+markers", name="Forecast",
        line=dict(color="#fd79a8", width=2.5, dash="dot"),
        marker=dict(size=9, color="#fd79a8", symbol="diamond",
                    line=dict(color="white", width=1.5)),
        hovertemplate=f"<b>%{{x|%b %d %Y}}</b><br>Forecast: %{{y:.3f}} {_UNIT}<extra></extra>"))
    fig.add_vline(x=datetime.now().timestamp() * 1000,
                  line_dash="dash", line_color="rgba(255,255,255,.18)",
                  annotation_text="Today",
                  annotation_font_color="rgba(255,255,255,.35)")
    _yaxis_prefix = "" if _is_canadian else "$"
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8ab4cc", size=12),
        xaxis=dict(gridcolor="rgba(255,255,255,.04)", tickformat="%b %d",
                   tickfont=dict(color="#4a6a7c")),
        yaxis=dict(gridcolor="rgba(255,255,255,.04)",
                   title=dict(text=f"Price ({_UNIT})", font=dict(color="#4a6a7c")),
                   tickprefix=_yaxis_prefix,
                   tickfont=dict(color="#4a6a7c")),
        legend=dict(bgcolor="rgba(0,0,0,.45)", bordercolor="rgba(255,255,255,.08)",
                    borderwidth=1),
        hovermode="x unified", margin=dict(l=10,r=10,t=14,b=10), height=380)
    st.plotly_chart(fig, use_container_width=True)

    # Monthly average bar
    st.markdown('<div class="sh">📅 Monthly Average Prices</div>', unsafe_allow_html=True)
    monthly_df = city_df.copy()
    monthly_df["period"] = monthly_df["date"].dt.to_period("M")
    monthly_gb = monthly_df.groupby("period")["price"].mean().reset_index()
    monthly_gb["label"] = monthly_gb["period"].astype(str)
    monthly_gb["avg"]   = monthly_gb["price"].apply(lambda p: _to_disp(p * multiplier))
    bar_fig = go.Figure(go.Bar(
        x=monthly_gb["label"], y=monthly_gb["avg"],
        marker=dict(color=monthly_gb["avg"],
                    colorscale=[[0,"#51cf66"],[0.5,"#ffd43b"],[1,"#ff6b6b"]],
                    showscale=False),
        hovertemplate=f"<b>%{{x}}</b><br>Avg: %{{y:.3f}} {_UNIT}<extra></extra>"))
    bar_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8ab4cc"),
        xaxis=dict(tickfont=dict(color="#4a6a7c"), tickangle=-45,
                   gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(gridcolor="rgba(255,255,255,.04)", tickprefix=_yaxis_prefix,
                   tickfont=dict(color="#4a6a7c")),
        margin=dict(l=10,r=10,t=10,b=60), height=280)
    st.plotly_chart(bar_fig, use_container_width=True)

    with st.expander("🧠 Model Feature Importance"):
        fi = predictor.get_feature_importance()
        if not fi.empty:
            fi_fig = go.Figure(go.Bar(
                x=fi.head(12)["importance"], y=fi.head(12)["feature"],
                orientation="h",
                marker=dict(color=fi.head(12)["importance"],
                            colorscale="Blues", showscale=False)))
            fi_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#8ab4cc"),
                xaxis=dict(gridcolor="rgba(255,255,255,.04)"),
                yaxis=dict(autorange="reversed", tickfont=dict(color="#6a8fa8")),
                height=300, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fi_fig, use_container_width=True)
        st.markdown("""
**Model:** Random Forest Regressor (scikit-learn, n_jobs=-1)  
**Features:** 1/2/7/14/30-day lags · 7/30-day rolling avg · WTI crude · cyclical date encoding · city label · momentum  
**Forecast:** auto-regressive — each predicted price feeds back as the next day's lag
""")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – LIVE MAP  (interactive gas station map)
# ══════════════════════════════════════════════════════════════════════════════
with tab_map:
    from src.data_collector import CITIES, usd_gal_to_cad_litre

    # Canadian cities only for the city picker (this map is Canada-focused)
    ca_cities = sorted([c for c in df["city"].unique() if CITIES.get(c, {}).get("country") == "CA"])
    def_map_idx = ca_cities.index("Oakville, ON") if "Oakville, ON" in ca_cities else 0

    # ── Controls row ──────────────────────────────────────────────────────
    mc1, mc2, mc3 = st.columns([2, 2, 1])
    with mc1:
        map_city = st.selectbox(
            "📍 Centre map on Canadian city",
            options=ca_cities,
            index=def_map_idx,
            key="map_city_sel",
        )
    with mc2:
        map_zoom = st.slider("Zoom level", 11, 17, 13, key="map_zoom_sl")
    with mc3:
        st.markdown("<br>", unsafe_allow_html=True)
        reload_map = st.button("🔄 Reload Map", key="map_reload")

    # City info for centring — always Canadian
    city_info   = CITIES.get(map_city, {})
    map_lat     = city_info.get("lat", 43.45)
    map_lon     = city_info.get("lon", -79.68)
    map_country = "CA"   # this map is Canada-focused

    # Instruction banner
    st.markdown(
        '<div class="info-pill">🍁 Gas stations plotted from <b>NRCan city price data</b> '
        'with realistic brand offsets. Prices anchored to live NRCan city averages. '
        'Click <b>📍 My Location</b> to centre on your position. '
        'Zoom into any station pin for the live price and 7-day forecast. '
        'Adjust the <b>Radius slider</b> to show nearby stations.</div>',
        unsafe_allow_html=True
    )

    # Build city_prices dict and render the Leaflet map
    city_prices = _station_map_prices(crude)
    map_html    = build_station_map_html(
        center_lat=map_lat,
        center_lon=map_lon,
        city_prices=city_prices,
        country="CA",
        zoom=map_zoom,
        height_px=620,
    )
    components.html(map_html, height=680, scrolling=False)

    # ── Canadian city rankings table ──────────────────────────────────────
    st.markdown('<div class="sh">🍁 Canadian Cities — Ranked by Today\'s Price (CAD/L)</div>',
                unsafe_allow_html=True)
    ca_rank = all_fc[all_fc["city"].isin(ca_cities)].copy()
    # Convert USD/gal → CAD/L for display
    _CAD_L = lambda usd: round(usd * 1.36 / 3.785, 3)
    ca_rank["Today (CAD/L)"]    = ca_rank["today_typed"].apply(_CAD_L)
    ca_rank["Tomorrow (CAD/L)"] = ca_rank["tomorrow_typed"].apply(_CAD_L)
    ca_rank["Δ (CAD/L)"]        = (ca_rank["tomorrow_typed"] - ca_rank["today_typed"]).apply(
                                      lambda x: f"{_CAD_L(x):+.3f}")
    ca_rank = ca_rank.sort_values("Today (CAD/L)")[["city", "state", "Today (CAD/L)", "Tomorrow (CAD/L)", "Δ (CAD/L)", "trend"]]
    ca_rank.columns = ["City", "Province", "Today (CAD/L)", "Tomorrow (CAD/L)", "Δ (CAD/L)", "Trend"]
    ca_rank["Today (CAD/L)"]    = ca_rank["Today (CAD/L)"].map("{:.3f}".format)
    ca_rank["Tomorrow (CAD/L)"] = ca_rank["Tomorrow (CAD/L)"].map("{:.3f}".format)
    st.dataframe(ca_rank.set_index("City"), use_container_width=True, height=320)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – CITY COMPARE
# ══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    defaults = ["Oakville, ON","Toronto, ON","Vancouver, BC","New York, NY","Los Angeles, CA"]
    defaults = [c for c in defaults if c in cities]
    cmp_cities = st.multiselect("Select cities to compare (up to 8)", cities,
                                default=defaults[:5], max_selections=8)

    if len(cmp_cities) >= 2:
        cmp_days = st.slider("Days", 7, 365, 60, key="cmp_days")
        palette  = ["#74b9ff","#fd79a8","#55efc4","#ffeaa7",
                    "#a29bfe","#fdcb6e","#e17055","#6c5ce7"]
        cmp_fig = go.Figure()
        for i, city in enumerate(cmp_cities):
            cdf = df[df["city"]==city].sort_values("date").tail(cmp_days)
            cmp_fig.add_trace(go.Scatter(
                x=cdf["date"], y=cdf["price"]*multiplier,
                mode="lines", name=city,
                line=dict(color=palette[i%len(palette)], width=2),
                hovertemplate=f"<b>{city}</b><br>%{{x|%b %d}}: $%{{y:.3f}}/gal<extra></extra>"))
        cmp_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8ab4cc"),
            xaxis=dict(gridcolor="rgba(255,255,255,.04)", tickformat="%b %d",
                       tickfont=dict(color="#4a6a7c")),
            yaxis=dict(gridcolor="rgba(255,255,255,.04)", tickprefix="$",
                       title="Price ($/gal)", tickfont=dict(color="#4a6a7c")),
            legend=dict(bgcolor="rgba(0,0,0,.5)",
                        bordercolor="rgba(255,255,255,.08)", borderwidth=1),
            hovermode="x unified",
            margin=dict(l=10,r=10,t=14,b=10), height=420)
        st.plotly_chart(cmp_fig, use_container_width=True)

        st.markdown('<div class="sh">📊 Stats Summary</div>', unsafe_allow_html=True)
        rows = []
        for city in cmp_cities:
            cdf    = df[df["city"]==city].sort_values("date").tail(cmp_days)
            prices = cdf["price"] * multiplier
            fc_row = all_fc[all_fc["city"]==city]
            tmrw   = float(fc_row["tomorrow_typed"].iloc[0]) if len(fc_row) else 0
            rows.append({
                "City":            city,
                "Current":         f"${prices.iloc[-1]:.3f}",
                "Tomorrow":        f"${tmrw:.3f}",
                f"{cmp_days}d Low":  f"${prices.min():.3f}",
                f"{cmp_days}d High": f"${prices.max():.3f}",
                f"{cmp_days}d Avg":  f"${prices.mean():.3f}",
                "Volatility (σ)":  f"${prices.std():.3f}",
            })
        st.dataframe(pd.DataFrame(rows).set_index("City"), use_container_width=True)
    else:
        st.info("Select at least 2 cities to compare.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab_analysis:
    ac1, ac2 = st.columns(2)
    with ac1:
        st.markdown('<div class="sh">🛢️ Gas vs WTI Crude (last 90 days)</div>',
                    unsafe_allow_html=True)
        oil_df = city_df.tail(90).copy()
        oil_df["price_disp"] = oil_df["price"].apply(lambda p: _to_disp(p * multiplier))
        fig_oil = go.Figure()
        fig_oil.add_trace(go.Scatter(
            x=oil_df["date"], y=oil_df["price_disp"],
            name=f"Gas ({_UNIT})", yaxis="y1",
            line=dict(color="#74b9ff", width=2),
            hovertemplate=f"%{{x|%b %d}}: %{{y:.3f}} {_UNIT}<extra>Gas</extra>"))
        fig_oil.add_trace(go.Scatter(
            x=oil_df["date"], y=oil_df["crude_oil"],
            name="WTI ($/bbl)", yaxis="y2",
            line=dict(color="#ffeaa7", width=2, dash="dot"),
            hovertemplate="%{x|%b %d}: $%{y:.2f}<extra>WTI</extra>"))
        fig_oil.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8ab4cc"),
            yaxis=dict(title=f"Gas ({_UNIT})", tickprefix=_yaxis_prefix,
                       gridcolor="rgba(255,255,255,.04)",
                       tickfont=dict(color="#4a6a7c")),
            yaxis2=dict(title="WTI ($/bbl)", tickprefix="$",
                        overlaying="y", side="right",
                        tickfont=dict(color="#4a6a7c")),
            legend=dict(bgcolor="rgba(0,0,0,.4)"),
            hovermode="x unified",
            margin=dict(l=10,r=10,t=10,b=10), height=300)
        st.plotly_chart(fig_oil, use_container_width=True)

    with ac2:
        st.markdown('<div class="sh">📆 Average by Day of Week</div>',
                    unsafe_allow_html=True)
        dow_df = city_df.copy()
        dow_df["price_typed"] = dow_df["price"].apply(lambda p: _to_disp(p * multiplier))
        dow_df["dow"] = dow_df["date"].dt.day_name()
        dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        dow_avg = dow_df.groupby("dow")["price_typed"].mean().reindex(dow_order)
        fig_dow = go.Figure(go.Bar(
            x=dow_avg.index, y=dow_avg.values,
            marker=dict(color=dow_avg.values,
                        colorscale=[[0,"#51cf66"],[0.5,"#ffd43b"],[1,"#ff6b6b"]],
                        showscale=False),
            hovertemplate="<b>%{x}</b><br>$%{y:.3f}/gal<extra></extra>"))
        fig_dow.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8ab4cc"),
            yaxis=dict(tickprefix=_yaxis_prefix, gridcolor="rgba(255,255,255,.04)",
                       tickfont=dict(color="#4a6a7c")),
            xaxis=dict(tickfont=dict(color="#4a6a7c")),
            margin=dict(l=10,r=10,t=10,b=10), height=300)
        st.plotly_chart(fig_dow, use_container_width=True)

    # Year-over-year
    st.markdown('<div class="sh">📅 Year-over-Year Comparison</div>', unsafe_allow_html=True)
    yoy_df = city_df.copy()
    yoy_df["price_typed"] = yoy_df["price"].apply(lambda p: _to_disp(p * multiplier))
    yoy_df["year"] = yoy_df["date"].dt.year
    yoy_df["doy"]  = yoy_df["date"].dt.dayofyear
    colors_yoy = {2022:"#a29bfe",2023:"#ffeaa7",2024:"#74b9ff",2025:"#fd79a8",2026:"#55efc4"}
    yoy_fig = go.Figure()
    for yr in sorted(yoy_df["year"].unique()):
        s = yoy_df[yoy_df["year"]==yr]
        yoy_fig.add_trace(go.Scatter(
            x=s["doy"], y=s["price_typed"], name=str(yr), mode="lines",
            line=dict(color=colors_yoy.get(yr,"#ffffff"), width=2),
            hovertemplate=f"<b>{yr}</b> Day %{{x}}: $%{{y:.3f}}<extra></extra>"))
    yoy_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8ab4cc"),
        xaxis=dict(title="Day of Year", gridcolor="rgba(255,255,255,.04)",
                   tickfont=dict(color="#4a6a7c")),
        yaxis=dict(tickprefix=_yaxis_prefix, gridcolor="rgba(255,255,255,.04)",
                   title=f"Price ({_UNIT})", tickfont=dict(color="#4a6a7c")),
        legend=dict(bgcolor="rgba(0,0,0,.4)", title="Year"),
        hovermode="x unified",
        margin=dict(l=10,r=10,t=10,b=10), height=320)
    st.plotly_chart(yoy_fig, use_container_width=True)

    # Seasonal box
    st.markdown('<div class="sh">📦 Price Distribution by Month</div>', unsafe_allow_html=True)
    sea_df = city_df.copy()
    sea_df["price_typed"]  = sea_df["price"].apply(lambda p: _to_disp(p * multiplier))
    sea_df["month_name"]   = sea_df["date"].dt.strftime("%b")
    month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    box_fig = go.Figure()
    for m in month_order:
        vals = sea_df[sea_df["month_name"]==m]["price_typed"]
        if len(vals):
            box_fig.add_trace(go.Box(
                y=vals, name=m,
                marker_color="#74b9ff", line_color="#4a6a7c",
                fillcolor="rgba(116,185,255,.12)", showlegend=False,
                hovertemplate=f"<b>{m}</b><br>$%{{y:.3f}}<extra></extra>"))
    box_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8ab4cc"),
        yaxis=dict(tickprefix=_yaxis_prefix, gridcolor="rgba(255,255,255,.04)",
                   tickfont=dict(color="#4a6a7c")),
        xaxis=dict(tickfont=dict(color="#4a6a7c")),
        margin=dict(l=10,r=10,t=10,b=10), height=300)
    st.plotly_chart(box_fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 – CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════
with tab_calc:
    st.markdown('<div class="sh">⛽ Fill-Up Cost Calculator</div>', unsafe_allow_html=True)
    cc1, cc2 = st.columns(2)
    with cc1:
        if _is_canadian:
            tank_size_l  = st.number_input("Tank size (litres)", 20.0, 120.0, 55.0, 1.0)
            tank_level   = st.slider("Current tank level (%)", 0, 100, 25)
            litres_need  = tank_size_l * (1 - tank_level/100)
            gallons_need = litres_need / _LITRES_PER_GAL  # for internal calc
        else:
            tank_size    = st.number_input("Tank size (gallons)", 5.0, 50.0, 15.0, 0.5)
            tank_level   = st.slider("Current tank level (%)", 0, 100, 25)
            gallons_need = tank_size * (1 - tank_level/100)
            litres_need  = gallons_need * _LITRES_PER_GAL
    with cc2:
        use_tmrw    = st.checkbox("Use tomorrow's predicted price")
        price_usd   = tomorrow_p if use_tmrw else today_p
        price_d_val = tomorrow_disp if use_tmrw else today_disp
        lbl_used    = "Tomorrow's forecast" if use_tmrw else "Today's price"
        total_cost  = gallons_need * price_usd   # internally USD
        total_disp_val = total_cost * (_CAD_PER_USD if _is_canadian else 1.0)
        savings_usd    = gallons_need * (today_p - tomorrow_p)
        savings_disp   = savings_usd  * (_CAD_PER_USD if _is_canadian else 1.0)
        vol_str     = f"{litres_need:.1f} L" if _is_canadian else f"{gallons_need:.1f} gal"
        curr_sym    = "C$" if _is_canadian else "$"

    st.markdown(
        f'<div class="calc-result">'
        f'<div class="kpi-label">{lbl_used}: {price_d_val:.3f} {_UNIT} · {vol_str} needed</div>'
        f'<div class="calc-big">{curr_sym}{total_disp_val:.2f}</div>'
        f'<div style="color:#5d7d90;font-size:.9em;margin-top:8px">'
        f'Fill-up cost · <b>{selected}</b> · <b>{gas_type}</b></div>'
        f'</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if abs(savings_disp) > 0.01:
        if savings_disp > 0:
            st.markdown(
                f'<div class="alert-green">💡 Waiting until tomorrow could save '
                f'<b>{curr_sym}{savings_disp:.2f}</b> (price predicted ▼ {abs(tomorrow_chg_disp):.3f} {_UNIT}).</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="alert-red">⚠️ Fill up today — tomorrow price rises '
                f'{tomorrow_chg_disp:+.3f} {_UNIT} (extra cost {curr_sym}{abs(savings_disp):.2f}).</div>',
                unsafe_allow_html=True)

    # Fill-up cost across all cities (always in local currency)
    st.markdown('<div class="sh">🗺️ Same Fill-Up Cost — All Cities</div>',
                unsafe_allow_html=True)
    fill_df = all_fc[["city","today_typed"]].copy()
    # For chart use USD fill cost for consistent comparison, label with $ or C$
    fill_df["fill_cost_usd"] = fill_df["today_typed"] * gallons_need
    fill_df["is_ca"] = fill_df["city"].str.endswith((" ON"," BC"," AB"," QC"," MB"))
    fill_df["fill_cost_disp"] = fill_df.apply(
        lambda r: r["fill_cost_usd"] * _CAD_PER_USD if r["is_ca"] else r["fill_cost_usd"], axis=1)
    fill_df = fill_df.sort_values("fill_cost_disp")
    fill_fig = go.Figure(go.Bar(
        x=fill_df["city"], y=fill_df["fill_cost_disp"],
        marker=dict(color=fill_df["fill_cost_disp"],
                    colorscale=[[0,"#51cf66"],[0.5,"#ffd43b"],[1,"#ff6b6b"]],
                    showscale=False),
        text=fill_df.apply(lambda r: f"{'C$' if r['is_ca'] else '$'}{r['fill_cost_disp']:.2f}", axis=1),
        textposition="outside",
        textfont=dict(color="#8ab4cc", size=9),
        hovertemplate="<b>%{x}</b><br>Fill-up: %{text}<extra></extra>"))
    fill_fig.add_hline(
        y=total_disp_val, line_dash="dash", line_color="rgba(253,121,168,.6)",
        annotation_text=f"{selected} ({curr_sym}{total_disp_val:.2f})",
        annotation_font_color="rgba(253,121,168,.8)")
    fill_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8ab4cc"),
        yaxis=dict(gridcolor="rgba(255,255,255,.04)",
                   title="Fill-up cost (local currency)", tickfont=dict(color="#4a6a7c")),
        xaxis=dict(tickfont=dict(color="#4a6a7c"), tickangle=-45),
        margin=dict(l=10,r=10,t=10,b=130), height=420)
    st.plotly_chart(fill_fig, use_container_width=True)

    # Monthly budget
    st.markdown('<div class="sh">📅 Monthly Fuel Budget</div>', unsafe_allow_html=True)
    mb1, mb2 = st.columns(2)
    with mb1:
        fills_mo = st.slider("Fill-ups per month", 1, 12, 4)
    with mb2:
        monthly_spend = fills_mo * total_disp_val
        st.markdown(
            f'<div class="calc-result" style="padding:16px">'
            f'<div class="kpi-label">{fills_mo} fill-ups/month at {price_d_val:.3f} {_UNIT}</div>'
            f'<div style="font-size:2.2em;font-weight:800;color:#74b9ff">{curr_sym}{monthly_spend:.2f}</div>'
            f'<div style="color:#5d7d90;font-size:.82em">estimated monthly fuel spend</div>'
            f'</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#1e3042;font-size:.75em'>"
    "GasWatch · Predictions are estimates only · Not financial advice · "
    "Sample data + live WTI via Yahoo Finance · "
    f"Updated: {datetime.now().strftime('%b %d %Y %H:%M')}"
    "</div>", unsafe_allow_html=True)

