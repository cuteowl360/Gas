"""
app.py – GasWatch
──────────────────
Weather-app-style gas price predictor built with Streamlit.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_collector import DataCollector
from src.predictor import GasPricePredictor

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GasWatch – Gas Price Predictor",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS  (dark gradient, weather-app cards)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Global ── */
.stApp {
    background: linear-gradient(145deg, #0d1b2a 0%, #1a2b3d 45%, #243447 100%);
    color: #dde8f0;
    font-family: 'Segoe UI', system-ui, sans-serif;
}
.block-container { padding-top: 0.8rem !important; }

/* ── App header ── */
.app-header  { text-align: center; padding: 6px 0 2px 0; }
.app-title   { font-size: 2.6em; font-weight: 800; letter-spacing: -1px;
               color: #ffffff; text-shadow: 0 2px 12px rgba(0,0,0,.6); }
.app-sub     { font-size: 1em; color: #7a9ab5; margin-top: -6px; }

/* ── Price cards ── */
.price-card {
    background: rgba(255,255,255,.07);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,.12);
    border-radius: 20px;
    padding: 22px 18px 18px 18px;
    text-align: center;
    box-shadow: 0 8px 30px rgba(0,0,0,.35);
    min-height: 155px;
}
.card-label  { font-size: .78em; text-transform: uppercase; letter-spacing: 1.8px;
               color: #607d8b; margin-bottom: 6px; }
.big-price   { font-size: 3em; font-weight: 800; line-height: 1.1;
               font-variant-numeric: tabular-nums; }
.card-change { font-size: .92em; margin-top: 6px; opacity: .88; }
.card-unit   { font-size: .72em; color: #435a68; margin-top: 5px;
               text-transform: uppercase; letter-spacing: 1px; }

/* ── Color helpers ── */
.c-up   { color: #ff6b6b; }
.c-down { color: #51cf66; }
.c-flat { color: #ffd43b; }
.c-blue { color: #74b9ff; }

/* ── Forecast mini-cards ── */
.fc-card {
    background: rgba(255,255,255,.055);
    border: 1px solid rgba(255,255,255,.10);
    border-radius: 14px;
    padding: 14px 8px 12px 8px;
    text-align: center;
    box-shadow: 0 4px 14px rgba(0,0,0,.2);
}
.fc-day   { font-size: .76em; text-transform: uppercase; letter-spacing: 1.5px;
            color: #5d7d90; margin-bottom: 5px; }
.fc-price { font-size: 1.55em; font-weight: 700; }
.fc-chg   { font-size: .78em; color: #7a9ab5; margin-top: 4px; }

/* ── Section header ── */
.sh {
    font-size: 1.05em; font-weight: 600; color: #b0c8de;
    padding: 10px 0 4px 2px;
    border-bottom: 1px solid rgba(255,255,255,.09);
    margin-bottom: 10px; letter-spacing: .4px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(10,22,34,.96) !important;
    border-right: 1px solid rgba(255,255,255,.07);
}

/* ── Alerts ── */
.alert-danger {
    background: rgba(255,107,107,.15); border: 1px solid rgba(255,107,107,.4);
    border-radius: 10px; padding: 10px 14px; color: #ff9999; font-size: .92em;
    margin-bottom: 8px;
}
.info-box {
    background: rgba(116,185,255,.09); border: 1px solid rgba(116,185,255,.28);
    border-radius: 10px; padding: 10px 14px; color: #90bedd; font-size: .84em;
}
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Initialise (cached across reruns – trains model on very first run)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ First-run setup: generating data & training model…")
def _initialize():
    collector = DataCollector()
    df = collector.load_or_generate_data()
    predictor = GasPricePredictor()
    if not predictor.load():
        predictor.train(df)
        predictor.save()
    return collector, predictor, df


collector, predictor, df = _initialize()


@st.cache_data(ttl=3600)
def _get_crude() -> float:
    return collector.get_crude_oil_price()


@st.cache_data(ttl=600, show_spinner=False)
def _all_city_forecasts(crude: float) -> pd.DataFrame:
    """Pre-compute today's price + 1-day forecast for every city."""
    rows = []
    city_groups = {c: g.sort_values("date") for c, g in df.groupby("city")}
    for city, cdf in city_groups.items():
        today = float(cdf["price"].iloc[-1])
        fc    = predictor.forecast_days(
            city=city, city_history=cdf,
            crude_price=crude, start_date=datetime.now(), days=1,
        )
        rows.append({"city": city, "today": today, "tomorrow": fc[0],
                     "lat":  cdf["lat"].iloc[-1], "lon": cdf["lon"].iloc[-1]})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⛽ GasWatch")
    st.markdown("*Real-time gas price forecasts*")
    st.markdown("---")

    st.markdown("### 📍 Location")
    cities      = sorted(df["city"].unique().tolist())
    default_idx = cities.index("Los Angeles, CA") if "Los Angeles, CA" in cities else 0
    selected    = st.selectbox("City", cities, index=default_idx, label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### ✏️ Manual Override")
    use_manual   = st.checkbox("Enter today's price manually")
    manual_price = 0.0
    if use_manual:
        manual_price = st.number_input(
            "Today's price ($/gal)", min_value=0.50, max_value=9.99,
            value=3.50, step=0.01, format="%.2f",
        )

    st.markdown("---")
    st.markdown("### ⚙️ Display Settings")
    fc_days     = st.slider("Forecast days ahead", 1, 7, 5)
    hist_days   = st.slider("History days to show", 7, 90, 30)
    show_map    = st.checkbox("Show price map", value=True)

    st.markdown("---")
    st.markdown("### 🔔 Price Alert")
    alert_on    = st.checkbox("Alert if price exceeds threshold")
    alert_limit = st.number_input(
        "Threshold ($/gal)", min_value=1.0, max_value=9.0,
        value=4.50, step=0.10, format="%.2f", disabled=not alert_on,
    )

    st.markdown("---")
    st.markdown(
        '<div class="info-box">💡 Gas prices use generated sample data.'
        " Live <b>WTI crude oil</b> price is fetched from Yahoo Finance."
        " Replace sample data with a real API (EIA, GasBuddy) for production.</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="app-header">'
    '<div class="app-title">⛽ GasWatch</div>'
    '<div class="app-sub">Gas Price Prediction Engine · ML-Powered</div>'
    "</div>",
    unsafe_allow_html=True,
)

c_loc, c_dt = st.columns([3, 1])
with c_loc:
    st.markdown(f"### 📍 {selected}")
with c_dt:
    st.markdown(
        f"<div style='text-align:right;color:#5d7d90;padding-top:10px'>"
        f"{datetime.now().strftime('%A, %b %d %Y')}</div>",
        unsafe_allow_html=True,
    )
st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# Data for selected city
# ─────────────────────────────────────────────────────────────────────────────
city_df   = df[df["city"] == selected].sort_values("date").copy()
crude     = _get_crude()

today_price     = manual_price if (use_manual and manual_price > 0) else float(city_df["price"].iloc[-1])
yesterday_price = float(city_df["price"].iloc[-2]) if len(city_df) >= 2 else today_price
week_ago_price  = float(city_df["price"].iloc[-8]) if len(city_df) >= 8 else today_price

forecast_prices = predictor.forecast_days(
    city=selected, city_history=city_df,
    crude_price=crude, start_date=datetime.now(), days=fc_days,
)
tomorrow_price = forecast_prices[0]


# ─────────────────────────────────────────────────────────────────────────────
# Helper: CSS class + arrow symbol
# ─────────────────────────────────────────────────────────────────────────────
def _cls(delta: float) -> str:
    return "c-up" if delta > 0.015 else ("c-down" if delta < -0.015 else "c-flat")


def _arrow(delta: float) -> str:
    return "▲" if delta > 0.015 else ("▼" if delta < -0.015 else "●")


daily_chg    = today_price   - yesterday_price
weekly_chg   = today_price   - week_ago_price
tomorrow_chg = tomorrow_price - today_price


# ─────────────────────────────────────────────────────────────────────────────
# Alerts
# ─────────────────────────────────────────────────────────────────────────────
if alert_on:
    if today_price >= alert_limit:
        st.markdown(
            f'<div class="alert-danger">🚨 <b>Price Alert!</b> '
            f"Current price <b>${today_price:.2f}/gal</b> exceeds your threshold of "
            f"<b>${alert_limit:.2f}/gal</b> in {selected}.</div>",
            unsafe_allow_html=True,
        )
    if tomorrow_price >= alert_limit:
        st.markdown(
            f'<div class="alert-danger">⚠️ <b>Forecast Alert!</b> '
            f"Tomorrow's predicted price <b>${tomorrow_price:.2f}/gal</b> may exceed "
            f"<b>${alert_limit:.2f}/gal</b>.</div>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Top KPI cards
# ─────────────────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(
        f'<div class="price-card">'
        f'<div class="card-label">Today\'s Price</div>'
        f'<div class="big-price {_cls(daily_chg)}">${today_price:.2f}</div>'
        f'<div class="card-change">{_arrow(daily_chg)} {daily_chg:+.3f} today</div>'
        f'<div class="card-unit">per gallon · Regular Unleaded</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

with k2:
    st.markdown(
        f'<div class="price-card">'
        f'<div class="card-label">Tomorrow\'s Forecast</div>'
        f'<div class="big-price {_cls(tomorrow_chg)}">${tomorrow_price:.2f}</div>'
        f'<div class="card-change">{_arrow(tomorrow_chg)} {tomorrow_chg:+.3f} projected</div>'
        f'<div class="card-unit">GBR · ML Prediction</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

with k3:
    st.markdown(
        f'<div class="price-card">'
        f'<div class="card-label">7-Day Change</div>'
        f'<div class="big-price {_cls(weekly_chg)}">{_arrow(weekly_chg)} {abs(weekly_chg):.3f}</div>'
        f'<div class="card-change">{weekly_chg:+.3f} vs last week</div>'
        f'<div class="card-unit">Weekly Trend</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

with k4:
    st.markdown(
        f'<div class="price-card">'
        f'<div class="card-label">WTI Crude Oil</div>'
        f'<div class="big-price c-blue">${crude:.2f}</div>'
        f'<div class="card-change">per barrel</div>'
        f'<div class="card-unit">West Texas Intermediate</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Historical + forecast chart
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="sh">📈 Price History & Forecast</div>', unsafe_allow_html=True)

chart_df      = city_df.tail(hist_days)
fc_dates      = [pd.Timestamp(datetime.now().date() + timedelta(days=i + 1)) for i in range(fc_days)]
upper_band    = [p + 0.04 + i * 0.008 for i, p in enumerate(forecast_prices)]
lower_band    = [p - 0.04 - i * 0.008 for i, p in enumerate(forecast_prices)]

fig = go.Figure()

# Shaded history
fig.add_trace(go.Scatter(
    x=chart_df["date"], y=chart_df["price"],
    mode="lines", name="Historical",
    line=dict(color="#74b9ff", width=2.5),
    fill="tozeroy", fillcolor="rgba(116,185,255,.07)",
    hovertemplate="<b>%{x|%b %d %Y}</b><br>$%{y:.3f}/gal<extra></extra>",
))

# Confidence band
fig.add_trace(go.Scatter(
    x=fc_dates + fc_dates[::-1],
    y=upper_band + lower_band[::-1],
    fill="toself", fillcolor="rgba(253,121,168,.10)",
    line=dict(color="rgba(0,0,0,0)"),
    name="Confidence Band", hoverinfo="skip",
))

# Forecast line
fig.add_trace(go.Scatter(
    x=[chart_df["date"].iloc[-1]] + fc_dates,
    y=[float(chart_df["price"].iloc[-1])] + forecast_prices,
    mode="lines+markers", name="Forecast",
    line=dict(color="#fd79a8", width=2.5, dash="dot"),
    marker=dict(size=9, color="#fd79a8", symbol="diamond",
                line=dict(color="white", width=1.5)),
    hovertemplate="<b>%{x|%b %d %Y}</b><br>Forecast: $%{y:.3f}/gal<extra></extra>",
))

fig.add_vline(
    x=datetime.now().strftime("%Y-%m-%d"),
    line_dash="dash", line_color="rgba(255,255,255,.25)",
    annotation_text="Today",
    annotation_font_color="rgba(255,255,255,.45)",
)

fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#b0c8de", size=12),
    xaxis=dict(gridcolor="rgba(255,255,255,.05)", tickformat="%b %d",
               tickfont=dict(color="#5d7d90")),
    yaxis=dict(gridcolor="rgba(255,255,255,.05)", title="Price ($/gal)",
               titlefont=dict(color="#5d7d90"), tickprefix="$",
               tickfont=dict(color="#5d7d90")),
    legend=dict(bgcolor="rgba(0,0,0,.35)", bordercolor="rgba(255,255,255,.1)",
                borderwidth=1, font=dict(color="#b0c8de")),
    hovermode="x unified",
    margin=dict(l=10, r=10, t=20, b=10),
    height=370,
)
st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Day-by-day forecast mini-cards
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="sh">📅 Day-by-Day Forecast</div>', unsafe_allow_html=True)

fc_labels = ["Tomorrow"] + [
    (datetime.now() + timedelta(days=i + 1)).strftime("%A")
    for i in range(1, fc_days)
]

fc_cols = st.columns(fc_days)
for col, label, price in zip(fc_cols, fc_labels, forecast_prices):
    chg = price - today_price
    with col:
        st.markdown(
            f'<div class="fc-card">'
            f'<div class="fc-day">{label}</div>'
            f'<div class="fc-price {_cls(chg)}">${price:.2f}</div>'
            f'<div class="fc-chg">{_arrow(chg)} {chg:+.3f}/gal</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# City comparison table  +  choropleth map
# ─────────────────────────────────────────────────────────────────────────────
all_fc = _all_city_forecasts(crude)
all_fc["chg"]   = all_fc["tomorrow"] - all_fc["today"]
all_fc["trend"] = all_fc["chg"].apply(
    lambda x: "🔴 Rising" if x > 0.015 else ("🟢 Falling" if x < -0.015 else "🟡 Stable")
)

col_tbl, col_map = st.columns([1, 1])

with col_tbl:
    st.markdown('<div class="sh">🏙️ City Comparison</div>', unsafe_allow_html=True)
    display = all_fc[["city", "today", "tomorrow", "trend"]].copy()
    display = display.sort_values("today")
    display.columns = ["City", "Today ($/gal)", "Tomorrow ($/gal)", "Trend"]
    display["Today ($/gal)"]    = display["Today ($/gal)"].map("${:.3f}".format)
    display["Tomorrow ($/gal)"] = display["Tomorrow ($/gal)"].map("${:.3f}".format)
    st.dataframe(display.set_index("City"), use_container_width=True, height=370)

with col_map:
    if show_map:
        st.markdown('<div class="sh">🗺️ Gas Price Map</div>', unsafe_allow_html=True)
        map_fig = px.scatter_mapbox(
            all_fc, lat="lat", lon="lon",
            color="today", size="today",
            hover_name="city",
            hover_data={"today": ":.3f", "lat": False, "lon": False},
            color_continuous_scale=["#51cf66", "#ffd43b", "#ff6b6b"],
            size_max=28, zoom=3.0,
            mapbox_style="carto-darkmatter",
            labels={"today": "$/gal"},
        )
        map_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=0, b=0),
            height=370,
            coloraxis_colorbar=dict(
                tickprefix="$", title="$/gal",
                titlefont=dict(color="#b0c8de"),
                tickfont=dict(color="#b0c8de"),
            ),
        )
        st.plotly_chart(map_fig, use_container_width=True)
    else:
        st.markdown('<div class="sh">🗺️ Gas Price Map</div>', unsafe_allow_html=True)
        st.info("Enable 'Show price map' in the sidebar to see the map view.")


# ─────────────────────────────────────────────────────────────────────────────
# Historical trend (price vs crude oil overlay)
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("📊 Price vs. Crude Oil Correlation (last 90 days)"):
    oil_df = city_df.tail(90)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=oil_df["date"], y=oil_df["price"],
        name="Gas ($/gal)", yaxis="y1",
        line=dict(color="#74b9ff", width=2),
    ))
    fig2.add_trace(go.Scatter(
        x=oil_df["date"], y=oil_df["crude_oil"],
        name="WTI Crude ($/bbl)", yaxis="y2",
        line=dict(color="#ffeaa7", width=2, dash="dot"),
    ))
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#b0c8de"),
        yaxis=dict(title="Gas ($/gal)", tickprefix="$",
                   gridcolor="rgba(255,255,255,.05)"),
        yaxis2=dict(title="WTI ($/bbl)", tickprefix="$",
                    overlaying="y", side="right",
                    gridcolor="rgba(255,255,255,.03)"),
        legend=dict(bgcolor="rgba(0,0,0,.35)"),
        hovermode="x unified",
        height=300, margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Model info
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("🧠 Model Details & Feature Importance"):
    fi = predictor.get_feature_importance()
    if not fi.empty:
        fi_fig = go.Figure(go.Bar(
            x=fi.head(12)["importance"],
            y=fi.head(12)["feature"],
            orientation="h",
            marker=dict(color="#74b9ff",
                        line=dict(color="rgba(255,255,255,.1)", width=0.5)),
        ))
        fi_fig.update_layout(
            title="Top-12 Feature Importances",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#b0c8de"),
            xaxis=dict(gridcolor="rgba(255,255,255,.05)"),
            yaxis=dict(autorange="reversed"),
            height=320, margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fi_fig, use_container_width=True)

    st.markdown(
        """
**Model:** Gradient Boosting Regressor (`sklearn`)  
**Features:** Lag prices (1 / 2 / 7 / 14 / 30 days), rolling averages (7 / 30 day),
WTI crude oil, cyclical date encoding (month & day-of-week sin/cos), city label encoding,
price momentum (1-day & 7-day diff)  
**Training data:** Generated sample data Jan 2022 – present  
**Forecast method:** Auto-regressive (each predicted day feeds as next lag)

> **For production use:** replace `data/prices.csv` with real data from
> [EIA API](https://www.eia.gov/opendata/), [GasBuddy API](https://www.gasbuddy.com/),
> or [Natural Resources Canada](https://natural-resources.canada.ca/).
"""
    )


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#2d4a5c;font-size:.78em'>"
    "GasWatch · Predictions are estimates only · Not financial advice · "
    "Data: sample generation + live WTI via Yahoo Finance"
    "</div>",
    unsafe_allow_html=True,
)
