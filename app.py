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
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_collector import DataCollector, GAS_TYPES
from src.predictor import GasPricePredictor

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

    st.markdown("---")
    st.markdown("**✏️ Manual Price Override**")
    use_manual = st.checkbox("Enter today's price manually")
    manual_price = 0.0
    if use_manual:
        manual_price = st.number_input("Today's price ($/gal)", 0.50, 9.99,
                                       3.50, 0.01, "%.2f")

    st.markdown("---")
    st.markdown("**⚙️ Settings**")
    fc_days   = st.slider("Forecast days", 1, 7, 5)
    hist_days = st.slider("History window (days)", 7, 365, 60)

    st.markdown("---")
    st.markdown("**🔔 Price Alert**")
    alert_on    = st.checkbox("Notify if price exceeds")
    alert_limit = st.number_input("Threshold ($/gal)", 1.0, 9.0,
                                  4.50, 0.10, "%.2f", disabled=not alert_on)

    st.markdown("---")
    st.markdown('<div class="info-pill">💡 Live <b>WTI crude oil</b> via Yahoo Finance.'
                ' Gas prices use generated sample data — replace <code>data/prices.csv</code>'
                ' with EIA or GasBuddy data for production.</div>', unsafe_allow_html=True)


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

# Canadian city: compute CAD/L equivalent for display
_is_canadian = (city_df["country"].iloc[-1] == "CA") if "country" in city_df.columns else False
_CAD_PER_USD  = 1.36
_LITRES_PER_GAL = 3.785
def _to_cad_litre(usd_gal): return usd_gal * _CAD_PER_USD / _LITRES_PER_GAL

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
    if today_p >= alert_limit:
        st.markdown(
            f'<div class="alert-red">🚨 <b>Alert!</b> Current {gas_type} price '
            f'<b>${today_p:.2f}/gal</b> exceeds <b>${alert_limit:.2f}/gal</b> in {selected}.</div>',
            unsafe_allow_html=True)
    elif tomorrow_p >= alert_limit:
        st.markdown(
            f'<div class="alert-red">⚠️ Tomorrow\'s predicted price '
            f'<b>${tomorrow_p:.2f}/gal</b> may exceed <b>${alert_limit:.2f}/gal</b>.</div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="alert-green">✅ Price <b>${today_p:.2f}/gal</b> is below your '
            f'<b>${alert_limit:.2f}/gal</b> threshold.</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# KPI cards
# ─────────────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)

# For Canadian cities show "CAD/L" label; for US show "per gallon"
if _is_canadian:
    _today_unit = f"{_to_cad_litre(today_p):.3f} CAD/L · " + gas_type
    _tmrw_unit  = f"{_to_cad_litre(tomorrow_p):.3f} CAD/L forecast"
else:
    _today_unit = "per gallon · " + gas_type
    _tmrw_unit  = "ML · Random Forest"

for col, label, val, sub, unit, cls in [
    (k1, "Today's Price",    f"${today_p:.2f}",
     f"{_arrow(daily_chg)} {daily_chg:+.3f} vs yesterday", _today_unit, _col(daily_chg)),
    (k2, "Tomorrow Forecast",f"${tomorrow_p:.2f}",
     f"{_arrow(tomorrow_chg)} {tomorrow_chg:+.3f} projected", _tmrw_unit, _col(tomorrow_chg)),
    (k3, "7-Day Change",     f"{_arrow(weekly_chg)} {abs(weekly_chg):.3f}",
     f"{weekly_chg:+.3f} vs last week", "Weekly trend", _col(weekly_chg)),
    (k4, "WTI Crude Oil",    f"${crude:.2f}",
     "per barrel", "West Texas Intermediate", "c-blue"),
    (k5, "US Avg Today",     f"${all_fc['today_typed'].mean():.2f}",
     f"across {len(all_fc)} cities", "US average · "+gas_type, "c-flat"),
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
    for idx, (col, lbl, price) in enumerate(zip(fc_cols, fc_labels, fc_prices)):
        chg      = price - today_p
        date_str = (datetime.now() + timedelta(days=idx+1)).strftime("%b %d")
        with col:
            st.markdown(
                f'<div class="fc-card">'
                f'<div class="fc-day">{lbl}</div>'
                f'<div class="fc-date">{date_str}</div>'
                f'<div class="fc-price {_col(chg)}">${price:.2f}</div>'
                f'<div class="fc-chg">{_arrow(chg)} {chg:+.3f}/gal</div>'
                f'</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # History + forecast chart
    st.markdown('<div class="sh">📈 Price History & Forecast</div>', unsafe_allow_html=True)

    chart_df = city_df.tail(hist_days).copy()
    chart_df["price_typed"] = chart_df["price"] * multiplier
    fc_dates = [pd.Timestamp(datetime.now().date() + timedelta(days=i+1)) for i in range(fc_days)]
    upper    = [p + 0.05 + i*0.009 for i, p in enumerate(fc_prices)]
    lower    = [p - 0.05 - i*0.009 for i, p in enumerate(fc_prices)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=chart_df["date"], y=chart_df["price_typed"],
        mode="lines", name="Historical",
        line=dict(color="#74b9ff", width=2.5),
        fill="tozeroy", fillcolor="rgba(116,185,255,.06)",
        hovertemplate="<b>%{x|%b %d %Y}</b><br>$%{y:.3f}/gal<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=fc_dates + fc_dates[::-1], y=upper + lower[::-1],
        fill="toself", fillcolor="rgba(253,121,168,.09)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Confidence Band", hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=[chart_df["date"].iloc[-1]] + fc_dates,
        y=[float(chart_df["price_typed"].iloc[-1])] + fc_prices,
        mode="lines+markers", name="Forecast",
        line=dict(color="#fd79a8", width=2.5, dash="dot"),
        marker=dict(size=9, color="#fd79a8", symbol="diamond",
                    line=dict(color="white", width=1.5)),
        hovertemplate="<b>%{x|%b %d %Y}</b><br>Forecast: $%{y:.3f}/gal<extra></extra>"))
    fig.add_vline(x=datetime.now().strftime("%Y-%m-%d"),
                  line_dash="dash", line_color="rgba(255,255,255,.18)",
                  annotation_text="Today",
                  annotation_font_color="rgba(255,255,255,.35)")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8ab4cc", size=12),
        xaxis=dict(gridcolor="rgba(255,255,255,.04)", tickformat="%b %d",
                   tickfont=dict(color="#4a6a7c")),
        yaxis=dict(gridcolor="rgba(255,255,255,.04)", title="Price ($/gal)",
                   titlefont=dict(color="#4a6a7c"), tickprefix="$",
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
    monthly_gb["avg"]   = monthly_gb["price"] * multiplier
    bar_fig = go.Figure(go.Bar(
        x=monthly_gb["label"], y=monthly_gb["avg"],
        marker=dict(color=monthly_gb["avg"],
                    colorscale=[[0,"#51cf66"],[0.5,"#ffd43b"],[1,"#ff6b6b"]],
                    showscale=False),
        hovertemplate="<b>%{x}</b><br>Avg: $%{y:.3f}/gal<extra></extra>"))
    bar_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8ab4cc"),
        xaxis=dict(tickfont=dict(color="#4a6a7c"), tickangle=-45,
                   gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(gridcolor="rgba(255,255,255,.04)", tickprefix="$",
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
# TAB 2 – LIVE MAP
# ══════════════════════════════════════════════════════════════════════════════
with tab_map:
    mc1, mc2 = st.columns([2,1])
    with mc1:
        map_view = st.radio("Show", ["Today's Prices","Tomorrow's Forecast","Price Change"],
                            horizontal=True, label_visibility="collapsed")
    with mc2:
        map_mode = st.radio("Style", ["Bubble Map","US State Heatmap"],
                            horizontal=True, label_visibility="collapsed")

    if map_view == "Today's Prices":
        mcol, mtitle = "today_typed", f"Today's {gas_type} Prices ($/gal)"
        cscale = [[0,"#51cf66"],[0.45,"#ffd43b"],[1,"#ff6b6b"]]
    elif map_view == "Tomorrow's Forecast":
        mcol, mtitle = "tomorrow_typed", f"Tomorrow's Predicted {gas_type} ($/gal)"
        cscale = [[0,"#51cf66"],[0.45,"#ffd43b"],[1,"#fd79a8"]]
    else:
        mcol, mtitle = "chg", "Predicted Price Change ($/gal)"
        cscale = [[0,"#51cf66"],[0.5,"#ffeaa7"],[1,"#ff6b6b"]]

    if map_mode == "Bubble Map":
        plot_df = all_fc.copy()
        plot_df["bubble_size"] = plot_df[mcol].clip(lower=0.01)
        map_fig = px.scatter_map(
            plot_df, lat="lat", lon="lon",
            color=mcol, size="bubble_size",
            hover_name="city",
            hover_data={"today_typed":":.3f","tomorrow_typed":":.3f",
                        "chg":":.3f","state":True,
                        "lat":False,"lon":False,"bubble_size":False},
            color_continuous_scale=cscale,
            size_max=40, zoom=2.4,
            center={"lat":47.0,"lon":-93.0},
            map_style="open-street-map",
            labels={"today_typed":"Today","tomorrow_typed":"Tomorrow",
                    "chg":"Δ Price","state":"State"})
        map_fig.update_traces(
            hovertemplate=(
                "<b>%{hovertext}</b><br>"
                "Today: $%{customdata[0]:.3f}<br>"
                "Tomorrow: $%{customdata[1]:.3f}<br>"
                "Change: %{customdata[2]:+.3f}<br>"
                "State: %{customdata[3]}<extra></extra>"),
            marker=dict(opacity=0.88))
        map_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            title=mtitle, title_font=dict(color="#8ab4cc", size=14),
            margin=dict(l=0,r=0,t=36,b=0), height=560,
            coloraxis_colorbar=dict(
                tickprefix="$", title="$/gal",
                titlefont=dict(color="#8ab4cc"),
                tickfont=dict(color="#8ab4cc")))
    else:
        state_avg = (all_fc.groupby("state")[mcol]
                     .mean().reset_index().rename(columns={mcol:"value"}))
        map_fig = px.choropleth(
            state_avg, locations="state", locationmode="USA-states",
            color="value", scope="usa",
            color_continuous_scale=cscale,
            title=mtitle,
            labels={"value":"$/gal"},
            hover_data={"value":":.3f"})
        map_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            geo=dict(bgcolor="rgba(0,0,0,0)",
                     landcolor="#1a2b3d", subunitcolor="#2d4a5c",
                     showlakes=True, lakecolor="#0d1e33"),
            title_font=dict(color="#8ab4cc", size=14),
            margin=dict(l=0,r=0,t=36,b=0), height=560,
            coloraxis_colorbar=dict(
                tickprefix="$", title="$/gal",
                titlefont=dict(color="#8ab4cc"),
                tickfont=dict(color="#8ab4cc")))

    st.plotly_chart(map_fig, use_container_width=True)

    # Ranked table
    rank_df = all_fc[["city","state","today_typed","tomorrow_typed","chg","trend"]].copy()
    rank_df = rank_df.sort_values("today_typed")
    rank_df.columns = ["City","State","Today ($/gal)","Tomorrow ($/gal)","Δ","Trend"]
    rank_df["Today ($/gal)"]    = rank_df["Today ($/gal)"].map("${:.3f}".format)
    rank_df["Tomorrow ($/gal)"] = rank_df["Tomorrow ($/gal)"].map("${:.3f}".format)
    rank_df["Δ"] = rank_df["Δ"].map("{:+.3f}".format)
    st.markdown('<div class="sh">📋 All Cities — Ranked Cheapest to Most Expensive</div>',
                unsafe_allow_html=True)
    st.dataframe(rank_df.set_index("City"), use_container_width=True, height=360)


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
        oil_df["price_typed"] = oil_df["price"] * multiplier
        fig_oil = go.Figure()
        fig_oil.add_trace(go.Scatter(
            x=oil_df["date"], y=oil_df["price_typed"],
            name="Gas ($/gal)", yaxis="y1",
            line=dict(color="#74b9ff", width=2),
            hovertemplate="%{x|%b %d}: $%{y:.3f}<extra>Gas</extra>"))
        fig_oil.add_trace(go.Scatter(
            x=oil_df["date"], y=oil_df["crude_oil"],
            name="WTI ($/bbl)", yaxis="y2",
            line=dict(color="#ffeaa7", width=2, dash="dot"),
            hovertemplate="%{x|%b %d}: $%{y:.2f}<extra>WTI</extra>"))
        fig_oil.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8ab4cc"),
            yaxis=dict(title="Gas ($/gal)", tickprefix="$",
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
        dow_df["price_typed"] = dow_df["price"] * multiplier
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
            yaxis=dict(tickprefix="$", gridcolor="rgba(255,255,255,.04)",
                       tickfont=dict(color="#4a6a7c")),
            xaxis=dict(tickfont=dict(color="#4a6a7c")),
            margin=dict(l=10,r=10,t=10,b=10), height=300)
        st.plotly_chart(fig_dow, use_container_width=True)

    # Year-over-year
    st.markdown('<div class="sh">📅 Year-over-Year Comparison</div>', unsafe_allow_html=True)
    yoy_df = city_df.copy()
    yoy_df["price_typed"] = yoy_df["price"] * multiplier
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
        yaxis=dict(tickprefix="$", gridcolor="rgba(255,255,255,.04)",
                   title="Price ($/gal)", tickfont=dict(color="#4a6a7c")),
        legend=dict(bgcolor="rgba(0,0,0,.4)", title="Year"),
        hovermode="x unified",
        margin=dict(l=10,r=10,t=10,b=10), height=320)
    st.plotly_chart(yoy_fig, use_container_width=True)

    # Seasonal box
    st.markdown('<div class="sh">📦 Price Distribution by Month</div>', unsafe_allow_html=True)
    sea_df = city_df.copy()
    sea_df["price_typed"]  = sea_df["price"] * multiplier
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
        yaxis=dict(tickprefix="$", gridcolor="rgba(255,255,255,.04)",
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
        tank_size    = st.number_input("Tank size (gallons)", 5.0, 50.0, 15.0, 0.5)
        tank_level   = st.slider("Current tank level (%)", 0, 100, 25)
        gallons_need = tank_size * (1 - tank_level/100)
    with cc2:
        use_tmrw   = st.checkbox("Use tomorrow's predicted price")
        price_used = tomorrow_p if use_tmrw else today_p
        lbl_used   = "Tomorrow's forecast" if use_tmrw else "Today's price"
        total_cost = gallons_need * price_used
        savings    = gallons_need * (today_p - tomorrow_p)

    st.markdown(
        f'<div class="calc-result">'
        f'<div class="kpi-label">{lbl_used}: ${price_used:.3f}/gal · {gallons_need:.1f} gal needed</div>'
        f'<div class="calc-big">${total_cost:.2f}</div>'
        f'<div style="color:#5d7d90;font-size:.9em;margin-top:8px">'
        f'Fill-up cost · <b>{selected}</b> · <b>{gas_type}</b></div>'
        f'</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if abs(savings) > 0.01:
        if savings > 0:
            st.markdown(
                f'<div class="alert-green">💡 Waiting until tomorrow could save '
                f'<b>${savings:.2f}</b> (price predicted ▼ {tomorrow_chg:+.3f}/gal).</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="alert-red">⚠️ Fill up today — tomorrow price rises '
                f'{tomorrow_chg:+.3f}/gal (extra cost ${abs(savings):.2f}).</div>',
                unsafe_allow_html=True)

    # Fill-up cost across all cities
    st.markdown('<div class="sh">🗺️ Same Fill-Up Cost — All Cities</div>',
                unsafe_allow_html=True)
    fill_df = all_fc[["city","today_typed"]].copy()
    fill_df["fill_cost"] = fill_df["today_typed"] * gallons_need
    fill_df = fill_df.sort_values("fill_cost")
    fill_fig = go.Figure(go.Bar(
        x=fill_df["city"], y=fill_df["fill_cost"],
        marker=dict(color=fill_df["fill_cost"],
                    colorscale=[[0,"#51cf66"],[0.5,"#ffd43b"],[1,"#ff6b6b"]],
                    showscale=False),
        text=fill_df["fill_cost"].map("${:.2f}".format),
        textposition="outside",
        textfont=dict(color="#8ab4cc", size=9),
        hovertemplate="<b>%{x}</b><br>Fill-up: $%{y:.2f}<extra></extra>"))
    fill_fig.add_hline(
        y=total_cost, line_dash="dash", line_color="rgba(253,121,168,.6)",
        annotation_text=f"{selected} (${total_cost:.2f})",
        annotation_font_color="rgba(253,121,168,.8)")
    fill_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8ab4cc"),
        yaxis=dict(tickprefix="$", gridcolor="rgba(255,255,255,.04)",
                   title="Fill-up cost ($)", tickfont=dict(color="#4a6a7c")),
        xaxis=dict(tickfont=dict(color="#4a6a7c"), tickangle=-45),
        margin=dict(l=10,r=10,t=10,b=130), height=420)
    st.plotly_chart(fill_fig, use_container_width=True)

    # Monthly budget
    st.markdown('<div class="sh">📅 Monthly Fuel Budget</div>', unsafe_allow_html=True)
    mb1, mb2 = st.columns(2)
    with mb1:
        fills_mo = st.slider("Fill-ups per month", 1, 12, 4)
    with mb2:
        monthly_spend = fills_mo * total_cost
        st.markdown(
            f'<div class="calc-result" style="padding:16px">'
            f'<div class="kpi-label">{fills_mo} fill-ups/month at ${price_used:.3f}/gal</div>'
            f'<div style="font-size:2.2em;font-weight:800;color:#74b9ff">${monthly_spend:.2f}</div>'
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

