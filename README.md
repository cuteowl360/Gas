# ⛽ GasWatch — Gas Price Prediction App

A **weather-app-style** gas price forecaster powered by machine learning.  
Select a US city, see today's gas price, and get an ML-predicted price for the next 1–7 days — all in a sleek dark dashboard.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-GBR-orange)

---

## Features

| Feature | Details |
|---|---|
| **Today's price** | From sample data (or enter manually) |
| **Tomorrow's forecast** | Gradient Boosting Regressor, auto-regressive |
| **Multi-day forecast** | Up to 7 days ahead with confidence band |
| **Historical chart** | Plotly line chart, configurable window |
| **WTI crude overlay** | Live price via yfinance |
| **City comparison table** | All 15 cities, today vs tomorrow |
| **Interactive map** | Scatter mapbox colored by price |
| **Price alerts** | Configurable $/gal threshold |
| **Feature importance** | Model interpretability chart |

---

## Quickstart (VS Code / local)

### 1. Clone and enter the repo
```bash
git clone https://github.com/cuteowl360/Gas.git
cd Gas
git checkout Yifan
```

### 2. Create a virtual environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

The browser opens at **http://localhost:8501**.

> **First run:** The app automatically generates ~3 years of sample price data
> (`data/prices.csv`) and trains the model (`models/gas_price_model.pkl`).
> This takes ~5–15 seconds. Subsequent runs load from cache instantly.

---

## Project Structure

```
Gas/
├── app.py                  ← Main Streamlit app (run this)
├── requirements.txt        ← Python dependencies
├── .streamlit/
│   └── config.toml         ← Dark theme + server config
├── src/
│   ├── data_collector.py   ← WTI fetcher + sample data generator
│   ├── preprocessor.py     ← Feature engineering (lags, rolling, cyclical)
│   └── predictor.py        ← GBR model: train / predict / forecast / persist
├── data/
│   └── prices.csv          ← Generated on first run (15 cities × ~1500 days)
└── models/
    └── gas_price_model.pkl ← Saved model (generated on first run)
```

---

## Machine Learning Model

**Algorithm:** `GradientBoostingRegressor` (scikit-learn)

**Features used:**
- Lag prices: 1, 2, 7, 14, 30 days
- Rolling averages: 7-day, 30-day
- WTI crude oil price ($/barrel)
- Cyclical date encoding: month sin/cos, day-of-week sin/cos
- Day of week, month, day of year
- City label encoding
- Price momentum (1-day diff, 7-day diff)

**Forecast method:** Auto-regressive — each predicted price is fed back as the next day's lag_1.

---

## Using Real Data (Production)

The app ships with generated sample data.  
To use real prices, swap `data/prices.csv` with data from:

| Source | Coverage | Notes |
|---|---|---|
| [EIA Retail Gasoline Prices API](https://www.eia.gov/opendata/) | USA (regional) | Free API key |
| [GasBuddy](https://www.gasbuddy.com/) | USA / Canada | Commercial API |
| [Natural Resources Canada](https://natural-resources.canada.ca/energy/fuel-prices/4593) | Canada | CSV downloads |

The CSV must contain columns: `date, city, price, crude_oil, lat, lon`.

---

## Deploy to Streamlit Cloud

1. Fork this repo on GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Select your fork, branch `Yifan`, file `app.py`
4. Click **Deploy**

---

## License

MIT
