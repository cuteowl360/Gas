"""
predictor.py – GasWatch
────────────────────────
Gradient Boosting Regressor for next-day gas price prediction.
Supports multi-day auto-regressive forecasting.
"""

import os
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from src.preprocessor import create_features, encode_city, FEATURE_COLS

_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(_ROOT, "models", "gas_price_model.pkl")


class GasPricePredictor:
    """Wraps a GradientBoostingRegressor with train / predict / persist helpers."""

    def __init__(self):
        self.model: GradientBoostingRegressor | None = None
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, df: pd.DataFrame) -> dict:
        """Train on a historical price DataFrame. Returns evaluation metrics."""
        feat_df = create_features(df).dropna(subset=FEATURE_COLS + ["price"])

        X = feat_df[FEATURE_COLS]
        y = feat_df["price"]

        # Chronological split (no shuffle)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.12, shuffle=False
        )

        self.model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.08,
            max_depth=5,
            subsample=0.8,
            min_samples_leaf=5,
            random_state=42,
        )
        self.model.fit(X_train, y_train)

        mae = mean_absolute_error(y_test, self.model.predict(X_test))
        return {"mae": round(mae, 4), "n_train": len(X_train), "n_test": len(X_test)}

    # ------------------------------------------------------------------
    # Single prediction
    # ------------------------------------------------------------------
    def predict(self, features: dict) -> float:
        """Predict from a feature dict. Returns price as float."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        X = pd.DataFrame([features])[FEATURE_COLS]
        return round(float(self.model.predict(X)[0]), 3)

    # ------------------------------------------------------------------
    # Multi-day forecast (auto-regressive)
    # ------------------------------------------------------------------
    def forecast_days(
        self,
        city: str,
        city_history: pd.DataFrame,
        crude_price: float,
        start_date: datetime,
        days: int = 5,
    ) -> list[float]:
        """
        Auto-regressively forecast `days` days ahead.

        Parameters
        ----------
        city         : city name (for encoding)
        city_history : sorted DataFrame with at least 'date' and 'price' columns
        crude_price  : WTI price assumed constant over the forecast window
        start_date   : the date representing "today" (day 0)
        days         : how many days ahead to forecast

        Returns
        -------
        List of predicted prices (length == days)
        """
        if self.model is None:
            raise ValueError("Model not trained.")

        city_code   = encode_city(city)
        hist_prices = city_history.sort_values("date")["price"].tolist()

        predictions: list[float] = []

        for step in range(days):
            forecast_date = start_date + timedelta(days=step + 1)
            n = len(hist_prices)

            def lag(k: int) -> float:
                idx = n - k
                return hist_prices[idx] if idx >= 0 else hist_prices[0]

            rolling_7  = float(np.mean(hist_prices[max(0, n - 7):n]))  if n >= 1 else lag(1)
            rolling_30 = float(np.mean(hist_prices[max(0, n - 30):n])) if n >= 1 else lag(1)

            dow   = forecast_date.weekday()
            month = forecast_date.month
            doy   = forecast_date.timetuple().tm_yday

            features = {
                "lag_1":        lag(1),
                "lag_2":        lag(2),
                "lag_7":        lag(7),
                "lag_14":       lag(14),
                "lag_30":       lag(30),
                "rolling_7":    rolling_7,
                "rolling_30":   rolling_30,
                "crude_oil":    crude_price,
                "day_of_week":  dow,
                "month":        month,
                "day_of_year":  doy,
                "month_sin":    np.sin(2 * np.pi * month / 12),
                "month_cos":    np.cos(2 * np.pi * month / 12),
                "dow_sin":      np.sin(2 * np.pi * dow   / 7),
                "dow_cos":      np.cos(2 * np.pi * dow   / 7),
                "city_code":    city_code,
                "price_diff_1": lag(1) - lag(2),
                "price_diff_7": lag(1) - lag(7),
            }

            pred = self.predict(features)
            pred = float(np.clip(pred, 1.50, 9.00))
            predictions.append(round(pred, 3))
            hist_prices.append(pred)   # feed prediction back as next lag

        return predictions

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------
    def get_feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            return pd.DataFrame()
        return (
            pd.DataFrame({"feature": FEATURE_COLS,
                          "importance": self.model.feature_importances_})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self) -> None:
        joblib.dump(self.model, MODEL_PATH)

    def load(self) -> bool:
        if os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODEL_PATH)
            return True
        return False
