"""Deterministic forward-forecast generator for the HTML report.

Loads the latest trained GBM models, fetches the most recent water-level and
weather data, builds the same feature vector used during training, and emits
forward-looking predictions for every configured horizon. Results are written to
output/models/forward_predictions.csv and output/models/forward_predictions.json
so the HTML report can plot a "latest measured + future predicted" chart.
"""

import json
import os
import pickle
from datetime import datetime, timedelta
from io import StringIO
from typing import Any, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

from thesiscrew.tools.pegelonline_tool import GetMeasurementsCSVTool
from thesiscrew.tools.dataset_tool import (
    KORNEUBURG_PEGEL_UUID,
    _open_meteo_past_url,
)

OUTPUT_DIR = os.environ.get("PEGELHUB_OUTPUT_DIR", "output")
DATA_DIR = os.environ.get("PEGELHUB_DATA_DIR", os.path.join(OUTPUT_DIR, "data"))
MODELS_DIR = os.environ.get("PEGELHUB_MODELS_DIR", os.path.join(OUTPUT_DIR, "models"))


class BuildForwardForecastsInput(BaseModel):
    recent_days: int = Field(
        default=14,
        description="Number of recent days to fetch for lag/rolling features.",
    )
    forecast_days: int = Field(
        default=8,
        description="Number of future days to fetch weather forecast for.",
    )


class BuildForwardForecastsTool(BaseTool):
    name: str = "build_forward_forecasts"
    description: str = (
        "Generate forward-looking water-level forecasts for all configured horizons "
        "using the latest trained GBM models. Fetches recent measurements and weather "
        "forecasts, builds an inference feature matrix, predicts each horizon, and "
        "writes output/models/forward_predictions.csv plus a JSON summary. "
        "Call this before building the HTML report so the report can show future forecasts."
    )
    args_schema: type[BaseModel] = BuildForwardForecastsInput

    def _run(
        self,
        recent_days: int = 14,
        forecast_days: int = 8,
    ) -> str:
        # ------------------------------------------------------------------
        # 1. Load project inputs
        # ------------------------------------------------------------------
        input_path = os.path.join(os.path.dirname(OUTPUT_DIR), "input", "research_area.json")
        if not os.path.exists(input_path):
            input_path = "input/research_area.json"
        setup = self._load_setup(input_path)

        # ------------------------------------------------------------------
        # 2. Fetch recent water level (Pegelonline)
        # ------------------------------------------------------------------
        end = datetime.utcnow()
        start = end - timedelta(days=recent_days)
        start_str = start.strftime("%Y-%m-%dT%H:%M:%S%z")
        end_str = end.strftime("%Y-%m-%dT%H:%M:%S%z")

        csv_text = GetMeasurementsCSVTool()._run(
            uuid=KORNEUBURG_PEGEL_UUID,
            timeseries="W",
            start=start_str,
            end=end_str,
        )
        if csv_text.startswith("Error"):
            return json.dumps({"error": f"Pegelonline fetch failed: {csv_text}"})

        df_w = pd.read_csv(StringIO(csv_text), sep=";")
        if df_w.shape[1] < 2:
            return json.dumps({"error": f"Unexpected Pegelonline CSV shape: {df_w.shape}"})
        df_w = df_w.iloc[:, :2].copy()
        df_w.columns = ["timestamp", "water_level"]
        df_w["timestamp"] = pd.to_datetime(df_w["timestamp"])
        df_w["water_level"] = pd.to_numeric(df_w["water_level"], errors="coerce")
        df_w = df_w.set_index("timestamp").resample("h").last()
        df_w["water_level"] = df_w["water_level"].ffill(limit=3)
        df_w = df_w.reset_index()

        # ------------------------------------------------------------------
        # 3. Fetch weather: recent + forecast
        # ------------------------------------------------------------------
        import ssl
        from urllib.request import urlopen, Request

        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        # Recent/historical weather to cover rolling windows.
        past_url = _open_meteo_past_url(
            latitude=48.38,
            longitude=16.34,
            past_days=recent_days,
        )
        try:
            req = Request(past_url, headers={"Accept": "application/json"})
            with urlopen(req, timeout=60, context=ctx) as resp:
                past_weather = json.loads(resp.read().decode())
        except Exception as e:
            return json.dumps({"error": f"Open-Meteo past weather failed: {e}"})

        # Future weather for forecast horizons.
        future_url = (
            "https://api.open-meteo.com/v1/forecast?"
            f"latitude=48.38&longitude=16.34"
            f"&hourly=temperature_2m,precipitation,rain,snowfall,snow_depth,wind_speed_10m"
            f"&forecast_days={forecast_days}"
            "&timezone=Europe%2FVienna"
        )
        try:
            req = Request(future_url, headers={"Accept": "application/json"})
            with urlopen(req, timeout=60, context=ctx) as resp:
                future_weather = json.loads(resp.read().decode())
        except Exception as e:
            return json.dumps({"error": f"Open-Meteo forecast weather failed: {e}"})

        def _weather_df(weather: dict) -> pd.DataFrame:
            hourly = weather.get("hourly", {})
            if not hourly.get("time"):
                return pd.DataFrame()
            return pd.DataFrame({
                "timestamp": pd.to_datetime(hourly["time"]),
                "temperature": hourly.get("temperature_2m"),
                "precipitation": hourly.get("precipitation"),
                "rain": hourly.get("rain"),
                "snowfall": hourly.get("snowfall"),
                "snow_depth": hourly.get("snow_depth"),
                "wind_speed": hourly.get("wind_speed_10m"),
            })

        df_past_w = _weather_df(past_weather)
        df_future_w = _weather_df(future_weather)
        df_weather = pd.concat([df_past_w, df_future_w], ignore_index=True)
        df_weather = df_weather.drop_duplicates(subset=["timestamp"], keep="last")

        # ------------------------------------------------------------------
        # 4. Merge and build inference feature matrix
        # ------------------------------------------------------------------
        df = pd.merge(df_w, df_weather, on="timestamp", how="outer").sort_values("timestamp")
        numeric_cols = [c for c in df.columns if c != "timestamp"]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        df[numeric_cols] = df[numeric_cols].ffill(limit=3)
        df = df.dropna(subset=["water_level"])

        if len(df) < 169:
            return json.dumps({
                "error": f"Not enough recent data for lag-168 features: {len(df)} rows",
                "rows": len(df),
            })

        features_df = self._build_inference_features(df)
        if features_df.empty:
            return json.dumps({"error": "Could not build inference features"})

        last_row = features_df.iloc[[-1]].copy()
        last_timestamp = last_row["timestamp"].iloc[0]

        # ------------------------------------------------------------------
        # 5. Predict each horizon with the latest trained model
        # ------------------------------------------------------------------
        os.makedirs(MODELS_DIR, exist_ok=True)
        feature_cols = [c for c in features_df.columns if c not in ("timestamp",)]
        X = last_row[feature_cols].values

        predictions = []
        for h in setup["horizons"]:
            model_path = os.path.join(MODELS_DIR, f"gbm_model_h{h}.pkl")
            if not os.path.exists(model_path):
                predictions.append({
                    "horizon": h,
                    "timestamp": (last_timestamp + timedelta(hours=h)).isoformat(),
                    "predicted": None,
                    "error": f"Model not found: {model_path}",
                })
                continue
            try:
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                yhat = model.predict(X)[0]
                predictions.append({
                    "horizon": h,
                    "timestamp": (last_timestamp + timedelta(hours=h)).isoformat(),
                    "predicted": round(float(yhat), 2),
                })
            except Exception as e:
                predictions.append({
                    "horizon": h,
                    "timestamp": (last_timestamp + timedelta(hours=h)).isoformat(),
                    "predicted": None,
                    "error": str(e),
                })

        # ------------------------------------------------------------------
        # 6. Write outputs: forward predictions + recent actuals for charting
        # ------------------------------------------------------------------
        forward_df = pd.DataFrame(predictions)
        forward_path = os.path.join(MODELS_DIR, "forward_predictions.csv")
        forward_df.to_csv(forward_path, index=False)

        recent = df[["timestamp", "water_level"]].tail(48).copy()
        recent.columns = ["timestamp", "actual"]
        recent["timestamp"] = recent["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        recent_path = os.path.join(MODELS_DIR, "forward_predictions_recent.csv")
        recent.to_csv(recent_path, index=False)

        json_summary = {
            "run_timestamp": last_timestamp.isoformat(),
            "station": setup["station"],
            "target": setup["target"],
            "predictions": predictions,
            "recent_actuals_file": recent_path,
            "forward_predictions_file": forward_path,
            "feature_row_columns": feature_cols,
        }
        json_path = os.path.join(MODELS_DIR, "forward_predictions.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_summary, f, indent=2, ensure_ascii=False, default=str)

        return json.dumps({
            "forward_predictions": forward_path,
            "recent_actuals": recent_path,
            "json_summary": json_path,
            "run_timestamp": last_timestamp.isoformat(),
            "predictions": predictions,
        }, indent=2, default=str)

    @staticmethod
    def _load_setup(input_path: str) -> dict[str, Any]:
        setup = {
            "station": "Korneuburg",
            "river": "Donau",
            "country": "Austria",
            "latitude": 48.345,
            "longitude": 16.337,
            "target": "water_level_cm",
            "horizons": [1, 6, 12, 24, 48, 72, 168],
        }
        if not os.path.exists(input_path):
            return setup
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            setup["station"] = data.get("primary_station", setup["station"])
            setup["river"] = data.get("river", setup["river"])
            setup["country"] = data.get("country", setup["country"])
            setup["latitude"] = data.get("latitude", setup["latitude"])
            setup["longitude"] = data.get("longitude", setup["longitude"])
            setup["target"] = data.get("target_variable", setup["target"])
            setup["horizons"] = data.get("forecast_horizons_hours", setup["horizons"])
        except Exception:
            pass
        return setup

    @staticmethod
    def _build_inference_features(df: pd.DataFrame) -> pd.DataFrame:
        """Build the same features as BuildFeatureMatrixTool but without targets."""
        target_col = "water_level"
        if target_col not in df.columns:
            return pd.DataFrame()

        # Endogenous / lagged features.
        for lag in [1, 3, 6, 12, 24, 168]:
            df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)

        for window in [3, 6, 24]:
            df[f"{target_col}_rolling_mean_{window}h"] = (
                df[target_col].rolling(window=window, min_periods=1).mean()
            )
        for window in [6, 24]:
            df[f"{target_col}_rolling_std_{window}h"] = (
                df[target_col].rolling(window=window, min_periods=1).std()
            )

        df[f"{target_col}_roc"] = df[target_col].diff()
        df[f"{target_col}_accel"] = df[f"{target_col}_roc"].diff()

        # Calendar features.
        ts = df["timestamp"].dt
        df["hour_sin"] = np.sin(2 * np.pi * ts.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * ts.hour / 24)
        df["dow_sin"] = np.sin(2 * np.pi * ts.dayofweek / 7)
        df["dow_cos"] = np.cos(2 * np.pi * ts.dayofweek / 7)
        df["doy_sin"] = np.sin(2 * np.pi * ts.dayofyear / 365.25)
        df["doy_cos"] = np.cos(2 * np.pi * ts.dayofyear / 365.25)
        df["is_weekend"] = (ts.dayofweek >= 5).astype(int)
        at_holidays = {
            "01-01", "01-06", "05-01", "08-15", "10-26", "11-01", "12-08",
            "12-25", "12-26",
        }
        df["is_holiday"] = df["timestamp"].dt.strftime("%m-%d").isin(at_holidays).astype(int)

        # Meteorological features.
        if "precipitation" in df.columns:
            for window in [3, 6, 12, 24, 48, 72]:
                df[f"precip_acc_{window}h"] = df["precipitation"].rolling(window=window, min_periods=1).sum()
        if "temperature" in df.columns:
            df["temp_rolling_24h"] = df["temperature"].rolling(window=24, min_periods=1).mean()
        if "snow_depth" in df.columns:
            df["snow_depth_lag_24"] = df["snow_depth"].shift(24)

        # Keep only rows with all required lags.
        df = df.dropna(subset=[f"{target_col}_lag_168"]).copy()
        return df
