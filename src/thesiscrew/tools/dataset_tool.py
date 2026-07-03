"""Dataset builder that fetches real-world data and writes a ready-to-model CSV.

Combines Pegelonline water-level measurements for Korneuburg with Open-Meteo
weather reanalysis. The resulting CSV is saved under output/data/processed and is
used by feature-engineering, baseline, and modeling agents.
"""

import json
import os
from datetime import datetime, timedelta
from io import StringIO
from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

from thesiscrew.tools.pegelonline_tool import GetMeasurementsCSVTool
from thesiscrew.tools.cache_util import disk_cache

OUTPUT_DIR = os.environ.get("PEGELHUB_OUTPUT_DIR", "output")
DATA_DIR = os.environ.get("PEGELHUB_DATA_DIR", os.path.join(OUTPUT_DIR, "data"))

# Korneuburg on the Austrian Danube (Pegelonline UUID discovered by the first agent).
KORNEUBURG_PEGEL_UUID = "ff44be4a-f934-446c-afb0-a1cf7702c48c"


def _open_meteo_past_url(
    latitude: float,
    longitude: float,
    past_days: int = 30,
    hourly_vars: Optional[list[str]] = None,
    timezone: str = "Europe/Vienna",
) -> str:
    """Build Open-Meteo forecast endpoint URL with past_days."""
    if hourly_vars is None:
        hourly_vars = [
            "temperature_2m", "precipitation", "rain",
            "snowfall", "snow_depth", "wind_speed_10m",
        ]
    params = [
        f"latitude={latitude}",
        f"longitude={longitude}",
        f"hourly={','.join(hourly_vars)}",
        f"past_days={past_days}",
        "forecast_days=1",
        f"timezone={timezone.replace('/', '%2F')}",
    ]
    return f"https://api.open-meteo.com/v1/forecast?{'&'.join(params)}"


class BuildKorneuburgDatasetInput(BaseModel):
    days: int = Field(
        default=30,
        description="Number of historical days to fetch. Pegelonline allows roughly 31 days per request.",
    )
    output_path: str = Field(
        default="processed/korneuburg_hourly.csv",
        description="Relative path under the data directory where the CSV will be written.",
    )


class BuildKorneuburgDatasetTool(BaseTool):
    name: str = "build_korneuburg_dataset"
    description: str = (
        "Fetch Pegelonline water level and Open-Meteo weather data for Korneuburg "
        "and write a single hourly CSV to output/data/processed/korneuburg_hourly.csv. "
        "Use this at the start of data ingestion so downstream agents have real data to model."
    )
    args_schema: type[BaseModel] = BuildKorneuburgDatasetInput

    @disk_cache(ttl_hours=24)
    def _run(self, days: int = 30, output_path: str = "processed/korneuburg_hourly.csv") -> str:
        end = datetime.utcnow()
        start = end - timedelta(days=days)
        start_str = start.strftime("%Y-%m-%dT%H:%M:%S%z")
        end_str = end.strftime("%Y-%m-%dT%H:%M:%S%z")

        # ------------------------------------------------------------------
        # Water level from Pegelonline (15-minute raw data)
        # ------------------------------------------------------------------
        csv_tool = GetMeasurementsCSVTool()
        csv_text = csv_tool._run(
            uuid=KORNEUBURG_PEGEL_UUID,
            timeseries="W",
            start=start_str,
            end=end_str,
        )
        if csv_text.startswith("Error"):
            return json.dumps({"error": csv_text})

        df_w = pd.read_csv(StringIO(csv_text), sep=";")
        if df_w.shape[1] < 2:
            return json.dumps({
                "error": f"Unexpected Pegelonline CSV shape: {df_w.shape}",
                "head": csv_text[:500],
            })
        df_w = df_w.iloc[:, :2].copy()
        df_w.columns = ["timestamp", "water_level"]
        df_w["timestamp"] = pd.to_datetime(df_w["timestamp"])
        df_w["water_level"] = pd.to_numeric(df_w["water_level"], errors="coerce")

        # Resample raw 15-min readings to hourly (last value) and forward-fill
        # small gaps up to 3 hours.
        df_w = df_w.set_index("timestamp").resample("h").last()
        df_w["water_level"] = df_w["water_level"].ffill(limit=3)
        df_w = df_w.reset_index()

        # ------------------------------------------------------------------
        # Weather from Open-Meteo (hourly reanalysis)
        # ------------------------------------------------------------------
        import ssl
        from urllib.request import urlopen, Request

        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        weather_url = _open_meteo_past_url(
            latitude=48.38,
            longitude=16.34,
            past_days=days,
        )
        try:
            req = Request(weather_url, headers={"Accept": "application/json"})
            with urlopen(req, timeout=60, context=ctx) as resp:
                weather = json.loads(resp.read().decode())
        except Exception as e:
            return json.dumps({"error": f"Could not fetch weather: {e}"})

        hourly = weather.get("hourly", {})
        if not hourly.get("time"):
            return json.dumps({"error": "Open-Meteo returned no hourly data", "response": weather})

        df_weather = pd.DataFrame({
            "timestamp": pd.to_datetime(hourly["time"]),
            "temperature": hourly.get("temperature_2m"),
            "precipitation": hourly.get("precipitation"),
            "rain": hourly.get("rain"),
            "snowfall": hourly.get("snowfall"),
            "snow_depth": hourly.get("snow_depth"),
            "wind_speed": hourly.get("wind_speed_10m"),
        })

        # ------------------------------------------------------------------
        # Merge and save
        # ------------------------------------------------------------------
        df = pd.merge(df_w, df_weather, on="timestamp", how="outer").sort_values("timestamp")
        numeric_cols = [c for c in df.columns if c != "timestamp"]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        # Forward-fill weather readings (they are already hourly) and small water gaps.
        df[numeric_cols] = df[numeric_cols].ffill(limit=3)
        df = df.dropna(subset=["water_level"])

        full_path = os.path.join(DATA_DIR, output_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        df.to_csv(full_path, index=False)

        return json.dumps({
            "output_path": output_path,
            "full_path": full_path,
            "rows": len(df),
            "columns": list(df.columns),
            "date_range": [str(df["timestamp"].min()), str(df["timestamp"].max())],
        }, indent=2, default=str)


class BuildFeatureMatrixInput(BaseModel):
    input_path: str = Field(
        default="processed/korneuburg_hourly.csv",
        description="Hourly CSV produced by build_korneuburg_dataset.",
    )
    output_path: str = Field(
        default="features/feature_matrix.csv",
        description="Relative path under data dir for the feature matrix.",
    )
    target_col: str = Field(default="water_level", description="Column to forecast.")
    horizons: list[int] = Field(
        default=[1, 6, 12, 24, 48, 72, 168],
        description="Forecast horizons in hours for target columns.",
    )


class BuildFeatureMatrixTool(BaseTool):
    name: str = "build_feature_matrix"
    description: str = (
        "Build a deterministic, model-ready feature matrix from the Korneuburg "
        "hourly dataset. Adds lag/rolling/calendar/meteorological features and "
        "chronological train/validation/test splits. Saves to "
        "output/data/features/feature_matrix.csv and the three split CSVs."
    )
    args_schema: type[BaseModel] = BuildFeatureMatrixInput

    @disk_cache(ttl_hours=24)
    def _run(
        self,
        input_path: str = "processed/korneuburg_hourly.csv",
        output_path: str = "features/feature_matrix.csv",
        target_col: str = "water_level",
        horizons: list[int] = [1, 6, 12, 24, 48, 72, 168],
    ) -> str:
        import numpy as np

        full_input = os.path.join(DATA_DIR, input_path)
        if not os.path.exists(full_input):
            return json.dumps({"error": f"Input dataset not found: {full_input}"})

        df = pd.read_csv(full_input, parse_dates=["timestamp"])
        if target_col not in df.columns:
            return json.dumps({"error": f"Target column '{target_col}' not in {input_path}"})

        # --- endogenous features ------------------------------------------------
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

        # --- calendar features --------------------------------------------------
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

        # --- meteorological features ------------------------------------------
        if "precipitation" in df.columns:
            for window in [3, 6, 12, 24, 48, 72]:
                df[f"precip_acc_{window}h"] = df["precipitation"].rolling(window=window, min_periods=1).sum()
        if "temperature" in df.columns:
            df["temp_rolling_24h"] = df["temperature"].rolling(window=24, min_periods=1).mean()
        if "snow_depth" in df.columns:
            df["snow_depth_lag_24"] = df["snow_depth"].shift(24)

        # --- target horizons ----------------------------------------------------
        max_horizon = max(horizons)
        for h in horizons:
            df[f"target_{h}h"] = df[target_col].shift(-h)

        # Drop rows that do not have all lags and at least one future target.
        df = df.dropna(subset=[f"{target_col}_lag_168"])
        target_cols = [f"target_{h}h" for h in horizons]
        df = df.dropna(subset=target_cols, how="all")

        # --- chronological split ----------------------------------------------
        df = df.sort_values("timestamp").reset_index(drop=True)
        n = len(df)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)
        train = df.iloc[:n_train]
        val = df.iloc[n_train:n_train + n_val]
        test = df.iloc[n_train + n_val:]

        out_dir = os.path.join(DATA_DIR, "features")
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.join(out_dir, "feature_matrix")
        df.to_csv(f"{base}.csv", index=False)
        train.to_csv(f"{base}_train.csv", index=False)
        val.to_csv(f"{base}_val.csv", index=False)
        test.to_csv(f"{base}_test.csv", index=False)

        return json.dumps({
            "feature_matrix": f"{base}.csv",
            "train": f"{base}_train.csv",
            "val": f"{base}_val.csv",
            "test": f"{base}_test.csv",
            "total_rows": n,
            "train_rows": len(train),
            "val_rows": len(val),
            "test_rows": len(test),
            "columns": len(df.columns),
            "horizons": horizons,
        }, indent=2, default=str)
