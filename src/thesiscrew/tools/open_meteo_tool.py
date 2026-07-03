"""CrewAI tools wrapping the Open-Meteo weather API.

Provides historical and forecast weather data (precipitation, temperature,
snow) for feature engineering. No API key required for non-commercial use.
"""

import json
import ssl
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
from typing import Type, Optional

from pydantic import BaseModel, Field
from crewai.tools import BaseTool

from thesiscrew.tools.cache_util import disk_cache

OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"

SSL_CONTEXT = ssl.create_default_context()
SSL_CONTEXT.check_hostname = False
SSL_CONTEXT.verify_mode = ssl.CERT_NONE


def _meteo_get(url: str) -> dict:
    req = Request(url, headers={"Accept": "application/json"})
    with urlopen(req, timeout=30, context=SSL_CONTEXT) as resp:
        return json.loads(resp.read().decode())


def _build_url(
    base: str, lat: float, lon: float,
    hourly: Optional[list[str]] = None,
    daily: Optional[list[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    timezone: str = "Europe/Vienna",
    forecast_days: Optional[int] = None,
) -> str:
    params = [f"latitude={lat}", f"longitude={lon}"]
    if hourly:
        params.append(f"hourly={','.join(hourly)}")
    if daily:
        params.append(f"daily={','.join(daily)}")
    if start_date:
        params.append(f"start_date={start_date}")
    if end_date:
        params.append(f"end_date={end_date}")
    if forecast_days:
        params.append(f"forecast_days={forecast_days}")
    params.append(f"timezone={timezone}")
    return f"{base}?{'&'.join(params)}"


# ── Historical Weather ─────────────────────────────────────────────────────

class HistoricalWeatherInput(BaseModel):
    latitude: float = Field(description="Station latitude (e.g. 48.38 for Korneuburg).")
    longitude: float = Field(description="Station longitude (e.g. 16.34 for Korneuburg).")
    start_date: str = Field(description="Start date YYYY-MM-DD.")
    end_date: str = Field(description="End date YYYY-MM-DD.")
    hourly_vars: Optional[list[str]] = Field(
        default=None,
        description="Hourly variables: temperature_2m, precipitation, rain, "
                    "snowfall, snow_depth, wind_speed_10m, etc.",
    )
    daily_vars: Optional[list[str]] = Field(
        default=None,
        description="Daily aggregates: temperature_2m_max, temperature_2m_min, "
                    "precipitation_sum, snowfall_sum, etc.",
    )
    timezone: str = Field(default="Europe/Vienna", description="Timezone string.")


class HistoricalWeatherTool(BaseTool):
    name: str = "get_historical_weather"
    description: str = (
        "Fetch historical weather data from Open-Meteo Archive API. "
        "Returns precipitation, temperature, snowfall, etc. for building "
        "training features. Outputs summary with time range and sample data."
    )
    args_schema: Type[BaseModel] = HistoricalWeatherInput

    @disk_cache(ttl_hours=168)
    def _run(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        hourly_vars: Optional[list[str]] = None,
        daily_vars: Optional[list[str]] = None,
        timezone: str = "Europe/Vienna",
    ) -> str:
        if hourly_vars is None:
            hourly_vars = [
                "temperature_2m", "precipitation", "rain",
                "snowfall", "snow_depth",
            ]
        if daily_vars is None:
            daily_vars = [
                "temperature_2m_max", "temperature_2m_min",
                "precipitation_sum", "snowfall_sum",
            ]
        url = _build_url(
            OPEN_METEO_ARCHIVE, latitude, longitude,
            hourly=hourly_vars, daily=daily_vars,
            start_date=start_date, end_date=end_date, timezone=timezone,
        )
        try:
            data = _meteo_get(url)
        except (HTTPError, URLError) as e:
            return f"Error fetching historical weather: {e}"
        summary = {
            "latitude": data.get("latitude"),
            "longitude": data.get("longitude"),
            "timezone": data.get("timezone"),
            "hourly_units": data.get("hourly_units"),
            "daily_units": data.get("daily_units"),
            "time_range": {
                "start": data.get("hourly", {}).get("time", [""])[0],
                "end": data.get("hourly", {}).get("time", [""])[-1],
            },
            "total_hourly_points": len(data.get("hourly", {}).get("time", [])),
            "total_daily_points": len(data.get("daily", {}).get("time", [])),
            "sample_first3": {
                k: v[:3] for k, v in data.get("hourly", {}).items()
                if isinstance(v, list)
            },
        }
        return json.dumps(summary, indent=2, ensure_ascii=False)


# ── Forecast Weather ───────────────────────────────────────────────────────

class ForecastWeatherInput(BaseModel):
    latitude: float = Field(description="Station latitude.")
    longitude: float = Field(description="Station longitude.")
    forecast_days: int = Field(default=7, description="Forecast days (1-16).")
    hourly_vars: Optional[list[str]] = Field(
        default=None,
        description="Hourly forecast variables.",
    )
    timezone: str = Field(default="Europe/Vienna", description="Timezone string.")


class ForecastWeatherTool(BaseTool):
    name: str = "get_forecast_weather"
    description: str = (
        "Fetch weather forecast from Open-Meteo Forecast API. "
        "Provides future precipitation, temperature, snowfall for forecast features."
    )
    args_schema: Type[BaseModel] = ForecastWeatherInput

    @disk_cache(ttl_hours=6)
    def _run(
        self,
        latitude: float,
        longitude: float,
        forecast_days: int = 7,
        hourly_vars: Optional[list[str]] = None,
        timezone: str = "Europe/Vienna",
    ) -> str:
        if hourly_vars is None:
            hourly_vars = [
                "temperature_2m", "precipitation", "rain",
                "snowfall", "snow_depth", "surface_pressure",
            ]
        url = _build_url(
            OPEN_METEO_FORECAST, latitude, longitude,
            hourly=hourly_vars, forecast_days=forecast_days, timezone=timezone,
        )
        try:
            data = _meteo_get(url)
        except (HTTPError, URLError) as e:
            return f"Error fetching forecast: {e}"
        summary = {
            "latitude": data.get("latitude"),
            "longitude": data.get("longitude"),
            "timezone": data.get("timezone"),
            "hourly_units": data.get("hourly_units"),
            "time_range": {
                "start": data.get("hourly", {}).get("time", [""])[0],
                "end": data.get("hourly", {}).get("time", [""])[-1],
            },
            "total_hourly_points": len(data.get("hourly", {}).get("time", [])),
            "sample_first5": {
                k: v[:5] for k, v in data.get("hourly", {}).items()
                if isinstance(v, list)
            },
        }
        return json.dumps(summary, indent=2, ensure_ascii=False)


# ── Korneuburg Convenience ─────────────────────────────────────────────────

class KorneuburgWeatherInput(BaseModel):
    start_date: str = Field(description="Start date YYYY-MM-DD.")
    end_date: str = Field(description="End date YYYY-MM-DD.")


class KorneuburgWeatherTool(BaseTool):
    name: str = "get_korneuburg_weather"
    description: str = (
        "Fetch historical weather for Korneuburg (48.38°N, 16.34°E). "
        "Pre-configured with target station coordinates and relevant variables."
    )
    args_schema: Type[BaseModel] = KorneuburgWeatherInput

    @disk_cache(ttl_hours=168)
    def _run(self, start_date: str, end_date: str) -> str:
        tool = HistoricalWeatherTool()
        return tool._run(
            latitude=48.38, longitude=16.34,
            start_date=start_date, end_date=end_date,
        )


class KorneuburgForecastInput(BaseModel):
    forecast_days: int = Field(default=7, description="Forecast days (1-16).")


class KorneuburgForecastTool(BaseTool):
    name: str = "get_korneuburg_forecast"
    description: str = (
        "Fetch weather forecast for Korneuburg (48.38°N, 16.34°E). "
        "Pre-configured with target station coordinates."
    )
    args_schema: Type[BaseModel] = KorneuburgForecastInput

    @disk_cache(ttl_hours=6)
    def _run(self, forecast_days: int = 7) -> str:
        tool = ForecastWeatherTool()
        return tool._run(
            latitude=48.38, longitude=16.34,
            forecast_days=forecast_days,
        )