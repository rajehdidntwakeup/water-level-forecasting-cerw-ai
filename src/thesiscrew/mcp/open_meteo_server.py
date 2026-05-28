"""MCP server exposing Open-Meteo API tools.

Open-Meteo provides free weather data (no API key for non-commercial use)
including historical and forecast precipitation, temperature, and snow data.
Base URL: https://api.open-meteo.com/v1/ (forecast)
Archive: https://archive-api.open-meteo.com/v1/ (historical)
"""

import json
import ssl
from urllib.request import urlopen, Request
from urllib.error import HTTPError
from typing import Optional

from mcp.server.fastmcp import FastMCP

OPEN_METEO_BASE = "https://api.open-meteo.com/v1"
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"

SSL_CONTEXT = ssl.create_default_context()
SSL_CONTEXT.check_hostname = False
SSL_CONTEXT.verify_mode = ssl.CERT_NONE

mcp = FastMCP("open_meteo")


def _get(url: str) -> dict:
    req = Request(url, headers={"Accept": "application/json"})
    with urlopen(req, timeout=30, context=SSL_CONTEXT) as resp:
        return json.loads(resp.read().decode())


def _build_url(base: str, latitude: float, longitude: float,
               hourly: Optional[list] = None,
               daily: Optional[list] = None,
               start_date: Optional[str] = None,
               end_date: Optional[str] = None,
               timezone: str = "Europe/Vienna",
               forecast_days: Optional[int] = None) -> str:
    params = [f"latitude={latitude}", f"longitude={longitude}"]
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


@mcp.tool()
def get_historical_weather(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    hourly_vars: Optional[list] = None,
    daily_vars: Optional[list] = None,
    timezone: str = "Europe/Vienna",
) -> str:
    """Fetch historical weather data from Open-Meteo Archive API.

    Useful for building training features: precipitation, temperature,
    snow depth, wind speed, etc.

    Args:
        latitude: Station latitude (e.g. 48.38 for Korneuburg).
        longitude: Station longitude (e.g. 16.34 for Korneuburg).
        start_date: Start date 'YYYY-MM-DD'.
        end_date: End date 'YYYY-MM-DD'.
        hourly_vars: Hourly variables to retrieve. Available:
            temperature_2m, relative_humidity_2m, precipitation,
            rain, snowfall, snow_depth, wind_speed_10m, wind_direction_10m,
            surface_pressure, cloud_cover, et0_fao56_evapotranspiration.
        daily_vars: Daily aggregates. Available:
            temperature_2m_max, temperature_2m_min, temperature_2m_mean,
            precipitation_sum, rain_sum, snowfall_sum, precipitation_hours,
            wind_speed_10m_max, et0_fao56_evapotranspiration_sum.
        timezone: Timezone string (default Europe/Vienna).
    """
    if hourly_vars is None:
        hourly_vars = [
            "temperature_2m", "precipitation", "rain",
            "snowfall", "snow_depth"
        ]
    if daily_vars is None:
        daily_vars = [
            "temperature_2m_max", "temperature_2m_min",
            "precipitation_sum", "snowfall_sum"
        ]
    url = _build_url(
        OPEN_METEO_ARCHIVE, latitude, longitude,
        hourly=hourly_vars, daily=daily_vars,
        start_date=start_date, end_date=end_date, timezone=timezone,
    )
    data = _get(url)
    # Summarize: return metadata + first/last few rows to keep output manageable
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
        "sample_hourly_first3": {
            k: v[:3] for k, v in data.get("hourly", {}).items()
            if isinstance(v, list)
        },
        "sample_hourly_last3": {
            k: v[-3:] for k, v in data.get("hourly", {}).items()
            if isinstance(v, list)
        },
    }
    return json.dumps(summary, indent=2, ensure_ascii=False)


@mcp.tool()
def get_forecast_weather(
    latitude: float,
    longitude: float,
    forecast_days: int = 7,
    hourly_vars: Optional[list] = None,
    timezone: str = "Europe/Vienna",
) -> str:
    """Fetch weather forecast data from Open-Meteo Forecast API.

    Provides future precipitation, temperature, snowfall etc.
    Critical for forecast features (precip_forecast_t+h).

    Args:
        latitude: Station latitude.
        longitude: Station longitude.
        forecast_days: Number of forecast days (1-16, default 7).
        hourly_vars: Hourly forecast variables.
        timezone: Timezone string.
    """
    if hourly_vars is None:
        hourly_vars = [
            "temperature_2m", "precipitation", "rain",
            "snowfall", "snow_depth", "surface_pressure"
        ]
    url = _build_url(
        OPEN_METEO_FORECAST, latitude, longitude,
        hourly=hourly_vars, forecast_days=forecast_days, timezone=timezone,
    )
    data = _get(url)
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
        "sample_hourly_first5": {
            k: v[:5] for k, v in data.get("hourly", {}).items()
            if isinstance(v, list)
        },
    }
    return json.dumps(summary, indent=2, ensure_ascii=False)


@mcp.tool()
def get_korneuburg_weather(
    start_date: str,
    end_date: str,
) -> str:
    """Convenience: fetch historical weather for Korneuburg (48.38, 16.34).

    Pre-configured with the target station coordinates and relevant variables
    for water-level forecasting.

    Args:
        start_date: Start date 'YYYY-MM-DD'.
        end_date: End date 'YYYY-MM-DD'.
    """
    return get_historical_weather(
        latitude=48.38, longitude=16.34,
        start_date=start_date, end_date=end_date,
    )


@mcp.tool()
def get_korneuburg_forecast(forecast_days: int = 7) -> str:
    """Convenience: fetch weather forecast for Korneuburg (48.38, 16.34).

    Args:
        forecast_days: Number of forecast days (1-16).
    """
    return get_forecast_weather(
        latitude=48.38, longitude=16.34,
        forecast_days=forecast_days,
    )


if __name__ == "__main__":
    mcp.run()