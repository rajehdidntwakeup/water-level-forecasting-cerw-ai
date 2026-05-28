"""MCP server exposing Pegelonline REST API tools.

Pegelonline (WSV) provides real-time and historical water level, discharge,
temperature, and forecast data for German waterways.
Base URL: https://www.pegelonline.wsv.de/webservices/rest-api/v2
No authentication required.
"""

import json
import ssl
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
from typing import Optional

from mcp.server.fastmcp import FastMCP

PEGELONLINE_BASE = "https://www.pegelonline.wsv.de/webservices/rest-api/v2"

SSL_CONTEXT = ssl.create_default_context()
SSL_CONTEXT.check_hostname = False
SSL_CONTEXT.verify_mode = ssl.CERT_NONE

mcp = FastMCP("pegelonline")


def _get(path: str, params: Optional[dict] = None) -> dict | list:
    """GET request to Pegelonline API, return parsed JSON."""
    url = f"{PEGELONLINE_BASE}{path}"
    if params:
        query = "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
        url = f"{url}?{query}"
    req = Request(url, headers={"Accept": "application/json"})
    with urlopen(req, timeout=30, context=SSL_CONTEXT) as resp:
        return json.loads(resp.read().decode())


@mcp.tool()
def list_stations(
    include_timeseries: bool = True,
    include_forecast_timeseries: bool = False,
    has_timeseries: Optional[str] = None,
) -> str:
    """List all available gauge stations from Pegelonline.

    Args:
        include_timeseries: Include available timeseries per station.
        include_forecast_timeseries: Include forecast timeseries availability.
        has_timeseries: Filter to stations with a specific timeseries type
            (e.g. 'W' for water level, 'Q' for discharge, 'WV' for forecast).
    """
    params = {}
    if include_timeseries:
        params["includeTimeseries"] = "true"
    if include_forecast_timeseries:
        params["includeForecastTimeseries"] = "true"
    if has_timeseries:
        params["hasTimeseries"] = has_timeseries
    data = _get("/stations.json", params)
    # Summarize to keep output manageable
    stations = []
    for s in data:
        entry = {
            "uuid": s.get("uuid"),
            "name": s.get("shortname") or s.get("longname"),
            "water": s.get("water", {}).get("longname"),
            "km": s.get("km"),
        }
        if include_timeseries and "timeseries" in s:
            ts_list = s["timeseries"]
            entry["timeseries"] = [t.get("shortname") for t in ts_list]
        stations.append(entry)
    return json.dumps(stations, indent=2, ensure_ascii=False)


@mcp.tool()
def get_station_detail(uuid: str) -> str:
    """Get detailed metadata and current measurements for a station.

    Args:
        uuid: Station UUID (obtain from list_stations).
    """
    data = _get(f"/stations/{uuid}.json", {"includeTimeseries": "true"})
    result = {
        "uuid": data.get("uuid"),
        "name": data.get("longname"),
        "shortname": data.get("shortname"),
        "water": data.get("water", {}).get("longname"),
        "km": data.get("km"),
        "latitude": data.get("latitude"),
        "longitude": data.get("longitude"),
        "federal_state": data.get("federalState"),
        "timeseries": [],
    }
    for ts in data.get("timeseries", []):
        ts_info = {
            "shortname": ts.get("shortname"),
            "longname": ts.get("longname"),
            "unit": ts.get("unit"),
            "equidistance": ts.get("equidistance"),
        }
        current = ts.get("currentMeasurement", {})
        ts_info["current_value"] = current.get("value")
        ts_info["current_timestamp"] = current.get("timestamp")
        result["timeseries"].append(ts_info)
    return json.dumps(result, indent=2, ensure_ascii=False)


@mcp.tool()
def get_measurements(
    uuid: str,
    timeseries: str = "W",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> str:
    """Fetch historical measurements for a station's timeseries.

    Args:
        uuid: Station UUID.
        timeseries: Timeseries code — 'W' (water level cm),
            'Q' (discharge m³/s), 'WT' (water temperature °C).
        start: Start timestamp ISO-8601 (e.g. '2024-01-01T00:00+01:00').
        end: End timestamp ISO-8601.
    """
    params = {}
    if start:
        params["start"] = start
    if end:
        params["end"] = end
    data = _get(f"/stations/{uuid}/{timeseries}/measurements.json", params)
    # Return as JSON — may be large, so limit to first 500 points
    trimmed = data[:500] if len(data) > 500 else data
    result = {
        "station_uuid": uuid,
        "timeseries": timeseries,
        "total_points": len(data),
        "returned_points": len(trimmed),
        "data": trimmed,
    }
    return json.dumps(result, indent=2, ensure_ascii=False)


@mcp.tool()
def get_measurements_csv(
    uuid: str,
    timeseries: str = "W",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> str:
    """Fetch historical measurements as CSV (machine-friendly bulk export).

    Args:
        uuid: Station UUID.
        timeseries: Timeseries code ('W', 'Q', 'WT', 'WV').
        start: Start timestamp ISO-8601.
        end: End timestamp ISO-8601.
    """
    params = {}
    if start:
        params["start"] = start
    if end:
        params["end"] = end
    url = f"{PEGELONLINE_BASE}/stations/{uuid}/{timeseries}/measurements.csv"
    if params:
        query = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{url}?{query}"
    req = Request(url, headers={"Accept": "text/csv"})
    with urlopen(req, timeout=60, context=SSL_CONTEXT) as resp:
        return resp.read().decode()


@mcp.tool()
def get_forecast_timeseries(
    uuid: str,
    timeseries: str = "WV",
) -> str:
    """Fetch forecast timeseries for a station (if available).

    Only available at select stations. Use list_stations with
    has_timeseries='WV' to discover which stations support forecasts.

    Args:
        uuid: Station UUID.
        timeseries: Forecast timeseries code (default 'WV').
    """
    try:
        data = _get(f"/stations/{uuid}/{timeseries}/measurements.json")
    except HTTPError as e:
        return json.dumps({"error": f"HTTP {e.code}: {e.reason}"})
    return json.dumps(data, indent=2, ensure_ascii=False)


@mcp.tool()
def get_water_bodies() -> str:
    """List all water bodies (rivers) available in Pegelonline."""
    data = _get("/water.json")
    waters = [{"longname": w.get("longname"), "shortname": w.get("shortname")}
              for w in data]
    return json.dumps(waters, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run()