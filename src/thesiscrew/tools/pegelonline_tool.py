"""CrewAI tools wrapping the Pegelonline REST API (WSV Germany).

Base URL: https://www.pegelonline.wsv.de/webservices/rest-api/v2
No authentication required.
"""

import json
import ssl
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
from typing import Type, Optional

from pydantic import BaseModel, Field
from crewai.tools import BaseTool

from thesiscrew.tools.cache_util import disk_cache


PEGELONLINE_BASE = "https://www.pegelonline.wsv.de/webservices/rest-api/v2"

SSL_CONTEXT = ssl.create_default_context()
SSL_CONTEXT.check_hostname = False
SSL_CONTEXT.verify_mode = ssl.CERT_NONE


def _pegel_get(path: str, params: Optional[dict[str, str]] = None) -> dict | list:
    url = f"{PEGELONLINE_BASE}{path}"
    if params:
        query = "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
        url = f"{url}?{query}"
    req = Request(url, headers={"Accept": "application/json"})
    with urlopen(req, timeout=30, context=SSL_CONTEXT) as resp:
        return json.loads(resp.read().decode())


# ── List Stations ──────────────────────────────────────────────────────────

class ListStationsInput(BaseModel):
    include_timeseries: bool = Field(
        default=True,
        description="Include available timeseries per station.",
    )
    include_forecast: bool = Field(
        default=False,
        description="Include forecast timeseries availability.",
    )
    has_timeseries: Optional[str] = Field(
        default=None,
        description="Filter stations by timeseries code: W (water level), "
                    "Q (discharge), WV (forecast).",
    )


class ListStationsTool(BaseTool):
    name: str = "list_stations"
    description: str = (
        "List all available gauge stations from Pegelonline. "
        "Returns station UUIDs, names, water bodies, and available timeseries. "
        "Use has_timeseries='WV' to find stations with official forecasts."
    )
    args_schema: Type[BaseModel] = ListStationsInput

    def _run(
        self,
        include_timeseries: bool = True,
        include_forecast: bool = False,
        has_timeseries: Optional[str] = None,
    ) -> str:
        params = {}
        if include_timeseries:
            params["includeTimeseries"] = "true"
        if include_forecast:
            params["includeForecastTimeseries"] = "true"
        if has_timeseries:
            params["hasTimeseries"] = has_timeseries
        try:
            data = _pegel_get("/stations.json", params)
        except (HTTPError, URLError) as e:
            return f"Error fetching stations: {e}"
        stations = []
        for s in data:
            entry = {
                "uuid": s.get("uuid"),
                "name": s.get("shortname") or s.get("longname"),
                "water": s.get("water", {}).get("longname"),
                "km": s.get("km"),
            }
            if include_timeseries and "timeseries" in s:
                entry["timeseries"] = [t.get("shortname") for t in s["timeseries"]]
            stations.append(entry)
        return json.dumps(stations, indent=2, ensure_ascii=False)


# ── Station Detail ─────────────────────────────────────────────────────────

class StationDetailInput(BaseModel):
    uuid: str = Field(description="Station UUID (from list_stations).")


class StationDetailTool(BaseTool):
    name: str = "station_detail"
    description: str = (
        "Get detailed metadata and current measurements for a Pegelonline station. "
        "Includes coordinates, timeseries info, and latest measured values."
    )
    args_schema: Type[BaseModel] = StationDetailInput

    def _run(self, uuid: str) -> str:
        try:
            data = _pegel_get(f"/stations/{uuid}.json", {"includeTimeseries": "true"})
        except (HTTPError, URLError) as e:
            return f"Error fetching station {uuid}: {e}"
        result = {
            "uuid": data.get("uuid"),
            "name": data.get("longname"),
            "shortname": data.get("shortname"),
            "water": data.get("water", {}).get("longname"),
            "km": data.get("km"),
            "latitude": data.get("latitude"),
            "longitude": data.get("longitude"),
            "timeseries": [],
        }
        for ts in data.get("timeseries", []):
            ts_info = {
                "shortname": ts.get("shortname"),
                "longname": ts.get("longname"),
                "unit": ts.get("unit"),
                "equidistance": ts.get("equidistance"),
            }
            cur = ts.get("currentMeasurement", {})
            ts_info["current_value"] = cur.get("value")
            ts_info["current_timestamp"] = cur.get("timestamp")
            result["timeseries"].append(ts_info)
        return json.dumps(result, indent=2, ensure_ascii=False)


# ── Measurements ───────────────────────────────────────────────────────────

class GetMeasurementsInput(BaseModel):
    uuid: str = Field(description="Station UUID.")
    timeseries: str = Field(
        default="W",
        description="Timeseries code: W (water level cm), "
                    "Q (discharge m³/s), WT (water temperature °C).",
    )
    start: Optional[str] = Field(
        default=None,
        description="Start timestamp ISO-8601 (e.g. '2024-01-01T00:00+01:00').",
    )
    end: Optional[str] = Field(
        default=None,
        description="End timestamp ISO-8601.",
    )


class GetMeasurementsTool(BaseTool):
    name: str = "get_measurements"
    description: str = (
        "Fetch historical measurements from Pegelonline as JSON. "
        "Use for water level (W), discharge (Q), or temperature (WT). "
        "Returns up to 500 data points."
    )
    args_schema: Type[BaseModel] = GetMeasurementsInput

    @disk_cache(ttl_hours=168)
    def _run(
        self,
        uuid: str,
        timeseries: str = "W",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> str:
        params = {}
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        try:
            data = _pegel_get(
                f"/stations/{uuid}/{timeseries}/measurements.json", params
            )
        except (HTTPError, URLError) as e:
            return f"Error fetching measurements: {e}"
        trimmed = data[:500] if len(data) > 500 else data
        result = {
            "station_uuid": uuid,
            "timeseries": timeseries,
            "total_points": len(data),
            "returned_points": len(trimmed),
            "data": trimmed,
        }
        return json.dumps(result, indent=2, ensure_ascii=False)


# ── Measurements CSV ───────────────────────────────────────────────────────

class GetMeasurementsCSVInput(BaseModel):
    uuid: str = Field(description="Station UUID.")
    timeseries: str = Field(
        default="W",
        description="Timeseries code (W, Q, WT, WV).",
    )
    start: Optional[str] = Field(default=None, description="Start ISO-8601.")
    end: Optional[str] = Field(default=None, description="End ISO-8601.")


class GetMeasurementsCSVTool(BaseTool):
    name: str = "get_measurements_csv"
    description: str = (
        "Fetch historical measurements from Pegelonline as CSV. "
        "Best for bulk data export. Returns raw CSV text."
    )
    args_schema: Type[BaseModel] = GetMeasurementsCSVInput

    @disk_cache(ttl_hours=168)
    def _run(
        self,
        uuid: str,
        timeseries: str = "W",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> str:
        params = {}
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        url = f"{PEGELONLINE_BASE}/stations/{uuid}/{timeseries}/measurements.csv"
        if params:
            query = "&".join(f"{k}={v}" for k, v in params.items())
            url = f"{url}?{query}"
        try:
            req = Request(url, headers={"Accept": "text/csv"})
            with urlopen(req, timeout=60, context=SSL_CONTEXT) as resp:
                return resp.read().decode()
        except (HTTPError, URLError) as e:
            return f"Error fetching CSV: {e}"


# ── Forecast Timeseries ────────────────────────────────────────────────────

class GetForecastInput(BaseModel):
    uuid: str = Field(description="Station UUID with forecast availability.")
    timeseries: str = Field(default="WV", description="Forecast code (default WV).")


class GetForecastTool(BaseTool):
    name: str = "get_forecast_timeseries"
    description: str = (
        "Fetch official forecast timeseries (WV) from Pegelonline. "
        "Only available at select stations. Use list_stations with "
        "has_timeseries='WV' to find eligible stations first."
    )
    args_schema: Type[BaseModel] = GetForecastInput

    @disk_cache(ttl_hours=6)
    def _run(self, uuid: str, timeseries: str = "WV") -> str:
        try:
            data = _pegel_get(
                f"/stations/{uuid}/{timeseries}/measurements.json"
            )
        except HTTPError as e:
            return json.dumps({"error": f"HTTP {e.code}: {e.reason}"})
        return json.dumps(data, indent=2, ensure_ascii=False)


# ── Water Bodies ───────────────────────────────────────────────────────────

class GetWaterBodiesInput(BaseModel):
    pass


class GetWaterBodiesTool(BaseTool):
    name: str = "get_water_bodies"
    description: str = "List all water bodies (rivers) available in Pegelonline."
    args_schema: Type[BaseModel] = GetWaterBodiesInput

    def _run(self) -> str:
        try:
            data = _pegel_get("/water.json")
        except (HTTPError, URLError) as e:
            return f"Error fetching water bodies: {e}"
        waters = [
            {"longname": w.get("longname"), "shortname": w.get("shortname")}
            for w in data
        ]
        return json.dumps(waters, indent=2, ensure_ascii=False)