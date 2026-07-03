"""CrewAI tools wrapping eHYD / Austrian hydrographic data access.

Provides station metadata, historical time-series, and characteristic
values (MW96, flood warning levels) for Austrian Danube gauges.
"""

import json
import ssl
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
from typing import Optional

from pydantic import BaseModel, Field
from crewai.tools import BaseTool

from thesiscrew.tools.cache_util import disk_cache

EHYD_BASE = "https://ehyd.gv.at"

SSL_CONTEXT = ssl.create_default_context()
SSL_CONTEXT.check_hostname = False
SSL_CONTEXT.verify_mode = ssl.CERT_NONE

AUSTRIAN_DANUBE_STATIONS = {
    "korneuburg": {
        "name": "Korneuburg",
        "river": "Donau",
        "latitude": 48.345,
        "longitude": 16.337,
        "eHYD_id": "207273",
        "role": "primary target station",
    },
    "wien_reichsbruecke": {
        "name": "Wien / Reichsbrücke",
        "river": "Donau",
        "latitude": 48.227,
        "longitude": 16.398,
        "eHYD_id": "207237",
        "role": "alternative station with forecast data",
    },
    "linz": {
        "name": "Linz-Donau",
        "river": "Donau",
        "latitude": 48.306,
        "longitude": 14.286,
        "eHYD_id": "207070",
        "role": "upstream station, travel time ~6-12h to Korneuburg",
    },
}


def _ehyd_get(url: str) -> dict | str:
    req = Request(url, headers={"Accept": "application/json", "User-Agent": "thesiscrew/1.0"})
    with urlopen(req, timeout=30, context=SSL_CONTEXT) as resp:
        content_type = resp.headers.get("Content-Type", "")
        body = resp.read().decode()
        if "json" in content_type:
            return json.loads(body)
        return body


# ── List Austrian Stations ─────────────────────────────────────────────────

class ListAustrianStationsInput(BaseModel):
    pass


class ListAustrianStationsTool(BaseTool):
    name: str = "list_austrian_stations"
    description: str = (
        "List predefined Austrian Danube stations with metadata. "
        "Returns station names, river, coordinates, eHYD IDs, and their "
        "role in the forecasting pipeline (target vs upstream)."
    )
    args_schema: type[BaseModel] = ListAustrianStationsInput

    def _run(self) -> str:
        return json.dumps(AUSTRIAN_DANUBE_STATIONS, indent=2, ensure_ascii=False)


# ── Station Metadata ───────────────────────────────────────────────────────

class StationMetadataInput(BaseModel):
    station_key: str = Field(
        default="korneuburg",
        description="Station key: 'korneuburg', 'wien_reichsbruecke', or 'linz'.",
    )


class StationMetadataTool(BaseTool):
    name: str = "get_station_metadata"
    description: str = (
        "Fetch metadata for an Austrian gauge station from eHYD. "
        "Includes gauge zero, characteristic values (MW96, warning levels), "
        "and available data types."
    )
    args_schema: type[BaseModel] = StationMetadataInput

    @disk_cache(ttl_hours=168)
    def _run(self, station_key: str = "korneuburg") -> str:
        station = AUSTRIAN_DANUBE_STATIONS.get(station_key)
        if not station:
            return json.dumps({
                "error": f"Unknown station key: {station_key}. "
                         f"Available: {list(AUSTRIAN_DANUBE_STATIONS.keys())}"
            })
        try:
            url = f"{EHYD_BASE}/api/v1/stations/{station['eHYD_id']}"
            data = _ehyd_get(url)
            return json.dumps(data, indent=2, ensure_ascii=False)
        except (HTTPError, URLError) as e:
            return json.dumps({
                "station": station,
                "note": "eHYD API may require browser session. "
                        "Use Pegelonline for German stations or manual CSV export.",
                "error": str(e),
            })


# ── Station Data ───────────────────────────────────────────────────────────

class StationDataInput(BaseModel):
    station_key: str = Field(
        default="korneuburg",
        description="Station key: 'korneuburg', 'wien_reichsbruecke', 'linz'.",
    )
    parameter: str = Field(
        default="W",
        description="Data type: W (water level), Q (discharge), WT (temperature).",
    )
    start_date: Optional[str] = Field(
        default=None, description="Start date YYYY-MM-DD.",
    )
    end_date: Optional[str] = Field(
        default=None, description="End date YYYY-MM-DD.",
    )


class StationDataTool(BaseTool):
    name: str = "get_station_data"
    description: str = (
        "Fetch historical time-series data from eHYD for an Austrian station. "
        "Returns water level, discharge, or temperature data."
    )
    args_schema: type[BaseModel] = StationDataInput

    @disk_cache(ttl_hours=168)
    def _run(
        self,
        station_key: str = "korneuburg",
        parameter: str = "W",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> str:
        station = AUSTRIAN_DANUBE_STATIONS.get(station_key)
        if not station:
            return json.dumps({"error": f"Unknown station key: {station_key}"})
        try:
            url = f"{EHYD_BASE}/api/v1/stations/{station['eHYD_id']}/data/{parameter}"
            params = []
            if start_date:
                params.append(f"start={start_date}")
            if end_date:
                params.append(f"end={end_date}")
            if params:
                url += "?" + "&".join(params)
            data = _ehyd_get(url)
            return json.dumps(data, indent=2, ensure_ascii=False)
        except (HTTPError, URLError) as e:
            return json.dumps({
                "station": station,
                "parameter": parameter,
                "note": "eHYD data may require manual CSV export from "
                        "https://ehyd.gv.at. API access may be limited.",
                "error": str(e),
                "fallback": "Use Pegelonline for Passau (German Danube) or "
                            "export CSV from eHYD portal.",
            })


# ── Characteristic Values ──────────────────────────────────────────────────

class CharacteristicValuesInput(BaseModel):
    station_key: str = Field(
        default="korneuburg",
        description="Station key: 'korneuburg', 'wien_reichsbruecke', 'linz'.",
    )


class CharacteristicValuesTool(BaseTool):
    name: str = "get_characteristic_values"
    description: str = (
        "Fetch characteristic values (MW96, flood warning levels) for an "
        "Austrian station. These thresholds define flood-level classes used "
        "in stratified evaluation and risk classification."
    )
    args_schema: type[BaseModel] = CharacteristicValuesInput

    def _run(self, station_key: str = "korneuburg") -> str:
        station = AUSTRIAN_DANUBE_STATIONS.get(station_key)
        if not station:
            return json.dumps({"error": f"Unknown station key: {station_key}"})
        try:
            url = (f"{EHYD_BASE}/api/v1/stations/{station['eHYD_id']}"
                   f"/characteristic-values")
            data = _ehyd_get(url)
            return json.dumps(data, indent=2, ensure_ascii=False)
        except (HTTPError, URLError) as e:
            return json.dumps({
                "station": station,
                "note": "Characteristic values may require manual lookup from "
                        "the eHYD web portal. Common Korneuburg values: "
                        "MW96 ~210cm, warning levels vary by station.",
                "error": str(e),
                "reference": "https://ehyd.gv.at — search station by name/number",
            })