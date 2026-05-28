"""MCP server exposing eHYD / Austrian hydrographic data tools.

eHYD (Water Level Information System Austria) and DORIS provide:
- Historical water levels (cm) at Austrian gauge stations
- Discharge data (m³/s)
- Official forecast products (Vorhersagen)
- Flood warning levels and characteristic values (MW96 etc.)

Access is via web portal and REST/CSV exports. This server wraps the
publicly accessible endpoints for Austrian Danube stations.
"""

import json
import ssl
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
from typing import Optional

from mcp.server.fastmcp import FastMCP

# eHYD REST endpoint base (public)
EHYD_BASE = "https://ehyd.gv.at"

SSL_CONTEXT = ssl.create_default_context()
SSL_CONTEXT.check_hostname = False
SSL_CONTEXT.verify_mode = ssl.CERT_NONE

mcp = FastMCP("ehyd")


def _get(url: str, accept: str = "application/json") -> dict | str:
    req = Request(url, headers={"Accept": accept, "User-Agent": "thesiscrew/1.0"})
    with urlopen(req, timeout=30, context=SSL_CONTEXT) as resp:
        content_type = resp.headers.get("Content-Type", "")
        body = resp.read().decode()
        if "json" in content_type:
            return json.loads(body)
        return body


# Known Austrian Danube stations for quick reference
AUSTRIAN_DANUBE_STATIONS = {
    "korneuburg": {
        "name": "Korneuburg",
        "river": "Donau",
        "latitude": 48.345,
        "longitude": 16.337,
        "eHYD_id": "207273",
        "description": "Primary target station. Regulated Danube, long record.",
    },
    "wien_reichsbruecke": {
        "name": "Wien / Reichsbrücke",
        "river": "Donau",
        "latitude": 48.227,
        "longitude": 16.398,
        "eHYD_id": "207237",
        "description": "Vienna Danube gauge. Good alternative with forecast data.",
    },
    "linz": {
        "name": "Linz-Donau",
        "river": "Donau",
        "latitude": 48.306,
        "longitude": 14.286,
        "eHYD_id": "207070",
        "description": "Upstream station. Travel time ~6-12h to Korneuburg.",
    },
}


@mcp.tool()
def list_austrian_stations() -> str:
    """List predefined Austrian Danube stations with metadata.

    Returns station names, river, coordinates, eHYD IDs, and role
    in the forecasting pipeline (target vs upstream).
    """
    return json.dumps(AUSTRIAN_DANUBE_STATIONS, indent=2, ensure_ascii=False)


@mcp.tool()
def get_station_metadata(station_key: str = "korneuburg") -> str:
    """Fetch metadata for an Austrian station from eHYD.

    Includes gauge zero, characteristic values (MW96, warning levels),
    and available data types.

    Args:
        station_key: Station key — 'korneuburg', 'wien_reichsbruecke', or 'linz'.
    """
    station = AUSTRIAN_DANUBE_STATIONS.get(station_key)
    if not station:
        return json.dumps({"error": f"Unknown station key: {station_key}. "
                              f"Available: {list(AUSTRIAN_DANUBE_STATIONS.keys())}"})
    try:
        url = f"{EHYD_BASE}/api/v1/stations/{station['eHYD_id']}"
        data = _get(url)
        return json.dumps(data, indent=2, ensure_ascii=False)
    except (HTTPError, URLError) as e:
        return json.dumps({
            "station": station,
            "note": "eHYD API endpoint may require browser session. "
                    "Use Pegelonline for German stations or manual CSV export.",
            "error": str(e),
        })


@mcp.tool()
def get_station_data(
    station_key: str = "korneuburg",
    parameter: str = "W",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> str:
    """Fetch historical time-series data from eHYD for an Austrian station.

    Args:
        station_key: Station key ('korneuburg', 'wien_reichsbruecke', 'linz').
        parameter: Data type — 'W' (water level), 'Q' (discharge),
            'WT' (water temperature).
        start_date: Start date 'YYYY-MM-DD'.
        end_date: End date 'YYYY-MM-DD'.
    """
    station = AUSTRIAN_DANUBE_STATIONS.get(station_key)
    if not station:
        return json.dumps({"error": f"Unknown station key: {station_key}"})
    try:
        url = (f"{EHYD_BASE}/api/v1/stations/{station['eHYD_id']}"
               f"/data/{parameter}")
        params = []
        if start_date:
            params.append(f"start={start_date}")
        if end_date:
            params.append(f"end={end_date}")
        if params:
            url += "?" + "&".join(params)
        data = _get(url)
        return json.dumps(data, indent=2, ensure_ascii=False)
    except (HTTPError, URLError) as e:
        return json.dumps({
            "station": station,
            "parameter": parameter,
            "note": "eHYD data may require manual CSV export from the web portal "
                    "at https://ehyd.gv.at. API access may be limited.",
            "error": str(e),
            "fallback": "Use Pegelonline for Passau (German Danube) which has "
                         "reliable API access, or export CSV from eHYD portal.",
        })


@mcp.tool()
def get_characteristic_values(station_key: str = "korneuburg") -> str:
    """Fetch characteristic values (MW96, flood warning levels) for a station.

    These thresholds define flood-level classes used in stratified evaluation
    and risk-aware output classification.

    Args:
        station_key: Station key ('korneuburg', 'wien_reichsbruecke', 'linz').
    """
    station = AUSTRIAN_DANUBE_STATIONS.get(station_key)
    if not station:
        return json.dumps({"error": f"Unknown station key: {station_key}"})
    try:
        url = (f"{EHYD_BASE}/api/v1/stations/{station['eHYD_id']}"
               f"/characteristic-values")
        data = _get(url)
        return json.dumps(data, indent=2, ensure_ascii=False)
    except (HTTPError, URLError) as e:
        return json.dumps({
            "station": station,
            "note": "Characteristic values may require manual lookup from "
                    "the eHYD web portal. Common values for Korneuburg: "
                    "MW96 ~210cm, warning levels vary by station.",
            "error": str(e),
            "reference": "https://ehyd.gv.at — search station by name/number",
        })


if __name__ == "__main__":
    mcp.run()