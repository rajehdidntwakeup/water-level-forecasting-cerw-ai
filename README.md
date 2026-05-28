# Water Level Forecasting Crew (ThesisCrew)

AI-powered water level forecasting system built with CrewAI for the PegelHub thesis project. A multi-agent crew discovers hydrometric data, engineers features, trains ML models, verifies against official forecasts, and compiles a thesis-ready report.

## Overview

Six specialized agents work sequentially through five phases:

| Agent | Role | Phase |
|-------|------|-------|
| `data_researcher` | Hydrometric Data Researcher | 1 — Data Discovery |
| `data_engineer` | Data Ingestion & Preprocessing Engineer | 1 — Data Ingestion |
| `feature_engineer` | Hydrological Feature Engineer | 2 — Feature Engineering |
| `model_developer` | ML Forecasting Model Developer | 2–3 — Baselines & Training |
| `verification_analyst` | Forecast Verification & Evaluation Analyst | 5 — Verification |
| `integration_specialist` | Forecast Integration & Deployment Specialist | 4 — Integration |
| `report_writer` | Technical Report Writer & Documentation Specialist | 5b — Report |

**Target station:** Korneuburg / Donau (Danube), Austria

**Forecast horizons:** +1h (nowcast) through +168h (7 days)

## Project Structure

```
src/thesiscrew/
├── crew.py                          # Crew orchestration, agent & task wiring
├── main.py                          # Entry point
├── config/
│   ├── agents.yaml                  # 7 agent definitions
│   └── tasks.yaml                   # 10 task definitions (5 phases)
├── tools/
│   ├── pegelonline_tool.py          # Pegelonline REST API (6 tools)
│   ├── open_meteo_tool.py           # Open-Meteo weather API (4 tools)
│   ├── ehyd_tool.py                 # eHYD Austrian data (4 tools)
│   ├── data_processing_tool.py      # Data I/O, cleaning, features (11 tools)
│   ├── model_evaluation_tool.py     # Baselines, walk-forward, metrics (5 tools)
│   ├── read_input_tool.py           # Input file reader
│   └── report_writer_tool.py        # Report MD writer, TOC, metrics, tables (6 tools)
├── mcp/
│   ├── pegelonline_server.py        # MCP server for Pegelonline API
│   ├── open_meteo_server.py         # MCP server for Open-Meteo API
│   ├── ehyd_server.py               # MCP server for eHYD API
│   ├── data_processing_server.py    # MCP server for data processing
│   └── report_server.py            # MCP server for report generation
└── knowledge/
    └── water-level-forecasting-plan.md   # Full forecasting methodology
```

## Data Sources

| Source | Data | API |
|--------|------|-----|
| **Pegelonline** (WSV, Germany) | Water level (W), discharge (Q), temperature (WT), forecast (WV) | REST API, no auth |
| **eHYD / DORIS** (Austria) | Historical water levels, discharge, characteristic values | REST/CSV |
| **Open-Meteo** | Historical & forecast precipitation, temperature, snow depth | REST API, no auth (non-commercial) |

## Installation

```powershell
# Clone and install dependencies
git clone <repo-url>
cd water-level-forecasting-cerw-ai
uv sync
```

Configure `.env`:
```
BASE_URL=http://localhost:11434
MODEL=ollama/qwen3.5:cloud
OLLAMA_API_KEY=your-api-key
```

## Usage

```powershell
# Run the full crew pipeline
uv run run_crew

# Or directly
uv run python src/thesiscrew/main.py
```

Input is read from `input/research_area.json` (station config, horizons, data sources).

## Tools (27 total)

### Data Discovery (Pegelonline)
- `ListStationsTool` — List all gauge stations, filter by timeseries
- `StationDetailTool` — Station metadata + current measurements
- `GetMeasurementsTool` — Historical measurements (JSON)
- `GetMeasurementsCSVTool` — Bulk historical data (CSV)
- `GetForecastTool` — Official forecast timeseries (WV)
- `GetWaterBodiesTool` — List all water bodies

### Weather (Open-Meteo)
- `HistoricalWeatherTool` — Historical precip, temp, snow
- `ForecastWeatherTool` — Future weather forecasts
- `KorneuburgWeatherTool` — Convenience: Korneuburg historical
- `KorneuburgForecastTool` — Convenience: Korneuburg forecast

### Austrian Data (eHYD)
- `ListAustrianStationsTool` — Predefined Danube station metadata
- `StationMetadataTool` — eHYD station details
- `StationDataTool` — Historical time-series data
- `CharacteristicValuesTool` — MW96, warning levels, thresholds

### Data Processing
- `ListDataFilesTool` — List files in data directory
- `CSVSummaryTool` / `ParquetSummaryTool` — Read data summaries
- `ResampleTool` — Resample to regular frequency
- `FillGapsTool` — Forward-fill / interpolate gaps
- `LagFeaturesTool` — Create lag features (t-1, t-6, t-24, t-168)
- `RollingFeaturesTool` — Rolling mean/std windows
- `CalendarFeaturesTool` — sin/cos hour, day-of-week, day-of-year
- `RateOfChangeTool` — First/second derivatives
- `ChronoSplitTool` — Chronological train/val/test split
- `ComputeMetricsTool` — RMSE, MAE, MAPE, NSE, bias

### Model Evaluation
- `PersistenceBaselineTool` — Persistence baseline metrics per horizon
- `WalkForwardTool` — Walk-forward validation split indices
- `StratifiedMetricsTool` — Metrics by flow regime / season
- `RegisterModelTool` / `ListModelsTool` — Model manifest tracking

### Report Writing
- `WriteReportTool` — Append/overwrite sections to `output/report.md`
- `ReadReportTool` — Read current report state
- `MarkdownTableTool` — Format data as Markdown tables
- `ReportTOCTool` — Generate table of contents
- `RenderMetricsTool` — Format metrics JSON as comparison tables
- `ReadArtifactTool` — Read pipeline artifacts (CSV, Parquet, JSON, MD)

## Output

The pipeline produces:

- `output/report.md` — Complete thesis-ready Markdown report
- `data/` — Ingested and processed datasets
- `models/` — Trained model artifacts and manifest

The report includes: executive summary, data discovery, ingestion details, feature manifest, baseline results, model architecture, verification metrics (RMSE, MAE, NSE, bias per horizon per regime), integration docs, reproducibility guide, and limitations.

## Pipeline Phases

1. **Data Discovery & Ingestion** — Query Pegelonline/eHYD/Open-Meteo, validate completeness, store in PegelHub schema
2. **Feature Engineering & Baselines** — Lag features, upstream spatial lags, precipitation windows, calendar encoding, persistence/ARIMA/linear baselines
3. **Model Development** — XGBoost/LightGBM + LSTM, walk-forward CV, SHAP analysis
4. **Integration & Frontend** — REST API, dashboard/Excel, scheduled predictions
5. **Verification & Documentation** — Stratified evaluation vs official forecasts, failure-mode analysis
6. **Report Compilation** — Self-contained Markdown document with all findings

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.