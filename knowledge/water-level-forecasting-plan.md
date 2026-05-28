# Water Level Forecasting Plan — PegelHub Use Case 4

**Date:** May 28, 2026  
**Assignment:** AI-Based Water Level Forecasting  
**Context:** PegelHub (TEVS — Technologies of Distributed Systems)  
**Reference Station (Example):** Korneuburg / Donau (Danube), Austria  

---

## 1. Executive Summary

This document outlines a comprehensive plan for building an AI-powered water level forecasting system within the PegelHub platform. The system uses publicly available hydrometric data from German and Austrian water authorities to predict river water levels hours, days, and ideally weeks into the future. The approach emphasizes **forecast verification against official prognostic data**, enabling meaningful model benchmarking and continuous improvement. If time permits, all base data will be stored in PegelHub and read from there; forecasts will be surfaced via a lightweight frontend.

**Key Principles:**
- **Data transparency:** Use only publicly available APIs and datasets.
- **Verifiability:** Leverage official forecast products to compute model skill.
- **Incremental delivery:** Start with simple time-series models, then add richer features.
- **PegelHub-native:** Store and version data inside PegelHub where feasible.

---

## 2. Target Station Selection & Justification

### Recommended Gauge: Korneuburg (Danube, Austria)

| Attribute | Value |
|-----------|-------|
| **River** | Donau (Danube) — largest navigable waterway in Austria |
| **Gauge Name** | Korneuburg |
| **Authority** | BMIMI / Hydrographie NÖ (Lower Austria) |
| **Navigability** | Yes, regulated by the Austrian Danube shipping authority |
| **Data Availability** | Historical water levels (cm), discharge (m³/s), official forecasts (Korneuburg Forecast) |
| **Why this station?** | Well-documented, long historical record, has both measured and forecast reference data, high practical relevance (flood protection, shipping) |

**Alternative stations** (if Korneuburg data is incomplete): Passau (Donau, Germany), Wien (Danube, Austria), or any gauge on the Rhine via Pegelonline (e.g., Koblenz, Kaub).

---

## 3. Data Sources

### 3.1 Austria — Hydrographie / BMIMI / DORIS

**System:** eHYD (Water Level Information System Austria) + DORIS (operational prognosis portal)  
**Data types available:**
- **Historical water levels** (cm) at gauge stations — daily and sub-hourly.
- **Discharge / Abfluss (m³/s)** — derived from rating curves.
- **Official forecast products** (Vorhersagen from hydrological models) — typically 24–48 h horizon.
- **Flood warning levels** (Hochwasserstufen), mean water (MW96/2001), characteristic thresholds.
- **Precipitation / meteorological inputs** — some forecasts include upstream precipitation data.

**Access patterns:**
- Web portal: `https://www.bmluk.gv.at/themen/wasser/...` — for verification charts and PDF forecasts.
- REST or SOAP-like services: eHYD provides data exports in CSV/JSON.
- For long-term historical data: request via BMLUK or hydrographical offices.

**How it will be used:**
1. **Training labels (Y):** Historical measured water levels at regular intervals (e.g., every 15 min, hourly, or daily).
2. **Forecast verification baseline:** The official "Most Likely Forecast" and confidence intervals become the benchmark against which our AI model is scored.
3. **Threshold calibration:** MW96 and warning levels help convert raw prediction into risk-aware output.

### 3.2 Germany — PEGELONLINE (WSV)

**System:** PEGELONLINE by the Waterways and Shipping Directorate General (Wasserstraßen- und Schifffahrtsverwaltung des Bundes)  
**Base URL:** `https://www.pegelonline.wsv.de/webservices/rest-api/v2`  
**No authentication required.**

**Data types available:**
- **Stations / Stammdaten:** UUID, name, water body name, km index, coordinates, operator.
- **Timeseries / Zeitreihen:**
  - `W` — Water level (Wasserstand) [cm]
  - `Q` — Discharge (Abfluss) [m³/s]
  - `WT` — Water temperature
  - `WV` — Water level forecast (Wasserstandvorhersage) ← **critical for verification**
- **Current measurements:** Latest 15-minute / hourly readings per station.
- **Historical measurements:** CSV/JSON via `/measurements` endpoint, up to 30 days via API; older data via bulk download.
- **Forecast timeseries:** Selected stations publish operational forecast time series.
- **Characteristic values:** Mean water, navigation water, flood thresholds per gauge.

**Key API endpoints for this project:**

| Endpoint | Purpose |
|----------|---------|
| `GET /stations.json?includeTimeseries=true` | Discover all stations and their data streams. |
| `GET /stations.json?includeForecastTimeseries=true&hasTimeseries=WV` | List stations with forecast availability. |
| `GET /water.json` | Discover water bodies and their stations. |
| `GET /stations/{uuid}.json` | Station metadata and current measurements. |
| `GET /stations/{uuid}/{timeseries}/measurements.json?start=...&end=...` | Historical readings for training. |
| `GET /stations/{uuid}/{timeseries}/measurements.csv` | Bulk historical data, machine-friendly. |

**How it will be used:**
1. **Primary target variable (Y):** Water level (`W`) from upstream/downstream gauges on the Rhine, Elbe, or especially the Danube (e.g., Passau).
2. **Predictor features (X):** Same station’s lagged water levels, same river’s upstream stations (propagation lag), discharge (`Q`), temperature (`WT`).
3. **Official forecast verification (Y_true vs Y_pred):** `WV` series at stations where forecasts are published.
4. **Spatial context:** Upstream stations (e.g., Linz → Korneuburg → Wien on the Danube) provide leading signals due to river wave travel time.

### 3.3 Open Weather / Meteorological Data (Optional, High Value)

**Sources:**
- **Open-Meteo API** (`https://open-meteo.com/`) — free, no API key for non-commercial, historical and forecast weather.
- **DWD (German Weather Service)**: historical rainfall for German catchments.
- **ZAMG (Austrian Weather Service)**: historical and forecast precipitation for Austrian river basins.

**Data types:**
- Precipitation (rainfall in mm/h or mm/day)
- Snow depth / snow water equivalent (SWE)
- Temperature (air, soil)
- Snowmelt / evapotranspiration indices

**How it will be used:**
- Rainfall-runoff models fundamentally need precipitation input.
- Upstream rainfall is a **leading indicator** of downstream water level rises with a lag of hours to days, depending on catchment size and season.
- Snow depth and temperature drive spring melt events (significant for Alpine catchments).
- This data bridges the gap between pure statistical time-series models and physically informed AI.

### 3.4 DEM / Topographic Data (Optional, Advanced)

**Sources:** NASA SRTM, EU-DEM, EU-Hydro.

**How it will be used:**
- Estimate river cross-sections and floodplain storage.
- Derive catchment area upstream of each gauge to normalize discharge.
- Compute time-of-travel between upstream and downstream gauges.

---

## 4. Data Storage & Architecture in PegelHub

### 4.1 Storage Philosophy

The assignment states: _"If time allows, the base data should be stored in the Pegelhub and read from there for the forecast."_ Because the plan targets a robust deliverable, **PegelHub storage is assumed in scope.**

### 4.2 Data Ingestion Pipeline

```
┌──────────────────┐      ┌───────────────┐      ┌─────────────────┐
│ Pegelonline API  │      │  BMIMI / eHYD │      │ Open-Meteo      │
│ (Germany)        │      │  (Austria)    │      │ (Weather)       │
└────────┬─────────┘      └───────┬───────┘      └────────┬────────┘
         │                        │                        │
         └────────────┬───────────┴───────────┬────────────┘
                      │    Ingestion Service   │
                      │   (ETL / Scheduled)    │
                      │   ┌──────────────────┐ │
                      │   │ Parse, validate, │ │
                      │   │ deduplicate,     │ │
                      │   │ normalize units  │ │
                      │   └──────────────────┘ │
                      └───────────┬────────────┘
                                  │
                      ┌───────────▼────────────┐
                      │      PegelHub DB       │
                      │  ┌────────────────┐    │
                      │  │ stations       │    │
                      │  │ timeseries_raw │    │
                      │  │ forecasts      │    │
                      │  │ weather_data   │    │
                      │  │ predictions    │    │
                      │  └────────────────┘    │
                      └───────────┬────────────┘
                                  │
                      ┌───────────▼────────────┐
                      │   Forecasting Engine     │
                      │   (Feature engineering │
                      │    + ML model)          │
                      └───────────┬────────────┘
                                  │
                      ┌───────────▼────────────┐
                      │   Frontend / Excel     │
                      │   / API Consumer       │
                      └────────────────────────┘
```

### 4.3 Proposed PegelHub Schema (Tables)

| Table | Columns | Purpose |
|-------|---------|---------|
| `stations` | `station_id`, `name`, `water_body`, `country`, `lat`, `lon`, `km`, `operator`, `source` | Master data for all gauges used. |
| `measurements` | `station_id`, `timestamp`, `parameter` (W/Q/WT), `value`, `unit`, `quality_flag` | Raw historical time series. Time granularity: 15 min, hourly, daily (normalized). |
| `official_forecasts` | `station_id`, `forecast_timestamp`, `valid_time`, `parameter`, `value`, `lower_ci`, `upper_ci`, `model_source` | Official forecast snapshots for verification. |
| `weather_data` | `station_id`, `timestamp`, `precipitation_mm`, `temperature_c`, `snow_depth_cm`, `source` | Gridded or point weather data aligned to catchments. |
| `model_predictions` | `station_id`, `run_timestamp`, `model_version`, `horizon_h`, `predicted_value`, `actual_value`, `error_metrics` | AI model outputs and backtesting results. |

---

## 5. Feature Engineering — How Data Becomes Model Input

This section is the core of the plan: **which data fields become which features (X) and labels (Y), and why.**

### 5.1 Target Variable(s) — Label (Y)

| Target | Description | Source |
|--------|-------------|--------|
| `water_level_t+h` | Water level at forecast target time `h` hours/days ahead. | Pegelonline `W` (Germany), eHYD (Austria). |
| `delta_water_level` | Change in water level (cm) over next horizon. | Derived. |
| `flood_level_class` | Binary / categorical: below MW, elevated, warning, alert. | Derived using characteristic thresholds. |

**Horizons:**
- **Nowcast / Short term:** +1h, +6h, +12h (shipping, immediate response)
- **Medium term:** +24h, +48h (operational flood management)
- **Long term:** +72h, +168h (1 week) — if the model architecture supports it.

### 5.2 Lagged Endogenous Features (Water Level History)

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `water_level_t-N` | Water level N hours/days ago at the **same** station. | Autoregressive dynamics, momentum, tide/wave periodicity. |
| `water_level_t-1d` (same_time_yesterday) | Water level at same time yesterday. | Diurnal and daily periodicity. |
| `water_level_t-7d` (same_time_last_week) | Weekly periodicity for seasonal effects. | Weekly cycles in consumption/usage. |
| `rolling_mean_3h/6h/24h` | Smoothed recent water level. | Noise reduction, capturing trends. |
| `rolling_std_6h/24h` | Recent volatility. | Indicates rising/falling trend confidence. |
| `rate_of_change` | `wl_t - wl_{t-1}` per hour. | First derivative; critical for detecting flood waves. |
| `acceleration` | Change in rate of change. | Second derivative; leading indicator of inflection. |

### 5.3 Spatial / Upstream-Downstream Features

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `upstream_level_t-lag` | Water level at upstream gauge with travel-time lag. | A flood wave at Linz arrives at Korneuburg after a known travel time (e.g., 6–12 h depending on Danube stage). |
| `upstream_rate_of_change` | Rate of rise at upstream gauge. | Sharp rises upstream predict downstream surges. |
| `downstream_level_t` | Water level downstream (a constraining boundary). | Backwater effects from downstream control structures or confluences. |
| `upstream_catchment_avg_rain` | Average rainfall over the upstream catchment. | New water volume entering the system. |

**Time-of-travel estimation:** Use first-peak cross-correlation or theoretical kinematic wave speed.

### 5.4 Meteorological / Hydrometeorological Features

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `precip_t` to `precip_t-72h` | Accumulated rainfall over lag windows (3h, 6h, 12h, 24h, 48h, 72h). | Rainfall-runoff generation; longer windows capture slower subsurface flow. |
| `precip_forecast_t+h` | Forecast precipitation at lead time. | Allows model to predict future inflow, not just react. |
| `temperature` | Air temperature (daily min/max, rolling mean). | Drives evaporation and snowmelt in Alpine regions. |
| `snow_depth` / `swe` | Snow water equivalent in upstream mountains. | Spring melt contributions; critical seasonally. |
| `potential_evapotranspiration` (PET) | Derived from temperature, humidity, solar radiation. | Water balance closure. |

### 5.5 Calendar / Cyclical Features

| Feature | Encoding | Rationale |
|---------|----------|-----------|
| Hour of day | sin/cos(2π × hour / 24) | Diurnal patterns (e.g., power-plant usage, temperature). |
| Day of week | sin/cos(2π × dow / 7) | Weekly operational patterns. |
| Day of year | sin/cos(2π × doy / 365.25) | Seasonal snowmelt, vegetation, and precipitation regimes. |
| Is_weekend / Is_holiday | Binary | Changed human/water-management patterns. |

### 5.6 Official Forecast as a Feature (Meta-Learning)

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `official_forecast_t+h` | Existing operational forecast for the same horizon. | State-of-the-art physics-based model output; the AI can learn a bias-correction residual. |
| `official_forecast_spread` | Upper CI − Lower CI. | Uncertainty signal; wide spread = model low confidence. |
| `forecast_age_hours` | How old is the official forecast? | Staleness degrades value. |

**Strategy:** If official forecasts are available only at select stations, train the AI to predict the **error of the official forecast** (delta-correction) rather than raw water level. This is called MFB (Model Output Statistics) or post-processing.

### 5.7 Derived / Engineered Features

| Feature | Description |
|---------|-------------|
| `antecedent_precipitation_index (API)` | Exponentially weighted moving average of rainfall; captures wetness state of the catchment. |
| `baseflow_separation` | Low-pass filtered water level representing groundwater contribution. |
| `flood_risk_score` | Combined function of current level, rate of rise, upstream levels, and rainfall. |
| `time_since_peak_hours` | Temporal distance from last local peak/trough. |

---

## 6. Data Preprocessing & Cleaning

| Step | Action | Justification |
|------|--------|-------------|
| **Temporal alignment** | Resample all sources to a regular grid (e.g., hourly) using mean/last-value interpolation. | ML models require aligned timestamps. |
| **Missing value handling** | Forward-fill for short gaps (< 3 h); interpolation for medium; discard for long. | Sensors may fail temporarily. |
| **Outlier removal** | Z-score > 4 or physically impossible jumps (e.g., +200 cm in 15 min). | Sensor malfunctions, ice jam artifacts. |
| **Unit standardization** | All water levels in **cm**; discharge in **m³/s**; rainfall in **mm**. | Prevents scale errors. |
| **Train/val/test split** | Time-based split (e.g., 70/15/15 by chronological order). | Prevents data leakage; respects temporal dependencies. |
| **Scaling** | Min-Max or StandardScaler per feature. | Required for neural networks and some tree ensembles. |

---

## 7. Model Architecture — Recommended Approach

The plan acknowledges time constraints in a university project. The following is a tiered strategy.

### 7.1 Tier 1 — Baseline (Must-Have)

| Model | Purpose | Input Features |
|-------|---------|----------------|
| **Persistence / Naïve** | Predict next value = current value. | `water_level_t` |
| **AR(I)MA / Exponential Smoothing** | Classical time-series baseline. | Lagged same-station water level. |
| **Linear Regression** | Simple multivariate baseline. | Lagged levels + upstream levels + rainfall. |

**Why:** Baselines are fast, interpretable, and provide skill-scoring anchors.

### 7.2 Tier 2 — Main AI Model (Should-Have)

| Model | Architecture | Why it fits |
|-------|-----------|-------------|
| **LSTM / GRU (Sequence-to-Vector)** | Recurrent network over sliding window of past features. | Naturally handles variable-length sequences, captures memory of rainfall antecedents. |
| **XGBoost / LightGBM** | Gradient boosted trees on tabular feature set. | Excellent with heterogeneous features (lagged, rainfall, categorical calendar). Fast to train, interpretable SHAP values. |
| **Random Forest Baseline** | Ensemble of decision trees. | Robust default, no heavy hyperparameter tuning needed. |

**Recommended primary model:** A **stacked ensemble** where a tree model (XGBoost) handles tabular features, and an LSTM processes the raw time-series window. Or, for simplicity, **XGBoost alone** with carefully engineered lag features.

### 7.3 Tier 3 — Advanced (Could-Have)

| Model | Architecture | When to use |
|-------|-----------|-------------|
| **Temporal Fusion Transformer (TFT)** | Multi-horizon attention-based forecasting with static + known future + observed inputs. | If you need unified multi-horizon outputs and built-in interpretability. |
| **N-BEATS / N-HiTS** | Deep learning specifically for univariate time series. | Good for baselines but lacks support for covariates like rainfall. |
| **LSTM with attention** | Seq2seq with attention over input windows. | If you want to visualize which past time steps contributed to the forecast. |

### 7.4 Forecast Horizons & Model Strategy

| Horizon | Strategy | Why |
|---------|----------|-----|
| +1h to +6h | Direct autoregressive model (few lags suffice) | Short-term dynamics dominated by wave propagation. |
| +24h to +48h | Include rainfall forecasts, upstream gauges with lag | Requires meteorological input and spatial memory. |
| +72h to +168h | Physics-informed approach; official forecast as primary feature; AI as post-processor | Long-term dominated by hydrological model skill; AI best used for bias correction. |

---

## 8. Verification & Model Evaluation

### 8.1 Why Verification Matters

The use case explicitly states: _"Explore available data that also includes original forecast data, so that appropriate verification can be carried out."_

This means the AI model must be benchmarked against **both:**
1. The official hydrological forecast (BMIMI / Hydrographie / Pegelonline WV).
2. Simple baselines (persistence, climatology, moving average).

### 8.2 Metrics

| Metric | Formula / Meaning | When to use |
|--------|-------------------|-------------|
| **RMSE** | `sqrt(mean((y - ŷ)²))` | Primary accuracy score; punishes large errors. |
| **MAE** | `mean(|y - ŷ|)` | Robust to outliers; interpretable in original units (cm). |
| **MAPE** | `mean(|(y - ŷ)| / y)` | Relative error; problematic near zero but useful for high-flow periods. |
| **R² / NSE (Nash-Sutcliffe Efficiency)** | `1 - Σ(y-ŷ)² / Σ(y-ȳ)²` | Standard hydrological skill metric. Values > 0.5 acceptable; > 0.8 good. |
| **Continuous Ranked Probability Score (CRPS)** | Scores probabilistic forecasts. | If model outputs distributions or confidence intervals. |
| **Bias** | `mean(ŷ - y)` | Systematic over-/under-prediction. |

### 8.3 Verification Setup

1. **Historical re-forecast:** For a past event (e.g., a known flood), run the AI model using data available up to that point, and compare to what actually happened and what the official model predicted.
2. **Walk-forward validation:** Slide a training window forward in time; evaluate on the next block.
3. **Stratified evaluation:** Separate scores for:
   - Low water, normal, elevated, warning, and alert levels.
   - Different seasons (winter, spring melt, summer storms).
   - Different upstream conditions (with vs. without upstream rainfall).

---

## 9. Frontend / Presentation

The use case allows **"a simple frontend"** or **"an Excel chart."** This plan covers both options.

### 9.1 Option A — Web Dashboard (Recommended)

| Component | Technology | Description |
|-----------|-----------|-------------|
| Backend API | Spring Boot / Java (matches PegelHub tech stack) or Python (FastAPI/Flask) | Serves forecasts, historical data, and station metadata. |
| Frontend | React / Vue.js or plain HTML+JS | Time-series charts, gauge selector, horizon selector. |
| Visualization | Apache ECharts, Recharts, or Plotly | Line chart: measured (solid), forecast (dashed), confidence band (shaded), official forecast (dotted), thresholds (horizontal lines). |

**Key UI elements:**
- Station selector (dropdown map).
- Forecast horizon controller (1h, 6h, 24h, 48h, 7d).
- Overlay toggle: show/hide official forecast, confidence intervals.
- Alert panel: current status vs. thresholds.
- Skill score panel: last 24h model accuracy vs. baseline.

### 9.2 Option B — Excel Report

| Component | Description |
|-----------|-------------|
| Data export | CSV with columns: `timestamp, measured_level, ai_forecast, official_forecast, lower_ci, upper_ci, error`. |
| Charts | Excel line chart with multiple series and a shaded confidence band (use secondary area chart for CI). |
| Summary table | RMSE, MAE, NSE per horizon for the selected period. |

**When to use:** If the project is heavily backend/data-science focused and frontend time is short.

---

## 10. Implementation Roadmap

### Phase 1 — Data Discovery & Ingestion (Week 1)
- [ ] Select target station (Korneuburg or alternative).
- [ ] Call Pegelonline REST API and eHYD portals.
- [ ] Document available parameters, temporal resolution, gaps, and forecast availability.
- [ ] Implement ingestion scripts into PegelHub database.
- [ ] Verify data completeness (e.g., > 2 years of hourly data desirable).

### Phase 2 — Feature Engineering & Baselines (Week 2)
- [ ] Build feature pipeline: lags, upstream lags, weather alignment, calendar encoding.
- [ ] Implement persistence, ARIMA, and linear regression baselines.
- [ ] Compute official forecast availability and ingest if present.

### Phase 3 — Model Development (Week 3–4)
- [ ] Train XGBoost / LSTM on historical data.
- [ ] Hyperparameter tuning via time-series cross-validation.
- [ ] Evaluate against baselines and official forecasts.
- [ ] Document feature importance (SHAP for trees; attention weights for LSTM).

### Phase 4 — Integration & Frontend (Week 4–5)
- [ ] Store model predictions in PegelHub `model_predictions` table.
- [ ] Build REST endpoint: `GET /api/forecast/{station}?horizon=24h`.
- [ ] Build simple web dashboard OR Excel report template.
- [ ] Add real-time or daily cron job to re-fetch new measurements and update predictions.

### Phase 5 — Verification & Documentation (Week 5–6)
- [ ] Run model on held-out test set; compute RMSE, MAE, NSE.
- [ ] Produce comparative chart: AI vs. official forecast vs. measured values.
- [ ] Write final report / README documenting data sources, features, model choice, and results.

---

## 11. Risk & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Forecast data (`WV`) unavailable for chosen station. | Verification baseline lost. | Choose station with confirmed forecast; or use nearby station with forecast as proxy. |
| Historical data gaps > 20%. | Biased model training. | Use alternative station; interpolate; or reduce model complexity. |
| Weather API rate limits. | Training data incomplete. | Cache daily; use static historical archives rather than live calls for training. |
| Project time overrun. | No frontend. | Fallback to Excel chart; make model evaluation the priority. |
| Distributed systems complexity. | PegelHub integration stalls. | Use local SQLite/PostgreSQL first; integrate later if time allows. |

---

## 12. Summary of Data Usage Matrix

| Data Category | Specific Fields | Used As | Model Role |
|---------------|---------------|---------|------------|
| **Water level (same station)** | `W` at t-1, t-6, t-24, t-168 | Lagged features | Autoregressive target history |
| **Water level (upstream)** | `W` at upstream gauge, lagged by travel time | Spatial lagged features | Leading wave signal |
| **Discharge** | `Q` at same station | Feature | Flow dynamics, cross-check rating |
| **Official forecast** | `WV` (most likely + CI) | Feature + benchmark | Meta-learning baseline + verification target |
| **Precipitation** | Accumulated rainfall over 3/6/12/24/48/72h | Meteorological feature | Anticipates inflow |
| **Temperature / Snow** | Daily temp, snow depth | Seasonal feature | Spring melt, evaporation |
| **Calendar** | Hour, weekday, day-of-year (sin/cos) | Cyclical features | Seasonal and diurnal patterns |
| **Characteristic values** | MW96, warning levels | Derived binary/class features | Risk stratification |

---

*End of Plan*

*Prepared for PegelHub — TEVS Course Assignment*
