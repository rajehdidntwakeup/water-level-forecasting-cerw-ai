# Water Level Forecasting — Crew Instructions

## Target

Forecast water levels at **Korneuburg (Donau/Danube, Austria)** using publicly
available hydrometric and meteorological data.

## Primary Station

| Attribute | Value |
|-----------|-------|
| Station | Korneuburg |
| River | Donau (Danube) |
| Country | Austria |
| Latitude | 48.345 |
| Longitude | 16.337 |
| eHYD ID | 207273 |

## Forecast Horizons

Nowcast (+1h, +6h, +12h), Medium-term (+24h, +48h), Long-term (+72h, +168h).

## Data Sources

1. **Pegelonline (WSV, Germany)** — REST API, no auth. Station metadata, water
   levels (W), discharge (Q), temperature (WT), forecasts (WV).
2. **eHYD / DORIS (Austria)** — Historical water levels, discharge, characteristic
   values (MW96, warning thresholds).
3. **Open-Meteo** — Historical and forecast weather (precipitation, temperature,
   snow depth). No API key for non-commercial use.

## Key Requirements

- Verify data against **official forecasts** (WV series from Pegelonline).
- Use **chronological** train/val/test splits (70/15/15). No random splits.
- Compute **NSE, RMSE, MAE, MAPE, bias** as primary metrics.
- Stratify evaluation by **flow regime** (low, normal, elevated, warning, alert).
- If time is short, prioritize model accuracy over frontend polish.

## Output

- Trained models (XGBoost, LSTM, baselines) with predictions on test set.
- Comparative metrics vs official forecast and baselines.
- Lightweight forecast visualization (web dashboard or Excel).
- Complete documentation (methodology, features, results, limitations).