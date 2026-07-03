"""CrewAI tools for data processing, feature engineering, and I/O.

Covers: CSV/Parquet I/O, resampling, gap filling, lag features,
rolling statistics, calendar encoding, rate-of-change, train/test splits.
"""

import json
import os
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field
from crewai.tools import BaseTool

OUTPUT_DIR = os.environ.get("PEGELHUB_OUTPUT_DIR", "output")
DATA_DIR = os.environ.get("PEGELHUB_DATA_DIR", os.path.join(OUTPUT_DIR, "data"))


def _resolve_path(filepath: str) -> str:
    """Resolve a relative path against the data directory."""
    return os.path.join(DATA_DIR, filepath) if not os.path.isabs(filepath) else filepath


def _read_csv_with_validation(filepath: str, required_cols: list[str] | None = None, parse_dates: list[str] | None = None):
    """Read a CSV file with clear existence and column validation.

    Raises FileNotFoundError or ValueError with actionable messages so
    guardrails and callbacks can catch them.
    """
    import pandas as pd
    full_path = _resolve_path(filepath)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"File not found: {filepath} (checked {full_path})")
    try:
        df = pd.read_csv(full_path, parse_dates=parse_dates)
    except Exception as e:
        raise ValueError(f"Could not read CSV {filepath}: {e}")
    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            available = ", ".join(df.columns.tolist())
            raise ValueError(f"Missing required columns in {filepath}: {missing}. Available: {available}")
    return df


# ── File Listing ───────────────────────────────────────────────────────────

class ListDataFilesInput(BaseModel):
    subdir: str = Field(
        default="",
        description="Subdirectory within data/ (e.g. 'raw', 'features').",
    )


class ListDataFilesTool(BaseTool):
    name: str = "list_data_files"
    description: str = "List available data files in the data directory."
    args_schema: type[BaseModel] = ListDataFilesInput

    def _run(self, subdir: str = "") -> str:
        # Strip redundant "data/" prefix agents sometimes add
        if subdir.startswith("data/") or subdir.startswith("data\\"):
            subdir = subdir[5:]
        path = os.path.join(DATA_DIR, subdir) if subdir else DATA_DIR
        if not os.path.isdir(path):
            return json.dumps({"error": f"Directory not found: {path}"})
        files = []
        for f in sorted(os.listdir(path)):
            fp = os.path.join(path, f)
            files.append({
                "name": f,
                "size_bytes": os.path.getsize(fp),
                "modified": datetime.fromtimestamp(os.path.getmtime(fp)).isoformat(),
            })
        return json.dumps({"directory": path, "files": files}, indent=2)


# ── CSV Summary ─────────────────────────────────────────────────────────────

class CSVSummaryInput(BaseModel):
    filepath: str = Field(description="Path to CSV file (relative to data dir).")
    max_rows: int = Field(default=20, description="Preview rows to return.")


class CSVSummaryTool(BaseTool):
    name: str = "read_csv_summary"
    description: str = (
        "Read a CSV file and return a summary: shape, columns, dtypes, "
        "null counts, and first few rows."
    )
    args_schema: type[BaseModel] = CSVSummaryInput

    def _run(self, filepath: str, max_rows: int = 20) -> str:
        try:
            df = _read_csv_with_validation(filepath)
            df = df.head(max_rows + 1000)
        except (FileNotFoundError, ValueError) as e:
            return json.dumps({"error": str(e)})
        except Exception as e:
            return json.dumps({"error": f"Unexpected error reading {filepath}: {e}"})
        summary = {
            "path": full_path,
            "shape": list(df.shape),
            "columns": list(df.columns),
            "dtypes": {col: str(dt) for col, dt in df.dtypes.items()},
            "null_counts": {col: int(df[col].isna().sum()) for col in df.columns},
            "head": df.head(max_rows).to_dict(orient="list"),
        }
        return json.dumps(summary, indent=2, default=str)


# ── Parquet Summary ────────────────────────────────────────────────────────

class ParquetSummaryInput(BaseModel):
    filepath: str = Field(description="Path to Parquet file (relative to data dir).")
    max_rows: int = Field(default=20, description="Preview rows to return.")


class ParquetSummaryTool(BaseTool):
    name: str = "read_parquet_summary"
    description: str = (
        "Read a Parquet file and return a summary: shape, columns, dtypes, "
        "null counts, and first few rows."
    )
    args_schema: type[BaseModel] = ParquetSummaryInput

    def _run(self, filepath: str, max_rows: int = 20) -> str:
        import pandas as pd
        full_path = _resolve_path(filepath)
        if not os.path.exists(full_path):
            return json.dumps({"error": f"File not found: {filepath} (checked {full_path})"})
        try:
            df = pd.read_parquet(full_path)
        except Exception as e:
            return json.dumps({"error": f"Could not read Parquet {filepath}: {e}"})
        summary = {
            "path": full_path,
            "shape": list(df.shape),
            "columns": list(df.columns),
            "dtypes": {col: str(dt) for col, dt in df.dtypes.items()},
            "null_counts": {col: int(df[col].isna().sum()) for col in df.columns},
            "head": df.head(max_rows).to_dict(orient="list"),
        }
        return json.dumps(summary, indent=2, default=str)


# ── Resample ───────────────────────────────────────────────────────────────

class ResampleInput(BaseModel):
    filepath: str = Field(description="Path to CSV file.")
    timestamp_col: str = Field(default="timestamp", description="Timestamp column name.")
    value_cols: Optional[list[str]] = Field(default=None, description="Columns to resample (None=all numeric).")
    freq: str = Field(default="h", description="Pandas freq string: 'h'=hourly, 'D'=daily.")
    method: str = Field(default="mean", description="Aggregation: 'mean', 'last', 'sum'.")


class ResampleTool(BaseTool):
    name: str = "resample_timeseries"
    description: str = (
        "Resample a time-series CSV to a regular frequency. "
        "Handles irregular timestamps by aggregating (mean/last/sum)."
    )
    args_schema: type[BaseModel] = ResampleInput

    def _run(
        self,
        filepath: str,
        timestamp_col: str = "timestamp",
        value_cols: Optional[list[str]] = None,
        freq: str = "h",
        method: str = "mean",
    ) -> str:
        try:
            df = _read_csv_with_validation(filepath, required_cols=[timestamp_col], parse_dates=[timestamp_col])
        except (FileNotFoundError, ValueError) as e:
            return json.dumps({"error": str(e)})
        df = df.set_index(timestamp_col)
        if value_cols:
            df = df[value_cols]
        agg_map = {"mean": "mean", "last": "last", "sum": "sum"}
        if method not in agg_map:
            return json.dumps({"error": f"Unknown method: {method}. Use mean/last/sum."})
        resampled = df.resample(freq).agg(agg_map[method])
        out_path = full_path.replace(".csv", f"_{freq}.csv")
        resampled.to_csv(out_path)
        return json.dumps({
            "input": filepath,
            "output": os.path.relpath(out_path),
            "original_shape": list(df.shape),
            "resampled_shape": list(resampled.shape),
            "freq": freq,
            "method": method,
        })


# ── Gap Filling ────────────────────────────────────────────────────────────

class FillGapsInput(BaseModel):
    filepath: str = Field(description="Path to CSV file.")
    timestamp_col: str = Field(default="timestamp", description="Timestamp column.")
    max_gap_hours: float = Field(default=3.0, description="Max gap to fill (hours). Longer gaps stay NaN.")
    method: str = Field(default="ffill", description="'ffill' or 'interpolate'.")


class FillGapsTool(BaseTool):
    name: str = "fill_gaps"
    description: str = (
        "Fill gaps in time-series data. Gaps shorter than max_gap_hours are "
        "filled via forward-fill or interpolation; longer gaps remain NaN."
    )
    args_schema: type[BaseModel] = FillGapsInput

    def _run(
        self,
        filepath: str,
        timestamp_col: str = "timestamp",
        max_gap_hours: float = 3.0,
        method: str = "ffill",
    ) -> str:
        try:
            df = _read_csv_with_validation(filepath, required_cols=[timestamp_col], parse_dates=[timestamp_col])
        except (FileNotFoundError, ValueError) as e:
            return json.dumps({"error": str(e)})
        nans_before = int(df.isna().sum().sum())
        if method == "ffill":
            df = df.set_index(timestamp_col)
            df = df.ffill(limit=int(max_gap_hours))
            df = df.reset_index()
        elif method == "interpolate":
            df = df.interpolate(method="linear", limit=int(max_gap_hours))
        else:
            return json.dumps({"error": f"Unknown method: {method}"})
        nans_after = int(df.isna().sum().sum())
        out_path = full_path.replace(".csv", "_filled.csv")
        df.to_csv(out_path, index=False)
        return json.dumps({
            "input": filepath,
            "output": os.path.relpath(out_path),
            "nans_before": nans_before,
            "nans_after": nans_after,
            "filled": nans_before - nans_after,
            "method": method,
            "max_gap_hours": max_gap_hours,
        })


# ── Lag Features ────────────────────────────────────────────────────────────

class LagFeaturesInput(BaseModel):
    filepath: str = Field(description="Path to CSV file with hourly data.")
    column: str = Field(description="Column to create lags for.")
    lags: list[int] = Field(
        default=[1, 3, 6, 12, 24, 168],
        description="Lag values in hours.",
    )


class LagFeaturesTool(BaseTool):
    name: str = "compute_lag_features"
    description: str = (
        "Add lag features for a column. Creates {column}_lag_{n} columns "
        "for each lag n (in hours)."
    )
    args_schema: type[BaseModel] = LagFeaturesInput

    def _run(
        self,
        filepath: str,
        column: str,
        lags: list[int] = [1, 3, 6, 12, 24, 168],
    ) -> str:
        try:
            df = _read_csv_with_validation(filepath, required_cols=[column])
        except (FileNotFoundError, ValueError) as e:
            return json.dumps({"error": str(e)})
        new_cols = []
        for lag in lags:
            col_name = f"{column}_lag_{lag}"
            df[col_name] = df[column].shift(lag)
            new_cols.append(col_name)
        out_path = full_path.replace(".csv", "_lagged.csv")
        df.to_csv(out_path, index=False)
        return json.dumps({
            "input": filepath,
            "output": os.path.relpath(out_path),
            "lags_added": new_cols,
            "total_columns": len(df.columns),
        })


# ── Rolling Features ────────────────────────────────────────────────────────

class RollingFeaturesInput(BaseModel):
    filepath: str = Field(description="Path to CSV file.")
    column: str = Field(description="Column to compute rolling stats for.")
    windows: list[int] = Field(default=[3, 6, 24], description="Window sizes in hours.")
    stats: list[str] = Field(default=["mean", "std"], description="Statistics: 'mean', 'std'.")


class RollingFeaturesTool(BaseTool):
    name: str = "compute_rolling_features"
    description: str = (
        "Add rolling window statistics (mean, std) for a column. "
        "Creates {column}_rolling_{stat}_{window}h columns."
    )
    args_schema: type[BaseModel] = RollingFeaturesInput

    def _run(
        self,
        filepath: str,
        column: str,
        windows: list[int] = [3, 6, 24],
        stats: list[str] = ["mean", "std"],
    ) -> str:
        try:
            df = _read_csv_with_validation(filepath, required_cols=[column])
        except (FileNotFoundError, ValueError) as e:
            return json.dumps({"error": str(e)})
        new_cols = []
        for w in windows:
            for s in stats:
                col_name = f"{column}_rolling_{s}_{w}h"
                df[col_name] = df[column].rolling(window=w, min_periods=1).agg(s)
                new_cols.append(col_name)
        out_path = full_path.replace(".csv", "_rolling.csv")
        df.to_csv(out_path, index=False)
        return json.dumps({
            "input": filepath,
            "output": os.path.relpath(out_path),
            "columns_added": new_cols,
            "total_columns": len(df.columns),
        })


# ── Calendar Features ──────────────────────────────────────────────────────

class CalendarFeaturesInput(BaseModel):
    filepath: str = Field(description="Path to CSV file.")
    timestamp_col: str = Field(default="timestamp", description="Timestamp column name.")


class CalendarFeaturesTool(BaseTool):
    name: str = "compute_calendar_features"
    description: str = (
        "Add cyclical calendar features: sin/cos encoded hour, day-of-week, "
        "day-of-year, plus binary is_weekend and is_holiday (Austrian holidays)."
    )
    args_schema: type[BaseModel] = CalendarFeaturesInput

    def _run(self, filepath: str, timestamp_col: str = "timestamp") -> str:
        import numpy as np
        try:
            df = _read_csv_with_validation(filepath, required_cols=[timestamp_col], parse_dates=[timestamp_col])
        except (FileNotFoundError, ValueError) as e:
            return json.dumps({"error": str(e)})
        ts = df[timestamp_col].dt
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
        df["is_holiday"] = df[timestamp_col].dt.strftime("%m-%d").isin(at_holidays).astype(int)
        out_path = full_path.replace(".csv", "_calendar.csv")
        df.to_csv(out_path, index=False)
        new_cols = [
            "hour_sin", "hour_cos", "dow_sin", "dow_cos",
            "doy_sin", "doy_cos", "is_weekend", "is_holiday",
        ]
        return json.dumps({
            "input": filepath,
            "output": os.path.relpath(out_path),
            "columns_added": new_cols,
            "total_columns": len(df.columns),
        })


# ── Rate of Change ─────────────────────────────────────────────────────────

class RateOfChangeInput(BaseModel):
    filepath: str = Field(description="Path to CSV file.")
    column: str = Field(description="Column to differentiate.")


class RateOfChangeTool(BaseTool):
    name: str = "compute_rate_of_change"
    description: str = (
        "Add rate of change (first derivative) and acceleration (second derivative) "
        "for a column. Creates {col}_roc and {col}_accel."
    )
    args_schema: type[BaseModel] = RateOfChangeInput

    def _run(self, filepath: str, column: str) -> str:
        try:
            df = _read_csv_with_validation(filepath, required_cols=[column])
        except (FileNotFoundError, ValueError) as e:
            return json.dumps({"error": str(e)})
        df[f"{column}_roc"] = df[column].diff()
        df[f"{column}_accel"] = df[f"{column}_roc"].diff()
        out_path = full_path.replace(".csv", "_derivatives.csv")
        df.to_csv(out_path, index=False)
        return json.dumps({
            "input": filepath,
            "output": os.path.relpath(out_path),
            "columns_added": [f"{column}_roc", f"{column}_accel"],
        })


# ── Chronological Train/Test Split ─────────────────────────────────────────

class ChronoSplitInput(BaseModel):
    filepath: str = Field(description="Path to CSV file.")
    timestamp_col: str = Field(default="timestamp", description="Timestamp column.")
    train_ratio: float = Field(default=0.7, description="Training fraction.")
    val_ratio: float = Field(default=0.15, description="Validation fraction.")
    test_ratio: float = Field(default=0.15, description="Test fraction.")


class ChronoSplitTool(BaseTool):
    name: str = "train_test_split_chronological"
    description: str = (
        "Split time-series data chronologically into train/val/test sets. "
        "No shuffling — preserves temporal order to prevent data leakage."
    )
    args_schema: type[BaseModel] = ChronoSplitInput

    def _run(
        self,
        filepath: str,
        timestamp_col: str = "timestamp",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> str:
        try:
            df = _read_csv_with_validation(filepath, required_cols=[timestamp_col], parse_dates=[timestamp_col])
        except (FileNotFoundError, ValueError) as e:
            return json.dumps({"error": str(e)})
        df = df.sort_values(timestamp_col).reset_index(drop=True)
        n = len(df)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train = df.iloc[:n_train]
        val = df.iloc[n_train:n_train + n_val]
        test = df.iloc[n_train + n_val:]
        base = full_path.replace(".csv", "")
        train.to_csv(f"{base}_train.csv", index=False)
        val.to_csv(f"{base}_val.csv", index=False)
        test.to_csv(f"{base}_test.csv", index=False)
        return json.dumps({
            "input": filepath,
            "total_rows": n,
            "train_rows": len(train),
            "val_rows": len(val),
            "test_rows": len(test),
            "train_range": [str(train[timestamp_col].iloc[0]), str(train[timestamp_col].iloc[-1])],
            "val_range": [str(val[timestamp_col].iloc[0]), str(val[timestamp_col].iloc[-1])],
            "test_range": [str(test[timestamp_col].iloc[0]), str(test[timestamp_col].iloc[-1])],
            "output_files": [
                os.path.relpath(f"{base}_train.csv"),
                os.path.relpath(f"{base}_val.csv"),
                os.path.relpath(f"{base}_test.csv"),
            ],
        })


# ── Metrics Computation ────────────────────────────────────────────────────

class ComputeMetricsInput(BaseModel):
    actual_filepath: str = Field(description="CSV with actual values.")
    predicted_filepath: str = Field(description="CSV with predicted values.")
    actual_col: str = Field(default="water_level", description="Actual values column.")
    predicted_col: str = Field(default="predicted", description="Predicted values column.")


class ComputeMetricsTool(BaseTool):
    name: str = "compute_metrics"
    description: str = (
        "Compute forecast verification metrics: RMSE, MAE, MAPE, NSE, bias. "
        "Standard hydrological skill metrics for comparing predictions vs actuals."
    )
    args_schema: type[BaseModel] = ComputeMetricsInput

    def _run(
        self,
        actual_filepath: str,
        predicted_filepath: str,
        actual_col: str = "water_level",
        predicted_col: str = "predicted",
    ) -> str:
        import numpy as np
        try:
            y = _read_csv_with_validation(actual_filepath, required_cols=[actual_col])[actual_col].values
            yhat = _read_csv_with_validation(predicted_filepath, required_cols=[predicted_col])[predicted_col].values
        except (FileNotFoundError, ValueError) as e:
            return json.dumps({"error": str(e)})
        mask = ~(np.isnan(y) | np.isnan(yhat))
        y = y[mask]
        yhat = yhat[mask]
        rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
        mae = float(np.mean(np.abs(y - yhat)))
        nonzero = y != 0
        mape = float(np.mean(np.abs((y[nonzero] - yhat[nonzero]) / y[nonzero])) * 100)
        nse = float(1 - np.sum((y - yhat) ** 2) / np.sum((y - np.mean(y)) ** 2))
        bias = float(np.mean(yhat - y))
        return json.dumps({
            "n_observations": int(len(y)),
            "RMSE_cm": round(rmse, 4),
            "MAE_cm": round(mae, 4),
            "MAPE_percent": round(mape, 4),
            "NSE": round(nse, 4),
            "bias_cm": round(bias, 4),
        }, indent=2)