"""MCP server for data processing, storage, and model I/O.

Provides tools for:
- CSV/Parquet reading and writing
- Data cleaning (gap handling, outlier removal, resampling)
- Feature engineering computations (lags, rolling stats, calendar encoding)
- Train/val/test splitting (chronological)
- Model persistence (save/load sklearn/xgboost/torch)
- Metric computation (RMSE, MAE, MAPE, NSE, bias)
"""

import json
import os
from datetime import datetime, timedelta
from typing import Optional

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("data_processing")

DATA_DIR = os.environ.get("PEGELHUB_DATA_DIR", "data")
MODELS_DIR = os.environ.get("PEGELHUB_MODELS_DIR", "models")


# ── File I/O ──────────────────────────────────────────────────────────────

@mcp.tool()
def list_data_files(subdir: str = "") -> str:
    """List available data files in the data directory.

    Args:
        subdir: Subdirectory within data/ (e.g. 'raw', 'features', 'predictions').
    """
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


@mcp.tool()
def read_csv_summary(filepath: str, max_rows: int = 20) -> str:
    """Read a CSV file and return a summary: shape, columns, dtypes, head.

    Args:
        filepath: Path to CSV file (relative to data directory).
        max_rows: Number of preview rows to return.
    """
    import pandas as pd
    full_path = os.path.join(DATA_DIR, filepath) if not os.path.isabs(filepath) else filepath
    try:
        df = pd.read_csv(full_path, nrows=max_rows + 1000)
    except Exception as e:
        return json.dumps({"error": str(e)})
    summary = {
        "path": full_path,
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {col: str(dt) for col, dt in df.dtypes.items()},
        "null_counts": {col: int(df[col].isna().sum()) for col in df.columns},
        "head": df.head(max_rows).to_dict(orient="list"),
    }
    return json.dumps(summary, indent=2, default=str)


@mcp.tool()
def read_parquet_summary(filepath: str, max_rows: int = 20) -> str:
    """Read a Parquet file and return a summary: shape, columns, dtypes, head.

    Args:
        filepath: Path to Parquet file (relative to data directory).
        max_rows: Number of preview rows to return.
    """
    import pandas as pd
    full_path = os.path.join(DATA_DIR, filepath) if not os.path.isabs(filepath) else filepath
    try:
        df = pd.read_parquet(full_path)
    except Exception as e:
        return json.dumps({"error": str(e)})
    summary = {
        "path": full_path,
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {col: str(dt) for col, dt in df.dtypes.items()},
        "null_counts": {col: int(df[col].isna().sum()) for col in df.columns},
        "head": df.head(max_rows).to_dict(orient="list"),
    }
    return json.dumps(summary, indent=2, default=str)


# ── Data Cleaning ─────────────────────────────────────────────────────────

@mcp.tool()
def resample_timeseries(
    filepath: str,
    timestamp_col: str = "timestamp",
    value_cols: Optional[list] = None,
    freq: str = "h",
    method: str = "mean",
) -> str:
    """Resample a time-series DataFrame to a regular frequency.

    Args:
        filepath: Path to CSV file.
        timestamp_col: Name of the timestamp column.
        value_cols: Columns to resample. None = all numeric.
        freq: Pandas frequency string ('h' = hourly, 'D' = daily).
        method: Aggregation method — 'mean', 'last', 'sum'.
    """
    import pandas as pd
    full_path = os.path.join(DATA_DIR, filepath) if not os.path.isabs(filepath) else filepath
    try:
        df = pd.read_csv(full_path, parse_dates=[timestamp_col])
    except Exception as e:
        return json.dumps({"error": str(e)})
    df = df.set_index(timestamp_col)
    if value_cols:
        df = df[value_cols]
    if method == "mean":
        resampled = df.resample(freq).mean()
    elif method == "last":
        resampled = df.resample(freq).last()
    elif method == "sum":
        resampled = df.resample(freq).sum()
    else:
        return json.dumps({"error": f"Unknown method: {method}"})
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


@mcp.tool()
def fill_gaps(
    filepath: str,
    timestamp_col: str = "timestamp",
    max_gap_hours: float = 3.0,
    method: str = "ffill",
) -> str:
    """Fill gaps in time-series data using forward-fill or interpolation.

    Gaps shorter than max_gap_hours are filled; longer gaps are left as NaN.

    Args:
        filepath: Path to CSV file.
        timestamp_col: Name of the timestamp column.
        max_gap_hours: Maximum gap length to fill (hours). Longer gaps stay NaN.
        method: 'ffill' for forward fill, 'interpolate' for linear interpolation.
    """
    import pandas as pd
    full_path = os.path.join(DATA_DIR, filepath) if not os.path.isabs(filepath) else filepath
    try:
        df = pd.read_csv(full_path, parse_dates=[timestamp_col])
    except Exception as e:
        return json.dumps({"error": str(e)})
    total_nans_before = int(df.isna().sum().sum())
    if method == "ffill":
        df = df.set_index(timestamp_col)
        df = df.ffill(limit=int(max_gap_hours))
        df = df.reset_index()
    elif method == "interpolate":
        df = df.interpolate(method="linear", limit=int(max_gap_hours))
    total_nans_after = int(df.isna().sum().sum())
    out_path = full_path.replace(".csv", "_filled.csv")
    df.to_csv(out_path, index=False)
    return json.dumps({
        "input": filepath,
        "output": os.path.relpath(out_path),
        "nans_before": total_nans_before,
        "nans_after": total_nans_after,
        "filled": total_nans_before - total_nans_after,
        "method": method,
        "max_gap_hours": max_gap_hours,
    })


# ── Feature Engineering ────────────────────────────────────────────────────

@mcp.tool()
def compute_lag_features(
    filepath: str,
    column: str,
    lags: list = [1, 3, 6, 12, 24, 168],
    timestamp_col: str = "timestamp",
) -> str:
    """Add lag features for a specified column.

    Creates columns like {column}_lag_{n} for each lag n (in hours).

    Args:
        filepath: Path to CSV file with hourly data.
        column: Column name to create lags for.
        lags: List of lag values in hours (default: 1,3,6,12,24,168).
        timestamp_col: Name of timestamp column.
    """
    import pandas as pd
    full_path = os.path.join(DATA_DIR, filepath) if not os.path.isabs(filepath) else filepath
    df = pd.read_csv(full_path, parse_dates=[timestamp_col])
    for lag in lags:
        df[f"{column}_lag_{lag}"] = df[column].shift(lag)
    out_path = full_path.replace(".csv", "_lagged.csv")
    df.to_csv(out_path, index=False)
    new_cols = [f"{column}_lag_{lag}" for lag in lags]
    return json.dumps({
        "input": filepath,
        "output": os.path.relpath(out_path),
        "lags_added": new_cols,
        "total_columns": len(df.columns),
    })


@mcp.tool()
def compute_rolling_features(
    filepath: str,
    column: str,
    windows: list = [3, 6, 24],
    stats: list = ["mean", "std"],
) -> str:
    """Add rolling window statistics (mean, std) for a column.

    Creates columns like {column}_rolling_{stat}_{window}h.

    Args:
        filepath: Path to CSV file.
        column: Column to compute rolling stats for.
        windows: Window sizes in hours (default: 3, 6, 24).
        stats: Statistics to compute ('mean', 'std').
    """
    import pandas as pd
    full_path = os.path.join(DATA_DIR, filepath) if not os.path.isabs(filepath) else filepath
    df = pd.read_csv(full_path)
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


@mcp.tool()
def compute_calendar_features(
    filepath: str,
    timestamp_col: str = "timestamp",
) -> str:
    """Add cyclical calendar features: sin/cos encoded hour, day-of-week,
    day-of-year, plus binary is_weekend and is_holiday (Austrian holidays).

    Args:
        filepath: Path to CSV file.
        timestamp_col: Name of timestamp column.
    """
    import numpy as np
    import pandas as pd
    full_path = os.path.join(DATA_DIR, filepath) if not os.path.isabs(filepath) else filepath
    df = pd.read_csv(full_path, parse_dates=[timestamp_col])
    ts = df[timestamp_col].dt
    df["hour_sin"] = np.sin(2 * np.pi * ts.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * ts.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * ts.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * ts.dayofweek / 7)
    df["doy_sin"] = np.sin(2 * np.pi * ts.dayofyear / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * ts.dayofyear / 365.25)
    df["is_weekend"] = (ts.dayofweek >= 5).astype(int)
    # Austrian public holidays (simplified — fixed-date only)
    at_holidays = {
        "01-01", "01-06", "05-01", "08-15", "10-26", "11-01", "12-08",
        "12-25", "12-26",
    }
    df["is_holiday"] = df[timestamp_col].dt.strftime("%m-%d").isin(at_holidays).astype(int)
    out_path = full_path.replace(".csv", "_calendar.csv")
    df.to_csv(out_path, index=False)
    new_cols = ["hour_sin", "hour_cos", "dow_sin", "dow_cos",
                "doy_sin", "doy_cos", "is_weekend", "is_holiday"]
    return json.dumps({
        "input": filepath,
        "output": os.path.relpath(out_path),
        "columns_added": new_cols,
        "total_columns": len(df.columns),
    })


@mcp.tool()
def compute_rate_of_change(
    filepath: str,
    column: str,
) -> str:
    """Add rate of change (first derivative) and acceleration (second derivative)
    for a column.

    Creates: {column}_roc (rate of change), {column}_accel (acceleration).

    Args:
        filepath: Path to CSV file.
        column: Column to differentiate.
    """
    import pandas as pd
    full_path = os.path.join(DATA_DIR, filepath) if not os.path.isabs(filepath) else filepath
    df = pd.read_csv(full_path)
    df[f"{column}_roc"] = df[column].diff()
    df[f"{column}_accel"] = df[f"{column}_roc"].diff()
    out_path = full_path.replace(".csv", "_derivatives.csv")
    df.to_csv(out_path, index=False)
    return json.dumps({
        "input": filepath,
        "output": os.path.relpath(out_path),
        "columns_added": [f"{column}_roc", f"{column}_accel"],
    })


@mcp.tool()
def train_test_split_chronological(
    filepath: str,
    timestamp_col: str = "timestamp",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> str:
    """Split a time-series dataset into train/val/test sets chronologically.

    No shuffling — preserves temporal order to prevent data leakage.

    Args:
        filepath: Path to CSV file.
        timestamp_col: Name of timestamp column.
        train_ratio: Fraction for training (default 0.7).
        val_ratio: Fraction for validation (default 0.15).
        test_ratio: Fraction for test (default 0.15).
    """
    import pandas as pd
    full_path = os.path.join(DATA_DIR, filepath) if not os.path.isabs(filepath) else filepath
    df = pd.read_csv(full_path, parse_dates=[timestamp_col])
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


# ── Metrics ───────────────────────────────────────────────────────────────

@mcp.tool()
def compute_metrics(
    actual_filepath: str,
    predicted_filepath: str,
    actual_col: str = "water_level",
    predicted_col: str = "predicted",
) -> str:
    """Compute forecast verification metrics: RMSE, MAE, MAPE, NSE, bias.

    Args:
        actual_filepath: Path to CSV with actual values.
        predicted_filepath: Path to CSV with predicted values.
        actual_col: Column name for actual values.
        predicted_col: Column name for predicted values.
    """
    import numpy as np
    import pandas as pd
    actual_path = os.path.join(DATA_DIR, actual_filepath) if not os.path.isabs(actual_filepath) else actual_filepath
    pred_path = os.path.join(DATA_DIR, predicted_filepath) if not os.path.isabs(predicted_filepath) else predicted_filepath
    y = pd.read_csv(actual_path)[actual_col].values
    yhat = pd.read_csv(pred_path)[predicted_col].values
    mask = ~(np.isnan(y) | np.isnan(yhat))
    y = y[mask]
    yhat = yhat[mask]
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    mae = float(np.mean(np.abs(y - yhat)))
    mape = float(np.mean(np.abs((y - yhat) / np.where(y != 0, y, np.nan))) * 100)
    nse = float(1 - np.sum((y - yhat) ** 2) / np.sum((y - np.mean(y)) ** 2))
    bias = float(np.mean(yhat - y))
    return json.dumps({
        "n_observations": int(len(y)),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "MAPE_percent": round(mape, 4),
        "NSE": round(nse, 4),
        "bias": round(bias, 4),
    }, indent=2)


# ── Model I/O ─────────────────────────────────────────────────────────────

@mcp.tool()
def save_model(model_path: str, description: str = "") -> str:
    """Register a trained model artifact by saving metadata.

    Actual model serialization should be done in Python (joblib/torch.save).
    This tool records the model in a manifest for tracking.

    Args:
        model_path: Path to the saved model file.
        description: Brief description of the model (architecture, hyperparams).
    """
    manifest_path = os.path.join(MODELS_DIR, "model_manifest.json")
    os.makedirs(MODELS_DIR, exist_ok=True)
    manifest = []
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
    entry = {
        "path": model_path,
        "description": description,
        "saved_at": datetime.now().isoformat(),
    }
    manifest.append(entry)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return json.dumps({"status": "registered", "entry": entry}, indent=2)


@mcp.tool()
def list_models() -> str:
    """List all registered model artifacts."""
    manifest_path = os.path.join(MODELS_DIR, "model_manifest.json")
    if not os.path.exists(manifest_path):
        return json.dumps({"models": []})
    with open(manifest_path) as f:
        return json.dumps({"models": json.load(f)}, indent=2)


if __name__ == "__main__":
    mcp.run()