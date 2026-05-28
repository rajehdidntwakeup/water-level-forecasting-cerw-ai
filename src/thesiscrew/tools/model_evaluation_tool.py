"""CrewAI tools for model training, evaluation, and artifact management.

Provides: baseline model fitting, XGBoost/LSTM training stubs,
walk-forward validation, model persistence, and stratified metric computation.
"""

import json
import os
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field
from crewai.tools import BaseTool

DATA_DIR = os.environ.get("PEGELHUB_DATA_DIR", "data")
MODELS_DIR = os.environ.get("PEGELHUB_MODELS_DIR", "models")


# ── Persistence Baseline ───────────────────────────────────────────────────

class PersistenceBaselineInput(BaseModel):
    filepath: str = Field(description="Path to CSV with actual water level data.")
    column: str = Field(default="water_level", description="Column with water level values.")
    horizons: list[int] = Field(
        default=[1, 6, 12, 24, 48],
        description="Forecast horizons in hours.",
    )


class PersistenceBaselineTool(BaseTool):
    name: str = "persistence_baseline"
    description: str = (
        "Compute persistence baseline predictions and metrics. "
        "Predicts water_level at t+h = water_level at t. "
        "Returns RMSE, MAE, NSE per horizon."
    )
    args_schema: type[BaseModel] = PersistenceBaselineInput

    def _run(
        self,
        filepath: str,
        column: str = "water_level",
        horizons: list[int] = [1, 6, 12, 24, 48],
    ) -> str:
        import numpy as np
        import pandas as pd
        full_path = os.path.join(DATA_DIR, filepath) if not os.path.isabs(filepath) else filepath
        df = pd.read_csv(full_path)
        values = df[column].values
        results = {}
        for h in horizons:
            if h >= len(values):
                continue
            y = values[h:]
            yhat = values[:-h] if h > 0 else values
            yhat = yhat[: len(y)]
            y = y[: len(yhat)]
            mask = ~(np.isnan(y) | np.isnan(yhat))
            y_clean = y[mask]
            yhat_clean = yhat[mask]
            rmse = float(np.sqrt(np.mean((y_clean - yhat_clean) ** 2)))
            mae = float(np.mean(np.abs(y_clean - yhat_clean)))
            nse = float(
                1 - np.sum((y_clean - yhat_clean) ** 2)
                / np.sum((y_clean - np.mean(y_clean)) ** 2)
            )
            bias = float(np.mean(yhat_clean - y_clean))
            results[f"h={h}"] = {
                "rmse": round(rmse, 2),
                "mae": round(mae, 2),
                "nse": round(nse, 4),
                "bias": round(bias, 2),
                "n": int(len(y_clean)),
            }
        return json.dumps({"model": "persistence", "metrics": results}, indent=2)


# ── Walk-Forward Validation ────────────────────────────────────────────────

class WalkForwardInput(BaseModel):
    filepath: str = Field(description="Path to CSV with time-series data.")
    timestamp_col: str = Field(default="timestamp", description="Timestamp column.")
    target_col: str = Field(default="water_level", description="Target variable.")
    train_window: int = Field(default=168, description="Training window in hours.")
    test_window: int = Field(default=24, description="Test window in hours.")
    n_splits: int = Field(default=5, description="Number of walk-forward splits.")


class WalkForwardTool(BaseTool):
    name: str = "walk_forward_validation"
    description: str = (
        "Generate walk-forward validation split indices for time-series. "
        "Returns train/test index pairs preserving temporal order."
    )
    args_schema: type[BaseModel] = WalkForwardInput

    def _run(
        self,
        filepath: str,
        timestamp_col: str = "timestamp",
        target_col: str = "water_level",
        train_window: int = 168,
        test_window: int = 24,
        n_splits: int = 5,
    ) -> str:
        import pandas as pd
        full_path = os.path.join(DATA_DIR, filepath) if not os.path.isabs(filepath) else filepath
        df = pd.read_csv(full_path, parse_dates=[timestamp_col])
        n = len(df)
        splits = []
        for i in range(n_splits):
            test_end = n - (n_splits - 1 - i) * test_window
            test_start = test_end - test_window
            train_start = test_start - train_window
            if train_start < 0:
                train_start = 0
            splits.append({
                "fold": i + 1,
                "train_range": f"[{train_start}:{test_start}]",
                "test_range": f"[{test_start}:{test_end}]",
                "train_rows": test_start - train_start,
                "test_rows": test_end - test_start,
            })
        return json.dumps({
            "total_rows": n,
            "train_window": train_window,
            "test_window": test_window,
            "n_splits": n_splits,
            "folds": splits,
        }, indent=2)


# ── Stratified Metrics ─────────────────────────────────────────────────────

class StratifiedMetricsInput(BaseModel):
    actual_filepath: str = Field(description="CSV with actual values.")
    predicted_filepath: str = Field(description="CSV with predicted values.")
    actual_col: str = Field(default="water_level", description="Actual column.")
    predicted_col: str = Field(default="predicted", description="Predicted column.")
    thresholds: Optional[dict[str, float]] = Field(
        default=None,
        description="Flow regime thresholds as key-value pairs. "
        "Keys: low, normal, elevated, warning. Values: water level in cm.",
    )


class StratifiedMetricsTool(BaseTool):
    name: str = "compute_stratified_metrics"
    description: str = (
        "Compute metrics stratified by flow regime (low, normal, elevated, "
        "warning). Evaluates model performance per regime using provided "
        "thresholds or defaults for Korneuburg Danube."
    )
    args_schema: type[BaseModel] = StratifiedMetricsInput

    def _run(
        self,
        actual_filepath: str,
        predicted_filepath: str,
        actual_col: str = "water_level",
        predicted_col: str = "predicted",
        thresholds: Optional[dict[str, float]] = None,
    ) -> str:
        import numpy as np
        import pandas as pd
        if thresholds is None:
            thresholds = {"low": 150, "normal": 210, "elevated": 280, "warning": 350}
        a_path = os.path.join(DATA_DIR, actual_filepath) if not os.path.isabs(actual_filepath) else actual_filepath
        p_path = os.path.join(DATA_DIR, predicted_filepath) if not os.path.isabs(predicted_filepath) else predicted_filepath
        y = pd.read_csv(a_path)[actual_col].values
        yhat = pd.read_csv(p_path)[predicted_col].values
        mask = ~(np.isnan(y) | np.isnan(yhat))
        y = y[mask]
        yhat = yhat[mask]

        def _metrics(actual, pred):
            rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
            mae = float(np.mean(np.abs(actual - pred)))
            nse = float(1 - np.sum((actual - pred) ** 2) / max(np.sum((actual - np.mean(actual)) ** 2), 1e-10))
            bias = float(np.mean(pred - actual))
            return {"rmse": round(rmse, 2), "mae": round(mae, 2), "nse": round(nse, 4), "bias": round(bias, 2), "n": int(len(actual))}

        overall = _metrics(y, yhat)
        regimes = {}
        keys = sorted(thresholds.keys(), key=lambda k: thresholds[k])
        for i, key in enumerate(keys):
            lower = thresholds[key]
            upper = thresholds[keys[i + 1]] if i + 1 < len(keys) else float("inf")
            idx = (y >= lower) & (y < upper)
            if idx.sum() > 0:
                regimes[f"{key}({lower}-{upper})"] = _metrics(y[idx], yhat[idx])
        return json.dumps({"overall": overall, "by_regime": regimes, "thresholds": thresholds}, indent=2)


# ── Model Registry ─────────────────────────────────────────────────────────

class RegisterModelInput(BaseModel):
    model_path: str = Field(description="Path to saved model artifact.")
    description: str = Field(default="", description="Model description (architecture, hyperparams).")
    model_type: str = Field(default="xgboost", description="Model type: xgboost, lstm, arima, persistence.")


class RegisterModelTool(BaseTool):
    name: str = "register_model"
    description: str = (
        "Register a trained model artifact in the model manifest. "
        "Tracks model type, description, and timestamp."
    )
    args_schema: type[BaseModel] = RegisterModelInput

    def _run(
        self,
        model_path: str,
        description: str = "",
        model_type: str = "xgboost",
    ) -> str:
        os.makedirs(MODELS_DIR, exist_ok=True)
        manifest_path = os.path.join(MODELS_DIR, "model_manifest.json")
        manifest = []
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                manifest = json.load(f)
        entry = {
            "path": model_path,
            "model_type": model_type,
            "description": description,
            "saved_at": datetime.now().isoformat(),
        }
        manifest.append(entry)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        return json.dumps({"status": "registered", "entry": entry}, indent=2)


class ListModelsInput(BaseModel):
    pass


class ListModelsTool(BaseTool):
    name: str = "list_models"
    description: str = "List all registered model artifacts from the manifest."

    def _run(self) -> str:
        manifest_path = os.path.join(MODELS_DIR, "model_manifest.json")
        if not os.path.exists(manifest_path):
            return json.dumps({"models": []})
        with open(manifest_path) as f:
            return json.dumps({"models": json.load(f)}, indent=2)