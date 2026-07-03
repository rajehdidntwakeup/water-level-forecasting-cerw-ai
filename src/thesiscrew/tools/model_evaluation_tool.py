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

OUTPUT_DIR = os.environ.get("PEGELHUB_OUTPUT_DIR", "output")
DATA_DIR = os.environ.get("PEGELHUB_DATA_DIR", os.path.join(OUTPUT_DIR, "data"))
MODELS_DIR = os.environ.get("PEGELHUB_MODELS_DIR", os.path.join(OUTPUT_DIR, "models"))


def _resolve_data_path(filepath: str) -> str:
    return os.path.join(DATA_DIR, filepath) if not os.path.isabs(filepath) else filepath


def _read_data_csv(filepath: str, required_cols: list[str]):
    full_path = _resolve_data_path(filepath)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"File not found: {filepath} (checked {full_path})")
    try:
        import pandas as pd
        df = pd.read_csv(full_path)
    except Exception as e:
        raise ValueError(f"Could not read CSV {filepath}: {e}")
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        available = ", ".join(df.columns.tolist())
        raise ValueError(f"Missing required columns in {filepath}: {missing}. Available: {available}")
    return df


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
        try:
            df = _read_data_csv(filepath, required_cols=[column])
        except (FileNotFoundError, ValueError) as e:
            return json.dumps({"error": str(e)})
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
        try:
            import pandas as pd
            df = _read_data_csv(filepath, required_cols=[timestamp_col, target_col])
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        except (FileNotFoundError, ValueError) as e:
            return json.dumps({"error": str(e)})
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
    thresholds_json: Optional[str] = Field(
        default=None,
        description="JSON string with flow regime thresholds, e.g. "
        "'{\"low\": 150, \"normal\": 210, \"elevated\": 280, \"warning\": 350}'. "
        "Omit to use defaults for Korneuburg Danube.",
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
        thresholds_json: Optional[str] = None,
    ) -> str:
        import numpy as np
        if thresholds_json is None:
            thresholds = {"low": 150, "normal": 210, "elevated": 280, "warning": 350}
        else:
            thresholds = json.loads(thresholds_json)
        try:
            y = _read_data_csv(actual_filepath, required_cols=[actual_col])[actual_col].values
            yhat = _read_data_csv(predicted_filepath, required_cols=[predicted_col])[predicted_col].values
        except (FileNotFoundError, ValueError) as e:
            return json.dumps({"error": str(e)})
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
    model_description: str = Field(default="", description="Short model description (architecture, hyperparams).")
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
        model_description: str = "",
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
            "description": model_description,
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
    args_schema: type[BaseModel] = ListModelsInput

    def _run(self) -> str:
        manifest_path = os.path.join(MODELS_DIR, "model_manifest.json")
        if not os.path.exists(manifest_path):
            return json.dumps({"models": []})
        with open(manifest_path) as f:
            return json.dumps({"models": json.load(f)}, indent=2)


# ── Gradient Boosting Forecaster ───────────────────────────────────────────

class TrainGBMInput(BaseModel):
    train_path: str = Field(
        default="features/feature_matrix_train.csv",
        description="Training CSV relative to data dir.",
    )
    test_path: str = Field(
        default="features/feature_matrix_test.csv",
        description="Test CSV relative to data dir.",
    )
    horizons: list[int] = Field(
        default=[1, 6, 12, 24, 48, 72, 168],
        description="Forecast horizons to train models for.",
    )
    target_prefix: str = Field(default="target_", description="Prefix for target columns.")


class TrainGradientBoostingTool(BaseTool):
    name: str = "train_gradient_boosting_forecaster"
    description: str = (
        "Train a scikit-learn HistGradientBoostingRegressor for each forecast "
        "horizon using the engineered feature matrix. Saves one model pickle and "
        "one predictions CSV per horizon under output/models. Returns RMSE, MAE, "
        "NSE per horizon."
    )
    args_schema: type[BaseModel] = TrainGBMInput

    def _run(
        self,
        train_path: str = "features/feature_matrix_train.csv",
        test_path: str = "features/feature_matrix_test.csv",
        horizons: list[int] = [1, 6, 12, 24, 48, 72, 168],
        target_prefix: str = "target_",
    ) -> str:
        import numpy as np
        import pandas as pd
        import pickle
        from sklearn.ensemble import HistGradientBoostingRegressor

        train_full = _resolve_data_path(train_path)
        test_full = _resolve_data_path(test_path)
        if not os.path.exists(train_full):
            return json.dumps({"error": f"Train file not found: {train_full}"})
        if not os.path.exists(test_full):
            return json.dumps({"error": f"Test file not found: {test_full}"})

        train_df = pd.read_csv(train_full)
        test_df = pd.read_csv(test_full)

        if "timestamp" not in train_df.columns:
            return json.dumps({"error": "timestamp column required in feature matrix"})

        # Feature columns: numeric, not targets, not timestamp.
        feature_cols = [
            c for c in train_df.columns
            if c != "timestamp" and not c.startswith(target_prefix)
        ]
        if not feature_cols:
            return json.dumps({"error": "No feature columns found in feature matrix"})

        os.makedirs(MODELS_DIR, exist_ok=True)
        metrics = {}
        for h in horizons:
            target_col = f"{target_prefix}{h}h"
            if target_col not in train_df.columns or target_col not in test_df.columns:
                metrics[f"h={h}"] = {"error": f"Target column {target_col} missing"}
                continue

            X_train = train_df[feature_cols].values
            y_train = train_df[target_col].values
            X_test = test_df[feature_cols].values
            y_test = test_df[target_col].values

            mask_train = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
            mask_test = ~(np.isnan(X_test).any(axis=1) | np.isnan(y_test))
            X_train = X_train[mask_train]
            y_train = y_train[mask_train]
            X_test = X_test[mask_test]
            y_test = y_test[mask_test]

            if len(X_train) == 0 or len(X_test) == 0:
                metrics[f"h={h}"] = {"error": "No valid rows after dropping NaNs"}
                continue

            model = HistGradientBoostingRegressor(max_iter=200, early_stopping=True, random_state=42)
            model.fit(X_train, y_train)
            yhat = model.predict(X_test)

            rmse = float(np.sqrt(np.mean((y_test - yhat) ** 2)))
            mae = float(np.mean(np.abs(y_test - yhat)))
            nse = float(1 - np.sum((y_test - yhat) ** 2) / max(np.sum((y_test - np.mean(y_test)) ** 2), 1e-10))
            bias = float(np.mean(yhat - y_test))

            pred_df = pd.DataFrame({
                "timestamp": test_df.loc[mask_test, "timestamp"].values,
                "actual": y_test,
                "predicted": yhat,
            })
            pred_path = os.path.join(MODELS_DIR, f"gbm_predictions_h{h}.csv")
            pred_df.to_csv(pred_path, index=False)

            model_path = os.path.join(MODELS_DIR, f"gbm_model_h{h}.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            metrics[f"h={h}"] = {
                "rmse": round(rmse, 2),
                "mae": round(mae, 2),
                "nse": round(nse, 4),
                "bias": round(bias, 2),
                "n": int(len(y_test)),
                "predictions": pred_path,
                "model": model_path,
            }

        return json.dumps({"model": "gradient_boosting", "metrics": metrics}, indent=2, default=str)