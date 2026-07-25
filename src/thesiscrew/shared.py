"""Shared helpers, models, guardrails, and callbacks for thesiscrew crews.

This module contains the pieces used by both the training crew and the inference
crew so the two pipeline classes stay DRY.
"""

from crewai import TaskOutput
from pydantic import BaseModel, Field
from typing import Tuple, Any, Dict, List

import json
import os
import sys
from datetime import datetime

# Load project .env so BASE_URL and MODEL are available
from dotenv import load_dotenv
load_dotenv(".env")

from thesiscrew.tools.dataset_tool import (
    BuildKorneuburgDatasetTool,
    BuildFeatureMatrixTool,
)
from thesiscrew.tools.model_evaluation_tool import (
    TrainGradientBoostingTool,
    PersistenceBaselineTool,
    WalkForwardTool,
    StratifiedMetricsTool,
)
from thesiscrew.tools.data_processing_tool import ComputeMetricsTool

OUTPUT_DIR = os.environ.get("PEGELHUB_OUTPUT_DIR", "output")
DATA_DIR = os.environ.get("PEGELHUB_DATA_DIR", os.path.join(OUTPUT_DIR, "data"))
MODELS_DIR = os.environ.get("PEGELHUB_MODELS_DIR", os.path.join(OUTPUT_DIR, "models"))
ARTIFACTS_DIR = os.path.join(OUTPUT_DIR, "artifacts")

# If the project .env points to a local endpoint (e.g. Ollama), configure LiteLLM
# so Agent(llm=...) calls route there instead of OpenAI.
BASE_URL = os.environ.get("BASE_URL", "")
if BASE_URL:
    import litellm
    litellm.api_base = BASE_URL
    # CrewAI/OpenAI provider needs this to form a valid URL when a non-OpenAI
    # base URL is in use.
    if "openai" not in BASE_URL.lower():
        os.environ.setdefault("OPENAI_API_BASE", BASE_URL)

# Default model: prefer MODEL from .env, fall back to OpenAI models.
DEFAULT_LLM = os.environ.get("MODEL", "openai/gpt-4o-mini")


def _ollama_model_name(model: str) -> str | None:
    """Strip the 'ollama/' prefix from a model string, if present."""
    if model.startswith("ollama/"):
        return model.split("ollama/", 1)[1]
    return None


def validate_ollama_model(model: str = DEFAULT_LLM, base_url: str = BASE_URL) -> None:
    """Fail fast if the configured Ollama model is not available or unresponsive.

    Avoids long timeouts when a model name is misspelled, not pulled, or a cloud
    endpoint is hanging.
    """
    ollama_name = _ollama_model_name(model)
    if not ollama_name or not base_url:
        return
    import urllib.request
    import json as _json
    try:
        req = urllib.request.Request(
            f"{base_url}/api/tags",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = _json.loads(resp.read().decode())
        available = {m.get("name", "") for m in data.get("models", [])}
        if ollama_name not in available:
            raise RuntimeError(
                f"Configured Ollama model '{ollama_name}' is not available at {base_url}. "
                f"Installed models: {sorted(available)}. "
                f"Either pull it with 'ollama pull {ollama_name}' or update MODEL in .env."
            )
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(
            f"Could not verify Ollama model availability at {base_url}: {e}. "
            f"Make sure Ollama is running and the model '{ollama_name}' is pulled."
        )

    # Preflight chat check: cloud stubs (e.g. glm-5.2:cloud) can be listed but
    # still hang on actual inference. Send a tiny prompt and ensure we get a
    # response within a short window before spending minutes on the full crew.
    try:
        chat_req = urllib.request.Request(
            f"{base_url}/api/chat",
            data=_json.dumps({
                "model": ollama_name,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        with urllib.request.urlopen(chat_req, timeout=30) as resp:
            chat_data = _json.loads(resp.read().decode())
        if not chat_data.get("message", {}).get("content"):
            raise RuntimeError(
                f"Ollama model '{ollama_name}' returned an empty response during preflight."
            )
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(
            f"Ollama model '{ollama_name}' failed the preflight chat test at {base_url}: {e}. "
            f"If you are using an Ollama Cloud model (tag ':cloud'), the remote endpoint "
            f"may be temporarily unavailable. Retry later, increase LITELLM_REQUEST_TIMEOUT, "
            f"or switch to a local model."
        )


# ---------------------------------------------------------------------------
# Structured output models
# ---------------------------------------------------------------------------

class HorizonMetrics(BaseModel):
    rmse: float = Field(default=0.0, description="Root mean squared error")
    mae: float = Field(default=0.0, description="Mean absolute error")
    nse: float = Field(default=0.0, description="Nash-Sutcliffe efficiency")
    bias: float = Field(default=0.0, description="Mean forecast bias")


class FeatureManifest(BaseModel):
    feature_files: List[str] = Field(default_factory=list, description="Paths to produced feature files")
    target_column: str = Field(default="water_level", description="Name of the target column")
    horizon_hours: List[int] = Field(default_factory=list, description="Forecast horizons in hours")
    split_dates: Dict[str, str] = Field(default_factory=dict, description="Train/validation/test split date boundaries")


class BaselineMetrics(BaseModel):
    model: str = Field(default="", description="Baseline model name")
    horizons: Dict[str, HorizonMetrics] = Field(default_factory=dict, description="Metrics per horizon")


class VerificationReport(BaseModel):
    best_model: str = Field(default="", description="Best performing model name")
    best_horizon: int = Field(default=-1, description="Best performing forecast horizon")
    beats_official: bool = Field(default=False, description="Whether AI beats official forecast")
    metrics_summary: str = Field(default="", description="Concise metrics summary")


class ArtifactManifest(BaseModel):
    models: List[str] = Field(default_factory=list)
    features: List[str] = Field(default_factory=list)
    reports: List[str] = Field(default_factory=list)
    metadata: Dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------

def guardrail_contains_sections(output: TaskOutput, required: list[str]) -> Tuple[bool, Any]:
    text = (output.raw or "").lower()
    missing = [r for r in required if r.lower() not in text]
    if missing:
        return (False, f"Missing required sections: {', '.join(missing)}")
    return (True, output.raw)


def validate_discovery_output(output: TaskOutput) -> Tuple[bool, Any]:
    required = ["pegelonline", "ehyd", "open-meteo", "completeness"]
    return guardrail_contains_sections(output, required)


def validate_feature_output(output: TaskOutput) -> Tuple[bool, Any]:
    required = ["feature", "lag", "train", "validation", "test"]
    ok, msg = guardrail_contains_sections(output, required)
    if not ok:
        return (ok, msg)
    if "feature" not in output.raw.lower():
        return (False, "Feature engineering output does not mention a feature file or manifest.")
    return (True, output.raw)


def validate_baseline_output(output: TaskOutput) -> Tuple[bool, Any]:
    required = ["persistence", "rmse", "mae", "nse"]
    return guardrail_contains_sections(output, required)


def validate_verification_output(output: TaskOutput) -> Tuple[bool, Any]:
    required = ["rmse", "mae", "nse", "walk-forward", "stratified"]
    return guardrail_contains_sections(output, required)


def validate_report_output(output: TaskOutput) -> Tuple[bool, Any]:
    required = [
        "executive summary",
        "data discovery",
        "feature engineering",
        "model development",
        "verification",
    ]
    return guardrail_contains_sections(output, required)


def _safe_print(message: str) -> None:
    """Print a message, replacing characters the console cannot encode."""
    try:
        print(message)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding if sys.stdout else "utf-8"
        print(message.encode(encoding or "utf-8", errors="replace").decode(encoding or "utf-8"))


# ---------------------------------------------------------------------------
# Callback mixin
# ---------------------------------------------------------------------------

class CrewCallbacks:
    """Mixin providing input validation, logging, checkpointing, and rendering."""

    def _validate_inputs(self, inputs: dict) -> dict:
        """Validate required inputs and seed output directories before kickoff."""
        validate_ollama_model()
        required = [
            "primary_station",
            "forecast_horizons_hours",
            "data_sources",
            "target_variable",
        ]
        for key in required:
            if key not in inputs:
                raise ValueError(f"Missing required input: {key}")

        subdirs = [
            "",
            "data",
            "data/raw",
            "data/features",
            "data/processed",
            "data/models",
            "models",
            "artifacts",
            ".cache",
            ".checkpoints",
        ]
        for subdir in subdirs:
            os.makedirs(os.path.join(OUTPUT_DIR, subdir), exist_ok=True)

        # Pre-fetch a real-world dataset and feature matrix so downstream agents
        # do not stall when an LLM agent only writes a narrative report. The
        # tools are idempotent and disk-cached for 24h.
        dataset_path = os.path.join(DATA_DIR, "processed", "korneuburg_hourly.csv")
        if not os.path.exists(dataset_path):
            try:
                _safe_print("[preflight] Building Korneuburg hourly dataset...")
                result = BuildKorneuburgDatasetTool()._run(days=30)
                _safe_print(f"[preflight] {result}")
            except Exception as e:
                _safe_print(f"[preflight] Dataset build warning (crew will continue): {e}")

        feature_path = os.path.join(DATA_DIR, "features", "feature_matrix.csv")
        if os.path.exists(dataset_path) and not os.path.exists(feature_path):
            try:
                _safe_print("[preflight] Building feature matrix...")
                result = BuildFeatureMatrixTool()._run()
                _safe_print(f"[preflight] {result}")
            except Exception as e:
                _safe_print(f"[preflight] Feature matrix warning (crew will continue): {e}")

        pred_path = os.path.join(MODELS_DIR, "gbm_predictions_h1.csv")
        if os.path.exists(feature_path) and not os.path.exists(pred_path):
            try:
                _safe_print("[preflight] Training gradient boosting forecaster...")
                result = TrainGradientBoostingTool()._run()
                _safe_print(f"[preflight] {result}")
            except Exception as e:
                _safe_print(f"[preflight] Model training warning (crew will continue): {e}")

        self._preflight_verification()

        return inputs

    def _preflight_verification(self) -> None:
        """Pre-compute verification numbers so the LLM only synthesises them."""
        import json as _json

        dataset_path = os.path.join(DATA_DIR, "processed", "korneuburg_hourly.csv")
        feature_path = os.path.join(DATA_DIR, "features", "feature_matrix.csv")
        test_path = os.path.join(DATA_DIR, "features", "feature_matrix_test.csv")
        verification_path = os.path.join(MODELS_DIR, "verification_inputs.json")

        if not os.path.exists(feature_path):
            return

        os.makedirs(MODELS_DIR, exist_ok=True)

        # If already computed, refresh only if predictions are newer.
        if os.path.exists(verification_path):
            try:
                mtime_ver = os.path.getmtime(verification_path)
                mtime_pred = os.path.getmtime(os.path.join(MODELS_DIR, "gbm_predictions_h1.csv")) \
                    if os.path.exists(os.path.join(MODELS_DIR, "gbm_predictions_h1.csv")) else 0
                if mtime_ver >= mtime_pred:
                    _safe_print("[preflight] Verification inputs are up to date.")
                    return
            except Exception:
                pass

        _safe_print("[preflight] Computing verification artifacts...")
        inputs: Dict[str, Any] = {}

        # Tools resolve relative paths against DATA_DIR, so pass absolute paths
        # for files that live outside that directory (e.g. models/).
        feature_abs = os.path.abspath(feature_path)
        test_abs = os.path.abspath(test_path)

        try:
            persistence = PersistenceBaselineTool()._run(
                filepath=feature_abs,
                column="water_level",
                horizons=[1, 6, 12, 24, 48, 72, 168],
            )
            inputs["persistence"] = _json.loads(persistence)
        except Exception as e:
            _safe_print(f"[preflight] persistence baseline warning: {e}")
            inputs["persistence"] = {"error": str(e)}

        inputs["gbm_overall"] = {}
        for h in [1, 6, 12, 24, 48, 72, 168]:
            pred_file = os.path.join(MODELS_DIR, f"gbm_predictions_h{h}.csv")
            if not os.path.exists(pred_file):
                continue
            pred_abs = os.path.abspath(pred_file)
            try:
                metrics = ComputeMetricsTool()._run(
                    actual_filepath=test_abs,
                    predicted_filepath=pred_abs,
                    actual_col="water_level",
                    predicted_col="predicted",
                )
                inputs["gbm_overall"][f"h={h}"] = _json.loads(metrics)
            except Exception as e:
                _safe_print(f"[preflight] metrics h={h} warning: {e}")
                inputs["gbm_overall"][f"h={h}"] = {"error": str(e)}

        try:
            walkforward = WalkForwardTool()._run(
                filepath=feature_abs,
                timestamp_col="timestamp",
                target_col="water_level",
                train_window=168,
                test_window=24,
                n_splits=5,
            )
            inputs["walk_forward"] = _json.loads(walkforward)
        except Exception as e:
            _safe_print(f"[preflight] walk-forward warning: {e}")
            inputs["walk_forward"] = {"error": str(e)}

        inputs["stratified"] = {}
        for h in [1, 6, 12, 24, 48, 72, 168]:
            pred_file = os.path.join(MODELS_DIR, f"gbm_predictions_h{h}.csv")
            if not os.path.exists(pred_file):
                continue
            pred_abs = os.path.abspath(pred_file)
            try:
                stratified = StratifiedMetricsTool()._run(
                    actual_filepath=test_abs,
                    predicted_filepath=pred_abs,
                    actual_col="water_level",
                    predicted_col="predicted",
                )
                inputs["stratified"][f"h={h}"] = _json.loads(stratified)
            except Exception as e:
                _safe_print(f"[preflight] stratified h={h} warning: {e}")
                inputs["stratified"][f"h={h}"] = {"error": str(e)}

        inputs["metadata"] = {
            "dataset": dataset_path,
            "feature_matrix": feature_path,
            "test_split": test_path,
            "models_dir": MODELS_DIR,
            "computed_at": datetime.utcnow().isoformat() + "Z",
        }

        try:
            with open(verification_path, "w", encoding="utf-8") as f:
                _json.dump(inputs, f, indent=2, ensure_ascii=False)
            _safe_print(f"[preflight] Verification inputs written to {verification_path}")
        except Exception as e:
            _safe_print(f"[preflight] Could not write verification inputs: {e}")

    def _log_results(self, result):
        """Persist a concise run summary after the crew finishes."""
        token_usage = getattr(result, "token_usage", None)
        summary = {
            "tasks_completed": len(result.tasks_output) if result.tasks_output else 0,
            "token_usage": self._serialize_token_usage(token_usage),
            "summary": result.raw[:500] if result.raw else "",
        }
        summary_path = os.path.join(OUTPUT_DIR, "run_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        return result

    def _serialize_token_usage(self, token_usage) -> Any:
        """Convert CrewAI UsageMetrics to a JSON-serializable dict."""
        if token_usage is None:
            return None
        if isinstance(token_usage, dict):
            return token_usage
        if hasattr(token_usage, "model_dump"):
            try:
                return token_usage.model_dump()
            except Exception:
                pass
        if hasattr(token_usage, "dict"):
            try:
                return token_usage.dict()
            except Exception:
                pass
        return str(token_usage)

    def _on_task_complete(self, output: TaskOutput) -> None:
        _safe_print(f"TASK DONE: '{output.description}' by {output.agent}")
        self._append_metrics_log(output)
        self._save_checkpoint(output)

    def _on_step(self, step: Any) -> None:
        step_str = str(step)
        if len(step_str) > 120:
            step_str = step_str[:120] + "..."
        _safe_print(f"[step] {step_str}")

    def _append_metrics_log(self, output: TaskOutput) -> None:
        """Append a structured event for each completed task."""
        log_path = os.path.join(OUTPUT_DIR, "metrics_log.jsonl")
        event = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "agent": output.agent,
            "description": output.description,
            "output_format": output.output_format.value if output.output_format else None,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _save_checkpoint(self, output: TaskOutput) -> None:
        """Persist task output so a failed run can be inspected or resumed."""
        checkpoint_dir = os.path.join(OUTPUT_DIR, ".checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        safe_name = "".join(
            c if c.isalnum() or c in "-_" else "_" for c in (output.description or "task")
        )[:50]
        filename = f"{len([f for f in os.listdir(checkpoint_dir) if f.endswith('.json')]):03d}_{safe_name}.json"
        data = {
            "description": output.description,
            "agent": output.agent,
            "raw": output.raw,
        }
        if output.pydantic:
            data["pydantic"] = output.pydantic.model_dump()
        if output.json_dict:
            data["json_dict"] = output.json_dict
        path = os.path.join(checkpoint_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self._update_checkpoint_manifest(output)

    def _update_checkpoint_manifest(self, output: TaskOutput) -> None:
        """Maintain a simple manifest mapping output file to task metadata."""
        checkpoint_dir = os.path.join(OUTPUT_DIR, ".checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        manifest_path = os.path.join(checkpoint_dir, "completed_tasks.json")
        manifest: Dict[str, Any] = {}
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
            except Exception:
                manifest = {}
        output_file = getattr(output, "output_file", None)
        if output_file:
            manifest[output_file] = {
                "description": output.description,
                "agent": output.agent,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
        try:
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
        except Exception as e:
            _safe_print(f"[checkpoint] Could not update manifest: {e}")

    def _load_completed_output_files(self) -> set[str]:
        """Return output file paths that have a non-empty checkpoint manifest entry."""
        completed: set[str] = set()
        manifest_path = os.path.join(OUTPUT_DIR, ".checkpoints", "completed_tasks.json")
        if not os.path.exists(manifest_path):
            return completed
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            return completed
        for output_file, info in manifest.items():
            if not isinstance(info, dict):
                continue
            if output_file and os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                completed.add(output_file)
        return completed

    def _filter_tasks_for_resume(self, tasks: List[Any]) -> List[Any]:
        """Opt-in resume: skip tasks whose output files already exist.

        Enabled by setting environment variable CREW_RESUME=1.
        """
        resume = os.environ.get("CREW_RESUME", "").lower() in ("1", "true", "yes")
        if not resume:
            return tasks
        completed = self._load_completed_output_files()
        if not completed:
            return tasks
        _safe_print(f"[resume] Found {len(completed)} completed task(s); filtering tasks.")
        filtered = []
        for task in tasks:
            output_file = getattr(task, "output_file", None)
            if output_file and output_file in completed:
                _safe_print(f"[resume] skipping: {task.description[:80]}...")
                continue
            filtered.append(task)
        _safe_print(f"[resume] {len(filtered)} task(s) remaining.")
        return filtered

    def _load_checkpoints(self) -> dict:
        """Load checkpointed task outputs keyed by description."""
        checkpoint_dir = os.path.join(OUTPUT_DIR, ".checkpoints")
        if not os.path.isdir(checkpoint_dir):
            return {}
        checkpoints = {}
        for filename in sorted(os.listdir(checkpoint_dir)):
            if not filename.endswith(".json"):
                continue
            path = os.path.join(checkpoint_dir, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                checkpoints[data.get("description", filename)] = data
            except Exception:
                continue
        return checkpoints

    def _render_feature_manifest(self, output: TaskOutput) -> None:
        """Render a Markdown feature manifest from the structured task output."""
        manifest = output.pydantic
        if not isinstance(manifest, FeatureManifest):
            return
        lines = [
            "# Feature Engineering Report\n",
            f"**Target column:** {manifest.target_column}\n",
            f"**Horizons:** {', '.join(str(h) for h in manifest.horizon_hours)} hours\n",
            "## Feature files\n",
        ]
        for f in manifest.feature_files:
            lines.append(f"- `{f}`\n")
        if manifest.split_dates:
            lines.append("\n## Train/validation/test splits\n")
            for split, date in manifest.split_dates.items():
                lines.append(f"- **{split}:** {date}\n")
        md_path = os.path.join(OUTPUT_DIR, "phase2_feature_engineering.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("".join(lines))

    def _render_baseline_metrics(self, output: TaskOutput) -> None:
        """Render a Markdown baseline metrics report from the structured task output."""
        metrics = output.pydantic
        if not isinstance(metrics, BaselineMetrics):
            return
        lines = [
            f"# Baseline Modeling Report: {metrics.model}\n",
            "## Metrics per horizon\n",
            "| Horizon | RMSE | MAE | NSE | Bias |\n",
            "|---|---|---|---|---|\n",
        ]
        for horizon, hm in metrics.horizons.items():
            lines.append(f"| {horizon} | {hm.rmse} | {hm.mae} | {hm.nse} | {hm.bias} |\n")
        md_path = os.path.join(OUTPUT_DIR, "phase2_baseline_modeling.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("".join(lines))

    def _render_verification_report(self, output: TaskOutput) -> None:
        """Render a Markdown verification report from the structured task output."""
        report = output.pydantic
        if not isinstance(report, VerificationReport):
            return
        lines = [
            "# Verification Report\n",
            f"**Best model:** {report.best_model}\n",
            f"**Best horizon:** {report.best_horizon}h\n",
            f"**Beats official forecast:** {'Yes' if report.beats_official else 'No'}\n",
            "## Summary\n",
            f"{report.metrics_summary}\n",
        ]
        md_path = os.path.join(OUTPUT_DIR, "phase5_verification.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("".join(lines))

    def _write_artifact_manifest(self, result) -> None:
        """Scan output directories and write a manifest of durable artifacts."""
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        models = []
        features = []
        reports = []
        metadata = {
            "run_summary": os.path.join(OUTPUT_DIR, "run_summary.json"),
            "crew_log": os.path.join(OUTPUT_DIR, "crew_run.json"),
        }

        if os.path.isdir(MODELS_DIR):
            models = [
                os.path.join(MODELS_DIR, f)
                for f in os.listdir(MODELS_DIR)
                if os.path.isfile(os.path.join(MODELS_DIR, f))
            ]
        if os.path.isdir(os.path.join(DATA_DIR, "features")):
            features = [
                os.path.join(DATA_DIR, "features", f)
                for f in os.listdir(os.path.join(DATA_DIR, "features"))
                if os.path.isfile(os.path.join(DATA_DIR, "features", f))
            ]
        for report_name in [
            "phase2_feature_engineering.md",
            "phase2_baseline_modeling.md",
            "phase3_model_training.md",
            "phase5_verification.md",
            "phase5_final_documentation.md",
            "phase5_report.md",
        ]:
            report_path = os.path.join(OUTPUT_DIR, report_name)
            if os.path.exists(report_path):
                reports.append(report_path)

        manifest = ArtifactManifest(
            models=models,
            features=features,
            reports=reports,
            metadata=metadata,
        )
        manifest_path = os.path.join(ARTIFACTS_DIR, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest.model_dump(), f, indent=2)

        # Ensure the final HTML report exists and is valid, even if the
        # frontend specialist task misbehaved or was skipped on resume.
        self._ensure_final_report()

        return result

    def _ensure_final_report(self) -> None:
        """Deterministically rebuild output/final_report.html if it is missing or invalid."""
        try:
            from thesiscrew.tools.html_report_tool import BuildHtmlReportTool

            report_path = os.path.join(OUTPUT_DIR, "final_report.html")
            min_size = 20_000  # A real styled report is much larger than agent chatter.
            needs_rebuild = True

            if os.path.exists(report_path) and os.path.getsize(report_path) >= min_size:
                with open(report_path, "r", encoding="utf-8") as f:
                    content = f.read()
                needs_rebuild = not BuildHtmlReportTool.validate_html(content)

            if needs_rebuild:
                _safe_print("[final-report] Rebuilding output/final_report.html deterministically...")
                try:
                    from thesiscrew.tools.forward_forecast_tool import BuildForwardForecastsTool
                    _safe_print("[final-report] Refreshing forward forecasts first...")
                    BuildForwardForecastsTool()._run()
                except Exception as e:
                    _safe_print(f"[final-report] Forward forecast refresh warning: {e}")
                tool = BuildHtmlReportTool()
                result = tool._run(style="academic", include_charts=True, focus="forecasting")
                _safe_print(f"[final-report] {result}")
            else:
                _safe_print("[final-report] output/final_report.html is present and valid.")
        except Exception as e:
            _safe_print(f"[final-report] Could not ensure final report: {e}")


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

def agent_llm(agent_name: str, default: str | None = None) -> str:
    """Return agent-specific LLM, overridable via <AGENT_NAME>_LLM env var.

    If no override or explicit default is given, fall back to the MODEL/BASE_URL
    configured in the project .env.
    """
    env_key = f"{agent_name.upper()}_LLM"
    return os.environ.get(env_key, default or DEFAULT_LLM)


def supports_structured_outputs() -> bool:
    """Return True if the configured model reliably supports Pydantic/JSON outputs."""
    return DEFAULT_LLM.startswith(("openai/", "anthropic/"))
