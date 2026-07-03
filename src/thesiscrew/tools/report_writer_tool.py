"""CrewAI tools for the report_writer agent.

Provides:
- WriteReportTool: append/overwrite sections to output/data/data_output.md
  or output/models/models_output.md
- ReadReportTool: read current report state from either output file
- MarkdownTableTool: format structured data as Markdown tables
- ReportTOCTool: generate a table of contents from either report file
- RenderMetricsTool: format metric dicts as Markdown comparison tables
- ReadArtifactTool is imported from thesiscrew.tools.artifact_tool for reuse.
"""

import json
import os
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field
from crewai.tools import BaseTool

from thesiscrew.tools.artifact_tool import ReadArtifactTool, ReadArtifactInput

OUTPUT_DIR = os.environ.get("PEGELHUB_OUTPUT_DIR", "output")
DATA_DIR = os.environ.get("PEGELHUB_DATA_DIR", "data")

# Section → target file mapping
DATA_SECTIONS = {"executive_summary", "data_discovery", "ingestion", "feature_engineering"}
MODELS_SECTIONS = {"baselines", "model_development", "verification", "integration",
                   "reproducibility", "limitations"}

REPORT_FILES = {
    "data": os.path.join(OUTPUT_DIR, "data", "data_output.md"),
    "models": os.path.join(OUTPUT_DIR, "models", "models_output.md"),
}


def _resolve_target(section: str, target: Optional[str] = None) -> str:
    """Return 'data' or 'models' based on section name or explicit target."""
    if target in REPORT_FILES:
        return target
    if section in DATA_SECTIONS:
        return "data"
    if section in MODELS_SECTIONS:
        return "models"
    if section == "full":
        return "models"
    return "data"


# ── Write Report ───────────────────────────────────────────────────────────

class WriteReportInput(BaseModel):
    section: str = Field(
        description="Section name: executive_summary, data_discovery, ingestion, "
                    "feature_engineering, baselines, model_development, verification, "
                    "integration, reproducibility, limitations, or full."
    )
    content: str = Field(
        description="Markdown content for the section."
    )
    mode: str = Field(
        default="append",
        description="append or overwrite."
    )
    target: str = Field(
        default="auto",
        description="Which file to write to: 'data' for output/data/data_output.md, "
                    "'models' for output/models/models_output.md, or 'auto' to route "
                    "based on section name."
    )


class WriteReportTool(BaseTool):
    name: str = "write_report"
    description: str = (
        "Write or append Markdown content to the workflow report files. "
        "Data sections (executive_summary, data_discovery, ingestion, "
        "feature_engineering) go to output/data/data_output.md. "
        "Model sections (baselines, model_development, verification, "
        "integration, reproducibility, limitations) go to "
        "output/models/models_output.md. Use target='data' or "
        "target='models' to override; 'auto' routes by section name."
    )
    args_schema: type[BaseModel] = WriteReportInput

    def _run(self, section: str, content: str, mode: str = "append", target: str = "auto") -> str:
        resolved = _resolve_target(section, target if target != "auto" else None)
        report_path = REPORT_FILES[resolved]
        os.makedirs(os.path.dirname(report_path), exist_ok=True)

        label = "Data" if resolved == "data" else "Models"
        title = f"# Water Level Forecasting — {label} Report\n\n"

        if mode == "overwrite" or section == "full":
            header = (
                title
                + f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                f"**Station:** Korneuburg / Donau (Danube), Austria\n\n"
                f"---\n\n"
            )
            full_content = header + content
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(full_content)
            return f"Report written to {report_path} ({len(full_content)} chars)"

        section_header = f"\n\n---\n\n## {section.replace('_', ' ').title()}\n\n"
        existing = ""
        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                existing = f.read()

        with open(report_path, "w", encoding="utf-8") as f:
            if not existing:
                f.write(
                    title
                    + f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                    f"**Station:** Korneuburg / Donau (Danube), Austria\n\n"
                    f"---\n\n"
                )
            else:
                f.write(existing)
            f.write(section_header)
            f.write(content)
            f.write("\n")

        return f"Section '{section}' appended to {report_path}"


# ── Read Report ─────────────────────────────────────────────────────────────


class ReadReportInput(BaseModel):
    target: str = Field(
        default="data",
        description="Which report file to read: 'data' or 'models'."
    )


class ReadReportTool(BaseTool):
    name: str = "read_report"
    description: str = (
        "Read the current state of a workflow report file. "
        "target='data' reads output/data/data_output.md, "
        "target='models' reads output/models/models_output.md. "
        "Use this to review what has already been documented before "
        "appending new sections."
    )
    args_schema: type[BaseModel] = ReadReportInput

    def _run(self, target: str = "data") -> str:
        resolved = target if target in REPORT_FILES else "data"
        report_path = REPORT_FILES[resolved]
        if not os.path.exists(report_path):
            return f"Report file {report_path} does not exist yet. Use write_report to create it."
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()
        if len(content) > 8000:
            return (
                f"Report is {len(content)} chars. Showing first 8000:\n\n"
                f"{content[:8000]}\n\n... (truncated)"
            )
        return content


# ── Markdown Table Formatter ────────────────────────────────────────────────

class MarkdownTableInput(BaseModel):
    headers: list[str] = Field(description="Column headers for the table.")
    rows: list[list[str]] = Field(
        description="Table rows. Each row is a list of strings matching the headers. "
                    "Example: [['5.2', '3.8', '0.95'], ['4.1', '2.9', '0.88']]."
    )
    title: Optional[str] = Field(default=None, description="Optional table caption.")


class MarkdownTableTool(BaseTool):
    name: str = "markdown_table"
    description: str = (
        "Format structured data as a Markdown table. Provide column headers "
        "and row data, get a properly aligned Markdown table string. "
        "Optionally add a caption title."
    )
    args_schema: type[BaseModel] = MarkdownTableInput

    def _run(self, headers: list[str], rows: list[list[str]], title: Optional[str] = None) -> str:
        if not headers:
            return "Error: headers cannot be empty"
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(str(cell)))

        def pad(cell: str, width: int) -> str:
            return str(cell).ljust(width)

        lines = []
        if title:
            lines.append(f"*{title}*\n")
        header_line = "| " + " | ".join(pad(h, col_widths[i]) for i, h in enumerate(headers)) + " |"
        separator = "| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |"
        lines.append(header_line)
        lines.append(separator)
        for row in rows:
            cells = []
            for i in range(len(headers)):
                cell = row[i] if i < len(row) else ""
                cells.append(pad(str(cell), col_widths[i]))
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)


# ── Table of Contents Generator ─────────────────────────────────────────────

class TOCInput(BaseModel):
    target: str = Field(
        default="data",
        description="Which report file to generate TOC for: 'data' or 'models'."
    )


class ReportTOCTool(BaseTool):
    name: str = "report_toc"
    description: str = (
        "Generate a table of contents from a report file. "
        "target='data' scans output/data/data_output.md, "
        "target='models' scans output/models/models_output.md. "
        "Produces a linked Markdown TOC."
    )
    args_schema: type[BaseModel] = TOCInput

    def _run(self, target: str = "data") -> str:
        resolved = target if target in REPORT_FILES else "data"
        report_path = REPORT_FILES[resolved]
        if not os.path.exists(report_path):
            return f"Report file {report_path} does not exist yet. Use write_report first."
        with open(report_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        toc_lines = ["## Table of Contents\n"]
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("### "):
                title = stripped[4:].strip()
                anchor = title.lower().replace(" ", "-").replace(".", "")
                toc_lines.append(f"  - [{title}](#{anchor})")
            elif stripped.startswith("## ") and "Table of Contents" not in stripped and stripped != "# ":
                title = stripped[3:].strip()
                anchor = title.lower().replace(" ", "-").replace(".", "")
                toc_lines.append(f"- [{title}](#{anchor})")
        if len(toc_lines) <= 1:
            return "No headers found in report."
        return "\n".join(toc_lines)


# ── Metrics Renderer ────────────────────────────────────────────────────────

class RenderMetricsInput(BaseModel):
    metrics_json: str = Field(
        description="JSON string of metric results. Example: "
                    "'{\"model\": \"xgboost\", \"horizons\": {\"+1h\": {\"rmse\": 5.2}}}'."
    )
    comparison: bool = Field(
        default=False,
        description="True = multi-model comparison table. False = single-model table.",
    )


class RenderMetricsTool(BaseTool):
    name: str = "render_metrics"
    description: str = (
        "Format metric results as a Markdown comparison table. Accepts a JSON "
        "string of metrics and produces a clean table suitable for the report. "
        "Supports single-model and multi-model comparison formats."
    )
    args_schema: type[BaseModel] = RenderMetricsInput

    def _run(self, metrics_json: str, comparison: bool = False) -> str:
        try:
            data = json.loads(metrics_json)
        except json.JSONDecodeError as e:
            return f"Error parsing metrics JSON: {e}"

        if comparison and isinstance(data, dict):
            return self._render_comparison(data)
        elif isinstance(data, dict):
            if "horizons" in data:
                return self._render_horizon_table(data)
            return self._render_flat_table(data)
        return f"Unsupported format: {type(data)}"

    def _render_horizon_table(self, data: dict) -> str:
        model_name = data.get("model", "unknown")
        horizons = data.get("horizons", {})
        if not horizons:
            return f"No horizon data for model {model_name}"
        headers = ["Horizon", "RMSE (cm)", "MAE (cm)", "NSE", "Bias (cm)"]
        rows = []
        for h, m in horizons.items():
            rows.append([
                h,
                str(m.get("rmse", m.get("RMSE_cm", "N/A"))),
                str(m.get("mae", m.get("MAE_cm", "N/A"))),
                str(m.get("nse", m.get("NSE", "N/A"))),
                str(m.get("bias", m.get("bias_cm", "N/A"))),
            ])
        tool = MarkdownTableTool()
        return f"### {model_name}\n\n" + tool._run(headers=headers, rows=rows)

    def _render_flat_table(self, data: dict) -> str:
        headers = ["Metric", "Value"]
        rows = [[k, str(v)] for k, v in data.items()]
        tool = MarkdownTableTool()
        return tool._run(headers=headers, rows=rows)

    def _render_comparison(self, data: dict) -> str:
        headers = ["Model", "RMSE (cm)", "MAE (cm)", "NSE", "Bias (cm)"]
        rows = []
        for model_name, metrics in data.items():
            if isinstance(metrics, dict):
                rows.append([
                    model_name,
                    str(metrics.get("rmse", metrics.get("RMSE_cm", "N/A"))),
                    str(metrics.get("mae", metrics.get("MAE_cm", "N/A"))),
                    str(metrics.get("nse", metrics.get("NSE", "N/A"))),
                    str(metrics.get("bias", metrics.get("bias_cm", "N/A"))),
                ])
        tool = MarkdownTableTool()
        return tool._run(headers=headers, rows=rows, title="Model Comparison")


# ── Read Pipeline Artifact (shared implementation) ─────────────────────────
# ReadArtifactTool is defined in thesiscrew.tools.artifact_tool and imported
# above so it can be reused by any agent, not just the report writer.
ReadArtifactTool, ReadArtifactInput  # silence unused-import linters
