"""CrewAI tools for the report_writer agent.

Provides:
- WriteReportTool: append/overwrite sections to output/report.md
- ReadReportTool: read current report state
- MarkdownTableTool: format structured data as Markdown tables
- ReportTOCTool: generate a table of contents from the report
- RenderMetricsTool: format metric dicts as Markdown comparison tables
- ReadArtifactTool: read any pipeline artifact (CSV, Parquet, JSON, MD)
"""

import json
import os
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field
from crewai.tools import BaseTool

OUTPUT_DIR = os.environ.get("PEGELHUB_OUTPUT_DIR", "output")
DATA_DIR = os.environ.get("PEGELHUB_DATA_DIR", "data")


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


class WriteReportTool(BaseTool):
    name: str = "write_report"
    description: str = (
        "Write or append Markdown content to the workflow report file "
        "(output/report.md). Use this to document each phase's findings, "
        "metric tables, feature manifests, and conclusions as the pipeline "
        "progresses. Sections are appended in order; use 'full' to write "
        "the complete report at once."
    )
    args_schema: type[BaseModel] = WriteReportInput

    def _run(self, section: str, content: str, mode: str = "append") -> str:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        report_path = os.path.join(OUTPUT_DIR, "report.md")

        if mode == "overwrite" or section == "full":
            header = (
                f"# Water Level Forecasting Report\n\n"
                f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
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
                    f"# Water Level Forecasting Report\n\n"
                    f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
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

class _NoInput(BaseModel):
    pass


class ReadReportTool(BaseTool):
    name: str = "read_report"
    description: str = (
        "Read the current state of the workflow report file (output/report.md). "
        "Use this to review what has already been documented before appending "
        "new sections, ensuring no duplication and consistent formatting."
    )
    args_schema: type[BaseModel] = _NoInput

    def _run(self) -> str:
        report_path = os.path.join(OUTPUT_DIR, "report.md")
        if not os.path.exists(report_path):
            return "Report file does not exist yet. Use write_report to create it."
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
    rows: list[str] = Field(
        description="Row data as JSON string. Each element is a list of strings matching the headers. "
                    "Example: '[\"5.2\", \"3.8\", \"0.95\"]' for a single row, or pass the full list."
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

    def _run(self, headers: list[str], rows: list[str], title: Optional[str] = None) -> str:
        # Parse rows: if first element looks like JSON list, parse it
        parsed_rows = []
        if rows:
            first = rows[0]
            if isinstance(first, str) and first.strip().startswith("["):
                try:
                    parsed_rows = json.loads(first.strip())
                    if isinstance(parsed_rows, list) and parsed_rows and isinstance(parsed_rows[0], list):
                        pass  # already list of lists
                    elif isinstance(parsed_rows, list):
                        parsed_rows = [parsed_rows]
                except (json.JSONDecodeError, TypeError):
                    parsed_rows = [[str(c) for c in rows]]
            else:
                parsed_rows = [[str(c) for c in rows]]
        rows = parsed_rows
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

class ReportTOCTool(BaseTool):
    name: str = "report_toc"
    description: str = (
        "Generate a table of contents from the current report (output/report.md). "
        "Scans all ## and ### headers and produces a linked TOC in Markdown."
    )
    args_schema: type[BaseModel] = _NoInput

    def _run(self) -> str:
        report_path = os.path.join(OUTPUT_DIR, "report.md")
        if not os.path.exists(report_path):
            return "Report file does not exist yet. Use write_report first."
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


# ── Read Pipeline Artifact ──────────────────────────────────────────────────

class ReadArtifactInput(BaseModel):
    filepath: str = Field(
        description="Path to the artifact file (CSV, Parquet, JSON, or Markdown). "
                    "Relative to the project root.",
    )
    max_rows: int = Field(default=10, description="Max rows to preview for tabular data.")


class ReadArtifactTool(BaseTool):
    name: str = "read_artifact"
    description: str = (
        "Read any pipeline artifact file (CSV, Parquet, JSON, Markdown) and "
        "return a summary suitable for inclusion in the report. For tabular "
        "data, returns shape, columns, dtypes, and a preview. For JSON, "
        "returns formatted content. For Markdown, returns the content directly."
    )
    args_schema: type[BaseModel] = ReadArtifactInput

    def _run(self, filepath: str, max_rows: int = 10) -> str:
        full_path = filepath if os.path.isabs(filepath) else os.path.join(DATA_DIR, filepath)
        if not os.path.exists(full_path):
            # Also try output dir
            alt_path = os.path.join(OUTPUT_DIR, filepath)
            if os.path.exists(alt_path):
                full_path = alt_path
            else:
                return f"File not found: {filepath} (checked {DATA_DIR} and {OUTPUT_DIR})"

        ext = os.path.splitext(full_path)[1].lower()
        if ext == ".csv":
            return self._read_csv(full_path, max_rows)
        elif ext == ".parquet":
            return self._read_parquet(full_path, max_rows)
        elif ext == ".json":
            return self._read_json(full_path)
        elif ext in (".md", ".markdown", ".txt"):
            return self._read_text(full_path)
        else:
            return f"Unsupported file type: {ext}"

    def _read_csv(self, path: str, max_rows: int) -> str:
        import pandas as pd
        try:
            df = pd.read_csv(path, nrows=max_rows + 100)
        except Exception as e:
            return f"Error reading CSV: {e}"
        lines = [f"**File:** `{path}`"]
        lines.append(f"**Shape:** {df.shape[0]} rows x {df.shape[1]} columns")
        lines.append(f"**Columns:** {', '.join(df.columns.tolist())}")
        lines.append(f"**Dtypes:** {', '.join(f'{c}={str(dt)}' for c, dt in df.dtypes.items())}")
        nulls = df.isna().sum()
        if nulls.any():
            null_cols = nulls[nulls > 0]
            lines.append(f"**Nulls:** {', '.join(f'{c}={int(v)}' for c, v in null_cols.items())}")
        lines.append(f"\n**Preview (first {min(max_rows, len(df))} rows):**")
        tool = MarkdownTableTool()
        headers = df.columns.tolist()
        rows = df.head(max_rows).astype(str).values.tolist()
        lines.append(tool._run(headers=headers, rows=rows))
        return "\n".join(lines)

    def _read_parquet(self, path: str, max_rows: int) -> str:
        import pandas as pd
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            return f"Error reading Parquet: {e}"
        lines = [f"**File:** `{path}`"]
        lines.append(f"**Shape:** {df.shape[0]} rows x {df.shape[1]} columns")
        lines.append(f"**Columns:** {', '.join(df.columns.tolist())}")
        lines.append(f"\n**Preview (first {min(max_rows, len(df))} rows):**")
        tool = MarkdownTableTool()
        headers = df.columns.tolist()
        rows = df.head(max_rows).astype(str).values.tolist()
        lines.append(tool._run(headers=headers, rows=rows))
        return "\n".join(lines)

    def _read_json(self, path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return f"**File:** `{path}`\n\n```json\n{json.dumps(data, indent=2, ensure_ascii=False)}\n```"
        except Exception as e:
            return f"Error reading JSON: {e}"

    def _read_text(self, path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            return content
        except Exception as e:
            return f"Error reading text file: {e}"