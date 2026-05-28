"""MCP server for report generation and artifact reading.

Provides tools for writing, reading, and formatting the workflow report,
plus reading pipeline artifacts (CSV, Parquet, JSON, Markdown).
"""

import json
import os
from datetime import datetime
from typing import Optional

from mcp.server.fastmcp import FastMCP

OUTPUT_DIR = os.environ.get("PEGELHUB_OUTPUT_DIR", "output")
DATA_DIR = os.environ.get("PEGELHUB_DATA_DIR", os.path.join(OUTPUT_DIR, "data"))

mcp = FastMCP("report")


# ── Helper: Markdown table ─────────────────────────────────────────────────

def _md_table(headers: list[str], rows: list[list[str]], title: Optional[str] = None) -> str:
    if not headers:
        return "Error: headers cannot be empty"
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(str(cell)))
    lines = []
    if title:
        lines.append(f"*{title}*\n")
    lines.append("| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |")
    lines.append("| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |")
    for row in rows:
        cells = [str(row[i]).ljust(col_widths[i]) if i < len(row) else "".ljust(col_widths[i]) for i in range(len(headers))]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ── Write Report Section ────────────────────────────────────────────────────

@mcp.tool()
def write_report_section(section: str, content: str, mode: str = "append") -> str:
    """Write or append a Markdown section to the workflow report (output/report.md).

    Args:
        section: Section name (e.g. 'executive_summary', 'data_discovery',
                 'verification', 'full' to write entire report).
        content: Markdown content for the section.
        mode: 'append' to add section, 'overwrite' to replace entire file.
    """
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


# ── Read Report ──────────────────────────────────────────────────────────────

@mcp.tool()
def read_report() -> str:
    """Read the current workflow report (output/report.md).
    Returns the full content, or a truncated preview if very long.
    """
    report_path = os.path.join(OUTPUT_DIR, "report.md")
    if not os.path.exists(report_path):
        return "Report file does not exist yet. Use write_report_section to create it."
    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()
    if len(content) > 8000:
        return f"Report is {len(content)} chars. First 8000:\n\n{content[:8000]}\n\n... (truncated)"
    return content


# ── Generate TOC ─────────────────────────────────────────────────────────────

@mcp.tool()
def generate_toc() -> str:
    """Generate a table of contents from the current report (output/report.md).
    Scans ## and ### headers and produces a linked Markdown TOC.
    """
    report_path = os.path.join(OUTPUT_DIR, "report.md")
    if not os.path.exists(report_path):
        return "Report file does not exist yet."
    with open(report_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    toc_lines = ["## Table of Contents\n"]
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("### "):
            title = stripped[4:].strip()
            anchor = title.lower().replace(" ", "-").replace(".", "")
            toc_lines.append(f"  - [{title}](#{anchor})")
        elif stripped.startswith("## ") and "Table of Contents" not in stripped and not stripped.startswith("# "):
            title = stripped[3:].strip()
            anchor = title.lower().replace(" ", "-").replace(".", "")
            toc_lines.append(f"- [{title}](#{anchor})")
    if len(toc_lines) <= 1:
        return "No headers found in report."
    return "\n".join(toc_lines)


# ── Read Artifact ────────────────────────────────────────────────────────────

@mcp.tool()
def read_artifact(filepath: str, max_rows: int = 10) -> str:
    """Read a pipeline artifact (CSV, Parquet, JSON, Markdown) and return
    a summary suitable for the report.

    Args:
        filepath: Path to the file (relative to data/ or output/).
        max_rows: Max rows to preview for tabular data.
    """
    full_path = filepath if os.path.isabs(filepath) else os.path.join(DATA_DIR, filepath)
    if not os.path.exists(full_path):
        alt_path = os.path.join(OUTPUT_DIR, filepath)
        if os.path.exists(alt_path):
            full_path = alt_path
        else:
            return f"File not found: {filepath}"

    ext = os.path.splitext(full_path)[1].lower()

    if ext == ".csv":
        return _read_csv(full_path, max_rows)
    elif ext == ".parquet":
        return _read_parquet(full_path, max_rows)
    elif ext == ".json":
        return _read_json(full_path)
    elif ext in (".md", ".markdown", ".txt"):
        return _read_text(full_path)
    else:
        return f"Unsupported file type: {ext}"


def _read_csv(path: str, max_rows: int) -> str:
    import pandas as pd
    try:
        df = pd.read_csv(path, nrows=max_rows + 100)
    except Exception as e:
        return f"Error reading CSV: {e}"
    lines = [f"**File:** `{path}`"]
    lines.append(f"**Shape:** {df.shape[0]} rows x {df.shape[1]} columns")
    lines.append(f"**Columns:** {', '.join(df.columns.tolist())}")
    lines.append(f"**Preview (first {min(max_rows, len(df))} rows):**")
    headers = df.columns.tolist()
    rows = df.head(max_rows).astype(str).values.tolist()
    lines.append(_md_table(headers, rows))
    return "\n".join(lines)


def _read_parquet(path: str, max_rows: int) -> str:
    import pandas as pd
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        return f"Error reading Parquet: {e}"
    lines = [f"**File:** `{path}`"]
    lines.append(f"**Shape:** {df.shape[0]} rows x {df.shape[1]} columns")
    lines.append(f"**Preview (first {min(max_rows, len(df))} rows):**")
    headers = df.columns.tolist()
    rows = df.head(max_rows).astype(str).values.tolist()
    lines.append(_md_table(headers, rows))
    return "\n".join(lines)


def _read_json(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return f"**File:** `{path}`\n\n```json\n{json.dumps(data, indent=2, ensure_ascii=False)}\n```"
    except Exception as e:
        return f"Error reading JSON: {e}"


def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading text file: {e}"


# ── Render Metrics ──────────────────────────────────────────────────────────

@mcp.tool()
def render_metrics(metrics_json: str, comparison: bool = False) -> str:
    """Format metric results as a Markdown table suitable for the report.

    Args:
        metrics_json: JSON string with metrics. Single model:
            '{"model": "xgboost", "horizons": {"+1h": {"rmse": 5.2, ...}}}'
            Multi-model comparison: '{"xgboost": {"rmse": 5.2, ...}, ...}'
        comparison: If True, format as multi-model comparison table.
    """
    try:
        data = json.loads(metrics_json)
    except json.JSONDecodeError as e:
        return f"Error parsing metrics JSON: {e}"

    if comparison and isinstance(data, dict):
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
        return _md_table(headers, rows, title="Model Comparison")

    if isinstance(data, dict) and "horizons" in data:
        model_name = data.get("model", "unknown")
        horizons = data.get("horizons", {})
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
        return f"### {model_name}\n\n" + _md_table(headers, rows)

    if isinstance(data, dict):
        headers = ["Metric", "Value"]
        rows = [[k, str(v)] for k, v in data.items()]
        return _md_table(headers, rows)

    return f"Unsupported metrics format: {type(data)}"


# ── Markdown Table ───────────────────────────────────────────────────────────

@mcp.tool()
def markdown_table(headers: list[str], rows: list[list[str]], title: Optional[str] = None) -> str:
    """Format data as a Markdown table with aligned columns.

    Args:
        headers: Column headers.
        rows: Row data, each row is a list of strings.
        title: Optional caption above the table.
    """
    return _md_table(headers, rows, title)


if __name__ == "__main__":
    mcp.run()