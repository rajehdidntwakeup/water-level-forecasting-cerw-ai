"""Shared artifact-reading tool used by any agent that consumes prior outputs.

Provides ReadArtifactTool, which reads CSV, Parquet, JSON, or Markdown files
produced by upstream agents/tasks.
"""

import json
import os
from typing import Optional

from pydantic import BaseModel, Field
from crewai.tools import BaseTool


OUTPUT_DIR = os.environ.get("PEGELHUB_OUTPUT_DIR", "output")
DATA_DIR = os.environ.get("PEGELHUB_DATA_DIR", os.path.join(OUTPUT_DIR, "data"))
PROJECT_ROOT = os.environ.get(
    "PEGELHUB_PROJECT_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
)


class ReadArtifactInput(BaseModel):
    filepath: str = Field(
        description="Path to the artifact file (CSV, Parquet, JSON, or Markdown). "
                    "Relative paths are resolved against the project root first, then the data directory, then the output directory.",
    )
    max_rows: int = Field(default=10, description="Max rows to preview for tabular data.")


class ReadArtifactTool(BaseTool):
    name: str = "read_artifact"
    description: str = (
        "Read any pipeline artifact file (CSV, Parquet, JSON, Markdown) and "
        "return a summary suitable for inclusion in the report or for downstream tasks. "
        "For tabular data, returns shape, columns, dtypes, and a preview. "
        "For JSON, returns formatted content. For Markdown, returns the content directly. "
        "Use this to read the previous agent's result file before starting your work."
    )
    args_schema: type[BaseModel] = ReadArtifactInput

    def _run(self, filepath: str, max_rows: int = 10) -> str:
        candidates = []
        if os.path.isabs(filepath):
            candidates.append(filepath)
        else:
            # Allow agents to reference project-root-relative paths like
            # "output/phase1_station_discovery.md" directly.
            candidates.append(os.path.join(PROJECT_ROOT, filepath))
            candidates.append(os.path.join(DATA_DIR, filepath))
            candidates.append(os.path.join(OUTPUT_DIR, filepath))

        full_path = None
        checked = []
        for p in candidates:
            checked.append(os.path.normpath(p))
            if os.path.exists(p):
                full_path = p
                break

        if full_path is None:
            return f"File not found: {filepath} (checked {', '.join(checked)})"

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
        from thesiscrew.tools.report_writer_tool import MarkdownTableTool
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
        from thesiscrew.tools.report_writer_tool import MarkdownTableTool
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
