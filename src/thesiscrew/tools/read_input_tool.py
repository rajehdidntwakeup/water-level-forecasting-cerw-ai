"""CrewAI tool for reading research area input files.

Reads input/research_area.json and input/user_input.md to provide
context about the forecasting target and user requirements.
"""

import json
import os
from typing import Type

from pydantic import BaseModel, Field
from crewai.tools import BaseTool

PROJECT_ROOT = os.environ.get(
    "PEGELHUB_PROJECT_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")),
)


class ReadInputInput(BaseModel):
    file_key: str = Field(
        default="all",
        description="Which input to read: 'research_area', 'user_input', or 'all'.",
    )


class ReadInputTool(BaseTool):
    name: str = "read_input"
    description: str = (
        "Read project input files: research_area.json (station config, API "
        "endpoints) and user_input.md (requirements and constraints). "
        "Provides context about the forecasting target."
    )
    args_schema: Type[BaseModel] = ReadInputInput

    def _run(self, file_key: str = "all") -> str:
        results = {}
        input_dir = os.path.join(PROJECT_ROOT, "input")
        if file_key in ("research_area", "all"):
            ra_path = os.path.join(input_dir, "research_area.json")
            if os.path.exists(ra_path):
                with open(ra_path) as f:
                    results["research_area"] = json.load(f)
            else:
                results["research_area"] = {"error": f"File not found: {ra_path}"}
        if file_key in ("user_input", "all"):
            ui_path = os.path.join(input_dir, "user_input.md")
            if os.path.exists(ui_path):
                with open(ui_path) as f:
                    results["user_input"] = f.read()
            else:
                results["user_input"] = {"error": f"File not found: {ui_path}"}
        return json.dumps(results, indent=2, ensure_ascii=False)