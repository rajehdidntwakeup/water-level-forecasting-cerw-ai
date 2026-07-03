#!/usr/bin/env python
import sys

# Windows terminals default to cp1252 and choke on CrewAI's emoji/log output.
# Force UTF-8 with replacement so logs never crash the crew.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import argparse
import json
import os
import warnings

from datetime import datetime

from thesiscrew.crew import Thesiscrew
from thesiscrew.inference_crew import InferenceCrew

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


# --- Monkey-patch CrewAI tool-input validation ---
# glm-5.1:cloud generates tool calls as JSON arrays of dicts
# (e.g. [{"file_key":"all"}, {"subdir":""}]) instead of a single dict.
# CrewAI's _validate_tool_input rejects all non-dict results.
# This patch unwraps list-of-dicts so the first tool call executes.
def _patch_crewai_tool_validation():
    try:
        from crewai.tools.tool_usage import ToolUsage
        _original_validate = ToolUsage._validate_tool_input

        def _patched_validate(self, tool_input, **kwargs):
            try:
                return _original_validate(self, tool_input, **kwargs)
            except Exception:
                # Try parsing as JSON and unwrapping list-of-dicts
                import json as _json
                for parser in (
                    lambda s: _json.loads(s),
                    lambda s: __import__("ast").literal_eval(s),
                ):
                    try:
                        parsed = parser(tool_input)
                        break
                    except Exception:
                        continue
                else:
                    raise

                if isinstance(parsed, list):
                    # Merge all dicts into one (later keys win) or take first dict
                    merged = {}
                    for item in parsed:
                        if isinstance(item, dict):
                            merged.update(item)
                    if merged:
                        return merged
                    # If list contains non-dicts, take first element as scalar
                    if len(parsed) == 1:
                        return parsed[0]

                raise

        ToolUsage._validate_tool_input = _patched_validate
    except Exception as e:
        print(f"Warning: could not patch CrewAI tool validation: {e}")


_patch_crewai_tool_validation()


DEFAULT_INPUT_FILE = "input/research_area.json"


def _load_inputs(input_file: str = DEFAULT_INPUT_FILE) -> dict:
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: {input_file} is not a valid JSON file.")
        sys.exit(1)


def run(mode: str = "train", inputs: dict | None = None) -> None:
    """Run either the training or inference crew."""
    if inputs is None:
        inputs = _load_inputs()

    if mode == "infer":
        crew = InferenceCrew().crew()
        log_file = "output/inference_crew_run.json"
    else:
        crew = Thesiscrew().crew()
        log_file = "output/crew_run.json"

    try:
        result = crew.kickoff(inputs=inputs)
        print("\n\n=== Crew Execution Complete ===")
        print(f"Result: {result}")
        print(f"Execution log: {log_file}")
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train(
    n_iterations: int = 5,
    filename: str = "output/trained_agents_data.pkl",
    inputs: dict | None = None,
) -> None:
    """Run the training crew in interactive training mode."""
    if inputs is None:
        inputs = _load_inputs()
    crew = Thesiscrew().crew()
    try:
        crew.train(
            n_iterations=n_iterations,
            inputs=inputs,
            filename=filename,
        )
        print(f"\n\n=== Training Complete ===")
        print(f"Trained agent data saved to: {filename}")
    except Exception as e:
        raise Exception(f"An error occurred during training: {e}")


def test(n_iterations: int = 3, model: str = "gpt-4o-mini") -> None:
    """Run the crew in test/regression mode."""
    crew = Thesiscrew().crew()
    try:
        crew.test(n_iterations=n_iterations, model=model)
        print("\n\n=== Test Complete ===")
    except Exception as e:
        raise Exception(f"An error occurred during testing: {e}")


def replay(task_id: str | None = None) -> None:
    """Stub for crewai replay CLI command."""
    print("Replay is not implemented in this version.")


def run_with_trigger() -> None:
    """Stub for external trigger entry point."""
    run(mode="infer")


def main() -> None:
    """Main entry point with subcommands for train, infer, and test."""
    parser = argparse.ArgumentParser(
        description="Water-level forecasting thesis crew",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "infer"],
        default="train",
        help="Run the full training crew (train) or the lightweight inference crew (infer).",
    )
    parser.add_argument(
        "--input-file",
        default=DEFAULT_INPUT_FILE,
        help="Path to the input JSON file.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run in interactive training mode (crewai train).",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in regression test mode (crewai test).",
    )
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=5,
        help="Number of iterations for train or test.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Evaluation model for crewai test.",
    )
    parser.add_argument(
        "--trained-file",
        default="output/trained_agents_data.pkl",
        help="Output file for crewai train.",
    )

    args = parser.parse_args()
    inputs = _load_inputs(args.input_file)

    if args.test:
        test(n_iterations=args.n_iterations, model=args.model)
    elif args.train:
        train(n_iterations=args.n_iterations, filename=args.trained_file, inputs=inputs)
    else:
        run(mode=args.mode, inputs=inputs)


if __name__ == "__main__":
    main()
