#!/usr/bin/env python
import sys
import warnings
import argparse

from datetime import datetime

import json
from thesiscrew.crew import Thesiscrew
import os
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run() -> None:
    """
    Run the crew.
    """
    # Ensure output directories exist
    output_dir = os.environ.get("PEGELHUB_OUTPUT_DIR", "output")
    for subdir in ["data", "models"]:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    # Load inputs from the research_area.json file
    input_file = "input/research_area.json"
    try:
        with open(input_file, 'r') as f:
            inputs = json.load(f)
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: {input_file} is not a valid JSON file.")
        return

    try:
        crew = Thesiscrew().crew()
        result = crew.kickoff(inputs=inputs)
        print("\n\n=== Crew Execution Complete ===")
        print(f"Result: {result}")
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def main():
    """Main entry point."""
    run()


if __name__ == "__main__":
    main()


