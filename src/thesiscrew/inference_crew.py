from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

import json
import os

import litellm
import urllib3

from dotenv import load_dotenv
load_dotenv()

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
litellm.ssl_verify = False
litellm.request_timeout = 600

from thesiscrew.tools.pegelonline_tool import (
    GetMeasurementsTool,
    GetMeasurementsCSVTool,
)
from thesiscrew.tools.open_meteo_tool import (
    KorneuburgWeatherTool,
    KorneuburgForecastTool,
)
from thesiscrew.tools.ehyd_tool import StationDataTool
from thesiscrew.tools.data_processing_tool import (
    ListDataFilesTool,
    CSVSummaryTool,
    LagFeaturesTool,
    RollingFeaturesTool,
    CalendarFeaturesTool,
    RateOfChangeTool,
)
from thesiscrew.tools.model_evaluation_tool import ListModelsTool
from thesiscrew.tools.artifact_tool import ReadArtifactTool
from thesiscrew.tools.report_writer_tool import MarkdownTableTool

from thesiscrew.shared import CrewCallbacks, agent_llm, validate_ollama_model, OUTPUT_DIR, ARTIFACTS_DIR


@CrewBase
class InferenceCrew(CrewCallbacks):
    """Lightweight inference crew that refreshes forecasts using trained artifacts.

    Runs much faster than the full training crew because it skips discovery,
    baseline modeling, verification, and report writing.
    """

    agents: list[BaseAgent]
    tasks: list[Task]
    tasks_config = None

    def _validate_artifacts(self, inputs: dict) -> dict:
        """Ensure a trained artifact manifest exists before inference."""
        validate_ollama_model()
        required = ["primary_station", "forecast_horizons_hours"]
        for key in required:
            if key not in inputs:
                raise ValueError(f"Missing required input: {key}")

        manifest_path = os.path.join(ARTIFACTS_DIR, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"No artifact manifest found at {manifest_path}. "
                "Run the training crew first."
            )
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if not manifest.get("models"):
            raise ValueError("Artifact manifest contains no trained models.")

        subdirs = ["data/raw", "data/features", "data/processed", "models"]
        for subdir in subdirs:
            os.makedirs(os.path.join(OUTPUT_DIR, subdir), exist_ok=True)
        return inputs

    @agent
    def inference_data_engineer(self) -> Agent:
        return Agent(
            config={
                "role": "Inference Data Engineer",
                "goal": "Fetch the latest hydrometric and weather data needed for a forecast refresh.",
                "backstory": "You fetch only the most recent data window required for inference. You reuse the same station and parameters as the training run.",
            },
            llm=agent_llm("data_engineer", "openai/gpt-4o-mini"),
            verbose=True,
            allow_delegation=False,
            max_iter=8,
            max_execution_time=300,
            max_rpm=60,
            cache=True,
            tools=[
                GetMeasurementsTool(),
                GetMeasurementsCSVTool(),
                StationDataTool(),
                KorneuburgWeatherTool(),
                KorneuburgForecastTool(),
                ListDataFilesTool(),
                CSVSummaryTool(),
                ReadArtifactTool(),
            ],
        )

    @agent
    def inference_feature_engineer(self) -> Agent:
        return Agent(
            config={
                "role": "Inference Feature Engineer",
                "goal": "Build the same feature vector used during training from the latest data window.",
                "backstory": "You apply the feature manifest from training to new data. You do not invent new features; you replicate the training pipeline exactly.",
            },
            llm=agent_llm("feature_engineer", "openai/gpt-4o"),
            verbose=True,
            allow_delegation=False,
            max_iter=8,
            max_execution_time=300,
            max_rpm=60,
            cache=True,
            tools=[
                LagFeaturesTool(),
                RollingFeaturesTool(),
                CalendarFeaturesTool(),
                RateOfChangeTool(),
                CSVSummaryTool(),
                ReadArtifactTool(),
            ],
        )

    @agent
    def inference_model_engineer(self) -> Agent:
        return Agent(
            config={
                "role": "Inference Model Engineer",
                "goal": "Load the best trained model and generate water-level predictions for all horizons.",
                "backstory": "You load serialized models from output/models/, build a prediction batch, and emit a JSON forecast payload.",
            },
            llm=agent_llm("model_developer", "openai/gpt-4o"),
            verbose=True,
            allow_delegation=False,
            max_iter=8,
            max_execution_time=300,
            max_rpm=60,
            cache=True,
            tools=[
                ListModelsTool(),
                CSVSummaryTool(),
                ReadArtifactTool(),
                MarkdownTableTool(),
            ],
        )

    @agent
    def inference_integration_specialist(self) -> Agent:
        return Agent(
            config={
                "role": "Inference Integration Specialist",
                "goal": "Store the new predictions in the model_predictions table and expose them via the forecast API.",
                "backstory": "You update the production forecast endpoint with the latest predictions and confirm the API responds correctly.",
            },
            llm=agent_llm("integration_specialist", "openai/gpt-4o-mini"),
            verbose=True,
            allow_delegation=False,
            max_iter=6,
            max_execution_time=300,
            max_rpm=60,
            cache=True,
            tools=[
                ListModelsTool(),
                ListDataFilesTool(),
                ReadArtifactTool(),
            ],
        )

    @task
    def fetch_latest_data_task(self) -> Task:
        return Task(
            description=(
                "Fetch the latest hydrometric measurements and weather data for the primary station "
                "and upstream stations identified in input/research_area.json. Use the same "
                "parameters as the training run. Store raw results in output/data/raw/ and summarize "
                "what was fetched."
            ),
            expected_output="Raw data files in output/data/raw/ plus a short fetch summary.",
            agent=self.inference_data_engineer(),
            output_file="output/inference_fetch_latest.md",
            create_directory=True,
        )

    @task
    def build_inference_features_task(self) -> Task:
        return Task(
            description=(
                "Read the training feature manifest from output/data/feature_manifest.json and the latest "
                "raw data, then build an inference feature matrix with the same columns and lags. "
                "Store the result in output/data/features/inference_features.csv."
            ),
            expected_output="A CSV feature matrix aligned with the training manifest.",
            agent=self.inference_feature_engineer(),
            output_file="output/data/features/inference_features.csv",
            create_directory=True,
            context=[self.fetch_latest_data_task()],
        )

    @task
    def generate_predictions_task(self) -> Task:
        return Task(
            description=(
                "Load the best trained model from output/models/ using the artifact manifest. "
                "Generate predictions for all configured forecast horizons. "
                "Write a JSON file output/models/inference_predictions.json with the forecast payload."
            ),
            expected_output="JSON forecast payload with predictions per horizon.",
            agent=self.inference_model_engineer(),
            output_file="output/models/inference_predictions.json",
            create_directory=True,
            context=[self.build_inference_features_task()],
        )

    @task
    def serve_predictions_task(self) -> Task:
        return Task(
            description=(
                "Update the forecast API /api/forecast endpoint with the latest predictions. "
                "Confirm the endpoint returns 200 and document the sample response."
            ),
            expected_output="Updated API endpoint plus a sample JSON response.",
            agent=self.inference_integration_specialist(),
            output_file="output/inference_serve.md",
            create_directory=True,
            context=[self.generate_predictions_task()],
        )

    @crew
    def crew(self) -> Crew:
        openai_key = os.environ.get("OPENAI_API_KEY", "")
        use_memory = bool(openai_key) and openai_key.lower() != "na"
        if not use_memory:
            print("WARNING: OPENAI_API_KEY not set or is 'NA'; disabling crew memory.")
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            memory=use_memory,
            cache=True,
            embedder={"provider": "openai"} if use_memory else None,
            max_rpm=100,
            output_log_file="output/inference_crew_run.json",
            step_callback=self._on_step,
            task_callback=self._on_task_complete,
            before_kickoff_callbacks=[self._validate_artifacts],
            after_kickoff_callbacks=[self._log_results],
        )
