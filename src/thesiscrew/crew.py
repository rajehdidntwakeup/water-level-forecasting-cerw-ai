from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

import os

import litellm
import urllib3

from dotenv import load_dotenv
load_dotenv()

# --- SSL (Daystrom selbstsigniert) ---
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
litellm.ssl_verify = False

# --- Tools ---
from thesiscrew.tools.pegelonline_tool import (
    ListStationsTool,
    StationDetailTool,
    GetMeasurementsTool,
    GetMeasurementsCSVTool,
    GetForecastTool,
    GetWaterBodiesTool,
)
from thesiscrew.tools.open_meteo_tool import (
    HistoricalWeatherTool,
    ForecastWeatherTool,
    KorneuburgWeatherTool,
    KorneuburgForecastTool,
)
from thesiscrew.tools.ehyd_tool import (
    ListAustrianStationsTool,
    StationMetadataTool,
    StationDataTool,
    CharacteristicValuesTool,
)
from thesiscrew.tools.data_processing_tool import (
    ListDataFilesTool,
    CSVSummaryTool,
    ParquetSummaryTool,
    ResampleTool,
    FillGapsTool,
    LagFeaturesTool,
    RollingFeaturesTool,
    CalendarFeaturesTool,
    RateOfChangeTool,
    ChronoSplitTool,
    ComputeMetricsTool,
)
from thesiscrew.tools.model_evaluation_tool import (
    PersistenceBaselineTool,
    WalkForwardTool,
    StratifiedMetricsTool,
    RegisterModelTool,
    ListModelsTool,
)
from thesiscrew.tools.read_input_tool import ReadInputTool
from thesiscrew.tools.report_writer_tool import (
    WriteReportTool,
    ReadReportTool,
    MarkdownTableTool,
    ReportTOCTool,
    RenderMetricsTool,
    ReadArtifactTool,
)


@CrewBase
class Thesiscrew():
    """Water-level forecasting crew for PegelHub thesis project."""

    agents: list[BaseAgent]
    agents_config = "config/agents.yaml"
    tasks: list[Task]
    tasks_config = "config/tasks.yaml"

    # --- Phase 1: Data Discovery & Ingestion ---

    @agent
    def data_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['data_researcher'],
            verbose=True,
            tools=[
                ListStationsTool(),
                StationDetailTool(),
                GetForecastTool(),
                GetWaterBodiesTool(),
                ListAustrianStationsTool(),
                StationMetadataTool(),
                StationDataTool(),
                CharacteristicValuesTool(),
                HistoricalWeatherTool(),
                ForecastWeatherTool(),
                KorneuburgWeatherTool(),
                KorneuburgForecastTool(),
                ReadInputTool(),
            ],
        )

    @agent
    def data_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config['data_engineer'],
            verbose=True,
            tools=[
                GetMeasurementsTool(),
                GetMeasurementsCSVTool(),
                StationDataTool(),
                KorneuburgWeatherTool(),
                KorneuburgForecastTool(),
                ListDataFilesTool(),
                CSVSummaryTool(),
                ParquetSummaryTool(),
                ResampleTool(),
                FillGapsTool(),
                ReadInputTool(),
            ],
        )

    # --- Phase 2: Feature Engineering & Baselines ---

    @agent
    def feature_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config['feature_engineer'],
            verbose=True,
            tools=[
                LagFeaturesTool(),
                RollingFeaturesTool(),
                CalendarFeaturesTool(),
                RateOfChangeTool(),
                ChronoSplitTool(),
                CSVSummaryTool(),
                ParquetSummaryTool(),
                ListDataFilesTool(),
                KorneuburgWeatherTool(),
                ReadInputTool(),
            ],
        )

    @agent
    def model_developer(self) -> Agent:
        return Agent(
            config=self.agents_config['model_developer'],
            verbose=True,
            tools=[
                PersistenceBaselineTool(),
                WalkForwardTool(),
                RegisterModelTool(),
                ListModelsTool(),
                CSVSummaryTool(),
                ParquetSummaryTool(),
                ListDataFilesTool(),
                ReadInputTool(),
            ],
        )

    # --- Phase 4: Integration & Frontend ---

    @agent
    def integration_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config['integration_specialist'],
            verbose=True,
            tools=[
                ListDataFilesTool(),
                CSVSummaryTool(),
                ListModelsTool(),
                RegisterModelTool(),
                ReadInputTool(),
            ],
        )

    # --- Phase 5: Verification & Documentation ---

    @agent
    def verification_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['verification_analyst'],
            verbose=True,
            tools=[
                ComputeMetricsTool(),
                StratifiedMetricsTool(),
                WalkForwardTool(),
                PersistenceBaselineTool(),
                CSVSummaryTool(),
                ParquetSummaryTool(),
                ListDataFilesTool(),
                ReadInputTool(),
            ],
        )

    @agent
    def report_writer(self) -> Agent:
        return Agent(
            config=self.agents_config['report_writer'],
            verbose=True,
            tools=[
                ReadReportTool(),
                WriteReportTool(),
                MarkdownTableTool(),
                ReportTOCTool(),
                RenderMetricsTool(),
                ReadArtifactTool(),
                ReadInputTool(),
                ListDataFilesTool(),
                CSVSummaryTool(),
                ParquetSummaryTool(),
                ListModelsTool(),
            ],
        )

    # --- Tasks (Phase 1) ---

    @task
    def station_discovery_task(self) -> Task:
        return Task(
            config=self.tasks_config['station_discovery_task'],
        )

    @task
    def data_ingestion_task(self) -> Task:
        return Task(
            config=self.tasks_config['data_ingestion_task'],
        )

    # --- Tasks (Phase 2) ---

    @task
    def feature_engineering_task(self) -> Task:
        return Task(
            config=self.tasks_config['feature_engineering_task'],
        )

    @task
    def baseline_modeling_task(self) -> Task:
        return Task(
            config=self.tasks_config['baseline_modeling_task'],
        )

    # --- Tasks (Phase 3) ---

    @task
    def model_training_task(self) -> Task:
        return Task(
            config=self.tasks_config['model_training_task'],
        )

    # --- Tasks (Phase 4) ---

    @task
    def api_integration_task(self) -> Task:
        return Task(
            config=self.tasks_config['api_integration_task'],
        )

    @task
    def frontend_task(self) -> Task:
        return Task(
            config=self.tasks_config['frontend_task'],
        )

    # --- Tasks (Phase 5) ---

    @task
    def verification_task(self) -> Task:
        return Task(
            config=self.tasks_config['verification_task'],
        )

    @task
    def final_documentation_task(self) -> Task:
        return Task(
            config=self.tasks_config['final_documentation_task'],
        )

    @task
    def report_writing_task(self) -> Task:
        return Task(
            config=self.tasks_config['report_writing_task'],
        )

    # --- Crew ---

    @crew
    def crew(self) -> Crew:
        """Creates the water-level forecasting crew."""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )