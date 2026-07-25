from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

import os

import litellm
import urllib3


# --- Tools ---
from thesiscrew.tools.pegelonline_tool import (
    ListStationsTool,
    StationDetailTool,
    GetMeasurementsTool,
    GetMeasurementsCSVTool,
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
    TrainGradientBoostingTool,
)
from thesiscrew.tools.read_input_tool import ReadInputTool
from thesiscrew.tools.dataset_tool import (
    BuildKorneuburgDatasetTool,
    BuildFeatureMatrixTool,
)
from thesiscrew.tools.artifact_tool import ReadArtifactTool
from thesiscrew.tools.report_writer_tool import (
    WriteReportTool,
    ReadReportTool,
    MarkdownTableTool,
    ReportTOCTool,
    RenderMetricsTool,
)
from thesiscrew.tools.html_report_tool import BuildHtmlReportTool
from thesiscrew.tools.forward_forecast_tool import BuildForwardForecastsTool

import litellm
import urllib3
# SSL warning suppression for local/self-signed endpoints
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
litellm.ssl_verify = False
litellm.request_timeout = int(os.environ.get("LITELLM_REQUEST_TIMEOUT", "300"))

from thesiscrew.shared import (
    CrewCallbacks,
    DEFAULT_LLM,
    FeatureManifest,
    BaselineMetrics,
    VerificationReport,
    agent_llm,
    supports_structured_outputs,
    validate_discovery_output,
    validate_feature_output,
    validate_baseline_output,
    validate_verification_output,
    validate_report_output,
    OUTPUT_DIR,
)


@CrewBase
class Thesiscrew(CrewCallbacks):
    """Full training pipeline crew for water-level forecasting."""

    agents: list[BaseAgent]
    agents_config = "config/agents.yaml"
    tasks: list[Task]
    tasks_config = "config/tasks.yaml"

    # -----------------------------------------------------------------------
    # Agents
    # -----------------------------------------------------------------------

    @agent
    def data_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['data_researcher'],
            llm=agent_llm("data_researcher"),
            verbose=True,
            allow_delegation=False,
            output_file='output/data_researcher.md',
            max_retry_limit=2,
            max_iter=10,
            max_execution_time=600,
            max_rpm=60,
            cache=True,
            inject_date=True,
            tools=[
                ListStationsTool(),
                StationDetailTool(),
                ListAustrianStationsTool(),
                StationMetadataTool(),
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
            llm=agent_llm("data_engineer"),
            verbose=True,
            allow_delegation=False,
            output_file='output/data_engineer.md',
            max_retry_limit=2,
            max_iter=12,
            max_execution_time=600,
            max_rpm=60,
            cache=True,
            inject_date=True,
            tools=[
                BuildKorneuburgDatasetTool(),
                GetMeasurementsTool(),
                GetMeasurementsCSVTool(),
                StationDataTool(),
                KorneuburgWeatherTool(),
                KorneuburgForecastTool(),
                ListDataFilesTool(),
                CSVSummaryTool(),
                ResampleTool(),
                FillGapsTool(),
                ReadArtifactTool(),
            ],
        )

    @agent
    def feature_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config['feature_engineer'],
            llm=agent_llm("feature_engineer"),
            verbose=True,
            allow_delegation=False,
            output_file='output/feature_engineer.md',
            max_retry_limit=2,
            max_iter=15,
            max_execution_time=900,
            max_rpm=60,
            cache=True,
            inject_date=True,
            reasoning=True,
            max_reasoning_attempts=2,
            tools=[
                BuildKorneuburgDatasetTool(),
                BuildFeatureMatrixTool(),
                LagFeaturesTool(),
                RollingFeaturesTool(),
                CalendarFeaturesTool(),
                RateOfChangeTool(),
                ChronoSplitTool(),
                CSVSummaryTool(),
                ListDataFilesTool(),
                ReadArtifactTool(),
            ],
        )

    @agent
    def model_developer(self) -> Agent:
        return Agent(
            config=self.agents_config['model_developer'],
            llm=agent_llm("model_developer"),
            verbose=True,
            allow_delegation=False,
            output_file='output/model_developer.md',
            max_retry_limit=2,
            max_iter=20,
            max_execution_time=1200,
            max_rpm=60,
            cache=True,
            inject_date=True,
            reasoning=True,
            max_reasoning_attempts=2,
            tools=[
                TrainGradientBoostingTool(),
                PersistenceBaselineTool(),
                WalkForwardTool(),
                StratifiedMetricsTool(),
                RegisterModelTool(),
                ListModelsTool(),
                CSVSummaryTool(),
                ListDataFilesTool(),
                ReadArtifactTool(),
            ],
        )

    @agent
    def integration_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config['integration_specialist'],
            llm=agent_llm("integration_specialist"),
            verbose=True,
            allow_delegation=False,
            output_file='output/integration_specialist.md',
            max_retry_limit=2,
            max_iter=10,
            max_execution_time=600,
            max_rpm=60,
            cache=True,
            inject_date=True,
            tools=[
                ListDataFilesTool(),
                ListModelsTool(),
                CSVSummaryTool(),
                ReadArtifactTool(),
            ],
        )

    @agent
    def verification_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['verification_analyst'],
            llm=agent_llm("verification_analyst"),
            verbose=True,
            allow_delegation=False,
            output_file='output/verification_analyst.md',
            max_retry_limit=2,
            max_iter=6,
            max_execution_time=600,
            max_rpm=60,
            cache=True,
            inject_date=True,
            tools=[
                ComputeMetricsTool(),
                StratifiedMetricsTool(),
                WalkForwardTool(),
                PersistenceBaselineTool(),
                CSVSummaryTool(),
                ListDataFilesTool(),
                ReadArtifactTool(),
            ],
        )

    @agent
    def report_writer(self) -> Agent:
        return Agent(
            config=self.agents_config['report_writer'],
            llm=agent_llm("report_writer"),
            verbose=True,
            output_file='output/data_output.md',
            allow_delegation=False,
            max_retry_limit=2,
            max_iter=12,
            max_execution_time=1200,
            max_rpm=60,
            cache=True,
            inject_date=True,
            tools=[
                ReadReportTool(),
                WriteReportTool(),
                MarkdownTableTool(),
                ReportTOCTool(),
                RenderMetricsTool(),
                ReadArtifactTool(),
            ],
        )

    @agent
    def frontend_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config['frontend_specialist'],
            llm=agent_llm("frontend_specialist"),
            verbose=True,
            output_file='output/frontend_specialist.md',
            allow_delegation=False,
            max_retry_limit=2,
            max_iter=6,
            max_execution_time=600,
            max_rpm=60,
            cache=True,
            inject_date=True,
            tools=[
                ReadArtifactTool(),
                ListDataFilesTool(),
                BuildHtmlReportTool(),
                BuildForwardForecastsTool(),
            ],
        )

    # -----------------------------------------------------------------------
    # Tasks
    # -----------------------------------------------------------------------

    @task
    def station_discovery_task(self) -> Task:
        return Task(
            config=self.tasks_config['station_discovery_task'],
            output_file='output/phase1_station_discovery.md',
            create_directory=True,
            guardrail=validate_discovery_output,
            guardrail_max_retries=2,
        )

    @task
    def data_ingestion_task(self) -> Task:
        return Task(
            config=self.tasks_config['data_ingestion_task'],
            output_file='output/phase1_data_ingestion.md',
            create_directory=True,
        )

    @task
    def feature_engineering_task(self) -> Task:
        # Local/cloud Ollama models do not reliably emit Pydantic JSON, so fall
        # back to Markdown outputs and guardrails for those providers.
        if supports_structured_outputs():
            return Task(
                config=self.tasks_config['feature_engineering_task'],
                output_file='output/data/feature_manifest.json',
                create_directory=True,
                output_pydantic=FeatureManifest,
                guardrail=validate_feature_output,
                guardrail_max_retries=2,
                callback=self._render_feature_manifest,
            )
        return Task(
            config=self.tasks_config['feature_engineering_task'],
            output_file='output/phase2_feature_engineering.md',
            create_directory=True,
            guardrail=validate_feature_output,
            guardrail_max_retries=2,
        )

    @task
    def baseline_modeling_task(self) -> Task:
        if supports_structured_outputs():
            return Task(
                config=self.tasks_config['baseline_modeling_task'],
                output_file='output/models/baseline_metrics.json',
                create_directory=True,
                output_pydantic=BaselineMetrics,
                guardrail=validate_baseline_output,
                guardrail_max_retries=2,
                callback=self._render_baseline_metrics,
            )
        return Task(
            config=self.tasks_config['baseline_modeling_task'],
            output_file='output/phase2_baseline_modeling.md',
            create_directory=True,
            guardrail=validate_baseline_output,
            guardrail_max_retries=2,
        )

    @task
    def model_training_task(self) -> Task:
        return Task(
            config=self.tasks_config['model_training_task'],
            output_file='output/phase3_model_training.md',
            create_directory=True,
        )

    @task
    def api_integration_task(self) -> Task:
        return Task(
            config=self.tasks_config['api_integration_task'],
            output_file='output/phase4_api_integration.md',
            create_directory=True,
            async_execution=True,
        )

    @task
    def frontend_task(self) -> Task:
        return Task(
            config=self.tasks_config['frontend_task'],
            output_file='output/phase4_frontend.md',
            create_directory=True,
            async_execution=True,
        )

    @task
    def verification_baseline_task(self) -> Task:
        return Task(
            config=self.tasks_config['verification_baseline_task'],
            output_file='output/models/verification_baseline.json',
            create_directory=True,
            async_execution=True,
        )

    @task
    def verification_walkforward_task(self) -> Task:
        return Task(
            config=self.tasks_config['verification_walkforward_task'],
            output_file='output/models/verification_walkforward.json',
            create_directory=True,
            async_execution=True,
        )

    @task
    def verification_stratified_task(self) -> Task:
        return Task(
            config=self.tasks_config['verification_stratified_task'],
            output_file='output/models/verification_stratified.json',
            create_directory=True,
            async_execution=True,
        )

    @task
    def verification_synthesis_task(self) -> Task:
        return Task(
            config=self.tasks_config['verification_synthesis_task'],
            output_file='output/phase5_verification.md',
            create_directory=True,
            guardrail=validate_verification_output,
            guardrail_max_retries=2,
            callback=self._render_verification_report,
            context=[
                self.verification_baseline_task(),
                self.verification_walkforward_task(),
                self.verification_stratified_task(),
                self.model_training_task(),
                self.baseline_modeling_task(),
                self.data_ingestion_task(),
            ],
        )

    @task
    def final_documentation_task(self) -> Task:
        return Task(
            config=self.tasks_config['final_documentation_task'],
            output_file='output/phase5_final_documentation.md',
            create_directory=True,
        )

    @task
    def report_writing_task(self) -> Task:
        # Build context dynamically so resuming does not reference skipped tasks.
        candidate_context = [
            self.station_discovery_task(),
            self.data_ingestion_task(),
            self.feature_engineering_task(),
            self.baseline_modeling_task(),
            self.model_training_task(),
            self.verification_baseline_task(),
            self.verification_walkforward_task(),
            self.verification_stratified_task(),
            self.verification_synthesis_task(),
            self.final_documentation_task(),
        ]
        completed = self._load_completed_output_files()
        active_context = [
            t for t in candidate_context
            if getattr(t, "output_file", None) not in completed
        ]
        # Always include final docs and synthesis if they exist; otherwise the
        # report writer has nothing to compile. Fall back to all candidates when
        # resume is not active.
        if not active_context:
            active_context = candidate_context

        return Task(
            config=self.tasks_config['report_writing_task'],
            output_file='output/phase5_report.md',
            create_directory=True,
            guardrail=validate_report_output,
            guardrail_max_retries=2,
            context=active_context,
        )

    @task
    def frontend_final_report_task(self) -> Task:
        return Task(
            config=self.tasks_config['frontend_final_report_task'],
            output_file='output/phase5_frontend_final_report.md',
            create_directory=True,
            context=[
                self.report_writing_task(),
                self.verification_synthesis_task(),
                self.final_documentation_task(),
            ],
        )

    # -----------------------------------------------------------------------
    # Crew
    # -----------------------------------------------------------------------

    @crew
    def crew(self) -> Crew:
        """Creates the optimized water-level forecasting crew."""
        openai_key = os.environ.get("OPENAI_API_KEY", "")
        use_memory = bool(openai_key) and openai_key.lower() != "na"
        if not use_memory:
            print("WARNING: OPENAI_API_KEY not set or is 'NA'; disabling crew memory. "
                  "Set a valid key to enable cross-agent memory and context reuse.")

        tasks = self._filter_tasks_for_resume(self.tasks)

        return Crew(
            agents=self.agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
            memory=use_memory,
            cache=True,
            embedder={"provider": "openai"} if use_memory else None,
            max_rpm=100,
            output_log_file="output/crew_run.json",
            step_callback=self._on_step,
            task_callback=self._on_task_complete,
            before_kickoff_callbacks=[self._validate_inputs],
            after_kickoff_callbacks=[self._log_results, self._write_artifact_manifest],
        )
