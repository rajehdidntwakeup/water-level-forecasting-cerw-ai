# Optimization Plan for `thesiscrew`

This plan analyzes the current CrewAI-based water-level forecasting project and recommends concrete optimizations for agents, tasks, crew execution, tools, LLM usage, and production readiness. It draws on the official CrewAPI surface documented in [`docs/crewai-functions.md`](crewai-functions.md) and the current project files (`src/thesiscrew/crew.py`, `src/thesiscrew/config/agents.yaml`, `src/thesiscrew/config/tasks.yaml`, and tools).

---

## 1. Executive Summary

The current crew is a **single sequential pipeline** with 7 agents and 10 tasks split across 5 phases (data discovery → ingestion → feature engineering → modeling → integration → verification → reporting). While well-structured, there are several optimization opportunities:

1. **Agents are over-provisioned with tools** — each agent has many tools, increasing LLM decision fatigue and cost.
2. **No output guardrails** — tasks can silently produce malformed outputs, propagating errors downstream.
3. **No async/parallelization** — tasks run sequentially even when they could run in parallel (e.g., API integration + frontend are independent after models exist).
4. **No checkpointing** — long-running model training/evaluation can fail and lose progress.
5. **No structured outputs** — report compilation and data discovery rely on free-text Markdown instead of validated schemas.
6. **Tool patch is brittle** — monkey-patching CrewAI's `_validate_tool_input` fixes a model-specific issue but should be replaced with a more robust approach.
7. **No crew memory or knowledge sources** — agents do not share prior context across the workflow.

This document proposes a prioritized optimization roadmap.

---

## 2. Agent Optimization

### 2.1 Reduce Tool Bloat Per Agent

Current agents carry many overlapping tools. For example, `data_researcher` has 12 tools, including `ReadInputTool`, `HistoricalWeatherTool`, `KorneuburgWeatherTool`, etc. This increases token cost and tool-selection errors.

**Recommendation:**

- Assign each agent only the tools it absolutely needs for its primary responsibility.
- Introduce a shared `ReadInputTool()` and `ListDataFilesTool()` only where truly necessary.
- Group related tools into narrower agent roles.

**Example: Split `data_researcher` into two agents:**

```python
class Thesiscrew():
    @agent
    def hydrometric_researcher(self) -> Agent:
        return Agent(
            role="Hydrometric Station Researcher",
            goal="Discover Pegelonline and eHYD station metadata and availability.",
            backstory="...",
            tools=[
                ListStationsTool(),
                StationDetailTool(),
                ListAustrianStationsTool(),
                StationMetadataTool(),
                CharacteristicValuesTool(),
            ],
            max_iter=10,
            max_retry_limit=2,
            verbose=True,
        )

    @agent
    def meteorological_researcher(self) -> Agent:
        return Agent(
            role="Meteorological Data Researcher",
            goal="Discover and validate Open-Meteo weather and forecast data.",
            backstory="...",
            tools=[
                HistoricalWeatherTool(),
                ForecastWeatherTool(),
                KorneuburgWeatherTool(),
                KorneuburgForecastTool(),
            ],
            max_iter=10,
            max_retry_limit=2,
            verbose=True,
        )
```

### 2.2 Enable Agent Reasoning for Complex Tasks

The `model_developer`, `feature_engineer`, and `verification_analyst` perform multi-step reasoning. Enable `reasoning=True` so they plan before executing.

```python
Agent(
    role="ML Forecasting Model Developer",
    goal="...",
    reasoning=True,
    max_reasoning_attempts=3,
    max_iter=15,
)
```

### 2.3 Set Per-Agent Timeouts and Iteration Limits

All agents currently use `max_iter=15`. Lower this for simpler agents and raise it only for model training:

| Agent | Current `max_iter` | Recommended | Rationale |
|---|---|---|---|
| `data_researcher` | 15 | 10 | Metadata lookup is bounded. |
| `data_engineer` | 15 | 12 | Fetching and resampling is deterministic. |
| `feature_engineer` | 15 | 15 | Feature engineering may need iteration. |
| `model_developer` | 15 | 20–25 | Training loops may need more attempts. |
| `verification_analyst` | 15 | 12 | Verification is mostly computation. |
| `integration_specialist` | 15 | 10 | API wiring is bounded. |
| `report_writer` | 15 | 12 | Report assembly is bounded. |

Also add `max_execution_time` to prevent hung tool calls:

```python
Agent(
    ..., 
    max_execution_time=600,  # 10 minutes
    max_rpm=60,
)
```

### 2.4 Use Agent-Specific LLMs

Currently all agents inherit the default LLM. Use cheaper/faster models for simple agents and stronger models for model development and verification.

```python
Agent(
    role="Data Ingestion Engineer",
    llm="openai/gpt-4o-mini",  # fast, cheap
    ...
)

Agent(
    role="ML Forecasting Model Developer",
    llm="openai/gpt-4o",  # stronger reasoning
    ...
)
```

### 2.5 Avoid `allow_code_execution`

The project already avoids this deprecated option. Continue using tool-based code execution and external sandboxes if needed.

---

## 3. Task Optimization

### 3.1 Add Output Guardrails

Tasks currently have no `guardrail` or `guardrails`. Add lightweight validation to catch empty outputs, missing sections, or malformed JSON.

**Example guardrail for station discovery:**

```python
from typing import Tuple, Any
from crewai import TaskOutput

def validate_discovery_output(output: TaskOutput) -> Tuple[bool, Any]:
    text = output.raw or ""
    required = ["pegelonline", "eHYD", "Open-Meteo", "completeness"]
    missing = [r for r in required if r.lower() not in text.lower()]
    if missing:
        return (False, f"Missing required sections: {', '.join(missing)}")
    return (True, output.raw)
```

Apply it in `tasks.yaml` or programmatically:

```python
@task
def station_discovery_task(self) -> Task:
    return Task(
        config=self.tasks_config['station_discovery_task'],
        guardrail=validate_discovery_output,
        guardrail_max_retries=2,
    )
```

### 3.2 Use Structured Outputs Where Possible

Tasks like `feature_engineering_task` and `baseline_modeling_task` produce metrics and manifests. Replace free-text parsing with Pydantic outputs.

```python
from pydantic import BaseModel
from typing import Dict

class BaselineMetrics(BaseModel):
    model: str
    horizons: Dict[str, Dict[str, float]]  # horizon -> {rmse, mae, nse, bias}

@task
def baseline_modeling_task(self) -> Task:
    return Task(
        config=self.tasks_config['baseline_modeling_task'],
        output_pydantic=BaselineMetrics,
        output_file="output/models/baseline_metrics.json",
    )
```

### 3.3 Add Task Callbacks for Progress Tracking

Track task completion and emit structured logs:

```python
def on_task_complete(output):
    print(f"✅ Task '{output.description}' completed by {output.agent}")
    if output.output_format.value == "Pydantic":
        print(output.pydantic.model_dump_json())

Task(..., callback=on_task_complete)
```

### 3.4 Mark Tasks as `async_execution=True` Where Independent

Tasks that don't depend on previous task outputs can run asynchronously:

- `station_discovery_task` and `data_ingestion_task` are sequential in reality (ingestion needs discovery).
- But within a phase, tasks like `api_integration_task` and `frontend_task` could run in parallel **after** `model_training_task` completes.

Consider splitting Phase 4 into two tasks that run in parallel by assigning them `async_execution=True` and adding a synchronization task afterward.

### 3.5 Use `context` More Precisely

The `report_writing_task` currently receives **all** prior tasks as context. This inflates the prompt. Pass only the **outputs** actually needed.

```python
@task
def report_writing_task(self) -> Task:
    return Task(
        config=self.tasks_config['report_writing_task'],
        context=[
            self.station_discovery_task(),
            self.data_ingestion_task(),
            self.feature_engineering_task(),
            self.baseline_modeling_task(),
            self.model_training_task(),
            self.verification_task(),
            self.final_documentation_task(),
        ],
    )
```

Remove `api_integration_task` and `frontend_task` from the report context unless the report explicitly needs integration details.

### 3.6 Add `output_file` and `create_directory` to Key Tasks

Ensure important artifacts are persisted automatically:

```python
Task(
    config=self.tasks_config['feature_engineering_task'],
    output_file="output/data/features/feature_manifest.md",
    create_directory=True,
)
```

---

## 4. Crew-Level Optimization

### 4.1 Switch to Hierarchical Process with a Manager Agent

The current sequential process may cause issues when one agent fails or when tasks could be delegated. A **hierarchical** process with a dedicated manager agent can:

- Route tasks based on outputs.
- Handle failures gracefully.
- Decide when to parallelize.

```python
from crewai import Agent, Crew, Process

@agent
def manager(self) -> Agent:
    return Agent(
        role="Project Manager",
        goal="Coordinate the water-level forecasting pipeline, ensure each phase produces valid outputs, and delegate replanning when needed.",
        backstory="...",
        allow_delegation=True,
        llm="openai/gpt-4o",
    )

@crew
def crew(self) -> Crew:
    return Crew(
        agents=self.agents,
        tasks=self.tasks,
        process=Process.hierarchical,
        manager_agent=self.manager(),
        verbose=True,
        memory=True,
        planning=True,
        planning_llm="openai/gpt-4o-mini",
    )
```

### 4.2 Enable Crew Memory

Set `memory=True` so agents share short-term, long-term, and entity memory. This is especially useful for the report writer, which needs to recall decisions made in earlier phases.

```python
Crew(
    agents=self.agents,
    tasks=self.tasks,
    memory=True,
    embedder={"provider": "openai"},
    ...
)
```

### 4.3 Enable Checkpointing

Model training and data ingestion are long-running. Enable checkpointing to resume from failures:

```python
from crewai.state.checkpoint_config import CheckpointConfig

Crew(
    agents=self.agents,
    tasks=self.tasks,
    checkpoint=CheckpointConfig(
        location="output/.checkpoints",
        on_events=["task_completed", "crew_kickoff_failed"],
        max_checkpoints=10,
    ),
)
```

Resume later:

```python
crew = Crew.from_checkpoint("output/.checkpoints/latest.json")
crew.kickoff()
```

### 4.4 Add `before_kickoff_callbacks` and `after_kickoff_callbacks`

Use callbacks to validate inputs, seed directories, and summarize outputs.

```python
def validate_inputs(inputs):
    required = ["primary_station", "forecast_horizons_hours", "data_sources"]
    for key in required:
        if key not in inputs:
            raise ValueError(f"Missing required input: {key}")
    return inputs

def log_results(result):
    with open("output/run_summary.json", "w") as f:
        json.dump({
            "raw": result.raw[:500],
            "token_usage": result.token_usage,
            "tasks_completed": len(result.tasks_output),
        }, f, indent=2)

Crew(
    ...,
    before_kickoff_callbacks=[validate_inputs],
    after_kickoff_callbacks=[log_results],
)
```

### 4.5 Use `task_callback` and `step_callback`

Track execution granularly:

```python
Crew(
    ...,
    step_callback=lambda agent, step: print(f"{agent}: {step}"),
    task_callback=lambda task_output: print(f"Task done: {task_output.description}"),
)
```

### 4.6 Set `max_rpm` and `output_log_file`

Avoid rate limits and persist execution logs:

```python
Crew(
    ...,
    max_rpm=100,
    output_log_file="output/crew_run.json",
)
```

---

## 5. Tool Optimization

### 5.1 Replace Monkey Patch with Robust Tool Input Handling

The current `main.py` monkey-patches `ToolUsage._validate_tool_input` to handle list-of-dicts tool inputs from `glm-5.1:cloud`. This is fragile and may break on CrewAI updates.

**Recommendations:**

1. **Prefer a provider/model that emits single-dict tool calls** when possible.
2. If the model must be used, wrap the patch in a try/except and log warnings.
3. Add a dedicated wrapper tool that accepts lists and merges them into a single dict:

```python
class SafeReadInputTool(BaseTool):
    name: str = "safe_read_input"
    description: str = "Wrapper around ReadInputTool that normalizes list inputs."
    args_schema: type[BaseModel] = ReadInputInput

    def _run(self, file_key="all"):
        if isinstance(file_key, list):
            file_key = file_key[0] if file_key else "all"
        return ReadInputTool()._run(file_key)
```

### 5.2 Add Caching and Idempotency to Data Tools

Tools like `GetMeasurementsTool` and `HistoricalWeatherTool` fetch external data. Cache results to avoid redundant API calls:

```python
from crewai.tools import BaseTool
from functools import lru_cache
import hashlib
import json

class CachedGetMeasurementsTool(GetMeasurementsTool):
    def _run(self, **kwargs):
        cache_key = hashlib.md5(json.dumps(kwargs, sort_keys=True).encode()).hexdigest()
        cache_path = f"output/.cache/measurements_{cache_key}.json"
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                return f.read()
        result = super()._run(**kwargs)
        with open(cache_path, "w") as f:
            f.write(result)
        return result
```

Alternatively, rely on CrewAI's built-in `cache=True` on the agent.

### 5.3 Standardize Tool Error Responses

Tools currently return `json.dumps({"error": ...})`. Consider raising exceptions for truly fatal errors so guardrails/callbacks can catch them. Return structured errors only for recoverable cases.

### 5.4 Add Input Validation to Tools

Tools like `LagFeaturesTool` assume CSV files exist and contain the target column. Add validation:

```python
def _run(self, filepath, column, lags):
    full_path = ...
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"File not found: {full_path}")
    df = pd.read_csv(full_path)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found. Available: {list(df.columns)}")
    ...
```

---

## 6. Execution and Performance

### 6.1 Use `akickoff` for Non-Interactive Runs

If running the crew unattended (e.g., scheduled retraining), use the native async method:

```python
async def main():
    crew = Thesiscrew().crew()
    result = await crew.akickoff(inputs=inputs)
    print(result.raw)
```

### 6.2 Batch Execution with `kickoff_for_each`

If evaluating multiple stations or scenarios, use `kickoff_for_each`:

```python
inputs_list = [
    {"primary_station": "Korneuburg"},
    {"primary_station": "Wien / Reichsbrücke"},
    {"primary_station": "Linz-Donau"},
]
results = crew.kickoff_for_each(inputs=inputs_list)
```

### 6.3 Add `max_execution_time` and Timeouts

Set crew-wide and agent-level timeouts to avoid runaway tasks:

```python
Agent(..., max_execution_time=900)  # 15 min per agent step
Crew(..., max_rpm=120)
```

---

## 7. LLM and Cost Optimization

### 7.1 Use Tiered Models

| Task Tier | Suggested Model | Rationale |
|---|---|---|
| Data/metadata lookup | `openai/gpt-4o-mini` or `gemini/gemini-1.5-flash` | Cheap, fast |
| Feature engineering planning | `openai/gpt-4o` | Needs reasoning |
| Model code generation | `openai/gpt-4o` or `anthropic/claude-sonnet-4-6` | Strong code |
| Verification analysis | `openai/gpt-4o` | Strong analysis |
| Report writing | `openai/gpt-4o-mini` | Summarization |

### 7.2 Respect Context Window

Keep `respect_context_window=True` (default). For the `report_writing_task`, pass only essential prior outputs to avoid truncation.

### 7.3 Inject Current Date

For forecast freshness, enable `inject_date=True` on agents that reason about recent data:

```python
Agent(
    role="Hydrometric Data Researcher",
    inject_date=True,
    date_format="%Y-%m-%d",
)
```

---

## 8. Training and Testing

### 8.1 Use `crewai train` for Agent Prompt Tuning

After running the pipeline several times, collect feedback:

```bash
crewai train -n 5 -f output/trained_agents_data.pkl
```

Then load the trained data on normal runs:

```python
Crew(
    ...,
    # Note: CrewAI docs reference trained_agents_file; verify exact param name for your version
)
```

### 8.2 Use `crewai test` for Regression Testing

Before releasing changes:

```bash
crewai test -n 3 -m gpt-4o-mini
```

Track scores over time to detect regressions.

---

## 9. Production Readiness

### 9.1 Separate Training from Inference

The current pipeline mixes data discovery, feature engineering, model training, and API deployment in one crew. Consider splitting into two crews:

1. **Training Crew**: data → features → baselines → models → verification → trained artifacts.
2. **Inference/Integration Crew**: load trained model → fetch latest data → generate predictions → serve via API.

This avoids retraining on every scheduled inference run and reduces cost.

### 9.2 Add Health Checks and Logging

- Persist logs with `output_log_file=True` or a path.
- Use `before_kickoff_callbacks` to validate `research_area.json` schema.
- Use `after_kickoff_callbacks` to write a `run_status.json` with success/failure and key metrics.

### 9.3 Use `security_config`

For production deployments, configure fingerprinting/identity:

```python
from crewai.security import SecurityConfig

Crew(
    ...,
    security_config=SecurityConfig(),
)
```

---

## 10. Recommended Optimization Roadmap

### Phase A — Quick Wins (1–2 days)

1. **Prune tools per agent** (remove overlap).
2. **Add `max_execution_time` and `max_rpm`** to agents and crew.
3. **Enable `memory=True`** on the crew.
4. **Add guardrails** to discovery, feature engineering, and verification tasks.
5. **Reduce `report_writing_task` context** to essential tasks only.
6. **Add `output_log_file`** and `before/after_kickoff_callbacks`.

### Phase B — Structural Improvements (1 week)

1. **Introduce a manager agent** and switch to `Process.hierarchical`.
2. **Enable checkpointing** for long-running tasks.
3. **Use `output_pydantic`/`output_json`** for metric and manifest tasks.
4. **Split Phase 4 tasks** into parallel async tasks where possible.
5. **Add task callbacks** to emit structured progress events.

### Phase C — Production Hardening (2+ weeks)

1. **Split into Training Crew and Inference Crew**.
2. **Replace monkey patch** with safe tool wrappers.
3. **Add tool-level caching** for external API calls.
4. **Implement `crewai train` workflow** and persist trained agent guidance.
5. **Add `crewai test` regression suite**.
6. **Add OpenTelemetry tracing** (`tracing=True`) for production observability.

---

## 11. Example Optimized `crew.py` Snippet

```python
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.state.checkpoint_config import CheckpointConfig
from typing import Tuple, Any
from crewai import TaskOutput


def validate_discovery_output(output: TaskOutput) -> Tuple[bool, Any]:
    text = output.raw or ""
    required = ["pegelonline", "eHYD", "Open-Meteo", "completeness"]
    missing = [r for r in required if r.lower() not in text.lower()]
    if missing:
        return (False, f"Missing sections: {', '.join(missing)}")
    return (True, output.raw)


@CrewBase
class Thesiscrew():
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def data_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['data_researcher'],
            verbose=True,
            allow_delegation=False,
            reasoning=True,
            max_reasoning_attempts=2,
            max_iter=10,
            max_execution_time=600,
            max_retry_limit=2,
            cache=True,
            inject_date=True,
            tools=[
                ListStationsTool(),
                StationDetailTool(),
                GetMeasurementsTool(),
                GetForecastTool(),
                ListAustrianStationsTool(),
                StationMetadataTool(),
                CharacteristicValuesTool(),
                HistoricalWeatherTool(),
                ForecastWeatherTool(),
            ],
        )

    @task
    def station_discovery_task(self) -> Task:
        return Task(
            config=self.tasks_config['station_discovery_task'],
            guardrail=validate_discovery_output,
            guardrail_max_retries=2,
            output_file="output/data/station_discovery_report.md",
            create_directory=True,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.hierarchical,
            manager_llm="openai/gpt-4o",
            verbose=True,
            memory=True,
            cache=True,
            max_rpm=100,
            output_log_file="output/crew_run.json",
            checkpoint=CheckpointConfig(
                location="output/.checkpoints",
                on_events=["task_completed"],
                max_checkpoints=10,
            ),
            before_kickoff_callbacks=[self._validate_inputs],
            after_kickoff_callbacks=[self._log_results],
        )

    def _validate_inputs(self, inputs):
        required = ["primary_station", "forecast_horizons_hours", "data_sources"]
        for key in required:
            assert key in inputs, f"Missing required input: {key}"
        return inputs

    def _log_results(self, result):
        import json
        with open("output/run_summary.json", "w") as f:
            json.dump({
                "tasks": len(result.tasks_output),
                "token_usage": result.token_usage,
                "summary": result.raw[:500],
            }, f, indent=2)
        return result
```

---

## 12. Key Metrics to Track

After applying optimizations, monitor:

| Metric | How to Track |
|---|---|
| Cost per run | `result.token_usage` and provider billing |
| Run duration | Wall-clock time of `crew.kickoff()` |
| Success rate | Fraction of runs completing without guardrail failures |
| Retry rate | `max_retry_limit` hits and guardrail retries |
| Tool redundancy | Number of tool calls per task from logs |
| Model accuracy | Verification metrics (RMSE, MAE, NSE) vs official forecast |

---

## 13. Conclusion

The `thesiscrew` project is already well-organized, but it can be made faster, cheaper, more reliable, and more maintainable by:

- **Narrowing agent tool sets** and enabling reasoning/timeouts.
- **Adding guardrails, structured outputs, and callbacks** to tasks.
- **Switching to hierarchical process with a manager agent** and enabling memory/checkpointing.
- **Splitting training and inference** for production use.
- **Hardening tool inputs** and adding caching.

Start with Phase A quick wins for immediate cost and reliability improvements, then proceed to Phase B and C based on project timeline and budget.
