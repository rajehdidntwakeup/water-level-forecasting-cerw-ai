# Step-by-Step Optimization Plan for `thesiscrew`

**Goal:** Produce perfect water-level forecasting results and dramatically reduce crew runtime.  
**Scope:** Apply CrewAPI capabilities from [`docs/crewai-functions.md`](crewai-functions.md) and the recommendations from [`docs/optimizing.md`](optimizing.md) in a strict execution order that builds safely from quick wins to structural changes.  
**Outcomes expected:** lower cost, fewer failures, faster execution, validated artifacts, and a production-ready training/inference split.

---

## How to use this plan

1. Execute each phase in order. Do **not** skip Phase A — it provides the safety net for the larger changes in Phase B.
2. Verify the checkpoint at the end of each phase before moving to the next.
3. Track the metrics listed in §7 in every run so progress is measurable.
4. Where a code snippet uses `openai/gpt-4o`, substitute the provider/model you actually have configured if it is equally capable for reasoning/code.

---

## Phase A — Safety & Speed Foundations (Days 1–2)

*Objective:* Reduce cost and failure rate immediately without changing the agent topology.

### A1. Lock inputs and seed the run

**Files:** `src/thesiscrew/main.py`, `src/thesiscrew/crew.py`  
**Action:**
- Add a `before_kickoff_callbacks` to `Crew` that validates all required inputs (`primary_station`, `forecast_horizons_hours`, `data_sources`, etc.).
- In the callback, create `output/` and any expected subdirectories before any agent runs.

**Why:** Prevents half-runs caused by missing keys or missing folders; first failures are the cheapest to catch.

**Verification:** Run `crew.kickoff(inputs={})` and confirm it raises immediately with a clear message.

### A2. Reduce tool bloat per agent

**Files:** `src/thesiscrew/crew.py`  
**Action:**
- Inspect every agent's `tools=` list.
- Remove tools that the agent does not use in its role description.  Keep only the tools whose names appear in the agent's goal/backstory.
- Ensure `ReadInputTool` and `ListDataFilesTool` are only given to agents that actually read files.

**Why:** Every extra tool is an extra candidate in every LLM tool-decision call.  Fewer tools = faster tool selection and fewer hallucinated calls.

**Verification:** Count tool assignments before/after.  Aim for ≤ 5 tools per simple agent, ≤ 8 for the model developer.

### A3. Tune `max_iter`, `max_execution_time`, and `max_rpm`

**Files:** `src/thesiscrew/crew.py`, `src/thesiscrew/config/agents.yaml`  
**Action:**
Apply the following baseline limits:

| Agent | `max_iter` | `max_execution_time` (s) | `max_rpm` |
|---|---|---|---|
| `data_researcher` | 10 | 600 | 60 |
| `data_engineer` | 12 | 600 | 60 |
| `feature_engineer` | 15 | 900 | 60 |
| `model_developer` | 20 | 1,200 | 60 |
| `verification_analyst` | 12 | 600 | 60 |
| `integration_specialist` | 10 | 600 | 60 |
| `report_writer` | 12 | 600 | 60 |

Also set `Crew(..., max_rpm=100)` as a global ceiling.

**Why:** Bounded iteration prevents runaway loops; bounded wall-clock time prevents hung API calls.

**Verification:** Run the crew once and confirm no agent exceeds its time/iteration budget.

### A4. Add guardrails to critical validation tasks

**Files:** `src/thesiscrew/crew.py`  
**Action:**
Add a function-based `guardrail` (signature `Callable[[TaskOutput], Tuple[bool, Any]]`) to these tasks:
1. `station_discovery_task` — require the output to mention `pegelonline`, `eHYD`, and `Open-Meteo`.
2. `feature_engineering_task` — require a non-empty list of created feature files or a feature manifest.
3. `verification_task` — require RMSE/MAE/NSE metrics to be present and parseable.
4. `report_writing_task` — require sections for methodology, results, and conclusion.

Set `guardrail_max_retries=2` on each.

**Why:** Malformed outputs propagate silently otherwise.  Guardrails catch them at the task boundary.

**Verification:** Introduce an intentionally bad output (e.g., truncate a report) and confirm the guardrail retries and then fails loudly.

### A5. Persist outputs and logs automatically

**Files:** `src/thesiscrew/crew.py`  
**Action:**
- Add `output_file` and `create_directory=True` to every task that produces a durable artifact.
- Add `output_log_file="output/crew_run.json"` to `Crew`.
- Add an `after_kickoff_callbacks` that writes `output/run_summary.json` containing `result.raw[:500]`, `result.token_usage`, and `len(result.tasks_output)`.

**Why:** Persistent logs make debugging faster; you no longer need to re-run to inspect an output.

**Verification:** After a run, `output/` contains the expected files and `crew_run.json` is non-empty.

### A6. Enable crew memory

**Files:** `src/thesiscrew/crew.py`  
**Action:**
Add to `Crew`:

```python
memory=True,
embedder={"provider": "openai"},  # or your configured embedder
cache=True,
```

**Why:** Agents reuse prior context instead of recomputing or re-reading files; reduces redundant LLM calls and improves consistency.

**Verification:** Run twice with the same inputs; second run should complete faster and produce identical or better outputs.

### Phase A Checkpoint
- [ ] All agents have ≤ 8 tools and explicit timeouts.
- [ ] Guardrails cover discovery, features, verification, and report.
- [ ] `output/` is auto-created and all key artifacts are persisted.
- [ ] Crew memory and cache are enabled.
- [ ] At least one successful end-to-end run finishes in less time than the baseline.

---

## Phase B — Quality & Structure Improvements (Days 3–7)

*Objective:* Make outputs deterministic, reusable, and parallel where possible.

### B1. Introduce structured outputs for metric/manifest tasks

**Files:** `src/thesiscrew/crew.py`  
**Action:**
Define Pydantic models and attach them to tasks:

```python
from pydantic import BaseModel
from typing import Dict, List

class FeatureManifest(BaseModel):
    feature_files: List[str]
    target_column: str
    horizon_hours: List[int]

class BaselineMetrics(BaseModel):
    model: str
    horizons: Dict[str, Dict[str, float]]
```

Then:
- `feature_engineering_task` → `output_pydantic=FeatureManifest`
- `baseline_modeling_task` → `output_pydantic=BaselineMetrics`
- `verification_task` → `output_pydantic=BaselineMetrics` (or a dedicated `VerificationReport`)

**Why:** Free-text metrics are fragile.  Pydantic outputs give downstream tasks reliable, typed data and eliminate parsing regexes.

**Verification:** Inspect `task.output.pydantic` after a run; it should populate all fields.

### B2. Add per-agent reasoning for complex agents

**Files:** `src/thesiscrew/crew.py`  
**Action:**
Enable `reasoning=True` with a small attempt budget on:
- `feature_engineer`
- `model_developer`
- `verification_analyst`

Example:

```python
Agent(
    role="ML Forecasting Model Developer",
    reasoning=True,
    max_reasoning_attempts=2,
    max_iter=20,
    max_execution_time=1_200,
)
```

**Why:** These agents perform multi-step planning (feature design, model architecture, diagnostic analysis).  Reasoning reduces the number of execution iterations.

**Verification:** Compare iteration counts with and without `reasoning=True` over 3 runs; count should drop or quality should rise.

### B3. Use tiered LLMs

**Files:** `src/thesiscrew/crew.py`  
**Action:**
Assign models by cognitive load:

| Agent | Suggested model | Rationale |
|---|---|---|
| `data_researcher`, `data_engineer` | `openai/gpt-4o-mini` or equivalent | deterministic lookup |
| `feature_engineer` | `openai/gpt-4o` | planning + schema design |
| `model_developer` | `openai/gpt-4o` or `anthropic/claude-sonnet-4-6` | code generation |
| `verification_analyst` | `openai/gpt-4o` | analysis |
| `report_writer` | `openai/gpt-4o-mini` | summarization |

**Why:** Cheaper models for simple work cuts cost; stronger models only where they matter improves quality.

**Verification:** Run a cost comparison across the same inputs; cost should drop without degrading verification metrics.

### B4. Reduce report context to essential tasks only

**Files:** `src/thesiscrew/crew.py`  
**Action:**
Change `report_writing_task.context` to only the tasks whose outputs the report genuinely needs:

```python
context=[
    self.station_discovery_task(),
    self.data_ingestion_task(),
    self.feature_engineering_task(),
    self.baseline_modeling_task(),
    self.model_training_task(),
    self.verification_task(),
    self.final_documentation_task(),
]
```

Remove `api_integration_task` and `frontend_task` unless the report explicitly describes the API.

**Why:** Smaller context windows reduce token cost and prevent truncation of important earlier outputs.

**Verification:** Compare token usage of the report task before/after.

### B5. Parallelize independent Phase 4 tasks

**Files:** `src/thesiscrew/crew.py`, `src/thesiscrew/config/tasks.yaml`  
**Action:**
- Identify tasks that depend only on `model_training_task` and not on each other (e.g., `api_integration_task` and `frontend_task`).
- Set `async_execution=True` on those tasks.
- Add a lightweight synchronization task afterward that reads both outputs and produces a single deployment manifest.

**Why:** Independent work runs concurrently, reducing wall-clock runtime.

**Verification:** Add logging timestamps around these tasks; confirm overlap in execution.

### B6. Stabilize the tool-input monkey patch

**Files:** `src/thesiscrew/main.py`  
**Action:**
- Wrap the existing `ToolUsage._validate_tool_input` patch in `try/except` and emit a warning if CrewAI's internals change.
- OR create a safe wrapper tool (e.g., `SafeReadInputTool`) that normalizes list-of-dicts inputs before delegating to the real tool.
- Add a regression test that calls the patched/wrapped tool with a list input.

**Why:** The current patch is brittle.  A wrapper or guarded patch survives library upgrades.

**Verification:** Run the test suite; tool calls with list inputs succeed.

### Phase B Checkpoint
- [ ] Feature and verification tasks emit validated Pydantic outputs.
- [ ] Reasoning enabled on complex agents.
- [ ] Tiered LLMs assigned.
- [ ] Report context is minimal and sufficient.
- [ ] Independent tasks run asynchronously.
- [ ] Tool-input patch is guarded or replaced by wrappers.

---

## Phase C — Reliability & Resilience (Days 8–12)

*Objective:* Survive failures, resume work, and make the pipeline reproducible.

### C1. Add checkpointing

**Files:** `src/thesiscrew/crew.py`  
**Action:**
Add `CheckpointConfig` to the crew:

```python
from crewai.state.checkpoint_config import CheckpointConfig

Crew(
    ...,
    checkpoint=CheckpointConfig(
        location="output/.checkpoints",
        on_events=["task_completed", "crew_kickoff_failed"],
        max_checkpoints=10,
    ),
)
```

**Why:** Long-running model training/evaluation can fail mid-pipeline.  Checkpointing lets you resume instead of restarting.

**Verification:** Start a run, kill it after the third task, then resume with `Crew.from_checkpoint("output/.checkpoints/latest.json")` and confirm it continues.

### C2. Cache external data tools

**Files:** `src/thesiscrew/tools/*.py`  
**Action:**
- For `GetMeasurementsTool`, `HistoricalWeatherTool`, `ForecastWeatherTool`, etc., add an on-disk cache keyed by request parameters and date range.
- Store under `output/.cache/` with a TTL (e.g., 24 hours for forecasts, 7 days for historical measurements).
- Alternatively, rely on `Agent(..., cache=True)` and `Crew(..., cache=True)` first, then add disk caching if still redundant.

**Why:** Repeated API calls are the largest runtime cost.  Caching eliminates duplicate external requests.

**Verification:** Run the same inputs twice; second run should hit the cache and complete the data phase faster.

### C3. Harden tool error handling

**Files:** `src/thesiscrew/tools/*.py`  
**Action:**
- Add file-existence and column-existence checks at the start of every tool that reads CSVs.
- Raise `FileNotFoundError`/`ValueError` for fatal preconditions.
- Return structured `{"error": "..."}` JSON only for recoverable runtime errors.

**Why:** Clear failures give guardrails and callbacks useful signals; silent JSON errors are swallowed downstream.

**Verification:** Write unit tests that pass bad file paths and bad column names; confirm errors surface correctly.

### C4. Add task and step callbacks for observability

**Files:** `src/thesiscrew/crew.py`  
**Action:**
Add to `Crew`:

```python
step_callback=lambda agent, step: print(f"[{agent}] {step}"),
task_callback=lambda output: print(f"✅ {output.description} → {output.agent}"),
```

Also add a `callback` on the verification task that logs metrics to `output/metrics_log.jsonl`.

**Why:** You can see where time is spent and which tasks fail most often.

**Verification:** `output/crew_run.json` and `output/metrics_log.jsonl` contain timestamped events.

### Phase C Checkpoint
- [ ] Checkpointing survives a mid-run kill and resumes correctly.
- [ ] External data tools are cached.
- [ ] Tools raise clear errors for missing files/columns.
- [ ] Callbacks produce a trace of every step and task.

---

## Phase D — Production Architecture (Days 13+)

*Objective:* Separate training from inference so scheduled runs are fast and cheap.

### D1. Split the monolithic crew into two crews

**Files:** new `src/thesiscrew/training_crew.py` and `src/thesiscrew/inference_crew.py`  
**Action:**
Create:
1. **Training Crew** — data discovery → ingestion → feature engineering → baseline → model training → verification → artifact export.
2. **Inference Crew** — load trained artifacts → fetch latest data → feature engineering (same pipeline, no search) → predict → serve.

Both should share the same tool implementations but different agent/task graphs.

**Why:** You do not want to re-discover stations and retrain on every hourly forecast.  Inference becomes a fast, cheap crew.

**Verification:** Run training once, then run inference with only the latest timestamp; inference should complete in a small fraction of training time.

### D2. Serialize trained artifacts explicitly

**Files:** `src/thesiscrew/tools/modeling_tool.py`, `src/thesiscrew/inference_crew.py`  
**Action:**
- Persist model files, scalers, feature manifests, and station metadata under `output/artifacts/`.
- Write an `artifacts/manifest.json` that lists every file needed by the inference crew.

**Why:** Inference crew knows exactly what to load; no guesswork.

**Verification:** Delete everything except `output/artifacts/` and confirm the inference crew still works.

### D3. Add `crewai train` and `crewai test` workflows

**Files:** `pyproject.toml` or `README.md`, `src/thesiscrew/crew.py`  
**Action:**
- After several successful runs, collect human feedback and run:
  ```bash
  crewai train -n 5 -f output/trained_agents_data.pkl
  ```
- Load the trained data by setting `trained_agents_file="output/trained_agents_data.pkl"` or the `CREWAI_TRAINED_AGENTS_FILE` environment variable.
- Add a regression test command:
  ```bash
  crewai test -n 3 -m gpt-4o-mini
  ```
- Track task scores in CI.

**Why:** Trained guidance improves consistency; regression tests prevent silent quality degradation.

**Verification:** `trained_agents_data.pkl` exists and `crewai test` runs without errors.

### Phase D Checkpoint
- [ ] Training and inference crews are separate and both runnable.
- [ ] Artifacts are fully serialized and reloadable.
- [ ] `crewai train` produces a guidance file.
- [ ] `crewai test` passes and scores are recorded.

---

## Phase E — Optional Advanced Optimizations

*Objective:* Push quality and speed further once Phases A–D are stable.

### E1. Switch to hierarchical process with a manager agent
**Files:** `src/thesiscrew/crew.py`  
**Action:**
- Add a `manager` agent with `allow_delegation=True`.
- Change `Crew` to `process=Process.hierarchical`, `manager_agent=self.manager()`.
- Use `manager_llm="openai/gpt-4o"`.

**Why:** The manager can reroute tasks on failure and decide when to parallelize, improving robustness.

**Risk:** Adds overhead on short runs; only do this after A–D are stable.

### E2. Enable streaming for live monitoring
**Files:** `src/thesiscrew/main.py`  
**Action:**
Set `Crew(..., stream=True)` and consume the stream in `main()`.

**Why:** Useful for demos and debugging.

### E3. Add `inject_date` to forecast agents
**Files:** `src/thesiscrew/crew.py`  
**Action:**
Enable `inject_date=True` on `data_researcher` and `feature_engineer`.

**Why:** Agents know the current date without it being passed in the prompt, improving forecast freshness.

### E4. Add OpenTelemetry tracing
**Files:** `src/thesiscrew/crew.py`  
**Action:**
Set `tracing=True` on `Crew` in production.

**Why:** Distributed traces help diagnose slow tool calls and model retries in production.

---

## Implementation Order Summary

| Order | Phase | Key change | Why first |
|---|---|---|---|
| 1 | A1 | Input validation + directory seeding | Fail fast, fail cheap |
| 2 | A2 | Prune tools per agent | Faster decisions, lower cost |
| 3 | A3 | Timeouts and iteration limits | Stop runaway agents |
| 4 | A4 | Guardrails on critical tasks | Catch bad outputs early |
| 5 | A5 | Persist outputs and logs | Debugging speed |
| 6 | A6 | Crew memory + cache | Reuse context, avoid recomputation |
| 7 | B1 | Structured outputs | Deterministic downstream data |
| 8 | B2 | Agent reasoning | Reduce iterations for complex agents |
| 9 | B3 | Tiered LLMs | Right model for the right job |
| 10 | B4 | Minimal report context | Lower tokens, less truncation |
| 11 | B5 | Async independent tasks | Wall-clock speed |
| 12 | B6 | Stabilize tool-input patch | Robustness |
| 13 | C1 | Checkpointing | Resume from failure |
| 14 | C2 | External tool caching | Eliminate duplicate API calls |
| 15 | C3 | Tool error handling | Clear failure signals |
| 16 | C4 | Step/task callbacks | Observability |
| 17 | D1 | Split training/inference crews | Production efficiency |
| 18 | D2 | Artifact manifest | Reliable reload |
| 19 | D3 | Train/test workflows | Continuous quality |
| 20 | E1–E4 | Advanced options | Fine-tuning |

---

## Metrics to Track

Track these after every run to confirm improvement:

| Metric | How to measure | Target direction |
|---|---|---|
| Wall-clock runtime | `time` around `crew.kickoff()` | ↓ |
| Cost per run | Provider billing or `result.token_usage` | ↓ |
| Guardrail failure rate | `output/crew_run.json` failures / total tasks | ↓ |
| Retry rate | Count of `max_retry_limit` hits | ↓ |
| Tool calls per task | `output/crew_run.json` | ↓ |
| Verification RMSE / MAE / NSE | `verification_task` output | RMSE/MAE ↓, NSE ↑ |
| Forecast lead-time accuracy | Compare to official forecasts | ↑ |
| Run completion rate | Successful end-to-end runs / total runs | ↑ |

---

## Expected Results After Full Implementation

- **Runtime:** 30–60% reduction from tool pruning, caching, async execution, and reduced context.
- **Cost:** 40–70% reduction from tiered models, smaller contexts, and fewer retries.
- **Reliability:** Near-zero silent failures due to guardrails, structured outputs, and checkpointing.
- **Quality:** More consistent, reproducible metrics and reports because downstream tasks consume validated Pydantic objects instead of parsed Markdown.
- **Maintainability:** Training and inference separation, plus artifact manifests, make the project deployable as a real service.

---

## References

- [`docs/crewai-functions.md`](crewai-functions.md) — CrewAPI surface for Agent, Task, Crew, training, and testing.
- [`docs/optimizing.md`](optimizing.md) — Detailed optimization rationale and code examples.
- CrewAI docs: https://docs.crewai.com/
