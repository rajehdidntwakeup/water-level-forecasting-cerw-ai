# CrewAI Functions Reference

This document summarizes the public classes, constructors, methods, and attributes for **Agent**, **Task**, and **Crew** in [CrewAI](https://docs.crewai.com/). It is compiled from the official CrewAI documentation.

---

## Table of Contents

1. [Agent](#agent)
2. [Task](#task)
3. [Crew](#crew)
4. [Training and Testing](#training-and-testing)
5. [Output Objects](#output-objects)

---

## Agent

An `Agent` is an autonomous unit that performs tasks, makes decisions, uses tools, collaborates with other agents, maintains memory, and can delegate tasks when allowed.

### Class Signature

```python
from crewai import Agent

agent = Agent(
    role="...",
    goal="...",
    backstory="...",
    # optional parameters below
)
```

### Constructor Parameters

#### Required

| Parameter | Type | Description |
|-----------|------|-------------|
| `role` | `str` | Defines the agent's function and expertise within the crew. |
| `goal` | `str` | The individual objective guiding the agent's decision-making. |
| `backstory` | `str` | Provides context and personality. |

#### LLM & Tooling

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm` | `Union[str, LLM, Any]` | `OPENAI_MODEL_NAME` or `"gpt-4"` | Language model powering the agent. |
| `function_calling_llm` | `Optional[Any]` | `None` | Separate LLM used for tool calling. |
| `tools` | `List[BaseTool]` | `[]` | Capabilities/functions available to the agent. |
| `knowledge_sources` | `Optional[List[BaseKnowledgeSource]]` | `None` | Domain-specific knowledge bases. |
| `embedder` | `Optional[Dict[str, Any]]` | `None` | Configuration for the embedder. |
| `multimodal` | `bool` | `False` | Whether the agent supports multimodal capabilities. |

#### Execution Control

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_iter` | `int` | `20` | Maximum iterations before returning the best answer. |
| `max_rpm` | `Optional[int]` | `None` | Requests-per-minute limit. |
| `max_execution_time` | `Optional[int]` | `None` | Timeout in seconds. |
| `max_retry_limit` | `int` | `2` | Retries on error. |
| `verbose` | `bool` | `False` | Enable detailed execution logs. |
| `cache` | `bool` | `True` | Enable caching for tool usage. |
| `allow_delegation` | `bool` | `False` | Allow the agent to delegate tasks to other agents. |

#### Reasoning & Planning

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reasoning` | `bool` | `False` | Reflect and create a plan before executing. |
| `max_reasoning_attempts` | `Optional[int]` | `None` | Planning attempts; `None` means try until ready. |

#### Context & Memory

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `memory` | `bool` | `False` (inferred) | Maintain conversation history. |
| `respect_context_window` | `bool` | `True` | Summarize messages to keep them under the token limit. |

#### Date Injection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `inject_date` | `bool` | `False` | Auto-inject the current date into tasks. |
| `date_format` | `str` | `"%Y-%m-%d"` | Python `datetime` format string. |

#### Templates

| Parameter | Type | Description |
|-----------|------|-------------|
| `system_template` | `Optional[str]` | Custom system prompt template. |
| `prompt_template` | `Optional[str]` | Custom prompt template. |
| `response_template` | `Optional[str]` | Custom response template. |

> **Best practice:** When customizing templates, define both `system_template` and `prompt_template` together.

#### Callbacks & Model Compatibility

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `step_callback` | `Optional[Any]` | `None` | Function called after each agent step. |
| `use_system_prompt` | `Optional[bool]` | `True` | Set to `False` for older models like `o1`. |

#### Code Execution (Deprecated)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `allow_code_execution` | `Optional[bool]` | `False` | **Deprecated.** Use external sandboxes like E2B or Modal. |
| `code_execution_mode` | `Literal["safe", "unsafe"]` | `"safe"` | **Deprecated.** Use external sandboxes. |

### Methods

#### `kickoff(messages, response_format=None)`

Directly interact with an agent without a crew or task workflow.

| Parameter | Type | Description |
|-----------|------|-------------|
| `messages` | `Union[str, List[Dict[str, str]]]` | A string query or a list of `{role, content}` conversation messages. |
| `response_format` | `Optional[Type[Any]]` | Optional Pydantic model for structured output. |

**Returns:** A `LiteAgentOutput` with:

- `raw`: raw text output
- `pydantic`: parsed Pydantic model (if `response_format` was provided)
- `agent_role`: role of the agent
- `usage_metrics`: token usage

```python
result = researcher.kickoff("What are the latest developments in language models?")
print(result.raw)
```

#### `kickoff_async(messages, response_format=None)`

Asynchronous version of `kickoff()` with identical parameters and return type.

```python
result = await researcher.kickoff_async("What are the latest developments in AI?")
```

### JSONC Configuration

New projects can define agents in `agents/<agent_name>.jsonc` and reference them from `crew.jsonc`:

```jsonc
// agents/researcher.jsonc
{
  "role": "{topic} Senior Data Researcher",
  "goal": "Uncover cutting-edge developments in {topic}",
  "backstory": "You find the most relevant information...",
  "llm": "openai/gpt-4o",
  "tools": ["SerperDevTool"],
  "settings": {
    "verbose": true,
    "allow_delegation": false,
    "max_iter": 20
  }
}
```

Behavior options may be top-level or under `settings`; `settings` takes precedence.

---

## Task

A `Task` represents a unit of work assigned to an agent. Tasks define what needs to be done, the expected output, and how the result should be validated or stored.

### Class Signature

```python
from crewai import Task

task = Task(
    description="...",
    expected_output="...",
    # optional parameters below
)
```

### Constructor Parameters

#### Required

| Parameter | Type | Description |
|-----------|------|-------------|
| `description` | `str` | A clear, concise statement of what the task entails. |
| `expected_output` | `str` | A detailed description of what task completion looks like. |

#### Optional

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `Optional[str]` | `None` | Identifier for the task. |
| `agent` | `Optional[BaseAgent]` | `None` | Agent responsible for execution. |
| `tools` | `List[BaseTool]` | `[]` | Tools the agent is limited to for this task. Overrides the agent's default tool set. |
| `context` | `Optional[List["Task"]]` | `None` | Prior tasks whose outputs feed into this one. |
| `async_execution` | `Optional[bool]` | `False` | Run the task asynchronously. |
| `human_input` | `Optional[bool]` | `False` | Require human review of the final answer. |
| `markdown` | `Optional[bool]` | `False` | Instruct the agent to return Markdown-formatted output. |
| `config` | `Optional[Dict[str, Any]]` | `None` | Task-specific configuration. |
| `output_file` | `Optional[str]` | `None` | File path to store the output. |
| `create_directory` | `Optional[bool]` | `True` | Create directories for `output_file` if missing. |
| `output_json` | `Optional[Type[BaseModel]]` | `None` | Pydantic model for JSON output structure. |
| `output_pydantic` | `Optional[Type[BaseModel]]` | `None` | Pydantic model for direct output validation. |
| `callback` | `Optional[Any]` | `None` | Function/object executed after completion. |
| `guardrail` | `Optional[Callable]` | `None` | Single validation function for task output. |
| `guardrails` | `Optional[List[Callable]]` | `None` | List of validation functions. |
| `guardrail_max_retries` | `Optional[int]` | `3` | Retries on guardrail failure. |

> **Deprecated:** `max_retries` is deprecated; use `guardrail_max_retries` instead.

> **Validation:** Only one output format should be set per task (`output_json`, `output_pydantic`, etc.). Manual assignment of `id` is prevented.

### JSONC Task Definition

Tasks defined in `crew.jsonc` also support `input_files`, `response_model`, `converter_cls`, and conditional tasks via `"type": "ConditionalTask"` plus a `condition` field.

### TaskOutput

When a task finishes, its result is wrapped in `TaskOutput`.

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `description` | `str` | Task description. |
| `summary` | `Optional[str]` | Auto-generated summary from the first 10 words of the description. |
| `raw` | `str` | Default raw text output. |
| `pydantic` | `Optional[BaseModel]` | Structured Pydantic output. |
| `json_dict` | `Optional[Dict[str, Any]]` | Parsed JSON output. |
| `agent` | `str` | Name of the executing agent. |
| `output_format` | `OutputFormat` | `RAW`, `JSON`, or `Pydantic`. |
| `messages` | `list[LLMMessage]` | Last execution messages. |

#### Properties/Methods

| Property/Method | Description |
|-------------------|-------------|
| `json` | JSON string when `output_format` is `JSON`. |
| `to_dict()` | Converts Pydantic/JSON outputs to a dictionary. |
| `__str__` | Returns string, preferring Pydantic, then JSON, then raw. |

Access a task's output after crew execution:

```python
task_output = task.output
print(task_output.raw)
```

### Execution Flow

Tasks run inside a `Crew` using either a sequential or hierarchical process:

```python
from crewai import Crew, Process

crew = Crew(
    agents=[agent1, agent2],
    tasks=[task1, task2],
    process=Process.sequential  # or Process.hierarchical
)
```

### Guardrails

Guardrails validate or transform output before the next task proceeds.

#### Function-Based Guardrail

Signature: `Callable[[TaskOutput], Tuple[bool, Any]]`

```python
from typing import Tuple, Any
from crewai import TaskOutput

def validate(result: TaskOutput) -> Tuple[bool, Any]:
    if len(result.raw.split()) > 200:
        return (False, "Too long")
    return (True, result.raw.strip())
```

#### LLM-Based Guardrail

Pass a string to `guardrail` or `guardrails`; the agent's LLM validates output against your description. Requires an assigned `agent`.

#### Multiple Guardrails

Use `guardrails` as a sequential list. If `guardrails` is provided, `guardrail` is ignored.

```python
Task(
    description="Write a blog post about AI",
    expected_output="100–500 words, clean format",
    agent=blog_agent,
    guardrails=[validate_word_count, "engaging and suitable for a general audience"],
    guardrail_max_retries=3
)
```

### Callbacks

A `callback` runs after task completion:

```python
def callback_function(output: TaskOutput):
    print(f"Task completed: {output.description}\n{output.raw}")

task = Task(..., callback=callback_function)
```

### Context & Dependencies

Use `context` to feed one or more prior task outputs into another:

```python
analysis_task = Task(
    description="Analyze the research findings",
    expected_output="Analysis report",
    agent=analyst,
    context=[research_task]
)
```

### Asynchronous Execution

Set `async_execution=True` so the crew continues without waiting; downstream tasks use `context` to wait for the async result.

### Markdown Output

Set `markdown=True` to automatically append Markdown formatting instructions to the prompt.

### Structured Outputs

- `output_pydantic=MyModel` — validates output as a Pydantic model.
- `output_json=MyModel` — validates output as JSON matching the model.

Only one output type should be set per task.

### Tool Override

Tools defined on a task override the agent's default tool set.

### Directory Creation

`create_directory=True` automatically creates missing directories for `output_file`. Set to `False` to require a pre-existing directory; a missing directory raises `RuntimeError`.

---

## Crew

A `Crew` is a collaborative group of agents working together to achieve a set of tasks.

### Class Signature

```python
from crewai import Crew, Process

crew = Crew(
    agents=[...],
    tasks=[...],
    process=Process.sequential,
    # optional parameters below
)
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tasks` | `List[Task]` | required | Tasks assigned to the crew. |
| `agents` | `List[Agent]` | required | Agents participating in the crew. |
| `process` | `Process` | `Process.sequential` | Execution flow: `sequential` or `hierarchical`. |
| `verbose` | `bool` | `False` | Logging verbosity level. |
| `manager_llm` | `Optional[Any]` | `None` | LLM for the manager agent; mandatory in hierarchical mode. |
| `function_calling_llm` | `Optional[Any]` | `None` | Crew-wide LLM for tool function calls; each agent may override it. |
| `config` | `Optional[Dict[str, Any]]` | `None` | JSON or dict configuration settings. |
| `max_rpm` | `Optional[int]` | `None` | Requests-per-minute ceiling; overrides individual agent limits. |
| `memory` | `Optional[bool]` | `None` | Activates short-term, long-term, and entity memory. |
| `cache` | `bool` | `True` | Caches tool execution results. |
| `embedder` | `Optional[Dict[str, Any]]` | `{"provider": "openai"}` | Embedder settings, mainly used by memory. |
| `step_callback` | `Optional[Any]` | `None` | Function invoked after every agent step. |
| `task_callback` | `Optional[Any]` | `None` | Function invoked after each task finishes. |
| `share_crew` | `Optional[bool]` | `None` | Share crew data with CrewAI for library improvement. |
| `output_log_file` | `Union[bool, str]` | `None` | `True` saves `logs.txt`; a string/path saves to that file. |
| `manager_agent` | `Optional[Agent]` | `None` | Custom manager agent for hierarchical crews. |
| `prompt_file` | `Optional[str]` | `None` | Path to a prompt JSON file. |
| `planning` | `Optional[bool]` | `None` | Enables an `AgentPlanner` to plan tasks before iterations. |
| `planning_llm` | `Optional[Any]` | `None` | LLM used by the planner. |
| `knowledge_sources` | `Optional[List[BaseKnowledgeSource]]` | `None` | Crew-level knowledge sources shared across agents. |
| `stream` | `bool` | `False` | Enable live execution streaming. |
| `chat_llm` | `Optional[Any]` | `None` | LLM for the `crewai chat` CLI. |
| `before_kickoff_callbacks` | `List[Callable]` | `[]` | Callables run before the crew starts; can mutate inputs. |
| `after_kickoff_callbacks` | `List[Callable]` | `[]` | Callables run after the crew finishes; can mutate output. |
| `tracing` | `Optional[bool]` | `None` | OpenTelemetry tracing; `None` inherits environment settings. |
| `skills` | `Optional[Union[List[str], List[Skill]]]` | `None` | Skill search directories or preloaded `Skill` objects. |
| `security_config` | `SecurityConfig` | `SecurityConfig()` | Fingerprinting and identity management. |
| `checkpoint` | `Union[bool, None, CheckpointConfig]` | `None` | Enables state saving; see Checkpointing below. |

### Execution Methods

| Method | Description |
|--------|-------------|
| `kickoff(inputs={})` | Synchronous execution of the crew workflow. |
| `kickoff_for_each(inputs=List[dict])` | Runs the crew once per input dict, synchronously. |
| `akickoff(inputs={})` | Native `async` execution. |
| `akickoff_for_each(inputs=List[dict])` | Native `async` per input. |
| `kickoff_async(inputs={})` | Thread-based async wrapper around synchronous logic. |
| `kickoff_for_each_async(inputs=List[dict])` | Thread-based async per input. |

> For high-concurrency scenarios, the native `akickoff` variants are recommended over the thread-based `*_async` wrappers.

```python
result = crew.kickoff(inputs={"topic": "AI agents"})
print(result.raw)
```

### Checkpointing Methods

| Method | Description |
|--------|-------------|
| `Crew.from_checkpoint(path)` | Class method that restores a saved crew state. Call `kickoff()` afterward to continue. |

```python
crew = Crew.from_checkpoint(".checkpoints/latest.json")
crew.kickoff()
```

### Other Attributes

| Attribute | Description |
|-----------|-------------|
| `output` | `CrewOutput` instance after execution. |
| `usage_metrics` | Aggregated LLM usage metrics across tasks. |

### Usage Examples

#### Sequential Process

```python
from crewai import Agent, Crew, Task, Process

crew = Crew(
    agents=[Agent(role="Analyst", goal="Analyze data", backstory="...")],
    tasks=[Task(description="Collect data", expected_output="Report")],
    process=Process.sequential,
    verbose=True,
    memory=True,
)
result = crew.kickoff()
```

#### Hierarchical Process

```python
from crewai import Crew, Process

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.hierarchical,
    manager_llm="openai/gpt-4o",
)
crew.kickoff()
```

#### Streaming Execution

```python
crew = Crew(agents=[researcher], tasks=[task], stream=True)
stream = crew.kickoff(inputs={"topic": "AI"})

for chunk in stream:
    print(chunk.content, end="", flush=True)

result = stream.result
```

#### JSONC Project Configuration

```jsonc
// crew.jsonc
{
  "name": "Market Research Crew",
  "agents": ["researcher", "analyst"],
  "tasks": [
    {
      "name": "research",
      "description": "Research {topic}.",
      "expected_output": "Research notes.",
      "agent": "researcher"
    },
    {
      "name": "analysis",
      "description": "Analyze findings and write a report.",
      "expected_output": "Markdown report.",
      "agent": "analyst",
      "context": ["research"],
      "output_file": "output/report.md"
    }
  ],
  "process": "sequential",
  "verbose": true,
  "memory": true,
  "inputs": { "topic": "AI Agents" }
}
```

---

## Training and Testing

### `train()`

CrewAI provides a `train()` method for interactive training via CLI or Python.

**CLI usage:**

```bash
crewai train -n <n_iterations> -f <filename.pkl>
```

If `-f` is omitted, the output defaults to `trained_agents_data.pkl` in the current working directory.

**Programmatic usage:**

```python
YourCrewName_Crew().crew().train(
    n_iterations=n_iterations,
    inputs=inputs,
    filename=filename
)
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `n_iterations` | `int` | Positive integer number of training loops. |
| `inputs` | `dict` | Dictionary of inputs for the training process, e.g. `{"topic": "CrewAI Training"}`. |
| `filename` | `str` | Output `.pkl` file; must end with `.pkl`. |

#### Behavior

- Training sets `human_input = True` and disables delegation.
- Initializes `training_data.pkl` plus the trained output file.
- On each iteration records `initial_output`, `human_feedback`, and `improved_output` per agent, keyed by agent ID and iteration.
- Prior feedback is appended to prompts within the session.
- After training, CrewAI evaluates data per agent and produces consolidated results containing `suggestions`, `quality`, and `final_summary`, saved by `agent_role`.

#### Training Files

| File | Purpose |
|------|---------|
| `training_data.pkl` | Ephemeral data: `agent_id -> { iteration: { initial_output, human_feedback, improved_output } }`. |
| `trained_agents_data.pkl` | Persisted guidance: `agent_role -> { suggestions, quality, final_summary }`. |

#### Loading Trained Data

During normal runs, agents auto-load suggestions from `trained_agents_data.pkl`. For custom files:

```python
Crew(trained_agents_file="my_custom_trained.pkl")
```

Or set the environment variable `CREWAI_TRAINED_AGENTS_FILE`, or run:

```bash
crewai run -f my_custom_trained.pkl
```

> **Recommendation:** CrewAI recommends models with at least 7B parameters for reliable JSON output accuracy, evaluation quality, and instruction following.

### `test()`

CrewAI exposes testing through the `crewai test` CLI command.

```bash
crewai test
crewai test --n_iterations 5 --model gpt-4o
crewai test -n 5 -m gpt-4o
```

#### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_iterations` | `2` | Number of iterations to run. |
| `model` | `gpt-4o-mini` | Evaluation model; only OpenAI provider is available. |

#### Behavior

- Runs the crew for the specified number of iterations.
- Displays detailed performance metrics at the end, including per-task scores per run, average totals, associated agents, an overall crew score, and execution time in seconds.

---

## Output Objects

### `CrewOutput`

After `crew.kickoff()` finishes, the result is a `CrewOutput` object.

| Attribute/Method | Type | Description |
|------------------|------|-------------|
| `raw` | `str` | String output from the final task. |
| `pydantic` | `Optional[BaseModel]` | Structured Pydantic output. |
| `json_dict` | `Optional[Dict[str, Any]]` | JSON dictionary. |
| `tasks_output` | `List[TaskOutput]` | List of all task outputs. |
| `token_usage` | summary object | Token usage summary. |
| `json` | `str` | JSON string representation. |
| `to_dict()` | `dict` | Converts JSON/Pydantic outputs to a dictionary. |
| `__str__` | `str` | String representation, preferring Pydantic, then JSON, then raw. |

### `TaskOutput`

See the [TaskOutput](#taskoutput) section above under Task.

### `LiteAgentOutput`

Returned by `Agent.kickoff()` and `Agent.kickoff_async()`.

| Attribute | Description |
|-----------|-------------|
| `raw` | Raw text output. |
| `pydantic` | Parsed Pydantic model if `response_format` was provided. |
| `agent_role` | Role of the agent. |
| `usage_metrics` | Token usage. |

---

## Sources

- [CrewAI Documentation](https://docs.crewai.com/)
- [Agent Class Reference](https://docs.crewai.com/en/concepts/agents)
- [Task Class Reference](https://docs.crewai.com/en/concepts/tasks)
- [Crew Class Reference](https://docs.crewai.com/v1.15.1/en/concepts/crews.md)
- [Training](https://docs.crewai.com/v1.15.1/en/concepts/training.md)
- [Testing](https://docs.crewai.com/v1.15.1/en/concepts/testing.md)
- [AMP REST API — Kickoff](https://docs.crewai.com/v1.15.1/en/api-reference/kickoff.md)
