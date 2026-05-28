
# Phase 4 – Tool Integration (Web Search)

In this phase you will integrate an existing **web search tool** provided by CrewAI.

CrewAI offers a large collection of tools that allow agents to interact with external services such as:

- web search
- APIs
- file systems
- databases

An overview of available tools can be found here:

https://docs.crewai.com/en/tools/overview

This phase integrates the **Serper web search tool**, allowing the research agent to perform real internet searches.

---

## Step 0 – Prerequisites

First install the CrewAI tools package.

```bash
pip install "crewai[tools]"
```

Then import the web search tool in your project.

```python
from crewai_tools import SerperDevTool
```

---

## Step 1 – Create a Serper API Key

The web search tool requires an API key.

1. Register at:

https://serper.dev

2. Create an API key.

3. Add the key to your `.env` file:

```env
SERPER_API_KEY=your_api_key_here
```

---

## Step 2 – Update the Research Task

Open:

```
src/thesisCrew/config/tasks.yaml
```

Modify the **research task description** so the agent performs a web search.

Example:

```yaml
research_task:
  description: >
    Conduct a thorough internet search to find promising bachelor thesis topics
    that combine the research areas defined in {path}.

    Make sure you find interesting and relevant information.
    The current year is {current_year}.
  expected_output: >
    A list of suitable bachelor thesis topics.
  agent: research_assistant
```

This allows the agent to **use web search results** while generating thesis topics.

---

## Step 3 – Add the Web Search Tool to the Agent

Open:

```
src/thesisCrew/crew.py
```

Add the tool to the **Research Assistant agent**.

```python
tools=[ReadResearchAreasTool(), SerperDevTool()]
```

The research agent can now:

- read research domains from the JSON file
- perform web searches to discover relevant thesis ideas

---

## Step 4 – Save the Output to a File

Next we modify the **reporting task** so the generated proposal is written to a local file.

Open:

```
src/thesisCrew/crew.py
```

Modify the reporting task:

```python
@task
def reporting_task(self) -> Task:
    return Task(
        config=self.tasks_config['reporting_task'],
        output_file='output/report.md'
    )
```

This will save the generated thesis proposal to:

```
output/report.md
```

---

## Result of Phase 4

After completing this phase you will have:

- an agent that can **perform real web searches**
- integration of the **Serper web search API**
- an agent that combines:
  - local data (JSON research areas)
  - online knowledge (web search)
- automatic generation of a **report file**

```
output/report.md
```

---

# What You Learned

In this phase you learned:

- how to **integrate external tools in CrewAI**
- how agents can **search the internet**
- how to **combine multiple tools in one agent**
- how to **store agent results in files**


