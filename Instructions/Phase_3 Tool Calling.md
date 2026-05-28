
# Phase 3 – Tool Calling (Read Thesis Area)

In this phase you will learn how to **create custom tools in CrewAI**.

Tools in CrewAI are essentially **function calls that an LLM can use** to access external functionality or data.

To demonstrate this concept, the **research domains for the Bachelor thesis** will be stored in a **JSON configuration file**.  
The tool we create will allow an agent to **read a local JSON file and return the defined research areas**.

---

## Step 1 – Create the Input File

Create the file:

```
/input/research_areas.json
```

Insert the following content into the file:

```json
{
  "section": [],
  "areas": [
    "Agentic AI",
    "Home Automation",
    "Energy"
  ]
}
```

This file defines the **research domains** in which the thesis topics should be generated.

---

## Step 2 – Create the Tool

Create the file:

```
src/thesisCrew/tools/read_research_areas_tool.py
```

Write a tool that **reads a JSON file from a given path** and returns the research areas.

```python
from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import json


class ReadResearchAreasInput(BaseModel):
    """Input schema for ReadResearchAreasTool."""
    path: str = Field(..., description="Path to research areas file.")


class ReadResearchAreasTool(BaseTool):
    name: str = "Read Research Areas"
    description: str = (
        "Reads areas of research that the thesis topic should focus on"
    )
    args_schema: Type[BaseModel] = ReadResearchAreasInput


    def _run(self, path: str) -> str:

        with open(path) as f:
            data = json.load(f)

        return data["areas"]
```

---

## Step 3 – Register the Tool in the Crew

Open:

```
src/thesisCrew/crew.py
```

Add the import for the tool:

```python
from thesisCrew.tools.read_research_areas_tool import ReadResearchAreasTool
```

---

## Step 4 – Provide the Tool to the Research Agent

Add the tool to the **Research Assistant agent**.

```python
tools=[ReadResearchAreasTool()]
```

This allows the agent to **read the research areas from the JSON file**.

---

## Step 5 – Provide the Input Path

Open:

```
main.py
```

Define the input path when running the crew:

```python
inputs = {
    "path": "./input/research_areas.json"
}
```

---

## Step 6 – Use the Path in the Task Description

Open:

```
src/thesisCrew/config/tasks.yaml
```

Modify the task description so the agent knows where to find the research areas.

Example:

```yaml
research_task:
  description: >
    Conduct a thorough internet search to find promising bachelor thesis topics
    that combine the research areas defined in {path}.
  expected_output: >
    A list of suitable bachelor thesis topics.
  agent: research_assistant
```

---

## Result of Phase 3

After completing this phase you will have:

- A **custom CrewAI tool**
- A tool that **reads structured data from a JSON file**
- An agent that can **use the tool during reasoning**
- A system where **research areas influence the generated thesis topics**

---

## What You Learned

In this phase you learned:

- how **CrewAI tools work**
- how to **create custom tools**
- how agents can **access external data**
- how to **connect tools with agents and tasks**


