
# Phase 2 – Build Your First Crew

In this phase you will create your first **CrewAI multi-agent system**.

CrewAI is a framework for building collaborative AI agent systems in which multiple agents with different roles cooperate to solve complex tasks.

---

## Step 1 – Setup CrewAI in Visual Studio Code

### 1. Create a Project Folder

Create a **new folder** for your project and open it in **Visual Studio Code**.

---

### 2. Create a CrewAI Project

Open a terminal in VS Code and run:

```bash
crewai create crew thesisCrew
```

This command creates a complete project structure with the required configuration files and directories for agents, tasks, and tools.

The generated structure will look similar to this:

```
thesisCrew/
├── .env
├── pyproject.toml
├── README.md
└── src/
    └── thesisCrew/
        ├── main.py
        ├── crew.py
        ├── tools/
        └── config/
            ├── agents.yaml
            └── tasks.yaml
```

### Importatn Project Files

| File | Purpose |
|-----|------|
| `main.py` | Entry point of the application |
| `crew.py` | Defines the crew and connects agents and tasks |
| `config/agents.yaml` | Defines the agents |
| `config/tasks.yaml` | Defines the tasks |
| `.env` | Stores API keys |

---

### 3. Configure an LLM

During the project creation process, the CLI may ask you to select:

- an **LLM provider**
- a **model**

You may either select one or skip this step.

---

### 4. Add an API Key

Edit the file:

```
.env
```

Add your API key:

```env
OPENAI_API_KEY=sk-....
```


Note: We provide a lokal LLM hostet at our own sever. This LLM has very limited capabilits and computational power. However, if you have no other option, you can use also this. 
Just deefine a local LLM in your **crew.py** file:  

```
local_llm = LLM(
        model="openai/qwen-toolcalling",
        base_url="https://daystrom.ditm.at:4000/v1",
        api_key="sk-31ok2l0-iNcJ2ccNjJnBAw"
    )
```

---

## Step 2 – Create the Agents

Agents can either be defined directly in **Python code** or in a **YAML configuration file**.

Using **YAML** helps create cleaner and more maintainable code, which is the recommended approach in CrewAI.

---

### Edit the Agent Configuration

Open the file:

```
src/thesisCrew/config/agents.yaml
```

Add the following agents:

- **Research Assistant**  
  Searches for potential Bachelor thesis topics.

- **User Interactor**  
  Interacts with the user to select a topic.

- **Proposal Writer**  
  Writes a short thesis proposal for the selected topic.

---

### Agent Structure

Use the following structure:

```yaml
agent_name:
  role: >
    # your role
  goal: >
    # your goal description
  backstory: >
    # your backstory
```

---

### Example

```yaml
research_assistant:
  role: >
    Bachelor Thesis Research Assistant
  goal: >
    Find suitable research topics for a Bachelor thesis
  backstory: >
    You are an experienced research assistant who helps students
    identify interesting and feasible thesis topics.
```

---

### Add the Agents in `crew.py`

Open the file:

```
src/thesisCrew/crew.py
```

Add the agents to the crew.

⚠ **Important**

The annotated function name must match the agent name defined in the YAML file.

Example:

```python
@agent
def research_assistant(self) -> Agent:
```

CrewAI maps YAML definitions to Python functions using **matching names**.

---

## Step 3 – Define the Tasks

Now you will define the tasks that the agents should perform.

Open the file:

```
src/thesisCrew/config/tasks.yaml
```

Add the following tasks:

- **research_task**  
  Searches for possible Bachelor thesis topics.

- **ui_task**  
  Interacts with the user to select a topic.

- **reporting_task**  
  Generates a short thesis proposal.

---

### Task Structure

Use the following structure:

```yaml
task_name:
  description: >
    # your description
  expected_output: >
    # your expected output
  agent: agent_name
```

---

### Example

```yaml
research_task:
  description: >
    Search for possible Bachelor thesis topics in a given research domain.
  expected_output: >
    A list of suitable Bachelor thesis topics.
  agent: research_assistant
```

---

## Result of Phase 2

At the end of this phase you should have:

- A working **CrewAI project**
- **Three agents**
- **Three tasks**
- A functioning **multi-agent workflow**

Your agents will collaborate to:

1. "Search" for thesis topics (currently without access to the internet)  
2. Interact with the user  
3. Generate a thesis proposal  

---

