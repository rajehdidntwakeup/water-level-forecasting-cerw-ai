
# Phase 5 – Add MCP Integration (Scientific Paper Search)

In this phase you will integrate an **MCP (Model Context Protocol) server** into your CrewAI system.

The MCP server allows your agents to **access external tools through a standardized interface**.  
In this example, we integrate an MCP server that enables searching **scientific publications on arXiv**.

This allows your agents to find **relevant academic papers related to the bachelor thesis topic**.

---

# Step 1 – Preparation

First install the required MCP tools.

```bash
uv add crewai-tools[mcp]
```

Next install the **arXiv MCP server**, which is implemented in Python.

```bash
uv tool install arxiv-mcp-server
```

More information about the server:

https://pypi.org/project/arxiv-mcp-server/


---
## Step 2 - Agent and Task

Add a librarian agent:
```
librarian:
  role: >
    librarian
  goal: >
    Find the most fitting research paper for the thesis topic. 
  backstory: >
    You are a experienced librarian that is well known to find current research paper that are relevant for a specific topic.
```

Add a task: 
```
find_paper_task:
  description: >
    Based on the produced proposal for thesis, form a good search string to find the top 10 relevant research papers on arxiv that help the student to get 
    into the topic. If available, provide review papers that summarize the current state of the art in that specific field.  
  expected_output: >
    A list of the top 10 relevant paper with titel and DOI number. Use markdown fromat for the output  without '```'
  agent: librarian
```
## Step 3 – Create a Local MCP Integration

Create a new folder for MCP integrations next to the `tools` folder.

```
src/thesisCrew/mcp
```

---

### Create the MCP Server File

Create the file:

```
src/thesisCrew/mcp/mcp_server.py
```

Insert the following code:

```python
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters


def get_mcp_tools():

    server_params = [
        # arXiv MCP Server
        StdioServerParameters(
            command="uv",
            args=[
                "tool",
                "run",
                "arxiv-mcp-server",
                "--storage-path",
                "./papers"
            ]
        )
        # Example for adding a second MCP server
        # StdioServerParameters(
        #     command="python",
        #     args=["servers/filesystem_server.py"]
        # )
    ]

    adapter = MCPServerAdapter(server_params)

    return adapter.tools
```

This configuration starts the **arXiv MCP server** and exposes its tools to CrewAI.

Downloaded papers will be stored in:

```
./papers
```

---

## Step 4 – Import the MCP Tools

Open:

```
src/thesisCrew/crew.py
```

Add the required imports:

```python
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters
from thesisCrew.mcp.mcp_server import get_mcp_tools
```

---

## Step 5 – Load the MCP Tools

In `crew.py`, load the tools provided by the MCP server.

```python
mcp_tools = get_mcp_tools()
```

This retrieves all tools exposed by the MCP server.

---

## Step 6 – Create a Librarian Agent

Create a new agent that specializes in **searching scientific literature**.

Example:

```python
@agent
def librarian(self) -> Agent:
    return Agent(
        config=self.agents_config['librarian'],
        verbose=True,
        tools=mcp_tools
    )
```

Add task:
```python
    @task
    def find_paper_task(self) -> Task:
        return Task(
            config=self.tasks_config['find_paper_task'], # type: ignore[index]
            output_file='output/paper.md'
        )
```
The **Librarian Agent** can now:

- search for papers on **arXiv**
- retrieve academic publications
- support the thesis topic generation process

---

## Result of Phase 5

After completing this phase you will have:

- integrated an **MCP server**
- enabled your system to **search scientific publications**
- created a **Librarian agent** specialized in literature search
- extended your multi-agent system with **external research capabilities**

---

## What You Learned

In this phase you learned:

- how **MCP servers work**
- how to integrate **external MCP tools**
- how agents can **access scientific knowledge sources**
- how to extend a CrewAI system with **research capabilities**

