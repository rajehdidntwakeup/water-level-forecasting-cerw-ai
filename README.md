# Thesiscrew Crew

## Agentic AI Code Analysis System

A multi-agent crew built with CrewAI for automated Java code analysis, security scanning, and technical report generation.

## Overview

This project implements three specialized AI agents that work together to analyze Java directories, identify security vulnerabilities, and produce consolidated technical reports:

| Agent | Role | Key Tools |
|-------|------|-----------|
| `code_reviewer` | Senior Java Code Reviewer | `JavaDirectoryReviewTool`, `ASTChunkerTool`, `ReadInputTool` |
| `proposal_writer` | Code Security Agent | `JavaDirectoryVulnerabilityScannerTool`, `VulnerabilityScannerTool` |
| `summary_writer` | Technical Report Specialist | `SummaryBuilderTool` |

## Project Structure

```
src/thesiscrew/
├── crew.py                 # Crew orchestration and configuration
├── main.py                 # Entry point for running the crew
├── config/
│   ├── agents.yaml         # Agent roles, goals, and backstories
│   └── tasks.yaml          # Task definitions, dependencies, and outputs
├── tools/
│   ├── java_directory_review_tool.py    # Reads and reviews Java files in a directory
│   ├── java_directory_vulnerability_scanner_tool.py # Scans Java directories for security flaws
│   ├── ast_chunker_tool.py              # Parses code into AST-based chunks
│   ├── vulnerability_scanner_tool.py    # Core security scanning logic
│   ├── summary_builder_tool.py          # Consolidates reports into output.md
│   └── read_input_tool.py               # Reads project input context
└── knowledge/
    └── user_preference.txt   # User context for agents
```

## Installation

This project uses `uv` for dependency management.

1. Clone the repository
2. Install dependencies:
   ```powershell
   uv sync
   ```
3. Configure environment variables in `.env`:
   ```
   BASE_URL=http://localhost:11434
   MODEL=ollama/qwen3.5:cloud
   OLLAMA_API_KEY=your-api-key
   ```

## Usage

Run the crew:
```powershell
uv run run_crew
```
Or directly via the main script:
```powershell
uv run python src/thesiscrew/main.py
```

## Tools

### JavaDirectoryReviewTool
Navigates a specified directory to find all Java source files and performs a detailed review for bugs, anti-patterns, and best practices.

### JavaDirectoryVulnerabilityScannerTool
Specifically designed to scan Java projects for security vulnerabilities, identifying risks and suggesting remediation steps.

### ASTChunkerTool
Parses files and extracts structured code chunks (functions, classes, methods) with metadata for granular analysis.

### VulnerabilityScannerTool
Core security engine that scans for:
- Hardcoded secrets and credentials
- Dangerous function calls
- Weak cryptography
- SQL injection patterns

### SummaryBuilderTool
Consolidates all findings from the `code_reviewer` and `proposal_writer` agents. It automatically combines both a high-level **Executive Summary** and a **Detailed Analysis** into a single document.

## Output

The final consolidated report is written to:
`C:\Users\Rajehdidntwakeup\IdeaProjects\crew-ai\output\output.md`

This file includes:
1. **Executive Summary**: High-level findings and overall assessment.
2. **Detailed Analysis**: File-by-file breakdown of all identified issues and security vulnerabilities.

## Workshop Instructions

- [Phase 1: Intro](./Instructions/Phase_1%20Intro.md)
- [Phase 2: Build First Crew](./Instructions/Phase_2%20Build%20First%20Crew.md)
- [Phase 3: Tool Calling](./Instructions/Phase_3%20Tool%20Calling.md)
- [Phase 4: Web Search Integration](./Instructions/Phase_4%20CrewAI%20Web%20Serach%20Integration.md)
- [Phase 5: MCP Integration](./Instructions/Phase_5%20MCP%20Integration.md)
