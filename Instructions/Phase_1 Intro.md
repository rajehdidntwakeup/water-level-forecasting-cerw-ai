# Intro to CrewAI Framework

~ 30 minutes.

More infos: https://crewai.com/



# Assignment: AI Agent System for Bachelor Thesis Topic Discovery

## Objective

The goal of this assignment is to develop an **AI agent system using CrewAI** that assists students in identifying a suitable **Bachelor’s thesis topic** and generating an initial research proposal.

CrewAI is a framework for building collaborative AI agent systems in which specialized agents work together to solve complex tasks. 


---

# Tasks

## 1. Topic Generation

The system should generate **several possible Bachelor’s thesis topics** within a given research area.

The proposed topics should be relevant and suitable for a **Bachelor-level research project**.



## 2. User Interaction

The user selects one of the proposed topics.

The selected topic will then be used as the input for the following steps.



## 3. Thesis Proposal Generation

After a topic has been selected, the system should generate a short **thesis proposal** consisting of the following elements:

- Motivation  
- Problem Statement  
- Research Question  
- Proposed Solution  

The proposal should provide a clear overview of the research idea.


## 4. Literature Search

Finally, the system should search **arXiv** for relevant academic publications related to the selected topic.

The system should return **up to 10 relevant research papers**, preferably:

- survey papers  
- review papers  

These papers should help the student gain an initial overview of the research area.

For each paper, the system should provide:

- Title  
- Authors  
- Abstract  
- Link to the paper

---

# Technical Requirements

The implementation should demonstrate the following aspects.

## 1. Multi-Agent System

The system should be implemented using **CrewAI** and include **multiple agents** that collaborate to complete the task.

In CrewAI, agents represent autonomous entities with specific roles and goals that collaborate to perform tasks within a workflow. 



## 2. Tool Use

The system should demonstrate the use of **tools within agents**.

Tools extend the capabilities of agents by allowing them to perform actions such as web searches, API calls, or data processing. 

This includes:

### Custom Tools

Students should implement **at least one custom tool**, for example:

- reading data from a file  
- accessing an API  
- processing research topics

### Existing Tools

Students should integrate at least one **existing CrewAI tool**, such as:

- **Serper.dev** (https://serper.dev) for Google web search.



## 3. MCP Integration

The system should also demonstrate the **integration and use of MCP (Model Context Protocol)**.

MCP allows AI agents to interact with external tools, services, and data sources through standardized interfaces.


