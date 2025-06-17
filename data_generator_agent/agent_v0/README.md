
# Data Generator Agent

## Overview

This repository is dedicated to studying how to create agents with LangGraph running local LLMs (Large Language Models).  
The main focus is to build a data generator agent. Each folder contains a version of the agent with a specific objective.

---

## About the Code

The main Python code defines an **ExtractionAgent** that interacts with an LLM (Qwen3:4b) via the `ChatOllama` interface to perform a pipeline of tasks:

- **Entity Extraction:** Extracts structured metadata entities (such as theme, columns, rows, and specifications) from a user's input prompt using a chat prompt and a low-temperature LLM call.
- **Metadata Generation:** Based on the extracted entities, it generates a detailed JSON metadata schema describing CSV file columns, their types, descriptions, and formatting rules.
- **Intention Detection:** Classifies user input intentions as either `"modification"` (schema changes) or `"generation"` (data generation requests) using the LLM.
- **Data Generation:** Produces realistic CSV data matching the generated metadata schema, respecting constraints such as patterns, data types, and formats.
- **StateGraph Workflow:** These steps are orchestrated via a `StateGraph` which manages the flow between entity extraction, metadata generation, intention detection, and data generation. The graph loops back to entity extraction if modifications are requested, otherwise, it proceeds to data generation.

The project leverages the LangChain ecosystem and LangGraph framework for prompt management, LLM invocation, and pipeline state management.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-user/data-generator-agent.git
cd data-generator-agent

# (Optional but recommended) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install the Python dependencies
pip install -e .

# Pull the Qwen3:4b model inside the ollama Docker container
docker compose exec ollama ollama pull qwen3:4b
```

---

## Usage

After installing dependencies and pulling the model, you can run the agent version you want by navigating to its folder and executing the respective commands, usually with Docker Compose or Python directly.

---

## Dependencies

- Python >= 3.13  
- langchain-community >= 0.3.24  
- langchain-core >= 0.3.64  
- langgraph >= 0.4.8  
- pandas >= 2.3.0  

---

## Notes

- Make sure Docker and Docker Compose are installed and running to use the containers.
- The `ollama` container must be running for pulling models and running the LLM.
- This project is for educational purposes to understand LLM agents running locally with LangGraph.