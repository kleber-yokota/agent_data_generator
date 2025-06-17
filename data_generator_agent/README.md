# LLM Agent Study with LangGraph

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Purpose

This repository is dedicated to studying how to build LLM-based agents using [LangGraph](https://github.com/langchain-ai/langgraph), running local LLM models.

The goal is to deeply understand how agent architectures work — from defining workflows to managing memory and interaction patterns — while keeping everything running locally for full control and reproducibility.

## Why Local?

Running LLMs locally removes external dependencies, providing full visibility into:

- Model behavior
- Latency and performance
- Reproducibility
- Security and data privacy

This makes it easier to study how agents behave in realistic scenarios and production-like environments.

## Structure

A data-generation agent will be created as the base example.

Each folder in this repository is self-contained and includes everything needed to run that version of the agent:

- `docker-compose.yaml` for containerized setup (when applicable)  
- `requirements.txt` or `uv` for Python dependencies  
- Source code for the agent logic  
- Any additional files or instructions required to run or understand the purpose of that agent version

Each folder represents a version of the agent with a different goal, logic, or capability.  
This structure allows experimentation and learning through iterative improvements.

## Getting Started

### Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) or [pip](https://pip.pypa.io/)  
- (Optional) Docker and Docker Compose  
- Local LLM runner (such as [Ollama](https://github.com/ollama/ollama) or [vLLM](https://github.com/vllm-project/vllm))

### Running an Agent

Navigate to the folder of the agent version you want to run:



