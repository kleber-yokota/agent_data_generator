# Structured Data Generation with LLMs

This project explores techniques for generating structured data using Large Language Models (LLMs), with a focus on improving reliability, testing, and observability.

## 🔍 What’s Inside

- **Common pitfalls** when working with LLM-generated JSON and CSV outputs
- **Custom format enforcement (guardrails)** to ensure structured responses
- **Unit testability** of graph-based pipelines using LangGraph
- **Custom CSV parsing** from markdown blocks
- **Prompt injection vulnerabilities** and mitigation ideas
- **OpenLIT integration** for tracing LLM calls
- **ClickStack observability setup**, using:
  - ClickHouse for high-performance analytics
  - HyperDX as the UI layer
  - OpenTelemetry for trace collection

## 💡 Highlights

- Easily trace model interactions and outputs using [OpenLIT](https://github.com/openlit/openlit)
- Modular LangGraph setup with testable nodes
- Observability stack deployable via Docker Compose or Helm


