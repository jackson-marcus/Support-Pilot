# SupportPilot — Autonomous Customer Support Triage & Drafting Agent

<div align="center">

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-Registry-0194E2.svg?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests: Pytest](https://img.shields.io/badge/tests-pytest-blue.svg?logo=pytest&logoColor=white)](https://pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

> **Production customer support agent: calibrated multi-label intent classification, SLA priority matrix scoring, similar historical ticket retrieval, and knowledge-base grounded response drafting.**

---

## 📖 Executive Summary & Value Proposition

**`supportpilot`** is a production-grade, end-to-end machine learning system built with strict engineering discipline, reproducible pipelines, and enterprise MLOps best practices. It bridges the gap between theoretical statistical rigor and high-availability operational microservices.

## 🎧 Core Methodologies & System Architecture

### 1. Calibrated Multi-Label Intent Classification
- Multi-label classification categorizing incoming tickets into intent categories, product components, and sentiment urgency.

### 2. SLA Priority Matrix Scoring
- Computes ticket priority score dynamically factoring customer tier, churn risk, and sentiment degradation velocity.

### 3. Similar-Ticket Retrieval Engine
- Dense vector similarity search over historical resolved tickets to surface proven resolution runbooks for support agents.

### 4. Grounded Knowledge-Base Draft Generation
- Synthesizes polite, accurate resolution drafts citing official documentation articles with zero hallucinated URLs.

## 📊 Architecture & Pipeline

```mermaid
flowchart LR
    Ticket[Incoming Customer Ticket] --> Class[Multi-Label Intent & Sentiment Classifier]
    Ticket --> Score[Priority Matrix Scorer]
    Ticket --> Sim[Similar Ticket Vector Retrieval]
    Ticket --> KB[Knowledge-Base Grounded Drafter]
    Class & Score & Sim & KB --> API[FastAPI :8090] --> UI[Streamlit Support Workbench :8591]
```

## 🛠️ Tech Stack & Engineering Standards
- **AI & NLP:** Python 3.12, Scikit-Learn, Sentence-Transformers, Anthropic Claude / Ollama
- **Serving & UI:** FastAPI, Streamlit, MLflow
- **Testing:** Pytest coverage across triage rules, retrieval accuracy, and draft generation


---

## 🚀 Quickstart & Setup Guide

### 1. Prerequisites & Environment Setup
Using **[uv](https://docs.astral.sh/uv/)** for lightning-fast, reproducible dependency resolution:

```bash
# Clone the repository
git clone https://github.com/jackson-marcus/supportpilot.git
cd supportpilot

# Install dependencies and pre-commit hooks
uv sync --group dev
```

### 2. Run Test Suite & Code Quality Checks
```bash
# Run unit & integration tests with coverage
uv run pytest --cov

# Run ruff linter and formatting checks
uv run ruff check .
uv run ruff format --check .
```

### 3. Launch Services Locally
```bash
# Start FastAPI REST API (listening on port :8090)
make api
# Or: uv run uvicorn supportpilot.api.main:app --reload --port 8090

# Start interactive Streamlit dashboard (listening on port :8591)
make ui

# Launch local MLflow Experiment Tracking UI (listening on port :5009)
make mlflow
```

### 4. Run with Docker Compose
```bash
# Spin up the complete microservice stack
docker compose up --build
```

---

## 📂 Repository Layout

```
supportpilot/
├── .github/workflows/ci.yml       # GitHub Actions CI pipeline (lint, test, build)
├── configs/                      # Configuration files and hyperparameters
├── data/                         # Data directory (raw, interim, processed)
├── scripts/                      # Data generators and operational scripts
├── src/supportpilot/               # Core Python package
│   ├── api/                      # FastAPI routes, schemas, and endpoints
│   ├── models/                   # Statistical models, ML algorithms, and estimators
│   ├── ui/                       # Streamlit interactive application
│   └── settings.py               # Centralized configuration & environment loader
├── tests/                        # Comprehensive Pytest suite
├── docker-compose.yml            # Multi-service container orchestration
├── Dockerfile                    # Container definition for API service
├── Makefile                      # Standardized project tasks
└── pyproject.toml                # Pinned dependencies and tool configs
```

---

## 👤 Author & Contact

**Jackson Marcus**
- **Email:** [jackson.marcus.work@gmail.com](mailto:jackson.marcus.work@gmail.com)
- **Upwork:** [Jackson Marcus on Upwork](https://www.upwork.com/freelancers/~012235717501ad9c7b)
- **GitHub:** [@jackson-marcus](https://github.com/jackson-marcus)

*Available for machine learning engineering, MLOps, data science, and AI system architecture consulting and contract engagements.*

