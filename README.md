# SupportPilot — Support Triage & Drafting (ASGI Onion Middleware Architecture) <div align="center"> [![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-Registry-0194E2.svg?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests: Pytest](https://img.shields.io/badge/tests-pytest-blue.svg?logo=pytest&logoColor=white)](https://pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) </div> > **Customer support triage and grounded response drafting agent architected as an ASGI-Style Onion Middleware Pipeline — wrapping ticket contexts with composable layers for input sanitization, sentiment analysis, TF-IDF category routing, SLA priority scoring, and knowledge base retrieval.** --- ## 🏛️ Architecture Pattern: ASGI-Style Onion Middleware Pipeline Architecture Customer support triage pipelines require preprocessing, sentiment analysis, classification, priority matrix calculation, and knowledge retrieval before reaching the final draft generator. Procedural chaining leads to brittle code where inserting a guardrail or audit layer disrupts downstream logic. `supportpilot` structures the execution lifecycle as an **Onion Middleware Stack** where each layer intercepts request and response phases: ```mermaid
> **Note:** This is a portfolio project demonstrating software engineering patterns and ML concepts. Not intended for production use without further hardening. sequenceDiagram autonumber actor Client as Support Agent / Webhook participant M1 as InputSanitizationMiddleware participant M2 as SentimentScoringMiddleware participant M3 as CategoryClassifierMiddleware participant M4 as SLAPriorityMiddleware participant Terminal as Terminal Drafting Handler Client->>M1: TicketContext (raw input) M1->>M1: Sanitize text & strip dangerous chars M1->>M2: TicketContext (sanitized) M2->>M2: Extract polarity & negative cues M2->>M3: TicketContext (sentiment added) M3->>M3: TF-IDF classification & confidence M3->>M4: TicketContext (category assigned) M4->>M4: Compute priority score & SLA band (P1-P4) M4->>Terminal: TicketContext (fully enriched) Terminal->>Terminal: Retrieve KB docs & synthesize draft Terminal-->>M4: Return TicketContext (with draft) M4-->>M3: Return TicketContext M3-->>M2: Return TicketContext M2-->>M1: Return TicketContext M1-->>Client: Final Triage Result & Draft
``` ### Middleware Pipeline Features
- **Bidirectional Onion Interception**: Layers can inspect/mutate `TicketContext` both on entry and exit.
- **Short-Circuiting**: Layers (e.g. rate limiters, policy blockers) can halt execution immediately without calling `call_next`.
- **Trace Transparency**: `ctx.trace` records exact enter/exit timestamps and stage transformations for auditability. ### Module Organization
- **`middleware/context.py`**: `TicketContext` carrying mutable enrichment state.
- **`middleware/base.py`**: `Middleware` protocol and `MiddlewareStack` onion builder.
- **`middleware/layers.py`**: Concrete layers (`InputSanitization`, `SentimentScoring`, `CategoryClassifier`, `SLAPriority`).
- **`triage/`**: Machine learning classifiers and priority scorecard formulas.
- **`drafts/`**: Grounded knowledge-base draft generation. --- ## 🎧 Core Methodologies & SLA Scoring ### 1. Calibrated Multi-Label Classification
- Combines TF-IDF n-gram vectors with calibrated logistic regression to route tickets into billing, technical, account, or general queues. ### 2. SLA Priority Scorecard
- Computes priority score dynamically factoring sentiment urgency, enterprise plan tier, and outage keywords: $$\text{Priority} = w_s \cdot \max(-\text{Sentiment}, 0) + w_p \cdot \text{PlanWeight} + w_o \cdot \text{OutageFlag}$$ --- ## 🚀 Quickstart & Setup Guide ```bash
git clone https://github.com/jackson-marcus/supportpilot.git
cd supportpilot $env:UV_CACHE_DIR = "D:\ml-projects\.uv-cache"
uv sync --group dev # Run unit tests and middleware verification
uv run pytest -q
uv run ruff check . # Launch FastAPI (port :8090) + Streamlit workbench (port :8591)
make api
make ui
``` --- ## 📂 Repository Layout ```
supportpilot/
├── configs/ # Triage thresholds, SLA weights, LLM prompts
├── data/ # Support ticket dataset and knowledge base
├── src/supportpilot/ # Core Python package
│ ├── middleware/ # Onion Middleware Pipeline: context, stack, layers
│ ├── triage/ # Classifier training, inference, priority scoring
│ ├── retrieval/ # Similar ticket & knowledge base vector store
│ ├── drafts/ # Response synthesizer with citation grounding
│ ├── api/ # FastAPI REST endpoints
│ └── ui/ # Streamlit support workbench
├── tests/ # Comprehensive Pytest suite covering middleware and triage
├── docker-compose.yml
└── pyproject.toml
``` --- ## 👤 Author & Contact **Jackson Marcus**
- **Email:** [jackson.marcus.work@gmail.com](mailto:jackson.marcus.work@gmail.com)
- **Upwork:** [Jackson Marcus on Upwork](https://www.upwork.com/freelancers/~012235717501ad9c7b)
- **GitHub:** [@jackson-marcus](https://github.com/jackson-marcus) --- ## 👨‍💻 Author & Maintainer <div align="center"> ### **Jackson Marcus**
**Senior AI & Machine Learning Engineer**
*Building ML Systems, Agentic Architectures & Scalable Data Pipelines* [![GitHub Profile](https://img.shields.io/badge/GitHub-jackson--marcus-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/jackson-marcus)
[![Upwork Portfolio](https://img.shields.io/badge/Upwork-Top%20Rated%20Plus-14A800?style=for-the-badge&logo=upwork&logoColor=white)](https://www.upwork.com/freelancers/~012235717501ad9c7b)
[![Email Contact](https://img.shields.io/badge/Email-wajahatanees41%40gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:wajahatanees41@gmail.com) 📍 *Byron, GA, USA* </div>
