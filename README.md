# Multi-Agent AI Ecosystem for Market Intelligence

A comprehensive multi-agent system designed to automate market intelligence gathering, processing, and analysis. This project implements specialized AI agents that work together to provide insights from news, competitor activities, market trends, and consumer sentiment.

> 📄 **Production roadmap:** see [`PRODUCTION_PLAN.md`](./PRODUCTION_PLAN.md) for the full migration plan (LangGraph · LangChain · Oracle 23ai · LangSmith · Grafana · RAGAS).

## 🚀 Features

### Core Agents
- **News Summarization Agent**: Monitors RSS feeds and creates actionable news summaries
- **Competitor Analysis Agent**: Tracks competitor websites and activities through web scraping
- **Trend Identification Agent**: Identifies emerging trends through keyword frequency analysis
- **Consumer Sentiment Analysis Agent**: Analyzes public sentiment using basic classification

### Orchestration
- **Main Orchestrator**: Coordinates all agents in a unified workflow
- **Data Processing Pipeline**: Sequential processing with inter-agent communication

## 📁 Project Structure

```
├── agents/
│   ├── news_summarization_agent/
│   │   └── news_agent_mvp.py          # News monitoring and summarization
│   ├── competitor_analysis_agent/
│   │   └── competitor_agent_mvp.py    # Competitor tracking and analysis
│   ├── trend_identification_agent/
│   │   └── trend_agent_mvp.py         # Trend detection and analysis
│   └── consumer_sentiment_analysis_agent/
│       └── sentiment_agent_mvp.py     # Sentiment analysis engine
├── orchestration/
│   └── main_orchestrator_mvp.py       # Main coordination system
├── data/                              # Runtime output / cache
├── PRODUCTION_PLAN.md                 # Production-grade migration roadmap
├── page.tsx                           # Web UI component
├── documentation/
│   ├── AI Agent Ecosystem Architecture for Market Intelligence.md
│   ├── AI Agent Ecosystem_ Components and Roles.md
│   └── Critical Analysis of the AI Agent Ecosystem for Market Intelligence.md
├── assets/
│   ├── hero_image_symbiotic_intelligence.png
│   └── icon_*.png                     # Agent icons
└── README.md
```

> **Legacy vs. LLM:** the `agents/` and `orchestration/` directories at the repo root are the
> **legacy deterministic MVP** (runnable baseline). The **LLM-backed agents** live in
> `src/market_intelligence/agents/` (imported as `market_intelligence.agents.*`). The legacy
> tree will be removed in a later phase of `PRODUCTION_PLAN.md`.

## 🛠️ Current Implementation Status

### MVP Features (✅ Implemented)
- Basic RSS feed processing
- Simple web scraping capabilities
- Keyword-based trend analysis
- Rule-based sentiment classification
- Sequential agent orchestration

### Production Features (🚧 Planned — see `PRODUCTION_PLAN.md`)
- **LLM agents** via LangChain (`init_chat_model` → OpenAI) with typed Pydantic structured output
- **LangGraph** supervisor orchestration with checkpointing, retries, and human-in-the-loop
- **Oracle Autonomous Database 23ai** persistence + AI Vector Search (RAG)
- **Observability** via LangSmith + OpenTelemetry → Grafana Cloud
- **Evaluation** via RAGAS + LangSmith datasets (offline + online)
- **Streamlit demo UI** with live agent streaming

## 🚀 Quick Start

### Prerequisites
Python 3.10+ and the pinned dependencies in `pyproject.toml`.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Canonical install (package + runtime dependencies):
pip install -e .
# Development install (adds test tooling):
pip install -e ".[dev]"
```

> `requirements.txt` mirrors the runtime pins from `pyproject.toml` for environments that
> only need the dependencies, not the package itself (`pip install -r requirements.txt`).

### Running the System
Run from the repository root so the `agents` package is importable.

```bash
# Run individual agents
python agents/news_summarization_agent/news_agent_mvp.py
python agents/competitor_analysis_agent/competitor_agent_mvp.py
python agents/trend_identification_agent/trend_agent_mvp.py
python agents/consumer_sentiment_analysis_agent/sentiment_agent_mvp.py

# Run complete orchestrated system (from the repo root)
python -m orchestration.main_orchestrator_mvp
```

### Example Output
The orchestrator will:
1. Fetch and summarize recent news articles
2. Extract trends from the news content
3. Analyze sentiment of the content
4. Scrape competitor information from a predefined URL
5. Display coordinated results from all agents

## ⚙️ Configuration

The LLM agents (`src/market_intelligence/`) read settings from environment variables or a
`.env` file (copy `.env.example` to `.env`). Model name and timeouts are configurable:

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | _(empty)_ | OpenAI API key (required for live LLM runs) |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model used by all agents |
| `NEWS_REQUEST_TIMEOUT` | `15.0` | RSS fetch timeout (seconds) |
| `COMPETITOR_REQUEST_TIMEOUT` | `15.0` | Competitor page fetch timeout (seconds) |
| `NUM_ARTICLES` | `2` | Articles processed per run |

### 💰 Cost per run

**Placeholder** — not yet measured. A live run requires an `OPENAI_API_KEY`, which is not
available in this environment, so the token count and cost per orchestrator run have not
been benchmarked. Update this figure after a real run against `gpt-4o-mini`.

## 🖥️ Streamlit Demo (planned)

A Streamlit app will drive the LangGraph workflow and stream each agent's output into tabs
(News · Trends · Sentiment · Competitors), with a human-approval button for competitor profiles.
See **§5.6 Streamlit demo UI** and the **vertical-slice milestone** in `PRODUCTION_PLAN.md`.

```bash
streamlit run app.py
```

## 📊 Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  News Agent     │    │ Competitor Agent │    │  Trend Agent    │
│                 │    │                  │    │                 │
│ • RSS Parsing   │    │ • Web Scraping   │    │ • Keyword Freq  │
│ • Summarization │    │ • Data Extraction│    │ • Pattern Detect│
└─────────┬───────┘    └────────┬─────────┘    └─────────┬───────┘
          │                     │                        │
          │              ┌──────▼──────┐                 │
          └──────────────►│ Orchestrator├─────────────────┘
                         │             │
          ┌──────────────►│ Coordinates ├─────────────────┐
          │              │  All Agents │                 │
          │              └─────────────┘                 │
          │                                              │
┌─────────▼───────┐                              ┌───────▼─────────┐
│ Sentiment Agent │                              │   Data Storage  │
│                 │                              │                 │
│ • Text Analysis │                              │ • Results       │
│ • Classification│                              │ • Insights      │
└─────────────────┘                              └─────────────────┘
```

## 🔧 Technology Stack

**Current MVP Stack:**
- Python 3.x
- feedparser (RSS processing)
- requests + BeautifulSoup (web scraping)
- collections.Counter (basic analytics)

**Planned Production Stack:**
- LangGraph (stateful orchestration) + LangChain (typed agents & tools)
- OpenAI models (`init_chat_model` provider-agnostic)
- Oracle Autonomous Database 23ai — AI Vector Search (HNSW/IVF) via `langchain-oracledb`
- LangSmith (tracing, evals, prompts) + OpenTelemetry → Grafana Cloud (Tempo/Loki/Prometheus)
- RAGAS (faithfulness, answer relevancy, context precision/recall)
- FastAPI (serving) + Streamlit (demo UI)

## 📈 Roadmap

| Phase | Scope |
|---|---|
| 0 — Foundation | Packaging (`pyproject.toml` + `uv`), config, `structlog`, CI, tests |
| 1 — LLM-ify agents | Typed OpenAI agents + tools (`with_structured_output`) |
| 2 — Orchestration | LangGraph supervisor graph + checkpointing + HITL |
| 2.5 — Streamlit demo | Vertical slice: stream all agents into tabs + approve button |
| 3 — Data | Oracle 23ai schema, embeddings, vector search/RAG |
| 4 — Observability | LangSmith + OTel → Grafana Cloud |
| 5 — Evals | RAGAS + LangSmith datasets, CI gating |
| 6 — Serving & hardening | FastAPI, background jobs, Docker/K8s/OCI, security |

See `PRODUCTION_PLAN.md` for full detail and exit criteria.

## 🤝 Contributing

This project is designed as a comprehensive market intelligence solution. Future contributions could focus on:

- Enhanced agent capabilities
- Integration with additional data sources
- Advanced AI/ML model integration
- UI/UX improvements
- Performance optimization

## 📄 Documentation

- **Production roadmap:** [`PRODUCTION_PLAN.md`](./PRODUCTION_PLAN.md)
- **Architecture Overview**: System design and component relationships
- **Component Specifications**: Detailed role definitions for each agent
- **Critical Analysis**: SWOT analysis and relevance assessment

## 🔐 Security & Privacy

- Implement proper API key management
- Follow data privacy regulations (GDPR, CCPA)
- Secure web scraping practices
- Ethical AI model deployment

## 📞 Support

For questions about implementation or extending this system, refer to the comprehensive documentation provided in the project.

---

**Status**: MVP Implementation Complete | **Next Phase**: LLM Integration & Production Readiness (see `PRODUCTION_PLAN.md`)
