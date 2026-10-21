# Market Intelligence Ecosystem — Production-Grade Plan

> Evaluation and migration roadmap from the current MVP to a production system built on
> **LangGraph**, **LangChain**, **Oracle AI Database (Autonomous 23ai)**, **LangSmith**,
> **OpenTelemetry / Grafana Cloud**, and **RAGAS**.

---

## 1. Executive summary

The current repository (`home/ubuntu/market_intelligence_ecosystem`) is a functional but
non-production MVP: four deterministic "agents" and a linear, hardcoded orchestrator with
`print()`-based output, no LLM, no persistence, no observability, and no evaluation.

This plan defines a target architecture and a phased migration that turns it into a
production-grade **agentic market-intelligence platform**: LLM-backed typed agents
orchestrated by a stateful LangGraph supervisor, persisted in Oracle Autonomous DB 23ai with
AI Vector Search, fully observable via LangSmith + OpenTelemetry → Grafana Cloud, and
continuously evaluated with RAGAS + LangSmith.

### Confirmed decisions

| Decision | Choice |
|---|---|
| LLM provider | **OpenAI** (`init_chat_model` abstraction to keep swap-ability) |
| Database | **OCI Autonomous Database 23ai** (AI Vector Search, HNSW/IVF) |
| Observability | **LangSmith Cloud** + **Grafana Cloud** (OTel → Tempo/Loki/Prometheus) |
| Orchestration | **LangGraph** supervisor graph with checkpointing + HITL |
| Evals | **RAGAS** + LangSmith datasets/online evaluators |

---

## 2. Current-state assessment

### 2.1 Inventory

| Component | File | Implementation | Production gap |
|---|---|---|---|
| Orchestrator | `orchestration/main_orchestrator_mvp.py` | Linear script, `print()` | No state, retries, persistence, streaming, observability |
| News agent | `agents/news_summarization_agent/news_agent_mvp.py` | `feedparser` + "first N sentences" | No LLM summarization |
| Trend agent | `agents/trend_identification_agent/trend_agent_mvp.py` | Word-frequency counter | No semantic trends, no history |
| Sentiment agent | `agents/consumer_sentiment_analysis_agent/sentiment_agent_mvp.py` | Keyword-list matching | No aspects/confidence/LLM |
| Competitor agent | `agents/competitor_analysis_agent/competitor_agent_mvp.py` | `requests` + BeautifulSoup, one hardcoded URL | No search, no structured profile, brittle scraping |

### 2.2 Anti-patterns to eliminate

1. **Brittle path manipulation** — `sys.path.append` and multi-level `os.path.dirname` in
   `main_orchestrator_mvp.py:6-40`. Replace with a real package + `pyproject.toml`.
2. **Hardcoded configuration** — RSS URL, competitor URL, user-agent string embedded in source.
   Move to `pydantic-settings` + environment/secrets.
3. **Swallowed exceptions** — `competitor_agent_mvp.py:15-17` returns `None` on any
   `RequestException`. Replace with typed error handling, retries, and graceful fallback.
4. **No typed outputs** — agents return ad-hoc dicts. Use Pydantic structured output.
5. **No LLM** — every "LLM-based" path is a placeholder comment.
6. **No state/persistence** — nothing is stored; each run is ephemeral.
7. **No observability/eval** — `print()` is the only telemetry.
8. **No tests/CI/packaging** — no `pyproject.toml`, no linting, no type checking.

---

## 3. Target architecture

```
┌────────────────────── Ingestion layer ──────────────────────┐
│  RSS/news APIs (Tavily/NewsAPI) · social feeds · web crawl   │
│  (async jobs, retries, circuit breakers, dedupe)             │
└────────────────────────────┬─────────────────────────────────┘
                             ▼
┌────────────────── LangGraph supervisor graph ────────────────┐
│  NewsAgent ──► TrendAgent ──► SentimentAgent                  │
│       └────────────► CompetitorAgent (parallel fan-out)       │
│  Typed State · Checkpointer (resume/time-travel) · HITL ·     │
│  retries/fallbacks · streaming (astream_events)               │
└──────────────┬────────────────────────────┬───────────────────┘
               ▼                            ▼
┌── Oracle Autonomous DB 23ai ───────┐  ┌── Observability ───────────┐
│  Relational tables (structured)    │  │  LangSmith Cloud (traces,  │
│  VECTOR cols + HNSW/IVF indexes    │  │    threads, evals, prompts)│
│  Oracle Text (keyword search)      │  │  OpenTelemetry → Grafana   │
│  langchain-oracledb (OracleVS)     │  │    Cloud (Tempo/Loki/      │
└────────────────────────────────────┘  │    Prometheus/Pyroscope)   │
               ▼                        └────────────────────────────┘
┌── Serving ───────────┐   ┌── Evals ────────────────────────────────┐
│  FastAPI async API    │   │  RAGAS (faithfulness, relevancy,        │
│  Arq/Celery jobs      │   │    context precision/recall)            │
│  Docker / K8s / OCI   │   │  LangSmith offline datasets + online    │
└───────────────────────┘   │  evaluators + CI gating                 │
                            └─────────────────────────────────────────┘
```

---

## 4. Repository structure (target)

```
market_intelligence_ecosystem/
├── pyproject.toml
├── .env.example
├── docker-compose.yml
├── Dockerfile
├── Makefile
├── app.py                        # Streamlit entrypoint (streamlit run app.py)
├── ui/
│   ├── session.py                # graph + thread_id + checkpointer init (st.session_state)
│   ├── render.py                 # streaming loop + HITL approval wiring
│   ├── charts.py                 # Plotly helpers (trends, sentiment, competitors)
│   └── components/
│       ├── news.py               # NewsSummary table/cards
│       ├── trends.py             # TrendReport + confidence bar chart
│       ├── sentiment.py          # SentimentResult gauge + aspect breakdown
│       └── competitors.py        # CompetitorProfile comparison + approve button
├── src/
│   └── market_intelligence/
│       ├── __init__.py
│       ├── config.py                 # pydantic-settings
│       ├── logging.py                # structlog setup
│       ├── llm.py                    # init_chat_model + structured output helpers
│       ├── db/
│       │   ├── connection.py         # python-oracledb pool
│       │   ├── schema.sql            # DDL (tables + vector indexes)
│       │   ├── repositories/         # articles, trends, sentiments, competitors
│       │   └── vectorstore.py        # OracleVS + embeddings
│       ├── agents/
│       │   ├── news/                 # node + prompt + model
│       │   ├── trend/
│       │   ├── sentiment/
│       │   └── competitor/
│       ├── tools/
│       │   ├── news_search.py        # Tavily/NewsAPI
│       │   ├── web_scrape.py         # Trafilatura/Firecrawl
│       │   └── competitor_search.py
│       ├── graph/
│       │   ├── state.py              # Pydantic State
│       │   ├── supervisor.py         # graph definition
│       │   └── nodes.py              # node wrappers
│       ├── ingestion/
│       │   └── pipeline.py           # scheduled crawl + normalize
│       ├── api/
│       │   ├── main.py               # FastAPI app
│       │   └── routes/
│       ├── observability/
│       │   ├── otel.py               # OTel setup + exporters
│       │   └── langsmith.py          # LangSmith config
│       └── evals/
│           ├── datasets.py
│           ├── ragas_metrics.py
│           └── online_evaluators.py
├── tests/
│   ├── unit/
│   └── integration/
└── scripts/
    ├── migrate.py
    └── eval.py
```

---

## 5. Layer designs

### 5.1 LangGraph orchestration

Replace the linear orchestrator with a typed, stateful supervisor graph.

**State (Pydantic):**

```python
from typing import Annotated, Literal
from pydantic import BaseModel
from langgraph.graph.message import add_messages

class Article(BaseModel):
    id: str
    title: str
    url: str
    content: str
    source: str
    published_at: str

class MarketIntelligenceState(BaseModel):
    run_id: str
    target_market: str
    articles: list[Article] = []
    news_summaries: list[NewsSummary] = []
    trends: list[TrendReport] = []
    sentiments: list[SentimentResult] = []
    competitor_profiles: list[CompetitorProfile] = []
    pending_approval: CompetitorProfile | None = None
    status: Literal["running", "awaiting_approval", "complete", "failed"] = "running"
    error: str | None = None
```

**Graph sketch:**

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.base import BaseCheckpointSaver

def build_graph(checkpointer: BaseCheckpointSaver) -> StateGraph:
    g = StateGraph(MarketIntelligenceState)
    g.add_node("fetch_news", fetch_news_node)
    g.add_node("summarize", summarize_node)          # LLM map-reduce
    g.add_node("identify_trends", trends_node)
    g.add_node("analyze_sentiment", sentiment_node)
    g.add_node("research_competitors", competitors_node)  # fan-out
    g.add_node("review_profile", review_node)        # HITL
    g.add_node("persist", persist_node)

    g.add_edge(START, "fetch_news")
    g.add_edge("fetch_news", "summarize")
    g.add_edge("summarize", "identify_trends")
    g.add_edge("summarize", "analyze_sentiment")
    g.add_edge("identify_trends", "research_competitors")
    g.add_edge("analyze_sentiment", "research_competitors")
    g.add_edge("research_competitors", "review_profile")
    # HITL: only an explicit approval may reach the business writer.
    g.add_conditional_edges(
        "review_profile",
        lambda s: "persist" if s.approved else "rejected",
        {"persist": "persist", "rejected": END},
    )
    g.add_edge("persist", END)
    return g.compile(checkpointer=checkpointer)
```

Key capabilities:
- **Checkpointing** (`langgraph-checkpoint-postgres` for prod, `-sqlite` for local dev) for
  resume, audit, and time-travel debugging.
- **Human-in-the-loop** via `interrupt()` inside `review_profile` to approve competitor
  profiles **before business persistence/export**. The checkpoint already stores the pending
  profile; the gate controls the writer, not all storage.
- **Streaming** via `graph.astream_events(...)` to emit node/token events into OTel.
- **Retries** via LangGraph node `RetryPolicy`, and `fallback` chains for LLM failures.

### 5.2 LangChain agents + tools

- **Models** — `from langchain.chat_models import init_chat_model` keyed to `OPENAI_*` env
  vars; per-agent model config in settings (allows swapping provider later without code change).
- **Structured output** — each agent calls `.with_structured_output(<PydanticModel>)` so outputs
  are validated before entering the graph:

```python
class NewsSummary(BaseModel):
    headline: str
    summary: str
    key_facts: list[str]
    sentiment_hint: Literal["positive", "negative", "neutral"]

class TrendReport(BaseModel):
    trend_name: str
    keywords: list[str]
    confidence: float          # 0..1
    evidence: list[str]

class SentimentResult(BaseModel):
    overall: Literal["positive", "negative", "neutral", "mixed"]
    score: float               # -1..1
    aspects: list[dict]        # [{aspect, sentiment, confidence}]

class CompetitorProfile(BaseModel):
    company: str
    positioning: str
    products: list[str]
    pricing_model: str | None
    funding_stage: str | None
    strengths: list[str]
    weaknesses: list[str]
    sources: list[str]
```

- **Tools** — replace raw `requests`/`BeautifulSoup` with retriever/scraper tools the LLM can
  invoke: `search_news` (Tavily or NewsAPI), `scrape_url` (Trafilatura/Firecrawl), and
  `search_competitor`.
- **Prompts** — manage in **LangChain Hub / LangSmith Prompts** for versioning and A/B.

### 5.3 Oracle Autonomous DB 23ai (data layer)

#### 5.3.1 Relational schema (canonical tables)

```sql
CREATE TABLE news_articles (
    article_id     RAW(16) DEFAULT SYS_GUID() PRIMARY KEY,
    source         VARCHAR2(100) NOT NULL,
    title          VARCHAR2(500) NOT NULL,
    url            VARCHAR2(2000) NOT NULL,
    content        CLOB,
    summary        CLOB,
    embedding      VECTOR(*, FLOAT),
    published_at   TIMESTAMP WITH TIME ZONE,
    ingested_at    TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP
);

CREATE TABLE trends (
    trend_id       RAW(16) DEFAULT SYS_GUID() PRIMARY KEY,
    run_id         RAW(16) NOT NULL,
    trend_name     VARCHAR2(200) NOT NULL,
    keywords       JSON,
    confidence     NUMBER,
    evidence       JSON,
    detected_at    TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP
);

CREATE TABLE sentiment_records (
    sentiment_id   RAW(16) DEFAULT SYS_GUID() PRIMARY KEY,
    article_id     RAW(16) REFERENCES news_articles(article_id),
    overall        VARCHAR2(20),
    score          NUMBER,
    aspects        JSON,
    analyzed_at    TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP
);

CREATE TABLE competitor_profiles (
    profile_id     RAW(16) DEFAULT SYS_GUID() PRIMARY KEY,
    company        VARCHAR2(200) NOT NULL,
    profile        JSON,
    embedding      VECTOR(*, FLOAT),
    approved_by    VARCHAR2(100),
    approved_at    TIMESTAMP WITH TIME ZONE,
    created_at     TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP
);

CREATE TABLE analysis_runs (
    run_id         RAW(16) PRIMARY KEY,
    target_market  VARCHAR2(200),
    status         VARCHAR2(30),
    started_at     TIMESTAMP WITH TIME ZONE,
    finished_at    TIMESTAMP WITH TIME ZONE
);
```

#### 5.3.2 Vector indexes

```sql
CREATE VECTOR INDEX news_embedding_hnsw ON news_articles(embedding)
  ORGANIZATION INMEMORY NEIGHBOR GRAPH
  DISTANCE COSINE
  WITH TARGET ACCURACY 95;
```

#### 5.3.3 RAG / retrieval with `langchain-oracledb`

```python
import oracledb
from langchain_oracledb.vectorstores import oraclevs
from langchain_oracledb.vectorstores.oraclevs import OracleVS
from langchain_oracledb.embeddings import OracleEmbeddings

# Connection pool (python-oracledb thin mode)
pool = oracledb.create_pool(
    user=..., password=..., dsn="<host>:<port>/<service_name>",
    min=2, max=10, increment=1,
)

# DB-resident embeddings (no external embedding service)
embeddings = OracleEmbeddings(conn=pool, params={"provider": "database", "model": "DB_MODEL"})

vs = OracleVS(pool, table_name="news_articles", embedding_function=embeddings)

oraclevs.create_index(pool, vs, params={
    "idx_name": "news_embedding_hnsw", "idx_type": "HNSW", "accuracy": 95,
})

retrieved = vs.similarity_search("generative AI in healthcare", k=5)
```

Notes:
- Prefer **DB-resident embeddings** (`OracleEmbeddings`) to keep data + vectors in one system.
- Use **metadata filters** (`$eq`, `$between`, `$in`, `$and`/`$or`) for relational filtering during
  vector search (e.g. `source`, `published_at` window).
- On **26ai**, add `OracleHybridSearchRetriever` (keyword + semantic) and
  `OracleTextSearchRetriever` (Oracle Text `CONTAINS`). On 23ai, pair `OracleVS` with a
  `CREATE SEARCH INDEX`/Oracle Text for keyword-only retrieval.

### 5.4 Observability

#### 5.4.1 LangSmith Cloud
- Set `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT`.
- Native LangGraph integration gives per-node traces, threads, and token/cost attribution.
- Use **dashboards** and **rules/alerts** for quality monitoring, and **Engine** for automatic
  failure diagnosis.

#### 5.4.2 OpenTelemetry → Grafana Cloud
```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(
    OTLPSpanExporter(endpoint="<grafana-otlp-endpoint>", headers={"Authorization": "Basic ..."})
))
trace.set_tracer_provider(provider)
```
- OTLP spans → **Grafana Tempo**; structured logs (`structlog`) → **Loki**; metrics → **Prometheus**;
  CPU/mem → **Pyroscope**.
- Enable the **Grafana LLM observability app** for agent-run views.
- Replace every `print()` with `structlog` structured logging (run_id, node, model, latency).

**Key metrics to emit:** per-node latency, token usage/cost, error rate, retrieval recall,
embedding count, queue depth, and data-freshness (time since last successful ingestion).

### 5.5 Evaluation

#### 5.5.1 Offline (LangSmith datasets + RAGAS)
- Curate **golden datasets** per agent (news → reference summaries; sentiment → labeled posts;
  trends → expected trend sets; competitor → reference profiles).
- Metrics:
  - Retrieval path: **RAGAS** `faithfulness`, `answer_relevancy`, `context_precision`,
    `context_recall`.
  - Generation: `correctness`, LLM-as-judge for summary/trend/sentiment quality.
- Run via `langsmith evaluate` or `scripts/eval.py`; store a leaderboard of prompt/model versions.

#### 5.5.2 Online (continuous)
- LangSmith **online evaluators** to auto-flag hallucinations, empty outputs, and irrelevant
  retrieval in production traces.
- **Rules/webhooks** to alert on metric regressions.
- **CI gating:** block deploys when eval scores fall below threshold.

### 5.6 Streamlit demo UI

Streamlit is the recommended first front-end for demoing the agentic platform. It reuses the
exact same `build_graph()` with zero extra HTTP/JSON plumbing, streams each agent's output as it
lands, and maps LangGraph's human-in-the-loop `interrupt` directly onto a Streamlit button.

#### 5.6.1 Why Streamlit first (and the one gotcha)

- **No front-end code** — render the Pydantic outputs as `st.dataframe` / Plotly charts.
- **Streaming is natural** — iterate `graph.stream(..., stream_mode="updates")` and render each
  node's output as it completes.
- **HITL maps cleanly** — when the graph hits `interrupt`, show an approve button; on click,
  call `graph.invoke(None, config)` to resume.
- **Same observability** — it's the same Python process, so LangSmith traces and OTel → Grafana
  still capture every run; surface a "View trace in LangSmith" link in the sidebar.

**Gotcha:** Streamlit re-runs the script on every interaction and is synchronous, while LangGraph
async streaming needs an event loop. For a demo, avoid `astream_events` and drive runs with the
**synchronous** `graph.stream()` / `graph.invoke()`. Build the graph once, cache it in
`st.session_state`, and pin a stable `thread_id` so runs survive reruns.

#### 5.6.2 Exact file layout

```
app.py                            # Streamlit entrypoint — run: streamlit run app.py
ui/
├── session.py                    # graph + checkpointer + thread_id (st.session_state)
├── render.py                     # streaming loop + HITL approval wiring
├── charts.py                     # Plotly helpers (trends, sentiment, competitors)
└── components/
    ├── news.py                   # NewsSummary cards / dataframe
    ├── trends.py                 # TrendReport + confidence bar chart
    ├── sentiment.py              # SentimentResult gauge + aspect breakdown
    └── competitors.py            # CompetitorProfile comparison + approve button
```

#### 5.6.3 Entrypoint sketch

```python
# app.py
import streamlit as st
from market_intelligence.graph.supervisor import build_graph
from ui.session import init_session
from ui import render

st.set_page_config(page_title="Market Intelligence", layout="wide")

graph, config = init_session()          # builds graph once, stashes in session_state

with st.sidebar:
    st.title("Market Intelligence")
    market = st.text_input("Target market / topic", "generative AI in healthcare")
    run = st.button("Run analysis", type="primary")
    if st.session_state.get("last_run_id"):
        st.caption(f"Thread: {st.session_state.last_run_id}")

tabs = st.tabs(["News", "Trends", "Sentiment", "Competitors"])

if run:
    render.run_and_stream(graph, config, {"target_market": market}, tabs)
```

```python
# ui/session.py
import streamlit as st
from langgraph.checkpoint.sqlite import SqliteSaver

def init_session():
    if "graph" not in st.session_state:
        st.session_state.graph = build_graph(checkpointer=SqliteSaver.from_conn_string("checkpoints.db"))
    thread_id = st.session_state.setdefault("thread_id", "demo-1")
    return st.session_state.graph, {"configurable": {"thread_id": thread_id}}
```

```python
# ui/render.py
import streamlit as st

def run_and_stream(graph, config, inputs, tabs):
    with st.status("Running agents…", expanded=True) as status:
        for update in graph.stream(inputs, config, stream_mode="updates"):
            for node, data in update.items():
                st.write(f"**{node}** finished")
                # route `data` into the matching tab via ui.components.*
        status.update(label="Complete", state="complete")
        st.session_state.last_run_id = config["configurable"]["thread_id"]

    # Human-in-the-loop: approve competitor profile before persistence
    snap = graph.get_state(config)
    if snap.next and "review_profile" in snap.next:
        profile = snap.values.get("pending_approval")
        with tabs[3]:
            if profile:
                st.json(profile.model_dump())
                if st.button("Approve competitor profile", type="primary"):
                    graph.invoke(None, config)     # resume past the interrupt
```

#### 5.6.4 Tab rendering

| Tab | Renders | Visualization |
|---|---|---|
| News | `list[NewsSummary]` | `st.dataframe` / expander cards |
| Trends | `list[TrendReport]` | Plotly horizontal bar of `confidence`, keyword chips |
| Sentiment | `list[SentimentResult]` | Gauge of `score` + aspect breakdown table |
| Competitors | `list[CompetitorProfile]` | Comparison table + approve button (HITL) |

#### 5.6.5 Vertical-slice milestone (build this first)

A minimal end-to-end demo that proves the whole stack without needing Oracle or Grafana yet:

1. `NewsAgent` (OpenAI via `init_chat_model`) — fetch + summarize live news through the
   `search_news` tool.
2. `TrendAgent` + `SentimentAgent` — consume the summaries, emit typed `TrendReport` /
   `SentimentResult`.
3. `CompetitorAgent` — web-search tool + structured `CompetitorProfile`, with the **approve
   button** (HITL) before persistence.
4. `sqlite` checkpointer + `app.py` streaming loop, LangSmith tracing on.

**Vertical-slice exit criteria:** a viewer can type a market, watch all four agents stream their
outputs into tabs, and approve/reject the competitor profile — with the full run visible as a
LangSmith trace. Oracle persistence (Phase 3) and Grafana (Phase 4) layer onto the same UI
without changes.

---

## 6. Reliability, security, deployment

- **Packaging/config:** `pyproject.toml` + `uv`; `pydantic-settings`; `.env.example`; secrets in
  OCI Vault / env (never in source).
- **Error handling:** typed exceptions, LangGraph `RetryPolicy` with exponential backoff,
  circuit breakers on external APIs, idempotent ingestion (dedupe on `url` hash).
- **Serving:** FastAPI (async) exposing `/run`, `/runs/{id}`, `/trends`, `/competitors`;
  Arq/Celery for scheduled ingestion; WebSocket/SSE for streaming updates.
- **Containers:** Docker for dev; deploy to K8s or OCI Container Instances; autoscale workers.
- **Security:** least-privilege DB user, TLS to OCI, rate limiting, input validation, PII
  redaction in ingested content before persistence.
- **Data governance:** retention policies on `news_articles`, audit columns on
  `competitor_profiles` (approved_by/approved_at).

---

## 7. Phased roadmap

| Phase | Scope | Exit criteria |
|---|---|---|
| **0 — Foundation** | Package with `pyproject.toml` + `uv`; `pydantic-settings`; `structlog`; `ruff`/`mypy`/`pytest`/pre-commit; CI | Clean install, lint/type/test green |
| **1 — LLM-ify agents** | Replace 4 deterministic functions with typed OpenAI agents + tools (`init_chat_model`, `with_structured_output`) | Each agent returns validated Pydantic output; unit tests pass |
| **2 — Orchestration** | Build LangGraph supervisor graph + checkpointing + HITL + retries/fallbacks | Runs resumable, streamable, auditable |
| **2.5 — Streamlit demo (vertical slice)** | `app.py` + `ui/` components; stream node outputs; approve button (HITL); LangSmith tracing | Viewer streams all 4 agents into tabs and approves/rejects competitor profile |
| **3 — Data** | Oracle 23ai schema, embeddings, HNSW/IVF, RAG retrieval via `OracleVS` | Articles/trends/sentiments/competitors persisted & searchable |
| **4 — Observability** | LangSmith tracing + OTel → Grafana Cloud (Tempo/Loki/Prometheus) | End-to-end traces + dashboards live |
| **5 — Evals** | RAGAS + LangSmith datasets, online evaluators, CI gating | Eval reports in CI; regressions blocked |
| **6 — Serving & hardening** | FastAPI, background jobs, Docker/K8s/OCI, security, data governance | Production deployment, autoscaling, secured |

---

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| LLM cost/latency at scale | Batch ingestion, caching embeddings, map-reduce summarization, token budget caps, per-model config |
| Scraping fragility / ToS | Use managed fetch tools (Firecrawl/Trafilatura), respect `robots.txt`, rate limits |
| Vector search quality | Curate chunking strategy, HNSW accuracy tuning, hybrid search (26ai), eval gate before rollout |
| Oracle 23ai feature availability | Confirm AI Vector Search is enabled in the chosen Autonomous DB tier; test DB-resident embedding model availability |
| Hallucination in generated intelligence | RAGAS faithfulness + LLM-as-judge + human approval (HITL) before publishing competitor profiles |
| Checkpoint DB load | Separate checkpoint store (Postgres) or reuse Oracle; size/monitor |

---

## 9. References

- **LangGraph:** https://langchain-ai.github.io/langgraph/ (concepts, checkpointing, HITL, streaming)
- **LangChain chat models / structured output:** https://python.langchain.com/docs/
- **LangChain Oracle vector store:** https://python.langchain.com/docs/integrations/vectorstores/oracle/
  (`langchain-oracledb`, `OracleVS`, `OracleEmbeddings`, HNSW/IVF, filters, hybrid/text search)
- **Oracle AI Vector Search:** https://docs.oracle.com/en/database/oracle/oracle-database/23/vecse/
- **LangSmith Observability & Evaluation:** https://docs.langchain.com/ (traces, dashboards, rules, evals)
- **RAGAS:** https://docs.ragas.io/ (faithfulness, answer relevancy, context precision/recall)
- **Grafana LLM observability:** https://grafana.com/docs/grafana-cloud/ (Tempo, Loki, Prometheus, Pyroscope, OTLP)
- **OpenTelemetry Python:** https://opentelemetry.io/docs/languages/python/
- **Streamlit:** https://docs.streamlit.io/ (session state, `st.status`, `st.dataframe`, Plotly integration)
