"""LangGraph supervisor graph for the market-intelligence workflow.

The graph is deterministic: news -> (trends || sentiment) -> competitor -> HITL review
-> persist. ``interrupt()`` in the review node makes the run resumable from a checkpoint.
Approval gates **business persistence/export**, not checkpoint storage: the checkpoint
already contains the pending profile.

A rejected run goes straight to ``END`` and never reaches the writer. An invalid resume
payload re-interrupts the review node rather than approving by coercion.
"""
from functools import partial

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from market_intelligence.agents.llm import get_llm
from market_intelligence.budget import BudgetedLLM, RunTracker
from market_intelligence.config import settings
from market_intelligence.graph.nodes import (
    analyze_news_tone_node,
    fetch_news_node,
    identify_topics_node,
    persist_node,
    research_competitor_node,
    review_profile_node,
    start_run_node,
)
from market_intelligence.graph.state import MarketIntelligenceState
from market_intelligence.persistence import ProfileWriter, default_profile_writer
from market_intelligence.retry import is_transient_error

# Network/LLM nodes retry with exponential backoff, but only for transient
# failures. Invalid feeds, SSRF rejections and 4xx responses fail immediately.
RETRY_POLICY = RetryPolicy(
    max_attempts=3,
    initial_interval=0.5,
    backoff_factor=2.0,
    retry_on=is_transient_error,
)


def route_after_review(state: MarketIntelligenceState) -> str:
    """Route an explicit decision; a missing one returns to the interrupt."""
    if state.approved is None:
        return "review"
    return "persist" if state.approved else "rejected"


def build_graph(
    checkpointer: BaseCheckpointSaver | None = None,
    llm=None,
    writer: ProfileWriter | None = None,
    max_run_tokens: int | None = None,
    max_run_llm_calls: int | None = None,
):
    """Compile the workflow.

    Args:
        checkpointer: Enables checkpointing and the HITL interrupt/resume flow.
            Use ``InMemorySaver`` for tests, ``SqliteSaver`` for local runs, and a
            Postgres/Oracle saver in production.
        llm: Optional injected chat model (tests pass a fake). When omitted the agents
            build one from settings via ``get_llm()``.
        writer: Business-persistence implementation for approved runs. When
            omitted, the writer is selected from configuration: a Postgres
            writer if ``DATABASE_URL`` is set, otherwise one that fails loudly
            at persist time. ``NullProfileWriter`` is never selected implicitly;
            tests and the fixture demo pass it explicitly.
        max_run_tokens: Optional override for the per-run token ceiling.
        max_run_llm_calls: Optional override for the per-run LLM call ceiling.

    Note:
        The compiled graph owns one shared run tracker. Build one graph per
        concurrent run; a single graph is intended for sequential runs.
    """
    tracker = RunTracker(
        max_tokens=max_run_tokens or settings.max_run_tokens,
        max_calls=max_run_llm_calls or settings.max_run_llm_calls,
    )
    llm = BudgetedLLM(llm or get_llm(), tracker)

    graph = StateGraph(MarketIntelligenceState)

    graph.add_node(
        "start_run",
        partial(start_run_node, tracker=tracker),
    )
    graph.add_node(
        "fetch_news",
        partial(fetch_news_node, llm=llm),
        retry_policy=RETRY_POLICY,
    )
    graph.add_node(
        "identify_topics",
        partial(identify_topics_node, llm=llm),
        retry_policy=RETRY_POLICY,
    )
    graph.add_node(
        "analyze_news_tone",
        partial(analyze_news_tone_node, llm=llm),
        retry_policy=RETRY_POLICY,
    )
    graph.add_node(
        "research_competitor",
        partial(research_competitor_node, llm=llm),
        retry_policy=RETRY_POLICY,
    )
    graph.add_node("review_profile", review_profile_node)
    graph.add_node(
        "persist",
        partial(persist_node, writer=writer or default_profile_writer()),
    )

    graph.add_edge(START, "start_run")
    graph.add_edge("start_run", "fetch_news")
    # Fan out: topics and news tone both consume the summaries and run in parallel.
    graph.add_edge("fetch_news", "identify_topics")
    graph.add_edge("fetch_news", "analyze_news_tone")
    # Fan in: the competitor step waits for both branches.
    graph.add_edge("identify_topics", "research_competitor")
    graph.add_edge("analyze_news_tone", "research_competitor")
    graph.add_edge("research_competitor", "review_profile")
    # Approval gate: only an explicit approval may reach the business writer.
    graph.add_conditional_edges(
        "review_profile",
        route_after_review,
        {"review": "review_profile", "persist": "persist", "rejected": END},
    )
    graph.add_edge("persist", END)

    return graph.compile(checkpointer=checkpointer)
