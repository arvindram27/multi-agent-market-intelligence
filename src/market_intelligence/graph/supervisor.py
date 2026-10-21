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

from market_intelligence.graph.nodes import (
    analyze_sentiment_node,
    fetch_news_node,
    identify_trends_node,
    persist_node,
    research_competitor_node,
    review_profile_node,
)
from market_intelligence.graph.state import MarketIntelligenceState
from market_intelligence.persistence import NullProfileWriter, ProfileWriter

# Network/LLM nodes retry with exponential backoff before failing the run.
RETRY_POLICY = RetryPolicy(max_attempts=3, initial_interval=0.5, backoff_factor=2.0)


def route_after_review(state: MarketIntelligenceState) -> str:
    """Route an explicit decision; a missing one returns to the interrupt."""
    if state.approved is None:
        return "review"
    return "persist" if state.approved else "rejected"


def build_graph(
    checkpointer: BaseCheckpointSaver | None = None,
    llm=None,
    writer: ProfileWriter | None = None,
):
    """Compile the workflow.

    Args:
        checkpointer: Enables checkpointing and the HITL interrupt/resume flow.
            Use ``InMemorySaver`` for tests, ``SqliteSaver`` for local runs, and a
            Postgres/Oracle saver in production.
        llm: Optional injected chat model (tests pass a fake). When omitted the agents
            build one from settings via ``get_llm()``.
        writer: Business-persistence implementation for approved runs. Defaults to
            ``NullProfileWriter`` until the durable store lands; tests inject a spy
            to prove rejection never reaches it.
    """
    graph = StateGraph(MarketIntelligenceState)

    graph.add_node(
        "fetch_news",
        partial(fetch_news_node, llm=llm),
        retry_policy=RETRY_POLICY,
    )
    graph.add_node(
        "identify_trends",
        partial(identify_trends_node, llm=llm),
        retry_policy=RETRY_POLICY,
    )
    graph.add_node(
        "analyze_sentiment",
        partial(analyze_sentiment_node, llm=llm),
        retry_policy=RETRY_POLICY,
    )
    graph.add_node(
        "research_competitor",
        partial(research_competitor_node, llm=llm),
        retry_policy=RETRY_POLICY,
    )
    graph.add_node("review_profile", review_profile_node)
    graph.add_node("persist", partial(persist_node, writer=writer or NullProfileWriter()))

    graph.add_edge(START, "fetch_news")
    # Fan out: trends and sentiment both consume the summaries and run in parallel.
    graph.add_edge("fetch_news", "identify_trends")
    graph.add_edge("fetch_news", "analyze_sentiment")
    # Fan in: the competitor step waits for both branches.
    graph.add_edge("identify_trends", "research_competitor")
    graph.add_edge("analyze_sentiment", "research_competitor")
    graph.add_edge("research_competitor", "review_profile")
    # Approval gate: only an explicit approval may reach the business writer.
    graph.add_conditional_edges(
        "review_profile",
        route_after_review,
        {"review": "review_profile", "persist": "persist", "rejected": END},
    )
    graph.add_edge("persist", END)

    return graph.compile(checkpointer=checkpointer)
