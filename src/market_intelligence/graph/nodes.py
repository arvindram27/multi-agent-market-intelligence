"""Graph nodes. Each node wraps an existing agent function and returns a state update."""
import logging
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from market_intelligence.agents.competitor import fetch_and_analyze
from market_intelligence.agents.news import ingest_news
from market_intelligence.agents.sentiment import analyze_sentiment
from market_intelligence.agents.trend import identify_trends
from market_intelligence.graph.state import MarketIntelligenceState
from market_intelligence.persistence import NullProfileWriter, ProfileWriter

logger = logging.getLogger(__name__)


class InvalidDecisionError(ValueError):
    """Raised when an approval payload is not an explicit boolean decision."""


def parse_decision(decision: Any) -> bool:
    """Validate a resume payload and return the approval decision.

    Accepts only a JSON boolean or ``{"approved": <boolean>}``. Strings such as
    ``"false"`` and integers are rejected rather than coerced, because ``bool()``
    would turn both of those into ``True``.
    """
    if isinstance(decision, bool):
        return decision
    if isinstance(decision, dict) and isinstance(decision.get("approved"), bool):
        return decision["approved"]
    raise InvalidDecisionError(
        "Approval decision must be a boolean or {'approved': <boolean>}; "
        f"received {type(decision).__name__}."
    )


def start_run_node(state: MarketIntelligenceState, tracker=None) -> dict[str, Any]:
    """Reset per-run LLM counters before the first work node.

    This node has no retry policy and runs once per new thread. Resuming an
    approval does not pass through it, so the resumed run keeps the budget it
    already consumed. Keeping the reset out of ``fetch_news`` is deliberate:
    retries of that node must not get a fresh ceiling.
    """
    if tracker is not None:
        tracker.reset()
    return {}


def fetch_news_node(
    state: MarketIntelligenceState, llm=None
) -> dict[str, Any]:
    """Fetch the RSS feed and summarize each usable article."""
    report = ingest_news(
        state.rss_url, num_articles=state.num_articles, llm=llm
    )
    return {"summaries": report.summaries, "warnings": report.warnings}


def identify_trends_node(state: MarketIntelligenceState, llm=None) -> dict[str, Any]:
    """Extract trends from the article summaries."""
    return {"trends": identify_trends([s.summary for s in state.summaries], llm=llm)}


def analyze_sentiment_node(state: MarketIntelligenceState, llm=None) -> dict[str, Any]:
    """Classify sentiment for every article summary."""
    return {
        "sentiments": [analyze_sentiment(s.summary, llm=llm) for s in state.summaries]
    }


def research_competitor_node(state: MarketIntelligenceState, llm=None) -> dict[str, Any]:
    """Scrape the competitor page and extract a structured profile."""
    return {
        "competitor": fetch_and_analyze(state.competitor_url, llm=llm),
        "status": "awaiting_approval",
    }


def review_profile_node(state: MarketIntelligenceState) -> dict[str, Any]:
    """Pause for human approval of the competitor profile.

    Resuming with an explicit boolean moves the run to ``approved_pending_write``
    or ``rejected``. An invalid payload clears nothing, records an error, and
    routes back here for another interrupt; the thread stays resumable.
    """
    decision = interrupt(
        {
            "type": "approval_request",
            "competitor": state.competitor.model_dump() if state.competitor else None,
        }
    )
    try:
        approved = parse_decision(decision)
    except InvalidDecisionError as exc:
        logger.warning("Rejected invalid approval payload: %s", exc)
        return {"status": "awaiting_approval", "error": str(exc)}
    return {
        "approved": approved,
        "status": "approved_pending_write" if approved else "rejected",
        "error": None,
    }


def persist_node(
    state: MarketIntelligenceState,
    config: RunnableConfig | None = None,
    writer: ProfileWriter | None = None,
) -> dict[str, Any]:
    """Persist an approved run, then mark it complete.

    The status only becomes ``complete`` after the injected writer returns. A
    writer failure is captured in state (``persist_failed``) so the UI can show
    that nothing was saved instead of reporting success.
    """
    writer = writer or NullProfileWriter()
    run_id = (config or {}).get("configurable", {}).get("thread_id")
    if not run_id:
        return {
            "status": "persist_failed",
            "error": "Missing thread_id; the approved run was not persisted.",
        }

    try:
        persisted_run_id = writer.persist(run_id, state)
    except Exception as exc:  # surface any writer failure as a failed run
        logger.exception("Persistence failed for run %s", run_id)
        return {
            "status": "persist_failed",
            "error": f"{type(exc).__name__}: {exc}",
        }

    return {
        "status": "complete",
        "persisted_run_id": persisted_run_id or run_id,
        "error": None,
    }
