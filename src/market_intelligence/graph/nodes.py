"""Graph nodes. Each node wraps an existing agent function and returns a state update."""
from typing import Any

from langgraph.types import interrupt

from market_intelligence.agents.competitor import fetch_and_analyze
from market_intelligence.agents.news import fetch_and_summarize
from market_intelligence.agents.sentiment import analyze_sentiment
from market_intelligence.agents.trend import identify_trends
from market_intelligence.graph.state import MarketIntelligenceState


def fetch_news_node(state: MarketIntelligenceState, llm=None) -> dict[str, Any]:
    """Fetch the RSS feed and summarize each article."""
    return {
        "summaries": fetch_and_summarize(
            state.rss_url, num_articles=state.num_articles, llm=llm
        )
    }


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

    The graph interrupts here; resuming with ``Command(resume={"approved": True})``
    continues into ``persist``. On rejection the run stops without persisting.
    """
    decision = interrupt(
        {
            "type": "approval_request",
            "competitor": state.competitor.model_dump() if state.competitor else None,
        }
    )
    if isinstance(decision, dict):
        approved = bool(decision.get("approved"))
    else:
        approved = bool(decision)
    return {"approved": approved, "status": "complete" if approved else "rejected"}


def persist_node(state: MarketIntelligenceState) -> dict[str, Any]:
    """Extension point for Phase 3 persistence (Oracle). Approved runs land here."""
    return {}
