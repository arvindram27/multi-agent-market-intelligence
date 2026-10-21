"""Linear orchestration wiring for the market intelligence agents.

Phase 1 keeps orchestration as a plain sequential function. A LangGraph supervisor
(Phase 2) will replace this once checkpointing and human-in-the-loop are added.
"""
from market_intelligence.agents.competitor import fetch_and_analyze
from market_intelligence.agents.news import fetch_and_summarize
from market_intelligence.agents.sentiment import analyze_sentiment
from market_intelligence.agents.trend import identify_trends


def run_pipeline(
    rss_url: str,
    competitor_url: str,
    num_articles: int | None = None,
    llm=None,
) -> dict:
    """Run the full market-intelligence pipeline and return structured results."""
    summaries = fetch_and_summarize(rss_url, num_articles=num_articles, llm=llm)
    summary_texts = [s.summary for s in summaries]
    trends = identify_trends(summary_texts, llm=llm)
    sentiments = [analyze_sentiment(text, llm=llm) for text in summary_texts]
    competitor = fetch_and_analyze(competitor_url, llm=llm)
    return {
        "summaries": summaries,
        "trends": trends,
        "sentiments": sentiments,
        "competitor": competitor,
    }
