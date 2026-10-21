"""Linear orchestration wiring for the market intelligence agents.

The LangGraph supervisor is the main entry point. This sequential function is a
small legacy baseline used by tests and the old CLI path.
"""
from market_intelligence.agents.competitor import fetch_and_analyze
from market_intelligence.agents.news import fetch_and_summarize
from market_intelligence.agents.tone import analyze_news_tone
from market_intelligence.agents.topics import identify_topics


def run_pipeline(
    rss_url: str,
    competitor_url: str,
    num_articles: int | None = None,
    llm=None,
) -> dict:
    """Run the full pipeline and return summaries, topics, tones and competitor."""
    summaries = fetch_and_summarize(rss_url, num_articles=num_articles, llm=llm)
    topic_extraction = identify_topics(summaries, llm=llm)
    tones = [analyze_news_tone(summary, llm=llm) for summary in summaries]
    competitor = fetch_and_analyze(competitor_url, llm=llm)
    return {
        "summaries": summaries,
        "topics": topic_extraction.topics,
        "tones": tones,
        "competitor": competitor,
        "warnings": topic_extraction.warnings,
    }
