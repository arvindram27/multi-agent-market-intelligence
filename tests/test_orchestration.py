"""Orchestrator wiring test: pipeline chains all four agents with mocked HTTP/RSS."""
from market_intelligence.net import FetchResult
from market_intelligence.orchestration import run_pipeline
from market_intelligence.agents.schemas import (
    CompetitorProfile,
    NewsSummary,
    SentimentResult,
    TrendAnalysis,
    TrendReport,
)

FEED_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Test feed</title>
<item><title>T1</title><description>content one is a sufficiently long RSS excerpt for summarization</description></item>
</channel></rss>"""


def test_run_pipeline_wires_agents(fake_llm, monkeypatch):
    def fake_fetch(url, **kwargs):
        if "feed" in url:
            return FetchResult(url, FEED_XML, "application/rss+xml", 200)
        return FetchResult(
            url,
            b"<html><body>Acme corp is a leading widget maker.</body></html>",
            "text/html",
            200,
        )

    monkeypatch.setattr("market_intelligence.agents.news.fetch_bytes", fake_fetch)
    monkeypatch.setattr("market_intelligence.agents.competitor.fetch_bytes", fake_fetch)

    responses = {
        NewsSummary: NewsSummary(headline="h", summary="s1", sentiment_hint="neutral"),
        TrendAnalysis: TrendAnalysis(trends=[TrendReport(trend_name="AI", confidence=0.8)]),
        SentimentResult: SentimentResult(overall="positive", score=0.5),
        CompetitorProfile: CompetitorProfile(company="Acme", positioning="leader"),
    }
    llm = fake_llm(responses)

    result = run_pipeline(
        "http://example.com/feed.xml",
        "https://acme.com",
        num_articles=1,
        llm=llm,
    )

    assert len(result["summaries"]) == 1
    assert result["summaries"][0].summary == "s1"
    assert result["trends"][0].trend_name == "AI"
    assert result["sentiments"][0].overall == "positive"
    assert result["competitor"].company == "Acme"
    # Four LLM calls: summarize, trends, sentiment, competitor
    assert len(llm.calls) == 4
