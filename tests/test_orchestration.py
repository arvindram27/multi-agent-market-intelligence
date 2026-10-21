"""Orchestrator wiring test: pipeline chains all four agents with mocked HTTP/RSS."""
from market_intelligence.net import FetchResult
from market_intelligence.orchestration import run_pipeline
from market_intelligence.agents.schemas import (
    CompetitorProfileDraft,
    NewsDigest,
    NewsToneDraft,
    TopicAnalysis,
    TopicFinding,
)

FEED_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Test feed</title>
<item><title>T1</title><link>https://example.com/article-1</link><pubDate>Wed, 21 Oct 2026 08:00:00 GMT</pubDate><description>content one is a sufficiently long RSS excerpt for summarization</description></item>
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
        NewsDigest: NewsDigest(headline="h", summary="s1", tone_hint="neutral"),
        TopicAnalysis: TopicAnalysis(
            topics=[TopicFinding(topic_name="AI", model_confidence=0.8)]
        ),
        NewsToneDraft: NewsToneDraft(overall="positive", score=0.5),
        CompetitorProfileDraft: CompetitorProfileDraft(
            company="Acme", positioning="leader"
        ),
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
    assert result["summaries"][0].source_url == "https://example.com/article-1"
    assert result["topics"][0].topic_name == "AI"
    assert result["tones"][0].overall == "positive"
    assert result["competitor"].company == "Acme"
    assert result["competitor"].retrieved_sources
    # Four LLM calls: summarize, topics, tone, competitor
    assert len(llm.calls) == 4
