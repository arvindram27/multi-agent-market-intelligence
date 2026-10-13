"""Agent tests: each agent returns its structured-output schema via a fake LLM."""
from market_intelligence.agents.competitor import analyze_competitor
from market_intelligence.agents.news import summarize_article
from market_intelligence.agents.sentiment import analyze_sentiment
from market_intelligence.agents.trend import identify_trends
from market_intelligence.agents.schemas import (
    CompetitorProfile,
    NewsSummary,
    SentimentResult,
    TrendAnalysis,
    TrendReport,
)


def test_summarize_article_returns_schema(fake_llm):
    expected = NewsSummary(headline="h", summary="s", key_facts=["k"], sentiment_hint="neutral")
    llm = fake_llm({NewsSummary: expected})
    result = summarize_article("title", "content", llm=llm)
    assert result == expected
    assert isinstance(result, NewsSummary)


def test_identify_trends_returns_schema(fake_llm):
    expected = TrendAnalysis(trends=[TrendReport(trend_name="AI", confidence=0.9)])
    llm = fake_llm({TrendAnalysis: expected})
    result = identify_trends(["some text"], llm=llm)
    assert result == [expected.trends[0]]
    assert isinstance(result[0], TrendReport)


def test_analyze_sentiment_returns_schema(fake_llm):
    expected = SentimentResult(overall="positive", score=0.8, aspects=[{"aspect": "quality"}])
    llm = fake_llm({SentimentResult: expected})
    result = analyze_sentiment("great product", llm=llm)
    assert result == expected
    assert isinstance(result, SentimentResult)


def test_analyze_competitor_returns_schema(fake_llm):
    expected = CompetitorProfile(company="Acme", positioning="leader", products=["widget"])
    llm = fake_llm({CompetitorProfile: expected})
    result = analyze_competitor("page text", "https://acme.com", llm=llm)
    assert result == expected
    assert isinstance(result, CompetitorProfile)
