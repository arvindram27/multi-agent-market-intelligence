"""Schema validation tests: each agent's structured-output model."""
import pytest
from pydantic import ValidationError

from market_intelligence.agents.schemas import (
    CompetitorProfile,
    NewsSummary,
    SentimentResult,
    TrendAnalysis,
    TrendReport,
)


def test_news_summary_valid():
    s = NewsSummary(headline="h", summary="s", key_facts=["a"], sentiment_hint="neutral")
    assert s.headline == "h"


def test_news_summary_requires_fields():
    with pytest.raises(ValidationError):
        NewsSummary()


def test_trend_report_valid():
    t = TrendReport(trend_name="AI", keywords=["ml"], confidence=0.9, evidence=["e"])
    assert t.trend_name == "AI"


def test_trend_report_confidence_bounds():
    with pytest.raises(ValidationError):
        TrendReport(trend_name="t", confidence=1.5)


def test_sentiment_result_valid():
    s = SentimentResult(overall="mixed", score=0.2, aspects=[{"aspect": "price"}])
    assert s.overall == "mixed"


def test_sentiment_score_bounds():
    with pytest.raises(ValidationError):
        SentimentResult(overall="mixed", score=2.0)


def test_competitor_profile_valid():
    p = CompetitorProfile(company="Acme", positioning="leader", products=["x"])
    assert p.company == "Acme"


def test_trend_analysis_container():
    ta = TrendAnalysis(trends=[TrendReport(trend_name="AI", confidence=0.9)])
    assert len(ta.trends) == 1
