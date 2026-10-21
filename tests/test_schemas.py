"""Schema validation tests: each agent's structured-output model."""
import pytest
from pydantic import ValidationError

from market_intelligence.agents.schemas import (
    CompetitorProfile,
    NewsSummary,
    SentimentAspect,
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
    s = SentimentResult(
        overall="mixed",
        score=0.2,
        aspects=[SentimentAspect(aspect="price", sentiment="negative", score=-0.4)],
    )
    assert s.overall == "mixed"
    assert s.aspects[0].aspect == "price"


def test_sentiment_score_bounds():
    with pytest.raises(ValidationError):
        SentimentResult(overall="mixed", score=2.0)


def test_competitor_profile_valid():
    p = CompetitorProfile(company="Acme", positioning="leader", products=["x"])
    assert p.company == "Acme"


def test_trend_analysis_container():
    ta = TrendAnalysis(trends=[TrendReport(trend_name="AI", confidence=0.9)])
    assert len(ta.trends) == 1


def _assert_openai_strict_objects(node, path="schema"):
    """OpenAI strict structured output rejects bare objects and additionalProperties: true."""
    if isinstance(node, dict):
        if node.get("type") == "object":
            assert "properties" in node, f"object without properties at {path}"
            assert node.get("additionalProperties") is not True, (
                f"additionalProperties: true at {path}"
            )
        for key, value in node.items():
            _assert_openai_strict_objects(value, f"{path}.{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            _assert_openai_strict_objects(value, f"{path}[{i}]")


@pytest.mark.parametrize(
    "model", [NewsSummary, SentimentResult, SentimentAspect, TrendAnalysis, CompetitorProfile]
)
def test_schemas_are_openai_strict_compatible(model):
    # Regression guard: list[dict] produced {"additionalProperties": true}, which OpenAI
    # rejects with a 400 under the default json_schema structured-output method.
    _assert_openai_strict_objects(model.model_json_schema())
