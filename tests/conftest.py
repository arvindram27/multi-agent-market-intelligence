"""Shared pytest fixtures for the market intelligence test suite."""
import pytest

from market_intelligence.agents.schemas import (
    CompetitorProfile,
    CompetitorSource,
    NewsSummary,
    NewsTone,
    ToneAspect,
    TopicEvidence,
    TopicReport,
)
from market_intelligence.graph.state import MarketIntelligenceState

RETRIEVED_AT = "2026-10-21T12:00:00.123456+00:00"
PUBLISHED_AT = "2026-10-20T10:00:00+00:00"


class FakeStructuredLLM:
    """Fake LLM that returns a fixed Pydantic instance per output schema.

    Pass ``responses`` as a mapping of schema class -> instance. Each
    ``with_structured_output(schema)`` call records the prompt and returns the
    matching instance, so the whole suite runs offline with no API key.
    """

    def __init__(self, responses: dict | None = None):
        self.responses = responses or {}
        self.calls: list[str] = []

    def with_structured_output(self, schema, **kwargs):
        return _SchemaRunnable(self.responses[schema], self.calls)


class _SchemaRunnable:
    def __init__(self, response, calls):
        self._response = response
        self._calls = calls

    def invoke(self, prompt, **kwargs):
        self._calls.append(prompt)
        return self._response


@pytest.fixture
def fake_llm():
    """Return the FakeStructuredLLM class (instantiate with a responses dict)."""
    return FakeStructuredLLM


def build_sample_state(*, approved: bool | None = True, summary: str = "s1"):
    """A fully populated approved-run state, including one unresolved citation.

    The unresolved citation mirrors what the topic agent really produces: an
    article id the model invented, kept with ``verified=False``.
    """
    article = NewsSummary(
        headline="h",
        summary=summary,
        key_facts=["fact one"],
        tone_hint="neutral",
        article_id="a1",
        feed_url="https://example.com/feed.xml",
        source_title="Source one",
        source_url="https://example.com/one",
        published_at=PUBLISHED_AT,
        retrieved_at=RETRIEVED_AT,
        content_hash="a" * 64,
        excerpt="content one is a sufficiently long RSS excerpt",
    )
    tone = NewsTone(
        overall="positive",
        score=0.5,
        aspects=[ToneAspect(aspect="growth", tone="positive", score=0.25)],
        article_id="a1",
        headline="h",
        source_url="https://example.com/one",
        published_at=PUBLISHED_AT,
        retrieved_at=RETRIEVED_AT,
    )
    verified = TopicEvidence(
        article_id="a1",
        quote="content one",
        source_title="Source one",
        source_url="https://example.com/one",
        published_at=PUBLISHED_AT,
        verified=True,
    )
    unresolved = TopicEvidence(
        article_id="invented-article-id",
        quote="a quote the model made up",
        source_title="Unknown source",
        verified=False,
    )
    topic = TopicReport(
        topic_name="AI",
        keywords=["ai", "widgets"],
        model_confidence=0.8,
        evidence=[verified, unresolved],
        window_start=PUBLISHED_AT,
        window_end=PUBLISHED_AT,
        source_count=1,
    )
    competitor = CompetitorProfile(
        company="Acme",
        positioning="leader",
        products=["widget"],
        pricing_model="subscription",
        funding_stage="seed",
        strengths=["strong brand"],
        weaknesses=["small team"],
        sources=["https://acme.com"],
        retrieved_sources=[
            CompetitorSource(
                url="https://acme.com",
                retrieved_at="2026-10-21T12:05:00+00:00",
                content_hash="b" * 64,
            )
        ],
    )
    return MarketIntelligenceState(
        rss_url="https://example.com/feed.xml",
        competitor_url="https://acme.com",
        num_articles=1,
        summaries=[article],
        topics=[topic],
        tones=[tone],
        competitor=competitor,
        warnings=["one warning"],
        approved=approved,
    )


@pytest.fixture
def sample_state():
    """Return the sample-state builder so each test gets a fresh model."""
    return build_sample_state
