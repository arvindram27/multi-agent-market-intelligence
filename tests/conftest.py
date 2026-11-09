"""Shared pytest fixtures for the market intelligence test suite."""
import pytest

from market_intelligence.agents.schemas import (
    CompanyComparison,
    Comparison,
    ComparisonEvidence,
    ComparisonField,
    NewsSummary,
    NewsTone,
    PageSource,
    ToneAspect,
    TopicEvidence,
    TopicReport,
)
from market_intelligence.graph.state import MarketIntelligenceState

RETRIEVED_AT = "2026-10-21T12:00:00.123456+00:00"
PUBLISHED_AT = "2026-10-20T10:00:00+00:00"


class FakeStructuredLLM:
    """Fake LLM that returns a fixed Pydantic instance per output schema.

    Pass ``responses`` as a mapping of schema class -> instance. A schema may
    also map to a list of instances; each ``invoke`` then returns the next one,
    which lets the two per-side comparison calls receive different drafts.
    """

    def __init__(self, responses: dict | None = None):
        self.responses = responses or {}
        self.calls: list[str] = []

    def with_structured_output(self, schema, **kwargs):
        return _SchemaRunnable(self.responses[schema], self.calls)


class _SchemaRunnable:
    def __init__(self, response, calls):
        # Share the list so repeated with_structured_output() calls consume the
        # queue in order (the two per-side comparison calls need this).
        self._queue = response if isinstance(response, list) else None
        self._response = response
        self._calls = calls

    def invoke(self, prompt, **kwargs):
        self._calls.append(prompt)
        if self._queue is not None:
            return self._queue.pop(0)
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
    comparison = Comparison(
        subject=CompanyComparison(
            name="Acme",
            pages=[
                PageSource(
                    source_id="S1",
                    url="https://acme.com",
                    retrieved_at="2026-10-21T12:05:00+00:00",
                    content_hash="b" * 64,
                )
            ],
            product_focus=ComparisonField(
                value="widget",
                evidence=[
                    ComparisonEvidence(source_id="S1", quote="widget")
                ],
            ),
            pricing_model=ComparisonField(
                value="subscription",
                evidence=[
                    ComparisonEvidence(source_id="S1", quote="subscription")
                ],
            ),
            positioning=ComparisonField(
                value="leader",
                evidence=[ComparisonEvidence(source_id="S1", quote="leader")],
            ),
        ),
        competitor=CompanyComparison(
            name="Rival",
            pages=[
                PageSource(
                    source_id="S1",
                    url="https://rival.com",
                    retrieved_at="2026-10-21T12:05:00+00:00",
                    content_hash="c" * 64,
                )
            ],
            product_focus=ComparisonField(
                value="widget",
                evidence=[
                    ComparisonEvidence(source_id="S1", quote="widget")
                ],
            ),
            pricing_model=ComparisonField(),
            positioning=ComparisonField(
                value="challenger",
                evidence=[
                    ComparisonEvidence(source_id="S1", quote="challenger")
                ],
            ),
        ),
    )
    return MarketIntelligenceState(
        rss_url="https://example.com/feed.xml",
        subject_name="Acme",
        subject_pages=["https://acme.com"],
        competitor_name="Rival",
        competitor_pages=["https://rival.com"],
        num_articles=1,
        summaries=[article],
        topics=[topic],
        tones=[tone],
        comparison=comparison,
        warnings=["one warning"],
        approved=approved,
    )


@pytest.fixture
def sample_state():
    """Return the sample-state builder so each test gets a fresh model."""
    return build_sample_state
