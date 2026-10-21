"""Schema validation tests: LLM drafts and enriched provenance records."""
import pytest
from pydantic import ValidationError

from market_intelligence.agents.schemas import (
    CompetitorProfile,
    CompetitorProfileDraft,
    CompetitorSource,
    NewsDigest,
    NewsSummary,
    NewsTone,
    NewsToneDraft,
    ToneAspect,
    TopicAnalysis,
    TopicEvidence,
    TopicFinding,
    TopicReport,
)


def _summary(**overrides):
    values = dict(
        headline="h",
        summary="s",
        key_facts=["a"],
        tone_hint="neutral",
        article_id="abc123",
        feed_url="https://example.com/feed.xml",
        source_title="Source title",
        source_url="https://example.com/article",
        published_at="2026-10-21T08:00:00+00:00",
        retrieved_at="2026-10-21T09:00:00+00:00",
        content_hash="a" * 64,
        excerpt="A captured RSS excerpt.",
    )
    values.update(overrides)
    return NewsSummary(**values)


def test_news_digest_valid():
    digest = NewsDigest(
        headline="h", summary="s", key_facts=["a"], tone_hint="neutral"
    )
    assert digest.headline == "h"


def test_news_summary_requires_provenance():
    with pytest.raises(ValidationError):
        NewsSummary(
            headline="h", summary="s", tone_hint="neutral"
        )


def test_news_summary_allows_explicitly_missing_source_fields():
    summary = _summary(source_url=None, published_at=None)
    assert summary.source_url is None
    assert summary.published_at is None
    assert summary.content_kind == "rss_excerpt"


def test_topic_finding_and_report():
    finding = TopicFinding(
        topic_name="AI",
        keywords=["ml"],
        model_confidence=0.9,
        evidence=[],
    )
    assert finding.model_confidence == 0.9

    report = TopicReport(
        topic_name="AI",
        keywords=["ml"],
        model_confidence=0.9,
        evidence=[
            TopicEvidence(
                article_id="a",
                quote="evidence",
                source_title="Source",
                verified=True,
            )
        ],
        window_start="2026-10-20",
        window_end="2026-10-21",
        source_count=2,
    )
    assert report.evidence[0].verified is True
    assert report.source_count == 2


def test_topic_confidence_bounds():
    with pytest.raises(ValidationError):
        TopicReport(
            topic_name="t",
            model_confidence=1.5,
            source_count=1,
        )


def test_tone_draft_and_enriched_tone():
    draft = NewsToneDraft(
        overall="mixed",
        score=0.2,
        aspects=[ToneAspect(aspect="price", tone="negative", score=-0.4)],
    )
    assert draft.aspects[0].tone == "negative"

    tone = NewsTone(
        **draft.model_dump(),
        article_id="a",
        headline="h",
        retrieved_at="2026-10-21T09:00:00+00:00",
    )
    assert tone.article_id == "a"


def test_tone_score_bounds():
    with pytest.raises(ValidationError):
        NewsToneDraft(overall="mixed", score=2.0)


def test_competitor_profile_with_retrieved_source():
    source = CompetitorSource(
        url="https://acme.example/about",
        retrieved_at="2026-10-21T09:00:00+00:00",
        content_hash="b" * 64,
    )
    profile = CompetitorProfile(
        company="Acme",
        positioning="leader",
        products=["x"],
        sources=[source.url],
        retrieved_sources=[source],
    )
    assert profile.retrieved_sources[0].content_hash == "b" * 64


def test_topic_analysis_container():
    analysis = TopicAnalysis(
        topics=[TopicFinding(topic_name="AI", model_confidence=0.9)]
    )
    assert len(analysis.topics) == 1


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
    "model",
    [
        NewsDigest,
        TopicFinding,
        TopicAnalysis,
        NewsToneDraft,
        CompetitorProfileDraft,
    ],
)
def test_llm_schemas_are_openai_strict_compatible(model):
    # Regression guard: list[dict] produced {"additionalProperties": true}, which OpenAI
    # rejects with a 400 under the default json_schema structured-output method.
    _assert_openai_strict_objects(model.model_json_schema())
