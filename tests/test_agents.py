"""Agent tests: structured output plus application-owned provenance."""
import pytest

from market_intelligence.agents.competitor import FetchedPage, analyze_competitor
from market_intelligence.agents.news import summarize_article
from market_intelligence.agents.schemas import (
    CompetitorProfileDraft,
    NewsDigest,
    NewsSummary,
    NewsTone,
    NewsToneDraft,
    ToneAspect,
    TopicAnalysis,
    TopicEvidenceQuote,
    TopicFinding,
)
from market_intelligence.agents.tone import (
    ToneScoreContractError,
    analyze_news_tone,
)
from market_intelligence.agents.topics import identify_topics
from market_intelligence.retry import is_transient_error


def _summary(**overrides):
    values = dict(
        headline="h",
        summary="s",
        key_facts=["k"],
        tone_hint="neutral",
        article_id="article-1",
        feed_url="https://example.com/feed.xml",
        source_title="Source title",
        source_url="https://example.com/article",
        published_at="2026-10-21T08:00:00+00:00",
        retrieved_at="2026-10-21T09:00:00+00:00",
        content_hash="a" * 64,
        excerpt="Source text with enough detail for evidence.",
    )
    values.update(overrides)
    return NewsSummary(**values)


def test_summarize_article_returns_digest(fake_llm):
    expected = NewsDigest(
        headline="h", summary="s", key_facts=["k"], tone_hint="neutral"
    )
    llm = fake_llm({NewsDigest: expected})
    result = summarize_article("title", "content", llm=llm)
    assert result == expected
    assert isinstance(result, NewsDigest)


def test_identify_topics_verifies_evidence(fake_llm):
    finding = TopicFinding(
        topic_name="AI",
        keywords=["ml"],
        model_confidence=0.9,
        evidence=[
            TopicEvidenceQuote(
                article_id="article-1", quote="Source text with enough detail"
            )
        ],
    )
    llm = fake_llm({TopicAnalysis: TopicAnalysis(topics=[finding])})

    extraction = identify_topics([_summary()], llm=llm)

    assert extraction.warnings == []
    topic = extraction.topics[0]
    assert topic.topic_name == "AI"
    assert topic.model_confidence == 0.9
    assert topic.evidence[0].verified is True
    assert topic.evidence[0].source_url == "https://example.com/article"
    assert topic.source_count == 1


def test_identify_topics_marks_invalid_id_and_bad_quote(fake_llm):
    finding = TopicFinding(
        topic_name="AI",
        model_confidence=0.5,
        evidence=[
            TopicEvidenceQuote(article_id="missing", quote="anything"),
            TopicEvidenceQuote(article_id="article-1", quote="not in the source"),
        ],
    )
    llm = fake_llm({TopicAnalysis: TopicAnalysis(topics=[finding])})

    extraction = identify_topics([_summary()], llm=llm)

    assert all(not item.verified for item in extraction.topics[0].evidence)
    assert any("not in this run" in warning for warning in extraction.warnings)
    assert any("not found verbatim" in warning for warning in extraction.warnings)


def test_identify_topics_warns_when_no_evidence(fake_llm):
    finding = TopicFinding(topic_name="AI", model_confidence=0.5, evidence=[])
    llm = fake_llm({TopicAnalysis: TopicAnalysis(topics=[finding])})

    extraction = identify_topics([_summary()], llm=llm)

    assert any("no cited evidence" in warning for warning in extraction.warnings)


def test_analyze_news_tone_links_to_source(fake_llm):
    draft = NewsToneDraft(
        overall="positive",
        score=0.8,
        aspects=[ToneAspect(aspect="quality", tone="positive", score=0.7)],
    )
    llm = fake_llm({NewsToneDraft: draft})
    summary = _summary()

    result = analyze_news_tone(summary, llm=llm)

    assert result.overall == "positive"
    assert result.article_id == summary.article_id
    assert result.headline == summary.headline
    assert result.source_url == summary.source_url
    assert result.retrieved_at == summary.retrieved_at


def test_analyze_competitor_attaches_retrieved_source(fake_llm):
    draft = CompetitorProfileDraft(
        company="Acme", positioning="leader", products=["widget"]
    )
    llm = fake_llm({CompetitorProfileDraft: draft})
    page = FetchedPage(
        url="https://acme.com/final",
        text="Acme is a leading widget maker.",
        retrieved_at="2026-10-21T09:00:00+00:00",
        content_hash="c" * 64,
    )

    result = analyze_competitor(page, llm=llm)

    assert result.company == "Acme"
    assert result.sources == ["https://acme.com/final"]
    assert result.retrieved_sources[0].url == "https://acme.com/final"
    assert result.retrieved_sources[0].content_hash == "c" * 64


def test_summarize_prompt_labels_rss_excerpt(fake_llm):
    expected = NewsDigest(headline="h", summary="s", tone_hint="neutral")
    llm = fake_llm({NewsDigest: expected})
    summarize_article("Breaking: Market Rally", "Stocks surged today.", llm=llm)
    prompt = llm.calls[-1]
    assert "Breaking: Market Rally" in prompt
    assert "Stocks surged today." in prompt
    assert "RSS excerpt" in prompt


def test_identify_topics_prompt_contains_source_ids_and_excerpt(fake_llm):
    finding = TopicFinding(topic_name="AI", model_confidence=0.9)
    llm = fake_llm({TopicAnalysis: TopicAnalysis(topics=[finding])})
    identify_topics([_summary()], llm=llm)
    prompt = llm.calls[-1]
    assert "article_id=article-1" in prompt
    assert "Source text with enough detail" in prompt
    assert "not a trend" in prompt


def test_analyze_news_tone_prompt_contains_summary(fake_llm):
    draft = NewsToneDraft(overall="positive", score=0.8)
    llm = fake_llm({NewsToneDraft: draft})
    analyze_news_tone(_summary(summary="customers love the new feature"), llm=llm)
    assert "customers love the new feature" in llm.calls[-1]


def test_analyze_competitor_prompt_contains_page(fake_llm):
    draft = CompetitorProfileDraft(company="Acme", positioning="leader")
    llm = fake_llm({CompetitorProfileDraft: draft})
    page = FetchedPage(
        url="https://acme.com",
        text="Acme is a leading widget maker.",
        retrieved_at="2026-10-21T09:00:00+00:00",
        content_hash="d" * 64,
    )
    analyze_competitor(page, llm=llm)
    prompt = llm.calls[-1]
    assert "https://acme.com" in prompt
    assert "Acme is a leading widget maker." in prompt


# --------------------------------------------------------------------------
# Tone score contract (signed polarity)
# --------------------------------------------------------------------------
_SIGNED_POLARITY_SENTENCE = (
    "score is signed polarity from -1 (very negative) to +1 (very positive): "
    "positive tone requires score > 0, negative tone requires score < 0, and "
    "neutral or mixed may take any value in range."
)


def test_tone_rejects_negative_label_with_positive_score(fake_llm):
    draft = NewsToneDraft(overall="negative", score=0.8)
    llm = fake_llm({NewsToneDraft: draft})

    with pytest.raises(ToneScoreContractError) as excinfo:
        analyze_news_tone(_summary(summary="model-written body text"), llm=llm)

    message = str(excinfo.value)
    assert "article-1" in message
    assert "negative" in message
    assert "score < 0" in message
    assert "0.8" not in message
    assert "model-written body text" not in message


def test_tone_rejects_negative_label_with_positive_score_085(fake_llm):
    draft = NewsToneDraft(overall="negative", score=0.85)
    llm = fake_llm({NewsToneDraft: draft})

    with pytest.raises(ToneScoreContractError) as excinfo:
        analyze_news_tone(_summary(), llm=llm)

    message = str(excinfo.value)
    assert "article-1" in message
    assert "negative" in message
    assert "score < 0" in message
    assert "0.85" not in message


def test_tone_accepts_consistent_negative(fake_llm):
    draft = NewsToneDraft(overall="negative", score=-0.6)
    llm = fake_llm({NewsToneDraft: draft})

    result = analyze_news_tone(_summary(), llm=llm)

    assert isinstance(result, NewsTone)
    assert result.overall == "negative"
    assert result.score == pytest.approx(-0.6)
    assert result.article_id == "article-1"


def test_tone_accepts_consistent_positive(fake_llm):
    draft = NewsToneDraft(overall="positive", score=0.5)
    llm = fake_llm({NewsToneDraft: draft})

    result = analyze_news_tone(_summary(), llm=llm)

    assert isinstance(result, NewsTone)
    assert result.overall == "positive"
    assert result.score == pytest.approx(0.5)


def test_tone_boundary_zero_rejected_for_positive(fake_llm):
    draft = NewsToneDraft(overall="positive", score=0.0)
    llm = fake_llm({NewsToneDraft: draft})

    with pytest.raises(ToneScoreContractError) as excinfo:
        analyze_news_tone(_summary(), llm=llm)

    message = str(excinfo.value)
    assert "overall='positive' requires score > 0" in message
    assert "0.0" not in message


def test_tone_boundary_zero_rejected_for_negative(fake_llm):
    draft = NewsToneDraft(overall="negative", score=0.0)
    llm = fake_llm({NewsToneDraft: draft})

    with pytest.raises(ToneScoreContractError) as excinfo:
        analyze_news_tone(_summary(), llm=llm)

    assert "overall='negative' requires score < 0" in str(excinfo.value)


def test_tone_neutral_allows_small_positive_score(fake_llm):
    draft = NewsToneDraft(overall="neutral", score=0.05)
    llm = fake_llm({NewsToneDraft: draft})

    result = analyze_news_tone(_summary(), llm=llm)

    assert result.overall == "neutral"
    assert result.score == pytest.approx(0.05)


def test_tone_neutral_allows_large_score(fake_llm):
    # Known limit: neutral has no sign requirement, so a large score passes.
    draft = NewsToneDraft(overall="neutral", score=0.9)
    llm = fake_llm({NewsToneDraft: draft})

    result = analyze_news_tone(_summary(), llm=llm)

    assert result.overall == "neutral"
    assert result.score == pytest.approx(0.9)


def test_tone_mixed_allows_either_sign(fake_llm):
    for score in (-0.5, 0.5):
        draft = NewsToneDraft(overall="mixed", score=score)
        llm = fake_llm({NewsToneDraft: draft})

        result = analyze_news_tone(_summary(), llm=llm)

        assert result.overall == "mixed"
        assert result.score == pytest.approx(score)


def test_tone_rejects_contradictory_aspect(fake_llm):
    draft = NewsToneDraft(
        overall="positive",
        score=0.5,
        aspects=[ToneAspect(aspect="quality", tone="negative", score=0.4)],
    )
    llm = fake_llm({NewsToneDraft: draft})

    with pytest.raises(ToneScoreContractError) as excinfo:
        analyze_news_tone(_summary(), llm=llm)

    message = str(excinfo.value)
    assert "article-1" in message
    assert "aspect #0" in message
    assert "tone='negative' requires score < 0" in message
    assert "quality" not in message
    assert "0.4" not in message


def test_tone_accepts_consistent_aspects(fake_llm):
    draft = NewsToneDraft(
        overall="positive",
        score=0.5,
        aspects=[
            ToneAspect(aspect="growth", tone="positive", score=0.7),
            ToneAspect(aspect="risk", tone="neutral", score=0.0),
            ToneAspect(aspect="cost", tone="mixed", score=-0.4),
        ],
    )
    llm = fake_llm({NewsToneDraft: draft})

    result = analyze_news_tone(_summary(), llm=llm)

    assert [(item.tone, item.score) for item in result.aspects] == [
        ("positive", 0.7),
        ("neutral", 0.0),
        ("mixed", -0.4),
    ]


def test_tone_invalid_draft_uses_one_model_call(fake_llm):
    draft = NewsToneDraft(overall="negative", score=0.8)
    llm = fake_llm({NewsToneDraft: draft})

    with pytest.raises(ToneScoreContractError):
        analyze_news_tone(_summary(), llm=llm)

    assert len(llm.calls) == 1


def test_tone_contract_error_is_not_transient():
    assert is_transient_error(ToneScoreContractError("x")) is False


def test_tone_prompt_states_signed_polarity(fake_llm):
    draft = NewsToneDraft(overall="positive", score=0.5)
    llm = fake_llm({NewsToneDraft: draft})

    analyze_news_tone(_summary(), llm=llm)

    assert _SIGNED_POLARITY_SENTENCE in llm.calls[-1]


def test_score_fields_describe_signed_polarity():
    for model in (NewsToneDraft, ToneAspect):
        description = model.model_fields["score"].description or ""
        assert "signed polarity" in description.lower()
