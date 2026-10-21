"""Agent tests: structured output plus application-owned provenance."""
from market_intelligence.agents.competitor import FetchedPage, analyze_competitor
from market_intelligence.agents.news import summarize_article
from market_intelligence.agents.schemas import (
    CompetitorProfileDraft,
    NewsDigest,
    NewsSummary,
    NewsToneDraft,
    ToneAspect,
    TopicAnalysis,
    TopicEvidenceQuote,
    TopicFinding,
)
from market_intelligence.agents.tone import analyze_news_tone
from market_intelligence.agents.topics import identify_topics


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
