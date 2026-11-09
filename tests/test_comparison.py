"""Offline tests for the subject-vs-competitor comparison builder."""
import json

import pytest

from market_intelligence.agents.comparison import (
    ComparisonContractError,
    build_comparison,
)
from market_intelligence.agents.schemas import (
    ComparedFieldDraft,
    EvidenceCitationDraft,
    SideComparisonDraft,
)
from market_intelligence.config import settings
from market_intelligence.net import FetchResult


class FakeStructuredLLM:
    """Returns queued drafts per schema and records each prompt."""

    def __init__(self, responses):
        self._queues = {
            schema: list(drafts) for schema, drafts in responses.items()
        }
        self.calls: list[str] = []

    def with_structured_output(self, schema):
        return _Runnable(self._queues[schema], self.calls)


class _Runnable:
    def __init__(self, drafts, calls):
        self._drafts = drafts
        self._calls = calls

    def invoke(self, prompt):
        self._calls.append(prompt)
        return self._drafts.pop(0)


class FakeFetcher:
    """Serves a fixed visible-text page per URL and records every fetch."""

    def __init__(self, pages: dict[str, str]):
        self.pages = pages
        self.calls: list[str] = []

    def __call__(self, url, **kwargs):
        self.calls.append(url)
        text = self.pages[url]
        return FetchResult(
            url,
            f"<html><body>{text}</body></html>".encode("utf-8"),
            "text/html",
            200,
        )


def _field(value="widgets", quote="widgets", source_id="S1"):
    return ComparedFieldDraft(
        value=value,
        evidence=[EvidenceCitationDraft(source_id=source_id, quote=quote)],
    )


def _draft(value="widgets", quote="widgets", source_id="S1"):
    return SideComparisonDraft(
        product_focus=_field(value, quote, source_id),
        pricing_model=ComparedFieldDraft(),
        positioning=ComparedFieldDraft(),
    )


def _pages():
    return {
        "https://subject.example/product": "Acme makes widgets for enterprise teams.",
        "https://subject.example/pricing": "Acme charges a subscription.",
        "https://competitor.example/product": "Rival makes gadgets for consumers.",
    }


def _build(pages=None, drafts=None):
    pages = pages if pages is not None else _pages()
    subject_draft = (_draft() if drafts is None else drafts[0])
    competitor_draft = (
        _draft("gadgets", "gadgets") if drafts is None else drafts[1]
    )
    fetcher = FakeFetcher(pages)
    llm = FakeStructuredLLM(
        {SideComparisonDraft: [subject_draft, competitor_draft]}
    )
    comparison, warnings = build_comparison(
        "Acme",
        ["https://subject.example/product", "https://subject.example/pricing"],
        "Rival",
        ["https://competitor.example/product"],
        llm=llm,
        fetcher=fetcher,
    )
    return comparison, warnings, fetcher, llm


def test_fetch_and_invocation_budget():
    comparison, _, fetcher, llm = _build()
    assert fetcher.calls == [
        "https://subject.example/product",
        "https://subject.example/pricing",
        "https://competitor.example/product",
    ]
    assert len(llm.calls) == 2
    assert comparison.subject.pages[0].source_id == "S1"
    assert comparison.subject.pages[1].source_id == "S2"


def test_primaries_only_use_two_fetches():
    fetcher = FakeFetcher(_pages())
    llm = FakeStructuredLLM(
        {SideComparisonDraft: [_draft(), _draft("gadgets", "gadgets")]}
    )
    build_comparison(
        "Acme",
        ["https://subject.example/product"],
        "Rival",
        ["https://competitor.example/product"],
        llm=llm,
        fetcher=fetcher,
    )
    assert len(fetcher.calls) == 2


def test_duplicate_url_within_side_rejected_before_fetch():
    fetcher = FakeFetcher(_pages())
    llm = FakeStructuredLLM({SideComparisonDraft: [_draft(), _draft()]})
    with pytest.raises(ValueError, match="unique"):
        build_comparison(
            "Acme",
            ["https://subject.example/product", "https://subject.example/product"],
            "Rival",
            ["https://competitor.example/product"],
            llm=llm,
            fetcher=fetcher,
        )
    assert fetcher.calls == []
    assert llm.calls == []


def test_same_url_on_both_sides_is_allowed_per_side():
    fetcher = FakeFetcher(_pages())
    llm = FakeStructuredLLM(
        {SideComparisonDraft: [_draft(), _draft("widgets", "widgets")]}
    )
    comparison, _ = build_comparison(
        "Acme",
        ["https://subject.example/product"],
        "Rival",
        ["https://subject.example/product"],
        llm=llm,
        fetcher=fetcher,
    )
    assert len(fetcher.calls) == 2
    assert comparison.subject.pages[0].source_id == "S1"
    assert comparison.competitor.pages[0].source_id == "S1"


def test_page_count_limits():
    llm = FakeStructuredLLM({SideComparisonDraft: [_draft(), _draft()]})
    fetcher = FakeFetcher(_pages())
    with pytest.raises(ValueError, match="at most"):
        build_comparison(
            "Acme",
            ["https://a.example/1", "https://a.example/2", "https://a.example/3"],
            "Rival",
            ["https://competitor.example/product"],
            llm=llm,
            fetcher=fetcher,
        )
    with pytest.raises(ValueError, match="at least"):
        build_comparison(
            "Acme", [], "Rival", ["https://competitor.example/product"],
            llm=llm, fetcher=fetcher,
        )
    assert fetcher.calls == []


def test_quote_matches_after_whitespace_normalization():
    pages = {
        "https://subject.example/product": "Acme   makes\nwidgets for teams.",
        "https://subject.example/pricing": "Acme charges a subscription.",
        "https://competitor.example/product": "Rival makes gadgets.",
    }
    draft = _draft("widgets", "Acme makes widgets")
    comparison, warnings, _, _ = _build(
        pages=pages, drafts=[draft, _draft("gadgets", "gadgets")]
    )
    field = comparison.subject.product_focus
    assert field.value == "widgets"
    assert field.evidence[0].quote == "Acme makes widgets"
    assert warnings == []


def test_quote_is_case_sensitive():
    pages = {
        "https://subject.example/product": "Acme makes Widgets.",
        "https://subject.example/pricing": "Acme charges a subscription.",
        "https://competitor.example/product": "Rival makes gadgets.",
    }
    comparison, warnings, _, _ = _build(
        pages=pages,
        drafts=[_draft("widgets", "acme makes widgets"), _draft("gadgets", "gadgets")],
    )
    assert comparison.subject.product_focus.value is None
    assert any("invalid citation" in warning for warning in warnings)


def test_quote_beyond_the_cap_is_rejected():
    capped = "a" * settings.max_competitor_chars
    pages = {
        "https://subject.example/product": capped + "beyond cap marker",
        "https://subject.example/pricing": "Acme charges a subscription.",
        "https://competitor.example/product": "Rival makes gadgets.",
    }
    comparison, warnings, _, _ = _build(
        pages=pages,
        drafts=[
            _draft("marker", "beyond cap marker"),
            _draft("gadgets", "gadgets"),
        ],
    )
    assert comparison.subject.product_focus.value is None
    assert any("not supported by retrieved sources" in w for w in warnings)


def test_cross_side_quote_fails_against_the_other_side_pages():
    comparison, warnings, _, _ = _build(
        drafts=[
            _draft(),
            _draft("gadgets", "Acme makes widgets"),
        ],
    )
    # "Acme makes widgets" is a subject page quote; on the competitor side the
    # same S1 resolves to the competitor page, where the quote is absent.
    assert comparison.competitor.product_focus.value is None
    assert any(
        "competitor: product_focus" in warning for warning in warnings
    )


def test_orphan_evidence_is_discarded_with_warning():
    orphan = ComparedFieldDraft(
        value=None,
        evidence=[EvidenceCitationDraft(source_id="S1", quote="widgets")],
    )
    comparison, warnings, _, _ = _build(
        drafts=[SideComparisonDraft(
            product_focus=orphan,
            pricing_model=ComparedFieldDraft(),
            positioning=ComparedFieldDraft(),
        ), _draft("gadgets", "gadgets")]
    )
    assert comparison.subject.product_focus.value is None
    assert comparison.subject.product_focus.evidence == []
    assert any("orphan evidence" in warning for warning in warnings)


def test_invalid_citation_blanks_the_field():
    comparison, warnings, _, _ = _build(
        drafts=[_draft("widgets", "not on the page"), _draft("gadgets", "gadgets")]
    )
    assert comparison.subject.product_focus.value is None
    assert comparison.subject.product_focus.evidence == []
    assert any(
        "subject: product_focus: not supported by retrieved sources" in w
        for w in warnings
    )


class _ExplodingLLM:
    def with_structured_output(self, schema):
        return self

    def invoke(self, prompt):
        raise ValueError("truncated json")


def test_unparseable_response_fails_closed_with_typed_error():
    fetcher = FakeFetcher(_pages())
    with pytest.raises(ComparisonContractError):
        build_comparison(
            "Acme",
            ["https://subject.example/product"],
            "Rival",
            ["https://competitor.example/product"],
            llm=_ExplodingLLM(),
            fetcher=fetcher,
        )


def test_representative_response_stays_under_output_ceiling():
    try:
        import tiktoken

        encoding = tiktoken.encoding_for_model(settings.openai_model)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(
            "tiktoken encoding data not available locally; no fetch attempted "
            f"({type(exc).__name__})"
        )

    value = (
        "Our product is a clinical documentation copilot for NHS trusts, "
        "sold as per-seat SaaS with an implementation package. " * 2
    )[:160]
    quote = (
        "clinicians use the copilot to draft discharge summaries and the "
        "pricing page lists a per-seat subscription with a one-off "
        "implementation fee " * 2
    )[:160]
    assert len(value) == 160 and len(quote) == 160
    field = ComparedFieldDraft(
        value=value,
        evidence=[EvidenceCitationDraft(source_id="S1", quote=quote)],
    )
    draft = SideComparisonDraft(
        product_focus=field, pricing_model=field, positioning=field
    )
    payload = draft.model_dump_json()
    tokens = len(encoding.encode(payload))
    print(f"representative comparison tokens: {tokens} (ceiling 512)")
    assert tokens < 512
    assert 512 - tokens >= 32
    assert json.loads(payload)["product_focus"]["value"] == value
