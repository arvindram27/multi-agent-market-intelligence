"""Subject-vs-competitor comparison with mechanical citation validation.

The comparison is built from at most two explicitly supplied pages per side.
Nothing is crawled or discovered, and no search provider is used.
"""
from __future__ import annotations

import re

from market_intelligence.agents.competitor import fetch_page
from market_intelligence.agents.llm import get_llm
from market_intelligence.agents.schemas import (
    CompanyComparison,
    Comparison,
    ComparisonEvidence,
    ComparisonField,
    PageSource,
    SideComparisonDraft,
)
from market_intelligence.config import settings

MAX_PAGES_PER_SIDE = 2
MAX_VALUE_CHARS = 160
MAX_QUOTE_CHARS = 160
FIELD_NAMES = ("product_focus", "pricing_model", "positioning")


class ComparisonContractError(RuntimeError):
    """A comparison response was unparseable, truncated or over-limit.

    This is a fatal outcome: the run fails closed and nothing is persisted.
    No warning is recorded for a fatal failure.
    """


def _normalize(text: str) -> str:
    """Whitespace-normalize text for verbatim matching; case is significant."""
    return re.sub(r"\s+", " ", text).strip()


def _validate_pages(side: str, urls: list[str]) -> None:
    if not urls:
        raise ValueError(f"{side} requires at least one page URL")
    if len(urls) > MAX_PAGES_PER_SIDE:
        raise ValueError(
            f"{side} accepts at most {MAX_PAGES_PER_SIDE} page URLs"
        )
    cleaned = [url.strip() for url in urls]
    if any(not url for url in cleaned):
        raise ValueError(f"{side} page URLs must not be blank")
    if len(set(cleaned)) != len(cleaned):
        raise ValueError(f"{side} page URLs must be unique")


def _prompt_for_side(
    name: str,
    pages: list[tuple[str, object, str]],
) -> str:
    source_blocks = "\n\n".join(
        f"Source {source_id}: {page.url}\n{prompt_text}"
        for source_id, page, prompt_text in pages
    )
    return (
        "Compare one company's public pages for three fields: product focus, "
        "pricing model and positioning.\n"
        f"Company: {name}\n\n"
        "Use only the source IDs listed below. Populate a field only when the "
        "page text supports it; otherwise set value to null and evidence to an "
        "empty list. At most one citation per field. Values and quotes must be "
        "at most 160 characters each, and every quote must be copied exactly "
        "from the cited source (whitespace may differ).\n\n"
        f"{source_blocks}"
    )


def _validate_field(
    side: str,
    field_name: str,
    draft: object,
    page_texts: dict[str, str],
) -> tuple[ComparisonField, list[str]]:
    """Validate one field's value/citation pair.

    Non-fatal problems blank the field and append a warning naming only the
    side, field and source ID.
    """
    warnings: list[str] = []
    value = draft.value.strip() if draft.value else None
    citations = list(draft.evidence)

    if value is None:
        if citations:
            warnings.append(
                f"{side}: {field_name}: orphan evidence discarded because the "
                "value is empty"
            )
        return ComparisonField(), warnings

    if len(value) > MAX_VALUE_CHARS:
        raise ComparisonContractError(
            f"comparison value for {side}:{field_name} exceeds the limit"
        )
    if not citations:
        warnings.append(
            f"{side}: {field_name}: not supported by retrieved sources"
        )
        return ComparisonField(), warnings

    citation = citations[0]
    quote = _normalize(citation.quote)
    prompt_text = page_texts.get(citation.source_id)
    supported = (
        bool(quote)
        and len(quote) <= MAX_QUOTE_CHARS
        and prompt_text is not None
        and quote in _normalize(prompt_text)
    )
    if not supported:
        warnings.append(
            f"{side}: {field_name}: not supported by retrieved sources "
            f"(invalid citation {citation.source_id})"
        )
        return ComparisonField(), warnings

    return (
        ComparisonField(
            value=value,
            evidence=[
                ComparisonEvidence(source_id=citation.source_id, quote=quote)
            ],
        ),
        warnings,
    )


def _compare_side(
    side: str,
    name: str,
    urls: list[str],
    llm,
    fetcher,
) -> tuple[CompanyComparison, list[str]]:
    _validate_pages(side, urls)
    fetched: list[tuple[str, object, str]] = []
    for index, url in enumerate(urls, start=1):
        page = fetch_page(url, fetcher=fetcher)
        prompt_text = page.text[: settings.max_competitor_chars]
        fetched.append((f"S{index}", page, prompt_text))

    sources = [
        PageSource(
            source_id=source_id,
            url=page.url,
            retrieved_at=page.retrieved_at,
            content_hash=page.content_hash,
        )
        for source_id, page, _ in fetched
    ]
    page_texts = {
        source_id: prompt_text for source_id, _, prompt_text in fetched
    }

    structured = llm.with_structured_output(SideComparisonDraft)
    try:
        draft = structured.invoke(_prompt_for_side(name, fetched))
    except Exception as exc:
        # Fatal: parse failure, truncation or an over-limit response. It fails
        # the run closed and deliberately produces no warning.
        raise ComparisonContractError(
            f"comparison response for {side} was unparseable or over-limit"
        ) from exc

    fields: dict[str, ComparisonField] = {}
    warnings: list[str] = []
    for field_name in FIELD_NAMES:
        field, field_warnings = _validate_field(
            side,
            field_name,
            getattr(draft, field_name),
            page_texts,
        )
        fields[field_name] = field
        warnings.extend(field_warnings)

    return (
        CompanyComparison(name=name, pages=sources, **fields),
        warnings,
    )


def build_comparison(
    subject_name: str,
    subject_pages: list[str],
    competitor_name: str,
    competitor_pages: list[str],
    llm=None,
    fetcher=None,
) -> tuple[Comparison, list[str]]:
    """Build both sides and return the comparison plus non-fatal warnings.

    Exactly two structured invocations are made (one per side); each side
    performs one ``fetch_page`` call per supplied URL. Errors while fetching or
    parsing propagate and fail the run closed.
    """
    llm = llm or get_llm()
    subject, subject_warnings = _compare_side(
        "subject", subject_name, subject_pages, llm, fetcher
    )
    competitor, competitor_warnings = _compare_side(
        "competitor", competitor_name, competitor_pages, llm, fetcher
    )
    return (
        Comparison(subject=subject, competitor=competitor),
        [*subject_warnings, *competitor_warnings],
    )
