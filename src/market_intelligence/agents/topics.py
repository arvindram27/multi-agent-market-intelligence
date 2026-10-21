"""Topic extraction over one RSS snapshot.

This agent deliberately does not claim to detect trends over time. It extracts
topics from the captured excerpts and resolves each cited quote back to the
source article. A time-window baseline is a later feature.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from market_intelligence.agents.llm import get_llm
from market_intelligence.agents.schemas import (
    NewsSummary,
    TopicAnalysis,
    TopicEvidence,
    TopicReport,
)

_WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class TopicExtraction:
    """Topics plus explicit warnings about missing or unverifiable evidence."""

    topics: list[TopicReport]
    warnings: list[str] = field(default_factory=list)


def _normalize(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", (text or "").casefold()).strip()


def verify_quote(quote: str, excerpt: str) -> bool:
    """Return True when ``quote`` appears verbatim (ignoring case/whitespace) in ``excerpt``."""
    quote = (quote or "").strip()
    return bool(quote) and _normalize(quote) in _normalize(excerpt or "")


def _window(summaries: list[NewsSummary]) -> tuple[str | None, str | None]:
    dates = sorted(item.published_at for item in summaries if item.published_at)
    if not dates:
        return None, None
    return dates[0], dates[-1]


def _resolve_evidence(
    finding,
    by_id: dict[str, NewsSummary],
    warnings: list[str],
) -> list[TopicEvidence]:
    evidence: list[TopicEvidence] = []
    for citation in finding.evidence:
        source = by_id.get(citation.article_id)
        if source is None:
            warnings.append(
                f"Topic {finding.topic_name!r}: cited article ID "
                f"{citation.article_id!r} is not in this run; evidence is unverified."
            )
            evidence.append(
                TopicEvidence(
                    article_id=citation.article_id,
                    quote=citation.quote,
                    source_title="Unknown source",
                    verified=False,
                )
            )
            continue

        quote = (citation.quote or "").strip()
        verified = verify_quote(quote, source.excerpt)
        if not verified:
            warnings.append(
                f"Topic {finding.topic_name!r}: quote was not found verbatim in "
                f"{source.article_id}; evidence is marked unverified."
            )
        evidence.append(
            TopicEvidence(
                article_id=source.article_id,
                quote=quote or "(missing quote)",
                source_title=source.source_title,
                source_url=source.source_url,
                published_at=source.published_at,
                verified=verified,
            )
        )

    if not evidence:
        warnings.append(
            f"Topic {finding.topic_name!r} has no cited evidence; treat it as an "
            "unsupported model suggestion."
        )
    return evidence


def identify_topics(summaries: list[NewsSummary], llm=None) -> TopicExtraction:
    """Extract topics from the summaries and verify their cited evidence."""
    llm = llm or get_llm()
    structured = llm.with_structured_output(TopicAnalysis)
    sources = "\n\n".join(
        (
            f"[article_id={item.article_id}] {item.source_title}"
            f" (published: {item.published_at or 'unknown'})\n"
            f"RSS excerpt: {item.excerpt[:1500]}"
        )
        for item in summaries
    )
    prompt = (
        "Extract up to 5 topics from this single RSS snapshot. This is topic "
        "extraction, not a trend over time. For every topic, cite at least one "
        "source using its article_id and an exact quote copied verbatim from that "
        "source's RSS excerpt. Do not invent article IDs or quotes. "
        "model_confidence is your own 0-1 rating, not a calibrated probability.\n\n"
        f"Sources:\n{sources}"
    )
    analysis = structured.invoke(prompt)

    by_id = {item.article_id: item for item in summaries}
    window_start, window_end = _window(summaries)
    warnings: list[str] = []
    topics: list[TopicReport] = []
    for finding in analysis.topics:
        topics.append(
            TopicReport(
                topic_name=finding.topic_name,
                keywords=finding.keywords,
                model_confidence=finding.model_confidence,
                evidence=_resolve_evidence(finding, by_id, warnings),
                window_start=window_start,
                window_end=window_end,
                source_count=len(summaries),
            )
        )

    if not topics:
        warnings.append("The model returned no topics with supported evidence.")
    return TopicExtraction(topics=topics, warnings=warnings)
