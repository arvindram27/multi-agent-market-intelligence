"""Offline collection core: fetch each configured feed, parse, compute identity.

No database, no LLM, no CLI and no transactions: :func:`collect` performs
every fetch first and returns purely in-memory results, so the storage layer
(I1b) can write everything in one transaction afterwards. Fetches go through
the same bounded HTTP client as live mode, and per-entry parsing is identical
to ``agents.news``, so live and store provenance match byte for byte.

Disabled sources are never fetched: they produce a ``disabled`` per-source
result so the per-sweep snapshot can record the whole config.
"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

import feedparser

from market_intelligence.config import settings
from market_intelligence.feed_parsing import (
    FEED_CONTENT_TYPES,
    article_id,
    clean_excerpt,
    entry_key,
    entry_key_hash,
    published_at,
    version_hash,
)
from market_intelligence.ingestion.sources import FeedSource
from market_intelligence.net import (
    FetchResult,
    PermanentFetchError,
    TransientFetchError,
    fetch_bytes,
)


class MalformedFeedError(Exception):
    """The bytes a source returned are not a parseable feed."""


class EmptyFeedError(Exception):
    """A feed parsed successfully but contained no entries."""


@dataclass(frozen=True)
class CollectedArticle:
    """One parsed entry plus every field the storage layer (I1b) needs.

    ``feed_url`` is the final URL after redirects (``FetchResult.url``), the
    same value live mode records, so store-vs-live provenance parity holds by
    construction. ``seen_at`` is the observation time of this source's bytes,
    and ``retrieved_at_raw`` the first-ever observation of this version.
    """

    source_key: str
    title: str
    excerpt: str
    feed_url: str
    source_url: str | None
    content_kind: str
    published_at_raw: str | None
    retrieved_at_raw: str
    seen_at: str
    content_hash: str
    article_id: str
    entry_key: str
    entry_key_hash: str
    version_hash: str


@dataclass(frozen=True)
class SourceResult:
    """Per-source outcome plus exactly the config snapshot this sweep used."""

    source_key: str
    kind: str
    url: str
    label: str
    enabled: bool
    status: str  # "ok" | "failed" | "disabled"
    articles: tuple[CollectedArticle, ...] = ()
    warnings: tuple[str, ...] = ()
    error_class: str | None = None
    articles_seen: int = 0

    @property
    def articles_usable(self) -> int:
        return len(self.articles)


@dataclass(frozen=True)
class SweepResult:
    """Everything one sweep observed, in memory, before any storage call."""

    source_results: tuple[SourceResult, ...]
    status: str  # "complete" | "partial" | "failed"

    @property
    def sources_attempted(self) -> int:
        return sum(1 for result in self.source_results if result.status != "disabled")

    @property
    def sources_succeeded(self) -> int:
        return sum(1 for result in self.source_results if result.status == "ok")

    @property
    def sources_failed(self) -> int:
        return sum(1 for result in self.source_results if result.status == "failed")

    @property
    def articles_seen(self) -> int:
        return sum(result.articles_seen for result in self.source_results)

    @property
    def articles(self) -> tuple[CollectedArticle, ...]:
        return tuple(
            article for result in self.source_results for article in result.articles
        )


def collect(sources: Sequence[FeedSource], *, fetcher=None) -> SweepResult:
    """Collect every source in order. Never stores anything, never call an LLM.

    Each source is attempted once; a failure is recorded per source and the
    sweep continues with the remaining sources. Unexpected exceptions (bugs)
    propagate instead of being swallowed into results.
    """
    bounded_fetch = fetcher or fetch_bytes
    results = tuple(_collect_source(source, bounded_fetch) for source in sources)

    attempted = [result for result in results if result.status != "disabled"]
    if not any(result.articles for result in attempted):
        status = "failed"
    elif any(result.status == "failed" for result in attempted):
        status = "partial"
    else:
        status = "complete"
    return SweepResult(source_results=results, status=status)


def _collect_source(source: FeedSource, fetcher) -> SourceResult:
    if not source.enabled:
        return SourceResult(
            source_key=source.source_key,
            kind=source.kind,
            url=source.url,
            label=source.label,
            enabled=False,
            status="disabled",
        )

    warnings: list[str] = []
    try:
        result: FetchResult = fetcher(
            source.url,
            timeout=settings.news_request_timeout,
            max_bytes=settings.max_feed_bytes,
            allowed_content_types=FEED_CONTENT_TYPES,
            max_redirects=settings.max_redirects,
        )
        feed = feedparser.parse(result.content)
        # Observation time: captured once per source right after the bytes
        # were received and parsed, the same point live mode records as
        # retrieved_at. No database clock is involved.
        seen_at = datetime.now(timezone.utc).isoformat()
        if feed.bozo and not feed.entries:
            detail = str(getattr(feed, "bozo_exception", "parse error"))
            raise MalformedFeedError(f"Could not parse feed: {detail}")
        if not feed.entries:
            raise EmptyFeedError("Feed contained no entries.")
        if feed.bozo:
            warnings.append(
                "Feed reported a parse warning; processing the entries it returned."
            )
    except (TransientFetchError, PermanentFetchError, MalformedFeedError, EmptyFeedError) as exc:
        return SourceResult(
            source_key=source.source_key,
            kind=source.kind,
            url=source.url,
            label=source.label,
            enabled=True,
            status="failed",
            error_class=type(exc).__name__,
        )

    total_entries = len(feed.entries)
    cap = settings.ingest_max_entries_per_source
    if total_entries > cap:
        warnings.append(f"Feed returned {total_entries} entries; capped at {cap}.")

    articles: list[CollectedArticle] = []
    for index, entry in enumerate(feed.entries[:cap], start=1):
        # Parity with agents.news: the same untitled-entry fallback, so the
        # same feed bytes produce the same identity on both paths.
        source_title = (getattr(entry, "title", "") or "").strip() or f"Untitled article {index}"
        excerpt = clean_excerpt(
            getattr(entry, "summary", "") or getattr(entry, "description", "") or ""
        )
        if len(excerpt) < settings.min_article_chars:
            warnings.append(
                f"Skipped article {index} ({source_title!r}): no usable RSS excerpt."
            )
            continue

        source_url = (getattr(entry, "link", "") or "").strip() or None
        published_at_raw = published_at(entry)
        excerpt_hash = hashlib.sha256(excerpt.encode("utf-8")).hexdigest()
        key, key_warning = entry_key(entry, title=source_title)
        if key_warning:
            warnings.append(key_warning)

        articles.append(
            CollectedArticle(
                source_key=source.source_key,
                title=source_title,
                excerpt=excerpt,
                feed_url=result.url,
                source_url=source_url,
                content_kind="rss_excerpt",
                published_at_raw=published_at_raw,
                retrieved_at_raw=seen_at,
                seen_at=seen_at,
                content_hash=excerpt_hash,
                article_id=article_id(
                    source_url, result.url, published_at_raw, excerpt_hash
                ),
                entry_key=key,
                entry_key_hash=entry_key_hash(key),
                version_hash=version_hash(
                    title=source_title,
                    excerpt=excerpt,
                    source_url=source_url,
                    published_at_raw=published_at_raw,
                    feed_url=result.url,
                ),
            )
        )

    return SourceResult(
        source_key=source.source_key,
        kind=source.kind,
        url=source.url,
        label=source.label,
        enabled=True,
        status="ok",
        articles=tuple(articles),
        warnings=tuple(warnings),
        articles_seen=min(total_entries, cap),
    )
