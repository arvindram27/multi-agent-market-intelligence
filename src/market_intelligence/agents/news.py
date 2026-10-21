"""News summarization agent: fetch an RSS feed and summarize excerpts with an LLM."""
from __future__ import annotations

import hashlib
import html
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

import feedparser

from market_intelligence.agents.llm import get_llm
from market_intelligence.agents.schemas import NewsDigest, NewsSummary
from market_intelligence.config import settings
from market_intelligence.net import FetchResult, fetch_bytes

FEED_CONTENT_TYPES = frozenset(
    {
        "application/rss+xml",
        "application/atom+xml",
        "application/rdf+xml",
        "application/xml",
        "text/xml",
    }
)
_HTML_TAG_RE = re.compile(r"<[^<]+?>")


class IngestionError(Exception):
    """The feed could not produce usable articles."""


class MalformedFeedError(IngestionError):
    """The bytes returned by the server are not a parseable feed."""


class EmptyFeedError(IngestionError):
    """The feed parsed but contained no usable article excerpts."""


@dataclass
class NewsIngestion:
    """Successful summaries plus explicit, user-visible ingestion warnings."""

    summaries: list[NewsSummary]
    warnings: list[str] = field(default_factory=list)


def summarize_article(title: str, content: str, llm=None) -> NewsDigest:
    """Summarize one RSS excerpt into a structured :class:`NewsDigest`."""
    llm = llm or get_llm()
    structured = llm.with_structured_output(NewsDigest)
    prompt = (
        "Summarize the following RSS excerpt into a concise, factual digest. "
        "The text is an excerpt, not the full article, so do not invent details "
        "that are not present. Return a tone hint for the excerpt itself; this is "
        "not a measure of consumer sentiment.\n\n"
        f"Title: {title}\n\nRSS excerpt:\n{content}"
    )
    return structured.invoke(prompt)


def checked_num_articles(num_articles: int | None) -> int:
    """Return a bounded article count or raise for an invalid request."""
    if num_articles is None:
        num_articles = settings.num_articles
    if isinstance(num_articles, bool) or not isinstance(num_articles, int):
        raise ValueError("num_articles must be an integer.")
    if not 1 <= num_articles <= settings.max_articles:
        raise ValueError(
            f"num_articles must be between 1 and {settings.max_articles}."
        )
    return num_articles


def _clean_excerpt(value: str) -> str:
    text = _HTML_TAG_RE.sub(" ", value or "")
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def _entry_timestamp(entry, key: str) -> str | None:
    parsed = getattr(entry, key, None)
    if not parsed:
        return None
    try:
        return datetime(*parsed[:6], tzinfo=timezone.utc).isoformat()
    except (TypeError, ValueError):
        return None


def _published_at(entry) -> str | None:
    return _entry_timestamp(entry, "published_parsed") or _entry_timestamp(
        entry, "updated_parsed"
    )


def _article_id(source_url: str | None, feed_url: str, published_at: str | None, content_hash: str) -> str:
    key = f"{source_url or feed_url}|{published_at or ''}|{content_hash}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def ingest_news(
    rss_url: str,
    num_articles: int | None = None,
    llm=None,
    fetcher=None,
) -> NewsIngestion:
    """Fetch, validate and summarize a feed with provenance attached.

    The feed is fetched as bounded bytes (timeout, size, content type and SSRF
    checks) and parsed from memory. Each summary records the feed URL, article
    URL (when the feed supplies one), source title, publication time, retrieval
    time, excerpt hash and the cleaned RSS excerpt.
    """
    count = checked_num_articles(num_articles)
    fetcher = fetcher or fetch_bytes
    result: FetchResult = fetcher(
        rss_url,
        timeout=settings.news_request_timeout,
        max_bytes=settings.max_feed_bytes,
        allowed_content_types=FEED_CONTENT_TYPES,
        max_redirects=settings.max_redirects,
    )

    feed = feedparser.parse(result.content)
    retrieved_at = datetime.now(timezone.utc).isoformat()
    warnings: list[str] = []

    if feed.bozo and not feed.entries:
        detail = str(getattr(feed, "bozo_exception", "parse error"))
        raise MalformedFeedError(f"Could not parse RSS feed: {detail}")
    if not feed.entries:
        raise EmptyFeedError("RSS feed contained no entries.")
    if feed.bozo:
        warnings.append(
            "Feed reported a parse warning; processing the entries it returned."
        )

    summaries: list[NewsSummary] = []
    for index, entry in enumerate(feed.entries[:count], start=1):
        source_title = (getattr(entry, "title", "") or "").strip() or f"Untitled article {index}"
        excerpt = _clean_excerpt(
            getattr(entry, "summary", "") or getattr(entry, "description", "") or ""
        )
        if len(excerpt) < settings.min_article_chars:
            warnings.append(
                f"Skipped article {index} ({source_title!r}): no usable RSS excerpt."
            )
            continue

        source_url = (getattr(entry, "link", "") or "").strip() or None
        published_at = _published_at(entry)
        content_hash = hashlib.sha256(excerpt.encode("utf-8")).hexdigest()
        digest = summarize_article(source_title, excerpt, llm=llm)
        summaries.append(
            NewsSummary(
                **digest.model_dump(),
                article_id=_article_id(
                    source_url, result.url, published_at, content_hash
                ),
                feed_url=result.url,
                source_title=source_title,
                source_url=source_url,
                published_at=published_at,
                retrieved_at=retrieved_at,
                content_hash=content_hash,
                excerpt=excerpt,
            )
        )

    if not summaries:
        raise EmptyFeedError(
            "RSS feed had no usable article excerpts; nothing was sent to the model."
        )

    return NewsIngestion(summaries=summaries, warnings=warnings)


def fetch_and_summarize(
    rss_url: str, num_articles: int | None = None, llm=None
) -> list[NewsSummary]:
    """Backward-compatible wrapper returning only the summaries."""
    return ingest_news(rss_url, num_articles=num_articles, llm=llm).summaries
