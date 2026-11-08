"""Pure feed-parsing and identity helpers shared by every ingestion path.

Imported by both :mod:`market_intelligence.agents.news` (the live LLM path)
and :mod:`market_intelligence.ingestion` (the offline collection path). This
module must never import agents, persistence, the LLM stack or the database,
so the two paths share one implementation without coupling the collector to
the graph.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
from datetime import datetime, timezone

# Content types a feed response may carry. Agents.news re-exports this under
# the same name; the collector imports it from here, never from agents.
FEED_CONTENT_TYPES = frozenset(
    {
        "application/rss+xml",
        "application/atom+xml",
        "application/rdf+xml",
        "application/xml",
        "text/xml",
    }
)

# Identity caps. An overlong identifier is hashed (over its FULL original
# value), never truncated, so distinct overlong identifiers stay distinct.
ENTRY_KEY_MAX_LENGTH = 2048

_HTML_TAG_RE = re.compile(r"<[^<]+?>")


def clean_excerpt(value: str) -> str:
    """Collapse an HTML-bearing RSS excerpt to cleaned plain text."""
    text = _HTML_TAG_RE.sub(" ", value or "")
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def entry_timestamp(entry, key: str) -> str | None:
    """ISO-8601 UTC string for a parsed feed time struct, or None."""
    parsed = getattr(entry, key, None)
    if not parsed:
        return None
    try:
        return datetime(*parsed[:6], tzinfo=timezone.utc).isoformat()
    except (TypeError, ValueError):
        return None


def published_at(entry) -> str | None:
    """First publication time, then last-update time, else ``None``."""
    return entry_timestamp(entry, "published_parsed") or entry_timestamp(
        entry, "updated_parsed"
    )


def article_id(
    source_url: str | None, feed_url: str, published_at: str | None, content_hash: str
) -> str:
    """Live-mode provenance identity, unchanged from the moved implementation."""
    key = f"{source_url or feed_url}|{published_at or ''}|{content_hash}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def entry_key(entry, *, title: str) -> tuple[str, str | None]:
    """Stable entry identity with an honest derivation prefix.

    Feed-provided GUID when the feed supplies one, else the entry link, else
    a documented fallback hash of the title. A fallback is never presented as
    a real GUID, and an overlong identifier is stored as a hash of its FULL
    value so distinct identifiers stay distinct.
    """
    guid = (getattr(entry, "id", None) or getattr(entry, "guid", None) or "").strip()
    if guid:
        return _bounded("guid", guid)
    link = (getattr(entry, "link", None) or "").strip()
    if link:
        return _bounded("link", link)
    digest = hashlib.sha256(title.encode("utf-8")).hexdigest()
    return (
        f"sha256:{digest}",
        "entry has neither GUID nor link; using the documented fallback identity",
    )


def _bounded(kind: str, value: str) -> tuple[str, str | None]:
    key = f"{kind}:{value}"
    if len(key) <= ENTRY_KEY_MAX_LENGTH:
        return key, None
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return (
        f"{kind}-sha256:{digest}",
        f"{kind} identifier exceeded {ENTRY_KEY_MAX_LENGTH} chars; "
        "stored as a hash of the full value",
    )


def entry_key_hash(key: str) -> str:
    """Fixed-length sha256 hex of the full prefixed entry key."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def version_hash(
    *,
    title: str,
    excerpt: str,
    source_url: str | None,
    published_at_raw: str | None,
    feed_url: str,
) -> str:
    """Content-version identity over the fields store mode reports.

    sha256 over the UTF-8 bytes of the canonical JSON array
    ``[title, excerpt, source_url, published_at_raw, feed_url]``. A missing
    value stays JSON ``null`` and hashes differently from ``""``; the array
    shape means separators inside field values cannot collide. Any change to
    any field is therefore a new version row under the same entry key.
    """
    fields = [title, excerpt, source_url, published_at_raw, feed_url]
    canonical = json.dumps(fields, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
