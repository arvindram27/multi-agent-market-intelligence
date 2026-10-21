"""News ingestion tests: bounded fetch, provenance, parsing and bounds."""
import pytest

from market_intelligence.agents import news
from market_intelligence.agents.news import (
    EmptyFeedError,
    MalformedFeedError,
    checked_num_articles,
    ingest_news,
)
from market_intelligence.agents.schemas import NewsDigest
from market_intelligence.config import settings
from market_intelligence.net import FetchResult

VALID_FEED = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Test feed</title>
<item><title>First story</title><link>https://example.com/first</link><pubDate>Wed, 21 Oct 2026 08:00:00 GMT</pubDate><description>&lt;p&gt;A sufficiently long RSS excerpt about the first story for summarization.&lt;/p&gt;</description></item>
<item><title>Second story</title><link>https://example.com/second</link><pubDate>Thu, 22 Oct 2026 09:00:00 GMT</pubDate><description>A sufficiently long RSS excerpt about the second story for summarization.</description></item>
</channel></rss>"""

EMPTY_FEED = b"""<?xml version="1.0"?><rss version="2.0"><channel><title>Empty</title></channel></rss>"""

MALFORMED_FEED = b"not xml at all <<<"


class FakeFetcher:
    def __init__(self, content: bytes, content_type="application/rss+xml"):
        self.content = content
        self.content_type = content_type
        self.calls = []

    def __call__(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return FetchResult(url, self.content, self.content_type, 200)


def _responses():
    return {
        NewsDigest: NewsDigest(
            headline="h", summary="s", tone_hint="neutral"
        )
    }


def test_ingest_uses_the_bounded_fetcher(fake_llm):
    fetcher = FakeFetcher(VALID_FEED)
    report = ingest_news(
        "https://example.com/feed.xml",
        num_articles=2,
        llm=fake_llm(_responses()),
        fetcher=fetcher,
    )

    assert len(report.summaries) == 2
    assert report.warnings == []
    _, kwargs = fetcher.calls[0]
    assert kwargs["timeout"] == settings.news_request_timeout
    assert kwargs["max_bytes"] == settings.max_feed_bytes
    assert kwargs["max_redirects"] == settings.max_redirects
    assert "application/rss+xml" in kwargs["allowed_content_types"]


def test_ingest_attaches_source_provenance(fake_llm):
    report = ingest_news(
        "https://example.com/feed.xml",
        num_articles=1,
        llm=fake_llm(_responses()),
        fetcher=FakeFetcher(VALID_FEED),
    )
    item = report.summaries[0]

    assert item.article_id
    assert len(item.article_id) == 16
    assert item.feed_url == "https://example.com/feed.xml"
    assert item.source_url == "https://example.com/first"
    assert item.source_title == "First story"
    assert item.published_at == "2026-10-21T08:00:00+00:00"
    assert item.retrieved_at
    assert len(item.content_hash) == 64
    assert item.excerpt.startswith("A sufficiently long RSS excerpt")
    assert item.content_kind == "rss_excerpt"


def test_article_id_is_stable_for_the_same_source(fake_llm):
    first = ingest_news(
        "https://example.com/feed.xml",
        num_articles=1,
        llm=fake_llm(_responses()),
        fetcher=FakeFetcher(VALID_FEED),
    ).summaries[0]
    second = ingest_news(
        "https://example.com/feed.xml",
        num_articles=1,
        llm=fake_llm(_responses()),
        fetcher=FakeFetcher(VALID_FEED),
    ).summaries[0]
    assert first.article_id == second.article_id
    assert first.content_hash == second.content_hash


def test_missing_link_and_date_are_explicit(fake_llm):
    feed = b"""<?xml version="1.0"?><rss version="2.0"><channel>
    <item><title>No metadata</title><description>A sufficiently long RSS excerpt without link or publication date for testing.</description></item>
    </channel></rss>"""
    item = ingest_news(
        "https://example.com/feed.xml",
        num_articles=1,
        llm=fake_llm(_responses()),
        fetcher=FakeFetcher(feed),
    ).summaries[0]
    assert item.source_url is None
    assert item.published_at is None
    assert item.article_id


def test_feed_is_parsed_and_html_is_cleaned(fake_llm):
    llm = fake_llm(_responses())
    ingest_news(
        "https://example.com/feed.xml",
        num_articles=2,
        llm=llm,
        fetcher=FakeFetcher(VALID_FEED),
    )

    assert "First story" in llm.calls[0]
    assert "A sufficiently long RSS excerpt" in llm.calls[0]
    assert "<p>" not in llm.calls[0]
    assert "RSS excerpt" in llm.calls[0]


def test_empty_feed_is_explicit():
    with pytest.raises(EmptyFeedError, match="no entries"):
        ingest_news(
            "https://example.com/feed.xml",
            num_articles=1,
            llm=None,
            fetcher=FakeFetcher(EMPTY_FEED),
        )


def test_malformed_feed_is_explicit():
    with pytest.raises(MalformedFeedError, match="Could not parse"):
        ingest_news(
            "https://example.com/feed.xml",
            num_articles=1,
            llm=None,
            fetcher=FakeFetcher(MALFORMED_FEED),
        )


def test_short_excerpts_are_skipped_with_a_warning(fake_llm):
    feed = b"""<?xml version="1.0"?><rss version="2.0"><channel>
    <item><title>Usable</title><description>A sufficiently long RSS excerpt about the usable story for summarization.</description></item>
    <item><title>Empty</title><description>too short</description></item>
    </channel></rss>"""
    report = ingest_news(
        "https://example.com/feed.xml",
        num_articles=2,
        llm=fake_llm(_responses()),
        fetcher=FakeFetcher(feed),
    )

    assert len(report.summaries) == 1
    assert any("Skipped article 2" in warning for warning in report.warnings)


def test_all_unusable_articles_fail_instead_of_sending_empty_text(fake_llm):
    feed = b"""<?xml version="1.0"?><rss version="2.0"><channel>
    <item><title>Empty</title><description>too short</description></item>
    </channel></rss>"""
    llm = fake_llm(_responses())
    with pytest.raises(EmptyFeedError, match="no usable"):
        ingest_news(
            "https://example.com/feed.xml",
            num_articles=1,
            llm=llm,
            fetcher=FakeFetcher(feed),
        )
    assert llm.calls == []


@pytest.mark.parametrize("count", [0, 11, -1, True, "2"])
def test_article_count_is_bounded(count):
    with pytest.raises(ValueError):
        checked_num_articles(count)


def test_article_count_defaults_to_settings():
    assert checked_num_articles(None) == settings.num_articles


def test_fetch_and_summarize_wrapper(fake_llm, monkeypatch):
    monkeypatch.setattr(
        "market_intelligence.agents.news.fetch_bytes",
        FakeFetcher(VALID_FEED),
    )
    result = news.fetch_and_summarize(
        "https://example.com/feed.xml", num_articles=1, llm=fake_llm(_responses())
    )
    assert len(result) == 1
