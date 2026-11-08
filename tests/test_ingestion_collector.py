"""Ingestion collector tests: bounded fetches, per-source failures, identity parity."""
import ast
import re
from datetime import datetime
from pathlib import Path

import pytest

from market_intelligence.agents import news
from market_intelligence.agents.schemas import NewsDigest
from market_intelligence.config import settings
from market_intelligence.feed_parsing import FEED_CONTENT_TYPES
from market_intelligence.ingestion.collector import (
    EmptyFeedError,
    MalformedFeedError,
    collect,
)
from market_intelligence.ingestion.sources import FeedSource
from market_intelligence.net import FetchResult, PermanentFetchError, TransientFetchError

LONG_TEXT = "A sufficiently long RSS excerpt about the first story for summarization."
SECOND_TEXT = "A sufficiently long RSS excerpt about the second story for summarization."
FEED_URL = "https://f.example/feed.xml"


def _item(title: str, link: str, description: str) -> str:
    return (
        f"<item><title>{title}</title><link>{link}</link>"
        f"<pubDate>Wed, 21 Oct 2026 08:00:00 GMT</pubDate>"
        f"<description>{description}</description></item>"
    )


def _rss(*items: str) -> bytes:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n<rss version="2.0"><channel>'
        "<title>Feed</title>\n" + "".join(items) + "</channel></rss>"
    ).encode("utf-8")


class FakeFetcher:
    """Mimics ``net.fetch_bytes``: a validated FetchResult or error per URL."""

    def __init__(self, results: dict[str, object]):
        self.results = results
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, url: str, **kwargs):
        self.calls.append((url, dict(kwargs)))
        outcome = self.results[url]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _result(content: bytes, url: str = FEED_URL) -> FetchResult:
    return FetchResult(
        url=url, content=content, content_type="application/rss+xml", status_code=200
    )


def _source(key: str = "one", url: str = FEED_URL, enabled: bool = True) -> FeedSource:
    return FeedSource(source_key=key, kind="rss", url=url, label=key.upper(), enabled=enabled)


def _two_story_feed() -> bytes:
    return _rss(
        _item("First story", "https://example.com/first", LONG_TEXT),
        _item("Second story", "https://example.com/second", SECOND_TEXT),
    )


def test_collect_fetches_with_the_bounded_client_arguments():
    fetcher = FakeFetcher({FEED_URL: _result(_two_story_feed())})
    sweep = collect([_source()], fetcher=fetcher)

    assert sweep.status == "complete"
    assert len(fetcher.calls) == 1
    url, kwargs = fetcher.calls[0]
    assert url == FEED_URL
    assert set(kwargs) == {"timeout", "max_bytes", "allowed_content_types", "max_redirects"}
    assert kwargs["timeout"] == settings.news_request_timeout
    assert kwargs["max_bytes"] == settings.max_feed_bytes
    assert kwargs["max_redirects"] == settings.max_redirects
    # Exactly the live-mode allowlist object, not a copy: no drift possible.
    assert kwargs["allowed_content_types"] is FEED_CONTENT_TYPES
    assert kwargs["allowed_content_types"] is news.FEED_CONTENT_TYPES


def test_transient_failure_continues_with_remaining_sources():
    other_url = "https://other.example/feed.xml"
    fetcher = FakeFetcher(
        {
            FEED_URL: TransientFetchError("read timed out"),
            other_url: _result(_two_story_feed()),
        }
    )
    sweep = collect([_source("slow"), _source("healthy", other_url)], fetcher=fetcher)

    assert [c[0] for c in fetcher.calls] == [FEED_URL, other_url]
    assert sweep.status == "partial"
    slow, healthy = sweep.source_results
    assert slow.status == "failed"
    assert slow.error_class == "TransientFetchError"
    assert slow.articles == ()
    assert healthy.status == "ok"
    assert len(healthy.articles) == 2
    assert sweep.sources_attempted == 2
    assert sweep.sources_succeeded == 1
    assert sweep.sources_failed == 1


def test_permanent_failure_continues_with_remaining_sources():
    other_url = "https://other.example/feed.xml"
    fetcher = FakeFetcher(
        {FEED_URL: PermanentFetchError("blocked"), other_url: _result(_two_story_feed())}
    )
    sweep = collect([_source("bad"), _source("good", other_url)], fetcher=fetcher)

    assert sweep.status == "partial"
    bad = sweep.source_results[0]
    assert bad.status == "failed"
    assert bad.error_class == "PermanentFetchError"
    assert len(sweep.source_results[1].articles) == 2


def test_malformed_feed_is_recorded_as_failed():
    fetcher = FakeFetcher({FEED_URL: _result(b"this is not a feed at all")})
    sweep = collect([_source()], fetcher=fetcher)
    result = sweep.source_results[0]
    assert result.status == "failed"
    assert result.error_class == "MalformedFeedError"
    assert result.articles == ()
    assert sweep.status == "failed"


def test_empty_feed_is_recorded_as_failed():
    fetcher = FakeFetcher(
        {FEED_URL: _result(b'<rss version="2.0"><channel><title>T</title></channel></rss>')}
    )
    sweep = collect([_source()], fetcher=fetcher)
    assert sweep.source_results[0].error_class == "EmptyFeedError"
    assert sweep.source_results[0].status == "failed"
    assert sweep.status == "failed"


def test_zero_usable_articles_yields_failed_sweep():
    short = _rss(
        _item("First story", "https://example.com/first", "too short"),
    )
    fetcher = FakeFetcher({FEED_URL: _result(short)})
    sweep = collect([_source()], fetcher=fetcher)
    assert sweep.source_results[0].status == "ok"
    assert sweep.source_results[0].articles_usable == 0
    assert any("no usable RSS excerpt" in w for w in sweep.source_results[0].warnings)
    assert sweep.status == "failed"
    assert sweep.articles == ()


def test_per_source_cap_limits_entries_and_warns(monkeypatch):
    monkeypatch.setattr(settings, "ingest_max_entries_per_source", 2)
    six_items = _rss(*[_item(f"S {i}", f"https://example.com/{i}", LONG_TEXT) for i in range(6)])
    fetcher = FakeFetcher({FEED_URL: _result(six_items)})
    sweep = collect([_source()], fetcher=fetcher)

    assert len(sweep.articles) == 2
    assert sweep.source_results[0].articles_seen == 2
    assert any("capped at 2" in w for w in sweep.source_results[0].warnings)


def test_short_excerpts_are_skipped_with_warnings():
    fetcher = FakeFetcher(
        {
            FEED_URL: _result(
                _rss(
                    _item("Skipped story", "https://example.com/skip", "too short"),
                    _item("First story", "https://example.com/first", LONG_TEXT),
                )
            )
        }
    )
    sweep = collect([_source()], fetcher=fetcher)
    assert len(sweep.articles) == 1
    assert sweep.articles[0].title == "First story"
    assert [w for w in sweep.source_results[0].warnings if "no usable RSS excerpt" in w]


def test_one_seen_at_per_source_captured_at_parse_time():
    third_url = "https://third.example/feed.xml"
    fetcher = FakeFetcher(
        {FEED_URL: _result(_two_story_feed()), third_url: _result(_two_story_feed(), third_url)}
    )
    sweep = collect([_source("one"), _source("three", third_url)], fetcher=fetcher)

    first_pair = sweep.source_results[0].articles
    second_pair = sweep.source_results[1].articles
    assert {a.seen_at for a in first_pair} == {first_pair[0].seen_at}
    assert {a.seen_at for a in second_pair} == {second_pair[0].seen_at}
    for a in (*first_pair, *second_pair):
        assert a.seen_at == a.retrieved_at_raw
        # ISO with UTC offset.
        assert datetime.fromisoformat(a.seen_at).tzinfo is not None


def test_disabled_sources_are_not_fetched():
    enabled_url = FEED_URL
    disabled_url = "https://disabled.example/feed.xml"
    # A fake fetcher whose mapping holds no entry for the disabled URL: any
    # attempt to fetch it would KeyError the test into an error.
    fetcher = FakeFetcher({enabled_url: _result(_two_story_feed())})
    sweep = collect(
        [_source("active", enabled_url), _source("off", disabled_url, enabled=False)],
        fetcher=fetcher,
    )

    assert [c[0] for c in fetcher.calls] == [enabled_url]
    active, off = sweep.source_results
    assert active.status == "ok"
    assert off.status == "disabled"
    assert off.error_class is None
    assert off.url == disabled_url
    assert sweep.sources_attempted == 1
    assert sweep.status == "complete"


def test_collector_output_matches_live_ingest_news(fake_llm):
    fetcher = FakeFetcher({FEED_URL: _result(_two_story_feed())})
    llm = fake_llm(
        {NewsDigest: NewsDigest(headline="h", summary="s", key_facts=[], tone_hint="neutral")}
    )
    live = news.ingest_news(FEED_URL, num_articles=2, llm=llm, fetcher=fetcher)
    sweep = collect([_source()], fetcher=fetcher)

    assert len(sweep.articles) == 2
    for article, summary in zip(sweep.articles, live.summaries):
        assert article.article_id == summary.article_id
        assert article.content_hash == summary.content_hash
        assert article.excerpt == summary.excerpt
        assert article.published_at_raw == summary.published_at
        assert article.feed_url == summary.feed_url
        assert article.source_url == summary.source_url
        assert article.title == summary.source_title
        assert article.content_kind == "rss_excerpt"
        assert re.fullmatch(r"[0-9a-f]{16}", article.article_id)
        assert len(article.content_hash) == 64


def test_entry_identity_fields_are_present_and_consistent():
    fetcher = FakeFetcher({FEED_URL: _result(_two_story_feed())})
    sweep = collect([_source()], fetcher=fetcher)
    first, second = sweep.articles
    for article in (first, second):
        assert article.entry_key.startswith("link:")
        assert re.fullmatch(r"[0-9a-f]{64}", article.entry_key_hash)
        assert re.fullmatch(r"[0-9a-f]{64}", article.version_hash)
    # Distinct entries never share identity; reordering them not relevant here
    # but the mapping entry_key -> hash must be one-to-one on this feed.
    hashed = {a.entry_key: a.entry_key_hash for a in (first, second)}
    assert len(hashed) == 2
    assert first.version_hash != second.version_hash


def test_ingestion_package_imports_no_llm_or_db_boundary():
    import market_intelligence.ingestion as ingestion

    package_dir = Path(ingestion.__file__).resolve().parent
    banned_prefixes = ("market_intelligence.agents", "market_intelligence.persistence")
    banned_tokens = ("langchain", "langgraph", "openai", "psycopg", "sqlite", "llm")

    def imported_names(path: Path) -> set[str]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        found: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                found.add(node.module or "")
        return found

    for source in sorted(package_dir.glob("*.py")):
        for module_name in imported_names(source):
            offender = (source.name, module_name)
            assert not module_name.startswith(banned_prefixes), offender
            assert not any(token in module_name for token in banned_tokens), offender


def test_collector_never_raises_for_a_missing_fake_source_mapping():
    """Sanity on the test double itself: absent fetch results surface loudly."""
    fetcher = FakeFetcher({})
    with pytest.raises(KeyError):
        collect([_source()], fetcher=fetcher)


def test_malformed_and_empty_feeds_are_distinct_exception_types():
    assert not issubclass(MalformedFeedError, EmptyFeedError)
    assert not issubclass(EmptyFeedError, MalformedFeedError)
