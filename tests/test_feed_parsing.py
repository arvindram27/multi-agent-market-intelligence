"""Feed-parsing helper parity, entry identity and version hashing tests."""
import hashlib
import json
import re
from types import SimpleNamespace

from market_intelligence.agents import news
from market_intelligence.feed_parsing import (
    ENTRY_KEY_MAX_LENGTH,
    FEED_CONTENT_TYPES,
    article_id,
    clean_excerpt,
    entry_key,
    entry_key_hash,
    entry_timestamp,
    published_at,
    version_hash,
)


def _entry(**kwargs) -> SimpleNamespace:
    return SimpleNamespace(**kwargs)


def _excerpt_html() -> str:
    return "<p>A sufficiently long RSS excerpt &amp; more.</p>"


def test_clean_excerpt_strips_tags_unescapes_and_collapses_whitespace():
    assert clean_excerpt(_excerpt_html()) == "A sufficiently long RSS excerpt & more."
    assert clean_excerpt(None) == ""


def test_entry_timestamp_parses_structs_and_bails_on_garbage():
    import time as time_module

    parsed = time_module.struct_time((2026, 10, 21, 8, 0, 0, 3, 294, 0))
    assert entry_timestamp(_entry(published_parsed=parsed), "published_parsed") == (
        "2026-10-21T08:00:00+00:00"
    )
    assert entry_timestamp(_entry(published_parsed="garbage"), "published_parsed") is None
    assert entry_timestamp(_entry(published_parsed=None), "published_parsed") is None


def test_published_at_prefers_published_then_updated_then_none():
    import time as time_module

    pub = time_module.struct_time((2026, 10, 21, 8, 0, 0, 3, 294, 0))
    upd = time_module.struct_time((2026, 10, 22, 9, 0, 0, 4, 295, 0))
    entry = _entry(published_parsed=pub, updated_parsed=upd)
    assert published_at(entry) == "2026-10-21T08:00:00+00:00"
    assert published_at(_entry(updated_parsed=upd)) == "2026-10-22T09:00:00+00:00"
    assert published_at(_entry()) is None


def test_article_id_matches_its_definition():
    # Independent recomputation of the documented formula; NOT a call to
    # article_id(). article_id covers link, published_at and excerpt hash,
    # and deliberately NOT the title.
    expected = hashlib.sha256(
        ("https://example.com/one|2026-10-21T08:00:00+00:00|" + "c" * 64).encode("utf-8")
    ).hexdigest()[:16]
    assert article_id("https://example.com/one", "https://f", "2026-10-21T08:00:00+00:00", "c" * 64) == expected
    # No link: the feed URL is used instead.
    assert article_id(None, "https://f", None, "c" * 64) == hashlib.sha256(
        ("https://f||" + "c" * 64).encode("utf-8")
    ).hexdigest()[:16]


def test_entry_key_prefers_feed_guid():
    entry = _entry(id="tag:example.com,2026:one", link="https://example.com/first")
    key, warning = entry_key(entry, title="Whatever")
    assert key == "guid:tag:example.com,2026:one"
    assert warning is None


def test_entry_key_uses_link_when_no_guid():
    entry = _entry(id=None, guid=None, link="https://example.com/first")
    key, warning = entry_key(entry, title="Whatever")
    assert key == "link:https://example.com/first"
    assert warning is None


def test_entry_key_falls_back_to_title_hash_with_warning():
    entry = _entry(id=None, guid=None, link=None)
    key, warning = entry_key(entry, title="Same title")
    assert key == f"sha256:{hashlib.sha256('Same title'.encode('utf-8')).hexdigest()}"
    assert warning is not None
    assert "fallback" in warning


def test_overlong_guid_stored_as_hash_of_full_value():
    overlong = "g" * (ENTRY_KEY_MAX_LENGTH + 10)
    key, warning = entry_key(_entry(id=overlong), title="T")
    assert key.startswith("guid-sha256:")
    digest = key.removeprefix("guid-sha256:")
    assert digest == hashlib.sha256(overlong.encode("utf-8")).hexdigest()
    assert warning is not None
    assert "2048" in warning


def test_overlong_link_stored_as_hash_of_full_value():
    overlong = "https://example.com/" + "x" * (ENTRY_KEY_MAX_LENGTH + 10)
    key, warning = entry_key(_entry(id=None, link=overlong), title="T")
    assert key.startswith("link-sha256:")
    assert key.removeprefix("link-sha256:") == hashlib.sha256(
        overlong.encode("utf-8")
    ).hexdigest()
    assert warning is not None
    assert "2048" in warning


def test_overlong_keys_stay_under_the_cap_with_prefix_kept():
    overlong_guid = "g" * (ENTRY_KEY_MAX_LENGTH + 10)
    overlong_link = "https://example.com/" + "x" * (ENTRY_KEY_MAX_LENGTH + 10)
    # guid wins over link for the same entry.
    guid_key, guid_warning = entry_key(_entry(id=overlong_guid, link=overlong_link), title="T")
    assert guid_key.startswith("guid-sha256:")
    assert len(guid_key) <= ENTRY_KEY_MAX_LENGTH
    assert guid_warning is not None
    # With no guid, the overlong link is hashed in its own namespace.
    link_key, link_warning = entry_key(_entry(id=None, link=overlong_link), title="T")
    assert link_key.startswith("link-sha256:")
    assert len(link_key) <= ENTRY_KEY_MAX_LENGTH
    assert link_warning is not None


def test_distinct_overlong_guids_with_same_title_get_distinct_keys():
    first, first_warning = entry_key(_entry(id="a" * (ENTRY_KEY_MAX_LENGTH + 1)), title="Same title")
    second, second_warning = entry_key(_entry(id="b" * (ENTRY_KEY_MAX_LENGTH + 1)), title="Same title")
    assert first_warning and second_warning
    assert first != second


def test_entry_key_hash_is_sha256_hex_of_full_key():
    key = "guid:tag:example.com,2026:one"
    result = entry_key_hash(key)
    expected = hashlib.sha256(key.encode("utf-8")).hexdigest()
    assert result == expected
    assert re.fullmatch(r"[0-9a-f]{64}", result)


def test_version_hash_matches_canonical_json_array():
    # Independent recomputation of the documented formula: sha256 over the
    # UTF-8 bytes of the canonical JSON array, missing values kept as null.
    fields = ["Title", "Excerpt", None, None, "https://feeds.example/final"]
    expected = hashlib.sha256(
        json.dumps(fields, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    result = version_hash(
        title="Title",
        excerpt="Excerpt",
        source_url=None,
        published_at_raw=None,
        feed_url="https://feeds.example/final",
    )
    assert result == expected
    assert re.fullmatch(r"[0-9a-f]{64}", result)
    # A change to any single field (here: the delimiter-like comma inside a
    # value) changes the array encoding itself, so it changes the digest.
    result_with_comma = version_hash(
        title="Title,\"x1f\"",
        excerpt="Excerpt",
        source_url=None,
        published_at_raw=None,
        feed_url="https://feeds.example/final",
    )
    assert result_with_comma != result


def test_version_hash_null_differs_from_empty_string():
    null_version = version_hash(
        title="T", excerpt="E", source_url=None, published_at_raw=None, feed_url="F"
    )
    empty_version = version_hash(
        title="T", excerpt="E", source_url="", published_at_raw="", feed_url="F"
    )
    assert null_version != empty_version


def test_version_hash_changes_on_title_only_edit():
    base = dict(
        title="Original headline",
        excerpt="Same excerpt",
        source_url="https://example.com/one",
        published_at_raw="2026-10-21T08:00:00+00:00",
        feed_url="https://f.example",
    )
    renamed = {**base, "title": "Renamed headline"}
    assert version_hash(**base) != version_hash(**renamed)
    # Repeated computation is deterministic.
    assert version_hash(**base) == version_hash(**base)


def test_version_hash_hashes_non_ascii_as_utf8_bytes():
    fields = ["标题", "Überschau", None, None, "https://f.example"]
    ascii_escaped = hashlib.sha256(
        json.dumps(fields, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    )
    honest = hashlib.sha256(
        json.dumps(fields, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    )
    result = version_hash(
        title="标题",
        excerpt="Überschau",
        source_url=None,
        published_at_raw=None,
        feed_url="https://f.example",
    )
    # ensure_ascii=False must win: the digest covers raw UTF-8, not escapes.
    assert result == honest.hexdigest()
    assert result != ascii_escaped.hexdigest()


def test_news_module_aliases_share_the_feed_parsing_implementations():
    assert news._clean_excerpt is clean_excerpt
    assert news._entry_timestamp is entry_timestamp
    assert news._published_at is published_at
    assert news._article_id is article_id
    assert news.FEED_CONTENT_TYPES is FEED_CONTENT_TYPES
