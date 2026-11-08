"""Ingestion source configuration tests: TOML load and structural validation."""
import pytest

from market_intelligence.ingestion.sources import (
    FeedSource,
    SourceConfigError,
    load_sources,
    validate_sources,
)

VALID_TOML = f"""
[[sources]]
key = "bbc-business"
kind = "rss"
label = "BBC Business"
url = "https://feeds.bbci.co.uk/news/business/rss.xml"
enabled = true

[[sources]]
key = "the-verge"
kind = "atom"
label = "The Verge"
url = "https://www.theverge.com/rss/index.xml"
enabled = false

[[sources]]
key = "techcrunch"
kind = "rss"
label = "TechCrunch"
url = "https://techcrunch.com/feed/"
"""

_REPOSITORY_CONFIG = "ingestion_sources.toml"


def test_missing_required_key_is_rejected(tmp_path):
    data = {"sources": [{"kind": "rss", "url": "https://x.example", "label": "X"}]}
    with pytest.raises(SourceConfigError, match="missing required keys"):
        validate_sources(data)


def test_unknown_key_rejected():
    data = {
        "sources": [
            {
                "key": "x",
                "kind": "rss",
                "label": "X",
                "url": "https://x.example/feed",
                "priority": 1,
            }
        ]
    }
    with pytest.raises(SourceConfigError, match="unknown keys \\['priority'\\]"):
        validate_sources(data)


def test_unknown_top_level_key_rejected():
    data = {"sources": [_minimal_source()], "defaults": {}}
    with pytest.raises(SourceConfigError, match="unknown top-level keys"):
        validate_sources(data)


def test_non_http_scheme_rejected():
    data = {
        "sources": [{"key": "x", "kind": "rss", "label": "X", "url": "ftp://x.example/feed"}]
    }
    with pytest.raises(SourceConfigError, match="url must be http"):
        validate_sources(data)


def test_duplicate_source_key_rejected():
    data = {
        "sources": [
            _minimal_source(),
            {"key": "x", "kind": "atom", "label": "Y", "url": "https://y.example/feed"},
        ]
    }
    with pytest.raises(SourceConfigError, match="duplicate source_key 'x'"):
        validate_sources(data)


def test_duplicate_url_rejected():
    data = {
        "sources": [
            _minimal_source(),
            {"key": "y", "kind": "rss", "label": "Y", "url": "https://x.example/feed"},
        ]
    }
    with pytest.raises(SourceConfigError, match="duplicate url"):
        validate_sources(data)


def test_kind_beyond_rss_atom_rejected():
    data = {"sources": [{"key": "x", "kind": "html", "label": "X", "url": "https://x.example"}]}
    with pytest.raises(SourceConfigError, match="kind must be one of"):
        validate_sources(data)


def test_empty_sources_list_rejected():
    with pytest.raises(SourceConfigError, match="non-empty"):
        validate_sources({"sources": []})
    with pytest.raises(SourceConfigError, match="non-empty"):
        validate_sources({})


def test_disabled_source_is_retained_with_enabled_false():
    data = {
        "sources": [
            {"key": "x", "kind": "rss", "label": "X", "url": "https://x.example/feed"},
            {
                "key": "y",
                "kind": "atom",
                "label": "Y",
                "url": "https://y.example/feed",
                "enabled": False,
            },
        ]
    }
    sources = validate_sources(data)
    assert len(sources) == 2
    assert sources[0].enabled is True
    assert sources[1] == FeedSource(
        source_key="y",
        kind="atom",
        url="https://y.example/feed",
        label="Y",
        enabled=False,
    )
    # enabled defaults to True when absent.
    assert validate_sources({"sources": [_minimal_source()]})[0].enabled is True


def test_invalid_toml_reports_parse_error_with_path(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text("[[sources]\nkey = ", encoding="utf-8")
    with pytest.raises(SourceConfigError, match="invalid TOML"):
        load_sources(bad)
    with pytest.raises(SourceConfigError, match="could not read"):
        load_sources(tmp_path / "missing.toml")


def test_valid_config_loads_all_sources_in_order(tmp_path):
    path = tmp_path / "sources.toml"
    path.write_text(VALID_TOML, encoding="utf-8")
    sources = load_sources(path)
    assert [s.source_key for s in sources] == ["bbc-business", "the-verge", "techcrunch"]
    assert [s.kind for s in sources] == ["rss", "atom", "rss"]
    assert [s.enabled for s in sources] == [True, False, True]
    assert sources[0].label == "BBC Business"
    assert sources[0].url == "https://feeds.bbci.co.uk/news/business/rss.xml"


def test_repo_starter_config_is_valid():
    # The shipped starter list loads through the same validation as user files.
    sources = load_sources(_REPOSITORY_CONFIG)
    keys = [s.source_key for s in sources]
    assert len(sources) == len(set(keys))
    assert all(s.enabled for s in sources)
    assert "bbc-business" in keys


def _minimal_source() -> dict:
    return {"key": "x", "kind": "rss", "label": "X", "url": "https://x.example/feed"}
