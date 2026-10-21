"""Competitor fetch tests: bounded client, final URL and empty pages."""
import pytest

from market_intelligence.agents.competitor import (
    HTML_CONTENT_TYPES,
    fetch_and_analyze,
    fetch_page,
)
from market_intelligence.agents.schemas import CompetitorProfile
from market_intelligence.config import settings
from market_intelligence.net import FetchResult, PermanentFetchError


class FakeFetcher:
    def __init__(self, content: bytes, url="https://acme.example/about"):
        self.content = content
        self.url = url
        self.calls = []

    def __call__(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return FetchResult(self.url, self.content, "text/html", 200)


def test_fetch_page_uses_bounded_client():
    fetcher = FakeFetcher(b"<html><body><h1>Acme</h1><p>Widgets for teams.</p></body></html>")
    page = fetch_page("https://acme.example/about", fetcher=fetcher)

    assert "Acme" in page.text
    assert page.url == "https://acme.example/about"
    assert page.retrieved_at
    _, kwargs = fetcher.calls[0]
    assert kwargs["timeout"] == settings.competitor_request_timeout
    assert kwargs["max_bytes"] == settings.max_page_bytes
    assert kwargs["allowed_content_types"] == HTML_CONTENT_TYPES


def test_empty_page_is_rejected():
    fetcher = FakeFetcher(b"<html><body>   </body></html>")
    with pytest.raises(PermanentFetchError, match="no readable text"):
        fetch_page("https://acme.example/about", fetcher=fetcher)


def test_fetch_and_analyze_uses_the_final_url(fake_llm):
    fetcher = FakeFetcher(
        b"<html><body>Acme makes widgets for enterprise teams.</body></html>",
        url="https://acme.example/final",
    )
    llm = fake_llm(
        {
            CompetitorProfile: CompetitorProfile(
                company="Acme", positioning="widget maker"
            )
        }
    )

    profile = fetch_and_analyze(
        "https://acme.example/about", llm=llm, fetcher=fetcher
    )

    assert profile.company == "Acme"
    assert "https://acme.example/final" in llm.calls[-1]
