"""Competitor analysis agent: fetch a bounded page and extract a structured profile."""
from dataclasses import dataclass
from datetime import datetime, timezone

from bs4 import BeautifulSoup

from market_intelligence.agents.llm import get_llm
from market_intelligence.agents.schemas import CompetitorProfile
from market_intelligence.config import settings
from market_intelligence.net import FetchResult, PermanentFetchError, fetch_bytes

HTML_CONTENT_TYPES = frozenset({"text/html", "application/xhtml+xml"})


@dataclass(frozen=True)
class FetchedPage:
    """Visible page text plus the final URL and retrieval time."""

    url: str
    text: str
    retrieved_at: str


def fetch_page(url: str, fetcher=None) -> FetchedPage:
    """Fetch a competitor page through the bounded, SSRF-guarded client."""
    fetcher = fetcher or fetch_bytes
    result: FetchResult = fetcher(
        url,
        timeout=settings.competitor_request_timeout,
        max_bytes=settings.max_page_bytes,
        allowed_content_types=HTML_CONTENT_TYPES,
        max_redirects=settings.max_redirects,
    )
    soup = BeautifulSoup(result.content, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    if not text:
        raise PermanentFetchError("The competitor page contained no readable text.")
    return FetchedPage(
        url=result.url,
        text=text,
        retrieved_at=datetime.now(timezone.utc).isoformat(),
    )


def fetch_page_text(url: str, fetcher=None) -> str:
    """Backward-compatible wrapper returning only the visible page text."""
    return fetch_page(url, fetcher=fetcher).text


def analyze_competitor(page_text: str, url: str, llm=None) -> CompetitorProfile:
    """Extract a structured :class:`CompetitorProfile` from page text."""
    llm = llm or get_llm()
    structured = llm.with_structured_output(CompetitorProfile)
    prompt = (
        "Extract a structured competitor profile from the following webpage content.\n"
        f"URL: {url}\n\nContent:\n{page_text[: settings.max_competitor_chars]}"
    )
    return structured.invoke(prompt)


def fetch_and_analyze(url: str, llm=None, fetcher=None) -> CompetitorProfile:
    """Fetch a competitor page and return a structured profile."""
    page = fetch_page(url, fetcher=fetcher)
    return analyze_competitor(page.text, page.url, llm=llm)
