"""Competitor analysis agent: scrape a page and extract a structured profile."""
import requests
from bs4 import BeautifulSoup

from market_intelligence.agents.llm import get_llm
from market_intelligence.agents.schemas import CompetitorProfile
from market_intelligence.config import settings

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
)


def fetch_page_text(url: str) -> str:
    """Fetch a URL and return its visible text (for downstream LLM extraction)."""
    resp = requests.get(
        url, headers={"User-Agent": _USER_AGENT}, timeout=settings.competitor_request_timeout
    )
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, "html.parser")
    return soup.get_text(separator=" ", strip=True)


def analyze_competitor(page_text: str, url: str, llm=None) -> CompetitorProfile:
    """Extract a structured :class:`CompetitorProfile` from page text."""
    llm = llm or get_llm()
    structured = llm.with_structured_output(CompetitorProfile)
    prompt = (
        "Extract a structured competitor profile from the following webpage content.\n"
        f"URL: {url}\n\nContent:\n{page_text[:8000]}"
    )
    return structured.invoke(prompt)


def fetch_and_analyze(url: str, llm=None) -> CompetitorProfile:
    """Fetch a competitor page and return a structured profile."""
    return analyze_competitor(fetch_page_text(url), url, llm=llm)
