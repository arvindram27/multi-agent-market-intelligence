"""News summarization agent: fetch an RSS feed and summarize articles with an LLM."""
import re

import feedparser

from market_intelligence.agents.llm import get_llm
from market_intelligence.agents.schemas import NewsSummary
from market_intelligence.config import settings

_HTML_TAG_RE = re.compile(r"<[^<]+?>")


def summarize_article(title: str, content: str, llm=None) -> NewsSummary:
    """Summarize a single article into a structured :class:`NewsSummary`."""
    llm = llm or get_llm()
    structured = llm.with_structured_output(NewsSummary)
    prompt = (
        "Summarize the following news article into a concise, factual summary.\n\n"
        f"Title: {title}\n\nContent:\n{content}"
    )
    return structured.invoke(prompt)


def fetch_and_summarize(
    rss_url: str, num_articles: int | None = None, llm=None
) -> list[NewsSummary]:
    """Fetch an RSS feed and return an LLM summary for each article."""
    if num_articles is None:
        num_articles = settings.num_articles
    feed = feedparser.parse(rss_url)
    summaries: list[NewsSummary] = []
    for entry in feed.entries[:num_articles]:
        title = getattr(entry, "title", "No Title")
        content = getattr(entry, "summary", "") or getattr(entry, "description", "")
        content = _HTML_TAG_RE.sub("", content)
        summaries.append(summarize_article(title, content, llm=llm))
    return summaries
