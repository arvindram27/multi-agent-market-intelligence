"""News tone agent: classify the tone of a model-written news summary."""
from market_intelligence.agents.llm import get_llm
from market_intelligence.agents.schemas import NewsSummary, NewsTone, NewsToneDraft


def analyze_news_tone(summary: NewsSummary, llm=None) -> NewsTone:
    """Classify the tone of a news summary and link it to its source record.

    This is a model assessment of the summary text. It is not consumer
    sentiment and is not a calibrated measurement of any population.
    """
    llm = llm or get_llm()
    structured = llm.with_structured_output(NewsToneDraft)
    prompt = (
        "Classify the overall tone and aspect-level tone of the following "
        "model-written summary of an RSS excerpt. Return only an assessment of "
        "this text; it is not consumer sentiment and not a survey.\n\n"
        f"Headline: {summary.headline}\n"
        f"Summary: {summary.summary}\n"
        f"RSS excerpt: {summary.excerpt[:1200]}"
    )
    draft = structured.invoke(prompt)
    return NewsTone(
        **draft.model_dump(),
        article_id=summary.article_id,
        headline=summary.headline,
        source_url=summary.source_url,
        published_at=summary.published_at,
        retrieved_at=summary.retrieved_at,
    )
