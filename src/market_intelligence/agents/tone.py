"""News tone agent: classify the tone of a model-written news summary."""
from market_intelligence.agents.llm import get_llm
from market_intelligence.agents.schemas import NewsSummary, NewsTone, NewsToneDraft


class ToneScoreContractError(RuntimeError):
    """A model tone label and its signed score contradict each other."""


def _require_sign(
    label: str,
    score: float,
    article_id: str,
    aspect_index: int | None,
) -> None:
    """Raise when a positive/negative label has a non-matching signed score."""
    if label == "positive" and not score > 0:
        if aspect_index is None:
            raise ToneScoreContractError(
                f"tone score contract violated for article {article_id}: "
                "overall='positive' requires score > 0"
            )
        raise ToneScoreContractError(
            f"tone score contract violated for article {article_id}: "
            f"aspect #{aspect_index} tone='positive' requires score > 0"
        )
    if label == "negative" and not score < 0:
        if aspect_index is None:
            raise ToneScoreContractError(
                f"tone score contract violated for article {article_id}: "
                "overall='negative' requires score < 0"
            )
        raise ToneScoreContractError(
            f"tone score contract violated for article {article_id}: "
            f"aspect #{aspect_index} tone='negative' requires score < 0"
        )


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
        "this text; it is not consumer sentiment and not a survey. score is signed "
        "polarity from -1 (very negative) to +1 (very positive): positive tone "
        "requires score > 0, negative tone requires score < 0, and neutral or "
        "mixed may take any value in range.\n\n"
        f"Headline: {summary.headline}\n"
        f"Summary: {summary.summary}\n"
        f"RSS excerpt: {summary.excerpt[:1200]}"
    )
    draft = structured.invoke(prompt)
    _require_sign(draft.overall, draft.score, summary.article_id, None)
    for index, aspect in enumerate(draft.aspects):
        _require_sign(aspect.tone, aspect.score, summary.article_id, index)
    return NewsTone(
        **draft.model_dump(),
        article_id=summary.article_id,
        headline=summary.headline,
        source_url=summary.source_url,
        published_at=summary.published_at,
        retrieved_at=summary.retrieved_at,
    )
