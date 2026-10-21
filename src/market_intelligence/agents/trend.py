"""Trend identification agent: extract emerging trends from text with an LLM."""
from market_intelligence.agents.llm import get_llm
from market_intelligence.agents.schemas import TrendAnalysis, TrendReport


def identify_trends(texts: list[str], llm=None) -> list[TrendReport]:
    """Identify up to 5 market trends across the provided texts."""
    llm = llm or get_llm()
    structured = llm.with_structured_output(TrendAnalysis)
    combined = "\n\n".join(texts)
    prompt = (
        "Identify the top market trends from the following text. "
        "Return up to 5 trends.\n\n" + combined
    )
    return structured.invoke(prompt).trends
