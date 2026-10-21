"""Consumer sentiment analysis agent: classify sentiment with an LLM."""
from market_intelligence.agents.llm import get_llm
from market_intelligence.agents.schemas import SentimentResult


def analyze_sentiment(text: str, llm=None) -> SentimentResult:
    """Classify the sentiment of a text into a structured :class:`SentimentResult`."""
    llm = llm or get_llm()
    structured = llm.with_structured_output(SentimentResult)
    prompt = f"Analyze the sentiment of the following text.\n\n{text}"
    return structured.invoke(prompt)
