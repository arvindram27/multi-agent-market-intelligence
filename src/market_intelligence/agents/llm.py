"""LLM construction helper."""
from langchain.chat_models import init_chat_model

from market_intelligence.config import settings


def get_llm():
    """Return a chat model configured from settings.

    Raises ``RuntimeError`` with a clear message when no OpenAI API key is set
    (an empty string counts as missing).
    """
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and fill in "
            "OPENAI_API_KEY, or set the OPENAI_API_KEY environment variable."
        )
    return init_chat_model(settings.openai_model, api_key=settings.openai_api_key)
