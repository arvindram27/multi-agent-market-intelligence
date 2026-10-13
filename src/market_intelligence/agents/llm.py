"""LLM construction helper."""
from langchain.chat_models import init_chat_model

from market_intelligence.config import settings


def get_llm():
    """Return a chat model configured from settings.

    The OpenAI API key is only passed when set; otherwise the model reads the
    ``OPENAI_API_KEY`` environment variable at invoke time.
    """
    kwargs = {}
    if settings.openai_api_key:
        kwargs["api_key"] = settings.openai_api_key
    return init_chat_model(settings.openai_model, **kwargs)
