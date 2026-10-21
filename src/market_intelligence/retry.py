"""Retry classification: only transient failures are retried."""
from market_intelligence.net import TransientFetchError


def is_transient_error(exc: Exception) -> bool:
    """Return True when retrying the same operation could reasonably succeed.

    Permanent input problems (SSRF rejections, malformed feeds, 4xx responses)
    return False. LLM timeouts, connection failures, rate limits and 5xx
    responses are retried by LangGraph's node policy.
    """
    if isinstance(exc, TransientFetchError):
        return True

    try:
        import openai
    except ImportError:  # pragma: no cover - openai ships with langchain-openai
        return False

    return isinstance(
        exc,
        (
            openai.APITimeoutError,
            openai.RateLimitError,
            openai.APIConnectionError,
            openai.InternalServerError,
        ),
    )
