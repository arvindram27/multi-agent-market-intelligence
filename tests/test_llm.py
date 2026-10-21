"""get_llm() fail-fast tests: clear error when OPENAI_API_KEY is missing."""
import pytest

import market_intelligence.config as config_mod
from market_intelligence.agents.llm import get_llm


def test_get_llm_raises_clear_error_without_key(monkeypatch):
    # Clear the environment variable and override the settings singleton so that
    # no local .env or environment value is read.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(config_mod.settings, "openai_api_key", "")
    # NOTE: this override proves the message is clear, but not that a stray .env can't leak a key into other tests.

    with pytest.raises(RuntimeError) as exc_info:
        get_llm()

    message = str(exc_info.value)
    assert "OPENAI_API_KEY" in message
    assert ".env.example" in message
    # The message must never contain an actual key value.
    assert "sk-" not in message
