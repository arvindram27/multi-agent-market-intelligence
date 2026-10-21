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


def test_get_llm_applies_timeouts_and_output_cap(monkeypatch):
    monkeypatch.setattr(config_mod.settings, "openai_api_key", "unit-test-key")
    captured = {}

    def fake_init(model, **kwargs):
        captured["model"] = model
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "market_intelligence.agents.llm.init_chat_model", fake_init
    )

    get_llm()

    assert captured["model"] == config_mod.settings.openai_model
    assert captured["timeout"] == config_mod.settings.llm_request_timeout
    assert captured["max_tokens"] == config_mod.settings.llm_max_output_tokens
    assert captured["max_retries"] == 0
