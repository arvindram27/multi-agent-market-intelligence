"""Run-budget tests: call ceilings, token ceilings and LangChain usage shapes."""
from types import SimpleNamespace

import pytest

from market_intelligence.budget import (
    BudgetedLLM,
    RunTracker,
    TokenBudgetCallback,
    TokenBudgetExceeded,
    usage_from_response,
)


def test_call_ceiling_fails_closed():
    tracker = RunTracker(max_tokens=1000, max_calls=2)
    tracker.start_call()
    tracker.start_call()
    with pytest.raises(TokenBudgetExceeded, match="2-call"):
        tracker.start_call()


def test_token_ceiling_fails_closed():
    tracker = RunTracker(max_tokens=100, max_calls=10)
    tracker.add_usage(60, 30)
    assert tracker.total_tokens == 90
    with pytest.raises(TokenBudgetExceeded, match="100-token"):
        tracker.add_usage(20, 0)


def test_reset_clears_counters():
    tracker = RunTracker(max_tokens=100, max_calls=10)
    tracker.start_call()
    tracker.add_usage(10, 10)
    tracker.reset()
    assert (tracker.calls, tracker.prompt_tokens, tracker.completion_tokens) == (0, 0, 0)


def test_usage_from_llm_output():
    response = SimpleNamespace(
        llm_output={"token_usage": {"prompt_tokens": 12, "completion_tokens": 7}},
        generations=[],
    )
    assert usage_from_response(response) == (12, 7)


def test_usage_from_message_metadata():
    message = SimpleNamespace(
        usage_metadata={"input_tokens": 5, "output_tokens": 3}
    )
    response = SimpleNamespace(llm_output={}, generations=[[SimpleNamespace(message=message)]])
    assert usage_from_response(response) == (5, 3)


def test_callback_raises_through_langchain():
    tracker = RunTracker(max_tokens=10, max_calls=10)
    callback = TokenBudgetCallback(tracker)
    message = SimpleNamespace(usage_metadata={"input_tokens": 8, "output_tokens": 5})
    response = SimpleNamespace(llm_output={}, generations=[[SimpleNamespace(message=message)]])

    with pytest.raises(TokenBudgetExceeded):
        callback.on_llm_end(response)
    assert callback.raise_error is True


class _Structured:
    def __init__(self):
        self.configs = []

    def invoke(self, prompt, **kwargs):
        self.configs.append(kwargs.get("config", {}))
        return "result"


class _InnerLLM:
    def __init__(self):
        self.structured = _Structured()

    def with_structured_output(self, schema, **kwargs):
        return self.structured


def test_budgeted_llm_counts_calls_and_attaches_callback():
    inner = _InnerLLM()
    tracker = RunTracker(max_tokens=100, max_calls=2)
    wrapped = BudgetedLLM(inner, tracker)

    runnable = wrapped.with_structured_output(object)
    assert runnable.invoke("prompt") == "result"

    assert tracker.calls == 1
    callbacks = inner.structured.configs[0]["callbacks"]
    assert any(isinstance(cb, TokenBudgetCallback) for cb in callbacks)
