"""Per-run LLM call and token ceilings.

The tracker is shared by every agent in one compiled graph and reset when a run
starts at ``fetch_news``. A run fails closed once either ceiling is crossed, so
an outage or retry loop cannot spend without bound.
"""
from __future__ import annotations

import threading

from langchain_core.callbacks import BaseCallbackHandler


class TokenBudgetExceeded(RuntimeError):
    """The run crossed its call or token ceiling."""


class RunTracker:
    """Mutable counters for one graph run."""

    def __init__(self, max_tokens: int, max_calls: int):
        self.max_tokens = max_tokens
        self.max_calls = max_calls
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0
        self._lock = threading.RLock()

    @property
    def total_tokens(self) -> int:
        with self._lock:
            return self.prompt_tokens + self.completion_tokens

    def reset(self) -> None:
        with self._lock:
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.calls = 0

    def start_call(self) -> None:
        with self._lock:
            self.calls += 1
            if self.calls > self.max_calls:
                raise TokenBudgetExceeded(
                    f"Run exceeded its {self.max_calls}-call LLM ceiling."
                )

    def add_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        with self._lock:
            self.prompt_tokens += max(0, prompt_tokens)
            self.completion_tokens += max(0, completion_tokens)
            total = self.prompt_tokens + self.completion_tokens
            if total > self.max_tokens:
                raise TokenBudgetExceeded(
                    f"Run exceeded its {self.max_tokens}-token ceiling "
                    f"({total} tokens used)."
                )


def usage_from_response(response) -> tuple[int, int]:
    """Extract ``(prompt, completion)`` tokens across LangChain response shapes."""
    output = getattr(response, "llm_output", None) or {}
    usage = output.get("token_usage") or output.get("usage") or {}
    prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    completion = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    if prompt or completion:
        return prompt, completion

    for generation_list in getattr(response, "generations", None) or []:
        for generation in generation_list:
            message = getattr(generation, "message", None)
            metadata = getattr(message, "usage_metadata", None) or {}
            prompt += int(metadata.get("input_tokens") or 0)
            completion += int(metadata.get("output_tokens") or 0)
    return prompt, completion


class TokenBudgetCallback(BaseCallbackHandler):
    """Records usage and raises (propagating) when the run is over budget."""

    raise_error = True

    def __init__(self, tracker: RunTracker):
        self.tracker = tracker

    def on_llm_end(self, response, **kwargs) -> None:
        prompt, completion = usage_from_response(response)
        self.tracker.add_usage(prompt, completion)


class _BudgetedStructured:
    def __init__(self, runnable, tracker: RunTracker):
        self._runnable = runnable
        self._tracker = tracker

    def invoke(self, prompt, **kwargs):
        self._tracker.start_call()
        config = kwargs.pop("config", None) or {}
        callbacks = list(config.get("callbacks", [])) + [
            TokenBudgetCallback(self._tracker)
        ]
        return self._runnable.invoke(
            prompt,
            config={**config, "callbacks": callbacks},
            **kwargs,
        )


class BudgetedLLM:
    """Wrap a chat model so every structured call is counted against a run."""

    def __init__(self, llm, tracker: RunTracker):
        self._llm = llm
        self._tracker = tracker

    def with_structured_output(self, schema, **kwargs):
        return _BudgetedStructured(
            self._llm.with_structured_output(schema, **kwargs),
            self._tracker,
        )
