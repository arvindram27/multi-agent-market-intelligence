"""Gatekeeper-backed structured model calls for the offline G5b slice."""
from __future__ import annotations

import threading
import uuid
from collections.abc import Mapping
from typing import Any, Callable

from langchain_core.callbacks import BaseCallbackHandler

from market_intelligence.admission import LLMRequest
from market_intelligence.gatekeeper import Gatekeeper


class GatekeeperDenied(RuntimeError):
    """An LLM request was denied by admission before the model was called."""

    def __init__(self, decision):
        self.decision = decision
        self.reason = (
            decision.reason.value
            if decision is not None and hasattr(decision.reason, "value")
            else "unknown"
        )
        super().__init__("LLM call denied by Gatekeeper: " + self.reason)


_MISSING = object()


def _usage_pair(mapping: object):
    if not isinstance(mapping, Mapping):
        return None
    prompt = mapping.get("prompt_tokens", mapping.get("input_tokens", _MISSING))
    completion = mapping.get(
        "completion_tokens", mapping.get("output_tokens", _MISSING)
    )
    if prompt is _MISSING or completion is _MISSING:
        return None
    if type(prompt) is not int or type(completion) is not int:
        return None
    if prompt < 0 or completion < 0:
        return None
    return (prompt, completion)


def _response_usage(response):
    output = getattr(response, "llm_output", None)
    if isinstance(output, Mapping):
        for key in ("token_usage", "usage"):
            if key in output:
                return _usage_pair(output[key])
    generations = getattr(response, "generations", None) or []
    pairs = []
    found = False
    for generation_list in generations:
        for generation in generation_list:
            message = getattr(generation, "message", None)
            metadata = getattr(message, "usage_metadata", None)
            if metadata is None:
                continue
            found = True
            pair = _usage_pair(metadata)
            if pair is None:
                return None
            pairs.append(pair)
    if not found:
        return None
    return (
        sum(pair[0] for pair in pairs),
        sum(pair[1] for pair in pairs),
    )


class _UsageCapture(BaseCallbackHandler):
    """Collects raw provider usage without ever raising from a callback."""

    raise_error = False

    def __init__(self):
        self._prompt = 0
        self._completion = 0
        self._seen = False
        self._invalid = False

    def on_llm_end(self, response, **kwargs):
        try:
            pair = _response_usage(response)
        except Exception:
            pair = None
        if pair is None:
            self._invalid = True
            return
        self._seen = True
        self._prompt += pair[0]
        self._completion += pair[1]

    def usage(self):
        if self._invalid or not self._seen:
            return None
        return (self._prompt, self._completion)


class _GatedStructured:
    def __init__(self, adapter: "GatekeeperLLM", schema: type):
        self._adapter = adapter
        self._schema = schema

    def invoke(self, prompt, **kwargs):
        adapter = self._adapter
        gate = adapter._gatekeeper

        counted = adapter._count(prompt)
        request = LLMRequest(
            adapter._model_name,
            self._schema.__name__,
            counted,
            adapter._max_output_tokens,
        )
        decision_id = str(uuid.uuid4())
        with adapter._lock:
            result = gate.admit_llm(request, decision_id)
            adapter.events.append((decision_id, None))
            event_index = len(adapter.events) - 1
        decision = result.decision
        if decision is None or not decision.allow:
            raise GatekeeperDenied(decision)

        capture = _UsageCapture()
        try:
            config = dict(kwargs.pop("config", None) or {})
            callbacks = list(config.get("callbacks", []) or [])
            callbacks.append(capture)
            config["callbacks"] = callbacks
            kwargs["config"] = config

            inner = adapter._inner.with_structured_output(self._schema)
            result_value = inner.invoke(prompt, **kwargs)
        except Exception:
            adapter._record_unknown_outcome(decision_id, event_index)
            raise

        usage = capture.usage()
        outcome_id = str(uuid.uuid4())
        if usage is None:
            prompt_tokens, completion_tokens = None, None
        else:
            prompt_tokens, completion_tokens = usage
        with adapter._lock:
            gate.record_llm_outcome(
                decision_id,
                prompt_tokens,
                completion_tokens,
                outcome_id,
            )
            adapter.events[event_index] = (decision_id, outcome_id)
        return result_value


class GatekeeperLLM:
    """Wrap a structured-output model so each attempt passes through a Gatekeeper."""

    def __init__(
        self,
        inner,
        gatekeeper: Gatekeeper,
        *,
        model_name: str,
        max_output_tokens: int,
        token_counter: Callable[[Any], int] | None = None,
    ):
        self._inner = inner
        self._gatekeeper = gatekeeper
        self._model_name = model_name
        self._max_output_tokens = max_output_tokens
        self._token_counter = token_counter
        self._lock = threading.Lock()
        self.events: list[tuple[str, str | None]] = []

    def _count(self, prompt) -> int | None:
        if self._token_counter is None:
            return None
        try:
            value = self._token_counter(prompt)
        except Exception:
            return None
        if type(value) is not int or value < 0:
            return None
        return value

    def _record_unknown_outcome(self, decision_id: str, event_index: int) -> None:
        """Record unknown usage for an admitted call that raised.

        The reservation stays held. If this record cannot be written, the
        failure is swallowed so the original model error still reaches the
        caller, and the trace stays as it was before the call raised.
        """
        try:
            outcome_id = str(uuid.uuid4())
            with self._lock:
                self._gatekeeper.record_llm_outcome(
                    decision_id, None, None, outcome_id
                )
                self.events[event_index] = (decision_id, outcome_id)
        except Exception:
            return

    def with_structured_output(self, schema, **kwargs):
        return _GatedStructured(self, schema)
