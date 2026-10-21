"""Retry classification tests: transient failures retry, permanent ones do not."""
from typing import TypedDict

import httpx
import openai
import pytest
from langgraph.graph import END, START, StateGraph

from market_intelligence.graph.supervisor import RETRY_POLICY
from market_intelligence.net import PermanentFetchError, TransientFetchError
from market_intelligence.retry import is_transient_error


def test_transient_fetch_errors_are_retried():
    assert is_transient_error(TransientFetchError("timeout"))
    assert RETRY_POLICY.retry_on(TransientFetchError("timeout")) is True


def test_permanent_input_errors_are_not_retried():
    for error in (
        PermanentFetchError("bad feed"),
        ValueError("bad count"),
        RuntimeError("bug"),
    ):
        assert is_transient_error(error) is False
        assert RETRY_POLICY.retry_on(error) is False


def test_llm_timeouts_are_transient():
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    assert is_transient_error(openai.APITimeoutError(request=request)) is True
    assert RETRY_POLICY.retry_on(openai.APITimeoutError(request=request)) is True


class _State(TypedDict, total=False):
    attempts: int


def _single_node_graph(node):
    graph = StateGraph(_State)
    graph.add_node("work", node, retry_policy=RETRY_POLICY)
    graph.add_edge(START, "work")
    graph.add_edge("work", END)
    return graph.compile()


def test_graph_retries_transient_failure():
    calls = {"count": 0}

    def node(state):
        calls["count"] += 1
        if calls["count"] < 3:
            raise TransientFetchError("temporary")
        return {"attempts": calls["count"]}

    result = _single_node_graph(node).invoke({})

    assert calls["count"] == 3
    assert result["attempts"] == 3


def test_graph_does_not_retry_permanent_failure():
    calls = {"count": 0}

    def node(state):
        calls["count"] += 1
        raise PermanentFetchError("invalid input")

    with pytest.raises(PermanentFetchError, match="invalid input"):
        _single_node_graph(node).invoke({})

    assert calls["count"] == 1
