"""Shared pytest fixtures for the market intelligence test suite."""
import pytest


class FakeStructuredLLM:
    """Fake LLM that returns a fixed Pydantic instance per output schema.

    Pass ``responses`` as a mapping of schema class -> instance. Each
    ``with_structured_output(schema)`` call records the prompt and returns the
    matching instance, so the whole suite runs offline with no API key.
    """

    def __init__(self, responses: dict | None = None):
        self.responses = responses or {}
        self.calls: list[str] = []

    def with_structured_output(self, schema, **kwargs):
        return _SchemaRunnable(self.responses[schema], self.calls)


class _SchemaRunnable:
    def __init__(self, response, calls):
        self._response = response
        self._calls = calls

    def invoke(self, prompt, **kwargs):
        self._calls.append(prompt)
        return self._response


@pytest.fixture
def fake_llm():
    """Return the FakeStructuredLLM class (instantiate with a responses dict)."""
    return FakeStructuredLLM
