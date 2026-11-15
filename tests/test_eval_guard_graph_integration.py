"""Actual-graph integration for the outermost invocation guard.

Two in-memory tests exercise the real compiled graph, real agents and prompts,
real ``BudgetedLLM`` + ``GatekeeperLLM`` wrapping, the outer guard and the
approval interrupt. A local stub gate and stub structured client stand in for
the durable ledger and the provider; nothing here is a live runner, a validated
tokenizer, a spend cap or proof of provider model identity.

Authored only: these tests are NOT run by the step that wrote them.
"""
from __future__ import annotations

import hashlib
import threading
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import RetryPolicy

import market_intelligence.agents.competitor as competitor_module
import market_intelligence.agents.news as news_module
from eval.invocation_guard import (
    GRAPH_SCHEMA_NAMES,
    InvocationGuard,
    InvocationGuardFatalError,
)
from market_intelligence.agents.schemas import (
    ComparedFieldDraft,
    Comparison,
    EvidenceCitationDraft,
    NewsDigest,
    NewsToneDraft,
    SideComparisonDraft,
    TopicAnalysis,
    TopicEvidenceQuote,
    TopicFinding,
)
from market_intelligence.config import settings
from market_intelligence.feed_parsing import article_id
from market_intelligence.graph import build_graph, checkpoint_serde
from market_intelligence.net import FetchResult

RSS_URL = "https://example.com/feed.xml"
SUBJECT_URL = "https://example.org/acme"
COMPETITOR_URL = "https://example.net/rival"

ARTICLE_TITLE = "Acme reports strong widget demand"
ARTICLE_URL = "https://example.com/acme-growth"
PUBLISHED_AT = "2026-10-21T10:00:00+00:00"
EXCERPT = "Acme reported strong widget demand and steady subscription growth this quarter."
ARTICLE_ID = article_id(ARTICLE_URL, RSS_URL, PUBLISHED_AT, hashlib.sha256(EXCERPT.encode("utf-8")).hexdigest())

FEED_XML = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    "<rss version=\"2.0\"><channel><title>Synthetic feed</title><item>"
    f"<title>{ARTICLE_TITLE}</title>"
    f"<link>{ARTICLE_URL}</link>"
    "<description>" + EXCERPT + "</description>"
    "<pubDate>Wed, 21 Oct 2026 10:00:00 GMT</pubDate>"
    "</item></channel></rss>"
).encode("utf-8")

SUBJECT_HTML = (
    "<html><body>Acme builds widgets. Acme uses subscription pricing. "
    "Acme is the market leader.</body></html>"
).encode("utf-8")
COMPETITOR_HTML = (
    "<html><body>Rival builds gadgets. Rival uses usage pricing. "
    "Rival is the challenger.</body></html>"
).encode("utf-8")

THREAD_ID = "g11i-integration-run"


class SpyWriter:
    """Records persist calls and never touches storage."""

    def __init__(self):
        self.calls = []

    def persist(self, run_id, state):
        self.calls.append((run_id, state))
        return run_id


class StubGate:
    """In-memory admission/outcome echo; not a durable gate or settlement."""

    def __init__(self):
        self._lock = threading.Lock()
        self.admit_calls = []
        self.outcome_calls = []

    def admit_llm(self, request, event_id):
        with self._lock:
            self.admit_calls.append((request, event_id))
        decision = SimpleNamespace(
            allow=True,
            reserve_tokens=1024,
            reason=SimpleNamespace(value="allowed"),
        )
        return SimpleNamespace(status="applied", event_id=event_id, decision=decision)

    def record_llm_outcome(
        self, decision_event_id, prompt_tokens, completion_tokens, event_id
    ):
        with self._lock:
            self.outcome_calls.append(
                (decision_event_id, prompt_tokens, completion_tokens, event_id)
            )
        return SimpleNamespace(status="applied", event_id=event_id)


class StubStructuredClient:
    """Deterministic in-memory structured client with per-schema queues."""

    def __init__(self, responses, *, omit_usage_for=()):
        self._queues = {schema: list(values) for schema, values in responses.items()}
        self._omit_usage_for = frozenset(omit_usage_for)
        self._lock = threading.Lock()
        self.invokes = []
        self.wrapper_calls = []

    def with_structured_output(self, schema, **kwargs):
        with self._lock:
            self.wrapper_calls.append(schema.__name__)
        return _StubRunnable(self, schema)

    def _next(self, schema):
        with self._lock:
            queue = self._queues.get(schema)
            if not queue:
                raise AssertionError("no stub response queued for " + schema.__name__)
            return queue.pop(0)


class _StubRunnable:
    def __init__(self, client, schema):
        self._client = client
        self._schema = schema

    def invoke(self, prompt, **kwargs):
        client = self._client
        with client._lock:
            client.invokes.append(self._schema.__name__)
        value = client._next(self._schema)
        if self._schema.__name__ not in client._omit_usage_for:
            response = SimpleNamespace(
                llm_output={
                    "token_usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 2,
                    }
                },
                generations=[],
            )
            callbacks = list(kwargs.get("config", {}).get("callbacks", []) or [])
            if not callbacks:
                raise AssertionError("expected real budget/gatekeeper callbacks")
            for callback in callbacks:
                callback.on_llm_end(response)
        return value


def _side_draft(product, build_line, pricing_line, position_line):
    def field(value, quote):
        return ComparedFieldDraft(
            value=value,
            evidence=[EvidenceCitationDraft(source_id="S1", quote=quote)],
        )

    return SideComparisonDraft(
        product_focus=field(product, build_line),
        pricing_model=field("subscription" if "subscription" in pricing_line else "usage", pricing_line),
        positioning=field("leader" if "leader" in position_line else "challenger", position_line),
    )


def _responses(*, omit_usage_for=()):
    subject = _side_draft(
        "widgets",
        "Acme builds widgets",
        "uses subscription pricing",
        "market leader",
    )
    competitor = _side_draft(
        "gadgets",
        "Rival builds gadgets",
        "uses usage pricing",
        "the challenger",
    )
    return (
        {
            NewsDigest: [
                NewsDigest(
                    headline="Widget demand",
                    summary="Acme reports strong demand.",
                    key_facts=["strong widget demand"],
                    tone_hint="positive",
                )
            ],
            TopicAnalysis: [
                TopicAnalysis(
                    topics=[
                        TopicFinding(
                            topic_name="widget demand",
                            keywords=["widgets"],
                            model_confidence=0.8,
                            evidence=[
                                TopicEvidenceQuote(
                                    article_id=ARTICLE_ID,
                                    quote="strong widget demand",
                                )
                            ],
                        )
                    ]
                )
            ],
            NewsToneDraft: [
                NewsToneDraft(overall="positive", score=0.5, aspects=[])
            ],
            SideComparisonDraft: [subject, competitor],
        },
        frozenset(omit_usage_for),
    )


def _fake_fetch_factory():
    counts = {"feed": 0, "subject": 0, "competitor": 0}

    def fake_fetch(url, **kwargs):
        if url == RSS_URL:
            counts["feed"] += 1
            return FetchResult(
                url=url,
                content=FEED_XML,
                content_type="application/rss+xml",
                status_code=200,
            )
        if url == SUBJECT_URL:
            counts["subject"] += 1
            return FetchResult(
                url=url,
                content=SUBJECT_HTML,
                content_type="text/html",
                status_code=200,
            )
        if url == COMPETITOR_URL:
            counts["competitor"] += 1
            return FetchResult(
                url=url,
                content=COMPETITOR_HTML,
                content_type="text/html",
                status_code=200,
            )
        raise AssertionError("unexpected fetch URL " + url)

    return fake_fetch, counts


def _inputs():
    return {
        "rss_url": RSS_URL,
        "subject_name": "Acme",
        "subject_pages": [SUBJECT_URL],
        "competitor_name": "Rival",
        "competitor_pages": [COMPETITOR_URL],
        "num_articles": 1,
    }


def _config():
    return {"configurable": {"thread_id": THREAD_ID}}


def _build_graph(monkeypatch, *, omit_usage_for=()):
    responses, omit = _responses(omit_usage_for=omit_usage_for)
    client = StubStructuredClient(responses, omit_usage_for=omit)
    gate = StubGate()
    writer = SpyWriter()
    guard = InvocationGuard(
        expected_model=settings.openai_model,
        max_output_tokens=settings.llm_max_output_tokens,
        allowed_schema_names=GRAPH_SCHEMA_NAMES,
        estimator=lambda schema, prompt: 128,
    )
    fake_fetch, counts = _fake_fetch_factory()
    monkeypatch.setattr(news_module, "fetch_bytes", fake_fetch)
    monkeypatch.setattr(competitor_module, "fetch_bytes", fake_fetch)
    graph = build_graph(
        checkpointer=InMemorySaver(serde=checkpoint_serde()),
        llm=client,
        writer=writer,
        max_run_tokens=10_000,
        max_run_llm_calls=10,
        gatekeeper=guard.gate_proxy(gate),
        token_counter=guard.token_counter,
        retry_policy=RetryPolicy(max_attempts=1),
        structured_model_wrapper=guard.structured_model_wrapper,
    )
    return graph, guard, gate, client, writer, counts


def test_success_reaches_review_interrupt_with_five_guarded_calls(monkeypatch):
    graph, guard, gate, client, writer, counts = _build_graph(monkeypatch)

    result = graph.invoke(_inputs(), _config())
    snap = graph.get_state(_config())
    values = snap.values

    assert snap.next == ("review_profile",)
    assert values["status"] == "awaiting_approval"
    assert values.get("approved") is None
    assert values.get("persisted_run_id") is None
    assert writer.calls == []

    assert len(values["summaries"]) == 1
    assert values["summaries"][0].summary == "Acme reports strong demand."
    assert len(values["topics"]) == 1
    assert values["topics"][0].evidence[0].verified is True
    assert len(values["tones"]) == 1
    assert values["tones"][0].overall == "positive"

    comparison = values["comparison"]
    assert isinstance(comparison, Comparison)
    assert comparison.subject.name == "Acme"
    assert comparison.competitor.name == "Rival"
    assert comparison.subject.pages[0].url == SUBJECT_URL
    assert comparison.competitor.pages[0].url == COMPETITOR_URL
    assert comparison.subject.product_focus.value == "widgets"
    assert comparison.subject.product_focus.evidence[0].source_id == "S1"
    assert comparison.subject.product_focus.evidence[0].quote == "Acme builds widgets"
    assert comparison.subject.pricing_model.value == "subscription"
    assert comparison.subject.pricing_model.evidence[0].source_id == "S1"
    assert comparison.subject.pricing_model.evidence[0].quote == "uses subscription pricing"
    assert comparison.subject.positioning.value == "leader"
    assert comparison.subject.positioning.evidence[0].source_id == "S1"
    assert comparison.subject.positioning.evidence[0].quote == "market leader"
    assert comparison.competitor.product_focus.value == "gadgets"
    assert comparison.competitor.product_focus.evidence[0].source_id == "S1"
    assert comparison.competitor.product_focus.evidence[0].quote == "Rival builds gadgets"
    assert comparison.competitor.pricing_model.value == "usage"
    assert comparison.competitor.pricing_model.evidence[0].source_id == "S1"
    assert comparison.competitor.pricing_model.evidence[0].quote == "uses usage pricing"
    assert comparison.competitor.positioning.value == "challenger"
    assert comparison.competitor.positioning.evidence[0].source_id == "S1"
    assert comparison.competitor.positioning.evidence[0].quote == "the challenger"

    interrupts = result["__interrupt__"]
    assert len(interrupts) == 1
    payload = interrupts[0].value
    assert payload["type"] == "approval_request"
    assert payload["competitor"] is None

    assert counts == {"feed": 1, "subject": 1, "competitor": 1}

    invoked = list(client.invokes)
    assert len(invoked) == 5
    assert invoked.count("NewsDigest") == 1
    assert invoked.count("TopicAnalysis") == 1
    assert invoked.count("NewsToneDraft") == 1
    assert invoked.count("SideComparisonDraft") == 2
    assert set(invoked) == set(GRAPH_SCHEMA_NAMES)

    wrapper_calls = list(client.wrapper_calls)
    assert len(wrapper_calls) == 5
    assert wrapper_calls.count("NewsDigest") == 1
    assert wrapper_calls.count("TopicAnalysis") == 1
    assert wrapper_calls.count("NewsToneDraft") == 1
    assert wrapper_calls.count("SideComparisonDraft") == 2
    assert set(wrapper_calls) == set(invoked)

    assert invoked[0] == "NewsDigest"
    topic_index = invoked.index("TopicAnalysis")
    tone_index = invoked.index("NewsToneDraft")
    side_indices = [i for i, name in enumerate(invoked) if name == "SideComparisonDraft"]
    assert min(side_indices) > max(topic_index, tone_index)

    assert len(gate.admit_calls) == 5
    assert len(gate.outcome_calls) == 5
    evidence = guard.evidence()
    assert len(evidence) == 5
    for record in evidence:
        assert record.receipt_returned is True
        assert record.known_usage == (3, 2)
        assert record.unknown_usage is False
        assert record.accounting_incomplete is False
        assert record.fatal_reason is None
        assert record.accounting_exception_class is None
        assert record.exception_class is None
        assert record.estimated_request_tokens == 128
        assert record.admission_allowed is True
        assert record.admitted_reservation == 1024
        assert record.requested_decision_event_id is not None
        assert record.returned_decision_event_id == record.requested_decision_event_id
        assert record.requested_outcome_event_id is not None
        assert record.returned_outcome_event_id == record.requested_outcome_event_id
    assert guard.fatal_reason is None


def test_fan_out_unknown_usage_fails_stop_at_exactly_two_calls(monkeypatch):
    graph, guard, gate, client, writer, counts = _build_graph(
        monkeypatch, omit_usage_for=("TopicAnalysis", "NewsToneDraft")
    )

    with pytest.raises(InvocationGuardFatalError):
        graph.invoke(_inputs(), _config())

    assert guard.fatal_reason == "usage-unknown"
    snap = graph.get_state(_config())
    assert snap.next != ("review_profile",)
    assert writer.calls == []

    invoked = list(client.invokes)
    assert len(invoked) == 2
    assert invoked[0] == "NewsDigest"
    assert invoked[1] in {"TopicAnalysis", "NewsToneDraft"}
    assert "SideComparisonDraft" not in invoked

    assert len(gate.admit_calls) == 2
    assert len(gate.outcome_calls) == 2
    evidence = guard.evidence()
    assert len(evidence) == 2
    assert evidence[0].schema_name == "NewsDigest"
    assert evidence[0].known_usage == (3, 2)
    assert evidence[0].fatal_reason is None
    assert evidence[1].schema_name in {"TopicAnalysis", "NewsToneDraft"}
    assert evidence[1].unknown_usage is True
    assert evidence[1].known_usage is None
    assert evidence[1].admitted_reservation == 1024
    assert evidence[1].fatal_reason == "usage-unknown"

    outcomes = list(gate.outcome_calls)
    assert len(outcomes) == 2
    assert sum(1 for item in outcomes if item[1:3] == (None, None)) == 1
    assert sum(1 for item in outcomes if item[1:3] == (3, 2)) == 1
    by_decision = {item[0]: item for item in outcomes}
    for record in evidence:
        linked = by_decision[record.requested_decision_event_id]
        assert record.requested_outcome_event_id == linked[3]
        assert record.returned_outcome_event_id == linked[3]
    linked_unknown = by_decision[evidence[1].requested_decision_event_id]
    assert linked_unknown[1:3] == (None, None)
    assert evidence[1].receipt_returned is True
    assert evidence[1].accounting_incomplete is False
    linked_known = by_decision[evidence[0].requested_decision_event_id]
    assert linked_known[1:3] == (3, 2)

    wrapped = guard.structured_model_wrapper(client)
    with pytest.raises(InvocationGuardFatalError):
        wrapped.with_structured_output(NewsDigest).invoke("after fatal")
    assert len(client.invokes) == 2
    assert len(gate.admit_calls) == 2
    assert len(gate.outcome_calls) == 2
    assert len(guard.evidence()) == 2

    assert counts == {"feed": 1, "subject": 0, "competitor": 0}
