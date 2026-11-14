"""Offline controlled-run demo: the real graph and Gatekeeper, fake effects only.

Three tests show, with the real graph, the real Gatekeeper, a fake model,
patched fetch and a scratch audit store:
(A) a full run that reaches the approval interrupt and completes, with every
    model call admitted and its usage recorded, plus a readable audit summary;
(B) a clean restart in a new graph object and a new Gatekeeper over the same
    audit store and sqlite checkpoint, where earlier spend still counts and a
    call that would exceed the cap is denied before the model;
(C) a mid-run model failure that leaves an unknown-usage outcome with the
    reservation held, visibly reducing the remaining cap.

Nothing here is a real provider, a real tokenizer, Postgres or a network call.
A clean restart is not a crash; no crash recovery or lease takeover is shown.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

import market_intelligence.agents.competitor as competitor_module
import market_intelligence.agents.news as news_module
import market_intelligence.control as control
import market_intelligence.ledger as ledger
from market_intelligence.admission import PolicyConfig
from market_intelligence.agents.schemas import (
    ComparedFieldDraft,
    EvidenceCitationDraft,
    NewsDigest,
    NewsToneDraft,
    SideComparisonDraft,
    TopicAnalysis,
    TopicFinding,
)
from market_intelligence.config import settings
from market_intelligence.gatekeeper import Gatekeeper
from market_intelligence.gatekeeper_llm import GatekeeperDenied, GatekeeperLLM
from market_intelligence.graph import build_graph, checkpoint_serde
from market_intelligence.net import FetchResult
from market_intelligence.persistence import NullProfileWriter

T0 = datetime(2026, 11, 14, 0, 0, 0, tzinfo=timezone.utc)
RUN = "run-demo"
TTL = 300
# Each test passes ONE thread id to both the Gatekeeper's audit run and the
# graph's checkpoint config. That shared value is deliberate construction in
# the test, not an automatic binding: the Gatekeeper never reads the
# checkpoint, and LangGraph never reads the audit store.
FEED_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Test feed</title>
<item><title>T1</title><description>content one is a sufficiently long RSS excerpt for summarization</description></item>
</channel></rss>"""
PAGE_HTML = b"<html><body>Acme corp is a leading widget maker.</body></html>"

# One full run makes five structured model calls: one news digest, one topics
# call, one tone call and two per-side comparison drafts.
DEMO_CALLS = 5
# Fixed fake counter (10 prompt tokens), margin 0.25, no overhead, output 32:
# reserve = ceil(10 * 1.25) + 32 = 45 per call.
RESERVE_PER_CALL = 45
# Fixed fake usage per call: 10 prompt + 5 completion = 15 spent per call.
USAGE_PER_CALL = (10, 5)


def _policy(**overrides):
    values = dict(
        max_run_tokens=110,
        max_run_llm_calls=10,
        max_output_tokens=32,
        max_fetches_per_run=8,
        reservation_margin=0.25,
        provider_overhead_tokens=0,
        allowed_model="gpt-4o-mini",
        allowed_schemas=frozenset(
            {
                "NewsDigest",
                "TopicAnalysis",
                "NewsToneDraft",
                "SideComparisonDraft",
                "CompetitorProfileDraft",
            }
        ),
    )
    values.update(overrides)
    return PolicyConfig(**values)


class _Clock:
    def __init__(self):
        self.t = T0

    def __call__(self):
        return self.t

    def adv(self, seconds):
        self.t += timedelta(seconds=seconds)


def _make_gate(store, thread_id, policy=None, owner=None, clock=None):
    gate = Gatekeeper(
        store,
        policy or _policy(),
        run_id=RUN,
        thread_id=thread_id,
        owner_id=owner or str(uuid.uuid4()),
        lease_ttl_seconds=TTL,
        now=clock or _Clock(),
    )
    gate.start_run(str(uuid.uuid4()))
    gate.acquire_lease(str(uuid.uuid4()))
    return gate


def _fixed_token_counter_for_tests_only(prompt):
    return 10


def _run_state(store):
    rows = store._conn.execute(
        "SELECT payload FROM audit_events ORDER BY seq"
    ).fetchall()
    events = [
        control.event_from_payload(json.loads(row[0]), f"seq {index}")
        for index, row in enumerate(rows, start=1)
    ]
    return ledger.replay_events(events).runs[RUN]


def _summary(store, cap):
    """Replay the audit store and print a readable per-run summary."""
    state = _run_state(store)
    llm_decisions = [
        decision
        for decision in state.decisions.values()
        if decision.operation == "llm"
    ]
    allowed = sum(1 for decision in llm_decisions if decision.allow)
    denied = sum(1 for decision in llm_decisions if not decision.allow)
    spent = state.prompt_tokens + state.completion_tokens
    summary = {
        "calls": state.calls,
        "prompt_tokens": state.prompt_tokens,
        "completion_tokens": state.completion_tokens,
        "spent_tokens": spent,
        "reserved_tokens": state.reserved_tokens,
        "remaining_tokens": max(0, cap - spent - state.reserved_tokens),
        "allowed_decisions": allowed,
        "denied_decisions": denied,
        "reservation_statuses": sorted(
            reservation.status for reservation in state.reservations.values()
        ),
    }
    print("AUDIT SUMMARY " + json.dumps(summary, sort_keys=True))
    return summary


class _UsageRunnable:
    def __init__(self, model, response):
        self._model = model
        self._queue = response if isinstance(response, list) else None
        self._response = response

    def invoke(self, prompt, **kwargs):
        self._model.calls.append(prompt)
        if self._model.error is not None:
            raise self._model.error
        prompt_tokens, completion_tokens = self._model.usage
        response = SimpleNamespace(
            llm_output={
                "token_usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                }
            },
            generations=[],
        )
        for callback in kwargs.get("config", {}).get("callbacks", []):
            callback.on_llm_end(response)
        if self._queue is not None:
            return self._queue.pop(0)
        return self._response


class _UsageLLM:
    """Fake structured model that reports fixed usage through callbacks.

    Unlike the conftest FakeStructuredLLM it fires on_llm_end, so the adapter
    can record real-shaped (still fake) usage instead of usage_missing.
    """

    def __init__(self, responses, usage=USAGE_PER_CALL, error=None):
        self.responses = responses
        self.usage = usage
        self.error = error
        self.calls = []

    def with_structured_output(self, schema, **kwargs):
        return _UsageRunnable(self, self.responses[schema])


class _RecordingInner:
    """Minimal adapter-level model that counts calls; never reached on denial."""

    def __init__(self):
        self.calls = 0

    def with_structured_output(self, schema, **kwargs):
        inner = self

        class _Runnable:
            def invoke(self, prompt, **kwargs):
                inner.calls += 1
                return "ok"

        return _Runnable()


def _side_draft():
    quote = "leading widget maker"
    citation = [EvidenceCitationDraft(source_id="S1", quote=quote)]
    return SideComparisonDraft(
        product_focus=ComparedFieldDraft(value="widgets", evidence=citation),
        pricing_model=ComparedFieldDraft(),
        positioning=ComparedFieldDraft(value="widget maker", evidence=list(citation)),
    )


def _responses():
    return {
        NewsDigest: NewsDigest(headline="h", summary="s1", tone_hint="neutral"),
        TopicAnalysis: TopicAnalysis(
            topics=[TopicFinding(topic_name="AI", model_confidence=0.8)]
        ),
        NewsToneDraft: NewsToneDraft(overall="positive", score=0.5),
        SideComparisonDraft: [_side_draft(), _side_draft()],
    }


def _inputs():
    return {
        "rss_url": "http://example.com/feed.xml",
        "subject_name": "Acme",
        "subject_pages": ["https://acme.com"],
        "competitor_name": "Rival",
        "competitor_pages": ["https://rival.com"],
        "num_articles": 1,
    }


def _mock_io(monkeypatch):
    def fake_fetch(url, **kwargs):
        if "feed" in url:
            return FetchResult(url, FEED_XML, "application/rss+xml", 200)
        return FetchResult(url, PAGE_HTML, "text/html", 200)

    monkeypatch.setattr(news_module, "fetch_bytes", fake_fetch)
    monkeypatch.setattr(competitor_module, "fetch_bytes", fake_fetch)


def _demo_settings(monkeypatch):
    monkeypatch.setattr(settings, "openai_model", "gpt-4o-mini")
    monkeypatch.setattr(settings, "llm_max_output_tokens", 32)


def _config(thread_id):
    return {"configurable": {"thread_id": thread_id}}


def _values(snapshot):
    values = snapshot.values
    return values.model_dump() if hasattr(values, "model_dump") else values


@pytest.fixture
def store(tmp_path):
    value = control.AuditStore(tmp_path / "audit.db")
    yield value
    value.close()


def test_full_graph_run_records_every_call_and_shows_the_summary(
    store, monkeypatch
):
    _mock_io(monkeypatch)
    _demo_settings(monkeypatch)
    thread = "g7-demo-full"
    policy = _policy()
    gate = _make_gate(store, thread, policy=policy)
    model = _UsageLLM(_responses())
    graph = build_graph(
        checkpointer=InMemorySaver(serde=checkpoint_serde()),
        llm=model,
        writer=NullProfileWriter(),
        gatekeeper=gate,
        token_counter=_fixed_token_counter_for_tests_only,
    )

    graph.invoke(_inputs(), _config(thread))
    snap = graph.get_state(_config(thread))
    assert snap.next == ("review_profile",)
    assert _values(snap)["status"] == "awaiting_approval"

    graph.invoke(Command(resume={"approved": True}), _config(thread))
    snap = graph.get_state(_config(thread))
    assert snap.next == ()
    assert _values(snap)["status"] == "complete"

    assert len(model.calls) == DEMO_CALLS
    summary = _summary(store, policy.max_run_tokens)
    assert summary["calls"] == DEMO_CALLS
    assert (summary["prompt_tokens"], summary["completion_tokens"]) == (
        DEMO_CALLS * USAGE_PER_CALL[0],
        DEMO_CALLS * USAGE_PER_CALL[1],
    )
    assert summary["spent_tokens"] == DEMO_CALLS * sum(USAGE_PER_CALL)
    assert summary["reserved_tokens"] == 0
    assert summary["remaining_tokens"] == policy.max_run_tokens - DEMO_CALLS * sum(
        USAGE_PER_CALL
    )
    assert summary["allowed_decisions"] == DEMO_CALLS
    assert summary["denied_decisions"] == 0


def test_restart_with_a_new_graph_and_gatekeeper_keeps_earlier_spend(
    tmp_path, monkeypatch
):
    _mock_io(monkeypatch)
    _demo_settings(monkeypatch)
    thread = "g7-demo-restart"
    policy = _policy()
    store = control.AuditStore(tmp_path / "audit.db")
    db_path = str(tmp_path / "checkpoints.sqlite")
    try:
        gate_a = _make_gate(store, thread, policy=policy)
        model_a = _UsageLLM(_responses())
        conn = sqlite3.connect(db_path, check_same_thread=False)
        try:
            first = build_graph(
                checkpointer=SqliteSaver(conn, serde=checkpoint_serde()),
                llm=model_a,
                writer=NullProfileWriter(),
                gatekeeper=gate_a,
                token_counter=_fixed_token_counter_for_tests_only,
            )
            first.invoke(_inputs(), _config(thread))
            snap = first.get_state(_config(thread))
            assert snap.next == ("review_profile",)
        finally:
            conn.close()
        before = _summary(store, policy.max_run_tokens)

        # Clean release, not a crash: no takeover and no expiry is shown.
        gate_a.release_lease(str(uuid.uuid4()))

        gate_b = Gatekeeper(
            store,
            policy,
            run_id=RUN,
            thread_id=thread,
            owner_id=str(uuid.uuid4()),
            lease_ttl_seconds=TTL,
            now=_Clock(),
        )
        gate_b.acquire_lease(str(uuid.uuid4()))
        model_b = _UsageLLM(_responses())
        conn = sqlite3.connect(db_path, check_same_thread=False)
        try:
            second = build_graph(
                checkpointer=SqliteSaver(conn, serde=checkpoint_serde()),
                llm=model_b,
                writer=NullProfileWriter(),
                gatekeeper=gate_b,
                token_counter=_fixed_token_counter_for_tests_only,
            )
            second.invoke(Command(resume={"approved": True}), _config(thread))
            snap = second.get_state(_config(thread))
            assert snap.next == ()
            assert _values(snap)["status"] == "complete"
        finally:
            conn.close()

        after = _summary(store, policy.max_run_tokens)
        assert after == before
        assert after["spent_tokens"] == DEMO_CALLS * sum(USAGE_PER_CALL)

        # At the adapter boundary, a further call is denied before the model:
        # spent 75 + reserve 45 = 120 > cap 110.
        inner = _RecordingInner()
        adapter = GatekeeperLLM(
            inner,
            gate_b,
            model_name="gpt-4o-mini",
            max_output_tokens=32,
            token_counter=_fixed_token_counter_for_tests_only,
        )
        with pytest.raises(GatekeeperDenied) as info:
            adapter.with_structured_output(NewsDigest).invoke("one more")
        assert info.value.reason == "budget_exceeded"
        assert inner.calls == 0
    finally:
        store.close()


def test_failed_call_mid_graph_holds_its_reservation_and_shows_the_cap_effect(
    store, monkeypatch
):
    _mock_io(monkeypatch)
    _demo_settings(monkeypatch)
    thread = "g7-demo-failure"
    policy = _policy(max_run_tokens=80)
    gate = _make_gate(store, thread, policy=policy)
    error = RuntimeError("model failed")
    model = _UsageLLM(_responses(), error=error)
    graph = build_graph(
        checkpointer=InMemorySaver(serde=checkpoint_serde()),
        llm=model,
        writer=NullProfileWriter(),
        gatekeeper=gate,
        token_counter=_fixed_token_counter_for_tests_only,
    )

    # A plain RuntimeError is not transient (retry.py), so: one attempt.
    with pytest.raises(RuntimeError) as raised:
        graph.invoke(_inputs(), _config(thread))
    assert raised.value is error
    assert len(model.calls) == 1

    snap = graph.get_state(_config(thread))
    assert snap.next != ("review_profile",)
    # The first node never returned, so the checkpoint carries no status
    # update; the typed default is "running" and nothing reached approval.
    assert _values(snap).get("status", "running") == "running"

    summary = _summary(store, policy.max_run_tokens)
    assert summary["allowed_decisions"] == 1
    assert summary["denied_decisions"] == 0
    assert summary["spent_tokens"] == 0
    assert summary["reserved_tokens"] == RESERVE_PER_CALL
    assert summary["remaining_tokens"] == policy.max_run_tokens - RESERVE_PER_CALL
    assert summary["reservation_statuses"] == ["unknown"]
    outcome_id = gate.event_status(
        next(
            event_id
            for event_id, decision in _run_state(store).decisions.items()
            if decision.operation == "llm" and decision.allow
        )
    ).outcome_event_id
    assert outcome_id is not None
    assert gate.event_status(outcome_id).reason_code == "usage_missing"

    # The held unknown visibly consumes cap: the next call needs 45 more and
    # 45 held + 45 new > 80, so it is denied before the model runs.
    inner = _RecordingInner()
    adapter = GatekeeperLLM(
        inner,
        gate,
        model_name="gpt-4o-mini",
        max_output_tokens=32,
        token_counter=_fixed_token_counter_for_tests_only,
    )
    with pytest.raises(GatekeeperDenied) as info:
        adapter.with_structured_output(NewsDigest).invoke("again")
    assert info.value.reason == "budget_exceeded"
    assert inner.calls == 0
