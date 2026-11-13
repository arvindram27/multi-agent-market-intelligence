"""Offline tests for the Gatekeeper-backed structured model boundary."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import InMemorySaver

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
from market_intelligence.gatekeeper import Gatekeeper, GatekeeperLeaseError
from market_intelligence.gatekeeper_llm import GatekeeperDenied, GatekeeperLLM
from market_intelligence.graph import build_graph, checkpoint_serde
from market_intelligence.net import FetchResult
from market_intelligence.persistence import NullProfileWriter

T0 = datetime(2026, 11, 14, 0, 0, 0, tzinfo=timezone.utc)
RUN = "run-1"
THREAD = "thread-1"
TTL = 300
NO_USAGE = object()
FEED_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Test feed</title>
<item><title>T1</title><description>content one is a sufficiently long RSS excerpt for summarization</description></item>
</channel></rss>"""
PAGE_HTML = b"<html><body>Acme corp is a leading widget maker.</body></html>"


def cfg(**overrides):
    values = dict(
        max_run_tokens=20000,
        max_run_llm_calls=24,
        max_output_tokens=512,
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


class Clock:
    def __init__(self):
        self.t = T0

    def __call__(self):
        return self.t

    def adv(self, seconds):
        self.t += timedelta(seconds=seconds)


@pytest.fixture
def store(tmp_path):
    value = control.AuditStore(tmp_path / "audit.db")
    yield value
    value.close()


def make_gate(store, clock=None, owner=None, policy=None, ttl=TTL):
    gate = Gatekeeper(
        store,
        policy or cfg(),
        run_id=RUN,
        thread_id=THREAD,
        owner_id=owner or str(uuid.uuid4()),
        lease_ttl_seconds=ttl,
        now=clock or Clock(),
    )
    gate.start_run(str(uuid.uuid4()))
    gate.acquire_lease(str(uuid.uuid4()))
    return gate


def run_state(store):
    rows = store._conn.execute(
        "SELECT payload FROM audit_events ORDER BY seq"
    ).fetchall()
    events = [
        control.event_from_payload(json.loads(row[0]), f"seq {index}")
        for index, row in enumerate(rows, start=1)
    ]
    return ledger.replay_events(events).runs[RUN]


def fake_token_counter_for_tests_only(prompt):
    return 10


class FakeRunnable:
    def __init__(self, model):
        self.model = model

    def invoke(self, prompt, **kwargs):
        self.model.calls += 1
        if self.model.before_return is not None:
            self.model.before_return()
        if self.model.error is not None:
            raise self.model.error
        if self.model.usage is not NO_USAGE:
            usage = self.model.usage
            if isinstance(usage, tuple):
                usage = {
                    "prompt_tokens": usage[0],
                    "completion_tokens": usage[1],
                }
            response = SimpleNamespace(
                llm_output={"token_usage": usage},
                generations=[],
            )
            for callback in kwargs.get("config", {}).get("callbacks", []):
                callback.on_llm_end(response)
        return self.model.result


class FakeAdapterModel:
    def __init__(
        self,
        result="ok",
        error=None,
        usage=NO_USAGE,
        before_return=None,
    ):
        self.result = result
        self.error = error
        self.usage = usage
        self.before_return = before_return
        self.calls = 0

    def with_structured_output(self, schema, **kwargs):
        return FakeRunnable(self)


def make_adapter(gate, inner=None, counter=fake_token_counter_for_tests_only, max_output_tokens=512):
    inner = inner or FakeAdapterModel()
    return GatekeeperLLM(
        inner,
        gate,
        model_name="gpt-4o-mini",
        max_output_tokens=max_output_tokens,
        token_counter=counter,
    )


def test_denied_call_does_not_reach_model(store):
    gate = make_gate(store, policy=cfg(max_run_llm_calls=1))
    inner = FakeAdapterModel()
    adapter = make_adapter(gate, inner)
    adapter.with_structured_output(NewsDigest).invoke("first")
    with pytest.raises(GatekeeperDenied):
        adapter.with_structured_output(NewsDigest).invoke("second")
    assert inner.calls == 1


def test_missing_counter_fails_closed(store):
    gate = make_gate(store)
    inner = FakeAdapterModel()
    adapter = make_adapter(gate, inner, counter=None)
    with pytest.raises(GatekeeperDenied) as info:
        adapter.with_structured_output(NewsDigest).invoke("prompt")
    assert info.value.reason == "tokenizer_unavailable"
    assert inner.calls == 0


def test_callback_usage_is_recorded(store):
    gate = make_gate(store)
    adapter = make_adapter(gate, FakeAdapterModel(usage=(30, 20)))
    assert adapter.with_structured_output(NewsDigest).invoke("prompt") == "ok"
    state = run_state(store)
    assert (state.prompt_tokens, state.completion_tokens, state.reserved_tokens) == (
        30,
        20,
        0,
    )


def test_absent_usage_is_unknown_not_zero(store):
    gate = make_gate(store)
    adapter = make_adapter(gate, FakeAdapterModel(usage=NO_USAGE))
    adapter.with_structured_output(NewsDigest).invoke("prompt")
    state = run_state(store)
    assert state.reserved_tokens > 0
    reservation = next(iter(state.reservations.values()))
    assert reservation.status == "unknown"
    outcome_id = adapter.events[-1][1]
    assert outcome_id is not None
    assert gate.event_status(outcome_id).reason_code == "usage_missing"
    # Test-supplied usage closes the unknown; this is not provider proof.
    gate.reconcile_llm_outcome(outcome_id, 5, 5, str(uuid.uuid4()))
    state = run_state(store)
    assert (state.prompt_tokens, state.completion_tokens, state.reserved_tokens) == (
        5,
        5,
        0,
    )


def test_explicit_zero_usage_is_recorded_as_zero(store):
    gate = make_gate(store)
    adapter = make_adapter(gate, FakeAdapterModel(usage=(0, 0)))
    adapter.with_structured_output(NewsDigest).invoke("prompt")
    state = run_state(store)
    reservation = next(iter(state.reservations.values()))
    assert (state.prompt_tokens, state.completion_tokens, state.reserved_tokens) == (
        0,
        0,
        0,
    )
    assert reservation.status == "applied"


@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_tokens": -1, "completion_tokens": 5},
        {"prompt_tokens": True, "completion_tokens": 5},
        {"prompt_tokens": 5.0, "completion_tokens": 1},
        {"prompt_tokens": 5},
    ],
)
def test_invalid_reported_usage_is_never_zero(store, usage):
    gate = make_gate(store)
    adapter = make_adapter(gate, FakeAdapterModel(usage=usage))
    adapter.with_structured_output(NewsDigest).invoke("prompt")
    state = run_state(store)
    assert state.reserved_tokens > 0
    reservation = next(iter(state.reservations.values()))
    assert reservation.status == "unknown"
    outcome_id = adapter.events[-1][1]
    assert gate.event_status(outcome_id).reason_code == "usage_missing"


def test_model_exception_records_unknown_outcome_and_holds_the_reservation(store):
    gate = make_gate(
        store,
        policy=cfg(
            max_run_tokens=900,
            max_output_tokens=100,
            reservation_margin=0.0,
        ),
    )
    error = RuntimeError("model failed")
    inner = FakeAdapterModel(error=error)
    adapter = make_adapter(
        gate,
        inner,
        counter=lambda prompt: 400,
        max_output_tokens=100,
    )
    with pytest.raises(RuntimeError) as info:
        adapter.with_structured_output(NewsDigest).invoke("prompt")
    assert info.value is error
    assert inner.calls == 1
    decision_id, outcome_id = adapter.events[0]
    assert outcome_id is not None
    state = run_state(store)
    assert state.decisions[decision_id].allow is True
    assert gate.event_status(decision_id).outcome_event_id == outcome_id
    assert gate.event_status(outcome_id).reason_code == "usage_missing"
    reservation = next(iter(state.reservations.values()))
    assert reservation.status == "unknown"
    # Held, not freed and not recorded as zero spend.
    assert state.reserved_tokens > 0
    assert (state.prompt_tokens, state.completion_tokens) == (0, 0)
    with pytest.raises(GatekeeperDenied):
        adapter.with_structured_output(NewsDigest).invoke("again")


def test_failed_call_unknown_outcome_can_be_reconciled(store):
    gate = make_gate(store)
    adapter = make_adapter(gate, FakeAdapterModel(error=RuntimeError("model failed")))
    with pytest.raises(RuntimeError):
        adapter.with_structured_output(NewsDigest).invoke("prompt")
    outcome_id = adapter.events[0][1]
    assert outcome_id is not None
    assert run_state(store).reserved_tokens > 0
    # Test-supplied usage closes the unknown; this is not provider proof.
    gate.reconcile_llm_outcome(outcome_id, 5, 5, str(uuid.uuid4()))
    state = run_state(store)
    assert (state.prompt_tokens, state.completion_tokens, state.reserved_tokens) == (
        5,
        5,
        0,
    )


def test_model_exception_recording_failure_does_not_replace_the_original(store):
    clock = Clock()
    gate = make_gate(store, clock=clock)
    error = RuntimeError("model failed")
    inner = FakeAdapterModel(error=error, before_return=lambda: clock.adv(301))
    adapter = make_adapter(gate, inner)
    with pytest.raises(RuntimeError) as info:
        adapter.with_structured_output(NewsDigest).invoke("prompt")
    assert info.value is error
    assert inner.calls == 1
    decision_id, outcome_id = adapter.events[0]
    assert outcome_id is None
    assert gate.event_status(decision_id).outcome_event_id is None
    assert run_state(store).reserved_tokens > 0


def test_each_failed_invocation_holds_its_own_reservation(store):
    gate = make_gate(store)
    inner = FakeAdapterModel(error=RuntimeError("model failed"))
    adapter = make_adapter(gate, inner)
    held = []
    for _ in range(3):
        with pytest.raises(RuntimeError):
            adapter.with_structured_output(NewsDigest).invoke("prompt")
        held.append(run_state(store).reserved_tokens)
    assert inner.calls == 3
    assert held[0] > 0
    assert held == [held[0], 2 * held[0], 3 * held[0]]
    assert len({decision for decision, _ in adapter.events}) == 3
    assert len({outcome for _, outcome in adapter.events}) == 3
    assert all(outcome is not None for _, outcome in adapter.events)
    state = run_state(store)
    assert [r.status for r in state.reservations.values()] == ["unknown"] * 3


def test_outcome_failure_surfaces_without_success(store):
    clock = Clock()
    gate = make_gate(store, clock=clock)
    inner = FakeAdapterModel(
        usage=(10, 10),
        before_return=lambda: clock.adv(301),
    )
    adapter = make_adapter(gate, inner)
    with pytest.raises(GatekeeperLeaseError):
        adapter.with_structured_output(NewsDigest).invoke("prompt")
    assert inner.calls == 1
    assert adapter.events[-1][1] is None
    state = run_state(store)
    decision_id = adapter.events[-1][0]
    assert gate.event_status(decision_id).outcome_event_id is None
    assert state.reserved_tokens > 0


def test_replay_after_clean_release_counts_earlier_usage(tmp_path):
    policy = cfg(
        max_run_tokens=700,
        max_output_tokens=100,
        reservation_margin=0.0,
    )
    store = control.AuditStore(tmp_path / "audit.db")
    try:
        gate_a = make_gate(store, policy=policy)
        adapter_a = make_adapter(
            gate_a,
            FakeAdapterModel(usage=(200, 200)),
            counter=lambda prompt: 400,
            max_output_tokens=100,
        )
        assert adapter_a.with_structured_output(NewsDigest).invoke("prompt") == "ok"
        gate_a.release_lease(str(uuid.uuid4()))

        gate_b = Gatekeeper(
            store,
            policy,
            run_id=RUN,
            thread_id=THREAD,
            owner_id=str(uuid.uuid4()),
            lease_ttl_seconds=TTL,
            now=Clock(),
        )
        gate_b.acquire_lease(str(uuid.uuid4()))
        inner_b = FakeAdapterModel()
        adapter_b = make_adapter(
            gate_b,
            inner_b,
            counter=lambda prompt: 400,
            max_output_tokens=100,
        )
        with pytest.raises(GatekeeperDenied):
            adapter_b.with_structured_output(NewsDigest).invoke("prompt")
        assert inner_b.calls == 0
    finally:
        store.close()

    fresh_store = control.AuditStore(tmp_path / "fresh.db")
    try:
        fresh_gate = make_gate(fresh_store, policy=policy)
        inner_fresh = FakeAdapterModel()
        adapter_fresh = make_adapter(
            fresh_gate,
            inner_fresh,
            counter=lambda prompt: 400,
            max_output_tokens=100,
        )
        assert (
            adapter_fresh.with_structured_output(NewsDigest).invoke("prompt")
            == "ok"
        )
        assert inner_fresh.calls == 1
    finally:
        fresh_store.close()


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


def _fake_fetch(url, **kwargs):
    if "feed" in url:
        return FetchResult(url, FEED_XML, "application/rss+xml", 200)
    return FetchResult(url, PAGE_HTML, "text/html", 200)


def test_graph_denial_stops_at_the_real_boundary(
    store, fake_llm, monkeypatch
):
    monkeypatch.setattr(settings, "openai_model", "gpt-4o-mini")
    monkeypatch.setattr(settings, "llm_max_output_tokens", 512)
    monkeypatch.setattr(news_module, "fetch_bytes", _fake_fetch)
    monkeypatch.setattr(competitor_module, "fetch_bytes", _fake_fetch)
    gate = make_gate(store, policy=cfg(max_run_llm_calls=1))
    model = fake_llm(_responses())
    graph = build_graph(
        checkpointer=InMemorySaver(serde=checkpoint_serde()),
        llm=model,
        writer=NullProfileWriter(),
        gatekeeper=gate,
        token_counter=fake_token_counter_for_tests_only,
    )

    with pytest.raises(GatekeeperDenied):
        graph.invoke(_inputs(), {"configurable": {"thread_id": "g5b-graph"}})

    assert len(model.calls) == 1
    state = run_state(store)
    llm_decisions = [
        decision
        for decision in state.decisions.values()
        if decision.operation == "llm"
    ]
    assert sum(1 for decision in llm_decisions if decision.allow) == 1
    assert sum(1 for decision in llm_decisions if not decision.allow) >= 1
