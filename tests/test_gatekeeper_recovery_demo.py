"""Offline recovery demo: simulated abandonment, expired-lease takeover, resume.

Two tests show, with the real graph, the real Gatekeeper, a fake model,
patched fetch and scratch audit/checkpoint files:
(A) a run whose first model call raises mid-graph is abandoned in memory
    (NOT an operating-system kill), the injected clock passes the lease TTL,
    a new owner takes the expired lease over, the held unknown reservation is
    reconciled exactly once with caller-supplied figures, and a rebuilt graph
    resumes the same checkpoint thread to completion; and
(B) a live, unexpired lease refuses both acquire and takeover by another
    owner, appending nothing.

Honest limits of this demo:
- The "crash" is simulated: in-memory instances are discarded and an injected
  clock advances past the lease TTL. No real process is killed.
- Resume is at-least-once at the graph level: the failed node runs again.
  What is shown once-only is the settlement of the failed attempt's ONE
  reservation. That is a different claim from once-only execution of the
  node, and nothing here shows exactly-once external side effects.
- Reconciled usage figures are caller-supplied (framed as
  operator/provider-recovered); after a real process death the true figures
  are unknowable inside the demo.
- All model calls and token figures are fake, writes go to NullProfileWriter,
  and nothing fences a genuinely live old worker. The documented open risk
  (a failed-write report after takeover can allow a second external write)
  is untouched: this demo does not go near the write path.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
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
from market_intelligence.gatekeeper import (
    Gatekeeper,
    GatekeeperLeaseError,
    GatekeeperRequestError,
)
from market_intelligence.graph import build_graph, checkpoint_serde
from market_intelligence.lifecycle import plan_recovery
from market_intelligence.net import FetchResult
from market_intelligence.persistence import NullProfileWriter

OWNER_A = "00000000-0000-4000-8000-aaaaaaaaaaaa"
OWNER_B = "00000000-0000-4000-8000-bbbbbbbbbbbb"
T0 = datetime(2026, 11, 15, 0, 0, 0, tzinfo=timezone.utc)
RUN = "run-recovery-demo"
TTL = 300
# As in the G7 demo: the audit run id and the graph checkpoint thread id are
# the same string by deliberate test construction, not an automatic binding.
THREAD = "g8-recovery"

FEED_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Test feed</title>
<item><title>T1</title><description>content one is a sufficiently long RSS excerpt for summarization</description></item>
</channel></rss>"""
PAGE_HTML = b"<html><body>Acme corp is a leading widget maker.</body></html>"

# A full graph run makes five structured model calls: one news digest, one
# topics call, one tone call and two per-side comparison drafts.
DEMO_CALLS = 5
# Fixed fake counter (10 prompt tokens), margin 0.25, no overhead, output 32:
# reserve = ceil(10 * 1.25) + 32 = 45 per call.
RESERVE_PER_CALL = 45
# Fixed fake usage per call: 10 prompt + 5 completion = 15 spent per call.
USAGE_PER_CALL = (10, 5)
# The graph fans out after fetch_news: topics and news tone run in parallel
# branches (supervisor.py), so at most TWO reservations are ever held at once
# by construction; the two comparison drafts are sequential invokes inside one
# node after the join. The worst budget point is therefore spent-at-fan-out
# plus two concurrent reservations: (15 reconciled + 15 digest) + 2 * 45 = 120.
# Cap 150 keeps a 30-token margin over that grounded worst point instead of
# assuming an unconstrained schedule. G7 already showed cap-edge denial; this
# demo is about recovery accounting. The observed peak is printed per phase.
CAP = 150
EXPECTED_CALLS = 1 + DEMO_CALLS  # the failed first attempt plus the retry run
EXPECTED_SPENT = (1 + DEMO_CALLS) * sum(USAGE_PER_CALL)  # 90


def _policy(**overrides):
    values = dict(
        max_run_tokens=CAP,
        max_run_llm_calls=12,
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


def _events(store):
    rows = store._conn.execute(
        "SELECT payload FROM audit_events ORDER BY seq"
    ).fetchall()
    return [
        control.event_from_payload(json.loads(row[0]), f"seq {index}")
        for index, row in enumerate(rows, start=1)
    ]


def _run_state(store):
    return ledger.replay_events(_events(store)).runs[RUN]


def _event_count(store):
    return store._conn.execute("SELECT count(*) FROM audit_events").fetchone()[0]


def _max_concurrent_reservations(events):
    """Observed peak of simultaneously held llm reservations, from the chain.

    A reservation opens at its allowed llm decision. It closes at the outcome
    that applies real usage; a usage_missing outcome leaves it held (unknown)
    until a reconciliation event referencing that unknown outcome closes it.
    This is a printed observation of one particular run, not a schedule
    guarantee; the cap above is chosen so the test does not depend on it.
    """
    held = set()
    unknown_to_decision = {}
    peak = 0
    for event in events:
        if (
            event.event_kind == "decision"
            and event.operation == "llm"
            and event.decision == "allow"
        ):
            held.add(event.event_id)
            peak = max(peak, len(held))
        elif event.event_kind == "outcome" and event.operation == "llm":
            ref = event.references_event_id
            if event.reason_code == "reconciliation":
                decision_id = unknown_to_decision.get(ref)
                if decision_id is not None:
                    held.discard(decision_id)
            elif event.usage_missing:
                if ref in held:
                    unknown_to_decision[event.event_id] = ref
            else:
                held.discard(ref)
    return peak


def _summary(store, phase):
    """Replay the audit store and print a readable per-phase summary."""
    events = _events(store)
    state = ledger.replay_events(events).runs[RUN]
    llm_decisions = [
        decision
        for decision in state.decisions.values()
        if decision.operation == "llm"
    ]
    allowed = sum(1 for decision in llm_decisions if decision.allow)
    denied = sum(1 for decision in llm_decisions if not decision.allow)
    spent = state.prompt_tokens + state.completion_tokens
    lease = state.lease
    summary = {
        "phase": phase,
        "run_id": RUN,
        "thread_id": THREAD,
        "calls": state.calls,
        "prompt_tokens": state.prompt_tokens,
        "completion_tokens": state.completion_tokens,
        "spent_tokens": spent,
        "reserved_tokens": state.reserved_tokens,
        "remaining_tokens": max(0, CAP - spent - state.reserved_tokens),
        "allowed_decisions": allowed,
        "denied_decisions": denied,
        "reservation_statuses": sorted(
            reservation.status for reservation in state.reservations.values()
        ),
        "unknown_outcomes": len(state.unknown_outcomes),
        "reconciled": len(state.reconciled),
        "lease_owner": lease.owner_id if lease else None,
        "lease_epoch": lease.epoch if lease else None,
        "lease_expires_at": lease.expires_at if lease else None,
        "lease_released": lease.released if lease else None,
        "max_concurrent_reservations": _max_concurrent_reservations(events),
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


def test_recovery_demo_end_to_end(tmp_path, monkeypatch):
    _mock_io(monkeypatch)
    _demo_settings(monkeypatch)
    policy = _policy()
    clock = _Clock()
    store = control.AuditStore(tmp_path / "audit.db")
    db_path = str(tmp_path / "checkpoints.sqlite")
    try:
        # Phase A: the first model call raises mid-graph. A plain RuntimeError
        # is not transient (retry.py), so exactly one attempt is made and the
        # reservation is left held with usage unknown.
        gate_a = _make_gate(store, THREAD, policy=policy, owner=OWNER_A, clock=clock)
        model_a = _UsageLLM(_responses(), error=RuntimeError("model failed"))
        conn = sqlite3.connect(db_path, check_same_thread=False)
        try:
            graph_a = build_graph(
                checkpointer=SqliteSaver(conn, serde=checkpoint_serde()),
                llm=model_a,
                writer=NullProfileWriter(),
                gatekeeper=gate_a,
                token_counter=_fixed_token_counter_for_tests_only,
            )
            with pytest.raises(RuntimeError):
                graph_a.invoke(_inputs(), _config(THREAD))
            snap = graph_a.get_state(_config(THREAD))
            print("after failure, graph next:", snap.next)
        finally:
            conn.close()
        assert len(model_a.calls) == 1
        failed = _summary(store, "A_failed_first_call")
        assert failed["allowed_decisions"] == 1
        assert failed["denied_decisions"] == 0
        assert failed["spent_tokens"] == 0
        assert failed["reserved_tokens"] == RESERVE_PER_CALL
        assert failed["reservation_statuses"] == ["unknown"]
        assert failed["unknown_outcomes"] == 1
        assert failed["reconciled"] == 0

        # While the lease is live and unexpired, another owner can neither
        # acquire nor take over, and nothing is appended by the refusals.
        gate_b = Gatekeeper(
            store,
            policy,
            run_id=RUN,
            thread_id=THREAD,
            owner_id=OWNER_B,
            lease_ttl_seconds=TTL,
            now=clock,
        )
        before = _event_count(store)
        with pytest.raises(GatekeeperLeaseError):
            gate_b.acquire_lease(str(uuid.uuid4()))
        with pytest.raises(GatekeeperLeaseError):
            gate_b.take_over_lease(str(uuid.uuid4()))
        assert _event_count(store) == before
        assert plan_recovery(gate_b.current_lease(), clock(), "approved") == "wait_live"

        # Simulated abandonment: gate_a and graph_a are simply not used again.
        # This is in-memory instance abandonment with an injected clock, NOT
        # an operating-system kill; nothing is released, and a genuinely live
        # old worker is not fenced.
        clock.adv(TTL + 1)
        assert plan_recovery(gate_b.current_lease(), clock(), "approved") == "takeover"

        # Phase B: the new owner takes the expired lease over (epoch 2).
        taken = gate_b.take_over_lease(str(uuid.uuid4()))
        assert taken.status == "applied"
        lease = gate_b.current_lease()
        assert lease.state == "present"
        assert lease.owner_id == OWNER_B
        assert lease.epoch == 2
        assert lease.released is False
        after_takeover = _summary(store, "B_lease_taken_over")
        assert after_takeover["lease_owner"] == OWNER_B
        assert after_takeover["lease_epoch"] == 2
        assert after_takeover["reserved_tokens"] == RESERVE_PER_CALL

        # The held unknown usage is reconciled exactly once with
        # caller-supplied figures (framed as operator/provider-recovered;
        # after a real crash the true figures are unknowable here).
        unknown_id = next(iter(_run_state(store).unknown_outcomes))
        settled = gate_b.reconcile_llm_outcome(unknown_id, 10, 5, str(uuid.uuid4()))
        assert settled.status == "applied"
        reconciled = _summary(store, "C_unknown_usage_reconciled")
        assert reconciled["spent_tokens"] == sum(USAGE_PER_CALL)
        assert reconciled["reserved_tokens"] == 0
        assert reconciled["reservation_statuses"] == ["resolved"]
        assert reconciled["unknown_outcomes"] == 1
        assert reconciled["reconciled"] == 1

        # A second reconciliation of the same unknown outcome (new event id,
        # since reusing an id is an idempotent confirm, not a second attempt)
        # is refused and appends nothing.
        before = _event_count(store)
        with pytest.raises(GatekeeperRequestError):
            gate_b.reconcile_llm_outcome(unknown_id, 10, 5, str(uuid.uuid4()))
        assert _event_count(store) == before

        # Resume: a new graph object over the same checkpoint file, with a
        # working fake model. Whether LangGraph re-runs only the pending node
        # or more after an exception is its own behaviour; because the
        # failure was injected on the FIRST model call, the retry makes the
        # same five calls under either shape.
        model_b = _UsageLLM(_responses())
        conn = sqlite3.connect(db_path, check_same_thread=False)
        try:
            graph_b = build_graph(
                checkpointer=SqliteSaver(conn, serde=checkpoint_serde()),
                llm=model_b,
                writer=NullProfileWriter(),
                gatekeeper=gate_b,
                token_counter=_fixed_token_counter_for_tests_only,
            )
            graph_b.invoke(None, _config(THREAD))
            snap = graph_b.get_state(_config(THREAD))
            assert snap.next == ("review_profile",)
            graph_b.invoke(Command(resume={"approved": True}), _config(THREAD))
            snap = graph_b.get_state(_config(THREAD))
            assert snap.next == ()
            assert _values(snap)["status"] == "complete"
        finally:
            conn.close()
        assert len(model_b.calls) == DEMO_CALLS

        gate_b.release_lease(str(uuid.uuid4()))
        final = _summary(store, "D_resumed_and_complete")

        # The audit's allowed llm decisions: the failed attempt first, then
        # the retry's digest; topics and tone fan out in parallel so their
        # relative order is schedule-dependent; the two side drafts follow
        # the join.
        state = _run_state(store)
        schemas = [
            decision.schema_name
            for decision in state.decisions.values()
            if decision.operation == "llm" and decision.allow
        ]
        assert schemas[:2] == ["NewsDigest", "NewsDigest"]
        assert sorted(schemas[2:4]) == ["NewsToneDraft", "TopicAnalysis"]
        assert schemas[4:] == ["SideComparisonDraft", "SideComparisonDraft"]

        # Final accounting, per attempt: the reconciled caller-supplied usage
        # of the failed attempt (15) plus the five retry calls (5 x 15) = 90.
        # The failed attempt's ONE reservation was settled exactly once. The
        # logical node ran twice: at-least-once execution, shown as such.
        assert final["calls"] == EXPECTED_CALLS
        assert final["allowed_decisions"] == EXPECTED_CALLS
        assert final["denied_decisions"] == 0
        assert final["spent_tokens"] == EXPECTED_SPENT
        assert final["reserved_tokens"] == 0
        assert final["remaining_tokens"] == CAP - EXPECTED_SPENT
        assert final["reservation_statuses"] == sorted(
            ["applied"] * DEMO_CALLS + ["resolved"]
        )
        assert final["unknown_outcomes"] == 1
        assert final["reconciled"] == 1
        assert final["lease_owner"] == OWNER_B
        assert final["lease_epoch"] == 2
        assert final["lease_released"] is True
        assert final["max_concurrent_reservations"] >= 1
    finally:
        store.close()


def test_live_lease_refuses_acquire_and_takeover(tmp_path):
    policy = _policy()
    clock = _Clock()
    store = control.AuditStore(tmp_path / "audit.db")
    try:
        _make_gate(store, THREAD, policy=policy, owner=OWNER_A, clock=clock)
        other = Gatekeeper(
            store,
            policy,
            run_id=RUN,
            thread_id=THREAD,
            owner_id=OWNER_B,
            lease_ttl_seconds=TTL,
            now=clock,
        )
        before = _event_count(store)
        with pytest.raises(GatekeeperLeaseError):
            other.acquire_lease(str(uuid.uuid4()))
        with pytest.raises(GatekeeperLeaseError):
            other.take_over_lease(str(uuid.uuid4()))
        assert _event_count(store) == before
    finally:
        store.close()
