"""Offline tests for the G4b lifecycle coordinator (real Gatekeeper, temporary audit store, fake clock).

No LangGraph, no network, no app or UI. Pause and resume are stub callables.
"""
import json
import os
import pathlib
import uuid
from datetime import datetime, timedelta, timezone

import pytest

import market_intelligence.admission as adm
import market_intelligence.control as control
import market_intelligence.gatekeeper as gk
from market_intelligence.gatekeeper import (
    Gatekeeper,
    GatekeeperLeaseError,
    GatekeeperPolicyMismatch,
)
from market_intelligence.lifecycle import (
    _REFUSALS,
    Coordinator,
    IntentJournal,
    LifecycleBlocked,
    LifecycleError,
    LifecycleJournalError,
    Segment,
    derive_event_id,
    plan_recovery,
)
from market_intelligence.gatekeeper import LeaseStatus

T0 = datetime(2026, 11, 13, 0, 0, 0, tzinfo=timezone.utc)
RUN = "run-1"
TTL = 300
_STORES = []


class Clock:
    def __init__(self):
        self.t = T0

    def __call__(self):
        return self.t

    def adv(self, seconds):
        self.t += timedelta(seconds=seconds)


class Boom(RuntimeError):
    pass


def cfg(**overrides):
    values = dict(
        max_run_tokens=20000, max_run_llm_calls=24, max_output_tokens=512,
        max_fetches_per_run=8, reservation_margin=0.25, provider_overhead_tokens=0,
        allowed_model="gpt-4o-mini", allowed_schemas=frozenset({"NewsDigest"}),
    )
    values.update(overrides)
    return adm.PolicyConfig(**values)


@pytest.fixture(autouse=True)
def _close():
    yield
    while _STORES:
        _STORES.pop().close()


def store_at(path):
    s = control.AuditStore(path)
    _STORES.append(s)
    return s


def coord(tmp_path, store=None, clock=None, name="j1", fault=None, policy=None):
    store = store or store_at(tmp_path / "audit.db")
    clock = clock or Clock()
    c = Coordinator(
        store, policy or cfg(), IntentJournal(tmp_path / f"{name}.jsonl"),
        run_id=RUN, thread_id="thread-1", lease_ttl_seconds=TTL, now=clock, fault=fault,
    )
    return c, store, clock


def kinds(store):
    rows = store._conn.execute("SELECT payload FROM audit_events ORDER BY seq").fetchall()
    return [json.loads(r[0])["event_kind"] for r in rows]


def count(store):
    return store._conn.execute("SELECT count(*) FROM audit_events").fetchone()[0]


def owners(store):
    rows = store._conn.execute("SELECT payload FROM audit_events ORDER BY seq").fetchall()
    return [(json.loads(r[0])["event_kind"], json.loads(r[0]).get("lease_owner"), json.loads(r[0]).get("lease_epoch")) for r in rows]


def paused_run(tmp_path, **kw):
    c, store, clock = coord(tmp_path, **kw)
    assert c.run_first_segment(lambda: "paused") == "paused"
    return c, store, clock


def test_event_id_is_stable_uuid_and_owner_free():
    a = derive_event_id(RUN, "acquire", 2)
    assert a == derive_event_id(RUN, "acquire", 2)
    assert str(uuid.UUID(a)) == a
    assert len({a, derive_event_id(RUN, "acquire", 3), derive_event_id(RUN, "release", 2),
                derive_event_id(RUN, "takeover", 2), derive_event_id("run-2", "acquire", 2),
                derive_event_id(RUN, "start")}) == 6


def test_journal_is_append_only_and_torn_line_fails_closed(tmp_path, monkeypatch):
    j = IntentJournal(tmp_path / "j.jsonl")
    j.append({"op": "decision", "value": "approved"})
    IntentJournal(tmp_path / "j.jsonl").append({"op": "resume_finished"})
    assert [r["op"] for r in j.records()] == ["decision", "resume_finished"]

    def forbidden(*a, **k):
        raise AssertionError("the journal must never replace, rename or delete")

    for name in ("replace", "rename", "remove", "unlink"):
        monkeypatch.setattr(os, name, forbidden)
    monkeypatch.setattr(pathlib.Path, "unlink", forbidden)
    c, store, clock = coord(tmp_path, name="j2")
    c.run_first_segment(lambda: "paused")
    with open(tmp_path / "j.jsonl", "ab") as h:
        h.write(b'{"op": "dec')
    with pytest.raises(LifecycleJournalError):
        j.records()


def test_first_segment_starts_acquires_epoch_1_and_releases_once_on_pause(tmp_path):
    c, store, clock = paused_run(tmp_path)
    assert kinds(store) == ["run_started", "lease_acquired", "lease_released"]
    st = c.status()
    assert (st.state, st.released, st.epoch) == ("present", True, 1)


def test_approved_resume_uses_new_owner_and_epoch_from_audit_then_releases(tmp_path):
    c, store, clock = paused_run(tmp_path)
    calls = []
    assert c.decide({"approved": True}) == "approved"
    assert c.run_approved(lambda: calls.append(1) or "finished") == "finished"
    ev = owners(store)
    assert [e[0] for e in ev] == ["run_started", "lease_acquired", "lease_released", "lease_acquired", "lease_released"]
    assert ev[1][1] != ev[3][1] and ev[3][2] == 2 and ev[4][1] == ev[3][1]
    assert calls == [1]
    assert c.run_approved(lambda: calls.append(1) or "finished") == "already_finished"
    assert calls == [1] and count(store) == 5


def test_rejected_and_invalid_never_touch_the_lease(tmp_path):
    c, store, clock = paused_run(tmp_path, name="a")
    before = count(store)
    assert c.decide("nope") == "invalid"
    assert c.decide({"approved": "yes"}) == "invalid"
    assert c.decide({}) == "invalid"
    assert c.decision() is None
    assert c.run_approved(lambda: pytest.fail("resume must not run")) == "no_approved_decision"
    assert c.decide({"approved": False}) == "rejected"
    assert c.run_approved(lambda: pytest.fail("resume must not run")) == "no_approved_decision"
    with pytest.raises(LifecycleBlocked):
        c.acquire(needs_approved_decision=True)
    assert count(store) == before
    with pytest.raises(LifecycleBlocked):
        c.decide({"approved": True})


def test_acquire_needs_a_recorded_approval_and_a_started_run(tmp_path):
    c, store, clock = coord(tmp_path)
    with pytest.raises(LifecycleBlocked):
        c.acquire(needs_approved_decision=False)  # run_absent
    assert count(store) == 0
    c.start()
    with pytest.raises(LifecycleBlocked):
        c.acquire(needs_approved_decision=True)
    assert kinds(store) == ["run_started"]


def test_epoch_follows_audit_history_across_takeover(tmp_path):
    c, store, clock = coord(tmp_path)
    c.start()
    s1 = c.acquire(needs_approved_decision=False)  # epoch 1, never released (crash)
    c.decide({"approved": True})
    clock.adv(TTL + 1)
    s2 = c.take_over()
    assert (s1.epoch, s2.epoch) == (1, 2) and s1.owner != s2.owner
    c.release(s2)
    s3 = c.acquire(needs_approved_decision=True)
    assert s3.epoch == 3
    assert [e[2] for e in owners(store) if e[2]] == [1, 2, 2, 3]
    assert derive_event_id(RUN, "acquire", 3) != derive_event_id(RUN, "acquire", 2)


def test_unknown_commit_is_confirmed_by_same_id_and_same_owner(tmp_path):
    def fault(point):
        if point == "after_call":
            raise Boom

    c, store, clock = coord(tmp_path, fault=fault)
    with pytest.raises(Boom):
        c.start()
    # the append committed, the journal says only "intent"
    assert kinds(store) == ["run_started"]
    c._fault = lambda point: None
    assert c.resume_pending() == [(derive_event_id(RUN, "start"), "already_applied")]
    assert kinds(store) == ["run_started"]
    assert c._open_intents() == []


def test_crash_between_intent_and_call_is_applied_once_on_retry(tmp_path):
    state = {"boom": True}

    def fault(point):
        if point == "after_intent" and state["boom"]:
            state["boom"] = False
            raise Boom

    c, store, clock = coord(tmp_path, fault=fault)
    with pytest.raises(Boom):
        c.start()
    assert count(store) == 0
    assert c.resume_pending() == [(derive_event_id(RUN, "start"), "applied")]
    assert kinds(store) == ["run_started"]


def test_open_intent_blocks_new_lease_calls_until_resolved(tmp_path):
    state = {"boom": False}

    def fault(point):
        if point == "after_intent" and state["boom"]:
            state["boom"] = False
            raise Boom

    c, store, clock = coord(tmp_path, fault=fault)
    c.start()
    state["boom"] = True
    with pytest.raises(Boom):
        c.acquire(needs_approved_decision=False)
    with pytest.raises(LifecycleBlocked):
        c.acquire(needs_approved_decision=False)
    c.resume_pending()
    assert kinds(store) == ["run_started", "lease_acquired"]


def test_losing_racer_ends_its_intent_and_never_retries_the_taken_id(tmp_path):
    a, store, clock = paused_run(tmp_path, name="a")
    a.decide({"approved": True})
    hits = {}

    def b_fault(point):
        if point == "after_intent" and "a" not in hits:
            hits["a"] = a.acquire(needs_approved_decision=True)  # A wins the same target epoch

    b, _, _ = coord(tmp_path, store=store, clock=clock, name="b", fault=b_fault)
    b.decide({"approved": True})
    with pytest.raises(control.AuditIncomplete):
        b.acquire(needs_approved_decision=True)
    assert count(store) == 4 and hits["a"].epoch == 2
    assert b._open_intents() == []
    assert b.resume_pending() == []
    assert plan_recovery(b.status(), clock(), "approved") == "wait_live"
    with pytest.raises(LifecycleBlocked):
        b.acquire(needs_approved_decision=True)
    assert count(store) == 4
    assert [r["state"] for r in b._journal.records() if r["op"] == "lease_call"] == ["intent", "superseded"]


def test_new_process_waits_while_live_then_takes_over_with_new_owner(tmp_path):
    a, store, clock = coord(tmp_path, name="a")
    a.start()
    dead = a.acquire(needs_approved_decision=False)  # process dies holding the lease
    b, _, _ = coord(tmp_path, store=store, clock=clock, name="b")
    b.decide({"approved": True})
    n = count(store)
    with pytest.raises(LifecycleBlocked):
        b.run_approved(lambda: pytest.fail("must not resume while the lease is live"))
    assert count(store) == n
    clock.adv(TTL + 1)
    ran = []
    assert b.run_approved(lambda: ran.append(1) or "finished") == "finished"
    ev = owners(store)
    assert ev[-2][0] == "lease_taken_over" and ev[-2][1] != dead.owner and ev[-2][2] == 2
    assert ev[-1][0] == "lease_released" and ran == [1]


def test_resume_is_at_least_once_when_the_stub_fails_after_acquire(tmp_path):
    c, store, clock = paused_run(tmp_path, name="a")
    c.decide({"approved": True})
    runs = []

    def failing():
        runs.append("a")
        raise Boom

    with pytest.raises(Boom):
        c.run_approved(failing)
    assert c.status().released is False
    d, _, _ = coord(tmp_path, store=store, clock=clock, name="a")  # same journal, new process
    clock.adv(TTL + 1)
    assert d.run_approved(lambda: runs.append("b") or "finished") == "finished"
    assert runs == ["a", "b"]  # replay is possible; nothing here promises once-only
    assert kinds(store)[-2:] == ["lease_taken_over", "lease_released"]


def test_release_crash_after_resume_finished_does_not_rerun_the_stub(tmp_path):
    state = {"boom": True}

    def fault(point):
        if point == "after_resume" and state["boom"]:
            state["boom"] = False
            raise Boom

    c, store, clock = paused_run(tmp_path, fault=fault)
    c.decide({"approved": True})
    runs = []
    with pytest.raises(Boom):
        c.run_approved(lambda: runs.append(1) or "finished")
    assert c.status().released is False
    assert c.run_approved(lambda: runs.append(2) or "finished") == "already_finished"
    assert runs == [1] and c.status().released is True


def test_snapshot_is_stale_and_only_the_gatekeeper_call_refuses(tmp_path):
    c, store, clock = coord(tmp_path, name="a")
    c.start()
    seg = c.acquire(needs_approved_decision=False)
    snap = c.status()
    assert c.still_holds(seg) is True
    clock.adv(TTL + 1)
    other, _, _ = coord(tmp_path, store=store, clock=clock, name="b")
    other.decide({"approved": True})
    other.take_over()
    assert snap.owner_id == seg.owner and snap.released is False  # the old value did not change
    n = count(store)
    old = Gatekeeper(store, cfg(), run_id=RUN, thread_id="thread-1", owner_id=seg.owner,
                     lease_ttl_seconds=TTL, now=clock)
    with pytest.raises(GatekeeperLeaseError):
        old.release_lease(str(uuid.uuid4()))
    assert count(store) == n
    assert c.still_holds(seg) is False
    with pytest.raises(LifecycleBlocked):
        c.release(seg)


def test_changed_policy_on_the_same_journal_is_refused_before_any_call(tmp_path):
    c, store, clock = paused_run(tmp_path, name="a")
    c.decide({"approved": True})
    d, _, _ = coord(tmp_path, store=store, clock=clock, name="a", policy=cfg(max_run_tokens=999))
    n = count(store)
    with pytest.raises(LifecycleJournalError):
        d.run_approved(lambda: pytest.fail("must not resume"))
    assert count(store) == n


def test_changed_policy_on_a_fresh_journal_is_refused_by_the_gatekeeper(tmp_path):
    c, store, clock = paused_run(tmp_path, name="a")
    d, _, _ = coord(tmp_path, store=store, clock=clock, name="fresh", policy=cfg(max_run_tokens=999))
    d.decide({"approved": True})
    n = count(store)
    with pytest.raises(GatekeeperPolicyMismatch):
        d.run_approved(lambda: pytest.fail("must not resume"))
    assert count(store) == n
    assert [r["state"] for r in d._journal.records() if r["op"] == "lease_call"] == ["intent", "superseded"]


def test_journal_of_another_run_or_without_header_is_refused(tmp_path):
    other = IntentJournal(tmp_path / "other.jsonl")
    other.append({"op": "header", "run_id": "run-2", "thread_id": "thread-1",
                  "lease_ttl_seconds": TTL, "policy_sha256": "x"})
    other.append({"op": "decision", "value": "approved"})
    other.append({"op": "resume_finished"})
    c, store, clock = coord(tmp_path, name="other")
    with pytest.raises(LifecycleJournalError):
        c.decision()
    with pytest.raises(LifecycleJournalError):
        c.run_approved(lambda: pytest.fail("must not resume"))
    bare = IntentJournal(tmp_path / "bare.jsonl")
    bare.append({"op": "decision", "value": "approved"})
    d, _, _ = coord(tmp_path, store=store, name="bare")
    with pytest.raises(LifecycleJournalError):
        d.decision()
    assert count(store) == 0


def test_recovery_table():
    def st(state, released=None, exp=None):
        return LeaseStatus(state=state, released=released, expires_at=exp)

    now = T0
    live = (T0 + timedelta(seconds=1)).isoformat()
    old = (T0 - timedelta(seconds=1)).isoformat()
    assert plan_recovery(st("verification_failed"), now, "approved") == "stop_unverifiable"
    assert plan_recovery(st("run_absent"), now, "approved") == "stop_run_absent"
    assert plan_recovery(st("present", False, live), now, "approved") == "wait_live"
    assert plan_recovery(st("present", False, old), now, "approved") == "takeover"
    assert plan_recovery(st("present", False, old), now, None) == "needs_owner"
    assert plan_recovery(st("present", True, old), now, "approved") == "acquire"
    assert plan_recovery(st("present", True, old), now, None) == "wait_for_decision"
    assert plan_recovery(st("no_lease"), now, "approved") == "acquire"
    assert plan_recovery(st("no_lease"), now, "rejected") == "wait_for_decision"


def test_refused_retry_of_an_open_intent_is_terminal(tmp_path):
    c, store, clock = coord(tmp_path)
    c.start()
    c.acquire(needs_approved_decision=False)  # a live lease exists
    stale_owner = str(uuid.uuid4())
    c._append({"op": "lease_call", "call": "acquire", "event_id": derive_event_id(RUN, "acquire", 2),
               "owner": stale_owner, "epoch": 2, "incarnation": c._incarnation, "state": "intent"})
    n = count(store)
    assert c.resume_pending() == [(derive_event_id(RUN, "acquire", 2), "superseded")]
    assert c._open_intents() == [] and c.resume_pending() == [] and count(store) == n


def crash_at(point, after=0):
    state = {"n": 0}

    def fault(p):
        if p == point:
            state["n"] += 1
            if state["n"] > after:
                raise Boom

    return fault


def test_new_coordinator_never_reuses_an_owner_from_an_open_acquire_intent(tmp_path):
    c1, store, clock = coord(tmp_path, name="a")
    c1.start()
    c1._fault = crash_at("after_intent")
    with pytest.raises(Boom):
        c1.acquire(needs_approved_decision=False)
    n = count(store)
    dead_owner = c1._open_intents()[0]["owner"]
    c2, _, _ = coord(tmp_path, store=store, clock=clock, name="a")
    assert c2.resume_pending() == []
    assert count(store) == n
    with pytest.raises(LifecycleBlocked):
        c2.acquire(needs_approved_decision=False)
    assert c2.resolve_foreign() == [(derive_event_id(RUN, "acquire", 1), "absent")]
    assert count(store) == n and c2._open_intents() == []
    seg = c2.acquire(needs_approved_decision=False)
    assert seg.owner != dead_owner and seg.epoch == 1
    assert dead_owner not in [e[1] for e in owners(store)]


def test_new_coordinator_never_reuses_an_owner_from_open_start_or_release_intents(tmp_path):
    c1, store, clock = coord(tmp_path, name="a")
    c1._fault = crash_at("after_call")
    with pytest.raises(Boom):
        c1.start()  # committed, journal says intent only
    c2, _, _ = coord(tmp_path, store=store, clock=clock, name="a")
    assert c2.resume_pending() == [] and kinds(store) == ["run_started"]
    with pytest.raises(LifecycleBlocked):
        c2.start()
    assert c2.resolve_foreign() == [(derive_event_id(RUN, "start"), "present")]
    seg = c2.acquire(needs_approved_decision=False)
    c2._fault = crash_at("after_intent")
    with pytest.raises(Boom):
        c2.release(seg)
    c3, _, _ = coord(tmp_path, store=store, clock=clock, name="a")
    n = count(store)
    assert c3.resume_pending() == []
    c3.decide({"approved": True})
    assert c3.resolve_foreign() == [(derive_event_id(RUN, "release", 1), "absent")]
    assert count(store) == n and c3.status().released is False  # old owner's lease is only left to expire
    with pytest.raises(LifecycleBlocked):
        c3.run_approved(lambda: pytest.fail("must not resume while the lease is live"))


def test_finished_branch_does_not_release_a_lease_held_by_an_older_incarnation(tmp_path):
    c1, store, clock = paused_run(tmp_path, name="a", fault=crash_at("after_resume"))
    c1.decide({"approved": True})
    with pytest.raises(Boom):
        c1.run_approved(lambda: "finished")
    assert c1.status().released is False
    n = count(store)
    c2, _, _ = coord(tmp_path, store=store, clock=clock, name="a")
    assert c2.run_approved(lambda: pytest.fail("must not resume")) == "already_finished"
    assert count(store) == n and c2.status().released is False


def test_paused_outcome_releases_without_marking_finished_and_can_be_redriven(tmp_path):
    c, store, clock = paused_run(tmp_path)
    c.decide({"approved": True})
    assert c.run_approved(lambda: "paused") == "paused"
    assert c.status().released is True
    assert not any(r["op"] == "resume_finished" for r in c._records())
    assert c.run_approved(lambda: "finished") == "finished"
    assert [e[2] for e in owners(store) if e[2]] == [1, 1, 2, 2, 3, 3]


def test_unexpected_stub_outcome_leaves_the_lease_held_and_not_finished(tmp_path):
    c, store, clock = paused_run(tmp_path)
    c.decide({"approved": True})
    with pytest.raises(LifecycleError):
        c.run_approved(lambda: "bogus")
    assert c.status().released is False
    assert not any(r["op"] == "resume_finished" for r in c._records())


def test_start_is_blocked_by_an_open_intent(tmp_path):
    c, store, clock = coord(tmp_path, fault=crash_at("after_intent"))
    with pytest.raises(Boom):
        c.start()
    with pytest.raises(LifecycleBlocked):
        c.start()
    assert count(store) == 0


def test_unknown_commit_outcome_stays_open_and_is_not_a_refusal(tmp_path, monkeypatch):
    c, store, clock = coord(tmp_path)
    c.start()
    real = gk.Gatekeeper.acquire_lease

    def unknown(self, event_id):
        real(self, event_id)  # the append really committed
        raise control.ControlUnavailable("commit outcome unknown", append_state="unknown")

    monkeypatch.setattr(gk.Gatekeeper, "acquire_lease", unknown)
    with pytest.raises(control.ControlUnavailable):
        c.acquire(needs_approved_decision=False)
    states = [r["state"] for r in c._records() if r["op"] == "lease_call" and r["call"] == "acquire"]
    assert states == ["intent"]
    assert len(c._own_open()) == 1
    monkeypatch.setattr(gk.Gatekeeper, "acquire_lease", real)
    assert c.resume_pending() == [(derive_event_id(RUN, "acquire", 1), "already_applied")]
    assert kinds(store) == ["run_started", "lease_acquired"]


def test_every_gatekeeper_refusal_class_is_listed_and_unknowns_are_not():
    refusals = {
        v for v in vars(gk).values()
        if isinstance(v, type) and issubclass(v, gk.GatekeeperError) and v is not gk.GatekeeperError
    }
    assert refusals <= set(_REFUSALS)
    assert control.ControlUnavailable not in _REFUSALS and gk.GatekeeperError not in _REFUSALS


def raw_gate(store, clock, owner):
    return Gatekeeper(store, cfg(), run_id=RUN, thread_id="thread-1", owner_id=owner,
                      lease_ttl_seconds=TTL, now=clock)


def test_acquire_records_the_actual_epoch_when_the_snapshot_was_stale(tmp_path):
    state = {"go": False}
    holder = {}

    def fault(point):
        if point == "after_intent" and state["go"]:
            state["go"] = False
            x = raw_gate(holder["store"], holder["clock"], str(uuid.uuid4()))
            x.acquire_lease(str(uuid.uuid4()))  # epoch 2, ids that are not derived
            x.release_lease(str(uuid.uuid4()))

    c, store, clock = paused_run(tmp_path, fault=fault)
    holder.update(store=store, clock=clock)
    c.decide({"approved": True})
    state["go"] = True
    seg = c.acquire(needs_approved_decision=True)  # read: released epoch 1, intended 2, accepted at 3
    assert seg.epoch == 3
    done = [r for r in c._records() if r["op"] == "lease_call" and r["call"] == "acquire" and r["state"] == "done"][-1]
    assert (done["epoch"], done["actual_epoch"]) == (2, 3)
    assert c.still_holds(seg) is True
    c.release(seg)
    assert [e[2] for e in owners(store) if e[2]][-2:] == [3, 3]
    assert c.status().released is True and c.status().epoch == 3


def test_unknown_commit_after_a_stale_epoch_acquire_confirms_the_same_actual_epoch(tmp_path):
    state = {"go": False, "boom": False}
    holder = {}

    def fault(point):
        if point == "after_intent" and state["go"]:
            state["go"] = False
            x = raw_gate(holder["store"], holder["clock"], str(uuid.uuid4()))
            x.acquire_lease(str(uuid.uuid4()))
            x.release_lease(str(uuid.uuid4()))
            state["boom"] = True
        if point == "after_call" and state["boom"]:
            state["boom"] = False
            raise Boom

    c, store, clock = paused_run(tmp_path, fault=fault)
    holder.update(store=store, clock=clock)
    c.decide({"approved": True})
    state["go"] = True
    with pytest.raises(Boom):
        c.acquire(needs_approved_decision=True)
    assert c.resume_pending() == [(derive_event_id(RUN, "acquire", 2), "already_applied")]
    seg = c._journal_segment()
    assert seg is not None and seg.epoch == 3 and c.still_holds(seg) is True
    n = count(store)
    c.release(seg)
    assert count(store) == n + 1


def test_takeover_records_the_actual_epoch_when_the_snapshot_was_stale(tmp_path):
    state = {"go": False}
    holder = {}

    def fault(point):
        if point == "after_intent" and state["go"]:
            state["go"] = False
            store, clock, dead = holder["store"], holder["clock"], holder["dead"]
            raw_gate(store, clock, dead.owner).release_lease(str(uuid.uuid4()))  # old owner releases late
            x = raw_gate(store, clock, str(uuid.uuid4()))
            x.acquire_lease(str(uuid.uuid4()))  # epoch 2, then it too expires
            clock.adv(TTL + 1)

    c, store, clock = coord(tmp_path, fault=fault)
    c.start()
    dead = c.acquire(needs_approved_decision=False)
    holder.update(store=store, clock=clock, dead=dead)
    c.decide({"approved": True})
    clock.adv(TTL + 1)
    state["go"] = True
    seg = c.take_over()  # read: expired epoch 1, intended 2, accepted at 3
    assert seg.epoch == 3 and seg.owner != dead.owner
    assert c.still_holds(seg) is True
    c.release(seg)
    assert c.status().released is True and c.status().epoch == 3


def test_stub_is_not_run_when_the_segment_no_longer_holds_after_acquire(tmp_path):
    c, store, clock = paused_run(tmp_path)
    c.decide({"approved": True})
    orig = c.acquire

    def acquire_then_expire(**kw):
        seg = orig(**kw)
        clock.adv(TTL + 1)
        return seg

    c.acquire = acquire_then_expire
    with pytest.raises(LifecycleBlocked):
        c.run_approved(lambda: pytest.fail("the stub must not run"))
    assert not any(r["op"] == "resume_finished" for r in c._records())


def test_acquire_is_not_recorded_done_when_the_chain_does_not_show_this_owner(tmp_path, monkeypatch):
    c, store, clock = coord(tmp_path)
    c.start()
    real = c._reader.event_status

    class Seen:
        state = "present"
        lease_owner = str(uuid.uuid4())
        lease_epoch = 1

    monkeypatch.setattr(c._reader, "event_status", lambda event_id: Seen)
    with pytest.raises(LifecycleError):
        c.acquire(needs_approved_decision=False)
    assert len(c._own_open()) == 1
    monkeypatch.setattr(c._reader, "event_status", real)
    assert c.resume_pending() == [(derive_event_id(RUN, "acquire", 1), "already_applied")]


def test_first_segment_stub_is_not_run_when_the_segment_no_longer_holds(tmp_path):
    c, store, clock = coord(tmp_path)
    orig = c.acquire

    def acquire_then_expire(**kw):
        seg = orig(**kw)
        clock.adv(TTL + 1)
        return seg

    c.acquire = acquire_then_expire
    with pytest.raises(LifecycleBlocked):
        c.run_first_segment(lambda: pytest.fail("the stub must not run"))
