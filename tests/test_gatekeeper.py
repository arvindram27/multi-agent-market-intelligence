"""Offline tests for the gatekeeper (slice 2b-iii, first batch).

They use the real AuditStore, the real LedgerProjection and the real admission
functions, with a fake clock. Failure injection wraps the store connection.
"""
import dataclasses
import hashlib
import json
import os
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone

import pytest

import market_intelligence.admission as adm
import market_intelligence.control as control
import market_intelligence.ledger as ledger
from market_intelligence.gatekeeper import (
    EventStatus,
    Gatekeeper,
    GatekeeperClockError,
    GatekeeperError,
    GatekeeperLeaseError,
    GatekeeperPolicyMismatch,
    GatekeeperRequestError,
    GatekeeperStateError,
    GatekeeperUnknownRun,
)

T0 = datetime(2026, 11, 13, 0, 0, 0, tzinfo=timezone.utc)
OWNER_A = "00000000-0000-0000-0000-00000000000a"
OWNER_B = "00000000-0000-0000-0000-00000000000b"
INT64 = 2**63 - 1


def eid():
    return str(uuid.uuid4())


_STORES = []


def open_store(path):
    store = control.AuditStore(path)
    _STORES.append(store)
    return store


@pytest.fixture(autouse=True)
def _close_every_store():
    """Close each store a test opened, so no connection outlives its test (Windows keeps files open)."""
    yield
    while _STORES:
        _STORES.pop().close()


class Clock:
    def __init__(self, start=T0):
        self.t = start

    def __call__(self):
        return self.t

    def adv(self, seconds):
        self.t += timedelta(seconds=seconds)


class NoClock:
    """A clock that fails the test when it is read."""

    def __call__(self):
        raise AssertionError("the injected clock must not be read")


class NoDefaultClock:
    """Stands in for control's datetime: any datetime.now call (make_event's default) fails the test."""

    def __init__(self, real):
        self._real = real

    def now(self, *args, **kwargs):
        raise AssertionError("control.make_event read the system clock")

    def __getattr__(self, name):
        return getattr(self._real, name)


@pytest.fixture
def no_system_clock(monkeypatch):
    monkeypatch.setattr(control, "datetime", NoDefaultClock(control.datetime))


def silence(gate):
    """Replace the gatekeeper's injected clock with one that fails when read (the field production code calls)."""
    gate._now = NoClock()


def cfg(**overrides):
    values = dict(
        max_run_tokens=20000,
        max_run_llm_calls=24,
        max_output_tokens=512,
        max_fetches_per_run=8,
        reservation_margin=0.25,
        provider_overhead_tokens=0,
        allowed_model="gpt-4o-mini",
        allowed_schemas=frozenset({"NewsDigest"}),
    )
    values.update(overrides)
    return adm.PolicyConfig(**values)


def make(path, clock=None, owner=OWNER_A, run_id="run-1", **policy):
    path.mkdir(parents=True, exist_ok=True)
    store = open_store(path / "audit.db")
    clock = clock or Clock()
    gate = Gatekeeper(
        store,
        cfg(**policy),
        run_id=run_id,
        thread_id="thread-1",
        owner_id=owner,
        lease_ttl_seconds=300,
        now=clock,
    )
    return store, clock, gate


def boot(path, **policy):
    store, clock, gate = make(path, **policy)
    gate.start_run(eid())
    gate.acquire_lease(eid())
    return store, clock, gate


def other(store, clock, **kw):
    values = dict(
        run_id="run-1", thread_id="thread-1", owner_id=OWNER_A, lease_ttl_seconds=300, now=clock
    )
    policy = kw.pop("policy", cfg())
    values.update(kw)
    return Gatekeeper(store, policy, **values)


def req(count=100, **kw):
    values = dict(
        model="gpt-4o-mini",
        schema_name="NewsDigest",
        serialized_body_token_count=count,
        max_output_tokens=512,
    )
    values.update(kw)
    return adm.LLMRequest(**values)


def verify(store):
    assert control.verify_chain(store.path) == []
    assert ledger.verify_ledger(store.path) == []


def count(store):
    return store._conn.execute("SELECT count(*) FROM audit_events").fetchone()[0]


def last_event(store):
    row = store._conn.execute(
        "SELECT payload FROM audit_events ORDER BY seq DESC LIMIT 1"
    ).fetchone()
    return control.event_from_payload(json.loads(row[0]))


class Wrap:
    """Delegating connection that fails or lets COMMIT through, then fails."""

    def __init__(self, conn, mode):
        self._c = conn
        self.mode = mode

    def __getattr__(self, name):
        return getattr(self._c, name)

    def execute(self, sql, *args):
        if sql.strip().upper() == "COMMIT":
            if self.mode == "landed":
                self._c.execute("COMMIT")
            raise sqlite3.OperationalError("injected commit failure")
        return self._c.execute(sql, *args)


def fingerprint_of_files(directory):
    out = {}
    for name in sorted(os.listdir(directory)):
        full = os.path.join(directory, name)
        with open(full, "rb") as handle:
            out[name] = (os.stat(full).st_size, os.stat(full).st_mtime_ns, hashlib.sha256(handle.read()).hexdigest())
    return out


# ---------------------------------------------------------------- happy path


def test_happy_path_is_verified_by_both_verifiers(tmp_path):
    store, clock, gate = boot(tmp_path)
    clock.adv(10)
    decision_id = eid()
    result = gate.admit_llm(req(), decision_id)
    assert result.status == "applied" and result.decision.allow
    assert result.decision.reserve_tokens == 637
    clock.adv(5)
    assert gate.record_llm_outcome(decision_id, 98, 40, eid()).status == "applied"
    verify(store)


def test_constructor_rejects_bad_arguments(tmp_path):
    store = open_store(tmp_path / "x.db")
    good = dict(run_id="r", thread_id="t", owner_id=OWNER_A, lease_ttl_seconds=300, now=Clock())
    with pytest.raises(GatekeeperRequestError):
        Gatekeeper(object(), cfg(), **good)
    with pytest.raises(GatekeeperRequestError):
        Gatekeeper(store, object(), **good)
    with pytest.raises(GatekeeperRequestError):
        Gatekeeper(store, cfg(), **{**good, "lease_ttl_seconds": 0})
    with pytest.raises(GatekeeperRequestError):
        Gatekeeper(store, cfg(), **{**good, "lease_ttl_seconds": True})
    with pytest.raises(GatekeeperRequestError):
        Gatekeeper(store, cfg(), **{**good, "now": 5})
    with pytest.raises(GatekeeperRequestError):
        Gatekeeper(store, cfg(), **{**good, "run_id": ""})


def test_unknown_run_and_double_start(tmp_path):
    store, clock, gate = make(tmp_path)
    with pytest.raises(GatekeeperUnknownRun):
        gate.acquire_lease(eid())
    assert count(store) == 0
    gate.start_run(eid())
    with pytest.raises(GatekeeperStateError):
        gate.start_run(eid())


def test_naive_clock_is_refused(tmp_path):
    store, clock, gate = make(tmp_path)
    gate.start_run(eid())
    clock.t = datetime(2026, 11, 13, 0, 0, 0)
    n = count(store)
    with pytest.raises(GatekeeperClockError):
        gate.acquire_lease(eid())
    assert count(store) == n


# ---------------------------------------------------------------- duplicates


def test_duplicate_of_every_method_after_state_change_and_expiry(tmp_path, no_system_clock):
    store, clock, gate = make(tmp_path)
    ids = {name: eid() for name in ("start", "acquire", "renew", "release", "decision", "outcome")}
    gate.start_run(ids["start"])
    clock.adv(1)
    gate.acquire_lease(ids["acquire"])
    clock.adv(10)
    first = gate.admit_llm(req(), ids["decision"])
    clock.adv(1)
    gate.record_llm_outcome(ids["decision"], 98, 40, ids["outcome"])
    clock.adv(1)
    gate.renew_lease(ids["renew"])
    clock.adv(1)
    gate.release_lease(ids["release"])
    clock.adv(100000)
    silence(gate)
    n = count(store)
    calls = (
        lambda: gate.start_run(ids["start"]),
        lambda: gate.acquire_lease(ids["acquire"]),
        lambda: gate.renew_lease(ids["renew"]),
        lambda: gate.release_lease(ids["release"]),
        lambda: gate.admit_llm(req(), ids["decision"]),
        lambda: gate.record_llm_outcome(ids["decision"], 98, 40, ids["outcome"]),
    )
    for call in calls:
        assert call().status == "already_applied"
    again = gate.admit_llm(req(), ids["decision"])
    assert again.decision == first.decision and again.event_hash == first.event_hash
    assert count(store) == n
    verify(store)


def test_duplicate_rederives_from_the_state_before_the_stored_event(tmp_path):
    store, clock, gate = boot(tmp_path, max_run_tokens=1500)
    clock.adv(1)
    first, second = eid(), eid()
    assert gate.admit_llm(req(100), first).decision.allow
    assert gate.admit_llm(req(100), second).decision.allow
    third = gate.admit_llm(req(100), eid())
    assert not third.decision.allow  # budget now used up
    again = gate.admit_llm(req(100), first)
    assert again.status == "already_applied" and again.decision.allow


def test_duplicate_with_different_request_fields_is_rejected_where_stored(tmp_path):
    store, clock, gate = boot(tmp_path)
    clock.adv(10)
    decision_id = eid()
    gate.admit_llm(req(100), decision_id)
    for changed in (req(101), req(100, model="other-model"), req(100, schema_name="Other")):
        with pytest.raises(control.AuditIncomplete):
            gate.admit_llm(changed, decision_id)
    clock.adv(1)
    outcome_id = eid()
    gate.record_llm_outcome(decision_id, 98, 40, outcome_id)
    with pytest.raises(control.AuditIncomplete):
        gate.record_llm_outcome(decision_id, 98, 41, outcome_id)
    with pytest.raises(control.AuditIncomplete):
        gate.record_llm_outcome(decision_id, None, None, outcome_id)
    with pytest.raises(GatekeeperRequestError):
        gate.record_llm_outcome(eid(), 98, 40, outcome_id)
    verify(store)


def test_max_output_tokens_is_not_provable_on_a_duplicate(tmp_path):
    store, clock, gate = boot(tmp_path)
    clock.adv(10)
    decision_id = eid()
    gate.admit_llm(req(100, max_output_tokens=512), decision_id)
    # unit-level statement of a limit: a different value that still re-derives the
    # same decision is accepted as the same request
    assert gate.admit_llm(req(100, max_output_tokens=100), decision_id).status == "already_applied"
    # a value that would re-derive a different decision is rejected
    with pytest.raises(control.AuditIncomplete):
        gate.admit_llm(req(100, max_output_tokens=9999), decision_id)


def test_duplicate_with_wrong_kind_or_run_is_refused_before_append(tmp_path):
    store, clock, gate = boot(tmp_path)
    clock.adv(10)
    decision_id = eid()
    gate.admit_llm(req(), decision_id)
    n = count(store)
    with pytest.raises(GatekeeperRequestError):
        gate.acquire_lease(decision_id)
    with pytest.raises(GatekeeperRequestError):
        other(store, clock, run_id="run-2").admit_llm(req(), decision_id)
    with pytest.raises(GatekeeperRequestError):
        other(store, clock, thread_id="thread-2").admit_llm(req(), decision_id)
    assert count(store) == n


def test_duplicate_policy_owner_and_ttl_mismatch(tmp_path):
    store, clock, gate = boot(tmp_path)
    clock.adv(10)
    decision_id, outcome_id = eid(), eid()
    gate.admit_llm(req(), decision_id)
    clock.adv(1)
    gate.record_llm_outcome(decision_id, 98, 40, outcome_id)
    lease_id = store._conn.execute(
        "SELECT event_id FROM audit_events WHERE payload LIKE '%lease_acquired%'"
    ).fetchone()[0]
    bad = other(store, clock, policy=cfg(reservation_margin=0.5))
    for call in (
        lambda: bad.admit_llm(req(), decision_id),
        lambda: bad.record_llm_outcome(decision_id, 98, 40, outcome_id),
        lambda: bad.acquire_lease(eid()),
    ):
        with pytest.raises(GatekeeperPolicyMismatch):
            call()
    stranger = other(store, clock, owner_id=OWNER_B)
    with pytest.raises(GatekeeperRequestError):
        stranger.admit_llm(req(), decision_id)
    with pytest.raises(GatekeeperRequestError):
        stranger.record_llm_outcome(decision_id, 98, 40, outcome_id)
    with pytest.raises(control.AuditIncomplete):
        stranger.acquire_lease(lease_id)
    with pytest.raises(GatekeeperPolicyMismatch):
        other(store, clock, lease_ttl_seconds=60).acquire_lease(lease_id)


def test_duplicate_validates_the_current_projection(tmp_path):
    store, clock, gate = boot(tmp_path)
    clock.adv(10)
    decision_id = eid()
    gate.admit_llm(req(), decision_id)
    store._conn.execute("UPDATE run_ledger SET calls = calls + 1")
    with pytest.raises(ledger.LedgerProjectionMismatch):
        gate.admit_llm(req(), decision_id)


# ---------------------------------------------------------- policy and clock


def test_policy_mismatch_raises_before_append(tmp_path):
    store, clock, gate = boot(tmp_path)
    clock.adv(10)
    n = count(store)
    with pytest.raises(GatekeeperPolicyMismatch):
        other(store, clock, policy=cfg(reservation_margin=0.5)).admit_llm(req(), eid())
    with pytest.raises(GatekeeperPolicyMismatch):
        other(store, clock, policy=cfg(allowed_schemas=frozenset({"A", "B"}))).admit_llm(req(), eid())
    assert count(store) == n


def test_clock_must_not_move_behind_any_event_of_the_run(tmp_path):
    store, clock, gate = boot(tmp_path)
    clock.adv(100)
    gate.admit_llm(req(), eid())
    clock.adv(-50)
    n = count(store)
    with pytest.raises(GatekeeperClockError):
        gate.admit_llm(req(), eid())
    with pytest.raises(GatekeeperClockError):
        gate.renew_lease(eid())
    assert count(store) == n
    clock.adv(50)  # equal to the last event is allowed
    gate.admit_llm(req(), eid())


def test_expired_missing_or_foreign_lease_raises_without_a_write(tmp_path):
    store, clock, gate = make(tmp_path)
    gate.start_run(eid())
    n = count(store)
    with pytest.raises(GatekeeperLeaseError):
        gate.admit_llm(req(), eid())  # no lease yet
    gate.acquire_lease(eid())
    n = count(store)
    with pytest.raises(GatekeeperLeaseError):
        other(store, clock, owner_id=OWNER_B).admit_llm(req(), eid())
    clock.adv(1000)
    with pytest.raises(GatekeeperLeaseError):
        gate.admit_llm(req(), eid())
    assert count(store) == n


def test_tokenizer_unavailable_writes_a_deny_without_a_reservation(tmp_path):
    store, clock, gate = boot(tmp_path)
    clock.adv(10)
    result = gate.admit_llm(req(None), eid())
    assert not result.decision.allow
    assert result.decision.reason is adm.AdmissionReason.TOKENIZER_UNAVAILABLE
    event = last_event(store)
    assert event.reason_code == "tokenizer_unavailable" and event.serialized_body_token_count is None
    n = count(store)
    with pytest.raises(GatekeeperRequestError):
        gate.admit_llm(req(None, model="not-allowed"), eid())
    assert count(store) == n
    verify(store)


def test_denials_write_deny_events(tmp_path):
    store, clock, gate = boot(tmp_path, max_run_tokens=700)
    clock.adv(10)
    assert gate.admit_llm(req(100), eid()).decision.allow
    budget = gate.admit_llm(req(100), eid())
    assert budget.decision.reason is adm.AdmissionReason.BUDGET_EXCEEDED
    policy = gate.admit_llm(req(100, model="other"), eid())
    assert policy.decision.reason is adm.AdmissionReason.POLICY_DENIED
    assert last_event(store).decision == "deny"
    verify(store)


# --------------------------------------------------------------- outcomes


def outcome_case(path, prompt, completion, **policy):
    store, clock, gate = boot(path, **policy)
    clock.adv(10)
    decision_id = eid()
    gate.admit_llm(req(), decision_id)
    clock.adv(5)
    gate.record_llm_outcome(decision_id, prompt, completion, eid())
    verify(store)
    return store, last_event(store)


def test_outcome_reasons(tmp_path):
    assert outcome_case(tmp_path / "ok", 98, 40)[1].reason_code == "allowed"
    event = outcome_case(tmp_path / "over", 600, 100)[1]
    assert event.reason_code == "reservation_overrun" and event.overrun_tokens == 63
    missing = outcome_case(tmp_path / "missing", None, None)[1]
    assert missing.reason_code == "usage_missing" and missing.usage_prompt_tokens is None
    shapes = [(-1, 5), (True, 5), (5.0, 1), ("5", 1), (INT64 + 1, 0), (1, None), (None, 1), (INT64, 1)]
    for i, (prompt, completion) in enumerate(shapes):
        event = outcome_case(tmp_path / f"bad{i}", prompt, completion)[1]
        assert event.reason_code == "usage_out_of_range", (prompt, completion)
        assert event.error_class == "UsageOutOfRange"
        assert event.usage_prompt_tokens is None and event.usage_completion_tokens is None
        assert dict(event.counters) == {} and event.reservation_released == 0


def test_unusable_usage_keeps_the_reservation_held_and_unknown(tmp_path):
    store, event = outcome_case(tmp_path, -1, 5)
    row = store._conn.execute("SELECT reserved_tokens FROM run_ledger").fetchone()
    status = store._conn.execute("SELECT status FROM reservations").fetchone()
    assert row[0] == 637 and status[0] == "unknown"


def test_budget_breach_arithmetic(tmp_path):
    event = outcome_case(tmp_path, 19000, 1500)[1]
    assert event.reason_code == "budget_breach" and event.overrun_tokens == 20500 - 637
    assert event.counters["budget_deficit"] == 500 and event.counters["remaining_tokens"] == 0


def test_small_usage_after_an_earlier_breach_must_also_be_a_breach(tmp_path):
    store, clock, gate = boot(tmp_path, max_run_tokens=2000)
    clock.adv(1)
    first, second = eid(), eid()
    assert gate.admit_llm(req(100), first).decision.allow
    assert gate.admit_llm(req(100), second).decision.allow
    clock.adv(1)
    gate.record_llm_outcome(first, 1500, 700, eid())
    clock.adv(1)
    gate.record_llm_outcome(second, 1, 1, eid())
    event = last_event(store)
    assert event.reason_code == "budget_breach" and event.overrun_tokens is None
    verify(store)


def test_outcome_refuses_unknown_denied_or_closed_decisions(tmp_path):
    store, clock, gate = boot(tmp_path)
    clock.adv(10)
    denied = gate.admit_llm(req(None), eid())
    allowed = eid()
    gate.admit_llm(req(), allowed)
    clock.adv(1)
    n = count(store)
    with pytest.raises(GatekeeperRequestError):
        gate.record_llm_outcome(eid(), 1, 1, eid())
    with pytest.raises(GatekeeperRequestError):
        gate.record_llm_outcome(denied.event_id, 1, 1, eid())
    gate.record_llm_outcome(allowed, 1, 1, eid())
    with pytest.raises(GatekeeperRequestError):
        gate.record_llm_outcome(allowed, 1, 1, eid())  # second outcome, new id
    assert count(store) == n + 1


# ------------------------------------------------------------------- lease


def test_renew_extends_and_release_ends_the_lease(tmp_path):
    store, clock, gate = boot(tmp_path)
    clock.adv(100)
    gate.renew_lease(eid())
    clock.adv(250)  # past the first expiry, inside the renewed one
    assert gate.admit_llm(req(), eid()).decision.allow
    gate.release_lease(eid())
    with pytest.raises(GatekeeperLeaseError):
        gate.admit_llm(req(), eid())
    verify(store)


def test_expired_lease_cannot_be_renewed_but_can_be_released_and_handed_over(tmp_path):
    store, clock, gate = boot(tmp_path)
    clock.adv(10)
    decision_id = eid()
    gate.admit_llm(req(), decision_id)
    clock.adv(1000)  # the lease expired with the reservation still open
    n = count(store)
    with pytest.raises(GatekeeperLeaseError):
        gate.renew_lease(eid())
    with pytest.raises(GatekeeperLeaseError):
        gate.record_llm_outcome(decision_id, 98, 40, eid())
    assert count(store) == n
    gate.release_lease(eid())  # release after expiry with an open reservation
    second = other(store, clock, owner_id=OWNER_B)
    clock.adv(1)
    second.acquire_lease(eid())
    clock.adv(1)
    assert second.record_llm_outcome(decision_id, 98, 40, eid()).status == "applied"
    verify(store)


def test_same_owner_cannot_reacquire_and_only_one_live_lease(tmp_path):
    store, clock, gate = boot(tmp_path)
    with pytest.raises(GatekeeperLeaseError):
        gate.acquire_lease(eid())
    gate.release_lease(eid())
    with pytest.raises(GatekeeperLeaseError):
        gate.acquire_lease(eid())


def test_chain_itself_rejects_renewal_after_expiry(tmp_path):
    # unit level: the gatekeeper refuses first, so this builds the event by hand
    store, clock, gate = boot(tmp_path)
    clock.adv(1000)
    observed = clock()
    event = control.make_event(
        run_id="run-1",
        thread_id="thread-1",
        policy_version=control.POLICY_VERSION_H1_SLICE2,
        operation="run",
        event_kind="lease_renewed",
        decision="allow",
        reason_code="allowed",
        counters={},
        lease_owner=OWNER_A,
        lease_epoch=1,
        observed_now=observed.isoformat(),
        lease_ttl_seconds=300,
        lease_expires_at=(observed + timedelta(seconds=300)).isoformat(),
    )
    with pytest.raises(control.AuditIncomplete):
        with store.transaction(ledger.LedgerProjection()) as tx:
            tx.append_in_tx(event)


# ------------------------------------------------------------ commit failure


@pytest.mark.parametrize("mode,expected", [("landed", "already_applied"), ("absent", "applied")])
def test_unknown_commit_then_same_id_call_is_execute_or_confirm(tmp_path, mode, expected):
    store, clock, gate = boot(tmp_path)
    clock.adv(10)
    real = store._conn
    decision_id = eid()
    store._conn = Wrap(real, mode)
    with pytest.raises(control.ControlUnavailable) as caught:
        gate.admit_llm(req(), decision_id)
    assert caught.value.append_state == "unknown"
    store._conn = real
    clock.adv(2)
    assert gate.admit_llm(req(), decision_id).status == expected
    assert store._conn.execute(
        "SELECT count(*) FROM audit_events WHERE event_id = ?", (decision_id,)
    ).fetchone()[0] == 1
    verify(store)


@pytest.mark.parametrize("method", ["acquire", "renew", "release"])
@pytest.mark.parametrize("mode,expected", [("landed", "already_applied"), ("absent", "applied")])
def test_unknown_commit_on_each_lease_method(tmp_path, method, mode, expected):
    store, clock, gate = make(tmp_path)
    gate.start_run(eid())
    if method != "acquire":
        gate.acquire_lease(eid())
    clock.adv(10)
    call = {"acquire": gate.acquire_lease, "renew": gate.renew_lease, "release": gate.release_lease}[method]
    event_id = eid()
    real = store._conn
    store._conn = Wrap(real, mode)
    with pytest.raises(control.ControlUnavailable) as caught:
        call(event_id)
    assert caught.value.append_state == "unknown"
    store._conn = real
    status = gate.event_status(event_id)
    assert status.state == ("present" if mode == "landed" else "absent")
    clock.adv(2)
    assert call(event_id).status == expected
    verify(store)


# ------------------------------------------------------------- event status


def test_event_status_present_absent_and_nothing_is_written_or_read(tmp_path):
    store, clock, gate = boot(tmp_path)
    clock.adv(10)
    decision_id = eid()
    gate.admit_llm(req(), decision_id)
    n = count(store)
    silence(gate)
    status = gate.event_status(decision_id)
    assert isinstance(status, EventStatus)
    assert status.state == "present" and status.event_kind == "decision"
    assert status.run_matches is True and status.reason_code == "allowed" and status.seq == n
    assert gate.event_status(eid()).state == "absent"
    assert count(store) == n


def test_event_status_after_an_unknown_commit_both_ways(tmp_path):
    for mode, expected in (("landed", "present"), ("absent", "absent")):
        store, clock, gate = boot(tmp_path / mode)
        clock.adv(10)
        real = store._conn
        decision_id = eid()
        store._conn = Wrap(real, mode)
        with pytest.raises(control.ControlUnavailable):
            gate.admit_llm(req(), decision_id)
        store._conn = real
        n = count(store)
        assert gate.event_status(decision_id).state == expected
        assert count(store) == n


def test_event_status_other_run_and_empty_store(tmp_path):
    store, clock, gate = boot(tmp_path)
    clock.adv(10)
    decision_id = eid()
    gate.admit_llm(req(), decision_id)
    status = other(store, clock, run_id="run-2").event_status(decision_id)
    assert status.state == "present" and status.run_matches is False
    _, _, empty = make(tmp_path / "empty")
    assert empty.event_status(eid()).state == "absent"


def test_event_status_changes_no_file_for_an_empty_or_used_store(tmp_path):
    for name, used in (("empty", False), ("used", True)):
        directory = tmp_path / name
        if used:
            store, clock, gate = boot(directory)
            clock.adv(10)
            gate.admit_llm(req(), eid())
        else:
            store, clock, gate = make(directory)
        before = fingerprint_of_files(directory)
        assert gate.event_status(eid()).state == "absent"
        assert gate.event_status(eid()).state == "absent"
        assert fingerprint_of_files(directory) == before
        assert sorted(before) == ["audit.db"]  # no journal, wal, shm or second file


def test_event_status_on_empty_store_does_not_create_ledger_tables(tmp_path):
    store, clock, gate = make(tmp_path)
    gate.event_status(eid())
    names = {row[0] for row in store._conn.execute("SELECT name FROM sqlite_master")}
    assert names.isdisjoint(set(ledger.COLUMNS))


def test_event_status_verification_failures_are_not_absent(tmp_path):
    store, clock, gate = boot(tmp_path)
    clock.adv(10)
    decision_id = eid()
    gate.admit_llm(req(), decision_id)
    store._conn.execute("UPDATE run_ledger SET calls = calls + 5")
    status = gate.event_status(eid())
    assert status.state == "verification_failed" and status.problems
    store._conn.execute("UPDATE run_ledger SET calls = calls - 5")
    assert gate.event_status(decision_id).state == "present"
    store._conn.execute(
        "UPDATE audit_events SET payload = replace(payload, 'gpt-4o-mini', 'gpt-4o-minx') WHERE event_id = ?",
        (decision_id,),
    )
    assert gate.event_status(eid()).state == "verification_failed"


def test_event_status_schema_problems_are_verification_failed(tmp_path):
    cases = {
        "ledger table missing": "DROP TABLE reservations",
        "ledger table malformed": "ALTER TABLE run_ledger ADD COLUMN extra INTEGER",
        "extra index": "CREATE INDEX stray ON run_ledger (calls)",
    }
    for name, statement in cases.items():
        store, clock, gate = boot(tmp_path / name.replace(" ", "_"))
        clock.adv(10)
        gate.admit_llm(req(), eid())
        store._conn.execute(statement)
        status = gate.event_status(eid())
        assert status.state == "verification_failed", name
        assert status.problems, name


def test_event_status_audit_schema_problems(tmp_path):
    store, clock, gate = boot(tmp_path / "a")
    store._conn.execute("ALTER TABLE audit_head ADD COLUMN extra INTEGER")
    assert gate.event_status(eid()).state == "verification_failed"
    store2, clock2, gate2 = boot(tmp_path / "b")
    store2._conn.execute("DROP TABLE audit_head")
    assert gate2.event_status(eid()).state == "verification_failed"


def test_event_status_empty_chain_with_partial_ledger_tables_is_not_absent(tmp_path):
    store, clock, gate = make(tmp_path)
    store._conn.execute(ledger.RUN_LEDGER_SQL)
    assert gate.event_status(eid()).state == "verification_failed"


def test_event_status_empty_chain_with_complete_empty_ledger_is_absent(tmp_path):
    store, clock, gate = make(tmp_path)
    for statement in ledger.SCHEMA_STATEMENTS:
        store._conn.execute(statement)
    assert gate.event_status(eid()).state == "absent"


def test_event_status_missing_ledger_tables_for_a_non_empty_chain_fail(tmp_path):
    store, clock, gate = make(tmp_path)
    ev = control.make_event(
        run_id="run-1", thread_id="thread-1", operation="llm", event_kind="decision",
        decision="deny", reason_code="policy_denied", model="gpt-4o-mini", schema_name="NewsDigest",
    )
    store.append(ev)  # a legacy audit-only event: chain exists, no ledger tables
    assert gate.event_status(eid()).state == "verification_failed"


def test_event_status_corrupt_chain_is_not_absent(tmp_path):
    store, clock, gate = boot(tmp_path)
    store._conn.execute("UPDATE audit_events SET payload = payload || ' ' WHERE seq = 1")
    assert gate.event_status(eid()).state == "verification_failed"


def test_event_status_busy_database_is_verification_failed_then_recovers(tmp_path):
    store, clock, gate = boot(tmp_path)
    clock.adv(10)
    decision_id = eid()
    gate.admit_llm(req(), decision_id)
    blocker = sqlite3.connect(str(store.path), isolation_level=None, timeout=0.0)
    blocker.execute("BEGIN EXCLUSIVE")
    try:
        status = gate.event_status(decision_id)
    finally:
        blocker.execute("ROLLBACK")
        blocker.close()
    assert status.state == "verification_failed"
    assert gate.event_status(decision_id).state == "present"


def test_event_status_missing_file_is_verification_failed(tmp_path):
    store, clock, gate = boot(tmp_path)
    gate._db_path = tmp_path / "gone.db"
    status = gate.event_status(eid())
    assert status.state == "verification_failed"
    assert not (tmp_path / "gone.db").exists()


def test_event_status_uses_the_open_file_even_after_a_directory_change(tmp_path, monkeypatch):
    base = tmp_path / "work"
    other_dir = tmp_path / "elsewhere"
    base.mkdir()
    other_dir.mkdir()
    monkeypatch.chdir(base)
    store = open_store("rel.db")
    clock = Clock()
    gate = Gatekeeper(
        store, cfg(), run_id="run-1", thread_id="thread-1", owner_id=OWNER_A,
        lease_ttl_seconds=300, now=clock,
    )
    gate.start_run(eid())
    gate.acquire_lease(eid())
    clock.adv(10)
    decision_id = eid()
    gate.admit_llm(req(), decision_id)
    monkeypatch.chdir(other_dir)
    assert gate.event_status(decision_id).state == "present"
    assert os.listdir(other_dir) == []  # no second database file appeared


@pytest.mark.parametrize("name", ["with space", "hash#mark", "percent%25sign", "ampersand&q", "uni\u00e9dir"])
def test_event_status_path_quoting(tmp_path, name):
    store, clock, gate = boot(tmp_path / name)
    clock.adv(10)
    decision_id = eid()
    gate.admit_llm(req(), decision_id)
    assert gate.event_status(decision_id).state == "present"
    assert sorted(os.listdir(tmp_path / name)) == ["audit.db"]


# ------------------------------------------------------------ unit level


def test_unit_run_view_matches_the_ledger_counters(tmp_path):
    store, clock, gate = boot(tmp_path)
    clock.adv(10)
    decision_id = eid()
    gate.admit_llm(req(), decision_id)
    state = ledger.replay_events(
        [control.event_from_payload(json.loads(r[0])) for r in store._conn.execute("SELECT payload FROM audit_events ORDER BY seq")]
    )
    view = Gatekeeper._view(state.runs["run-1"])
    assert (view.calls, view.reserved_tokens, view.prompt_tokens, view.completion_tokens) == (1, 637, 0, 0)
    assert view.fetch_count == 0 and view.approved is False


@pytest.mark.parametrize("margin", [0.0, 0.25, 0.1, 1.0, 1 / 3])
@pytest.mark.parametrize("overhead", [0, 7])
@pytest.mark.parametrize("tokens", [0, 3, 100, 2**53 + 1])
def test_unit_reserve_parity_between_admission_and_ledger(tmp_path, margin, overhead, tokens):
    store, clock, gate = boot(
        tmp_path / f"{margin}-{overhead}-{tokens}".replace(".", "_"),
        reservation_margin=margin,
        provider_overhead_tokens=overhead,
        max_run_tokens=2**62,
    )
    clock.adv(1)
    result = gate.admit_llm(req(tokens), eid())
    assert result.decision.allow
    verify(store)  # the ledger replay would reject a reserve that differs


def test_gatekeeper_errors_share_one_base():
    for cls in (
        GatekeeperRequestError, GatekeeperUnknownRun, GatekeeperStateError,
        GatekeeperPolicyMismatch, GatekeeperLeaseError, GatekeeperClockError,
    ):
        assert issubclass(cls, GatekeeperError)


def test_module_has_no_network_tokenizer_or_clock_imports():
    import market_intelligence.gatekeeper as module

    source = open(module.__file__, encoding="utf-8").read()
    for word in ("requests", "socket", "urllib", "tiktoken", "openai", "time.time", "datetime.now", "utcnow"):
        assert word not in source


def test_counter_deficit_beyond_int64_is_out_of_range(tmp_path):
    store, clock, gate = boot(tmp_path, max_run_tokens=10000)
    clock.adv(1)
    first, second = eid(), eid()
    assert gate.admit_llm(req(100), first).decision.allow
    assert gate.admit_llm(req(100), second).decision.allow
    clock.adv(1)
    gate.record_llm_outcome(first, INT64 // 2, INT64 // 2, eid())
    assert last_event(store).reason_code == "budget_breach"
    clock.adv(1)
    gate.record_llm_outcome(second, INT64 // 2, 1, eid())  # each sum fits, the deficit does not
    assert last_event(store).reason_code == "usage_out_of_range"
    verify(store)


def test_unit_event_status_opens_the_file_read_only_and_skips_a_missing_file(tmp_path, monkeypatch):
    # UNIT LEVEL: SQLite would not let a read-only open write, so this pins the URI mode and the guard directly.
    import market_intelligence.gatekeeper as module

    store, clock, gate = boot(tmp_path)
    seen = []
    real_connect = sqlite3.connect

    def spy(target, *args, **kwargs):
        seen.append((target, kwargs))
        return real_connect(target, *args, **kwargs)

    monkeypatch.setattr(module.sqlite3, "connect", spy)
    assert gate.event_status(eid()).state == "absent"
    files = [item for item in seen if item[0] != ":memory:"]  # schema_problems builds an in-memory reference
    assert len(files) == 1
    target, kwargs = files[0]
    assert target.startswith("file:") and target.endswith("?mode=ro")
    assert kwargs.get("uri") is True and kwargs.get("timeout") == 0.0
    seen.clear()
    gate._db_path = tmp_path / "gone.db"
    assert gate.event_status(eid()).state == "verification_failed"
    assert [item for item in seen if item[0] != ":memory:"] == []  # no attempt to open a missing file


class CountingClock(Clock):
    def __init__(self, start=T0):
        super().__init__(start)
        self.reads = 0

    def __call__(self):
        self.reads += 1
        return self.t


def stored_events(store):
    rows = store._conn.execute("SELECT payload FROM audit_events ORDER BY seq").fetchall()
    return [control.event_from_payload(json.loads(row[0])) for row in rows]


def test_each_new_event_reads_the_injected_clock_once_and_never_the_system_clock(tmp_path, no_system_clock):
    clock = CountingClock()
    store, _, gate = make(tmp_path, clock=clock)
    first_decision = eid()
    steps = (
        lambda: gate.start_run(eid()),
        lambda: gate.acquire_lease(eid()),
        lambda: gate.admit_llm(req(), first_decision),
        lambda: gate.admit_llm(req(serialized_body_token_count=None), eid()),
        lambda: gate.record_llm_outcome(first_decision, 98, 40, eid()),
        lambda: gate.renew_lease(eid()),
        lambda: gate.release_lease(eid()),
    )
    for step in steps:
        before = clock.reads
        clock.adv(5)
        step()
        assert clock.reads == before + 1
    events = stored_events(store)
    assert [event.event_kind for event in events] == [
        "run_started", "lease_acquired", "decision", "decision", "outcome", "lease_renewed", "lease_released",
    ]
    for index, event in enumerate(events):
        stamped = T0 + timedelta(seconds=5 * (index + 1))
        assert datetime.fromisoformat(event.recorded_at) == stamped
        if event.event_kind == "run_started":
            assert event.observed_now is None
        else:
            assert datetime.fromisoformat(event.observed_now) == stamped
    verify(store)


def test_a_refused_call_does_not_fall_back_to_the_system_clock(tmp_path, no_system_clock):
    store, clock, gate = boot(tmp_path)
    with pytest.raises(GatekeeperRequestError):
        gate.admit_llm(req(serialized_body_token_count=None, model="other-model"), eid())
    with pytest.raises(GatekeeperUnknownRun):
        other(store, clock, run_id="run-2").acquire_lease(eid())
    verify(store)


def test_duplicates_and_event_status_never_read_either_clock(tmp_path, no_system_clock):
    clock = CountingClock()
    store, _, gate = make(tmp_path, clock=clock)
    ids = [eid() for _ in range(6)]
    gate.start_run(ids[0])
    clock.adv(1)
    gate.acquire_lease(ids[1])
    clock.adv(1)
    gate.admit_llm(req(), ids[2])
    clock.adv(1)
    gate.record_llm_outcome(ids[2], 98, 40, ids[3])
    clock.adv(1)
    gate.renew_lease(ids[4])
    clock.adv(1)
    gate.release_lease(ids[5])
    reads = clock.reads
    rows = count(store)
    silence(gate)
    for call in (
        lambda: gate.start_run(ids[0]),
        lambda: gate.acquire_lease(ids[1]),
        lambda: gate.admit_llm(req(), ids[2]),
        lambda: gate.record_llm_outcome(ids[2], 98, 40, ids[3]),
        lambda: gate.renew_lease(ids[4]),
        lambda: gate.release_lease(ids[5]),
    ):
        assert call().status == "already_applied"
    for event_id in ids:
        assert gate.event_status(event_id).state == "present"
    assert gate.event_status(eid()).state == "absent"
    assert clock.reads == reads and count(store) == rows
    verify(store)


class FailOn:
    """Delegating connection that raises sqlite3.OperationalError for one statement, optionally also for ROLLBACK."""

    def __init__(self, conn, needle, fail_rollback=False):
        self._c = conn
        self.needle = needle
        self.fail_rollback = fail_rollback

    def __getattr__(self, name):
        return getattr(self._c, name)

    def execute(self, sql, *args):
        text = " ".join(sql.split())
        if self.needle in text:
            raise sqlite3.OperationalError("injected statement failure")
        if self.fail_rollback and text.upper() == "ROLLBACK":
            raise sqlite3.OperationalError("injected rollback failure")
        return self._c.execute(sql, *args)


READ_CHAIN = "FROM audit_events ORDER BY seq"
INSERT_EVENT = "INSERT INTO audit_events"


@pytest.mark.parametrize("needle", [READ_CHAIN, INSERT_EVENT])
def test_sqlite_error_before_the_hooks_propagates_unchanged_after_a_confirmed_rollback(tmp_path, needle):
    # The verified read runs in the method body and the insert in control's append, both before any
    # ledger hook. A confirmed rollback leaves the original sqlite3 error; it is not ControlUnavailable.
    store, clock, gate = boot(tmp_path)
    clock.adv(10)
    real = store._conn
    rows = count(store)
    store._conn = FailOn(real, needle)
    with pytest.raises(sqlite3.OperationalError, match="injected statement failure") as caught:
        gate.admit_llm(req(), eid())
    assert not isinstance(caught.value, control.ControlUnavailable)
    store._conn = real
    assert count(store) == rows
    assert real.in_transaction is False
    verify(store)


@pytest.mark.parametrize("needle", [READ_CHAIN, INSERT_EVENT])
def test_sqlite_error_with_a_failed_rollback_is_unknown(tmp_path, needle):
    store, clock, gate = boot(tmp_path)
    clock.adv(10)
    real = store._conn
    store._conn = FailOn(real, needle, fail_rollback=True)
    with pytest.raises(control.ControlUnavailable) as caught:
        gate.admit_llm(req(), eid())
    assert caught.value.append_state == "unknown"
    store._conn = real
    if real.in_transaction:
        real.execute("ROLLBACK")  # test-internal cleanup of its own connection
    verify(store)


# ============================================================ G2a: approval and fetch


def freq(host="example.com", **kw):
    values = dict(host=host, max_bytes=1000, timeout_seconds=10, max_redirects=3)
    values.update(kw)
    return adm.FetchRequest(**values)


def run_state(store):
    rows = store._conn.execute("SELECT payload FROM audit_events ORDER BY seq").fetchall()
    events = [control.event_from_payload(json.loads(row[0])) for row in rows]
    return ledger.replay_events(events).runs["run-1"]


def expire(clock):
    clock.adv(301)


# ---- approval


def test_approve_records_one_target_under_the_live_lease(tmp_path):
    store, clock, gate = boot(tmp_path)
    clock.adv(5)
    approval = eid()
    result = gate.approve("target-1", approval)
    assert result.status == "applied"
    event = last_event(store)
    assert event.event_kind == "run_approved" and event.operation == "run"
    assert event.approved_target_id == "target-1"
    assert (event.lease_owner, event.lease_epoch) == (OWNER_A, 1)
    assert datetime.fromisoformat(event.observed_now) == T0 + timedelta(seconds=5)
    assert run_state(store).approved_target == "target-1"
    status = gate.event_status(approval)
    assert status.state == "present" and status.approved_target_id == "target-1"
    assert (status.lease_owner, status.lease_epoch) == (OWNER_A, 1)
    verify(store)


def test_second_approval_with_a_new_event_id_is_refused_even_for_the_same_target(tmp_path):
    store, clock, gate = boot(tmp_path)
    gate.approve("target-1", eid())
    rows = count(store)
    for target in ("target-1", "target-2"):
        with pytest.raises(GatekeeperStateError):
            gate.approve(target, eid())
    assert count(store) == rows and run_state(store).approved_target == "target-1"
    verify(store)


def test_approval_duplicate_is_confirmed_without_a_clock_and_a_different_target_is_refused(tmp_path, no_system_clock):
    store, clock, gate = boot(tmp_path)
    clock.adv(1)
    approval = eid()
    first = gate.approve("target-1", approval)
    clock.adv(1)
    gate.admit_fetch(freq(), eid())
    expire(clock)
    silence(gate)
    rows = count(store)
    again = gate.approve("target-1", approval)
    assert again.status == "already_applied" and again.event_hash == first.event_hash
    with pytest.raises(control.AuditIncomplete):
        gate.approve("target-2", approval)
    assert count(store) == rows
    verify(store)


def test_approval_duplicate_by_another_owner_is_refused(tmp_path):
    store, clock, gate = boot(tmp_path)
    approval = eid()
    gate.approve("target-1", approval)
    rows = count(store)
    with pytest.raises(GatekeeperRequestError):
        other(store, clock, owner_id=OWNER_B).approve("target-1", approval)
    assert count(store) == rows


def test_approval_needs_a_live_lease_owned_by_the_caller(tmp_path):
    store, clock, gate = make(tmp_path)
    gate.start_run(eid())
    with pytest.raises(GatekeeperLeaseError):
        gate.approve("target-1", eid())
    gate.acquire_lease(eid())
    rows = count(store)
    with pytest.raises(GatekeeperLeaseError):
        other(store, clock, owner_id=OWNER_B).approve("target-1", eid())
    expire(clock)
    with pytest.raises(GatekeeperLeaseError):
        gate.approve("target-1", eid())
    assert count(store) == rows and run_state(store).approved_target is None
    verify(store)


@pytest.mark.parametrize(
    "target", ["", " x", "x ", "x" * 129, "a\nb", "a\x00b", None, 5, b"x"]
)
def test_approval_rejects_unstorable_target_ids_before_append(tmp_path, target):
    store, clock, gate = boot(tmp_path)
    rows = count(store)
    with pytest.raises(GatekeeperRequestError):
        gate.approve(target, eid())
    assert count(store) == rows


def test_approval_accepts_a_128_character_target(tmp_path):
    store, clock, gate = boot(tmp_path)
    gate.approve("x" * 128, eid())
    assert run_state(store).approved_target == "x" * 128
    verify(store)


def test_approval_unknown_run_policy_mismatch_and_clock(tmp_path):
    store, clock, gate = boot(tmp_path)
    rows = count(store)
    with pytest.raises(GatekeeperUnknownRun):
        other(store, clock, run_id="run-2").approve("t", eid())
    with pytest.raises(GatekeeperPolicyMismatch):
        other(store, clock, policy=cfg(max_fetches_per_run=9)).approve("t", eid())
    clock.adv(-10)
    with pytest.raises(GatekeeperClockError):
        gate.approve("t", eid())
    assert count(store) == rows


# ---- fetch admission


def test_fetch_allow_stores_the_normalized_host_and_uses_the_cap_at_decision(tmp_path):
    store, clock, gate = boot(tmp_path, max_fetches_per_run=2)
    clock.adv(1)
    first = gate.admit_fetch(freq("Example.COM"), eid())
    assert first.decision.allow and first.decision.reason is adm.AdmissionReason.ALLOWED
    event = last_event(store)
    assert (event.event_kind, event.operation, event.decision) == ("decision", "fetch", "allow")
    assert event.target_host == "example.com" and event.reason_code == "allowed"
    assert event.reserve_tokens is None and event.serialized_body_token_count is None
    assert run_state(store).fetch_count == 1
    gate.admit_fetch(freq("b.example.com"), eid())
    assert run_state(store).fetch_count == 2
    denied = gate.admit_fetch(freq("c.example.com"), eid())
    assert not denied.decision.allow and denied.decision.reason is adm.AdmissionReason.BUDGET_EXCEEDED
    event = last_event(store)
    assert event.decision == "deny" and event.reason_code == "budget_exceeded"
    assert event.target_host == "c.example.com"
    assert run_state(store).fetch_count == 2
    verify(store)


@pytest.mark.parametrize(
    "host", ["a_b.com", "a.com.", "-a.com", "host:80", "http://x", "a b", "a/b", "a@b", "a?b", "a#b", "a\\b"]
)
def test_fetch_with_a_host_that_cannot_be_stored_is_denied_without_a_host(tmp_path, host):
    store, clock, gate = boot(tmp_path)
    result = gate.admit_fetch(freq(host), eid())
    assert not result.decision.allow
    assert result.decision.reason is adm.AdmissionReason.INVALID_TARGET
    event = last_event(store)
    assert event.decision == "deny" and event.reason_code == "invalid_target"
    assert event.target_host is None
    assert run_state(store).fetch_count == 0
    verify(store)


def test_unit_requests_the_admission_layer_cannot_build_never_reach_the_gatekeeper():
    # admission.FetchRequest rejects these hosts at construction, so no deny event can exist for them.
    for host in ("", " a.com", "a.com ", "a\x01b", "x" * 254):
        with pytest.raises((TypeError, ValueError)):
            freq(host)


@pytest.mark.parametrize(
    "kw",
    [
        dict(timeout_seconds=0),
        dict(timeout_seconds=-1),
        dict(timeout_seconds=float("nan")),
        dict(timeout_seconds=float("inf")),
        dict(max_bytes=0),
        dict(max_redirects=-1),
    ],
)
def test_fetch_with_bad_limits_is_denied_and_keeps_the_host(tmp_path, kw):
    store, clock, gate = boot(tmp_path)
    result = gate.admit_fetch(freq("Example.com", **kw), eid())
    assert not result.decision.allow and result.decision.reason is adm.AdmissionReason.INVALID_TARGET
    event = last_event(store)
    assert event.reason_code == "invalid_target" and event.target_host == "example.com"
    assert run_state(store).fetch_count == 0
    verify(store)


HOST_CASES = [
    "example.com", "Example.COM", "a.b.c", "1.2.3.4", "999.999.999.999", "1.2.3", "01.2.3.4",
    "xn--bcher-kva.example", "a-b.com", "-a.com", "a-.com", "a..b", ".a", "a.", "a_b", "a b",
    "a:80", "a/b", "a@b", "a?b", "a#b", "a\\b", "http://a", "\u212a.com", "caf\u00e9.com",
    "a" * 63 + ".com", "a" * 64 + ".com", ("a" * 60 + ".") * 4 + "com", "::1", "[::1]", "x",
]


@pytest.mark.parametrize("host", HOST_CASES)
def test_unit_host_rules_match_between_admission_and_control(tmp_path, host):
    store, clock, gate = make(tmp_path)
    try:
        request = freq(host)
    except (TypeError, ValueError):
        return
    storable = gate._stored_host(request.host) is not None
    assert adm._host_is_syntactic(request.host) == storable


def test_unit_a_kelvin_sign_host_is_stored_as_ascii_lowercase(tmp_path):
    # Admission and control both lowercase first, so the stored host can differ from the caller's string.
    store, clock, gate = boot(tmp_path)
    gate.admit_fetch(freq("\u212a.com"), eid())
    assert last_event(store).target_host == "k.com"


def test_unit_an_allowed_fetch_whose_host_cannot_be_stored_is_refused(tmp_path, monkeypatch):
    store, clock, gate = boot(tmp_path)
    monkeypatch.setattr(Gatekeeper, "_stored_host", staticmethod(lambda host: None))
    rows = count(store)
    with pytest.raises(GatekeeperRequestError):
        gate.admit_fetch(freq(), eid())
    assert count(store) == rows


def test_fetch_duplicate_is_rebuilt_from_the_state_before_the_stored_event(tmp_path, no_system_clock):
    store, clock, gate = boot(tmp_path, max_fetches_per_run=1)
    first_id = eid()
    clock.adv(1)
    first = gate.admit_fetch(freq(), first_id)
    clock.adv(1)
    assert not gate.admit_fetch(freq("b.example.com"), eid()).decision.allow
    expire(clock)
    silence(gate)
    rows = count(store)
    again = gate.admit_fetch(freq(), first_id)
    assert again.status == "already_applied" and again.decision.allow
    assert again.event_hash == first.event_hash and count(store) == rows
    verify(store)


def test_fetch_duplicate_does_not_prove_the_request_is_unchanged(tmp_path):
    store, clock, gate = boot(tmp_path)
    denied_id, allowed_id = eid(), eid()
    gate.admit_fetch(freq("a_b.com"), denied_id)
    gate.admit_fetch(freq("good.example.com", max_bytes=1000), allowed_id)
    rows = count(store)
    # a different unstorable host, or different limits, hash to the same stored event: confirmed
    assert gate.admit_fetch(freq("c_d.com"), denied_id).status == "already_applied"
    assert gate.admit_fetch(freq("good.example.com", max_bytes=5, timeout_seconds=1, max_redirects=0), allowed_id).status == "already_applied"
    # a different storable host differs in the stored event: refused
    with pytest.raises(control.AuditIncomplete):
        gate.admit_fetch(freq("other.example.com"), allowed_id)
    assert count(store) == rows


def test_fetch_refusals_write_nothing(tmp_path):
    store, clock, gate = make(tmp_path)
    with pytest.raises(GatekeeperUnknownRun):
        gate.admit_fetch(freq(), eid())
    gate.start_run(eid())
    with pytest.raises(GatekeeperLeaseError):
        gate.admit_fetch(freq(), eid())
    gate.acquire_lease(eid())
    rows = count(store)
    for bad in (None, "example.com", req()):
        with pytest.raises(GatekeeperRequestError):
            gate.admit_fetch(bad, eid())
    with pytest.raises(GatekeeperLeaseError):
        other(store, clock, owner_id=OWNER_B).admit_fetch(freq(), eid())
    with pytest.raises(GatekeeperPolicyMismatch):
        other(store, clock, policy=cfg(max_fetches_per_run=9)).admit_fetch(freq(), eid())
    clock.adv(-1)
    with pytest.raises(GatekeeperClockError):
        gate.admit_fetch(freq(), eid())
    clock.adv(1)
    expire(clock)
    with pytest.raises(GatekeeperLeaseError):
        gate.admit_fetch(freq(), eid())
    assert count(store) == rows and run_state(store).fetch_count == 0
    verify(store)


def test_fetch_duplicate_by_another_owner_is_refused(tmp_path):
    store, clock, gate = boot(tmp_path)
    decision_id = eid()
    gate.admit_fetch(freq(), decision_id)
    with pytest.raises(GatekeeperRequestError):
        other(store, clock, owner_id=OWNER_B).admit_fetch(freq(), decision_id)


# ---- fetch outcome


@pytest.mark.parametrize("result_class", sorted(control.RESULT_CLASSES))
def test_fetch_outcome_records_the_decision_host_and_the_caller_result_class(tmp_path, result_class):
    store, clock, gate = boot(tmp_path)
    decision_id, outcome_id = eid(), eid()
    gate.admit_fetch(freq("Fetch-Host.Example.ORG"), decision_id)
    clock.adv(3)
    result = gate.record_fetch_outcome(decision_id, result_class, outcome_id)
    assert result.status == "applied" and result.decision is None
    event = last_event(store)
    assert (event.event_kind, event.operation, event.decision, event.reason_code) == (
        "outcome", "fetch", "allow", "allowed",
    )
    assert event.target_host == "fetch-host.example.org" and event.result_class == result_class
    assert event.references_event_id == decision_id and not event.counters
    assert event.error_class is None
    status = gate.event_status(outcome_id)
    assert status.state == "present"
    assert (status.result_class, status.target_host, status.references_event_id) == (
        result_class, "fetch-host.example.org", decision_id,
    )
    verify(store)


def test_fetch_outcome_can_carry_an_error_class_name(tmp_path):
    store, clock, gate = boot(tmp_path)
    decision_id, outcome_id = eid(), eid()
    gate.admit_fetch(freq(), decision_id)
    gate.record_fetch_outcome(decision_id, "network_error", outcome_id, error_class="market_intelligence.net.TransientFetchError")
    assert last_event(store).error_class == "market_intelligence.net.TransientFetchError"
    assert gate.event_status(outcome_id).error_class == "market_intelligence.net.TransientFetchError"
    verify(store)


def test_fetch_outcome_refusals_write_nothing(tmp_path):
    store, clock, gate = boot(tmp_path, max_fetches_per_run=2)
    llm_id, denied_id, capped_id, ok_id = eid(), eid(), eid(), eid()
    gate.admit_llm(req(), llm_id)
    gate.admit_fetch(freq("a_b.com"), denied_id)
    gate.admit_fetch(freq(), ok_id)
    gate.admit_fetch(freq("b.example.com"), eid())
    assert not gate.admit_fetch(freq("c.example.com"), capped_id).decision.allow
    rows = count(store)
    for decision_id in (eid(), denied_id, capped_id, llm_id, "", None):
        with pytest.raises(GatekeeperRequestError):
            gate.record_fetch_outcome(decision_id, "ok", eid())
    for bad in ("bogus", None, 5, "", "OK"):
        with pytest.raises(GatekeeperRequestError):
            gate.record_fetch_outcome(ok_id, bad, eid())
    for bad in ("a b", "1x", "", 5, "a..b"):
        with pytest.raises(GatekeeperRequestError):
            gate.record_fetch_outcome(ok_id, "ok", eid(), error_class=bad)
    assert count(store) == rows
    gate.record_fetch_outcome(ok_id, "ok", eid())
    rows = count(store)
    with pytest.raises(GatekeeperRequestError):
        gate.record_fetch_outcome(ok_id, "http_error", eid())
    assert count(store) == rows
    verify(store)


def test_fetch_outcome_needs_a_live_lease_owned_by_the_caller(tmp_path):
    store, clock, gate = boot(tmp_path)
    decision_id = eid()
    gate.admit_fetch(freq(), decision_id)
    rows = count(store)
    with pytest.raises(GatekeeperLeaseError):
        other(store, clock, owner_id=OWNER_B).record_fetch_outcome(decision_id, "ok", eid())
    with pytest.raises(GatekeeperUnknownRun):
        other(store, clock, run_id="run-2").record_fetch_outcome(decision_id, "ok", eid())
    with pytest.raises(GatekeeperPolicyMismatch):
        other(store, clock, policy=cfg(max_fetches_per_run=9)).record_fetch_outcome(decision_id, "ok", eid())
    clock.adv(-1)
    with pytest.raises(GatekeeperClockError):
        gate.record_fetch_outcome(decision_id, "ok", eid())
    clock.adv(1)
    expire(clock)
    with pytest.raises(GatekeeperLeaseError):
        gate.record_fetch_outcome(decision_id, "ok", eid())
    assert count(store) == rows


def test_fetch_outcome_duplicate_uses_the_state_before_the_stored_event(tmp_path, no_system_clock):
    store, clock, gate = boot(tmp_path)
    decision_id, outcome_id = eid(), eid()
    clock.adv(1)
    gate.admit_fetch(freq(), decision_id)
    second_decision = eid()
    gate.admit_fetch(freq("b.example.com"), second_decision)
    clock.adv(1)
    first = gate.record_fetch_outcome(decision_id, "timeout", outcome_id, error_class="TimeoutError")
    clock.adv(1)
    gate.admit_fetch(freq("c.example.com"), eid())
    expire(clock)
    silence(gate)
    rows = count(store)
    again = gate.record_fetch_outcome(decision_id, "timeout", outcome_id, error_class="TimeoutError")
    assert again.status == "already_applied" and again.event_hash == first.event_hash
    with pytest.raises(control.AuditIncomplete):
        gate.record_fetch_outcome(decision_id, "ok", outcome_id, error_class="TimeoutError")
    with pytest.raises(GatekeeperRequestError):
        gate.record_fetch_outcome(eid(), "timeout", outcome_id)
    with pytest.raises(GatekeeperRequestError):
        gate.record_fetch_outcome(decision_id, "timeout", eid())  # a second outcome under a new id
    with pytest.raises(GatekeeperRequestError):
        gate.record_fetch_outcome(second_decision, "timeout", outcome_id)  # known allowed decision, wrong reference
    assert count(store) == rows
    verify(store)


def test_a_second_owner_closes_an_earlier_fetch_decision_after_handover(tmp_path):
    store, clock, gate = boot(tmp_path)
    decision_id = eid()
    gate.admit_fetch(freq(), decision_id)
    expire(clock)
    gate.release_lease(eid())
    second = other(store, clock, owner_id=OWNER_B)
    second.acquire_lease(eid())
    outcome_id = eid()
    assert second.record_fetch_outcome(decision_id, "unknown", outcome_id).status == "applied"
    event = last_event(store)
    assert (event.lease_owner, event.lease_epoch) == (OWNER_B, 2)
    # the owner that made the decision can still confirm its own earlier event id; the second owner cannot
    other_decision = eid()
    rows = count(store)
    with pytest.raises(GatekeeperRequestError):
        second.admit_fetch(freq(), decision_id)
    assert gate.admit_fetch(freq(), decision_id).status == "already_applied"
    assert count(store) == rows and other_decision
    verify(store)


# ---- failure injection and clock for the new methods


def _g2a_calls(gate, ids):
    return {
        "approve": lambda: gate.approve("target-1", ids["approve"]),
        "fetch": lambda: gate.admit_fetch(freq(), ids["fetch"]),
        "outcome": lambda: gate.record_fetch_outcome(ids["fetch"], "ok", ids["outcome"]),
    }


@pytest.mark.parametrize("method", ["approve", "fetch", "outcome"])
@pytest.mark.parametrize("mode,expected", [("landed", "already_applied"), ("absent", "applied")])
def test_unknown_commit_on_each_g2a_method_is_execute_or_confirm(tmp_path, method, mode, expected):
    store, clock, gate = boot(tmp_path)
    ids = {name: eid() for name in ("approve", "fetch", "outcome")}
    calls = _g2a_calls(gate, ids)
    if method == "outcome":
        calls["fetch"]()
    clock.adv(10)
    real = store._conn
    store._conn = Wrap(real, mode)
    with pytest.raises(control.ControlUnavailable) as caught:
        calls[method]()
    assert caught.value.append_state == "unknown"
    store._conn = real
    key = {"approve": "approve", "fetch": "fetch", "outcome": "outcome"}[method]
    assert gate.event_status(ids[key]).state == ("present" if mode == "landed" else "absent")
    clock.adv(2)
    assert calls[method]().status == expected
    assert store._conn.execute(
        "SELECT count(*) FROM audit_events WHERE event_id = ?", (ids[key],)
    ).fetchone()[0] == 1
    verify(store)


@pytest.mark.parametrize("needle", [READ_CHAIN, INSERT_EVENT])
@pytest.mark.parametrize("method", ["approve", "fetch", "outcome"])
def test_sqlite_error_in_a_g2a_method_propagates_unchanged_after_a_confirmed_rollback(tmp_path, method, needle):
    store, clock, gate = boot(tmp_path)
    ids = {name: eid() for name in ("approve", "fetch", "outcome")}
    calls = _g2a_calls(gate, ids)
    if method == "outcome":
        calls["fetch"]()
    clock.adv(10)
    real = store._conn
    rows = count(store)
    store._conn = FailOn(real, needle)
    with pytest.raises(sqlite3.OperationalError, match="injected statement failure") as caught:
        calls[method]()
    assert not isinstance(caught.value, control.ControlUnavailable)
    store._conn = real
    assert count(store) == rows and real.in_transaction is False
    verify(store)


def test_each_new_g2a_event_reads_the_injected_clock_once_and_duplicates_never(tmp_path, no_system_clock):
    clock = CountingClock()
    store, _, gate = make(tmp_path, clock=clock)
    gate.start_run(eid())
    gate.acquire_lease(eid())
    ids = {name: eid() for name in ("approve", "fetch", "deny", "outcome")}
    steps = (
        lambda: gate.approve("target-1", ids["approve"]),
        lambda: gate.admit_fetch(freq(), ids["fetch"]),
        lambda: gate.admit_fetch(freq("a_b.com"), ids["deny"]),
        lambda: gate.record_fetch_outcome(ids["fetch"], "ok", ids["outcome"]),
    )
    for step in steps:
        before = clock.reads
        clock.adv(5)
        step()
        assert clock.reads == before + 1
    reads, rows = clock.reads, count(store)
    silence(gate)
    for step in steps:
        assert step().status == "already_applied"
    for event_id in ids.values():
        assert gate.event_status(event_id).state == "present"
    assert clock.reads == reads and count(store) == rows
    events = stored_events(store)[2:]
    for index, event in enumerate(events):
        stamped = T0 + timedelta(seconds=5 * (index + 1))
        assert datetime.fromisoformat(event.recorded_at) == stamped
        assert datetime.fromisoformat(event.observed_now) == stamped
    verify(store)


def test_event_status_reports_the_new_fields_and_stays_read_only(tmp_path):
    store, clock, gate = boot(tmp_path)
    ids = {name: eid() for name in ("approve", "fetch", "deny", "outcome")}
    gate.approve("target-1", ids["approve"])
    gate.admit_fetch(freq("Example.com"), ids["fetch"])
    gate.admit_fetch(freq("a_b.com"), ids["deny"])
    gate.record_fetch_outcome(ids["fetch"], "ok", ids["outcome"])
    before = fingerprint_of_files(tmp_path)
    rows = count(store)
    assert gate.event_status(ids["deny"]).target_host is None
    assert gate.event_status(ids["deny"]).reason_code == "invalid_target"
    assert gate.event_status(ids["fetch"]).target_host == "example.com"
    assert gate.event_status(ids["approve"]).approved_target_id == "target-1"
    assert gate.event_status(ids["outcome"]).result_class == "ok"
    assert gate.event_status(eid()).approved_target_id is None
    assert fingerprint_of_files(tmp_path) == before and count(store) == rows


def test_g2a_leaves_llm_admission_and_the_g1_surface_unchanged(tmp_path):
    store, clock, gate = boot(tmp_path)
    gate.approve("target-1", eid())
    gate.admit_fetch(freq(), eid())
    decision_id = eid()
    result = gate.admit_llm(req(), decision_id)
    assert result.decision.allow
    gate.record_llm_outcome(decision_id, 98, 40, eid())
    state = run_state(store)
    assert state.fetch_count == 1 and state.calls == 1 and state.approved_target == "target-1"
    verify(store)


# ============================================================ G2b: write admission and write outcomes

HASH = "ab" * 32


def wreq(target="target-1"):
    return adm.WriteRequest(target_id=target)


def approved(path, target="target-1", **policy):
    store, clock, gate = boot(path, **policy)
    gate.approve(target, eid())
    clock.adv(1)
    return store, clock, gate


def test_write_allowed_for_the_approved_target_records_target_hash_and_policy(tmp_path):
    store, clock, gate = approved(tmp_path)
    decision_id = eid()
    result = gate.admit_write(wreq(), decision_id, content_hash=HASH.upper())
    assert result.status == "applied" and result.decision.allow
    event = last_event(store)
    assert (event.event_kind, event.operation, event.decision, event.reason_code) == (
        "decision", "write", "allow", "allowed",
    )
    assert event.target_id == "target-1" and event.content_hash == HASH and event.target_host is None
    assert event.max_run_tokens == 20000 and event.allowed_model == "gpt-4o-mini" and not event.counters
    assert (event.lease_owner, event.lease_epoch) == (OWNER_A, 1)
    state = run_state(store)
    assert (state.calls, state.fetch_count, state.reserved_tokens) == (0, 0, 0)
    assert state.decisions[decision_id].operation == "write" and state.decisions[decision_id].target_id == "target-1"
    status = gate.event_status(decision_id)
    assert (status.state, status.target_id, status.content_hash) == ("present", "target-1", HASH)
    verify(store)


def test_write_without_a_content_hash_stores_none(tmp_path):
    store, clock, gate = approved(tmp_path)
    gate.admit_write(wreq(), eid())
    assert last_event(store).content_hash is None
    verify(store)


def test_write_without_approval_is_a_recorded_unapproved_deny(tmp_path):
    store, clock, gate = boot(tmp_path)
    decision_id = eid()
    result = gate.admit_write(wreq(), decision_id, content_hash=HASH)
    assert result.status == "applied" and not result.decision.allow
    event = last_event(store)
    assert (event.decision, event.reason_code, event.operation) == ("deny", "unapproved_write", "write")
    assert event.target_id == "target-1" and event.content_hash == HASH
    assert run_state(store).decisions[decision_id].allow is False
    verify(store)


@pytest.mark.parametrize("target", ["target-2", "Target-1", "target-1x", "t"])
def test_write_to_another_target_is_a_recorded_unapproved_deny(tmp_path, target):
    store, clock, gate = approved(tmp_path)
    result = gate.admit_write(wreq(target), eid())
    assert not result.decision.allow
    event = last_event(store)
    assert (event.decision, event.reason_code, event.target_id) == ("deny", "unapproved_write", target)
    verify(store)


def test_denied_writes_never_block_a_later_allowed_write(tmp_path):
    store, clock, gate = approved(tmp_path)
    assert not gate.admit_write(wreq("other-1"), eid()).decision.allow
    assert not gate.admit_write(wreq("other-2"), eid()).decision.allow
    assert gate.admit_write(wreq(), eid()).decision.allow
    verify(store)


def test_write_requests_the_audit_layer_cannot_store_raise_before_append(tmp_path):
    store, clock, gate = approved(tmp_path)
    rows = count(store)
    for bad in (adm.WriteRequest(target_id=None), wreq("x" * 129), "target-1", None, 5):
        with pytest.raises(GatekeeperRequestError):
            gate.admit_write(bad, eid())
    for bad_hash in ("zz", "a" * 63, "a" * 65, 5, "", b"ab" * 32):
        with pytest.raises(GatekeeperRequestError):
            gate.admit_write(wreq(), eid(), content_hash=bad_hash)
    for bad_id in ("", None, 5):
        with pytest.raises(GatekeeperRequestError):
            gate.admit_write(wreq(), bad_id)
    assert count(store) == rows
    assert gate.admit_write(wreq("x" * 128), eid()).decision.allow is False
    verify(store)


def test_write_needs_a_live_lease_owned_by_the_caller_a_matching_policy_and_a_forward_clock(tmp_path):
    store, clock, gate = approved(tmp_path)
    rows = count(store)
    with pytest.raises(GatekeeperLeaseError):
        other(store, clock, owner_id=OWNER_B).admit_write(wreq(), eid())
    with pytest.raises(GatekeeperUnknownRun):
        other(store, clock, run_id="run-2").admit_write(wreq(), eid())
    with pytest.raises(GatekeeperPolicyMismatch):
        other(store, clock, policy=cfg(max_fetches_per_run=9)).admit_write(wreq(), eid())
    clock.adv(-2)
    with pytest.raises(GatekeeperClockError):
        gate.admit_write(wreq(), eid())
    clock.adv(2)
    expire(clock)
    with pytest.raises(GatekeeperLeaseError):
        gate.admit_write(wreq(), eid())
    assert count(store) == rows


def test_a_second_allowed_write_is_refused_while_the_first_has_no_outcome(tmp_path):
    clock = CountingClock()
    store, _, gate = make(tmp_path, clock=clock)
    gate.start_run(eid())
    gate.acquire_lease(eid())
    gate.approve("target-1", eid())
    first = eid()
    clock.adv(1)
    assert gate.admit_write(wreq(), first).decision.allow
    rows, reads = count(store), clock.reads
    with pytest.raises(GatekeeperStateError, match=first) as caught:
        gate.admit_write(wreq(), eid())
    assert "no outcome" in str(caught.value)
    assert count(store) == rows and clock.reads == reads
    assert not gate.admit_write(wreq("other"), eid()).decision.allow  # a deny is still recorded
    verify(store)


@pytest.mark.parametrize("value", ["committed", "unknown"])
def test_a_second_allowed_write_is_refused_after_committed_or_unknown(tmp_path, value):
    store, clock, gate = approved(tmp_path)
    first = eid()
    gate.admit_write(wreq(), first)
    gate.record_write_outcome(first, value, eid())
    rows = count(store)
    with pytest.raises(GatekeeperStateError, match=first) as caught:
        gate.admit_write(wreq(), eid())
    assert value in str(caught.value)
    assert count(store) == rows
    verify(store)


def test_failed_lets_exactly_one_new_write_through_and_the_new_one_blocks_in_turn(tmp_path):
    store, clock, gate = approved(tmp_path)
    first, second = eid(), eid()
    gate.admit_write(wreq(), first)
    gate.record_write_outcome(first, "failed", eid())
    assert gate.admit_write(wreq(), second).decision.allow
    with pytest.raises(GatekeeperStateError, match=second):
        gate.admit_write(wreq(), eid())
    gate.record_write_outcome(second, "failed", eid())
    third = eid()
    assert gate.admit_write(wreq(), third).decision.allow
    gate.record_write_outcome(third, "committed", eid())
    with pytest.raises(GatekeeperStateError, match=third):
        gate.admit_write(wreq(), eid())
    verify(store)


def test_unit_the_guard_looks_at_every_earlier_allowed_write_not_only_the_latest(tmp_path):
    store, clock, gate = approved(tmp_path)
    first, second = eid(), eid()
    gate.admit_write(wreq(), first)
    gate.record_write_outcome(first, "failed", eid())
    gate.admit_write(wreq(), second)
    gate.record_write_outcome(second, "failed", eid())
    events = stored_events(store)
    assert gate._blocking_write(events) is None
    # no input reaches this through the public methods, so the unit check removes an earlier outcome by hand
    pruned = [e for e in events if not (e.event_kind == "outcome" and e.references_event_id == first)]
    assert gate._blocking_write(pruned) == (first, "no outcome")
    flipped = [
        dataclasses.replace(e, write_outcome="unknown")
        if (e.event_kind == "outcome" and e.references_event_id == first)
        else e
        for e in events
    ]
    assert gate._blocking_write(flipped) == (first, "unknown")


def test_writes_of_another_run_in_the_same_chain_do_not_block_this_run(tmp_path):
    store, clock, gate = approved(tmp_path)
    second = other(store, clock, run_id="run-2", owner_id=OWNER_B)
    second.start_run(eid())
    second.acquire_lease(eid())
    second.approve("target-1", eid())
    clock.adv(1)
    foreign = eid()
    assert second.admit_write(wreq(), foreign).decision.allow  # open in run-2
    assert gate.admit_write(wreq(), eid()).decision.allow  # run-1 unaffected
    with pytest.raises(GatekeeperStateError):
        gate.admit_write(wreq(), eid())  # now run-1 is blocked by its own write
    with pytest.raises(GatekeeperStateError, match=foreign):
        second.admit_write(wreq(), eid())  # and run-2 by its own
    verify(store)


@pytest.mark.parametrize("value", ["committed", "failed", "unknown"])
def test_write_outcome_records_the_caller_value_and_derives_retry_safe(tmp_path, value):
    store, clock, gate = approved(tmp_path)
    decision_id, outcome_id = eid(), eid()
    gate.admit_write(wreq(), decision_id, content_hash=HASH)
    clock.adv(3)
    result = gate.record_write_outcome(decision_id, value, outcome_id)
    assert result.status == "applied" and result.decision is None
    event = last_event(store)
    assert (event.event_kind, event.operation, event.decision, event.reason_code) == (
        "outcome", "write", "allow", "allowed",
    )
    assert event.write_outcome == value and event.retry_safe is (value == "failed")
    assert event.references_event_id == decision_id and not event.counters
    assert event.target_id is None and event.content_hash is None and event.target_host is None
    assert event.result_class is None and event.reserve_tokens is None
    assert (event.lease_owner, event.lease_epoch) == (OWNER_A, 1)
    state = run_state(store)
    assert decision_id in state.normal_outcomes and state.reserved_tokens == 0 and not state.unknown_outcomes
    status = gate.event_status(outcome_id)
    assert (status.write_outcome, status.references_event_id, status.target_id) == (value, decision_id, None)
    assert gate.event_status(decision_id).outcome_event_id == outcome_id
    verify(store)


def test_write_outcome_refusals_write_nothing(tmp_path):
    store, clock, gate = approved(tmp_path)
    llm_id, fetch_id, deny_id, ok_id = eid(), eid(), eid(), eid()
    gate.admit_llm(req(), llm_id)
    gate.admit_fetch(freq(), fetch_id)
    gate.admit_write(wreq("other"), deny_id)
    gate.admit_write(wreq(), ok_id)
    rows = count(store)
    for decision_id in (eid(), deny_id, fetch_id, llm_id, "", None, 5, []):
        with pytest.raises(GatekeeperRequestError):
            gate.record_write_outcome(decision_id, "committed", eid())
    for bad in ("bogus", None, 5, "", "COMMITTED", "retry_safe", True):
        with pytest.raises(GatekeeperRequestError):
            gate.record_write_outcome(ok_id, bad, eid())
    assert count(store) == rows
    gate.record_write_outcome(ok_id, "committed", eid())
    rows = count(store)
    for value in ("committed", "failed", "unknown"):
        with pytest.raises(GatekeeperRequestError):
            gate.record_write_outcome(ok_id, value, eid())
    assert count(store) == rows
    verify(store)


def test_write_outcome_needs_a_live_lease_owned_by_the_caller(tmp_path):
    store, clock, gate = approved(tmp_path)
    decision_id = eid()
    gate.admit_write(wreq(), decision_id)
    rows = count(store)
    with pytest.raises(GatekeeperLeaseError):
        other(store, clock, owner_id=OWNER_B).record_write_outcome(decision_id, "failed", eid())
    with pytest.raises(GatekeeperUnknownRun):
        other(store, clock, run_id="run-2").record_write_outcome(decision_id, "failed", eid())
    with pytest.raises(GatekeeperPolicyMismatch):
        other(store, clock, policy=cfg(max_fetches_per_run=9)).record_write_outcome(decision_id, "failed", eid())
    clock.adv(-2)
    with pytest.raises(GatekeeperClockError):
        gate.record_write_outcome(decision_id, "failed", eid())
    clock.adv(2)
    expire(clock)
    with pytest.raises(GatekeeperLeaseError):
        gate.record_write_outcome(decision_id, "failed", eid())
    assert count(store) == rows


def test_write_decision_duplicate_uses_the_state_before_the_stored_event_and_no_clock(tmp_path, no_system_clock):
    store, clock, gate = approved(tmp_path)
    decision_id = eid()
    clock.adv(1)
    first = gate.admit_write(wreq(), decision_id, content_hash=HASH)
    clock.adv(1)
    gate.record_write_outcome(decision_id, "unknown", eid())  # the run is now blocked for new writes
    gate.admit_fetch(freq(), eid())
    expire(clock)
    silence(gate)
    rows = count(store)
    again = gate.admit_write(wreq(), decision_id, content_hash=HASH.upper())
    assert again.status == "already_applied" and again.event_hash == first.event_hash and again.decision.allow
    assert count(store) == rows
    verify(store)


def test_a_denied_write_duplicate_stays_a_deny_after_the_approval_exists(tmp_path, no_system_clock):
    store, clock, gate = boot(tmp_path)
    deny_id = eid()
    clock.adv(1)
    assert not gate.admit_write(wreq(), deny_id, content_hash=HASH).decision.allow
    clock.adv(1)
    gate.approve("target-1", eid())
    silence(gate)
    again = gate.admit_write(wreq(), deny_id, content_hash=HASH)
    assert again.status == "already_applied" and not again.decision.allow
    verify(store)


def test_a_repeated_write_event_id_with_a_changed_request_is_refused_not_confirmed(tmp_path):
    store, clock, gate = approved(tmp_path)
    allowed_id, deny_id = eid(), eid()
    gate.admit_write(wreq(), allowed_id, content_hash=HASH)
    gate.admit_write(wreq("other"), deny_id, content_hash=HASH)
    rows = count(store)
    for event_id, target, digest in (
        (allowed_id, "target-2", HASH),
        (allowed_id, "target-1", "cd" * 32),
        (allowed_id, "target-1", None),
        (deny_id, "other", "cd" * 32),
        (deny_id, "other", None),
        (deny_id, "another", HASH),
        (deny_id, "target-1", HASH),
    ):
        with pytest.raises(control.AuditIncomplete):
            gate.admit_write(wreq(target), event_id, content_hash=digest)
    assert count(store) == rows
    verify(store)


def test_a_repeated_outcome_event_id_with_a_different_value_is_refused_not_confirmed(tmp_path, no_system_clock):
    store, clock, gate = approved(tmp_path)
    decision_id, outcome_id, other_decision = eid(), eid(), eid()
    gate.admit_write(wreq(), decision_id)
    first = gate.record_write_outcome(decision_id, "committed", outcome_id)
    clock.adv(1)
    gate.admit_fetch(freq(), eid())
    expire(clock)
    silence(gate)
    rows = count(store)
    again = gate.record_write_outcome(decision_id, "committed", outcome_id)
    assert again.status == "already_applied" and again.event_hash == first.event_hash
    for value in ("failed", "unknown"):
        with pytest.raises(control.AuditIncomplete):
            gate.record_write_outcome(decision_id, value, outcome_id)
    with pytest.raises(GatekeeperRequestError):
        gate.record_write_outcome(eid(), "committed", outcome_id)  # unknown decision
    with pytest.raises(GatekeeperRequestError):
        gate.record_write_outcome(other_decision, "committed", outcome_id)
    with pytest.raises(GatekeeperRequestError):
        gate.record_write_outcome(decision_id, "committed", eid())  # a second outcome under a new id
    assert count(store) == rows
    verify(store)


def test_write_duplicates_by_another_owner_are_refused(tmp_path):
    store, clock, gate = approved(tmp_path)
    decision_id, outcome_id = eid(), eid()
    gate.admit_write(wreq(), decision_id)
    gate.record_write_outcome(decision_id, "failed", outcome_id)
    second = other(store, clock, owner_id=OWNER_B)
    with pytest.raises(GatekeeperRequestError):
        second.admit_write(wreq(), decision_id)
    with pytest.raises(GatekeeperRequestError):
        second.record_write_outcome(decision_id, "failed", outcome_id)


def test_a_second_owner_closes_an_earlier_write_after_handover_and_unknown_still_blocks(tmp_path):
    store, clock, gate = approved(tmp_path)
    decision_id = eid()
    gate.admit_write(wreq(), decision_id)
    expire(clock)
    gate.release_lease(eid())
    second = other(store, clock, owner_id=OWNER_B)
    second.acquire_lease(eid())
    outcome_id = eid()
    assert second.record_write_outcome(decision_id, "unknown", outcome_id).status == "applied"
    event = last_event(store)
    assert (event.lease_owner, event.lease_epoch, event.write_outcome) == (OWNER_B, 2, "unknown")
    rows = count(store)
    with pytest.raises(GatekeeperStateError, match="unknown"):
        second.admit_write(wreq(), eid())
    assert gate.admit_write(wreq(), decision_id).status == "already_applied"  # the first owner can still confirm
    assert count(store) == rows
    verify(store)


def _g2b_calls(gate, ids):
    return {
        "write": lambda: gate.admit_write(wreq(), ids["write"], content_hash=HASH),
        "outcome": lambda: gate.record_write_outcome(ids["write"], "failed", ids["outcome"]),
    }


@pytest.mark.parametrize("method", ["write", "outcome"])
@pytest.mark.parametrize("mode,expected", [("landed", "already_applied"), ("absent", "applied")])
def test_unknown_commit_on_each_write_method_is_execute_or_confirm(tmp_path, method, mode, expected):
    store, clock, gate = approved(tmp_path)
    ids = {name: eid() for name in ("write", "outcome")}
    calls = _g2b_calls(gate, ids)
    if method == "outcome":
        calls["write"]()
    clock.adv(10)
    real = store._conn
    store._conn = Wrap(real, mode)
    with pytest.raises(control.ControlUnavailable) as caught:
        calls[method]()
    assert caught.value.append_state == "unknown"
    store._conn = real
    assert gate.event_status(ids[method]).state == ("present" if mode == "landed" else "absent")
    clock.adv(2)
    assert calls[method]().status == expected
    assert store._conn.execute(
        "SELECT count(*) FROM audit_events WHERE event_id = ?", (ids[method],)
    ).fetchone()[0] == 1
    verify(store)


@pytest.mark.parametrize("needle", [READ_CHAIN, INSERT_EVENT])
@pytest.mark.parametrize("method", ["write", "outcome"])
def test_sqlite_error_in_a_write_method_propagates_unchanged_after_a_confirmed_rollback(tmp_path, method, needle):
    store, clock, gate = approved(tmp_path)
    ids = {name: eid() for name in ("write", "outcome")}
    calls = _g2b_calls(gate, ids)
    if method == "outcome":
        calls["write"]()
    clock.adv(10)
    real = store._conn
    rows = count(store)
    store._conn = FailOn(real, needle)
    with pytest.raises(sqlite3.OperationalError, match="injected statement failure") as caught:
        calls[method]()
    assert not isinstance(caught.value, control.ControlUnavailable)
    store._conn = real
    assert count(store) == rows and real.in_transaction is False
    verify(store)


def test_each_new_write_event_reads_the_injected_clock_once_and_duplicates_never(tmp_path, no_system_clock):
    clock = CountingClock()
    store, _, gate = make(tmp_path, clock=clock)
    gate.start_run(eid())
    gate.acquire_lease(eid())
    gate.approve("target-1", eid())
    ids = {name: eid() for name in ("deny", "write", "outcome")}
    steps = (
        lambda: gate.admit_write(wreq("other"), ids["deny"]),
        lambda: gate.admit_write(wreq(), ids["write"], content_hash=HASH),
        lambda: gate.record_write_outcome(ids["write"], "unknown", ids["outcome"]),
    )
    for step in steps:
        before = clock.reads
        clock.adv(5)
        step()
        assert clock.reads == before + 1
    reads, rows = clock.reads, count(store)
    silence(gate)
    for step in steps:
        assert step().status == "already_applied"
    assert clock.reads == reads and count(store) == rows
    events = stored_events(store)[3:]
    for index, event in enumerate(events):
        stamped = T0 + timedelta(seconds=5 * (index + 1))
        assert datetime.fromisoformat(event.recorded_at) == stamped
        assert datetime.fromisoformat(event.observed_now) == stamped
    verify(store)


def test_event_status_reports_the_write_fields_and_the_outcome_id_and_stays_read_only(tmp_path):
    store, clock, gate = approved(tmp_path)
    ids = {name: eid() for name in ("deny", "write", "outcome", "fetch", "fetch_outcome")}
    gate.admit_write(wreq("other"), ids["deny"], content_hash=HASH)
    gate.admit_write(wreq(), ids["write"], content_hash=HASH)
    gate.admit_fetch(freq(), ids["fetch"])
    gate.record_fetch_outcome(ids["fetch"], "ok", ids["fetch_outcome"])
    before = fingerprint_of_files(tmp_path)
    rows = count(store)
    status = gate.event_status(ids["write"])
    assert (status.target_id, status.content_hash, status.write_outcome, status.outcome_event_id) == (
        "target-1", HASH, None, None,
    )
    assert gate.event_status(ids["deny"]).reason_code == "unapproved_write"
    assert gate.event_status(ids["fetch"]).outcome_event_id == ids["fetch_outcome"]
    gate.record_write_outcome(ids["write"], "committed", ids["outcome"])
    assert gate.event_status(ids["write"]).outcome_event_id == ids["outcome"]
    assert gate.event_status(ids["deny"]).outcome_event_id is None
    assert gate.event_status(ids["outcome"]).outcome_event_id is None
    assert gate.event_status(eid()).write_outcome is None
    before = fingerprint_of_files(tmp_path)
    rows = count(store)
    gate.event_status(ids["write"])
    assert fingerprint_of_files(tmp_path) == before and count(store) == rows


def test_write_leaves_llm_fetch_and_approval_behaviour_unchanged(tmp_path):
    store, clock, gate = approved(tmp_path)
    write_id = eid()
    gate.admit_write(wreq(), write_id)
    decision_id = eid()
    assert gate.admit_llm(req(), decision_id).decision.allow
    gate.record_llm_outcome(decision_id, 98, 40, eid())
    assert gate.admit_fetch(freq(), eid()).decision.allow
    with pytest.raises(GatekeeperStateError):
        gate.approve("target-2", eid())
    state = run_state(store)
    assert (state.calls, state.fetch_count, state.approved_target) == (1, 1, "target-1")
    assert state.reserved_tokens == 0 and write_id in state.decisions and write_id not in state.normal_outcomes
    verify(store)


# ============================================================ G2c: lease takeover and llm reconciliation


def lease_events(store):
    return [event for event in stored_events(store) if event.event_kind.startswith("lease_")]


def counting_boot(path, **policy):
    clock = CountingClock()
    store, clock, gate = make(path, clock=clock, **policy)
    gate.start_run(eid())
    gate.acquire_lease(eid())
    return store, clock, gate


def unknown_llm(path, kind="missing", **policy):
    store, clock, gate = boot(path, **policy)
    clock.adv(10)
    decision_id = eid()
    gate.admit_llm(req(), decision_id)
    clock.adv(5)
    unknown_id = eid()
    prompt, completion = (None, None) if kind == "missing" else (-1, 5)
    gate.record_llm_outcome(decision_id, prompt, completion, unknown_id)
    return store, clock, gate, decision_id, unknown_id


def taken_over(store, clock):
    """Expire the live lease and let OWNER_B take it over; returns the second gatekeeper."""
    expire(clock)
    second = other(store, clock, owner_id=OWNER_B)
    second.take_over_lease(eid())
    clock.adv(1)
    return second


def test_takeover_of_an_expired_lease_by_a_different_owner(tmp_path):
    store, clock, gate = boot(tmp_path)
    first_lease = lease_events(store)[0]
    expire(clock)
    second = other(store, clock, owner_id=OWNER_B)
    takeover_id = eid()
    result = second.take_over_lease(takeover_id)
    assert result.status == "applied" and result.decision is None
    event = last_event(store)
    assert (event.event_kind, event.operation, event.decision, event.reason_code) == (
        "lease_taken_over", "run", "allow", "allowed",
    )
    assert (event.lease_owner, event.lease_epoch) == (OWNER_B, 2)
    assert (event.previous_lease_owner, event.previous_lease_epoch) == (OWNER_A, 1)
    assert event.references_event_id == first_lease.event_id
    assert event.lease_ttl_seconds == 300 and not event.counters
    assert _expires(event) == _expires(first_lease) + timedelta(seconds=301)
    lease = run_state(store).lease
    assert (lease.owner_id, lease.epoch, lease.released, lease.last_event_id) == (OWNER_B, 2, False, takeover_id)
    verify(store)


def _expires(event):
    return datetime.fromisoformat(event.lease_expires_at)


def test_takeover_is_allowed_exactly_at_expiry_and_refused_one_second_before(tmp_path):
    store, clock, gate = boot(tmp_path)
    clock.adv(299)
    second = other(store, clock, owner_id=OWNER_B)
    rows = count(store)
    with pytest.raises(GatekeeperLeaseError, match="not expired"):
        second.take_over_lease(eid())
    assert count(store) == rows
    clock.adv(1)  # now == expires_at
    assert second.take_over_lease(eid()).status == "applied"
    verify(store)


def test_takeover_is_refused_for_the_current_owner_without_reading_the_clock(tmp_path):
    store, clock, gate = counting_boot(tmp_path)
    expire(clock)
    rows, reads = count(store), clock.reads
    with pytest.raises(GatekeeperLeaseError, match="different from the current lease owner"):
        gate.take_over_lease(eid())
    assert count(store) == rows and clock.reads == reads
    verify(store)


def test_takeover_needs_an_unreleased_lease_and_a_started_run(tmp_path):
    store, clock, gate = boot(tmp_path)
    expire(clock)
    gate.release_lease(eid())  # a released lease is acquired, not taken over
    second = other(store, clock, owner_id=OWNER_B)
    rows = count(store)
    with pytest.raises(GatekeeperLeaseError, match="no unreleased lease"):
        second.take_over_lease(eid())
    store2, clock2, gate2 = make(tmp_path / "nolease")
    gate2.start_run(eid())
    with pytest.raises(GatekeeperLeaseError, match="no unreleased lease"):
        gate2.take_over_lease(eid())
    store3, clock3, gate3 = make(tmp_path / "norun")
    with pytest.raises(GatekeeperUnknownRun):
        gate3.take_over_lease(eid())
    assert count(store) == rows and count(store2) == 1 and count(store3) == 0


def test_takeover_references_the_last_lease_event_even_when_it_is_a_renewal(tmp_path):
    store, clock, gate = boot(tmp_path)
    clock.adv(100)
    renew_id = eid()
    gate.renew_lease(renew_id)  # expires at T0+400
    clock.adv(301)
    second = other(store, clock, owner_id=OWNER_B)
    second.take_over_lease(eid())
    event = last_event(store)
    assert event.references_event_id == renew_id and event.previous_lease_epoch == 1
    assert event.lease_epoch == 2
    verify(store)


def test_a_takeover_can_follow_a_takeover(tmp_path):
    store, clock, gate = boot(tmp_path)
    second = taken_over(store, clock)
    expire(clock)
    third = other(store, clock, owner_id="00000000-0000-0000-0000-00000000000c")
    third.take_over_lease(eid())
    event = last_event(store)
    assert (event.lease_epoch, event.previous_lease_owner, event.previous_lease_epoch) == (3, OWNER_B, 2)
    assert event.references_event_id == lease_events(store)[1].event_id
    with pytest.raises(GatekeeperLeaseError):
        second.renew_lease(eid())
    verify(store)


def test_after_takeover_the_old_owner_is_locked_out_and_the_new_owner_acts(tmp_path):
    store, clock, gate = boot(tmp_path)
    second = taken_over(store, clock)
    rows = count(store)
    for call in (
        lambda: gate.admit_llm(req(), eid()),
        lambda: gate.renew_lease(eid()),
        lambda: gate.release_lease(eid()),
        lambda: gate.take_over_lease(eid()),
    ):
        with pytest.raises(GatekeeperLeaseError):
            call()
    assert count(store) == rows
    assert second.admit_llm(req(), eid()).decision.allow
    verify(store)


def test_expiry_is_judged_only_by_the_new_owners_own_clock(tmp_path):
    # Known limit: a fast clock in the new owner takes a lease that is live by the old owner's clock.
    store, clock, gate = boot(tmp_path)
    fast = other(store, Clock(T0 + timedelta(seconds=400)), owner_id=OWNER_B)
    assert fast.take_over_lease(eid()).status == "applied"
    clock.adv(400)
    with pytest.raises(GatekeeperLeaseError, match="another owner"):
        gate.admit_llm(req(), eid())  # the old owner is fenced from now on
    verify(store)


def test_takeover_duplicate_is_confirmed_without_a_clock_and_a_foreign_one_is_refused(tmp_path, no_system_clock):
    store, clock, gate = boot(tmp_path)
    expire(clock)
    clock2 = CountingClock(clock())
    second = other(store, clock2, owner_id=OWNER_B)
    takeover_id = eid()
    second.take_over_lease(takeover_id)
    second.admit_llm(req(), eid())
    clock2.adv(1000)
    rows, reads = count(store), clock2.reads
    assert second.take_over_lease(takeover_id).status == "already_applied"
    assert count(store) == rows and clock2.reads == reads
    third = other(store, clock2, owner_id="00000000-0000-0000-0000-00000000000c")
    with pytest.raises(control.AuditIncomplete, match="different logical fingerprint"):
        third.take_over_lease(takeover_id)
    assert count(store) == rows
    verify(store)


def test_takeover_event_id_of_another_kind_is_refused(tmp_path):
    store, clock, gate = boot(tmp_path)
    clock.adv(10)
    decision_id = eid()
    gate.admit_llm(req(), decision_id)
    expire(clock)
    second = other(store, clock, owner_id=OWNER_B)
    rows = count(store)
    with pytest.raises(GatekeeperRequestError, match="different event kind"):
        second.take_over_lease(decision_id)
    with pytest.raises(GatekeeperRequestError):
        second.take_over_lease("")
    assert count(store) == rows


def test_event_status_reports_a_takeover_event(tmp_path):
    store, clock, gate = boot(tmp_path)
    expire(clock)
    second = other(store, clock, owner_id=OWNER_B)
    takeover_id = eid()
    second.take_over_lease(takeover_id)
    status = second.event_status(takeover_id)
    assert (status.state, status.event_kind, status.operation, status.run_matches) == (
        "present", "lease_taken_over", "run", True,
    )
    assert (status.lease_owner, status.lease_epoch) == (OWNER_B, 2)
    assert status.references_event_id == lease_events(store)[0].event_id


def test_the_new_owner_records_outcomes_for_decisions_made_under_the_old_epoch(tmp_path):
    store, clock, gate = boot(tmp_path)
    clock.adv(10)
    decision_id = eid()
    gate.admit_llm(req(), decision_id)
    second = taken_over(store, clock)
    assert second.record_llm_outcome(decision_id, 98, 40, eid()).status == "applied"
    event = last_event(store)
    assert (event.lease_owner, event.lease_epoch, event.reason_code) == (OWNER_B, 2, "allowed")
    assert run_state(store).reserved_tokens == 0
    verify(store)


def test_takeover_leaves_the_failed_write_rule_unchanged_and_that_unlocks_a_new_write(tmp_path):
    # Known policy risk, pinned on purpose: the old worker may have written before it died, yet a
    # reported "failed" from the new owner lets a new write through (G2b rule, unchanged in G2c).
    store, clock, gate = approved(tmp_path)
    first = eid()
    assert gate.admit_write(wreq(), first).decision.allow
    second = taken_over(store, clock)
    with pytest.raises(GatekeeperStateError, match="no outcome"):
        second.admit_write(wreq(), eid())  # still blocked until an outcome is recorded
    second.record_write_outcome(first, "failed", eid())
    assert second.admit_write(wreq(), eid()).decision.allow
    verify(store)


def test_takeover_does_not_clear_a_write_unknown_either(tmp_path):
    store, clock, gate = approved(tmp_path)
    first = eid()
    gate.admit_write(wreq(), first)
    gate.record_write_outcome(first, "unknown", eid())
    second = taken_over(store, clock)
    with pytest.raises(GatekeeperStateError, match="unknown"):
        second.admit_write(wreq(), eid())
    verify(store)


# ---- reconciliation


@pytest.mark.parametrize("kind", ["missing", "range"])
def test_reconcile_settles_an_unknown_reservation_from_reported_usage(tmp_path, kind):
    store, clock, gate, decision_id, unknown_id = unknown_llm(tmp_path, kind)
    before = run_state(store)
    assert before.reserved_tokens == 637 and before.reservations[decision_id].status == "unknown"
    clock.adv(1)
    reconcile_id = eid()
    result = gate.reconcile_llm_outcome(unknown_id, 98, 40, reconcile_id)
    assert result.status == "applied" and result.decision is None
    event = last_event(store)
    assert (event.event_kind, event.operation, event.decision, event.reason_code) == (
        "outcome", "llm", "allow", "reconciliation",
    )
    assert event.references_event_id == unknown_id
    assert (event.model, event.schema_name) == ("gpt-4o-mini", "NewsDigest")
    assert (event.usage_prompt_tokens, event.usage_completion_tokens) == (98, 40)
    assert event.reservation_released == 637 and event.overrun_tokens is None
    assert dict(event.counters) == {
        "calls": 1,
        "prompt_tokens": 98,
        "completion_tokens": 40,
        "reserved_tokens": 0,
        "remaining_tokens": 20000 - 138,
        "budget_deficit": 0,
    }
    state = run_state(store)
    assert state.reserved_tokens == 0 and (state.prompt_tokens, state.completion_tokens) == (98, 40)
    assert state.reservations[decision_id].status == "resolved" and unknown_id in state.reconciled
    verify(store)


def test_reconcile_by_the_new_owner_after_takeover_frees_the_held_tokens(tmp_path):
    store, clock, gate, decision_id, unknown_id = unknown_llm(tmp_path)
    second = taken_over(store, clock)
    assert second.reconcile_llm_outcome(unknown_id, 98, 40, eid()).status == "applied"
    event = last_event(store)
    assert (event.lease_owner, event.lease_epoch) == (OWNER_B, 2)
    assert run_state(store).reserved_tokens == 0
    verify(store)


def test_reconcile_may_leave_a_token_deficit_and_carries_no_overrun(tmp_path):
    store, clock, gate, decision_id, unknown_id = unknown_llm(tmp_path, max_run_tokens=1000)
    clock.adv(1)
    gate.reconcile_llm_outcome(unknown_id, 900, 500, eid())
    event = last_event(store)
    assert event.reason_code == "reconciliation" and event.overrun_tokens is None
    assert event.counters["budget_deficit"] == 400 and event.counters["remaining_tokens"] == 0
    verify(store)


def test_reconcile_usage_above_the_reservation_without_a_deficit_has_no_overrun(tmp_path):
    store, clock, gate, decision_id, unknown_id = unknown_llm(tmp_path)
    clock.adv(1)
    gate.reconcile_llm_outcome(unknown_id, 600, 100, eid())  # 700 > 637 held, balance still positive
    event = last_event(store)
    assert event.overrun_tokens is None and event.counters["budget_deficit"] == 0
    verify(store)


@pytest.mark.parametrize(
    "prompt,completion",
    [(None, None), (1, None), (None, 1), (-1, 5), (True, 5), (5.0, 1), ("5", 1), (INT64 + 1, 0), (0, INT64 + 1), (INT64, INT64)],
)
def test_reconcile_refuses_missing_or_unusable_usage_and_writes_nothing(tmp_path, prompt, completion):
    store, clock, gate, decision_id, unknown_id = unknown_llm(tmp_path)
    rows = count(store)
    clock.adv(1)
    with pytest.raises(GatekeeperError):
        gate.reconcile_llm_outcome(unknown_id, prompt, completion, eid())
    assert count(store) == rows
    state = run_state(store)
    assert state.reserved_tokens == 637 and state.reservations[decision_id].status == "unknown"
    verify(store)


def test_reconcile_refuses_anything_that_is_not_an_open_llm_unknown(tmp_path):
    store, clock, gate, decision_id, unknown_id = unknown_llm(tmp_path)
    normal = eid()
    other_decision = eid()
    gate.admit_llm(req(), other_decision)
    gate.record_llm_outcome(other_decision, 98, 40, normal)
    rows = count(store)
    clock.adv(1)
    for bad in (decision_id, normal, other_decision, eid(), "x"):
        with pytest.raises(GatekeeperRequestError):
            gate.reconcile_llm_outcome(bad, 98, 40, eid())
    with pytest.raises(GatekeeperRequestError):
        gate.reconcile_llm_outcome("", 98, 40, eid())
    with pytest.raises(GatekeeperRequestError):
        gate.reconcile_llm_outcome(None, 98, 40, eid())
    assert count(store) == rows
    verify(store)


def test_reconcile_works_only_once_per_unknown(tmp_path):
    store, clock, gate, decision_id, unknown_id = unknown_llm(tmp_path)
    clock.adv(1)
    first = eid()
    gate.reconcile_llm_outcome(unknown_id, 98, 40, first)
    rows = count(store)
    with pytest.raises(GatekeeperRequestError, match="not unknown"):
        gate.reconcile_llm_outcome(unknown_id, 98, 40, eid())
    assert gate.reconcile_llm_outcome(unknown_id, 98, 40, first).status == "already_applied"
    assert count(store) == rows
    verify(store)


def test_reconcile_needs_a_live_lease_owned_by_the_caller(tmp_path):
    store, clock, gate, decision_id, unknown_id = unknown_llm(tmp_path)
    foreign = other(store, clock, owner_id=OWNER_B)
    rows = count(store)
    clock.adv(1)
    with pytest.raises(GatekeeperLeaseError):
        foreign.reconcile_llm_outcome(unknown_id, 98, 40, eid())  # no takeover yet
    expire(clock)
    with pytest.raises(GatekeeperLeaseError):
        gate.reconcile_llm_outcome(unknown_id, 98, 40, eid())  # expired
    assert count(store) == rows
    with pytest.raises(GatekeeperLeaseError):
        foreign.reconcile_llm_outcome(unknown_id, 98, 40, eid())  # expired and not taken over
    assert count(store) == rows
    verify(store)


def test_reconcile_refuses_a_write_unknown(tmp_path):
    store, clock, gate = approved(tmp_path)
    write_id, outcome_id = eid(), eid()
    gate.admit_write(wreq(), write_id)
    gate.record_write_outcome(write_id, "unknown", outcome_id)
    rows = count(store)
    for target in (write_id, outcome_id):
        with pytest.raises(GatekeeperRequestError):
            gate.reconcile_llm_outcome(target, 1, 1, eid())
    assert count(store) == rows
    verify(store)


def test_reconcile_reads_the_clock_once_for_a_new_event_and_never_for_a_duplicate_or_refusal(tmp_path, no_system_clock):
    clock = CountingClock()
    store, clock, gate = make(tmp_path, clock=clock)
    gate.start_run(eid())
    gate.acquire_lease(eid())
    clock.adv(10)
    decision_id, unknown_id, reconcile_id = eid(), eid(), eid()
    gate.admit_llm(req(), decision_id)
    gate.record_llm_outcome(decision_id, None, None, unknown_id)
    reads = clock.reads
    with pytest.raises(GatekeeperRequestError):
        gate.reconcile_llm_outcome(unknown_id, None, None, eid())
    assert clock.reads == reads
    gate.reconcile_llm_outcome(unknown_id, 98, 40, reconcile_id)
    assert clock.reads == reads + 1
    clock.adv(1000)  # even an expired lease cannot stop a confirmation
    before = clock.reads
    assert gate.reconcile_llm_outcome(unknown_id, 98, 40, reconcile_id).status == "already_applied"
    assert clock.reads == before


def test_reconcile_duplicate_with_other_usage_reference_or_owner_is_not_confirmed(tmp_path):
    store, clock, gate, decision_id, unknown_id = unknown_llm(tmp_path)
    clock.adv(1)
    reconcile_id = eid()
    gate.reconcile_llm_outcome(unknown_id, 98, 40, reconcile_id)
    rows = count(store)
    with pytest.raises(control.AuditIncomplete, match="different logical fingerprint"):
        gate.reconcile_llm_outcome(unknown_id, 99, 40, reconcile_id)
    with pytest.raises(GatekeeperRequestError, match="different"):
        gate.reconcile_llm_outcome(decision_id, 98, 40, reconcile_id)
    foreign = other(store, clock, owner_id=OWNER_B)
    with pytest.raises(GatekeeperRequestError, match="another lease owner"):
        foreign.reconcile_llm_outcome(unknown_id, 98, 40, reconcile_id)
    assert count(store) == rows
    verify(store)


def test_reconcile_event_id_of_another_kind_is_refused(tmp_path):
    store, clock, gate, decision_id, unknown_id = unknown_llm(tmp_path)
    clock.adv(1)
    rows = count(store)
    with pytest.raises(GatekeeperRequestError, match="different event kind"):
        gate.reconcile_llm_outcome(unknown_id, 98, 40, decision_id)
    assert count(store) == rows


def test_reconciled_tokens_count_against_the_budget_for_later_admission(tmp_path):
    store, clock, gate, decision_id, unknown_id = unknown_llm(tmp_path, max_run_tokens=1200)
    clock.adv(1)
    assert not gate.admit_llm(req(100), eid()).decision.allow  # 637 still held while unknown
    gate.reconcile_llm_outcome(unknown_id, 20, 10, eid())
    clock.adv(1)
    assert gate.admit_llm(req(100), eid()).decision.allow
    verify(store)


def _g2c_calls(gate, second, ids):
    return {
        "takeover": lambda: second.take_over_lease(ids["takeover"]),
        "reconcile": lambda: gate.reconcile_llm_outcome(ids["unknown"], 98, 40, ids["reconcile"]),
    }


@pytest.mark.parametrize("method", ["takeover", "reconcile"])
@pytest.mark.parametrize("mode,expected", [("landed", "already_applied"), ("absent", "applied")])
def test_unknown_commit_on_each_g2c_method_is_execute_or_confirm(tmp_path, method, mode, expected):
    store, clock, gate, decision_id, unknown_id = unknown_llm(tmp_path)
    ids = {"takeover": eid(), "reconcile": eid(), "unknown": unknown_id}
    if method == "takeover":
        expire(clock)
        actor = other(store, clock, owner_id=OWNER_B)
    else:
        clock.adv(1)
        actor = gate
    calls = _g2c_calls(gate, actor, ids)
    real = store._conn
    store._conn = Wrap(real, mode)
    with pytest.raises(control.ControlUnavailable) as caught:
        calls[method]()
    assert caught.value.append_state == "unknown"
    store._conn = real
    assert actor.event_status(ids[method]).state == ("present" if mode == "landed" else "absent")
    clock.adv(2)
    assert calls[method]().status == expected
    assert store._conn.execute(
        "SELECT count(*) FROM audit_events WHERE event_id = ?", (ids[method],)
    ).fetchone()[0] == 1
    verify(store)


@pytest.mark.parametrize("needle", [READ_CHAIN, INSERT_EVENT])
@pytest.mark.parametrize("method", ["takeover", "reconcile"])
def test_sqlite_error_in_a_g2c_method_propagates_unchanged_after_a_confirmed_rollback(tmp_path, method, needle):
    store, clock, gate, decision_id, unknown_id = unknown_llm(tmp_path)
    ids = {"takeover": eid(), "reconcile": eid(), "unknown": unknown_id}
    if method == "takeover":
        expire(clock)
        actor = other(store, clock, owner_id=OWNER_B)
    else:
        clock.adv(1)
        actor = gate
    calls = _g2c_calls(gate, actor, ids)
    real = store._conn
    rows = count(store)
    store._conn = FailOn(real, needle)
    with pytest.raises(sqlite3.OperationalError, match="injected statement failure") as caught:
        calls[method]()
    assert not isinstance(caught.value, control.ControlUnavailable)
    store._conn = real
    assert count(store) == rows and real.in_transaction is False
    verify(store)


def test_the_two_new_methods_leave_earlier_behaviour_unchanged(tmp_path):
    store, clock, gate, decision_id, unknown_id = unknown_llm(tmp_path)
    second = taken_over(store, clock)
    second.reconcile_llm_outcome(unknown_id, 98, 40, eid())
    assert second.admit_fetch(freq(), eid()).decision.allow
    with pytest.raises(GatekeeperLeaseError):
        second.acquire_lease(eid())  # a live lease still blocks acquire
    second.release_lease(eid())
    state = run_state(store)
    assert (state.calls, state.fetch_count, state.reserved_tokens) == (1, 1, 0)
    assert state.lease.released and state.lease.epoch == 2
    verify(store)
