"""Offline tests for the gatekeeper (slice 2b-iii, first batch).

They use the real AuditStore, the real LedgerProjection and the real admission
functions, with a fake clock. Failure injection wraps the store connection.
"""
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
