"""Offline tests for the pure ledger replay and read-only verifier (2b-ii-A)."""
import hashlib
import json
import sqlite3
import uuid

import pytest

import market_intelligence.admission as admission
import market_intelligence.control as control
import market_intelligence.ledger as ledger
from market_intelligence.control import AuditStore, ControlUnavailable


OWNER_1 = "00000000-0000-0000-0000-000000000001"


def _policy():
    return {
        "policy_margin": 0.25,
        "policy_margin_num": 1,
        "policy_margin_den": 4,
        "provider_overhead_tokens": 0,
        "max_run_tokens": 20000,
        "max_run_llm_calls": 24,
        "max_output_tokens": 512,
        "max_fetches_per_run": 8,
        "lease_ttl_seconds": 300,
        "allowed_model": "gpt-4o-mini",
        "allowed_schemas": ("NewsDigest",),
        "estimate_kind": "policy_margin_v1",
    }


def _event(**overrides):
    values = {
        "run_id": "run-ledger",
        "thread_id": "thread-ledger",
        "policy_version": control.POLICY_VERSION_H1_SLICE2,
    }
    values.update(overrides)
    return control.make_event(**values)


def _start(store):
    store.append(
        _event(
            operation="run",
            event_kind="run_started",
            decision="allow",
            reason_code="allowed",
            counters={},
            **_policy(),
        )
    )
    acquired = _event(
        operation="run",
        event_kind="lease_acquired",
        decision="allow",
        reason_code="allowed",
        counters={},
        lease_owner=OWNER_1,
        lease_epoch=1,
        observed_now="2026-11-12T00:00:00+00:00",
        lease_expires_at="2026-11-12T00:05:00+00:00",
        lease_ttl_seconds=300,
    )
    store.append(acquired)
    return acquired


def _decision(store, *, model="gpt-4o-mini", schema="NewsDigest", reserve=637, **overrides):
    values = {
        "operation": "llm",
        "event_kind": "decision",
        "decision": "allow",
        "reason_code": "allowed",
        "counters": {},
        "model": model,
        "schema_name": schema,
        "serialized_body_token_count": 100,
        "reserve_tokens": reserve,
        "lease_owner": OWNER_1,
        "lease_epoch": 1,
        "observed_now": "2026-11-12T00:00:10+00:00",
    }
    values.update(_policy())
    values.update(overrides)
    event = _event(**values)
    store.append(event)
    return event


def _counters(prompt=98, completion=40, reserved=0):
    return _counters_for(20000, prompt, completion, reserved)


def _counters_for(max_tokens, prompt=98, completion=40, reserved=0):
    balance = max_tokens - prompt - completion - reserved
    return {
        "calls": 1,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "reserved_tokens": reserved,
        "remaining_tokens": max(0, balance),
        "budget_deficit": max(0, -balance),
    }


def _outcome(store, decision, **overrides):
    values = {
        "operation": "llm",
        "event_kind": "outcome",
        "decision": "allow",
        "reason_code": "allowed",
        "counters": _counters(),
        "model": "gpt-4o-mini",
        "schema_name": "NewsDigest",
        "usage_prompt_tokens": 98,
        "usage_completion_tokens": 40,
        "reservation_released": 637,
        "references_event_id": decision.event_id,
        "lease_owner": OWNER_1,
        "lease_epoch": 1,
        "observed_now": "2026-11-12T00:00:20+00:00",
    }
    values.update(overrides)
    event = _event(**values)
    store.append(event)
    return event


def _events(db):
    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT seq, payload FROM audit_events ORDER BY seq"
        ).fetchall()
    events = []
    for seq, payload in rows:
        events.append(control.event_from_payload(json.loads(payload), f"seq {seq}"))
    return events


def _insert_payload_direct(db, payload):
    """Insert a hash-valid chain row directly. Used only by the chain-owned counter tests, for a row AuditStore.append refuses; it models a tampered database, not a path the ledger can receive through append."""
    with sqlite3.connect(db) as conn:
        last = conn.execute(
            "SELECT seq, event_hash FROM audit_events ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        prev_hash = last[1] if last else None
        digest = control._event_hash(payload, prev_hash)
        text = control.canonical_json({**payload, "prev_hash": prev_hash})
        seq = (last[0] + 1) if last else 1
        conn.execute(
            "INSERT INTO audit_events (event_id, prev_hash, event_hash, payload) "
            "VALUES (?, ?, ?, ?)",
            (payload["event_id"], prev_hash, digest, text),
        )
        conn.execute(
            "UPDATE audit_head SET seq = ?, event_hash = ? WHERE singleton = 1",
            (seq, digest),
        )


def _state(db):
    return ledger.replay_events(_events(db))


def _insert_ledger_rows(db, state):
    with sqlite3.connect(db) as conn:
        conn.executescript(ledger.SCHEMA_SQL)
        head = conn.execute(
            "SELECT seq, event_hash FROM audit_head WHERE singleton = 1"
        ).fetchone()
        rows = ledger.state_to_rows(state, head)
        for table in ("run_policy", "run_ledger", "reservations", "run_leases", "ledger_head"):
            values = rows[table]
            if not values:
                continue
            placeholders = ",".join("?" for _ in values[0])
            conn.executemany(
                f"INSERT INTO {table} VALUES ({placeholders})", values
            )


def _valid_chain(tmp_path):
    db = tmp_path / "audit-ledger.sqlite3"
    store = AuditStore(db)
    _start(store)
    decision = _decision(store)
    _outcome(store, decision)
    store.close()
    assert control.verify_chain(db) == []
    return db, decision


def test_next_state_run_started_creates_zero_ledger():
    event = _event(
        operation="run",
        event_kind="run_started",
        decision="allow",
        reason_code="allowed",
        counters={},
        **_policy(),
    )
    state = ledger.next_state(ledger.LedgerState(), event)
    rows = ledger.state_to_rows(state, (1, "a" * 64))
    assert rows["run_ledger"] == [("run-ledger", 0, 0, 0, 0, 0)]
    assert rows["run_policy"][0][1] == "thread-ledger"


def test_allowed_llm_decision_opens_reservation(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    _start(store)
    decision = _decision(store)
    store.close()
    state = _state(db)
    run = state.runs["run-ledger"]
    assert run.calls == 1
    assert run.reserved_tokens == 637
    assert run.reservations[decision.event_id].status == "open"


def test_allowed_fetch_decision_increments_fetch_count(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    _start(store)
    store.append(
        _event(
            operation="fetch",
            event_kind="decision",
            decision="allow",
            reason_code="allowed",
            counters={},
            target_host="example.com",
            lease_owner=OWNER_1,
            lease_epoch=1,
            observed_now="2026-11-12T00:00:10+00:00",
            **_policy(),
        )
    )
    store.close()
    assert _state(db).runs["run-ledger"].fetch_count == 1


def test_allowed_write_decision_does_not_count(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    _start(store)
    store.append(
        _event(
            operation="run",
            event_kind="run_approved",
            decision="allow",
            reason_code="allowed",
            counters={},
            lease_owner=OWNER_1,
            lease_epoch=1,
            observed_now="2026-11-12T00:00:05+00:00",
            approved_target_id="target-1",
        )
    )
    store.append(
        _event(
            operation="write",
            event_kind="decision",
            decision="allow",
            reason_code="allowed",
            counters={},
            target_id="target-1",
            lease_owner=OWNER_1,
            lease_epoch=1,
            observed_now="2026-11-12T00:00:10+00:00",
            **_policy(),
        )
    )
    store.close()
    run = _state(db).runs["run-ledger"]
    assert run.calls == 0 and run.fetch_count == 0 and run.reserved_tokens == 0


def test_normal_outcome_updates_usage_and_releases(tmp_path):
    db, decision = _valid_chain(tmp_path)
    state = _state(db)
    run = state.runs["run-ledger"]
    assert run.calls == 1
    assert run.prompt_tokens == 98
    assert run.completion_tokens == 40
    assert run.reserved_tokens == 0
    assert run.reservations[decision.event_id].status == "applied"
    _insert_ledger_rows(db, state)
    assert ledger.verify_ledger(db) == []


def test_overrun_and_budget_breach(tmp_path):
    db = tmp_path / "audit-a.sqlite3"
    store = AuditStore(db)
    _start(store)
    decision = _decision(store)
    _outcome(
        store,
        decision,
        reason_code="reservation_overrun",
        counters=_counters(prompt=120, completion=600),
        usage_prompt_tokens=120,
        usage_completion_tokens=600,
        overrun_tokens=83,
    )
    store.close()
    assert control.verify_chain(db) == []
    run = _state(db).runs["run-ledger"]
    assert run.prompt_tokens == 120 and run.completion_tokens == 600
    assert run.reserved_tokens == 0
    assert ledger._counter_snapshot(run) == _counters(prompt=120, completion=600)

    db2 = tmp_path / "audit-b.sqlite3"
    store2 = AuditStore(db2)
    _start(store2)
    decision2 = _decision(store2)
    _outcome(
        store2,
        decision2,
        reason_code="budget_breach",
        counters=_counters(prompt=12000, completion=9000),
        usage_prompt_tokens=12000,
        usage_completion_tokens=9000,
        overrun_tokens=20363,
    )
    store2.close()
    assert control.verify_chain(db2) == []
    run2 = _state(db2).runs["run-ledger"]
    assert run2.reserved_tokens == 0
    snapshot = ledger._counter_snapshot(run2)
    assert snapshot == _counters(prompt=12000, completion=9000)
    assert snapshot["budget_deficit"] == 1000
    assert snapshot["remaining_tokens"] == 0


def test_usage_missing_and_out_of_range_hold_reservation(tmp_path):
    for reason in ("usage_missing", "usage_out_of_range"):
        db = tmp_path / f"audit-{reason}.sqlite3"
        store = AuditStore(db)
        _start(store)
        decision = _decision(store)
        extra = {"error_class": "UsageOutOfRange"} if reason == "usage_out_of_range" else {}
        store.append(
            _event(
                operation="llm",
                event_kind="outcome",
                decision="allow",
                reason_code=reason,
                counters={},
                model="gpt-4o-mini",
                schema_name="NewsDigest",
                usage_missing=True,
                reservation_released=0,
                references_event_id=decision.event_id,
                lease_owner=OWNER_1,
                lease_epoch=1,
                observed_now="2026-11-12T00:00:20+00:00",
                **extra,
            )
        )
        store.close()
        assert control.verify_chain(db) == []
        run = _state(db).runs["run-ledger"]
        assert run.reserved_tokens == 637
        assert run.reservations[decision.event_id].status == "unknown"


def test_released_no_effect_and_reconciliation_with_usage(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    _start(store)
    decision = _decision(store)
    _outcome(
        store,
        decision,
        reason_code="released_no_effect",
        counters={},
        usage_prompt_tokens=None,
        usage_completion_tokens=None,
        reservation_released=637,
    )
    store.close()
    assert _state(db).runs["run-ledger"].reservations[decision.event_id].status == "released"

    db2 = tmp_path / "audit-b.sqlite3"
    store2 = AuditStore(db2)
    _start(store2)
    decision2 = _decision(store2)
    unknown = _outcome(
        store2,
        decision2,
        reason_code="usage_missing",
        counters={},
        usage_prompt_tokens=None,
        usage_completion_tokens=None,
        reservation_released=0,
        usage_missing=True,
    )
    _outcome(
        store2,
        decision2,
        reason_code="reconciliation",
        counters=_counters(),
        references_event_id=unknown.event_id,
    )
    store2.close()
    run = _state(db2).runs["run-ledger"]
    assert run.reservations[decision2.event_id].status == "resolved"
    assert run.reserved_tokens == 0


def test_usage_less_reconciliation_is_rejected(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    _start(store)
    decision = _decision(store)
    unknown = _outcome(
        store,
        decision,
        reason_code="usage_missing",
        counters={},
        usage_prompt_tokens=None,
        usage_completion_tokens=None,
        reservation_released=0,
        usage_missing=True,
    )
    store.append(
        _event(
            operation="llm",
            event_kind="outcome",
            decision="allow",
            reason_code="reconciliation",
            counters={},
            model="gpt-4o-mini",
            schema_name="NewsDigest",
            references_event_id=unknown.event_id,
            lease_owner=OWNER_1,
            lease_epoch=1,
            observed_now="2026-11-12T00:00:30+00:00",
        )
    )
    store.close()
    assert control.verify_chain(db) == []
    with pytest.raises(ledger.LedgerIncompleteReconciliation, match="reconciliation without usage does not settle"):
        _state(db)


def test_denied_decision_outcome_and_release_mismatch_are_rejected(tmp_path):
    db = tmp_path / "audit-d.sqlite3"
    store = AuditStore(db)
    _start(store)
    denied = _event(
        operation="llm",
        event_kind="decision",
        decision="deny",
        reason_code="policy_denied",
        counters={},
        model="gpt-4o-mini",
        schema_name="NewsDigest",
        serialized_body_token_count=100,
        lease_owner=OWNER_1,
        lease_epoch=1,
        observed_now="2026-11-12T00:00:10+00:00",
        **_policy(),
    )
    store.append(denied)
    _outcome(store, denied)
    store.close()
    assert control.verify_chain(db) == []
    with pytest.raises(ledger.LedgerInvalidTransition, match="outcome references a denied decision"):
        _state(db)

    db2 = tmp_path / "audit-r.sqlite3"
    store2 = AuditStore(db2)
    _start(store2)
    decision = _decision(store2)
    _outcome(store2, decision, reservation_released=1)
    store2.close()
    assert control.verify_chain(db2) == []
    with pytest.raises(ledger.LedgerInvalidEvent, match="release does not equal the held reservation"):
        _state(db2)


def test_counter_and_model_mismatches_are_rejected(tmp_path):
    db = tmp_path / "audit-c.sqlite3"
    store = AuditStore(db)
    _start(store)
    decision = _decision(store)
    _outcome(store, decision, counters=dict(_counters(), calls=99))
    store.close()
    assert control.verify_chain(db) == []
    with pytest.raises(ledger.LedgerInvalidEvent, match="outcome counters do not match derived ledger counters"):
        _state(db)

    db2 = tmp_path / "audit-m.sqlite3"
    store2 = AuditStore(db2)
    _start(store2)
    decision2 = _decision(store2)
    _outcome(store2, decision2, model="other-model")
    store2.close()
    assert control.verify_chain(db2) == []
    with pytest.raises(ledger.LedgerInvalidEvent, match="outcome model/schema differs from decision"):
        _state(db2)


def test_fetch_target_mismatch_is_rejected(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    _start(store)
    decision = _event(
        operation="fetch",
        event_kind="decision",
        decision="allow",
        reason_code="allowed",
        counters={},
        target_host="example.com",
        lease_owner=OWNER_1,
        lease_epoch=1,
        observed_now="2026-11-12T00:00:10+00:00",
        **_policy(),
    )
    store.append(decision)
    store.append(
        _event(
            operation="fetch",
            event_kind="outcome",
            decision="allow",
            reason_code="allowed",
            counters={},
            target_host="other.com",
            result_class="ok",
            references_event_id=decision.event_id,
            lease_owner=OWNER_1,
            lease_epoch=1,
            observed_now="2026-11-12T00:00:20+00:00",
        )
    )
    store.close()
    assert control.verify_chain(db) == []
    with pytest.raises(ledger.LedgerInvalidEvent, match="fetch outcome target differs from decision"):
        _state(db)


def test_reserve_formula_matches_admission():
    policy = admission.PolicyConfig(
        max_run_tokens=20000,
        max_run_llm_calls=24,
        max_output_tokens=512,
        max_fetches_per_run=8,
        reservation_margin=0.25,
        provider_overhead_tokens=0,
        allowed_model="gpt-4o-mini",
        allowed_schemas=frozenset({"NewsDigest"}),
    )
    snapshot = ledger.PolicySnapshot(
        margin_num=1,
        margin_den=4,
        provider_overhead_tokens=0,
        max_run_tokens=20000,
        max_run_llm_calls=24,
        max_output_tokens=512,
        max_fetches_per_run=8,
        lease_ttl_seconds=300,
        allowed_model="gpt-4o-mini",
        allowed_schemas=("NewsDigest",),
        estimate_kind="policy_margin_v1",
        policy_version=control.POLICY_VERSION_H1_SLICE2,
    )
    for count in (0, 3, 100, 2**53 + 1):
        assert ledger._reserve(snapshot, count) == admission._reserve(policy, count)


def test_verify_ledger_matching_and_tamper(tmp_path):
    db, _ = _valid_chain(tmp_path)
    state = _state(db)
    _insert_ledger_rows(db, state)
    assert ledger.verify_ledger(db) == []

    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE run_ledger SET calls = 99")
    problems = ledger.verify_ledger(db)
    assert any("run_ledger" in p for p in problems)


def test_verify_ledger_missing_table_and_chain_first(tmp_path):
    db, _ = _valid_chain(tmp_path)
    state = _state(db)
    _insert_ledger_rows(db, state)
    with sqlite3.connect(db) as conn:
        conn.execute("DROP TABLE run_ledger")
    problems = ledger.verify_ledger(db)
    assert any("run_ledger" in p and "missing" in p for p in problems)

    db2 = tmp_path / "audit-broken.sqlite3"
    store = AuditStore(db2)
    _start(store)
    store.close()
    with sqlite3.connect(db2) as conn:
        conn.execute("UPDATE audit_events SET payload = '{}' WHERE seq = 1")
    problems = ledger.verify_ledger(db2)
    assert problems
    assert problems == control.verify_chain(db2)
    assert not any("ledger table" in p for p in problems)


def test_verify_ledger_skips_table_compare_on_ledger_error(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    _start(store)
    denied = _event(
        operation="llm",
        event_kind="decision",
        decision="deny",
        reason_code="policy_denied",
        counters={},
        model="gpt-4o-mini",
        schema_name="NewsDigest",
        serialized_body_token_count=100,
        lease_owner=OWNER_1,
        lease_epoch=1,
        observed_now="2026-11-12T00:00:10+00:00",
        **_policy(),
    )
    store.append(denied)
    _outcome(store, denied)
    store.close()
    assert control.verify_chain(db) == []
    problems = ledger.verify_ledger(db)
    assert problems == ["seq 4: outcome references a denied decision"]
    assert not any("run_policy" in p for p in problems)


def test_verify_ledger_is_read_only(tmp_path):
    db, _ = _valid_chain(tmp_path)
    _insert_ledger_rows(db, _state(db))
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    assert ledger.verify_ledger(db) == []
    after = hashlib.sha256(db.read_bytes()).hexdigest()
    assert before == after


def test_verify_ledger_busy_database(tmp_path):
    db, _ = _valid_chain(tmp_path)
    _insert_ledger_rows(db, _state(db))
    writer = sqlite3.connect(db, timeout=0)
    try:
        writer.execute("BEGIN EXCLUSIVE")
        problems = ledger.verify_ledger(db)
        assert problems == ["cannot read ledger snapshot: database is busy or locked"]
    finally:
        writer.rollback()
        writer.close()


def test_control_wrappers_match_private_behavior(tmp_path):
    db, _ = _valid_chain(tmp_path)
    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT seq, event_id, prev_hash, event_hash, payload "
            "FROM audit_events ORDER BY seq"
        ).fetchall()
        head = conn.execute(
            "SELECT seq, event_hash FROM audit_head WHERE singleton = 1"
        ).fetchone()
    assert control.chain_problems(rows, head, allow_empty=False) == []
    event = control.event_from_payload(json.loads(rows[0][4]))
    assert isinstance(event, control.AuditEvent)
    assert event.event_id == rows[0][1]
    with pytest.raises(ValueError, match="payload fails the event schema"):
        control.event_from_payload({"event_id": str(uuid.uuid4())})
    parsed = json.loads(rows[0][4])
    parsed["unexpected_field"] = 1
    with pytest.raises(ValueError, match="seq 1: payload has unknown fields"):
        control.event_from_payload(parsed, "seq 1")


def test_checked_add_overflow_is_typed():
    with pytest.raises(ledger.LedgerOverflow, match="calls is outside the checked int64 domain"):
        ledger._add(2**63 - 1, 1, "calls")


def test_lease_release_keeps_expiry_and_marker(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    _start(store)
    store.append(
        _event(
            operation="run",
            event_kind="lease_released",
            decision="allow",
            reason_code="allowed",
            counters={},
            lease_owner=OWNER_1,
            lease_epoch=1,
            observed_now="2026-11-12T00:00:30+00:00",
            lease_ttl_seconds=300,
        )
    )
    store.close()
    lease = _state(db).runs["run-ledger"].lease
    assert lease.released is True
    assert lease.expires_at == "2026-11-12T00:05:00+00:00"
    assert lease.released_at == "2026-11-12T00:00:30+00:00"


def test_allowed_model_mismatch_is_rejected(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    _start(store)
    _decision(store, model="other-model")
    store.close()
    assert control.verify_chain(db) == []
    with pytest.raises(ledger.LedgerInvalidEvent, match="allowed llm decision model is not allowed"):
        _state(db)


def test_expected_counters_after_decision_and_outcome(tmp_path):
    """expected_counters runs the full rule path and raises ledger errors on rule failures."""
    db, decision = _valid_chain(tmp_path)
    events = _events(db)
    run_started = next(event for event in events if event.event_kind == "run_started")
    lease = next(event for event in events if event.event_kind == "lease_acquired")
    decision_event = next(event for event in events if event.event_id == decision.event_id)
    outcome_event = next(event for event in events if event.event_kind == "outcome")

    state_before = ledger.next_state(ledger.next_state(ledger.LedgerState(), run_started), lease)
    decision_expected = dict(ledger.expected_counters(state_before, decision_event))
    state_after_decision = ledger.next_state(state_before, decision_event)
    decision_state = ledger.next_state(state_before, decision_event)
    assert decision_expected == dict(
        ledger._counter_snapshot(decision_state.runs["run-ledger"])
    )
    assert decision_expected["calls"] == 1
    assert decision_expected["reserved_tokens"] == 637
    assert decision_expected["remaining_tokens"] == 20000 - 637

    outcome_expected = dict(ledger.expected_counters(state_after_decision, outcome_event))
    state_after_outcome = ledger.next_state(state_after_decision, outcome_event)
    outcome_snapshot = dict(
        ledger._counter_snapshot(state_after_outcome.runs["run-ledger"])
    )
    assert outcome_expected == dict(outcome_event.counters)
    assert outcome_expected == outcome_snapshot


def _start_with_policy(store, **overrides):
    policy = dict(_policy())
    policy.update(overrides)
    store.append(
        _event(
            operation="run",
            event_kind="run_started",
            decision="allow",
            reason_code="allowed",
            counters={},
            **policy,
        )
    )
    store.append(
        _event(
            operation="run",
            event_kind="lease_acquired",
            decision="allow",
            reason_code="allowed",
            counters={},
            lease_owner=OWNER_1,
            lease_epoch=1,
            observed_now="2026-11-12T00:00:00+00:00",
            lease_expires_at="2026-11-12T00:05:00+00:00",
            lease_ttl_seconds=300,
        )
    )


def test_token_cap_at_limit_and_one_over(tmp_path):
    db = tmp_path / "audit-ok.sqlite3"
    store = AuditStore(db)
    _start_with_policy(store, max_run_tokens=637)
    _decision(store, max_run_tokens=637)
    store.close()
    assert control.verify_chain(db) == []
    assert _state(db).runs["run-ledger"].reserved_tokens == 637

    db2 = tmp_path / "audit-over.sqlite3"
    store2 = AuditStore(db2)
    _start_with_policy(store2, max_run_tokens=636)
    _decision(store2, max_run_tokens=636)
    store2.close()
    assert control.verify_chain(db2) == []
    with pytest.raises(ledger.LedgerInvalidTransition, match="token cap would be exceeded"):
        _state(db2)


def test_token_cap_counts_prior_reservation_and_usage(tmp_path):
    db = tmp_path / "audit-two.sqlite3"
    store = AuditStore(db)
    _start_with_policy(store, max_run_tokens=1000)
    _decision(store, max_run_tokens=1000)
    _decision(store, max_run_tokens=1000, observed_now="2026-11-12T00:00:11+00:00")
    store.close()
    assert control.verify_chain(db) == []
    with pytest.raises(ledger.LedgerInvalidTransition, match="token cap would be exceeded"):
        _state(db)

    db2 = tmp_path / "audit-usage.sqlite3"
    store2 = AuditStore(db2)
    _start_with_policy(store2, max_run_tokens=1000)
    first = _decision(store2, max_run_tokens=1000)
    _outcome(
        store2,
        first,
        counters=_counters_for(1000, prompt=500, completion=100),
        usage_prompt_tokens=500,
        usage_completion_tokens=100,
    )
    _decision(store2, max_run_tokens=1000, observed_now="2026-11-12T00:00:30+00:00")
    store2.close()
    assert control.verify_chain(db2) == []
    with pytest.raises(ledger.LedgerInvalidTransition, match="token cap would be exceeded"):
        _state(db2)


def test_call_and_fetch_caps(tmp_path):
    db = tmp_path / "audit-call.sqlite3"
    store = AuditStore(db)
    _start_with_policy(store, max_run_llm_calls=1)
    _decision(store, max_run_llm_calls=1)
    _decision(store, max_run_llm_calls=1, observed_now="2026-11-12T00:00:11+00:00")
    store.close()
    assert control.verify_chain(db) == []
    with pytest.raises(ledger.LedgerInvalidTransition, match="llm call cap would be exceeded"):
        _state(db)

    db2 = tmp_path / "audit-fetch.sqlite3"
    store2 = AuditStore(db2)
    _start_with_policy(store2, max_fetches_per_run=1)
    for observed in ("2026-11-12T00:00:10+00:00", "2026-11-12T00:00:11+00:00"):
        store2.append(
            _event(
                operation="fetch",
                event_kind="decision",
                decision="allow",
                reason_code="allowed",
                counters={},
                target_host="example.com",
                lease_owner=OWNER_1,
                lease_epoch=1,
                observed_now=observed,
                **{**_policy(), "max_fetches_per_run": 1},
            )
        )
    store2.close()
    assert control.verify_chain(db2) == []
    with pytest.raises(ledger.LedgerInvalidTransition, match="fetch cap would be exceeded"):
        _state(db2)


def test_allowed_usage_over_held_and_released_mismatch(tmp_path):
    db = tmp_path / "audit-allowed.sqlite3"
    store = AuditStore(db)
    _start(store)
    decision = _decision(store)
    _outcome(
        store,
        decision,
        counters=_counters(prompt=600, completion=100),
        usage_prompt_tokens=600,
        usage_completion_tokens=100,
    )
    store.close()
    assert control.verify_chain(db) == []
    with pytest.raises(ledger.LedgerInvalidEvent, match="allowed usage exceeds the held reservation"):
        _state(db)

    db2 = tmp_path / "audit-release.sqlite3"
    store2 = AuditStore(db2)
    _start(store2)
    decision2 = _decision(store2)
    _outcome(store2, decision2, reservation_released=1)
    store2.close()
    assert control.verify_chain(db2) == []
    with pytest.raises(ledger.LedgerInvalidEvent, match="release does not equal the held reservation"):
        _state(db2)


def test_reconciliation_negative_balance_is_allowed(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    _start_with_policy(store, max_run_tokens=1000)
    decision = _decision(store, max_run_tokens=1000)
    unknown = _outcome(
        store,
        decision,
        reason_code="usage_missing",
        counters={},
        usage_prompt_tokens=None,
        usage_completion_tokens=None,
        reservation_released=0,
        usage_missing=True,
    )
    _outcome(
        store,
        decision,
        reason_code="reconciliation",
        counters={
            "calls": 1,
            "prompt_tokens": 1200,
            "completion_tokens": 0,
            "reserved_tokens": 0,
            "remaining_tokens": 0,
            "budget_deficit": 200,
        },
        usage_prompt_tokens=1200,
        usage_completion_tokens=0,
        references_event_id=unknown.event_id,
    )
    store.close()
    assert control.verify_chain(db) == []
    run = _state(db).runs["run-ledger"]
    assert run.prompt_tokens == 1200
    assert run.reserved_tokens == 0
    assert ledger._counter_snapshot(run)["budget_deficit"] == 200


def test_reserve_and_schema_mismatches(tmp_path):
    db = tmp_path / "audit-reserve.sqlite3"
    store = AuditStore(db)
    _start(store)
    _decision(store, reserve=638)
    store.close()
    assert control.verify_chain(db) == []
    with pytest.raises(ledger.LedgerInvalidEvent, match="decision reserve does not match the policy formula"):
        _state(db)

    db2 = tmp_path / "audit-schema.sqlite3"
    store2 = AuditStore(db2)
    _start(store2)
    _decision(store2, schema="Other")
    store2.close()
    assert control.verify_chain(db2) == []
    with pytest.raises(ledger.LedgerInvalidEvent, match="allowed llm decision schema is not allowed"):
        _state(db2)


def test_immutable_state_and_counter_expected_error(tmp_path):
    db, decision = _valid_chain(tmp_path)
    state = _state(db)
    with pytest.raises(TypeError):
        state.runs["other"] = None
    run = state.runs["run-ledger"]
    with pytest.raises(TypeError):
        run.reservations["x"] = None
    with pytest.raises(TypeError):
        run.decisions["x"] = None
    with pytest.raises(ledger.LedgerInvalidTransition, match="event has no counter snapshot"):
        ledger.expected_counters(
            state,
            next(event for event in _events(db) if event.event_kind == "lease_acquired"),
        )


def test_verify_ledger_exact_storage_problems(tmp_path):
    db, _ = _valid_chain(tmp_path)
    _insert_ledger_rows(db, _state(db))
    with sqlite3.connect(db) as conn:
        conn.execute("DROP TABLE run_ledger")
    problems = ledger.verify_ledger(db)
    assert problems == ["ledger table run_ledger is missing"]

    db2 = tmp_path / "audit-b.sqlite3"
    writer = sqlite3.connect(db2)
    writer.execute("CREATE TABLE audit_events (seq INTEGER, event_id TEXT, prev_hash TEXT, event_hash TEXT, payload TEXT)")
    writer.execute("CREATE TABLE audit_head (singleton INTEGER, seq INTEGER, event_hash TEXT)")
    writer.execute("BEGIN EXCLUSIVE")
    try:
        assert ledger.verify_ledger(db2) == ["cannot read ledger snapshot: database is busy or locked"]
    finally:
        writer.rollback()
        writer.close()


def test_compare_table_strict_types_and_columns():
    assert ledger.COLUMNS["run_policy"] == (
        "run_id", "thread_id", "policy_version", "estimate_kind", "margin_num",
        "margin_den", "provider_overhead_tokens", "max_run_tokens", "max_run_llm_calls",
        "max_output_tokens", "max_fetches_per_run", "lease_ttl_seconds", "allowed_model",
        "allowed_schemas_json",
    )
    row = ["r", "t", "v", "e", "1", "4", 0, 1, 1, 1, 1, 300, "m", '["s"]']
    assert ledger._compare_table("run_policy", [tuple(row)], [tuple(row)]) == []
    changed = list(row)
    changed[6] = 1.0
    problems = ledger._compare_table("run_policy", [tuple(row)], [tuple(changed)])
    assert problems == ["run_policy r: column provider_overhead_tokens differs from ledger replay"]


def test_released_no_effect_release_mismatch_is_rejected(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    _start(store)
    decision = _decision(store)
    _outcome(
        store,
        decision,
        reason_code="released_no_effect",
        counters={},
        usage_prompt_tokens=None,
        usage_completion_tokens=None,
        reservation_released=1,
    )
    store.close()
    assert control.verify_chain(db) == []
    with pytest.raises(
        ledger.LedgerInvalidEvent, match="release does not equal the held reservation"
    ):
        _state(db)


def test_llm_call_cap_at_max_and_one_over(tmp_path):
    db = tmp_path / "audit-cap.sqlite3"
    store = AuditStore(db)
    _start_with_policy(store, max_run_llm_calls=2)
    _decision(store, max_run_llm_calls=2)
    _decision(store, max_run_llm_calls=2, observed_now="2026-11-12T00:00:11+00:00")
    store.close()
    assert control.verify_chain(db) == []
    run = _state(db).runs["run-ledger"]
    assert run.calls == 2
    assert run.reserved_tokens == 1274

    db2 = tmp_path / "audit-cap-over.sqlite3"
    store2 = AuditStore(db2)
    _start_with_policy(store2, max_run_llm_calls=1)
    _decision(store2, max_run_llm_calls=1)
    _decision(store2, max_run_llm_calls=1, observed_now="2026-11-12T00:00:11+00:00")
    store2.close()
    assert control.verify_chain(db2) == []
    with pytest.raises(ledger.LedgerInvalidTransition, match="llm call cap would be exceeded"):
        _state(db2)


def test_fetch_cap_at_max_and_one_over(tmp_path):
    db = tmp_path / "audit-fetch.sqlite3"
    store = AuditStore(db)
    _start_with_policy(store, max_fetches_per_run=2)
    for observed in ("2026-11-12T00:00:10+00:00", "2026-11-12T00:00:11+00:00"):
        store.append(
            _event(
                operation="fetch",
                event_kind="decision",
                decision="allow",
                reason_code="allowed",
                counters={},
                target_host="example.com",
                lease_owner=OWNER_1,
                lease_epoch=1,
                observed_now=observed,
                **{**_policy(), "max_fetches_per_run": 2},
            )
        )
    store.close()
    assert control.verify_chain(db) == []
    assert _state(db).runs["run-ledger"].fetch_count == 2

    db2 = tmp_path / "audit-fetch-over.sqlite3"
    store2 = AuditStore(db2)
    _start_with_policy(store2, max_fetches_per_run=1)
    for observed in ("2026-11-12T00:00:10+00:00", "2026-11-12T00:00:11+00:00"):
        store2.append(
            _event(
                operation="fetch",
                event_kind="decision",
                decision="allow",
                reason_code="allowed",
                counters={},
                target_host="example.com",
                lease_owner=OWNER_1,
                lease_epoch=1,
                observed_now=observed,
                **{**_policy(), "max_fetches_per_run": 1},
            )
        )
    store2.close()
    assert control.verify_chain(db2) == []
    with pytest.raises(ledger.LedgerInvalidTransition, match="fetch cap would be exceeded"):
        _state(db2)


def test_allowed_usage_equal_to_held_is_accepted(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    _start(store)
    decision = _decision(store)
    _outcome(
        store,
        decision,
        counters=_counters(prompt=500, completion=137),
        usage_prompt_tokens=500,
        usage_completion_tokens=137,
    )
    store.close()
    assert control.verify_chain(db) == []
    run = _state(db).runs["run-ledger"]
    assert run.prompt_tokens == 500
    assert run.completion_tokens == 137
    assert run.reserved_tokens == 0
    assert run.reservations[decision.event_id].status == "applied"
    assert ledger._counter_snapshot(run) == _counters(prompt=500, completion=137)


@pytest.mark.parametrize(
    "case_id, extra",
    [
        ("calls", {"calls": 99}),
        ("prompt_tokens", {"prompt_tokens": 99, "remaining_tokens": 19861}),
        ("completion_tokens", {"completion_tokens": 41, "remaining_tokens": 19861}),
        ("reserved_tokens", {"reserved_tokens": 1, "remaining_tokens": 19861}),
    ],
)
def test_ledger_owned_counter_keys_reject_at_ledger(tmp_path, case_id, extra):
    db = tmp_path / f"audit-{case_id}.sqlite3"
    store = AuditStore(db)
    _start(store)
    decision = _decision(store)
    _outcome(store, decision, counters=dict(_counters(), **extra))
    store.close()
    assert control.verify_chain(db) == []
    with pytest.raises(
        ledger.LedgerInvalidEvent,
        match="outcome counters do not match derived ledger counters",
    ):
        _state(db)


@pytest.mark.parametrize(
    "case_id, key, value",
    [
        ("remaining_tokens", "remaining_tokens", 19000),
        ("budget_deficit", "budget_deficit", 5),
    ],
)
def test_chain_owned_counter_keys_reject_at_chain(tmp_path, case_id, key, value):
    db = tmp_path / f"audit-{case_id}.sqlite3"
    store = AuditStore(db)
    _start(store)
    decision = _decision(store)
    counters = dict(_counters())
    counters[key] = value
    outcome = _event(
        operation="llm",
        event_kind="outcome",
        decision="allow",
        reason_code="allowed",
        counters=counters,
        model="gpt-4o-mini",
        schema_name="NewsDigest",
        usage_prompt_tokens=98,
        usage_completion_tokens=40,
        reservation_released=637,
        references_event_id=decision.event_id,
        lease_owner=OWNER_1,
        lease_epoch=1,
        observed_now="2026-11-12T00:00:20+00:00",
    )
    with pytest.raises(ControlUnavailable, match="audit append failed before commit") as excinfo:
        store.append(outcome)
    assert excinfo.value.append_state == "known_rolled_back"
    assert isinstance(excinfo.value.__cause__, control.AuditIncomplete)
    assert f"{key} equation mismatch" in str(excinfo.value.__cause__)
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT count(*) FROM audit_events").fetchone()[0] == 3
    assert control.verify_chain(db) == []
    store.close()
    _insert_payload_direct(db, outcome.to_payload())
    problems = control.verify_chain(db)
    assert len(problems) == 1
    assert problems[0].endswith(f"{key} equation mismatch")
    assert ledger.verify_ledger(db) == problems


def test_outcome_schema_differs_from_decision(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    _start(store)
    decision = _decision(store)
    _outcome(store, decision, schema_name="OtherSchema")
    store.close()
    assert control.verify_chain(db) == []
    with pytest.raises(
        ledger.LedgerInvalidEvent, match="outcome model/schema differs from decision"
    ):
        _state(db)


def test_completion_only_reconciliation_is_incomplete(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    _start(store)
    decision = _decision(store)
    unknown = _outcome(
        store,
        decision,
        reason_code="usage_missing",
        counters={},
        usage_prompt_tokens=None,
        usage_completion_tokens=None,
        reservation_released=0,
        usage_missing=True,
    )
    store.append(
        _event(
            operation="llm",
            event_kind="outcome",
            decision="allow",
            reason_code="reconciliation",
            counters={},
            model="gpt-4o-mini",
            schema_name="NewsDigest",
            usage_completion_tokens=5,
            references_event_id=unknown.event_id,
            lease_owner=OWNER_1,
            lease_epoch=1,
            observed_now="2026-11-12T00:00:30+00:00",
        )
    )
    store.close()
    assert control.verify_chain(db) == []
    with pytest.raises(ledger.LedgerIncompleteReconciliation, match="reconciliation with one usage figure does not settle"):
        _state(db)


def test_fetch_outcome_target_normalization_is_accepted(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    _start(store)
    decision = _event(
        operation="fetch",
        event_kind="decision",
        decision="allow",
        reason_code="allowed",
        counters={},
        target_host="example.com",
        lease_owner=OWNER_1,
        lease_epoch=1,
        observed_now="2026-11-12T00:00:10+00:00",
        **_policy(),
    )
    store.append(decision)
    store.append(
        _event(
            operation="fetch",
            event_kind="outcome",
            decision="allow",
            reason_code="allowed",
            counters={},
            target_host="EXAMPLE.com",
            result_class="ok",
            references_event_id=decision.event_id,
            lease_owner=OWNER_1,
            lease_epoch=1,
            observed_now="2026-11-12T00:00:20+00:00",
        )
    )
    store.close()
    assert control.verify_chain(db) == []
    assert _state(db).runs["run-ledger"].fetch_count == 1


@pytest.mark.parametrize(
    "case_id, sql, expected",
    [
        ("run_policy_column", "UPDATE run_policy SET max_run_tokens = 19999",
         ["run_policy run-ledger: column max_run_tokens differs from ledger replay"]),
        ("run_ledger_counter", "UPDATE run_ledger SET reserved_tokens = 1",
         ["run_ledger run-ledger: column reserved_tokens differs from ledger replay"]),
        ("reservation_status", "UPDATE reservations SET status = 'open'",
         ["reservations {decision}: column status differs from ledger replay"]),
        ("reservation_closed_at_null", "UPDATE reservations SET closed_at = NULL",
         ["reservations {decision}: column closed_at differs from ledger replay"]),
        ("lease_released", "UPDATE run_leases SET released = 1",
         ["run_leases run-ledger: column released differs from ledger replay"]),
        ("head_seq", "UPDATE ledger_head SET seq = 3",
         ["ledger_head 1: column seq differs from ledger replay"]),
        ("head_hash", "UPDATE ledger_head SET event_hash = '0000000000000000000000000000000000000000000000000000000000000000'",
         ["ledger_head 1: column event_hash differs from ledger replay"]),
        ("reservation_extra", "INSERT INTO reservations VALUES ('run-ledger', 'extra-decision', 1, 'open', '2026-11-12T00:00:10+00:00', NULL, NULL)",
         ["reservations: unexpected row extra-decision"]),
        ("reservation_missing", "DELETE FROM reservations",
         ["reservations: missing row {decision}"]),
        ("run_ledger_missing", "DELETE FROM run_ledger",
         ["run_ledger: missing row run-ledger"]),
        ("run_ledger_extra", "INSERT INTO run_ledger VALUES ('other-run', 0, 0, 0, 0, 0)",
         ["run_ledger: unexpected row other-run"]),
        ("lease_missing", "DELETE FROM run_leases",
         ["run_leases: missing row run-ledger"]),
    ],
    ids=[
        "run_policy_column", "run_ledger_counter", "reservation_status",
        "reservation_closed_at_null", "lease_released", "head_seq", "head_hash",
        "reservation_extra", "reservation_missing", "run_ledger_missing",
        "run_ledger_extra", "lease_missing",
    ],
)
def test_verify_ledger_reports_exact_row_tampering(tmp_path, case_id, sql, expected):
    db, decision = _valid_chain(tmp_path)
    _insert_ledger_rows(db, _state(db))
    assert control.verify_chain(db) == []
    assert ledger.verify_ledger(db) == []
    with sqlite3.connect(db) as conn:
        audit_before = conn.execute(
            "SELECT seq, event_id, prev_hash, event_hash, payload "
            "FROM audit_events ORDER BY seq"
        ).fetchall()
        head_before = conn.execute(
            "SELECT seq, event_hash FROM audit_head"
        ).fetchone()
    with sqlite3.connect(db) as conn:
        conn.execute(sql.replace("{decision}", str(decision.event_id)))
    expected_problems = [p.replace("{decision}", str(decision.event_id)) for p in expected]
    with sqlite3.connect(db) as conn:
        audit_after = conn.execute(
            "SELECT seq, event_id, prev_hash, event_hash, payload "
            "FROM audit_events ORDER BY seq"
        ).fetchall()
        head_after = conn.execute(
            "SELECT seq, event_hash FROM audit_head"
        ).fetchone()
    assert audit_after == audit_before
    assert head_after == head_before
    assert control.verify_chain(db) == []
    assert ledger.verify_ledger(db) == expected_problems


def test_verify_ledger_reports_table_problems_in_table_order(tmp_path):
    db, decision = _valid_chain(tmp_path)
    _insert_ledger_rows(db, _state(db))
    assert control.verify_chain(db) == []
    assert ledger.verify_ledger(db) == []
    with sqlite3.connect(db) as conn:
        audit_before = conn.execute(
            "SELECT seq, event_id, prev_hash, event_hash, payload "
            "FROM audit_events ORDER BY seq"
        ).fetchall()
        head_before = conn.execute(
            "SELECT seq, event_hash FROM audit_head"
        ).fetchone()
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE ledger_head SET seq = 3")
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE run_ledger SET calls = 99")
    with sqlite3.connect(db) as conn:
        audit_after = conn.execute(
            "SELECT seq, event_id, prev_hash, event_hash, payload "
            "FROM audit_events ORDER BY seq"
        ).fetchall()
        head_after = conn.execute(
            "SELECT seq, event_hash FROM audit_head"
        ).fetchone()
    assert audit_after == audit_before
    assert head_after == head_before
    assert control.verify_chain(db) == []
    assert ledger.verify_ledger(db) == [
        "run_ledger run-ledger: column calls differs from ledger replay",
        "ledger_head 1: column seq differs from ledger replay",
    ]


def test_verify_ledger_reports_all_five_missing_tables(tmp_path):
    db, _ = _valid_chain(tmp_path)
    conn = sqlite3.connect(db)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name IN "
            "('run_policy','run_ledger','reservations','run_leases','ledger_head')"
        ).fetchall()
    finally:
        conn.close()
    assert rows == []
    assert control.verify_chain(db) == []
    assert ledger.verify_ledger(db) == [
        "ledger table run_policy is missing",
        "ledger table run_ledger is missing",
        "ledger table reservations is missing",
        "ledger table run_leases is missing",
        "ledger table ledger_head is missing",
    ]


@pytest.mark.parametrize(
    "table",
    ["run_policy", "run_ledger", "reservations", "run_leases", "ledger_head"],
    ids=["run_policy", "run_ledger", "reservations", "run_leases", "ledger_head"],
)
def test_ledger_columns_match_schema_pragma(table):
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(ledger.SCHEMA_SQL)
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    finally:
        conn.close()
    columns = tuple(row[1] for row in rows)
    assert columns
    assert columns == ledger.COLUMNS[table]
    assert ledger.TABLE_KEYS[table] in ledger.COLUMNS[table]
    assert set(ledger.COLUMNS) == set(ledger.TABLE_KEYS) == {
        "run_policy",
        "run_ledger",
        "reservations",
        "run_leases",
        "ledger_head",
    }


def test_verify_ledger_leaves_no_journal_or_wal_file(tmp_path):
    db, _ = _valid_chain(tmp_path)
    _insert_ledger_rows(db, _state(db))
    before = sorted(path.name for path in tmp_path.iterdir())
    assert control.verify_chain(db) == []
    assert ledger.verify_ledger(db) == []
    after = sorted(path.name for path in tmp_path.iterdir())
    assert after == before
    assert not any(name.endswith(("-journal", "-wal", "-shm")) for name in before + after)


def _two_run_chain(tmp_path):
    db = tmp_path / "audit-two-runs.sqlite3"
    store = AuditStore(db)
    _start(store)
    store.append(
        _event(
            run_id="run-b",
            thread_id="thread-b",
            operation="run",
            event_kind="run_started",
            decision="allow",
            reason_code="allowed",
            counters={},
            **_policy(),
        )
    )
    store.append(
        _event(
            run_id="run-b",
            thread_id="thread-b",
            operation="run",
            event_kind="lease_acquired",
            decision="allow",
            reason_code="allowed",
            counters={},
            lease_owner=OWNER_1,
            lease_epoch=1,
            observed_now="2026-11-12T00:00:00+00:00",
            lease_expires_at="2026-11-12T00:05:00+00:00",
            lease_ttl_seconds=300,
        )
    )
    decision_a = _decision(store)
    decision_b = _decision(store, run_id="run-b", thread_id="thread-b")
    _outcome(store, decision_a)
    _outcome(store, decision_b, run_id="run-b", thread_id="thread-b")
    store.close()
    assert control.verify_chain(db) == []
    return db, decision_a, decision_b


def test_verify_ledger_two_interleaved_runs_match_replay(tmp_path):
    db, decision_a, decision_b = _two_run_chain(tmp_path)
    _insert_ledger_rows(db, _state(db))
    assert [event.run_id for event in _events(db)] == [
        "run-ledger", "run-ledger", "run-b", "run-b",
        "run-ledger", "run-b", "run-ledger", "run-b",
    ]
    assert sorted(_state(db).runs) == ["run-b", "run-ledger"]
    assert control.verify_chain(db) == []
    assert ledger.verify_ledger(db) == []
    conn = sqlite3.connect(db)
    try:
        policy = conn.execute("SELECT run_id FROM run_policy ORDER BY run_id").fetchall()
        run_ledger = conn.execute("SELECT run_id FROM run_ledger ORDER BY run_id").fetchall()
        leases = conn.execute("SELECT run_id FROM run_leases ORDER BY run_id").fetchall()
        reservations = conn.execute(
            "SELECT decision_event_id, run_id FROM reservations ORDER BY decision_event_id"
        ).fetchall()
        head = conn.execute("SELECT seq FROM ledger_head").fetchall()
    finally:
        conn.close()
    assert policy == [("run-b",), ("run-ledger",)]
    assert run_ledger == [("run-b",), ("run-ledger",)]
    assert leases == [("run-b",), ("run-ledger",)]
    assert reservations == sorted([
        (decision_a.event_id, "run-ledger"),
        (decision_b.event_id, "run-b"),
    ])
    assert head == [(8,)]


@pytest.mark.parametrize(
    "case_id, statements, expected",
    [
        ("run_b_counter", ["UPDATE run_ledger SET calls = 99 WHERE run_id = 'run-b'"],
         ["run_ledger run-b: column calls differs from ledger replay"]),
        ("run_a_policy", ["UPDATE run_policy SET max_run_tokens = 19999 WHERE run_id = 'run-ledger'"],
         ["run_policy run-ledger: column max_run_tokens differs from ledger replay"]),
        ("run_b_lease", ["UPDATE run_leases SET released = 1 WHERE run_id = 'run-b'"],
         ["run_leases run-b: column released differs from ledger replay"]),
        ("run_b_reservation", ["UPDATE reservations SET status = 'open' WHERE run_id = 'run-b'"],
         ["reservations {decision_b}: column status differs from ledger replay"]),
        ("run_b_policy_missing", ["DELETE FROM run_policy WHERE run_id = 'run-b'"],
         ["run_policy: missing row run-b"]),
        ("extra_run_policy_only", [
            "INSERT INTO run_policy SELECT 'other-run', thread_id, policy_version, "
            "estimate_kind, margin_num, margin_den, provider_overhead_tokens, "
            "max_run_tokens, max_run_llm_calls, max_output_tokens, max_fetches_per_run, "
            "lease_ttl_seconds, allowed_model, allowed_schemas_json FROM run_policy "
            "WHERE run_id = 'run-ledger'"
        ], ["run_policy: unexpected row other-run"]),
        ("extra_run_policy_and_ledger", [
            "INSERT INTO run_policy SELECT 'other-run', thread_id, policy_version, "
            "estimate_kind, margin_num, margin_den, provider_overhead_tokens, "
            "max_run_tokens, max_run_llm_calls, max_output_tokens, max_fetches_per_run, "
            "lease_ttl_seconds, allowed_model, allowed_schemas_json FROM run_policy "
            "WHERE run_id = 'run-ledger'",
            "INSERT INTO run_ledger VALUES ('other-run', 0, 0, 0, 0, 0)",
        ], ["run_policy: unexpected row other-run", "run_ledger: unexpected row other-run"]),
        ("both_runs_counter_order", [
            "UPDATE run_ledger SET calls = 99 WHERE run_id = 'run-b'",
            "UPDATE run_ledger SET calls = 98 WHERE run_id = 'run-ledger'",
        ], [
            "run_ledger run-b: column calls differs from ledger replay",
            "run_ledger run-ledger: column calls differs from ledger replay",
        ]),
    ],
    ids=[
        "run_b_counter", "run_a_policy", "run_b_lease", "run_b_reservation",
        "run_b_policy_missing", "extra_run_policy_only", "extra_run_policy_and_ledger",
        "both_runs_counter_order",
    ],
)
def test_verify_ledger_reports_per_run_table_tampering(tmp_path, case_id, statements, expected):
    db, decision_a, decision_b = _two_run_chain(tmp_path)
    _insert_ledger_rows(db, _state(db))
    assert control.verify_chain(db) == []
    assert ledger.verify_ledger(db) == []
    conn = sqlite3.connect(db)
    try:
        audit_before = conn.execute(
            "SELECT seq, event_id, prev_hash, event_hash, payload "
            "FROM audit_events ORDER BY seq"
        ).fetchall()
        head_before = conn.execute(
            "SELECT seq, event_hash FROM audit_head"
        ).fetchone()
    finally:
        conn.close()
    conn = sqlite3.connect(db)
    try:
        for statement in statements:
            conn.execute(statement.replace("{decision_b}", str(decision_b.event_id)))
        conn.commit()
    finally:
        conn.close()
    conn = sqlite3.connect(db)
    try:
        audit_after = conn.execute(
            "SELECT seq, event_id, prev_hash, event_hash, payload "
            "FROM audit_events ORDER BY seq"
        ).fetchall()
        head_after = conn.execute(
            "SELECT seq, event_hash FROM audit_head"
        ).fetchone()
    finally:
        conn.close()
    assert audit_after == audit_before
    assert head_after == head_before
    assert control.verify_chain(db) == []
    expected_problems = [p.replace("{decision_b}", str(decision_b.event_id)) for p in expected]
    assert ledger.verify_ledger(db) == expected_problems


@pytest.mark.parametrize(
    "case_id, kind, outcome_kwargs, expected",
    [
        (
            "overrun",
            "outcome",
            {
                "reason_code": "reservation_overrun",
                "counters": _counters(prompt=120, completion=600),
                "usage_prompt_tokens": 120,
                "usage_completion_tokens": 600,
                "overrun_tokens": 83,
            },
            {
                "calls": 1,
                "prompt_tokens": 120,
                "completion_tokens": 600,
                "reserved_tokens": 0,
                "remaining_tokens": 19280,
                "budget_deficit": 0,
            },
        ),
        (
            "breach",
            "outcome",
            {
                "reason_code": "budget_breach",
                "counters": _counters(prompt=12000, completion=9000),
                "usage_prompt_tokens": 12000,
                "usage_completion_tokens": 9000,
                "overrun_tokens": 20363,
            },
            {
                "calls": 1,
                "prompt_tokens": 12000,
                "completion_tokens": 9000,
                "reserved_tokens": 0,
                "remaining_tokens": 0,
                "budget_deficit": 1000,
            },
        ),
        (
            "reconciliation",
            "reconciliation",
            {"reason_code": "reconciliation", "counters": _counters()},
            {
                "calls": 1,
                "prompt_tokens": 98,
                "completion_tokens": 40,
                "reserved_tokens": 0,
                "remaining_tokens": 19862,
                "budget_deficit": 0,
            },
        ),
    ],
    ids=["overrun", "breach", "reconciliation"],
)
def test_expected_counters_for_usage_outcomes(tmp_path, case_id, kind, outcome_kwargs, expected):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    _start(store)
    decision = _decision(store)
    if kind == "reconciliation":
        unknown = _outcome(
            store,
            decision,
            reason_code="usage_missing",
            counters={},
            usage_prompt_tokens=None,
            usage_completion_tokens=None,
            reservation_released=0,
            usage_missing=True,
        )
        outcome_kwargs = dict(outcome_kwargs, references_event_id=unknown.event_id)
    _outcome(store, decision, **outcome_kwargs)
    store.close()
    assert control.verify_chain(db) == []
    events = _events(db)
    outcome = events[-1]
    assert outcome.event_kind == "outcome"
    assert outcome.reason_code == outcome_kwargs["reason_code"]
    before = ledger.replay_events(events[:-1])
    assert dict(ledger.expected_counters(before, outcome)) == expected
    assert dict(outcome.counters) == expected
    assert ledger._counter_snapshot(ledger.next_state(before, outcome).runs["run-ledger"]) == expected


def test_expected_counters_ignores_event_counters_while_next_state_rejects(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    _start(store)
    decision = _decision(store)
    _outcome(store, decision, counters=dict(_counters(), calls=99))
    store.close()
    assert control.verify_chain(db) == []
    events = _events(db)
    outcome = events[-1]
    before = ledger.replay_events(events[:-1])
    assert dict(outcome.counters)["calls"] == 99
    assert dict(ledger.expected_counters(before, outcome)) == _counters()
    with pytest.raises(
        ledger.LedgerInvalidEvent,
        match="outcome counters do not match derived ledger counters",
    ):
        ledger.next_state(before, outcome)


@pytest.mark.parametrize(
    "case_id, func_name, args, exc_name, match, expected",
    [
        ("add_at_max", "_add", (2**63 - 2, 1, "calls"), None, None, 2**63 - 1),
        ("add_zero", "_add", (0, 0, "calls"), None, None, 0),
        ("checked_negative", "_checked", (-1, "calls"), "LedgerOverflow",
         "calls is outside the checked int64 domain", None),
        ("checked_bool", "_checked", (True, "calls"), "LedgerOverflow",
         "calls is outside the checked int64 domain", None),
        ("checked_above_max", "_checked", (2**63, "calls"), "LedgerOverflow",
         "calls is outside the checked int64 domain", None),
        ("sub_to_zero", "_sub", (5, 5, "reserved_tokens"), None, None, 0),
        ("sub_negative", "_sub", (5, 6, "reserved_tokens"), "LedgerInvalidTransition",
         "reserved_tokens would become negative", None),
    ],
    ids=[
        "add_at_max", "add_zero", "checked_negative", "checked_bool",
        "checked_above_max", "sub_to_zero", "sub_negative",
    ],
)
def test_checked_arithmetic_boundaries(case_id, func_name, args, exc_name, match, expected):
    if exc_name is None:
        assert getattr(ledger, func_name)(*args) == expected
    else:
        with pytest.raises(getattr(ledger, exc_name), match=match):
            getattr(ledger, func_name)(*args)


def test_reserve_formula_value_and_overflow():
    policy = ledger.PolicySnapshot(
        margin_num=1,
        margin_den=4,
        provider_overhead_tokens=0,
        max_run_tokens=20000,
        max_run_llm_calls=24,
        max_output_tokens=512,
        max_fetches_per_run=8,
        lease_ttl_seconds=300,
        allowed_model="gpt-4o-mini",
        allowed_schemas=("NewsDigest",),
        estimate_kind="policy_margin_v1",
        policy_version=control.POLICY_VERSION_H1_SLICE2,
    )
    assert ledger._reserve(policy, 100) == 637
    with pytest.raises(
        ledger.LedgerOverflow, match="reserve_tokens is outside the checked int64 domain"
    ):
        ledger._reserve(policy, 2**63 - 1)


@pytest.mark.parametrize("case_id", ["applied_outcome", "rejected_duplicate_run_started"])
def test_next_state_leaves_input_state_unchanged(tmp_path, case_id):
    db, decision = _valid_chain(tmp_path)
    events = _events(db)
    before = ledger.replay_events(events[:-1])
    head = (len(events) - 1, "a" * 64)
    runs_before = dict(before.runs)
    run_before = before.runs["run-ledger"]
    rows_before = ledger.state_to_rows(before, head)
    if case_id == "applied_outcome":
        after = ledger.next_state(before, events[-1])
        assert after is not before
        assert ledger.state_to_rows(after, head) != rows_before
    else:
        assert events[0].event_kind == "run_started"
        with pytest.raises(
            ledger.LedgerInvalidTransition, match="duplicate run_started for ledger run"
        ):
            ledger.next_state(before, events[0])
    assert dict(before.runs) == runs_before
    assert before.runs["run-ledger"] is run_before
    assert ledger.state_to_rows(before, head) == rows_before


@pytest.mark.parametrize(
    "head, allow_empty, expected",
    [
        (None, False, ["audit chain is empty"]),
        (None, True, []),
        ((1, "a" * 64), True, ["recorded head exists without events"]),
        ((1, "a" * 64), False, ["recorded head exists without events"]),
    ],
    ids=[
        "empty_no_head_strict", "empty_no_head_allowed",
        "empty_head_allowed", "empty_head_strict",
    ],
)
def test_chain_problems_empty_chain_cases(head, allow_empty, expected):
    assert control.chain_problems([], head, allow_empty=allow_empty) == expected
    assert control._chain_problems([], head, allow_empty=allow_empty) == expected


def test_chain_problems_multi_problem_list_and_wrapper_parity(tmp_path):
    db, _ = _valid_chain(tmp_path)
    conn = sqlite3.connect(db)
    try:
        rows = conn.execute(
            "SELECT seq, event_id, prev_hash, event_hash, payload "
            "FROM audit_events ORDER BY seq"
        ).fetchall()
        head = conn.execute(
            "SELECT seq, event_hash FROM audit_head WHERE singleton = 1"
        ).fetchone()
    finally:
        conn.close()
    assert len(rows) == 4 and head == (4, rows[3][3])
    assert control.chain_problems(rows, head, allow_empty=False) == []
    mutated = [list(row) for row in rows]
    mutated[1][3] = "0" * 64
    mutated[3][0] = 5
    mutated = [tuple(row) for row in mutated]
    expected = [
        "seq 2: event_hash does not match payload",
        "seq 3: broken prev_hash linkage",
        "seq 5: sequence gap",
        "recorded head does not match the last event",
    ]
    assert control.chain_problems(mutated, head, allow_empty=False) == expected
    assert control._chain_problems(mutated, head, allow_empty=False) == expected


def test_event_from_payload_failure_uses_the_given_label():
    with pytest.raises(ValueError) as caught:
        control.event_from_payload({"event_id": str(uuid.uuid4())}, "seq 7")
    assert str(caught.value) == "seq 7: payload fails the event schema"


def test_valid_chain_with_renewal_and_takeover_matches_ledger_replay(tmp_path):
    owner_2 = "00000000-0000-0000-0000-000000000002"
    db = tmp_path / "audit-leases.sqlite3"
    store = AuditStore(db)
    _start(store)
    renewed = _event(
        operation="run",
        event_kind="lease_renewed",
        decision="allow",
        reason_code="allowed",
        counters={},
        lease_owner=OWNER_1,
        lease_epoch=1,
        observed_now="2026-11-12T00:01:00+00:00",
        lease_expires_at="2026-11-12T00:06:00+00:00",
        lease_ttl_seconds=300,
    )
    store.append(renewed)
    taken = _event(
        operation="run",
        event_kind="lease_taken_over",
        decision="allow",
        reason_code="allowed",
        counters={},
        lease_owner=owner_2,
        lease_epoch=2,
        previous_lease_owner=OWNER_1,
        previous_lease_epoch=1,
        references_event_id=renewed.event_id,
        observed_now="2026-11-12T00:07:00+00:00",
        lease_expires_at="2026-11-12T00:12:00+00:00",
        lease_ttl_seconds=300,
    )
    store.append(taken)
    decision = _decision(
        store,
        lease_owner=owner_2,
        lease_epoch=2,
        observed_now="2026-11-12T00:07:10+00:00",
    )
    _outcome(
        store,
        decision,
        lease_owner=owner_2,
        lease_epoch=2,
        observed_now="2026-11-12T00:07:20+00:00",
    )
    released = _event(
        operation="run",
        event_kind="lease_released",
        decision="allow",
        reason_code="allowed",
        counters={},
        lease_owner=owner_2,
        lease_epoch=2,
        observed_now="2026-11-12T00:08:00+00:00",
        lease_ttl_seconds=300,
    )
    store.append(released)
    store.close()
    assert control.verify_chain(db) == []
    assert [event.event_kind for event in _events(db)] == [
        "run_started", "lease_acquired", "lease_renewed", "lease_taken_over",
        "decision", "outcome", "lease_released",
    ]
    state = _state(db)
    lease = state.runs["run-ledger"].lease
    assert lease.owner_id == owner_2
    assert lease.epoch == 2
    assert lease.expires_at == "2026-11-12T00:12:00+00:00"
    assert lease.released is True
    assert lease.released_at == "2026-11-12T00:08:00+00:00"
    assert lease.last_event_id == released.event_id
    _insert_ledger_rows(db, state)
    assert ledger.verify_ledger(db) == []


def test_verify_ledger_unreadable_table_is_not_reported_as_missing(tmp_path):
    db, _ = _valid_chain(tmp_path)
    _insert_ledger_rows(db, _state(db))
    assert ledger.verify_ledger(db) == []
    conn = sqlite3.connect(db)
    try:
        conn.execute("ALTER TABLE run_ledger RENAME TO run_ledger_saved")
        conn.execute("CREATE VIEW run_ledger AS SELECT 'x' AS run_id")
        conn.commit()
    finally:
        conn.close()
    assert ledger.verify_ledger(db) == [
        "cannot read ledger table run_ledger: OperationalError"
    ]


def test_verify_ledger_connects_read_only_with_zero_timeout(tmp_path, monkeypatch):
    db, _ = _valid_chain(tmp_path)
    _insert_ledger_rows(db, _state(db))
    real_connect = sqlite3.connect
    calls = []

    def recording_connect(*args, **kwargs):
        calls.append((args, kwargs))
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", recording_connect)
    assert ledger.verify_ledger(db) == []
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert len(args) == 1
    assert args[0].startswith("file:") and args[0].endswith("?mode=ro")
    assert kwargs == {"uri": True, "timeout": 0.0}
    assert type(kwargs["timeout"]) is float


@pytest.mark.parametrize(
    "fail_on, error, expected",
    [
        (None, None, []),
        (
            "connect",
            sqlite3.OperationalError("unable to open database file"),
            ["cannot open ledger database read-only: OperationalError"],
        ),
        (
            "FROM audit_events",
            sqlite3.OperationalError("disk I/O error"),
            ["cannot read ledger snapshot: OperationalError"],
        ),
        (
            "FROM run_ledger",
            sqlite3.OperationalError("database is locked"),
            ["cannot read ledger snapshot: database is busy or locked"],
        ),
        (
            "FROM run_ledger",
            sqlite3.DatabaseError("database is busy"),
            ["cannot read ledger snapshot: database is busy or locked"],
        ),
        (
            "FROM run_ledger",
            sqlite3.OperationalError("disk I/O error"),
            ["cannot read ledger table run_ledger: OperationalError"],
        ),
        (
            "FROM run_ledger",
            sqlite3.DatabaseError("database disk image is malformed"),
            ["cannot read ledger table run_ledger: DatabaseError"],
        ),
    ],
    ids=[
        "passthrough_control", "open_error", "audit_read_error",
        "table_busy_operational", "table_busy_database_error",
        "table_unreadable_operational", "table_unreadable_database_error",
    ],
)
def test_verify_ledger_sqlite_errors_per_read_phase(
    tmp_path, monkeypatch, fail_on, error, expected
):
    db, _ = _valid_chain(tmp_path)
    _insert_ledger_rows(db, _state(db))
    assert control.verify_chain(db) == []
    real_connect = sqlite3.connect

    class _FailingConnection(sqlite3.Connection):
        def execute(self, sql, *args):
            if fail_on not in (None, "connect") and fail_on in sql:
                raise error
            return super().execute(sql, *args)

    def failing_connect(*args, **kwargs):
        if fail_on == "connect":
            raise error
        return real_connect(*args, factory=_FailingConnection, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", failing_connect)
    assert ledger.verify_ledger(db) == expected
    probe = real_connect(db, timeout=0)
    try:
        probe.isolation_level = None
        probe.execute("BEGIN EXCLUSIVE")
        probe.execute("ROLLBACK")
    finally:
        probe.close()


@pytest.mark.parametrize(
    "marker",
    [
        "FROM run_policy",
        "FROM run_ledger",
        "FROM reservations",
        "FROM run_leases",
        "FROM ledger_head",
    ],
    ids=["run_policy", "run_ledger", "reservations", "run_leases", "ledger_head"],
)
def test_verify_ledger_snapshot_blocks_interleaved_writer(tmp_path, monkeypatch, marker):
    db, _ = _valid_chain(tmp_path)
    _insert_ledger_rows(db, _state(db))
    assert control.verify_chain(db) == []
    assert ledger.verify_ledger(db) == []
    real_connect = sqlite3.connect

    def read_calls():
        probe = real_connect(db, timeout=0)
        try:
            assert probe.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
            return probe.execute("SELECT calls FROM run_ledger").fetchall()
        finally:
            probe.close()

    before = read_calls()
    assert before
    fired = []
    attempts = []

    class _InterleavingConnection(sqlite3.Connection):
        def execute(self, sql, *args):
            if marker in sql and not fired:
                fired.append(sql)
                stage = "begin"
                writer = real_connect(db, timeout=0, isolation_level=None)
                try:
                    try:
                        writer.execute("BEGIN IMMEDIATE")
                        stage = "update"
                        writer.execute("UPDATE run_ledger SET calls = calls + 1")
                        stage = "commit"
                        writer.execute("COMMIT")
                        stage = "committed"
                        attempts.append((stage, ""))
                    except sqlite3.OperationalError as exc:
                        attempts.append((stage, str(exc)))
                        writer.execute("ROLLBACK")
                finally:
                    writer.close()
            return super().execute(sql, *args)

    def interleaving_connect(*args, **kwargs):
        return real_connect(*args, factory=_InterleavingConnection, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", interleaving_connect)
    assert ledger.verify_ledger(db) == []
    assert len(fired) == 1
    assert len(attempts) == 1
    assert attempts[0][0] == "commit"
    assert "locked" in attempts[0][1]
    assert read_calls() == before


def test_interleaved_writer_commits_when_no_read_transaction_is_held(tmp_path):
    db, _ = _valid_chain(tmp_path)
    _insert_ledger_rows(db, _state(db))
    assert ledger.verify_ledger(db) == []
    reader = sqlite3.connect(db, timeout=0)
    try:
        assert reader.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        before = reader.execute("SELECT calls FROM run_ledger").fetchall()
        assert before
        reader.execute("SELECT seq FROM audit_events ORDER BY seq").fetchall()
        writer = sqlite3.connect(db, timeout=0, isolation_level=None)
        try:
            writer.execute("BEGIN IMMEDIATE")
            writer.execute("UPDATE run_ledger SET calls = calls + 1")
            writer.execute("COMMIT")
        finally:
            writer.close()
        after = reader.execute("SELECT calls FROM run_ledger").fetchall()
    finally:
        reader.close()
    assert after == [(before[0][0] + 1,)]
    assert ledger.verify_ledger(db) == [
        "run_ledger run-ledger: column calls differs from ledger replay"
    ]


@pytest.mark.parametrize(
    "margin, overhead, expected_at_100",
    [
        (0.1, 17, 641),
        (0.75, 0, 687),
        (0.75, 4096, 7855),
        (1.0, 17, 746),
        (0.0, 5, 617),
    ],
    ids=[
        "margin_0_1_overhead_17", "margin_0_75_no_overhead",
        "margin_0_75_overhead_4096", "margin_1_0_overhead_17",
        "margin_0_0_overhead_5",
    ],
)
def test_reserve_matches_admission_with_overhead_and_float_margins(
    margin, overhead, expected_at_100
):
    policy = admission.PolicyConfig(
        max_run_tokens=20000,
        max_run_llm_calls=24,
        max_output_tokens=512,
        max_fetches_per_run=8,
        reservation_margin=margin,
        provider_overhead_tokens=overhead,
        allowed_model="gpt-4o-mini",
        allowed_schemas=frozenset({"NewsDigest"}),
    )
    numerator, denominator = margin.as_integer_ratio()
    snapshot = ledger.PolicySnapshot(
        margin_num=numerator,
        margin_den=denominator,
        provider_overhead_tokens=overhead,
        max_run_tokens=20000,
        max_run_llm_calls=24,
        max_output_tokens=512,
        max_fetches_per_run=8,
        lease_ttl_seconds=300,
        allowed_model="gpt-4o-mini",
        allowed_schemas=("NewsDigest",),
        estimate_kind="policy_margin_v1",
        policy_version=control.POLICY_VERSION_H1_SLICE2,
    )
    for count in (0, 3, 100, 2**53 + 1):
        assert ledger._reserve(snapshot, count) == admission._reserve(policy, count)
    assert ledger._reserve(snapshot, 100) == expected_at_100
    assert admission._reserve(policy, 100) == expected_at_100


@pytest.mark.parametrize("operation", ["fetch", "write"], ids=["fetch", "write"])
def test_denied_fetch_and_write_outcomes_are_rejected(tmp_path, operation):
    if operation == "fetch":
        target = {"target_host": "example.com"}
        outcome_fields = {"target_host": "example.com", "result_class": "ok"}
    else:
        target = {"target_id": "target-1"}
        outcome_fields = {"write_outcome": "failed"}
    for label, verdict, reason in (
        ("allowed", "allow", "allowed"),
        ("denied", "deny", "policy_denied"),
    ):
        db = tmp_path / f"audit-{operation}-{label}.sqlite3"
        store = AuditStore(db)
        _start(store)
        store.append(
            _event(
                operation="run",
                event_kind="run_approved",
                decision="allow",
                reason_code="allowed",
                counters={},
                lease_owner=OWNER_1,
                lease_epoch=1,
                observed_now="2026-11-12T00:00:05+00:00",
                approved_target_id="target-1",
            )
        )
        decision = _event(
            operation=operation,
            event_kind="decision",
            decision=verdict,
            reason_code=reason,
            counters={},
            lease_owner=OWNER_1,
            lease_epoch=1,
            observed_now="2026-11-12T00:00:10+00:00",
            **target,
            **_policy(),
        )
        store.append(decision)
        store.append(
            _event(
                operation=operation,
                event_kind="outcome",
                decision="allow",
                reason_code="allowed",
                counters={},
                references_event_id=decision.event_id,
                lease_owner=OWNER_1,
                lease_epoch=1,
                observed_now="2026-11-12T00:00:20+00:00",
                **outcome_fields,
            )
        )
        store.close()
        assert control.verify_chain(db) == []
        if verdict == "allow":
            run = _state(db).runs["run-ledger"]
            assert decision.event_id in run.normal_outcomes
        else:
            with pytest.raises(
                ledger.LedgerInvalidTransition,
                match="outcome references a denied decision",
            ):
                _state(db)


def test_allowed_llm_decision_without_count_passes_chain_and_is_rejected_by_ledger(tmp_path):
    db = tmp_path / "audit-no-count.sqlite3"
    store = AuditStore(db)
    _start(store)
    _decision(
        store,
        reason_code="tokenizer_unavailable",
        serialized_body_token_count=None,
    )
    store.close()
    assert control.verify_chain(db) == []
    with pytest.raises(
        ledger.LedgerInvalidTransition,
        match="allowed llm decision lacks a serialized count",
    ):
        _state(db)
    assert ledger.verify_ledger(db) == [
        "seq 3: allowed llm decision lacks a serialized count"
    ]


def test_released_no_effect_returns_reservation_to_zero(tmp_path):
    db = tmp_path / "audit-released.sqlite3"
    store = AuditStore(db)
    _start(store)
    decision = _decision(store)
    outcome = _outcome(
        store,
        decision,
        reason_code="released_no_effect",
        counters={},
        usage_prompt_tokens=None,
        usage_completion_tokens=None,
        reservation_released=637,
    )
    store.close()
    assert control.verify_chain(db) == []
    state = _state(db)
    run = state.runs["run-ledger"]
    assert run.calls == 1
    assert run.prompt_tokens == 0
    assert run.completion_tokens == 0
    assert run.reserved_tokens == 0
    assert run.fetch_count == 0
    assert run.normal_outcomes == frozenset({decision.event_id})
    reservation = run.reservations[decision.event_id]
    assert reservation.reserved_tokens == 637
    assert reservation.status == "released"
    assert reservation.opened_at == "2026-11-12T00:00:10+00:00"
    assert reservation.closed_at == "2026-11-12T00:00:20+00:00"
    assert reservation.last_event_id == outcome.event_id
    assert ledger._counter_snapshot(run) == {
        "calls": 1,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reserved_tokens": 0,
        "remaining_tokens": 20000,
        "budget_deficit": 0,
    }
    rows = ledger.state_to_rows(state, (4, "a" * 64))
    assert rows["run_ledger"] == [("run-ledger", 1, 0, 0, 0, 0)]


def test_ledger_rows_match_independent_literals_for_applied_chain(tmp_path):
    db = tmp_path / "audit-literal-rows.sqlite3"
    store = AuditStore(db)
    acquired = _start(store)
    decision = _decision(store)
    outcome = _outcome(store, decision)
    store.close()
    assert control.verify_chain(db) == []
    conn = sqlite3.connect(db)
    try:
        head = conn.execute(
            "SELECT seq, event_hash FROM audit_head WHERE singleton = 1"
        ).fetchone()
    finally:
        conn.close()
    assert head[0] == 4
    columns = {
        "run_policy": (
            "run_id", "thread_id", "policy_version", "estimate_kind", "margin_num",
            "margin_den", "provider_overhead_tokens", "max_run_tokens",
            "max_run_llm_calls", "max_output_tokens", "max_fetches_per_run",
            "lease_ttl_seconds", "allowed_model", "allowed_schemas_json",
        ),
        "run_ledger": (
            "run_id", "calls", "prompt_tokens", "completion_tokens",
            "reserved_tokens", "fetch_count",
        ),
        "reservations": (
            "run_id", "decision_event_id", "reserved_tokens", "status",
            "opened_at", "closed_at", "last_event_id",
        ),
        "run_leases": (
            "run_id", "owner_id", "epoch", "expires_at", "released",
            "released_at", "last_event_id",
        ),
        "ledger_head": ("singleton", "seq", "event_hash"),
    }
    literal = {
        "run_policy": [(
            "run-ledger", "thread-ledger", control.POLICY_VERSION_H1_SLICE2,
            "policy_margin_v1", "1", "4", 0, 20000, 24, 512, 8, 300,
            "gpt-4o-mini", '["NewsDigest"]',
        )],
        "run_ledger": [("run-ledger", 1, 98, 40, 0, 0)],
        "reservations": [(
            "run-ledger", decision.event_id, 637, "applied",
            "2026-11-12T00:00:10+00:00", "2026-11-12T00:00:20+00:00",
            outcome.event_id,
        )],
        "run_leases": [(
            "run-ledger", OWNER_1, 1, "2026-11-12T00:05:00+00:00", 0, None,
            acquired.event_id,
        )],
        "ledger_head": [(1, 4, head[1])],
    }
    assert ledger.state_to_rows(_state(db), head) == literal
    conn = sqlite3.connect(db)
    try:
        conn.executescript(ledger.SCHEMA_SQL)
        for table, rows in literal.items():
            names = ", ".join(columns[table])
            marks = ", ".join("?" for _ in columns[table])
            for row in rows:
                conn.execute(
                    f"INSERT INTO {table} ({names}) VALUES ({marks})", row
                )
        conn.commit()
    finally:
        conn.close()
    assert control.verify_chain(db) == []
    assert ledger.verify_ledger(db) == []
