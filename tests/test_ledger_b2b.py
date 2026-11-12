"""Offline tests for the real ledger participant inside the audit transaction (2b-ii-B2b)."""
import json
import sqlite3
import threading

import pytest

import market_intelligence.control as control
import market_intelligence.ledger as ledger
from market_intelligence.control import AuditStore, ControlUnavailable

OWNER_1 = "00000000-0000-0000-0000-000000000001"
LEDGER_TABLES = {"run_policy", "run_ledger", "reservations", "run_leases", "ledger_head"}


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


def _run_started():
    return _event(
        operation="run",
        event_kind="run_started",
        decision="allow",
        reason_code="allowed",
        counters={},
        **_policy(),
    )


def _lease():
    return _event(
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


def _allow_decision():
    values = {
        "operation": "llm",
        "event_kind": "decision",
        "decision": "allow",
        "reason_code": "allowed",
        "counters": {},
        "model": "gpt-4o-mini",
        "schema_name": "NewsDigest",
        "serialized_body_token_count": 100,
        "reserve_tokens": 637,
        "lease_owner": OWNER_1,
        "lease_epoch": 1,
        "observed_now": "2026-11-12T00:00:10+00:00",
    }
    values.update(_policy())
    return _event(**values)


def _deny_decision():
    return _event(
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


def _outcome_for(decision):
    return _event(
        operation="llm",
        event_kind="outcome",
        decision="allow",
        reason_code="allowed",
        counters={
            "calls": 1,
            "prompt_tokens": 98,
            "completion_tokens": 40,
            "reserved_tokens": 0,
            "remaining_tokens": 19862,
            "budget_deficit": 0,
        },
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


class _Wrap:
    """Delegates every attribute to the real connection; fails or counts on SQL prefixes."""

    def __init__(self, real, marker=None, *, after_execute=False, also=()):
        self._real = real
        self._marker = marker
        self._after_execute = after_execute
        self._also = tuple(also)
        self.seen = []

    def execute(self, sql, params=()):
        head = sql.strip().upper()
        self.seen.append(head)
        if self._marker is not None and head.startswith(self._marker):
            if self._after_execute:
                self._real.execute(sql, params)
            raise sqlite3.OperationalError(f"injected failure at {self._marker}")
        for extra in self._also:
            if head.startswith(extra):
                raise sqlite3.OperationalError(f"injected failure at {extra}")
        return self._real.execute(sql, params)

    def __getattr__(self, name):
        return getattr(self._real, name)


class _Cursor:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class _HeadWrap(_Wrap):
    """Answers the third audit_head read (the participant's, in finalize) with another head.

    For a one-event transaction the reads are: control's chain read, the
    participant's first-hook read, then finalize.
    """

    def __init__(self, real):
        super().__init__(real)
        self.head_reads = 0

    def execute(self, sql, params=()):
        if sql.strip().upper().startswith("SELECT SEQ, EVENT_HASH FROM AUDIT_HEAD"):
            self.head_reads += 1
            if self.head_reads == 3:
                return _Cursor((99, "f" * 64))
        return self._real.execute(sql, params)


def _lock_is_free():
    seen = []

    def probe():
        got = control._APPEND_LOCK.acquire(blocking=False)
        seen.append(got)
        if got:
            control._APPEND_LOCK.release()

    thread = threading.Thread(target=probe)
    thread.start()
    thread.join()
    return seen[0]


def _store(tmp_path, name="audit.sqlite3"):
    db = tmp_path / name
    return AuditStore(db), db


def _commit(store, *events):
    """Commit events through the real participant in one transaction."""
    with store.transaction(ledger.LedgerProjection()) as tx:
        for event in events:
            tx.append_in_tx(event)


def _audit_count(db):
    conn = sqlite3.connect(db)
    try:
        return conn.execute("SELECT count(*) FROM audit_events").fetchone()[0]
    finally:
        conn.close()


def _ledger_tables(db):
    conn = sqlite3.connect(db)
    try:
        names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
        return names & LEDGER_TABLES
    finally:
        conn.close()


def _ledger_rows(db):
    conn = sqlite3.connect(db)
    try:
        out = {}
        for name in sorted(_ledger_tables(db)):
            out[name] = conn.execute(f"SELECT * FROM {name} ORDER BY 1").fetchall()
        return out
    finally:
        conn.close()


def _both_clean(db):
    assert control.verify_chain(db) == []
    assert ledger.verify_ledger(db) == []


def test_empty_chain_transaction_commits_events_and_ledger_together(tmp_path):
    store, db = _store(tmp_path)
    decision = _allow_decision()
    try:
        _commit(store, _run_started(), _lease(), decision, _outcome_for(decision))
        assert _lock_is_free()
    finally:
        store.close()
    assert _audit_count(db) == 4
    assert _ledger_tables(db) == LEDGER_TABLES
    _both_clean(db)


def test_second_transaction_extends_an_already_projected_chain(tmp_path):
    store, db = _store(tmp_path)
    decision = _allow_decision()
    try:
        _commit(store, _run_started(), _lease())
        _commit(store, decision, _outcome_for(decision))
    finally:
        store.close()
    assert _audit_count(db) == 4
    _both_clean(db)


def test_zero_appends_create_no_ledger_tables_and_write_nothing(tmp_path):
    store, db = _store(tmp_path)
    try:
        with store.transaction(ledger.LedgerProjection()):
            pass
        assert _lock_is_free()
    finally:
        store.close()
    assert _audit_count(db) == 0
    assert _ledger_tables(db) == set()


def test_final_state_is_applied_once_for_a_multi_event_transaction(tmp_path):
    store, db = _store(tmp_path)
    real = store._conn
    wrapper = _Wrap(real)
    store._conn = wrapper
    try:
        _commit(store, _run_started(), _lease())
    finally:
        store._conn = real
        store.close()
    heads = [s for s in wrapper.seen if s.startswith("INSERT INTO LEDGER_HEAD")]
    assert len(heads) == 1
    _both_clean(db)


def test_ledger_invalid_event_rolls_back_and_stays_typed(tmp_path):
    store, db = _store(tmp_path)
    denied = _deny_decision()
    try:
        _commit(store, _run_started(), _lease(), denied)
        before = _ledger_rows(db)
        with pytest.raises(ledger.LedgerInvalidTransition, match="denied decision"):
            with store.transaction(ledger.LedgerProjection()) as tx:
                tx.append_in_tx(_outcome_for(denied))
        assert _lock_is_free()
    finally:
        store.close()
    assert _audit_count(db) == 3
    assert _ledger_rows(db) == before
    _both_clean(db)


def test_failure_creating_the_schema_after_the_audit_insert_is_known_rolled_back(tmp_path):
    store, db = _store(tmp_path)
    real = store._conn
    store._conn = _Wrap(real, "CREATE TABLE")
    try:
        with pytest.raises(ControlUnavailable) as caught:
            _commit(store, _run_started())
        assert caught.value.append_state == "known_rolled_back"
        assert _lock_is_free()
    finally:
        store._conn = real
        store.close()
    assert _audit_count(db) == 0
    assert _ledger_tables(db) == set()


def test_failure_during_final_apply_rolls_back_audit_and_ledger(tmp_path):
    store, db = _store(tmp_path)
    real = store._conn
    store._conn = _Wrap(real, "INSERT INTO RUN_POLICY")
    try:
        with pytest.raises(ControlUnavailable) as caught:
            _commit(store, _run_started())
        assert caught.value.append_state == "known_rolled_back"
        assert _lock_is_free()
    finally:
        store._conn = real
        store.close()
    assert _audit_count(db) == 0
    assert _ledger_tables(db) == set()


def test_commit_failure_before_execute_is_unknown_and_nothing_is_written(tmp_path):
    store, db = _store(tmp_path)
    real = store._conn
    wrapper = _Wrap(real, "COMMIT")
    store._conn = wrapper
    try:
        with pytest.raises(ControlUnavailable) as caught:
            _commit(store, _run_started())
        assert caught.value.append_state == "unknown"
        assert sum(1 for sql in wrapper.seen if sql.startswith("COMMIT")) == 1
        assert _lock_is_free()
    finally:
        store._conn = real
        store.close()
    assert _audit_count(db) == 0
    assert _ledger_tables(db) == set()


def test_commit_failure_after_execute_is_unknown_and_both_are_present(tmp_path):
    store, db = _store(tmp_path)
    real = store._conn
    wrapper = _Wrap(real, "COMMIT", after_execute=True)
    store._conn = wrapper
    try:
        with pytest.raises(ControlUnavailable) as caught:
            _commit(store, _run_started())
        assert caught.value.append_state == "unknown"
        assert sum(1 for sql in wrapper.seen if sql.startswith("COMMIT")) == 1
    finally:
        store._conn = real
        store.close()
    assert _audit_count(db) == 1
    _both_clean(db)


def test_rule_rejection_with_rollback_failure_is_unknown_and_keeps_the_cause(tmp_path):
    store, db = _store(tmp_path)
    denied = _deny_decision()
    try:
        _commit(store, _run_started(), _lease(), denied)
        real = store._conn
        store._conn = _Wrap(real, "ROLLBACK")
        with pytest.raises(ControlUnavailable) as caught:
            with store.transaction(ledger.LedgerProjection()) as tx:
                tx.append_in_tx(_outcome_for(denied))
        assert caught.value.append_state == "unknown"
        assert isinstance(caught.value.__cause__, ledger.LedgerInvalidTransition)
        assert real.in_transaction is True
        real.execute("ROLLBACK")
        store._conn = real
    finally:
        store.close()
    assert _audit_count(db) == 3


def test_exact_duplicate_returns_already_applied_and_writes_nothing(tmp_path):
    store, db = _store(tmp_path)
    first = _run_started()
    try:
        _commit(store, first, _lease())
        before = _ledger_rows(db)
        with store.transaction(ledger.LedgerProjection()) as tx:
            assert tx.append_in_tx(first).status == "already_applied"
            with pytest.raises(control.TransactionAborted):
                tx.append_in_tx(_lease())
    finally:
        store.close()
    assert _audit_count(db) == 2
    assert _ledger_rows(db) == before
    _both_clean(db)


def test_duplicate_after_a_pending_event_rolls_the_pending_event_back(tmp_path):
    store, db = _store(tmp_path)
    first = _run_started()
    lease = _lease()
    try:
        _commit(store, first)
        before = _ledger_rows(db)
        with store.transaction(ledger.LedgerProjection()) as tx:
            tx.append_in_tx(lease)
            assert tx.append_in_tx(first).status == "already_applied"
    finally:
        store.close()
    assert _audit_count(db) == 1
    assert _ledger_rows(db) == before
    _both_clean(db)


def test_duplicate_of_a_pending_event_rolls_everything_back(tmp_path):
    store, db = _store(tmp_path)
    started = _run_started()
    try:
        with store.transaction(ledger.LedgerProjection()) as tx:
            tx.append_in_tx(started)
            assert tx.append_in_tx(started).status == "already_applied"
    finally:
        store.close()
    assert _audit_count(db) == 0
    assert _ledger_tables(db) == set()


def test_duplicate_with_a_missing_projection_is_bootstrap_required(tmp_path):
    store, db = _store(tmp_path)
    first = _run_started()
    try:
        store.append(first)
        with pytest.raises(ledger.LedgerBootstrapRequired):
            with store.transaction(ledger.LedgerProjection()) as tx:
                tx.append_in_tx(first)
        assert _lock_is_free()
    finally:
        store.close()
    assert _audit_count(db) == 1
    assert _ledger_tables(db) == set()


def test_new_event_on_an_audit_only_chain_is_bootstrap_required(tmp_path):
    store, db = _store(tmp_path)
    try:
        store.append(_run_started())
        with pytest.raises(ledger.LedgerBootstrapRequired, match="1 events"):
            _commit(store, _lease())
    finally:
        store.close()
    assert _audit_count(db) == 1
    assert _ledger_tables(db) == set()


def test_new_event_with_committed_empty_ledger_tables_is_bootstrap_required(tmp_path):
    store, db = _store(tmp_path)
    try:
        store.append(_run_started())
        setup = sqlite3.connect(db, isolation_level=None)
        setup.execute("BEGIN IMMEDIATE")
        ledger.ensure_schema(setup)
        setup.execute("COMMIT")
        setup.close()
        with pytest.raises(ledger.LedgerBootstrapRequired):
            _commit(store, _lease())
    finally:
        store.close()
    assert _audit_count(db) == 1
    assert _ledger_tables(db) == LEDGER_TABLES
    assert all(rows == [] for rows in _ledger_rows(db).values())


def test_tampered_projection_is_a_typed_mismatch_and_nothing_is_written(tmp_path):
    store, db = _store(tmp_path)
    try:
        _commit(store, _run_started(), _lease())
        tamper = sqlite3.connect(db)
        tamper.execute("UPDATE run_ledger SET calls = 2")
        tamper.commit()
        tamper.close()
        before = _ledger_rows(db)
        with pytest.raises(ledger.LedgerProjectionMismatch) as caught:
            _commit(store, _allow_decision())
        assert caught.value.problems == (
            "run_ledger run-ledger: column calls differs from ledger replay",
        )
    finally:
        store.close()
    assert _audit_count(db) == 2
    assert _ledger_rows(db) == before


def test_schema_difference_is_a_typed_mismatch(tmp_path):
    store, db = _store(tmp_path)
    try:
        _commit(store, _run_started())
        tamper = sqlite3.connect(db)
        tamper.execute(
            "CREATE TRIGGER trg AFTER INSERT ON run_ledger BEGIN SELECT 1; END"
        )
        tamper.commit()
        tamper.close()
        with pytest.raises(ledger.LedgerProjectionMismatch):
            _commit(store, _lease())
    finally:
        store.close()
    assert _audit_count(db) == 1


def test_committed_prefix_that_fails_ledger_replay_is_reported_with_its_seq(tmp_path):
    store, db = _store(tmp_path)
    denied = _deny_decision()
    try:
        for event in (_run_started(), _lease(), denied, _outcome_for(denied)):
            store.append(event)
        assert control.verify_chain(db) == []
        expected = ledger.verify_ledger(db)
        assert expected == ["seq 4: outcome references a denied decision"]
        with pytest.raises(ledger.LedgerProjectionMismatch) as caught:
            _commit(store, _allow_decision())
        assert list(caught.value.problems) == expected
    finally:
        store.close()
    assert _audit_count(db) == 4
    assert _ledger_tables(db) == set()


def test_participant_cannot_be_reused_or_used_before_start(tmp_path):
    store, db = _store(tmp_path)
    participant = ledger.LedgerProjection()
    try:
        with store.transaction(participant) as tx:
            tx.append_in_tx(_run_started())
        with pytest.raises(ledger.LedgerUsageError, match="already used"):
            store.transaction(participant)
        assert _lock_is_free()
        assert store._conn.in_transaction is False
        with store.transaction() as tx:
            tx.append_in_tx(_lease())
    finally:
        store.close()
    assert _audit_count(db) == 2
    fresh = ledger.LedgerProjection()
    with pytest.raises(ledger.LedgerUsageError, match="not started"):
        fresh.finalize()
    with pytest.raises(ledger.LedgerUsageError, match="not started"):
        fresh.applied(None, None)


def test_broken_chain_never_reaches_the_ledger(tmp_path):
    store, db = _store(tmp_path)
    try:
        store.append(_run_started())
        tamper = sqlite3.connect(db)
        tamper.execute("UPDATE audit_events SET payload = payload || ' '")
        tamper.commit()
        tamper.close()
        with pytest.raises(control.AuditIncomplete):
            _commit(store, _lease())
        assert _lock_is_free()
    finally:
        store.close()
    assert _ledger_tables(db) == set()


def test_stored_payload_round_trips_to_the_event_the_ledger_replays(tmp_path):
    store, db = _store(tmp_path)
    started = _run_started()
    try:
        _commit(store, started)
    finally:
        store.close()
    conn = sqlite3.connect(db)
    try:
        payload = conn.execute("SELECT payload FROM audit_events").fetchone()[0]
    finally:
        conn.close()
    assert control.event_from_payload(json.loads(payload)).event_id == started.event_id


def test_head_that_differs_from_the_last_applied_event_is_a_typed_mismatch(tmp_path):
    store, db = _store(tmp_path)
    real = store._conn
    store._conn = _HeadWrap(real)
    try:
        with pytest.raises(ledger.LedgerProjectionMismatch, match="audit head differs"):
            _commit(store, _run_started())
        assert _lock_is_free()
        assert real.in_transaction is False
    finally:
        store._conn = real
        store.close()
    assert _audit_count(db) == 0
    assert _ledger_tables(db) == set()


def test_audit_insert_failure_propagates_the_original_error_after_a_checked_rollback(tmp_path):
    store, db = _store(tmp_path)
    real = store._conn
    store._conn = _Wrap(real, "INSERT INTO AUDIT_EVENTS")
    try:
        with pytest.raises(sqlite3.OperationalError, match="injected failure"):
            _commit(store, _run_started())
        assert _lock_is_free()
        assert real.in_transaction is False
    finally:
        store._conn = real
        store.close()
    assert _audit_count(db) == 0
    assert _ledger_tables(db) == set()


def test_duplicate_with_a_diverged_projection_is_a_typed_mismatch(tmp_path):
    store, db = _store(tmp_path)
    first = _run_started()
    try:
        _commit(store, first)
        tamper = sqlite3.connect(db)
        tamper.execute("UPDATE run_ledger SET calls = 2")
        tamper.commit()
        tamper.close()
        before = _ledger_rows(db)
        with pytest.raises(ledger.LedgerProjectionMismatch) as caught:
            _commit(store, first)
        assert caught.value.problems == (
            "run_ledger run-ledger: column calls differs from ledger replay",
        )
        assert _lock_is_free()
    finally:
        store.close()
    assert _audit_count(db) == 1
    assert _ledger_rows(db) == before


def test_zero_append_finalize_is_terminal(tmp_path):
    store, db = _store(tmp_path)
    participant = ledger.LedgerProjection()
    try:
        with store.transaction(participant):
            pass
        with pytest.raises(ledger.LedgerUsageError, match="already failed"):
            participant.finalize()
        with pytest.raises(ledger.LedgerUsageError, match="already failed"):
            participant.applied(None, None)
        with pytest.raises(ledger.LedgerUsageError, match="already failed"):
            participant.already_applied(None, None)
    finally:
        store.close()
    assert _audit_count(db) == 0
    assert _ledger_tables(db) == set()


def test_hooks_after_finalize_or_already_applied_are_rejected(tmp_path):
    store, db = _store(tmp_path)
    done = ledger.LedgerProjection()
    dup = ledger.LedgerProjection()
    first = _run_started()
    try:
        with store.transaction(done) as tx:
            tx.append_in_tx(first)
        with pytest.raises(ledger.LedgerUsageError, match="already failed"):
            done.applied(None, None)
        with pytest.raises(ledger.LedgerUsageError, match="already failed"):
            done.already_applied(None, None)
        with pytest.raises(ledger.LedgerUsageError, match="already failed"):
            done.finalize()
        with store.transaction(dup) as tx:
            assert tx.append_in_tx(first).status == "already_applied"
        with pytest.raises(ledger.LedgerUsageError, match="already failed"):
            dup.applied(None, None)
        with pytest.raises(ledger.LedgerUsageError, match="already failed"):
            dup.finalize()
    finally:
        store.close()
    assert _audit_count(db) == 1
    _both_clean(db)


def _open_unit_transaction(db):
    conn = sqlite3.connect(db, isolation_level=None)
    conn.execute("BEGIN IMMEDIATE")
    return conn


def _stored_rows(db):
    conn = sqlite3.connect(db)
    try:
        return conn.execute(
            "SELECT seq, event_id, prev_hash, event_hash, payload "
            "FROM audit_events ORDER BY seq"
        ).fetchall()
    finally:
        conn.close()


def test_unit_applied_guards_reject_a_stored_event_control_would_never_pass(tmp_path):
    """Direct hook calls with hand-built StoredEvent values (unit level, not end to end)."""
    store, db = _store(tmp_path)
    try:
        _commit(store, _run_started())
        with store.transaction() as tx:
            tx.append_in_tx(_lease())
    finally:
        store.close()
    rows = _stored_rows(db)
    first_row, last_row = rows[0], rows[1]

    def stored(row, **changes):
        values = dict(zip(("seq", "event_id", "prev_hash", "event_hash", "payload"), row))
        values.update(changes)
        return control.StoredEvent(**values)

    cases = [
        (stored(first_row), "not the last audit row"),
        (stored(last_row, prev_hash="0" * 64), "does not extend the prefix"),
    ]
    for candidate, text in cases:
        conn = _open_unit_transaction(db)
        try:
            participant = ledger.LedgerProjection()
            participant.start(conn)
            with pytest.raises(ledger.LedgerUsageError, match=text):
                participant.applied(candidate, None)
            with pytest.raises(ledger.LedgerUsageError, match="already failed"):
                participant.applied(candidate, None)
        finally:
            conn.execute("ROLLBACK")
            conn.close()

    later = [
        (stored(last_row, seq=5), "does not extend the last event"),
        (stored(last_row, seq=3, prev_hash=last_row[3]), "already applied"),
    ]
    for candidate, text in later:
        conn = _open_unit_transaction(db)
        try:
            participant = ledger.LedgerProjection()
            participant.start(conn)
            participant.applied(stored(last_row), None)
            with pytest.raises(ledger.LedgerUsageError, match=text):
                participant.applied(candidate, None)
            with pytest.raises(ledger.LedgerUsageError, match="already failed"):
                participant.finalize()
        finally:
            conn.execute("ROLLBACK")
            conn.close()
    assert _audit_count(db) == 2
    assert control.verify_chain(db) == []
