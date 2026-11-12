"""Offline tests for the optional projection participant on AuditTransaction (2b-ii-B2a)."""
import json
import sqlite3
import threading

import pytest

import market_intelligence.control as control
import market_intelligence.ledger as ledger
from market_intelligence.control import (
    AuditIncomplete,
    AuditStore,
    ControlUnavailable,
    TransactionAborted,
)


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


def _run_started(run_id):
    return control.make_event(
        run_id=run_id,
        thread_id="thread-b2a",
        policy_version=control.POLICY_VERSION_H1_SLICE2,
        operation="run",
        event_kind="run_started",
        decision="allow",
        reason_code="allowed",
        counters={},
        **_policy(),
    )


class _Wrap:
    """Delegates every attribute to the real connection; fails on one SQL word."""

    def __init__(self, real, marker=None, *, after_execute=False, also=()):
        self._real = real
        self._marker = marker
        self._after_execute = after_execute
        self._also = tuple(also)
        self.injected = 0

    def execute(self, sql, params=()):
        head = sql.strip().upper()
        if self._marker is not None and head.startswith(self._marker):
            self.injected += 1
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


class _ReadWrap(_Wrap):
    """Answers the stored-row read from control with a missing or altered row."""

    def __init__(self, real, mode):
        super().__init__(real)
        self._mode = mode

    def execute(self, sql, params=()):
        cursor = self._real.execute(sql, params)
        head = sql.strip().upper()
        if head.startswith("SELECT SEQ, EVENT_ID") and "WHERE EVENT_ID" in head:
            if self._mode == "missing":
                return _Cursor(None)
            row = list(cursor.fetchone())
            row[3] = "0" * 64
            return _Cursor(tuple(row))
        return cursor


class _Rejected(control.ProjectionRejected):
    pass


class _Fake:
    """Participant that records calls and writes a scratch table in the transaction."""

    def __init__(self, fail_at=None, error=None):
        self.calls = []
        self.stored = []
        self.fail_at = fail_at
        self.error = error
        self.conn = None

    def _maybe_fail(self, where):
        if self.fail_at == where:
            raise self.error

    def start(self, conn):
        if self.conn is not None:
            raise RuntimeError("participant already bound to a transaction")
        self._maybe_fail("start")
        self.calls.append("start")
        self.conn = conn
        conn.execute("CREATE TABLE scratch (x INTEGER)")

    def applied(self, stored, result):
        self.calls.append(("applied", stored.seq))
        self.stored.append((stored, result))
        self.conn.execute("INSERT INTO scratch VALUES (?)", (stored.seq,))
        self._maybe_fail("applied")

    def already_applied(self, event, result):
        self.calls.append(("already_applied", event.event_id, result.status))
        self.conn.execute("INSERT INTO scratch VALUES (-1)")
        self._maybe_fail("already_applied")

    def finalize(self):
        self.calls.append("finalize")
        self.conn.execute("INSERT INTO scratch VALUES (99)")
        self._maybe_fail("finalize")


def _lock_is_free():
    """True when another thread can take the process-wide audit lock."""
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


def _state(db):
    """(audit event count, scratch table exists, scratch values) from a new connection."""
    conn = sqlite3.connect(db)
    try:
        events = conn.execute("SELECT count(*) FROM audit_events").fetchone()[0]
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE name = 'scratch'"
        ).fetchall()
        values = (
            [row[0] for row in conn.execute("SELECT x FROM scratch ORDER BY rowid")]
            if exists
            else []
        )
        return events, bool(exists), values
    finally:
        conn.close()


def _store(tmp_path):
    db = tmp_path / "audit.sqlite3"
    return AuditStore(db), db


def test_plain_transaction_commits_and_releases_lock(tmp_path):
    store, db = _store(tmp_path)
    try:
        with store.transaction() as tx:
            assert tx.append_in_tx(_run_started("r1")).status == "applied"
        assert _lock_is_free()
    finally:
        store.close()
    assert _state(db) == (1, False, [])
    assert control.verify_chain(db) == []


def test_commit_failure_before_execute_is_unknown_and_releases_lock_once(tmp_path):
    store, db = _store(tmp_path)
    real = store._conn
    wrapper = _Wrap(real, "COMMIT")
    store._conn = wrapper
    control._APPEND_LOCK.acquire()
    try:
        with pytest.raises(ControlUnavailable) as caught:
            with store.transaction() as tx:
                tx.append_in_tx(_run_started("r1"))
        assert caught.value.append_state == "unknown"
        assert wrapper.injected == 1
        control._APPEND_LOCK.release()
        with pytest.raises(RuntimeError):
            control._APPEND_LOCK.release()
        assert _lock_is_free()
    finally:
        store._conn = real
        store.close()
    assert _state(db)[0] == 0


def test_commit_failure_after_execute_is_unknown_and_row_is_present(tmp_path):
    store, db = _store(tmp_path)
    real = store._conn
    store._conn = _Wrap(real, "COMMIT", after_execute=True)
    try:
        with pytest.raises(ControlUnavailable) as caught:
            with store.transaction() as tx:
                tx.append_in_tx(_run_started("r1"))
        assert caught.value.append_state == "unknown"
        assert _lock_is_free()
    finally:
        store._conn = real
        store.close()
    assert _state(db)[0] == 1


def test_body_error_with_rollback_failure_is_unknown(tmp_path):
    store, db = _store(tmp_path)
    real = store._conn
    store._conn = _Wrap(real, "ROLLBACK")
    try:
        with pytest.raises(ControlUnavailable) as caught:
            with store.transaction() as tx:
                tx.append_in_tx(_run_started("r1"))
                raise KeyError("body failed")
        assert caught.value.append_state == "unknown"
        assert isinstance(caught.value.__context__, KeyError)
        assert _lock_is_free()
        assert real.in_transaction is True
        real.execute("ROLLBACK")
    finally:
        store._conn = real
        store.close()
    assert _state(db)[0] == 0


def test_already_applied_with_rollback_failure_is_unknown(tmp_path):
    store, db = _store(tmp_path)
    event = _run_started("r0")
    store.append(event)
    real = store._conn
    store._conn = _Wrap(real, "ROLLBACK")
    try:
        with pytest.raises(ControlUnavailable) as caught:
            with store.transaction() as tx:
                tx.append_in_tx(event)
        assert caught.value.append_state == "unknown"
        assert _lock_is_free()
        assert real.in_transaction is True
        real.execute("ROLLBACK")
    finally:
        store._conn = real
        store.close()
    assert _state(db)[0] == 1


def test_begin_failure_is_known_rolled_back_and_releases_lock(tmp_path):
    store, db = _store(tmp_path)
    real = store._conn
    store._conn = _Wrap(real, "BEGIN")
    try:
        with pytest.raises(ControlUnavailable) as caught:
            store.transaction(_Fake())
        assert caught.value.append_state == "known_rolled_back"
        assert _lock_is_free()
    finally:
        store._conn = real
        store.close()


def test_participant_sees_ordered_hooks_and_commits_with_the_audit_rows(tmp_path):
    store, db = _store(tmp_path)
    fake = _Fake()
    try:
        with store.transaction(fake) as tx:
            first = tx.append_in_tx(_run_started("ra"))
            second = tx.append_in_tx(_run_started("rb"))
        assert _lock_is_free()
    finally:
        store.close()
    assert fake.calls == ["start", ("applied", 1), ("applied", 2), "finalize"]
    (stored_one, result_one), (stored_two, result_two) = fake.stored
    assert stored_one.prev_hash is None
    assert stored_two.prev_hash == stored_one.event_hash
    assert stored_one.event_hash == first.event_hash == result_one.event_hash
    assert stored_two.event_hash == second.event_hash == result_two.event_hash
    assert result_one.status == "applied"
    assert control.event_from_payload(
        json.loads(stored_one.payload)
    ).event_id == stored_one.event_id
    assert _state(db) == (2, True, [1, 2, 99])
    assert control.verify_chain(db) == []


def test_zero_appends_still_calls_finalize_once(tmp_path):
    store, db = _store(tmp_path)
    fake = _Fake()
    try:
        with store.transaction(fake):
            pass
    finally:
        store.close()
    assert fake.calls == ["start", "finalize"]
    assert _state(db) == (0, True, [99])


def test_duplicate_after_pending_calls_hook_before_abort_and_rolls_everything_back(tmp_path):
    store, db = _store(tmp_path)
    committed = _run_started("r0")
    store.append(committed)
    fake = _Fake()
    try:
        with store.transaction(fake) as tx:
            tx.append_in_tx(_run_started("r1"))
            result = tx.append_in_tx(committed)
            assert result.status == "already_applied"
            assert tx.result == result
            with pytest.raises(TransactionAborted):
                tx.append_in_tx(_run_started("r2"))
        assert _lock_is_free()
    finally:
        store.close()
    assert fake.calls == [
        "start",
        ("applied", 2),
        ("already_applied", committed.event_id, "already_applied"),
    ]
    assert _state(db) == (1, False, [])


def test_duplicate_of_a_pending_event_aborts_the_whole_transaction(tmp_path):
    store, db = _store(tmp_path)
    pending = _run_started("r1")
    fake = _Fake()
    try:
        with store.transaction(fake) as tx:
            tx.append_in_tx(pending)
            assert tx.append_in_tx(pending).status == "already_applied"
    finally:
        store.close()
    assert fake.calls == [
        "start",
        ("applied", 1),
        ("already_applied", pending.event_id, "already_applied"),
    ]
    assert _state(db) == (0, False, [])


def test_rule_rejection_at_every_hook_rolls_back_and_stays_typed(tmp_path):
    for where in ("start", "applied", "already_applied", "finalize"):
        case = tmp_path / where
        case.mkdir()
        store, db = _store(case)
        committed = _run_started("r0")
        store.append(committed)
        fake = _Fake(fail_at=where, error=_Rejected(where))
        try:
            with pytest.raises(_Rejected):
                if where == "start":
                    store.transaction(fake)
                else:
                    with store.transaction(fake) as tx:
                        tx.append_in_tx(
                            committed if where == "already_applied" else _run_started("r1")
                        )
            assert _lock_is_free()
        finally:
            store.close()
        assert _state(db) == (1, False, []), where
        assert control.verify_chain(db) == [], where


def test_finalize_sqlite_error_is_known_rolled_back_not_commit_unknown(tmp_path):
    store, db = _store(tmp_path)
    fake = _Fake(fail_at="finalize", error=sqlite3.OperationalError("disk problem"))
    try:
        with pytest.raises(ControlUnavailable) as caught:
            with store.transaction(fake) as tx:
                tx.append_in_tx(_run_started("r1"))
        assert caught.value.append_state == "known_rolled_back"
        assert "commit outcome" not in str(caught.value)
        assert _lock_is_free()
    finally:
        store.close()
    assert _state(db) == (0, False, [])


def test_hook_error_with_rollback_failure_is_unknown(tmp_path):
    store, db = _store(tmp_path)
    real = store._conn
    store._conn = _Wrap(real, "ROLLBACK")
    fake = _Fake(fail_at="finalize", error=_Rejected("rejected"))
    try:
        with pytest.raises(ControlUnavailable) as caught:
            with store.transaction(fake) as tx:
                tx.append_in_tx(_run_started("r1"))
        assert caught.value.append_state == "unknown"
        assert isinstance(caught.value.__cause__, _Rejected)
        assert _lock_is_free()
        assert real.in_transaction is True
        real.execute("ROLLBACK")
    finally:
        store._conn = real
        store.close()
    assert _state(db)[0] == 0


def test_unexpected_hook_error_is_reraised_unchanged_after_checked_rollback(tmp_path):
    store, db = _store(tmp_path)
    error = RuntimeError("wiring bug")
    fake = _Fake(fail_at="applied", error=error)
    try:
        with pytest.raises(RuntimeError) as caught:
            with store.transaction(fake) as tx:
                tx.append_in_tx(_run_started("r1"))
        assert caught.value is error
        assert _lock_is_free()
    finally:
        store.close()
    assert _state(db) == (0, False, [])


def test_broken_chain_never_reaches_the_applied_hook(tmp_path):
    store, db = _store(tmp_path)
    store.append(_run_started("r0"))
    tamper = sqlite3.connect(db)
    tamper.execute("UPDATE audit_events SET payload = payload || ' '")
    tamper.commit()
    tamper.close()
    fake = _Fake()
    try:
        with pytest.raises(AuditIncomplete):
            with store.transaction(fake) as tx:
                tx.append_in_tx(_run_started("r1"))
        assert _lock_is_free()
    finally:
        store.close()
    assert fake.calls == ["start"]


def test_start_failure_rolls_back_and_releases_lock_even_without_rollback(tmp_path):
    store, db = _store(tmp_path)
    error = RuntimeError("participant reused")
    try:
        with pytest.raises(RuntimeError) as caught:
            store.transaction(_Fake(fail_at="start", error=error))
        assert caught.value is error
        assert _lock_is_free()
        assert store._conn.in_transaction is False
        real = store._conn
        store._conn = _Wrap(real, "ROLLBACK")
        with pytest.raises(ControlUnavailable) as unconfirmed:
            store.transaction(_Fake(fail_at="start", error=error))
        assert unconfirmed.value.append_state == "unknown"
        assert _lock_is_free()
        assert real.in_transaction is True
        real.execute("ROLLBACK")
        store._conn = real
    finally:
        store.close()


def test_ledger_errors_inherit_the_projection_rejection_base(tmp_path):
    assert issubclass(ledger.LedgerError, control.ProjectionRejected)
    assert issubclass(ledger.LedgerError, ValueError)
    assert issubclass(ledger.LedgerProjectionMismatch, control.ProjectionRejected)
    assert issubclass(ledger.LedgerInvalidTransition, control.ProjectionRejected)
    assert not issubclass(ledger.LedgerUsageError, control.ProjectionRejected)
    assert not issubclass(ledger.LedgerUsageError, ledger.LedgerError)


def test_append_failure_with_rollback_failure_is_unknown_before_and_after_insert(tmp_path):
    for label, after in (("before", False), ("after", True)):
        case = tmp_path / label
        case.mkdir()
        store, db = _store(case)
        real = store._conn
        store._conn = _Wrap(
            real, "INSERT INTO AUDIT_EVENTS", after_execute=after, also=("ROLLBACK",)
        )
        try:
            with pytest.raises(ControlUnavailable) as caught:
                with store.transaction() as tx:
                    tx.append_in_tx(_run_started("r1"))
            assert caught.value.append_state == "unknown", label
            assert isinstance(caught.value.__cause__, sqlite3.OperationalError), label
            assert _lock_is_free(), label
            assert real.in_transaction is True, label
            real.execute("ROLLBACK")
        finally:
            store._conn = real
            store.close()
        assert _state(db)[0] == 0, label


def test_broken_chain_with_rollback_failure_is_unknown_and_keeps_the_cause(tmp_path):
    store, db = _store(tmp_path)
    store.append(_run_started("r0"))
    tamper = sqlite3.connect(db)
    tamper.execute("UPDATE audit_events SET payload = payload || ' '")
    tamper.commit()
    tamper.close()
    real = store._conn
    store._conn = _Wrap(real, "ROLLBACK")
    try:
        with pytest.raises(ControlUnavailable) as caught:
            with store.transaction() as tx:
                tx.append_in_tx(_run_started("r1"))
        assert caught.value.append_state == "unknown"
        assert isinstance(caught.value.__cause__, AuditIncomplete)
        assert _lock_is_free()
        assert real.in_transaction is True
        real.execute("ROLLBACK")
    finally:
        store._conn = real
        store.close()


def test_participant_cannot_be_reused_and_a_second_start_writes_nothing(tmp_path):
    store, db = _store(tmp_path)
    fake = _Fake()
    try:
        with store.transaction(fake) as tx:
            tx.append_in_tx(_run_started("r1"))
        with pytest.raises(RuntimeError, match="already bound"):
            store.transaction(fake)
        assert _lock_is_free()
        assert store._conn.in_transaction is False
        with store.transaction() as tx:
            tx.append_in_tx(_run_started("r2"))
    finally:
        store.close()
    assert fake.calls == ["start", ("applied", 1), "finalize"]
    assert _state(db) == (2, True, [1, 99])


def test_stored_row_mismatch_or_missing_never_reaches_applied_and_rolls_back(tmp_path):
    for mode in ("missing", "hash"):
        case = tmp_path / mode
        case.mkdir()
        store, db = _store(case)
        real = store._conn
        store._conn = _ReadWrap(real, mode)
        fake = _Fake()
        try:
            with pytest.raises(AuditIncomplete):
                with store.transaction(fake) as tx:
                    tx.append_in_tx(_run_started("r1"))
            assert _lock_is_free(), mode
            assert real.in_transaction is False, mode
        finally:
            store._conn = real
            store.close()
        assert fake.calls == ["start"], mode
        assert _state(db) == (0, False, []), mode
