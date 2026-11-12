"""Offline tests for ledger schema creation and projection writes (2b-ii-B1)."""
import ast
import hashlib
import inspect
import json
import sqlite3
import textwrap

import pytest

import market_intelligence.control as control
import market_intelligence.ledger as ledger
from market_intelligence.control import AuditStore


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


def _b1_chain(tmp_path, name="audit.sqlite3"):
    db = tmp_path / name
    store = AuditStore(db)
    acquired = _start(store)
    decision = _decision(store)
    outcome = _outcome(store, decision)
    store.close()
    assert control.verify_chain(db) == []
    return db, acquired, decision, outcome


def _b1_apply_full(tmp_path, name="audit-applied.sqlite3"):
    db, acquired, decision, outcome = _b1_chain(tmp_path, name)
    events = _events(db)
    state = ledger.replay_events(events)
    conn = sqlite3.connect(db, isolation_level=None)
    try:
        conn.execute("BEGIN IMMEDIATE")
        ledger.ensure_schema(conn)
        head = conn.execute(
            "SELECT seq, event_hash FROM audit_head WHERE singleton = 1"
        ).fetchone()
        ledger.apply_state(conn, state, head)
        conn.execute("COMMIT")
    finally:
        conn.close()
    return db, state, head, acquired, decision, outcome


def _b1_rows(conn, table):
    statements = {
        "run_policy": (
            "SELECT run_id, thread_id, policy_version, estimate_kind, margin_num, "
            "margin_den, provider_overhead_tokens, max_run_tokens, max_run_llm_calls, "
            "max_output_tokens, max_fetches_per_run, lease_ttl_seconds, allowed_model, "
            "allowed_schemas_json FROM run_policy ORDER BY run_id"
        ),
        "run_ledger": (
            "SELECT run_id, calls, prompt_tokens, completion_tokens, reserved_tokens, "
            "fetch_count FROM run_ledger ORDER BY run_id"
        ),
        "reservations": (
            "SELECT run_id, decision_event_id, reserved_tokens, status, opened_at, "
            "closed_at, last_event_id FROM reservations ORDER BY decision_event_id"
        ),
        "run_leases": (
            "SELECT run_id, owner_id, epoch, expires_at, released, released_at, "
            "last_event_id FROM run_leases ORDER BY run_id"
        ),
        "ledger_head": (
            "SELECT singleton, seq, event_hash FROM ledger_head ORDER BY singleton"
        ),
    }
    return conn.execute(statements[table]).fetchall()


def _b1_literal_full(acquired, decision, outcome, head_hash):
    return {
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
        "ledger_head": [(1, 4, head_hash)],
    }


def _b1_literal_three(acquired, decision, head_hash):
    return {
        "run_policy": [(
            "run-ledger", "thread-ledger", control.POLICY_VERSION_H1_SLICE2,
            "policy_margin_v1", "1", "4", 0, 20000, 24, 512, 8, 300,
            "gpt-4o-mini", '["NewsDigest"]',
        )],
        "run_ledger": [("run-ledger", 1, 0, 0, 637, 0)],
        "reservations": [(
            "run-ledger", decision.event_id, 637, "open",
            "2026-11-12T00:00:10+00:00", None, None,
        )],
        "run_leases": [(
            "run-ledger", OWNER_1, 1, "2026-11-12T00:05:00+00:00", 0, None,
            acquired.event_id,
        )],
        "ledger_head": [(1, 3, head_hash)],
    }


def test_schema_statements_and_columns():
    assert ledger.SCHEMA_SQL == "".join(ledger.SCHEMA_STATEMENTS)
    created = []
    for statement in ledger.SCHEMA_STATEMENTS:
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute(statement)
            names = conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
            ).fetchall()
            assert len(names) == 1
            created.append(names[0][0])
        finally:
            conn.close()
    assert created == list(ledger.COLUMNS)


def test_ensure_schema_requires_an_active_transaction(tmp_path):
    db = tmp_path / "audit.sqlite3"
    AuditStore(db).close()
    conn = sqlite3.connect(db, isolation_level=None)
    try:
        with pytest.raises(ledger.LedgerUsageError):
            ledger.ensure_schema(conn)
        names = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
        assert not any(name in {row[0] for row in names} for name in ledger.COLUMNS)
    finally:
        conn.close()


def test_ensure_schema_does_not_commit_preexisting_work(tmp_path):
    db = tmp_path / "audit.sqlite3"
    conn = sqlite3.connect(db, isolation_level=None)
    try:
        conn.execute("CREATE TABLE keep (x INTEGER)")
        conn.execute("INSERT INTO keep VALUES (1)")
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("INSERT INTO keep VALUES (2)")
        ledger.ensure_schema(conn)
        assert conn.in_transaction is True
        conn.execute("ROLLBACK")
        assert conn.in_transaction is False
        assert conn.execute("SELECT x FROM keep").fetchall() == [(1,)]
        names = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )}
        assert not any(name in names for name in ledger.COLUMNS)

        second = tmp_path / "audit-second.sqlite3"
        conn2 = sqlite3.connect(second, isolation_level=None)
        try:
            conn2.execute("BEGIN IMMEDIATE")
            ledger.ensure_schema(conn2)
            conn2.execute("COMMIT")
            assert ledger.schema_problems(conn2) == []
            assert {row[0] for row in conn2.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )} >= set(ledger.COLUMNS)
        finally:
            conn2.close()
    finally:
        conn.close()


def test_ensure_schema_twice_in_one_transaction(tmp_path):
    db = tmp_path / "audit.sqlite3"
    conn = sqlite3.connect(db, isolation_level=None)
    try:
        conn.execute("BEGIN IMMEDIATE")
        ledger.ensure_schema(conn)
        ledger.ensure_schema(conn)
        assert ledger.schema_problems(conn) == []
        assert conn.in_transaction is True
    finally:
        conn.rollback()
        conn.close()


def test_ensure_schema_accepts_test_created_schema(tmp_path):
    db = tmp_path / "audit.sqlite3"
    conn = sqlite3.connect(db, isolation_level=None)
    try:
        conn.executescript(ledger.SCHEMA_SQL)
        conn.execute("BEGIN IMMEDIATE")
        ledger.ensure_schema(conn)
        assert ledger.schema_problems(conn) == []
    finally:
        conn.rollback()
        conn.close()


@pytest.mark.parametrize(
    "setup, expected",
    [
        (["CREATE TABLE run_ledger (run_id TEXT PRIMARY KEY)"], "run_ledger"),
        ([
            "CREATE TABLE run_ledger (run_id TEXT PRIMARY KEY, calls TEXT NOT NULL, "
            "prompt_tokens INTEGER NOT NULL, completion_tokens INTEGER NOT NULL, "
            "reserved_tokens INTEGER NOT NULL, fetch_count INTEGER NOT NULL)"
        ], "run_ledger"),
        ([
            "CREATE TABLE reservations (run_id TEXT NOT NULL, decision_event_id TEXT "
            "PRIMARY KEY, reserved_tokens INTEGER NOT NULL, status TEXT NOT NULL, "
            "opened_at TEXT NOT NULL, closed_at TEXT NOT NULL, last_event_id TEXT)"
        ], "reservations"),
        ([
            "CREATE TABLE ledger_head (singleton INTEGER PRIMARY KEY, seq INTEGER NOT "
            "NULL, event_hash TEXT)"
        ], "ledger_head"),
        ([
            "CREATE TRIGGER trg_run_ledger AFTER INSERT ON run_ledger BEGIN SELECT 1; END"
        ], "trigger"),
        ([
            "CREATE INDEX ix_reservations_run ON reservations(run_id)"
        ], "index"),
        ([
            "CREATE TABLE run_ledger (run_id TEXT PRIMARY KEY, calls INTEGER NOT NULL, "
            "prompt_tokens INTEGER NOT NULL, completion_tokens INTEGER NOT NULL, "
            "reserved_tokens INTEGER NOT NULL, fetch_count INTEGER NOT NULL) WITHOUT ROWID"
        ], "run_ledger"),
        (["CREATE TEMP TABLE run_ledger (x INTEGER)"], "TEMP"),
    ],
    ids=[
        "different_columns", "calls_text", "closed_at_not_null",
        "head_without_check", "trigger", "extra_index", "without_rowid", "temp_table",
    ],
)
def test_ensure_schema_rejects_adversarial_schema(tmp_path, setup, expected):
    db = tmp_path / "audit-adversarial.sqlite3"
    conn = sqlite3.connect(db, isolation_level=None)
    try:
        if expected in {"trigger", "index"}:
            for statement in ledger.SCHEMA_STATEMENTS:
                conn.execute(statement)
        for statement in setup:
            conn.execute(statement)
        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(ledger.LedgerProjectionMismatch) as caught:
            ledger.ensure_schema(conn)
        assert expected.lower() in str(caught.value).lower()
        assert conn.in_transaction is True
        if expected == "TEMP":
            assert conn.execute(
                "SELECT name FROM sqlite_temp_master WHERE name = 'run_ledger'"
            ).fetchone() is not None
        else:
            assert conn.execute(
                "SELECT name FROM sqlite_master WHERE name IN "
                "('run_ledger','reservations','ledger_head','trg_run_ledger',"
                "'ix_reservations_run')"
            ).fetchall()
        conn.execute("ROLLBACK")
        assert conn.in_transaction is False
        if expected == "TEMP":
            assert conn.execute(
                "SELECT name FROM sqlite_temp_master WHERE name = 'run_ledger'"
            ).fetchone() is not None
        else:
            assert conn.execute(
                "SELECT name FROM sqlite_master WHERE name IN "
                "('run_ledger','reservations','ledger_head','trg_run_ledger',"
                "'ix_reservations_run')"
            ).fetchall()
    finally:
        conn.close()

    control_db = tmp_path / "audit-clean.sqlite3"
    clean = sqlite3.connect(control_db, isolation_level=None)
    try:
        clean.execute("BEGIN IMMEDIATE")
        ledger.ensure_schema(clean)
        assert ledger.schema_problems(clean) == []
        clean.execute("ROLLBACK")
    finally:
        clean.close()


def test_apply_state_preconditions_and_bad_heads(tmp_path):
    db = tmp_path / "audit.sqlite3"
    AuditStore(db).close()
    conn = sqlite3.connect(db, isolation_level=None)
    try:
        with pytest.raises(ledger.LedgerUsageError):
            ledger.apply_state(conn, ledger.LedgerState(), (1, "h"))
        conn.execute("BEGIN IMMEDIATE")
        for bad in (None, (0, "h"), (True, "h"), (1, ""), (1, 5), (1, "h", "x"), [1, "h"]):
            with pytest.raises(ledger.LedgerUsageError):
                ledger.apply_state(conn, ledger.LedgerState(), bad)
            assert conn.in_transaction is True
        with pytest.raises(ledger.LedgerUsageError):
            ledger.apply_state(conn, {"runs": {}}, (1, "h"))
        assert conn.in_transaction is True
        conn.execute("ROLLBACK")
    finally:
        conn.close()


def test_projection_errors_and_exception_hierarchy(tmp_path):
    db, acquired, decision, outcome = _b1_chain(tmp_path)
    state = ledger.replay_events(_events(db))
    conn = sqlite3.connect(db, isolation_level=None)
    try:
        conn.execute("BEGIN IMMEDIATE")
        ledger.ensure_schema(conn)
        with pytest.raises(ledger.LedgerUsageError):
            ledger.projection_problems(conn, state, None)
        conn.execute("ROLLBACK")
    finally:
        conn.close()
    assert not isinstance(ledger.LedgerUsageError(), ledger.LedgerError)
    assert isinstance(ledger.LedgerProjectionMismatch(["x"]), ledger.LedgerError)


def test_apply_state_sqlite_error_propagates_and_leaves_earlier_writes(tmp_path):
    db, acquired, decision, outcome = _b1_chain(tmp_path)
    state = ledger.replay_events(_events(db))
    conn = sqlite3.connect(db, isolation_level=None)
    try:
        for statement in (
            ledger.RUN_POLICY_SQL,
            ledger.RUN_LEDGER_SQL,
            ledger.RUN_LEASES_SQL,
            ledger.LEDGER_HEAD_SQL,
        ):
            conn.execute(statement)
        head = conn.execute(
            "SELECT seq, event_hash FROM audit_head WHERE singleton = 1"
        ).fetchone()
        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(sqlite3.OperationalError, match="reservations"):
            ledger.apply_state(conn, state, head)
        assert conn.in_transaction is True
        assert conn.execute("SELECT count(*) FROM run_policy").fetchone()[0] == 1
        assert conn.execute("SELECT count(*) FROM run_ledger").fetchone()[0] == 1
        conn.execute("ROLLBACK")
        assert conn.execute("SELECT count(*) FROM run_policy").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM run_ledger").fetchone()[0] == 0
    finally:
        conn.close()


def test_apply_state_literal_rows_and_verifiers(tmp_path):
    db, state, head, acquired, decision, outcome = _b1_apply_full(tmp_path)
    conn = sqlite3.connect(db)
    try:
        literal = _b1_literal_full(acquired, decision, outcome, head[1])
        for table in ledger.COLUMNS:
            assert _b1_rows(conn, table) == literal[table]
        assert ledger.projection_problems(conn, state, head) == []
    finally:
        conn.close()
    assert control.verify_chain(db) == []
    assert ledger.verify_ledger(db) == []


def test_apply_state_upserts_without_deleting(tmp_path):
    db, acquired, decision, outcome = _b1_chain(tmp_path)
    events = _events(db)
    state3 = ledger.replay_events(events[:3])
    state4 = ledger.replay_events(events)
    conn = sqlite3.connect(db, isolation_level=None)
    try:
        head3 = conn.execute(
            "SELECT seq, event_hash FROM audit_events WHERE seq = 3"
        ).fetchone()
        conn.execute("BEGIN IMMEDIATE")
        ledger.ensure_schema(conn)
        ledger.apply_state(conn, state3, head3)
        conn.execute("COMMIT")
        literal3 = _b1_literal_three(acquired, decision, head3[1])
        for table in ledger.COLUMNS:
            assert _b1_rows(conn, table) == literal3[table]

        head4 = conn.execute(
            "SELECT seq, event_hash FROM audit_head WHERE singleton = 1"
        ).fetchone()
        conn.execute("BEGIN IMMEDIATE")
        ledger.apply_state(conn, state4, head4)
        conn.execute("COMMIT")
        literal4 = _b1_literal_full(acquired, decision, outcome, head4[1])
        for table in ledger.COLUMNS:
            assert _b1_rows(conn, table) == literal4[table]
            assert len(_b1_rows(conn, table)) == len(literal4[table])

        conn.execute("BEGIN IMMEDIATE")
        ledger.apply_state(conn, state4, head4)
        conn.execute("COMMIT")
        for table in ledger.COLUMNS:
            assert _b1_rows(conn, table) == literal4[table]
    finally:
        conn.close()


def test_apply_state_respects_rollback_and_commit(tmp_path):
    db, acquired, decision, outcome = _b1_chain(tmp_path)
    events = _events(db)
    state3 = ledger.replay_events(events[:3])
    state4 = ledger.replay_events(events)
    conn = sqlite3.connect(db, isolation_level=None)
    try:
        head3 = conn.execute(
            "SELECT seq, event_hash FROM audit_events WHERE seq = 3"
        ).fetchone()
        conn.execute("BEGIN IMMEDIATE")
        ledger.ensure_schema(conn)
        ledger.apply_state(conn, state3, head3)
        conn.execute("COMMIT")
        literal3 = _b1_literal_three(acquired, decision, head3[1])

        head4 = conn.execute(
            "SELECT seq, event_hash FROM audit_head WHERE singleton = 1"
        ).fetchone()
        conn.execute("BEGIN IMMEDIATE")
        ledger.apply_state(conn, state4, head4)
        assert conn.in_transaction is True
        conn.execute("ROLLBACK")
        for table in ledger.COLUMNS:
            assert _b1_rows(conn, table) == literal3[table]

        conn.execute("BEGIN IMMEDIATE")
        ledger.apply_state(conn, state4, head4)
        conn.execute("COMMIT")
        literal4 = _b1_literal_full(acquired, decision, outcome, head4[1])
        for table in ledger.COLUMNS:
            assert _b1_rows(conn, table) == literal4[table]
    finally:
        conn.close()


def test_upsert_sql_is_built_from_columns_and_keys():
    assert list(ledger.UPSERT_SQL) == list(ledger.COLUMNS)
    for table, columns in ledger.COLUMNS.items():
        key = ledger.TABLE_KEYS[table]
        expected = (
            f"INSERT INTO {table} ({', '.join(columns)}) VALUES "
            f"({', '.join('?' for _ in columns)}) ON CONFLICT({key}) DO UPDATE SET "
            + ", ".join(
                f"{column} = excluded.{column}" for column in columns if column != key
            )
        )
        assert ledger.UPSERT_SQL[table] == expected
        assert "DELETE" not in ledger.UPSERT_SQL[table]


def test_no_forbidden_calls_or_sql_in_source(tmp_path):
    functions = [
        ledger.ensure_schema,
        ledger.apply_state,
        ledger.schema_problems,
        ledger.projection_problems,
        ledger.projection_is_empty,
    ]
    for function in functions:
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        body = tree.body[0].body
        if body and isinstance(body[0], ast.Expr) and isinstance(
            body[0].value, ast.Constant
        ) and isinstance(body[0].value.value, str):
            body = body[1:]
        errors = []
        for statement in body:
            for node in ast.walk(statement):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if node.func.attr in {"executescript", "commit", "rollback"}:
                        errors.append(node.func.attr)
                    if node.func.attr == "close" and function is not ledger.schema_problems:
                        errors.append("close")
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    text = node.value.strip().upper()
                    if text.startswith(
                        ("BEGIN", "COMMIT", "ROLLBACK", "SAVEPOINT", "DELETE", "DROP", "ALTER")
                    ):
                        errors.append(node.value)
        assert errors == []

    schema_source = textwrap.dedent(inspect.getsource(ledger.schema_problems))
    tree = ast.parse(schema_source)
    private_ref = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            call = node.value
            if (
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "connect"
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "sqlite3"
                and call.args
                and isinstance(call.args[0], ast.Constant)
                and call.args[0].value == ":memory:"
            ):
                private_ref = node.targets[0].id
    assert private_ref is not None
    close_calls = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "close"
        ):
            close_calls.append(node.func.value)
    assert len(close_calls) == 1
    assert isinstance(close_calls[0], ast.Name) and close_calls[0].id == private_ref

    db = tmp_path / "audit.sqlite3"
    AuditStore(db).close()
    conn = sqlite3.connect(db, isolation_level=None)
    try:
        conn.execute("BEGIN IMMEDIATE")
        ledger.ensure_schema(conn)
        assert ledger.schema_problems(conn) == []
        assert conn.in_transaction is True
        conn.execute("CREATE TABLE scratch (x INTEGER)")
        conn.execute("INSERT INTO scratch VALUES (1)")
        conn.execute("ROLLBACK")
    finally:
        conn.close()

    conn = sqlite3.connect(db, isolation_level=None)
    try:
        conn.execute("BEGIN IMMEDIATE")
        ledger.ensure_schema(conn)
        conn.execute("COMMIT")
        conn.execute(
            "CREATE TRIGGER trg AFTER INSERT ON run_ledger BEGIN SELECT 1; END"
        )
        conn.execute("BEGIN IMMEDIATE")
        assert ledger.schema_problems(conn)
        assert conn.in_transaction is True
        assert conn.execute("SELECT 1").fetchone() == (1,)
        conn.execute("CREATE TABLE scratch (x INTEGER)")
        conn.execute("INSERT INTO scratch VALUES (2)")
        conn.execute("ROLLBACK")
    finally:
        conn.close()


@pytest.mark.parametrize(
    "sql, expected",
    [
        ("UPDATE run_ledger SET calls = 2",
         "run_ledger run-ledger: column calls differs from ledger replay"),
        ("DELETE FROM reservations", "reservations: missing row {decision}"),
        ("INSERT INTO run_leases VALUES ('extra', 'x', 1, 'z', 0, NULL, NULL)",
         "run_leases: unexpected row extra"),
        ("DELETE FROM ledger_head", "ledger_head: missing row 1"),
        ("UPDATE ledger_head SET seq = 5",
         "ledger_head 1: column seq differs from ledger replay"),
    ],
    ids=["run_ledger_calls", "reservations_missing", "extra_lease", "head_missing", "head_seq"],
)
def test_projection_tamper_problems(tmp_path, sql, expected):
    db, state, head, acquired, decision, outcome = _b1_apply_full(
        tmp_path, f"audit-{expected[:4]}.sqlite3"
    )
    conn = sqlite3.connect(db, isolation_level=None)
    try:
        assert ledger.projection_problems(conn, state, head) == []
        conn.execute(sql)
        conn.commit()
        expected_problems = [expected.replace("{decision}", decision.event_id)]
        assert ledger.projection_problems(conn, state, head) == expected_problems
    finally:
        conn.close()


def test_empty_chain_projection_rules(tmp_path):
    db = tmp_path / "audit.sqlite3"
    AuditStore(db).close()
    conn = sqlite3.connect(db, isolation_level=None)
    try:
        conn.execute("BEGIN IMMEDIATE")
        ledger.ensure_schema(conn)
        assert ledger.projection_problems(conn, ledger.LedgerState(), None) == []
        assert ledger.projection_is_empty(conn) is True
        conn.execute(
            "INSERT INTO run_ledger VALUES ('stray', 0, 0, 0, 0, 0)"
        )
        assert ledger.projection_problems(
            conn, ledger.LedgerState(), None
        ) == ["run_ledger: unexpected row stray"]
        assert ledger.projection_is_empty(conn) is False
        conn.execute("DELETE FROM run_ledger")
        conn.execute("INSERT INTO ledger_head VALUES (1, 1, NULL)")
        assert ledger.projection_problems(
            conn, ledger.LedgerState(), None
        ) == ["ledger_head: unexpected row 1"]
        conn.execute("UPDATE ledger_head SET event_hash = ?", ("0" * 64,))
        assert ledger.projection_problems(
            conn, ledger.LedgerState(), None
        ) == ["ledger_head: unexpected row 1"]
        with pytest.raises(ledger.LedgerUsageError):
            ledger.apply_state(conn, ledger.LedgerState(), None)
        assert conn.in_transaction is True
        conn.execute("ROLLBACK")
    finally:
        conn.close()


@pytest.mark.parametrize("table", list(ledger.COLUMNS), ids=list(ledger.COLUMNS))
def test_projection_is_empty_false_for_one_row(tmp_path, table):
    db = tmp_path / f"audit-{table}.sqlite3"
    conn = sqlite3.connect(db, isolation_level=None)
    try:
        conn.execute("BEGIN IMMEDIATE")
        ledger.ensure_schema(conn)
        values = {
            "run_policy": (
                "r", "t", control.POLICY_VERSION_H1_SLICE2, "k", "1", "4",
                0, 1, 1, 1, 1, 1, "m", "[]",
            ),
            "run_ledger": ("r", 0, 0, 0, 0, 0),
            "reservations": ("r", "d", 0, "open", "t", None, None),
            "run_leases": ("r", "o", 1, "t", 0, None, None),
            "ledger_head": (1, 1, None),
        }[table]
        conn.execute(
            f"INSERT INTO {table} VALUES ({', '.join('?' for _ in values)})", values
        )
        assert ledger.projection_is_empty(conn) is False
        conn.execute("ROLLBACK")
    finally:
        conn.close()


def test_projection_is_empty_missing_table_and_fresh(tmp_path):
    db = tmp_path / "audit-missing.sqlite3"
    conn = sqlite3.connect(db, isolation_level=None)
    try:
        for statement in ledger.SCHEMA_STATEMENTS[:-1]:
            conn.execute(statement)
        assert ledger.projection_is_empty(conn) is False
        conn.execute("BEGIN IMMEDIATE")
        ledger.ensure_schema(conn)
        assert ledger.projection_is_empty(conn) is True
        conn.execute("ROLLBACK")
    finally:
        conn.close()
    applied, state, head, acquired, decision, outcome = _b1_apply_full(
        tmp_path, "audit-applied-empty.sqlite3"
    )
    conn = sqlite3.connect(applied)
    try:
        assert ledger.projection_is_empty(conn) is False
    finally:
        conn.close()


def test_apply_state_cannot_detect_wrong_state_and_head(tmp_path):
    db, acquired, decision, outcome = _b1_chain(tmp_path)
    events = _events(db)
    state3 = ledger.replay_events(events[:3])
    conn = sqlite3.connect(db, isolation_level=None)
    try:
        head4 = conn.execute(
            "SELECT seq, event_hash FROM audit_head WHERE singleton = 1"
        ).fetchone()
        conn.execute("BEGIN IMMEDIATE")
        ledger.ensure_schema(conn)
        ledger.apply_state(conn, state3, head4)
        conn.execute("COMMIT")
    finally:
        conn.close()
    assert ledger.verify_ledger(db) != []


def test_read_helpers_propagate_non_missing_sqlite_errors(tmp_path):
    db = tmp_path / "audit-locked.sqlite3"
    writer = sqlite3.connect(db, isolation_level=None)
    reader = None
    try:
        for statement in ledger.SCHEMA_STATEMENTS:
            writer.execute(statement)
        reader = sqlite3.connect(db, isolation_level=None, timeout=0)
        writer.execute("BEGIN EXCLUSIVE")
        with pytest.raises(sqlite3.OperationalError) as caught:
            ledger.projection_is_empty(reader)
        assert "locked" in str(caught.value).lower()
        with pytest.raises(sqlite3.OperationalError) as caught:
            ledger.projection_problems(reader, ledger.LedgerState(), None)
        assert type(caught.value) is sqlite3.OperationalError
        assert "locked" in str(caught.value).lower()
        assert reader.in_transaction is False
        writer.execute("ROLLBACK")
        assert ledger.projection_is_empty(reader) is True
        assert ledger.projection_problems(reader, ledger.LedgerState(), None) == []
    finally:
        if reader is not None:
            reader.close()
        writer.close()


def test_verify_ledger_does_not_modify_the_file(tmp_path):
    db, state, head, acquired, decision, outcome = _b1_apply_full(
        tmp_path, "audit-readonly.sqlite3"
    )
    before = (db.stat().st_size, hashlib.sha256(db.read_bytes()).hexdigest())
    assert ledger.verify_ledger(db) == []
    after = (db.stat().st_size, hashlib.sha256(db.read_bytes()).hexdigest())
    assert after == before
