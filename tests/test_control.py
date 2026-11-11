"""Offline tests for the append-only H1 audit foundation (Slice 1, repaired)."""
import hashlib
import json
import sqlite3
import threading
from pathlib import Path

import pytest

from market_intelligence.control import (
    AuditEvent,
    AuditStore,
    ControlUnavailable,
    ReasonCode,
    canonical_json,
    make_event,
    verify_chain,
)


def _event(**overrides):
    values = {
        "operation": "fetch",
        "event_kind": "decision",
        "decision": "allow",
        "reason_code": ReasonCode.ALLOWED,
        "run_id": "run-1",
        "thread_id": "thread-1",
        "target_host": "example.com",
        "target_id": "op-1",
        "counters": {"calls": 1, "reserved_tokens": 10},
    }
    values.update(overrides)
    return make_event(**values)


def _manifest(directory: Path) -> dict[str, tuple[int, str]]:
    """File names, sizes and SHA-256 only; never file contents."""
    manifest: dict[str, tuple[int, str]] = {}
    for item in sorted(directory.iterdir()):
        if item.is_file():
            digest = hashlib.sha256(item.read_bytes()).hexdigest()
            manifest[item.name] = (item.stat().st_size, digest)
    return manifest


def _print_manifest(label: str, manifest: dict[str, tuple[int, str]]) -> None:
    print(f"{label}:")
    for name, (size, digest) in sorted(manifest.items()):
        print(f"  {name} size={size} sha256={digest}")


def _rows(path: Path) -> int:
    with sqlite3.connect(path) as conn:
        return conn.execute("SELECT count(*) FROM audit_events").fetchone()[0]


def test_append_and_verify_a_valid_chain(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    try:
        store.append(_event(target_id="op-1"))
        store.append(_event(event_kind="outcome", error_class="None", target_id="op-1"))
        store.append(
            _event(
                operation="llm",
                event_kind="outcome",
                model="gpt-4o-mini",
                schema_name="NewsDigest",
                target_host=None,
                target_id="llm-1",
                content_hash="a" * 64,
            )
        )
    finally:
        store.close()

    assert verify_chain(db) == []


def test_event_schema_is_bounded(tmp_path):
    with pytest.raises(TypeError):
        AuditEvent(
            event_id="00000000-0000-0000-0000-000000000001",
            recorded_at="2026-11-11T00:00:00+00:00",
            run_id="run-1",
            thread_id="thread-1",
            operation="fetch",
            event_kind="decision",
            decision="allow",
            reason_code="allowed",
            policy_version="h1-slice1",
            extra_field="not allowed",
        )

    with pytest.raises(ValueError):
        _event(reason_code="free text reason")
    with pytest.raises(ValueError):
        _event(counters={"unknown_counter": 1})
    with pytest.raises(ValueError):
        _event(operation="network")
    with pytest.raises(ValueError):
        _event(target_host="https://example.com/path?q=secret")
    with pytest.raises(ValueError):
        _event(operation="llm", event_kind="decision")
    with pytest.raises(ValueError):
        _event(model="gpt-4o-mini", schema_name="NewsDigest")
    with pytest.raises(ValueError):
        _event(operation="write", event_kind="outcome")
    with pytest.raises(ValueError):
        _event(operation="fetch", event_kind="outcome", write_outcome="committed")
    with pytest.raises(ValueError):
        _event(error_class="RuntimeError")


def test_serializer_rejects_non_exact_json():
    with pytest.raises(ValueError):
        canonical_json({"value": float("nan")})
    with pytest.raises(ValueError):
        canonical_json({"value": float("inf")})
    with pytest.raises(ValueError):
        canonical_json({"value": "\ud800"})
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_tampered_payload_is_detected(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    try:
        store.append(_event(target_id="op-1"))
        store.append(_event(event_kind="outcome", target_id="op-1"))
    finally:
        store.close()

    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE audit_events SET payload = ? WHERE seq = 2",
            ('{"tampered":true}',),
        )

    problems = verify_chain(db)
    assert any("event_hash does not match payload" in problem for problem in problems)


def test_broken_linkage_and_genesis_are_detected(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    try:
        store.append(_event(target_id="op-1"))
        store.append(_event(event_kind="outcome", target_id="op-1"))
    finally:
        store.close()

    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE audit_events SET prev_hash = ? WHERE seq = 2",
            ("0" * 64,),
        )
    problems = verify_chain(db)
    assert any("broken prev_hash linkage" in problem for problem in problems)

    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE audit_events SET prev_hash = ? WHERE seq = 1",
            ("0" * 64,),
        )
    problems = verify_chain(db)
    assert any("genesis event has a prev_hash" in problem for problem in problems)


def test_append_failure_raises_control_unavailable_and_leaves_no_row(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    real_connection = store._conn

    class _BrokenConnection:
        def execute(self, *args, **kwargs):
            raise sqlite3.OperationalError("injected failure")

        def close(self):
            pass

    try:
        store._conn = _BrokenConnection()
        with pytest.raises(ControlUnavailable):
            store.append(_event())
    finally:
        store._conn = real_connection
        store.close()

    assert _rows(db) == 0


def test_concurrent_appends_are_serialized(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    errors: list[BaseException] = []

    def worker(index: int) -> None:
        try:
            for offset in range(5):
                store.append(
                    _event(
                        event_id=None,
                        target_id=f"op-{index}-{offset}",
                    )
                )
        except BaseException as exc:  # collected and asserted below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(10)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        store.close()

    assert errors == []
    assert _rows(db) == 50
    assert verify_chain(db) == []


def test_unknown_write_outcome_and_reconciliation_are_append_only(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    unknown = _event(
        operation="write",
        event_kind="outcome",
        write_outcome="unknown",
        target_host=None,
        target_id=None,
        counters={"calls": 0},
    )
    try:
        store.append(unknown)
        with sqlite3.connect(db) as conn:
            original = conn.execute(
                "SELECT payload, event_hash FROM audit_events WHERE seq = 1"
            ).fetchone()

        reconciliation = _event(
            operation="write",
            event_kind="outcome",
            write_outcome="committed",
            reason_code=ReasonCode.RECONCILIATION,
            target_host=None,
            target_id=None,
            references_event_id=unknown.event_id,
            counters={"calls": 0},
        )
        store.append(reconciliation)
    finally:
        store.close()

    assert unknown.retry_safe is False
    assert reconciliation.retry_safe is False
    assert verify_chain(db) == []
    with sqlite3.connect(db) as conn:
        after = conn.execute(
            "SELECT payload, event_hash FROM audit_events WHERE seq = 1"
        ).fetchone()
        count = conn.execute("SELECT count(*) FROM audit_events").fetchone()[0]
    assert after == original
    assert count == 2


def test_verify_is_read_only_and_never_creates_the_file(tmp_path):
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir()
    db = audit_dir / "audit.sqlite3"
    store = AuditStore(db)
    try:
        store.append(_event(target_id="op-1"))
        store.append(_event(event_kind="outcome", target_id="op-1"))
    finally:
        store.close()

    before = _manifest(audit_dir)
    _print_manifest("MANIFEST_BEFORE", before)
    assert verify_chain(db) == []
    after = _manifest(audit_dir)
    _print_manifest("MANIFEST_AFTER", after)

    assert after == before
    missing = audit_dir / "does-not-exist.sqlite3"
    problems = verify_chain(missing)
    assert problems
    assert not missing.exists()


def test_append_refuses_to_extend_a_broken_chain(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    try:
        store.append(_event(target_id="op-1"))
        store.append(_event(event_kind="outcome", target_id="op-1"))
    finally:
        store.close()

    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE audit_events SET payload = ? WHERE seq = 2", ("{}",))

    reopened = AuditStore(db)
    try:
        with pytest.raises(ControlUnavailable) as excinfo:
            reopened.append(_event(target_id="op-2"))
    finally:
        reopened.close()

    assert excinfo.value.append_state == "known_rolled_back"
    assert excinfo.value.reason_code == "chain_broken"
    assert _rows(db) == 2


def test_append_refuses_head_without_events(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    try:
        store.append(_event())
    finally:
        store.close()

    with sqlite3.connect(db) as conn:
        conn.execute("DELETE FROM audit_events")

    reopened = AuditStore(db)
    try:
        with pytest.raises(ControlUnavailable) as excinfo:
            reopened.append(_event(target_id="op-2"))
    finally:
        reopened.close()

    assert excinfo.value.reason_code == "chain_broken"
    assert _rows(db) == 0


def test_verifier_checks_payload_against_row_columns(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    try:
        store.append(_event())
        store.append(_event(event_kind="outcome"))
    finally:
        store.close()

    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE audit_events SET event_id = ? WHERE seq = 2",
            ("00000000-0000-0000-0000-000000000099",),
        )

    problems = verify_chain(db)
    assert any("payload event_id does not match the row" in p for p in problems)


def test_verifier_revalidates_payload_schema_and_canonical_text(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    try:
        store.append(_event())
    finally:
        store.close()

    with sqlite3.connect(db) as conn:
        payload, _ = conn.execute(
            "SELECT payload, event_hash FROM audit_events WHERE seq = 1"
        ).fetchone()
        parsed = json.loads(payload)
        parsed["extra_field"] = "not allowed"
        text = canonical_json(parsed)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        conn.execute(
            "UPDATE audit_events SET payload = ?, event_hash = ? WHERE seq = 1",
            (text, digest),
        )
        conn.execute(
            "UPDATE audit_head SET event_hash = ? WHERE singleton = 1", (digest,)
        )

    problems = verify_chain(db)
    assert any("payload has unknown fields" in p for p in problems)

    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE audit_events SET payload = ? WHERE seq = 1",
            (json.dumps(parsed, indent=2),),
        )
    problems = verify_chain(db)
    assert any("payload text is not canonical JSON" in p for p in problems)


def test_verifier_requires_boolean_retry_safe_and_exact_form(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    committed = _event(
        operation="write",
        event_kind="outcome",
        write_outcome="committed",
        target_host=None,
        target_id=None,
        counters={"calls": 0},
    )
    try:
        store.append(committed)
    finally:
        store.close()

    with sqlite3.connect(db) as conn:
        payload, _ = conn.execute(
            "SELECT payload, event_hash FROM audit_events WHERE seq = 1"
        ).fetchone()
        parsed = json.loads(payload)
        parsed["retry_safe"] = 0
        text = canonical_json(parsed)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        conn.execute(
            "UPDATE audit_events SET payload = ?, event_hash = ? WHERE seq = 1",
            (text, digest),
        )
        conn.execute(
            "UPDATE audit_head SET event_hash = ? WHERE singleton = 1", (digest,)
        )

    problems = verify_chain(db)
    assert any("retry_safe must be a JSON boolean or null" in p for p in problems)
    assert any("exact canonical form" in p for p in problems)


def test_retry_safe_is_persisted_and_matches_derivation(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    failed = _event(
        operation="write",
        event_kind="outcome",
        write_outcome="failed",
        target_host=None,
        target_id=None,
        counters={"calls": 0},
    )
    committed = _event(
        operation="write",
        event_kind="outcome",
        write_outcome="committed",
        target_host=None,
        target_id=None,
        counters={"calls": 0},
    )
    try:
        store.append(failed)
        store.append(committed)
    finally:
        store.close()

    assert failed.retry_safe is True
    assert committed.retry_safe is False
    with sqlite3.connect(db) as conn:
        payloads = [
            json.loads(row[0])
            for row in conn.execute(
                "SELECT payload FROM audit_events ORDER BY seq"
            ).fetchall()
        ]
    assert payloads[0]["retry_safe"] is True
    assert payloads[1]["retry_safe"] is False
    assert verify_chain(db) == []


def test_counters_are_immutable_after_construction():
    event = _event(counters={"calls": 1, "reserved_tokens": 10})
    with pytest.raises(TypeError):
        event.counters["calls"] = 99
    assert dict(event.counters) == {"calls": 1, "reserved_tokens": 10}


def test_free_text_and_host_bounds():
    with pytest.raises(ValueError):
        _event(run_id="x" * 129)
    with pytest.raises(ValueError):
        _event(thread_id="x" * 129)
    with pytest.raises(ValueError):
        _event(target_id="x" * 129)
    with pytest.raises(ValueError):
        _event(policy_version="x" * 65)
    with pytest.raises(ValueError):
        _event(
            operation="llm",
            event_kind="decision",
            model="m" * 129,
            schema_name="NewsDigest",
        )
    with pytest.raises(ValueError):
        _event(operation="fetch", event_kind="outcome", error_class="Error: message")
    with pytest.raises(ValueError):
        _event(target_host="example.com:443")
    with pytest.raises(ValueError):
        _event(target_host="exa_mple.com")
    with pytest.raises(ValueError):
        _event(target_host="ex\u00e4mple.com")
    accepted = _event(
        operation="fetch",
        event_kind="outcome",
        error_class="package.CustomError",
    )
    assert accepted.error_class == "package.CustomError"


class _FailOnSQL:
    """Delegates to a real connection but fails on a chosen SQL prefix."""

    def __init__(self, real, marker, *, after_execute=False):
        self._real = real
        self._marker = marker
        self._after_execute = after_execute

    def execute(self, sql, params=()):
        if self._marker in sql.upper():
            if self._after_execute:
                self._real.execute(sql, params)
            raise sqlite3.OperationalError(f"injected failure at {self._marker}")
        return self._real.execute(sql, params)

    def close(self):
        self._real.close()


def test_append_failure_before_commit_is_known_rolled_back(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    real = store._conn
    try:
        store._conn = _FailOnSQL(real, "INSERT INTO AUDIT_EVENTS")
        with pytest.raises(ControlUnavailable) as excinfo:
            store.append(_event())
    finally:
        store._conn = real
        store.close()

    assert excinfo.value.append_state == "known_rolled_back"
    assert _rows(db) == 0


def test_append_head_write_failure_rolls_back_inserted_event(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    real = store._conn
    try:
        store._conn = _FailOnSQL(real, "INSERT INTO AUDIT_HEAD")
        with pytest.raises(ControlUnavailable) as excinfo:
            store.append(_event())
    finally:
        store._conn = real
        store.close()

    assert excinfo.value.append_state == "known_rolled_back"
    assert _rows(db) == 0


def test_append_commit_failure_is_unknown_and_not_auto_retried(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    real = store._conn
    try:
        store._conn = _FailOnSQL(real, "COMMIT")
        with pytest.raises(ControlUnavailable) as excinfo:
            store.append(_event())
    finally:
        store._conn = real
        store.close()

    assert excinfo.value.append_state == "unknown"
    assert _rows(db) == 0


def test_append_commit_applied_then_error_is_unknown(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    real = store._conn
    try:
        store._conn = _FailOnSQL(real, "COMMIT", after_execute=True)
        with pytest.raises(ControlUnavailable) as excinfo:
            store.append(_event())
    finally:
        store._conn = real
        store.close()

    assert excinfo.value.append_state == "unknown"
    assert _rows(db) == 1
    assert verify_chain(db) == []


def test_read_only_uri_handles_space_and_hash(tmp_path):
    audit_dir = tmp_path / "audit dir #1"
    audit_dir.mkdir()
    db = audit_dir / "audit #1.sqlite3"
    store = AuditStore(db)
    try:
        store.append(_event())
    finally:
        store.close()

    assert verify_chain(db) == []
