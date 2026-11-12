"""Offline tests for the append-only H1 audit foundation (Slice 1, repaired)."""
import hashlib
import json
import re
import sqlite3
import threading
import uuid
from pathlib import Path

import pytest

import market_intelligence.control as control_module

from market_intelligence.control import (
    AuditEvent,
    AuditIncomplete,
    AuditStore,
    ControlUnavailable,
    POLICY_VERSION_H1_SLICE2,
    ReasonCode,
    TransactionAborted,
    canonical_decimal,
    canonical_json,
    logical_fingerprint,
    make_event,
    parse_canonical_decimal,
    verify_chain,
)


OWNER_1 = "00000000-0000-0000-0000-000000000001"
OWNER_2 = "00000000-0000-0000-0000-000000000002"
OWNER_OTHER = "00000000-0000-0000-0000-000000000003"


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


# --- Slice 2b-i: 2b event model and atomic append -------------------------


def _v2_policy(**overrides):
    values = {
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
    values.update(overrides)
    return values


def _v2_event(**overrides):
    values = {
        "run_id": "run-2b",
        "thread_id": "thread-2b",
        "policy_version": POLICY_VERSION_H1_SLICE2,
    }
    values.update(overrides)
    return make_event(**values)


def _v2_start(store, *, owner=OWNER_1, observed="2026-11-12T00:00:00+00:00"):
    started = _v2_event(
        operation="run",
        event_kind="run_started",
        decision="allow",
        reason_code="allowed",
        counters={},
        **_v2_policy(),
    )
    store.append(started)
    acquired = _v2_event(
        operation="run",
        event_kind="lease_acquired",
        decision="allow",
        reason_code="allowed",
        counters={},
        lease_owner=owner,
        lease_epoch=1,
        observed_now=observed,
        lease_expires_at="2026-11-12T00:05:00+00:00",
        lease_ttl_seconds=300,
    )
    store.append(acquired)
    return started, acquired


def _v2_llm_decision(store, *, owner=OWNER_1, observed="2026-11-12T00:00:10+00:00"):
    event = _v2_event(
        operation="llm",
        event_kind="decision",
        decision="allow",
        reason_code="allowed",
        counters={},
        model="gpt-4o-mini",
        schema_name="NewsDigest",
        serialized_body_token_count=100,
        reserve_tokens=637,
        lease_owner=owner,
        lease_epoch=1,
        observed_now=observed,
        **_v2_policy(),
    )
    store.append(event)
    return event


def _usage_counters(prompt=98, completion=40, reserved=0):
    balance = 20000 - prompt - completion - reserved
    return {
        "calls": 1,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "reserved_tokens": reserved,
        "remaining_tokens": max(0, balance),
        "budget_deficit": max(0, -balance),
    }


def _v2_outcome(store, decision, **overrides):
    values = {
        "operation": "llm",
        "event_kind": "outcome",
        "decision": "allow",
        "reason_code": "allowed",
        "counters": _usage_counters(),
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
    event = _v2_event(**values)
    store.append(event)
    return event


def _insert_payload_direct(db, payload):
    """Insert a payload as the next chain row with consistent hashes/head (no append())."""
    with sqlite3.connect(db) as conn:
        last = conn.execute(
            "SELECT seq, event_hash FROM audit_events ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        prev_hash = last[1] if last else None
        digest = control_module._event_hash(payload, prev_hash)
        text = canonical_json({**payload, "prev_hash": prev_hash})
        seq = (last[0] + 1) if last else 1
        conn.execute(
            "INSERT INTO audit_events (event_id, prev_hash, event_hash, payload) "
            "VALUES (?, ?, ?, ?)",
            (payload["event_id"], prev_hash, digest, text),
        )
        conn.execute(
            "INSERT INTO audit_head (singleton, seq, event_hash) "
            "VALUES (1, ?, ?) "
            "ON CONFLICT(singleton) DO UPDATE SET "
            "seq = excluded.seq, event_hash = excluded.event_hash",
            (seq, digest),
        )


def _insert_row_direct(db, event):
    _insert_payload_direct(db, event.to_payload())


def _tamper_row_payload(db, seq, **changes):
    """Rewrite the HEAD row's payload in place, keeping prev_hash and head consistent.

    Head-only: the target row MUST be the chain's last row (highest seq), because this
    helper points audit_head at the tampered row's new digest. Tampering a non-head row
    would corrupt the head linkage and mask the intended problem, so assert it.
    """
    with sqlite3.connect(db) as conn:
        head = conn.execute("SELECT MAX(seq) FROM audit_events").fetchone()[0]
        assert seq == head, "_tamper_row_payload target must be the head row"
        stored = json.loads(
            conn.execute(
                "SELECT payload FROM audit_events WHERE seq = ?", (seq,)
            ).fetchone()[0]
        )
        prev_hash = stored["prev_hash"]
        payload = {key: value for key, value in stored.items() if key != "prev_hash"}
        payload.update(changes)
        digest = control_module._event_hash(payload, prev_hash)
        text = canonical_json({**payload, "prev_hash": prev_hash})
        conn.execute(
            "UPDATE audit_events SET event_hash = ?, payload = ? WHERE seq = ?",
            (digest, text, seq),
        )
        conn.execute(
            "UPDATE audit_head SET event_hash = ? WHERE seq = ?", (digest, seq),
        )


def test_transaction_appends_without_nested_begin(tmp_path):
    store = AuditStore(tmp_path / "audit.sqlite3")
    try:
        with store.transaction() as tx:
            result = tx.append_in_tx(
                _v2_event(
                    operation="run",
                    event_kind="run_started",
                    decision="allow",
                    reason_code="allowed",
                    counters={},
                    **_v2_policy(),
                )
            )
            assert result.status == "applied"
            assert tx.head() is not None
    finally:
        store.close()
    assert verify_chain(tmp_path / "audit.sqlite3") == []


def test_transaction_rolls_back_on_exception(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    try:
        with pytest.raises(RuntimeError):
            with store.transaction() as tx:
                tx.append_in_tx(
                    _v2_event(
                        operation="run",
                        event_kind="run_started",
                        decision="allow",
                        reason_code="allowed",
                        counters={},
                        **_v2_policy(),
                    )
                )
                raise RuntimeError("stop")
    finally:
        store.close()
    assert _rows(db) == 0
    assert verify_chain(db) == ["audit chain is empty"]


def test_idempotent_re_read_by_fingerprint_is_success_without_second_effect(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    event = _event(target_id="op-1")
    try:
        store.append(event)
        with store.transaction() as tx:
            result = tx.append_in_tx(event)
            assert result.status == "already_applied"
            with pytest.raises(TransactionAborted):
                tx.append_in_tx(event)
    finally:
        store.close()
    assert _rows(db) == 1
    assert verify_chain(db) == []


def test_conflicting_fingerprint_is_rejected(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    event = _event(target_id="op-1")
    conflicting = _event(target_id="op-2", event_id=event.event_id)
    try:
        store.append(event)
        with pytest.raises(AuditIncomplete):
            with store.transaction() as tx:
                tx.append_in_tx(conflicting)
    finally:
        store.close()
    assert _rows(db) == 1
    assert verify_chain(db) == []


def test_public_append_still_rejects_duplicate_event_id(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    event = _event()
    try:
        store.append(event)
        with pytest.raises(ControlUnavailable):
            store.append(event)
    finally:
        store.close()
    assert _rows(db) == 1


def test_fingerprint_excludes_counters_but_includes_semantic_fields():
    base = _event(target_id="op-1")
    other_counter = _event(target_id="op-1", counters={"calls": 9}, event_id=base.event_id)
    other_semantic = _event(target_id="op-2", event_id=base.event_id)
    assert logical_fingerprint(base) == logical_fingerprint(other_counter)
    assert logical_fingerprint(base) != logical_fingerprint(other_semantic)


def test_2b_fields_require_2b_version():
    with pytest.raises(ValueError):
        _v2_event(
            operation="run",
            event_kind="run_started",
            decision="allow",
            reason_code="allowed",
            counters={},
            policy_version="h1-slice1",
            **_v2_policy(),
        )


def test_run_started_requires_complete_policy_and_no_binding():
    with pytest.raises(ValueError):
        _v2_event(
            operation="run",
            event_kind="run_started",
            decision="allow",
            reason_code="allowed",
            counters={},
            **{**_v2_policy(), "max_run_tokens": None},
        )
    with pytest.raises(ValueError):
        _v2_event(
            operation="run",
            event_kind="run_started",
            decision="allow",
            reason_code="allowed",
            counters={},
            lease_owner=OWNER_1,
            lease_epoch=1,
            observed_now="2026-11-12T00:00:00+00:00",
            **_v2_policy(),
        )


def test_policy_ratio_must_match_float_ratio():
    with pytest.raises(ValueError):
        _v2_event(
            operation="run",
            event_kind="run_started",
            decision="allow",
            reason_code="allowed",
            counters={},
            **{**_v2_policy(), "policy_margin": 0.1, "policy_margin_num": 1, "policy_margin_den": 10},
        )


def test_run_approved_requires_target_and_is_single_scope(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    try:
        _v2_start(store)
        with pytest.raises(ValueError):
            _v2_event(
                operation="run",
                event_kind="run_approved",
                decision="allow",
                reason_code="allowed",
                counters={},
                lease_owner=OWNER_1,
                lease_epoch=1,
                observed_now="2026-11-12T00:00:10+00:00",
            )
        approved = _v2_event(
            operation="run",
            event_kind="run_approved",
            decision="allow",
            reason_code="allowed",
            counters={},
            lease_owner=OWNER_1,
            lease_epoch=1,
            observed_now="2026-11-12T00:00:10+00:00",
            approved_target_id="write-target-1",
        )
        store.append(approved)
        second = _v2_event(
            operation="run",
            event_kind="run_approved",
            decision="allow",
            reason_code="allowed",
            counters={},
            lease_owner=OWNER_1,
            lease_epoch=1,
            observed_now="2026-11-12T00:00:11+00:00",
            approved_target_id="write-target-2",
        )
        with pytest.raises(ControlUnavailable):
            store.append(second)
    finally:
        store.close()
    assert verify_chain(db) == []


def test_lease_release_keeps_epoch_and_rejects_old_holder(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    try:
        _v2_start(store)
        store.append(
            _v2_event(
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
        with pytest.raises(ControlUnavailable):
            _v2_llm_decision(store, observed="2026-11-12T00:00:40+00:00")
        store.append(
            _v2_event(
                operation="run",
                event_kind="lease_acquired",
                decision="allow",
                reason_code="allowed",
                counters={},
                lease_owner=OWNER_2,
                lease_epoch=2,
                observed_now="2026-11-12T00:00:45+00:00",
                lease_expires_at="2026-11-12T00:05:45+00:00",
                lease_ttl_seconds=300,
            )
        )
    finally:
        store.close()
    assert verify_chain(db) == []


def test_decision_binding_mismatch_is_rejected(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    try:
        _v2_start(store)
        bad = _v2_event(
            operation="llm",
            event_kind="decision",
            decision="allow",
            reason_code="allowed",
            counters={},
            model="gpt-4o-mini",
            schema_name="NewsDigest",
            serialized_body_token_count=100,
            reserve_tokens=637,
            lease_owner=OWNER_OTHER,
            lease_epoch=1,
            observed_now="2026-11-12T00:00:10+00:00",
            **_v2_policy(),
        )
        with pytest.raises(ControlUnavailable):
            store.append(bad)
    finally:
        store.close()
    assert verify_chain(db) == []


def test_lease_expiry_arithmetic_and_expired_binding(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    try:
        started = _v2_event(
            operation="run",
            event_kind="run_started",
            decision="allow",
            reason_code="allowed",
            counters={},
            **_v2_policy(),
        )
        store.append(started)
        with pytest.raises(ControlUnavailable):
            store.append(
                _v2_event(
                    operation="run",
                    event_kind="lease_acquired",
                    decision="allow",
                    reason_code="allowed",
                    counters={},
                    lease_owner=OWNER_1,
                    lease_epoch=1,
                    observed_now="2026-11-12T00:00:00+00:00",
                    lease_expires_at="2026-11-12T00:07:00+00:00",
                    lease_ttl_seconds=300,
                )
            )
        acquired = _v2_event(
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
        expired_decision = _v2_event(
            operation="llm",
            event_kind="decision",
            decision="allow",
            reason_code="allowed",
            counters={},
            model="gpt-4o-mini",
            schema_name="NewsDigest",
            serialized_body_token_count=100,
            reserve_tokens=637,
            lease_owner=OWNER_1,
            lease_epoch=1,
            observed_now="2026-11-12T00:05:00+00:00",
            **_v2_policy(),
        )
        with pytest.raises(ControlUnavailable):
            store.append(expired_decision)
        assert verify_chain(db) == []
    finally:
        store.close()
    assert verify_chain(db) == []

    db2 = tmp_path / "audit-expired.sqlite3"
    store2 = AuditStore(db2)
    _v2_start(store2)
    expired_decision_b = _v2_event(
        operation="llm",
        event_kind="decision",
        decision="allow",
        reason_code="allowed",
        counters={},
        model="gpt-4o-mini",
        schema_name="NewsDigest",
        serialized_body_token_count=100,
        reserve_tokens=637,
        lease_owner=OWNER_1,
        lease_epoch=1,
        observed_now="2026-11-12T00:05:00+00:00",
        **_v2_policy(),
    )
    _insert_row_direct(db2, expired_decision_b)
    store2.close()
    problems = verify_chain(db2)
    assert len(problems) == 1 and "lease expired at observed_now" in problems[0]


def test_lease_takeover_requires_expiry_and_reference(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    try:
        _, acquired = _v2_start(store)
        takeover = _v2_event(
            operation="run",
            event_kind="lease_taken_over",
            decision="allow",
            reason_code="allowed",
            counters={},
            lease_owner=OWNER_2,
            lease_epoch=2,
            observed_now="2026-11-12T00:06:00+00:00",
            lease_expires_at="2026-11-12T00:11:00+00:00",
            lease_ttl_seconds=300,
            previous_lease_owner=OWNER_1,
            previous_lease_epoch=1,
            references_event_id=acquired.event_id,
        )
        store.append(takeover)
        with pytest.raises(ControlUnavailable):
            _v2_llm_decision(store, owner=OWNER_1, observed="2026-11-12T00:06:10+00:00")
    finally:
        store.close()
    assert verify_chain(db) == []


def test_lease_renewal_and_backward_clock_are_checked(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    try:
        _v2_start(store, observed="2026-11-12T00:00:00+00:00")
        renewal = _v2_event(
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
        store.append(renewal)
        backward = _v2_event(
            operation="run",
            event_kind="lease_renewed",
            decision="allow",
            reason_code="allowed",
            counters={},
            lease_owner=OWNER_1,
            lease_epoch=1,
            observed_now="2026-11-12T00:00:30+00:00",
            lease_expires_at="2026-11-12T00:05:30+00:00",
            lease_ttl_seconds=300,
        )
        with pytest.raises(ControlUnavailable):
            store.append(backward)
    finally:
        store.close()
    assert verify_chain(db) == []


def test_outcome_usage_and_balance_equations(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    try:
        _v2_start(store)
        decision = _v2_llm_decision(store)
        _v2_outcome(store, decision)
    finally:
        store.close()
    assert verify_chain(db) == []

    with sqlite3.connect(db) as conn:
        payload, _ = conn.execute(
            "SELECT payload, event_hash FROM audit_events ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        parsed = json.loads(payload)
        parsed["counters"]["remaining_tokens"] = 1
        text = canonical_json(parsed)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        conn.execute(
            "UPDATE audit_events SET payload = ?, event_hash = ? WHERE seq = 4",
            (text, digest),
        )
        conn.execute(
            "UPDATE audit_head SET event_hash = ? WHERE singleton = 1", (digest,)
        )
    assert any("remaining_tokens" in p for p in verify_chain(db))


def test_budget_breach_outcome_requires_negative_balance(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    try:
        _v2_start(store)
        decision = _v2_llm_decision(store)
        counters = _usage_counters(prompt=12000, completion=9000)
        _v2_outcome(
            store,
            decision,
            reason_code="budget_breach",
            counters=counters,
            usage_prompt_tokens=12000,
            usage_completion_tokens=9000,
            overrun_tokens=20363,
        )
        with pytest.raises(ControlUnavailable):
            _v2_outcome(
                store,
                decision,
                reason_code="allowed",
                counters=_usage_counters(prompt=1000, completion=1000),
                usage_prompt_tokens=1000,
                usage_completion_tokens=1000,
                references_event_id=decision.event_id,
                observed_now="2026-11-12T00:00:30+00:00",
                event_id=str(uuid.uuid4()),
            )
    finally:
        store.close()
    assert verify_chain(db) == []


def test_reservation_overrun_requires_positive_overrun(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    try:
        _v2_start(store)
        decision = _v2_llm_decision(store)
        with pytest.raises(ValueError):
            _v2_event(
                operation="llm",
                event_kind="outcome",
                decision="allow",
                reason_code="reservation_overrun",
                counters=_usage_counters(prompt=120, completion=600),
                model="gpt-4o-mini",
                schema_name="NewsDigest",
                usage_prompt_tokens=120,
                usage_completion_tokens=600,
                reservation_released=637,
                overrun_tokens=0,
                references_event_id=decision.event_id,
                lease_owner=OWNER_1,
                lease_epoch=1,
                observed_now="2026-11-12T00:00:20+00:00",
            )
        _v2_outcome(
            store,
            decision,
            reason_code="reservation_overrun",
            counters=_usage_counters(prompt=120, completion=600),
            usage_prompt_tokens=120,
            usage_completion_tokens=600,
            overrun_tokens=83,
        )
    finally:
        store.close()
    assert verify_chain(db) == []


def test_usage_missing_marks_unknown_without_zero_usage(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    try:
        _v2_start(store)
        decision = _v2_llm_decision(store)
        unknown = _v2_event(
            operation="llm",
            event_kind="outcome",
            decision="allow",
            reason_code="usage_missing",
            counters={},
            model="gpt-4o-mini",
            schema_name="NewsDigest",
            usage_missing=True,
            reservation_released=0,
            references_event_id=decision.event_id,
            lease_owner=OWNER_1,
            lease_epoch=1,
            observed_now="2026-11-12T00:00:20+00:00",
        )
        store.append(unknown)
        reconciled = _v2_event(
            operation="llm",
            event_kind="outcome",
            decision="allow",
            reason_code="reconciliation",
            counters=_usage_counters(),
            model="gpt-4o-mini",
            schema_name="NewsDigest",
            usage_prompt_tokens=98,
            usage_completion_tokens=40,
            reservation_released=637,
            references_event_id=unknown.event_id,
            lease_owner=OWNER_1,
            lease_epoch=1,
            observed_now="2026-11-12T00:00:30+00:00",
        )
        store.append(reconciled)
        with pytest.raises(ControlUnavailable):
            store.append(
                _v2_event(
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
                    observed_now="2026-11-12T00:00:40+00:00",
                )
            )
    finally:
        store.close()
    assert verify_chain(db) == []


def test_second_normal_outcome_is_chain_problem(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    try:
        _v2_start(store)
        decision = _v2_llm_decision(store)
        _v2_outcome(store, decision)
        with pytest.raises(ControlUnavailable):
            _v2_outcome(
                store,
                decision,
                observed_now="2026-11-12T00:00:30+00:00",
            )
    finally:
        store.close()
    assert verify_chain(db) == []


def test_fetch_outcome_requires_result_class_and_no_reservation(tmp_path):
    with pytest.raises(ValueError):
        _v2_event(
            operation="fetch",
            event_kind="outcome",
            decision="allow",
            reason_code="allowed",
            counters={},
            target_host="example.com",
            references_event_id=str(uuid.uuid4()),
            lease_owner=OWNER_1,
            lease_epoch=1,
            observed_now="2026-11-12T00:00:20+00:00",
        )
    with pytest.raises(ValueError):
        _v2_event(
            operation="write",
            event_kind="outcome",
            decision="allow",
            reason_code="allowed",
            counters={},
            write_outcome="committed",
            references_event_id=str(uuid.uuid4()),
            lease_owner=OWNER_1,
            lease_epoch=1,
            observed_now="2026-11-12T00:00:20+00:00",
            reserve_tokens=1,
        )

    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    try:
        _v2_start(store)
        decision = _v2_event(
            operation="fetch",
            event_kind="decision",
            decision="allow",
            reason_code="allowed",
            counters={},
            target_host="example.com",
            lease_owner=OWNER_1,
            lease_epoch=1,
            observed_now="2026-11-12T00:00:10+00:00",
            **_v2_policy(),
        )
        store.append(decision)
        outcome = _v2_event(
            operation="fetch",
            event_kind="outcome",
            decision="allow",
            reason_code="allowed",
            counters={},
            target_host="example.com",
            result_class="ok",
            references_event_id=decision.event_id,
            lease_owner=OWNER_1,
            lease_epoch=1,
            observed_now="2026-11-12T00:00:20+00:00",
        )
        store.append(outcome)
    finally:
        store.close()
    assert verify_chain(db) == []


def test_decision_thread_binding_is_checked(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    try:
        _v2_start(store)
        bad = _v2_event(
            operation="llm",
            event_kind="decision",
            decision="allow",
            reason_code="allowed",
            counters={},
            thread_id="other-thread",
            model="gpt-4o-mini",
            schema_name="NewsDigest",
            serialized_body_token_count=100,
            reserve_tokens=637,
            lease_owner=OWNER_1,
            lease_epoch=1,
            observed_now="2026-11-12T00:00:10+00:00",
            **_v2_policy(),
        )
        with pytest.raises(ControlUnavailable):
            store.append(bad)
    finally:
        store.close()
    assert verify_chain(db) == []


def test_legacy_event_after_run_started_is_chain_problem(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    try:
        _v2_start(store)
        with pytest.raises(ControlUnavailable):
            store.append(_event(run_id="run-2b", thread_id="thread-2b"))
    finally:
        store.close()
    assert verify_chain(db) == []


def test_legacy_only_chain_remains_verify_only(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    try:
        store.append(_event(run_id="legacy-run", thread_id="legacy-thread"))
        store.append(
            _event(
                run_id="legacy-run",
                thread_id="legacy-thread",
                event_kind="outcome",
            )
        )
    finally:
        store.close()
    assert verify_chain(db) == []


def test_transaction_rejects_invalid_candidate_and_rolls_back(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    try:
        bad = _v2_event(
            operation="llm",
            event_kind="decision",
            decision="allow",
            reason_code="allowed",
            counters={},
            model="gpt-4o-mini",
            schema_name="NewsDigest",
            serialized_body_token_count=100,
            reserve_tokens=637,
            lease_owner=OWNER_1,
            lease_epoch=1,
            observed_now="2026-11-12T00:00:10+00:00",
            **_v2_policy(),
        )
        with pytest.raises(AuditIncomplete):
            with store.transaction() as tx:
                tx.append_in_tx(bad)
    finally:
        store.close()
    assert _rows(db) == 0


def test_transaction_does_not_commit_until_exit(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    try:
        with store.transaction() as tx:
            tx.append_in_tx(
                _v2_event(
                    operation="run",
                    event_kind="run_started",
                    decision="allow",
                    reason_code="allowed",
                    counters={},
                    **_v2_policy(),
                )
            )
            assert _rows(db) == 0
        assert _rows(db) == 1
    finally:
        store.close()
    assert verify_chain(db) == []


def test_denied_llm_tokenizer_unavailable_carries_no_reserve():
    event = _v2_event(
        operation="llm",
        event_kind="decision",
        decision="deny",
        reason_code="tokenizer_unavailable",
        counters={},
        model="gpt-4o-mini",
        schema_name="NewsDigest",
        serialized_body_token_count=None,
        reserve_tokens=None,
        lease_owner=OWNER_1,
        lease_epoch=1,
        observed_now="2026-11-12T00:00:10+00:00",
        **_v2_policy(),
    )
    assert event.reserve_tokens is None


def test_v2_counters_and_usage_are_bounded_to_int64():
    with pytest.raises(ValueError):
        _v2_event(
            operation="run",
            event_kind="run_started",
            decision="allow",
            reason_code="allowed",
            counters={"calls": 2**63},
            **_v2_policy(),
        )
    with pytest.raises(ValueError):
        _v2_event(
            operation="llm",
            event_kind="outcome",
            decision="allow",
            reason_code="allowed",
            counters=_usage_counters(),
            model="gpt-4o-mini",
            schema_name="NewsDigest",
            usage_prompt_tokens=2**63,
            usage_completion_tokens=1,
            reservation_released=637,
            references_event_id=str(uuid.uuid4()),
            lease_owner=OWNER_1,
            lease_epoch=1,
            observed_now="2026-11-12T00:00:20+00:00",
        )


def test_usage_out_of_range_requires_typed_error_class():
    with pytest.raises(ValueError):
        _v2_event(
            operation="llm",
            event_kind="outcome",
            decision="allow",
            reason_code="usage_out_of_range",
            counters={},
            model="gpt-4o-mini",
            schema_name="NewsDigest",
            usage_missing=True,
            error_class="OtherError",
            reservation_released=0,
            references_event_id=str(uuid.uuid4()),
            lease_owner=OWNER_1,
            lease_epoch=1,
            observed_now="2026-11-12T00:00:20+00:00",
        )


def test_released_no_effect_requires_positive_release():
    with pytest.raises(ValueError):
        _v2_event(
            operation="llm",
            event_kind="outcome",
            decision="allow",
            reason_code="released_no_effect",
            counters={},
            model="gpt-4o-mini",
            schema_name="NewsDigest",
            reservation_released=0,
            references_event_id=str(uuid.uuid4()),
            lease_owner=OWNER_1,
            lease_epoch=1,
            observed_now="2026-11-12T00:00:20+00:00",
        )


def test_v2_write_decision_requires_target():
    with pytest.raises(ValueError):
        _v2_event(
            operation="write",
            event_kind="decision",
            decision="allow",
            reason_code="allowed",
            counters={},
            lease_owner=OWNER_1,
            lease_epoch=1,
            observed_now="2026-11-12T00:00:10+00:00",
            **_v2_policy(),
        )


def test_transaction_poison_after_idempotent_blocks_commit(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    e0 = _event(target_id="e0")
    e1 = _event(event_kind="outcome", target_id="e1")
    try:
        store.append(e0)
        with store.transaction() as tx:
            assert tx.append_in_tx(e1).status == "applied"
            result = tx.append_in_tx(e0)
            assert result.status == "already_applied"
            with pytest.raises(TransactionAborted):
                tx.append_in_tx(e1)
    finally:
        store.close()
    assert _rows(db) == 1
    with sqlite3.connect(db) as conn:
        ids = [row[0] for row in conn.execute("SELECT event_id FROM audit_events")]
        head = conn.execute(
            "SELECT seq, event_hash FROM audit_head WHERE singleton = 1"
        ).fetchone()
        stored_hash = conn.execute(
            "SELECT event_hash FROM audit_events WHERE event_id = ?", (e0.event_id,)
        ).fetchone()[0]
    assert ids == [e0.event_id]
    assert head == (1, stored_hash)
    assert e1.event_id not in ids
    assert verify_chain(db) == []


def test_canonical_decimal_round_trip_and_rejection():
    for value in (0, 1, 10**399):
        text = canonical_decimal(value)
        assert parse_canonical_decimal(text) == value
    for text in ("+1", "-1", "01", " 1", "1 ", "", "x", "9" * 401):
        with pytest.raises(ValueError):
            parse_canonical_decimal(text)
    with pytest.raises(ValueError):
        canonical_decimal(-1)


def test_mixed_legacy_and_v2_chain_verifies(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    try:
        store.append(_event(run_id="legacy-run", thread_id="legacy-thread"))
        store.append(
            _event(
                run_id="legacy-run",
                thread_id="legacy-thread",
                event_kind="outcome",
            )
        )
        _v2_start(store)
        _v2_llm_decision(store)
    finally:
        store.close()
    assert verify_chain(db) == []


def test_verifier_rejects_duplicate_run_started(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    try:
        _v2_start(store)
        duplicate = _v2_event(
            operation="run",
            event_kind="run_started",
            decision="allow",
            reason_code="allowed",
            counters={},
            **_v2_policy(),
        )
        with sqlite3.connect(db) as conn:
            last = conn.execute(
                "SELECT seq, event_hash FROM audit_events ORDER BY seq DESC LIMIT 1"
            ).fetchone()
            payload = duplicate.to_payload()
            prev_hash = last[1]
            digest = control_module._event_hash(payload, prev_hash)
            text = canonical_json({**payload, "prev_hash": prev_hash})
            seq = last[0] + 1
            conn.execute(
                "INSERT INTO audit_events (event_id, prev_hash, event_hash, payload) "
                "VALUES (?, ?, ?, ?)",
                (duplicate.event_id, prev_hash, digest, text),
            )
            conn.execute(
                "UPDATE audit_head SET seq = ?, event_hash = ? WHERE singleton = 1",
                (seq, digest),
            )
    finally:
        store.close()
    assert any("duplicate run_started" in p for p in verify_chain(db))


def test_verifier_rejects_policy_snapshot_mismatch(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    try:
        _v2_start(store)
        decision = _v2_llm_decision(store)
        with sqlite3.connect(db) as conn:
            row = conn.execute(
                "SELECT seq, prev_hash, payload FROM audit_events WHERE event_id = ?",
                (decision.event_id,),
            ).fetchone()
            payload = json.loads(row[2])
            payload["policy_margin"] = 0.5
            payload["policy_margin_num"] = 1
            payload["policy_margin_den"] = 2
            digest = control_module._event_hash(payload, row[1])
            text = canonical_json({**payload, "prev_hash": row[1]})
            conn.execute(
                "UPDATE audit_events SET payload = ?, event_hash = ? WHERE seq = ?",
                (text, digest, row[0]),
            )
            conn.execute(
                "UPDATE audit_head SET event_hash = ? WHERE singleton = 1", (digest,)
            )
    finally:
        store.close()
    assert any("policy snapshot mismatch" in p for p in verify_chain(db))


def test_already_applied_skips_context_revalidation(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    try:
        _v2_start(store)
        original = _v2_llm_decision(store)
        store.append(
            _v2_event(
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
        before = _rows(db)
        with store.transaction() as tx:
            result = tx.append_in_tx(original)
            assert result.status == "already_applied"
            with pytest.raises(TransactionAborted):
                tx.append_in_tx(original)
    finally:
        store.close()
    assert _rows(db) == before
    assert verify_chain(db) == []


def test_integrity_conflict_without_row_is_typed_problem(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    store.append(_event(target_id="e0"))
    real = store._conn

    class _EmptyCursor:
        def fetchone(self):
            return None

    class _IntegrityNoRow:
        def __init__(self, real_conn):
            self._real = real_conn

        def execute(self, sql, params=()):
            upper = sql.upper()
            if "INSERT INTO AUDIT_EVENTS" in upper:
                raise sqlite3.IntegrityError("forced event_id conflict")
            if "SELECT EVENT_HASH, PAYLOAD FROM AUDIT_EVENTS WHERE EVENT_ID" in upper:
                return _EmptyCursor()
            return self._real.execute(sql, params)

        def close(self):
            self._real.close()

    try:
        store._conn = _IntegrityNoRow(real)
        with pytest.raises(AuditIncomplete):
            with store.transaction() as tx:
                tx.append_in_tx(_event(target_id="e1"))
    finally:
        store._conn = real
        store.close()
    assert _rows(db) == 1
    assert verify_chain(db) == []


def test_lease_acquire_while_active_or_unreleased_is_rejected(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    try:
        _, acquired = _v2_start(store)
        with pytest.raises(ControlUnavailable):
            store.append(
                _v2_event(
                    operation="run",
                    event_kind="lease_acquired",
                    decision="allow",
                    reason_code="allowed",
                    counters={},
                    lease_owner=OWNER_2,
                    lease_epoch=2,
                    observed_now="2026-11-12T00:00:20+00:00",
                    lease_expires_at="2026-11-12T00:05:20+00:00",
                    lease_ttl_seconds=300,
                )
            )
        with pytest.raises(ControlUnavailable):
            store.append(
                _v2_event(
                    operation="run",
                    event_kind="lease_acquired",
                    decision="allow",
                    reason_code="allowed",
                    counters={},
                    lease_owner=OWNER_2,
                    lease_epoch=2,
                    observed_now="2026-11-12T00:06:00+00:00",
                    lease_expires_at="2026-11-12T00:11:00+00:00",
                    lease_ttl_seconds=300,
                )
            )
        store.append(
            _v2_event(
                operation="run",
                event_kind="lease_taken_over",
                decision="allow",
                reason_code="allowed",
                counters={},
                lease_owner=OWNER_2,
                lease_epoch=2,
                observed_now="2026-11-12T00:06:00+00:00",
                lease_expires_at="2026-11-12T00:11:00+00:00",
                lease_ttl_seconds=300,
                previous_lease_owner=OWNER_1,
                previous_lease_epoch=1,
                references_event_id=acquired.event_id,
            )
        )
    finally:
        store.close()
    assert verify_chain(db) == []


def test_lease_renewal_after_expiry_is_chain_problem(tmp_path):
    renewal = _v2_event(
        operation="run",
        event_kind="lease_renewed",
        decision="allow",
        reason_code="allowed",
        counters={},
        lease_owner=OWNER_1,
        lease_epoch=1,
        observed_now="2026-11-12T00:05:00+00:00",
        lease_expires_at="2026-11-12T00:10:00+00:00",
        lease_ttl_seconds=300,
    )
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    _v2_start(store)
    with pytest.raises(ControlUnavailable):
        store.append(renewal)
    assert _rows(db) == 2
    store.close()
    assert verify_chain(db) == []

    db2 = tmp_path / "audit-b.sqlite3"
    store2 = AuditStore(db2)
    _v2_start(store2)
    _insert_row_direct(db2, renewal)
    store2.close()
    problems = verify_chain(db2)
    assert len(problems) == 1 and "renewal after expiry" in problems[0]


def test_lease_wrong_owner_renewal_or_release_is_chain_problem(tmp_path):
    bad_renewal = _v2_event(
        operation="run",
        event_kind="lease_renewed",
        decision="allow",
        reason_code="allowed",
        counters={},
        lease_owner=OWNER_2,
        lease_epoch=1,
        observed_now="2026-11-12T00:01:00+00:00",
        lease_expires_at="2026-11-12T00:06:00+00:00",
        lease_ttl_seconds=300,
    )
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    _v2_start(store)
    with pytest.raises(ControlUnavailable):
        store.append(bad_renewal)
    assert _rows(db) == 2
    store.close()
    assert verify_chain(db) == []

    db2 = tmp_path / "audit-b.sqlite3"
    store2 = AuditStore(db2)
    _v2_start(store2)
    _insert_row_direct(db2, bad_renewal)
    store2.close()
    problems = verify_chain(db2)
    assert len(problems) == 1 and "renewal owner/epoch mismatch" in problems[0]

    bad_release = _v2_event(
        operation="run",
        event_kind="lease_released",
        decision="allow",
        reason_code="allowed",
        counters={},
        lease_owner=OWNER_2,
        lease_epoch=1,
        observed_now="2026-11-12T00:01:00+00:00",
        lease_ttl_seconds=300,
    )
    db3 = tmp_path / "audit-c.sqlite3"
    store3 = AuditStore(db3)
    _v2_start(store3)
    with pytest.raises(ControlUnavailable):
        store3.append(bad_release)
    assert _rows(db3) == 2
    store3.close()
    assert verify_chain(db3) == []

    db4 = tmp_path / "audit-d.sqlite3"
    store4 = AuditStore(db4)
    _v2_start(store4)
    _insert_row_direct(db4, bad_release)
    store4.close()
    problems = verify_chain(db4)
    assert len(problems) == 1 and "release owner/epoch mismatch" in problems[0]


def test_write_decision_uses_recorded_approval_scope_only(tmp_path):
    def start_and_approve(db, target):
        store = AuditStore(db)
        _v2_start(store)
        store.append(
            _v2_event(
                operation="run",
                event_kind="run_approved",
                decision="allow",
                reason_code="allowed",
                counters={},
                lease_owner=OWNER_1,
                lease_epoch=1,
                observed_now="2026-11-12T00:00:05+00:00",
                approved_target_id=target,
            )
        )
        return store

    def write_decision(target, observed="2026-11-12T00:00:10+00:00", **overrides):
        values = dict(
            operation="write",
            event_kind="decision",
            decision="allow",
            reason_code="allowed",
            counters={},
            target_id=target,
            lease_owner=OWNER_1,
            lease_epoch=1,
            observed_now=observed,
            **_v2_policy(),
        )
        values.update(overrides)
        return _v2_event(**values)

    db = tmp_path / "audit.sqlite3"
    store = start_and_approve(db, "write-target-1")
    decision = write_decision("write-target-1")
    store.append(decision)
    store.append(
        _v2_event(
            operation="write",
            event_kind="outcome",
            decision="allow",
            reason_code="allowed",
            counters={},
            write_outcome="committed",
            references_event_id=decision.event_id,
            lease_owner=OWNER_1,
            lease_epoch=1,
            observed_now="2026-11-12T00:00:20+00:00",
        )
    )
    store.close()
    assert verify_chain(db) == []

    db2 = tmp_path / "audit-b.sqlite3"
    store2 = start_and_approve(db2, "write-target-1")
    mismatched = write_decision("write-target-2")
    with pytest.raises(ControlUnavailable):
        store2.append(mismatched)
    assert _rows(db2) == 3
    store2.close()
    assert verify_chain(db2) == []

    db3 = tmp_path / "audit-c.sqlite3"
    store3 = start_and_approve(db3, "write-target-1")
    mismatched_b = write_decision("write-target-2")
    _insert_row_direct(db3, mismatched_b)
    store3.close()
    problems = verify_chain(db3)
    assert len(problems) == 1 and "allowed write lacks earlier approved target" in problems[0]

    db4 = tmp_path / "audit-d.sqlite3"
    store4 = AuditStore(db4)
    _v2_start(store4)
    store4.append(
        write_decision(
            "write-target-9",
            decision="deny",
            reason_code="unapproved_write",
        )
    )
    store4.close()
    assert verify_chain(db4) == []


def test_denied_decisions_carry_no_reservation_fields(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    _v2_start(store)
    store.append(
        _v2_event(
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
            **_v2_policy(),
        )
    )
    store.append(
        _v2_event(
            operation="fetch",
            event_kind="decision",
            decision="deny",
            reason_code="policy_denied",
            counters={},
            target_host="example.com",
            lease_owner=OWNER_1,
            lease_epoch=1,
            observed_now="2026-11-12T00:00:20+00:00",
            **_v2_policy(),
        )
    )
    store.append(
        _v2_event(
            operation="write",
            event_kind="decision",
            decision="deny",
            reason_code="unapproved_write",
            counters={},
            target_id="write-target-1",
            lease_owner=OWNER_1,
            lease_epoch=1,
            observed_now="2026-11-12T00:00:30+00:00",
            **_v2_policy(),
        )
    )
    store.close()
    assert verify_chain(db) == []


def test_outcome_snapshot_equation_breadth(tmp_path):
    def outcome(decision, **overrides):
        values = dict(
            operation="llm",
            event_kind="outcome",
            decision="allow",
            reason_code="allowed",
            counters=_usage_counters(),
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
        values.update(overrides)
        return _v2_event(**values)

    db = tmp_path / "audit-a.sqlite3"
    store = AuditStore(db)
    _v2_start(store)
    decision = _v2_llm_decision(store)
    store.close()
    _insert_row_direct(
        db, outcome(decision, reason_code="budget_breach", counters=_usage_counters())
    )
    problems = verify_chain(db)
    assert len(problems) == 1 and "budget_breach without a negative balance" in problems[0]

    db2 = tmp_path / "audit-b.sqlite3"
    store2 = AuditStore(db2)
    _v2_start(store2)
    decision2 = _v2_llm_decision(store2)
    store2.close()
    _insert_row_direct(
        db2,
        outcome(
            decision2,
            usage_prompt_tokens=19900,
            usage_completion_tokens=200,
            counters={
                "calls": 1,
                "prompt_tokens": 19900,
                "completion_tokens": 200,
                "reserved_tokens": 0,
                "remaining_tokens": 0,
                "budget_deficit": 100,
            },
        ),
    )
    problems = verify_chain(db2)
    assert len(problems) == 1 and "non-breach outcome has a negative balance" in problems[0]

    db3 = tmp_path / "audit-c.sqlite3"
    store3 = AuditStore(db3)
    _v2_start(store3)
    decision3 = _v2_llm_decision(store3)
    store3.close()
    _insert_row_direct(
        db3,
        outcome(
            decision3,
            counters={
                "calls": 1,
                "prompt_tokens": 98,
                "completion_tokens": 40,
                "reserved_tokens": 0,
                "remaining_tokens": 19862,
                "budget_deficit": 5,
            },
        ),
    )
    problems = verify_chain(db3)
    assert len(problems) == 1 and "budget_deficit equation mismatch" in problems[0]

    with pytest.raises(ValueError):
        outcome(
            decision3,
            counters={
                "calls": 1,
                "prompt_tokens": 98,
                "completion_tokens": 40,
                "reserved_tokens": 0,
                "remaining_tokens": 19862,
            },
        )


def test_new_verifier_rejects_2b_constructs_on_a_legacy_payload(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    store.append(_event(target_id="op-1"))
    store.close()
    payload = _event(target_id="op-2").to_payload()
    payload["event_kind"] = "run_started"
    _insert_payload_direct(db, payload)
    problems = verify_chain(db)
    assert len(problems) == 1 and "payload fails the event schema" in problems[0]


def test_v2_kinds_reject_foreign_content_fields():
    binding = dict(lease_owner=OWNER_1, lease_epoch=1, observed_now="2026-11-12T00:00:10+00:00")
    content = ["target_host", "target_id", "content_hash"]
    values = {"target_host": "example.com", "target_id": "t-1", "content_hash": "a" * 64}

    def rejects(message, **kwargs):
        with pytest.raises(ValueError, match=re.escape(message)):
            _v2_event(**kwargs)

    # Positive controls: each base shape constructs fine without the foreign fields.
    base_started = dict(operation="run", event_kind="run_started", decision="allow",
                        reason_code="allowed", counters={})
    _v2_event(**base_started, **_v2_policy())
    base_approved = dict(operation="run", event_kind="run_approved", decision="allow",
                         reason_code="allowed", counters={}, approved_target_id="t-1")
    _v2_event(**base_approved, **binding)
    base_lease = dict(operation="run", decision="allow", reason_code="allowed", counters={},
                      lease_ttl_seconds=300, lease_expires_at="2026-11-12T00:05:00+00:00")
    _v2_event(event_kind="lease_acquired", **base_lease, **binding)
    _v2_event(event_kind="lease_renewed", **base_lease, **binding)
    _v2_event(event_kind="lease_taken_over", previous_lease_owner=OWNER_2,
              previous_lease_epoch=1, references_event_id=str(uuid.uuid4()),
              **base_lease, **binding)
    _v2_event(operation="run", event_kind="lease_released", decision="allow",
              reason_code="allowed", counters={}, lease_ttl_seconds=300, **binding)
    base_llm_outcome = dict(operation="llm", event_kind="outcome", decision="allow",
                            reason_code="allowed", counters=_usage_counters(),
                            model="gpt-4o-mini", schema_name="NewsDigest",
                            usage_prompt_tokens=98, usage_completion_tokens=40,
                            reservation_released=637,
                            references_event_id=str(uuid.uuid4()))
    _v2_event(**base_llm_outcome, **binding)
    _v2_event(operation="fetch", event_kind="outcome", decision="allow",
              reason_code="allowed", counters={}, target_host="example.com",
              result_class="ok", references_event_id=str(uuid.uuid4()), **binding)
    _v2_event(operation="write", event_kind="outcome", decision="allow",
              reason_code="allowed", counters={}, write_outcome="committed",
              references_event_id=str(uuid.uuid4()), **binding)
    # tokenizer_unavailable deny WITHOUT a count stays valid (E6 adds the inverse only).
    _v2_event(operation="llm", event_kind="decision", decision="deny",
              reason_code="tokenizer_unavailable", counters={},
              model="gpt-4o-mini", schema_name="NewsDigest", **binding, **_v2_policy())
    _v2_event(operation="llm", event_kind="outcome", decision="allow",
              reason_code="usage_out_of_range", error_class="UsageOutOfRange",
              counters={}, model="gpt-4o-mini", schema_name="NewsDigest",
              reservation_released=0, usage_missing=True,
              references_event_id=str(uuid.uuid4()), **binding)
    _v2_event(**dict(base_llm_outcome, error_class="UpstreamTimeout"), **binding)
    _v2_event(operation="fetch", event_kind="outcome", decision="allow",
              reason_code="allowed", counters={}, target_host="example.com",
              result_class="ok", error_class="UpstreamTimeout",
              references_event_id=str(uuid.uuid4()), **binding)
    _v2_event(operation="write", event_kind="outcome", decision="allow",
              reason_code="allowed", counters={}, write_outcome="committed",
              error_class="UpstreamTimeout",
              references_event_id=str(uuid.uuid4()), **binding)

    for name in content:
        rejects(f"run_started must not carry {name}",
                **base_started, **{name: values[name]}, **_v2_policy())
    for name in ["references_event_id"] + content:
        extra = {name: str(uuid.uuid4())} if name == "references_event_id" else {name: values[name]}
        rejects(f"run_approved must not carry {name}", **base_approved, **extra, **binding)
    for kind in ("lease_acquired", "lease_renewed"):
        for name in content:
            rejects(f"{kind} must not carry {name}",
                    event_kind=kind, **base_lease, **{name: values[name]}, **binding)
    for name in content:
        rejects(f"lease_taken_over must not carry {name}",
                event_kind="lease_taken_over", previous_lease_owner=OWNER_2,
                previous_lease_epoch=1, references_event_id=str(uuid.uuid4()),
                **base_lease, **{name: values[name]}, **binding)
    for name in ["references_event_id"] + content:
        extra = {name: str(uuid.uuid4())} if name == "references_event_id" else {name: values[name]}
        rejects(f"lease_released must not carry {name}",
                operation="run", event_kind="lease_released", decision="allow",
                reason_code="allowed", counters={}, lease_ttl_seconds=300,
                **extra, **binding)
    for name in content:
        rejects(f"llm outcome must not carry {name}", **base_llm_outcome,
                **{name: values[name]}, **binding)
    for name in ("target_id", "content_hash"):
        rejects(f"fetch outcome must not carry {name}",
                operation="fetch", event_kind="outcome", decision="allow",
                reason_code="allowed", counters={}, target_host="example.com",
                result_class="ok", references_event_id=str(uuid.uuid4()),
                **{name: values[name]}, **binding)
    for name in content:
        rejects(f"write outcome must not carry {name}",
                operation="write", event_kind="outcome", decision="allow",
                reason_code="allowed", counters={}, write_outcome="committed",
                references_event_id=str(uuid.uuid4()),
                **{name: values[name]}, **binding)
    rejects("tokenizer_unavailable denial carries no serialized count",
            operation="llm", event_kind="decision", decision="deny",
            reason_code="tokenizer_unavailable", counters={},
            model="gpt-4o-mini", schema_name="NewsDigest",
            serialized_body_token_count=100, **binding, **_v2_policy())


def test_verifier_rejects_tampered_payloads_with_foreign_fields(tmp_path):
    db = tmp_path / "audit.sqlite3"
    store = AuditStore(db)
    store.append(
        _v2_event(operation="run", event_kind="run_started", decision="allow",
                  reason_code="allowed", counters={}, **_v2_policy())
    )
    store.close()
    _tamper_row_payload(db, 1, target_id="t-1")
    problems = verify_chain(db)
    assert len(problems) == 1 and "payload fails the event schema" in problems[0]

    db2 = tmp_path / "audit-b.sqlite3"
    store2 = AuditStore(db2)
    _v2_start(store2)
    decision = _v2_llm_decision(store2)
    _v2_outcome(store2, decision)
    store2.close()
    _tamper_row_payload(db2, 4, content_hash="a" * 64)
    problems = verify_chain(db2)
    assert len(problems) == 1 and "payload fails the event schema" in problems[0]
