"""Append-only audit foundation for the H1 control layer (Slice 1, repaired).

No admission, policy, reservation, tokenizer or controller-wiring logic lives
here. This module provides the bounded audit-event model, the append-only SQLite
store and a read-only whole-chain verifier.
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

POLICY_VERSION = "h1-slice1"
MAX_TEXT_LEN = 128
MAX_POLICY_LEN = 64
MAX_HOST_LEN = 253

OPERATIONS = frozenset({"fetch", "llm", "write"})
EVENT_KINDS = frozenset({"decision", "outcome"})
DECISIONS = frozenset({"allow", "deny"})
WRITE_OUTCOMES = frozenset({"committed", "failed", "unknown"})
COUNTER_KEYS = frozenset(
    {
        "calls",
        "prompt_tokens",
        "completion_tokens",
        "reserved_tokens",
        "remaining_tokens",
    }
)
APPEND_STATES = frozenset({"known_rolled_back", "unknown"})

_HOST_PATTERN = re.compile(
    r"^(?=.{1,253}$)"
    r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)*"
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$"
)
_ERROR_CLASS_PATTERN = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$"
)

_APPEND_LOCK = threading.RLock()
_PAYLOAD_EXTRA_KEYS = frozenset({"prev_hash", "retry_safe"})


class ReasonCode(str, Enum):
    """Fixed, stable reason codes; audit events never carry free-text reasons."""

    ALLOWED = "allowed"
    POLICY_DENIED = "policy_denied"
    UNAPPROVED_WRITE = "unapproved_write"
    INVALID_TARGET = "invalid_target"
    AUDIT_UNAVAILABLE = "audit_unavailable"
    CHAIN_BROKEN = "chain_broken"
    RECONCILIATION = "reconciliation"


REASON_CODES = frozenset(item.value for item in ReasonCode)


class ControlUnavailable(RuntimeError):
    """The audit store could not complete an append; the caller must not act.

    ``append_state`` is ``known_rolled_back`` only when the store can prove the
    transaction did not commit; a commit-stage failure is ``unknown`` and must
    never be automatically resubmitted.
    """

    def __init__(
        self,
        message: str,
        *,
        append_state: str,
        reason_code: str | ReasonCode = ReasonCode.AUDIT_UNAVAILABLE,
    ):
        if append_state not in APPEND_STATES:
            raise ValueError(f"unknown append_state {append_state!r}")
        reason = reason_code.value if isinstance(reason_code, ReasonCode) else reason_code
        if reason not in REASON_CODES:
            raise ValueError(f"unknown reason_code {reason_code!r}")
        self.append_state = append_state
        self.reason_code = reason
        super().__init__(message)


class AuditIncomplete(RuntimeError):
    """An audit outcome is incomplete or inconsistent with the stored chain."""


class _BrokenChain(Exception):
    """Internal signal: the existing chain failed verification."""

    def __init__(self, problems: list[str]):
        self.problems = tuple(problems)
        super().__init__("; ".join(problems))


def canonical_json(value: Any) -> str:
    """Return deterministic JSON that rejects NaN/Infinity and non-UTF-8 text."""
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        text.encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ValueError(f"value cannot be represented as exact JSON: {exc}") from exc
    return text


def _text(value: object, name: str, *, max_length: int = MAX_TEXT_LEN) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty trimmed string")
    if len(value) > max_length:
        raise ValueError(f"{name} exceeds {max_length} characters")
    if any(ord(char) < 32 for char in value):
        raise ValueError(f"{name} must not contain control characters")
    return value


def _timestamp(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("recorded_at must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("recorded_at must be an ISO-8601 string") from exc
    if parsed.tzinfo is None:
        raise ValueError("recorded_at must carry a timezone")
    return parsed.astimezone(timezone.utc).isoformat()


def _uuid(value: object, name: str) -> str:
    text = _text(value, name)
    try:
        return str(uuid.UUID(text))
    except ValueError as exc:
        raise ValueError(f"{name} must be a UUID") from exc


def _optional_text(
    value: object, name: str, *, max_length: int = MAX_TEXT_LEN
) -> str | None:
    if value is None:
        return None
    return _text(value, name, max_length=max_length)


def _normalized_host(value: object) -> str | None:
    if value is None:
        return None
    host = _text(value, "target_host", max_length=MAX_HOST_LEN).lower()
    if not (_HOST_PATTERN.fullmatch(host) or _is_ipv4(host)):
        raise ValueError(
            "target_host must be a DNS name or IPv4 literal, without a port"
        )
    return host


def _is_ipv4(value: str) -> bool:
    try:
        ipaddress.IPv4Address(value)
    except ValueError:
        return False
    return True


def _content_hash(value: object) -> str | None:
    if value is None:
        return None
    text = _text(value, "content_hash")
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text.lower()):
        raise ValueError("content_hash must be a 64-character hexadecimal SHA-256")
    return text.lower()


def _error_class(value: object) -> str | None:
    if value is None:
        return None
    text = _text(value, "error_class")
    if not _ERROR_CLASS_PATTERN.fullmatch(text):
        raise ValueError("error_class must be a dotted exception class name")
    return text


@dataclass(frozen=True)
class AuditEvent:
    """One validated audit event.

    The dataclass constructor is the field allowlist: unknown keyword arguments
    raise ``TypeError`` and every listed field is validated in ``__post_init__``.
    The payload also stores the derived ``retry_safe`` flag and ``prev_hash``.
    """

    event_id: str
    recorded_at: str
    run_id: str
    thread_id: str
    operation: str
    event_kind: str
    decision: str
    reason_code: str
    policy_version: str
    counters: Mapping[str, int] = field(default_factory=dict)
    model: str | None = None
    schema_name: str | None = None
    error_class: str | None = None
    usage_missing: bool = False
    target_host: str | None = None
    target_id: str | None = None
    content_hash: str | None = None
    write_outcome: str | None = None
    references_event_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _uuid(self.event_id, "event_id"))
        object.__setattr__(self, "recorded_at", _timestamp(self.recorded_at))
        object.__setattr__(self, "run_id", _text(self.run_id, "run_id"))
        object.__setattr__(self, "thread_id", _text(self.thread_id, "thread_id"))
        object.__setattr__(
            self,
            "policy_version",
            _text(self.policy_version, "policy_version", max_length=MAX_POLICY_LEN),
        )

        if self.operation not in OPERATIONS:
            raise ValueError(f"unknown operation {self.operation!r}")
        if self.event_kind not in EVENT_KINDS:
            raise ValueError(f"unknown event_kind {self.event_kind!r}")
        if self.decision not in DECISIONS:
            raise ValueError(f"unknown decision {self.decision!r}")

        reason = (
            self.reason_code.value
            if isinstance(self.reason_code, ReasonCode)
            else self.reason_code
        )
        if reason not in REASON_CODES:
            raise ValueError(f"unknown reason_code {self.reason_code!r}")
        object.__setattr__(self, "reason_code", reason)

        if not isinstance(self.counters, Mapping):
            raise ValueError("counters must be a mapping")
        clean_counters: dict[str, int] = {}
        for key, value in self.counters.items():
            if key not in COUNTER_KEYS:
                raise ValueError(f"unknown counter {key!r}")
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"counter {key!r} must be a non-negative integer")
            clean_counters[key] = value
        object.__setattr__(
            self, "counters", MappingProxyType(dict(sorted(clean_counters.items())))
        )

        model = _optional_text(self.model, "model")
        schema_name = _optional_text(self.schema_name, "schema_name")
        if self.operation == "llm":
            if model is None or schema_name is None:
                raise ValueError("LLM events require model and schema_name")
        elif model is not None or schema_name is not None:
            raise ValueError("model/schema_name are only valid for llm events")

        if not isinstance(self.usage_missing, bool):
            raise ValueError("usage_missing must be a boolean")
        if self.usage_missing and not (
            self.operation == "llm" and self.event_kind == "outcome"
        ):
            raise ValueError("usage_missing is only valid for llm outcomes")

        object.__setattr__(self, "target_host", _normalized_host(self.target_host))
        object.__setattr__(
            self, "target_id", _optional_text(self.target_id, "target_id")
        )
        object.__setattr__(self, "content_hash", _content_hash(self.content_hash))
        object.__setattr__(self, "error_class", _error_class(self.error_class))
        if self.error_class is not None and self.event_kind != "outcome":
            raise ValueError("error_class is only valid for outcome events")

        if self.write_outcome is None:
            if self.operation == "write" and self.event_kind == "outcome":
                raise ValueError("write outcomes require write_outcome")
        else:
            if self.operation != "write" or self.event_kind != "outcome":
                raise ValueError("write_outcome is only valid for write outcomes")
            if self.write_outcome not in WRITE_OUTCOMES:
                raise ValueError(f"unknown write_outcome {self.write_outcome!r}")

        if self.references_event_id is not None:
            object.__setattr__(
                self,
                "references_event_id",
                _uuid(self.references_event_id, "references_event_id"),
            )

    @property
    def retry_safe(self) -> bool | None:
        """Whether a write outcome is safe to retry.

        ``failed`` means a confirmed no-effect outcome (the transaction failed
        and rollback was confirmed), so it is the only retry-safe write result.
        ``committed`` and ``unknown`` return ``False``; non-write events return
        ``None``. The derived value is persisted in the payload and revalidated
        by the verifier.
        """
        if self.write_outcome is None:
            return None
        return self.write_outcome == "failed"

    def to_payload(self) -> dict[str, Any]:
        """Return only the bounded, JSON-safe event fields plus derived fields."""
        payload: dict[str, Any] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, Enum):
                value = value.value
            if item.name == "counters":
                value = dict(sorted(value.items()))
            payload[item.name] = value
        payload["retry_safe"] = self.retry_safe
        return payload


def make_event(
    *,
    operation: str,
    event_kind: str,
    decision: str,
    reason_code: str | ReasonCode,
    run_id: str,
    thread_id: str,
    recorded_at: str | None = None,
    event_id: str | None = None,
    counters: Mapping[str, int] | None = None,
    model: str | None = None,
    schema_name: str | None = None,
    error_class: str | None = None,
    usage_missing: bool = False,
    target_host: str | None = None,
    target_id: str | None = None,
    content_hash: str | None = None,
    write_outcome: str | None = None,
    references_event_id: str | None = None,
    policy_version: str = POLICY_VERSION,
) -> AuditEvent:
    """Construct an event with a generated id and UTC timestamp by default."""
    return AuditEvent(
        event_id=event_id or str(uuid.uuid4()),
        recorded_at=recorded_at or datetime.now(timezone.utc).isoformat(),
        run_id=run_id,
        thread_id=thread_id,
        operation=operation,
        event_kind=event_kind,
        decision=decision,
        reason_code=reason_code,
        policy_version=policy_version,
        counters=counters or {},
        model=model,
        schema_name=schema_name,
        error_class=error_class,
        usage_missing=usage_missing,
        target_host=target_host,
        target_id=target_id,
        content_hash=content_hash,
        write_outcome=write_outcome,
        references_event_id=references_event_id,
    )


_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_events (
    seq        INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id   TEXT NOT NULL UNIQUE,
    prev_hash  TEXT,
    event_hash TEXT NOT NULL UNIQUE,
    payload    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_head (
    singleton  INTEGER PRIMARY KEY CHECK (singleton = 1),
    seq        INTEGER NOT NULL,
    event_hash TEXT NOT NULL
);
"""


def _event_hash(payload: dict[str, Any], prev_hash: str | None) -> str:
    body = canonical_json({**payload, "prev_hash": prev_hash})
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _payload_event(
    parsed: Mapping[str, Any], label: str, problems: list[str]
) -> AuditEvent | None:
    """Reconstruct a validated event from a stored payload (without metadata)."""
    allowed = {item.name for item in fields(AuditEvent)} | _PAYLOAD_EXTRA_KEYS
    extra = set(parsed) - allowed
    if extra:
        problems.append(f"{label}: payload has unknown fields")
        return None
    event_fields = {
        key: value
        for key, value in parsed.items()
        if key not in _PAYLOAD_EXTRA_KEYS
    }
    try:
        event = AuditEvent(**event_fields)
    except (TypeError, ValueError):
        problems.append(f"{label}: payload fails the event schema")
        return None
    return event


def _chain_problems(
    rows: list[tuple[int, str, str | None, str, str]],
    head: tuple[int, str] | None,
    *,
    allow_empty: bool,
) -> list[str]:
    """Verify every stored row and the recorded head; returns problems."""
    problems: list[str] = []
    if not rows:
        if head is not None:
            problems.append("recorded head exists without events")
        elif not allow_empty:
            problems.append("audit chain is empty")
        return problems

    seen_ids: set[str] = set()
    previous_hash: str | None = None
    previous_seq: int | None = None
    for index, (seq, event_id, prev_hash, event_hash, payload) in enumerate(rows):
        label = f"seq {seq}"
        if event_id in seen_ids:
            problems.append(f"{label}: duplicate event_id")
        seen_ids.add(event_id)
        if previous_seq is not None and seq != previous_seq + 1:
            problems.append(f"{label}: sequence gap")
        previous_seq = seq

        if index == 0:
            if prev_hash is not None:
                problems.append(f"{label}: genesis event has a prev_hash")
        elif prev_hash != previous_hash:
            problems.append(f"{label}: broken prev_hash linkage")

        try:
            parsed = json.loads(payload)
        except (TypeError, ValueError):
            problems.append(f"{label}: payload is not valid JSON")
            previous_hash = event_hash
            continue
        if not isinstance(parsed, dict):
            problems.append(f"{label}: payload is not a JSON object")
            previous_hash = event_hash
            continue

        if parsed.get("event_id") != event_id:
            problems.append(f"{label}: payload event_id does not match the row")
        if "prev_hash" not in parsed:
            problems.append(f"{label}: payload is missing prev_hash")
        if parsed.get("prev_hash") != prev_hash:
            problems.append(f"{label}: payload prev_hash does not match the row")
        if "retry_safe" not in parsed:
            problems.append(f"{label}: payload is missing retry_safe")
        else:
            stored_retry_safe = parsed["retry_safe"]
            if stored_retry_safe is not None and type(stored_retry_safe) is not bool:
                problems.append(
                    f"{label}: retry_safe must be a JSON boolean or null"
                )

        event = _payload_event(parsed, label, problems)
        if event is not None:
            expected_retry = event.retry_safe
            if parsed.get("retry_safe") != expected_retry:
                problems.append(
                    f"{label}: payload retry_safe does not match its write outcome"
                )
            rebuilt = event.to_payload()
            rebuilt["prev_hash"] = prev_hash
            try:
                rebuilt_payload = canonical_json(rebuilt)
            except ValueError:
                problems.append(
                    f"{label}: reconstructed payload is not canonical JSON"
                )
            else:
                if rebuilt_payload != payload:
                    problems.append(
                        f"{label}: payload is not the exact canonical form of "
                        "the reconstructed event"
                    )

        try:
            canonical = canonical_json(parsed)
        except ValueError:
            problems.append(f"{label}: payload is not canonical JSON")
        else:
            if canonical != payload:
                problems.append(f"{label}: payload text is not canonical JSON")

        try:
            recomputed = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        except UnicodeEncodeError:
            problems.append(f"{label}: payload is not UTF-8")
            recomputed = ""
        if recomputed != event_hash:
            problems.append(f"{label}: event_hash does not match payload")

        previous_hash = event_hash

    if head is None:
        problems.append("recorded head is missing")
    elif head != (rows[-1][0], rows[-1][3]):
        problems.append("recorded head does not match the last event")
    return problems


def _rollback(conn) -> bool:
    try:
        conn.execute("ROLLBACK")
        return True
    except Exception:
        return False


class AuditStore:
    """Single-writer append-only SQLite audit store.

    The store owns one connection. ``append`` holds a process-wide lock and a
    SQLite ``BEGIN IMMEDIATE`` transaction while it verifies the existing chain,
    reads the head and inserts the new event. There is no update or delete
    method.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._conn = sqlite3.connect(
            str(self.path), isolation_level=None, check_same_thread=False
        )
        try:
            self._conn.execute("PRAGMA journal_mode=DELETE")
            self._conn.execute("PRAGMA synchronous=FULL")
            self._conn.executescript(_SCHEMA)
        except Exception as exc:
            self._conn.close()
            raise ControlUnavailable(
                "could not initialize the audit store",
                append_state="known_rolled_back",
            ) from exc

    def close(self) -> None:
        self._conn.close()

    def append(self, event: AuditEvent) -> str:
        """Append one event and return its stored ``event_hash``.

        The existing chain is re-verified inside the same transaction before the
        new event is added. A corrupted chain is refused with
        ``ReasonCode.CHAIN_BROKEN``. A failure before COMMIT is reported as
        ``known_rolled_back`` when rollback is confirmed; a COMMIT-stage failure
        is reported as ``unknown`` and must not be automatically resubmitted.
        """
        if not isinstance(event, AuditEvent):
            raise TypeError("append requires an AuditEvent")
        with _APPEND_LOCK:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
            except Exception as exc:
                raise ControlUnavailable(
                    "could not start the audit transaction",
                    append_state="known_rolled_back",
                ) from exc

            try:
                rows = self._conn.execute(
                    "SELECT seq, event_id, prev_hash, event_hash, payload "
                    "FROM audit_events ORDER BY seq"
                ).fetchall()
                head = self._conn.execute(
                    "SELECT seq, event_hash FROM audit_head WHERE singleton = 1"
                ).fetchone()
                problems = _chain_problems(rows, head, allow_empty=True)
                if problems:
                    raise _BrokenChain(problems)

                prev_hash = rows[-1][3] if rows else None
                payload = event.to_payload()
                digest = _event_hash(payload, prev_hash)
                self._conn.execute(
                    "INSERT INTO audit_events "
                    "(event_id, prev_hash, event_hash, payload) VALUES (?, ?, ?, ?)",
                    (
                        event.event_id,
                        prev_hash,
                        digest,
                        canonical_json({**payload, "prev_hash": prev_hash}),
                    ),
                )
                seq = self._conn.execute(
                    "SELECT last_insert_rowid()"
                ).fetchone()[0]
                self._conn.execute(
                    "INSERT INTO audit_head (singleton, seq, event_hash) "
                    "VALUES (1, ?, ?) "
                    "ON CONFLICT(singleton) DO UPDATE SET "
                    "seq = excluded.seq, event_hash = excluded.event_hash",
                    (seq, digest),
                )
            except _BrokenChain as exc:
                rolled_back = _rollback(self._conn)
                raise ControlUnavailable(
                    "refusing to extend a broken audit chain: "
                    + "; ".join(exc.problems),
                    append_state="known_rolled_back" if rolled_back else "unknown",
                    reason_code=ReasonCode.CHAIN_BROKEN,
                ) from exc
            except Exception as exc:
                rolled_back = _rollback(self._conn)
                raise ControlUnavailable(
                    "audit append failed before commit",
                    append_state="known_rolled_back" if rolled_back else "unknown",
                ) from exc

            try:
                self._conn.execute("COMMIT")
            except Exception as exc:
                # The COMMIT may or may not have been applied server-side; do
                # not claim a clean rollback and do not auto-resubmit.
                _rollback(self._conn)
                raise ControlUnavailable(
                    "audit commit outcome is unknown",
                    append_state="unknown",
                ) from exc
            return digest


def verify_chain(path: str | Path) -> list[str]:
    """Read-only verification of the whole audit chain.

    Returns a list of human-readable problems; an empty list means the saved
    chain is internally consistent. The file must already exist: this function
    never creates it and performs only SELECTs.
    """
    target = Path(path)
    if not target.exists():
        return [f"audit database does not exist: {target.name}"]

    uri = target.resolve().as_uri() + "?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as exc:
        return [f"cannot open audit database read-only: {type(exc).__name__}"]

    try:
        try:
            # One read transaction on this connection: events and head are read
            # from the same consistent snapshot.
            conn.execute("BEGIN")
            rows = conn.execute(
                "SELECT seq, event_id, prev_hash, event_hash, payload "
                "FROM audit_events ORDER BY seq"
            ).fetchall()
            head = conn.execute(
                "SELECT seq, event_hash FROM audit_head WHERE singleton = 1"
            ).fetchone()
            conn.execute("COMMIT")
        except sqlite3.Error as exc:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            return [f"cannot read audit chain: {type(exc).__name__}"]
    finally:
        conn.close()

    return _chain_problems(rows, head, allow_empty=False)
