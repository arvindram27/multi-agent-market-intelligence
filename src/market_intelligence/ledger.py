"""Pure ledger replay and read-only verifier for H1 slice 2b-ii-A.

No production path in this module writes a database. The SQL constants are
executed only by tests. ``next_state`` is pure and never reads a clock or calls
admission decide_* functions.

Limits: the chain does not store the request's max_output_tokens, tokenizer
state, fetch request limits, write content identity, or the cause of a denied
decision. There is no external anchor for the audit hash chain. ``verify_ledger``
is O(n) in chain length. A database with an audit chain and no ledger tables is
reported, never repaired. No global clock monotonic rule is added here.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import market_intelligence.control as control

INT64_MAX = 2**63 - 1

RUN_POLICY_SQL = """
CREATE TABLE IF NOT EXISTS run_policy (
    run_id                  TEXT PRIMARY KEY,
    thread_id               TEXT NOT NULL,
    policy_version          TEXT NOT NULL,
    estimate_kind           TEXT NOT NULL,
    margin_num              TEXT NOT NULL,
    margin_den              TEXT NOT NULL,
    provider_overhead_tokens INTEGER NOT NULL,
    max_run_tokens          INTEGER NOT NULL,
    max_run_llm_calls       INTEGER NOT NULL,
    max_output_tokens       INTEGER NOT NULL,
    max_fetches_per_run     INTEGER NOT NULL,
    lease_ttl_seconds       INTEGER NOT NULL,
    allowed_model           TEXT NOT NULL,
    allowed_schemas_json    TEXT NOT NULL
);
"""

RUN_LEDGER_SQL = """
CREATE TABLE IF NOT EXISTS run_ledger (
    run_id              TEXT PRIMARY KEY,
    calls               INTEGER NOT NULL,
    prompt_tokens       INTEGER NOT NULL,
    completion_tokens   INTEGER NOT NULL,
    reserved_tokens     INTEGER NOT NULL,
    fetch_count         INTEGER NOT NULL
);
"""

RESERVATIONS_SQL = """
CREATE TABLE IF NOT EXISTS reservations (
    run_id              TEXT NOT NULL,
    decision_event_id   TEXT PRIMARY KEY,
    reserved_tokens     INTEGER NOT NULL,
    status              TEXT NOT NULL,
    opened_at           TEXT NOT NULL,
    closed_at           TEXT,
    last_event_id       TEXT
);
"""

RUN_LEASES_SQL = """
CREATE TABLE IF NOT EXISTS run_leases (
    run_id              TEXT PRIMARY KEY,
    owner_id            TEXT NOT NULL,
    epoch               INTEGER NOT NULL,
    expires_at          TEXT NOT NULL,
    released            INTEGER NOT NULL,
    released_at         TEXT,
    last_event_id       TEXT
);
"""

LEDGER_HEAD_SQL = """
CREATE TABLE IF NOT EXISTS ledger_head (
    singleton  INTEGER PRIMARY KEY CHECK (singleton = 1),
    seq        INTEGER NOT NULL,
    event_hash TEXT
);
"""

SCHEMA_SQL = (
    RUN_POLICY_SQL
    + RUN_LEDGER_SQL
    + RESERVATIONS_SQL
    + RUN_LEASES_SQL
    + LEDGER_HEAD_SQL
)

COLUMNS = {
    "run_policy": (
        "run_id",
        "thread_id",
        "policy_version",
        "estimate_kind",
        "margin_num",
        "margin_den",
        "provider_overhead_tokens",
        "max_run_tokens",
        "max_run_llm_calls",
        "max_output_tokens",
        "max_fetches_per_run",
        "lease_ttl_seconds",
        "allowed_model",
        "allowed_schemas_json",
    ),
    "run_ledger": (
        "run_id",
        "calls",
        "prompt_tokens",
        "completion_tokens",
        "reserved_tokens",
        "fetch_count",
    ),
    "reservations": (
        "run_id",
        "decision_event_id",
        "reserved_tokens",
        "status",
        "opened_at",
        "closed_at",
        "last_event_id",
    ),
    "run_leases": (
        "run_id",
        "owner_id",
        "epoch",
        "expires_at",
        "released",
        "released_at",
        "last_event_id",
    ),
    "ledger_head": ("singleton", "seq", "event_hash"),
}

TABLE_KEYS = {
    "run_policy": "run_id",
    "run_ledger": "run_id",
    "reservations": "decision_event_id",
    "run_leases": "run_id",
    "ledger_head": "singleton",
}


class _BusyDatabase(Exception):
    """Internal signal for a locked/busy SQLite read."""


class LedgerError(ValueError):
    """Base class for deterministic ledger rule failures."""


class LedgerInvalidEvent(LedgerError):
    """The event cannot be used by the ledger."""


class LedgerInvalidTransition(LedgerError):
    """The event is not valid in the current ledger state."""


class LedgerIncompleteReconciliation(LedgerError):
    """A reconciliation did not supply enough evidence to settle."""


class LedgerOverflow(LedgerError):
    """A ledger integer would leave the signed 64-bit domain."""


def _checked(value: int, label: str) -> int:
    if type(value) is not int or value < 0 or value > INT64_MAX:
        raise LedgerOverflow(f"{label} is outside the checked int64 domain")
    return value


def _add(left: int, right: int, label: str) -> int:
    return _checked(left + right, label)


def _sub(left: int, right: int, label: str) -> int:
    value = left - right
    if type(value) is not int or value < 0:
        raise LedgerInvalidTransition(f"{label} would become negative")
    return _checked(value, label)


def _reserve(policy: "PolicySnapshot", token_count: int) -> int:
    scaled = (
        (token_count + policy.provider_overhead_tokens)
        * (policy.margin_num + policy.margin_den)
        + policy.margin_den
        - 1
    ) // policy.margin_den
    return _checked(scaled + policy.max_output_tokens, "reserve_tokens")


@dataclass(frozen=True)
class PolicySnapshot:
    margin_num: int
    margin_den: int
    provider_overhead_tokens: int
    max_run_tokens: int
    max_run_llm_calls: int
    max_output_tokens: int
    max_fetches_per_run: int
    lease_ttl_seconds: int
    allowed_model: str
    allowed_schemas: tuple[str, ...]
    estimate_kind: str
    policy_version: str


@dataclass(frozen=True)
class Reservation:
    decision_event_id: str
    run_id: str
    reserved_tokens: int
    status: str
    opened_at: str
    closed_at: str | None
    last_event_id: str | None


@dataclass(frozen=True)
class DecisionRecord:
    event_id: str
    run_id: str
    operation: str
    allow: bool
    reserve_tokens: int | None
    model: str | None
    schema_name: str | None
    target_host: str | None
    target_id: str | None
    observed_now: str


@dataclass(frozen=True)
class LeaseRecord:
    owner_id: str
    epoch: int
    expires_at: str
    released: bool
    released_at: str | None
    last_event_id: str


@dataclass(frozen=True)
class RunState:
    run_id: str
    thread_id: str
    policy: PolicySnapshot
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reserved_tokens: int = 0
    fetch_count: int = 0
    reservations: Mapping[str, Reservation] = None  # type: ignore[assignment]
    decisions: Mapping[str, DecisionRecord] = None  # type: ignore[assignment]
    normal_outcomes: frozenset[str] = frozenset()
    unknown_outcomes: Mapping[str, str] = None  # type: ignore[assignment]
    reconciled: frozenset[str] = frozenset()
    approved_target: str | None = None
    lease: LeaseRecord | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "reservations", MappingProxyType(dict(self.reservations or {}))
        )
        object.__setattr__(
            self, "decisions", MappingProxyType(dict(self.decisions or {}))
        )
        object.__setattr__(
            self,
            "unknown_outcomes",
            MappingProxyType(dict(self.unknown_outcomes or {})),
        )
        object.__setattr__(self, "normal_outcomes", frozenset(self.normal_outcomes))


@dataclass(frozen=True)
class LedgerState:
    runs: Mapping[str, RunState] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(self, "runs", MappingProxyType(dict(self.runs or {})))


def _policy_from_event(event: control.AuditEvent) -> PolicySnapshot:
    values = (
        event.policy_margin_num,
        event.policy_margin_den,
        event.provider_overhead_tokens,
        event.max_run_tokens,
        event.max_run_llm_calls,
        event.max_output_tokens,
        event.max_fetches_per_run,
        event.lease_ttl_seconds,
        event.allowed_model,
        event.allowed_schemas,
        event.estimate_kind,
        event.policy_version,
    )
    if any(value is None for value in values):
        raise LedgerInvalidEvent("run_started policy snapshot is incomplete")
    return PolicySnapshot(
        margin_num=event.policy_margin_num,
        margin_den=event.policy_margin_den,
        provider_overhead_tokens=event.provider_overhead_tokens,
        max_run_tokens=event.max_run_tokens,
        max_run_llm_calls=event.max_run_llm_calls,
        max_output_tokens=event.max_output_tokens,
        max_fetches_per_run=event.max_fetches_per_run,
        lease_ttl_seconds=event.lease_ttl_seconds,
        allowed_model=event.allowed_model,
        allowed_schemas=tuple(event.allowed_schemas),
        estimate_kind=event.estimate_kind,
        policy_version=event.policy_version,
    )


def _balance(run: RunState) -> int:
    return (
        run.policy.max_run_tokens
        - run.prompt_tokens
        - run.completion_tokens
        - run.reserved_tokens
    )


def _counter_snapshot(run: RunState) -> dict[str, int]:
    balance = _balance(run)
    return {
        "calls": run.calls,
        "prompt_tokens": run.prompt_tokens,
        "completion_tokens": run.completion_tokens,
        "reserved_tokens": run.reserved_tokens,
        "remaining_tokens": max(0, balance),
        "budget_deficit": max(0, -balance),
    }


def _replace_run(state: LedgerState, run: RunState) -> LedgerState:
    runs = dict(state.runs)
    runs[run.run_id] = run
    return LedgerState(runs=runs)


def _require_run(state: LedgerState, run_id: str) -> RunState:
    try:
        return state.runs[run_id]
    except KeyError as exc:
        raise LedgerInvalidTransition(f"unknown run {run_id!r}") from exc


def _require_decision(run: RunState, event_id: str) -> DecisionRecord:
    try:
        return run.decisions[event_id]
    except KeyError as exc:
        raise LedgerInvalidTransition(f"unknown decision {event_id!r}") from exc


def _add_reservation(run: RunState, reservation: Reservation) -> RunState:
    reservations = dict(run.reservations)
    reservations[reservation.decision_event_id] = reservation
    return replace(run, reservations=reservations)


def _replace_reservation(run: RunState, reservation: Reservation) -> RunState:
    reservations = dict(run.reservations)
    reservations[reservation.decision_event_id] = reservation
    return replace(run, reservations=reservations)


def _transition(
    state: LedgerState, event: control.AuditEvent, *, check_counters: bool
) -> LedgerState:
    """Single rule engine used by both next_state and expected_counters."""
    if event.policy_version != control.POLICY_VERSION_H1_SLICE2:
        return state

    if event.event_kind == "run_started":
        if event.run_id in state.runs:
            raise LedgerInvalidTransition("duplicate run_started for ledger run")
        policy = _policy_from_event(event)
        run = RunState(run_id=event.run_id, thread_id=event.thread_id, policy=policy)
        return _replace_run(state, run)

    run = _require_run(state, event.run_id)

    if event.event_kind == "run_approved":
        if run.approved_target is not None:
            raise LedgerInvalidTransition("second approval in ledger state")
        return _replace_run(state, replace(run, approved_target=event.approved_target_id))

    if event.event_kind in {"lease_acquired", "lease_renewed", "lease_taken_over", "lease_released"}:
        if event.event_kind == "lease_released":
            if run.lease is None or run.lease.released:
                raise LedgerInvalidTransition("release without a live lease in ledger state")
            lease = replace(
                run.lease,
                released=True,
                released_at=event.observed_now,
                last_event_id=event.event_id,
            )
            return _replace_run(state, replace(run, lease=lease))
        lease = LeaseRecord(
            owner_id=event.lease_owner,
            epoch=event.lease_epoch,
            expires_at=event.lease_expires_at,
            released=False,
            released_at=None,
            last_event_id=event.event_id,
        )
        return _replace_run(state, replace(run, lease=lease))

    if event.event_kind == "decision":
        decision = DecisionRecord(
            event_id=event.event_id,
            run_id=event.run_id,
            operation=event.operation,
            allow=event.decision == "allow",
            reserve_tokens=event.reserve_tokens,
            model=event.model,
            schema_name=event.schema_name,
            target_host=event.target_host,
            target_id=event.target_id,
            observed_now=event.observed_now,
        )
        decisions = dict(run.decisions)
        decisions[event.event_id] = decision
        run = replace(run, decisions=decisions)
        if not decision.allow:
            return _replace_run(state, run)

        if event.operation == "llm":
            if event.serialized_body_token_count is None:
                raise LedgerInvalidTransition("allowed llm decision lacks a serialized count")
            if event.model != run.policy.allowed_model:
                raise LedgerInvalidEvent("allowed llm decision model is not allowed")
            if event.schema_name not in run.policy.allowed_schemas:
                raise LedgerInvalidEvent("allowed llm decision schema is not allowed")
            expected_reserve = _reserve(run.policy, event.serialized_body_token_count)
            if event.reserve_tokens != expected_reserve:
                raise LedgerInvalidEvent("decision reserve does not match the policy formula")
            if _add(run.calls, 1, "calls") > run.policy.max_run_llm_calls:
                raise LedgerInvalidTransition("llm call cap would be exceeded")
            candidate = replace(
                run,
                calls=_add(run.calls, 1, "calls"),
                reserved_tokens=_add(run.reserved_tokens, event.reserve_tokens, "reserved_tokens"),
            )
            if _balance(candidate) < 0:
                raise LedgerInvalidTransition("token cap would be exceeded")
            run = candidate
            reservation = Reservation(
                decision_event_id=event.event_id,
                run_id=event.run_id,
                reserved_tokens=event.reserve_tokens,
                status="open",
                opened_at=event.observed_now,
                closed_at=None,
                last_event_id=None,
            )
            run = _add_reservation(run, reservation)
            return _replace_run(state, run)

        if event.operation == "fetch":
            if _add(run.fetch_count, 1, "fetch_count") > run.policy.max_fetches_per_run:
                raise LedgerInvalidTransition("fetch cap would be exceeded")
            return _replace_run(
                state, replace(run, fetch_count=_add(run.fetch_count, 1, "fetch_count"))
            )

        return _replace_run(state, run)

    if event.event_kind == "outcome":
        ref = event.references_event_id
        if ref is None:
            raise LedgerInvalidTransition("outcome lacks a reference")

        if event.reason_code == "reconciliation":
            if (event.usage_prompt_tokens is None) != (event.usage_completion_tokens is None):
                raise LedgerIncompleteReconciliation(
                    "reconciliation with one usage figure does not settle"
                )
            if event.usage_prompt_tokens is None and event.usage_completion_tokens is None:
                raise LedgerIncompleteReconciliation("reconciliation without usage does not settle")
            try:
                decision_id = run.unknown_outcomes[ref]
            except KeyError as exc:
                raise LedgerInvalidTransition("reconciliation does not reference a known unknown outcome") from exc
            decision = _require_decision(run, decision_id)
            if not decision.allow or decision.operation != event.operation:
                raise LedgerInvalidTransition("reconciliation decision is not allowed/same operation")
            reservation = run.reservations.get(decision_id)
            if reservation is None or reservation.status != "unknown":
                raise LedgerInvalidTransition("reconciliation reservation is not unknown")
            if ref in run.reconciled:
                raise LedgerInvalidTransition("second reconciliation for the same unknown")
            return _apply_llm_usage(
                state,
                run,
                decision,
                reservation,
                event,
                target_status="resolved",
                allow_negative=True,
                require_overrun_absent=True,
                check_counters=check_counters,
            )

        decision = _require_decision(run, ref)
        if not decision.allow:
            raise LedgerInvalidTransition("outcome references a denied decision")
        if decision.operation != event.operation:
            raise LedgerInvalidTransition("outcome operation differs from decision")

        if event.operation == "llm":
            reservation = run.reservations.get(decision.event_id)
            if reservation is None:
                raise LedgerInvalidTransition("llm outcome has no reservation")
            if reservation.status != "open":
                raise LedgerInvalidTransition("llm outcome reservation is not open")
            if ref in run.normal_outcomes:
                raise LedgerInvalidTransition("second normal outcome for decision")
            if decision.model != event.model or decision.schema_name != event.schema_name:
                raise LedgerInvalidEvent("outcome model/schema differs from decision")
            if event.reason_code == "usage_missing" or event.reason_code == "usage_out_of_range":
                if event.reservation_released != 0:
                    raise LedgerInvalidEvent("unknown outcome release must be zero")
                status = "unknown"
                run = _replace_reservation(
                    run,
                    replace(
                        reservation,
                        status=status,
                        last_event_id=event.event_id,
                    ),
                )
                unknown = dict(run.unknown_outcomes)
                unknown[event.event_id] = decision.event_id
                run = replace(
                    run,
                    unknown_outcomes=unknown,
                    normal_outcomes=run.normal_outcomes | {ref},
                )
                return _replace_run(state, run)
            if event.reason_code == "released_no_effect":
                if event.reservation_released != reservation.reserved_tokens:
                    raise LedgerInvalidEvent("release does not equal the held reservation")
                run = _replace_reservation(
                    run,
                    replace(
                        reservation,
                        status="released",
                        closed_at=event.observed_now,
                        last_event_id=event.event_id,
                    ),
                )
                run = replace(
                    run,
                    reserved_tokens=_sub(run.reserved_tokens, reservation.reserved_tokens, "reserved_tokens"),
                    normal_outcomes=run.normal_outcomes | {ref},
                )
                return _replace_run(state, run)
            if event.reason_code in {"allowed", "reservation_overrun", "budget_breach"}:
                if event.reservation_released != reservation.reserved_tokens:
                    raise LedgerInvalidEvent("release does not equal the held reservation")
                return _apply_llm_usage(
                    state,
                    run,
                    decision,
                    reservation,
                    event,
                    target_status="applied",
                    allow_negative=event.reason_code == "budget_breach",
                    require_overrun_absent=False,
                    check_counters=check_counters,
                )
            raise LedgerInvalidTransition("unsupported llm outcome reason for ledger")

        if event.operation == "fetch":
            if event.target_host is not None and decision.target_host is not None:
                if event.target_host != decision.target_host:
                    raise LedgerInvalidEvent("fetch outcome target differs from decision")
            return _replace_run(state, replace(run, normal_outcomes=run.normal_outcomes | {ref}))

        return _replace_run(state, replace(run, normal_outcomes=run.normal_outcomes | {ref}))

    raise LedgerInvalidTransition(f"unsupported event kind {event.event_kind!r}")


def next_state(state: LedgerState, event: control.AuditEvent) -> LedgerState:
    """Apply one already-chain-verified event to the pure ledger state."""
    return _transition(state, event, check_counters=True)


def _apply_llm_usage(
    state: LedgerState,
    run: RunState,
    decision: DecisionRecord,
    reservation: Reservation,
    event: control.AuditEvent,
    *,
    target_status: str,
    allow_negative: bool,
    require_overrun_absent: bool,
    check_counters: bool,
) -> LedgerState:
    held = reservation.reserved_tokens
    if event.reservation_released != held:
        raise LedgerInvalidEvent("release does not equal the held reservation")
    usage_prompt = event.usage_prompt_tokens
    usage_completion = event.usage_completion_tokens
    if usage_prompt is None or usage_completion is None:
        raise LedgerInvalidEvent("usage-carrying outcome lacks usage")
    usage_total = _add(usage_prompt, usage_completion, "usage total")
    if event.reason_code == "allowed" and usage_total > held:
        raise LedgerInvalidEvent("allowed usage exceeds the held reservation")
    if event.reason_code == "reservation_overrun":
        expected = usage_total - held
        if expected <= 0 or event.overrun_tokens != expected:
            raise LedgerInvalidEvent("reservation_overrun amount is not exact")
    elif event.reason_code == "budget_breach":
        expected = max(0, usage_total - held)
        if event.overrun_tokens is not None and event.overrun_tokens != expected:
            raise LedgerInvalidEvent("budget_breach overrun amount is not exact")
    elif require_overrun_absent and event.overrun_tokens is not None:
        raise LedgerInvalidEvent("reconciliation must not carry overrun_tokens")

    prompt = _add(run.prompt_tokens, usage_prompt, "prompt_tokens")
    completion = _add(run.completion_tokens, usage_completion, "completion_tokens")
    reserved = _sub(run.reserved_tokens, held, "reserved_tokens")
    candidate = replace(
        run,
        prompt_tokens=prompt,
        completion_tokens=completion,
        reserved_tokens=reserved,
    )
    balance = _balance(candidate)
    if event.reason_code == "budget_breach":
        if balance >= 0:
            raise LedgerInvalidTransition("budget_breach requires a negative balance")
    elif not allow_negative and balance < 0:
        raise LedgerInvalidTransition("ledger balance would become negative")
    expected_counters = _counter_snapshot(candidate)
    if check_counters and dict(event.counters) != expected_counters:
        raise LedgerInvalidEvent("outcome counters do not match derived ledger counters")
    candidate = _replace_reservation(
        candidate,
        replace(
            reservation,
            status=target_status,
            closed_at=event.observed_now,
            last_event_id=event.event_id,
        ),
    )
    candidate = replace(candidate, normal_outcomes=candidate.normal_outcomes | {decision.event_id})
    if target_status == "resolved":
        candidate = replace(
            candidate,
            reconciled=candidate.reconciled | {event.references_event_id},
        )
    return _replace_run(state, candidate)


def expected_counters(state: LedgerState, event: control.AuditEvent) -> Mapping[str, int]:
    """Return expected six counters by running the shared transition without checks."""
    allowed_event = (
        event.event_kind == "decision"
        and event.operation == "llm"
        and event.decision == "allow"
    )
    usage_outcome = (
        event.event_kind == "outcome"
        and event.operation == "llm"
        and event.reason_code
        in {"allowed", "reservation_overrun", "budget_breach", "reconciliation"}
        and event.usage_prompt_tokens is not None
        and event.usage_completion_tokens is not None
    )
    if not (allowed_event or usage_outcome):
        raise LedgerInvalidTransition("event has no counter snapshot")
    result = _transition(state, event, check_counters=False)
    return _counter_snapshot(_require_run(result, event.run_id))


def replay_events(events: list[control.AuditEvent]) -> LedgerState:
    state = LedgerState()
    for event in events:
        state = next_state(state, event)
    return state


def _schemas_json(schemas: tuple[str, ...]) -> str:
    return json.dumps(list(schemas), separators=(",", ":"), ensure_ascii=False)


def state_to_rows(
    state: LedgerState, audit_head: tuple[int, str]
) -> dict[str, list[tuple[Any, ...]]]:
    """Map ledger state to the five tables as plain tuples (no DB access)."""
    policy_rows: list[tuple[Any, ...]] = []
    ledger_rows: list[tuple[Any, ...]] = []
    reservation_rows: list[tuple[Any, ...]] = []
    lease_rows: list[tuple[Any, ...]] = []
    for run_id in sorted(state.runs):
        run = state.runs[run_id]
        policy = run.policy
        policy_rows.append(
            (
                run_id,
                run.thread_id,
                policy.policy_version,
                policy.estimate_kind,
                str(policy.margin_num),
                str(policy.margin_den),
                policy.provider_overhead_tokens,
                policy.max_run_tokens,
                policy.max_run_llm_calls,
                policy.max_output_tokens,
                policy.max_fetches_per_run,
                policy.lease_ttl_seconds,
                policy.allowed_model,
                _schemas_json(policy.allowed_schemas),
            )
        )
        ledger_rows.append(
            (
                run_id,
                run.calls,
                run.prompt_tokens,
                run.completion_tokens,
                run.reserved_tokens,
                run.fetch_count,
            )
        )
        for decision_id in sorted(run.reservations):
            reservation = run.reservations[decision_id]
            reservation_rows.append(
                (
                    reservation.run_id,
                    reservation.decision_event_id,
                    reservation.reserved_tokens,
                    reservation.status,
                    reservation.opened_at,
                    reservation.closed_at,
                    reservation.last_event_id,
                )
            )
        if run.lease is not None:
            lease = run.lease
            lease_rows.append(
                (
                    run_id,
                    lease.owner_id,
                    lease.epoch,
                    lease.expires_at,
                    1 if lease.released else 0,
                    lease.released_at,
                    lease.last_event_id,
                )
            )
    return {
        "run_policy": policy_rows,
        "run_ledger": ledger_rows,
        "reservations": reservation_rows,
        "run_leases": lease_rows,
        "ledger_head": [(1, audit_head[0], audit_head[1])],
    }


def _read_audit(conn: sqlite3.Connection):
    rows = conn.execute(
        "SELECT seq, event_id, prev_hash, event_hash, payload "
        "FROM audit_events ORDER BY seq"
    ).fetchall()
    head = conn.execute(
        "SELECT seq, event_hash FROM audit_head WHERE singleton = 1"
    ).fetchone()
    return rows, head


def _is_busy(exc: sqlite3.Error) -> bool:
    text = str(exc).lower()
    return "locked" in text or "busy" in text


def _read_tables(conn: sqlite3.Connection):
    """Return (tables, problems); missing tables are data, not exceptions."""
    tables: dict[str, list[tuple[Any, ...]]] = {}
    problems: list[str] = []
    for name, sql in (
        (
            "run_policy",
            "SELECT run_id, thread_id, policy_version, estimate_kind, margin_num, "
            "margin_den, provider_overhead_tokens, max_run_tokens, max_run_llm_calls, "
            "max_output_tokens, max_fetches_per_run, lease_ttl_seconds, allowed_model, "
            "allowed_schemas_json FROM run_policy ORDER BY run_id",
        ),
        (
            "run_ledger",
            "SELECT run_id, calls, prompt_tokens, completion_tokens, reserved_tokens, "
            "fetch_count FROM run_ledger ORDER BY run_id",
        ),
        (
            "reservations",
            "SELECT run_id, decision_event_id, reserved_tokens, status, opened_at, "
            "closed_at, last_event_id FROM reservations ORDER BY decision_event_id",
        ),
        (
            "run_leases",
            "SELECT run_id, owner_id, epoch, expires_at, released, released_at, "
            "last_event_id FROM run_leases ORDER BY run_id",
        ),
        (
            "ledger_head",
            "SELECT singleton, seq, event_hash FROM ledger_head ORDER BY singleton",
        ),
    ):
        try:
            tables[name] = conn.execute(sql).fetchall()
        except sqlite3.OperationalError as exc:
            if _is_busy(exc):
                raise _BusyDatabase(str(exc)) from exc
            if "no such table" in str(exc).lower():
                problems.append(f"ledger table {name} is missing")
            else:
                problems.append(f"cannot read ledger table {name}: {type(exc).__name__}")
        except sqlite3.Error as exc:
            if _is_busy(exc):
                raise _BusyDatabase(str(exc)) from exc
            problems.append(f"cannot read ledger table {name}: {type(exc).__name__}")
    return tables, problems


def _compare_table(name, expected, live) -> list[str]:
    key = TABLE_KEYS[name]
    key_index = COLUMNS[name].index(key)
    expected_by_key = {row[key_index]: row for row in expected}
    live_by_key = {row[key_index]: row for row in live}
    problems: list[str] = []
    for row_key in sorted(set(expected_by_key) | set(live_by_key), key=lambda value: str(value)):
        if row_key not in live_by_key:
            problems.append(f"{name}: missing row {row_key}")
            continue
        if row_key not in expected_by_key:
            problems.append(f"{name}: unexpected row {row_key}")
            continue
        want = expected_by_key[row_key]
        got = live_by_key[row_key]
        for index, column in enumerate(COLUMNS[name]):
            if type(want[index]) is not type(got[index]) or want[index] != got[index]:
                problems.append(f"{name} {row_key}: column {column} differs from ledger replay")
    return problems


def verify_ledger(path: str | Path) -> list[str]:
    """Read-only replay + exact table comparison; never writes, repairs, or retries.

    SQLite is opened with timeout=0.0; a busy/locked read is reported immediately
    rather than waited on inside the sqlite3 library.
    """
    target = Path(path)
    if not target.exists():
        return [f"audit database does not exist: {target.name}"]
    uri = target.resolve().as_uri() + "?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=0.0)
    except sqlite3.Error as exc:
        return [f"cannot open ledger database read-only: {type(exc).__name__}"]

    try:
        try:
            conn.execute("BEGIN")
            rows, head = _read_audit(conn)
            problems = control.chain_problems(rows, head, allow_empty=False)
            if problems:
                conn.execute("COMMIT")
                return problems

            events: list[control.AuditEvent] = []
            for seq, event_id, prev_hash, event_hash, payload in rows:
                try:
                    event = control.event_from_payload(json.loads(payload), f"seq {seq}")
                except (TypeError, ValueError) as exc:
                    conn.execute("COMMIT")
                    return [f"seq {seq}: {exc}"]
                events.append(event)

            state = LedgerState()
            for seq, event in zip([row[0] for row in rows], events):
                try:
                    state = next_state(state, event)
                except LedgerError as exc:
                    conn.execute("COMMIT")
                    return [f"seq {seq}: {exc}"]

            tables, storage_problems = _read_tables(conn)
            conn.execute("COMMIT")
        except _BusyDatabase:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            return ["cannot read ledger snapshot: database is busy or locked"]
        except sqlite3.Error as exc:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            if _is_busy(exc):
                return ["cannot read ledger snapshot: database is busy or locked"]
            return [f"cannot read ledger snapshot: {type(exc).__name__}"]
    finally:
        conn.close()

    if storage_problems:
        return storage_problems

    expected = state_to_rows(state, head)
    problems: list[str] = []
    for name in ("run_policy", "run_ledger", "reservations", "run_leases", "ledger_head"):
        problems.extend(_compare_table(name, expected[name], tables.get(name, [])))
    return problems
