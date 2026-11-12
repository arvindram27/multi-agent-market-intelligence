"""Pure ledger replay and read-only verifier for H1 slice 2b-ii-A.

``ensure_schema`` and ``apply_state`` write only on a caller-supplied connection
inside a caller-owned transaction; they never begin, commit or roll back, and
``verify_ledger`` and the read helpers stay read-only. Only LedgerProjection calls the write
functions, and nothing calls it yet. ``next_state`` is pure and never reads a clock or calls
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


class LedgerError(control.ProjectionRejected):
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


SCHEMA_STATEMENTS = (
    RUN_POLICY_SQL,
    RUN_LEDGER_SQL,
    RESERVATIONS_SQL,
    RUN_LEASES_SQL,
    LEDGER_HEAD_SQL,
)


def _build_upsert_sql() -> dict[str, str]:
    statements: dict[str, str] = {}
    for table, columns in COLUMNS.items():
        key = TABLE_KEYS[table]
        marks = ", ".join("?" for _ in columns)
        updates = ", ".join(
            f"{column} = excluded.{column}" for column in columns if column != key
        )
        statements[table] = (
            f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({marks}) "
            f"ON CONFLICT({key}) DO UPDATE SET {updates}"
        )
    return statements


UPSERT_SQL = _build_upsert_sql()


class LedgerProjectionMismatch(LedgerError):
    """The live schema is not exactly the expected ledger schema."""

    def __init__(self, problems: list[str]):
        self.problems = tuple(problems)
        super().__init__("; ".join(self.problems))


class LedgerUsageError(RuntimeError):
    """The caller broke a syntactic precondition of a projection function."""


def _validate_projection_head(audit_head: object):
    if type(audit_head) is not tuple or len(audit_head) != 2:
        raise LedgerUsageError("audit_head must be a (seq, event_hash) tuple")
    seq, event_hash = audit_head
    if type(seq) is not int or seq < 1:
        raise LedgerUsageError("audit_head seq must be an int >= 1")
    if type(event_hash) is not str or not event_hash:
        raise LedgerUsageError("audit_head event_hash must be a non-empty str")
    return seq, event_hash


def schema_problems(conn: sqlite3.Connection) -> list[str]:
    """Compare the live ledger schema with a private reference schema.

    Read only. Never commits or rolls back, and never closes the supplied
    connection. Limits: views that merely reference a ledger table are not
    detected, attached databases are not inspected, and unrelated objects are
    ignored.
    """
    names = tuple(COLUMNS)
    query = (
        "SELECT type, name, tbl_name, sql FROM sqlite_master "
        "WHERE tbl_name IN (?, ?, ?, ?, ?) ORDER BY type, name"
    )
    temp_query = query.replace("sqlite_master", "sqlite_temp_master")
    reference = sqlite3.connect(":memory:")
    try:
        for statement in SCHEMA_STATEMENTS:
            reference.execute(statement)
        reference_rows = reference.execute(query, names).fetchall()
    finally:
        reference.close()
    live_rows = conn.execute(query, names).fetchall()
    temp_rows = conn.execute(temp_query, names).fetchall()

    def indexed(rows):
        return {(row[0], row[1]): (row[2], row[3]) for row in rows}

    expected = indexed(reference_rows)
    actual = indexed(live_rows)
    problems: list[str] = []
    for object_type, name in sorted(expected):
        tbl_name, sql = expected[(object_type, name)]
        if (object_type, name) not in actual:
            problems.append(f"missing {object_type} {name} on {tbl_name}")
        elif actual[(object_type, name)] != (tbl_name, sql):
            problems.append(f"different {object_type} {name} on {tbl_name}")
    for object_type, name in sorted(set(actual) - set(expected)):
        tbl_name, _ = actual[(object_type, name)]
        problems.append(f"unexpected {object_type} {name} on {tbl_name}")
    for object_type, name, tbl_name, _ in temp_rows:
        problems.append(f"TEMP {object_type} {name} on {tbl_name}")
    return problems


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create the five ledger tables only inside the caller's open transaction.

    No commit, no rollback: the caller owns the transaction. Table creation is
    executed one statement at a time so it stays inside that transaction.
    """
    if not conn.in_transaction:
        raise LedgerUsageError("ensure_schema requires an active transaction")
    for statement in SCHEMA_STATEMENTS:
        conn.execute(statement)
    problems = schema_problems(conn)
    if problems:
        raise LedgerProjectionMismatch(problems)


def projection_problems(
    conn: sqlite3.Connection,
    state: LedgerState,
    audit_head: tuple[int, str] | None,
) -> list[str]:
    """Compare live ledger tables with state_to_rows.

    Read only. Never commits or rolls back. An empty chain is represented by
    audit_head None; its expected content is zero rows in all five tables.
    """
    if not isinstance(state, LedgerState):
        raise LedgerUsageError("state must be a LedgerState")
    if audit_head is None:
        if state.runs:
            raise LedgerUsageError("empty-chain head contradicts a non-empty state")
        expected = {name: [] for name in COLUMNS}
    else:
        _validate_projection_head(audit_head)
        expected = state_to_rows(state, audit_head)
    try:
        tables, storage_problems = _read_tables(conn)
    except _BusyDatabase as exc:
        cause = exc.__cause__
        if isinstance(cause, sqlite3.Error):
            raise cause
        raise
    problems = list(storage_problems)
    for name in COLUMNS:
        if name in tables:
            problems.extend(_compare_table(name, expected[name], tables[name]))
    return problems


def projection_is_empty(conn: sqlite3.Connection) -> bool:
    """True only when all five tables exist and all hold zero rows.

    Read only. Never commits or rolls back. Only a missing table gives False;
    any other sqlite error propagates unchanged.
    """
    for name in COLUMNS:
        try:
            row = conn.execute(f"SELECT 1 FROM {name} LIMIT 1").fetchone()
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc).lower():
                return False
            raise
        if row is not None:
            return False
    return True


def apply_state(
    conn: sqlite3.Connection,
    state: LedgerState,
    audit_head: tuple[int, str],
) -> None:
    """Upsert state_to_rows into the five ledger tables.

    No commit, no rollback: the caller owns the transaction and must have
    ensured the schema and checked projection_problems for the pre-event state.
    The caller also owns that state is the replay of a verified chain up to
    exactly audit_head. It never deletes rows.
    """
    if not conn.in_transaction:
        raise LedgerUsageError("apply_state requires an active transaction")
    if not isinstance(state, LedgerState):
        raise LedgerUsageError("state must be a LedgerState")
    _validate_projection_head(audit_head)
    rows = state_to_rows(state, audit_head)
    for table in COLUMNS:
        statement = UPSERT_SQL[table]
        for row in rows[table]:
            conn.execute(statement, row)


class LedgerBootstrapRequired(LedgerError):
    """The audit chain has events but the ledger projection tables are empty."""


def _replay_prefix(rows):
    """Replay chain rows through next_state; report a failure like verify_ledger."""
    state = LedgerState()
    for seq, _event_id, _prev_hash, _event_hash, payload in rows:
        try:
            event = control.event_from_payload(json.loads(payload), f"seq {seq}")
            state = next_state(state, event)
        except (TypeError, ValueError) as exc:
            raise LedgerProjectionMismatch([f"seq {seq}: {exc}"]) from exc
    head = (rows[-1][0], rows[-1][3]) if rows else None
    return state, head


class LedgerProjection:
    """Single-use participant that keeps the ledger tables equal to the chain.

    Control calls start, applied, already_applied and finalize inside one open
    audit transaction. This class never begins, commits or rolls back. It reads
    the committed chain only after control has verified it (the first applied or
    already_applied call), advances the pure ledger state per inserted event,
    and writes the final state once in finalize.
    """

    def __init__(self) -> None:
        self._used = False
        self._conn: sqlite3.Connection | None = None
        self._dead = False
        self._ready = False
        self._base_state: LedgerState | None = None
        self._base_head: tuple[int, str] | None = None
        self._state: LedgerState | None = None
        self._last: tuple[int, str] | None = None
        self._pending_ids: set[str] = set()

    def start(self, conn: sqlite3.Connection) -> None:
        if self._used:
            raise LedgerUsageError("ledger projection participant already used")
        self._used = True
        self._conn = conn

    def _live_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise LedgerUsageError("ledger projection participant was not started")
        if self._dead:
            raise LedgerUsageError("ledger projection participant already failed")
        return self._conn

    def _prepare(self, conn: sqlite3.Connection, prefix) -> None:
        ensure_schema(conn)
        state, head = _replay_prefix(prefix)
        if prefix and projection_is_empty(conn):
            raise LedgerBootstrapRequired(
                f"audit chain has {len(prefix)} events but the ledger tables are empty"
            )
        problems = projection_problems(conn, state, head)
        if problems:
            raise LedgerProjectionMismatch(problems)
        self._base_state = state
        self._base_head = head
        self._ready = True

    def applied(self, stored: control.StoredEvent, result: control.AppendResult) -> None:
        conn = self._live_conn()
        try:
            event = control.event_from_payload(
                json.loads(stored.payload), f"seq {stored.seq}"
            )
            if not self._ready:
                rows, _head = _read_audit(conn)
                if not rows or rows[-1][0] != stored.seq or rows[-1][3] != stored.event_hash:
                    raise LedgerUsageError("stored event is not the last audit row")
                prefix = [row for row in rows if row[0] < stored.seq]
                expected_prev = prefix[-1][3] if prefix else None
                if stored.prev_hash != expected_prev:
                    raise LedgerUsageError("stored event does not extend the prefix")
                self._prepare(conn, prefix)
                state = self._base_state
            else:
                if self._last is None or (
                    stored.seq != self._last[0] + 1 or stored.prev_hash != self._last[1]
                ):
                    raise LedgerUsageError("stored event does not extend the last event")
                if stored.event_id in self._pending_ids:
                    raise LedgerUsageError("stored event id was already applied")
                state = self._state
            self._state = next_state(state, event)
            self._last = (stored.seq, stored.event_hash)
            self._pending_ids.add(stored.event_id)
        except BaseException:
            self._dead = True
            raise

    def already_applied(self, event: control.AuditEvent, result: control.AppendResult) -> None:
        conn = self._live_conn()
        self._dead = True
        if not self._ready:
            rows, _head = _read_audit(conn)
            self._prepare(conn, rows)

    def finalize(self) -> None:
        conn = self._live_conn()
        self._dead = True
        if self._last is None:
            return
        row = conn.execute(
            "SELECT seq, event_hash FROM audit_head WHERE singleton = 1"
        ).fetchone()
        if row is None or (row[0], row[1]) != self._last:
            raise LedgerProjectionMismatch(
                ["audit head differs from the last applied event"]
            )
        apply_state(conn, self._state, (row[0], row[1]))
