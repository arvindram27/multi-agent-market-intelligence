"""Gatekeeper for the H1 control layer (slice 2b-iii, first batch).

One Gatekeeper serves one run. Every public method that writes opens exactly one
``AuditStore.transaction(ledger.LedgerProjection())``, verifies the committed
chain and the ledger tables inside that transaction, applies the pure admission
rules to a RunView derived from the verified ledger state, builds one audit
event and appends it, so the audit row and the ledger rows commit or roll back
together. ``event_status`` is read-only.

Contract
- Every writing method takes a caller-supplied ``event_id``. A call is
  execute-or-confirm: if the id is already in the chain the call returns
  ``already_applied`` and writes nothing, otherwise it runs as a NEW request
  (new clock reading, current state, it may be denied or refused). The
  gatekeeper never retries by itself. ``event_status`` tells a caller whether an
  id is present, absent or could not be verified, without making a decision.
- A duplicate is rebuilt from the request, the ledger state before the stored
  event and the stored time and lease binding, then compared by control with the
  stored event through its logical fingerprint. It reads no clock and skips the
  lease, clock and reservation checks that apply to new events. It does not
  prove the request is unchanged: the chain does not store max_output_tokens, the
  cause of a denial beyond its reason code, malformed usage values or request
  bytes.
- New events must not move observed_now back: it must be at least the largest
  observed_now of any event of this run. This is stricter than the chain, which
  only compares lease events. A second owner whose clock is behind is refused
  until its clock catches up.
- Lease: a decision or outcome needs a live lease owned by this gatekeeper. A
  lease that has already expired cannot be renewed (the chain rejects it). The
  owner may still release it, and a different owner id may then acquire and
  record outcomes for earlier decisions. If the owner is gone, closing an open
  reservation needs takeover, which is not in this batch. There is no heartbeat.
- The gatekeeper never estimates a token count, never calls a tokenizer or a
  provider, and never reads a clock other than the injected ``now``: exactly once
  per NEW event, never for a duplicate or for event_status. That one reading is
  both ``recorded_at`` and, for lease, decision and outcome events, ``observed_now``
  (run_started sets only ``recorded_at``). ``recorded_at`` is always passed to
  make_event explicitly, so its system-clock default never runs; a duplicate is
  rebuilt with the stored ``recorded_at``.
- Usage that cannot be stored (wrong type, negative, above int64, one figure
  missing, or a derived total that would leave int64) is recorded as
  ``usage_out_of_range`` with no figures. Both figures None is ``usage_missing``.
  The bad values themselves are not kept.
- The verified read replays the whole chain, and the participant replays it again
  on the first append, so each call is O(n) twice.
- Fetch, write, approval, takeover and reconciliation are not in this batch.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import market_intelligence.admission as admission
import market_intelligence.control as control
import market_intelligence.ledger as ledger

INT64_MAX = 2**63 - 1

_AUDIT_COLUMNS = {
    "audit_events": ("seq", "event_id", "prev_hash", "event_hash", "payload"),
    "audit_head": ("singleton", "seq", "event_hash"),
}


class GatekeeperError(RuntimeError):
    """Base class for gatekeeper refusals raised before anything is appended."""


class GatekeeperRequestError(GatekeeperError):
    pass


class GatekeeperUnknownRun(GatekeeperError):
    pass


class GatekeeperStateError(GatekeeperError):
    pass


class GatekeeperPolicyMismatch(GatekeeperError):
    pass


class GatekeeperLeaseError(GatekeeperError):
    pass


class GatekeeperClockError(GatekeeperError):
    pass


@dataclass(frozen=True)
class GatekeeperResult:
    status: str  # "applied" or "already_applied"
    event_id: str
    event_hash: str
    decision: admission.Decision | None = None


@dataclass(frozen=True)
class EventStatus:
    state: str  # "present", "absent" or "verification_failed"
    problems: tuple[str, ...] = ()
    seq: int | None = None
    event_hash: str | None = None
    event_kind: str | None = None
    operation: str | None = None
    run_matches: bool | None = None
    observed_now: str | None = None
    decision: str | None = None
    reason_code: str | None = None


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _usable_int(value: object) -> bool:
    return type(value) is int and 0 <= value <= INT64_MAX


class Gatekeeper:
    def __init__(
        self,
        store: control.AuditStore,
        policy: admission.PolicyConfig,
        *,
        run_id: str,
        thread_id: str,
        owner_id: str,
        lease_ttl_seconds: int,
        now: Callable[[], datetime],
        estimate_kind: str = "policy_margin_v1",
    ) -> None:
        if not isinstance(store, control.AuditStore):
            raise GatekeeperRequestError("store must be an AuditStore")
        if not isinstance(policy, admission.PolicyConfig):
            raise GatekeeperRequestError("policy must be a PolicyConfig")
        if not callable(now):
            raise GatekeeperRequestError("now must be callable")
        if type(lease_ttl_seconds) is not int or not 1 <= lease_ttl_seconds <= INT64_MAX // 4:
            raise GatekeeperRequestError("lease_ttl_seconds must be a positive int")
        for name, value in (
            ("run_id", run_id),
            ("thread_id", thread_id),
            ("owner_id", owner_id),
            ("estimate_kind", estimate_kind),
        ):
            if not isinstance(value, str) or not value:
                raise GatekeeperRequestError(f"{name} must be a non-empty string")
        self._store = store
        self._policy = policy
        self._run_id = run_id
        self._thread_id = thread_id
        self._owner = owner_id
        self._ttl = lease_ttl_seconds
        self._now = now
        self._estimate_kind = estimate_kind
        self._db_path = self._main_database_path(store)

    # ------------------------------------------------------------------ setup

    @staticmethod
    def _main_database_path(store: control.AuditStore) -> Path:
        """Absolute path of the database file the store actually has open.

        Taken from SQLite itself, once, so event_status cannot open a second file
        if the working directory changes after the store was created.
        """
        rows = store._conn.execute("PRAGMA database_list").fetchall()
        for _seq, name, file in rows:
            if name == "main" and file:
                path = Path(file)
                if not path.is_absolute():
                    raise GatekeeperRequestError("store database path is not absolute")
                return path
        raise GatekeeperRequestError("store has no file-backed main database")

    def _policy_fields(self) -> dict[str, Any]:
        policy = self._policy
        numerator, denominator = policy.reservation_margin.as_integer_ratio()
        return dict(
            policy_margin=policy.reservation_margin,
            policy_margin_num=numerator,
            policy_margin_den=denominator,
            provider_overhead_tokens=policy.provider_overhead_tokens,
            max_run_tokens=policy.max_run_tokens,
            max_run_llm_calls=policy.max_run_llm_calls,
            max_output_tokens=policy.max_output_tokens,
            max_fetches_per_run=policy.max_fetches_per_run,
            lease_ttl_seconds=self._ttl,
            allowed_model=policy.allowed_model,
            allowed_schemas=tuple(sorted(policy.allowed_schemas)),
            estimate_kind=self._estimate_kind,
        )

    def _policy_matches(self, run: ledger.RunState) -> bool:
        snap = run.policy
        want = self._policy_fields()
        return (
            snap.margin_num,
            snap.margin_den,
            snap.provider_overhead_tokens,
            snap.max_run_tokens,
            snap.max_run_llm_calls,
            snap.max_output_tokens,
            snap.max_fetches_per_run,
            snap.lease_ttl_seconds,
            snap.allowed_model,
            snap.allowed_schemas,
            snap.estimate_kind,
            snap.policy_version,
        ) == (
            want["policy_margin_num"],
            want["policy_margin_den"],
            want["provider_overhead_tokens"],
            want["max_run_tokens"],
            want["max_run_llm_calls"],
            want["max_output_tokens"],
            want["max_fetches_per_run"],
            want["lease_ttl_seconds"],
            want["allowed_model"],
            want["allowed_schemas"],
            want["estimate_kind"],
            control.POLICY_VERSION_H1_SLICE2,
        )

    def _event(self, recorded_at: str, **values: Any) -> control.AuditEvent:
        # recorded_at is always passed, so make_event never reads the system clock.
        return control.make_event(
            recorded_at=recorded_at,
            run_id=self._run_id,
            thread_id=self._thread_id,
            policy_version=control.POLICY_VERSION_H1_SLICE2,
            **values,
        )

    @staticmethod
    def _view(run: ledger.RunState) -> admission.RunView:
        return admission.RunView(
            calls=run.calls,
            prompt_tokens=run.prompt_tokens,
            completion_tokens=run.completion_tokens,
            reserved_tokens=run.reserved_tokens,
            fetch_count=run.fetch_count,
            approved=run.approved_target is not None,
        )

    # --------------------------------------------------------- verified read

    @staticmethod
    def _read_verified(conn: sqlite3.Connection):
        """Verify chain and ledger on the open transaction connection.

        Returns the decoded events, the ledger state before each event, and the
        final state. Does not use any participant state.
        """
        rows = conn.execute(
            "SELECT seq, event_id, prev_hash, event_hash, payload "
            "FROM audit_events ORDER BY seq"
        ).fetchall()
        head_row = conn.execute(
            "SELECT seq, event_hash FROM audit_head WHERE singleton = 1"
        ).fetchone()
        head = (head_row[0], head_row[1]) if head_row else None
        problems = control.chain_problems(rows, head, allow_empty=True)
        if problems:
            raise control.AuditIncomplete("; ".join(problems))
        events: list[control.AuditEvent] = []
        before: list[ledger.LedgerState] = []
        state = ledger.LedgerState()
        for row in rows:
            event = control.event_from_payload(json.loads(row[4]), f"seq {row[0]}")
            events.append(event)
            before.append(state)
            try:
                state = ledger.next_state(state, event)
            except ledger.LedgerError as exc:
                raise ledger.LedgerProjectionMismatch([f"seq {row[0]}: {exc}"]) from exc
        ledger.ensure_schema(conn)
        if rows and ledger.projection_is_empty(conn):
            raise ledger.LedgerBootstrapRequired(
                f"audit chain has {len(rows)} events but the ledger tables are empty"
            )
        mismatches = ledger.projection_problems(conn, state, head)
        if mismatches:
            raise ledger.LedgerProjectionMismatch(mismatches)
        return events, before, state

    def _stamp(self) -> datetime:
        value = self._now()
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise GatekeeperClockError("now must return a timezone-aware datetime")
        return value.astimezone(timezone.utc)

    def _check_monotonic(self, now: datetime, events: list[control.AuditEvent]) -> None:
        seen = [
            _utc(event.observed_now)
            for event in events
            if event.run_id == self._run_id and event.observed_now
        ]
        if seen and now < max(seen):
            raise GatekeeperClockError("observed_now would move behind an earlier event")

    def _owned_lease(
        self, run: ledger.RunState, now: datetime, *, require_unexpired: bool
    ) -> ledger.LeaseRecord:
        lease = run.lease
        if lease is None or lease.released:
            raise GatekeeperLeaseError("the run has no live lease")
        if lease.owner_id != self._owner:
            raise GatekeeperLeaseError("the live lease belongs to another owner")
        if require_unexpired and _utc(lease.expires_at) <= now:
            raise GatekeeperLeaseError("the lease has expired")
        return lease

    # ---------------------------------------------------------------- driver

    def _execute(self, event_id, kind, operation, build, *, run_must_exist=True):
        if not isinstance(event_id, str) or not event_id:
            raise GatekeeperRequestError("event_id must be a non-empty string")
        with self._store.transaction(ledger.LedgerProjection()) as tx:
            events, before, state = self._read_verified(tx._conn)
            index = {event.event_id: i for i, event in enumerate(events)}
            stored = None
            prior = state
            if event_id in index:
                position = index[event_id]
                stored = events[position]
                prior = before[position]
                if stored.run_id != self._run_id or stored.thread_id != self._thread_id:
                    raise GatekeeperRequestError("event_id belongs to a different run")
                if stored.event_kind != kind or stored.operation != operation:
                    raise GatekeeperRequestError("event_id belongs to a different event kind")
            elif run_must_exist and self._run_id not in state.runs:
                raise GatekeeperUnknownRun(self._run_id)
            event, decision = build(state, prior, events, stored)
            result = tx.append_in_tx(event)
            return GatekeeperResult(result.status, event.event_id, result.event_hash, decision)

    def _run_of(self, state: ledger.LedgerState) -> ledger.RunState:
        run = state.runs.get(self._run_id)
        if run is None:
            raise GatekeeperUnknownRun(self._run_id)
        if not self._policy_matches(run):
            raise GatekeeperPolicyMismatch("policy differs from the run_started snapshot")
        return run

    # ------------------------------------------------------------ run, lease

    def start_run(self, event_id: str) -> GatekeeperResult:
        def build(state, prior, events, stored):
            if stored is None and self._run_id in state.runs:
                raise GatekeeperStateError("the run was already started")
            if stored is None:
                recorded = self._stamp().isoformat()
            else:
                recorded = stored.recorded_at
            event = self._event(
                recorded,
                event_id=event_id,
                operation="run",
                event_kind="run_started",
                decision="allow",
                reason_code="allowed",
                counters={},
                **self._policy_fields(),
            )
            return event, None

        return self._execute(event_id, "run_started", "run", build, run_must_exist=False)

    def _lease(self, event_id: str, kind: str) -> GatekeeperResult:
        def build(state, prior, events, stored):
            run = self._run_of(state)
            if stored is not None:
                observed = _utc(stored.observed_now)
                recorded = stored.recorded_at
                epoch = stored.lease_epoch
            else:
                observed = self._stamp()
                self._check_monotonic(observed, events)
                recorded = observed.isoformat()
                lease = run.lease
                if kind == "lease_acquired":
                    if lease is not None and not lease.released:
                        raise GatekeeperLeaseError("a live lease already exists")
                    if lease is not None and lease.owner_id == self._owner:
                        raise GatekeeperLeaseError("acquire needs an owner id that did not hold the previous lease")
                    epoch = 1 if lease is None else lease.epoch + 1
                else:
                    lease = self._owned_lease(
                        run, observed, require_unexpired=kind == "lease_renewed"
                    )
                    epoch = lease.epoch
                    if kind == "lease_renewed" and _utc(lease.expires_at) >= observed + timedelta(seconds=self._ttl):
                        raise GatekeeperLeaseError("renewal would not advance the expiry")
            expires = (observed + timedelta(seconds=self._ttl)).isoformat()
            event = self._event(
                recorded,
                event_id=event_id,
                operation="run",
                event_kind=kind,
                decision="allow",
                reason_code="allowed",
                counters={},
                lease_owner=self._owner,
                lease_epoch=epoch,
                observed_now=observed.isoformat(),
                lease_expires_at=None if kind == "lease_released" else expires,
                lease_ttl_seconds=self._ttl,
            )
            return event, None

        return self._execute(event_id, kind, "run", build)

    def acquire_lease(self, event_id: str) -> GatekeeperResult:
        return self._lease(event_id, "lease_acquired")

    def renew_lease(self, event_id: str) -> GatekeeperResult:
        return self._lease(event_id, "lease_renewed")

    def release_lease(self, event_id: str) -> GatekeeperResult:
        return self._lease(event_id, "lease_released")

    # ------------------------------------------------------------------- llm

    def admit_llm(self, request: admission.LLMRequest, event_id: str) -> GatekeeperResult:
        if not isinstance(request, admission.LLMRequest):
            raise GatekeeperRequestError("request must be an LLMRequest")

        def build(state, prior, events, stored):
            run = self._run_of(state)
            basis = self._run_of(prior)
            decision = admission.decide_llm(request, self._view(basis), self._policy)
            if (
                request.serialized_body_token_count is None
                and decision.reason is admission.AdmissionReason.POLICY_DENIED
            ):
                raise GatekeeperRequestError(
                    "a request without a token count that is also policy denied has no representable event"
                )
            if stored is None:
                now = self._stamp()
                self._check_monotonic(now, events)
                lease = self._owned_lease(run, now, require_unexpired=True)
                observed, owner, epoch = now.isoformat(), lease.owner_id, lease.epoch
                recorded = observed
            else:
                observed, owner, epoch = stored.observed_now, stored.lease_owner, stored.lease_epoch
                recorded = stored.recorded_at
                if owner != self._owner:
                    raise GatekeeperRequestError("the stored event belongs to another lease owner")
            event = self._event(
                recorded,
                event_id=event_id,
                operation="llm",
                event_kind="decision",
                decision="allow" if decision.allow else "deny",
                reason_code=decision.reason.value,
                counters={},
                model=request.model,
                schema_name=request.schema_name,
                serialized_body_token_count=request.serialized_body_token_count,
                reserve_tokens=decision.reserve_tokens,
                lease_owner=owner,
                lease_epoch=epoch,
                observed_now=observed,
                **self._policy_fields(),
            )
            return event, decision

        return self._execute(event_id, "decision", "llm", build)

    def record_llm_outcome(
        self,
        decision_event_id: str,
        usage_prompt_tokens: object,
        usage_completion_tokens: object,
        event_id: str,
    ) -> GatekeeperResult:
        if not isinstance(decision_event_id, str) or not decision_event_id:
            raise GatekeeperRequestError("decision_event_id must be a non-empty string")

        def build(state, prior, events, stored):
            run = self._run_of(state)
            basis = self._run_of(prior)
            if stored is not None and stored.references_event_id != decision_event_id:
                raise GatekeeperRequestError("event_id references a different decision")
            decision = basis.decisions.get(decision_event_id)
            if decision is None or not decision.allow or decision.operation != "llm":
                raise GatekeeperRequestError("the decision is not a known allowed llm decision")
            reservation = basis.reservations.get(decision_event_id)
            if reservation is None or (stored is None and reservation.status != "open"):
                raise GatekeeperRequestError("the decision has no open reservation")
            held = reservation.reserved_tokens
            if stored is None:
                now = self._stamp()
                self._check_monotonic(now, events)
                lease = self._owned_lease(run, now, require_unexpired=True)
                observed, owner, epoch = now.isoformat(), lease.owner_id, lease.epoch
                recorded = observed
            else:
                observed, owner, epoch = stored.observed_now, stored.lease_owner, stored.lease_epoch
                recorded = stored.recorded_at
                if owner != self._owner:
                    raise GatekeeperRequestError("the stored event belongs to another lease owner")
            common = dict(
                event_id=event_id,
                operation="llm",
                event_kind="outcome",
                model=decision.model,
                schema_name=decision.schema_name,
                references_event_id=decision_event_id,
                lease_owner=owner,
                lease_epoch=epoch,
                observed_now=observed,
            )
            prompt, completion = usage_prompt_tokens, usage_completion_tokens
            if prompt is None and completion is None:
                return (
                    self._event(
                        recorded,
                        decision="allow",
                        reason_code="usage_missing",
                        counters={},
                        usage_missing=True,
                        reservation_released=0,
                        **common,
                    ),
                    None,
                )
            outcome = self._usage_outcome(basis, held, prompt, completion)
            if outcome is None:
                return (
                    self._event(
                        recorded,
                        decision="allow",
                        reason_code="usage_out_of_range",
                        counters={},
                        usage_missing=True,
                        reservation_released=0,
                        error_class="UsageOutOfRange",
                        **common,
                    ),
                    None,
                )
            reason, overrun, counters = outcome
            return (
                self._event(
                    recorded,
                    decision="allow",
                    reason_code=reason,
                    counters=counters,
                    usage_prompt_tokens=prompt,
                    usage_completion_tokens=completion,
                    reservation_released=held,
                    overrun_tokens=overrun,
                    **common,
                ),
                None,
            )

        return self._execute(event_id, "outcome", "llm", build)

    @staticmethod
    def _usage_outcome(run: ledger.RunState, held: int, prompt: object, completion: object):
        """Return (reason, overrun_tokens, counters), or None when usage is unusable."""
        if not (_usable_int(prompt) and _usable_int(completion)):
            return None
        total = prompt + completion
        new_prompt = run.prompt_tokens + prompt
        new_completion = run.completion_tokens + completion
        reserved = run.reserved_tokens - held
        if reserved < 0:
            return None
        if max(total, new_prompt, new_completion) > INT64_MAX:
            return None
        balance = run.policy.max_run_tokens - new_prompt - new_completion - reserved
        if max(0, -balance) > INT64_MAX:
            return None
        over = total - held
        if balance < 0:
            reason, overrun = "budget_breach", (over if over > 0 else None)
        elif over > 0:
            reason, overrun = "reservation_overrun", over
        else:
            reason, overrun = "allowed", None
        counters = {
            "calls": run.calls,
            "prompt_tokens": new_prompt,
            "completion_tokens": new_completion,
            "reserved_tokens": reserved,
            "remaining_tokens": max(0, balance),
            "budget_deficit": max(0, -balance),
        }
        return reason, overrun, counters

    # ---------------------------------------------------------- event status

    def event_status(self, event_id: str) -> EventStatus:
        """Read-only: is this event id in a verified chain?

        Opens its own read-only connection to the database file the store has
        open, reads chain, ledger tables and schema in one snapshot, and never
        writes, creates a file, runs DDL, reads the clock, calls admission or
        appends. ``absent`` is reported only when the chain and the schema
        verified; anything unverifiable is ``verification_failed``.
        """

        def failed(*problems: str) -> EventStatus:
            return EventStatus("verification_failed", tuple(problems))

        path = self._db_path
        if not path.exists():
            return failed("audit database file does not exist")
        try:
            conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=0.0)
        except sqlite3.Error as exc:
            return failed(f"cannot open the audit database read-only: {type(exc).__name__}")
        try:
            try:
                conn.execute("BEGIN")
                found = self._snapshot(conn, event_id)
                conn.execute("COMMIT")
            except (sqlite3.Error, ValueError, TypeError, ledger.LedgerError) as exc:
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                return failed(f"cannot verify the audit database: {type(exc).__name__}")
        finally:
            conn.close()
        return found

    def _snapshot(self, conn: sqlite3.Connection, event_id: str):
        for table, columns in _AUDIT_COLUMNS.items():
            info = conn.execute(f"PRAGMA table_info({table})").fetchall()
            if tuple(row[1] for row in info) != columns:
                return EventStatus(
                    "verification_failed", (f"audit table {table} is missing or malformed",)
                )
        rows = conn.execute(
            "SELECT seq, event_id, prev_hash, event_hash, payload "
            "FROM audit_events ORDER BY seq"
        ).fetchall()
        head_row = conn.execute(
            "SELECT seq, event_hash FROM audit_head WHERE singleton = 1"
        ).fetchone()
        head = (head_row[0], head_row[1]) if head_row else None
        problems = control.chain_problems(rows, head, allow_empty=True)
        if problems:
            return EventStatus("verification_failed", tuple(problems))
        events = [
            control.event_from_payload(json.loads(row[4]), f"seq {row[0]}") for row in rows
        ]
        state = ledger.replay_events(events)
        ledger_names = tuple(ledger.COLUMNS)
        marks = ", ".join("?" for _ in ledger_names)
        present = conn.execute(
            f"SELECT count(*) FROM sqlite_master WHERE name IN ({marks})", ledger_names
        ).fetchone()[0]
        if rows or present:
            schema = ledger.schema_problems(conn)
            if schema:
                return EventStatus("verification_failed", tuple(schema))
            mismatches = ledger.projection_problems(conn, state, head)
            if mismatches:
                return EventStatus("verification_failed", tuple(mismatches))
        for row, event in zip(rows, events):
            if row[1] == event_id:
                return EventStatus(
                    "present",
                    (),
                    row[0],
                    row[3],
                    event.event_kind,
                    event.operation,
                    event.run_id == self._run_id and event.thread_id == self._thread_id,
                    event.observed_now,
                    event.decision,
                    event.reason_code,
                )
        return EventStatus("absent")
