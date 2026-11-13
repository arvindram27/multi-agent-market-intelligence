"""Offline run-lifecycle coordinator proof (G4b). Not wired into the app, UI or graph.

This module drives the real Gatekeeper lease calls around an approval pause. It has
no LangGraph import: pausing and resuming are stub callables supplied by the caller.

What it does and does not give:
- The audit store stays the only authority for who held the lease. The intent journal
  is an append-only local file written before each lease call; it is not authority.
- current_lease() is a value read at one moment. A check made from it before acting
  is advisory. Only the Gatekeeper's own transactional checks are enforced, and
  nothing here fences an old worker or an external effect.
- Resume is at-least-once: a stub that raised after a decision may run again later.
  Nothing here makes persistence exactly-once or at-most-once.
- A single writer per journal file is assumed. Two writers on one file are not handled.
- An owner id is reused only by the same Coordinator object (same incarnation) to retry
  its own open intent. A new Coordinator on the same journal never reuses an owner id
  from it: it only reads those records and closes foreign open intents without calling.
- The journal is bound to one run id, thread id, lease TTL and policy digest; a journal
  from another run or policy is refused before any lease call.
- Only definite refusals close an intent. An unknown outcome leaves it open.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import market_intelligence.admission as admission
import market_intelligence.control as control
from market_intelligence.gatekeeper import (
    Gatekeeper,
    GatekeeperClockError,
    GatekeeperLeaseError,
    GatekeeperPolicyMismatch,
    GatekeeperRequestError,
    GatekeeperStateError,
    GatekeeperUnknownRun,
    LeaseStatus,
)

NAMESPACE = uuid.UUID("5a0f2c7e-3b1d-4c6a-9e84-7d2b1f0a6c35")


class LifecycleError(RuntimeError):
    pass


class LifecycleBlocked(LifecycleError):
    """The contract says do not act now (live lease, no decision, unverifiable state)."""


class LifecycleJournalError(LifecycleError):
    """The intent journal is unreadable or torn; nothing may continue from it."""


@dataclass(frozen=True)
class Segment:
    owner: str
    epoch: int


def derive_event_id(run_id: str, event_class: str, epoch: int | None = None) -> str:
    """Stable UUID text from run id, event class and target epoch. The owner is not an input."""
    return str(uuid.uuid5(NAMESPACE, json.dumps([run_id, event_class, epoch])))


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


class IntentJournal:
    """Append-only JSON Lines file. It never truncates, replaces, renames or deletes."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)

    def append(self, record: dict) -> None:
        line = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
        with open(self.path, "ab") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        data = self.path.read_bytes()
        if data and not data.endswith(b"\n"):
            raise LifecycleJournalError("the journal ends with a partial line")
        out = []
        for number, raw in enumerate(data.split(b"\n")[:-1], start=1):
            try:
                value = json.loads(raw.decode("utf-8"))
            except ValueError as exc:
                raise LifecycleJournalError(f"journal line {number} is not valid JSON") from exc
            if not isinstance(value, dict) or "op" not in value:
                raise LifecycleJournalError(f"journal line {number} has no op")
            out.append(value)
        return out


def plan_recovery(status: LeaseStatus, now: datetime, decision: str | None) -> str:
    """Pure decision table. Returns an action name; it never calls anything."""
    if status.state == "verification_failed":
        return "stop_unverifiable"
    if status.state == "run_absent":
        return "stop_run_absent"
    if status.state == "present" and not status.released:
        if _utc(status.expires_at) > now:
            return "wait_live"
        return "takeover" if decision == "approved" else "needs_owner"
    # no_lease, or present and released
    return "acquire" if decision == "approved" else "wait_for_decision"


# Definite refusals only: the Gatekeeper raises these before anything is appended, and
# AuditIncomplete reaches the caller only after a confirmed rollback. ControlUnavailable
# (any append_state) and every other exception are NOT here: they leave the intent open.
_REFUSALS = (
    GatekeeperLeaseError,
    GatekeeperStateError,
    GatekeeperRequestError,
    GatekeeperUnknownRun,
    GatekeeperPolicyMismatch,
    GatekeeperClockError,
    control.AuditIncomplete,
)


def _policy_digest(policy: admission.PolicyConfig) -> str:
    values = {
        k: (sorted(v) if isinstance(v, (set, frozenset)) else v)
        for k, v in dataclasses.asdict(policy).items()
    }
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode("utf-8")).hexdigest()


class Coordinator:
    def __init__(
        self,
        store: control.AuditStore,
        policy: admission.PolicyConfig,
        journal: IntentJournal,
        *,
        run_id: str,
        thread_id: str,
        lease_ttl_seconds: int,
        now: Callable[[], datetime],
        new_owner: Callable[[], str] = lambda: str(uuid.uuid4()),
        fault: Callable[[str], None] | None = None,
    ) -> None:
        self._store = store
        self._policy = policy
        self._journal = journal
        self._run_id = run_id
        self._thread_id = thread_id
        self._ttl = lease_ttl_seconds
        self._now = now
        self._new_owner = new_owner
        self._fault = fault or (lambda point: None)
        self._incarnation = str(uuid.uuid4())
        self._header = {
            "op": "header",
            "run_id": run_id,
            "thread_id": thread_id,
            "lease_ttl_seconds": lease_ttl_seconds,
            "policy_sha256": _policy_digest(policy),
        }
        self._reader = self._gate(self._new_owner())

    def _gate(self, owner: str) -> Gatekeeper:
        return Gatekeeper(
            self._store,
            self._policy,
            run_id=self._run_id,
            thread_id=self._thread_id,
            owner_id=owner,
            lease_ttl_seconds=self._ttl,
            now=self._now,
        )

    # ------------------------------------------------------------ journal

    def _records(self) -> list[dict]:
        records = self._journal.records()
        if records and records[0] != self._header:
            raise LifecycleJournalError("the journal is bound to a different run, thread, TTL or policy")
        if any(r["op"] == "header" for r in records[1:]):
            raise LifecycleJournalError("the journal has a second header")
        return records

    def _append(self, record: dict) -> None:
        if not self._records():
            self._journal.append(self._header)
        self._journal.append(record)

    # ------------------------------------------------------------- reads

    def status(self) -> LeaseStatus:
        return self._reader.current_lease()

    def decision(self) -> str | None:
        found = None
        for rec in self._records():
            if rec["op"] == "decision":
                found = rec["value"]
        return found

    def still_holds(self, segment: Segment) -> bool:
        """Advisory only: true if a snapshot taken now shows this segment as the live lease."""
        st = self.status()
        return (
            st.state == "present"
            and st.released is False
            and st.owner_id == segment.owner
            and st.epoch == segment.epoch
            and _utc(st.expires_at) > self._now().astimezone(timezone.utc)
        )

    def _open_intents(self) -> list[dict]:
        state: dict[str, dict] = {}
        for rec in self._records():
            if rec["op"] == "lease_call":
                if rec["state"] == "intent":
                    state[rec["event_id"]] = rec
                else:
                    state.pop(rec["event_id"], None)
        return list(state.values())

    def _own_open(self) -> list[dict]:
        return [r for r in self._open_intents() if r["incarnation"] == self._incarnation]

    def _foreign_open(self) -> list[dict]:
        return [r for r in self._open_intents() if r["incarnation"] != self._incarnation]

    # ----------------------------------------------------------- one lease call

    def _attempt(self, call: str, owner: str | None, epoch: int | None, gate: Gatekeeper) -> str:
        event_id = derive_event_id(self._run_id, call, epoch)
        base = {
            "op": "lease_call", "call": call, "event_id": event_id, "owner": owner,
            "epoch": epoch, "incarnation": self._incarnation,
        }
        self._append({**base, "state": "intent"})
        self._fault("after_intent")
        return self._execute(base, gate)[0]

    def _attempt_epoch(self, call: str, owner: str, intended_epoch: int) -> int:
        """Run an acquire or takeover and return the epoch the audit chain actually recorded."""
        event_id = derive_event_id(self._run_id, call, intended_epoch)
        base = {
            "op": "lease_call", "call": call, "event_id": event_id, "owner": owner,
            "epoch": intended_epoch, "incarnation": self._incarnation,
        }
        self._append({**base, "state": "intent"})
        self._fault("after_intent")
        actual = self._execute(base, self._gate(owner))[1]
        return actual

    def _execute(self, base: dict, gate: Gatekeeper) -> tuple[str, int | None]:
        method = {
            "start": "start_run",
            "acquire": "acquire_lease",
            "release": "release_lease",
            "takeover": "take_over_lease",
        }[base["call"]]
        try:
            result = getattr(gate, method)(base["event_id"])
        except _REFUSALS as exc:
            self._append({**base, "state": "superseded", "why": type(exc).__name__})
            raise
        self._fault("after_call")
        actual = None
        if base["call"] in ("acquire", "takeover"):
            # The epoch was read from a snapshot before the call and can be stale; the Gatekeeper
            # decides the real epoch. Read what the chain recorded for this event id.
            seen = self._reader.event_status(base["event_id"])
            if seen.state != "present" or seen.lease_owner != base["owner"] or seen.lease_epoch is None:
                raise LifecycleError("the audit chain does not show this call as this owner's lease event")
            actual = seen.lease_epoch
        self._append({**base, "state": "done", "status": result.status, "actual_epoch": actual})
        return result.status, actual

    def resume_pending(self) -> list[tuple[str, str]]:
        """Retry each open intent written by THIS Coordinator object, once, with its own owner.

        This is the only place an owner id is reused: same event id, same incarnation.
        A definite refusal closes that intent for good. An unknown outcome leaves it open
        and raises. Intents from another incarnation are never touched here.
        """
        outcomes = []
        for rec in self._own_open():
            base = {k: rec[k] for k in ("op", "call", "event_id", "owner", "epoch", "incarnation")}
            gate = self._gate(rec["owner"]) if rec["owner"] else self._reader
            try:
                outcomes.append((rec["event_id"], self._execute(base, gate)[0]))
            except _REFUSALS:
                outcomes.append((rec["event_id"], "superseded"))
        return outcomes

    def resolve_foreign(self) -> list[tuple[str, str]]:
        """Close open intents written by another incarnation WITHOUT calling the Gatekeeper.

        The owner id in such an intent is never reused. The audit store is only read
        (event_status) and its answer is recorded. If the other process is still alive and
        later commits, the audit chain decides; this process stays on current_lease.
        """
        outcomes = []
        for rec in self._foreign_open():
            seen = self._reader.event_status(rec["event_id"])
            base = {k: rec[k] for k in ("op", "call", "event_id", "owner", "epoch", "incarnation")}
            self._append({**base, "state": "abandoned", "audit": seen.state})
            outcomes.append((rec["event_id"], seen.state))
        return outcomes

    # ------------------------------------------------------------ lifecycle

    def start(self) -> str:
        if self._open_intents():
            raise LifecycleBlocked("an earlier intent is still open; resolve it first")
        return self._attempt("start", None, None, self._reader)

    def _require_ready(self) -> LeaseStatus:
        if self._open_intents():
            raise LifecycleBlocked("an earlier intent is still open; resolve it first")
        st = self.status()
        if st.state in ("verification_failed", "run_absent"):
            raise LifecycleBlocked(f"lease state is {st.state}")
        return st

    def acquire(self, *, needs_approved_decision: bool) -> Segment:
        if needs_approved_decision and self.decision() != "approved":
            raise LifecycleBlocked("no approved decision is recorded")
        st = self._require_ready()
        if st.state == "present" and not st.released:
            raise LifecycleBlocked("an unreleased lease exists")
        epoch = 1 if st.state == "no_lease" else st.epoch + 1
        owner = self._new_owner()
        actual = self._attempt_epoch("acquire", owner, epoch)
        return Segment(owner, actual)

    def take_over(self) -> Segment:
        if self.decision() != "approved":
            raise LifecycleBlocked("no approved decision is recorded")
        st = self._require_ready()
        if st.state != "present" or st.released:
            raise LifecycleBlocked("there is no unreleased lease to take over")
        if _utc(st.expires_at) > self._now().astimezone(timezone.utc):
            raise LifecycleBlocked("the lease is still live by this process's clock")
        owner = self._new_owner()
        epoch = st.epoch + 1
        actual = self._attempt_epoch("takeover", owner, epoch)
        return Segment(owner, actual)

    def release(self, segment: Segment) -> str:
        st = self._require_ready()
        if not (
            st.state == "present"
            and st.released is False
            and st.owner_id == segment.owner
            and st.epoch == segment.epoch
        ):
            raise LifecycleBlocked("the segment does not hold the lease now")
        return self._attempt("release", segment.owner, segment.epoch, self._gate(segment.owner))

    # --------------------------------------------------------- decision, flows

    def decide(self, payload: object) -> str:
        """Record an approval payload before any lease call. Invalid input records nothing."""
        if not isinstance(payload, dict) or type(payload.get("approved")) is not bool:
            return "invalid"
        value = "approved" if payload["approved"] else "rejected"
        existing = self.decision()
        if existing is not None and existing != value:
            raise LifecycleBlocked("a different decision is already recorded")
        if existing is None:
            self._append({"op": "decision", "value": value})
        return value

    def run_first_segment(self, start_graph: Callable[[], str]) -> str:
        self.start()
        segment = self.acquire(needs_approved_decision=False)
        self._confirm_holding(segment)
        outcome = start_graph()
        if outcome not in ("paused", "finished"):
            raise LifecycleError(f"unexpected graph outcome {outcome!r}")
        self.release(segment)
        return outcome

    def _confirm_holding(self, segment: Segment) -> None:
        """Advisory read before the stub runs: the lease must still show this segment. Not a fence."""
        if not self.still_holds(segment):
            raise LifecycleBlocked("the lease does not show this segment as holder; the stub was not run")

    def _journal_segment(self) -> Segment | None:
        held = None
        for rec in self._records():
            if (
                rec["op"] == "lease_call"
                and rec["state"] == "done"
                and rec["incarnation"] == self._incarnation
            ):
                if rec["call"] in ("acquire", "takeover"):
                    held = Segment(rec["owner"], rec["actual_epoch"])
                elif rec["call"] == "release" and held and held.epoch == rec["epoch"]:
                    held = None
        return held

    def run_approved(self, resume_graph: Callable[[], str]) -> str:
        """Resume after an approval. Stub contract: resume_graph returns "finished" or "paused".

        "finished": mark resume_finished, then release. "paused" (the graph stopped again):
        release, do not mark finished, return "paused". Anything else, or an exception: the
        lease stays held, nothing is marked finished, and the caller decides. Resume is
        at-least-once; nothing here makes it once-only.
        """
        if self.decision() != "approved":
            return "no_approved_decision"
        self.resume_pending()
        self.resolve_foreign()
        finished = any(r["op"] == "resume_finished" for r in self._records())
        if finished:
            segment = self._journal_segment()
            if segment is not None and self.still_holds(segment):
                self.release(segment)
            return "already_finished"
        action = plan_recovery(self.status(), self._now().astimezone(timezone.utc), "approved")
        if action == "acquire":
            segment = self.acquire(needs_approved_decision=True)
        elif action == "takeover":
            segment = self.take_over()
        else:
            raise LifecycleBlocked(action)
        self._confirm_holding(segment)
        outcome = resume_graph()
        if outcome == "finished":
            self._append({"op": "resume_finished"})
            self._fault("after_resume")
            self.release(segment)
            return "finished"
        if outcome == "paused":
            self.release(segment)
            return "paused"
        raise LifecycleError(f"unexpected graph outcome {outcome!r}; the lease stays held")
