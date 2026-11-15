"""Outermost invocation guard for gatekeeper-backed structured model calls.

This module is standard-library only. It holds no transport, no provider client
and no gate implementation: it wraps an already-constructed structured model and
a narrow gate proxy, and it keeps in-memory evidence for one item/graph run.

Scope and non-claims:

* This is an offline-testable foundation, not a live runner.
* The estimator is an injected ESTIMATE of a full request. It is not a
  validated provider tokenizer, not a serialized-request measurement and not a
  spend cap.
* A delegate return is only ``record_call_returned``. It is not proof that a
  reservation was settled or that final ledger state was verified.
* Gate start/lease creation, settings identity alignment and provider model
  identity attestation are outside this component.

The guard serializes admitted invocations with one re-entrant lock. The lock is
held across estimation, admission, the inner model call, outcome recording and
the final fatal/receipt recheck, so a sibling that waits on the lock observes a
latched fatal condition before it can reach admission or transport.
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass

INT64_MAX = (1 << 63) - 1

GRAPH_SCHEMA_NAMES = frozenset(
    {
        "NewsDigest",
        "TopicAnalysis",
        "NewsToneDraft",
        "SideComparisonDraft",
    }
)


class InvocationGuardError(RuntimeError):
    """Base class for guard failures.

    Messages are fixed strings. They never carry prompts, responses, provider
    error text or secret material.
    """


class InvocationGuardReentryError(InvocationGuardError):
    """A nested admitted invocation was attempted inside one guarded call."""


class InvocationGuardContractError(InvocationGuardError):
    """An estimator, admission, usage or receipt contract was violated."""


class InvocationGuardFatalError(InvocationGuardError):
    """A fatal condition is latched; no further admitted call may start."""


@dataclass(frozen=True)
class InvocationEvidence:
    """Immutable snapshot of one guarded invocation.

    No prompt text, raw model response, provider error text or secret is kept.
    Requested and returned event IDs are separate fields. ``admission_allowed``
    records the delegate's explicit allow/deny value even when the attempt is
    rejected. ``admitted_reservation`` is captured only for a valid positively
    allowed decision; when accounting is incomplete it is the unresolved
    exposure figure, not a known charge and not a guaranteed maximum liability.
    ``exception_class`` is the primary/inner exception type; a separate
    ``accounting_exception_class`` records an outcome-writing failure.
    """

    invocation_id: str
    schema_name: str
    requested_decision_event_id: str | None
    returned_decision_event_id: str | None
    admission_allowed: bool | None
    admitted_reservation: int | None
    requested_outcome_event_id: str | None
    returned_outcome_event_id: str | None
    estimated_request_tokens: int | None
    known_usage: tuple[int, int] | None
    receipt_returned: bool
    unknown_usage: bool
    accounting_incomplete: bool
    fatal_reason: str | None
    exception_class: str | None
    accounting_exception_class: str | None
    overrun_tokens: int | None


@dataclass
class _Active:
    """Mutable per-invocation context; only the guard mutates it under lock."""

    invocation_id: str
    schema_name: str
    prompt: object
    estimated_request_tokens: int | None = None
    admission_attempted: bool = False
    requested_decision_event_id: str | None = None
    returned_decision_event_id: str | None = None
    admission_allowed: bool | None = None
    admitted_reservation: int | None = None
    requested_outcome_event_id: str | None = None
    returned_outcome_event_id: str | None = None
    known_usage: tuple[int, int] | None = None
    receipt_returned: bool = False
    unknown_usage: bool = False
    accounting_incomplete: bool = False
    fatal_reason: str | None = None
    exception_class: str | None = None
    accounting_exception_class: str | None = None
    overrun_tokens: int | None = None
    outcome_calls: int = 0
    finished: bool = False


def _valid_usage(prompt_tokens: object, completion_tokens: object) -> bool:
    """True only for exact non-negative signed-int64 integers with a bounded sum."""

    if type(prompt_tokens) is not int or type(completion_tokens) is not int:
        return False
    if prompt_tokens < 0 or completion_tokens < 0:
        return False
    if prompt_tokens > INT64_MAX or completion_tokens > INT64_MAX:
        return False
    return prompt_tokens + completion_tokens <= INT64_MAX


def _valid_positive_int(value: object) -> bool:
    return type(value) is int and 0 < value <= INT64_MAX


class InvocationGuard:
    """One guard instance per future single item/graph run.

    The guard owns a shared lock, the fatal latch, the active invocation context
    and the evidence list. Nothing here is global or shared between runs.
    """

    def __init__(
        self,
        *,
        expected_model: str,
        max_output_tokens: int,
        allowed_schema_names: frozenset[str],
        estimator,
    ) -> None:
        if not isinstance(expected_model, str) or not expected_model.strip():
            raise InvocationGuardContractError("expected_model must be a non-empty string")
        if expected_model != expected_model.strip():
            raise InvocationGuardContractError("expected_model must be trimmed")
        if not _valid_positive_int(max_output_tokens):
            raise InvocationGuardContractError("max_output_tokens must be a positive int")
        if not isinstance(allowed_schema_names, frozenset):
            raise InvocationGuardContractError("allowed_schema_names must be a frozenset")
        if allowed_schema_names != GRAPH_SCHEMA_NAMES:
            raise InvocationGuardContractError(
                "allowed_schema_names must equal the four graph schema names"
            )
        if not callable(estimator):
            raise InvocationGuardContractError("estimator must be callable")
        self._expected_model = expected_model
        self._max_output_tokens = max_output_tokens
        self._estimator = estimator
        self._lock = threading.RLock()
        self._fatal: str | None = None
        self._active: _Active | None = None
        self._evidence: list[InvocationEvidence] = []

    @property
    def fatal_reason(self) -> str | None:
        """The latched fatal reason code, or None."""

        return self._fatal

    def evidence(self) -> tuple[InvocationEvidence, ...]:
        """Return an immutable copy of the recorded evidence."""

        return tuple(self._evidence)

    def structured_model_wrapper(self, model):
        """Return the outermost wrapper to pass to ``build_graph``."""

        return _GuardedModel(self, model)

    def gate_proxy(self, gate):
        """Return the narrow two-method proxy to give ``GatekeeperLLM``."""

        if gate is None:
            raise InvocationGuardContractError("gate must be provided")
        return _GateProxy(self, gate)

    def token_counter(self, prompt):
        """Prompt-only callback for ``build_graph(token_counter=...)``.

        GatekeeperLLM calls this with only the prompt. The guard binds it to the
        active invocation context and refuses anything else, so a callback
        outside the guarded call (or with a different prompt object) fails
        closed instead of returning a fabricated count.
        """

        with self._lock:
            active = self._active
            if (
                active is None
                or active.estimated_request_tokens is None
                or prompt is not active.prompt
            ):
                raise InvocationGuardContractError(
                    "prompt counter used outside its active invocation"
                )
            return active.estimated_request_tokens

    def _latch(self, active: _Active | None, reason: str) -> None:
        if active is not None and active.fatal_reason is None:
            active.fatal_reason = reason
        if self._fatal is None:
            self._fatal = reason

    def _finish(self, active: _Active) -> None:
        if active.finished:
            return
        active.finished = True
        self._evidence.append(
            InvocationEvidence(
                invocation_id=active.invocation_id,
                schema_name=active.schema_name,
                requested_decision_event_id=active.requested_decision_event_id,
                returned_decision_event_id=active.returned_decision_event_id,
                admission_allowed=active.admission_allowed,
                admitted_reservation=active.admitted_reservation,
                requested_outcome_event_id=active.requested_outcome_event_id,
                returned_outcome_event_id=active.returned_outcome_event_id,
                estimated_request_tokens=active.estimated_request_tokens,
                known_usage=active.known_usage,
                receipt_returned=active.receipt_returned,
                unknown_usage=active.unknown_usage,
                accounting_incomplete=active.accounting_incomplete,
                fatal_reason=active.fatal_reason,
                exception_class=active.exception_class,
                accounting_exception_class=active.accounting_exception_class,
                overrun_tokens=active.overrun_tokens,
            )
        )

    def _estimate(self, schema, prompt) -> int:
        try:
            value = self._estimator(schema, prompt)
        except InvocationGuardReentryError:
            self._latch(self._active, "reentry")
            raise
        except Exception as exc:
            if self._active is not None:
                self._active.exception_class = type(exc).__name__
                self._latch(self._active, "estimator-error")
            raise InvocationGuardContractError("estimator failed") from None
        if not _valid_positive_int(value):
            self._latch(self._active, "estimator-invalid")
            raise InvocationGuardContractError("invalid estimate")
        return value


class _GuardedModel:
    """Outermost structured-model wrapper."""

    def __init__(self, guard: InvocationGuard, inner) -> None:
        self._guard = guard
        self._inner = inner

    def with_structured_output(self, schema, **kwargs):
        if kwargs:
            self._guard._latch(None, "structured-options-not-supported")
            raise InvocationGuardContractError(
                "structured-output options are not supported"
            )
        return _GuardedStructured(self._guard, self._inner, schema)


class _GuardedStructured:
    """One lazily created structured runnable for a schema."""

    def __init__(self, guard: InvocationGuard, inner, schema) -> None:
        self._guard = guard
        self._inner = inner
        self._schema = schema

    def invoke(self, prompt, **kwargs):
        guard = self._guard
        with guard._lock:
            if guard._fatal is not None:
                raise InvocationGuardFatalError("invocation guard is latched")
            if guard._active is not None:
                guard._latch(guard._active, "reentry")
                raise InvocationGuardReentryError("nested admitted invocation")
            name = getattr(self._schema, "__name__", None)
            active = _Active(
                invocation_id=str(uuid.uuid4()),
                schema_name=name if isinstance(name, str) else "",
                prompt=prompt,
            )
            guard._active = active
            try:
                if not isinstance(name, str) or name not in GRAPH_SCHEMA_NAMES:
                    guard._latch(active, "schema-not-allowed")
                    raise InvocationGuardContractError("schema is not allowed")
                active.estimated_request_tokens = guard._estimate(self._schema, prompt)
                if guard._fatal is not None:
                    raise InvocationGuardFatalError(
                        "invocation guard is latched after estimation"
                    )
                try:
                    inner = self._inner.with_structured_output(self._schema)
                    if guard._fatal is not None:
                        raise InvocationGuardFatalError(
                            "invocation guard is latched before the inner call"
                        )
                    value = inner.invoke(prompt, **kwargs)
                except Exception as exc:
                    active.exception_class = type(exc).__name__
                    if guard._fatal is None:
                        guard._latch(active, "inner-error")
                    raise
                if guard._fatal is not None:
                    guard._latch(active, "return-after-fatal")
                    active.exception_class = "InvocationGuardFatalError"
                    raise InvocationGuardFatalError(
                        "guarded call returned after a fatal condition"
                    )
                if not active.receipt_returned:
                    guard._latch(active, "outcome-missing")
                    raise InvocationGuardContractError("no outcome receipt")
                return value
            finally:
                guard._finish(active)
                guard._active = None


class _GateProxy:
    """Narrow gate proxy exposing only ``admit_llm`` and ``record_llm_outcome``.

    No ``__getattr__`` delegation is defined, so no other gate mutation method is
    reachable through this object.
    """

    def __init__(self, guard: InvocationGuard, gate) -> None:
        self._guard = guard
        self._gate = gate

    def admit_llm(self, request, event_id):
        guard = self._guard
        with guard._lock:
            active = guard._active
            if guard._fatal is not None:
                raise InvocationGuardFatalError("invocation guard is latched")
            if active is None:
                guard._latch(None, "admission-outside-invocation")
                raise InvocationGuardContractError("admission outside an invocation")
            if active.admission_attempted:
                guard._latch(active, "duplicate-admission")
                raise InvocationGuardContractError("duplicate admission attempt")
            if not isinstance(event_id, str) or not event_id:
                guard._latch(active, "event-id-invalid")
                raise InvocationGuardContractError("decision event id is invalid")
            if not _request_matches(request, guard, active):
                guard._latch(active, "admission-mismatch")
                raise InvocationGuardContractError("admission request does not match")
            active.admission_attempted = True
            active.requested_decision_event_id = event_id
            try:
                result = self._gate.admit_llm(request, event_id)
            except Exception as exc:
                active.exception_class = type(exc).__name__
                guard._latch(active, "admission-error")
                raise
            decision = getattr(result, "decision", None)
            allow = getattr(decision, "allow", None)
            if allow is not True and allow is not False:
                guard._latch(active, "admission-decision-invalid")
                raise InvocationGuardContractError("admission decision is invalid")
            active.admission_allowed = allow
            reserve = getattr(decision, "reserve_tokens", None)
            if allow is True and _valid_positive_int(reserve):
                active.admitted_reservation = reserve
            returned_id = getattr(result, "event_id", None)
            if isinstance(returned_id, str):
                active.returned_decision_event_id = returned_id
            if not isinstance(returned_id, str) or returned_id != event_id:
                guard._latch(active, "admission-receipt-mismatch")
                raise InvocationGuardContractError("admission receipt mismatch")
            if allow is not True:
                guard._latch(active, "admission-denied")
                raise InvocationGuardFatalError("admission denied")
            if not _valid_positive_int(reserve):
                guard._latch(active, "reservation-invalid")
                raise InvocationGuardContractError("invalid admission reservation")
            active.admitted_reservation = reserve
            return result

    def record_llm_outcome(
        self,
        decision_event_id,
        prompt_tokens,
        completion_tokens,
        event_id,
    ):
        guard = self._guard
        with guard._lock:
            active = guard._active
            if active is None:
                guard._latch(None, "outcome-outside-invocation")
                raise InvocationGuardContractError("outcome outside an invocation")
            if (
                active.requested_decision_event_id is None
                or decision_event_id != active.requested_decision_event_id
            ):
                guard._latch(active, "outcome-decision-mismatch")
                raise InvocationGuardContractError("outcome does not match its decision")
            if active.outcome_calls:
                guard._latch(active, "duplicate-outcome")
                raise InvocationGuardContractError("duplicate outcome")
            if not isinstance(event_id, str) or not event_id:
                guard._latch(active, "event-id-invalid")
                raise InvocationGuardContractError("outcome event id is invalid")
            if (
                active.admission_allowed is not True
                or not _valid_positive_int(active.admitted_reservation)
            ):
                guard._latch(active, "outcome-without-admission")
                raise InvocationGuardContractError(
                    "outcome requires an allowed admission with a valid reservation"
                )
            active.outcome_calls = 1
            active.requested_outcome_event_id = event_id
            valid = _valid_usage(prompt_tokens, completion_tokens)
            if valid:
                active.known_usage = (prompt_tokens, completion_tokens)
                if (
                    prompt_tokens + completion_tokens
                    > active.admitted_reservation
                ):
                    active.overrun_tokens = (
                        prompt_tokens
                        + completion_tokens
                        - active.admitted_reservation
                    )
                    guard._latch(active, "usage-overrun")
            else:
                active.unknown_usage = True
                guard._latch(active, "usage-unknown")
            try:
                result = self._gate.record_llm_outcome(
                    decision_event_id,
                    prompt_tokens,
                    completion_tokens,
                    event_id,
                )
            except Exception as exc:
                active.accounting_incomplete = True
                active.accounting_exception_class = type(exc).__name__
                guard._latch(active, "outcome-record-error")
                raise
            returned_id = getattr(result, "event_id", None)
            if isinstance(returned_id, str):
                active.returned_outcome_event_id = returned_id
            if not isinstance(returned_id, str) or returned_id != event_id:
                active.accounting_incomplete = True
                guard._latch(active, "outcome-receipt-mismatch")
                raise InvocationGuardContractError("outcome receipt mismatch")
            active.receipt_returned = True
            return result


def _request_matches(request, guard: InvocationGuard, active: _Active) -> bool:
    model = getattr(request, "model", None)
    schema_name = getattr(request, "schema_name", None)
    cap = getattr(request, "max_output_tokens", None)
    count = getattr(request, "serialized_body_token_count", None)
    return (
        isinstance(model, str)
        and model == guard._expected_model
        and isinstance(schema_name, str)
        and schema_name == active.schema_name
        and _valid_positive_int(cap)
        and cap == guard._max_output_tokens
        and _valid_positive_int(count)
        and count == active.estimated_request_tokens
    )
