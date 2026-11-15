"""Offline tests for the outermost invocation guard.

Authored and statically inspected only. These tests are NOT run by the correction
Build that wrote them. No network, credentials, datasets, manifest, reports,
temporary directories or filesystem writes are used here. All gates, clients and
estimators are in-memory stubs; the real project wrappers are used only where an
integration assertion needs them.
"""
import inspect
import threading
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from eval.invocation_guard import (
    GRAPH_SCHEMA_NAMES,
    INT64_MAX,
    InvocationEvidence,
    InvocationGuard,
    InvocationGuardContractError,
    InvocationGuardFatalError,
    InvocationGuardReentryError,
)
from market_intelligence.budget import BudgetedLLM, RunTracker
from market_intelligence.gatekeeper_llm import GatekeeperLLM
from market_intelligence.graph import supervisor


class NewsDigest:
    """Stand-in for the graph's news schema (name is what the guard checks)."""


class TopicAnalysis:
    """Stand-in for the graph's topic schema."""


class NewsToneDraft:
    """Stand-in for the graph's tone schema."""


class SideComparisonDraft:
    """Stand-in for the graph's comparison schema."""


class BogusSchema:
    """A schema name the guard must refuse."""


class StubGate:
    """In-memory stand-in for Gatekeeper (admission and outcome only)."""

    def __init__(
        self,
        *,
        allow=True,
        reserve=12,
        admission="ok",
        admit_receipt=None,
        outcome_receipt=None,
        fail_admit=False,
        fail_outcome=False,
    ):
        self.allow = allow
        self.reserve = reserve
        self.admission = admission
        self.admit_receipt = admit_receipt
        self.outcome_receipt = outcome_receipt
        self.fail_admit = fail_admit
        self.fail_outcome = fail_outcome
        self.admit_calls = []
        self.outcome_calls = []

    def admit_llm(self, request, event_id):
        self.admit_calls.append((request, event_id))
        if self.fail_admit:
            raise RuntimeError("admission delegate failed")
        if self.admission == "missing":
            return SimpleNamespace(status="applied", event_id=event_id, decision=None)
        decision = SimpleNamespace(
            allow=self.allow,
            reserve_tokens=self.reserve,
            reason=SimpleNamespace(value="stub"),
        )
        receipt = self.admit_receipt if self.admit_receipt is not None else event_id
        return SimpleNamespace(status="applied", event_id=receipt, decision=decision)

    def record_llm_outcome(
        self, decision_event_id, prompt_tokens, completion_tokens, event_id
    ):
        self.outcome_calls.append(
            (decision_event_id, prompt_tokens, completion_tokens, event_id)
        )
        if self.fail_outcome:
            raise RuntimeError("outcome delegate failed")
        receipt = (
            self.outcome_receipt
            if self.outcome_receipt is not None
            else event_id
        )
        return SimpleNamespace(status="applied", event_id=receipt)


class StubClient:
    """In-memory structured client that can emit usage or raise."""

    def __init__(self, *, value="model-value", error=None, usage=(3, 2), on_invoke=None):
        self.value = value
        self.error = error
        self.usage = usage
        self.on_invoke = on_invoke
        self.calls = 0
        self.schemas = []

    def with_structured_output(self, schema, **kwargs):
        return _StubRunnable(self, schema)


class _StubRunnable:
    def __init__(self, client, schema):
        self._client = client
        self._schema = schema

    def invoke(self, prompt, **kwargs):
        client = self._client
        client.calls += 1
        client.schemas.append(self._schema.__name__)
        if client.on_invoke is not None:
            client.on_invoke()
        if client.error is not None:
            raise client.error
        if client.usage is not None:
            response = SimpleNamespace(
                llm_output={
                    "token_usage": {
                        "prompt_tokens": client.usage[0],
                        "completion_tokens": client.usage[1],
                    }
                },
                generations=[],
            )
            for callback in kwargs.get("config", {}).get("callbacks", []) or []:
                callback.on_llm_end(response)
        return client.value


def _guard(expected_model="gpt-test", max_output_tokens=32, estimator=None):
    return InvocationGuard(
        expected_model=expected_model,
        max_output_tokens=max_output_tokens,
        allowed_schema_names=GRAPH_SCHEMA_NAMES,
        estimator=estimator or (lambda schema, prompt: 10),
    )


def _stack(
    *,
    guard=None,
    gate=None,
    client=None,
    model_name="gpt-test",
    max_output_tokens=32,
    token_counter="guard",
):
    guard = guard or _guard()
    gate = gate or StubGate()
    client = client or StubClient()
    proxy = guard.gate_proxy(gate)
    counter = guard.token_counter if token_counter == "guard" else token_counter
    llm = GatekeeperLLM(
        BudgetedLLM(client, RunTracker(max_tokens=10_000, max_calls=32)),
        proxy,
        model_name=model_name,
        max_output_tokens=max_output_tokens,
        token_counter=counter,
    )
    return guard, gate, client, proxy, guard.structured_model_wrapper(llm)


class RecordingGraph:
    """Records node registrations instead of compiling a LangGraph."""

    last = None

    def __init__(self, state):
        self.state = state
        self.nodes = {}
        RecordingGraph.last = self

    def add_node(self, name, action, **kwargs):
        self.nodes[name] = {"action": action, **kwargs}

    def add_edge(self, *args, **kwargs):
        return None

    def add_conditional_edges(self, *args, **kwargs):
        return None

    def compile(self, checkpointer=None):
        return self


class ScriptedInner:
    """Drives the proxy directly, allowing malformed admission/outcome patterns."""

    def __init__(
        self,
        guard,
        proxy,
        *,
        usage=(3, 2),
        outcome="once",
        admit_calls=1,
        admit_after_outcome=False,
        outcome_after_failed_admission=False,
        transport=True,
        cap=None,
        count=None,
        constructor_error=None,
        on_before_invoke=None,
        on_after_admit=None,
        value="scripted",
    ):
        self._guard = guard
        self._proxy = proxy
        self._usage = usage
        self._outcome = outcome
        self._admit_calls = admit_calls
        self._admit_after_outcome = admit_after_outcome
        self._outcome_after_failed_admission = outcome_after_failed_admission
        self._transport = transport
        self._cap = cap
        self._count = count
        self._constructor_error = constructor_error
        self._on_before_invoke = on_before_invoke
        self._on_after_admit = on_after_admit
        self._value = value
        self.transport_count = 0
        self.constructor_calls = 0

    def with_structured_output(self, schema, **kwargs):
        self.constructor_calls += 1
        if self._constructor_error is not None:
            raise self._constructor_error
        return _ScriptedRunnable(self, schema)


class _ScriptedRunnable:
    def __init__(self, inner, schema):
        self._inner = inner
        self._schema = schema

    def invoke(self, prompt, **kwargs):
        inner = self._inner
        guard = inner._guard
        proxy = inner._proxy
        if inner._on_before_invoke is not None:
            inner._on_before_invoke()
        cap = guard._max_output_tokens if inner._cap is None else inner._cap
        count = guard.token_counter(prompt) if inner._count is None else inner._count
        request = SimpleNamespace(
            model=guard._expected_model,
            schema_name=self._schema.__name__,
            max_output_tokens=cap,
            serialized_body_token_count=count,
        )
        decision_id = "decision-" + self._schema.__name__
        outcome_id = "outcome-" + self._schema.__name__
        try:
            proxy.admit_llm(request, decision_id)
        except Exception:
            if not inner._outcome_after_failed_admission:
                raise
            try:
                proxy.record_llm_outcome(
                    decision_id, inner._usage[0], inner._usage[1], outcome_id
                )
            except Exception:
                pass
            return inner._value
        if inner._admit_calls > 1:
            proxy.admit_llm(request, decision_id)
            return inner._value
        if inner._transport:
            inner.transport_count += 1
        if inner._on_after_admit is not None:
            inner._on_after_admit()
        if inner._outcome == "none":
            return inner._value
        if inner._outcome == "mismatch":
            proxy.record_llm_outcome(
                "wrong-decision", inner._usage[0], inner._usage[1], outcome_id
            )
            return inner._value
        if inner._outcome == "twice":
            proxy.record_llm_outcome(
                decision_id, inner._usage[0], inner._usage[1], outcome_id
            )
            proxy.record_llm_outcome(
                decision_id, inner._usage[0], inner._usage[1], outcome_id + "-2"
            )
            return inner._value
        proxy.record_llm_outcome(
            decision_id, inner._usage[0], inner._usage[1], outcome_id
        )
        if inner._admit_after_outcome:
            proxy.admit_llm(request, "late-decision")
        return inner._value


def _scripted(**kwargs):
    guard = kwargs.pop("guard", None) or _guard()
    gate = kwargs.pop("gate", None) or StubGate()
    proxy = guard.gate_proxy(gate)
    inner = ScriptedInner(guard, proxy, **kwargs)
    return guard, gate, proxy, inner, guard.structured_model_wrapper(inner)


def test_build_graph_defaults_and_retry_policy_unchanged():
    signature = inspect.signature(supervisor.build_graph)
    assert signature.parameters["retry_policy"].default is None
    assert signature.parameters["structured_model_wrapper"].default is None
    assert supervisor.RETRY_POLICY.max_attempts == 3


def test_one_attempt_policy_applies_to_three_nodes_and_wrapper_is_outermost(
    monkeypatch,
):
    monkeypatch.setattr(supervisor, "StateGraph", RecordingGraph)
    one_attempt = supervisor.RetryPolicy(max_attempts=1)
    gate = StubGate()
    captured = {}

    def wrapper(model):
        captured["model"] = model
        return model

    supervisor.build_graph(
        llm=StubClient(),
        writer=SimpleNamespace(persist=lambda *args: "writer"),
        gatekeeper=gate,
        structured_model_wrapper=wrapper,
        retry_policy=one_attempt,
    )
    nodes = RecordingGraph.last.nodes
    for name in ("fetch_news", "identify_topics", "analyze_news_tone"):
        assert nodes[name]["retry_policy"] is one_attempt
        assert nodes[name]["retry_policy"].max_attempts == 1
    assert "retry_policy" not in nodes["research_competitor"]
    assert isinstance(captured["model"], GatekeeperLLM)
    assert captured["model"]._gatekeeper is gate

    supervisor.build_graph(
        llm=StubClient(),
        writer=SimpleNamespace(persist=lambda *args: "writer"),
    )
    default_nodes = RecordingGraph.last.nodes
    for name in ("fetch_news", "identify_topics", "analyze_news_tone"):
        assert default_nodes[name]["retry_policy"] is supervisor.RETRY_POLICY
    assert "retry_policy" not in default_nodes["research_competitor"]


@pytest.mark.parametrize("bad_estimate", [None, 0, -1, True, INT64_MAX + 1])
def test_invalid_estimates_fail_closed_before_admission(bad_estimate):
    guard = _guard(estimator=lambda schema, prompt: bad_estimate)
    _, gate, client, _, wrapped = _stack(guard=guard)
    with pytest.raises(InvocationGuardContractError):
        wrapped.with_structured_output(NewsDigest).invoke("prompt")
    assert gate.admit_calls == []
    assert client.calls == 0
    assert guard.fatal_reason == "estimator-invalid"


def test_missing_or_raising_estimator_fails_closed():
    with pytest.raises(InvocationGuardContractError):
        InvocationGuard(
            expected_model="gpt-test",
            max_output_tokens=32,
            allowed_schema_names=GRAPH_SCHEMA_NAMES,
            estimator=None,
        )

    def boom(schema, prompt):
        raise RuntimeError("estimator detail")

    guard = _guard(estimator=boom)
    _, gate, client, _, wrapped = _stack(guard=guard)
    with pytest.raises(InvocationGuardContractError):
        wrapped.with_structured_output(NewsDigest).invoke("prompt")
    assert gate.admit_calls == []
    assert client.calls == 0
    assert guard.fatal_reason == "estimator-error"


def test_schema_model_and_cap_mismatches_never_admit():
    guard = _guard()
    _, gate, client, _, wrapped = _stack(guard=guard)
    with pytest.raises(InvocationGuardContractError):
        wrapped.with_structured_output(BogusSchema).invoke("prompt")
    assert gate.admit_calls == []
    assert client.calls == 0

    guard = _guard()
    _, gate, client, _, wrapped = _stack(guard=guard, model_name="other-model")
    with pytest.raises(InvocationGuardContractError):
        wrapped.with_structured_output(NewsDigest).invoke("prompt")
    assert gate.admit_calls == []
    assert client.calls == 0

    guard = _guard()
    _, gate, client, _, wrapped = _stack(guard=guard, max_output_tokens=64)
    with pytest.raises(InvocationGuardContractError):
        wrapped.with_structured_output(NewsDigest).invoke("prompt")
    assert gate.admit_calls == []
    assert client.calls == 0

    guard = _guard()
    _, gate, client, _, wrapped = _stack(guard=guard, token_counter=lambda prompt: 11)
    with pytest.raises(InvocationGuardContractError):
        wrapped.with_structured_output(NewsDigest).invoke("prompt")
    assert gate.admit_calls == []
    assert client.calls == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cap": True},
        {"cap": 32.0},
        {"count": 10.0},
        {"count": True},
    ],
)
def test_exact_integer_types_reject_numeric_equality_traps(kwargs):
    guard, gate, _, inner, wrapped = _scripted(**kwargs)
    with pytest.raises(InvocationGuardContractError):
        wrapped.with_structured_output(NewsDigest).invoke("prompt")
    assert gate.admit_calls == []
    assert inner.transport_count == 0
    assert guard.fatal_reason == "admission-mismatch"


def test_estimator_receives_the_active_schema_once():
    seen = []

    def estimator(schema, prompt):
        seen.append(schema.__name__)
        return 7

    guard = _guard(estimator=estimator)
    _, gate, client, _, wrapped = _stack(guard=guard)
    assert wrapped.with_structured_output(NewsDigest).invoke("prompt") == "model-value"
    assert seen == ["NewsDigest"]
    assert len(gate.admit_calls) == 1
    assert client.calls == 1


def test_success_records_estimate_reservation_and_receipt():
    guard, gate, client, _, wrapped = _stack()
    result = wrapped.with_structured_output(NewsDigest).invoke("prompt")
    assert result == "model-value"
    evidence = guard.evidence()
    assert len(evidence) == 1
    record = evidence[0]
    assert isinstance(record, InvocationEvidence)
    assert record.schema_name == "NewsDigest"
    assert record.estimated_request_tokens == 10
    assert record.admitted_reservation == 12
    assert record.admission_allowed is True
    assert record.requested_decision_event_id is not None
    assert record.returned_decision_event_id == record.requested_decision_event_id
    assert record.requested_outcome_event_id is not None
    assert record.returned_outcome_event_id == record.requested_outcome_event_id
    assert record.known_usage == (3, 2)
    assert record.receipt_returned is True
    assert record.unknown_usage is False
    assert record.accounting_incomplete is False
    assert record.fatal_reason is None
    assert record.exception_class is None
    assert record.accounting_exception_class is None
    assert guard.fatal_reason is None
    assert len(gate.admit_calls) == 1
    assert len(gate.outcome_calls) == 1
    with pytest.raises(InvocationGuardContractError):
        guard.token_counter("prompt")


def test_prompt_identity_is_rejected():
    holder = {}

    def wrong_prompt_estimator(schema, prompt):
        return holder["guard"].token_counter("a different prompt")

    guard = _guard(estimator=wrong_prompt_estimator)
    holder["guard"] = guard
    _, gate, client, _, wrapped = _stack(guard=guard)
    with pytest.raises(InvocationGuardContractError):
        wrapped.with_structured_output(NewsDigest).invoke("prompt")
    assert gate.admit_calls == []
    assert client.calls == 0


def test_caught_nested_reentry_in_estimator_still_latches():
    holder = {}

    def nested_estimator(schema, prompt):
        try:
            holder["wrapped"].with_structured_output(TopicAnalysis).invoke("nested")
        except InvocationGuardReentryError:
            pass
        return 10

    guard = _guard(estimator=nested_estimator)
    holder["guard"] = guard
    _, gate, _, inner, wrapped = _scripted(guard=guard)
    holder["wrapped"] = wrapped
    with pytest.raises(InvocationGuardFatalError):
        wrapped.with_structured_output(NewsDigest).invoke("prompt")
    assert guard.fatal_reason == "reentry"
    assert inner.constructor_calls == 0
    assert gate.admit_calls == []
    assert inner.transport_count == 0


def test_caught_nested_reentry_inside_inner_call_still_latches():
    holder = {}

    def swallow_nested():
        try:
            holder["wrapped"].with_structured_output(TopicAnalysis).invoke("nested")
        except InvocationGuardReentryError:
            pass

    guard, gate, _, inner, wrapped = _scripted()
    holder["wrapped"] = wrapped
    inner._on_after_admit = swallow_nested
    with pytest.raises(InvocationGuardFatalError):
        wrapped.with_structured_output(NewsDigest).invoke("prompt")
    assert guard.fatal_reason == "reentry"
    assert len(gate.admit_calls) == 1
    assert inner.transport_count == 1
    assert len(gate.outcome_calls) == 1


def test_missing_usage_latches_and_keeps_the_reservation():
    guard, gate, client, _, wrapped = _stack(client=StubClient(usage=None))
    with pytest.raises(InvocationGuardFatalError):
        wrapped.with_structured_output(NewsDigest).invoke("prompt")
    record = guard.evidence()[0]
    assert record.unknown_usage is True
    assert record.known_usage is None
    assert record.admitted_reservation == 12
    assert record.receipt_returned is True
    assert guard.fatal_reason == "usage-unknown"
    assert gate.outcome_calls[0][1:3] == (None, None)


@pytest.mark.parametrize(
    "usage",
    [
        (True, 2),
        (3.0, 2),
        (-1, 2),
        (INT64_MAX, INT64_MAX),
    ],
)
def test_malformed_usage_is_forwarded_unchanged_and_never_succeeds(usage):
    guard, gate, _, inner, wrapped = _scripted(usage=usage)
    with pytest.raises(InvocationGuardFatalError):
        wrapped.with_structured_output(NewsDigest).invoke("prompt")
    record = guard.evidence()[0]
    assert record.unknown_usage is True
    assert record.known_usage is None
    assert record.receipt_returned is True
    assert len(gate.outcome_calls) == 1
    assert gate.outcome_calls[0][1:3] == usage
    assert guard.fatal_reason == "usage-unknown"
    assert inner.transport_count == 1


def test_distinct_provider_and_accounting_exception_types_are_kept():
    provider_error = ValueError("provider detail")
    guard, gate, client, _, wrapped = _stack(
        gate=StubGate(fail_outcome=True), client=StubClient(error=provider_error)
    )
    with pytest.raises(ValueError) as excinfo:
        wrapped.with_structured_output(NewsDigest).invoke("prompt")
    assert excinfo.value is provider_error
    record = guard.evidence()[0]
    assert record.exception_class == "ValueError"
    assert record.accounting_exception_class == "RuntimeError"
    assert record.accounting_incomplete is True
    assert record.admitted_reservation == 12
    assert record.receipt_returned is False
    assert guard.fatal_reason is not None
    assert client.calls == 1
    assert len(gate.admit_calls) == 1
    with pytest.raises(InvocationGuardFatalError):
        wrapped.with_structured_output(TopicAnalysis).invoke("second")
    assert client.calls == 1


def test_outcome_write_error_is_retained_and_later_calls_refused():
    guard, gate, client, _, wrapped = _stack(gate=StubGate(fail_outcome=True))
    with pytest.raises(RuntimeError):
        wrapped.with_structured_output(NewsDigest).invoke("prompt")
    record = guard.evidence()[0]
    assert record.accounting_incomplete is True
    assert record.accounting_exception_class == "RuntimeError"
    assert record.exception_class == "RuntimeError"
    assert guard.fatal_reason == "outcome-record-error"
    with pytest.raises(InvocationGuardFatalError):
        wrapped.with_structured_output(TopicAnalysis).invoke("second")
    assert client.calls == 1
    assert len(gate.admit_calls) == 1
    assert len(gate.outcome_calls) == 1


def test_overrun_is_recorded_before_outcome_failure():
    guard, gate, _, inner, wrapped = _scripted(
        usage=(10, 10), gate=StubGate(reserve=5, fail_outcome=True)
    )
    with pytest.raises(RuntimeError):
        wrapped.with_structured_output(NewsDigest).invoke("prompt")
    record = guard.evidence()[0]
    assert record.overrun_tokens == 15
    assert record.known_usage == (10, 10)
    assert record.accounting_incomplete is True
    assert guard.fatal_reason == "usage-overrun"
    assert gate.outcome_calls[0][1:3] == (10, 10)


def test_duplicate_admission_before_outcome_is_refused():
    guard, gate, _, inner, wrapped = _scripted(admit_calls=2)
    with pytest.raises(InvocationGuardContractError):
        wrapped.with_structured_output(NewsDigest).invoke("prompt")
    assert guard.fatal_reason == "duplicate-admission"
    assert len(gate.admit_calls) == 1
    assert inner.transport_count == 0


def test_duplicate_admission_after_outcome_is_refused():
    guard, gate, _, inner, wrapped = _scripted(admit_after_outcome=True)
    with pytest.raises(InvocationGuardContractError):
        wrapped.with_structured_output(NewsDigest).invoke("prompt")
    assert guard.fatal_reason == "duplicate-admission"
    assert len(gate.admit_calls) == 1
    assert len(gate.outcome_calls) == 1
    assert inner.transport_count == 1


def test_missing_decision_is_refused():
    guard, gate, _, inner, wrapped = _scripted(
        gate=StubGate(admission="missing")
    )
    with pytest.raises(InvocationGuardContractError):
        wrapped.with_structured_output(NewsDigest).invoke("prompt")
    assert guard.fatal_reason == "admission-decision-invalid"
    assert guard.evidence()[0].admission_allowed is None
    assert len(gate.admit_calls) == 1
    assert inner.transport_count == 0


@pytest.mark.parametrize("reserve", [True, 0, -1, 12.0])
def test_invalid_reservation_is_refused_after_recording_the_allow(reserve):
    guard, gate, _, inner, wrapped = _scripted(
        gate=StubGate(reserve=reserve)
    )
    with pytest.raises(InvocationGuardContractError):
        wrapped.with_structured_output(NewsDigest).invoke("prompt")
    record = guard.evidence()[0]
    assert record.admission_allowed is True
    assert record.admitted_reservation is None
    assert guard.fatal_reason == "reservation-invalid"
    assert inner.transport_count == 0


def test_admission_receipt_mismatch_keeps_the_valid_reservation():
    guard, gate, _, inner, wrapped = _scripted(
        gate=StubGate(admit_receipt="wrong-id")
    )
    with pytest.raises(InvocationGuardContractError):
        wrapped.with_structured_output(NewsDigest).invoke("prompt")
    record = guard.evidence()[0]
    assert record.admitted_reservation == 12
    assert record.returned_decision_event_id == "wrong-id"
    assert guard.fatal_reason == "admission-receipt-mismatch"
    assert inner.transport_count == 0


def test_denied_admission_never_reaches_transport():
    guard, gate, _, inner, wrapped = _scripted(gate=StubGate(allow=False))
    with pytest.raises(InvocationGuardFatalError):
        wrapped.with_structured_output(NewsDigest).invoke("prompt")
    record = guard.evidence()[0]
    assert record.admission_allowed is False
    assert record.admitted_reservation is None
    assert guard.fatal_reason == "admission-denied"
    assert inner.transport_count == 0
    assert len(gate.admit_calls) == 1


def test_non_string_admission_receipt_is_not_retained():
    guard, gate, _, inner, wrapped = _scripted(gate=StubGate(admit_receipt=123))
    with pytest.raises(InvocationGuardContractError):
        wrapped.with_structured_output(NewsDigest).invoke("prompt")
    record = guard.evidence()[0]
    assert record.returned_decision_event_id is None
    assert record.admitted_reservation == 12
    assert guard.fatal_reason == "admission-receipt-mismatch"
    assert inner.transport_count == 0


@pytest.mark.parametrize(
    "gate_kwargs,allowed,reason",
    [
        ({"allow": False}, False, "admission-denied"),
        ({"admission": "missing"}, None, "admission-decision-invalid"),
        ({"reserve": True}, True, "reservation-invalid"),
    ],
)
@pytest.mark.parametrize("usage", [(3, 2), (None, None)])
def test_outcome_after_failed_admission_is_refused(
    gate_kwargs, allowed, reason, usage
):
    guard, gate, _, inner, wrapped = _scripted(
        gate=StubGate(**gate_kwargs),
        usage=usage,
        outcome_after_failed_admission=True,
    )
    with pytest.raises(InvocationGuardFatalError):
        wrapped.with_structured_output(NewsDigest).invoke("prompt")
    record = guard.evidence()[0]
    assert record.admission_allowed is allowed
    assert record.admitted_reservation is None
    assert record.requested_outcome_event_id is None
    assert record.receipt_returned is False
    assert guard.fatal_reason == reason
    assert gate.outcome_calls == []
    assert inner.transport_count == 0


def test_missing_outcome_is_refused():
    guard, gate, _, _, wrapped = _scripted(outcome="none")
    with pytest.raises(InvocationGuardContractError):
        wrapped.with_structured_output(NewsDigest).invoke("prompt")
    assert guard.fatal_reason == "outcome-missing"
    assert guard.evidence()[0].receipt_returned is False
    assert len(gate.admit_calls) == 1


def test_multiple_outcomes_are_refused():
    guard, gate, _, _, wrapped = _scripted(outcome="twice")
    with pytest.raises(InvocationGuardContractError):
        wrapped.with_structured_output(NewsDigest).invoke("prompt")
    assert guard.fatal_reason == "duplicate-outcome"
    assert len(gate.outcome_calls) == 1


def test_mismatched_outcome_decision_is_refused():
    guard, gate, _, _, wrapped = _scripted(outcome="mismatch")
    with pytest.raises(InvocationGuardContractError):
        wrapped.with_structured_output(NewsDigest).invoke("prompt")
    assert guard.fatal_reason == "outcome-decision-mismatch"
    assert gate.outcome_calls == []


def test_outcome_receipt_mismatch_is_refused_and_marks_accounting():
    guard, gate, _, _, wrapped = _scripted(
        gate=StubGate(outcome_receipt="wrong-id")
    )
    with pytest.raises(InvocationGuardContractError):
        wrapped.with_structured_output(NewsDigest).invoke("prompt")
    record = guard.evidence()[0]
    assert record.accounting_incomplete is True
    assert record.returned_outcome_event_id == "wrong-id"
    assert record.requested_outcome_event_id == "outcome-NewsDigest"
    assert record.receipt_returned is False
    assert guard.fatal_reason == "outcome-receipt-mismatch"


def test_non_string_outcome_receipt_is_not_retained():
    guard, gate, _, _, wrapped = _scripted(
        gate=StubGate(outcome_receipt=123)
    )
    with pytest.raises(InvocationGuardContractError):
        wrapped.with_structured_output(NewsDigest).invoke("prompt")
    record = guard.evidence()[0]
    assert record.returned_outcome_event_id is None
    assert record.accounting_incomplete is True
    assert record.receipt_returned is False
    assert guard.fatal_reason == "outcome-receipt-mismatch"


def test_lazy_constructor_error_latches_before_admission():
    guard, gate, _, inner, wrapped = _scripted(
        constructor_error=ValueError("constructor detail")
    )
    with pytest.raises(ValueError):
        wrapped.with_structured_output(NewsDigest).invoke("prompt")
    assert guard.fatal_reason == "inner-error"
    assert gate.admit_calls == []
    assert inner.transport_count == 0


def test_structured_output_options_are_rejected_before_admission():
    guard, gate, _, inner, wrapped = _scripted()
    with pytest.raises(InvocationGuardContractError):
        wrapped.with_structured_output(NewsDigest, method="json_mode")
    assert guard.fatal_reason == "structured-options-not-supported"
    assert gate.admit_calls == []
    assert inner.transport_count == 0


def test_fatal_latch_blocks_a_coordinated_sibling_attempt():
    rise = threading.Event()
    release = threading.Event()
    outcome = {}

    def on_invoke():
        rise.set()
        if not release.wait(5):
            raise RuntimeError("coordinated release timed out")

    guard, gate, client, _, wrapped = _stack(
        client=StubClient(error=RuntimeError("first failure"), on_invoke=on_invoke)
    )
    started = threading.Event()

    def first():
        try:
            wrapped.with_structured_output(NewsDigest).invoke("first")
        except BaseException as exc:  # noqa: BLE001 - class only
            outcome["first"] = type(exc).__name__

    def second():
        started.set()
        try:
            wrapped.with_structured_output(TopicAnalysis).invoke("second")
        except BaseException as exc:  # noqa: BLE001 - class only
            outcome["second"] = type(exc).__name__

    first_thread = threading.Thread(target=first)
    second_thread = threading.Thread(target=second)
    started_threads = []
    first_thread.start()
    started_threads.append(first_thread)
    try:
        assert rise.wait(5)
        second_thread.start()
        started_threads.append(second_thread)
        assert started.wait(5)
        assert len(gate.admit_calls) == 1
        assert client.calls == 1
    finally:
        release.set()
        for thread in started_threads:
            thread.join(5)
    assert all(not thread.is_alive() for thread in started_threads)
    assert outcome["first"] == "RuntimeError"
    assert outcome["second"] == "InvocationGuardFatalError"
    assert len(gate.admit_calls) == 1
    assert client.calls == 1


def test_evidence_is_immutable_and_proxy_is_narrow():
    guard, gate, proxy, _, wrapped = _scripted()
    assert wrapped.with_structured_output(NewsDigest).invoke("prompt") == "scripted"
    record = guard.evidence()[0]
    with pytest.raises(FrozenInstanceError):
        record.known_usage = (9, 9)
    assert not hasattr(proxy, "start_run")
    assert not hasattr(proxy, "acquire_lease")
    assert not hasattr(proxy, "reconcile_llm_outcome")
    assert not hasattr(proxy, "__getattr__")
    assert len(gate.admit_calls) == 1
    assert len(gate.outcome_calls) == 1
