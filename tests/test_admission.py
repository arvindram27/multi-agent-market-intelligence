"""Offline tests for the pure admission decision layer (Slice 2a)."""
import ast
import inspect
import math
from pathlib import Path

import pytest

from market_intelligence.admission import (
    AdmissionReason,
    Decision,
    FetchRequest,
    LLMRequest,
    PolicyConfig,
    RunView,
    WriteRequest,
    decide_fetch,
    decide_llm,
    decide_write,
)

SOURCE = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "market_intelligence"
    / "admission.py"
)


def _policy(**overrides):
    values = {
        "max_run_tokens": 20_000,
        "max_run_llm_calls": 24,
        "max_output_tokens": 512,
        "max_fetches_per_run": 8,
        "reservation_margin": 0.25,
        "provider_overhead_tokens": 0,
        "allowed_model": "gpt-4o-mini",
        "allowed_schemas": frozenset({"NewsDigest", "TopicAnalysis"}),
    }
    values.update(overrides)
    return PolicyConfig(**values)


def _view(**overrides):
    values = {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reserved_tokens": 0,
        "fetch_count": 0,
        "approved": False,
    }
    values.update(overrides)
    return RunView(**values)


def _llm(**overrides):
    values = {
        "model": "gpt-4o-mini",
        "schema_name": "NewsDigest",
        "serialized_body_token_count": 100,
        "max_output_tokens": 512,
    }
    values.update(overrides)
    return LLMRequest(**values)


def _fetch(**overrides):
    values = {
        "host": "example.com",
        "max_bytes": 1000,
        "timeout_seconds": 5.0,
        "max_redirects": 3,
    }
    values.update(overrides)
    return FetchRequest(**values)


def test_reservation_math_is_exact_integer_rational():
    policy = _policy(max_output_tokens=100, reservation_margin=0.25)
    decision = decide_llm(_llm(max_output_tokens=100), _view(), policy)
    assert decision.allow is True
    assert decision.reason is AdmissionReason.ALLOWED
    assert decision.reserve_tokens == 225  # ceil(100 * 1.25) + 100

    fractional = _policy(max_output_tokens=10, reservation_margin=0.1)
    decision = decide_llm(
        _llm(serialized_body_token_count=3, max_output_tokens=10),
        _view(),
        fractional,
    )
    assert decision.reserve_tokens == 14  # ceil(3 * 1.1) + 10

    overhead = _policy(
        max_output_tokens=20,
        reservation_margin=0.5,
        provider_overhead_tokens=5,
    )
    decision = decide_llm(
        _llm(serialized_body_token_count=7, max_output_tokens=20),
        _view(),
        overhead,
    )
    assert decision.reserve_tokens == 38  # ceil((7 + 5) * 1.5) + 20


def test_reservation_uses_policy_cap_not_request_cap():
    policy = _policy(max_output_tokens=512)
    decision = decide_llm(
        _llm(max_output_tokens=1), _view(), policy
    )
    assert decision.allow is True
    assert decision.reserve_tokens == 125 + 512


def test_request_cap_above_policy_cap_is_policy_denied():
    policy = _policy(max_output_tokens=100)
    decision = decide_llm(_llm(max_output_tokens=101), _view(), policy)
    assert decision.allow is False
    assert decision.reason is AdmissionReason.POLICY_DENIED
    assert decision.reserve_tokens is None


def test_boundary_equality_is_allowed_and_one_over_is_denied():
    request = _llm(serialized_body_token_count=100, max_output_tokens=100)
    policy = _policy(max_run_tokens=225, max_output_tokens=100)
    decision = decide_llm(request, _view(), policy)
    assert decision.allow is True
    assert decision.reserve_tokens == 225

    just_over = _policy(max_run_tokens=224, max_output_tokens=100)
    decision = decide_llm(request, _view(), just_over)
    assert decision.allow is False
    assert decision.reason is AdmissionReason.BUDGET_EXCEEDED


def test_outstanding_reservation_counts_in_budget():
    policy = _policy(max_run_tokens=300, max_output_tokens=100)
    decision = decide_llm(
        _llm(serialized_body_token_count=100, max_output_tokens=100),
        _view(reserved_tokens=76),
        policy,
    )
    assert decision.allow is False
    assert decision.reason is AdmissionReason.BUDGET_EXCEEDED


def test_call_boundary_and_one_over():
    policy = _policy(max_run_llm_calls=2)
    request = _llm(serialized_body_token_count=10)
    assert decide_llm(request, _view(calls=1), policy).allow is True
    decision = decide_llm(request, _view(calls=2), policy)
    assert decision.allow is False
    assert decision.reason is AdmissionReason.BUDGET_EXCEEDED


def test_tokenizer_unavailable_and_zero_count():
    decision = decide_llm(
        _llm(serialized_body_token_count=None), _view(), _policy()
    )
    assert decision.allow is False
    assert decision.reason is AdmissionReason.TOKENIZER_UNAVAILABLE

    decision = decide_llm(
        _llm(serialized_body_token_count=0), _view(), _policy()
    )
    assert decision.allow is True
    assert decision.reserve_tokens == 512  # ceil(0 * 1.25) + policy cap


def test_unknown_model_and_schema_are_policy_denied():
    policy = _policy()
    decision = decide_llm(_llm(model="other-model"), _view(), policy)
    assert decision.allow is False
    assert decision.reason is AdmissionReason.POLICY_DENIED

    decision = decide_llm(_llm(schema_name="OtherSchema"), _view(), policy)
    assert decision.allow is False
    assert decision.reason is AdmissionReason.POLICY_DENIED


def test_large_counts_stay_exact_above_2_to_the_53():
    token_count = 2**53 + 1
    policy = _policy(
        max_run_tokens=token_count * 10,
        max_output_tokens=7,
        reservation_margin=0.25,
        provider_overhead_tokens=0,
    )
    decision = decide_llm(
        _llm(serialized_body_token_count=token_count, max_output_tokens=7),
        _view(),
        policy,
    )
    expected = ((token_count * 5) + 3) // 4 + 7  # ceil(count * 1.25) + 7
    assert decision.allow is True
    assert decision.reserve_tokens == expected


def test_fetch_syntactic_host_and_limits():
    policy = _policy()
    assert decide_fetch(_fetch(host="127.0.0.1"), _view(), policy).allow is True
    for host in (
        "https://example.com",
        "example.com:443",
        "exa_mple.com",
        "user@example.com",
        "exa mple.com",
        "::1",
        "",
    ):
        if host == "":
            with pytest.raises(ValueError):
                _fetch(host=host)
            continue
        decision = decide_fetch(_fetch(host=host), _view(), policy)
        assert decision.allow is False
        assert decision.reason is AdmissionReason.INVALID_TARGET

    for overrides in (
        {"max_bytes": 0},
        {"max_bytes": -1},
        {"timeout_seconds": 0},
        {"timeout_seconds": -1.0},
        {"timeout_seconds": float("nan")},
        {"timeout_seconds": float("inf")},
        {"max_redirects": -1},
    ):
        decision = decide_fetch(_fetch(**overrides), _view(), policy)
        assert decision.allow is False
        assert decision.reason is AdmissionReason.INVALID_TARGET


def test_huge_integer_timeout_is_finite_and_allowed():
    decision = decide_fetch(
        _fetch(timeout_seconds=10**309), _view(fetch_count=0), _policy()
    )
    assert decision.allow is True
    assert decision.reason is AdmissionReason.ALLOWED

    negative = decide_fetch(
        _fetch(timeout_seconds=-(10**309)), _view(), _policy()
    )
    assert negative.allow is False
    assert negative.reason is AdmissionReason.INVALID_TARGET


def test_fetch_cap_boundary():
    policy = _policy(max_fetches_per_run=2)
    assert decide_fetch(_fetch(), _view(fetch_count=1), policy).allow is True
    decision = decide_fetch(_fetch(), _view(fetch_count=2), policy)
    assert decision.allow is False
    assert decision.reason is AdmissionReason.BUDGET_EXCEEDED


def test_write_requires_explicit_approval():
    policy = _policy()
    decision = decide_write(WriteRequest(target_id=None), _view(approved=False), policy)
    assert decision.allow is False
    assert decision.reason is AdmissionReason.UNAPPROVED_WRITE
    assert decide_write(
        WriteRequest(target_id="run-1"), _view(approved=True), policy
    ).allow is True


def test_policy_construction_validation():
    with pytest.raises((TypeError, ValueError)):
        _policy(max_run_tokens=True)
    with pytest.raises(ValueError):
        _policy(max_run_tokens=0)
    with pytest.raises(ValueError):
        _policy(max_run_llm_calls=0)
    with pytest.raises(ValueError):
        _policy(max_output_tokens=0)
    with pytest.raises(ValueError):
        _policy(max_fetches_per_run=0)
    with pytest.raises(TypeError):
        _policy(reservation_margin=1)
    with pytest.raises(ValueError):
        _policy(reservation_margin=float("nan"))
    with pytest.raises(ValueError):
        _policy(reservation_margin=float("inf"))
    with pytest.raises(ValueError):
        _policy(reservation_margin=1.1)
    with pytest.raises(ValueError):
        _policy(provider_overhead_tokens=-1)
    with pytest.raises(ValueError):
        _policy(allowed_model="")
    with pytest.raises(TypeError):
        _policy(allowed_schemas=["NewsDigest"])
    with pytest.raises(ValueError):
        _policy(allowed_schemas=frozenset({""}))
    with pytest.raises(ValueError):
        _policy(allowed_model="x" * 257)


def test_run_view_construction_validation():
    for field in (
        "calls",
        "prompt_tokens",
        "completion_tokens",
        "reserved_tokens",
        "fetch_count",
    ):
        with pytest.raises(ValueError):
            _view(**{field: -1})
        with pytest.raises(TypeError):
            _view(**{field: True})
    with pytest.raises(TypeError):
        _view(approved=1)


def test_request_construction_validation():
    with pytest.raises(TypeError):
        _fetch(max_bytes=True)
    with pytest.raises(TypeError):
        _fetch(max_redirects=True)
    with pytest.raises(TypeError):
        _fetch(timeout_seconds="5")
    with pytest.raises(TypeError):
        _llm(serialized_body_token_count=True)
    with pytest.raises(ValueError):
        _llm(serialized_body_token_count=-1)
    with pytest.raises(ValueError):
        _llm(max_output_tokens=0)
    with pytest.raises(ValueError):
        _llm(max_output_tokens=-1)
    with pytest.raises(ValueError):
        WriteRequest(target_id="")
    with pytest.raises(ValueError):
        WriteRequest(target_id="x" * 257)
    assert WriteRequest(target_id=None).target_id is None


def test_decision_construction_contract():
    with pytest.raises(TypeError):
        Decision(allow=1, reason=AdmissionReason.ALLOWED)
    with pytest.raises(TypeError):
        Decision(allow=True, reason="allowed")
    with pytest.raises(ValueError):
        Decision(allow=False, reason=AdmissionReason.POLICY_DENIED, reserve_tokens=3)


def test_decisions_are_plain_values_not_exceptions_for_semantics():
    policy = _policy(max_fetches_per_run=1)
    results = [
        decide_fetch(_fetch(host="bad_host"), _view(), policy),
        decide_llm(_llm(model="wrong"), _view(), policy),
        decide_llm(_llm(serialized_body_token_count=None), _view(), policy),
        decide_write(WriteRequest(None), _view(approved=False), policy),
    ]
    assert all(isinstance(result, Decision) and not result.allow for result in results)


def test_decide_functions_reject_wrong_class_arguments():
    with pytest.raises(TypeError):
        decide_fetch(_llm(), _view(), _policy())
    with pytest.raises(TypeError):
        decide_fetch(_fetch(), _policy(), _policy())
    with pytest.raises(TypeError):
        decide_fetch(_fetch(), _view(), _view())

    with pytest.raises(TypeError):
        decide_llm(_fetch(), _view(), _policy())
    with pytest.raises(TypeError):
        decide_llm(_llm(), _policy(), _policy())
    with pytest.raises(TypeError):
        decide_llm(_llm(), _view(), _view())

    with pytest.raises(TypeError):
        decide_write(_fetch(), _view(approved=True), _policy())
    with pytest.raises(TypeError):
        decide_write(WriteRequest(None), _policy(), _policy())
    with pytest.raises(TypeError):
        decide_write(WriteRequest(None), _view(approved=True), _view())


def test_module_is_pure_by_static_import_check_and_signature():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    forbidden = {
        "socket",
        "sqlite3",
        "urllib",
        "http",
        "langchain",
        "openai",
        "tiktoken",
        "requests",
        "psycopg",
        "market_intelligence",
    }
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module.split(".")[0])
    assert not (set(imports) & forbidden)

    for function in (decide_fetch, decide_llm, decide_write):
        parameters = list(inspect.signature(function).parameters.values())
        assert [parameter.name for parameter in parameters] == [
            "request",
            "view",
            "policy",
        ]
        assert all(
            parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
            for parameter in parameters
        )
