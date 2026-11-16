"""Focused offline tests for the pure provider-contract module.

Authored only in this Build step; never executed here.
"""
import dataclasses
import inspect
import socket

import pytest

import eval.provider_contract as contract

_DEFAULT = object()

_ALWAYS_BLOCKERS = (
    "MISSING_VERIFIED_BOUND",
    "PROVIDER_IDENTITY_UNATTESTED",
)


def _profile(**overrides):
    values = {
        "provider_id": "openai",
        "model_id": "declared-model-x",
        "schema_names": frozenset({"digest", "topic", "tone", "comparison"}),
        "max_output_tokens": 1024,
        "sdk_max_retries": 0,
        "graph_max_attempts": 1,
        "structured_mode": "declared_structured_output_v1",
    }
    values.update(overrides)
    return contract.ExecutionProfile(**values)


def _policy(**overrides):
    values = {
        "allowed_model": "declared-model-x",
        "allowed_schema_names": frozenset(
            {"digest", "topic", "tone", "comparison"}
        ),
        "max_output_tokens": 1024,
        "required_structured_mode": "declared_structured_output_v1",
    }
    values.update(overrides)
    return contract.PolicySnapshot(**values)


def _estimate(**overrides):
    values = {
        "estimator_id": "est-1",
        "estimator_version": "1.0",
        "tokenizer_id": "tok-a",
        "tokenizer_model_map": frozenset({("declared-model-x", "tok-a")}),
        "method_id": "declared_method_v1",
        "covered_components": contract.ESTIMATE_COMPONENTS,
        "stated_overhead_tokens": 0,
        "reserve_margin_percent": 0,
        "scope_exclusions": (),
        "request_shape_id": "shape-v1",
        "library_id": "openai",
        "library_version": "3.29.0",
    }
    values.update(overrides)
    return contract.EstimateMetadata(**values)


def _evidence(**overrides):
    values = {
        "model_id": "declared-model-x",
        "estimator_id": "est-1",
        "estimator_version": "1.0",
        "method_id": "declared_method_v1",
        "schema_names": frozenset({"digest", "topic", "tone", "comparison"}),
        "request_shape_id": "shape-v1",
        "library_id": "openai",
        "library_version": "3.29.0",
        "evidence_reference": "opaque-ref",
    }
    values.update(overrides)
    return contract.EvidenceRecord(**values)


def _run(profile=None, policy=None, estimate=_DEFAULT, mode="offline_synthetic"):
    return contract.evaluate_contract(
        _profile() if profile is None else profile,
        _policy() if policy is None else policy,
        _estimate() if estimate is _DEFAULT else estimate,
        mode,
    )


def _assert_result(
    result,
    *,
    offline,
    codes,
    evidence_status="absent",
    mode="offline_synthetic",
):
    blockers = list(_ALWAYS_BLOCKERS)
    if evidence_status == "stale_binding":
        blockers.append("STALE_BINDING")
    elif evidence_status == "structurally_consistent":
        blockers.append("UNTRUSTED_EVIDENCE_SOURCE")
    if mode == "offline_synthetic":
        blockers.append("NON_LIVE_MODE")
    assert result.offline_eligible is offline
    assert result.live_eligible is False
    assert result.codes == codes
    assert result.evidence_status == evidence_status
    assert result.live_blockers == tuple(blockers)


def test_row_01():
    result = _run()
    _assert_result(result, offline=True, codes=(), evidence_status="absent")


def test_row_02():
    result = _run(profile=_profile(model_id="other-model"))
    _assert_result(
        result,
        offline=False,
        codes=("MODEL_MISMATCH", "TOKENIZER_MISMATCH"),
    )


def test_row_03():
    result = _run(profile=_profile(max_output_tokens=512))
    _assert_result(result, offline=False, codes=("CAP_MISMATCH",))


@pytest.mark.parametrize(
    "cap,expected",
    [
        (0, ("CAP_INVALID",)),
        (-5, ("CAP_INVALID",)),
        (2**63, ("CAP_INVALID",)),
        (True, ("PROFILE_INVALID",)),
        (1.5, ("PROFILE_INVALID",)),
    ],
)
def test_row_04(cap, expected):
    result = _run(profile=_profile(max_output_tokens=cap))
    _assert_result(result, offline=False, codes=expected)


@pytest.mark.parametrize(
    "profile,policy,expected",
    [
        (_profile(schema_names=frozenset({"a", "b", "c"})), None, ("SCHEMA_SET_MISMATCH",)),
        (
            _profile(schema_names=frozenset({"a", "b", "c", "d", "e"})),
            None,
            ("SCHEMA_SET_MISMATCH",),
        ),
        (
            _profile(schema_names=frozenset({"digest2", "topic", "tone", "comparison"})),
            None,
            ("SCHEMA_SET_MISMATCH",),
        ),
        (_profile(schema_names=["digest", "topic", "tone", "comparison"]), None, ("SCHEMA_SET_INVALID",)),
        (
            _profile(schema_names=frozenset({"digest", "topic", "tone", " "})),
            None,
            ("SCHEMA_SET_INVALID",),
        ),
        (
            None,
            _policy(allowed_schema_names=frozenset({"digest", "topic", "tone"})),
            ("POLICY_INVALID",),
        ),
    ],
)
def test_row_05(profile, policy, expected):
    result = _run(profile=profile, policy=policy)
    _assert_result(result, offline=False, codes=expected)


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({"sdk_max_retries": 1}, ("RETRY_SHAPE_UNSUPPORTED",)),
        ({"graph_max_attempts": 2}, ("RETRY_SHAPE_UNSUPPORTED",)),
        ({"sdk_max_retries": True}, ("PROFILE_INVALID",)),
    ],
)
def test_row_06(overrides, expected):
    result = _run(profile=_profile(**overrides))
    _assert_result(result, offline=False, codes=expected)


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("json_mode", ("STRUCTURED_MODE_UNSUPPORTED",)),
        (" declared_structured_output_v1 ", ("PROFILE_INVALID",)),
    ],
)
def test_row_07(mode, expected):
    result = _run(profile=_profile(structured_mode=mode))
    _assert_result(result, offline=False, codes=expected)


def test_row_08():
    result = _run(mode="production")
    _assert_result(
        result,
        offline=False,
        codes=("MODE_UNSUPPORTED",),
        mode="production",
    )


@pytest.mark.parametrize(
    "estimate,expected",
    [
        ("MISSING", ("ESTIMATE_MISSING",)),
        (
            _estimate(covered_components=frozenset({"prompt", "alien"})),
            ("ESTIMATE_COMPONENT_UNKNOWN",),
        ),
        (_estimate(stated_overhead_tokens=1.5), ("METADATA_INVALID",)),
        (
            _estimate(covered_components=frozenset({"prompt", "alien", " padded "})),
            ("ESTIMATE_COMPONENT_UNKNOWN", "METADATA_INVALID"),
        ),
        (
            _estimate(covered_components=frozenset({"", "alien"})),
            ("ESTIMATE_COMPONENT_UNKNOWN", "METADATA_INVALID"),
        ),
    ],
)
def test_row_09(estimate, expected):
    if estimate == "MISSING":
        result = _run(estimate=None)
    else:
        result = _run(estimate=estimate)
    _assert_result(result, offline=False, codes=expected)


def test_row_10():
    result = _run(
        estimate=_estimate(
            tokenizer_model_map=frozenset({("declared-model-x", "tok-b")})
        )
    )
    _assert_result(result, offline=False, codes=("TOKENIZER_MISMATCH",))


def test_row_11():
    result = _run(estimate=_estimate(validation_evidence=_evidence(model_id="other-model")))
    _assert_result(
        result,
        offline=True,
        codes=(),
        evidence_status="stale_binding",
    )

    plain = _run(estimate=_estimate(validation_evidence="not-a-record"))
    _assert_result(
        plain,
        offline=True,
        codes=("EVIDENCE_INVALID",),
        evidence_status="stale_binding",
    )


def test_row_12():
    result = _run(estimate=_estimate(validation_evidence=_evidence()))
    _assert_result(
        result,
        offline=True,
        codes=(),
        evidence_status="structurally_consistent",
    )


def test_row_13():
    result = _run(mode="live")
    _assert_result(result, offline=True, codes=(), mode="live")


def test_purity_no_file_or_socket_io(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("unexpected I/O")

    monkeypatch.setattr("builtins.open", boom)
    monkeypatch.setattr(socket, "socket", boom)
    result = contract.evaluate_contract(
        _profile(), _policy(), _estimate(), "offline_synthetic"
    )
    _assert_result(result, offline=True, codes=())


def test_public_surface_exact():
    expected = [
        "ExecutionProfile",
        "PolicySnapshot",
        "EstimateMetadata",
        "EvidenceRecord",
        "ContractResult",
        "evaluate_contract",
        "SUPPORTED_STRUCTURED_MODES",
        "SUPPORTED_MODES",
        "ESTIMATE_COMPONENTS",
        "REQUIRED_SDK_MAX_RETRIES",
        "REQUIRED_GRAPH_MAX_ATTEMPTS",
        "INT64_MAX",
    ]
    assert contract.__all__ == expected
    for name in expected:
        assert hasattr(contract, name)
    public = {name for name in vars(contract) if not name.startswith("_")}
    assert public == set(expected)
    for name in (
        "get_llm",
        "settings",
        "Settings",
        "factory",
        "create_client",
        "runner",
        "main",
        "transport",
    ):
        assert not hasattr(contract, name)


def test_result_is_frozen():
    result = _run()
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.offline_eligible = False


def test_no_bypass_field():
    for record in (
        contract.ExecutionProfile,
        contract.PolicySnapshot,
        contract.EstimateMetadata,
        contract.EvidenceRecord,
        contract.ContractResult,
    ):
        for field in dataclasses.fields(record):
            lowered = field.name.lower()
            assert "allow_live" not in lowered
            assert "trusted" not in lowered
            assert "bypass" not in lowered
            assert "skip" not in lowered
    for name in inspect.signature(contract.evaluate_contract).parameters:
        lowered = name.lower()
        assert "allow_live" not in lowered
        assert "trusted" not in lowered
        assert "bypass" not in lowered
        assert "skip" not in lowered


_INVALID_CASES = []


def _add_case(builder, expected, case_id):
    _INVALID_CASES.append(pytest.param(builder, expected, id=case_id))


_PROFILE_STR_FIELDS = ("provider_id", "model_id", "structured_mode")
_POLICY_STR_FIELDS = ("allowed_model", "required_structured_mode")
_ESTIMATE_STR_FIELDS = (
    "estimator_id",
    "estimator_version",
    "tokenizer_id",
    "method_id",
    "request_shape_id",
    "library_id",
    "library_version",
)

for _record, _fields, _code in (
    ("profile", _PROFILE_STR_FIELDS, ("PROFILE_INVALID",)),
    ("policy", _POLICY_STR_FIELDS, ("POLICY_INVALID",)),
    ("estimate", _ESTIMATE_STR_FIELDS, ("METADATA_INVALID",)),
):
    for _field in _fields:
        for _value, _tag in ((5, "int5"), ("", "empty"), (" padded ", "padded")):
            def _builder(f=_field, v=_value, r=_record):
                overrides = {f: v}
                return (
                    _profile(**overrides) if r == "profile" else _profile(),
                    _policy(**overrides) if r == "policy" else _policy(),
                    _estimate(**overrides) if r == "estimate" else _estimate(),
                    "offline_synthetic",
                )

            _add_case(_builder, _code, "str-%s-%s-%s" % (_record, _field, _tag))

for _field in ("max_output_tokens", "sdk_max_retries", "graph_max_attempts"):
    for _value, _tag in ((True, "bool"), (1.5, "float"), ("3", "str")):
        def _builder(f=_field, v=_value):
            return (_profile(**{f: v}), _policy(), _estimate(), "offline_synthetic")

        _add_case(_builder, ("PROFILE_INVALID",), "profile-int-%s-%s" % (_field, _tag))

for _value, _tag in ((True, "bool"), (1.5, "float"), ("3", "str")):
    def _builder(v=_value):
        return (_profile(), _policy(max_output_tokens=v), _estimate(), "offline_synthetic")

    _add_case(_builder, ("POLICY_INVALID",), "policy-cap-%s" % _tag)

for _field in ("stated_overhead_tokens", "reserve_margin_percent"):
    for _value, _tag in ((True, "bool"), (1.5, "float"), ("3", "str")):
        def _builder(f=_field, v=_value):
            return (_profile(), _policy(), _estimate(**{f: v}), "offline_synthetic")

        _add_case(_builder, ("METADATA_INVALID",), "estimate-int-%s-%s" % (_field, _tag))

for _value, _tag in ((0, "zero"), (-1, "negative"), (2**63, "overflow")):
    def _builder(v=_value):
        return (_profile(max_output_tokens=v), _policy(), _estimate(), "offline_synthetic")

    _add_case(_builder, ("CAP_INVALID",), "profile-cap-%s" % _tag)

    def _builder(v=_value):
        return (_profile(), _policy(max_output_tokens=v), _estimate(), "offline_synthetic")

    _add_case(_builder, ("CAP_INVALID",), "policy-cap-range-%s" % _tag)

for _value, _tag in ((-1, "negative"), (2**63, "overflow")):
    def _builder(v=_value):
        return (_profile(), _policy(), _estimate(stated_overhead_tokens=v), "offline_synthetic")

    _add_case(_builder, ("METADATA_INVALID",), "overhead-range-%s" % _tag)

for _value, _tag in ((-1, "negative"), (101, "over100")):
    def _builder(v=_value):
        return (_profile(), _policy(), _estimate(reserve_margin_percent=v), "offline_synthetic")

    _add_case(_builder, ("METADATA_INVALID",), "margin-range-%s" % _tag)

_SCHEMA_VALUES = (
    (["a", "b", "c", "d"], "list"),
    ("abcd", "str"),
    (frozenset({5}), "int-element"),
    (frozenset({""}), "empty-element"),
    (frozenset({" padded "}), "padded-element"),
)
for _value, _tag in _SCHEMA_VALUES:
    def _builder(v=_value):
        return (_profile(schema_names=v), _policy(), _estimate(), "offline_synthetic")

    _add_case(_builder, ("SCHEMA_SET_INVALID",), "profile-schema-%s" % _tag)

    def _builder(v=_value):
        return (_profile(), _policy(allowed_schema_names=v), _estimate(), "offline_synthetic")

    _add_case(_builder, ("SCHEMA_SET_INVALID",), "policy-schema-%s" % _tag)

def _policy_three_names():
    return (
        _profile(),
        _policy(allowed_schema_names=frozenset({"a", "b", "c"})),
        _estimate(),
        "offline_synthetic",
    )


_add_case(_policy_three_names, ("POLICY_INVALID",), "policy-schema-three")

_MAP_VALUES = (
    (["a", "b"], "list"),
    ("abc", "str"),
    (frozenset({"abc"}), "non-tuple-element"),
    (frozenset({("a",)}), "tuple-len1"),
    (frozenset({("a", "b", "c")}), "tuple-len3"),
    (frozenset({("a", 5)}), "nonstr-item"),
    (frozenset({("a", " b ")}), "padded-item"),
)
for _value, _tag in _MAP_VALUES:
    def _builder(v=_value):
        return (_profile(), _policy(), _estimate(tokenizer_model_map=v), "offline_synthetic")

    _add_case(_builder, ("METADATA_INVALID",), "map-%s" % _tag)

_COVERED_VALUES = (
    (["prompt"], "list", ("METADATA_INVALID",)),
    ("prompt", "str", ("METADATA_INVALID",)),
    (frozenset({"prompt", 5}), "nonstr-element", ("METADATA_INVALID",)),
    (frozenset(), "empty", ("ESTIMATE_COMPONENT_UNKNOWN",)),
    (frozenset({"alien"}), "alien", ("ESTIMATE_COMPONENT_UNKNOWN",)),
    (frozenset({""}), "empty-str", ("METADATA_INVALID",)),
    (frozenset({" padded "}), "padded-str", ("METADATA_INVALID",)),
)
for _value, _tag, _code in _COVERED_VALUES:
    def _builder(v=_value):
        return (_profile(), _policy(), _estimate(covered_components=v), "offline_synthetic")

    _add_case(_builder, _code, "covered-%s" % _tag)

_SCOPE_VALUES = (
    (["x"], "list"),
    (("x", 5), "nonstr-element"),
    (("",), "empty-str"),
    ((" padded ",), "padded-str"),
)
for _value, _tag in _SCOPE_VALUES:
    def _builder(v=_value):
        return (_profile(), _policy(), _estimate(scope_exclusions=v), "offline_synthetic")

    _add_case(_builder, ("METADATA_INVALID",), "scope-%s" % _tag)

_MODE_VALUES = (
    (None, "none"),
    (5, "int"),
    ("", "empty"),
    (" live ", "padded"),
)
for _value, _tag in _MODE_VALUES:
    def _builder(v=_value):
        return (_profile(), _policy(), _estimate(), v)

    _add_case(_builder, ("MODE_UNSUPPORTED",), "mode-%s" % _tag)

def _profile_json_mode():
    return (
        _profile(structured_mode="json_mode"),
        _policy(),
        _estimate(),
        "offline_synthetic",
    )


_add_case(_profile_json_mode, ("STRUCTURED_MODE_UNSUPPORTED",), "profile-json-mode")

def _policy_json_mode():
    return (
        _profile(),
        _policy(required_structured_mode="json_mode"),
        _estimate(),
        "offline_synthetic",
    )


_add_case(_policy_json_mode, ("POLICY_INVALID",), "policy-json-mode")

def _profile_none():
    return (None, _policy(), _estimate(), "offline_synthetic")


_add_case(_profile_none, ("PROFILE_INVALID",), "refusal-profile-none")

def _policy_none():
    return (_profile(), None, _estimate(), "offline_synthetic")


_add_case(_policy_none, ("POLICY_INVALID",), "refusal-policy-none")

def _profile_wrong_type():
    return (_policy(), _policy(), _estimate(), "offline_synthetic")


_add_case(_profile_wrong_type, ("PROFILE_INVALID",), "refusal-profile-policy")

def _policy_wrong_type():
    return (_profile(), _profile(), _estimate(), "offline_synthetic")


_add_case(_policy_wrong_type, ("POLICY_INVALID",), "refusal-policy-profile")

def _both_none():
    return (None, None, _estimate(), "offline_synthetic")


_add_case(_both_none, ("POLICY_INVALID", "PROFILE_INVALID"), "refusal-both-none")

def _estimate_none():
    return (_profile(), _policy(), None, "offline_synthetic")


_add_case(_estimate_none, ("ESTIMATE_MISSING",), "estimate-none")

def _estimate_wrong_type():
    return (_profile(), _policy(), _profile(), "offline_synthetic")


_add_case(_estimate_wrong_type, ("METADATA_INVALID",), "estimate-wrong-type")

def _retry_sdk():
    return (_profile(sdk_max_retries=1), _policy(), _estimate(), "offline_synthetic")


_add_case(_retry_sdk, ("RETRY_SHAPE_UNSUPPORTED",), "retry-sdk")

def _retry_graph():
    return (_profile(graph_max_attempts=2), _policy(), _estimate(), "offline_synthetic")


_add_case(_retry_graph, ("RETRY_SHAPE_UNSUPPORTED",), "retry-graph")


@pytest.mark.parametrize("build,expected", _INVALID_CASES)
def test_invalid_field_types(build, expected):
    profile, policy, estimate, mode = build()
    result = contract.evaluate_contract(profile, policy, estimate, mode)
    _assert_result(result, offline=False, codes=expected, mode=mode)


_EVIDENCE_CASES = []


def _add_evidence_case(case, case_id):
    _EVIDENCE_CASES.append(pytest.param(case, id=case_id))


for _field in (
    "model_id",
    "estimator_id",
    "estimator_version",
    "method_id",
    "request_shape_id",
    "library_id",
    "library_version",
    "evidence_reference",
):
    for _value, _tag in ((5, "int5"), ("", "empty"), (" padded ", "padded")):
        _add_evidence_case(
            ("field", (_field, _value)),
            "evidence-%s-%s" % (_field, _tag),
        )

for _value, _tag in (
    (["a", "b", "c", "d"], "list"),
    (frozenset({5}), "int-element"),
    (frozenset({" padded "}), "padded-element"),
):
    _add_evidence_case(("schema", _value), "evidence-schema-%s" % _tag)

for _field in (
    "model_id",
    "estimator_id",
    "estimator_version",
    "method_id",
    "schema_names",
    "request_shape_id",
    "library_id",
    "library_version",
):
    _add_evidence_case(("binding", _field), "binding-%s" % _field)

_add_evidence_case(("skipped", 5), "skip-estimator-int")
_add_evidence_case(("skipped", " padded "), "skip-estimator-padded")
_add_evidence_case(("skipped-cap", None), "skip-cap-zero")


@pytest.mark.parametrize("case", _EVIDENCE_CASES)
def test_evidence_binding_mismatches(case):
    kind, payload = case
    if kind == "binding":
        if payload == "schema_names":
            replacement = frozenset({"a", "b", "c", "d"})
        elif payload in ("estimator_version", "library_version"):
            replacement = "9.9"
        else:
            replacement = "other-" + payload
        evidence = dataclasses.replace(_evidence(), **{payload: replacement})
        result = _run(estimate=_estimate(validation_evidence=evidence))
        _assert_result(
            result,
            offline=True,
            codes=(),
            evidence_status="stale_binding",
        )
    elif kind == "field":
        field, value = payload
        evidence = dataclasses.replace(_evidence(), **{field: value})
        result = _run(estimate=_estimate(validation_evidence=evidence))
        _assert_result(
            result,
            offline=True,
            codes=("EVIDENCE_INVALID",),
            evidence_status="stale_binding",
        )
    elif kind == "schema":
        evidence = dataclasses.replace(_evidence(), schema_names=payload)
        result = _run(estimate=_estimate(validation_evidence=evidence))
        _assert_result(
            result,
            offline=True,
            codes=("EVIDENCE_INVALID",),
            evidence_status="stale_binding",
        )
    elif kind == "skipped":
        result = _run(
            estimate=_estimate(
                estimator_id=payload,
                validation_evidence=_evidence(),
            )
        )
        _assert_result(result, offline=False, codes=("METADATA_INVALID",))
    else:
        result = _run(
            profile=_profile(max_output_tokens=0),
            estimate=_estimate(validation_evidence=_evidence()),
        )
        _assert_result(result, offline=False, codes=("CAP_INVALID",))
