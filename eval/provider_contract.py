"""Pure, offline provider-contract validation for a future evaluation adapter.

Declared labels are caller claims, not verified identity or serialization.
No caller input can make live_eligible True; there is no bypass field.
Trusted verifier provenance is unresolved, so live eligibility is always False.
This module performs no I/O and imports no project, settings or SDK code.
Evidence state never affects offline eligibility; it is reported only.
"""
from dataclasses import dataclass as _dataclass

__all__ = [
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

SUPPORTED_STRUCTURED_MODES = frozenset({"declared_structured_output_v1"})
SUPPORTED_MODES = frozenset({"offline_synthetic", "live"})
ESTIMATE_COMPONENTS = frozenset(
    {
        "prompt",
        "message_framing",
        "schema_payload",
        "tool_payload",
        "provider_overhead",
    }
)
REQUIRED_SDK_MAX_RETRIES = 0
REQUIRED_GRAPH_MAX_ATTEMPTS = 1
INT64_MAX = 2**63 - 1

_PROFILE_INVALID = "PROFILE_INVALID"
_POLICY_INVALID = "POLICY_INVALID"
_METADATA_INVALID = "METADATA_INVALID"
_EVIDENCE_INVALID = "EVIDENCE_INVALID"
_ESTIMATE_MISSING = "ESTIMATE_MISSING"
_MODEL_MISMATCH = "MODEL_MISMATCH"
_CAP_INVALID = "CAP_INVALID"
_CAP_MISMATCH = "CAP_MISMATCH"
_SCHEMA_SET_INVALID = "SCHEMA_SET_INVALID"
_SCHEMA_SET_MISMATCH = "SCHEMA_SET_MISMATCH"
_RETRY_SHAPE_UNSUPPORTED = "RETRY_SHAPE_UNSUPPORTED"
_STRUCTURED_MODE_UNSUPPORTED = "STRUCTURED_MODE_UNSUPPORTED"
_MODE_UNSUPPORTED = "MODE_UNSUPPORTED"
_ESTIMATE_COMPONENT_UNKNOWN = "ESTIMATE_COMPONENT_UNKNOWN"
_TOKENIZER_MISMATCH = "TOKENIZER_MISMATCH"

_MISSING_VERIFIED_BOUND = "MISSING_VERIFIED_BOUND"
_PROVIDER_IDENTITY_UNATTESTED = "PROVIDER_IDENTITY_UNATTESTED"
_STALE_BINDING = "STALE_BINDING"
_UNTRUSTED_EVIDENCE_SOURCE = "UNTRUSTED_EVIDENCE_SOURCE"
_NON_LIVE_MODE = "NON_LIVE_MODE"

_ABSENT = "absent"
_STALE = "stale_binding"
_STRUCTURALLY_CONSISTENT = "structurally_consistent"


@_dataclass(frozen=True)
class ExecutionProfile:
    provider_id: str
    model_id: str
    schema_names: frozenset
    max_output_tokens: int
    sdk_max_retries: int
    graph_max_attempts: int
    structured_mode: str


@_dataclass(frozen=True)
class PolicySnapshot:
    allowed_model: str
    allowed_schema_names: frozenset
    max_output_tokens: int
    required_structured_mode: str


@_dataclass(frozen=True)
class EstimateMetadata:
    estimator_id: str
    estimator_version: str
    tokenizer_id: str
    tokenizer_model_map: frozenset
    method_id: str
    covered_components: frozenset
    stated_overhead_tokens: int
    reserve_margin_percent: int
    scope_exclusions: tuple
    request_shape_id: str
    library_id: str
    library_version: str
    validation_evidence: object = None


@_dataclass(frozen=True)
class EvidenceRecord:
    model_id: str
    estimator_id: str
    estimator_version: str
    method_id: str
    schema_names: frozenset
    request_shape_id: str
    library_id: str
    library_version: str
    evidence_reference: str


@_dataclass(frozen=True)
class ContractResult:
    offline_eligible: bool
    live_eligible: bool
    codes: tuple
    live_blockers: tuple
    evidence_status: str


def _valid_string(value):
    return type(value) is str and len(value) > 0 and value == value.strip()


def _int_in_range(value, minimum, maximum):
    return type(value) is int and minimum <= value <= maximum


def _frozenset_of_strings(value):
    if type(value) is not frozenset:
        return False
    for item in value:
        if type(item) is not str:
            return False
    return True


def _strings_valid(items):
    for item in items:
        if not _valid_string(item):
            return False
    return True


def _profile_shape(profile):
    codes = set()
    if type(profile) is not ExecutionProfile:
        return False, {_PROFILE_INVALID}, None
    for name in ("provider_id", "model_id"):
        if type(getattr(profile, name)) is not str:
            codes.add(_PROFILE_INVALID)
    if type(profile.schema_names) is not frozenset:
        codes.add(_SCHEMA_SET_INVALID)
    else:
        for item in profile.schema_names:
            if type(item) is not str:
                codes.add(_SCHEMA_SET_INVALID)
                break
    for name in ("max_output_tokens", "sdk_max_retries", "graph_max_attempts"):
        if type(getattr(profile, name)) is not int:
            codes.add(_PROFILE_INVALID)
    if type(profile.structured_mode) is not str:
        codes.add(_PROFILE_INVALID)
    return not codes, codes, None


def _policy_shape(policy):
    codes = set()
    if type(policy) is not PolicySnapshot:
        return False, {_POLICY_INVALID}, None
    if type(policy.allowed_model) is not str:
        codes.add(_POLICY_INVALID)
    if type(policy.allowed_schema_names) is not frozenset:
        codes.add(_SCHEMA_SET_INVALID)
    else:
        for item in policy.allowed_schema_names:
            if type(item) is not str:
                codes.add(_SCHEMA_SET_INVALID)
                break
    if type(policy.max_output_tokens) is not int:
        codes.add(_POLICY_INVALID)
    if type(policy.required_structured_mode) is not str:
        codes.add(_POLICY_INVALID)
    return not codes, codes, None


def _estimate_shape(estimate):
    codes = set()
    if estimate is None:
        return False, {_ESTIMATE_MISSING}, None
    if type(estimate) is not EstimateMetadata:
        return False, {_METADATA_INVALID}, None
    for name in (
        "estimator_id",
        "estimator_version",
        "tokenizer_id",
        "method_id",
        "request_shape_id",
        "library_id",
        "library_version",
    ):
        if type(getattr(estimate, name)) is not str:
            codes.add(_METADATA_INVALID)
    if type(estimate.tokenizer_model_map) is not frozenset:
        codes.add(_METADATA_INVALID)
    else:
        for item in estimate.tokenizer_model_map:
            if type(item) is not tuple or len(item) != 2:
                codes.add(_METADATA_INVALID)
                break
            if type(item[0]) is not str or type(item[1]) is not str:
                codes.add(_METADATA_INVALID)
                break
    if type(estimate.covered_components) is not frozenset:
        codes.add(_METADATA_INVALID)
    else:
        for item in estimate.covered_components:
            if type(item) is not str:
                codes.add(_METADATA_INVALID)
                break
    for name in ("stated_overhead_tokens", "reserve_margin_percent"):
        if type(getattr(estimate, name)) is not int:
            codes.add(_METADATA_INVALID)
    if type(estimate.scope_exclusions) is not tuple:
        codes.add(_METADATA_INVALID)
    else:
        for item in estimate.scope_exclusions:
            if type(item) is not str:
                codes.add(_METADATA_INVALID)
                break
    return not codes, codes, None


def _profile_values(profile):
    codes = set()
    model_ok = _valid_string(profile.model_id)
    schema_ok = _frozenset_of_strings(profile.schema_names) and _strings_valid(
        profile.schema_names
    )
    cap_ok = _int_in_range(profile.max_output_tokens, 1, INT64_MAX)
    struct_ok = _valid_string(profile.structured_mode)
    provider_ok = _valid_string(profile.provider_id)
    if not provider_ok:
        codes.add(_PROFILE_INVALID)
    if not model_ok:
        codes.add(_PROFILE_INVALID)
    if not schema_ok:
        codes.add(_SCHEMA_SET_INVALID)
    if not cap_ok:
        codes.add(_CAP_INVALID)
    if not struct_ok:
        codes.add(_PROFILE_INVALID)
    elif profile.structured_mode not in SUPPORTED_STRUCTURED_MODES:
        codes.add(_STRUCTURED_MODE_UNSUPPORTED)
        struct_ok = False
    return (
        not codes,
        {
            "model": model_ok,
            "schema": schema_ok,
            "cap": cap_ok,
            "structured": struct_ok,
        },
        codes,
    )


def _policy_values(policy):
    codes = set()
    model_ok = _valid_string(policy.allowed_model)
    schema_ok = _frozenset_of_strings(policy.allowed_schema_names)
    if schema_ok and not _strings_valid(policy.allowed_schema_names):
        schema_ok = False
    if not schema_ok:
        codes.add(_SCHEMA_SET_INVALID)
    elif len(policy.allowed_schema_names) != 4:
        codes.add(_POLICY_INVALID)
        schema_ok = False
    cap_ok = _int_in_range(policy.max_output_tokens, 1, INT64_MAX)
    struct_ok = _valid_string(policy.required_structured_mode)
    if not model_ok:
        codes.add(_POLICY_INVALID)
    if not cap_ok:
        codes.add(_CAP_INVALID)
    if not struct_ok:
        codes.add(_POLICY_INVALID)
    elif policy.required_structured_mode not in SUPPORTED_STRUCTURED_MODES:
        codes.add(_POLICY_INVALID)
        struct_ok = False
    return (
        not codes,
        {"model": model_ok, "schema": schema_ok, "cap": cap_ok, "structured": struct_ok},
        codes,
    )


def _estimate_values(estimate):
    codes = set()
    strings_ok = True
    for name in (
        "estimator_id",
        "estimator_version",
        "method_id",
        "request_shape_id",
        "library_id",
        "library_version",
    ):
        if not _valid_string(getattr(estimate, name)):
            strings_ok = False
    if not strings_ok:
        codes.add(_METADATA_INVALID)
    tokenizer_ok = _valid_string(estimate.tokenizer_id)
    if not tokenizer_ok:
        codes.add(_METADATA_INVALID)
    if tokenizer_ok:
        for pair in estimate.tokenizer_model_map:
            if not _valid_string(pair[0]) or not _valid_string(pair[1]):
                tokenizer_ok = False
                codes.add(_METADATA_INVALID)
                break
    covered_ok = True
    if not _frozenset_of_strings(estimate.covered_components):
        covered_ok = False
        codes.add(_METADATA_INVALID)
    elif not estimate.covered_components:
        codes.add(_ESTIMATE_COMPONENT_UNKNOWN)
        covered_ok = False
    else:
        for item in estimate.covered_components:
            if not _valid_string(item):
                codes.add(_METADATA_INVALID)
                covered_ok = False
                continue
            if item not in ESTIMATE_COMPONENTS:
                codes.add(_ESTIMATE_COMPONENT_UNKNOWN)
                covered_ok = False
    if not _int_in_range(estimate.stated_overhead_tokens, 0, INT64_MAX):
        codes.add(_METADATA_INVALID)
    if not _int_in_range(estimate.reserve_margin_percent, 0, 100):
        codes.add(_METADATA_INVALID)
    scope_ok = type(estimate.scope_exclusions) is tuple and _strings_valid(
        estimate.scope_exclusions
    )
    if not scope_ok:
        codes.add(_METADATA_INVALID)
    return (
        not codes,
        {
            "tokenizer": tokenizer_ok,
            "covered": covered_ok,
            "strings": strings_ok,
        },
        codes,
    )


def _evidence_is_valid(evidence):
    if type(evidence) is not EvidenceRecord:
        return False
    for name in (
        "model_id",
        "estimator_id",
        "estimator_version",
        "method_id",
        "request_shape_id",
        "library_id",
        "library_version",
        "evidence_reference",
    ):
        if not _valid_string(getattr(evidence, name)):
            return False
    if not _frozenset_of_strings(evidence.schema_names):
        return False
    return _strings_valid(evidence.schema_names)


def evaluate_contract(profile, policy, estimate, mode):
    codes = set()

    profile_shape_ok, profile_shape_codes, _ = _profile_shape(profile)
    policy_shape_ok, policy_shape_codes, _ = _policy_shape(policy)
    estimate_shape_ok, estimate_shape_codes, _ = _estimate_shape(estimate)
    codes.update(profile_shape_codes)
    codes.update(policy_shape_codes)
    codes.update(estimate_shape_codes)

    if type(mode) is not str or mode not in SUPPORTED_MODES:
        codes.add(_MODE_UNSUPPORTED)

    profile_clean = False
    policy_clean = False
    estimate_clean = False
    p_flags = {}
    pol_flags = {}
    e_flags = {}

    if profile_shape_ok:
        profile_clean, p_flags, profile_value_codes = _profile_values(profile)
        codes.update(profile_value_codes)
    if policy_shape_ok:
        policy_clean, pol_flags, policy_value_codes = _policy_values(policy)
        codes.update(policy_value_codes)
    if estimate_shape_ok:
        estimate_clean, e_flags, estimate_value_codes = _estimate_values(estimate)
        codes.update(estimate_value_codes)

    if profile_shape_ok and policy_shape_ok:
        if (
            p_flags.get("model")
            and pol_flags.get("model")
            and profile.model_id != policy.allowed_model
        ):
            codes.add(_MODEL_MISMATCH)
        if (
            p_flags.get("cap")
            and pol_flags.get("cap")
            and profile.max_output_tokens != policy.max_output_tokens
        ):
            codes.add(_CAP_MISMATCH)
        if (
            p_flags.get("schema")
            and pol_flags.get("schema")
            and profile.schema_names != policy.allowed_schema_names
        ):
            codes.add(_SCHEMA_SET_MISMATCH)
    if profile_shape_ok:
        if (
            profile.sdk_max_retries != REQUIRED_SDK_MAX_RETRIES
            or profile.graph_max_attempts != REQUIRED_GRAPH_MAX_ATTEMPTS
        ):
            codes.add(_RETRY_SHAPE_UNSUPPORTED)
    if profile_shape_ok and policy_shape_ok:
        if (
            p_flags.get("structured")
            and pol_flags.get("structured")
            and profile.structured_mode != policy.required_structured_mode
        ):
            codes.add(_STRUCTURED_MODE_UNSUPPORTED)
    if profile_shape_ok and estimate_shape_ok:
        if p_flags.get("model") and e_flags.get("tokenizer"):
            pair = (profile.model_id, estimate.tokenizer_id)
            if pair not in estimate.tokenizer_model_map:
                codes.add(_TOKENIZER_MISMATCH)

    evidence_status = _ABSENT
    if (
        profile_shape_ok
        and profile_clean
        and policy_shape_ok
        and policy_clean
        and estimate_shape_ok
        and estimate_clean
    ):
        evidence = estimate.validation_evidence
        if evidence is not None:
            if not _evidence_is_valid(evidence):
                codes.add(_EVIDENCE_INVALID)
                evidence_status = _STALE
            else:
                matching = (
                    evidence.model_id == profile.model_id
                    and evidence.estimator_id == estimate.estimator_id
                    and evidence.estimator_version == estimate.estimator_version
                    and evidence.method_id == estimate.method_id
                    and evidence.schema_names == profile.schema_names
                    and evidence.request_shape_id == estimate.request_shape_id
                    and evidence.library_id == estimate.library_id
                    and evidence.library_version == estimate.library_version
                )
                evidence_status = (
                    _STRUCTURALLY_CONSISTENT if matching else _STALE
                )

    sorted_codes = tuple(sorted(codes))
    offline_eligible = sorted_codes == () or sorted_codes == (_EVIDENCE_INVALID,)
    blockers = [_MISSING_VERIFIED_BOUND, _PROVIDER_IDENTITY_UNATTESTED]
    if evidence_status == _STALE:
        blockers.append(_STALE_BINDING)
    if evidence_status == _STRUCTURALLY_CONSISTENT:
        blockers.append(_UNTRUSTED_EVIDENCE_SOURCE)
    if type(mode) is str and mode == "offline_synthetic":
        blockers.append(_NON_LIVE_MODE)

    return ContractResult(
        offline_eligible=offline_eligible,
        live_eligible=False,
        codes=sorted_codes,
        live_blockers=tuple(blockers),
        evidence_status=evidence_status,
    )
