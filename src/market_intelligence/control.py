"""Append-only audit foundation for the H1 control layer (Slice 1, repaired).

No admission, policy, reservation, tokenizer or controller-wiring logic lives
here. This module provides the bounded audit-event model, the append-only SQLite
store and a read-only whole-chain verifier.
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import re
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field, fields
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

POLICY_VERSION = "h1-slice1"
POLICY_VERSION_H1_SLICE2 = "h1-slice2"
MAX_TEXT_LEN = 128
MAX_POLICY_LEN = 64
MAX_HOST_LEN = 253
MAX_RATIO_DECIMAL = 400
INT64_MAX = 2**63 - 1
INT64_MIN = -(2**63)

LEGACY_OPERATIONS = frozenset({"fetch", "llm", "write"})
OPERATIONS = frozenset({"fetch", "llm", "write", "run"})
LEGACY_EVENT_KINDS = frozenset({"decision", "outcome"})
V2_EVENT_KINDS = frozenset(
    {
        "run_started",
        "run_approved",
        "lease_acquired",
        "lease_renewed",
        "lease_taken_over",
        "lease_released",
    }
)
EVENT_KINDS = LEGACY_EVENT_KINDS | V2_EVENT_KINDS
DECISIONS = frozenset({"allow", "deny"})
WRITE_OUTCOMES = frozenset({"committed", "failed", "unknown"})
LEGACY_COUNTER_KEYS = frozenset(
    {
        "calls",
        "prompt_tokens",
        "completion_tokens",
        "reserved_tokens",
        "remaining_tokens",
    }
)
COUNTER_KEYS = LEGACY_COUNTER_KEYS | {"budget_deficit"}
V2_REASON_CODES = frozenset(
    {
        "budget_exceeded",
        "tokenizer_unavailable",
        "lease_conflict",
        "ledger_ambiguous",
        "released_no_effect",
        "usage_missing",
        "reservation_overrun",
        "budget_breach",
        "usage_out_of_range",
    }
)
RESULT_CLASSES = frozenset(
    {"ok", "http_error", "timeout", "invalid_content", "network_error", "unknown"}
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
_DECIMAL_RE = re.compile(r"^(0|[1-9][0-9]*)$")

_APPEND_LOCK = threading.RLock()
_PAYLOAD_EXTRA_KEYS = frozenset({"prev_hash", "retry_safe"})
_NEW_FIELD_NAMES = frozenset(
    {
        "serialized_body_token_count",
        "reserve_tokens",
        "usage_prompt_tokens",
        "usage_completion_tokens",
        "reservation_released",
        "policy_margin",
        "policy_margin_num",
        "policy_margin_den",
        "provider_overhead_tokens",
        "max_run_tokens",
        "max_run_llm_calls",
        "max_output_tokens",
        "max_fetches_per_run",
        "lease_ttl_seconds",
        "allowed_model",
        "allowed_schemas",
        "estimate_kind",
        "observed_now",
        "lease_owner",
        "lease_epoch",
        "lease_expires_at",
        "previous_lease_owner",
        "previous_lease_epoch",
        "approved_target_id",
        "result_class",
        "overrun_tokens",
    }
)


class ReasonCode(str, Enum):
    """Fixed, stable reason codes; audit events never carry free-text reasons."""

    ALLOWED = "allowed"
    POLICY_DENIED = "policy_denied"
    UNAPPROVED_WRITE = "unapproved_write"
    INVALID_TARGET = "invalid_target"
    AUDIT_UNAVAILABLE = "audit_unavailable"
    CHAIN_BROKEN = "chain_broken"
    RECONCILIATION = "reconciliation"
    BUDGET_EXCEEDED = "budget_exceeded"
    TOKENIZER_UNAVAILABLE = "tokenizer_unavailable"
    LEASE_CONFLICT = "lease_conflict"
    LEDGER_AMBIGUOUS = "ledger_ambiguous"
    RELEASED_NO_EFFECT = "released_no_effect"
    USAGE_MISSING = "usage_missing"
    RESERVATION_OVERRUN = "reservation_overrun"
    BUDGET_BREACH = "budget_breach"
    USAGE_OUT_OF_RANGE = "usage_out_of_range"


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


def _int64(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if value < minimum or value > INT64_MAX:
        raise ValueError(f"{name} must be between {minimum} and {INT64_MAX}")
    return value


def _ratio_int(value: object, name: str) -> int:
    if type(value) is not int or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if not 0 <= value:
        raise ValueError(f"{name} must be non-negative")
    if len(str(value)) > MAX_RATIO_DECIMAL:
        raise ValueError(f"{name} exceeds the ratio decimal bound")
    return value


def canonical_decimal(value: int) -> str:
    """Canonical decimal TEXT for a non-negative ratio integer."""
    if type(value) is not int or isinstance(value, bool) or value < 0:
        raise ValueError("ratio integer must be a non-negative int")
    text = str(value)
    if len(text) > MAX_RATIO_DECIMAL:
        raise ValueError("ratio integer exceeds the decimal bound")
    return text


def parse_canonical_decimal(text: object) -> int:
    """Strict inverse of canonical_decimal: no sign, zeros or whitespace."""
    if type(text) is not str or not _DECIMAL_RE.fullmatch(text):
        raise ValueError("ratio TEXT is not canonical decimal")
    if len(text) > MAX_RATIO_DECIMAL:
        raise ValueError("ratio TEXT exceeds the decimal bound")
    value = int(text)
    if str(value) != text:
        raise ValueError("ratio TEXT does not round-trip")
    return value


def _optional_timestamp(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _timestamp(value)


def _schemas(value: object) -> tuple[str, ...] | None:
    if value is None:
        return None
    if type(value) is not tuple:
        raise TypeError("allowed_schemas must be a tuple")
    cleaned: list[str] = []
    for item in value:
        cleaned.append(_text(item, "allowed_schemas item"))
    if len(cleaned) > 32:
        raise ValueError("allowed_schemas exceeds 32 entries")
    if cleaned != sorted(set(cleaned)):
        raise ValueError("allowed_schemas must be sorted and unique")
    return tuple(cleaned)


def _has_binding(event: "AuditEvent") -> bool:
    return any(
        value is not None
        for value in (event.lease_owner, event.lease_epoch, event.observed_now)
    )


def _binding_complete(event: "AuditEvent") -> bool:
    return all(
        value is not None
        for value in (event.lease_owner, event.lease_epoch, event.observed_now)
    )


def _has_policy(event: "AuditEvent") -> bool:
    return all(
        getattr(event, name) is not None
        for name in (
            "policy_margin",
            "policy_margin_num",
            "policy_margin_den",
            "provider_overhead_tokens",
            "max_run_tokens",
            "max_run_llm_calls",
            "max_output_tokens",
            "max_fetches_per_run",
            "lease_ttl_seconds",
            "allowed_model",
            "allowed_schemas",
            "estimate_kind",
        )
    )


def _validate_ratio(event: "AuditEvent") -> None:
    margin = event.policy_margin
    if type(margin) is not float or not math.isfinite(margin) or not 0.0 <= margin <= 1.0:
        raise ValueError("policy_margin must be a finite float in [0, 1]")
    numerator, denominator = margin.as_integer_ratio()
    if event.policy_margin_num != numerator or event.policy_margin_den != denominator:
        raise ValueError("policy_margin_num/den must equal policy_margin.as_integer_ratio()")
    _ratio_int(event.policy_margin_num, "policy_margin_num")
    _ratio_int(event.policy_margin_den, "policy_margin_den")
    if event.policy_margin_den < 1 or event.policy_margin_num > event.policy_margin_den:
        raise ValueError("policy_margin ratio is out of range")


def _validate_v2_shape(event: "AuditEvent") -> None:
    kind = event.event_kind
    present = {
        name for name in _NEW_FIELD_NAMES if getattr(event, name) is not None
    }
    allowed_new = {
        "run_started": {
            "policy_margin",
            "policy_margin_num",
            "policy_margin_den",
            "provider_overhead_tokens",
            "max_run_tokens",
            "max_run_llm_calls",
            "max_output_tokens",
            "max_fetches_per_run",
            "lease_ttl_seconds",
            "allowed_model",
            "allowed_schemas",
            "estimate_kind",
        },
        "run_approved": {"observed_now", "lease_owner", "lease_epoch", "approved_target_id"},
        "lease_acquired": {"observed_now", "lease_owner", "lease_epoch", "lease_expires_at", "lease_ttl_seconds"},
        "lease_renewed": {"observed_now", "lease_owner", "lease_epoch", "lease_expires_at", "lease_ttl_seconds"},
        "lease_taken_over": {
            "observed_now",
            "lease_owner",
            "lease_epoch",
            "lease_expires_at",
            "lease_ttl_seconds",
            "previous_lease_owner",
            "previous_lease_epoch",
        },
        "lease_released": {"observed_now", "lease_owner", "lease_epoch", "lease_expires_at", "lease_ttl_seconds"},
        "decision": {
            "observed_now",
            "lease_owner",
            "lease_epoch",
            "policy_margin",
            "policy_margin_num",
            "policy_margin_den",
            "provider_overhead_tokens",
            "max_run_tokens",
            "max_run_llm_calls",
            "max_output_tokens",
            "max_fetches_per_run",
            "lease_ttl_seconds",
            "allowed_model",
            "allowed_schemas",
            "estimate_kind",
            "serialized_body_token_count",
            "reserve_tokens",
        },
        "outcome": {
            "observed_now",
            "lease_owner",
            "lease_epoch",
            "usage_prompt_tokens",
            "usage_completion_tokens",
            "reservation_released",
            "result_class",
            "overrun_tokens",
        },
    }
    extra = present - allowed_new.get(kind, set())
    if extra:
        raise ValueError(f"extra 2b fields for {kind}: {sorted(extra)}")
    if kind == "run_started":
        if event.operation != "run" or event.decision != "allow":
            raise ValueError("run_started must be a run/allow event")
        if event.reason_code != ReasonCode.ALLOWED.value:
            raise ValueError("run_started requires reason allowed")
        if event.counters or _has_binding(event):
            raise ValueError("run_started has no counters or lease binding")
        if not _has_policy(event):
            raise ValueError("run_started requires a full policy snapshot")
        _validate_ratio(event)
        for name in (
            "serialized_body_token_count",
            "reserve_tokens",
            "usage_prompt_tokens",
            "usage_completion_tokens",
            "reservation_released",
            "previous_lease_owner",
            "previous_lease_epoch",
            "approved_target_id",
            "result_class",
            "overrun_tokens",
            "lease_expires_at",
            "error_class",
            "write_outcome",
            "references_event_id",
            "target_host",
            "target_id",
            "content_hash",
        ):
            if getattr(event, name) is not None:
                raise ValueError(f"run_started must not carry {name}")
        return

    if kind == "run_approved":
        if event.operation != "run" or event.decision != "allow":
            raise ValueError("run_approved must be a run/allow event")
        if event.reason_code != ReasonCode.ALLOWED.value:
            raise ValueError("run_approved requires reason allowed")
        if event.counters or not _binding_complete(event):
            raise ValueError("run_approved requires a lease binding and no counters")
        if event.approved_target_id is None:
            raise ValueError("run_approved requires an explicit approved_target_id")
        if _has_policy(event):
            raise ValueError("run_approved must not carry a policy snapshot")
        for name in (
            "references_event_id",
            "target_host",
            "target_id",
            "content_hash",
        ):
            if getattr(event, name) is not None:
                raise ValueError(f"run_approved must not carry {name}")
        return

    if kind in {"lease_acquired", "lease_renewed", "lease_taken_over", "lease_released"}:
        if event.operation != "run" or event.decision != "allow":
            raise ValueError("lease events must be run/allow events")
        if event.reason_code != ReasonCode.ALLOWED.value:
            raise ValueError("lease events require reason allowed")
        if event.counters or not _binding_complete(event):
            raise ValueError("lease events require a lease binding and no counters")
        if event.lease_ttl_seconds is None:
            raise ValueError("lease events require lease_ttl_seconds")
        if kind != "lease_released" and event.lease_expires_at is None:
            raise ValueError("lease event requires lease_expires_at")
        if kind in {"lease_acquired", "lease_renewed"}:
            if event.previous_lease_owner is not None or event.previous_lease_epoch is not None:
                raise ValueError("acquire/renew must not carry previous lease fields")
            if event.references_event_id is not None:
                raise ValueError("acquire/renew must not reference another event")
            for name in ("target_host", "target_id", "content_hash"):
                if getattr(event, name) is not None:
                    raise ValueError(f"{kind} must not carry {name}")
        if kind == "lease_taken_over":
            if event.previous_lease_owner is None or event.previous_lease_epoch is None:
                raise ValueError("takeover requires previous owner and epoch")
            if event.references_event_id is None:
                raise ValueError("takeover requires a lease reference")
            for name in ("target_host", "target_id", "content_hash"):
                if getattr(event, name) is not None:
                    raise ValueError(f"lease_taken_over must not carry {name}")
        if kind == "lease_released":
            if event.previous_lease_owner is not None or event.previous_lease_epoch is not None:
                raise ValueError("release must not carry previous lease fields")
            for name in (
                "references_event_id",
                "target_host",
                "target_id",
                "content_hash",
            ):
                if getattr(event, name) is not None:
                    raise ValueError(f"lease_released must not carry {name}")
        if _has_policy(event):
            raise ValueError("lease events must not carry a policy snapshot")
        return

    if kind == "decision":
        if event.operation not in LEGACY_OPERATIONS:
            raise ValueError("decision operation must be fetch/llm/write")
        if not _binding_complete(event):
            raise ValueError("decision requires a lease binding")
        if not _has_policy(event):
            raise ValueError("decision requires a full policy snapshot")
        _validate_ratio(event)
        if event.operation == "llm":
            if event.serialized_body_token_count is None and event.reason_code != ReasonCode.TOKENIZER_UNAVAILABLE.value:
                raise ValueError("llm decision requires a serialized token count")
            if (
                event.serialized_body_token_count is not None
                and event.reason_code == ReasonCode.TOKENIZER_UNAVAILABLE.value
            ):
                raise ValueError("tokenizer_unavailable denial carries no serialized count")
            if event.serialized_body_token_count is not None:
                _int64(event.serialized_body_token_count, "serialized_body_token_count")
            if event.decision == "allow":
                if event.reserve_tokens is None:
                    raise ValueError("allowed llm decision requires reserve_tokens")
            elif event.reserve_tokens is not None:
                raise ValueError("denied llm decision must not carry reserve_tokens")
        else:
            if event.serialized_body_token_count is not None or event.reserve_tokens is not None:
                raise ValueError("fetch/write decisions carry no reservation fields")
        if event.operation == "write" and event.target_id is None:
            raise ValueError("write decision requires a target_id")
        return

    if kind == "outcome":
        if event.operation not in LEGACY_OPERATIONS:
            raise ValueError("outcome operation must be fetch/llm/write")
        if not _binding_complete(event):
            raise ValueError("outcome requires a lease binding")
        if event.references_event_id is None:
            raise ValueError("outcome requires references_event_id")
        if _has_policy(event):
            raise ValueError("outcome must not carry a policy snapshot")
        reason = event.reason_code
        if event.operation == "llm":
            if event.model is None or event.schema_name is None:
                raise ValueError("llm outcome requires model and schema_name")
            if reason in {
                ReasonCode.USAGE_MISSING.value,
                ReasonCode.USAGE_OUT_OF_RANGE.value,
        }:
                if event.usage_prompt_tokens is not None or event.usage_completion_tokens is not None:
                    raise ValueError("unknown outcomes must not carry usage figures")
                if event.reservation_released != 0:
                    raise ValueError("unknown outcomes require reservation_released=0")
                if event.overrun_tokens is not None:
                    raise ValueError("unknown outcomes carry no overrun")
                if event.counters:
                    raise ValueError("unknown outcomes carry empty counters")
                if not event.usage_missing:
                    raise ValueError("unknown outcomes set usage_missing")
                if reason == ReasonCode.USAGE_OUT_OF_RANGE.value and event.error_class != "UsageOutOfRange":
                    raise ValueError("usage_out_of_range requires UsageOutOfRange error_class")
            elif reason == ReasonCode.RELEASED_NO_EFFECT.value:
                if event.usage_missing or event.usage_prompt_tokens is not None or event.usage_completion_tokens is not None:
                    raise ValueError("released_no_effect carries no usage")
                if event.reservation_released is None or event.reservation_released <= 0:
                    raise ValueError("released_no_effect requires a release amount")
                if event.overrun_tokens is not None or event.counters:
                    raise ValueError("released_no_effect has no usage/overrun/counters")
            elif reason in {
                ReasonCode.ALLOWED.value,
                ReasonCode.RESERVATION_OVERRUN.value,
                ReasonCode.BUDGET_BREACH.value,
                ReasonCode.RECONCILIATION.value,
            }:
                if event.usage_missing:
                    raise ValueError("usage-carrying outcome must not set usage_missing")
                if event.usage_prompt_tokens is None or event.usage_completion_tokens is None:
                    if reason == ReasonCode.RECONCILIATION.value:
                        if event.reservation_released not in (None, 0) or event.counters:
                            raise ValueError("usage-less reconciliation must not settle")
                    else:
                        raise ValueError("usage-carrying outcome requires both usage figures")
                if event.usage_prompt_tokens is not None:
                    _int64(event.usage_prompt_tokens, "usage_prompt_tokens")
                    _int64(event.usage_completion_tokens, "usage_completion_tokens")
                    if event.reservation_released is None:
                        raise ValueError("usage-carrying outcome requires reservation_released")
                    _int64(event.reservation_released, "reservation_released")
                    if reason == ReasonCode.RESERVATION_OVERRUN.value:
                        if event.overrun_tokens is None or event.overrun_tokens <= 0:
                            raise ValueError("reservation_overrun requires positive overrun_tokens")
                    elif event.overrun_tokens is not None and reason != ReasonCode.BUDGET_BREACH.value:
                        raise ValueError("overrun_tokens only on overrun/breach")
                    if not event.counters:
                        raise ValueError("usage-carrying outcome requires six counters")
                    if set(event.counters) != COUNTER_KEYS:
                        raise ValueError("usage-carrying outcome requires all six counters")
            else:
                raise ValueError("unsupported llm outcome reason")
            for name in ("target_host", "target_id", "content_hash"):
                if getattr(event, name) is not None:
                    raise ValueError(f"llm outcome must not carry {name}")
            return
        if event.operation == "fetch":
            if reason != ReasonCode.ALLOWED.value:
                raise ValueError("fetch outcome reason must be allowed")
            if event.target_host is None or event.result_class not in RESULT_CLASSES:
                raise ValueError("fetch outcome requires target_host and result_class")
            if event.counters:
                raise ValueError("fetch outcome has no counters")
            if any(
                getattr(event, name) is not None
                for name in (
                    "serialized_body_token_count",
                    "reserve_tokens",
                    "usage_prompt_tokens",
                    "usage_completion_tokens",
                    "reservation_released",
                    "overrun_tokens",
                    "write_outcome",
                )
            ):
                raise ValueError("fetch outcome carries no reservation/write fields")
            for name in ("target_id", "content_hash"):
                if getattr(event, name) is not None:
                    raise ValueError(f"fetch outcome must not carry {name}")
            return
        if event.operation == "write":
            if event.write_outcome is None:
                raise ValueError("write outcome requires write_outcome")
            if event.counters:
                raise ValueError("write outcome has no counters")
            if any(
                getattr(event, name) is not None
                for name in (
                    "serialized_body_token_count",
                    "reserve_tokens",
                    "usage_prompt_tokens",
                    "usage_completion_tokens",
                    "reservation_released",
                    "overrun_tokens",
                    "result_class",
                )
            ):
                raise ValueError("write outcome carries no llm/fetch fields")
            for name in ("target_host", "target_id", "content_hash"):
                if getattr(event, name) is not None:
                    raise ValueError(f"write outcome must not carry {name}")
            return
        raise ValueError("unsupported outcome operation")

    raise ValueError(f"unsupported v2 event kind {kind!r}")


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
    serialized_body_token_count: int | None = None
    reserve_tokens: int | None = None
    usage_prompt_tokens: int | None = None
    usage_completion_tokens: int | None = None
    reservation_released: int | None = None
    policy_margin: float | None = None
    policy_margin_num: int | None = None
    policy_margin_den: int | None = None
    provider_overhead_tokens: int | None = None
    max_run_tokens: int | None = None
    max_run_llm_calls: int | None = None
    max_output_tokens: int | None = None
    max_fetches_per_run: int | None = None
    lease_ttl_seconds: int | None = None
    allowed_model: str | None = None
    allowed_schemas: tuple[str, ...] | None = None
    estimate_kind: str | None = None
    observed_now: str | None = None
    lease_owner: str | None = None
    lease_epoch: int | None = None
    lease_expires_at: str | None = None
    previous_lease_owner: str | None = None
    previous_lease_epoch: int | None = None
    approved_target_id: str | None = None
    result_class: str | None = None
    overrun_tokens: int | None = None

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

        v2 = self.policy_version == POLICY_VERSION_H1_SLICE2
        if not v2:
            if self.operation not in LEGACY_OPERATIONS:
                raise ValueError(f"unknown legacy operation {self.operation!r}")
            if self.event_kind not in LEGACY_EVENT_KINDS:
                raise ValueError(f"unknown legacy event_kind {self.event_kind!r}")
            if any(getattr(self, name) is not None for name in _NEW_FIELD_NAMES):
                raise ValueError("2b fields require policy_version h1-slice2")
            if any(key not in LEGACY_COUNTER_KEYS for key in self.counters):
                raise ValueError("2b counter keys require policy_version h1-slice2")
        else:
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
        if not v2 and reason in V2_REASON_CODES:
            raise ValueError("2b reason codes require policy_version h1-slice2")
        object.__setattr__(self, "reason_code", reason)

        if not isinstance(self.counters, Mapping):
            raise ValueError("counters must be a mapping")
        clean_counters: dict[str, int] = {}
        for key, value in self.counters.items():
            if key not in COUNTER_KEYS:
                raise ValueError(f"unknown counter {key!r}")
            if v2:
                if type(value) is not int or isinstance(value, bool) or not 0 <= value <= INT64_MAX:
                    raise ValueError(f"counter {key!r} must be an int in [0, 2**63-1]")
            elif isinstance(value, bool) or not isinstance(value, int) or value < 0:
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
            if self.operation == "write" and self.event_kind == "outcome" and not v2:
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

        if v2:
            for name in (
                "serialized_body_token_count",
                "reserve_tokens",
                "usage_prompt_tokens",
                "usage_completion_tokens",
                "reservation_released",
                "provider_overhead_tokens",
                "max_run_tokens",
                "max_run_llm_calls",
                "max_output_tokens",
                "max_fetches_per_run",
                "lease_ttl_seconds",
                "lease_epoch",
                "previous_lease_epoch",
                "overrun_tokens",
            ):
                value = getattr(self, name)
                if value is not None:
                    minimum = 1 if name in {
                        "max_run_tokens",
                        "max_run_llm_calls",
                        "max_output_tokens",
                        "max_fetches_per_run",
                        "lease_ttl_seconds",
                        "lease_epoch",
                        "previous_lease_epoch",
                    } else 0
                    object.__setattr__(self, name, _int64(value, name, minimum=minimum))
            if self.policy_margin is not None:
                if type(self.policy_margin) is not float or not math.isfinite(self.policy_margin) or not 0.0 <= self.policy_margin <= 1.0:
                    raise ValueError("policy_margin must be a finite float in [0, 1]")
            for name in ("policy_margin_num", "policy_margin_den"):
                value = getattr(self, name)
                if value is not None:
                    object.__setattr__(self, name, _ratio_int(value, name))
            object.__setattr__(
                self, "allowed_model", _optional_text(self.allowed_model, "allowed_model")
            )
            object.__setattr__(
                self, "allowed_schemas", _schemas(self.allowed_schemas)
            )
            object.__setattr__(
                self, "estimate_kind", _optional_text(self.estimate_kind, "estimate_kind")
            )
            object.__setattr__(
                self,
                "approved_target_id",
                _optional_text(self.approved_target_id, "approved_target_id"),
            )
            object.__setattr__(
                self,
                "observed_now",
                _optional_timestamp(self.observed_now, "observed_now"),
            )
            object.__setattr__(
                self,
                "lease_expires_at",
                _optional_timestamp(self.lease_expires_at, "lease_expires_at"),
            )
            for name in ("lease_owner", "previous_lease_owner"):
                value = getattr(self, name)
                if value is not None:
                    object.__setattr__(self, name, _uuid(value, name))
            if self.result_class is not None:
                if self.result_class not in RESULT_CLASSES:
                    raise ValueError("unknown result_class")
            _validate_v2_shape(self)

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
            if item.name in _NEW_FIELD_NAMES and value is None:
                continue
            if isinstance(value, Enum):
                value = value.value
            if item.name == "counters":
                value = dict(sorted(value.items()))
            if item.name == "allowed_schemas" and value is not None:
                value = list(value)
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
    serialized_body_token_count: int | None = None,
    reserve_tokens: int | None = None,
    usage_prompt_tokens: int | None = None,
    usage_completion_tokens: int | None = None,
    reservation_released: int | None = None,
    policy_margin: float | None = None,
    policy_margin_num: int | None = None,
    policy_margin_den: int | None = None,
    provider_overhead_tokens: int | None = None,
    max_run_tokens: int | None = None,
    max_run_llm_calls: int | None = None,
    max_output_tokens: int | None = None,
    max_fetches_per_run: int | None = None,
    lease_ttl_seconds: int | None = None,
    allowed_model: str | None = None,
    allowed_schemas: tuple[str, ...] | None = None,
    estimate_kind: str | None = None,
    observed_now: str | None = None,
    lease_owner: str | None = None,
    lease_epoch: int | None = None,
    lease_expires_at: str | None = None,
    previous_lease_owner: str | None = None,
    previous_lease_epoch: int | None = None,
    approved_target_id: str | None = None,
    result_class: str | None = None,
    overrun_tokens: int | None = None,
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
        serialized_body_token_count=serialized_body_token_count,
        reserve_tokens=reserve_tokens,
        usage_prompt_tokens=usage_prompt_tokens,
        usage_completion_tokens=usage_completion_tokens,
        reservation_released=reservation_released,
        policy_margin=policy_margin,
        policy_margin_num=policy_margin_num,
        policy_margin_den=policy_margin_den,
        provider_overhead_tokens=provider_overhead_tokens,
        max_run_tokens=max_run_tokens,
        max_run_llm_calls=max_run_llm_calls,
        max_output_tokens=max_output_tokens,
        max_fetches_per_run=max_fetches_per_run,
        lease_ttl_seconds=lease_ttl_seconds,
        allowed_model=allowed_model,
        allowed_schemas=allowed_schemas,
        estimate_kind=estimate_kind,
        observed_now=observed_now,
        lease_owner=lease_owner,
        lease_epoch=lease_epoch,
        lease_expires_at=lease_expires_at,
        previous_lease_owner=previous_lease_owner,
        previous_lease_epoch=previous_lease_epoch,
        approved_target_id=approved_target_id,
        result_class=result_class,
        overrun_tokens=overrun_tokens,
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
    if isinstance(event_fields.get("allowed_schemas"), list):
        event_fields["allowed_schemas"] = tuple(event_fields["allowed_schemas"])
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
    valid_events: list[AuditEvent] = []
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
            valid_events.append(event)
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
    problems.extend(_verify_2b_events(valid_events))
    return problems


def chain_problems(
    rows: list[tuple[int, str, str | None, str, str]],
    head: tuple[int, str] | None,
    *,
    allow_empty: bool,
) -> list[str]:
    """Public read-only wrapper over _chain_problems (same rows and order)."""
    return _chain_problems(rows, head, allow_empty=allow_empty)


def event_from_payload(
    parsed: Mapping[str, Any], label: str = "payload"
) -> AuditEvent:
    """Public read-only wrapper over _payload_event; raises ValueError on failure."""
    problems: list[str] = []
    event = _payload_event(parsed, label, problems)
    if event is None:
        raise ValueError(problems[0] if problems else f"{label}: payload fails the event schema")
    return event


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _policy_snapshot(event: AuditEvent) -> tuple[object, ...]:
    return (
        event.policy_margin,
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
    )


def _verify_2b_events(events: list[AuditEvent]) -> list[str]:
    """Minimal in-memory 2b run/lease/reference state machine (no ledger tables)."""
    problems: list[str] = []
    runs: dict[str, dict[str, Any]] = {}
    legacy_runs: set[str] = set()
    by_id: dict[str, AuditEvent] = {}

    for event in events:
        label = f"event {event.event_id}"
        if event.policy_version != POLICY_VERSION_H1_SLICE2:
            if event.event_kind in {"decision", "outcome"} and event.run_id in runs:
                problems.append(f"{label}: legacy mutation in a bound run")
            legacy_runs.add(event.run_id)
            by_id[event.event_id] = event
            continue

        if event.event_kind == "run_started":
            if event.run_id in legacy_runs:
                problems.append(f"{label}: run_started after legacy events for this run")
            if event.run_id in runs:
                problems.append(f"{label}: duplicate run_started")
                continue
            runs[event.run_id] = {
                "thread": event.thread_id,
                "policy": event,
                "approved": None,
                "decisions": {},
                "normal_outcomes": {},
                "unknown_outcomes": {},
                "reconciled": set(),
                "lease": None,
                "last_lease_event": None,
                "last_lease_observed": None,
            }
            by_id[event.event_id] = event
            continue

        state = runs.get(event.run_id)
        if state is None:
            problems.append(f"{label}: v2 event before run_started")
            by_id[event.event_id] = event
            continue
        if event.thread_id != state["thread"]:
            problems.append(f"{label}: thread binding mismatch")

        if event.event_kind in {"run_approved", "decision", "outcome"}:
            lease = state["lease"]
            observed = _dt(event.observed_now)
            if lease is None or lease["released"]:
                problems.append(f"{label}: state-changing event without a live lease")
            else:
                if event.lease_owner != lease["owner"] or event.lease_epoch != lease["epoch"]:
                    problems.append(f"{label}: lease binding mismatch")
                if _dt(lease["expires_at"]) <= observed:
                    problems.append(f"{label}: lease expired at observed_now")
            if state["last_lease_observed"] is not None and observed < state["last_lease_observed"]:
                problems.append(f"{label}: observed_now moved backward")

        if event.event_kind == "run_approved":
            if state["approved"] is not None:
                problems.append(f"{label}: second run_approved for this run")
            else:
                state["approved"] = event.approved_target_id
            by_id[event.event_id] = event
            continue

        if event.event_kind in {"lease_acquired", "lease_renewed", "lease_taken_over", "lease_released"}:
            observed = _dt(event.observed_now)
            if state["last_lease_observed"] is not None and observed < state["last_lease_observed"]:
                problems.append(f"{label}: lease observed_now moved backward")
            if event.lease_ttl_seconds != state["policy"].lease_ttl_seconds:
                problems.append(f"{label}: lease_ttl_seconds differs from run_started")
            lease = state["lease"]
            ttl = event.lease_ttl_seconds
            expected_expiry = (observed + timedelta(seconds=ttl)).isoformat()
            if event.event_kind == "lease_acquired":
                if lease is not None and not lease["released"]:
                    problems.append(f"{label}: acquire while a live lease exists")
                if lease is None:
                    if event.lease_epoch != 1:
                        problems.append(f"{label}: first acquire must use epoch 1")
                else:
                    if event.lease_epoch != lease["epoch"] + 1:
                        problems.append(f"{label}: acquire epoch must be previous + 1")
                    if event.lease_owner == lease["owner"]:
                        problems.append(f"{label}: acquire requires a fresh owner")
                if event.lease_expires_at != expected_expiry:
                    problems.append(f"{label}: acquire expiry arithmetic mismatch")
                state["lease"] = {
                    "owner": event.lease_owner,
                    "epoch": event.lease_epoch,
                    "expires_at": event.lease_expires_at,
                    "released": False,
                }
            elif event.event_kind == "lease_renewed":
                if lease is None or lease["released"]:
                    problems.append(f"{label}: renewal without a live lease")
                else:
                    if event.lease_owner != lease["owner"] or event.lease_epoch != lease["epoch"]:
                        problems.append(f"{label}: renewal owner/epoch mismatch")
                    if observed >= _dt(lease["expires_at"]):
                        problems.append(f"{label}: renewal after expiry")
                    if _dt(event.lease_expires_at) <= _dt(lease["expires_at"]):
                        problems.append(f"{label}: renewal expiry did not advance")
                    if event.lease_expires_at != expected_expiry:
                        problems.append(f"{label}: renewal expiry arithmetic mismatch")
                state["lease"] = {
                    "owner": event.lease_owner,
                    "epoch": event.lease_epoch,
                    "expires_at": event.lease_expires_at,
                    "released": False,
                }
            elif event.event_kind == "lease_taken_over":
                if lease is None or lease["released"]:
                    problems.append(f"{label}: takeover without an unreleased lease")
                else:
                    if _dt(lease["expires_at"]) > observed:
                        problems.append(f"{label}: takeover evidence is not expired")
                    if event.previous_lease_owner != lease["owner"] or event.previous_lease_epoch != lease["epoch"]:
                        problems.append(f"{label}: takeover previous fields mismatch")
                    if event.lease_epoch != lease["epoch"] + 1:
                        problems.append(f"{label}: takeover epoch must be previous + 1")
                    if event.references_event_id != state["last_lease_event"]:
                        problems.append(f"{label}: takeover does not reference the expired lease event")
                if event.lease_expires_at != expected_expiry:
                    problems.append(f"{label}: takeover expiry arithmetic mismatch")
                state["lease"] = {
                    "owner": event.lease_owner,
                    "epoch": event.lease_epoch,
                    "expires_at": event.lease_expires_at,
                    "released": False,
                }
            else:  # lease_released
                if lease is None or lease["released"]:
                    problems.append(f"{label}: release without a live lease")
                elif event.lease_owner != lease["owner"] or event.lease_epoch != lease["epoch"]:
                    problems.append(f"{label}: release owner/epoch mismatch")
                if lease is not None:
                    state["lease"] = {**lease, "released": True}
            state["last_lease_event"] = event.event_id
            state["last_lease_observed"] = observed
            by_id[event.event_id] = event
            continue

        if event.event_kind == "decision":
            if _policy_snapshot(event) != _policy_snapshot(state["policy"]):
                problems.append(f"{label}: decision policy snapshot mismatch")
            if event.operation == "write" and event.decision == "allow":
                if state["approved"] is None or event.target_id != state["approved"]:
                    problems.append(f"{label}: allowed write lacks earlier approved target")
            state["decisions"][event.event_id] = event
            by_id[event.event_id] = event
            continue

        if event.event_kind == "outcome":
            ref = by_id.get(event.references_event_id)
            if ref is None or ref.run_id != event.run_id or ref.thread_id != event.thread_id:
                problems.append(f"{label}: outcome reference is not an earlier event of this run/thread")
                by_id[event.event_id] = event
                continue
            if event.reason_code == ReasonCode.RECONCILIATION.value:
                if ref.event_kind != "outcome" or ref.reason_code not in {
                    ReasonCode.USAGE_MISSING.value,
                    ReasonCode.USAGE_OUT_OF_RANGE.value,
                }:
                    problems.append(f"{label}: reconciliation must reference an unknown-marking outcome")
                if ref.operation != event.operation:
                    problems.append(f"{label}: reconciliation operation differs from its outcome")
                if event.references_event_id in state["reconciled"]:
                    problems.append(f"{label}: second reconciliation for the same unknown")
                state["reconciled"].add(event.references_event_id)
            else:
                if ref.event_kind != "decision" or ref.operation != event.operation:
                    problems.append(f"{label}: outcome must reference its decision")
                if event.references_event_id in state["normal_outcomes"]:
                    problems.append(f"{label}: second normal outcome for the same decision")
                state["normal_outcomes"][event.references_event_id] = event

            if event.operation == "llm" and (
                event.reason_code
                in {
                    ReasonCode.ALLOWED.value,
                    ReasonCode.RESERVATION_OVERRUN.value,
                    ReasonCode.BUDGET_BREACH.value,
                    ReasonCode.RECONCILIATION.value,
                }
                and event.usage_prompt_tokens is not None
            ):
                counters = event.counters
                required = COUNTER_KEYS
                if set(counters) != required:
                    problems.append(f"{label}: usage snapshot must carry all six counters")
                else:
                    max_tokens = state["policy"].max_run_tokens
                    balance = (
                        max_tokens
                        - counters["prompt_tokens"]
                        - counters["completion_tokens"]
                        - counters["reserved_tokens"]
                    )
                    if counters["remaining_tokens"] != max(0, balance):
                        problems.append(f"{label}: remaining_tokens equation mismatch")
                    if counters["budget_deficit"] != max(0, -balance):
                        problems.append(f"{label}: budget_deficit equation mismatch")
                    if event.reason_code in {
                        ReasonCode.ALLOWED.value,
                        ReasonCode.RESERVATION_OVERRUN.value,
                    } and balance < 0:
                        problems.append(f"{label}: non-breach outcome has a negative balance")
                    if event.reason_code == ReasonCode.BUDGET_BREACH.value and balance >= 0:
                        problems.append(f"{label}: budget_breach without a negative balance")
                    if event.overrun_tokens is not None:
                        expected_overrun = max(
                            0,
                            event.usage_prompt_tokens
                            + event.usage_completion_tokens
                            - event.reservation_released,
                        )
                        if event.overrun_tokens != expected_overrun:
                            problems.append(f"{label}: overrun_tokens equation mismatch")
                if event.reason_code == ReasonCode.RESERVATION_OVERRUN.value and event.overrun_tokens is None:
                    problems.append(f"{label}: reservation_overrun requires overrun_tokens")
            if event.reason_code in {
                ReasonCode.USAGE_MISSING.value,
                ReasonCode.USAGE_OUT_OF_RANGE.value,
            }:
                state["unknown_outcomes"][event.event_id] = event
            by_id[event.event_id] = event
            continue

    return problems


def _rollback(conn) -> bool:
    try:
        conn.execute("ROLLBACK")
        return True
    except Exception:
        return False


@dataclass(frozen=True)
class AppendResult:
    status: str
    event_hash: str | None


class TransactionAborted(RuntimeError):
    """The transaction was aborted by an idempotent re-read; the handle is dead."""


def _strip_chain_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in {"prev_hash", "recorded_at", "counters"}
    }


def logical_fingerprint(value: AuditEvent | Mapping[str, Any]) -> str:
    """Semantic fingerprint excluding prev_hash, recorded_at and counters."""
    payload = value.to_payload() if isinstance(value, AuditEvent) else dict(value)
    return canonical_json(_strip_chain_fields(payload))


class AuditStore:
    """Single-writer append-only SQLite audit store.

    The store owns one connection. ``append`` opens and commits its own
    transaction (published behavior unchanged). The gatekeeper uses
    :meth:`transaction` to run several operations - and ``append_in_tx`` - under
    one transaction on the same connection. There is no update or delete method.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._conn = sqlite3.connect(
            str(self.path), isolation_level=None, check_same_thread=False
        )
        try:
            self._conn.execute("PRAGMA journal_mode=DELETE")
            self._conn.execute("PRAGMA synchronous=FULL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(_SCHEMA)
        except Exception as exc:
            self._conn.close()
            raise ControlUnavailable(
                "could not initialize the audit store",
                append_state="known_rolled_back",
            ) from exc

    def close(self) -> None:
        self._conn.close()

    def _read_chain(self):
        rows = self._conn.execute(
            "SELECT seq, event_id, prev_hash, event_hash, payload "
            "FROM audit_events ORDER BY seq"
        ).fetchall()
        head = self._conn.execute(
            "SELECT seq, event_hash FROM audit_head WHERE singleton = 1"
        ).fetchone()
        return rows, head

    def _append_in_tx(self, event: AuditEvent, *, idempotent: bool) -> AppendResult:
        rows, head = self._read_chain()
        problems = _chain_problems(rows, head, allow_empty=True)
        if problems:
            raise _BrokenChain(problems)

        if idempotent:
            existing = self._conn.execute(
                "SELECT event_hash, payload FROM audit_events WHERE event_id = ?",
                (event.event_id,),
            ).fetchone()
            if existing is not None:
                stored_payload = json.loads(existing[1])
                if logical_fingerprint(stored_payload) != logical_fingerprint(event):
                    raise AuditIncomplete(
                        "event_id exists with a different logical fingerprint"
                    )
                return AppendResult("already_applied", existing[0])

        prev_hash = rows[-1][3] if rows else None
        payload = event.to_payload()
        digest = _event_hash(payload, prev_hash)
        payload_text = canonical_json({**payload, "prev_hash": prev_hash})
        candidate_seq = (rows[-1][0] + 1) if rows else 1
        candidate = [
            *rows,
            (candidate_seq, event.event_id, prev_hash, digest, payload_text),
        ]
        candidate_problems = _chain_problems(
            candidate, (candidate_seq, digest), allow_empty=False
        )
        if candidate_problems:
            raise AuditIncomplete(
                "event would break the audit chain: "
                + "; ".join(candidate_problems)
            )
        try:
            self._conn.execute(
                "INSERT INTO audit_events "
                "(event_id, prev_hash, event_hash, payload) VALUES (?, ?, ?, ?)",
                (event.event_id, prev_hash, digest, payload_text),
            )
        except sqlite3.IntegrityError:
            if not idempotent:
                raise
            existing = self._conn.execute(
                "SELECT event_hash, payload FROM audit_events WHERE event_id = ?",
                (event.event_id,),
            ).fetchone()
            if existing is None:
                raise AuditIncomplete(
                    "event_id integrity conflict without a stored event"
                )
            stored_payload = json.loads(existing[1])
            if logical_fingerprint(stored_payload) != logical_fingerprint(event):
                raise AuditIncomplete(
                    "event_id exists with a different logical fingerprint"
                )
            return AppendResult("already_applied", existing[0])
        self._conn.execute(
            "INSERT INTO audit_head (singleton, seq, event_hash) "
            "VALUES (1, ?, ?) "
            "ON CONFLICT(singleton) DO UPDATE SET "
            "seq = excluded.seq, event_hash = excluded.event_hash",
            (candidate_seq, digest),
        )
        return AppendResult("applied", digest)

    def transaction(self) -> "AuditTransaction":
        """Open one gatekeeper-owned write transaction on this connection."""
        return AuditTransaction(self)

    def append(self, event: AuditEvent) -> str:
        """Append one event in its own transaction (published behavior).

        A duplicate ``event_id`` is a public-append failure, as before; only the
        gatekeeper transaction's explicit idempotent re-read treats an identical
        fingerprint as already applied.
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
                result = self._append_in_tx(event, idempotent=False)
            except _BrokenChain as exc:
                rolled_back = _rollback(self._conn)
                raise ControlUnavailable(
                    "refusing to extend a broken audit chain: "
                    + "; ".join(exc.problems),
                    append_state="known_rolled_back" if rolled_back else "unknown",
                    reason_code=ReasonCode.CHAIN_BROKEN,
                ) from exc
            except AuditIncomplete as exc:
                rolled_back = _rollback(self._conn)
                raise ControlUnavailable(
                    "audit append failed before commit",
                    append_state="known_rolled_back" if rolled_back else "unknown",
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
                _rollback(self._conn)
                raise ControlUnavailable(
                    "audit commit outcome is unknown",
                    append_state="unknown",
                ) from exc
            return result.event_hash or ""


class AuditTransaction:
    """One write transaction; poisoned after an idempotent re-read."""

    def __init__(self, store: AuditStore):
        self._store = store
        self._conn = store._conn
        self._lock = _APPEND_LOCK
        self._lock.acquire()
        try:
            self._conn.execute("BEGIN IMMEDIATE")
        except Exception as exc:
            self._lock.release()
            raise ControlUnavailable(
                "could not start the audit transaction",
                append_state="known_rolled_back",
            ) from exc
        self._aborted = False
        self._closed = False
        self.result: AppendResult | None = None

    def _ensure_open(self) -> None:
        if self._closed:
            raise TransactionAborted("transaction is closed")
        if self._aborted:
            raise TransactionAborted("transaction was aborted by an idempotent re-read")

    def _abort(self) -> None:
        if not self._aborted:
            try:
                self._conn.execute("ROLLBACK")
            except Exception:
                pass
            self._aborted = True

    def append_in_tx(self, event: AuditEvent) -> AppendResult:
        """Append inside the open transaction, with idempotent re-read."""
        self._ensure_open()
        if not isinstance(event, AuditEvent):
            raise TypeError("append_in_tx requires an AuditEvent")
        try:
            result = self._store._append_in_tx(event, idempotent=True)
        except _BrokenChain as exc:
            self._abort()
            raise AuditIncomplete(
                "refusing to extend a broken audit chain: " + "; ".join(exc.problems)
            ) from exc
        except Exception:
            self._abort()
            raise
        if result.status == "already_applied":
            self.result = result
            self._abort()
        return result

    def head(self) -> tuple[int, str] | None:
        self._ensure_open()
        return self._conn.execute(
            "SELECT seq, event_hash FROM audit_head WHERE singleton = 1"
        ).fetchone()

    def __enter__(self) -> "AuditTransaction":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._closed:
            return False
        if self._aborted:
            self._closed = True
            self._lock.release()
            return False
        try:
            if exc_type is not None:
                self._conn.execute("ROLLBACK")
            else:
                self._conn.execute("COMMIT")
        except Exception as commit_exc:
            try:
                self._conn.execute("ROLLBACK")
            except Exception:
                pass
            self._closed = True
            self._lock.release()
            raise ControlUnavailable(
                "audit commit outcome is unknown",
                append_state="unknown",
            ) from commit_exc
        finally:
            self._closed = True
            self._lock.release()
        return False


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
