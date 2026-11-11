"""Pure admission decisions for the H1 control layer (Slice 2a).

This module contains no I/O, no database, no network and no delegate. A caller
supplies policy values explicitly, a trusted snapshot of run counters, and a
request; the decide functions return allow/deny plus a typed reason and, for an
allowed LLM call, an estimated token reservation.

The reservation is an estimate, not a spend guarantee. This layer does not own
the ledger, lease, serializer, tokenizer or audit record; those remain later
slices. RunView is a trusted caller snapshot: this module makes no claim that
the snapshot is consistent, atomic or correctly leased.
"""
from __future__ import annotations

import ipaddress
import math
import re
from dataclasses import dataclass
from enum import Enum

MAX_TEXT_LEN = 256
MAX_HOST_LEN = 253

_HOST_PATTERN = re.compile(
    r"^(?=.{1,253}$)"
    r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)*"
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$"
)


class AdmissionReason(str, Enum):
    """Typed reasons local to the pure admission layer."""

    ALLOWED = "allowed"
    POLICY_DENIED = "policy_denied"
    INVALID_TARGET = "invalid_target"
    UNAPPROVED_WRITE = "unapproved_write"
    BUDGET_EXCEEDED = "budget_exceeded"
    TOKENIZER_UNAVAILABLE = "tokenizer_unavailable"


def _plain_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an int")
    return value


def _int(value: object, name: str, *, minimum: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an int")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a bool")
    return value


def _text(value: object, name: str, *, max_length: int = MAX_TEXT_LEN) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty trimmed string")
    if len(value) > max_length:
        raise ValueError(f"{name} exceeds {max_length} characters")
    if any(ord(char) < 32 for char in value):
        raise ValueError(f"{name} must not contain control characters")
    return value


def _number(value: object, name: str) -> int | float:
    if type(value) not in (int, float):
        raise TypeError(f"{name} must be a number")
    return value


@dataclass(frozen=True)
class PolicyConfig:
    """Explicit caller policy; there are no live-policy defaults in this module."""

    max_run_tokens: int
    max_run_llm_calls: int
    max_output_tokens: int
    max_fetches_per_run: int
    reservation_margin: float
    provider_overhead_tokens: int
    allowed_model: str
    allowed_schemas: frozenset[str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "max_run_tokens", _int(self.max_run_tokens, "max_run_tokens", minimum=1)
        )
        object.__setattr__(
            self,
            "max_run_llm_calls",
            _int(self.max_run_llm_calls, "max_run_llm_calls", minimum=1),
        )
        object.__setattr__(
            self,
            "max_output_tokens",
            _int(self.max_output_tokens, "max_output_tokens", minimum=1),
        )
        object.__setattr__(
            self,
            "max_fetches_per_run",
            _int(self.max_fetches_per_run, "max_fetches_per_run", minimum=1),
        )
        if type(self.reservation_margin) is not float:
            raise TypeError("reservation_margin must be a float")
        if not math.isfinite(self.reservation_margin):
            raise ValueError("reservation_margin must be finite")
        if not 0.0 <= self.reservation_margin <= 1.0:
            raise ValueError("reservation_margin must be between 0 and 1")
        object.__setattr__(
            self,
            "provider_overhead_tokens",
            _int(
                self.provider_overhead_tokens,
                "provider_overhead_tokens",
                minimum=0,
            ),
        )
        object.__setattr__(
            self, "allowed_model", _text(self.allowed_model, "allowed_model")
        )
        if type(self.allowed_schemas) is not frozenset:
            raise TypeError("allowed_schemas must be a frozenset")
        for schema in self.allowed_schemas:
            _text(schema, "allowed_schemas item")


@dataclass(frozen=True)
class RunView:
    """Trusted caller snapshot of the run so far."""

    calls: int
    prompt_tokens: int
    completion_tokens: int
    reserved_tokens: int
    fetch_count: int
    approved: bool

    def __post_init__(self) -> None:
        for name in (
            "calls",
            "prompt_tokens",
            "completion_tokens",
            "reserved_tokens",
            "fetch_count",
        ):
            object.__setattr__(self, name, _int(getattr(self, name), name, minimum=0))
        object.__setattr__(self, "approved", _bool(self.approved, "approved"))


@dataclass(frozen=True)
class FetchRequest:
    host: str
    max_bytes: int
    timeout_seconds: int | float
    max_redirects: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "host", _text(self.host, "host", max_length=MAX_HOST_LEN)
        )
        object.__setattr__(
            self, "max_bytes", _plain_int(self.max_bytes, "max_bytes")
        )
        object.__setattr__(
            self,
            "timeout_seconds",
            _number(self.timeout_seconds, "timeout_seconds"),
        )
        object.__setattr__(
            self, "max_redirects", _plain_int(self.max_redirects, "max_redirects")
        )


@dataclass(frozen=True)
class LLMRequest:
    model: str
    schema_name: str
    serialized_body_token_count: int | None
    max_output_tokens: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "model", _text(self.model, "model"))
        object.__setattr__(
            self, "schema_name", _text(self.schema_name, "schema_name")
        )
        if self.serialized_body_token_count is not None:
            object.__setattr__(
                self,
                "serialized_body_token_count",
                _int(
                    self.serialized_body_token_count,
                    "serialized_body_token_count",
                    minimum=0,
                ),
            )
        object.__setattr__(
            self,
            "max_output_tokens",
            _int(self.max_output_tokens, "max_output_tokens", minimum=1),
        )


@dataclass(frozen=True)
class WriteRequest:
    target_id: str | None

    def __post_init__(self) -> None:
        if self.target_id is not None:
            object.__setattr__(
                self, "target_id", _text(self.target_id, "target_id")
            )


@dataclass(frozen=True)
class Decision:
    allow: bool
    reason: AdmissionReason
    reserve_tokens: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "allow", _bool(self.allow, "allow"))
        if not isinstance(self.reason, AdmissionReason):
            raise TypeError("reason must be an AdmissionReason")
        if self.reserve_tokens is not None:
            object.__setattr__(
                self,
                "reserve_tokens",
                _int(self.reserve_tokens, "reserve_tokens", minimum=0),
            )
        if not self.allow and self.reserve_tokens is not None:
            raise ValueError("a denied decision must not carry a reservation")


def _valid_host(host: str) -> bool:
    return _HOST_PATTERN.fullmatch(host.lower()) is not None


def _valid_ipv4(host: str) -> bool:
    try:
        ipaddress.IPv4Address(host)
    except ValueError:
        return False
    return True


def _host_is_syntactic(host: str) -> bool:
    # Syntactic only: no DNS, no public/private-IP judgement. The existing
    # fetch guard in net.py remains responsible for SSRF/DNS checks.
    if any(marker in host for marker in ("://", "/", "?", "#", "@", "\\", ":")):
        return False
    if any(char.isspace() for char in host):
        return False
    return _valid_ipv4(host) or _valid_host(host)


def _reserve(policy: PolicyConfig, token_count: int) -> int:
    """Exact integer-rational ceiling of (tokens + overhead) * (1 + margin)."""
    numerator, denominator = policy.reservation_margin.as_integer_ratio()
    scaled = (
        (token_count + policy.provider_overhead_tokens)
        * (numerator + denominator)
        + denominator
        - 1
    ) // denominator
    return scaled + policy.max_output_tokens


def decide_fetch(request: FetchRequest, view: RunView, policy: PolicyConfig) -> Decision:
    """Decide a fetch syntactically and against the fetch cap; no DNS or SSRF."""
    if not isinstance(request, FetchRequest):
        raise TypeError("request must be a FetchRequest")
    if not isinstance(view, RunView):
        raise TypeError("view must be a RunView")
    if not isinstance(policy, PolicyConfig):
        raise TypeError("policy must be a PolicyConfig")
    if not _host_is_syntactic(request.host):
        return Decision(False, AdmissionReason.INVALID_TARGET)
    if type(request.timeout_seconds) is float and not math.isfinite(
        request.timeout_seconds
    ):
        return Decision(False, AdmissionReason.INVALID_TARGET)
    if request.timeout_seconds <= 0:
        return Decision(False, AdmissionReason.INVALID_TARGET)
    if request.max_bytes <= 0 or request.max_redirects < 0:
        return Decision(False, AdmissionReason.INVALID_TARGET)
    if view.fetch_count + 1 > policy.max_fetches_per_run:
        return Decision(False, AdmissionReason.BUDGET_EXCEEDED)
    return Decision(True, AdmissionReason.ALLOWED)


def decide_llm(request: LLMRequest, view: RunView, policy: PolicyConfig) -> Decision:
    """Decide an LLM call using the caller-provided token count (no tokenizer)."""
    if not isinstance(request, LLMRequest):
        raise TypeError("request must be an LLMRequest")
    if not isinstance(view, RunView):
        raise TypeError("view must be a RunView")
    if not isinstance(policy, PolicyConfig):
        raise TypeError("policy must be a PolicyConfig")
    if request.max_output_tokens > policy.max_output_tokens:
        return Decision(False, AdmissionReason.POLICY_DENIED)
    if request.model != policy.allowed_model:
        return Decision(False, AdmissionReason.POLICY_DENIED)
    if request.schema_name not in policy.allowed_schemas:
        return Decision(False, AdmissionReason.POLICY_DENIED)
    if request.serialized_body_token_count is None:
        return Decision(False, AdmissionReason.TOKENIZER_UNAVAILABLE)

    reserve = _reserve(policy, request.serialized_body_token_count)
    used = view.prompt_tokens + view.completion_tokens + view.reserved_tokens
    if used + reserve > policy.max_run_tokens:
        return Decision(False, AdmissionReason.BUDGET_EXCEEDED)
    if view.calls + 1 > policy.max_run_llm_calls:
        return Decision(False, AdmissionReason.BUDGET_EXCEEDED)
    return Decision(True, AdmissionReason.ALLOWED, reserve_tokens=reserve)


def decide_write(request: WriteRequest, view: RunView, policy: PolicyConfig) -> Decision:
    """Decide a business write; target_id carries no authorization meaning."""
    if not isinstance(request, WriteRequest):
        raise TypeError("request must be a WriteRequest")
    if not isinstance(view, RunView):
        raise TypeError("view must be a RunView")
    if not isinstance(policy, PolicyConfig):
        raise TypeError("policy must be a PolicyConfig")
    if view.approved is not True:
        return Decision(False, AdmissionReason.UNAPPROVED_WRITE)
    return Decision(True, AdmissionReason.ALLOWED)
