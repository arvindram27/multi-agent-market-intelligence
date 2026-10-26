"""Business-persistence seam for approved runs.

The graph treats persistence as an injected dependency so tests can prove that a
rejected run never reaches it. ``PostgresProfileWriter`` is the durable
implementation; this module holds the contract, the explicit test/demo no-op, and
the configuration-based selection used when no writer is injected.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Iterable, Protocol

from market_intelligence.config import settings

if TYPE_CHECKING:
    from market_intelligence.graph.state import MarketIntelligenceState


class PersistenceError(RuntimeError):
    """A write failed and the database was reconciled as not containing it."""


class PersistenceNotConfigured(PersistenceError):
    """No durable store is configured, so an approved run cannot be saved."""


class CommitOutcomeUnknown(PersistenceError):
    """The commit may or may not have landed.

    Reconciliation could not observe a matching snapshot, but absence is not
    proof that the original transaction will not commit. Re-persisting the
    *unchanged* snapshot (same ``run_id`` and ``content_hash``) is safe because
    it is deduplicated; changed content is stored as a new version.
    """


class UnapprovedRunError(PersistenceError):
    """Refused to persist a run that was not explicitly approved."""


class ProfileWriter(Protocol):
    """Persists an approved run and returns its durable identifier."""

    def persist(self, run_id: str, state: "MarketIntelligenceState") -> str:
        """Write the approved profile and related records for ``run_id``."""
        ...


class NullProfileWriter:
    """Explicit no-op writer for tests and fixture demos ONLY.

    Normal execution never selects this writer: ``default_profile_writer``
    returns a Postgres writer when ``DATABASE_URL`` is set and an
    :class:`UnconfiguredProfileWriter` otherwise. This class exists so a test or
    the fixture demo can opt out of storage on purpose.
    """

    def persist(self, run_id: str, state: "MarketIntelligenceState") -> str:
        return run_id


class UnconfiguredProfileWriter:
    """Fails loudly instead of pretending an approved run was stored."""

    def __init__(self, reason: str | None = None):
        self.reason = reason or (
            "DATABASE_URL is not set; the approved run was not persisted. "
            "Configure the local Postgres store and retry the approval."
        )

    def persist(self, run_id: str, state: "MarketIntelligenceState") -> str:
        raise PersistenceNotConfigured(self.reason)


def persistence_configured() -> bool:
    """True when a database DSN is configured for business persistence."""
    return bool(settings.database_url.get_secret_value().strip())


def default_profile_writer() -> ProfileWriter:
    """Select the writer from configuration. Never silently no-ops."""
    dsn = settings.database_url.get_secret_value().strip()
    if not dsn:
        return UnconfiguredProfileWriter()
    try:
        from market_intelligence.persistence_postgres import PostgresProfileWriter
    except ImportError as exc:  # pragma: no cover - depends on install state
        return UnconfiguredProfileWriter(
            f"DATABASE_URL is set but the Postgres driver is unavailable ({exc}); "
            "the approved run was not persisted."
        )
    return PostgresProfileWriter(
        dsn,
        connect_timeout=settings.db_connect_timeout,
    )


_DSN_URI_RE = re.compile(r"(?i)\bpostgres(?:ql)?://[^\s'\"<>]+")
_PASSWORD_KV_RE = re.compile(
    r"(?i)\bpassword\s*=\s*(?:'[^']*'|\"[^\"]*\"|[^\s;'\"]+)"
)
# Any URL that embeds credentials in its authority section, not just Postgres.
_URL_CREDENTIALS_RE = re.compile(
    r"(?i)\b([a-z][a-z0-9+.\-]*://)[^/\s:@'\"<>]+:[^@\s/'\"<>]+@"
)


def redact_secrets(text: str, secrets: Iterable[str] = ()) -> str:
    """Remove DSNs and passwords from text that may reach state, UI or logs.

    Replacement is explicit (not just exception wrapping): every known secret is
    removed, then URI-shaped and ``password=`` shaped fragments are masked.
    """
    if not text:
        return ""
    cleaned = str(text)
    # Longest first so a full DSN is removed before its password fragment.
    for secret in sorted((s for s in secrets if s), key=len, reverse=True):
        cleaned = cleaned.replace(secret, "***")
    cleaned = _URL_CREDENTIALS_RE.sub(r"\1***:***@", cleaned)
    cleaned = _DSN_URI_RE.sub("postgresql://***", cleaned)
    cleaned = _PASSWORD_KV_RE.sub("password=***", cleaned)
    return cleaned


def sanitize_for_output(value: object, secrets: Iterable[str] = ()) -> str:
    """Public boundary for text that leaves persistence for state, UI or logs.

    Callers such as the graph must not trust a writer to have scrubbed its own
    exception. This applies pattern-based redaction to any exception or string
    and is safe to apply twice. It never formats a traceback or exception
    chain; only the type name and the redacted message are returned.
    """
    if value is None:
        return ""
    if isinstance(value, BaseException):
        message = redact_secrets(str(value), secrets)
        name = type(value).__name__
        return f"{name}: {message}" if message else name
    return redact_secrets(str(value), secrets)
