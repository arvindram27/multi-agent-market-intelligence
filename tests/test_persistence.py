"""Offline contracts: writer selection, approval guard, hashing and redaction.

No sockets are opened here; the connection factory is injected or never reached.
"""
import logging

import psycopg
import pytest
from pydantic import SecretStr

from market_intelligence.config import settings
from market_intelligence.graph.nodes import persist_node
from market_intelligence.persistence import (
    PersistenceError,
    PersistenceNotConfigured,
    UnapprovedRunError,
    UnconfiguredProfileWriter,
    default_profile_writer,
    redact_secrets,
)
from market_intelligence.persistence_postgres import (
    PostgresProfileWriter,
    business_payload,
    canonical_json,
    content_hash,
)


def _run_config(thread_id="run-1"):
    return {"configurable": {"thread_id": thread_id}}


def test_missing_configuration_fails_loudly(monkeypatch, sample_state):
    monkeypatch.setattr(settings, "database_url", SecretStr(""))

    result = persist_node(sample_state(), _run_config())

    assert result["status"] == "persist_failed"
    assert "DATABASE_URL" in result["error"]
    assert "persisted_run_id" not in result


def test_default_writer_is_never_the_null_writer(monkeypatch):
    monkeypatch.setattr(settings, "database_url", SecretStr(""))
    assert isinstance(default_profile_writer(), UnconfiguredProfileWriter)

    monkeypatch.setattr(
        settings,
        "database_url",
        SecretStr("postgresql://market_intel:fake@127.0.0.1:1/db"),  # pragma: allowlist secret
    )
    assert isinstance(default_profile_writer(), PostgresProfileWriter)


def test_unconfigured_writer_raises_rather_than_pretending(sample_state):
    with pytest.raises(PersistenceNotConfigured, match="DATABASE_URL"):
        UnconfiguredProfileWriter().persist("run-1", sample_state())


def test_writer_refuses_unapproved_state(sample_state):
    def forbidden_factory():  # must never be reached
        raise AssertionError("the writer connected before checking approval")

    writer = PostgresProfileWriter(
        "postgresql://market_intel:fake@127.0.0.1:1/db",  # pragma: allowlist secret
        connection_factory=forbidden_factory,
    )
    with pytest.raises(UnapprovedRunError):
        writer.persist("run-1", sample_state(approved=False))
    with pytest.raises(UnapprovedRunError):
        writer.persist("run-1", sample_state(approved=None))


def test_canonical_hash_is_stable_and_content_sensitive(sample_state):
    first = sample_state()
    same = sample_state()
    changed = sample_state()
    changed.summaries[0].summary = "a different summary"

    def digest(state):
        return content_hash(canonical_json(business_payload(state)))

    assert digest(first) == digest(same)
    assert digest(first) != digest(changed)
    assert len(digest(first)) == 64
    # Bookkeeping must not influence identity.
    first.approved = None
    first.status = "persist_unknown"
    first.error = "irrelevant"
    assert digest(first) == digest(same)
    assert "status" not in business_payload(first)


def test_redaction_removes_fake_credentials():
    secret = "fake-password-9f3a"  # pragma: allowlist secret
    dsn = f"postgresql://market_intel:{secret}@127.0.0.1:5432/market_intelligence"
    message = f"could not connect to {dsn}; password={secret} rejected"

    cleaned = redact_secrets(message, [dsn, secret])

    assert secret not in cleaned
    assert dsn not in cleaned
    assert "127.0.0.1" not in cleaned


def test_connection_failure_never_leaks_credentials(sample_state, caplog):
    secret = "fake-password-7c1d"  # pragma: allowlist secret
    dsn = f"postgresql://market_intel:{secret}@127.0.0.1:5999/market_intelligence"

    def failing_factory():
        raise psycopg.OperationalError(f"could not connect to {dsn}")

    writer = PostgresProfileWriter(dsn, connection_factory=failing_factory)
    with caplog.at_level(logging.ERROR):
        with pytest.raises(PersistenceError) as info:
            writer.persist("run-1", sample_state())

    assert secret not in str(info.value)
    assert dsn not in str(info.value)
    assert secret not in caplog.text
    assert dsn not in caplog.text
    # The original exception must not ride along in a chained traceback.
    assert info.value.__cause__ is None
    assert info.value.__suppress_context__ is True
