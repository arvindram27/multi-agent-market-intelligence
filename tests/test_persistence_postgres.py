"""Real-Postgres integration tests for durable approved-run persistence.

These tests need a running local Postgres (docker compose up -d db) and a
separate test database. Run them without ``--disable-socket``:

    .venv\\Scripts\\python.exe -m pytest -q -m integration

Every read/write here opens an independent connection so durability is proven
against the server, not against a cached connection object.
"""
import os
import re
import threading

import psycopg
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from market_intelligence.agents.schemas import (
    CompetitorProfileDraft,
    NewsDigest,
    NewsToneDraft,
    TopicAnalysis,
    TopicFinding,
)
from market_intelligence.config import settings
from market_intelligence.db import (
    SCHEMA_VERSION,
    apply_schema,
    verify_schema,
)
from market_intelligence.graph import build_graph, checkpoint_serde
from market_intelligence.graph.state import MarketIntelligenceState
from market_intelligence.net import FetchResult
from market_intelligence.persistence import PersistenceError, UnapprovedRunError
from market_intelligence.persistence_postgres import (
    PostgresProfileWriter,
    business_payload,
    canonical_json,
    content_hash,
)

pytestmark = [pytest.mark.integration, pytest.mark.enable_socket]


# --------------------------------------------------------------------------
# fixtures and small helpers
# --------------------------------------------------------------------------
def _test_dsn() -> str:
    return (
        os.environ.get("TEST_DATABASE_URL")
        or settings.test_database_url.get_secret_value()
    ).strip()


def _ensure_database(dsn: str) -> str:
    info = psycopg.conninfo.conninfo_to_dict(dsn)
    dbname = info.get("dbname") or ""
    if not dbname.endswith("_test"):
        pytest.fail(
            f"refusing to use {dbname!r} for destructive tests; "
            "the test database name must end with '_test'"
        )
    if not re.fullmatch(r"[A-Za-z0-9_]+", dbname):
        pytest.fail(f"unsupported test database name: {dbname!r}")
    admin = {**info, "dbname": "postgres"}
    with psycopg.connect(**admin, autocommit=True, connect_timeout=3) as conn:
        exists = conn.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (dbname,)
        ).fetchone()
        if not exists:
            conn.execute(f'CREATE DATABASE "{dbname}"')
    return dsn


@pytest.fixture(scope="session")
def pg_dsn():
    dsn = _test_dsn()
    if not dsn:
        pytest.skip("TEST_DATABASE_URL is not set")
    try:
        dsn = _ensure_database(dsn)
        with psycopg.connect(dsn, autocommit=True, connect_timeout=3) as conn:
            apply_schema(conn)
            problems = verify_schema(conn)
    except psycopg.Error as exc:
        pytest.skip(f"local Postgres unavailable: {type(exc).__name__}")
    if problems:
        pytest.fail(f"test database schema is incompatible: {problems}")
    return dsn


@pytest.fixture
def db(pg_dsn):
    """Isolated test database, emptied before each test (never the dev DB)."""
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        conn.execute("TRUNCATE runs RESTART IDENTITY CASCADE")
    return pg_dsn


def _rows(dsn, sql, params=()):
    with psycopg.connect(dsn, autocommit=True) as conn:
        return conn.execute(sql, params).fetchall()


def _one(dsn, sql, params=()):
    rows = _rows(dsn, sql, params)
    return rows[0][0] if rows else None


def _count(dsn, table):
    return _one(dsn, f"SELECT count(*) FROM {table}")


def _counts(dsn):
    return {
        table: _count(dsn, table)
        for table in (
            "runs",
            "run_articles",
            "run_topics",
            "run_topic_evidence",
            "run_tones",
            "competitor_profiles",
        )
    }


def _stored_payload(dsn, run_id):
    row = _rows(dsn, "SELECT payload FROM runs WHERE run_id = %s", (run_id,))
    return row[0][0] if row else None


def _state_from_values(values) -> MarketIntelligenceState:
    if isinstance(values, MarketIntelligenceState):
        return values
    return MarketIntelligenceState(**values)


# --------------------------------------------------------------------------
# 1. approved write, child counts, payload round-trip, provenance
# --------------------------------------------------------------------------
def test_approved_write_persists_every_child(db, sample_state):
    state = sample_state()
    # Values described as exact must survive without decimal rounding.
    state.topics[0].model_confidence = 0.30000000000000004
    state.tones[0].score = -0.1234567890123
    writer = PostgresProfileWriter(db)

    returned = writer.persist("run-approved", state)

    assert returned == "run-approved"
    counts = _counts(db)
    print("child counts:", counts)
    assert counts == {
        "runs": 1,
        "run_articles": 1,
        "run_topics": 1,
        "run_topic_evidence": 2,
        "run_tones": 1,
        "competitor_profiles": 1,
    }

    header = _rows(
        db,
        "SELECT run_id, content_hash, schema_version, num_articles "
        "FROM runs WHERE run_id = %s",
        ("run-approved",),
    )[0]
    print("run header:", header)
    assert header[1] == content_hash(canonical_json(business_payload(state)))
    assert header[2] == SCHEMA_VERSION

    article = _rows(
        db,
        "SELECT article_id, published_at, published_at_raw, retrieved_at, "
        "retrieved_at_raw, content_hash FROM run_articles",
    )[0]
    print("article provenance:", article)
    assert article[0] == "a1"
    assert article[1] is not None and article[1].microsecond == 0
    assert article[2] == "2026-10-20T10:00:00+00:00"
    assert article[3].microsecond == 123456
    assert article[4] == "2026-10-21T12:00:00.123456+00:00"
    assert article[5] == "a" * 64

    evidence = _rows(
        db,
        "SELECT article_id, matched_article_id, verified "
        "FROM run_topic_evidence ORDER BY verified",
    )
    print("topic evidence:", evidence)
    assert ("a1", "a1", True) in evidence
    assert ("invented-article-id", None, False) in evidence

    confidence = _one(db, "SELECT model_confidence FROM run_topics")
    score = _one(db, "SELECT score FROM run_tones")
    print("exact numeric values:", confidence, score)
    assert confidence == 0.30000000000000004
    assert score == -0.1234567890123

    stored = _stored_payload(db, "run-approved")
    rebuilt = MarketIntelligenceState(**stored)
    assert business_payload(rebuilt) == business_payload(state)


def test_unparseable_published_at_is_preserved(db, sample_state):
    state = sample_state()
    state.summaries[0].published_at = "not-a-timestamp"

    PostgresProfileWriter(db).persist("run-raw-ts", state)

    row = _rows(
        db, "SELECT published_at, published_at_raw FROM run_articles"
    )[0]
    print("unparseable timestamp row:", row)
    assert row[0] is None
    assert row[1] == "not-a-timestamp"
    assert _stored_payload(db, "run-raw-ts")["summaries"][0]["published_at"] == (
        "not-a-timestamp"
    )


# --------------------------------------------------------------------------
# 2. identical retry and changed content
# --------------------------------------------------------------------------
def test_identical_retry_does_not_duplicate(db, sample_state):
    state = sample_state()
    writer = PostgresProfileWriter(db)

    first = writer.persist("run-retry", state)
    after_first = _counts(db)
    second = writer.persist("run-retry", state)
    after_second = _counts(db)

    print("counts after first:", after_first)
    print("counts after retry:", after_second)
    assert first == second == "run-retry"
    assert after_first == after_second
    assert after_second["runs"] == 1


def test_changed_content_creates_a_new_visible_version(db, sample_state):
    writer = PostgresProfileWriter(db)
    original = sample_state()
    writer.persist("run-versioned", original)

    changed = sample_state(summary="s1 but materially different")
    writer.persist("run-versioned", changed)

    versions = _rows(
        db,
        "SELECT id, content_hash, payload->'summaries'->0->>'summary' "
        "FROM runs WHERE run_id = %s ORDER BY id",
        ("run-versioned",),
    )
    latest = _rows(
        db,
        "SELECT id, content_hash FROM latest_runs WHERE run_id = %s",
        ("run-versioned",),
    )
    print("versions:", versions)
    print("latest view:", latest)
    assert len(versions) == 2
    assert versions[0][2] == "s1"
    assert versions[1][2] == "s1 but materially different"
    assert _count(db, "run_articles") == 2
    assert latest[0][0] == versions[-1][0]


def test_concurrent_identical_writes_store_once(db, sample_state):
    barrier = threading.Barrier(2)
    results, errors = [], []

    def attempt():
        try:
            # Independent writer => independent connection.
            writer = PostgresProfileWriter(db)
            barrier.wait(timeout=10)
            results.append(writer.persist("run-concurrent", sample_state()))
        except Exception as exc:  # collected and asserted below
            errors.append(exc)

    threads = [threading.Thread(target=attempt) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    counts = _counts(db)
    print("concurrent results:", results)
    print("concurrent counts:", counts)
    assert not errors, errors
    assert results == ["run-concurrent", "run-concurrent"]
    assert counts["runs"] == 1
    assert counts["run_articles"] == 1
    assert counts["run_topic_evidence"] == 2


# --------------------------------------------------------------------------
# 3. failure, recovery and rollback
# --------------------------------------------------------------------------
def test_failure_then_recovery(db, sample_state):
    info = psycopg.conninfo.conninfo_to_dict(db)
    bad_dsn = psycopg.conninfo.make_conninfo(**{**info, "port": "1"})
    failing = PostgresProfileWriter(bad_dsn, connect_timeout=2)

    with pytest.raises(PersistenceError) as excinfo:
        failing.persist("run-recover", sample_state())

    print("failure message:", str(excinfo.value))
    assert _count(db, "runs") == 0
    assert info.get("password", "") not in str(excinfo.value)

    PostgresProfileWriter(db).persist("run-recover", sample_state())
    assert _count(db, "runs") == 1


def test_rejected_run_is_never_persisted(db, fake_llm, monkeypatch):
    graph, config = _graph_for(db, fake_llm, monkeypatch)
    graph.invoke(_inputs(), config)
    graph.invoke(Command(resume={"approved": False}), config)

    values = _state_from_values(graph.get_state(config).values)
    print("rejected status:", values.status)
    assert values.status == "rejected"
    assert _count(db, "runs") == 0


def test_writer_guard_rejects_unapproved_state_without_writing(db, sample_state):
    with pytest.raises(UnapprovedRunError):
        PostgresProfileWriter(db).persist("run-unapproved", sample_state(approved=False))
    assert _count(db, "runs") == 0


def test_child_constraint_failure_rolls_back_the_snapshot(db, sample_state):
    state = sample_state()
    # A tone for an article that is not in this snapshot violates the FK.
    state.tones = [state.tones[0].model_copy(update={"article_id": "ghost-article"})]

    with pytest.raises(PersistenceError):
        PostgresProfileWriter(db).persist("run-rollback", state)

    print("counts after rollback:", _counts(db))
    assert _count(db, "runs") == 0
    assert _count(db, "run_articles") == 0


# --------------------------------------------------------------------------
# 4. lost commit confirmation
# --------------------------------------------------------------------------
class _LostAckTransaction:
    """Commit for real, then pretend the acknowledgement was lost."""

    def __init__(self, inner):
        self._inner = inner

    def __enter__(self):
        return self._inner.__enter__()

    def __exit__(self, exc_type, exc, tb):
        result = self._inner.__exit__(exc_type, exc, tb)
        if exc_type is None:
            raise psycopg.OperationalError(
                "connection lost before commit confirmation was received"
            )
        return result


class _LostAckConnection:
    def __init__(self, conn):
        self._conn = conn

    def transaction(self):
        return _LostAckTransaction(self._conn.transaction())

    def execute(self, *args, **kwargs):
        return self._conn.execute(*args, **kwargs)

    def close(self):
        return self._conn.close()


def test_lost_commit_confirmation_reconciles_without_duplicates(db, sample_state):
    used = {"first": False}

    def factory():
        conn = psycopg.connect(db, autocommit=True)
        if not used["first"]:
            used["first"] = True
            return _LostAckConnection(conn)
        return conn

    writer = PostgresProfileWriter(db, connection_factory=factory)
    returned = writer.persist("run-lost-ack", sample_state())

    counts = _counts(db)
    print("counts after reconciled lost ack:", counts)
    assert returned == "run-lost-ack"
    assert counts["runs"] == 1

    # A retry after the unknown outcome stays deduplicated.
    writer.persist("run-lost-ack", sample_state())
    assert _count(db, "runs") == 1


def test_lost_commit_and_lost_reconciliation_is_unknown(db, sample_state):
    used = {"first": False}

    def factory():
        if used["first"]:
            raise psycopg.OperationalError("cannot reconnect to reconcile")
        used["first"] = True
        return _LostAckConnection(psycopg.connect(db, autocommit=True))

    writer = PostgresProfileWriter(db, connection_factory=factory)
    from market_intelligence.persistence import CommitOutcomeUnknown

    with pytest.raises(CommitOutcomeUnknown):
        writer.persist("run-unknown", sample_state())

    # The outcome really was unknown: the row committed and must not be lost.
    print("row present after unknown outcome:", _count(db, "runs"))
    assert _count(db, "runs") == 1

    PostgresProfileWriter(db).persist("run-unknown", sample_state())
    assert _count(db, "runs") == 1


# --------------------------------------------------------------------------
# 5. setup is initial setup, not migrations
# --------------------------------------------------------------------------
def test_setup_twice_is_safe_and_incompatible_schema_is_detected(db):
    with psycopg.connect(db, autocommit=True) as conn:
        apply_schema(conn)
        apply_schema(conn)
        assert verify_schema(conn) == []

        # Probe an incompatible existing shape inside a transaction that is
        # rolled back, so no real object is ever dropped or altered.
        conn.execute("BEGIN")
        conn.execute("CREATE SCHEMA setup_probe")
        conn.execute("SET LOCAL search_path TO setup_probe")
        conn.execute("CREATE TABLE runs (id bigint)")
        problems = verify_schema(conn)
        conn.execute("ROLLBACK")

    print("incompatible-schema problems:", problems)
    assert any("runs" in problem and "missing columns" in problem for problem in problems)
    assert (
        _one(
            db,
            "SELECT count(*) FROM information_schema.schemata "
            "WHERE schema_name = 'setup_probe'",
        )
        == 0
    )


# --------------------------------------------------------------------------
# 6. graph-level approved run
# --------------------------------------------------------------------------
FEED_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Test feed</title>
<item><title>T1</title><description>content one is a sufficiently long RSS excerpt for summarization</description></item>
</channel></rss>"""


def _responses():
    return {
        NewsDigest: NewsDigest(headline="h", summary="s1", tone_hint="neutral"),
        TopicAnalysis: TopicAnalysis(
            topics=[TopicFinding(topic_name="AI", model_confidence=0.8)]
        ),
        NewsToneDraft: NewsToneDraft(overall="positive", score=0.5),
        CompetitorProfileDraft: CompetitorProfileDraft(
            company="Acme", positioning="leader"
        ),
    }


def _mock_io(monkeypatch):
    def fake_fetch(url, **kwargs):
        if "feed" in url:
            return FetchResult(url, FEED_XML, "application/rss+xml", 200)
        return FetchResult(
            url,
            b"<html><body>Acme corp is a leading widget maker.</body></html>",
            "text/html",
            200,
        )

    monkeypatch.setattr("market_intelligence.agents.news.fetch_bytes", fake_fetch)
    monkeypatch.setattr("market_intelligence.agents.competitor.fetch_bytes", fake_fetch)


def _inputs():
    return {
        "rss_url": "http://example.com/feed.xml",
        "competitor_url": "https://acme.com",
        "num_articles": 1,
    }


def _graph_for(db, fake_llm, monkeypatch):
    _mock_io(monkeypatch)
    graph = build_graph(
        checkpointer=InMemorySaver(serde=checkpoint_serde()),
        llm=fake_llm(_responses()),
        writer=PostgresProfileWriter(db),
    )
    return graph, {"configurable": {"thread_id": "pg-graph-run"}}


def test_graph_approved_run_is_durable_after_api_returns(db, fake_llm, monkeypatch):
    graph, config = _graph_for(db, fake_llm, monkeypatch)
    graph.invoke(_inputs(), config)
    graph.invoke(Command(resume={"approved": True}), config)

    values = _state_from_values(graph.get_state(config).values)
    print("graph status:", values.status)
    assert values.status == "complete"
    assert values.persisted_run_id == "pg-graph-run"

    header = _rows(
        db,
        "SELECT run_id, content_hash FROM runs WHERE run_id = %s",
        ("pg-graph-run",),
    )
    print("persisted header from a new connection:", header)
    assert len(header) == 1
    assert header[0][1] == content_hash(
        canonical_json(business_payload(values))
    )

    stored = _stored_payload(db, "pg-graph-run")
    rebuilt = MarketIntelligenceState(**stored)
    assert business_payload(rebuilt) == business_payload(values)
    assert _count(db, "run_articles") == 1
    assert _count(db, "run_topics") == 1
    assert _count(db, "competitor_profiles") == 1
