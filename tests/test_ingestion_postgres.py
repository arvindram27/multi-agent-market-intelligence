"""Real-Postgres integration tests for the I1b ingestion sweep writer.

These tests need a running local Postgres (``docker compose up -d db``) and a
separate test database. Run them without ``--disable-socket``:

    .venv\\Scripts\\python.exe -m pytest -q -m integration

The CI gate never runs this file: the offline suite selects it out with
``-m "not integration"``; it must only import cleanly at collection. Every
read/write here opens independent connections so durability is proven against
the server, not against a cached object.
"""
import os
import re
import threading
from contextlib import contextmanager

import psycopg
import pytest

from market_intelligence.config import settings
from market_intelligence.db.ingestion import (
    apply_ingestion_schema,
    verify_ingestion_schema,
)
from market_intelligence.ingestion import INGEST_SWEEP_LOCK_KEY
from market_intelligence.ingestion.collector import CollectedArticle, SourceResult, SweepResult
from market_intelligence.ingestion.store import (
    IngestionSweepWriter,
    SweepLockedError,
    SweepOutcomeUnknown,
    SweepStoreError,
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
            apply_ingestion_schema(conn)
            problems = verify_ingestion_schema(conn)
    except psycopg.Error as exc:
        pytest.skip(f"local Postgres unavailable: {type(exc).__name__}")
    if problems:
        pytest.fail(f"test database schema is incompatible: {problems}")
    return dsn


@pytest.fixture
def db(pg_dsn):
    """Isolated ingestion tables, emptied before each test."""
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        conn.execute(
            "TRUNCATE ingest_run_articles, raw_articles, ingest_runs, "
            "ingest_sources RESTART IDENTITY CASCADE"
        )
    return pg_dsn


def _rows(dsn, sql, params=()):
    with psycopg.connect(dsn, autocommit=True) as conn:
        return conn.execute(sql, params).fetchall()


def _one(dsn, sql, params=()):
    rows = _rows(dsn, sql, params)
    return rows[0][0] if rows else None


def _counts(dsn):
    return {
        table: _one(dsn, f"SELECT count(*) FROM {table}")
        for table in (
            "ingest_runs",
            "ingest_sources",
            "raw_articles",
            "ingest_run_articles",
        )
    }


_SEEN_T1 = "2026-11-09T10:00:00+00:00"
_SEEN_T2 = "2026-11-09T11:00:00+00:00"
_SEEN_T3 = "2026-11-09T12:00:00+00:00"


def _article(
    *,
    entry_key: str = "link:https://example.com/first",
    seen_at: str = _SEEN_T1,
    **overrides,
) -> CollectedArticle:
    fields = {
        "source_key": "one",
        "title": "First story",
        "excerpt": "A sufficiently long RSS excerpt about the first story.",
        "feed_url": "https://f.example/feed.xml",
        "source_url": "https://example.com/first",
        "content_kind": "rss_excerpt",
        "published_at_raw": None,
        "retrieved_at_raw": seen_at,
        "seen_at": seen_at,
        "content_hash": "a" * 64,
        "article_id": "0123456789abcdef",
        "entry_key": entry_key,
        "entry_key_hash": "b" * 64,
        "version_hash": "c" * 64,
    }
    fields.update(overrides)
    return CollectedArticle(**fields)


def _source_result(key: str, *, status: str = "ok", articles=(), enabled: bool = True,
                   error_class: str | None = None, articles_seen: int = 0,
                   url: str | None = None):
    return SourceResult(
        source_key=key,
        kind="rss",
        url=url or f"https://f.example/{key}.xml",
        label=key.upper(),
        enabled=enabled,
        status=status,
        articles=tuple(articles),
        warnings=(),
        error_class=error_class,
        articles_seen=articles_seen,
    )


def _writer(dsn, connection_factory=None) -> IngestionSweepWriter:
    return IngestionSweepWriter(dsn, connection_factory=connection_factory)


# --------------------------------------------------------------------------
# 1. initial setup, not migrations
# --------------------------------------------------------------------------
def test_apply_schema_twice_is_safe_and_verify_is_clean(db):
    with psycopg.connect(db, autocommit=True) as conn:
        apply_ingestion_schema(conn)
        apply_ingestion_schema(conn)
        assert verify_ingestion_schema(conn) == []


def test_verify_rejects_wrong_column_type(db):
    """Probe an incompatible shape in a rolled-back transaction."""

    @contextmanager
    def probe():
        conn = psycopg.connect(db, autocommit=True)
        try:
            conn.execute("BEGIN")
            conn.execute("CREATE SCHEMA ingest_probe_type")
            conn.execute("SET LOCAL search_path TO ingest_probe_type")
            apply_ingestion_schema(conn)
            yield conn
        finally:
            conn.execute("ROLLBACK")
            conn.close()
        with psycopg.connect(db, autocommit=True) as check:
            assert check.execute(
                "SELECT 1 FROM information_schema.schemata "
                "WHERE schema_name = 'ingest_probe_type'"
            ).fetchone() is None

    with probe() as conn:
        conn.execute(
            "ALTER TABLE raw_articles ALTER COLUMN title TYPE varchar(2048)"
        )
        problems = verify_ingestion_schema(conn)
    print("wrong-type problems:", problems)
    assert any(
        "raw_articles" in problem and "wrong column types" in problem for problem in problems
    )


def test_verify_rejects_missing_identity_constraint(db):
    conn = psycopg.connect(db, autocommit=True)
    try:
        conn.execute("BEGIN")
        conn.execute("CREATE SCHEMA ingest_probe_identity")
        conn.execute("SET LOCAL search_path TO ingest_probe_identity")
        apply_ingestion_schema(conn)
        conn.execute(
            "ALTER TABLE raw_articles DROP CONSTRAINT raw_articles_identity"
        )
        problems = verify_ingestion_schema(conn)
        conn.execute("ROLLBACK")
    finally:
        conn.close()
    print("missing-identity problems:", problems)
    assert any("raw_articles_identity" in problem for problem in problems)


def test_verify_rejects_missing_latest_articles_view(db):
    conn = psycopg.connect(db, autocommit=True)
    try:
        conn.execute("BEGIN")
        conn.execute("CREATE SCHEMA ingest_probe_view")
        conn.execute("SET LOCAL search_path TO ingest_probe_view")
        apply_ingestion_schema(conn)
        conn.execute("DROP VIEW latest_articles")
        problems = verify_ingestion_schema(conn)
        conn.execute("ROLLBACK")
    finally:
        conn.close()
    print("missing-view problems:", problems)
    assert any("latest_articles view is missing" in problem for problem in problems)


# --------------------------------------------------------------------------
# 2. one sweep, one transaction
# --------------------------------------------------------------------------
def _single_article_sweep(article=None):
    article = article or _article()
    return SweepResult(
        source_results=(
            _source_result("one", articles=(article,), articles_seen=1),
        ),
        status="complete",
    )


def test_sweep_write_persists_run_sources_articles_and_memberships(db):
    result = _writer(db).write_sweep(_single_article_sweep())

    print("write result:", result)
    assert result.outcome == "committed"
    assert result.articles_stored == 1
    counts = _counts(db)
    assert counts == {
        "ingest_runs": 1,
        "ingest_sources": 1,
        "raw_articles": 1,
        "ingest_run_articles": 1,
    }
    run = _rows(
        db,
        "SELECT sweep_id, status, sources_attempted, sources_succeeded, "
        "articles_seen FROM ingest_runs",
    )[0]
    print("run row:", run)
    assert run[0] == result.sweep_id
    assert run[1] == "complete"
    assert (run[2], run[3]) == (1, 1)
    assert _one(db, "SELECT seen_at FROM ingest_run_articles") is not None


def test_run_counts_stored_and_replayed_recorded(db):
    article = _article()
    writer = _writer(db)
    writer.write_sweep(SweepResult(
        source_results=(_source_result("one", articles=(article,), articles_seen=1),),
        status="complete",
    ))
    seen_again = _article(seen_at=_SEEN_T2)
    writer.write_sweep(SweepResult(
        source_results=(_source_result("one", articles=(seen_again,), articles_seen=1),),
        status="complete",
    ))

    stored_rows = _rows(
        db,
        "SELECT articles_stored, articles_replayed FROM ingest_runs ORDER BY id",
    )
    print("run counts:", stored_rows)
    assert stored_rows[0] == (1, 0)
    assert stored_rows[1] == (0, 1)


def test_reseeing_the_same_version_replays_and_advances_last_seen(db):
    writer = _writer(db)
    first = writer.write_sweep(
        SweepResult(
            source_results=(_source_result("one", articles=(_article(),), articles_seen=1),),
            status="complete",
        )
    )
    assert first.articles_stored == 1 and first.articles_replayed == 0

    second = writer.write_sweep(
        SweepResult(
            source_results=(
                _source_result("one", articles=(_article(seen_at=_SEEN_T2),), articles_seen=1),
            ),
            status="complete",
        )
    )
    print("second sweep result:", second)
    assert second.articles_stored == 0 and second.articles_replayed == 1
    assert _one(db, "SELECT count(*) FROM raw_articles") == 1
    assert _one(
        db, "SELECT count(*) FROM ingest_run_articles"
    ) == 2  # one membership per run
    last_seen = _one(db, "SELECT last_seen_at FROM raw_articles")
    print("last_seen after replay:", last_seen)
    assert last_seen.isoformat() == "2026-11-09T11:00:00+00:00"


def test_changed_content_creates_versions_and_latest_view_returns_last(db):
    """A -> B -> A reversion: latest_articles returns A (newest last_observation)."""
    writer = _writer(db)
    first = _article(entry_key="link:https://example.com/same", seen_at=_SEEN_T1,
                     version_hash="c" * 64)
    second = _article(entry_key="link:https://example.com/same", seen_at=_SEEN_T2,
                      version_hash="d" * 64, article_id="fedcba9876543210")
    again = _article(entry_key="link:https://example.com/same", seen_at=_SEEN_T3,
                     version_hash="c" * 64)
    source = _source_result("one", articles=(first, second, again), articles_seen=3)

    writer.write_sweep(SweepResult(source_results=(source,), status="complete"))

    counts = _counts(db)
    print("counts after A/B/A:", counts)
    assert counts["raw_articles"] == 2
    assert counts["ingest_run_articles"] == 2
    latest = _rows(
        db,
        "SELECT source_id, entry_key, version_hash, last_seen_at FROM latest_articles",
    )
    print("latest_articles rows:", latest)
    assert len(latest) == 1
    assert latest[0][2] == "c" * 64
    assert latest[0][3].isoformat() == "2026-11-09T12:00:00+00:00"


def test_repeated_sweeps_replay_without_duplicate_versions(db):
    writer = _writer(db)
    for seen in (_SEEN_T1, _SEEN_T2, _SEEN_T3):
        writer.write_sweep(
            SweepResult(
                source_results=(
                    _source_result("one", articles=(_article(seen_at=seen),), articles_seen=1),
                ),
                status="complete",
            )
        )
    counts = _counts(db)
    print("counts after three identical sweeps:", counts)
    assert counts == {
        "ingest_runs": 3,
        "ingest_sources": 1,
        "raw_articles": 1,
        "ingest_run_articles": 3,
    }


def test_unparseable_published_at_is_preserved_and_published_at_is_null(db):
    article = _article(published_at_raw="not-a-timestamp")
    _writer(db).write_sweep(_single_article_sweep(article))

    published, published_raw = _rows(
        db, "SELECT published_at, published_at_raw FROM raw_articles"
    )[0]
    print("unparseable published_at row:", published, published_raw)
    assert published is None
    assert published_raw == "not-a-timestamp"


def test_source_metadata_syncs_between_sweeps(db):
    writer = _writer(db)
    writer.write_sweep(
        SweepResult(
            source_results=(_source_result("one", articles=(_article(),), articles_seen=1),),
            status="complete",
        )
    )
    writer.write_sweep(
        SweepResult(
            source_results=(
                _source_result("one", url="https://f.example/one-moved.xml"),
            ),
            status="complete",
        )
    )
    url_count = _one(db, "SELECT count(*) FROM ingest_sources")
    url = _one(db, "SELECT url FROM ingest_sources")
    print("sources after re-sync:", url_count, url)
    assert url_count == 1
    assert url == "https://f.example/one-moved.xml"
    assert _one(
        db, "SELECT feed_url FROM raw_articles"
    ) == "https://f.example/feed.xml"  # the historical article row keeps its own feed_url


def test_snapshot_records_failed_and_disabled_sources(db):
    source = _source_result("one", articles=(_article(),), articles_seen=1)
    failed = _source_result("bad", status="failed", error_class="TransientFetchError")
    disabled = _source_result("off", status="disabled", enabled=False)
    _writer(db).write_sweep(
        SweepResult(source_results=(source, failed, disabled), status="partial")
    )

    snapshot = _one(db, "SELECT source_results FROM ingest_runs")
    keys = [row["source_key"] for row in snapshot]
    print("snapshot keys:", keys)
    assert keys == ["one", "bad", "off"]
    run = _rows(db, "SELECT status, sources_attempted, sources_succeeded FROM ingest_runs")[0]
    assert run == ("partial", 2, 1)


# --------------------------------------------------------------------------
# 3. failures roll back inside the real server
# --------------------------------------------------------------------------
def test_entry_key_shape_violation_rolls_back_everything(db):
    article = _article(entry_key="bogus:not-a-shape")
    with pytest.raises(SweepStoreError, match="rolled back") as excinfo:
        _writer(db).write_sweep(_single_article_sweep(article))

    print("constraint failure message:", excinfo.value)
    counts = _counts(db)
    print("counts after rollback:", counts)
    assert counts == {
        "ingest_runs": 0,
        "ingest_sources": 0,
        "raw_articles": 0,
        "ingest_run_articles": 0,
    }


def test_article_id_length_violation_rolls_back_and_recovers(db):
    article = _article(article_id="short")
    with pytest.raises(SweepStoreError, match="rolled back"):
        _writer(db).write_sweep(_single_article_sweep(article))
    assert _counts(db)["ingest_runs"] == 0

    # A well-formed write right afterwards proves the writer recovered.
    result = _writer(db).write_sweep(_single_article_sweep())
    assert result.outcome == "committed"


def test_connect_failure_is_known_failure_without_writing(db):
    info = psycopg.conninfo.conninfo_to_dict(db)
    bad_dsn = psycopg.conninfo.make_conninfo(**{**info, "port": "1"})

    with pytest.raises(SweepStoreError, match="no rows were stored") as excinfo:
        _writer(bad_dsn).write_sweep(_single_article_sweep())

    print("connect failure message:", excinfo.value)
    assert _counts(db)["ingest_runs"] == 0
    password = info.get("password")
    if password:
        assert password not in str(excinfo.value)


# --------------------------------------------------------------------------
# 4. advisory-lock serialization
# --------------------------------------------------------------------------
def test_advisory_lock_contention_fails_fast_before_writes(db):
    holder = psycopg.connect(db, autocommit=True)
    try:
        holder.execute("BEGIN")
        holder.execute(
            "SELECT pg_try_advisory_xact_lock(%s)", (INGEST_SWEEP_LOCK_KEY,)
        )
        with pytest.raises(SweepLockedError, match="advisory lock"):
            _writer(db).write_sweep(_single_article_sweep())
        assert _counts(db)["ingest_runs"] == 0
    finally:
        holder.close()


def test_advisory_lock_is_released_by_commit(db):
    holder = psycopg.connect(db, autocommit=True)
    try:
        holder.execute("BEGIN")
        holder.execute(
            "SELECT pg_try_advisory_xact_lock(%s)", (INGEST_SWEEP_LOCK_KEY,)
        )
        holder.execute("ROLLBACK")
    finally:
        holder.close()
    result = _writer(db).write_sweep(_single_article_sweep())
    assert result.outcome == "committed"
    assert _counts(db)["ingest_runs"] == 1


# --------------------------------------------------------------------------
# 5. lost commit acknowledgement
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


def test_lost_commit_confirmation_reconciles(db):
    used = {"first": False}

    def factory():
        conn = psycopg.connect(db, autocommit=True)
        if not used["first"]:
            used["first"] = True
            return _LostAckConnection(conn)
        return conn

    result = _writer(db, factory).write_sweep(_single_article_sweep())

    print("reconciled result:", result)
    assert result.outcome == "reconciled"
    assert result.articles_stored is None
    counts = _counts(db)
    print("counts after reconciled lost ack:", counts)
    assert counts["ingest_runs"] == 1
    assert counts["raw_articles"] == 1


def test_reconciled_sweep_was_not_written_twice(db):
    # The sweep writes once: a second write_sweep call is a NEW sweep id, but
    # the version row deduplicates by identity and membership rows belong to
    # their own run.
    first = _writer(db).write_sweep(_single_article_sweep())
    second = _writer(db).write_sweep(_single_article_sweep(_article(seen_at=_SEEN_T2)))
    print("first:", first.outcome, "second:", second.outcome)
    assert first.outcome == "committed"
    assert second.outcome == "committed"
    assert _one(db, "SELECT count(*) FROM raw_articles") == 1
    assert _one(db, "SELECT articles_replayed FROM ingest_runs ORDER BY id LIMIT 1") == 0


def test_lost_commit_and_lost_reconciliation_is_unknown(db):
    used = {"first": False}

    def factory():
        if used["first"]:
            raise psycopg.OperationalError("cannot reconnect to reconcile")
        used["first"] = True
        return _LostAckConnection(psycopg.connect(db, autocommit=True))

    with pytest.raises(SweepOutcomeUnknown, match="may or may not"):
        _writer(db, factory).write_sweep(_single_article_sweep())

    # The outcome really was unknown: the row committed and must not be lost.
    print("runs after unknown outcome:", _counts(db))
    assert _counts(db)["ingest_runs"] == 1
    assert _counts(db)["raw_articles"] == 1


class _DeferredCommitTransaction:
    """Hold the server transaction open, then commit after the client gave up."""

    def __init__(self, owner: "_DeferredCommitConnection", inner):
        self._owner = owner
        self._inner = inner

    def __enter__(self):
        return self._inner.__enter__()

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            return self._inner.__exit__(exc_type, exc, tb)
        owner = self._owner
        owner.pending = True

        def finish():
            try:
                owner.release.wait(timeout=10)
                self._inner.__exit__(None, None, None)
            finally:
                owner.pending = False
                try:
                    owner.conn.close()
                finally:
                    owner.committed.set()

        threading.Thread(target=finish, daemon=True).start()
        raise psycopg.OperationalError(
            "connection lost before commit confirmation was received"
        )


class _DeferredCommitConnection:
    """COMMIT is acknowledged only after the test releases it."""

    def __init__(self, conn, release: threading.Event, committed: threading.Event):
        self.conn = conn
        self.release = release
        self.committed = committed
        self.pending = False

    def transaction(self):
        return _DeferredCommitTransaction(self, self.conn.transaction())

    def execute(self, *args, **kwargs):
        return self.conn.execute(*args, **kwargs)

    def close(self):
        if self.pending:
            return
        self.conn.close()


def test_deferred_commit_is_unknown_then_lands_and_replays_dedup(db):
    release = threading.Event()
    committed = threading.Event()
    used = {"first": False}

    def factory():
        if not used["first"]:
            used["first"] = True
            return _DeferredCommitConnection(
                psycopg.connect(db, autocommit=True), release, committed
            )
        return psycopg.connect(db, autocommit=True)

    try:
        with pytest.raises(SweepOutcomeUnknown, match="may or may not") as info:
            _writer(db, factory).write_sweep(_single_article_sweep())
        print("message while unresolved:", info.value)
        assert "may or may not" in str(info.value)
        assert _counts(db)["ingest_runs"] == 0
        assert committed.is_set() is False
    finally:
        release.set()
        assert committed.wait(timeout=10), "deferred commit did not finish"

    print("counts after deferred commit landed:", _counts(db))
    assert _counts(db)["ingest_runs"] == 1
    assert _counts(db)["raw_articles"] == 1

    # A later identical sweep replays the same version row; nothing duplicates.
    replay = _writer(db).write_sweep(_single_article_sweep(_article(seen_at=_SEEN_T2)))
    assert replay.outcome == "committed"
    assert replay.articles_stored == 0 and replay.articles_replayed == 1
    assert _counts(db)["raw_articles"] == 1
