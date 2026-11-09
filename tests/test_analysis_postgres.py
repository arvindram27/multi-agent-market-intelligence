"""Real-Postgres integration tests for the I2/M1 ingestion analysis section.

AUTHORED ONLY in the M1 gate: this file is collected (and deselected by the
``integration`` marker in the offline suite) but never executed until the
separate database-authority gate. Importing this module performs no settings
loading, no DSN resolution and no database side effects: the only import-time
behaviour is ``pytest.importorskip``; every project and configuration import
happens inside a fixture or test body.

Run with the established local pattern:

    .venv\\Scripts\\python.exe -m pytest -q -m integration
"""
import os
import re
from datetime import datetime, timezone

import pytest

psycopg = pytest.importorskip("psycopg")

pytestmark = [pytest.mark.integration, pytest.mark.enable_socket]

_T1 = "2026-11-09T10:00:00+00:00"
_T2 = "2026-11-09T11:00:00+00:00"
_T_OLD = "2026-11-08T10:00:00+00:00"
_T_BOUNDARY = "2026-11-08T13:00:00+00:00"
_T_FRESH = "2026-11-09T12:00:00+00:00"
_GENERATED = datetime(2026, 11, 9, 13, 0, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------
# fixtures (established pattern from tests/test_ingestion_postgres.py)
# --------------------------------------------------------------------------
def _test_dsn() -> str:
    from market_intelligence.config import settings

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
            pytest.fail(
                f"test database {dbname!r} does not exist; "
                "the execution gate never creates databases"
            )
    return dsn


@pytest.fixture(scope="session")
def pg_dsn():
    from market_intelligence.db.ingestion import (
        apply_ingestion_schema,
        verify_ingestion_schema,
    )

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


# --------------------------------------------------------------------------
# collectors fixtures, imported lazily so collection stays side-effect free
# --------------------------------------------------------------------------
def _article(
    *,
    entry_key: str = "link:https://example.com/first",
    version_hash: str = "c" * 64,
    article_id: str = "0123456789abcdef",
    seen_at: str = _T1,
    source_key: str = "one",
    **overrides,
):
    from market_intelligence.ingestion.collector import CollectedArticle

    fields = {
        "source_key": source_key,
        "title": "First story",
        "excerpt": "A sufficiently long RSS excerpt about the first story.",
        "feed_url": "https://f.example/feed.xml",
        "source_url": "https://example.com/first",
        "content_kind": "rss_excerpt",
        "published_at_raw": None,
        "retrieved_at_raw": seen_at,
        "seen_at": seen_at,
        "content_hash": "a" * 64,
        "article_id": article_id,
        "entry_key": entry_key,
        "entry_key_hash": "b" * 64,
        "version_hash": version_hash,
    }
    fields.update(overrides)
    return CollectedArticle(**fields)


def _source_result(
    key: str,
    *,
    status: str = "ok",
    articles=(),
    enabled: bool = True,
    error_class: str | None = None,
    articles_seen: int | None = None,
):
    from market_intelligence.ingestion.collector import SourceResult

    return SourceResult(
        source_key=key,
        kind="rss",
        url=f"https://f.example/{key}.xml",
        label=key.upper(),
        enabled=enabled,
        status=status,
        articles=tuple(articles),
        warnings=(),
        error_class=error_class,
        articles_seen=len(articles) if articles_seen is None else articles_seen,
    )


def _sweep(source_results, status: str = "complete"):
    from market_intelligence.ingestion.collector import SweepResult

    return SweepResult(source_results=tuple(source_results), status=status)


def _single_article_sweep(article=None):
    article = article or _article()
    return _sweep(
        [_source_result("one", articles=(article,), articles_seen=1)],
        status="complete",
    )


def _rows(dsn, sql, params=()):
    with psycopg.connect(dsn, autocommit=True) as conn:
        return conn.execute(sql, params).fetchall()


# --------------------------------------------------------------------------
# readout over real seeded sweeps
# --------------------------------------------------------------------------
def test_volumes_latest_entries_and_multi_version(db):
    from market_intelligence.analysis.queries import load_ingestion
    from market_intelligence.analysis.report import build_report
    from market_intelligence.db.analytics import open_read_connection
    from market_intelligence.ingestion.store import IngestionSweepWriter

    writer = IngestionSweepWriter(db)
    writer.write_sweep(
        _sweep((_source_result("one", articles=(_article(seen_at=_T1),), articles_seen=1),))
    )
    writer.write_sweep(
        _sweep(
            (
                _source_result(
                    "one",
                    articles=(
                        _article(seen_at=_T2),
                        _article(
                            seen_at=_T2,
                            version_hash="d" * 64,
                            article_id="fedcba9876543210",
                        ),
                    ),
                    articles_seen=2,
                ),
            )
        )
    )

    with open_read_connection(db, connect_timeout=3) as conn:
        rows = load_ingestion(conn, recent_limit=10)
    report = build_report(
        rows, generated_at=_GENERATED, stale_after_hours=2, recent_limit=10
    )

    assert report.ingestion.available is True
    source = next(
        item for item in report.ingestion.sources if item.source_key == "one"
    )
    assert source.versions == 2
    assert source.entries == 1
    assert source.latest_entries == 1
    assert source.age_seconds == 2 * 3600
    assert source.stale is False  # exactly 2h at a 2h threshold is not stale

    assert report.ingestion.totals.sweeps == 2
    assert report.ingestion.totals.complete == 2
    assert report.ingestion.totals.articles_stored == 2
    assert report.ingestion.totals.articles_replayed == 1
    assert report.ingestion.totals.replay_rate == pytest.approx(1 / 3)
    assert [
        (item.source_key, item.entries) for item in report.ingestion.multi_version
    ] == [("one", 1)]


def test_freshness_contract_against_real_rows(db):
    from market_intelligence.analysis.queries import load_ingestion
    from market_intelligence.analysis.report import build_report
    from market_intelligence.db.analytics import open_read_connection
    from market_intelligence.ingestion.store import IngestionSweepWriter

    IngestionSweepWriter(db).write_sweep(
        _sweep(
            (
                _source_result(
                    "fresh",
                    articles=(_article(source_key="fresh", seen_at=_T_FRESH),),
                    articles_seen=1,
                ),
                _source_result(
                    "old",
                    articles=(_article(source_key="old", seen_at=_T_OLD),),
                    articles_seen=1,
                ),
                _source_result(
                    "boundary",
                    articles=(
                        _article(source_key="boundary", seen_at=_T_BOUNDARY),
                    ),
                    articles_seen=1,
                ),
                _source_result("off", enabled=False, status="disabled"),
            )
        )
    )

    with open_read_connection(db, connect_timeout=3) as conn:
        rows = load_ingestion(conn, recent_limit=5)
    report = build_report(
        rows, generated_at=_GENERATED, stale_after_hours=24, recent_limit=5
    )
    sources = {item.source_key: item for item in report.ingestion.sources}

    assert sources["fresh"].age_seconds == 3600
    assert sources["fresh"].stale is False
    assert sources["old"].age_seconds == 27 * 3600
    assert sources["old"].stale is True
    assert sources["boundary"].age_seconds == 24 * 3600
    assert sources["boundary"].stale is False
    # A disabled source with no observations keeps a null stale field.
    assert sources["off"].last_seen_at is None
    assert sources["off"].stale is None


def test_empty_database_is_available_with_zero_totals(db):
    from market_intelligence.analysis.queries import load_ingestion
    from market_intelligence.analysis.report import build_report
    from market_intelligence.db.analytics import open_read_connection

    with open_read_connection(db, connect_timeout=3) as conn:
        rows = load_ingestion(conn, recent_limit=5)
    report = build_report(rows, generated_at=_GENERATED, recent_limit=5)

    assert report.ingestion.available is True
    assert report.ingestion.sources == ()
    assert report.ingestion.totals.sweeps == 0
    assert report.ingestion.totals.replay_rate is None


def test_read_only_connection_rejects_writes(db):
    from market_intelligence.db.analytics import open_read_connection

    conn = open_read_connection(db, connect_timeout=3)
    try:
        with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
            conn.execute(
                "INSERT INTO ingest_sources "
                "(source_key, kind, url, label, enabled) "
                "VALUES ('refused', 'rss', 'https://x.example/feed.xml', 'X', true)"
            )
    finally:
        conn.close()


def test_malformed_source_results_are_counted_and_never_crash(db):
    from psycopg.types.json import Jsonb

    from market_intelligence.analysis.queries import load_ingestion
    from market_intelligence.analysis.report import build_report
    from market_intelligence.db.analytics import open_read_connection

    insert = (
        "INSERT INTO ingest_runs (sweep_id, started_at, finished_at, status, "
        "sources_attempted, sources_succeeded, sources_failed, articles_seen, "
        "articles_stored, articles_replayed, source_results) "
        "VALUES (%s, %s, %s, 'complete', 0, 0, 0, 0, 0, 0, %s)"
    )
    with psycopg.connect(db, autocommit=True) as conn:
        conn.execute(
            insert,
            (
                "malformed-object",
                _GENERATED,
                _GENERATED,
                Jsonb({"status": "ok"}),
            ),
        )
        conn.execute(
            insert,
            (
                "malformed-elements",
                _GENERATED,
                _GENERATED,
                Jsonb(["scalar", {"status": "ok"}, {"note": "no status"}]),
            ),
        )

    with open_read_connection(db, connect_timeout=3) as conn:
        rows = load_ingestion(conn, recent_limit=10)
    report = build_report(
        rows, generated_at=_GENERATED, stale_after_hours=24, recent_limit=10
    )
    outcomes = report.ingestion.outcomes
    assert outcomes.malformed_snapshots == 1
    statuses = dict(outcomes.status_counts)
    assert statuses["ok"] == 1
    assert statuses["<missing>"] == 2
    assert outcomes.elements_with_no_error_class == 3
    assert outcomes.error_classes == ()


def test_cli_reads_the_test_database(db, capsys, monkeypatch):
    import json

    from market_intelligence.analysis.__main__ import main
    from market_intelligence.ingestion.store import IngestionSweepWriter

    IngestionSweepWriter(db).write_sweep(_single_article_sweep())
    # Keep the CLI test free of saved configuration: the timeout comes from the
    # test, not from settings.
    monkeypatch.setattr(
        "market_intelligence.analysis.__main__._resolve_connect_timeout",
        lambda: 3.0,
    )

    assert main(["--dsn", db, "--format", "json", "--recent-sweeps", "5"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ingestion"]["available"] is True
    assert payload["ingestion"]["totals"]["sweeps"] == 1
    assert payload["ingestion"]["sources"][0]["versions"] == 1
