"""Real-Postgres integration tests for the I2/M2 pipeline analysis section.

AUTHORED ONLY in the M2 build gate: this file is collected by the offline suite
(and deselected by the ``integration`` marker) but never executed here. Its
execution needs a separate scoped database-authority approval. Importing this
module performs no settings loading, no DSN resolution and no database side
effects: the only import-time behaviour is ``pytest.importorskip``; every
project and configuration import happens inside a fixture or test body.

This file touches ONLY the Phase-1 pipeline store (no ingestion objects are
created, applied or probed).

Run with the established local pattern (when approved):

    .venv\\Scripts\\python.exe -m pytest -q -m integration
"""
import os
import re
from datetime import datetime, timezone

import pytest

psycopg = pytest.importorskip("psycopg")

pytestmark = [pytest.mark.integration, pytest.mark.enable_socket]

_GENERATED = datetime(2026, 11, 9, 13, 0, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------
# fixtures (established M1 pattern; no auto-creation)
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
    from market_intelligence.db import (
        apply_schema,
        verify_schema,
    )

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
    """Isolated pipeline tables, emptied before each test."""
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        conn.execute(
            "TRUNCATE run_topic_evidence, run_tones, run_topics, "
            "run_articles, competitor_profiles, runs "
            "RESTART IDENTITY CASCADE"
        )
    return pg_dsn


def _persist_two_versions(dsn, sample_state):
    from market_intelligence.persistence_postgres import (
        PostgresProfileWriter,
    )

    writer = PostgresProfileWriter(dsn)
    writer.persist("run-m2", sample_state())
    writer.persist("run-m2", sample_state(summary="s2"))
    return writer


def _load(dsn):
    from market_intelligence.analysis.queries import load_pipeline
    from market_intelligence.db.analytics import open_read_connection

    with open_read_connection(dsn, connect_timeout=3) as conn:
        return load_pipeline(conn, recent_limit=5)


# --------------------------------------------------------------------------
# readout over real persisted versions
# --------------------------------------------------------------------------
def test_latest_vs_all_version_counts(db, sample_state):
    _persist_two_versions(db, sample_state)
    rows = _load(db)

    assert rows.run_totals.run_versions == 2
    assert rows.run_totals.run_ids == 1
    assert rows.run_ids_with_multiple_versions == 1
    assert rows.article_counts.article_rows_all_versions == 2
    assert rows.article_counts.article_rows_latest_versions == 1
    assert rows.topic_totals.topic_rows_all_versions == 2
    assert rows.topic_totals.topic_rows_latest_versions == 1
    assert rows.topic_totals.distinct_topics_latest_versions == 1
    assert rows.evidence_rows_all_versions == 4
    assert rows.evidence_totals.evidence_rows_latest_versions == 2
    assert rows.tone_totals.tone_rows_all_versions == 2
    assert rows.tone_totals.tone_rows_latest_versions == 1
    assert len(rows.recent_runs) == 1
    assert rows.recent_runs[0].run_id == "run-m2"


def test_topics_keywords_and_confidence(db, sample_state):
    _persist_two_versions(db, sample_state)
    rows = _load(db)

    assert [topic.topic_name for topic in rows.topics] == ["AI"]
    topic = rows.topics[0]
    assert topic.topic_occurrences_latest_versions == 1
    assert topic.distinct_runs_latest_versions == 1
    assert topic.avg_model_confidence_latest_versions == pytest.approx(0.8)
    assert topic.avg_source_count_latest_versions == pytest.approx(1.0)
    keywords = {item.keyword for item in rows.keywords}
    assert keywords == {"ai", "widgets"}
    assert rows.keywords_nonarray_rows_latest_versions == 0


def test_evidence_verification_and_unknown_source(db, sample_state):
    _persist_two_versions(db, sample_state)
    rows = _load(db)

    latest = rows.evidence_totals
    assert latest.evidence_rows_latest_versions == 2
    assert latest.verified_latest_versions == 1
    assert latest.unverified_latest_versions == 1
    assert latest.unknown_source_citations_latest_versions == 1
    assert rows.aspects_nonarray_rows_latest_versions == 0


def test_tone_distribution_score_range_and_aspects(db, sample_state):
    _persist_two_versions(db, sample_state)
    rows = _load(db)

    assert [(item.overall, item.tone_occurrences_latest_versions)
            for item in rows.tone_distribution] == [("positive", 1)]
    assert rows.tone_distribution[0].avg_score_latest_versions == pytest.approx(
        0.5
    )
    assert rows.tone_score_range.min_score_latest_versions == pytest.approx(0.5)
    assert rows.tone_score_range.max_score_latest_versions == pytest.approx(0.5)
    assert [(item.aspect, item.aspect_occurrences_latest_versions)
            for item in rows.aspects] == [("growth", 1)]


def test_initialized_but_empty_store_reads_zero_rows(db):
    from market_intelligence.analysis.queries import load_pipeline
    from market_intelligence.analysis.report import build_analysis_report
    from market_intelligence.db.analytics import open_read_connection

    with open_read_connection(db, connect_timeout=3) as conn:
        rows = load_pipeline(conn, recent_limit=5)
    report = build_analysis_report(
        None,
        rows,
        include_ingestion=False,
        include_pipeline=True,
        generated_at=_GENERATED,
    )
    pipeline = report.pipeline
    assert pipeline is not None and pipeline.available is True
    assert pipeline.run_versions == 0
    assert pipeline.verification_rate_latest_versions is None
    assert pipeline.min_score_latest_versions is None


def test_read_only_connection_rejects_pipeline_writes(db):
    from market_intelligence.db.analytics import open_read_connection

    conn = open_read_connection(db, connect_timeout=3)
    try:
        with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
            conn.execute(
                "INSERT INTO runs (run_id, content_hash, rss_url, "
                "competitor_url, num_articles, payload) "
                "VALUES ('refused', %s, 'rss', 'competitor', 1, '{}'::jsonb)",
                ("a" * 64,),
            )
    finally:
        conn.close()


def test_cli_pipeline_dataset_end_to_end(db, capsys, monkeypatch, sample_state):
    import json

    from market_intelligence.analysis.__main__ import main
    from market_intelligence.persistence_postgres import (
        PostgresProfileWriter,
    )

    PostgresProfileWriter(db).persist("run-cli", sample_state())
    # The connect timeout comes from the test, not from saved configuration.
    monkeypatch.setattr(
        "market_intelligence.analysis.__main__._resolve_connect_timeout",
        lambda: 3.0,
    )
    assert (
        main(
            [
                "--dsn",
                db,
                "--dataset",
                "pipeline",
                "--format",
                "json",
                "--recent-runs",
                "5",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert list(payload) == ["report_version", "generated_at", "pipeline"]
    assert payload["pipeline"]["available"] is True
    assert payload["pipeline"]["run_versions"] == 1
