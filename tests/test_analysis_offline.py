"""Offline (DB-less, socket-less) checks for the I2/M1 analysis layer.

Everything here runs with ``--disable-socket`` and no database: fake rows feed
the pure shaping code, and the CLI is exercised against a monkeypatched
connection/loader. The only DSN-shaped value is a conspicuously synthetic
fixture; the real-Postgres coverage lives in ``tests/test_analysis_postgres.py``
and is never executed by this file.
"""
import ast
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from market_intelligence.analysis import REPORT_VERSION, build_report
from market_intelligence.analysis import __main__ as cli
from market_intelligence.analysis import queries as queries_module
from market_intelligence.analysis import report as report_module
from market_intelligence.analysis.queries import (
    IngestionRows,
    LatestEntryRow,
    MultiVersionRow,
    RecentSweepRow,
    SourceOutcomeRow,
    SourceVolumeRow,
    SweepTotalsRow,
)
import psycopg

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = REPO_ROOT / "src" / "market_intelligence" / "analysis"
ANALYTICS_PATH = REPO_ROOT / "src" / "market_intelligence" / "db" / "analytics.py"

FIXED = datetime(2026, 11, 9, 12, 0, 0, tzinfo=timezone.utc)
SYNTHETIC_DSN = (
    "postgresql://synthetic-user:synthetic-fake-password@invalid:1/synthetic_db"  # pragma: allowlist secret
)
FORBIDDEN_SQL = (
    "INSERT",
    "UPDATE",
    "DELETE",
    "MERGE",
    "TRUNCATE",
    "DROP",
    "ALTER",
    "CREATE",
    "GRANT",
    "REVOKE",
    "COPY",
)


# --------------------------------------------------------------------------
# static guarantees: psycopg boundary and SELECT-only SQL
# --------------------------------------------------------------------------
def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_analysis_package_does_not_import_psycopg():
    files = sorted(ANALYSIS_DIR.glob("*.py"))
    assert files, "analysis package files are missing"
    for path in files:
        assert "psycopg" not in _imported_roots(path), path.name
    # The facade is the single direct psycopg importer for the analysis layer.
    assert "psycopg" in _imported_roots(ANALYTICS_PATH)


def test_sql_constants_are_select_only():
    constants = {
        name: value
        for name, value in vars(queries_module).items()
        if name.startswith("SQL_") and isinstance(value, str)
    }
    assert len(constants) >= 7
    for name, sql in constants.items():
        text = " ".join(sql.split()).upper()
        assert text.startswith("SELECT "), name
        for word in FORBIDDEN_SQL:
            assert word not in text, (name, word)


# --------------------------------------------------------------------------
# pure shaping
# --------------------------------------------------------------------------
def _rows(
    *,
    sources=(),
    latest_entries=(),
    totals=None,
    recent_sweeps=(),
    outcomes=(),
    multi_version=(),
    malformed_snapshots=0,
) -> IngestionRows:
    return IngestionRows(
        sources=tuple(sources),
        latest_entries=tuple(latest_entries),
        totals=totals or SweepTotalsRow(0, 0, 0, 0, 0, 0, 0),
        recent_sweeps=tuple(recent_sweeps),
        outcomes=tuple(outcomes),
        multi_version=tuple(multi_version),
        malformed_snapshots=malformed_snapshots,
    )


def _standard_rows() -> IngestionRows:
    return _rows(
        sources=(
            SourceVolumeRow(
                1,
                "one",
                "One",
                True,
                2,
                1,
                FIXED - timedelta(days=3),
                FIXED - timedelta(hours=25),
            ),
            SourceVolumeRow(2, "empty", "Empty", True, 0, 0, None, None),
        ),
        latest_entries=(LatestEntryRow("one", 1),),
        totals=SweepTotalsRow(2, 1, 1, 0, 10, 6, 4),
        recent_sweeps=(
            RecentSweepRow(
                "sweep-2",
                "partial",
                FIXED - timedelta(hours=2),
                FIXED - timedelta(hours=1),
                2,
                1,
                1,
                6,
                4,
                2,
            ),
            RecentSweepRow(
                "sweep-1",
                "complete",
                FIXED - timedelta(hours=4),
                FIXED - timedelta(hours=3),
                2,
                2,
                0,
                4,
                2,
                2,
            ),
        ),
        outcomes=(
            SourceOutcomeRow("sweep-2", "ok", None),
            SourceOutcomeRow("sweep-2", "failed", "TransientFetchError"),
            SourceOutcomeRow("sweep-1", "ok", None),
        ),
        multi_version=(MultiVersionRow("one", 1),),
    )


def test_sources_merge_latest_entries_and_freshness_contract():
    rows = _rows(
        sources=(
            SourceVolumeRow(
                1,
                "one",
                "One",
                True,
                2,
                1,
                FIXED - timedelta(days=2),
                FIXED - timedelta(hours=25),
            ),
            SourceVolumeRow(
                2,
                "boundary",
                "Boundary",
                True,
                1,
                1,
                FIXED - timedelta(hours=24),
                FIXED - timedelta(hours=24),
            ),
            SourceVolumeRow(
                3,
                "future",
                "Future",
                True,
                1,
                1,
                FIXED,
                FIXED + timedelta(hours=5),
            ),
            SourceVolumeRow(4, "empty", "Empty", True, 0, 0, None, None),
            SourceVolumeRow(
                5, "off", "Off", False, 1, 1,
                FIXED - timedelta(days=4), FIXED - timedelta(hours=48),
            ),
        ),
        latest_entries=(LatestEntryRow("one", 7),),
    )
    report = build_report(
        rows, generated_at=FIXED, stale_after_hours=24, recent_limit=10
    )
    sources = {source.source_key: source for source in report.ingestion.sources}

    assert sources["one"].versions == 2
    assert sources["one"].entries == 1
    assert sources["one"].latest_entries == 7
    assert sources["one"].age_seconds == 25 * 3600
    assert sources["one"].stale is True

    # Exactly at the threshold is not stale: the comparison is strict.
    assert sources["boundary"].age_seconds == 24 * 3600
    assert sources["boundary"].stale is False

    # A future timestamp still reports the timestamp, clamps the age to 0 and
    # is not stale.
    assert sources["future"].last_seen_at == FIXED + timedelta(hours=5)
    assert sources["future"].age_seconds == 0
    assert sources["future"].stale is False

    # No observations: the stale field is present and null (never omitted).
    assert sources["empty"].last_seen_at is None
    assert sources["empty"].age_seconds is None
    assert sources["empty"].stale is None

    # A disabled source with observations is never stale, but its age renders.
    assert sources["off"].age_seconds == 48 * 3600
    assert sources["off"].stale is False


def test_replay_rate_math_and_zero_denominator():
    report = build_report(
        _rows(
            totals=SweepTotalsRow(3, 2, 1, 0, 10, 6, 4),
            recent_sweeps=(
                RecentSweepRow(
                    "zero",
                    "complete",
                    FIXED,
                    FIXED,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ),
            ),
        ),
        generated_at=FIXED,
    )
    assert report.ingestion.totals.replay_rate == pytest.approx(0.4)
    assert report.ingestion.recent_sweeps[0].replay_rate is None
    text = report.render_text()
    assert "replay_rate 40.0%" in text
    # A None rate renders as the n/a table cell, never 0%.
    assert (
        "zero | complete | 2026-11-09T12:00:00+00:00 | 0 | 0 | 0 | 0 | 0 | 0 | n/a"
        in text
    )
    assert "replay_rate 0.0%" not in text
    assert "| 0.0%" not in text

    empty = build_report(_rows(), generated_at=FIXED)
    assert empty.ingestion.totals.replay_rate is None


def test_outcome_counts_unknown_missing_and_malformed():
    rows = _rows(
        outcomes=(
            SourceOutcomeRow("s1", "ok", None),
            SourceOutcomeRow("s1", "ok", None),
            SourceOutcomeRow("s1", "ok", "E1"),
            SourceOutcomeRow("s1", "failed", "TransientFetchError"),
            SourceOutcomeRow("s1", "disabled", None),
            SourceOutcomeRow("s1", "some-new-status", None),
            SourceOutcomeRow("s1", "<missing>", None),
            SourceOutcomeRow("s1", "<missing>", None),
        ),
        malformed_snapshots=2,
    )
    report = build_report(rows, generated_at=FIXED)
    outcomes = report.ingestion.outcomes
    assert outcomes.status_counts == (
        ("ok", 3),
        ("<missing>", 2),
        ("disabled", 1),
        ("failed", 1),
        ("some-new-status", 1),
    )
    assert outcomes.error_classes == ("E1", "TransientFetchError")
    assert outcomes.elements_with_no_error_class == 6
    assert outcomes.malformed_snapshots == 2
    assert "malformed source_results snapshots: 2" in report.render_text()


def test_empty_store_report_is_available_with_zero_totals():
    report = build_report(_rows(), generated_at=FIXED, recent_limit=5)
    ingestion = report.ingestion
    assert ingestion.available is True
    assert ingestion.sources == ()
    assert ingestion.totals.sweeps == 0
    assert ingestion.totals.replay_rate is None
    text = report.render_text()
    assert "Ingestion store: available" in text
    assert "sources: 0 (enabled 0, disabled 0)" in text
    assert "sweeps: 0 (complete 0, partial 0, failed 0, other 0)" in text
    assert "multi-version entries: none" in text
    assert "recent sweeps: none" in text


def test_unavailable_store_reports_not_initialized():
    report = build_report(None, generated_at=FIXED)
    assert report.ingestion.available is False
    assert report.ingestion.totals is None
    assert "not initialized (available: false)" in report.render_text()
    payload = report.to_dict()
    assert payload["ingestion"]["available"] is False
    assert payload["ingestion"]["totals"] is None


def test_json_key_order_iso_timestamps_and_round_trip():
    payload = build_report(_standard_rows(), generated_at=FIXED).to_dict()
    assert list(payload) == ["report_version", "generated_at", "ingestion"]
    ingestion = payload["ingestion"]
    assert list(ingestion) == [
        "available",
        "recent_limit",
        "sources",
        "totals",
        "recent_sweeps",
        "outcomes",
        "multi_version",
    ]
    assert list(ingestion["sources"][0]) == [
        "source_key",
        "label",
        "enabled",
        "versions",
        "entries",
        "latest_entries",
        "first_seen_at",
        "last_seen_at",
        "age_seconds",
        "stale",
    ]
    assert payload["report_version"] == REPORT_VERSION
    assert payload["generated_at"] == "2026-11-09T12:00:00+00:00"
    assert ingestion["sources"][0]["stale"] is True
    # Nulls are preserved, not omitted.
    assert ingestion["sources"][1]["stale"] is None
    assert ingestion["sources"][1]["last_seen_at"] is None
    assert json.loads(json.dumps(payload)) == payload


def test_text_rendering_is_deterministic():
    report = build_report(_standard_rows(), generated_at=FIXED)
    assert report.render_text() == report.render_text()


# --------------------------------------------------------------------------
# CLI model (fake connection, no socket, no database)
# --------------------------------------------------------------------------
class _FakeConnection:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _patch_cli(monkeypatch, *, present, problems=(), rows=None):
    monkeypatch.setattr(
        cli,
        "open_read_connection",
        lambda dsn, *, connect_timeout: _FakeConnection(),
    )
    # The connect timeout is injected so the fake-CLI path never loads saved
    # configuration or resolves a real DSN.
    monkeypatch.setattr(cli, "_resolve_connect_timeout", lambda: 5.0)
    monkeypatch.setattr(cli, "table_present", lambda conn, name: present[name])
    monkeypatch.setattr(
        cli, "verify_ingestion_schema", lambda conn: list(problems)
    )
    if rows is not None:
        monkeypatch.setattr(
            cli, "load_ingestion", lambda conn, *, recent_limit: rows
        )


def test_cli_reports_unavailable_when_no_object_is_present(
    monkeypatch, capsys
):
    _patch_cli(
        monkeypatch,
        present={name: False for name in queries_module.INGEST_OBJECTS},
    )
    assert cli.main(["--dsn", SYNTHETIC_DSN, "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ingestion"]["available"] is False


def test_cli_fails_closed_on_partial_schema(monkeypatch, capsys):
    present = {
        name: name == "ingest_sources" for name in queries_module.INGEST_OBJECTS
    }
    _patch_cli(
        monkeypatch,
        present=present,
        problems=["ingest_runs is missing"],
    )
    assert cli.main(["--dsn", SYNTHETIC_DSN]) == 1
    captured = capsys.readouterr()
    assert "ingest_runs is missing" in captured.err
    assert captured.out == ""


def test_cli_full_report_in_text_and_json(monkeypatch, capsys):
    _patch_cli(
        monkeypatch,
        present={name: True for name in queries_module.INGEST_OBJECTS},
        rows=_standard_rows(),
    )
    monkeypatch.setattr(report_module, "utcnow", lambda: FIXED)

    assert cli.main(["--dsn", SYNTHETIC_DSN]) == 0
    text = capsys.readouterr().out
    assert "Ingestion store: available" in text
    assert "replay_rate 40.0%" in text
    assert "one | One | True | 2 | 1 | 1" in text

    assert cli.main(["--dsn", SYNTHETIC_DSN, "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["report_version"] == REPORT_VERSION
    assert payload["generated_at"] == "2026-11-09T12:00:00+00:00"
    assert payload["ingestion"]["totals"]["replay_rate"] == pytest.approx(0.4)


def test_cli_output_is_byte_identical_with_a_frozen_clock(monkeypatch, capsys):
    _patch_cli(
        monkeypatch,
        present={name: True for name in queries_module.INGEST_OBJECTS},
        rows=_standard_rows(),
    )
    monkeypatch.setattr(report_module, "utcnow", lambda: FIXED)

    assert cli.main(["--dsn", SYNTHETIC_DSN]) == 0
    first = capsys.readouterr().out
    assert cli.main(["--dsn", SYNTHETIC_DSN]) == 0
    second = capsys.readouterr().out
    assert first == second


def test_cli_exits_2_without_a_configured_dsn(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_resolve_dsn", lambda explicit: "")
    assert cli.main([]) == 2
    err = capsys.readouterr().err
    assert "DATABASE_URL" in err
    assert "--dsn" in err


def test_cli_redacts_connection_errors(monkeypatch, capsys):
    def boom(dsn, *, connect_timeout):
        raise psycopg.OperationalError(f"could not connect using {SYNTHETIC_DSN}")

    monkeypatch.setattr(cli, "open_read_connection", boom)
    monkeypatch.setattr(cli, "_resolve_connect_timeout", lambda: 5.0)

    assert cli.main(["--dsn", SYNTHETIC_DSN]) == 1
    captured = capsys.readouterr()
    assert "OperationalError" in captured.err
    assert "synthetic-fake-password" not in captured.out + captured.err
    assert SYNTHETIC_DSN not in captured.out + captured.err


def test_cli_redacts_a_bare_password_in_connection_errors(monkeypatch, capsys):
    def boom(dsn, *, connect_timeout):
        raise psycopg.OperationalError(
            "authentication rejected for password synthetic-fake-password"
        )

    monkeypatch.setattr(cli, "open_read_connection", boom)
    monkeypatch.setattr(cli, "_resolve_connect_timeout", lambda: 5.0)

    assert cli.main(["--dsn", SYNTHETIC_DSN]) == 1
    captured = capsys.readouterr()
    assert "synthetic-fake-password" not in captured.out + captured.err
    assert "OperationalError" in captured.err


def test_dsn_secrets_parses_forms_and_fails_closed():
    from market_intelligence.db.analytics import dsn_secrets

    # URI form: password extracted.
    assert dsn_secrets(SYNTHETIC_DSN) == [SYNTHETIC_DSN, "synthetic-fake-password"]
    # URI form, percent-encoded: the decoded value is the secret.
    uri = "postgresql://u:p%40ss%3Aword@invalid:1/d"  # pragma: allowlist secret
    assert dsn_secrets(uri) == [uri, "p@ss:word"]
    # Keyword form.
    kw = "host=invalid user=u password=kw-secret dbname=d"
    assert dsn_secrets(kw) == [kw, "kw-secret"]
    # Malformed input fails closed to the DSN alone, without raising.
    for bad in ("not a dsn at all", "postgresql://[::1", "password="):
        secrets = dsn_secrets(bad)
        assert secrets[0] == bad
        assert all(isinstance(value, str) for value in secrets)
    # No password anywhere: only the DSN itself.
    assert dsn_secrets("postgresql://u@invalid:1/d") == ["postgresql://u@invalid:1/d"]


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--stale-hours", "0"),
        ("--stale-hours", "-1"),
        ("--stale-hours", "nan"),
        ("--stale-hours", "inf"),
        ("--recent-sweeps", "0"),
        ("--recent-sweeps", "101"),
    ],
)
def test_cli_rejects_invalid_threshold_arguments(flag, value):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--dsn", SYNTHETIC_DSN, flag, value])
    assert excinfo.value.code == 2
