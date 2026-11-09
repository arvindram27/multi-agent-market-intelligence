"""Offline (DB-less) checks for the I1b ingestion store, facade and CLI.

Boundary AST checks for the new ingestion files, static checks on
``ingestion_schema.sql``, init-script ``--check`` behaviour against a stubbed
connection, and mocked writer/CLI tests. No socket and no database are used:
every connection object here is a fake. The real-Postgres coverage is
``tests/test_ingestion_postgres.py`` and needs a separate local approval.
"""
import ast
import argparse
import importlib
import re
from pathlib import Path
from types import SimpleNamespace

import psycopg
import pytest
from psycopg.pq import TransactionStatus

from market_intelligence.config import SecretStr, settings
from market_intelligence.db.ingestion import INGESTION_SCHEMA_PATH
from market_intelligence.ingestion.collector import (
    CollectedArticle,
    SourceResult,
    SweepResult,
)
from market_intelligence.ingestion.store import (
    IngestionSweepWriter,
    SweepLockedError,
    SweepOutcomeUnknown,
    SweepStoreError,
    SweepWriteResult,
)

DSN = "postgresql://writer:S3cretPassword@localhost:5432/mi_test"  # pragma: allowlist secret
SEEN = "2026-11-09T12:00:00+00:00"


# --------------------------------------------------------------------------
# Fixtures for the collector dataclasses (built directly, no network).
# --------------------------------------------------------------------------
def _article(**overrides) -> CollectedArticle:
    fields: dict = {
        "source_key": "one",
        "title": "First story",
        "excerpt": "A sufficiently long RSS excerpt about the first story.",
        "feed_url": "https://f.example/feed.xml",
        "source_url": "https://example.com/first",
        "content_kind": "rss_excerpt",
        "published_at_raw": None,
        "retrieved_at_raw": SEEN,
        "seen_at": SEEN,
        "content_hash": "a" * 64,
        "article_id": "0123456789abcdef",
        "entry_key": "link:https://example.com/first",
        "entry_key_hash": "b" * 64,
        "version_hash": "c" * 64,
    }
    fields.update(overrides)
    return CollectedArticle(**fields)


def _source_result(
    key: str = "one",
    *,
    status: str = "ok",
    articles: tuple = (),
    url: str = "https://f.example/feed.xml",
    enabled: bool = True,
    error_class: str | None = None,
    articles_seen: int = 0,
) -> SourceResult:
    return SourceResult(
        source_key=key,
        kind="rss",
        url=url,
        label=key.upper(),
        enabled=enabled,
        status=status,
        articles=tuple(articles),
        warnings=(),
        error_class=error_class,
        articles_seen=articles_seen,
    )


def _sweep(source_results, status: str = "complete") -> SweepResult:
    return SweepResult(source_results=tuple(source_results), status=status)


def _single_article_sweep(article: CollectedArticle | None = None) -> SweepResult:
    article = article or _article()
    return _sweep(
        [_source_result(articles=(article,), articles_seen=1)],
        status="complete",
    )


# --------------------------------------------------------------------------
# Fake psycopg-facing connection (the store never imports psycopg itself).
# --------------------------------------------------------------------------
class _FakeResult:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class _FakeTransaction:
    """Mimics psycopg: body exceptions roll back; a clean exit may commit."""

    def __init__(self, conn: "_FakeConn"):
        self._conn = conn

    def __enter__(self) -> "_FakeTransaction":
        self._conn.info.transaction_status = TransactionStatus.INTRANS
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._conn.info.transaction_status = TransactionStatus.IDLE
        if exc_type is None and self._conn.commit_error is not None:
            raise self._conn.commit_error
        return False


class _FakeConn:
    """Records statements; answers by recognizable SQL markers."""

    def __init__(
        self,
        *,
        lock: bool = True,
        report_insert_row: bool = True,
        status_row=None,
    ):
        self.executed: list[tuple[str, tuple | None]] = []
        self.info = SimpleNamespace(transaction_status=TransactionStatus.IDLE)
        self.lock = lock
        self.report_insert_row = report_insert_row
        self.status_row = status_row
        self.commit_error: Exception | None = None
        self.fail_on: str | None = None
        self.closed = False
        self.broken = False
        self._ids = iter(range(1, 100000))

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        upper = sql.upper()
        if self.fail_on is not None and self.fail_on.upper() in upper:
            raise psycopg.OperationalError("boom")
        if "PG_TRY_ADVISORY_XACT_LOCK" in upper:
            return _FakeResult((self.lock,))
        if "SET_CONFIG" in upper:
            return _FakeResult((params[0] if params else None,))
        if "INSERT INTO INGEST_RUN_ARTICLES" in upper:
            return _FakeResult(None)
        if "INSERT INTO INGEST_RUNS" in upper:
            return _FakeResult((next(self._ids),))
        if "INSERT INTO INGEST_SOURCES" in upper:
            return _FakeResult((next(self._ids),))
        if "INSERT INTO RAW_ARTICLES" in upper:
            row = (next(self._ids),) if self.report_insert_row else None
            return _FakeResult(row)
        if "SELECT ID FROM RAW_ARTICLES" in upper:
            return _FakeResult((next(self._ids),))
        if "SELECT STATUS FROM INGEST_RUNS" in upper:
            return _FakeResult(self.status_row)
        return _FakeResult(None)

    def transaction(self) -> _FakeTransaction:
        return _FakeTransaction(self)

    def close(self) -> None:
        self.closed = True

    def inserts(self) -> list[tuple[str, tuple | None]]:
        return [entry for entry in self.executed if "INSERT" in entry[0].upper()]


class _ConnFactory:
    """Hands out prepared connections and counts the calls, for reconcile."""

    def __init__(self, *conns: _FakeConn):
        self.conns = list(conns)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.conns.pop(0)


def _writer(*conns: _FakeConn, **writer_kwargs):
    factory = _ConnFactory(*conns)
    writer = IngestionSweepWriter(
        DSN, connection_factory=factory, **writer_kwargs
    )
    return writer, factory


def _cli():
    return importlib.import_module("market_intelligence.ingestion.__main__")


# --------------------------------------------------------------------------
# Boundary: the new ingestion files lock to the reviewed import set.
# --------------------------------------------------------------------------
_ALLOWED_FIRST_CHILDREN = {"ingestion", "feed_parsing", "db", "config", "net"}


def _imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level != 0:
                modules.append("." * node.level + (node.module or ""))
            else:
                modules.append(node.module or "")
    return modules


def _allowed_project_module(name: str) -> bool:
    parts = name.split(".")
    return (
        len(parts) >= 2
        and parts[0] == "market_intelligence"
        and parts[1] in _ALLOWED_FIRST_CHILDREN
    )


def _ingestion_dir() -> Path:
    import market_intelligence.ingestion as ingestion

    return Path(ingestion.__file__).resolve().parent


def test_boundary_of_new_ingestion_files():
    # Project-module restriction: only the reviewed allowlist. Stdlib and
    # third-party imports (including the declared psycopg) are permitted by
    # rule, but the locked design keeps psycopg out of the ingestion package
    # entirely: it lives behind the db facade.
    package_dir = _ingestion_dir()
    for name in ("store.py", "__main__.py"):
        modules = _imported_modules(package_dir / name)
        project = [m for m in modules if m.startswith("market_intelligence")]
        assert project, f"{name} must not be decoupled from the project"
        for module_name in project:
            assert _allowed_project_module(module_name), (name, module_name)
        for module_name in modules:
            assert not module_name.startswith(
                ("psycopg", "market_intelligence.agents",
                 "market_intelligence.graph", "market_intelligence.persistence")
            ), (name, module_name)
            assert "llm" not in module_name, (name, module_name)


# --------------------------------------------------------------------------
# Static checks on the generated ingestion_schema.sql.
# --------------------------------------------------------------------------
def _schema_text() -> str:
    return re.sub(r"--[^\n]*", "", INGESTION_SCHEMA_PATH.read_text(encoding="utf-8"))


def _schema_normalized() -> str:
    return " ".join(_schema_text().split())


def test_schema_tables_and_no_destructive_statements():
    stripped = _schema_text()
    tables = re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", stripped)
    assert set(tables) == {
        "ingest_schema_meta",
        "ingest_sources",
        "ingest_runs",
        "raw_articles",
        "ingest_run_articles",
    }
    assert len(tables) == 5
    assert not re.search(r"(?im)^\s*(DROP|ALTER)\b", stripped)
    assert not re.search(r"(?i)create\s+or\s+replace", stripped)
    assert "INSERT INTO ingest_schema_meta" in stripped
    assert "ON DELETE CASCADE" not in stripped


def test_schema_view_guard_and_definition():
    stripped = _stripped = _schema_text()
    normalized = _schema_normalized()
    assert re.search(r"to_regclass\('latest_articles'\) IS NULL", stripped)
    assert "CREATE VIEW latest_articles" in stripped
    assert "distinct on (source_id, entry_key)" in normalized.lower()
    assert (
        "order by source_id, entry_key, last_seen_at desc, id desc"
        in normalized.lower()
    )
    assert (
        "source_id, entry_key, entry_key_hash, version_hash, article_id, id, "
        "content_hash, first_seen_at, last_seen_at" in normalized
    )


def test_schema_entry_key_checks_and_reviewed_indexes():
    stripped = _schema_text()
    assert (
        "CHECK (entry_key ~ '^(guid(-sha256)?|link(-sha256)?|sha256):')"
        in stripped
    )
    assert "raw_articles_entry_key_length" in stripped
    assert "CHECK (length(entry_key) <= 2048)" in stripped
    assert "CONSTRAINT raw_articles_identity" in stripped
    assert "UNIQUE (source_id, entry_key_hash, version_hash)" in stripped
    for name, columns in (
        ("raw_articles_by_entry", "(source_id, entry_key_hash)"),
        ("raw_articles_by_article_id", "(source_id, article_id)"),
        ("raw_articles_by_published", "(published_at DESC)"),
        ("ingest_run_articles_by_article", "(raw_article_id)"),
    ):
        assert name in stripped and columns in stripped, (name, columns)
    assert "raw_articles_by_source_seen" not in stripped
    assert stripped.count("ON DELETE RESTRICT") == 3


def test_schema_no_clock_defaults_on_observation_times():
    stripped = _schema_text()
    for column in ("retrieved_at", "first_seen_at", "last_seen_at", "seen_at"):
        match = re.search(
            rf"(?i)(?m)^\s*{column}\s+timestamptz NOT NULL,\s*$", stripped
        )
        assert match is not None, column
        assert "DEFAULT" not in match.group(0), column


# --------------------------------------------------------------------------
# Init script: --check never applies; apply path; exit codes; redaction.
# --------------------------------------------------------------------------
class _InitConn:
    def __init__(self):
        self.executed: list[str] = []

    def execute(self, sql, params=None):
        self.executed.append(sql)
        return _FakeResult(None)

    def __enter__(self):
        return self

    def __exit__(self, *args) -> bool:
        return False


def _patch_init(monkeypatch, *, connect_raises=None):
    import scripts.init_ingestion_db as init_mod

    conn = _InitConn()
    calls: list[str] = []

    def connect(*args, **kwargs):
        calls.append("connect")
        if connect_raises is not None:
            raise connect_raises
        return conn

    monkeypatch.setattr(
        init_mod, "psycopg", SimpleNamespace(connect=connect, Error=psycopg.Error)
    )
    monkeypatch.setattr(
        init_mod,
        "apply_ingestion_schema",
        lambda c: calls.append("apply"),
    )
    return init_mod, conn, calls


def test_init_check_paths_verify_only(monkeypatch, capsys):
    init_mod, _conn, calls = _patch_init(monkeypatch)
    monkeypatch.setattr(init_mod.settings, "database_url", SecretStr(DSN))
    results = {"problems": []}

    def fake_verify(_conn):
        calls.append("verify")
        return results["problems"]

    monkeypatch.setattr(init_mod, "verify_ingestion_schema", fake_verify)
    monkeypatch.setattr(
        init_mod,
        "require_ingestion_schema",
        lambda _conn: (_ for _ in ()).throw(
            AssertionError("must not apply under --check")
        ),
    )

    assert init_mod.main(["--check"]) == 0
    results["problems"] = ["bad view"]
    assert init_mod.main(["--check"]) == 1
    assert "bad view" in capsys.readouterr().err
    assert calls == ["connect", "verify", "connect", "verify"]


def test_init_apply_runs_require(monkeypatch, capsys):
    init_mod, _conn, calls = _patch_init(monkeypatch)
    monkeypatch.setattr(init_mod.settings, "database_url", SecretStr(DSN))
    monkeypatch.setattr(
        init_mod,
        "require_ingestion_schema",
        lambda _conn: (calls.append("require"), [])[1],
    )
    assert init_mod.main([]) == 0
    assert calls == ["connect", "require"]
    assert "applied and verified" in capsys.readouterr().out


def test_init_refuses_without_a_dsn_and_never_connects(monkeypatch, capsys):
    init_mod, _conn, calls = _patch_init(
        monkeypatch, connect_raises=AssertionError("must not connect")
    )
    monkeypatch.setattr(init_mod.settings, "database_url", SecretStr(""))
    assert init_mod.main([]) == 2
    assert calls == []
    assert "No database configured" in capsys.readouterr().err


def test_init_database_errors_are_reported_and_redacted(monkeypatch, capsys):
    init_mod, _conn, calls = _patch_init(
        monkeypatch,
        connect_raises=psycopg.OperationalError(f"refused to reach {DSN}"),
    )
    assert init_mod.main(["--dsn", DSN, "--check"]) == 1
    err = capsys.readouterr().err
    assert calls == ["connect"]
    assert "database error" in err
    assert DSN not in err
    assert "***" in err


def test_init_schema_mismatch_returns_one(monkeypatch, capsys):
    import scripts.init_ingestion_db as init_mod

    from market_intelligence.db import SchemaMismatchError

    def connect(*args, **kwargs):
        raise SchemaMismatchError("ingestion schema is incompatible:\n  broken")

    monkeypatch.setattr(
        init_mod, "psycopg", SimpleNamespace(connect=connect, Error=psycopg.Error)
    )
    assert init_mod.main(["--dsn", DSN, "--check"]) == 1
    assert "broken" in capsys.readouterr().err


# --------------------------------------------------------------------------
# Mocked writer behaviour (the locked classification and write order).
# --------------------------------------------------------------------------
def test_lock_contention_fails_fast_before_writes():
    conn = _FakeConn(lock=False)
    writer, factory = _writer(conn)
    with pytest.raises(SweepLockedError, match="advisory lock"):
        writer.write_sweep(_single_article_sweep())
    assert factory.calls == 1
    assert len(conn.executed) == 1
    assert "pg_try_advisory_xact_lock" in conn.executed[0][0].lower()
    assert conn.inserts() == []
    assert conn.info.transaction_status == TransactionStatus.IDLE


def test_required_timestamps_validate_before_any_insert():
    bad = _article(retrieved_at_raw="not-a-timestamp")
    conn = _FakeConn()
    writer, _factory = _writer(conn)
    with pytest.raises(
        SweepStoreError, match="retrieved_at is required and could not be parsed"
    ) as excinfo:
        writer.write_sweep(_single_article_sweep(bad))
    # The raw timestamp never appears in the error text.
    assert "not-a-timestamp" not in str(excinfo.value)
    statements = [sql for sql, _params in conn.executed]
    assert [s.lower().split()[1] for s in statements] == [
        "pg_try_advisory_xact_lock(%s)",
        "set_config('statement_timeout',",
    ]
    assert conn.inserts() == []
    # Verified rollback: the confirmed IDLE connection proves no COMMIT ran.
    assert conn.info.transaction_status == TransactionStatus.IDLE


def test_connect_failure_is_a_known_failure():
    calls: list[int] = []

    def failing_factory():
        calls.append(1)
        raise psycopg.OperationalError("no route to host")

    writer = IngestionSweepWriter(DSN, connection_factory=failing_factory)
    with pytest.raises(SweepStoreError, match="no rows were stored") as excinfo:
        writer.write_sweep(_single_article_sweep())
    # Connect failures are definitive; no reconcile connection is opened.
    assert calls == [1]
    assert DSN not in str(excinfo.value)


def test_statement_order_lock_then_set_config_then_inserts():
    conn = _FakeConn()
    writer, _factory = _writer(conn)
    result = writer.write_sweep(_single_article_sweep())
    lowered = [sql.strip().lower() for sql, _params in conn.executed]
    assert "pg_try_advisory_xact_lock(%s)" in lowered[0]
    assert lowered[1] == "select set_config('statement_timeout', %s, true)"
    first_insert = next(
        index for index, sql in enumerate(lowered) if sql.startswith("insert")
    )
    assert first_insert > 1
    assert conn.executed[1][1] == ("5000",)
    assert result.outcome == "committed"
    assert result.status == "complete"
    assert result.sources_synced == 1
    assert result.articles_seen == 1
    assert result.articles_stored == 1
    assert result.articles_replayed == 0
    assert result.sweep_id


def test_full_sweep_counts_sources_run_articles_and_memberships():
    first = _article()
    second = _article(
        source_key="two",
        title="Second story",
        entry_key="link:https://example.com/second",
        entry_key_hash="d" * 64,
        version_hash="e" * 64,
        article_id="fedcba9876543210",
    )
    failed = _source_result(
        "bad", status="failed", error_class="TransientFetchError", articles_seen=0
    )
    sweep = _sweep(
        [
            _source_result("one", articles=(first,), articles_seen=1),
            _source_result("two", articles=(second,), articles_seen=1),
            failed,
        ],
        status="partial",
    )
    conn = _FakeConn()
    writer, _factory = _writer(conn)
    result = writer.write_sweep(sweep)
    assert result.outcome == "committed"
    assert result.sources_synced == 3
    assert result.articles_seen == 2
    assert result.articles_stored == 2
    assert result.articles_replayed == 0
    insert_sqls = [sql for sql, _params in conn.inserts()]
    assert (
        sum("insert into ingest_run_articles" in sql.lower() for sql in insert_sqls)
        == 2
    )
    run_params = next(
        params
        for sql, params in conn.inserts()
        if "insert into ingest_runs" in sql.lower()
    )
    snapshot = getattr(run_params[10], "obj", run_params[10])
    assert [row["source_key"] for row in snapshot] == ["one", "two", "bad"]


def test_body_failure_with_confirmed_rollback_is_known():
    conn = _FakeConn()
    conn.fail_on = "INSERT INTO RAW_ARTICLES"
    writer, _factory = _writer(conn)
    with pytest.raises(
        SweepStoreError, match="rolled back; no rows were stored"
    ) as excinfo:
        writer.write_sweep(_single_article_sweep())
    assert "OperationalError: boom" in str(excinfo.value)
    assert conn.info.transaction_status == TransactionStatus.IDLE
    assert conn.inserts()  # the failing INSERT was attempted, then rolled back


def test_commit_failure_reconciles_present_terminal_row():
    first = _FakeConn()
    first.commit_error = psycopg.OperationalError("connection lost at commit")
    second = _FakeConn(status_row=("complete",))
    writer, factory = _writer(first, second)
    result = writer.write_sweep(_single_article_sweep())
    assert factory.calls == 2
    assert result.outcome == "reconciled"
    assert result.sweep_id
    assert result.articles_stored is None
    reconcile_sqls = [sql for sql, _params in second.executed]
    assert any("select status from ingest_runs" in s.lower() for s in reconcile_sqls)


def test_commit_failure_without_terminal_row_stays_unknown():
    first = _FakeConn()
    first.commit_error = psycopg.OperationalError("connection lost at commit")
    second = _FakeConn(status_row=None)
    writer, _factory = _writer(first, second)
    with pytest.raises(
        SweepOutcomeUnknown, match="may or may not have been stored"
    ):
        writer.write_sweep(_single_article_sweep())


# --------------------------------------------------------------------------
# CLI surface.
# --------------------------------------------------------------------------
def test_cli_surface_and_help_exit_zero(capsys):
    cli = _cli()
    parser = cli.build_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    args = parser.parse_args(
        ["--sources", "custom.toml", "--dsn", "postgresql://x/db"]
    )
    assert args.sources == "custom.toml"
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--help"])
    assert exit_info.value.code == 0
    assert "sweep" in capsys.readouterr().out


def test_cli_missing_dsn_returns_two(monkeypatch, capsys):
    cli = _cli()
    monkeypatch.setattr(settings, "database_url", SecretStr(""))

    def bomb(*args, **kwargs):
        raise AssertionError("collect must not run before the DSN is checked")

    monkeypatch.setattr(cli, "collect", bomb)
    assert cli.main([]) == 2
    assert "No database configured" in capsys.readouterr().err


def test_cli_mocked_pipeline_stores_the_sweep(monkeypatch, capsys):
    cli = _cli()
    seen = _single_article_sweep()

    monkeypatch.setattr(settings, "database_url", SecretStr(""))
    monkeypatch.setattr(cli, "load_sources", lambda path: ["stub"])
    monkeypatch.setattr(cli, "collect", lambda sources: seen)

    written: dict = {}

    class _StubWriter:
        def __init__(self, dsn):
            self.dsn = dsn

        def write_sweep(self, sweep):
            written["sweep"] = sweep
            return SweepWriteResult(
                sweep_id="abc",
                status=sweep.status,
                outcome="committed",
                sources_synced=1,
                articles_seen=1,
                articles_stored=1,
                articles_replayed=0,
            )

    monkeypatch.setattr(cli, "IngestionSweepWriter", _StubWriter)
    assert (
        cli.main(["--sources", "x.toml", "--dsn", "postgresql://user:pw@h/db"])  # pragma: allowlist secret
        == 0
    )
    assert written["sweep"].status == "complete"
    assert "outcome=committed" in capsys.readouterr().out
