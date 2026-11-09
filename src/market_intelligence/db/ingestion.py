"""Ingestion schema application, verification and Postgres-facing helpers.

This is the only ingestion-feature module allowed to import psycopg: the I1a
boundary test bans it inside the ``ingestion`` package (which stays
untouched), so ``ingestion.store`` and ``ingestion.__main__`` reach Postgres
exclusively through this facade. ``market_intelligence.db`` itself never
imports psycopg.

Verification composes the generic ``db.verify_schema`` machinery
(parameterized, defaults byte-identical to the runs schema checks) with the
ingestion-specific rules: five primary keys, the unique-index set scoped to
``raw_articles``, the reviewed non-unique index set, the no-clock-defaults
rule for observation timestamps, and the ``entry_key`` CHECK constraints. A
mismatch is reported, never fixed in place: ``require_ingestion_schema``
raises instead of dropping or rewriting anything.
"""
from __future__ import annotations

from pathlib import Path

import psycopg
from psycopg.pq import TransactionStatus
from psycopg.types.json import Jsonb  # noqa: F401 (re-exported for the store)

from market_intelligence.db import (  # noqa: F401
    SchemaMismatchError,
    verify_schema,
)
from market_intelligence.db import Constraint as _Constraint

INGESTION_SCHEMA_VERSION = 1
INGESTION_SCHEMA_PATH = Path(__file__).with_name("ingestion_schema.sql")

# psycopg-facing surface the ingestion package imports instead of psycopg.
DatabaseError = psycopg.Error


def open_connection(dsn: str, *, connect_timeout: float = 5.0):
    """Open an autocommit connection; the writer opens transactions itself."""
    return psycopg.connect(dsn, autocommit=True, connect_timeout=connect_timeout)


def is_transaction_closed(conn) -> bool:
    """True when the connection is healthy and holds no open transaction.

    Only meaningful after a transaction block has unwound from a body
    failure: a healthy, idle connection means psycopg issued ROLLBACK and no
    COMMIT was attempted. A broken connection or a still-open transaction is
    ambiguous.
    """
    try:
        return (
            not conn.closed
            and not conn.broken
            and conn.info.transaction_status == TransactionStatus.IDLE
        )
    except Exception:  # pragma: no cover - defensive for exotic connections
        return False


# --------------------------------------------------------------------------
# Structural specification of the ingestion schema.
# --------------------------------------------------------------------------

INGEST_REQUIRED_COLUMNS: dict[str, dict[str, str]] = {
    "ingest_schema_meta": {
        "singleton": "boolean",
        "schema_version": "integer",
        "applied_at": "timestamp with time zone",
    },
    "ingest_sources": {
        "id": "bigint",
        "source_key": "text",
        "kind": "text",
        "url": "text",
        "label": "text",
        "enabled": "boolean",
        "created_at": "timestamp with time zone",
        "updated_at": "timestamp with time zone",
    },
    "ingest_runs": {
        "id": "bigint",
        "sweep_id": "text",
        "started_at": "timestamp with time zone",
        "finished_at": "timestamp with time zone",
        "status": "text",
        "sources_attempted": "integer",
        "sources_succeeded": "integer",
        "sources_failed": "integer",
        "articles_seen": "integer",
        "articles_stored": "integer",
        "articles_replayed": "integer",
        "source_results": "jsonb",
    },
    "raw_articles": {
        "id": "bigint",
        "source_id": "bigint",
        "entry_key": "text",
        "entry_key_hash": "text",
        "version_hash": "text",
        "article_id": "text",
        "title": "text",
        "excerpt": "text",
        "content_kind": "text",
        "content_hash": "text",
        "feed_url": "text",
        "source_url": "text",
        "published_at": "timestamp with time zone",
        "published_at_raw": "text",
        "retrieved_at": "timestamp with time zone",
        "retrieved_at_raw": "text",
        "first_seen_at": "timestamp with time zone",
        "last_seen_at": "timestamp with time zone",
    },
    "ingest_run_articles": {
        "run_id": "bigint",
        "raw_article_id": "bigint",
        "seen_at": "timestamp with time zone",
    },
}

_INGEST_REQUIRED_CONSTRAINTS: dict[str, tuple] = {
    "ingest_schema_meta": (_Constraint("p", ("singleton",)),),
    "ingest_sources": (
        _Constraint("p", ("id",)),
        _Constraint("u", ("source_key",), name="ingest_sources_key"),
    ),
    "ingest_runs": (
        _Constraint("p", ("id",)),
        _Constraint("u", ("sweep_id",), name="ingest_runs_sweep"),
    ),
    "raw_articles": (
        _Constraint("p", ("id",)),
        _Constraint(
            "u",
            ("source_id", "entry_key_hash", "version_hash"),
            name="raw_articles_identity",
        ),
        _Constraint("f", ("source_id",), "ingest_sources", ("id",), "r"),
    ),
    "ingest_run_articles": (
        _Constraint("p", ("run_id", "raw_article_id")),
        _Constraint("f", ("run_id",), "ingest_runs", ("id",), "r"),
        _Constraint("f", ("raw_article_id",), "raw_articles", ("id",), "r"),
    ),
}

INGEST_REQUIRED_VIEWS: dict[str, tuple[str, ...]] = {
    "latest_articles": (
        "source_id",
        "entry_key",
        "entry_key_hash",
        "version_hash",
        "article_id",
        "id",
        "content_hash",
        "first_seen_at",
        "last_seen_at",
    ),
}

INGEST_REQUIRED_VIEW_DEFINITIONS: dict[str, tuple[str, ...]] = {
    "latest_articles": (
        "distinct on (source_id, entry_key)",
        "order by source_id, entry_key, last_seen_at desc, id desc",
    ),
}

# Unique indexes raw_articles must have (scoped negative check: any other
# unique index there, or any missing one, is a problem).
_RAW_UNIQUE_EXPECTATIONS = (
    ("raw_articles_pkey", "(id)", True),
    ("raw_articles_identity", "(source_id, entry_key_hash, version_hash)", False),
)

# Reviewed non-unique index set (name, columns as pg_get_indexdef renders it).
_REVIEWED_INDEXES = (
    ("raw_articles_by_entry", "(source_id, entry_key_hash)", "raw_articles"),
    ("raw_articles_by_article_id", "(source_id, article_id)", "raw_articles"),
    ("raw_articles_by_published", "(published_at desc)", "raw_articles"),
    ("ingest_run_articles_by_article", "(raw_article_id)", "ingest_run_articles"),
)

_INDEX_SQL = """
SELECT t.relname AS table_name,
       i.indisunique,
       i.indisprimary,
       pg_get_indexdef(i.indexrelid) AS definition
FROM pg_index i
JOIN pg_class t ON t.oid = i.indrelid
JOIN pg_namespace n ON n.oid = t.relnamespace
WHERE n.nspname = %s
  AND t.relname IN ('raw_articles', 'ingest_run_articles')
"""

_TIMESTAMP_COLUMN_SQL = """
SELECT table_name, column_name, column_default, is_nullable
FROM information_schema.columns
WHERE table_schema = %s
  AND ((table_name = 'raw_articles'
        AND column_name IN ('retrieved_at', 'first_seen_at', 'last_seen_at'))
    OR (table_name = 'ingest_run_articles' AND column_name = 'seen_at'))
"""

_CHECK_CONSTRAINT_DEFS_SQL = """
SELECT pg_get_constraintdef(c.oid)
FROM pg_constraint c
JOIN pg_class t ON t.oid = c.conrelid
JOIN pg_namespace n ON n.oid = t.relnamespace
WHERE n.nspname = %s AND t.relname = 'raw_articles' AND c.contype = 'c'
"""


def apply_ingestion_schema(conn) -> None:
    """Create missing objects. Safe to re-run; never alters or drops."""
    conn.execute(INGESTION_SCHEMA_PATH.read_text(encoding="utf-8"))


def verify_ingestion_schema(conn, schema: str | None = None) -> list[str]:
    """Return human-readable incompatibilities, or an empty list when compatible."""
    schema = schema or conn.execute("SELECT current_schema()").fetchone()[0]
    problems = verify_schema(
        conn,
        schema,
        required_columns=INGEST_REQUIRED_COLUMNS,
        required_constraints=_INGEST_REQUIRED_CONSTRAINTS,
        required_views=INGEST_REQUIRED_VIEWS,
        required_view_definitions=INGEST_REQUIRED_VIEW_DEFINITIONS,
        version_table="ingest_schema_meta",
        expected_version=INGESTION_SCHEMA_VERSION,
    )
    _check_index_shapes(conn, schema, problems)
    _check_observation_timestamps(conn, schema, problems)
    _check_entry_key_constraints(conn, schema, problems)
    return problems


def require_ingestion_schema(conn, schema: str | None = None) -> list[str]:
    """Apply initial setup and return incompatibilities (empty when compatible)."""
    apply_ingestion_schema(conn)
    problems = verify_ingestion_schema(conn, schema)
    if problems:
        raise SchemaMismatchError(
            "ingestion schema is incompatible; no data was changed:\n  "
            + "\n  ".join(problems)
        )
    return problems


# --------------------------------------------------------------------------
# Ingestion-specific verification helpers.
# --------------------------------------------------------------------------
def _check_index_shapes(conn, schema: str, problems: list[str]) -> None:
    definitions: list[str] = []
    raw_unique: list[tuple[bool, str]] = []
    for table, is_unique, is_primary, definition in conn.execute(
        _INDEX_SQL, (schema,)
    ).fetchall():
        normalized = " ".join((definition or "").split()).lower()
        definitions.append(normalized)
        if table == "raw_articles" and is_unique:
            raw_unique.append((is_primary, normalized))

    if len(raw_unique) != len(_RAW_UNIQUE_EXPECTATIONS):
        problems.append(
            f"{schema}.raw_articles: expected exactly two unique indexes "
            f"(the primary key and raw_articles_identity), found "
            f"{len(raw_unique)}: "
            + ("; ".join(d for _, d in raw_unique) or "none")
        )
    for _name, columns, is_primary in _RAW_UNIQUE_EXPECTATIONS:
        if not any(defn.endswith(columns) for primary, defn in raw_unique
                   if primary == is_primary):
            problems.append(
                f"{schema}.raw_articles: missing unique index over {columns}"
            )

    for name, columns, table in _REVIEWED_INDEXES:
        marker = f"{table} using btree {columns}"
        if not any(name in defn and marker in defn for defn in definitions):
            problems.append(
                f"{schema}.{name}: missing the reviewed non-unique index "
                f"({name} on {table} {columns})"
            )


def _check_observation_timestamps(conn, schema: str, problems: list[str]) -> None:
    """No clock defaults on seen_at/first_seen_at/last_seen_at; retrieved_at NOT NULL."""
    found: dict[tuple[str, str], tuple[object, str]] = {}
    for table, column, default, nullable in conn.execute(
        _TIMESTAMP_COLUMN_SQL, (schema,)
    ).fetchall():
        found[(table, column)] = (default, nullable)

    no_default = (
        ("raw_articles", "first_seen_at"),
        ("raw_articles", "last_seen_at"),
        ("ingest_run_articles", "seen_at"),
    )
    for table, column in no_default + (("raw_articles", "retrieved_at"),):
        if (table, column) not in found:
            continue  # a missing column is already reported by verify_schema
        default, nullable = found[(table, column)]
        if (table, column) == ("raw_articles", "retrieved_at"):
            if nullable != "NO":
                problems.append(
                    f"{schema}.{table}.{column} must be NOT NULL; the typed "
                    "observation time is a collector guarantee"
                )
        elif default is not None:
            problems.append(
                f"{schema}.{table}.{column} must have no default; observed "
                f"times are supplied by the sweep writer (found {default!r})"
            )


def _check_entry_key_constraints(conn, schema: str, problems: list[str]) -> None:
    shapes = [
        " ".join(str(row[0] or "").split())
        for row in conn.execute(_CHECK_CONSTRAINT_DEFS_SQL, (schema,)).fetchall()
    ]
    joined = " ".join(shapes).lower()
    compact = joined.replace(" ", "")
    if "^(guid(-sha256)?|link(-sha256)?|sha256):" not in compact:
        problems.append(
            f"{schema}.raw_articles: the entry_key shape CHECK accepting "
            "guid(-sha256)/link(-sha256)/sha256 prefixes is missing; found "
            + ("; ".join(shapes) or "no CHECK constraints")
        )
    if "length(entry_key))<=2048" not in compact and "length(entry_key)<=2048" not in compact:
        problems.append(
            f"{schema}.raw_articles: the entry_key length CHECK "
            "(<= 2048 chars) is missing"
        )
