"""Initial-schema setup helpers for the local Postgres store.

``schema.sql`` is an initial setup script, not a migration framework. It only
creates missing objects. :func:`verify_schema` compares the live database with
the shape this code expects — column types, primary/foreign-key and unique
constraint definitions, and the ``latest_runs`` view — and reports an
incompatible existing shape so setup can stop without deleting anything.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SCHEMA_VERSION = 1
SCHEMA_PATH = Path(__file__).with_name("schema.sql")

# Columns we write and read. ``information_schema.columns.data_type`` reports
# bigserial as bigint, so that is the expected type for identity columns.
REQUIRED_COLUMNS: dict[str, dict[str, str]] = {
    "schema_meta": {
        "singleton": "boolean",
        "schema_version": "integer",
        "applied_at": "timestamp with time zone",
    },
    "runs": {
        "id": "bigint",
        "run_id": "text",
        "content_hash": "text",
        "schema_version": "integer",
        "rss_url": "text",
        "competitor_url": "text",
        "num_articles": "integer",
        "warnings": "jsonb",
        "payload": "jsonb",
        "stored_at": "timestamp with time zone",
    },
    "run_articles": {
        "run_version": "bigint",
        "article_id": "text",
        "ordinal": "integer",
        "headline": "text",
        "summary": "text",
        "key_facts": "jsonb",
        "tone_hint": "text",
        "feed_url": "text",
        "source_title": "text",
        "source_url": "text",
        "published_at": "timestamp with time zone",
        "published_at_raw": "text",
        "retrieved_at": "timestamp with time zone",
        "retrieved_at_raw": "text",
        "content_hash": "text",
        "excerpt": "text",
        "content_kind": "text",
    },
    "run_topics": {
        "run_version": "bigint",
        "ordinal": "integer",
        "topic_name": "text",
        "keywords": "jsonb",
        "model_confidence": "double precision",
        "window_start": "text",
        "window_end": "text",
        "source_count": "integer",
    },
    "run_topic_evidence": {
        "id": "bigint",
        "run_version": "bigint",
        "topic_ordinal": "integer",
        "article_id": "text",
        "matched_article_id": "text",
        "quote": "text",
        "source_title": "text",
        "source_url": "text",
        "published_at": "timestamp with time zone",
        "published_at_raw": "text",
        "verified": "boolean",
    },
    "run_tones": {
        "run_version": "bigint",
        "article_id": "text",
        "headline": "text",
        "overall": "text",
        "score": "double precision",
        "aspects": "jsonb",
        "source_url": "text",
        "published_at": "timestamp with time zone",
        "published_at_raw": "text",
        "retrieved_at": "timestamp with time zone",
        "retrieved_at_raw": "text",
    },
    "competitor_profiles": {
        "run_version": "bigint",
        "company": "text",
        "positioning": "text",
        "products": "jsonb",
        "pricing_model": "text",
        "funding_stage": "text",
        "strengths": "jsonb",
        "weaknesses": "jsonb",
        "sources": "jsonb",
        "retrieved_sources": "jsonb",
        "retrieved_at": "timestamp with time zone",
    },
}

_DELETE_ACTIONS = {
    "a": "NO ACTION",
    "r": "RESTRICT",
    "c": "CASCADE",
    "n": "SET NULL",
    "d": "SET DEFAULT",
}


@dataclass(frozen=True)
class Constraint:
    """One required constraint, described structurally and as DDL text."""

    kind: str  # "p" primary key, "u" unique, "f" foreign key
    columns: tuple[str, ...]
    ref_table: str | None = None
    ref_columns: tuple[str, ...] = ()
    on_delete: str | None = None
    name: str | None = None

    def definition(self) -> str:
        columns = ", ".join(self.columns)
        if self.kind == "p":
            return f"PRIMARY KEY ({columns})"
        if self.kind == "u":
            return f"UNIQUE ({columns})"
        action = _DELETE_ACTIONS.get(self.on_delete or "a", "NO ACTION")
        return (
            f"FOREIGN KEY ({columns}) REFERENCES {self.ref_table} "
            f"({', '.join(self.ref_columns)}) ON DELETE {action}"
        )


REQUIRED_CONSTRAINTS: dict[str, tuple[Constraint, ...]] = {
    "schema_meta": (Constraint("p", ("singleton",)),),
    "runs": (
        Constraint("p", ("id",)),
        Constraint(
            "u",
            ("run_id", "content_hash"),
            name="runs_identity",
        ),
    ),
    "run_articles": (
        Constraint("p", ("run_version", "article_id")),
        Constraint("u", ("run_version", "ordinal")),
        Constraint("f", ("run_version",), "runs", ("id",), "c"),
    ),
    "run_topics": (
        Constraint("p", ("run_version", "ordinal")),
        Constraint("f", ("run_version",), "runs", ("id",), "c"),
    ),
    "run_topic_evidence": (
        Constraint("p", ("id",)),
        Constraint(
            "f",
            ("run_version", "topic_ordinal"),
            "run_topics",
            ("run_version", "ordinal"),
            "c",
        ),
        Constraint(
            "f",
            ("run_version", "matched_article_id"),
            "run_articles",
            ("run_version", "article_id"),
            "c",
        ),
    ),
    "run_tones": (
        Constraint("p", ("run_version", "article_id")),
        Constraint("f", ("run_version",), "runs", ("id",), "c"),
        Constraint(
            "f",
            ("run_version", "article_id"),
            "run_articles",
            ("run_version", "article_id"),
            "c",
        ),
    ),
    "competitor_profiles": (
        Constraint("p", ("run_version",)),
        Constraint("f", ("run_version",), "runs", ("id",), "c"),
    ),
}

# latest_runs must expose these columns and select the highest run version per
# run id. Shape and definition are checked here; the integration suite also
# proves the behaviour by inserting two versions.
REQUIRED_VIEWS: dict[str, tuple[str, ...]] = {
    "latest_runs": ("run_id", "id", "content_hash", "stored_at"),
}
REQUIRED_VIEW_DEFINITION: dict[str, tuple[str, ...]] = {
    "latest_runs": ("distinct on (run_id)", "order by run_id, id desc"),
}

_CONSTRAINT_SQL = """
SELECT t.relname,
       c.conname,
       c.contype,
       ARRAY(
           SELECT a.attname
           FROM unnest(c.conkey) WITH ORDINALITY AS k(attnum, ord)
           JOIN pg_attribute a
             ON a.attrelid = c.conrelid AND a.attnum = k.attnum
           ORDER BY k.ord
       ) AS columns,
       ft.relname AS ref_table,
       ARRAY(
           SELECT a.attname
           FROM unnest(c.confkey) WITH ORDINALITY AS k(attnum, ord)
           JOIN pg_attribute a
             ON a.attrelid = c.confrelid AND a.attnum = k.attnum
           ORDER BY k.ord
       ) AS ref_columns,
       c.confdeltype,
       pg_get_constraintdef(c.oid) AS definition
FROM pg_constraint c
JOIN pg_class t ON t.oid = c.conrelid
JOIN pg_namespace n ON n.oid = t.relnamespace
LEFT JOIN pg_class ft ON ft.oid = c.confrelid
WHERE n.nspname = %s AND c.contype IN ('p', 'u', 'f')
"""


class SchemaMismatchError(RuntimeError):
    """The existing database does not match the schema this code expects."""


def apply_schema(conn) -> None:
    """Create missing objects. Safe to run repeatedly; never alters or drops."""
    conn.execute(SCHEMA_PATH.read_text(encoding="utf-8"))


def _present_columns(conn, schema: str) -> dict[str, dict[str, str]]:
    rows = conn.execute(
        "SELECT table_name, column_name, data_type "
        "FROM information_schema.columns WHERE table_schema = %s",
        (schema,),
    ).fetchall()
    present: dict[str, dict[str, str]] = {}
    for table, column, data_type in rows:
        present.setdefault(table, {})[column] = data_type
    return present


def _actual_constraints(conn, schema: str) -> dict[str, list[dict]]:
    rows = conn.execute(_CONSTRAINT_SQL, (schema,)).fetchall()
    actual: dict[str, list[dict]] = {}
    for (
        table,
        name,
        kind,
        columns,
        ref_table,
        ref_columns,
        on_delete,
        definition,
    ) in rows:
        actual.setdefault(table, []).append(
            {
                "name": name,
                "kind": kind,
                "columns": tuple(columns or ()),
                "ref_table": ref_table,
                "ref_columns": tuple(ref_columns or ()),
                "on_delete": on_delete,
                "definition": definition,
            }
        )
    return actual


def _matches(expected: Constraint, actual: dict) -> bool:
    if expected.kind != actual["kind"]:
        return False
    if expected.columns != actual["columns"]:
        return False
    if expected.kind != "f":
        return True
    return (
        expected.ref_table == actual["ref_table"]
        and expected.ref_columns == actual["ref_columns"]
        and (expected.on_delete or "a") == (actual["on_delete"] or "a")
    )


def _check_constraints(
    conn,
    schema: str,
    present: dict[str, dict[str, str]],
    problems: list[str],
) -> None:
    actual = _actual_constraints(conn, schema)
    for table, required in REQUIRED_CONSTRAINTS.items():
        if table not in present:
            continue  # the missing table is already reported
        found = actual.get(table, [])
        for expected in required:
            if any(_matches(expected, item) for item in found):
                continue
            by_name = (
                next(
                    (item for item in found if item["name"] == expected.name),
                    None,
                )
                if expected.name
                else None
            )
            if by_name is not None:
                problems.append(
                    f"{schema}.{table}: constraint {expected.name} has "
                    f"definition `{by_name['definition']}`, expected "
                    f"`{expected.definition()}`"
                )
                continue
            existing = (
                "; ".join(sorted(item["definition"] for item in found))
                or "none"
            )
            problems.append(
                f"{schema}.{table}: missing {expected.definition()}; "
                f"existing constraints: {existing}"
            )


def _check_views(
    conn,
    schema: str,
    present: dict[str, dict[str, str]],
    problems: list[str],
) -> None:
    views = {
        row[0]
        for row in conn.execute(
            "SELECT table_name FROM information_schema.views "
            "WHERE table_schema = %s",
            (schema,),
        ).fetchall()
    }
    definitions = {
        row[0]: row[1]
        for row in conn.execute(
            "SELECT viewname, definition FROM pg_views WHERE schemaname = %s",
            (schema,),
        ).fetchall()
    }

    for view, required_columns in REQUIRED_VIEWS.items():
        if view not in views:
            problems.append(f"{schema}.{view} view is missing")
            continue
        available = present.get(view, {})
        missing = sorted(set(required_columns) - set(available))
        if missing:
            problems.append(
                f"{schema}.{view} view is missing columns: {', '.join(missing)}"
            )
            continue
        normalized = " ".join((definitions.get(view) or "").split()).lower()
        for fragment in REQUIRED_VIEW_DEFINITION.get(view, ()):
            if fragment not in normalized:
                problems.append(
                    f"{schema}.{view} view definition does not contain "
                    f"`{fragment}`; actual definition: "
                    f"{definitions.get(view) or 'unavailable'}"
                )


def verify_schema(conn, schema: str | None = None) -> list[str]:
    """Return human-readable incompatibilities, or an empty list when compatible.

    Read-only: safe for ``init_db.py --check``. Checks required columns and
    their PostgreSQL types, primary/unique/foreign-key definitions (rendered as
    DDL), and the required views.
    """
    schema = schema or conn.execute("SELECT current_schema()").fetchone()[0]
    present = _present_columns(conn, schema)

    problems: list[str] = []
    for table, required in REQUIRED_COLUMNS.items():
        if table not in present:
            problems.append(f"{schema}.{table} is missing")
            continue
        missing = sorted(set(required) - set(present[table]))
        if missing:
            problems.append(
                f"{schema}.{table} is missing columns: {', '.join(missing)}"
            )
            continue
        wrong = [
            f"{column} is {present[table][column]!r}, expected {expected!r}"
            for column, expected in required.items()
            if present[table][column] != expected
        ]
        if wrong:
            problems.append(
                f"{schema}.{table} has wrong column types: " + "; ".join(wrong)
            )

    _check_constraints(conn, schema, present, problems)
    _check_views(conn, schema, present, problems)

    if "schema_meta" not in present:
        return problems
    try:
        row = conn.execute(
            "SELECT schema_version FROM schema_meta WHERE singleton"
        ).fetchone()
    except Exception as exc:  # table exists with an incompatible shape
        problems.append(f"{schema}.schema_meta cannot be read: {exc}")
        return problems

    if row is None:
        problems.append(f"{schema}.schema_meta has no version row")
    elif row[0] != SCHEMA_VERSION:
        problems.append(
            f"{schema}.schema_meta version is {row[0]}, "
            f"this code expects {SCHEMA_VERSION}"
        )
    return problems


def require_schema(conn, schema: str | None = None) -> list[str]:
    """Apply initial setup and return incompatibilities (empty when compatible)."""
    apply_schema(conn)
    problems = verify_schema(conn, schema)
    if problems:
        raise SchemaMismatchError(
            "database schema is incompatible; no data was changed:\n  "
            + "\n  ".join(problems)
        )
    return problems
