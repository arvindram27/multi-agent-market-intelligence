"""Initial-schema setup helpers for the local Postgres store.

``schema.sql`` is an initial setup script, not a migration framework. It only
creates missing objects. :func:`verify_schema` compares the live database with
the columns this code expects and reports an incompatible existing shape so
setup can stop without deleting anything.
"""
from __future__ import annotations

from pathlib import Path

SCHEMA_VERSION = 1
SCHEMA_PATH = Path(__file__).with_name("schema.sql")

REQUIRED_COLUMNS: dict[str, frozenset[str]] = {
    "schema_meta": frozenset({"singleton", "schema_version", "applied_at"}),
    "runs": frozenset(
        {
            "id",
            "run_id",
            "content_hash",
            "schema_version",
            "rss_url",
            "competitor_url",
            "num_articles",
            "warnings",
            "payload",
            "stored_at",
        }
    ),
    "run_articles": frozenset(
        {
            "run_version",
            "article_id",
            "ordinal",
            "headline",
            "summary",
            "key_facts",
            "tone_hint",
            "feed_url",
            "source_title",
            "source_url",
            "published_at",
            "published_at_raw",
            "retrieved_at",
            "retrieved_at_raw",
            "content_hash",
            "excerpt",
            "content_kind",
        }
    ),
    "run_topics": frozenset(
        {
            "run_version",
            "ordinal",
            "topic_name",
            "keywords",
            "model_confidence",
            "window_start",
            "window_end",
            "source_count",
        }
    ),
    "run_topic_evidence": frozenset(
        {
            "id",
            "run_version",
            "topic_ordinal",
            "article_id",
            "matched_article_id",
            "quote",
            "source_title",
            "source_url",
            "published_at",
            "published_at_raw",
            "verified",
        }
    ),
    "run_tones": frozenset(
        {
            "run_version",
            "article_id",
            "headline",
            "overall",
            "score",
            "aspects",
            "source_url",
            "published_at",
            "published_at_raw",
            "retrieved_at",
            "retrieved_at_raw",
        }
    ),
    "competitor_profiles": frozenset(
        {
            "run_version",
            "company",
            "positioning",
            "products",
            "pricing_model",
            "funding_stage",
            "strengths",
            "weaknesses",
            "sources",
            "retrieved_sources",
            "retrieved_at",
        }
    ),
}


class SchemaMismatchError(RuntimeError):
    """The existing database does not match the schema this code expects."""


def apply_schema(conn) -> None:
    """Create missing objects. Safe to run repeatedly; never alters or drops."""
    conn.execute(SCHEMA_PATH.read_text(encoding="utf-8"))


def verify_schema(conn, schema: str | None = None) -> list[str]:
    """Return human-readable incompatibilities, or an empty list when compatible."""
    schema = schema or conn.execute("SELECT current_schema()").fetchone()[0]
    rows = conn.execute(
        "SELECT table_name, column_name FROM information_schema.columns "
        "WHERE table_schema = %s",
        (schema,),
    ).fetchall()
    present: dict[str, set[str]] = {}
    for table, column in rows:
        present.setdefault(table, set()).add(column)

    problems: list[str] = []
    for table, required in REQUIRED_COLUMNS.items():
        if table not in present:
            problems.append(f"{schema}.{table} is missing")
            continue
        missing = sorted(required - present[table])
        if missing:
            problems.append(
                f"{schema}.{table} is missing columns: {', '.join(missing)}"
            )

    if "schema_meta" in present:
        try:
            row = conn.execute(
                "SELECT schema_version FROM schema_meta WHERE singleton"
            ).fetchone()
        except Exception as exc:  # table exists with an incompatible shape
            problems.append(f"{schema}.schema_meta cannot be read: {exc}")
        else:
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
