"""Read-only SQL queries for the I2 ingestion analysis section.

Nothing in this module imports psycopg: every function receives an open
connection from :mod:`market_intelligence.db.analytics`. All statements are
single ``SELECT``s; the caller guarantees a read-only session.

The SQL text is kept in module-level ``SQL_*`` constants so the offline suite
can statically assert that it is SELECT-only. Every query has a full
``ORDER BY`` with a tiebreaker so repeated readouts are deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

# The five ingestion objects whose presence decides whether the store is
# initialized. A subset of them is an incompatible (partial) schema.
INGEST_OBJECTS: tuple[str, ...] = (
    "ingest_sources",
    "ingest_runs",
    "raw_articles",
    "ingest_run_articles",
    "latest_articles",
)

SQL_SOURCES = """SELECT s.id, s.source_key, s.label, s.enabled,
       count(ra.id) AS versions,
       count(DISTINCT ra.entry_key_hash) AS entries,
       min(ra.first_seen_at) AS first_seen_at,
       max(ra.last_seen_at) AS last_seen_at
FROM ingest_sources s
LEFT JOIN raw_articles ra ON ra.source_id = s.id
GROUP BY s.id, s.source_key, s.label, s.enabled
ORDER BY s.source_key"""

SQL_LATEST_ENTRIES = """SELECT s.source_key, count(la.id) AS latest_entries
FROM ingest_sources s
LEFT JOIN latest_articles la ON la.source_id = s.id
GROUP BY s.source_key
ORDER BY s.source_key"""

SQL_RUN_TOTALS = """SELECT count(*) AS sweeps,
       count(*) FILTER (WHERE status = 'complete') AS complete,
       count(*) FILTER (WHERE status = 'partial') AS partial,
       count(*) FILTER (WHERE status = 'failed') AS failed,
       coalesce(sum(articles_seen), 0) AS seen,
       coalesce(sum(articles_stored), 0) AS stored,
       coalesce(sum(articles_replayed), 0) AS replayed
FROM ingest_runs"""

SQL_RECENT_SWEEPS = """SELECT sweep_id, status, started_at, finished_at,
       sources_attempted, sources_succeeded, sources_failed,
       articles_seen, articles_stored, articles_replayed
FROM ingest_runs
ORDER BY finished_at DESC, id DESC
LIMIT %s"""

# A WHERE clause cannot be relied on to guard set-returning-function
# evaluation, so a non-array source_results must be turned into an empty array
# by the CASE (it yields zero element rows instead of an error).
SQL_SOURCE_OUTCOMES = """SELECT r.sweep_id,
       coalesce(e->>'status', '<missing>') AS status,
       e->>'error_class' AS error_class
FROM ingest_runs r
CROSS JOIN LATERAL jsonb_array_elements(
  CASE WHEN jsonb_typeof(r.source_results) = 'array'
       THEN r.source_results ELSE '[]'::jsonb END) AS e
WHERE r.id IN (SELECT id FROM ingest_runs
               ORDER BY finished_at DESC, id DESC LIMIT %s)
ORDER BY r.finished_at DESC, r.id DESC, 2, 3"""

SQL_MALFORMED_SNAPSHOTS = """SELECT count(*) AS malformed FROM (
  SELECT source_results FROM ingest_runs
  ORDER BY finished_at DESC, id DESC LIMIT %s) t
WHERE jsonb_typeof(t.source_results) IS DISTINCT FROM 'array'"""

SQL_MULTI_VERSION = """SELECT s.source_key, count(*) AS entries
FROM (SELECT source_id, entry_key_hash FROM raw_articles
      GROUP BY source_id, entry_key_hash HAVING count(*) > 1) m
JOIN ingest_sources s ON s.id = m.source_id
GROUP BY s.source_key
ORDER BY 2 DESC, 1"""


@dataclass(frozen=True)
class SourceVolumeRow:
    """One source with its stored version/entry counts and observation window."""

    source_id: int
    source_key: str
    label: str
    enabled: bool
    versions: int
    entries: int
    first_seen_at: datetime | None
    last_seen_at: datetime | None


@dataclass(frozen=True)
class LatestEntryRow:
    """Entry-level count from the ``latest_articles`` view, per source key."""

    source_key: str
    latest_entries: int


@dataclass(frozen=True)
class SweepTotalsRow:
    sweeps: int
    complete: int
    partial: int
    failed: int
    articles_seen: int
    articles_stored: int
    articles_replayed: int


@dataclass(frozen=True)
class RecentSweepRow:
    sweep_id: str
    status: str
    started_at: datetime
    finished_at: datetime
    sources_attempted: int
    sources_succeeded: int
    sources_failed: int
    articles_seen: int
    articles_stored: int
    articles_replayed: int


@dataclass(frozen=True)
class SourceOutcomeRow:
    """One element of a sweep's ``source_results`` snapshot."""

    sweep_id: str
    status: str
    error_class: str | None


@dataclass(frozen=True)
class MultiVersionRow:
    source_key: str
    entries: int


@dataclass(frozen=True)
class IngestionRows:
    """Raw, unshaped rows for the ingestion report section."""

    sources: tuple[SourceVolumeRow, ...]
    latest_entries: tuple[LatestEntryRow, ...]
    totals: SweepTotalsRow
    recent_sweeps: tuple[RecentSweepRow, ...]
    outcomes: tuple[SourceOutcomeRow, ...]
    multi_version: tuple[MultiVersionRow, ...]
    malformed_snapshots: int


def load_ingestion(conn, *, recent_limit: int = 20) -> IngestionRows:
    """Run every ingestion-section query in a fixed order (all SELECT-only)."""
    sources = tuple(
        SourceVolumeRow(*row) for row in conn.execute(SQL_SOURCES).fetchall()
    )
    latest_entries = tuple(
        LatestEntryRow(*row) for row in conn.execute(SQL_LATEST_ENTRIES).fetchall()
    )
    totals_row = conn.execute(SQL_RUN_TOTALS).fetchone()
    totals = (
        SweepTotalsRow(*totals_row)
        if totals_row is not None
        else SweepTotalsRow(0, 0, 0, 0, 0, 0, 0)
    )
    recent_sweeps = tuple(
        RecentSweepRow(*row)
        for row in conn.execute(SQL_RECENT_SWEEPS, (recent_limit,)).fetchall()
    )
    outcomes = tuple(
        SourceOutcomeRow(*row)
        for row in conn.execute(SQL_SOURCE_OUTCOMES, (recent_limit,)).fetchall()
    )
    multi_version = tuple(
        MultiVersionRow(*row)
        for row in conn.execute(SQL_MULTI_VERSION).fetchall()
    )
    malformed_row = conn.execute(
        SQL_MALFORMED_SNAPSHOTS, (recent_limit,)
    ).fetchone()
    malformed_snapshots = int(malformed_row[0]) if malformed_row else 0
    return IngestionRows(
        sources=sources,
        latest_entries=latest_entries,
        totals=totals,
        recent_sweeps=recent_sweeps,
        outcomes=outcomes,
        multi_version=multi_version,
        malformed_snapshots=malformed_snapshots,
    )
