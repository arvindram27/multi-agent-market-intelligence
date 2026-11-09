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


# --------------------------------------------------------------------------
# Pipeline (Phase-1 approved-run) store: read-only queries for I2/M2.
#
# These constants never reference ingestion objects and the ingestion constants
# above never reference pipeline objects: the two stores stay independent and
# are never joined. One SELECT per constant; list queries are bounded and
# ordered.
# --------------------------------------------------------------------------

PIPELINE_OBJECTS: tuple[str, ...] = (
    "runs",
    "run_articles",
    "run_topics",
    "run_topic_evidence",
    "run_tones",
    "competitor_profiles",
    "latest_runs",
)

# Every bounded list in the pipeline section (per-topic, keywords, aspects)
# uses this cap; totals and distinct counts stay unbounded so overflow remains
# visible as total minus listed.
PIPELINE_TOP_LIMIT = 20

SQL_PIPELINE_RUN_TOTALS = """SELECT count(*) AS run_versions,
       count(DISTINCT run_id) AS run_ids,
       max(stored_at) AS newest_stored_at
FROM runs"""

SQL_PIPELINE_MULTI_VERSION_RUN_IDS = """SELECT count(*) AS run_ids_with_multiple_versions
FROM (SELECT run_id FROM runs GROUP BY run_id HAVING count(*) > 1) AS multi"""

SQL_PIPELINE_RECENT_RUNS = """SELECT l.run_id, l.id AS run_version_id, r.stored_at, r.num_articles
FROM latest_runs l
JOIN runs r ON r.id = l.id
ORDER BY r.stored_at DESC, l.id DESC
LIMIT %s"""

SQL_PIPELINE_ARTICLE_COUNTS = """SELECT (SELECT count(*) FROM run_articles) AS article_rows_all_versions,
       (SELECT count(*)
        FROM latest_runs l JOIN run_articles a ON a.run_version = l.id
       ) AS article_rows_latest_versions"""

SQL_PIPELINE_TOPIC_TOTALS = """SELECT (SELECT count(*) FROM run_topics) AS topic_rows_all_versions,
       (SELECT count(*)
        FROM latest_runs l JOIN run_topics t ON t.run_version = l.id
       ) AS topic_rows_latest_versions,
       (SELECT count(DISTINCT t.topic_name)
        FROM latest_runs l JOIN run_topics t ON t.run_version = l.id
       ) AS distinct_topics_latest_versions"""

SQL_PIPELINE_PER_TOPIC_LATEST = """SELECT t.topic_name,
       count(*) AS topic_occurrences_latest_versions,
       count(DISTINCT l.run_id) AS distinct_runs_latest_versions,
       avg(t.model_confidence) AS avg_model_confidence_latest_versions,
       avg(t.source_count) AS avg_source_count_latest_versions
FROM latest_runs l
JOIN run_topics t ON t.run_version = l.id
GROUP BY t.topic_name
ORDER BY topic_occurrences_latest_versions DESC, t.topic_name ASC
LIMIT %s"""

SQL_PIPELINE_KEYWORDS_TOP = """SELECT CASE WHEN jsonb_typeof(k.elem) = 'string'
            THEN k.elem #>> '{}' ELSE '<missing>' END AS keyword,
       count(*) AS keyword_occurrences_latest_versions
FROM latest_runs l
JOIN run_topics t ON t.run_version = l.id
CROSS JOIN LATERAL jsonb_array_elements(
  CASE WHEN jsonb_typeof(t.keywords) = 'array'
       THEN t.keywords ELSE '[]'::jsonb END) AS k(elem)
GROUP BY 1
ORDER BY keyword_occurrences_latest_versions DESC, keyword ASC
LIMIT %s"""

SQL_PIPELINE_KEYWORDS_NONARRAY = """SELECT count(*) AS keywords_nonarray_rows_latest_versions
FROM latest_runs l
JOIN run_topics t ON t.run_version = l.id
WHERE jsonb_typeof(t.keywords) IS DISTINCT FROM 'array'"""

SQL_PIPELINE_EVIDENCE_TOTALS_LATEST = """SELECT count(*) AS evidence_rows_latest_versions,
       count(*) FILTER (WHERE e.verified) AS verified_latest_versions,
       count(*) FILTER (WHERE NOT e.verified) AS unverified_latest_versions,
       count(*) FILTER (WHERE NOT e.verified AND e.source_title = 'Unknown source')
         AS unknown_source_citations_latest_versions
FROM latest_runs l
JOIN run_topic_evidence e ON e.run_version = l.id"""

SQL_PIPELINE_EVIDENCE_ALL = """SELECT count(*) AS evidence_rows_all_versions FROM run_topic_evidence"""

SQL_PIPELINE_TONE_TOTALS = """SELECT (SELECT count(*) FROM run_tones) AS tone_rows_all_versions,
       (SELECT count(*)
        FROM latest_runs l JOIN run_tones t ON t.run_version = l.id
       ) AS tone_rows_latest_versions"""

SQL_PIPELINE_TONE_DISTRIBUTION = """SELECT t.overall,
       count(*) AS tone_occurrences_latest_versions,
       avg(t.score) AS avg_score_latest_versions
FROM latest_runs l
JOIN run_tones t ON t.run_version = l.id
GROUP BY t.overall
ORDER BY t.overall ASC"""

SQL_PIPELINE_TONE_SCORE_RANGE = """SELECT min(t.score) AS min_score_latest_versions,
       max(t.score) AS max_score_latest_versions,
       avg(t.score) AS avg_score_latest_versions
FROM latest_runs l
JOIN run_tones t ON t.run_version = l.id"""

SQL_PIPELINE_ASPECTS_TOP = """SELECT CASE WHEN jsonb_typeof(a.elem) = 'object'
            THEN coalesce(a.elem ->> 'aspect', '<missing>')
            ELSE '<missing>' END AS aspect,
       count(*) AS aspect_occurrences_latest_versions
FROM latest_runs l
JOIN run_tones t ON t.run_version = l.id
CROSS JOIN LATERAL jsonb_array_elements(
  CASE WHEN jsonb_typeof(t.aspects) = 'array'
       THEN t.aspects ELSE '[]'::jsonb END) AS a(elem)
GROUP BY 1
ORDER BY aspect_occurrences_latest_versions DESC, aspect ASC
LIMIT %s"""

SQL_PIPELINE_ASPECTS_NONARRAY = """SELECT count(*) AS aspects_nonarray_rows_latest_versions
FROM latest_runs l
JOIN run_tones t ON t.run_version = l.id
WHERE jsonb_typeof(t.aspects) IS DISTINCT FROM 'array'"""


@dataclass(frozen=True)
class PipelineRunTotalsRow:
    run_versions: int
    run_ids: int
    newest_stored_at: datetime | None


@dataclass(frozen=True)
class PipelineMultiVersionRow:
    run_ids_with_multiple_versions: int


@dataclass(frozen=True)
class PipelineRecentRunRow:
    run_id: str
    run_version_id: int
    stored_at: datetime
    num_articles: int


@dataclass(frozen=True)
class PipelineArticleCountsRow:
    article_rows_all_versions: int
    article_rows_latest_versions: int


@dataclass(frozen=True)
class PipelineTopicTotalsRow:
    topic_rows_all_versions: int
    topic_rows_latest_versions: int
    distinct_topics_latest_versions: int


@dataclass(frozen=True)
class PipelinePerTopicRow:
    topic_name: str
    topic_occurrences_latest_versions: int
    distinct_runs_latest_versions: int
    avg_model_confidence_latest_versions: float
    avg_source_count_latest_versions: object  # numeric; shaped to float


@dataclass(frozen=True)
class PipelineKeywordRow:
    keyword: str
    keyword_occurrences_latest_versions: int


@dataclass(frozen=True)
class PipelineEvidenceTotalsRow:
    evidence_rows_latest_versions: int
    verified_latest_versions: int
    unverified_latest_versions: int
    unknown_source_citations_latest_versions: int


@dataclass(frozen=True)
class PipelineToneTotalsRow:
    tone_rows_all_versions: int
    tone_rows_latest_versions: int


@dataclass(frozen=True)
class PipelineToneDistributionRow:
    overall: str
    tone_occurrences_latest_versions: int
    avg_score_latest_versions: float


@dataclass(frozen=True)
class PipelineToneScoreRangeRow:
    min_score_latest_versions: float | None
    max_score_latest_versions: float | None
    avg_score_latest_versions: float | None


@dataclass(frozen=True)
class PipelineAspectRow:
    aspect: str
    aspect_occurrences_latest_versions: int


@dataclass(frozen=True)
class PipelineRows:
    """Raw, unshaped rows for the pipeline report section."""

    run_totals: PipelineRunTotalsRow
    run_ids_with_multiple_versions: int
    recent_runs: tuple[PipelineRecentRunRow, ...]
    article_counts: PipelineArticleCountsRow
    topic_totals: PipelineTopicTotalsRow
    topics: tuple[PipelinePerTopicRow, ...]
    keywords: tuple[PipelineKeywordRow, ...]
    keywords_nonarray_rows_latest_versions: int
    evidence_totals: PipelineEvidenceTotalsRow
    evidence_rows_all_versions: int
    tone_totals: PipelineToneTotalsRow
    tone_distribution: tuple[PipelineToneDistributionRow, ...]
    tone_score_range: PipelineToneScoreRangeRow
    aspects: tuple[PipelineAspectRow, ...]
    aspects_nonarray_rows_latest_versions: int


def _one_or_zero(conn, sql: str, params=()) -> int:
    row = conn.execute(sql, params).fetchone()
    return int(row[0]) if row is not None and row[0] is not None else 0


def load_pipeline(conn, *, recent_limit: int = 20) -> PipelineRows:
    """Run every pipeline-section query in a fixed order (all SELECT-only)."""
    totals_row = conn.execute(SQL_PIPELINE_RUN_TOTALS).fetchone()
    run_totals = (
        PipelineRunTotalsRow(*totals_row)
        if totals_row is not None
        else PipelineRunTotalsRow(0, 0, None)
    )
    multi = _one_or_zero(conn, SQL_PIPELINE_MULTI_VERSION_RUN_IDS)
    recent_runs = tuple(
        PipelineRecentRunRow(*row)
        for row in conn.execute(SQL_PIPELINE_RECENT_RUNS, (recent_limit,)).fetchall()
    )
    article_row = conn.execute(SQL_PIPELINE_ARTICLE_COUNTS).fetchone()
    article_counts = (
        PipelineArticleCountsRow(*article_row)
        if article_row is not None
        else PipelineArticleCountsRow(0, 0)
    )
    topic_totals_row = conn.execute(SQL_PIPELINE_TOPIC_TOTALS).fetchone()
    topic_totals = (
        PipelineTopicTotalsRow(*topic_totals_row)
        if topic_totals_row is not None
        else PipelineTopicTotalsRow(0, 0, 0)
    )
    topics = tuple(
        PipelinePerTopicRow(*row)
        for row in conn.execute(
            SQL_PIPELINE_PER_TOPIC_LATEST, (PIPELINE_TOP_LIMIT,)
        ).fetchall()
    )
    keywords = tuple(
        PipelineKeywordRow(*row)
        for row in conn.execute(
            SQL_PIPELINE_KEYWORDS_TOP, (PIPELINE_TOP_LIMIT,)
        ).fetchall()
    )
    keywords_nonarray = _one_or_zero(conn, SQL_PIPELINE_KEYWORDS_NONARRAY)
    evidence_row = conn.execute(SQL_PIPELINE_EVIDENCE_TOTALS_LATEST).fetchone()
    evidence_totals = (
        PipelineEvidenceTotalsRow(*evidence_row)
        if evidence_row is not None
        else PipelineEvidenceTotalsRow(0, 0, 0, 0)
    )
    evidence_all = _one_or_zero(conn, SQL_PIPELINE_EVIDENCE_ALL)
    tone_totals_row = conn.execute(SQL_PIPELINE_TONE_TOTALS).fetchone()
    tone_totals = (
        PipelineToneTotalsRow(*tone_totals_row)
        if tone_totals_row is not None
        else PipelineToneTotalsRow(0, 0)
    )
    tone_distribution = tuple(
        PipelineToneDistributionRow(*row)
        for row in conn.execute(SQL_PIPELINE_TONE_DISTRIBUTION).fetchall()
    )
    score_row = conn.execute(SQL_PIPELINE_TONE_SCORE_RANGE).fetchone()
    tone_score_range = (
        PipelineToneScoreRangeRow(*score_row)
        if score_row is not None
        else PipelineToneScoreRangeRow(None, None, None)
    )
    aspects = tuple(
        PipelineAspectRow(*row)
        for row in conn.execute(
            SQL_PIPELINE_ASPECTS_TOP, (PIPELINE_TOP_LIMIT,)
        ).fetchall()
    )
    aspects_nonarray = _one_or_zero(conn, SQL_PIPELINE_ASPECTS_NONARRAY)
    return PipelineRows(
        run_totals=run_totals,
        run_ids_with_multiple_versions=multi,
        recent_runs=recent_runs,
        article_counts=article_counts,
        topic_totals=topic_totals,
        topics=topics,
        keywords=keywords,
        keywords_nonarray_rows_latest_versions=keywords_nonarray,
        evidence_totals=evidence_totals,
        evidence_rows_all_versions=evidence_all,
        tone_totals=tone_totals,
        tone_distribution=tone_distribution,
        tone_score_range=tone_score_range,
        aspects=aspects,
        aspects_nonarray_rows_latest_versions=aspects_nonarray,
    )
