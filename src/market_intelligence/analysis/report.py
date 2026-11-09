"""Pure shaping and rendering for the I2 analysis report.

No psycopg and no connection here: :func:`build_report` (ingestion only, M1) and
:func:`build_analysis_report` (composition with the M2 pipeline section) turn
raw rows from :mod:`market_intelligence.analysis.queries` into frozen report
objects that render deterministically to text or JSON.

Stale contract (single definition): every source row always carries ``stale``.
It is ``None`` exactly when the source has no observations (``last_seen_at`` is
``None``). Otherwise ``stale = enabled and age_seconds > stale_hours``, with
``age_seconds`` clamped to 0 for future timestamps (the timestamp is still
reported). Replay rates are ``None`` when the denominator is 0.

Pipeline contract: every metric carries its scope. ``*_all_versions`` is
occurrence-level (rows across all stored run versions), ``run_ids`` is
run-level, and ``*_latest_versions`` is latest-version-only (joined to
``latest_runs``). ``latest_runs`` is the highest ``runs.id`` per ``run_id``
(insertion order); ``stored_at`` is statement time and any freshness derived
from it is advisory.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from market_intelligence.analysis.queries import (
    IngestionRows,
    MultiVersionRow,
    PipelineAspectRow,
    PipelineKeywordRow,
    PipelinePerTopicRow,
    PipelineRows,
    PipelineToneDistributionRow,
    RecentSweepRow,
    SourceOutcomeRow,
    SourceVolumeRow,
    SweepTotalsRow,
)

REPORT_VERSION = 1

_SECONDS_PER_HOUR = 3600


def utcnow() -> datetime:
    """Injectable clock seam for deterministic output in tests."""
    return datetime.now(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return _as_utc(value).isoformat()


def _iso_or_na(value: datetime | None) -> str:
    return "n/a" if value is None else _iso(value)


def _int_or_na(value: int | None) -> str:
    return "n/a" if value is None else str(value)


def _stale_text(value: bool | None) -> str:
    if value is None:
        return "n/a"
    return "yes" if value else "no"


def _rate_text(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _number_or_na(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _replay_rate(stored: int, replayed: int) -> float | None:
    denominator = stored + replayed
    if denominator <= 0:
        return None
    return round(replayed / denominator, 6)


def _verification_rate(verified: int, total: int) -> float | None:
    if total <= 0:
        return None
    return round(verified / total, 6)


# --------------------------------------------------------------------------
# Ingestion section (M1)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SourceReport:
    source_key: str
    label: str
    enabled: bool
    versions: int
    entries: int
    latest_entries: int
    first_seen_at: datetime | None
    last_seen_at: datetime | None
    age_seconds: int | None
    stale: bool | None


@dataclass(frozen=True)
class SweepTotalsReport:
    sweeps: int
    complete: int
    partial: int
    failed: int
    other: int
    articles_seen: int
    articles_stored: int
    articles_replayed: int
    replay_rate: float | None


@dataclass(frozen=True)
class RecentSweepReport:
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
    replay_rate: float | None


@dataclass(frozen=True)
class OutcomeReport:
    # (status literal, count) ordered by count desc then status asc.
    status_counts: tuple[tuple[str, int], ...]
    error_classes: tuple[str, ...]
    elements_with_no_error_class: int
    malformed_snapshots: int


@dataclass(frozen=True)
class MultiVersionReport:
    source_key: str
    entries: int


@dataclass(frozen=True)
class IngestionReport:
    available: bool
    recent_limit: int
    sources: tuple[SourceReport, ...]
    totals: SweepTotalsReport | None
    recent_sweeps: tuple[RecentSweepReport, ...]
    outcomes: OutcomeReport | None
    multi_version: tuple[MultiVersionReport, ...]


# --------------------------------------------------------------------------
# Pipeline section (M2), all labels carry their scope
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PipelineRecentRunReport:
    run_id: str
    run_version_id: int
    stored_at: datetime
    num_articles: int


@dataclass(frozen=True)
class PipelineTopicReport:
    topic_name: str
    topic_occurrences_latest_versions: int
    distinct_runs_latest_versions: int
    avg_model_confidence_latest_versions: float | None
    avg_source_count_latest_versions: float | None


@dataclass(frozen=True)
class PipelineKeywordReport:
    keyword: str
    keyword_occurrences_latest_versions: int


@dataclass(frozen=True)
class PipelineToneDistributionReport:
    overall: str
    tone_occurrences_latest_versions: int
    avg_score_latest_versions: float | None


@dataclass(frozen=True)
class PipelineAspectReport:
    aspect: str
    aspect_occurrences_latest_versions: int


@dataclass(frozen=True)
class PipelineReport:
    available: bool
    recent_limit: int
    run_versions: int
    run_ids: int
    run_ids_with_multiple_versions: int
    newest_stored_at: datetime | None
    age_seconds: int | None
    recent_runs: tuple[PipelineRecentRunReport, ...]
    article_rows_all_versions: int
    article_rows_latest_versions: int
    topic_rows_all_versions: int
    topic_rows_latest_versions: int
    distinct_topics_latest_versions: int
    topics: tuple[PipelineTopicReport, ...]
    keywords: tuple[PipelineKeywordReport, ...]
    keywords_nonarray_rows_latest_versions: int
    evidence_rows_all_versions: int
    evidence_rows_latest_versions: int
    verified_latest_versions: int
    unverified_latest_versions: int
    verification_rate_latest_versions: float | None
    unknown_source_citations_latest_versions: int
    tone_rows_all_versions: int
    tone_rows_latest_versions: int
    tone_distribution: tuple[PipelineToneDistributionReport, ...]
    min_score_latest_versions: float | None
    max_score_latest_versions: float | None
    avg_score_latest_versions: float | None
    aspects: tuple[PipelineAspectReport, ...]
    aspects_nonarray_rows_latest_versions: int


def _empty_pipeline_report(recent_limit: int) -> PipelineReport:
    return PipelineReport(
        available=False,
        recent_limit=recent_limit,
        run_versions=0,
        run_ids=0,
        run_ids_with_multiple_versions=0,
        newest_stored_at=None,
        age_seconds=None,
        recent_runs=(),
        article_rows_all_versions=0,
        article_rows_latest_versions=0,
        topic_rows_all_versions=0,
        topic_rows_latest_versions=0,
        distinct_topics_latest_versions=0,
        topics=(),
        keywords=(),
        keywords_nonarray_rows_latest_versions=0,
        evidence_rows_all_versions=0,
        evidence_rows_latest_versions=0,
        verified_latest_versions=0,
        unverified_latest_versions=0,
        verification_rate_latest_versions=None,
        unknown_source_citations_latest_versions=0,
        tone_rows_all_versions=0,
        tone_rows_latest_versions=0,
        tone_distribution=(),
        min_score_latest_versions=None,
        max_score_latest_versions=None,
        avg_score_latest_versions=None,
        aspects=(),
        aspects_nonarray_rows_latest_versions=0,
    )


@dataclass(frozen=True)
class AnalysisReport:
    generated_at: datetime
    ingestion: IngestionReport | None
    report_version: int = REPORT_VERSION
    pipeline: PipelineReport | None = None

    def to_dict(self) -> dict[str, Any]:
        """Fixed-key-order JSON-ready payload; excluded stores stay omitted."""
        payload: dict[str, Any] = {
            "report_version": self.report_version,
            "generated_at": _iso(self.generated_at),
        }
        if self.ingestion is not None:
            payload["ingestion"] = _ingestion_dict(self.ingestion)
        if self.pipeline is not None:
            payload["pipeline"] = _pipeline_dict(self.pipeline)
        return payload

    def render_text(self) -> str:
        lines = [
            "Market intelligence - SQL analysis",
            f"report_version: {self.report_version}",
            f"generated_at: {_iso(self.generated_at)}",
            "",
        ]
        sections: list[list[str]] = []
        if self.ingestion is not None:
            sections.append(_ingestion_lines(self.ingestion))
        if self.pipeline is not None:
            sections.append(_pipeline_lines(self.pipeline))
        for index, section in enumerate(sections):
            if index and section:
                lines.append("")
            lines.extend(section)
        return "\n".join(lines) + "\n"


def render_text(report: AnalysisReport) -> str:
    """Render a report to its deterministic text form."""
    return report.render_text()


# --------------------------------------------------------------------------
# Ingestion shaping (unchanged M1 behaviour and bytes)
# --------------------------------------------------------------------------
def _source_report(
    row: SourceVolumeRow,
    latest_entries: int,
    generated_at: datetime,
    stale_after_hours: float,
) -> SourceReport:
    if row.last_seen_at is None:
        age_seconds: int | None = None
        stale: bool | None = None
    else:
        age_seconds = int(
            max(0.0, (generated_at - _as_utc(row.last_seen_at)).total_seconds())
        )
        stale = bool(
            row.enabled and age_seconds > stale_after_hours * _SECONDS_PER_HOUR
        )
    return SourceReport(
        source_key=row.source_key,
        label=row.label,
        enabled=row.enabled,
        versions=row.versions,
        entries=row.entries,
        latest_entries=latest_entries,
        first_seen_at=row.first_seen_at,
        last_seen_at=row.last_seen_at,
        age_seconds=age_seconds,
        stale=stale,
    )


def _totals_report(row: SweepTotalsRow) -> SweepTotalsReport:
    other = row.sweeps - row.complete - row.partial - row.failed
    return SweepTotalsReport(
        sweeps=row.sweeps,
        complete=row.complete,
        partial=row.partial,
        failed=row.failed,
        other=other,
        articles_seen=row.articles_seen,
        articles_stored=row.articles_stored,
        articles_replayed=row.articles_replayed,
        replay_rate=_replay_rate(row.articles_stored, row.articles_replayed),
    )


def _recent_sweep_report(row: RecentSweepRow) -> RecentSweepReport:
    return RecentSweepReport(
        sweep_id=row.sweep_id,
        status=row.status,
        started_at=row.started_at,
        finished_at=row.finished_at,
        sources_attempted=row.sources_attempted,
        sources_succeeded=row.sources_succeeded,
        sources_failed=row.sources_failed,
        articles_seen=row.articles_seen,
        articles_stored=row.articles_stored,
        articles_replayed=row.articles_replayed,
        replay_rate=_replay_rate(row.articles_stored, row.articles_replayed),
    )


def _outcomes_report(
    rows: tuple[SourceOutcomeRow, ...], malformed_snapshots: int
) -> OutcomeReport:
    counts: dict[str, int] = {}
    error_classes: set[str] = set()
    no_error_class = 0
    for row in rows:
        counts[row.status] = counts.get(row.status, 0) + 1
        if row.error_class is None:
            no_error_class += 1
        else:
            error_classes.add(row.error_class)
    status_counts = tuple(
        sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    )
    return OutcomeReport(
        status_counts=status_counts,
        error_classes=tuple(sorted(error_classes)),
        elements_with_no_error_class=no_error_class,
        malformed_snapshots=malformed_snapshots,
    )


def _ingestion_report(
    rows: IngestionRows | None,
    generated_at: datetime,
    stale_after_hours: float,
    recent_limit: int,
) -> IngestionReport:
    if rows is None:
        return IngestionReport(
            available=False,
            recent_limit=recent_limit,
            sources=(),
            totals=None,
            recent_sweeps=(),
            outcomes=None,
            multi_version=(),
        )
    latest = {row.source_key: row.latest_entries for row in rows.latest_entries}
    sources = tuple(
        _source_report(
            row,
            latest.get(row.source_key, 0),
            generated_at,
            stale_after_hours,
        )
        for row in rows.sources
    )
    multi_version: tuple[MultiVersionReport, ...] = tuple(
        MultiVersionReport(source_key=row.source_key, entries=row.entries)
        for row in rows.multi_version
    )
    return IngestionReport(
        available=True,
        recent_limit=recent_limit,
        sources=sources,
        totals=_totals_report(rows.totals),
        recent_sweeps=tuple(_recent_sweep_report(row) for row in rows.recent_sweeps),
        outcomes=_outcomes_report(rows.outcomes, rows.malformed_snapshots),
        multi_version=multi_version,
    )


# --------------------------------------------------------------------------
# Pipeline shaping
# --------------------------------------------------------------------------
def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _pipeline_report(
    rows: PipelineRows | None,
    generated_at: datetime,
    recent_limit: int,
) -> PipelineReport:
    if rows is None:
        return _empty_pipeline_report(recent_limit)

    newest = rows.run_totals.newest_stored_at
    if newest is None:
        age_seconds: int | None = None
    else:
        age_seconds = int(
            max(0.0, (generated_at - _as_utc(newest)).total_seconds())
        )

    topics = tuple(
        PipelineTopicReport(
            topic_name=row.topic_name,
            topic_occurrences_latest_versions=(
                row.topic_occurrences_latest_versions
            ),
            distinct_runs_latest_versions=row.distinct_runs_latest_versions,
            avg_model_confidence_latest_versions=_optional_float(
                row.avg_model_confidence_latest_versions
            ),
            avg_source_count_latest_versions=_optional_float(
                row.avg_source_count_latest_versions
            ),
        )
        for row in rows.topics
    )
    keywords = tuple(
        PipelineKeywordReport(
            keyword=row.keyword,
            keyword_occurrences_latest_versions=(
                row.keyword_occurrences_latest_versions
            ),
        )
        for row in rows.keywords
    )
    tone_distribution = tuple(
        PipelineToneDistributionReport(
            overall=row.overall,
            tone_occurrences_latest_versions=(
                row.tone_occurrences_latest_versions
            ),
            avg_score_latest_versions=_optional_float(
                row.avg_score_latest_versions
            ),
        )
        for row in rows.tone_distribution
    )
    aspects = tuple(
        PipelineAspectReport(
            aspect=row.aspect,
            aspect_occurrences_latest_versions=(
                row.aspect_occurrences_latest_versions
            ),
        )
        for row in rows.aspects
    )
    return PipelineReport(
        available=True,
        recent_limit=recent_limit,
        run_versions=rows.run_totals.run_versions,
        run_ids=rows.run_totals.run_ids,
        run_ids_with_multiple_versions=(
            rows.run_ids_with_multiple_versions
        ),
        newest_stored_at=newest,
        age_seconds=age_seconds,
        recent_runs=tuple(
            PipelineRecentRunReport(
                run_id=row.run_id,
                run_version_id=row.run_version_id,
                stored_at=row.stored_at,
                num_articles=row.num_articles,
            )
            for row in rows.recent_runs
        ),
        article_rows_all_versions=(
            rows.article_counts.article_rows_all_versions
        ),
        article_rows_latest_versions=(
            rows.article_counts.article_rows_latest_versions
        ),
        topic_rows_all_versions=rows.topic_totals.topic_rows_all_versions,
        topic_rows_latest_versions=(
            rows.topic_totals.topic_rows_latest_versions
        ),
        distinct_topics_latest_versions=(
            rows.topic_totals.distinct_topics_latest_versions
        ),
        topics=topics,
        keywords=keywords,
        keywords_nonarray_rows_latest_versions=(
            rows.keywords_nonarray_rows_latest_versions
        ),
        evidence_rows_all_versions=rows.evidence_rows_all_versions,
        evidence_rows_latest_versions=(
            rows.evidence_totals.evidence_rows_latest_versions
        ),
        verified_latest_versions=(
            rows.evidence_totals.verified_latest_versions
        ),
        unverified_latest_versions=(
            rows.evidence_totals.unverified_latest_versions
        ),
        verification_rate_latest_versions=_verification_rate(
            rows.evidence_totals.verified_latest_versions,
            rows.evidence_totals.evidence_rows_latest_versions,
        ),
        unknown_source_citations_latest_versions=(
            rows.evidence_totals.unknown_source_citations_latest_versions
        ),
        tone_rows_all_versions=rows.tone_totals.tone_rows_all_versions,
        tone_rows_latest_versions=rows.tone_totals.tone_rows_latest_versions,
        tone_distribution=tone_distribution,
        min_score_latest_versions=_optional_float(
            rows.tone_score_range.min_score_latest_versions
        ),
        max_score_latest_versions=_optional_float(
            rows.tone_score_range.max_score_latest_versions
        ),
        avg_score_latest_versions=_optional_float(
            rows.tone_score_range.avg_score_latest_versions
        ),
        aspects=aspects,
        aspects_nonarray_rows_latest_versions=(
            rows.aspects_nonarray_rows_latest_versions
        ),
    )


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------
def build_analysis_report(
    ingestion_rows: IngestionRows | None,
    pipeline_rows: PipelineRows | None,
    *,
    include_ingestion: bool,
    include_pipeline: bool,
    generated_at: datetime | None = None,
    stale_after_hours: float = 24,
    recent_limit: int = 20,
    recent_runs: int = 20,
) -> AnalysisReport:
    """Compose the selected store sections into one deterministic report.

    ``ingestion_rows=None`` / ``pipeline_rows=None`` means the selected store's
    objects are all absent (the section reports ``available: false``). Excluded
    stores are omitted from both text and JSON.
    """
    generated = _as_utc(generated_at or utcnow())
    ingestion = (
        _ingestion_report(
            ingestion_rows, generated, stale_after_hours, recent_limit
        )
        if include_ingestion
        else None
    )
    pipeline = (
        _pipeline_report(pipeline_rows, generated, recent_runs)
        if include_pipeline
        else None
    )
    return AnalysisReport(
        generated_at=generated,
        ingestion=ingestion,
        pipeline=pipeline,
    )


def build_report(
    rows: IngestionRows | None,
    *,
    generated_at: datetime | None = None,
    stale_after_hours: float = 24,
    recent_limit: int = 20,
) -> AnalysisReport:
    """M1-compatible wrapper: ingestion-only report, byte-identical output."""
    return build_analysis_report(
        rows,
        None,
        include_ingestion=True,
        include_pipeline=False,
        generated_at=generated_at,
        stale_after_hours=stale_after_hours,
        recent_limit=recent_limit,
    )


# --------------------------------------------------------------------------
# JSON payloads
# --------------------------------------------------------------------------
def _ingestion_dict(ingestion: IngestionReport) -> dict[str, Any]:
    return {
        "available": ingestion.available,
        "recent_limit": ingestion.recent_limit,
        "sources": [
            {
                "source_key": source.source_key,
                "label": source.label,
                "enabled": source.enabled,
                "versions": source.versions,
                "entries": source.entries,
                "latest_entries": source.latest_entries,
                "first_seen_at": (
                    None if source.first_seen_at is None else _iso(source.first_seen_at)
                ),
                "last_seen_at": (
                    None if source.last_seen_at is None else _iso(source.last_seen_at)
                ),
                "age_seconds": source.age_seconds,
                "stale": source.stale,
            }
            for source in ingestion.sources
        ],
        "totals": (
            None
            if ingestion.totals is None
            else {
                "sweeps": ingestion.totals.sweeps,
                "complete": ingestion.totals.complete,
                "partial": ingestion.totals.partial,
                "failed": ingestion.totals.failed,
                "other": ingestion.totals.other,
                "articles_seen": ingestion.totals.articles_seen,
                "articles_stored": ingestion.totals.articles_stored,
                "articles_replayed": ingestion.totals.articles_replayed,
                "replay_rate": ingestion.totals.replay_rate,
            }
        ),
        "recent_sweeps": [
            {
                "sweep_id": sweep.sweep_id,
                "status": sweep.status,
                "started_at": _iso(sweep.started_at),
                "finished_at": _iso(sweep.finished_at),
                "sources_attempted": sweep.sources_attempted,
                "sources_succeeded": sweep.sources_succeeded,
                "sources_failed": sweep.sources_failed,
                "articles_seen": sweep.articles_seen,
                "articles_stored": sweep.articles_stored,
                "articles_replayed": sweep.articles_replayed,
                "replay_rate": sweep.replay_rate,
            }
            for sweep in ingestion.recent_sweeps
        ],
        "outcomes": (
            None
            if ingestion.outcomes is None
            else {
                "status_counts": dict(ingestion.outcomes.status_counts),
                "error_classes": list(ingestion.outcomes.error_classes),
                "elements_with_no_error_class": (
                    ingestion.outcomes.elements_with_no_error_class
                ),
                "malformed_snapshots": ingestion.outcomes.malformed_snapshots,
            }
        ),
        "multi_version": [
            {"source_key": item.source_key, "entries": item.entries}
            for item in ingestion.multi_version
        ],
    }


def _pipeline_dict(pipeline: PipelineReport) -> dict[str, Any]:
    if not pipeline.available:
        return {"available": False}
    return {
        "available": True,
        "recent_limit": pipeline.recent_limit,
        "run_versions": pipeline.run_versions,
        "run_ids": pipeline.run_ids,
        "run_ids_with_multiple_versions": (
            pipeline.run_ids_with_multiple_versions
        ),
        "newest_stored_at": (
            None
            if pipeline.newest_stored_at is None
            else _iso(pipeline.newest_stored_at)
        ),
        "age_seconds": pipeline.age_seconds,
        "recent_runs": [
            {
                "run_id": run.run_id,
                "run_version_id": run.run_version_id,
                "stored_at": _iso(run.stored_at),
                "num_articles": run.num_articles,
            }
            for run in pipeline.recent_runs
        ],
        "article_rows_all_versions": pipeline.article_rows_all_versions,
        "article_rows_latest_versions": pipeline.article_rows_latest_versions,
        "topic_rows_all_versions": pipeline.topic_rows_all_versions,
        "topic_rows_latest_versions": pipeline.topic_rows_latest_versions,
        "distinct_topics_latest_versions": (
            pipeline.distinct_topics_latest_versions
        ),
        "topics": [
            {
                "topic_name": topic.topic_name,
                "topic_occurrences_latest_versions": (
                    topic.topic_occurrences_latest_versions
                ),
                "distinct_runs_latest_versions": (
                    topic.distinct_runs_latest_versions
                ),
                "avg_model_confidence_latest_versions": (
                    topic.avg_model_confidence_latest_versions
                ),
                "avg_source_count_latest_versions": (
                    topic.avg_source_count_latest_versions
                ),
            }
            for topic in pipeline.topics
        ],
        "keywords": [
            {
                "keyword": item.keyword,
                "keyword_occurrences_latest_versions": (
                    item.keyword_occurrences_latest_versions
                ),
            }
            for item in pipeline.keywords
        ],
        "keywords_nonarray_rows_latest_versions": (
            pipeline.keywords_nonarray_rows_latest_versions
        ),
        "evidence_rows_all_versions": pipeline.evidence_rows_all_versions,
        "evidence_rows_latest_versions": pipeline.evidence_rows_latest_versions,
        "verified_latest_versions": pipeline.verified_latest_versions,
        "unverified_latest_versions": pipeline.unverified_latest_versions,
        "verification_rate_latest_versions": (
            pipeline.verification_rate_latest_versions
        ),
        "unknown_source_citations_latest_versions": (
            pipeline.unknown_source_citations_latest_versions
        ),
        "tone_rows_all_versions": pipeline.tone_rows_all_versions,
        "tone_rows_latest_versions": pipeline.tone_rows_latest_versions,
        "tone_distribution": [
            {
                "overall": item.overall,
                "tone_occurrences_latest_versions": (
                    item.tone_occurrences_latest_versions
                ),
                "avg_score_latest_versions": item.avg_score_latest_versions,
            }
            for item in pipeline.tone_distribution
        ],
        "min_score_latest_versions": pipeline.min_score_latest_versions,
        "max_score_latest_versions": pipeline.max_score_latest_versions,
        "avg_score_latest_versions": pipeline.avg_score_latest_versions,
        "aspects": [
            {
                "aspect": item.aspect,
                "aspect_occurrences_latest_versions": (
                    item.aspect_occurrences_latest_versions
                ),
            }
            for item in pipeline.aspects
        ],
        "aspects_nonarray_rows_latest_versions": (
            pipeline.aspects_nonarray_rows_latest_versions
        ),
    }


# --------------------------------------------------------------------------
# Text sections
# --------------------------------------------------------------------------
def _ingestion_lines(ingestion: IngestionReport) -> list[str]:
    if not ingestion.available:
        return ["Ingestion store: not initialized (available: false)"]

    lines: list[str] = []
    enabled = sum(1 for source in ingestion.sources if source.enabled)
    lines.append("Ingestion store: available")
    lines.append(f"recent_limit: {ingestion.recent_limit}")
    lines.append(
        f"sources: {len(ingestion.sources)} "
        f"(enabled {enabled}, disabled {len(ingestion.sources) - enabled})"
    )
    if ingestion.sources:
        lines.append(
            "source_key | label | enabled | versions | entries | "
            "latest_entries | first_seen_at | last_seen_at | "
            "age_seconds | stale"
        )
        for source in ingestion.sources:
            lines.append(
                " | ".join(
                    [
                        source.source_key,
                        source.label,
                        str(source.enabled),
                        str(source.versions),
                        str(source.entries),
                        str(source.latest_entries),
                        _iso_or_na(source.first_seen_at),
                        _iso_or_na(source.last_seen_at),
                        _int_or_na(source.age_seconds),
                        _stale_text(source.stale),
                    ]
                )
            )
    else:
        lines.append("sources: none observed")

    totals = ingestion.totals
    assert totals is not None  # available reports always carry totals
    lines.append("")
    lines.append(
        f"sweeps: {totals.sweeps} (complete {totals.complete}, "
        f"partial {totals.partial}, failed {totals.failed}, "
        f"other {totals.other})"
    )
    lines.append(
        f"articles: seen {totals.articles_seen}, stored "
        f"{totals.articles_stored}, replayed {totals.articles_replayed}, "
        f"replay_rate {_rate_text(totals.replay_rate)}"
    )

    outcomes = ingestion.outcomes
    assert outcomes is not None  # available reports always carry outcomes
    statuses = (
        ", ".join(f"{status}={count}" for status, count in outcomes.status_counts)
        or "none"
    )
    lines.append(
        f"source outcomes (recent {ingestion.recent_limit} sweeps): {statuses}"
    )
    lines.append(f"error classes: {', '.join(outcomes.error_classes) or 'none'}")
    lines.append(
        f"elements with no error_class: {outcomes.elements_with_no_error_class}"
    )
    lines.append(
        f"malformed source_results snapshots: {outcomes.malformed_snapshots}"
    )

    lines.append("")
    if ingestion.multi_version:
        lines.append("multi-version entries:")
        for item in ingestion.multi_version:
            lines.append(f"  {item.source_key}: {item.entries}")
    else:
        lines.append("multi-version entries: none")

    lines.append("")
    if ingestion.recent_sweeps:
        lines.append("recent sweeps:")
        lines.append(
            "sweep_id | status | finished_at | attempted | succeeded | "
            "failed | seen | stored | replayed | replay_rate"
        )
        for sweep in ingestion.recent_sweeps:
            lines.append(
                " | ".join(
                    [
                        sweep.sweep_id,
                        sweep.status,
                        _iso(sweep.finished_at),
                        str(sweep.sources_attempted),
                        str(sweep.sources_succeeded),
                        str(sweep.sources_failed),
                        str(sweep.articles_seen),
                        str(sweep.articles_stored),
                        str(sweep.articles_replayed),
                        _rate_text(sweep.replay_rate),
                    ]
                )
            )
    else:
        lines.append("recent sweeps: none")
    return lines


def _pipeline_lines(pipeline: PipelineReport) -> list[str]:
    if not pipeline.available:
        return ["Pipeline store: not initialized (available: false)"]

    lines: list[str] = [
        "Pipeline store: available",
        f"recent_limit: {pipeline.recent_limit}",
        (
            "run versions (all versions): "
            f"{pipeline.run_versions}; run ids (run-level): "
            f"{pipeline.run_ids}; run ids with multiple versions "
            f"(run-level): {pipeline.run_ids_with_multiple_versions}"
        ),
        (
            "newest stored_at (all versions, statement time; advisory): "
            f"{_iso_or_na(pipeline.newest_stored_at)}; age_seconds: "
            f"{_int_or_na(pipeline.age_seconds)}"
        ),
        (
            "articles (all versions): "
            f"{pipeline.article_rows_all_versions}; articles (latest "
            f"versions): {pipeline.article_rows_latest_versions}"
        ),
        (
            "topics (all versions): "
            f"{pipeline.topic_rows_all_versions}; topics (latest versions): "
            f"{pipeline.topic_rows_latest_versions}; distinct topic names "
            f"(latest versions): {pipeline.distinct_topics_latest_versions}"
        ),
    ]

    lines.append("topics (latest versions, top 20 shown):")
    if pipeline.topics:
        lines.append(
            "topic_name | topic_occurrences_latest_versions | "
            "distinct_runs_latest_versions | "
            "avg_model_confidence_latest_versions | "
            "avg_source_count_latest_versions"
        )
        for topic in pipeline.topics:
            lines.append(
                " | ".join(
                    [
                        topic.topic_name,
                        str(topic.topic_occurrences_latest_versions),
                        str(topic.distinct_runs_latest_versions),
                        _number_or_na(
                            topic.avg_model_confidence_latest_versions
                        ),
                        _number_or_na(topic.avg_source_count_latest_versions),
                    ]
                )
            )
    else:
        lines.append("none")

    lines.append("keywords (latest versions, top 20):")
    if pipeline.keywords:
        lines.append("keyword | keyword_occurrences_latest_versions")
        for item in pipeline.keywords:
            lines.append(
                f"{item.keyword} | "
                f"{item.keyword_occurrences_latest_versions}"
            )
    else:
        lines.append("none")
    lines.append(
        "malformed keywords rows (latest versions): "
        f"{pipeline.keywords_nonarray_rows_latest_versions}"
    )

    lines.append(
        "evidence (latest versions): rows "
        f"{pipeline.evidence_rows_latest_versions}, verified "
        f"{pipeline.verified_latest_versions}, unverified "
        f"{pipeline.unverified_latest_versions}, verification_rate "
        f"{_rate_text(pipeline.verification_rate_latest_versions)}, "
        "unknown_source_citations "
        f"{pipeline.unknown_source_citations_latest_versions}"
    )
    lines.append(
        "evidence rows (all versions): "
        f"{pipeline.evidence_rows_all_versions}"
    )

    lines.append(
        "tones (all versions): "
        f"{pipeline.tone_rows_all_versions}; tones (latest versions): "
        f"{pipeline.tone_rows_latest_versions}"
    )
    lines.append("tone distribution (latest versions):")
    if pipeline.tone_distribution:
        lines.append(
            "overall | tone_occurrences_latest_versions | "
            "avg_score_latest_versions"
        )
        for item in pipeline.tone_distribution:
            lines.append(
                " | ".join(
                    [
                        item.overall,
                        str(item.tone_occurrences_latest_versions),
                        _number_or_na(item.avg_score_latest_versions),
                    ]
                )
            )
    else:
        lines.append("none")
    lines.append(
        "tone score range (latest versions): min "
        f"{_number_or_na(pipeline.min_score_latest_versions)}, max "
        f"{_number_or_na(pipeline.max_score_latest_versions)}, avg "
        f"{_number_or_na(pipeline.avg_score_latest_versions)}"
    )

    lines.append("aspects (latest versions, top 20):")
    if pipeline.aspects:
        lines.append("aspect | aspect_occurrences_latest_versions")
        for item in pipeline.aspects:
            lines.append(
                f"{item.aspect} | "
                f"{item.aspect_occurrences_latest_versions}"
            )
    else:
        lines.append("none")
    lines.append(
        "malformed aspects rows (latest versions): "
        f"{pipeline.aspects_nonarray_rows_latest_versions}"
    )

    lines.append("")
    if pipeline.recent_runs:
        lines.append(
            "recent runs (latest version per run_id, statement time):"
        )
        lines.append("run_id | run_version_id | stored_at | num_articles")
        for run in pipeline.recent_runs:
            lines.append(
                " | ".join(
                    [
                        run.run_id,
                        str(run.run_version_id),
                        _iso(run.stored_at),
                        str(run.num_articles),
                    ]
                )
            )
    else:
        lines.append("recent runs: none")
    return lines
