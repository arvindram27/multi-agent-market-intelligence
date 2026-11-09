"""Pure shaping and rendering for the I2 ingestion analysis report.

No psycopg and no connection here: :func:`build_report` turns the raw rows from
:mod:`market_intelligence.analysis.queries` into frozen report objects, and the
same objects render deterministically to text or JSON.

Stale contract (single definition): every source row always carries ``stale``.
It is ``None`` exactly when the source has no observations (``last_seen_at`` is
``None``). Otherwise ``stale = enabled and age_seconds > stale_hours``, with
``age_seconds`` clamped to 0 for future timestamps (the timestamp is still
reported). Replay rates are ``None`` when the denominator is 0.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from market_intelligence.analysis.queries import (
    IngestionRows,
    MultiVersionRow,
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


def _replay_rate(stored: int, replayed: int) -> float | None:
    denominator = stored + replayed
    if denominator <= 0:
        return None
    return round(replayed / denominator, 6)


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


@dataclass(frozen=True)
class AnalysisReport:
    generated_at: datetime
    ingestion: IngestionReport
    report_version: int = REPORT_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Fixed-key-order JSON-ready payload."""
        ingestion = self.ingestion
        payload: dict[str, Any] = {
            "report_version": self.report_version,
            "generated_at": _iso(self.generated_at),
            "ingestion": {
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
                            None
                            if source.first_seen_at is None
                            else _iso(source.first_seen_at)
                        ),
                        "last_seen_at": (
                            None
                            if source.last_seen_at is None
                            else _iso(source.last_seen_at)
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
                        "malformed_snapshots": (
                            ingestion.outcomes.malformed_snapshots
                        ),
                    }
                ),
                "multi_version": [
                    {"source_key": item.source_key, "entries": item.entries}
                    for item in ingestion.multi_version
                ],
            },
        }
        return payload

    def render_text(self) -> str:
        ingestion = self.ingestion
        lines = [
            "Market intelligence - SQL analysis",
            f"report_version: {self.report_version}",
            f"generated_at: {_iso(self.generated_at)}",
            "",
        ]
        if not ingestion.available:
            lines.append("Ingestion store: not initialized (available: false)")
            return "\n".join(lines) + "\n"

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
        lines.append(
            f"error classes: {', '.join(outcomes.error_classes) or 'none'}"
        )
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
        return "\n".join(lines) + "\n"


def render_text(report: AnalysisReport) -> str:
    """Render a report to its deterministic text form."""
    return report.render_text()


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


def build_report(
    rows: IngestionRows | None,
    *,
    generated_at: datetime | None = None,
    stale_after_hours: float = 24,
    recent_limit: int = 20,
) -> AnalysisReport:
    """Shape raw rows into the deterministic ingestion readout.

    ``rows=None`` means none of the ingestion objects exists: the report is
    still produced with ``available: false``. ``generated_at`` is the clock
    seam; when omitted the module-level :func:`utcnow` is used.
    """
    generated = _as_utc(generated_at or utcnow())
    if rows is None:
        return AnalysisReport(
            generated_at=generated,
            ingestion=IngestionReport(
                available=False,
                recent_limit=recent_limit,
                sources=(),
                totals=None,
                recent_sweeps=(),
                outcomes=None,
                multi_version=(),
            ),
        )

    latest = {row.source_key: row.latest_entries for row in rows.latest_entries}
    sources = tuple(
        _source_report(
            row,
            latest.get(row.source_key, 0),
            generated,
            stale_after_hours,
        )
        for row in rows.sources
    )
    multi_version: tuple[MultiVersionReport, ...] = tuple(
        MultiVersionReport(source_key=row.source_key, entries=row.entries)
        for row in rows.multi_version
    )
    return AnalysisReport(
        generated_at=generated,
        ingestion=IngestionReport(
            available=True,
            recent_limit=recent_limit,
            sources=sources,
            totals=_totals_report(rows.totals),
            recent_sweeps=tuple(
                _recent_sweep_report(row) for row in rows.recent_sweeps
            ),
            outcomes=_outcomes_report(rows.outcomes, rows.malformed_snapshots),
            multi_version=multi_version,
        ),
    )
