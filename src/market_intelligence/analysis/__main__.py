"""Command-line readout of the durable analysis stores (landmark I2).

Usage:
    python -m market_intelligence.analysis [--dsn DSN] [--format text|json]
        [--dataset ingestion|pipeline|both] [--stale-hours 24]
        [--recent-sweeps 20] [--recent-runs 20]

``--dataset`` defaults to ``ingestion`` (M1 behaviour, byte-identical output;
the pipeline store is not probed). ``pipeline`` and ``both`` add the M2
pipeline section. The DSN comes from ``--dsn`` or ``DATABASE_URL`` and is
never rewritten or printed. Each connection is a server-enforced read-only
session (see :mod:`market_intelligence.db.analytics`). Exit codes: 0 report
produced (any mix of absent/clean stores), 2 no DSN or invalid arguments,
1 connection error or an incompatible present schema (nothing is modified).
"""
from __future__ import annotations

import argparse
import json
import math
import sys

from market_intelligence.analysis.queries import (
    INGEST_OBJECTS,
    PIPELINE_OBJECTS,
    load_ingestion,
    load_pipeline,
)
from market_intelligence.analysis.report import build_analysis_report
from market_intelligence.db import verify_schema
from market_intelligence.db.analytics import (
    DatabaseError,
    dsn_secrets,
    open_read_connection,
    table_present,
)
from market_intelligence.db.ingestion import verify_ingestion_schema
from market_intelligence.persistence import redact_secrets


def _stale_hours(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"must be a finite number greater than 0, got {value!r}"
        ) from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError(
            f"must be a finite number greater than 0, got {value!r}"
        )
    return parsed


def _bounded_count(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"must be an integer between 1 and 100, got {value!r}"
        ) from exc
    if not 1 <= parsed <= 100:
        raise argparse.ArgumentTypeError(
            f"must be an integer between 1 and 100, got {value!r}"
        )
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m market_intelligence.analysis",
        description="Read-only SQL analysis of the durable analysis stores.",
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help="Postgres DSN (defaults to DATABASE_URL; never printed).",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format (default: text).",
    )
    parser.add_argument(
        "--dataset",
        choices=("ingestion", "pipeline", "both"),
        default="ingestion",
        help="Which store(s) to read (default: ingestion; pipeline is not "
        "probed on the default path).",
    )
    parser.add_argument(
        "--stale-hours",
        type=_stale_hours,
        default=24.0,
        help="Source freshness threshold in hours (ingestion only; "
        "default: 24).",
    )
    parser.add_argument(
        "--recent-sweeps",
        type=_bounded_count,
        default=20,
        help="How many recent sweeps to detail, 1-100 (ingestion only; "
        "default: 20).",
    )
    parser.add_argument(
        "--recent-runs",
        type=_bounded_count,
        default=20,
        help="How many recent pipeline runs to detail, 1-100 "
        "(default: 20).",
    )
    return parser


def _resolve_dsn(explicit: str | None) -> str:
    if explicit is not None:
        return explicit.strip()
    # Imported lazily so importing this module never loads configuration.
    from market_intelligence.config import settings

    return settings.database_url.get_secret_value().strip()


def _resolve_connect_timeout() -> float:
    from market_intelligence.config import settings

    return settings.db_connect_timeout


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dsn = _resolve_dsn(args.dsn)
    if not dsn:
        print(
            "No database configured: set DATABASE_URL or pass --dsn.",
            file=sys.stderr,
        )
        return 2

    include_ingestion = args.dataset in ("ingestion", "both")
    include_pipeline = args.dataset in ("pipeline", "both")

    try:
        with open_read_connection(
            dsn, connect_timeout=_resolve_connect_timeout()
        ) as conn:
            ingestion_rows = None
            if include_ingestion:
                present = [
                    table_present(conn, name) for name in INGEST_OBJECTS
                ]
                if any(present):
                    problems = verify_ingestion_schema(conn)
                    if problems:
                        print(
                            "ingestion schema is incompatible; no data was "
                            "changed:\n  " + "\n  ".join(problems),
                            file=sys.stderr,
                        )
                        return 1
                    ingestion_rows = load_ingestion(
                        conn, recent_limit=args.recent_sweeps
                    )

            pipeline_rows = None
            if include_pipeline:
                present = [
                    table_present(conn, name) for name in PIPELINE_OBJECTS
                ]
                if any(present):
                    problems = verify_schema(conn)
                    if problems:
                        print(
                            "pipeline schema is incompatible; no data was "
                            "changed:\n  " + "\n  ".join(problems),
                            file=sys.stderr,
                        )
                        return 1
                    pipeline_rows = load_pipeline(
                        conn, recent_limit=args.recent_runs
                    )

            report = build_analysis_report(
                ingestion_rows,
                pipeline_rows,
                include_ingestion=include_ingestion,
                include_pipeline=include_pipeline,
                stale_after_hours=args.stale_hours,
                recent_limit=args.recent_sweeps,
                recent_runs=args.recent_runs,
            )
    except (DatabaseError, OSError) as exc:
        print(
            redact_secrets(f"{type(exc).__name__}: {exc}", dsn_secrets(dsn)),
            file=sys.stderr,
        )
        return 1

    if args.format == "json":
        print(json.dumps(report.to_dict(), indent=2))
    else:
        sys.stdout.write(report.render_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
