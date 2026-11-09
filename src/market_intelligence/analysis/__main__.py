"""Command-line readout of the durable ingestion store (landmark I2, M1).

Usage:
    python -m market_intelligence.analysis [--dsn DSN] [--format text|json]
        [--stale-hours 24] [--recent-sweeps 20]

The DSN comes from ``--dsn`` or ``DATABASE_URL`` and is never rewritten or
printed. Each connection is a server-enforced read-only session (see
:mod:`market_intelligence.db.analytics`). Exit codes: 0 report produced
(including an empty database), 2 no DSN or invalid threshold arguments,
1 connection error or an incompatible present schema (nothing is modified).
"""
from __future__ import annotations

import argparse
import json
import math
import sys

from market_intelligence.analysis.queries import INGEST_OBJECTS, load_ingestion
from market_intelligence.analysis.report import build_report
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


def _recent_sweeps(value: str) -> int:
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
        description="Read-only SQL analysis of the durable ingestion store.",
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
        "--stale-hours",
        type=_stale_hours,
        default=24.0,
        help="Source freshness threshold in hours (default: 24).",
    )
    parser.add_argument(
        "--recent-sweeps",
        type=_recent_sweeps,
        default=20,
        help="How many recent sweeps to detail, 1-100 (default: 20).",
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

    try:
        with open_read_connection(
            dsn, connect_timeout=_resolve_connect_timeout()
        ) as conn:
            present = [table_present(conn, name) for name in INGEST_OBJECTS]
            if not any(present):
                report = build_report(
                    None,
                    stale_after_hours=args.stale_hours,
                    recent_limit=args.recent_sweeps,
                )
            else:
                problems = verify_ingestion_schema(conn)
                if problems:
                    print(
                        "ingestion schema is incompatible; no data was changed:\n  "
                        + "\n  ".join(problems),
                        file=sys.stderr,
                    )
                    return 1
                rows = load_ingestion(conn, recent_limit=args.recent_sweeps)
                report = build_report(
                    rows,
                    stale_after_hours=args.stale_hours,
                    recent_limit=args.recent_sweeps,
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
