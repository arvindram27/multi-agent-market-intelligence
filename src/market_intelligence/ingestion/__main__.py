"""CLI for one ingestion sweep (landmark I1b).

``python -m market_intelligence.ingestion`` loads the configured feed
sources, collects every feed (bounded network client, no LLM) and then
writes the whole sweep in a single Postgres transaction. Fetching completes
before the transaction starts, so a store failure never leaves the
collection half-done mid-transaction.

Exit codes: 0 stored (or reconciled), 1 failure, 2 configuration problems
(no DSN, invalid source config). DSN/password-shaped fragments are redacted
locally because this package never imports persistence.
"""
from __future__ import annotations

import argparse
import re
import sys

from market_intelligence.config import settings
from market_intelligence.db.ingestion import DatabaseError
from market_intelligence.ingestion.collector import collect
from market_intelligence.ingestion.sources import SourceConfigError, load_sources
from market_intelligence.ingestion.store import (
    IngestionSweepWriter,
    SweepLockedError,
    SweepOutcomeUnknown,
    SweepStoreError,
)

# Mirrors persistence.redact_secrets for this package's messages: explicit
# secrets first, then URI and password= shaped fragments.
_URL_CREDENTIALS_RE = re.compile(
    r"(?i)\b([a-z][a-z0-9+.\-]*://)[^/\s:@'\"<>]+:[^@\s/'\"<>]+@"
)
_PASSWORD_KV_RE = re.compile(
    r"(?i)\bpassword\s*=\s*(?:'[^']*'|\"[^\"]*\"|[^\s;'\"]+)"
)


def _redact_message(text: str, dsn: str | None) -> str:
    cleaned = str(text or "")
    if dsn:
        cleaned = cleaned.replace(dsn, "***")
    cleaned = _URL_CREDENTIALS_RE.sub(r"\1***:***@", cleaned)
    cleaned = _PASSWORD_KV_RE.sub("password=***", cleaned)
    return cleaned


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m market_intelligence.ingestion",
        description="Collect every configured feed and store one sweep.",
    )
    parser.add_argument(
        "--sources",
        default="ingestion_sources.toml",
        help="TOML source list (default: ingestion_sources.toml)",
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help="Postgres DSN (defaults to DATABASE_URL)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    dsn = (args.dsn or settings.database_url.get_secret_value()).strip()
    if not dsn:
        print(
            "No database configured: set DATABASE_URL or pass --dsn.",
            file=sys.stderr,
        )
        return 2

    try:
        sources = load_sources(args.sources)
    except SourceConfigError as exc:
        print(f"invalid source configuration: {exc}", file=sys.stderr)
        return 2

    # Everything is fetched first; only parsing results survive to this line.
    sweep = collect(sources)
    print(
        f"sweep {sweep.status}: "
        f"attempted={sweep.sources_attempted} "
        f"succeeded={sweep.sources_succeeded} "
        f"failed={sweep.sources_failed} "
        f"articles_seen={sweep.articles_seen}"
    )

    writer = IngestionSweepWriter(dsn)
    try:
        result = writer.write_sweep(sweep)
    except SweepLockedError as exc:
        print(_redact_message(f"error: {exc}", dsn), file=sys.stderr)
        return 1
    except SweepStoreError as exc:
        print(_redact_message(f"error: {exc}", dsn), file=sys.stderr)
        return 1
    except SweepOutcomeUnknown as exc:
        print(
            _redact_message(f"warning: {exc}", dsn),
            file=sys.stderr,
        )
        return 1
    except DatabaseError as exc:  # defensive: the writer's contract has no leak
        print(
            _redact_message(f"error: {type(exc).__name__}: {exc}", dsn),
            file=sys.stderr,
        )
        return 1

    stored = "-" if result.articles_stored is None else result.articles_stored
    replayed = "-" if result.articles_replayed is None else result.articles_replayed
    print(
        f"stored: sweep_id={result.sweep_id} outcome={result.outcome} "
        f"stored={stored} replayed={replayed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
