"""Initial setup for the local Postgres store (NOT a migration tool).

Applies ``market_intelligence/db/schema.sql`` and then verifies that the live
database matches the shape this code expects. Re-running is safe: missing
objects are created, existing objects are left alone, and an incompatible
existing shape is reported instead of being dropped or rewritten.

Usage:
    python scripts/init_db.py                 # apply + verify DATABASE_URL
    python scripts/init_db.py --check         # verify only
    python scripts/init_db.py --dsn <dsn>     # explicit target (tests, CI)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import psycopg  # noqa: E402

from market_intelligence.config import settings  # noqa: E402
from market_intelligence.db import (  # noqa: E402
    SCHEMA_VERSION,
    SchemaMismatchError,
    require_schema,
    verify_schema,
)
from market_intelligence.persistence import redact_secrets  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", help="Postgres DSN (defaults to DATABASE_URL)")
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the existing schema without applying setup",
    )
    args = parser.parse_args(argv)

    dsn = (args.dsn or settings.database_url.get_secret_value()).strip()
    if not dsn:
        print(
            "No database configured: set DATABASE_URL or pass --dsn.",
            file=sys.stderr,
        )
        return 2

    try:
        with psycopg.connect(dsn, autocommit=True, connect_timeout=5) as conn:
            problems = verify_schema(conn) if args.check else require_schema(conn)
    except SchemaMismatchError as exc:
        print(redact_secrets(str(exc), [dsn]), file=sys.stderr)
        return 1
    except psycopg.Error as exc:
        print(
            "database error: " + redact_secrets(f"{type(exc).__name__}: {exc}", [dsn]),
            file=sys.stderr,
        )
        return 1

    if problems:
        print(
            "database schema is incompatible; no data was changed:\n  "
            + "\n  ".join(problems),
            file=sys.stderr,
        )
        return 1

    action = "verified" if args.check else "applied and verified"
    print(f"schema version {SCHEMA_VERSION} {action}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
