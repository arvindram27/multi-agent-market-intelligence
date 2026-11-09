"""Read-only Postgres access for the SQL analysis layer (landmark I2, M1).

This is the only analysis-facing module that imports psycopg, mirroring
``market_intelligence.db.ingestion`` for the ingestion package: the
``analysis`` package stays psycopg-free and receives an open connection.

Every connection is a **server-enforced read-only session**
(``default_transaction_read_only=on``) with a statement timeout and an
application name. That makes the server reject writes in this session; it is
defense in depth alongside a future read-only DB role, not an immutable
guarantee. This module issues no ``SET``, no DDL, no temporary tables and no
``EXPLAIN ANALYZE``.

``market_intelligence.db`` itself (``db/__init__.py``) stays psycopg-free.
"""
from __future__ import annotations

import re

import psycopg

DatabaseError = psycopg.Error

DEFAULT_STATEMENT_TIMEOUT_MS = 30_000

# Table names are module constants, never user input; this guard keeps a future
# caller from turning ``to_regclass`` into an injection point.
_IDENTIFIER = re.compile(r"[a-z_][a-z0-9_]*\Z")


def open_read_connection(
    dsn: str,
    *,
    connect_timeout: float,
    statement_timeout_ms: int = DEFAULT_STATEMENT_TIMEOUT_MS,
):
    """Open an autocommit session the server enforces as read-only."""
    options = (
        "-c default_transaction_read_only=on "
        f"-c statement_timeout={int(statement_timeout_ms)} "
        "-c application_name=market_intelligence_analysis"
    )
    return psycopg.connect(
        dsn,
        autocommit=True,
        connect_timeout=connect_timeout,
        options=options,
    )


def dsn_secrets(dsn: str) -> list[str]:
    """Secret values that must never reach output: the DSN and its password.

    The password is parsed with libpq's own conninfo parser, which supports
    both the URI form (percent-decoded) and the keyword form. Parsing fails
    closed to the DSN alone; nothing is logged and no file or environment is
    read here.
    """
    secrets = [dsn]
    try:
        password = psycopg.conninfo.conninfo_to_dict(dsn).get("password")
    except psycopg.Error:
        password = None
    if password:
        secrets.append(password)
    return secrets


def table_present(conn, name: str) -> bool:
    """True when ``name`` resolves to an existing relation in this database."""
    if not _IDENTIFIER.fullmatch(name):
        raise ValueError(f"unsafe table name: {name!r}")
    row = conn.execute("SELECT to_regclass(%s) IS NOT NULL", (name,)).fetchone()
    return bool(row and row[0])
