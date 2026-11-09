"""One-transaction durable writer for collected feed sweeps (landmark I1b).

:func:`IngestionSweepWriter.write_sweep` persists an entire
:class:`~SweepResult` in a single Postgres transaction: run header, source
sync, raw articles and membership rows either all commit together or not at
all. The writer never imports psycopg and never touches the agents graph, the
LLM stack or persistence; all database access goes through
``market_intelligence.db.ingestion``, which the I1a AST boundary test and the
offline boundary test prove.

Write order inside the transaction is fixed: acquire the advisory lock, then
validate every required observation timestamp of the sweep's articles, then
begin the INSERTs (run row, source upserts, article upserts, membership
rows). A timestamp validation failure therefore precedes the first INSERT;
the rollback behind it is verified by the offline mocked test, not asserted
as an unconditional guarantee here.

Outcome classification mirrors ``persistence_postgres``:

* the connection could not be opened -> known failure, nothing written;
* the transaction body failed and the connection is confirmed IDLE ->
  known failure with the rollback named (no COMMIT was issued);
* anything raised while COMMIT was in flight, or an unconfirmed rollback ->
  reconcile on a second connection by ``sweep_id``: a present, terminal run
  row means success (``outcome="reconciled"``); absence means the outcome is
  genuinely unknown and is never reported as "failed".
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from market_intelligence.db.ingestion import (
    DatabaseError,
    Jsonb,
    is_transaction_closed,
    open_connection,
)
from market_intelligence.ingestion import INGEST_SWEEP_LOCK_KEY
from market_intelligence.ingestion.collector import CollectedArticle, SweepResult

_TERMINAL_SWEEP_STATUSES = ("complete", "partial", "failed")

_INSERT_SOURCE = """
INSERT INTO ingest_sources (source_key, kind, url, label, enabled)
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT (source_key) DO UPDATE
SET kind = EXCLUDED.kind,
    url = EXCLUDED.url,
    label = EXCLUDED.label,
    enabled = EXCLUDED.enabled,
    updated_at = clock_timestamp()
RETURNING id
"""

_INSERT_RUN = """
INSERT INTO ingest_runs (
    sweep_id, started_at, finished_at, status, sources_attempted,
    sources_succeeded, sources_failed, articles_seen, articles_stored,
    articles_replayed, source_results
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
RETURNING id
"""

_INSERT_ARTICLE = """
INSERT INTO raw_articles (
    source_id, entry_key, entry_key_hash, version_hash, article_id, title,
    excerpt, content_kind, content_hash, feed_url, source_url, published_at,
    published_at_raw, retrieved_at, retrieved_at_raw, first_seen_at, last_seen_at
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (source_id, entry_key_hash, version_hash) DO NOTHING
RETURNING id
"""

_SELECT_ARTICLE_ID = """
SELECT id FROM raw_articles
WHERE source_id = %s AND entry_key_hash = %s AND version_hash = %s
"""

_ADVANCE_LAST_SEEN = """
UPDATE raw_articles SET last_seen_at = GREATEST(last_seen_at, %s) WHERE id = %s
"""

_INSERT_MEMBERSHIP = """
INSERT INTO ingest_run_articles (run_id, raw_article_id, seen_at)
VALUES (%s, %s, %s)
ON CONFLICT (run_id, raw_article_id) DO NOTHING
"""

_RECORD_RUN_COUNTS = """
UPDATE ingest_runs SET articles_stored = %s, articles_replayed = %s
WHERE id = %s
"""

_RECONCILE_SQL = "SELECT status FROM ingest_runs WHERE sweep_id = %s"


class SweepStoreError(Exception):
    """Known write failure: no commit was issued, so nothing was stored."""


class SweepLockedError(SweepStoreError):
    """Another sweep holds the ingestion advisory lock; nothing was written."""


class SweepOutcomeUnknown(Exception):
    """The connection failed at COMMIT; the write may or may not be stored."""


class _TransactionAborted(Exception):
    """The transaction body failed and psycopg confirmed a rollback."""

    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(detail)


def _optional_timestamp(value: str | None) -> datetime | None:
    """Parse a feed-supplied publication time; absent/garbage stays ``None``.

    ``published_at_raw`` always keeps the verbatim string, so an unparseable
    value is preserved rather than dropped.
    """
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _required_timestamp(value: str | None, field: str) -> datetime:
    """Parse a collector-guaranteed observation time, naming the column only.

    Observation times (``retrieved_at``, ``first_seen_at``, ``last_seen_at``,
    membership ``seen_at``) are produced by the collector itself; a parse
    failure is a writer bug, so it raises and the transaction is rolled back
    rather than storing NULL. The raw timestamp value is deliberately kept
    out of the error text.
    """
    parsed = _optional_timestamp(value)
    if parsed is None:
        raise SweepStoreError(f"{field} is required and could not be parsed")
    return parsed


@dataclass(frozen=True)
class SweepWriteResult:
    """Outcome of one ``write_sweep`` call.

    Counters are ``None`` for a ``reconciled`` outcome: the values computed
    inside the failed transaction cannot be trusted, and only the committed
    run row was confirmed observable.
    """

    sweep_id: str
    status: str
    outcome: str  # "committed" | "reconciled"
    sources_synced: int | None
    articles_seen: int | None
    articles_stored: int | None
    articles_replayed: int | None


class IngestionSweepWriter:
    """Durable writer for one collected sweep. One call, one transaction."""

    def __init__(
        self,
        dsn: str,
        *,
        connect_timeout: float = 5.0,
        statement_timeout_ms: int = 5000,
        connection_factory: Callable[[], object] | None = None,
    ):
        self._dsn = dsn
        self._connect_timeout = connect_timeout
        self._statement_timeout_ms = int(statement_timeout_ms)
        self._connection_factory = connection_factory or (
            lambda: open_connection(self._dsn, connect_timeout=self._connect_timeout)
        )

    # -- public contract ---------------------------------------------------
    def write_sweep(self, sweep: SweepResult) -> SweepWriteResult:
        if sweep.status not in _TERMINAL_SWEEP_STATUSES:
            raise SweepStoreError(f"sweep status {sweep.status!r} is not a "
                                  "storable terminal status")

        # A failure to connect proves nothing was written: no transaction was
        # ever started, so this is a known failure.
        try:
            conn = self._connection_factory()
        except (DatabaseError, OSError) as exc:
            raise SweepStoreError(
                "could not connect to the database; no rows were stored: "
                f"{type(exc).__name__}: {exc}"
            ) from None

        sweep_id = uuid.uuid4().hex
        try:
            summary = self._write(conn, sweep_id, sweep)
        except _TransactionAborted as aborted:
            # psycopg rolled the transaction back and the connection confirmed
            # it holds no open transaction: no COMMIT was issued.
            raise SweepStoreError(
                "the sweep write failed and was rolled back; no rows were "
                "stored: " + aborted.detail
            ) from None
        except (DatabaseError, OSError) as exc:
            # Any failure from COMMIT, or a rollback we could not confirm, is
            # genuinely ambiguous; absence after the fact is not proof.
            reconciled = self._reconcile(sweep_id)
            if reconciled is True:
                return SweepWriteResult(
                    sweep_id=sweep_id,
                    status=sweep.status,
                    outcome="reconciled",
                    sources_synced=None,
                    articles_seen=None,
                    articles_stored=None,
                    articles_replayed=None,
                )
            raise SweepOutcomeUnknown(
                "the connection failed while a commit was in flight and no "
                "terminal run row was visible afterwards. The sweep may or "
                "may not have been stored; re-running the same sweep writes "
                "a new sweep id, so check the database before retrying. "
                f"({type(exc).__name__})"
            ) from None
        finally:
            # Cleanup must never change an established outcome.
            try:
                conn.close()
            except Exception:  # pragma: no cover - defensive
                pass

        return SweepWriteResult(
            sweep_id=sweep_id,
            status=sweep.status,
            outcome="committed",
            **summary,
        )

    # -- internals ---------------------------------------------------------
    def _write(self, conn, sweep_id: str, sweep: SweepResult) -> dict:
        """Run the single write transaction. Returns the committed counters."""
        body_error: BaseException | None = None
        try:
            with conn.transaction():
                try:
                    self._configure_transaction(conn)
                    parsed = self._validate_required_timestamps(sweep)
                    totals = self._insert_all(conn, sweep_id, sweep, parsed)
                except BaseException as exc:
                    body_error = exc
                    raise
        except BaseException as exc:
            if body_error is None:
                # Raised while opening the transaction or by COMMIT itself;
                # whether the server committed is unknown.
                raise
            if exc is not body_error:
                # The rollback itself failed (or the connection broke while
                # rolling back): the transaction may still be open.
                raise
            if not is_transaction_closed(conn):
                raise
            if isinstance(body_error, (DatabaseError, OSError)):
                raise _TransactionAborted(f"{type(body_error).__name__}: "
                                          f"{body_error}") from None
            raise  # application-level error (e.g. validation), already rolled back
        return totals

    def _configure_transaction(self, conn) -> None:
        """Advisory lock first, then the transaction-local statement timeout.

        ``pg_try_advisory_xact_lock`` is transaction-scoped: it is released
        automatically by commit or rollback, so there is no manual unlock and
        no session leak.
        """
        acquired = conn.execute(
            "SELECT pg_try_advisory_xact_lock(%s)", (INGEST_SWEEP_LOCK_KEY,)
        ).fetchone()
        if not acquired or not acquired[0]:
            raise SweepLockedError(
                "another sweep holds the ingestion advisory lock; "
                "nothing was written"
            )
        conn.execute(
            "SELECT set_config('statement_timeout', %s, true)",
            (str(self._statement_timeout_ms),),
        )

    def _validate_required_timestamps(self, sweep: SweepResult):
        """Validate every required observation time before any INSERT.

        Returns per article: ``(article, retrieved_at, first_seen_at,
        last_seen_at, seen_at`` parsed datetimes, all collector-guaranteed.
        """
        parsed = []
        for article in sweep.articles:
            retrieved_at = _required_timestamp(
                article.retrieved_at_raw, "retrieved_at"
            )
            first_seen_at = _required_timestamp(
                article.retrieved_at_raw, "first_seen_at"
            )
            seen_at = _required_timestamp(article.seen_at, "seen_at")
            last_seen_at = _required_timestamp(article.seen_at, "last_seen_at")
            parsed.append(
                (article, retrieved_at, first_seen_at, last_seen_at, seen_at)
            )
        return parsed

    def _insert_all(self, conn, sweep_id: str, sweep: SweepResult, parsed):
        write_time = datetime.now(timezone.utc)
        snapshot = [
            {
                "source_key": result.source_key,
                "kind": result.kind,
                "url": result.url,
                "label": result.label,
                "enabled": result.enabled,
                "status": result.status,
                "error_class": result.error_class,
                "articles_seen": result.articles_seen,
                "articles": len(result.articles),
                "warnings": list(result.warnings),
            }
            for result in sweep.source_results
        ]
        run_row = conn.execute(
            _INSERT_RUN,
            (
                sweep_id,
                write_time,
                write_time,
                sweep.status,
                sweep.sources_attempted,
                sweep.sources_succeeded,
                sweep.sources_failed,
                sweep.articles_seen,
                0,
                0,
                Jsonb(snapshot),
            ),
        ).fetchone()
        run_id = run_row[0]

        source_ids: dict[str, int] = {}
        for result in sweep.source_results:
            row = conn.execute(
                _INSERT_SOURCE,
                (
                    result.source_key,
                    result.kind,
                    result.url,
                    result.label,
                    result.enabled,
                ),
            ).fetchone()
            source_ids[result.source_key] = row[0]

        stored = 0
        replayed = 0
        for (article, retrieved_at, first_seen_at, last_seen_at, seen_at) in parsed:
            published_at = _optional_timestamp(article.published_at_raw)
            inserted = conn.execute(
                _INSERT_ARTICLE,
                (
                    source_ids[article.source_key],
                    article.entry_key,
                    article.entry_key_hash,
                    article.version_hash,
                    article.article_id,
                    article.title,
                    article.excerpt,
                    article.content_kind,
                    article.content_hash,
                    article.feed_url,
                    article.source_url,
                    published_at,
                    article.published_at_raw,
                    retrieved_at,
                    article.retrieved_at_raw,
                    first_seen_at,
                    last_seen_at,
                ),
            ).fetchone()
            if inserted is not None:
                raw_article_id = inserted[0]
                stored += 1
            else:
                # Same (source_id, entry_key_hash, version_hash) already
                # committed: keep the single version row and advance its
                # last observation time to this sweep's seen_at.
                existing = conn.execute(
                    _SELECT_ARTICLE_ID,
                    (
                        source_ids[article.source_key],
                        article.entry_key_hash,
                        article.version_hash,
                    ),
                ).fetchone()
                raw_article_id = existing[0]
                replayed += 1
                conn.execute(_ADVANCE_LAST_SEEN, (last_seen_at, raw_article_id))
            conn.execute(
                _INSERT_MEMBERSHIP, (run_id, raw_article_id, seen_at)
            )

        conn.execute(_RECORD_RUN_COUNTS, (stored, replayed, run_id))
        return {
            "sources_synced": len(sweep.source_results),
            "articles_seen": sweep.articles_seen,
            "articles_stored": stored,
            "articles_replayed": replayed,
        }

    def _reconcile(self, sweep_id: str) -> bool | None:
        """True only for a present, terminal run row; otherwise ``None``.

        ``None`` means "not observed", which is *not* the same as "not
        stored": the original transaction may still commit. The caller must
        keep the outcome unknown.
        """
        try:
            conn = self._connection_factory()
        except Exception:
            return None
        try:
            row = conn.execute(_RECONCILE_SQL, (sweep_id,)).fetchone()
        except Exception:
            return None
        finally:
            try:
                conn.close()
            except Exception:  # pragma: no cover - defensive
                pass
        if row is None:
            return None
        return row[0] in _TERMINAL_SWEEP_STATUSES


def sweep_snapshot_json(snapshot: object) -> str:
    """Deterministic JSON for the per-sweep source snapshot (test/debug aid)."""
    return json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
