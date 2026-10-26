"""Postgres implementation of the approved-run persistence contract.

One ``persist`` call is one outer transaction: the run header and every child
row either commit together or not at all. The transaction uses an autocommit
connection plus an explicit ``conn.transaction()`` block, so the block is a real
top-level transaction whose exit issues COMMIT. (Opening ``transaction()`` on a
connection that is already inside a transaction would only create a savepoint
and release it; that is not used here.)

Failure classification follows the transaction phase that was actually reached:

* the connection could not be opened -> no transaction started, so nothing was
  written (known failure);
* the transaction body failed and psycopg rolled it back, and the connection
  reports no open transaction afterwards -> no commit was issued (known
  failure, with the rollback named in the message);
* anything raised while COMMIT itself was in flight -> the outcome is unknown.
  A ``SELECT`` that finds no row on another connection is *not* proof that the
  original transaction cannot still commit; it only means the row was not
  visible at that instant. Absence therefore stays ``persist_unknown``.

Re-persisting the *unchanged* snapshot (same ``run_id``, same ``content_hash``)
is deduplicated by the unique constraint. Changed business content is stored as
a new version. The writer never mutates ``state`` and never regenerates
provenance values or the hash, so a retry reuses the frozen snapshot.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable

import psycopg
from psycopg.pq import TransactionStatus
from psycopg.types.json import Jsonb

from market_intelligence.db import SCHEMA_VERSION
from market_intelligence.persistence import (
    CommitOutcomeUnknown,
    PersistenceError,
    UnapprovedRunError,
    redact_secrets,
)

if TYPE_CHECKING:
    from market_intelligence.graph.state import MarketIntelligenceState

logger = logging.getLogger(__name__)

# Prefixing the hash keeps a future payload-version change from colliding with
# snapshots written by this version.
CANONICAL_HASH_PREFIX = "mi-run-v1\n"

_INSERT_RUN = """
INSERT INTO runs (
    run_id, content_hash, schema_version, rss_url, competitor_url,
    num_articles, warnings, payload
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (run_id, content_hash) DO NOTHING
RETURNING id
"""

_INSERT_ARTICLE = """
INSERT INTO run_articles (
    run_version, article_id, ordinal, headline, summary, key_facts, tone_hint,
    feed_url, source_title, source_url, published_at, published_at_raw,
    retrieved_at, retrieved_at_raw, content_hash, excerpt, content_kind
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

_INSERT_TOPIC = """
INSERT INTO run_topics (
    run_version, ordinal, topic_name, keywords, model_confidence,
    window_start, window_end, source_count
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
"""

_INSERT_EVIDENCE = """
INSERT INTO run_topic_evidence (
    run_version, topic_ordinal, article_id, matched_article_id, quote,
    source_title, source_url, published_at, published_at_raw, verified
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

_INSERT_TONE = """
INSERT INTO run_tones (
    run_version, article_id, headline, overall, score, aspects, source_url,
    published_at, published_at_raw, retrieved_at, retrieved_at_raw
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

_INSERT_COMPETITOR = """
INSERT INTO competitor_profiles (
    run_version, company, positioning, products, pricing_model, funding_stage,
    strengths, weaknesses, sources, retrieved_sources, retrieved_at
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""


def canonical_json(payload: dict[str, Any]) -> str:
    """Deterministic JSON used for the snapshot identity.

    Sorted keys, no insignificant whitespace, UTF-8 by default and no NaN/inf:
    two equal business payloads always produce the same bytes.
    """
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def content_hash(canonical: str) -> str:
    """SHA-256 of the version prefix plus the canonical payload."""
    return hashlib.sha256(
        (CANONICAL_HASH_PREFIX + canonical).encode("utf-8")
    ).hexdigest()


def business_payload(state: "MarketIntelligenceState") -> dict[str, Any]:
    """The exact business content that defines snapshot identity.

    Graph bookkeeping (``status``, ``error``, ``approved``, ``persisted_run_id``)
    is deliberately excluded: a retry after success must hash the same.
    """
    return {
        "rss_url": state.rss_url,
        "competitor_url": state.competitor_url,
        "num_articles": state.num_articles,
        "warnings": list(state.warnings),
        "summaries": [item.model_dump(mode="json") for item in state.summaries],
        "topics": [item.model_dump(mode="json") for item in state.topics],
        "tones": [item.model_dump(mode="json") for item in state.tones],
        "competitor": (
            state.competitor.model_dump(mode="json") if state.competitor else None
        ),
    }


def _as_timestamp(value: str | None) -> datetime | None:
    """Parse an ISO timestamp, keeping microsecond precision.

    Returns ``None`` when the source value is absent or not parseable. Callers
    also store the raw string, so an unparseable value is preserved rather than
    dropped.
    """
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _dsn_secrets(dsn: str) -> tuple[str, ...]:
    """The full DSN and its password, for explicit redaction."""
    secrets = [dsn]
    try:
        info = psycopg.conninfo.conninfo_to_dict(dsn)
    except Exception:
        return tuple(secrets)
    password = info.get("password")
    if password:
        secrets.append(password)
    return tuple(secrets)


class _TransactionAborted(Exception):
    """The transaction body failed and psycopg confirmed a rollback."""

    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(detail)


def _transaction_is_closed(conn) -> bool:
    """True when the connection is healthy and holds no open transaction.

    Only used after the transaction block has unwound from a body failure: a
    healthy, idle connection means psycopg issued ROLLBACK and no COMMIT was
    attempted. A broken connection or a still-open transaction is ambiguous.
    """
    try:
        return (
            not conn.closed
            and not conn.broken
            and conn.info.transaction_status == TransactionStatus.IDLE
        )
    except Exception:  # pragma: no cover - defensive for exotic connections
        return False


def _check_invariants(state: "MarketIntelligenceState") -> None:
    """Fail closed when a state was built outside the graph and is inconsistent.

    The pipeline guarantees two evidence invariants:

    * ``TopicEvidence.verified`` is set only when the topic agent resolved the
      citation to a captured article and matched the quote verbatim, so a
      verified citation must name an article in this snapshot;
    * every ``NewsTone`` was assessed from a captured summary, so its
      ``article_id`` must name an article in this snapshot.

    Foreign keys enforce the same rules in the database. This check exists so an
    externally constructed state fails with a clear message before any write,
    and it never relaxes the constraints or drops evidence.
    """
    article_ids = {item.article_id for item in state.summaries}

    for tone in state.tones:
        if tone.article_id not in article_ids:
            raise PersistenceError(
                "evidence integrity invariant violated: a tone references "
                f"article {tone.article_id!r}, which is not in this snapshot; "
                "refusing to persist"
            )

    for topic in state.topics:
        for evidence in topic.evidence:
            if evidence.verified and evidence.article_id not in article_ids:
                raise PersistenceError(
                    "evidence integrity invariant violated: verified topic "
                    f"evidence references article {evidence.article_id!r}, "
                    "which is not in this snapshot; refusing to persist"
                )


class PostgresProfileWriter:
    """Durable writer for approved runs. One call, one committed transaction."""

    def __init__(
        self,
        dsn: str,
        *,
        connect_timeout: float = 5.0,
        statement_timeout_ms: int = 5000,
        connection_factory: Callable[[], Any] | None = None,
    ):
        self._dsn = dsn
        self._connect_timeout = connect_timeout
        self._statement_timeout_ms = int(statement_timeout_ms)
        self._secrets = _dsn_secrets(dsn)
        self._connection_factory = connection_factory or self._connect

    # -- public contract ---------------------------------------------------
    def persist(self, run_id: str, state: "MarketIntelligenceState") -> str:
        if state.approved is not True:
            # Defense in depth: the graph should never route an unapproved run
            # here, but the writer refuses on its own too.
            raise UnapprovedRunError(
                "refusing to persist a run that was not explicitly approved"
            )
        _check_invariants(state)

        payload = business_payload(state)
        canonical = canonical_json(payload)
        digest = content_hash(canonical)

        # A failure to connect proves nothing was written: no transaction was
        # ever started, so this is a known failure.
        try:
            conn = self._connection_factory()
        except (psycopg.Error, OSError) as exc:
            safe = self._sanitize(exc)
            logger.error("Could not connect for run %s: %s", run_id, safe)
            raise PersistenceError(
                "could not connect to the database; no rows were stored: " + safe
            ) from None

        try:
            outcome = self._write(conn, run_id, digest, canonical, state)
        except _TransactionAborted as aborted:
            # psycopg rolled the transaction back and the connection confirmed
            # it holds no open transaction: no COMMIT was issued.
            logger.error(
                "Write for run %s was rolled back: %s", run_id, aborted.detail
            )
            raise PersistenceError(
                "the database write failed and was rolled back; "
                "no snapshot was stored: " + aborted.detail
            ) from None
        except (psycopg.Error, OSError) as exc:
            # Any failure from COMMIT, or a rollback we could not confirm, is
            # genuinely ambiguous. Absence is not proof of a rollback.
            safe = self._sanitize(exc)
            logger.error(
                "Postgres write for run %s has an ambiguous commit outcome: %s",
                run_id,
                safe,
            )
            reconciled = self._reconcile(run_id, digest, canonical)
            if reconciled is True:
                logger.warning(
                    "Run %s: a matching snapshot is present; reconciled to "
                    "success without rewriting.",
                    run_id,
                )
                return run_id
            raise CommitOutcomeUnknown(
                "the connection failed while a commit was in flight and no "
                "matching snapshot was visible afterwards. The write may or may "
                "not have been stored. Re-persisting the unchanged snapshot "
                "(same run id and content hash) is safe and deduplicated; "
                "changed content is stored as a new version. " + safe
            ) from None
        finally:
            # Cleanup must never change an outcome that was already established.
            try:
                conn.close()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "Could not close the connection for run %s: %s",
                    run_id,
                    self._sanitize(exc),
                )

        logger.info(
            "Approved run %s: %s snapshot %s", run_id, outcome, digest[:12]
        )
        return run_id

    # -- internals ---------------------------------------------------------
    def _connect(self):
        return psycopg.connect(
            self._dsn,
            connect_timeout=self._connect_timeout,
            autocommit=True,
        )

    def _sanitize(self, value: object) -> str:
        return redact_secrets(str(value), self._secrets)

    def _write(
        self,
        conn,
        run_id: str,
        digest: str,
        canonical: str,
        state: "MarketIntelligenceState",
    ) -> str:
        """Write one snapshot. Returns ``"inserted"`` or ``"existing"``.

        Failures are classified by the phase that raised: an error from the
        transaction body with a confirmed rollback becomes
        :class:`_TransactionAborted`; an error from COMMIT (or from a rollback
        that could not be confirmed) propagates as ambiguous.
        """
        payload = json.loads(canonical)
        body_error: BaseException | None = None
        outcome = "inserted"
        try:
            with conn.transaction():
                try:
                    # Bound a stuck database without affecting the session
                    # default.
                    conn.execute(
                        f"SET LOCAL statement_timeout = "
                        f"{self._statement_timeout_ms}"
                    )
                    row = conn.execute(
                        _INSERT_RUN,
                        (
                            run_id,
                            digest,
                            SCHEMA_VERSION,
                            state.rss_url,
                            state.competitor_url,
                            state.num_articles,
                            Jsonb(list(state.warnings)),
                            Jsonb(payload),
                        ),
                    ).fetchone()
                    if row is None:
                        # Identical snapshot already committed: write nothing.
                        outcome = "existing"
                        return outcome
                    version_id = row[0]
                    self._insert_articles(conn, version_id, state)
                    self._insert_topics(conn, version_id, state)
                    self._insert_tones(conn, version_id, state)
                    self._insert_competitor(conn, version_id, state)
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
            if not _transaction_is_closed(conn):
                raise
            if isinstance(body_error, (psycopg.Error, OSError)):
                raise _TransactionAborted(self._sanitize(body_error)) from None
            raise
        return outcome

    def _insert_articles(self, conn, version_id: int, state) -> None:
        for ordinal, item in enumerate(state.summaries):
            conn.execute(
                _INSERT_ARTICLE,
                (
                    version_id,
                    item.article_id,
                    ordinal,
                    item.headline,
                    item.summary,
                    Jsonb(list(item.key_facts)),
                    item.tone_hint,
                    item.feed_url,
                    item.source_title,
                    item.source_url,
                    _as_timestamp(item.published_at),
                    item.published_at,
                    _as_timestamp(item.retrieved_at),
                    item.retrieved_at,
                    item.content_hash,
                    item.excerpt,
                    item.content_kind,
                ),
            )

    def _insert_topics(self, conn, version_id: int, state) -> None:
        for ordinal, topic in enumerate(state.topics):
            conn.execute(
                _INSERT_TOPIC,
                (
                    version_id,
                    ordinal,
                    topic.topic_name,
                    Jsonb(list(topic.keywords)),
                    topic.model_confidence,
                    topic.window_start,
                    topic.window_end,
                    topic.source_count,
                ),
            )
            for evidence in topic.evidence:
                conn.execute(
                    _INSERT_EVIDENCE,
                    (
                        version_id,
                        ordinal,
                        evidence.article_id,
                        # Evidence invariant: the topic agent only marks a
                        # citation verified after resolving it to a captured
                        # article, so only those carry the composite foreign
                        # key. Unresolved citations are preserved with a NULL
                        # match and verified = false.
                        evidence.article_id if evidence.verified else None,
                        evidence.quote,
                        evidence.source_title,
                        evidence.source_url,
                        _as_timestamp(evidence.published_at),
                        evidence.published_at,
                        evidence.verified,
                    ),
                )

    def _insert_tones(self, conn, version_id: int, state) -> None:
        for tone in state.tones:
            conn.execute(
                _INSERT_TONE,
                (
                    version_id,
                    tone.article_id,
                    tone.headline,
                    tone.overall,
                    tone.score,
                    Jsonb([aspect.model_dump(mode="json") for aspect in tone.aspects]),
                    tone.source_url,
                    _as_timestamp(tone.published_at),
                    tone.published_at,
                    _as_timestamp(tone.retrieved_at),
                    tone.retrieved_at,
                ),
            )

    def _insert_competitor(self, conn, version_id: int, state) -> None:
        profile = state.competitor
        if profile is None:
            return
        retrieved_at = None
        for source in profile.retrieved_sources:
            retrieved_at = _as_timestamp(source.retrieved_at)
            if retrieved_at is not None:
                break
        conn.execute(
            _INSERT_COMPETITOR,
            (
                version_id,
                profile.company,
                profile.positioning,
                Jsonb(list(profile.products)),
                profile.pricing_model,
                profile.funding_stage,
                Jsonb(list(profile.strengths)),
                Jsonb(list(profile.weaknesses)),
                Jsonb(list(profile.sources)),
                Jsonb(
                    [
                        source.model_dump(mode="json")
                        for source in profile.retrieved_sources
                    ]
                ),
                retrieved_at,
            ),
        )

    def _reconcile(
        self, run_id: str, digest: str, canonical: str
    ) -> bool | None:
        """True only for a present, matching snapshot; otherwise ``None``.

        ``None`` means "not observed", which is *not* the same as "not stored":
        the original transaction may still commit. The caller must keep the
        outcome unknown.
        """
        try:
            conn = self._connection_factory()
        except Exception as exc:
            logger.error(
                "Could not reconnect to reconcile run %s: %s",
                run_id,
                self._sanitize(exc),
            )
            return None

        try:
            row = conn.execute(
                "SELECT payload FROM runs "
                "WHERE run_id = %s AND content_hash = %s",
                (run_id, digest),
            ).fetchone()
        except Exception as exc:
            logger.error(
                "Reconciliation query failed for run %s: %s",
                run_id,
                self._sanitize(exc),
            )
            return None
        finally:
            # A failed cleanup must not mask "not observed" as success.
            try:
                conn.close()
            except Exception:  # pragma: no cover - defensive
                pass

        if row is None:
            logger.info(
                "Reconciliation for run %s found no snapshot yet; the commit "
                "outcome remains unknown.",
                run_id,
            )
            return None

        stored = row[0]
        if isinstance(stored, str):
            stored = json.loads(stored)
        try:
            matches = canonical_json(stored) == canonical
        except (TypeError, ValueError):
            matches = False
        if not matches:
            logger.error(
                "Reconciliation found run %s with a different payload; the "
                "commit outcome remains unknown.",
                run_id,
            )
            return None
        return True
