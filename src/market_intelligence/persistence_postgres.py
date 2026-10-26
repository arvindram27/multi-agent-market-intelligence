"""Postgres implementation of the approved-run persistence contract.

One ``persist`` call is one outer transaction: the run header and every child
row either commit together or not at all. The transaction uses an autocommit
connection plus an explicit ``conn.transaction()`` block, so the block is a real
top-level transaction whose exit issues COMMIT. (Opening ``transaction()`` on a
connection that is already inside a transaction would only create a savepoint
and release it; that is not used here.)

Retries are idempotent: the unique ``(run_id, content_hash)`` constraint decides
whether a snapshot already exists. A failed connection around commit is
reconciled against the database before the writer claims success or failure.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable

import psycopg
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

        payload = business_payload(state)
        canonical = canonical_json(payload)
        digest = content_hash(canonical)

        # A failure to connect proves nothing was written: report a known
        # failure instead of an unknown outcome.
        try:
            conn = self._connection_factory()
        except (psycopg.Error, OSError) as exc:
            safe = self._sanitize(exc)
            logger.error("Could not connect for run %s: %s", run_id, safe)
            raise PersistenceError(
                "could not connect to the database; no rows were stored: " + safe
            ) from None

        try:
            self._write(conn, run_id, digest, canonical, state)
        except (psycopg.Error, OSError) as exc:
            safe = self._sanitize(exc)
            logger.error("Postgres write failed for run %s: %s", run_id, safe)
            outcome = self._reconcile(run_id, digest)
            if outcome is True:
                logger.warning(
                    "Run %s was stored even though the client lost the commit "
                    "confirmation; reconciled without rewriting.",
                    run_id,
                )
                return run_id
            if outcome is False:
                raise PersistenceError(
                    "database write failed and no snapshot was stored: " + safe
                ) from None
            raise CommitOutcomeUnknown(
                "the database connection failed around commit and the snapshot "
                "could not be reconciled; the write may or may not have been "
                "stored. Re-approving is safe and deduplicated. " + safe
            ) from None
        finally:
            conn.close()
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
    ) -> None:
        payload = json.loads(canonical)
        with conn.transaction():
            # Bound a stuck database without affecting the session default.
            conn.execute(
                f"SET LOCAL statement_timeout = {self._statement_timeout_ms}"
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
                # Identical snapshot already committed. Nothing to write.
                return
            version_id = row[0]
            self._insert_articles(conn, version_id, state)
            self._insert_topics(conn, version_id, state)
            self._insert_tones(conn, version_id, state)
            self._insert_competitor(conn, version_id, state)

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
                        # Only verified citations resolve to a captured article;
                        # unresolved citations are preserved with a NULL match.
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

    def _snapshot_exists(self, run_id: str, digest: str) -> bool:
        conn = self._connection_factory()
        try:
            row = conn.execute(
                "SELECT 1 FROM runs WHERE run_id = %s AND content_hash = %s",
                (run_id, digest),
            ).fetchone()
            return row is not None
        finally:
            conn.close()

    def _reconcile(self, run_id: str, digest: str) -> bool | None:
        """True = committed, False = verified absent, None = cannot tell."""
        try:
            return self._snapshot_exists(run_id, digest)
        except Exception as exc:
            logger.error(
                "Could not reconcile run %s after a failed write: %s",
                run_id,
                self._sanitize(exc),
            )
            return None
