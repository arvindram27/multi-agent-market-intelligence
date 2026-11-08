"""Raw feed ingestion core (I1).

Feed configuration, bounded collection and identity computation. The storage
layer (one transaction per sweep), the CLI and the Postgres tests are
landmark I1b.

This package must never import the agents graph, the LLM stack, persistence
or the database: it is fetch-and-parse code plus identity hashing. The AST
boundary test in ``tests/test_ingestion_collector.py`` proves it.
"""
from market_intelligence.ingestion.collector import (
    CollectedArticle,
    EmptyFeedError,
    MalformedFeedError,
    SourceResult,
    SweepResult,
    collect,
)
from market_intelligence.ingestion.sources import (
    FeedSource,
    SourceConfigError,
    load_sources,
    validate_sources,
)

# One fixed, application-wide advisory key: sweeps are serialized
# database-wide with pg_try_advisory_xact_lock (transaction-scoped, acquired
# before any write, fail-fast on contention; released automatically by commit
# or rollback, so there are no session leaks and no manual unlock).
# 0x4D414D490001 == ASCII "MAMI" followed by version 1.
INGEST_SWEEP_LOCK_KEY = 84_942_864_842_753

__all__ = [
    "CollectedArticle",
    "EmptyFeedError",
    "FeedSource",
    "INGEST_SWEEP_LOCK_KEY",
    "MalformedFeedError",
    "SourceConfigError",
    "SourceResult",
    "SweepResult",
    "collect",
    "load_sources",
    "validate_sources",
]
