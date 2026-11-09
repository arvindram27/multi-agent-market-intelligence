-- Initial ingestion database setup for raw feed sweeps (landmark I1b).
--
-- Generated from the reviewed I1 plan DDL with only the pinned corrections
-- applied (hashed-prefix entry_key CHECK `^(guid(-sha256)?|link(-sha256)?|sha256):`
-- plus length("entry_key") <= 2048, "retrieved_at" NOT NULL, the reviewed
-- three-index set, "ingest_run_articles.raw_article_id" with both foreign
-- keys ON DELETE RESTRICT, and the guarded "latest_articles" view ordered by
-- "last_seen_at DESC"). This is NOT a migration system: "IF NOT EXISTS"
-- everywhere, no DROP, no ALTER and no CREATE OR REPLACE. Re-running is safe
-- and never alters existing objects; db.verify_ingestion_schema judges an
-- existing shape and the init script refuses on mismatch rather than
-- dropping or rewriting anything.
--
-- Identity: one raw_articles row per (source_id, entry_key_hash,
-- version_hash). Distinct content versions stay as separate rows; the same
-- content re-seen in later sweeps only advances last_seen_at. "latest_articles"
-- is defined on last observation time so an A -> B -> A reversion returns A.

CREATE TABLE IF NOT EXISTS ingest_schema_meta (
    singleton      boolean     PRIMARY KEY DEFAULT true CHECK (singleton),
    schema_version integer     NOT NULL,
    applied_at     timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ingest_sources (
    id         bigserial   PRIMARY KEY,
    source_key text        NOT NULL,
    kind       text        NOT NULL CHECK (kind IN ('rss', 'atom')),
    url        text        NOT NULL,
    label      text        NOT NULL,
    enabled    boolean     NOT NULL,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT ingest_sources_key UNIQUE (source_key)
);

CREATE TABLE IF NOT EXISTS ingest_runs (
    id                bigserial   PRIMARY KEY,
    sweep_id          text        NOT NULL,
    started_at        timestamptz NOT NULL,
    finished_at       timestamptz NOT NULL,
    status            text        NOT NULL
        CHECK (status IN ('complete', 'partial', 'failed')),
    sources_attempted integer     NOT NULL CHECK (sources_attempted >= 0),
    sources_succeeded integer     NOT NULL CHECK (sources_succeeded >= 0),
    sources_failed    integer     NOT NULL CHECK (sources_failed >= 0),
    articles_seen     integer     NOT NULL CHECK (articles_seen >= 0),
    -- articles_stored counts raw_articles rows created by this sweep;
    -- articles_replayed counts existing version rows this sweep re-saw.
    articles_stored   integer     NOT NULL CHECK (articles_stored >= 0),
    articles_replayed integer     NOT NULL CHECK (articles_replayed >= 0),
    -- Explicit per-sweep snapshot of the whole source config, including
    -- disabled and failed sources. Metadata sync is future-only: later
    -- sweeps update ingest_sources, never this historical snapshot.
    source_results    jsonb       NOT NULL,
    CONSTRAINT ingest_runs_sweep UNIQUE (sweep_id)
);

CREATE TABLE IF NOT EXISTS raw_articles (
    id               bigserial   PRIMARY KEY,
    source_id        bigint      NOT NULL
        REFERENCES ingest_sources(id) ON DELETE RESTRICT,
    entry_key        text        NOT NULL
        CHECK (entry_key ~ '^(guid(-sha256)?|link(-sha256)?|sha256):'),
    entry_key_hash   text        NOT NULL CHECK (length(entry_key_hash) = 64),
    version_hash     text        NOT NULL CHECK (length(version_hash) = 64),
    article_id       text        NOT NULL CHECK (length(article_id) = 16),
    title            text        NOT NULL,
    excerpt          text        NOT NULL,
    content_kind     text        NOT NULL,
    content_hash     text        NOT NULL CHECK (length(content_hash) = 64),
    feed_url         text        NOT NULL,
    source_url       text,
    -- Optional publication time: feed-supplied, may be absent or unparseable,
    -- and the verbatim raw value is always kept beside the typed parse.
    published_at     timestamptz,
    published_at_raw text,
    -- Observation times are collector-guaranteed; they are required values
    -- written by the sweep writer (never clock defaults) and must parse.
    retrieved_at     timestamptz NOT NULL,
    retrieved_at_raw text        NOT NULL,
    first_seen_at    timestamptz NOT NULL,
    last_seen_at     timestamptz NOT NULL,
    CONSTRAINT raw_articles_entry_key_length
        CHECK (length(entry_key) <= 2048),
    CONSTRAINT raw_articles_identity
        UNIQUE (source_id, entry_key_hash, version_hash)
);

CREATE INDEX IF NOT EXISTS raw_articles_by_entry
    ON raw_articles (source_id, entry_key_hash);
CREATE INDEX IF NOT EXISTS raw_articles_by_article_id
    ON raw_articles (source_id, article_id);
CREATE INDEX IF NOT EXISTS raw_articles_by_published
    ON raw_articles (published_at DESC);

CREATE TABLE IF NOT EXISTS ingest_run_articles (
    run_id         bigint      NOT NULL
        REFERENCES ingest_runs(id) ON DELETE RESTRICT,
    raw_article_id bigint      NOT NULL
        REFERENCES raw_articles(id) ON DELETE RESTRICT,
    seen_at        timestamptz NOT NULL,
    PRIMARY KEY (run_id, raw_article_id)
);

CREATE INDEX IF NOT EXISTS ingest_run_articles_by_article
    ON ingest_run_articles (raw_article_id);

-- "Latest" = newest version of each entry by its last observation time
-- (last_seen_at), exactly as reviewed. No DROP and no CREATE OR REPLACE: an
-- existing view is left alone here and judged by verify_ingestion_schema,
-- so drift is refused instead of masked.
DO $$
BEGIN
    IF to_regclass('latest_articles') IS NULL THEN
        CREATE VIEW latest_articles AS
        SELECT DISTINCT ON (source_id, entry_key)
               source_id, entry_key, entry_key_hash, version_hash,
               article_id, id, content_hash, first_seen_at, last_seen_at
        FROM raw_articles
        ORDER BY source_id, entry_key, last_seen_at DESC, id DESC;
    END IF;
END
$$;

INSERT INTO ingest_schema_meta (singleton, schema_version)
VALUES (true, 1)
ON CONFLICT (singleton) DO NOTHING;
