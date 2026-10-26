-- Initial database setup for the durable approved-run store.
--
-- This is NOT a migration system. It creates the objects a fresh database needs
-- and is safe to re-run because every object uses IF NOT EXISTS. It will not
-- alter an existing table: `python scripts/init_db.py` verifies the resulting
-- schema afterwards and refuses to continue when an older/incompatible shape is
-- present, rather than dropping or rewriting data.
--
-- Identity: one row in `runs` per (run_id, canonical business payload). Two
-- writes are the same snapshot when both run_id and content_hash match; the
-- unique constraint is what makes retries idempotent and concurrent duplicates
-- impossible.

CREATE TABLE IF NOT EXISTS schema_meta (
    singleton      boolean     PRIMARY KEY DEFAULT true CHECK (singleton),
    schema_version integer     NOT NULL,
    applied_at     timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS runs (
    id             bigserial   PRIMARY KEY,
    run_id         text        NOT NULL,
    content_hash   text        NOT NULL CHECK (length(content_hash) = 64),
    schema_version integer     NOT NULL DEFAULT 1,
    rss_url        text        NOT NULL,
    competitor_url text        NOT NULL,
    num_articles   integer     NOT NULL,
    warnings       jsonb       NOT NULL DEFAULT '[]'::jsonb,
    -- Exact business payload used to compute content_hash. Reconstructed models
    -- are compared against this on read.
    payload        jsonb       NOT NULL,
    -- Statement time, NOT commit time and NOT approval time. Ordering across
    -- concurrent writers is advisory; see latest_runs.
    stored_at      timestamptz NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT runs_identity UNIQUE (run_id, content_hash)
);

CREATE INDEX IF NOT EXISTS runs_by_run ON runs (run_id, id DESC);
CREATE INDEX IF NOT EXISTS runs_by_hash ON runs (content_hash);

CREATE TABLE IF NOT EXISTS run_articles (
    run_version     bigint      NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    article_id      text        NOT NULL,
    ordinal         integer     NOT NULL,
    headline        text        NOT NULL,
    summary         text        NOT NULL,
    key_facts       jsonb       NOT NULL DEFAULT '[]'::jsonb,
    tone_hint       text        NOT NULL,
    feed_url        text        NOT NULL,
    source_title    text        NOT NULL,
    source_url      text,
    -- Typed timestamp for querying plus the verbatim source value so a value we
    -- cannot parse is preserved rather than dropped.
    published_at    timestamptz,
    published_at_raw text,
    retrieved_at    timestamptz,
    retrieved_at_raw text       NOT NULL,
    content_hash    text        NOT NULL,
    excerpt         text        NOT NULL,
    content_kind    text        NOT NULL,
    PRIMARY KEY (run_version, article_id),
    UNIQUE (run_version, ordinal)
);

CREATE TABLE IF NOT EXISTS run_topics (
    run_version      bigint           NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    ordinal          integer          NOT NULL,
    topic_name       text             NOT NULL,
    keywords         jsonb            NOT NULL DEFAULT '[]'::jsonb,
    model_confidence double precision NOT NULL
        CHECK (model_confidence >= 0 AND model_confidence <= 1),
    window_start     text,
    window_end       text,
    source_count     integer          NOT NULL CHECK (source_count >= 0),
    PRIMARY KEY (run_version, ordinal)
);

-- The topic agent intentionally keeps citations it could not resolve ("Unknown
-- source", verified = false). A plain foreign key on article_id would force us
-- to drop those rows or abort the run, so the raw citation is always kept and
-- matched_article_id carries the foreign key only when the citation was
-- verified against a captured article.
CREATE TABLE IF NOT EXISTS run_topic_evidence (
    id                bigserial NOT NULL PRIMARY KEY,
    run_version       bigint    NOT NULL,
    topic_ordinal     integer   NOT NULL,
    article_id        text      NOT NULL,
    matched_article_id text,
    quote             text      NOT NULL,
    source_title      text      NOT NULL,
    source_url        text,
    published_at      timestamptz,
    published_at_raw  text,
    verified          boolean   NOT NULL,
    FOREIGN KEY (run_version, topic_ordinal)
        REFERENCES run_topics (run_version, ordinal) ON DELETE CASCADE,
    FOREIGN KEY (run_version, matched_article_id)
        REFERENCES run_articles (run_version, article_id) ON DELETE CASCADE,
    CHECK (matched_article_id IS NULL OR verified)
);

CREATE INDEX IF NOT EXISTS topic_evidence_by_topic
    ON run_topic_evidence (run_version, topic_ordinal);

CREATE TABLE IF NOT EXISTS run_tones (
    run_version      bigint           NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    article_id       text             NOT NULL,
    headline         text             NOT NULL,
    overall          text             NOT NULL
        CHECK (overall IN ('positive', 'negative', 'neutral', 'mixed')),
    score            double precision NOT NULL CHECK (score >= -1 AND score <= 1),
    aspects          jsonb            NOT NULL DEFAULT '[]'::jsonb,
    source_url       text,
    published_at     timestamptz,
    published_at_raw text,
    retrieved_at     timestamptz,
    retrieved_at_raw text,
    PRIMARY KEY (run_version, article_id),
    FOREIGN KEY (run_version, article_id)
        REFERENCES run_articles (run_version, article_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS competitor_profiles (
    run_version       bigint      PRIMARY KEY REFERENCES runs(id) ON DELETE CASCADE,
    company           text        NOT NULL,
    positioning       text        NOT NULL,
    products          jsonb       NOT NULL DEFAULT '[]'::jsonb,
    pricing_model     text,
    funding_stage     text,
    strengths         jsonb       NOT NULL DEFAULT '[]'::jsonb,
    weaknesses        jsonb       NOT NULL DEFAULT '[]'::jsonb,
    sources           jsonb       NOT NULL DEFAULT '[]'::jsonb,
    -- RetrievedSource records: url, retrieved_at, content_hash, kept verbatim.
    retrieved_sources jsonb       NOT NULL DEFAULT '[]'::jsonb,
    retrieved_at      timestamptz
);

-- "Latest" means the highest run version id for a run_id, i.e. insertion order.
-- bigserial values are assigned at INSERT time and can commit out of order, and
-- stored_at is statement time, so neither is a commit-order or approval-order
-- guarantee. Both versions always remain queryable.
CREATE OR REPLACE VIEW latest_runs AS
SELECT DISTINCT ON (run_id) run_id, id, content_hash, stored_at
FROM runs
ORDER BY run_id, id DESC;

INSERT INTO schema_meta (singleton, schema_version)
VALUES (true, 1)
ON CONFLICT (singleton) DO NOTHING;
