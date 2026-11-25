-- TrustRAG schema for PostgreSQL 16+ with pgvector. Run as the table owner (not as the app role).
-- Row-Level Security is the second lock: the application filters every query itself, and the
-- database refuses to return rows outside the session's tenant, ACL entries and clearance even if
-- an application query forgets its WHERE clause. See docs/adr/0002-postgres-row-level-security.md.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS trustrag_chunks (
    tenant_id      text        NOT NULL,
    doc_id         text        NOT NULL,
    n              integer     NOT NULL,
    title          text        NOT NULL,
    body           text        NOT NULL,
    acl            text[]      NOT NULL,
    classification smallint    NOT NULL CHECK (classification BETWEEN 0 AND 3),
    flags          text[]      NOT NULL DEFAULT '{}',
    embedding      vector(256) NOT NULL,
    tsv            tsvector    GENERATED ALWAYS AS (to_tsvector('english', title || ' ' || body)) STORED,
    PRIMARY KEY (tenant_id, doc_id, n)
);

CREATE INDEX IF NOT EXISTS trustrag_chunks_acl ON trustrag_chunks USING gin (acl);
CREATE INDEX IF NOT EXISTS trustrag_chunks_tsv ON trustrag_chunks USING gin (tsv);
-- No approximate vector index on purpose: with pgvector < 0.8, filtering after an HNSW/IVFFlat scan
-- can silently return fewer than k authorised rows. Exact scans are correct; add an index with
-- iterative scans (pgvector >= 0.8) when the corpus needs it.

ALTER TABLE trustrag_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE trustrag_chunks FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS trustrag_read ON trustrag_chunks;
CREATE POLICY trustrag_read ON trustrag_chunks FOR SELECT USING (
    tenant_id = current_setting('trustrag.tenant', true)
    AND acl && string_to_array(current_setting('trustrag.principals', true), ',')
    AND classification <= coalesce(nullif(current_setting('trustrag.clearance', true), '')::int, -1)
);

DROP POLICY IF EXISTS trustrag_admin ON trustrag_chunks;
CREATE POLICY trustrag_admin ON trustrag_chunks FOR ALL
    USING (tenant_id = current_setting('trustrag.tenant', true) AND current_setting('trustrag.admin', true) = 'on')
    WITH CHECK (tenant_id = current_setting('trustrag.tenant', true) AND current_setting('trustrag.admin', true) = 'on');
