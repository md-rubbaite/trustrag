# Threat model: multi-tenant enterprise RAG

**Assets:** documents and their chunks, embeddings, access metadata, answers, the audit trail.
**Actors:** authenticated users of a tenant, tenant administrators, other tenants, authors of documents
(possibly malicious), the LLM provider.

| # | Threat | Control in TrustRAG | Residual risk |
|---|---|---|---|
| R1 | Cross-tenant retrieval | Tenant from verified token only; filter inside the store query; RLS on the table ([ADR 0002](adr/0002-postgres-row-level-security.md)) | Compromise of the token secret or IdP |
| R2 | Within-tenant over-sharing (wrong group or clearance) | ACL + clearance checked before ranking ([ADR 0001](adr/0001-authorise-inside-retrieval.md)); deny by default | ACLs are only as good as the source system's permissions |
| R3 | Stale access after revocation | Access changes update every chunk in one transaction; no result cache | Answers already delivered cannot be recalled |
| R4 | Prompt injection in documents claiming privileges | Access decided from metadata only; instruction-like chunks flagged, never quoted, labelled for the LLM | An LLM may still be misled within what the user may see |
| R5 | Secrets and personal data in documents | Redacted before chunking and embedding; again on answers | Pattern-based; unusual formats need rules |
| R6 | Invented or misattributed citations | Citation verification; unverified answers withheld ([ADR 0003](adr/0003-extractive-default-and-citation-verification.md)) | Lexical support check, not entailment |
| R7 | Ranking side channel (scores reveal hidden words) | BM25 statistics from authorised candidates only; `ts_rank_cd` on Postgres | Timing differences are not addressed |
| R8 | ACL entry smuggling (`x,group:hr`) | Safe-name rules on principals, ACLs and token claims; commas rejected before RLS settings | — |
| R9 | Forgotten filter in new code | RLS returns no rows without a scope and only permitted rows with one; service re-checks every hit | Superuser or BYPASSRLS roles; refused at start-up |
| R10 | Token forgery or downgrade | HS256 only, constant-time compare, `exp` and `iss` required, secrets ≥ 32 bytes | Use an IdP (OIDC) in production |
| R11 | Data sent to the LLM provider | Only authorised, redacted chunks are sent; provider is pluggable (can be self-hosted) | Provider retention policies |
| R12 | Repudiation | Audit trail of ingestion, access changes and queries (question stored as SHA-256) | Not tamper-evident yet; see roadmap |
