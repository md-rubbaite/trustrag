# Changelog

All notable changes are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-11-25

First release (MVP).

### Added
- Access model:
  - tenants, ACL entries (`user:`, `group:`, `tenant:*`) and classification against clearance;
  - deny by default, with safe-name validation.
- In-memory and PostgreSQL/pgvector stores:
  - authorisation inside the query, before ranking;
  - RLS policies and a start-up check of the connection role.
- Ingestion pipeline:
  - redaction of secrets and personal data before chunking and embedding;
  - instruction-like text flagged;
  - sentence-aware chunking.
- Hybrid ranking:
  - BM25 statistics computed only over authorised candidates;
  - a deterministic hashing embedder.
- Answers:
  - extractive answerer by default, plus an OpenAI-compatible LLM answerer;
  - citation verification, with unverified answers withheld;
  - output redaction.
- HS256 bearer tokens and a FastAPI HTTP API; CLI commands `demo`, `eval`, `token` and `serve`.
- Leakage and quality evaluation against an independent oracle and a permission-unaware baseline.
- Design docs: ADRs 0001–0003 and a threat model.
- CI:
  - Python 3.11–3.13;
  - PostgreSQL 17 + pgvector job;
  - actions pinned to commit SHAs.
