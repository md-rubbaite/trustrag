# ADR 0002: PostgreSQL Row-Level Security as the second lock

- **Status:** Accepted
- **Date:** 2026-11-25

## Context

Application filters are only as good as every query that uses them. One forgotten `WHERE` clause in a
new feature, a report or an admin script can expose every tenant. Data stores used for RAG in
production usually hold more than one tenant.

## Decision

1. `trustrag_chunks` has `ENABLE` and `FORCE ROW LEVEL SECURITY`. The read policy allows a row only
   when the transaction's `trustrag.tenant`, `trustrag.principals` and `trustrag.clearance` settings
   match the row's tenant, ACL (array overlap) and classification.
2. Settings are transaction-local (`set_config(..., true)`), set by `PostgresStore` at the start of
   every transaction, so a pooled connection cannot carry one user's scope into another's query.
3. Writes use a separate policy that requires `trustrag.admin = on` and the same tenant (`WITH CHECK`),
   so even an admin session cannot write rows into another tenant. Searches always run with admin off.
4. The connection role must not be a superuser, own the table or have BYPASSRLS (RLS would not apply).
   `PostgresStore` checks this at start-up and refuses to run otherwise.
5. Application queries still filter explicitly. RLS is the second lock, not the only one.

## Alternatives considered

- **Schema or database per tenant:** strong isolation, operationally heavy at many tenants, and does
  not cover ACLs within a tenant.
- **Views with security barriers:** weaker than RLS against a role that can query the base table.

## Consequences

- Tests run an unfiltered `SELECT` as the app role and check that it returns exactly the rows an
  independent oracle allows for each user, and nothing without a scope.
- No approximate vector index is created yet. With pgvector before 0.8, filtering after an HNSW or
  IVFFlat scan can return fewer than k authorised rows. Exact scans are correct at this corpus size.
  Add iterative index scans (pgvector 0.8+) when the corpus grows.
