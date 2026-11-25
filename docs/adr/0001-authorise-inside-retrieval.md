# ADR 0001: Authorise inside retrieval, before ranking, from metadata only

- **Status:** Accepted
- **Date:** 2026-11-25

## Context

Most RAG prototypes index everything into one vector store and filter after retrieval, or rely on the
model to "not mention" what the user should not see. Both leak: post-filtering returns fewer results
and exposes forbidden chunks to the model, and models follow text they are given. Enterprise
deployments must also isolate tenants and respect document classifications.

## Decision

1. **Access is decided from metadata only:** tenant, ACL entries (`user:`, `group:`, `tenant:*`) and
   classification against clearance. Document text can never change it; instruction-like text is flagged
   and never quoted.
2. **The filter runs inside the store's query, before ranking.** A chunk the principal may not see is
   never a candidate, so it cannot be ranked, passed to the model or cited.
3. **Ranking statistics come from the authorised candidates only.** BM25 IDF is computed over what
   the user may see, so adding a confidential document cannot change a public document's score (no
   side channel on the existence of words in hidden documents). The Postgres path uses `ts_rank_cd`,
   which has no corpus statistics.
4. **Deny by default.** An empty ACL is refused at ingestion. Principals, ACL entries and token claims
   use a safe character set, so a crafted name such as `x,group:hr` cannot smuggle extra entries.
5. **Defence in depth in the service.** Every hit is re-checked against the principal; a store that
   returns a forbidden chunk is caught, the chunk dropped and the event audited.
6. **The tenant comes from the verified token only,** never from the request body or document id.

## Alternatives considered

- **One index per tenant or per group:** strong isolation, but ACLs that cross groups multiply indexes,
  and revocation means re-indexing. Kept as an option for very large tenants; not the default.
- **Post-filtering:** rejected (leaks to the model, unstable k).
- **Encrypting embeddings per tenant:** does not solve within-tenant ACLs; embeddings can still leak
  content to anyone who can query them.

## Consequences

Revocation and reclassification take effect on the next query (no cache of results). The leakage
evaluation checks results against an independent oracle and compares with a permission-unaware
baseline on the same index.
