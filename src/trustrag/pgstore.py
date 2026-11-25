"""PostgreSQL + pgvector store with Row-Level Security (optional: `pip install trustrag[postgres]`).

Every operation runs in its own transaction and first sets transaction-local settings
(`trustrag.tenant`, `trustrag.principals`, `trustrag.clearance`, `trustrag.admin`). The RLS policies in
`sql/schema.sql` read those settings, so the database enforces the same rule as the application:
right tenant, matching ACL entry, sufficient clearance. Searches always run with `trustrag.admin`
off, even for administrators, so an admin's searches are filtered like anyone else's.

The application role must not be a superuser, must not own the table and must not have BYPASSRLS.
`check_role()` verifies that at start-up and refuses to run otherwise.
"""

from __future__ import annotations

import threading
from importlib import resources
from typing import Any

from .models import Chunk, Level, Principal
from .store import hybrid_scores

try:
    import psycopg

    HAVE_PSYCOPG = True
except ImportError:  # pragma: no cover
    HAVE_PSYCOPG = False


def _vec(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


def schema_sql() -> str:
    return resources.files("trustrag").joinpath("sql/schema.sql").read_text(encoding="utf-8")


def install_schema(owner_dsn: str, app_role: str) -> None:
    """Create table and policies as the owner, and grant the app role the minimum it needs."""
    if not app_role.replace("_", "").isalnum():
        raise ValueError("app role name must be alphanumeric/underscore")
    with psycopg.connect(owner_dsn, autocommit=True) as conn:
        conn.execute(schema_sql())
        conn.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE ON trustrag_chunks TO {app_role}")


class UnsafeRoleError(RuntimeError):
    pass


class PostgresStore:
    def __init__(self, dsn: str, candidate_pool: int = 50, verify_role: bool = True):
        if not HAVE_PSYCOPG:
            raise RuntimeError("PostgresStore needs psycopg: pip install 'trustrag[postgres]'")
        self._conn = psycopg.connect(dsn, autocommit=False)
        self._lock = threading.Lock()
        self.pool = candidate_pool
        if verify_role:
            try:
                self.check_role()
            except Exception:
                self._conn.close()
                raise

    def close(self) -> None:
        self._conn.close()

    def check_role(self) -> None:
        with self._lock, self._conn.transaction():
            row = self._conn.execute(
                """SELECT r.rolsuper, r.rolbypassrls, c.relowner = r.oid
                   FROM pg_roles r, pg_class c
                   WHERE r.rolname = current_user AND c.relname = 'trustrag_chunks'"""
            ).fetchone()
        if row is None:
            raise UnsafeRoleError("table trustrag_chunks not found; run install_schema first")
        superuser, bypass, owner = row
        if superuser or bypass or owner:
            raise UnsafeRoleError(
                "the TrustRAG connection role must not be a superuser, the table owner or have BYPASSRLS; "
                "Row-Level Security would not apply"
            )

    def _scope(self, tenant: str, principals: frozenset[str] = frozenset(), clearance: int = -1, admin: bool = False):
        for e in principals:
            if "," in e:
                raise ValueError("principal entries must not contain commas")
        self._conn.execute(
            "SELECT set_config('trustrag.tenant', %s, true), set_config('trustrag.principals', %s, true),"
            " set_config('trustrag.clearance', %s, true), set_config('trustrag.admin', %s, true)",
            (tenant, ",".join(sorted(principals)), str(clearance), "on" if admin else "off"),
        )

    # -------------------------------------------------------------- writes (admin scope)

    def upsert_document(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        tenant, doc_id = chunks[0].tenant, chunks[0].doc_id
        with self._lock, self._conn.transaction():
            self._scope(tenant, admin=True)
            self._conn.execute("DELETE FROM trustrag_chunks WHERE tenant_id = %s AND doc_id = %s", (tenant, doc_id))
            with self._conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO trustrag_chunks"
                    " (tenant_id, doc_id, n, title, body, acl, classification, flags, embedding)"
                    " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector)",
                    [
                        (
                            c.tenant,
                            c.doc_id,
                            c.n,
                            c.title,
                            c.text,
                            sorted(c.acl),
                            int(c.classification),
                            list(c.flags),
                            _vec(c.embedding),
                        )
                        for c in chunks
                    ],
                )

    def delete_document(self, tenant: str, doc_id: str) -> int:
        with self._lock, self._conn.transaction():
            self._scope(tenant, admin=True)
            cur = self._conn.execute(
                "DELETE FROM trustrag_chunks WHERE tenant_id = %s AND doc_id = %s", (tenant, doc_id)
            )
            return cur.rowcount

    def update_access(self, tenant: str, doc_id: str, acl: frozenset[str], classification: Level) -> int:
        with self._lock, self._conn.transaction():
            self._scope(tenant, admin=True)
            cur = self._conn.execute(
                "UPDATE trustrag_chunks SET acl = %s, classification = %s WHERE tenant_id = %s AND doc_id = %s",
                (sorted(acl), int(classification), tenant, doc_id),
            )
            return cur.rowcount

    # -------------------------------------------------------------- reads (principal scope)

    def search(self, p: Principal, query: str, query_vec: list[float], k: int) -> list[tuple[Chunk, float]]:
        entries = p.entries()
        sql = """
            WITH vec AS (
                SELECT doc_id, n FROM trustrag_chunks
                WHERE tenant_id = %(t)s AND acl && %(acl)s AND classification <= %(c)s
                ORDER BY embedding <=> %(v)s::vector LIMIT %(pool)s
            ), lex AS (
                SELECT doc_id, n FROM trustrag_chunks
                WHERE tenant_id = %(t)s AND acl && %(acl)s AND classification <= %(c)s
                  AND tsv @@ plainto_tsquery('english', %(q)s)
                ORDER BY ts_rank_cd(tsv, plainto_tsquery('english', %(q)s)) DESC LIMIT %(pool)s
            )
            SELECT ch.doc_id, ch.n, ch.title, ch.body, ch.acl, ch.classification, ch.flags,
                   1 - (ch.embedding <=> %(v)s::vector) AS sim,
                   ts_rank_cd(ch.tsv, plainto_tsquery('english', %(q)s)) AS lex
            FROM trustrag_chunks ch
            JOIN (SELECT * FROM vec UNION SELECT * FROM lex) cand USING (doc_id, n)
            WHERE ch.tenant_id = %(t)s
        """
        params: dict[str, Any] = {
            "t": p.tenant,
            "acl": sorted(entries),
            "c": int(p.clearance),
            "v": _vec(query_vec),
            "q": query,
            "pool": max(self.pool, k * 5),
        }
        with self._lock, self._conn.transaction():
            self._scope(p.tenant, entries, int(p.clearance), admin=False)
            rows = self._conn.execute(sql, params).fetchall()
        chunks, lex, sim = [], [], []
        for doc_id, n, title, body, acl, cls, flags, s, lx in rows:
            chunks.append(Chunk(doc_id, p.tenant, n, title, body, frozenset(acl), Level(cls), [], tuple(flags)))
            lex.append(float(lx))
            sim.append(float(s))
        return hybrid_scores(chunks, lex, sim, k)

    def count(self, tenant: str) -> int:
        with self._lock, self._conn.transaction():
            self._scope(tenant, admin=True)
            return self._conn.execute("SELECT count(*) FROM trustrag_chunks").fetchone()[0]
