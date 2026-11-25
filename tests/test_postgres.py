"""PostgreSQL + pgvector + Row-Level Security. Skipped unless TRUSTRAG_PG_OWNER_DSN and TRUSTRAG_PG_APP_DSN are set.

The owner DSN creates the schema; the app DSN must be a role without superuser/BYPASSRLS that does not
own the table. CI runs these against a pgvector service container.
"""

from __future__ import annotations

import os

import pytest

OWNER = os.environ.get("TRUSTRAG_PG_OWNER_DSN")
APP = os.environ.get("TRUSTRAG_PG_APP_DSN")
pytestmark = pytest.mark.skipif(not (OWNER and APP), reason="PostgreSQL DSNs not set")

if OWNER and APP:
    import psycopg

    from trustrag.evaluation import evaluate, load_dataset, load_into, principal
    from trustrag.models import TENANT_WIDE, Document, Level, Principal
    from trustrag.pgstore import PostgresStore, UnsafeRoleError, install_schema
    from trustrag.service import RagService

ADMIN = None


@pytest.fixture
def store():
    install_schema(OWNER, "trustrag_app")
    with psycopg.connect(OWNER, autocommit=True) as c:
        c.execute("TRUNCATE trustrag_chunks")
    s = PostgresStore(APP)
    yield s
    s.close()


def _admin(tenant="acme"):
    return Principal(tenant, "ops", roles=frozenset({"admin"}))


def test_owner_or_superuser_connections_are_refused(store):
    with pytest.raises(UnsafeRoleError):
        PostgresStore(OWNER)


def test_evaluation_on_postgres_has_zero_leaks(store):
    r = evaluate(store=store)
    assert r.ok, r.to_markdown()
    assert r.leaked_chunks == 0 and r.baseline_forbidden > 10
    assert r.recall == 1.0 and r.denied_correctly == (6, 6)


def test_rls_hides_rows_even_from_an_unfiltered_query(store):
    """Second lock: a query with no WHERE clause, as the app role, still sees only permitted rows."""
    ds = load_dataset()
    load_into(RagService(store=store), ds)
    with psycopg.connect(APP) as c:
        assert c.execute("SELECT count(*) FROM trustrag_chunks").fetchone()[0] == 0  # no session scope
    for user in ("alice", "bob", "carol", "dave", "erin"):
        p = principal(ds, user)
        with psycopg.connect(APP) as c, c.transaction():
            c.execute(
                "SELECT set_config('trustrag.tenant', %s, true), set_config('trustrag.principals', %s, true),"
                " set_config('trustrag.clearance', %s, true)",
                (p.tenant, ",".join(sorted(p.entries())), str(int(p.clearance))),
            )
            seen = {r[0] for r in c.execute("SELECT DISTINCT doc_id FROM trustrag_chunks").fetchall()}
        from trustrag.evaluation import oracle_allows

        expected = {d["id"] for d in ds["documents"] if oracle_allows(ds, user, d["id"])}
        assert seen == expected, user


def test_rls_blocks_cross_tenant_writes(store):
    with psycopg.connect(APP) as c:
        c.execute("SELECT set_config('trustrag.tenant', 'acme', false), set_config('trustrag.admin', 'on', false)")
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            c.execute(
                "INSERT INTO trustrag_chunks (tenant_id, doc_id, n, title, body, acl, classification, embedding)"
                " VALUES ('globex', 'x', 0, 't', 'b', '{tenant:*}', 0, %s::vector)",
                ("[" + ",".join(["0"] * 256) + "]",),
            )


def test_admin_searches_are_filtered_like_anyone_elses(store):
    svc = RagService(store=store)
    svc.ingest(
        _admin(),
        Document("sal", "acme", "Salaries", "Senior band 130,000.", frozenset({"group:hr"}), Level.CONFIDENTIAL),
    )
    assert svc.retrieve(_admin(), "senior band") == []


def test_revocation_and_deletion_on_postgres(store):
    svc = RagService(store=store)
    alice = Principal("acme", "alice", frozenset({"engineering"}), Level.INTERNAL)
    svc.ingest(
        _admin(),
        Document(
            "rb",
            "acme",
            "Runbook",
            "Deploy the payment service with the release workflow.",
            frozenset({"group:engineering"}),
        ),
    )
    assert svc.retrieve(alice, "deploy payment service")
    assert svc.set_access(_admin(), "rb", frozenset({"group:sre"}), Level.INTERNAL) == 1
    assert svc.retrieve(alice, "deploy payment service") == []
    svc.set_access(_admin(), "rb", frozenset({TENANT_WIDE}), Level.INTERNAL)
    assert svc.retrieve(alice, "deploy payment service")
    assert svc.delete(_admin(), "rb") == 1 and svc.store.count("acme") == 0


def test_comma_in_scope_entries_is_rejected(store):
    with pytest.raises(ValueError):
        store._scope("acme", frozenset({"group:a,group:hr"}))
