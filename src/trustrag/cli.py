"""Command line: demo | eval | token | serve."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__
from .evaluation import evaluate, load_dataset, load_into, principal
from .models import Level, Principal
from .service import RagService


def _store(dsn: str | None):
    if not dsn:
        return None
    from .pgstore import PostgresStore

    return PostgresStore(dsn)


def cmd_demo(a: argparse.Namespace) -> int:
    ds = load_dataset(a.dataset)
    service = RagService(store=_store(a.postgres))
    load_into(service, ds)
    question = a.question or "What are the salary bands for senior engineers?"
    for user in a.users or ["alice", "bob"]:
        r = service.query(principal(ds, user), question)
        print(f"\n{user} ({ds['users'][user]['tenant']}, groups={ds['users'][user]['groups']}) asks: {question}")
        print(f"  answer:  {r.answer}")
        print(f"  sources: {[s.key for s in r.sources] or '-'}   verified: {r.verification.ok}")
    return 0


def cmd_eval(a: argparse.Namespace) -> int:
    report = evaluate(load_dataset(a.dataset), store=_store(a.postgres), k=a.k)
    if a.json:
        Path(a.json).write_text(json.dumps(report.to_json(), indent=2) + "\n", encoding="utf-8")
    sys.stdout.write(report.to_markdown())
    return 0 if report.ok else 1


def cmd_token(a: argparse.Namespace) -> int:
    from .auth import issue_token

    secret = os.environ.get("TRUSTRAG_TOKEN_SECRET", "")
    p = Principal(a.tenant, a.user, frozenset(a.groups), Level.parse(a.clearance), frozenset(a.roles))
    print(issue_token(secret, p, ttl_seconds=a.ttl))
    return 0


def cmd_serve(a: argparse.Namespace) -> int:
    import uvicorn

    from .api import create_app

    service = RagService(store=_store(a.postgres))
    if a.load_sample:
        load_into(service, load_dataset())
    uvicorn.run(create_app(service), host=a.host, port=a.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="trustrag", description="Permission-aware RAG")
    p.add_argument("--version", action="version", version=f"trustrag {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("demo", help="ask one question as several users")
    s.add_argument("--question")
    s.add_argument("--users", nargs="*")
    s.add_argument("--dataset")
    s.add_argument("--postgres", help="PostgreSQL DSN for the app role (default: in-memory store)")
    s.set_defaults(fn=cmd_demo)

    s = sub.add_parser("eval", help="leakage and quality evaluation")
    s.add_argument("--dataset")
    s.add_argument("--postgres")
    s.add_argument("--k", type=int, default=5)
    s.add_argument("--json")
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("token", help="issue a signed bearer token (needs TRUSTRAG_TOKEN_SECRET)")
    s.add_argument("--tenant", required=True)
    s.add_argument("--user", required=True)
    s.add_argument("--groups", nargs="*", default=[])
    s.add_argument("--roles", nargs="*", default=[])
    s.add_argument("--clearance", default="internal")
    s.add_argument("--ttl", type=int, default=3600)
    s.set_defaults(fn=cmd_token)

    s = sub.add_parser("serve", help="run the HTTP API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--postgres")
    s.add_argument("--load-sample", action="store_true")
    s.set_defaults(fn=cmd_serve)

    a = p.parse_args(argv)
    try:
        return a.fn(a)
    except (ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
