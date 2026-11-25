"""Leakage and quality evaluation on a labelled corpus.

For every query, run as a specific user:

- **leakage:** every retrieved chunk is checked against an *independent* oracle written directly
  from the dataset (tenant, ACL list, clearance), not with `Chunk.visible_to`, so a bug in the
  access model is not checked by the same code. Must be zero.
- **baseline:** the same query against a naive RAG (same index, no access control) shows how many
  forbidden chunks permission-unaware retrieval would have put in front of the model.
- **recall@k:** share of labelled relevant documents retrieved (queries with relevant documents only).
- **answers:** citation verification passes; no secret or personal data appears in any answer;
  queries with nothing relevant get the "not found" answer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from .answer import NOT_FOUND
from .models import Document, Level, Principal
from .service import RagService
from .store import MemoryStore, Store

LEVELS = {"public": 0, "internal": 1, "confidential": 2, "restricted": 3}


def load_dataset(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        return json.loads(resources.files("trustrag").joinpath("data/sample.json").read_text(encoding="utf-8"))
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _acl(raw: str | list[str]) -> list[str]:
    return [raw] if isinstance(raw, str) else list(raw)


def documents(ds: dict[str, Any]) -> list[Document]:
    return [
        Document(
            d["id"], d["tenant"], d["title"], d["text"], frozenset(_acl(d["acl"])), Level.parse(d["classification"])
        )
        for d in ds["documents"]
    ]


def principal(ds: dict[str, Any], user: str) -> Principal:
    u = ds["users"][user]
    return Principal(u["tenant"], user, frozenset(u["groups"]), Level.parse(u["clearance"]))


def oracle_allows(ds: dict[str, Any], user: str, doc_id: str) -> bool:
    """Ground truth written from the dataset alone."""
    u = ds["users"][user]
    d = next(x for x in ds["documents"] if x["id"] == doc_id)
    if d["tenant"] != u["tenant"]:
        return False
    if LEVELS[d["classification"]] > LEVELS[u["clearance"]]:
        return False
    for entry in _acl(d["acl"]):
        if entry == "tenant:*" or entry == f"user:{user}" or entry in {f"group:{g}" for g in u["groups"]}:
            return True
    return False


def load_into(service: RagService, ds: dict[str, Any]) -> None:
    for tenant in sorted({d["tenant"] for d in ds["documents"]}):
        admin = Principal(tenant, "ingest-bot", roles=frozenset({"admin"}))
        for doc in (x for x in documents(ds) if x.tenant == tenant):
            service.ingest(admin, doc)


@dataclass
class QueryOutcome:
    id: str
    user: str
    retrieved_docs: list[str]
    leaked: list[str]
    baseline_forbidden: int
    relevant: list[str]
    found_relevant: list[str]
    answer: str
    verified: bool
    sensitive_in_answer: bool
    answered_from_relevant: bool = False


@dataclass
class EvalReport:
    k: int
    outcomes: list[QueryOutcome] = field(default_factory=list)

    @property
    def leaked_chunks(self) -> int:
        return sum(len(o.leaked) for o in self.outcomes)

    @property
    def baseline_forbidden(self) -> int:
        return sum(o.baseline_forbidden for o in self.outcomes)

    @property
    def recall(self) -> float:
        graded = [o for o in self.outcomes if o.relevant]
        return sum(len(o.found_relevant) / len(o.relevant) for o in graded) / max(len(graded), 1)

    @property
    def denied_correctly(self) -> tuple[int, int]:
        denied = [o for o in self.outcomes if not o.relevant]
        return sum(o.answer == NOT_FOUND for o in denied), len(denied)

    @property
    def answered(self) -> tuple[int, int]:
        graded = [o for o in self.outcomes if o.relevant]
        return sum(o.answered_from_relevant for o in graded), len(graded)

    @property
    def verified(self) -> tuple[int, int]:
        return sum(o.verified for o in self.outcomes), len(self.outcomes)

    @property
    def sensitive_answers(self) -> int:
        return sum(o.sensitive_in_answer for o in self.outcomes)

    @property
    def ok(self) -> bool:
        return self.leaked_chunks == 0 and self.sensitive_answers == 0 and self.verified[0] == self.verified[1]

    def to_markdown(self) -> str:
        d_ok, d_n = self.denied_correctly
        v_ok, v_n = self.verified
        lines = [
            "# TrustRAG evaluation",
            "",
            f"- Queries: **{len(self.outcomes)}** (k = {self.k})",
            f"- Unauthorised chunks retrieved: **{self.leaked_chunks}**",
            f"- Unauthorised chunks a permission-unaware RAG would have retrieved: **{self.baseline_forbidden}**",
            f"- Recall@{self.k} on queries with relevant documents: **{self.recall:.2f}**",
            f"- Queries with nothing the user may see that answered 'not found': **{d_ok} / {d_n}**",
            f"- Queries with relevant documents answered from one of them: **{self.answered[0]} / {self.answered[1]}**",
            f"- Answers passing citation verification: **{v_ok} / {v_n}**",
            f"- Answers containing secrets or personal data: **{self.sensitive_answers}**",
            f"- Result: **{'PASS' if self.ok else 'FAIL'}**",
            "",
            "| Query | User | Retrieved | Baseline forbidden | Relevant found |",
            "|---|---|---|---|---|",
        ]
        for o in self.outcomes:
            rel = f"{len(o.found_relevant)}/{len(o.relevant)}" if o.relevant else "n/a"
            lines.append(
                f"| {o.id} | {o.user} | {', '.join(o.retrieved_docs) or '-'} | {o.baseline_forbidden} | {rel} |"
            )
        return "\n".join(lines) + "\n"

    def to_json(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "leaked_chunks": self.leaked_chunks,
            "baseline_forbidden": self.baseline_forbidden,
            "recall": round(self.recall, 4),
            "denied_correctly": list(self.denied_correctly),
            "answered_from_relevant": list(self.answered),
            "verified": list(self.verified),
            "sensitive_answers": self.sensitive_answers,
            "ok": self.ok,
            "queries": [o.__dict__ for o in self.outcomes],
        }


def _naive_hits(store: MemoryStore, service: RagService, tenantless_question: str, k: int) -> list[str]:
    """Retrieval with no access control at all: every chunk of every tenant is a candidate."""
    from .store import bm25, hybrid_scores
    from .text import cosine

    every = [c for cs in store._chunks.values() for c in cs]
    vec = service.embedder.embed([tenantless_question])[0]
    hits = hybrid_scores(every, bm25(every, tenantless_question), [cosine(vec, c.embedding) for c in every], k)
    return [c.doc_id for c, _ in hits]


def evaluate(ds: dict[str, Any] | None = None, store: Store | None = None, k: int = 5) -> EvalReport:
    from .sensitive import redact

    ds = ds or load_dataset()
    service = RagService(store=store or MemoryStore())
    load_into(service, ds)
    baseline_store = MemoryStore()
    baseline = RagService(store=baseline_store, embedder=service.embedder)
    load_into(baseline, ds)
    report = EvalReport(k)
    for q in ds["queries"]:
        p = principal(ds, q["user"])
        hits = service.retrieve(p, q["question"], k)
        docs = [c.doc_id for c, _ in hits]
        leaked = [c.key for c, _ in hits if not oracle_allows(ds, q["user"], c.doc_id)]
        naive = _naive_hits(baseline_store, baseline, q["question"], k)
        result = service.query(p, q["question"], k)
        _, sensitive = redact(result.answer, pii=True)
        report.outcomes.append(
            QueryOutcome(
                id=q["id"],
                user=q["user"],
                retrieved_docs=list(dict.fromkeys(docs)),
                leaked=leaked,
                baseline_forbidden=sum(1 for d in naive if not oracle_allows(ds, q["user"], d)),
                relevant=q["relevant"],
                found_relevant=[d for d in q["relevant"] if d in docs],
                answer=result.answer,
                verified=result.verification.ok,
                sensitive_in_answer=bool(sensitive),
                answered_from_relevant=result.answer != NOT_FOUND
                and any(result.sources[n - 1].doc_id in q["relevant"] for n in result.verification.cited),
            )
        )
    return report
