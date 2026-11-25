"""RagService: the one place where ingestion, access changes and queries are authorised and audited."""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .answer import Answerer, ExtractiveAnswerer, Source, Verification, number_sources, verify_citations
from .models import Chunk, Document, Level, Principal
from .sensitive import instruction_flags, redact
from .store import MemoryStore, Store
from .text import Embedder, HashingEmbedder, chunk_text


class Forbidden(Exception):
    pass


@dataclass
class IngestReport:
    doc_id: str
    chunks: int
    redactions: dict[str, int]
    flagged_chunks: int


@dataclass
class QueryResult:
    answer: str
    sources: list[Source]
    verification: Verification
    redacted_in_answer: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": [
                {"n": s.n, "doc_id": s.doc_id, "chunk": s.key, "title": s.title, "flags": list(s.flags)}
                for s in self.sources
            ],
            "verification": {
                "ok": self.verification.ok,
                "cited": self.verification.cited,
                "invalid_refs": self.verification.invalid_refs,
                "unsupported": self.verification.unsupported,
                "uncited": self.verification.uncited,
            },
            "redacted_in_answer": self.redacted_in_answer,
        }


class AuditTrail:
    """JSON Lines; the question is stored as a hash, sources as chunk keys."""

    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self.records: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def write(self, event: str, p: Principal, **fields: Any) -> None:
        rec = {"ts": datetime.now(UTC).isoformat(), "event": event, "tenant": p.tenant, "user": p.user, **fields}
        with self._lock:
            self.records.append(rec)
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, sort_keys=True) + "\n")


class RagService:
    def __init__(
        self,
        store: Store | None = None,
        embedder: Embedder | None = None,
        answerer: Answerer | None = None,
        audit: AuditTrail | None = None,
        redact_pii: bool = True,
        max_k: int = 8,
        block_unverified: bool = True,
    ):
        self.store = store or MemoryStore()
        self.embedder = embedder or HashingEmbedder()
        self.answerer = answerer or ExtractiveAnswerer()
        self.audit = audit or AuditTrail(None)
        self.redact_pii = redact_pii
        self.max_k = max_k
        self.block_unverified = block_unverified

    # ------------------------------------------------------------------ administration

    def _require_admin(self, p: Principal, tenant: str) -> None:
        if not p.is_admin() or p.tenant != tenant:
            self.audit.write("admin.denied", p, target_tenant=tenant)
            raise Forbidden("administrator of this tenant required")

    def ingest(self, p: Principal, doc: Document) -> IngestReport:
        self._require_admin(p, doc.tenant)
        if not doc.acl:
            raise ValueError(f"document {doc.id} has an empty ACL; nobody could read it (deny by default)")
        pieces = chunk_text(doc.text)
        redactions: dict[str, int] = {}
        chunks: list[Chunk] = []
        clean_texts = []
        for piece in pieces:
            clean, found = redact(piece, pii=self.redact_pii)
            for label in found:
                redactions[label] = redactions.get(label, 0) + 1
            clean_texts.append(clean)
        vectors = self.embedder.embed([f"{doc.title}. {t}" for t in clean_texts])
        for n, (text, vec) in enumerate(zip(clean_texts, vectors, strict=True)):
            chunks.append(
                Chunk(
                    doc.id,
                    doc.tenant,
                    n,
                    doc.title,
                    text,
                    doc.acl,
                    doc.classification,
                    vec,
                    tuple(instruction_flags(text)),
                )
            )
        self.store.upsert_document(chunks)
        report = IngestReport(doc.id, len(chunks), redactions, sum(1 for c in chunks if c.flags))
        self.audit.write("document.ingested", p, doc=doc.id, chunks=len(chunks), redactions=redactions)
        return report

    def set_access(self, p: Principal, doc_id: str, acl: frozenset[str], classification: Level) -> int:
        self._require_admin(p, p.tenant)
        Document(doc_id, p.tenant, "-", "-", acl, classification)  # validates the ACL entries
        if not acl:
            raise ValueError("empty ACL; delete the document instead")
        n = self.store.update_access(p.tenant, doc_id, acl, classification)
        self.audit.write("document.access_changed", p, doc=doc_id, acl=sorted(acl), classification=classification.name)
        return n

    def delete(self, p: Principal, doc_id: str) -> int:
        self._require_admin(p, p.tenant)
        n = self.store.delete_document(p.tenant, doc_id)
        self.audit.write("document.deleted", p, doc=doc_id, chunks=n)
        return n

    # ------------------------------------------------------------------ query

    def retrieve(self, p: Principal, question: str, k: int = 5) -> list[tuple[Chunk, float]]:
        k = max(1, min(k, self.max_k))
        vec = self.embedder.embed([question])[0]
        hits = self.store.search(p, question, vec, k)
        # Defence in depth: a store bug must not leak. Re-check every hit against the principal.
        leaked = [c.key for c, _ in hits if not c.visible_to(p)]
        if leaked:
            self.audit.write("store.leak_blocked", p, chunks=leaked)
            hits = [(c, s) for c, s in hits if c.visible_to(p)]
        return hits

    def query(self, p: Principal, question: str, k: int = 5) -> QueryResult:
        hits = self.retrieve(p, question, k)
        sources = number_sources(hits)
        answer = self.answerer.answer(question, sources)
        verification = verify_citations(answer, sources)
        if self.block_unverified and not verification.ok:
            answer = "The generated answer could not be verified against your sources, so it was withheld."
        answer, found = redact(answer, pii=self.redact_pii)
        self.audit.write(
            "query",
            p,
            question_sha256=hashlib.sha256(question.encode()).hexdigest(),
            sources=[s.key for s in sources],
            verified=verification.ok,
            redacted=found,
        )
        return QueryResult(answer, sources, verification, found)
