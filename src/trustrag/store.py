"""Chunk stores. Authorisation is part of every read: a store cannot return a chunk the principal may
not see, and ranking statistics are computed only over chunks the principal may see.

Why the second point matters: BM25's inverse document frequency depends on every document in the
index. If it were computed over the whole tenant, the score of a public document would change when a
confidential document containing the same rare word is added: a measurable side channel that reveals
the existence of words in documents the user cannot read. MemoryStore therefore builds IDF from the
authorised candidate set; PostgresStore uses ts_rank_cd, which has no corpus statistics.
"""

from __future__ import annotations

import math
import threading
from collections import Counter
from typing import Protocol

from .models import Chunk, Level, Principal
from .text import cosine, tokens


class Store(Protocol):
    def upsert_document(self, chunks: list[Chunk]) -> None: ...

    def delete_document(self, tenant: str, doc_id: str) -> int: ...

    def update_access(self, tenant: str, doc_id: str, acl: frozenset[str], classification: Level) -> int: ...

    def search(self, p: Principal, query: str, query_vec: list[float], k: int) -> list[tuple[Chunk, float]]: ...

    def count(self, tenant: str) -> int: ...


def hybrid_scores(
    chunks: list[Chunk], lexical: list[float], semantic: list[float], k: int, lexical_weight: float = 0.5
) -> list[tuple[Chunk, float]]:
    """Min-max normalise both signals over the candidates and combine them. Chunks with neither a
    lexical match nor a positive similarity are dropped, so unrelated text is not returned."""
    if not chunks:
        return []

    def norm(xs: list[float]) -> list[float]:
        lo, hi = min(xs), max(xs)
        return [0.0 if hi == lo else (x - lo) / (hi - lo) for x in xs]

    lex, sem = norm(lexical), norm(semantic)
    scored = [
        (c, lexical_weight * a + (1 - lexical_weight) * b)
        for c, a, b, raw_lex, raw_sem in zip(chunks, lex, sem, lexical, semantic, strict=True)
        if raw_lex > 0 or raw_sem > 0.15
    ]
    scored.sort(key=lambda x: (-x[1], x[0].key))
    return scored[:k]


def bm25(candidates: list[Chunk], query: str) -> list[float]:
    """BM25 (k1=1.2, b=0.75) with statistics from the candidate set only."""
    q = tokens(query)
    docs = [tokens(c.title + " " + c.text) for c in candidates]
    n = len(docs)
    avg = sum(len(d) for d in docs) / max(n, 1) or 1.0
    df = Counter(t for d in docs for t in set(d))
    out = []
    for d in docs:
        tf = Counter(d)
        s = 0.0
        for t in q:
            if tf[t]:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * tf[t] * 2.2 / (tf[t] + 1.2 * (1 - 0.75 + 0.75 * len(d) / avg))
        out.append(s)
    return out


class MemoryStore:
    def __init__(self) -> None:
        self._chunks: dict[tuple[str, str], list[Chunk]] = {}
        self._lock = threading.Lock()

    def upsert_document(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        tenant, doc_id = chunks[0].tenant, chunks[0].doc_id
        if any(c.tenant != tenant or c.doc_id != doc_id for c in chunks):
            raise ValueError("all chunks of one upsert must belong to the same document and tenant")
        with self._lock:
            self._chunks[(tenant, doc_id)] = list(chunks)

    def delete_document(self, tenant: str, doc_id: str) -> int:
        with self._lock:
            return len(self._chunks.pop((tenant, doc_id), []))

    def update_access(self, tenant: str, doc_id: str, acl: frozenset[str], classification: Level) -> int:
        with self._lock:
            chunks = self._chunks.get((tenant, doc_id), [])
            for c in chunks:
                c.acl = acl
                c.classification = classification
            return len(chunks)

    def search(self, p: Principal, query: str, query_vec: list[float], k: int) -> list[tuple[Chunk, float]]:
        with self._lock:
            allowed = [c for (t, _), cs in self._chunks.items() if t == p.tenant for c in cs if c.visible_to(p)]
        return hybrid_scores(allowed, bm25(allowed, query), [cosine(query_vec, c.embedding) for c in allowed], k)

    def count(self, tenant: str) -> int:
        with self._lock:
            return sum(len(cs) for (t, _), cs in self._chunks.items() if t == tenant)
