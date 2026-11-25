"""Tokenising, chunking and a deterministic hashing embedder.

The hashing embedder is lexical (words + character trigrams hashed into a fixed-size vector). It is
used so the project runs offline and tests are reproducible. It is not a semantic model: swap in a
real embedding model through the `Embedder` protocol for production quality.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol

WORD = re.compile(r"[a-z0-9]+")
_STOP_WORDS = (
    "a an and are as at be by for from has have how i in is it its of on or our the their this to was we what "
    "when where which who will with you your do does can"
)
STOP = frozenset(_STOP_WORDS.split(" "))


def _stem(t: str) -> str:
    """Very light normalisation: plural 's' only (engineers -> engineer, bands -> band)."""
    return t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t


def tokens(text: str) -> list[str]:
    return [_stem(t) for t in WORD.findall(text.lower()) if t not in STOP]


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n{2,}", text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, max_words: int = 120, overlap_sentences: int = 1) -> list[str]:
    """Split on sentence boundaries into chunks of at most ~max_words, overlapping by whole sentences."""
    sents = sentences(text)
    chunks: list[str] = []
    i = 0
    while i < len(sents):
        words, j = 0, i
        while j < len(sents) and (words == 0 or words + len(sents[j].split()) <= max_words):
            words += len(sents[j].split())
            j += 1
        chunks.append(" ".join(sents[i:j]))
        if j >= len(sents):
            break
        i = max(i + 1, j - overlap_sentences)
    return chunks


class Embedder(Protocol):
    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class HashingEmbedder:
    def __init__(self, dim: int = 256):
        self.dim = dim

    def _bucket(self, feature: str) -> tuple[int, float]:
        h = hashlib.blake2b(feature.encode(), digest_size=8).digest()
        n = int.from_bytes(h, "big")
        return n % self.dim, 1.0 if (n >> 63) & 1 else -1.0

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out = []
        for text in texts:
            v = [0.0] * self.dim
            toks = tokens(text)
            feats = toks + [f"#{t[i : i + 3]}" for t in toks if len(t) > 3 for i in range(len(t) - 2)]
            for f in feats:
                idx, sign = self._bucket(f)
                v[idx] += sign * (1.0 if not f.startswith("#") else 0.5)
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))
