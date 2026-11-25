"""Core types: who is asking (Principal) and what may be retrieved (Document, Chunk).

Access model (deny by default):

- Every document belongs to exactly one tenant. A principal can never see another tenant's data,
  whatever the ACL says.
- A document's ACL is a set of entries: ``user:<id>``, ``group:<name>`` or ``tenant:*`` (everyone in
  the tenant). An empty ACL means nobody can retrieve the document.
- A document has a classification; a principal has a clearance. Both use the same ordered levels.
  Retrieval requires ACL match AND clearance >= classification.

Text inside documents never changes any of this. Authorisation is decided from metadata only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import IntEnum

SAFE_NAME = re.compile(r"^[A-Za-z0-9._@-]{1,128}$")


class Level(IntEnum):
    PUBLIC = 0
    INTERNAL = 1
    CONFIDENTIAL = 2
    RESTRICTED = 3

    @classmethod
    def parse(cls, value: str | int | Level) -> Level:
        if isinstance(value, Level):
            return value
        if isinstance(value, int):
            return cls(value)
        try:
            return cls[str(value).strip().upper()]
        except KeyError as e:
            raise ValueError(f"unknown level '{value}' (use {', '.join(m.name.lower() for m in cls)})") from e


TENANT_WIDE = "tenant:*"


def valid_acl_entry(entry: str) -> bool:
    if entry == TENANT_WIDE:
        return True
    kind, _, name = entry.partition(":")
    return kind in ("user", "group") and bool(SAFE_NAME.match(name))


@dataclass(frozen=True)
class Principal:
    tenant: str
    user: str
    groups: frozenset[str] = frozenset()
    clearance: Level = Level.INTERNAL
    roles: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        for value in (self.tenant, self.user, *self.groups, *self.roles):
            if not SAFE_NAME.match(value):
                raise ValueError(f"unsafe principal name {value!r}")

    def entries(self) -> frozenset[str]:
        """ACL entries that grant this principal access."""
        return frozenset({f"user:{self.user}", TENANT_WIDE, *(f"group:{g}" for g in self.groups)})

    def is_admin(self) -> bool:
        return "admin" in self.roles


@dataclass
class Document:
    id: str
    tenant: str
    title: str
    text: str
    acl: frozenset[str]
    classification: Level = Level.INTERNAL
    source: str = ""

    def __post_init__(self) -> None:
        bad = [e for e in self.acl if not valid_acl_entry(e)]
        if bad:
            raise ValueError(f"document {self.id}: invalid ACL entries {sorted(bad)}")
        if not self.tenant or not self.id:
            raise ValueError("document needs an id and a tenant")


@dataclass
class Chunk:
    doc_id: str
    tenant: str
    n: int
    title: str
    text: str
    acl: frozenset[str]
    classification: Level
    embedding: list[float] = field(default_factory=list, repr=False)
    flags: tuple[str, ...] = ()  # e.g. "instruction-like-text" found at ingestion

    @property
    def key(self) -> str:
        return f"{self.doc_id}#{self.n}"

    def visible_to(self, p: Principal) -> bool:
        return self.tenant == p.tenant and bool(self.acl & p.entries()) and self.classification <= p.clearance
