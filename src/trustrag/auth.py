"""Bearer tokens (JWT, HS256) carrying the caller's tenant, groups, clearance and roles.

The tenant a request is served for comes from the verified token only, never from the request body.
HS256 with a shared secret keeps the demo self-contained; in production put an identity provider in
front (OIDC) and map its claims to `Principal` the same way.

Verification rejects: other algorithms (including "none"), bad signatures (constant-time compare),
missing or past `exp`, wrong `iss`, and names outside a safe character set (so a crafted group name
cannot smuggle extra ACL entries, for example "x,group:hr").
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import time
from typing import Any

from .models import Level, Principal

NAME = re.compile(r"^[A-Za-z0-9._@-]{1,128}$")
ISSUER = "trustrag"


class AuthError(Exception):
    pass


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _check_secret(secret: str) -> bytes:
    raw = secret.encode()
    if len(raw) < 32:
        raise AuthError("token secret must be at least 32 bytes")
    return raw


def issue_token(secret: str, principal: Principal, ttl_seconds: int = 3600, now: float | None = None) -> str:
    now = time.time() if now is None else now
    header = {"alg": "HS256", "typ": "JWT"}
    claims = {
        "iss": ISSUER,
        "sub": principal.user,
        "tenant": principal.tenant,
        "groups": sorted(principal.groups),
        "clearance": principal.clearance.name.lower(),
        "roles": sorted(principal.roles),
        "iat": int(now),
        "exp": int(now + ttl_seconds),
    }
    signing_input = (
        _b64(json.dumps(header, separators=(",", ":")).encode())
        + "."
        + _b64(json.dumps(claims, separators=(",", ":")).encode())
    )
    sig = hmac.new(_check_secret(secret), signing_input.encode(), hashlib.sha256).digest()
    return signing_input + "." + _b64(sig)


def _name(value: Any, what: str) -> str:
    if not isinstance(value, str) or not NAME.match(value):
        raise AuthError(f"invalid {what}")
    return value


def verify_token(secret: str, token: str, now: float | None = None) -> Principal:
    now = time.time() if now is None else now
    try:
        h64, c64, s64 = token.split(".")
        header = json.loads(_unb64(h64))
        claims = json.loads(_unb64(c64))
        sig = _unb64(s64)
    except (ValueError, json.JSONDecodeError) as e:
        raise AuthError("malformed token") from e
    if not isinstance(header, dict) or header.get("alg") != "HS256":
        raise AuthError("unsupported token algorithm")
    expected = hmac.new(_check_secret(secret), f"{h64}.{c64}".encode(), hashlib.sha256).digest()
    if not hmac.compare_digest(sig, expected):
        raise AuthError("bad token signature")
    if not isinstance(claims, dict) or claims.get("iss") != ISSUER:
        raise AuthError("wrong token issuer")
    exp = claims.get("exp")
    if not isinstance(exp, int) or exp <= now:
        raise AuthError("token expired or without exp")
    groups = claims.get("groups", [])
    roles = claims.get("roles", [])
    if not isinstance(groups, list) or not isinstance(roles, list):
        raise AuthError("groups and roles must be lists")
    try:
        clearance = Level.parse(claims.get("clearance", "internal"))
    except ValueError as e:
        raise AuthError("invalid clearance") from e
    return Principal(
        tenant=_name(claims.get("tenant"), "tenant"),
        user=_name(claims.get("sub"), "subject"),
        groups=frozenset(_name(g, "group") for g in groups),
        clearance=clearance,
        roles=frozenset(_name(r, "role") for r in roles),
    )
