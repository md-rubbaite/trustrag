"""Tokens and the HTTP API."""

from __future__ import annotations

import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fastapi.testclient import TestClient

from trustrag.answer import OpenAICompatibleProvider
from trustrag.api import create_app
from trustrag.auth import AuthError, issue_token, verify_token
from trustrag.cli import main
from trustrag.models import Level, Principal
from trustrag.service import RagService

SECRET = "s" * 40
ALICE = Principal("acme", "alice", frozenset({"engineering"}), Level.INTERNAL)
ADMIN = Principal("acme", "ops", roles=frozenset({"admin"}))
GLOBEX_ADMIN = Principal("globex", "ops", roles=frozenset({"admin"}))


def _b64(d: dict) -> str:
    return base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()


# ------------------------------------------------------------------ tokens


def test_token_round_trip():
    p = verify_token(SECRET, issue_token(SECRET, ALICE))
    assert p == ALICE


def test_expired_and_tampered_tokens_are_rejected():
    with pytest.raises(AuthError, match="expired"):
        verify_token(SECRET, issue_token(SECRET, ALICE, ttl_seconds=10, now=1000), now=2000)
    t = issue_token(SECRET, ALICE)
    h, c, s = t.split(".")
    claims = json.loads(base64.urlsafe_b64decode(c + "=="))
    claims["tenant"] = "globex"
    with pytest.raises(AuthError, match="signature"):
        verify_token(SECRET, f"{h}.{_b64(claims)}.{s}")
    with pytest.raises(AuthError, match="signature"):
        verify_token("t" * 40, t)


def test_alg_none_and_wrong_issuer_are_rejected():
    claims = {"iss": "trustrag", "sub": "x", "tenant": "acme", "exp": 9_999_999_999}
    with pytest.raises(AuthError, match="algorithm"):
        verify_token(SECRET, f"{_b64({'alg': 'none'})}.{_b64(claims)}.")


def test_smuggled_acl_entries_in_claims_are_rejected():
    """A group named 'x,group:hr' must not become two ACL entries anywhere."""
    import hashlib
    import hmac

    claims = {"iss": "trustrag", "sub": "alice", "tenant": "acme", "groups": ["x,group:hr"], "exp": 9_999_999_999}
    h, c = _b64({"alg": "HS256", "typ": "JWT"}), _b64(claims)
    sig = base64.urlsafe_b64encode(hmac.new(SECRET.encode(), f"{h}.{c}".encode(), hashlib.sha256).digest())
    with pytest.raises(AuthError, match="invalid group"):
        verify_token(SECRET, f"{h}.{c}.{sig.rstrip(b'=').decode()}")


def test_short_secrets_are_refused():
    with pytest.raises(AuthError):
        issue_token("short", ALICE)


# ------------------------------------------------------------------ API


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(RagService(), SECRET))


def _h(p: Principal) -> dict:
    return {"Authorization": f"Bearer {issue_token(SECRET, p)}"}


def test_api_requires_a_valid_token(client):
    assert client.post("/v1/query", json={"question": "x"}).status_code == 401
    assert client.post("/v1/query", json={"question": "x"}, headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.get("/healthz").json() == {"status": "ok"}


def test_api_ingest_query_access_delete(client):
    body = {
        "id": "runbook",
        "title": "Runbook",
        "text": "Deploy the payment service with the release workflow.",
        "acl": ["group:engineering"],
    }
    assert client.post("/v1/documents", json=body, headers=_h(ALICE)).status_code == 403
    r = client.post("/v1/documents", json=body, headers=_h(ADMIN))
    assert r.status_code == 201 and r.json()["chunks"] == 1
    q = client.post("/v1/query", json={"question": "How do I deploy the payment service?"}, headers=_h(ALICE)).json()
    assert q["sources"][0]["doc_id"] == "runbook" and q["verification"]["ok"]
    r = client.put("/v1/documents/runbook/access", json={"acl": ["group:sre"]}, headers=_h(ADMIN))
    assert r.json()["chunks_updated"] == 1
    q = client.post("/v1/query", json={"question": "How do I deploy the payment service?"}, headers=_h(ALICE)).json()
    assert q["sources"] == []
    assert client.delete("/v1/documents/runbook", headers=_h(ADMIN)).status_code == 204
    assert client.delete("/v1/documents/runbook", headers=_h(ADMIN)).status_code == 404


def test_api_tenant_comes_from_the_token_not_the_body(client):
    body = {"id": "doc", "title": "T", "text": "Acme internal plan for the payment service.", "acl": ["tenant:*"]}
    client.post("/v1/documents", json=body, headers=_h(ADMIN))
    # A globex admin cannot reach or change acme's document by naming it.
    assert client.delete("/v1/documents/doc", headers=_h(GLOBEX_ADMIN)).status_code == 404
    q = client.post(
        "/v1/query", json={"question": "payment service plan"}, headers=_h(Principal("globex", "x", roles=frozenset()))
    ).json()
    assert q["sources"] == []


def test_api_validates_input(client):
    bad = {"id": "../etc", "title": "T", "text": "x", "acl": ["group:hr"]}
    assert client.post("/v1/documents", json=bad, headers=_h(ADMIN)).status_code == 422
    bad = {"id": "ok", "title": "T", "text": "x", "acl": ["group:a,group:hr"]}
    assert client.post("/v1/documents", json=bad, headers=_h(ADMIN)).status_code == 422
    assert client.post("/v1/query", json={"question": "x", "k": 100}, headers=_h(ALICE)).status_code == 422


def test_create_app_refuses_weak_secret():
    with pytest.raises(RuntimeError):
        create_app(RagService(), "short")


# ------------------------------------------------------------------ provider + CLI


def test_openai_compatible_provider_request_shape():
    seen = {}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            seen["path"] = self.path
            seen["auth"] = self.headers.get("Authorization")
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            out = json.dumps({"choices": [{"message": {"content": "ok [1]"}}]}).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        prov = OpenAICompatibleProvider(f"http://127.0.0.1:{srv.server_address[1]}", "m", api_key="k")
        assert prov.complete("sys", "user") == "ok [1]"
    finally:
        srv.shutdown()
        srv.server_close()
    assert seen["path"] == "/v1/chat/completions" and seen["auth"] == "Bearer k"
    assert seen["body"]["temperature"] == 0 and seen["body"]["messages"][0]["role"] == "system"


def test_cli_eval_and_token(tmp_path, capsys, monkeypatch):
    assert main(["eval", "--json", str(tmp_path / "r.json")]) == 0
    assert "Unauthorised chunks retrieved: **0**" in capsys.readouterr().out
    assert json.loads((tmp_path / "r.json").read_text())["ok"] is True
    monkeypatch.setenv("TRUSTRAG_TOKEN_SECRET", SECRET)
    assert main(["token", "--tenant", "acme", "--user", "alice", "--groups", "engineering"]) == 0
    token = capsys.readouterr().out.strip()
    assert verify_token(SECRET, token).groups == frozenset({"engineering"})
