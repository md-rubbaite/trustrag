"""HTTP API (FastAPI). The caller's tenant and permissions come only from the bearer token.

POST   /v1/query                     {question, k}             any authenticated user
POST   /v1/documents                 {id, title, text, acl, classification}   tenant admin
PUT    /v1/documents/{id}/access     {acl, classification}     tenant admin
DELETE /v1/documents/{id}                                      tenant admin
GET    /healthz
"""

import os
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .auth import AuthError, verify_token
from .models import Document, Level, Principal
from .service import Forbidden, RagService


class QueryIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=8)


class DocumentIn(BaseModel):
    id: str = Field(pattern=r"^[A-Za-z0-9._-]{1,128}$")
    title: str = Field(min_length=1, max_length=300)
    text: str = Field(min_length=1, max_length=200_000)
    acl: list[str] = Field(min_length=1, max_length=200)
    classification: str = "internal"


class AccessIn(BaseModel):
    acl: list[str] = Field(min_length=1, max_length=200)
    classification: str = "internal"


def create_app(service: RagService, secret: str | None = None) -> FastAPI:
    secret = secret or os.environ.get("TRUSTRAG_TOKEN_SECRET", "")
    if len(secret.encode()) < 32:
        raise RuntimeError("set TRUSTRAG_TOKEN_SECRET (at least 32 bytes)")
    app = FastAPI(title="TrustRAG", version="0.1.0", docs_url="/docs", redoc_url=None)

    def principal(authorization: Annotated[str | None, Header()] = None) -> Principal:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(401, "bearer token required", headers={"WWW-Authenticate": "Bearer"})
        try:
            return verify_token(secret, authorization.removeprefix("Bearer ").strip())
        except AuthError as e:
            raise HTTPException(401, str(e), headers={"WWW-Authenticate": "Bearer"}) from e

    Caller = Annotated[Principal, Depends(principal)]

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/query")
    def query(body: QueryIn, p: Caller) -> dict:
        return service.query(p, body.question, body.k).to_json()

    @app.post("/v1/documents", status_code=201)
    def ingest(body: DocumentIn, p: Caller) -> dict:
        try:
            doc = Document(
                body.id, p.tenant, body.title, body.text, frozenset(body.acl), Level.parse(body.classification)
            )
            r = service.ingest(p, doc)
        except Forbidden as e:
            raise HTTPException(403, str(e)) from e
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        return {"doc_id": r.doc_id, "chunks": r.chunks, "redactions": r.redactions, "flagged_chunks": r.flagged_chunks}

    @app.put("/v1/documents/{doc_id}/access")
    def access(doc_id: str, body: AccessIn, p: Caller) -> dict:
        try:
            n = service.set_access(p, doc_id, frozenset(body.acl), Level.parse(body.classification))
        except Forbidden as e:
            raise HTTPException(403, str(e)) from e
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        if n == 0:
            raise HTTPException(404, "document not found")
        return {"doc_id": doc_id, "chunks_updated": n}

    @app.delete("/v1/documents/{doc_id}", status_code=204)
    def delete(doc_id: str, p: Caller) -> None:
        try:
            n = service.delete(p, doc_id)
        except Forbidden as e:
            raise HTTPException(403, str(e)) from e
        if n == 0:
            raise HTTPException(404, "document not found")

    return app
