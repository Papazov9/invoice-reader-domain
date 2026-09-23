"""Request/response models for the HTTP API (kept separate from domain models)."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from app.domain import (
    CompanyGroup,
    CompanyInfo,
    DocCandidate,
    DuplicateMatch,
    Invoice,
    MatchEvidence,
    ValidationResult,
)
from app.rag.base import Citation, RetrievedChunk


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    ocr: dict[str, object]


class ExtractXmlRequest(BaseModel):
    xml: str = Field(description="Raw XML content")
    label: str = ""


class ExtractXmlResponse(BaseModel):
    documents: list[DocCandidate]
    invoices: list[Invoice]
    groups: list[CompanyGroup]


class ExtractCsvRequest(BaseModel):
    csv: str = Field(description="Raw CSV content")
    label: str = "csv"


class ExtractCsvResponse(BaseModel):
    invoices: list[Invoice]
    groups: list[CompanyGroup]


class GroupRequest(BaseModel):
    invoices: list[Invoice]


class GroupResponse(BaseModel):
    groups: list[CompanyGroup]


class ExtractTextRequest(BaseModel):
    text: str = Field(description="Raw invoice text (e.g. OCR output)")
    doc_id: str = "invoice"
    source: str = "manual"
    # Which party the document is read from: "auto" infers from sale/purchase
    # direction, "supplier"/"recipient" force the choice.
    perspective: str = "auto"


class InvoiceResponse(BaseModel):
    invoice: Invoice


class CompanyLookupResponse(BaseModel):
    company: CompanyInfo


class ValidateResponse(BaseModel):
    invoice_id: str
    results: list[ValidationResult]
    is_valid: bool


class CompareRequest(BaseModel):
    a: DocCandidate
    b: DocCandidate


class CompareResponse(BaseModel):
    evidence: MatchEvidence


class DuplicatesRequest(BaseModel):
    query: DocCandidate
    candidates: list[DocCandidate]
    top_k: int = Field(default=5, ge=1, le=100)


class DuplicatesResponse(BaseModel):
    matches: list[DuplicateMatch]


# auth


class RegisterRequest(BaseModel):
    email: str
    password: str = Field(min_length=8)
    tenant_name: str = Field(min_length=1, description="Firm / workspace name")


class LoginRequest(BaseModel):
    email: str
    password: str


class UserOut(BaseModel):
    id: str
    email: str
    role: str
    tenant_id: str
    tenant_name: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


# workspace (persisted, tenant-scoped)


class PersistRequest(BaseModel):
    invoices: list[Invoice]


class PersistResponse(BaseModel):
    stored: int


class WorkspaceInvoicesResponse(BaseModel):
    invoices: list[Invoice]


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=50)
    company_key: Optional[str] = None


class SearchHit(BaseModel):
    invoice: Invoice
    score: float


class SearchResponse(BaseModel):
    hits: list[SearchHit]


# RAG + chat orchestration


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = Field(default=6, ge=1, le=30)
    company_key: Optional[str] = None


class RetrieveResponse(BaseModel):
    chunks: list[RetrievedChunk]


class ChatTurn(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatTurn] = Field(default_factory=list)
    company_key: Optional[str] = None
    top_k: int = Field(default=6, ge=1, le=30)


class ChatResponse(BaseModel):
    reply: str
    citations: list[Citation] = Field(default_factory=list)
    model: str
    context: list[RetrievedChunk] = Field(default_factory=list)
    # agent fields (Phase 2)
    cards: list[dict] = Field(default_factory=list)
    refused: bool = False
    tool_trace: list[dict] = Field(default_factory=list)
