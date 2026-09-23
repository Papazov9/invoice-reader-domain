"""Domain models.

TextUnit/DocCandidate are the flat ingestion intermediate; Invoice is the richer
persisted object. Money uses Decimal so arithmetic validation is exact.
"""

from __future__ import annotations

from decimal import Decimal
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class TextUnit(BaseModel):
    # kind drives how the NLP layer treats the text: field names get
    # identifier-split, values are preserved.
    kind: str = Field(description="fieldName | value | context")
    text: str


class DocCandidate(BaseModel):
    # Schema-agnostic: any XML/OCR source collapses to this shape.
    id: str
    units: list[TextUnit] = Field(default_factory=list)

    def joined_text(self) -> str:
        return " ".join(u.text for u in self.units if u.text)


class Party(BaseModel):
    name: Optional[str] = None
    vat_number: Optional[str] = None  # e.g. BG123456789
    eik: Optional[str] = None  # Bulgarian company id (9 digits)
    address: Optional[str] = None
    # Where the party data came from: extracted from the document, looked up in the
    # commercial register, or a merge of both. recovered_fields lists the keys filled
    # or corrected from the register so the UI can flag them.
    source: str = "extracted"  # extracted | register | merged | vision
    recovered_fields: list[str] = Field(default_factory=list)


class LineItem(BaseModel):
    description: Optional[str] = None
    quantity: Optional[Decimal] = None
    unit_price: Optional[Decimal] = None
    amount: Optional[Decimal] = None
    code: Optional[str] = None  # supplier article/lot number (Номер/Партида)
    unit: Optional[str] = None  # unit of measure (Мярка), e.g. "бр.", "кг", "л"


class TaxLine(BaseModel):
    # VAT breakdown row: base taxed at rate yields amount.
    rate: Decimal  # fractional, e.g. Decimal("0.20") for 20%
    base: Optional[Decimal] = None
    amount: Optional[Decimal] = None


class ExtractedField(BaseModel):
    value: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class Invoice(BaseModel):
    id: str
    source: str = Field(default="unknown", description="xml | ocr | csv | manual | ...")

    # Bulgarian accounting document classification.
    doc_type: str = "invoice"      # see tools/ingest/document_types.DocumentType
    direction: str = "unknown"     # sale | purchase | unknown
    reverse_charge: bool = False   # обратно начисляване (чл. 117 / чл. 163 ЗДДС)

    number: Optional[str] = None
    date: Optional[str] = None  # ISO YYYY-MM-DD when normalizable
    currency: str = "BGN"

    supplier: Party = Field(default_factory=Party)
    recipient: Party = Field(default_factory=Party)
    # Which party the document is read from. "supplier" preserves the historical
    # grouping behaviour; "auto" lets ingestion resolve it from the direction.
    perspective: str = "supplier"  # supplier | recipient | auto

    line_items: list[LineItem] = Field(default_factory=list)
    tax_lines: list[TaxLine] = Field(default_factory=list)

    net_amount: Optional[Decimal] = None
    vat_amount: Optional[Decimal] = None
    total_amount: Optional[Decimal] = None

    # Per-field extraction confidence, keyed by field name.
    field_confidence: dict[str, float] = Field(default_factory=dict)
    # True when the vision model contributed any field (poor scan / photo fallback), so
    # the UI can flag the document as model-read.
    vision_used: bool = False
    # Document-type specific fields that don't fit the invoice shape: fiscal device
    # number, IBAN, statement period/balances, customs MRN, etc.
    extra: dict[str, str] = Field(default_factory=dict)

    # Identified counterparty company (set during ingestion, see tools/ingest/company.py).
    company_key: Optional[str] = None
    company_name: Optional[str] = None

    def to_doc_candidate(self) -> DocCandidate:
        """Project the invoice back to a DocCandidate for comparison/dedup."""
        units: list[TextUnit] = []

        def add(field_name: str, value: object) -> None:
            if value not in (None, "", []):
                units.append(TextUnit(kind="fieldName", text=field_name))
                units.append(TextUnit(kind="value", text=str(value)))

        add("invoiceNumber", self.number)
        add("documentDate", self.date)
        add("supplierName", self.supplier.name)
        add("supplierVAT", self.supplier.vat_number)
        add("supplierEIK", self.supplier.eik)
        add("recipientName", self.recipient.name)
        add("recipientVAT", self.recipient.vat_number)
        add("netAmount", self.net_amount)
        add("vatAmount", self.vat_amount)
        add("totalAmount", self.total_amount)
        return DocCandidate(id=self.id, units=units)


class MatchEvidence(BaseModel):
    # Interpretable breakdown of why two documents scored as they did.
    word_similarity: float
    char_similarity: float
    fused_score: float

    def as_strings(self) -> list[str]:
        return [
            f"tfidf_word={self.word_similarity:.3f}",
            f"tfidf_char={self.char_similarity:.3f}",
            f"fused={self.fused_score:.3f}",
        ]


class DuplicateMatch(BaseModel):
    candidate_id: str
    score: float = Field(ge=0.0, le=1.0)
    is_duplicate: bool
    evidence: MatchEvidence


# companies (per-company working sets)


class Company(BaseModel):
    # An identified business entity, used to group invoices into working sets.
    key: str  # stable identity: vat:... / eik:... / name:... / unknown
    name: Optional[str] = None
    vat: Optional[str] = None
    eik: Optional[str] = None
    invoice_count: int = 0


class CompanyInfo(BaseModel):
    # Registry record for an EIK, used to recover or confirm a counterparty.
    eik: str
    name: Optional[str] = None
    vat_number: Optional[str] = None
    status: Optional[str] = None
    country: Optional[str] = None
    city: Optional[str] = None
    zip_code: Optional[str] = None
    address_line1: Optional[str] = None
    manager: Optional[str] = None
    source: str = "register"


class CompanyGroup(BaseModel):
    company: Company
    invoices: list[Invoice] = Field(default_factory=list)


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ValidationResult(BaseModel):
    rule_id: str
    passed: bool
    severity: Severity
    message: str
    evidence: dict[str, str] = Field(default_factory=dict)
