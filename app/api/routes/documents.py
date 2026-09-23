"""Ingestion endpoints: XML extraction, text extraction, and PDF/OCR upload."""

from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.api.schemas import (
    CompanyLookupResponse,
    ExtractCsvRequest,
    ExtractCsvResponse,
    ExtractTextRequest,
    ExtractXmlRequest,
    ExtractXmlResponse,
    InvoiceResponse,
)
from app.tools.ingest import (
    extract_document,
    extract_from_image_bytes,
    extract_from_pdf_bytes,
    group_by_company,
    invoices_from_xml_content,
    lookup_company,
    lookup_status,
    ocr_status,
    parse_csv,
    parse_xml,
    validate_eik,
)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/extract-xml", response_model=ExtractXmlResponse)
def extract_xml(req: ExtractXmlRequest) -> ExtractXmlResponse:
    """Parse any XML (XXE-safe) into typed invoices grouped by company. Handles the
    Controlisy/ExportedData schema and generic record-based schemas (SAP, UBL, ...)."""
    docs = parse_xml(req.xml, req.label)
    invoices = invoices_from_xml_content(req.xml, req.label)
    groups = group_by_company(invoices)
    return ExtractXmlResponse(documents=docs, invoices=invoices, groups=groups)


@router.post("/extract-csv", response_model=ExtractCsvResponse)
def extract_csv(req: ExtractCsvRequest) -> ExtractCsvResponse:
    """Parse a CSV export / VAT ledger into typed invoices grouped by company."""
    invoices = parse_csv(req.csv, req.label)
    return ExtractCsvResponse(invoices=invoices, groups=group_by_company(invoices))


@router.post("/extract-text", response_model=InvoiceResponse)
def extract_text(req: ExtractTextRequest) -> InvoiceResponse:
    """Extract structured fields from raw text (e.g. pasted OCR output), routing by
    detected document type."""
    invoice = extract_document(
        req.text, req.doc_id, req.source, perspective=req.perspective
    )
    return InvoiceResponse(invoice=invoice)


@router.get("/company/{eik}", response_model=CompanyLookupResponse)
def company_lookup(eik: str) -> CompanyLookupResponse:
    """Look up a counterparty in the commercial register by EIK."""
    if not validate_eik(eik):
        raise HTTPException(status_code=400, detail="invalid EIK")
    if not lookup_status().get("available"):
        raise HTTPException(status_code=503, detail="company lookup not available")
    info = lookup_company(eik)
    if info is None:
        raise HTTPException(status_code=404, detail=f"company {eik} not found")
    return CompanyLookupResponse(company=info)


@router.post("/extract-pdf", response_model=InvoiceResponse)
async def extract_pdf(
    file: UploadFile = File(...),
    perspective: str = Form("auto"),
    vision: bool = Form(True),
) -> InvoiceResponse:
    """OCR a PDF (Bulgarian+English), preprocess the pages, and extract structured
    fields; a poor scan falls back to the vision model and the register. Set vision=false
    (e.g. for bulk uploads) to skip the slow vision fallback."""
    if not ocr_status().get("available"):
        raise HTTPException(status_code=503, detail="OCR not available on this server")
    content = await file.read()
    doc_id = (file.filename or "invoice").rsplit(".", 1)[0]
    try:
        invoice = extract_from_pdf_bytes(content, doc_id, source="ocr", perspective=perspective, use_vision=vision)
    except Exception as exc:  # pragma: no cover - depends on OCR env
        raise HTTPException(status_code=422, detail=f"OCR failed: {exc}") from exc
    return InvoiceResponse(invoice=invoice)


@router.post("/extract-image", response_model=InvoiceResponse)
async def extract_image(
    file: UploadFile = File(...),
    perspective: str = Form("auto"),
    vision: bool = Form(True),
) -> InvoiceResponse:
    """OCR a photographed/scanned image (JPG/PNG/TIFF…) — EXIF-corrected, preprocessed and
    column-reflowed like a PDF page — and extract structured fields; a poor photo falls
    back to the vision model and the register. Set vision=false to skip it (bulk)."""
    if not ocr_status().get("available"):
        raise HTTPException(status_code=503, detail="OCR not available on this server")
    content = await file.read()
    doc_id = (file.filename or "image").rsplit(".", 1)[0]
    try:
        invoice = extract_from_image_bytes(content, doc_id, source="ocr", perspective=perspective, use_vision=vision)
    except Exception as exc:  # pragma: no cover - depends on OCR env
        raise HTTPException(status_code=422, detail=f"image OCR failed: {exc}") from exc
    return InvoiceResponse(invoice=invoice)
