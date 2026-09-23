"""Top-level document extraction: classify, then route to type-aware handling.

Every accounting document collapses to the same Invoice shape so the API, persistence
and the frontend stay uniform; type-specific values (fiscal device number, IBAN,
statement balances, customs MRN, ...) live in Invoice.extra. The invoice family reuses
the full invoice extractor unchanged, so that path is identical to before.

extract_from_pdf_bytes is the OCR coordinator: recognise, extract, optionally consult
the vision model on a poor scan, then recover counterparties from the register.
"""

from __future__ import annotations

import re

from app.domain import Invoice

from .company import tag_company
from .document_types import DocumentType
from .invoice_extractor import extract_invoice_from_text, recover_parties
from .llm_assist import assist_fields, merge_fields, should_assist
from .ocr import extract_ocr_from_image_bytes, extract_ocr_from_pdf_bytes
from .vision_extract import extract_invoice_via_vision, merge_into_invoice, should_use_vision


def extract_document(
    text: str,
    doc_id: str,
    source: str = "ocr",
    *,
    perspective: str = "auto",
    low_conf_tokens: set[str] | None = None,
) -> Invoice:
    """Extract a typed Invoice, augmenting it with fields specific to its document type."""
    invoice = extract_invoice_from_text(
        text, doc_id, source, perspective=perspective, low_conf_tokens=low_conf_tokens
    )
    augment = _AUGMENTERS.get(invoice.doc_type)
    if augment:
        augment(invoice, text)
        # The augmenter may have added parties/EIKs (e.g. customs exporter/importer);
        # recover their canonical names from the register and re-tag the company.
        recover_parties(invoice)
        tag_company(invoice)

    # Hard cases (no type, missing parties): let the LLM fill weak NON-AMOUNT fields,
    # then re-recover/re-tag. Amounts stay rule-computed. No-ops without a model.
    if should_assist(invoice):
        fields = assist_fields(text, invoice.doc_type)
        if fields:
            merge_fields(invoice, fields)
            recover_parties(invoice)
            tag_company(invoice)
    return invoice


def extract_from_pdf_bytes(
    content: bytes,
    doc_id: str,
    source: str = "ocr",
    *,
    perspective: str = "auto",
    use_vision: bool = True,
) -> Invoice:
    """OCR a PDF then extract; consult the vision model when the scan is poor. The slow
    vision fallback can be disabled (e.g. for bulk uploads) via use_vision=False."""
    ocr = extract_ocr_from_pdf_bytes(content)
    invoice = _ocr_then_extract(ocr, doc_id, source, perspective, use_vision)
    # A digital PDF read via its embedded text has no page images, so the normal vision
    # fallback can't run. When such a read comes out critically incomplete (an atypical
    # layout the rules couldn't parse), render the pages now and let vision fill the gaps.
    if (use_vision and not invoice.vision_used and not ocr.page_images
            and should_use_vision(invoice, 1.0)):
        from .ocr import render_pdf_page_images

        # Just the first page: it carries the header, parties and totals for essentially all
        # invoices, and a vision model reads one clean page far better than several (extra
        # pages are usually duplicate copies that only degrade the reading).
        images = render_pdf_page_images(content, max_pages=1)
        if images:
            _apply_vision(invoice, images, doc_id)
    return invoice


def extract_from_image_bytes(
    content: bytes,
    doc_id: str,
    source: str = "ocr",
    *,
    perspective: str = "auto",
    use_vision: bool = True,
) -> Invoice:
    """OCR a photographed/scanned image then extract; same pipeline as a scanned PDF."""
    return _ocr_then_extract(extract_ocr_from_image_bytes(content), doc_id, source, perspective, use_vision)


def _ocr_then_extract(ocr, doc_id: str, source: str, perspective: str, use_vision: bool = True) -> Invoice:
    invoice = extract_document(
        ocr.text, doc_id, source, perspective=perspective, low_conf_tokens=ocr.low_conf_tokens
    )
    if use_vision and ocr.page_images and should_use_vision(invoice, ocr.mean_conf):
        _apply_vision(invoice, ocr.page_images, doc_id)
    return invoice


def _apply_vision(invoice: Invoice, page_images: list[bytes], doc_id: str) -> None:
    """Consult the vision model on the given page images and merge what it reads, then
    re-recover parties (a surfaced EIK) and re-tag the company."""
    fields = extract_invoice_via_vision(page_images, doc_id)
    if fields:
        merge_into_invoice(invoice, fields)
        recover_parties(invoice)
        tag_company(invoice)


def _first(text: str, *patterns: str) -> str | None:
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return (m.group(1) if m.groups() else m.group(0)).strip()
    return None


def _augment_receipt(invoice: Invoice, text: str) -> None:
    device = _first(
        text,
        r"(?:ФУ|ФП|ФМ|Фискално\s+устройство|Фиск\.?\s*устр\.?)\s*[:№]?\s*([A-ZА-Я]{2}\d{6,})",
        r"\b([A-ZА-Я]{2}\d{6}-\d{4})\b",
    )
    if device:
        invoice.extra["fiscal_device"] = device


_IBAN = r"BG\d{2}[A-Z]{4}[0-9A-Z]{14}"


def _augment_bank_statement(invoice: Invoice, text: str) -> None:
    iban = _first(text, rf"\b({_IBAN})\b")
    if iban:
        invoice.extra["iban"] = iban.replace(" ", "")
    opening = _first(text, r"Начално\s+салдо\s*[:\-]?\s*(-?[\d  ]+[.,]\d{2})")
    closing = _first(text, r"Крайно\s+салдо\s*[:\-]?\s*(-?[\d  ]+[.,]\d{2})")
    if opening:
        invoice.extra["opening_balance"] = opening
    if closing:
        invoice.extra["closing_balance"] = closing
    period = _first(text, r"(?:период|за\s+период)\s*[:\-]?\s*(.{4,40}?\d{4})")
    if period:
        invoice.extra["period"] = period


def _augment_customs(invoice: Invoice, text: str) -> None:
    # MRN: 2-digit year, 2-letter country, 14 alphanumerics. It is the document's id.
    mrn = _first(text, r"\b(\d{2}[A-Z]{2}[A-Z0-9]{14})\b", r"MRN\s*[:№]?\s*([A-Z0-9]{14,18})")
    if mrn:
        invoice.extra["mrn"] = mrn
        # The MRN is the customs document's reference; it outranks any invoice number
        # the generic pass may have picked up from a referenced commercial invoice.
        invoice.number = mrn
        invoice.field_confidence["number"] = 0.9

    # Exporter -> supplier, importer -> recipient. Names end before the country code.
    exporter = _first(text, r"Износител\s+(.+?)\s+[A-Z]{2},")
    importer = _first(text, r"Вносител\s+(.+?)\s+[A-Z]{2},")
    if exporter and not invoice.supplier.name:
        invoice.supplier.name = exporter
    if importer and not invoice.recipient.name:
        invoice.recipient.name = importer
    # Importer EORI carries the BG EIK; recovery can then restore the full legal name.
    imp_eik = _first(text, r"Вносител[^\n]*?BG[A-Z]?(\d{9})")
    if imp_eik and not invoice.recipient.eik:
        invoice.recipient.eik = imp_eik

    value = _first(text, r"(?:Митническа\s+стойност|Customs\s+value)\s*\w*\s*([\d  ]+[.,]\d{2})")
    if value:
        invoice.extra["customs_value"] = value
    vat = _first(text, r"B00\s*-?\s*ДДС\s+([\d  ]+[.,]\d{2})")
    if vat:
        invoice.extra["vat"] = vat


def _augment_expense_protocol(invoice: Invoice, text: str) -> None:
    article = _first(text, r"чл\.?\s*(\d+[а-я]?)")
    if article:
        invoice.extra["article"] = f"чл. {article}"
    if invoice.reverse_charge:
        invoice.extra["reverse_charge"] = "true"


_AUGMENTERS = {
    DocumentType.FISCAL_RECEIPT.value: _augment_receipt,
    DocumentType.BANK_STATEMENT.value: _augment_bank_statement,
    DocumentType.CUSTOMS_DECLARATION.value: _augment_customs,
    DocumentType.EXPENSE_REPORT.value: _augment_expense_protocol,
    DocumentType.PROTOCOL.value: _augment_expense_protocol,
}
