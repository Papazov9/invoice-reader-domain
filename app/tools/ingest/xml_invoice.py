"""Map a schema-agnostic DocCandidate (from XML) into a typed Invoice.

ERP/accounting XML exports use wildly different field names. This resolver recognises
a broad set of EN + BG synonyms (camelCase, snake_case, Cyrillic) and routes each
fieldName/value pair to the right Invoice slot, so XML records become first-class
Invoices with company identity, just like OCR output.
"""

from __future__ import annotations

import re
from decimal import Decimal

from app.domain import DocCandidate, Invoice, Party, TaxLine

from .company import tag_company
from .currency import currency_from_field_name, normalize_currency
from .document_types import (
    DocumentType,
    detect_direction,
    detect_document_type,
    detect_reverse_charge,
)
from .invoice_extractor import clean_amount, normalize_date

_TOKEN = re.compile(r"[^0-9a-zа-я]+", re.UNICODE)


def _tok(name: str) -> str:
    """Normalize a field name to a comparison token (lowercase, alnum only)."""
    return _TOKEN.sub("", name.lower())


# Slot resolution, checked in priority order. Each rule: (slot, predicate(token)).
def _has(*subs: str):
    return lambda t: any(s in t for s in subs)


_SUPPLIER = ("supplier", "contractor", "seller", "vendor", "partner", "доставчик", "продавач")
_RECIPIENT = ("recipient", "customer", "buyer", "client", "получател", "купувач")

_RULES: list[tuple[str, object]] = [
    ("supplier_vat", lambda t: ("vat" in t or "ддс" in t) and _has(*_SUPPLIER)(t)),
    ("recipient_vat", lambda t: ("vat" in t or "ддс" in t) and _has(*_RECIPIENT)(t)),
    ("recipient_eik", lambda t: ("eik" in t or "bulstat" in t) and _has(*_RECIPIENT)(t)),
    ("supplier_eik", lambda t: ("eik" in t or "bulstat" in t or "uic" in t) and _has(*_SUPPLIER)(t)),
    ("supplier_name", lambda t: _has("suppliername", "contractorname", "sellername", "vendorname")(t) or t in _SUPPLIER),
    ("recipient_name", lambda t: _has("recipientname", "customername", "buyername", "clientname")(t) or t in _RECIPIENT),
    ("number", lambda t: t in {"documentnumber", "invoicenumber", "docnumber", "docno", "number", "invno", "invoiceno", "s1"} or (("number" in t or t.endswith("no")) and _has("doc", "invoice", "faktura", "фактура")(t))),
    ("date", lambda t: "date" in t or t in {"datum", "дата"} or _has("издаване", "issuedate", "taxpointdate")(t)),
    ("supplier_eik", lambda t: t in {"eik", "bulstat", "uic", "еик"}),  # bare EIK goes to supplier
    ("net", lambda t: _has("netamount", "taxbase", "danachnaosnova", "данъчнаоснова", "baseamount", "sumawithoutvat", "taxexclusive", "lineextension", "nettotal")(t) or t in {"net", "нето"}),
    ("vat_amount", lambda t: _has("vatamount", "taxamount", "vatvalue", "taxpayable", "vattotal")(t) or t in {"vat", "ддс", "dds"} or ("ддс" in t and "размер" in t) or "размернаданъка" in t),
    ("total", lambda t: "total" in t or _has("обща", "крайнасума", "заплащане", "payableamount", "taxinclusive", "grosstotal", "grandtotal", "grossamount", "grossvalue")(t)),
    ("currency", lambda t: t in {"currency", "valuta", "ccy", "валута", "currencycode"}),
    ("doctype", lambda t: t in {"documenttype", "doctype", "типдокумент", "виддокумент", "типнадокумента", "invoicetype"}),
]


def _resolve_slot(field_name: str) -> str | None:
    t = _tok(field_name)
    if not t:
        return None
    for slot, pred in _RULES:
        if pred(t):  # type: ignore[operator]
            return slot
    return None


# Explicit document-type field values (e.g. SAP InvoiceType="Standard").
_DOCTYPE_VALUES: dict[str, DocumentType] = {
    "standard": DocumentType.INVOICE,
    "invoice": DocumentType.INVOICE,
    "factura": DocumentType.INVOICE,
    "creditnote": DocumentType.CREDIT_NOTE,
    "credit": DocumentType.CREDIT_NOTE,
    "debitnote": DocumentType.DEBIT_NOTE,
    "debit": DocumentType.DEBIT_NOTE,
    "proforma": DocumentType.PROFORMA,
}


def _classify(fields: dict[str, str], blob: str, total: Decimal | None) -> DocumentType:
    raw = _tok(fields.get("doctype", ""))
    if raw in _DOCTYPE_VALUES:
        return _DOCTYPE_VALUES[raw]
    detected = detect_document_type(f"{fields.get('doctype', '')} {blob}")
    if total is not None and total < 0:
        return DocumentType.CREDIT_NOTE
    if detected == DocumentType.OTHER and fields.get("number") and (
        "net" in fields or "total" in fields
    ):
        return DocumentType.INVOICE
    return detected


def doc_candidate_to_invoice(doc: DocCandidate, default_currency: str | None = None) -> Invoice:
    """Resolve a DocCandidate's fieldName/value pairs into a typed, company-tagged Invoice."""
    fields: dict[str, str] = {}
    names: dict[str, str] = {}  # slot to original field name (for currency suffix)
    units = doc.units
    for i, unit in enumerate(units):
        if unit.kind != "fieldName":
            continue
        nxt = units[i + 1] if i + 1 < len(units) else None
        value = nxt.text if (nxt and nxt.kind == "value") else ""
        if not value:
            continue
        slot = _resolve_slot(unit.text)
        if slot and slot not in fields:  # first match wins
            fields[slot] = value.strip()
            names[slot] = unit.text

    def dec(slot: str) -> Decimal | None:
        return clean_amount(fields[slot]) if slot in fields else None

    blob = doc.joined_text()
    total = dec("total")
    doc_type = _classify(fields, blob, total)
    currency = (
        normalize_currency(fields.get("currency"))
        or currency_from_field_name(names.get("total"))
        or currency_from_field_name(names.get("net"))
        or currency_from_field_name(names.get("vat_amount"))
        or default_currency
        or "BGN"
    )

    invoice = Invoice(
        id=doc.id,
        source="xml",
        doc_type=doc_type.value,
        direction=detect_direction(blob, hint=doc.id).value,
        reverse_charge=detect_reverse_charge(blob),
        number=fields.get("number"),
        date=normalize_date(fields["date"]) if "date" in fields else None,
        currency=currency,
        supplier=Party(
            name=fields.get("supplier_name"),
            vat_number=fields.get("supplier_vat"),
            eik=fields.get("supplier_eik"),
        ),
        recipient=Party(
            name=fields.get("recipient_name"),
            vat_number=fields.get("recipient_vat"),
            eik=fields.get("recipient_eik"),
        ),
        net_amount=dec("net"),
        vat_amount=dec("vat_amount"),
        total_amount=total,
        field_confidence={slot: 0.95 for slot in fields},  # structured source, high confidence
    )

    if invoice.net_amount and invoice.vat_amount and invoice.net_amount > 0:
        rate = (invoice.vat_amount / invoice.net_amount).quantize(Decimal("0.01"))
        invoice.tax_lines.append(
            TaxLine(rate=rate, base=invoice.net_amount, amount=invoice.vat_amount)
        )

    return tag_company(invoice)


def invoices_from_xml(documents: list[DocCandidate], default_currency: str | None = None) -> list[Invoice]:
    return [doc_candidate_to_invoice(d, default_currency) for d in documents]


def invoices_from_xml_content(xml: str, label: str = "") -> list[Invoice]:
    """Parse XML into typed invoices, dispatching to the Controlisy handler when the
    ExportedData schema is detected, otherwise the generic record resolver."""
    from .controlisy import looks_like_controlisy, parse_controlisy
    from .currency import file_currency_hint
    from .ms_rowset import looks_like_rowset, parse_rowset
    from .xml_parser import parse_xml

    hint = file_currency_hint(xml)  # e.g. SAP <Currency> (file-level fallback)
    if looks_like_controlisy(xml):
        return parse_controlisy(xml, label, default_currency=hint)
    if looks_like_rowset(xml):
        return parse_rowset(xml, label, default_currency=hint)
    return invoices_from_xml(parse_xml(xml, label), default_currency=hint)
