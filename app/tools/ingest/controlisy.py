"""Parser for the Controlisy / ExportedData XML schema.

The schema normalises contractors into a lookup table and references them from each
document, with nested accounting and VAT rows:

    <ExportedData>
      <Contractors><Contractor ca_contractorId=".." contractorName=".." .../></Contractors>
      <Documents>
        <Document documentNumber=".." documentDate=".." netAmountBGN=".." ...
                  ca_contractorId=".." ca_docTypeID="..">
          <Accountings>
            <Accounting amountBGN=".."><AccountingDetail .../></Accounting>
            <VAT taxBase=".." vatRate="20" vatOperationName="Продажби 20%" .../>
          </Accountings>
        </Document>
      </Documents>
    </ExportedData>

We resolve the contractor by id (rather than scraping the noisy accounting lines),
read the document totals, and classify the document (credit note on negative totals,
direction from the journal/reason, reverse charge / ВОП-ВОД from the VAT operation).
"""

from __future__ import annotations

from decimal import Decimal

from defusedxml.ElementTree import fromstring as safe_fromstring

from app.domain import Invoice, Party, TaxLine

from .company import tag_company
from .currency import normalize_currency
from .document_types import Direction, DocumentType, detect_direction, detect_reverse_charge
from .invoice_extractor import clean_amount, normalize_date


def looks_like_controlisy(xml: str) -> bool:
    return "exporteddata" in xml[:600].lower()


def _local(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def _contractor_table(root) -> dict[str, dict[str, str]]:
    table: dict[str, dict[str, str]] = {}
    for el in root.iter():
        if _local(el.tag) == "Contractor":
            cid = el.attrib.get("ca_contractorId", "")
            if cid:
                table[cid] = {
                    "name": el.attrib.get("contractorName") or None,
                    "eik": el.attrib.get("contractorEIK") or None,
                    "vat": el.attrib.get("contractorVATNumber") or None,
                }
    return table


def _tax_lines(document) -> list[TaxLine]:
    lines: list[TaxLine] = []
    seen: set[tuple] = set()
    for el in document.iter():
        if _local(el.tag) != "VAT":
            continue
        rate_raw = el.attrib.get("vatRate", "")
        try:
            rate = (Decimal(rate_raw) / Decimal(100)).quantize(Decimal("0.01")) if rate_raw != "" else None
        except Exception:
            rate = None
        if rate is None:
            continue
        base = clean_amount(el.attrib.get("taxBase", ""))
        amount = clean_amount(el.attrib.get("vatAmountBGN", ""))
        key = (rate, base, amount)
        if key in seen:  # the export sometimes repeats the same VAT row
            continue
        seen.add(key)
        lines.append(TaxLine(rate=rate, base=base, amount=amount))
    return lines


def _vat_operation_text(document) -> str:
    parts: list[str] = []
    for el in document.iter():
        if _local(el.tag) == "VAT":
            for key in ("vatOperationName", "vatOperationIden", "vatOperationAdditionalName"):
                if el.attrib.get(key):
                    parts.append(el.attrib[key])
    return " ".join(parts)


def _amount(attrs: dict, base: str) -> tuple[Decimal | None, str | None]:
    """Read an amount that carries its currency in the field name (...BGN / ...EUR)."""
    for suffix in ("BGN", "EUR"):
        value = attrs.get(base + suffix)
        if value not in (None, ""):
            return clean_amount(value), suffix
    return None, None


def parse_controlisy(xml: str, label: str = "", default_currency: str | None = None) -> list[Invoice]:
    root = safe_fromstring(xml.encode("utf-8") if isinstance(xml, str) else xml)
    contractors = _contractor_table(root)

    journal_direction = detect_direction("", hint=f"{label} {root.attrib.get('reason', '')}")

    invoices: list[Invoice] = []
    for doc in root.iter():
        if _local(doc.tag) != "Document":
            continue
        a = doc.attrib
        number = a.get("documentNumber") or None
        if number is None and not (a.get("netAmountBGN") or a.get("netAmountEUR")):
            continue

        total, cur_t = _amount(a, "totalAmount")
        net, cur_n = _amount(a, "netAmount")
        vat, _ = _amount(a, "vatAmount")
        currency = cur_t or cur_n or normalize_currency(a.get("currency")) or default_currency or "BGN"

        contractor = contractors.get(a.get("ca_contractorId", ""), {})
        party = Party(name=contractor.get("name"), eik=contractor.get("eik"), vat_number=contractor.get("vat"))

        reason = a.get("reason", "")
        direction = journal_direction
        if direction == Direction.UNKNOWN:
            direction = detect_direction(reason)

        # Counterparty goes on the side opposite to our own firm.
        supplier = party if direction == Direction.PURCHASE else Party()
        recipient = party if direction != Direction.PURCHASE else Party()

        vat_text = _vat_operation_text(doc)
        doc_type = DocumentType.INVOICE
        if total is not None and total < 0:
            doc_type = DocumentType.CREDIT_NOTE

        invoice = Invoice(
            id=f"{label}-{number}" if label and number else (number or a.get("ca_docId", "")),
            source="xml",
            doc_type=doc_type.value,
            direction=direction.value,
            reverse_charge=detect_reverse_charge(f"{reason} {vat_text}"),
            number=number,
            date=normalize_date(a.get("documentDate", "")) if a.get("documentDate") else None,
            currency=currency,
            supplier=supplier,
            recipient=recipient,
            net_amount=net,
            vat_amount=vat,
            total_amount=total,
            tax_lines=_tax_lines(doc),
            field_confidence={
                k: 0.98
                for k, v in {
                    "number": number, "date": a.get("documentDate"),
                    "net_amount": net, "vat_amount": vat, "total_amount": total,
                }.items()
                if v not in (None, "")
            },
        )
        invoices.append(tag_company(invoice))
    return invoices
