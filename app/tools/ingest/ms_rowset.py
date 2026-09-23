"""Parser for the NAP VAT-ledger MS ADO Rowset XML (дневник за продажби/покупки).

These exports use the urn:schemas-microsoft-com:rowset namespace: an <s:Schema>
block followed by <z:row .../> data rows whose columns follow the НАП ledger cells.

Observed sales-ledger column mapping (base + VAT = total):
    DocNo = number, DocDate = date, Name = counterparty, VATNo = counterparty VAT,
    S11 = net (данъчна основа), S12 = VAT (начислен данък), S10 = total
"""

from __future__ import annotations

from decimal import Decimal

from defusedxml.ElementTree import fromstring as safe_fromstring

from app.domain import Invoice, Party, TaxLine

from .company import tag_company
from .currency import normalize_currency
from .document_types import Direction, DocumentType, detect_direction
from .invoice_extractor import clean_amount, normalize_date


def looks_like_rowset(xml: str) -> bool:
    head = xml[:1500].lower()
    return "schemas-microsoft-com:rowset" in head or "rowsetschema" in head


def _local(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def parse_rowset(xml: str, label: str = "", default_currency: str | None = None) -> list[Invoice]:
    root = safe_fromstring(xml.encode("utf-8") if isinstance(xml, str) else xml)
    direction = detect_direction("", hint=label)

    invoices: list[Invoice] = []
    for el in root.iter():
        if _local(el.tag) != "row":
            continue  # skips the <s:ElementType name='row'> schema node
        a = {_local(k): v for k, v in el.attrib.items()}
        if "DocNo" not in a and "Name" not in a:
            continue

        net = clean_amount(a.get("S11", ""))
        vat = clean_amount(a.get("S12", "")) or clean_amount(a.get("S14", ""))
        total = clean_amount(a.get("S10", ""))
        if total is None and net is not None and vat is not None:
            total = net + vat
        if net is None and total is not None and vat is not None:
            net = total - vat

        number = a.get("DocNo") or None
        currency = normalize_currency(a.get("Currency")) or default_currency or "BGN"
        party = Party(name=a.get("Name") or None, vat_number=a.get("VATNo") or None)
        supplier = party if direction == Direction.PURCHASE else Party()
        recipient = party if direction != Direction.PURCHASE else Party()

        doc_type = DocumentType.CREDIT_NOTE if (total is not None and total < 0) else DocumentType.INVOICE

        invoice = Invoice(
            id=f"{label}-{number}" if label and number else (number or a.get("DocNo", "")),
            source="xml",
            doc_type=doc_type.value,
            direction=direction.value,
            number=number,
            date=normalize_date(a["DocDate"].split("T")[0]) if a.get("DocDate") else None,
            currency=currency,
            supplier=supplier,
            recipient=recipient,
            net_amount=net,
            vat_amount=vat,
            total_amount=total,
            field_confidence={
                k: 0.95
                for k, present in {
                    "number": number,
                    "date": a.get("DocDate"),
                    "net_amount": net is not None,
                    "vat_amount": vat is not None,
                    "total_amount": total is not None,
                }.items()
                if present
            },
        )
        if net and vat and net > 0:
            rate = (vat / net).quantize(Decimal("0.01"))
            invoice.tax_lines.append(TaxLine(rate=rate, base=net, amount=vat))
        invoices.append(tag_company(invoice))
    return invoices
