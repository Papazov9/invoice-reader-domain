"""CSV ingestion for spreadsheet exports and VAT ledgers (дневници за покупки/продажби).

Each row becomes a DocCandidate (header as fieldName, cell as value) and goes through
the same field resolver as XML, so the EN/BG synonyms all apply.
"""

from __future__ import annotations

import csv
import io

from app.domain import DocCandidate, Invoice, TextUnit

from .xml_invoice import doc_candidate_to_invoice


def parse_csv(content: str, label: str = "csv") -> list[Invoice]:
    if not content.strip():
        return []

    sample = content[:4096]
    try:
        dialect: type[csv.Dialect] | csv.Dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel

    reader = csv.DictReader(io.StringIO(content), dialect=dialect)
    invoices: list[Invoice] = []
    for idx, row in enumerate(reader):
        units: list[TextUnit] = []
        for header, cell in row.items():
            value = "" if cell is None else str(cell).strip()
            if header and value:
                units.append(TextUnit(kind="fieldName", text=str(header)))
                units.append(TextUnit(kind="value", text=value))
        if not units:
            continue

        invoice = doc_candidate_to_invoice(DocCandidate(id=f"{label}-{idx}", units=units))
        invoice.source = "csv"
        if invoice.number:
            invoice.id = f"{label}-{invoice.number}"
        invoices.append(invoice)
    return invoices
