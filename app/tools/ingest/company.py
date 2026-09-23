"""Company identification and per-company grouping.

Derives a stable company identity from an invoice's supplier/recipient data so invoices
can be grouped into per-company working sets. Keyed by the strongest available signal:
VAT number, else EIK, else normalized legal name.
"""

from __future__ import annotations

import re

from app.domain import Company, CompanyGroup, Invoice, Party

# Legal-form suffixes stripped when normalizing a company name for matching.
_SUFFIXES = [
    "еоод", "оод", "еад", "ад", "ет", "сд", "дззд", "адсиц",
    "ltd", "llc", "ltd.", "gmbh", "ad", "ood", "eood",
]
_SUFFIX_RE = re.compile(r"\b(" + "|".join(map(re.escape, _SUFFIXES)) + r")\b", re.IGNORECASE)
_NON_ALNUM = re.compile(r"[^0-9a-zA-Zа-яА-Я]+", re.UNICODE)


def normalize_company_name(name: str | None) -> str:
    """Normalize a legal name for matching: drop suffixes/punctuation/case/space."""
    if not name:
        return ""
    s = name.lower()
    s = _SUFFIX_RE.sub(" ", s)
    s = _NON_ALNUM.sub(" ", s)
    return " ".join(s.split()).strip()


def _norm_vat(vat: str | None) -> str | None:
    if not vat:
        return None
    v = re.sub(r"\s+", "", vat).upper()
    return v or None


def _norm_eik(eik: str | None) -> str | None:
    if not eik:
        return None
    e = re.sub(r"\D", "", eik)
    return e or None


def party_key(party: Party) -> str | None:
    """Stable identity for a party, or None if it carries no usable signal."""
    vat = _norm_vat(party.vat_number)
    if vat:
        return f"vat:{vat}"
    eik = _norm_eik(party.eik)
    if eik:
        return f"eik:{eik}"
    norm = normalize_company_name(party.name)
    if norm:
        return f"name:{norm}"
    return None


def _identity_party(invoice: Invoice) -> Party:
    """The party that defines the invoice's company. The chosen perspective decides which
    side is primary; we still fall back to the other when the primary carries no signal."""
    if getattr(invoice, "perspective", "supplier") == "recipient":
        primary, secondary = invoice.recipient, invoice.supplier
    else:
        primary, secondary = invoice.supplier, invoice.recipient
    if party_key(primary):
        return primary
    if party_key(secondary):
        return secondary
    return primary


def company_of(invoice: Invoice) -> Company:
    """Build the Company identity for an invoice (used for grouping/tagging)."""
    party = _identity_party(invoice)
    key = party_key(party) or "unknown"
    return Company(
        key=key,
        name=party.name,
        vat=_norm_vat(party.vat_number),
        eik=_norm_eik(party.eik),
    )


def tag_company(invoice: Invoice) -> Invoice:
    """Set company_key / company_name on the invoice in place and return it."""
    company = company_of(invoice)
    invoice.company_key = company.key
    invoice.company_name = company.name or company.vat or company.eik or "Unknown company"
    return invoice


def group_by_company(invoices: list[Invoice]) -> list[CompanyGroup]:
    """Group invoices into per-company working sets, preserving first-seen order."""
    groups: dict[str, CompanyGroup] = {}
    for inv in invoices:
        company = company_of(inv)
        grp = groups.get(company.key)
        if grp is None:
            grp = CompanyGroup(company=company, invoices=[])
            groups[company.key] = grp
        grp.invoices.append(inv)
    for grp in groups.values():
        grp.company.invoice_count = len(grp.invoices)
    # Largest groups first; "unknown" always last.
    return sorted(
        groups.values(),
        key=lambda g: (g.company.key == "unknown", -g.company.invoice_count),
    )
