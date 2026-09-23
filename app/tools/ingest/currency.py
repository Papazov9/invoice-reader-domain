"""Currency detection from the document's own data.

Bulgaria's euro adoption means invoices may be in EUR or BGN (and transition documents
often show both), so currency is never assumed. It is derived per document from explicit
currency fields, amount field-name suffixes (...BGN/...EUR), the "Словом:" words line, or
currency tokens next to the totals.
"""

from __future__ import annotations

import re

# token (substring, lowercased) to ISO code; EUR checked before BGN.
_TOKENS: list[tuple[tuple[str, ...], str]] = [
    (("eur", "€", "евро", "евр", "цент", "ευρ"), "EUR"),
    (("bgn", "bgl", "лева", "лев", "лв", "lv"), "BGN"),
]


def normalize_currency(value: str | None) -> str | None:
    if not value:
        return None
    t = value.strip().lower()
    for tokens, code in _TOKENS:
        if any(tok in t for tok in tokens):
            return code
    return None


def currency_from_field_name(name: str | None) -> str | None:
    """Currency implied by an amount field name, e.g. netAmountBGN / totalAmountEUR."""
    if not name:
        return None
    n = name.lower()
    if "eur" in n or "евро" in n:
        return "EUR"
    if "bgn" in n or "bgl" in n or "лева" in n or n.endswith("лв"):
        return "BGN"
    return None


# Lines that merely state the EUR<->BGN conversion (transition notices) must not decide
# the currency — they always mention euro even on a plain BGN invoice.
_CONV_LINE = re.compile(
    r"курс|превалут|изчислен|стойност\s+в\s+евро|1\s*(?:eur|€)\s*=|=\s*1[.,]95583", re.IGNORECASE
)
_TOTAL_LINE = re.compile(
    r"(обща\s+стойност|общо\s+за\s+плащане|обща\s+сума\s+за\s+плащане|всичко\s+за\s+плащане|"
    r"сума\s+за\s+плащане|крайна\s+сума)[^\n]*",
    re.IGNORECASE,
)


def detect_currency_text(text: str) -> str | None:
    """Currency of an OCR/plain-text invoice. Order: an explicit "Валута:" field, then the
    'Словом:' legal-amount words, then the primary total line, then a majority vote — always
    ignoring EUR/BGN conversion-notice lines (common on euro-transition invoices)."""
    if not text:
        return None
    m = re.search(r"валута\s*[:\-]?\s*([A-Za-zА-Яа-я€]{2,})", text, re.IGNORECASE)
    if m:
        c = normalize_currency(m.group(1))
        if c:
            return c
    m = re.search(r"словом[^\n]*", text, re.IGNORECASE)
    if m and not _CONV_LINE.search(m.group(0)):
        c = normalize_currency(m.group(0))
        if c:
            return c
    for m in _TOTAL_LINE.finditer(text):
        if _CONV_LINE.search(m.group(0)):
            continue
        c = normalize_currency(m.group(0))
        if c:
            return c
    bgn = eur = 0
    for line in text.splitlines():
        if _CONV_LINE.search(line):
            continue
        c = normalize_currency(line)
        if c == "BGN":
            bgn += 1
        elif c == "EUR":
            eur += 1
    if bgn or eur:
        return "BGN" if bgn >= eur else "EUR"
    return None


def file_currency_hint(xml: str) -> str | None:
    """A file-level currency (e.g. SAP <Currency>EUR</Currency> or a currency attr)."""
    for pat in (r"<Currency>\s*([^<]+?)\s*</Currency>", r'currency(?:code)?="([^"]+)"'):
        m = re.search(pat, xml, re.IGNORECASE)
        if m:
            c = normalize_currency(m.group(1))
            if c:
                return c
    return None
