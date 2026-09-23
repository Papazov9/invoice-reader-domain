"""Line-item (row) extraction from an invoice's ``pdftotext -layout`` text.

Digital PDFs from Bulgarian suppliers print the item table as fixed-width columns, so
``pdftotext -layout`` preserves each cell at a stable character offset. Rather than guess
row structure with a single regex (fragile across vendors), we locate the *header* row,
read the character offset of each recognised column label, and slice every data row at
those offsets. This adapts to whatever column order a given invoice uses.

Deterministic and free: no OCR-specific logic and no model. Populates Invoice.line_items;
amounts stay Decimal so the downstream arithmetic/validation is exact. Image/photo scans,
whose OCR text does not preserve columns, are left to the vision fallback.
"""

from __future__ import annotations

import re
from decimal import Decimal

from app.domain import LineItem

from .invoice_extractor import clean_amount

# Column-label vocabulary. Each group maps to a logical cell; matching is accent/case
# folded and done on whole words so "Кол." matches but "Колона" does not falsely win.
_COLUMN_LABELS: dict[str, tuple[str, ...]] = {
    "code": ("код", "артикулен", "кат.№", "sku", "code", "ref"),
    "description": (
        "наименование", "описание", "стока", "стоки", "услуга", "услуги",
        "артикул", "продукт", "description", "item", "goods", "продукти",
    ),
    "unit": ("мярка", "мяр", "мер", "ед.мярка", "unit", "u/m", "м-ка"),
    "quantity": ("количество", "кол", "к-во", "брой", "qty", "quantity", "бр."),
    "unit_price": ("цена", "ед.цена", "единична", "price"),
    "amount": ("стойност", "сума", "amount", "value", "нето", "линия"),
}

# A row at or after one of these lines is no longer part of the item table.
_TERMINATORS = re.compile(
    r"(данъчна\s+основа|данъчна\s+ставка|междинна\s+сума|обща\s+сума|общо\s+за\s+плащане"
    r"|сума\s+за\s+плащане|сума\s+в\s+(?:лева|евро|bgn|eur)|словом|начин\s+на\s+плащане"
    r"|основание|забележ|iban|банка\s*:|подпис|съставил|получил|"
    r"subtotal|sub-total|grand\s+total|amount\s+due|total\s+due|обща\s+стойност\s+без)",
    re.IGNORECASE,
)

_HEADER_MIN_GROUPS = 2  # a real header names at least this many known columns
_ACCENT = str.maketrans("", "", "")


def _norm(word: str) -> str:
    return word.strip(" .:№|/\\").lower()


def _find_labels(line: str) -> dict[str, int]:
    """Return {column_name: char_offset} for every recognised label on this line."""
    found: dict[str, int] = {}
    for m in re.finditer(r"[^\s|]+(?:\s[^\s|]+)?", line):
        token = _norm(m.group(0))
        if not token:
            continue
        for col, labels in _COLUMN_LABELS.items():
            if col in found:
                continue
            # whole-word / prefix match against the label vocabulary
            if any(token == lab or token.startswith(lab + ".") or token == lab.rstrip(".")
                   for lab in labels):
                found[col] = m.start()
    return found


def _looks_like_header(labels: dict[str, int]) -> bool:
    if len(labels) < _HEADER_MIN_GROUPS:
        return False
    # Must anchor the money column and something describing the row.
    return "amount" in labels and ("description" in labels or "quantity" in labels or "unit_price" in labels)


def _spans(labels: dict[str, int], width: int) -> list[tuple[str, int, int]]:
    """Ordered [ (col, start, end) ] covering the line by the label offsets."""
    ordered = sorted(labels.items(), key=lambda kv: kv[1])
    spans: list[tuple[str, int, int]] = []
    for i, (col, start) in enumerate(ordered):
        end = ordered[i + 1][1] if i + 1 < len(ordered) else width
        # widen a hair so a value that starts slightly under its label is not clipped
        spans.append((col, max(0, start - 1), end))
    return spans


def _cell(line: str, start: int, end: int) -> str:
    return line[start:end].strip() if start < len(line) else ""


def _num(cell: str) -> Decimal | None:
    """A cell is numeric only if it actually carries digits (skips 'Бр', '', '---')."""
    if not cell or not any(c.isdigit() for c in cell):
        return None
    return clean_amount(cell)


def parse_line_items(text: str) -> list[LineItem]:
    """Extract table rows from column-preserving invoice text. Empty list when no item
    table is found (image OCR, atypical layout) so callers can fall back to vision."""
    lines = text.splitlines()
    header_idx = -1
    labels: dict[str, int] = {}
    for i, line in enumerate(lines):
        cand = _find_labels(line)
        if _looks_like_header(cand):
            header_idx, labels = i, cand
            break
    if header_idx < 0:
        return []

    width = max((len(l) for l in lines[header_idx:]), default=0)
    spans = _spans(labels, width)
    amount_span = next((s for s in spans if s[0] == "amount"), None)
    desc_span = next((s for s in spans if s[0] == "description"), None)
    if amount_span is None:
        return []

    items: list[LineItem] = []
    for line in lines[header_idx + 1:]:
        if _TERMINATORS.search(line):
            break
        if not line.strip():
            continue
        cells = {col: _cell(line, s, e) for col, s, e in spans}
        amount = _num(cells.get("amount", ""))

        if amount is None:
            # A continuation line (wrapped description) has text in the desc column but
            # no money — append it to the previous item instead of dropping it.
            if items and desc_span is not None:
                cont = _cell(line, desc_span[1], desc_span[2])
                if cont and not any(c.isdigit() for c in cont[:3]):
                    prev = items[-1]
                    prev.description = f"{prev.description} {cont}".strip() if prev.description else cont
            continue

        qty = _num(cells.get("quantity", ""))
        price = _num(cells.get("unit_price", ""))
        desc = cells.get("description", "").strip()
        code = cells.get("code", "").strip()
        # When there is no dedicated code column the leading token of a No/Код-prefixed
        # description is noise; keep the human-readable part only.
        if not desc and code:
            desc, code = code, ""
        if not desc and not qty and not price:
            # bare number line (e.g. a running subtotal) — not an item
            continue

        items.append(LineItem(
            description=desc or None,
            quantity=qty,
            unit_price=price,
            amount=amount,
        ))
    return items
