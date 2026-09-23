"""Parse Bulgarian amounts written in words (the 'Словом:' line on invoices).

Recovers the total when the numeric line is garbled by OCR, e.g.
'двеста и осем лева и 32 ст' becomes Decimal('208.32'). Also serves as an independent
source to cross-check the numeric total.
"""

from __future__ import annotations

import re
from decimal import Decimal

_UNITS = {
    "нула": 0, "едно": 1, "един": 1, "една": 1, "два": 2, "две": 2, "три": 3,
    "четири": 4, "пет": 5, "шест": 6, "седем": 7, "осем": 8, "девет": 9,
}
_TEENS = {
    "десет": 10, "единадесет": 11, "единайсет": 11, "дванадесет": 12, "дванайсет": 12,
    "тринадесет": 13, "тринайсет": 13, "четиринадесет": 14, "четиринайсет": 14,
    "петнадесет": 15, "петнайсет": 15, "шестнадесет": 16, "шестнайсет": 16,
    "седемнадесет": 17, "седемнайсет": 17, "осемнадесет": 18, "осемнайсет": 18,
    "деветнадесет": 19, "деветнайсет": 19,
}
_TENS = {
    "двадесет": 20, "двайсет": 20, "тридесет": 30, "трийсет": 30, "четиридесет": 40,
    "четирийсет": 40, "петдесет": 50, "шестдесет": 60, "шейсет": 60, "седемдесет": 70,
    "осемдесет": 80, "деветдесет": 90,
}
_HUNDREDS = {
    "сто": 100, "двеста": 200, "триста": 300, "четиристотин": 400, "петстотин": 500,
    "шестстотин": 600, "седемстотин": 700, "осемстотин": 800, "деветстотин": 900,
}


def _group(words: list[str]) -> int:
    """Parse a 0-999 group of number words (ignoring the connector 'и')."""
    total = 0
    for w in words:
        if w in _HUNDREDS:
            total += _HUNDREDS[w]
        elif w in _TEENS:
            total += _TEENS[w]
        elif w in _TENS:
            total += _TENS[w]
        elif w in _UNITS:
            total += _UNITS[w]
    return total


def parse_amount_words(phrase: str) -> Decimal | None:
    """Parse a Bulgarian amount phrase into a Decimal, or None if unparseable."""
    if not phrase:
        return None
    low = phrase.lower()

    cents = 0
    m = re.search(r"и\s*(\d{1,2})\s*(?:ст|стотинки)", low)
    if m:
        cents = int(m.group(1))

    # Leva words: everything up to the first token that starts with 'лев'/'лв'.
    leva_part = re.split(r"\bле[вв]|\bлв", low)[0]
    tokens = re.findall(r"[а-я]+", leva_part)
    if not tokens:
        return None

    if any(t.startswith("хиляд") for t in tokens):
        idx = next(i for i, t in enumerate(tokens) if t.startswith("хиляд"))
        thousands = _group(tokens[:idx]) or 1
        leva = thousands * 1000 + _group(tokens[idx + 1:])
    else:
        leva = _group(tokens)

    if leva == 0 and cents == 0:
        return None
    return Decimal(leva) + (Decimal(cents) / Decimal(100))


def total_from_words(text: str) -> Decimal | None:
    """Extract the 'Словом:' total written in words from invoice text."""
    m = re.search(r"словом\s*[:\-]?\s*(.+)", text, re.IGNORECASE)
    if not m:
        return None
    phrase = re.split(r"сума\s+за\s+плащане|\n", m.group(1), flags=re.IGNORECASE)[0]
    return parse_amount_words(phrase)
