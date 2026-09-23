"""Identifier tokenization.

Splits compound technical identifiers (XML field names, GL codes) into space-separated
words so word-level TF-IDF can match them semantically. Handles camelCase, PascalCase,
snake_case, kebab-case, SCREAMING_SNAKE, abbreviation runs, and letter/digit boundaries.

    invoiceNumber   becomes "invoice number"
    XMLParser       becomes "xml parser"
    totalAmountBGN  becomes "total amount bgn"
    VAT_number      becomes "vat number"
    doc-no-2        becomes "doc no 2"
"""

from __future__ import annotations

import re

_SEPARATORS = re.compile(r"[_\-./]+")
# lower/digit then Upper: camelCase boundary (invoiceNumber)
_CAMEL = re.compile(r"(?<=[a-zа-я0-9])(?=[A-ZА-Я])")
# Upper then Upper+lower: end of an acronym (XMLParser splits to XML|Parser)
_ACRONYM = re.compile(r"(?<=[A-ZА-Я])(?=[A-ZА-Я][a-zа-я])")
# letter/digit boundaries
_LETTER_DIGIT = re.compile(r"(?<=[A-Za-zА-Яа-я])(?=\d)")
_DIGIT_LETTER = re.compile(r"(?<=\d)(?=[A-Za-zА-Яа-я])")
_WS = re.compile(r"\s+")


def split_identifier(text: str, *, lowercase: bool = True) -> str:
    if not text:
        return ""
    s = _SEPARATORS.sub(" ", text)
    s = _CAMEL.sub(" ", s)
    s = _ACRONYM.sub(" ", s)
    s = _LETTER_DIGIT.sub(" ", s)
    s = _DIGIT_LETTER.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s.lower() if lowercase else s
