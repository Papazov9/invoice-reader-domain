"""Text normalization.

Lowercases, strips punctuation/symbols, and collapses whitespace while preserving
Cyrillic (Bulgarian) and digits. Unicode-aware \\w keeps letters from any script.
"""

from __future__ import annotations

import re

_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+", re.UNICODE)


def normalize(text: str, *, lowercase: bool = True) -> str:
    """Return a normalized form suitable for vectorization/matching.

    >>> normalize("Фактура №: 2000002487!!")
    'фактура 2000002487'
    """
    if not text:
        return ""
    out = _NON_WORD.sub(" ", text)
    out = _WS.sub(" ", out).strip()
    return out.lower() if lowercase else out
