"""Deterministic candidate generation for trainable field selection.

Each generator yields every plausible candidate for a field together with text-based
features (label context, position, ids present, magnitude). A trained selector then PICKS
which candidate is the number / date / supplier / recipient / net / vat / total — but the
candidate's *value* is always a verbatim span from the document, so amounts stay exact and
auditable. Pure and deterministic; features work the same on embedded text and OCR text
(no coordinates needed).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from decimal import Decimal

from .invoice_extractor import _AMT, _COMPANY_SUFFIX, _clean_name, _valid_iso, clean_amount, normalize_date

_AMT_RE = re.compile(_AMT)
_VAT_RE = re.compile(r"BG\s*\d{9,10}(?!\d)", re.IGNORECASE)
_EIK_NEAR = re.compile(r"(?:ЕИК|ЕИН|БУЛСТАТ)\s*[:\-№]?\s*(\d{9}(?:\d{4})?)(?!\d)", re.IGNORECASE)
_NAME_RE = re.compile(
    rf"([А-Яа-яA-Za-z0-9\-\"'.,]+(?:\s+[А-Яа-яA-Za-z0-9\-\"'.,]+){{0,5}}?)\s+({_COMPANY_SUFFIX})\b",
    re.IGNORECASE,
)
_ROLE_RE = re.compile(
    r"доставчик|продавач|изпълнител|получател|купувач|клиент|износител|вносител",
    re.IGNORECASE,
)
_DATE_RE = re.compile(r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}[-./]\d{1,2}[-./]\d{1,2}")
_NUM_RE = re.compile(r"\b\d{6,15}\b")
_TOKEN_RE = re.compile(r"[a-zа-я]+", re.IGNORECASE)


def _line_idx(text: str, pos: int) -> int:
    return text.count("\n", 0, pos)


def _ctx_tokens(text: str, pos: int, n: int = 8) -> list[str]:
    """Up to n word tokens immediately preceding pos (the label context)."""
    return _TOKEN_RE.findall(text[max(0, pos - 80):pos].lower())[-n:]


def _ctx_features(text: str, pos: int) -> dict[str, float]:
    return {f"w:{t}": 1.0 for t in _ctx_tokens(text, pos)}


# --- amounts ------------------------------------------------------------------

@dataclass
class AmountCandidate:
    value: Decimal
    raw: str
    pos: int
    line_idx: int
    feats: dict = field(default_factory=dict)

    def features(self) -> dict:
        return {**self.feats}


def amount_candidates(text: str) -> list[AmountCandidate]:
    total_lines = max(1, text.count("\n") + 1)
    raw_cands: list[AmountCandidate] = []
    for m in _AMT_RE.finditer(text):
        val = clean_amount(m.group(1))
        if val is None:
            continue
        raw_cands.append(AmountCandidate(value=val, raw=m.group(1).strip(), pos=m.start(), line_idx=_line_idx(text, m.start())))
    if not raw_cands:
        return []
    biggest = max(c.value for c in raw_cands)
    vals = {c.value for c in raw_cands}
    for c in raw_cands:
        # does this value participate in a net+vat==total triple among the candidates?
        in_triple = any((c.value + v) in vals or (c.value - v) in vals for v in vals if v != c.value)
        c.feats = {
            "value_log": math.log1p(float(abs(c.value))),
            "line_frac": c.line_idx / total_lines,
            "is_largest": 1.0 if c.value == biggest else 0.0,
            "in_triple": 1.0 if in_triple else 0.0,
            **_ctx_features(text, c.pos),
        }
    return raw_cands


# --- parties ------------------------------------------------------------------

@dataclass
class PartyCandidate:
    name: str
    eik: str | None
    vat: str | None
    pos: int
    line_idx: int
    label: str
    order: int
    feats: dict = field(default_factory=dict)

    def features(self) -> dict:
        return {**self.feats}


def party_candidates(text: str) -> list[PartyCandidate]:
    total_lines = max(1, text.count("\n") + 1)
    roles = [(m.start(), m.group(0).lower()) for m in _ROLE_RE.finditer(text)]
    cands: list[PartyCandidate] = []
    for order, m in enumerate(_NAME_RE.finditer(text)):
        name = _clean_name(f"{m.group(1)} {m.group(2)}")
        if not name:
            continue
        start = m.start()
        # nearest role label within 60 chars before the name
        label = "none"
        for rpos, rname in roles:
            if 0 <= start - rpos <= 60:
                label = rname
        window = text[start: start + 160]
        eik_m = _EIK_NEAR.search(window)
        vat_m = _VAT_RE.search(window)
        cands.append(PartyCandidate(
            name=name, eik=eik_m.group(1) if eik_m else None,
            vat=vat_m.group(0).replace(" ", "").upper() if vat_m else None,
            pos=start, line_idx=_line_idx(text, start), label=label, order=order,
        ))
    for c in cands:
        c.feats = {
            "label": c.label,
            "has_eik": 1.0 if c.eik else 0.0,
            "has_vat": 1.0 if c.vat else 0.0,
            "order": float(c.order),
            "line_frac": c.line_idx / total_lines,
            **_ctx_features(text, c.pos),
        }
    return cands


# --- number / date (generic value candidates) ---------------------------------

@dataclass
class ValueCandidate:
    value: str
    pos: int
    line_idx: int
    feats: dict = field(default_factory=dict)

    def features(self) -> dict:
        return {**self.feats}


def number_candidates(text: str, exclude: set[str] | None = None) -> list[ValueCandidate]:
    exclude = exclude or set()
    total_lines = max(1, text.count("\n") + 1)
    out: list[ValueCandidate] = []
    for i, m in enumerate(_NUM_RE.finditer(text)):
        if m.group(0) in exclude:
            continue
        c = ValueCandidate(value=m.group(0), pos=m.start(), line_idx=_line_idx(text, m.start()))
        c.feats = {
            "len": float(len(m.group(0))),
            "line_frac": c.line_idx / total_lines,
            "order": float(i),
            **_ctx_features(text, m.start()),
        }
        out.append(c)
    return out


def date_candidates(text: str) -> list[ValueCandidate]:
    total_lines = max(1, text.count("\n") + 1)
    out: list[ValueCandidate] = []
    for i, m in enumerate(_DATE_RE.finditer(text)):
        iso = normalize_date(m.group(0))
        if not _valid_iso(iso):  # skip garbled OCR dates (month 42, year 2825, …)
            continue
        c = ValueCandidate(value=iso, pos=m.start(), line_idx=_line_idx(text, m.start()))
        c.feats = {
            "line_frac": c.line_idx / total_lines,
            "order": float(i),
            **_ctx_features(text, m.start()),
        }
        out.append(c)
    return out
