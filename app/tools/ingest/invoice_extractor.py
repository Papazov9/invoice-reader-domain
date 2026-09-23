"""Bulgarian invoice field extraction from OCR/plain text.

Locale-aware regex for Фактура / ЕИК / ДДС / Доставчик / Получател etc., returning
each field with a confidence score and building a typed Invoice. A labelled-pattern
match scores ~0.9, a positional fallback ~0.5, a miss 0.0. The low scores are where
an LLM extraction fallback can kick in.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from app.core import get_settings
from app.domain import ExtractedField, Invoice, Party, TaxLine

from .bg_amount_words import total_from_words
from .company import tag_company
from .company_lookup import lookup_company
from .currency import detect_currency_text
from .document_types import (
    Direction,
    classify_document_type,
    detect_direction,
    detect_reverse_charge,
)
from .eik import validate_eik
from .ocr import normalize_token

_HIGH = 0.9
_LOW = 0.5


def clean_amount(raw: str) -> Decimal | None:
    """Normalize a localized amount string to a Decimal.

    Handles thousands separators and EU decimal commas: "16 143,38" -> 16143.38.
    """
    if not raw:
        return None
    s = raw.replace(" ", "").replace(" ", "")
    if "," in s and "." in s:
        # Last separator is the decimal point.
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".")
    s = re.sub(r"[^\d.]", "", s)
    if not s:
        return None
    try:
        value = Decimal(s)
    except InvalidOperation:
        return None
    # Preserve sign: credit notes carry negative amounts; (123) is also negative.
    r = raw.strip()
    if r.startswith("-") or (r.startswith("(") and r.endswith(")")):
        value = -value
    return value


def normalize_date(raw: str) -> str:
    """Normalize ``dd.mm.yyyy`` / ``dd/mm/yy`` / ``dd-mm-yy`` / ``yyyy-mm-dd`` to ISO
    ``yyyy-mm-dd``. Two-digit years are read as 20yy (current accounting docs)."""
    m = re.match(r"(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})", raw)
    if m:
        d, mo, y = m.groups()
        if len(y) == 2:
            y = "20" + y
        return f"{y}-{mo.zfill(2)}-{d.zfill(2)}"
    m = re.match(r"(\d{4})[-./](\d{1,2})[-./](\d{1,2})", raw)
    if m:
        y, mo, d = m.groups()
        return f"{y}-{mo.zfill(2)}-{d.zfill(2)}"
    return raw


# field extractors


# The label may carry a slashed alternative ("ЕИК/ЕГН:", "БУЛСТАТ/ЕИК:", "ЕИК / VAT"): the
# recipient block of many templates labels the company id "ЕИК/ЕГН", and without consuming
# that suffix the digits right after it are missed and a repeated id elsewhere wins instead.
_EIK_RE = re.compile(
    r"(?:ЕИК|ЕИН|БУЛСТАТ)(?:\s*/\s*[A-Za-zА-Яа-я.]+)?\s*[:\-№]?\s*(\d{9}(?:\d{4})?)(?!\d)",
    re.IGNORECASE,
)


def _eik_list(text: str) -> list[str]:
    """EIKs in order of appearance (deduped). ЕИК / ЕИН / БУЛСТАТ are interchangeable
    labels for the company id (9 or 13 digits); the lookahead stops the match from
    swallowing an adjacent column's digits."""
    out: list[str] = []
    for m in _EIK_RE.findall(text):
        if m not in out:
            out.append(m)
    return out


def _all_eik(text: str) -> set[str]:
    return set(_eik_list(text))


def _all_vat(text: str) -> list[str]:
    # BG VAT is BG + exactly 9 or 10 digits; the lookahead prevents grabbing trailing
    # digits from an adjacent number (which would corrupt the company key).
    return [m.replace(" ", "").upper()
            for m in re.findall(r"BG\s*\d{9,10}(?!\d)", text, re.IGNORECASE)]


def _is_referenced(text: str, start: int) -> bool:
    """True when the number at `start` is a citation of another document ("Към фактура
    № X", "Възоснова на: X") rather than this document's own number."""
    return bool(re.search(r"\bкъм\b|възоснова|to\s+invoice", text[max(0, start - 16):start], re.IGNORECASE))


def _bogus_number(n: str) -> bool:
    """A run of identical digits (000000…, 999999999999999) is a template placeholder, not
    a real document number."""
    return len(set(n)) <= 1


def extract_invoice_number(text: str) -> ExtractedField:
    exclude = _all_eik(text) | {v.removeprefix("BG") for v in _all_vat(text)}
    # OCR often garbles "№" into "Ne", "Ме", "No", "N°", so allow a few non-digit
    # characters between the label and the number rather than a fixed marker.
    labelled = [
        # any document-type label (фактура / кредитно-дебитно известие / проформа /
        # протокол) followed by the number; tolerant of a garbled № marker
        r"(?:Фактура|Кредитно\s+известие|Дебитно\s+известие|Проформа(?:\s+фактура)?|"
        r"Опростена\s+фактура|Протокол)[^\d\n]{0,6}(\d{7,15})",
        r"\bНомер\s*[:\-]?\s*(\d{6,15})",      # "Номер: 0000005002" / "Номер 0067915794"
        r"\bNo\.?\s*[:\-]?\s*(\d{7,15})",      # "No. 0400153377"
        r"Invoice[^\d\n]{0,6}(\d{6,15})",
        r"№\s*[:\-]?\s*(\d{10,15})",
    ]
    for pat in labelled:
        for m in re.finditer(pat, text, re.IGNORECASE):
            if m.group(1) not in exclude and not _is_referenced(text, m.start()) and not _bogus_number(m.group(1)):
                return ExtractedField(value=m.group(1), confidence=_HIGH)
    # fallback: a standalone 10-digit number (common BG invoice shape)
    for m in re.finditer(r"\b(\d{10})\b", text):
        if m.group(1) not in exclude and not _is_referenced(text, m.start()) and not _bogus_number(m.group(1)):
            return ExtractedField(value=m.group(1), confidence=_LOW)
    return ExtractedField(value=None, confidence=0.0)


# dd.mm.yyyy / dd.mm.yy / dd-mm-yy / dd/mm/yyyy, or ISO yyyy-mm-dd. Two-digit years are
# common on real invoices ("21.06.25"), so the year is \d{2,4} and the separators include "-".
_DATE_PAT = r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}[-./]\d{1,2}[-./]\d{1,2}"


def _valid_iso(s: str) -> bool:
    """Reject impossible dates (garbled OCR like month 42 or year 2825) so a wrong date is
    never reported — a later valid candidate or the vision/selector fallback wins instead."""
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", s or "")
    if not m:
        return False
    y, mo, d = (int(x) for x in m.groups())
    return 2000 <= y <= 2099 and 1 <= mo <= 12 and 1 <= d <= 31


def extract_date(text: str) -> ExtractedField:
    # Prefer an explicitly labelled issue date; skip any candidate that isn't a real date.
    for label in (r"Дата\s+на\s+издаване", r"Дата\s+на\s+(?:дан[\.ъ][^:\n]{0,18})?събитие", r"Дата", r"Date", r"Issued"):
        for m in re.finditer(label + r"\s*[:\-]?\s*(" + _DATE_PAT + r")", text, re.IGNORECASE):
            iso = normalize_date(m.group(1))
            if _valid_iso(iso):
                return ExtractedField(value=iso, confidence=_HIGH)
    for m in re.finditer(r"(" + _DATE_PAT + r")", text):
        iso = normalize_date(m.group(1))
        if _valid_iso(iso):
            return ExtractedField(value=iso, confidence=_LOW)
    return ExtractedField(value=None, confidence=0.0)


_COMPANY_SUFFIX = r"(?:ЕООД|ООД|ЕТ|АД|СД|ЕАД|Ltd|LLC|ЛТД)"
# Party-role labels that can leak into a captured name and must be trimmed.
_LEADING_LABEL = re.compile(
    r"^(?:доставчик|получател|продавач|купувач|клиент|изпълнител)\b[\s:.\-]*",
    re.IGNORECASE,
)


def _clean_name(raw: str) -> str:
    name = " ".join(raw.split()).strip(" .,;:-\"'")
    prev = None
    while prev != name:  # strip repeated leading role labels
        prev = name
        name = _LEADING_LABEL.sub("", name).strip(" .,;:-\"'")
    return name


def _extract_party_name(text: str, labels: str) -> ExtractedField:
    # The legal-form suffix must be its own token (a space before it, a boundary after),
    # otherwise a short suffix like "ЕТ" would match inside a name such as "БЕТА".
    pat = rf"(?:{labels})\s*[:\-]?\s*([А-Яа-яA-Za-z0-9\s\-\"'.,]+?)\s+({_COMPANY_SUFFIX})\b"
    m = re.search(pat, text, re.IGNORECASE)
    if m:
        name = _clean_name(f"{m.group(1)} {m.group(2)}")
        return ExtractedField(value=name or None, confidence=_HIGH if name else 0.0)
    return ExtractedField(value=None, confidence=0.0)


_SUPPLIER_LABELS = "Доставчик|ДОСТАВЧИК|Продавач|Изпълнител"
_RECIPIENT_LABELS = "Получател|ПОЛУЧАТЕЛ|Купувач|Клиент"


def extract_supplier_name(text: str) -> ExtractedField:
    return _extract_party_name(text, _SUPPLIER_LABELS)


def extract_recipient_name(text: str) -> ExtractedField:
    return _extract_party_name(text, _RECIPIENT_LABELS)


_NAME_FINDER = re.compile(
    rf"[А-Яа-яA-Za-z0-9\-\"'.,]+(?:\s+[А-Яа-яA-Za-z0-9\-\"'.,]+){{0,5}}?\s+{_COMPANY_SUFFIX}\b",
    re.IGNORECASE,
)


def _column_split_blocks(text: str) -> tuple[str, str] | None:
    """Handle the common two-column header (supplier label and recipient label printed
    side-by-side, with the two company names stacked beneath them in two columns). Embedded
    text (pdftotext -layout) keeps both columns on the same line, so a naive label-order
    split interleaves the names. Split the party region at the second company's column.
    Returns (supplier_block, recipient_block) or None when it's not a two-column header.
    """
    lines = text.split("\n")
    for i, line in enumerate(lines):
        ms = re.search(_SUPPLIER_LABELS, line, re.IGNORECASE)
        mr = re.search(_RECIPIENT_LABELS, line, re.IGNORECASE)
        if not (ms and mr) or abs(ms.start() - mr.start()) <= 3:
            continue
        region = lines[i:i + 12]
        # Preferred: split at the second company's column (two names share a line).
        cut = next((m[1].start() for l in region
                    if len(m := list(_NAME_FINDER.finditer(l))) >= 2), None)
        # Fallback: the names sit on separate lines but the two columns are still interleaved
        # on every row below (each row holds left-party then right-party fields — the case
        # that silently swapped КВК-style invoices). Split at the right-hand label's column,
        # which is where the right column of data begins.
        if cut is None:
            cut = max(ms.start(), mr.start())
        if cut <= 0:
            continue
        left = "\n".join(l[:cut] for l in region)
        right = "\n".join(l[cut:] for l in region)
        return (left, right) if ms.start() < mr.start() else (right, left)
    return None


def _split_party_blocks(text: str) -> tuple[str, str]:
    """Split the document into the supplier and recipient sections by their role labels.

    Handles side-by-side columns and either top-to-bottom ordering; returns ("", "") for a
    block that has no label.
    """
    columns = _column_split_blocks(text)
    if columns is not None:
        return columns
    sup = re.search(_SUPPLIER_LABELS, text, re.IGNORECASE)
    rec = re.search(_RECIPIENT_LABELS, text, re.IGNORECASE)
    if sup and rec:
        if sup.start() < rec.start():
            return text[sup.start():rec.start()], text[rec.start():]
        return text[sup.start():], text[rec.start():sup.start()]
    if sup:
        return text[sup.start():], ""
    if rec:
        return "", text[rec.start():]
    return "", ""


def _assign_owners(
    sup_block: str, rec_block: str, all_values: list[str], finder
) -> tuple[str | None, str | None]:
    """Assign extracted values (EIKs/VATs) to supplier vs recipient by block, with a
    global best-effort fallback when the sections didn't separate them."""
    sup = next(iter(finder(sup_block)), None)
    rec = next(iter(finder(rec_block)), None)
    if sup is None and rec is None and all_values:
        sup = all_values[0]
        rec = all_values[1] if len(all_values) > 1 else None
    elif sup is None and rec is not None:
        sup = next((v for v in all_values if v != rec), None)
    elif rec is None and sup is not None:
        rec = next((v for v in all_values if v != sup), None)
    return sup, rec


def _apply_low_conf(field: ExtractedField, low_conf_tokens: set[str] | None) -> ExtractedField:
    """Down-weight a captured field if any of its words were recognised with low OCR
    confidence, so recovery (register / vision) knows to revisit it."""
    if not low_conf_tokens or not field.value:
        return field
    tokens = {normalize_token(t) for t in field.value.split()}
    if tokens & low_conf_tokens:
        field.confidence = min(field.confidence, _LOW)
    return field


# Tolerant separator: optional colon/dash, optional currency (BGN / лв / EUR), then amount.
# Handles real layouts like "ОБЩА СТОЙНОСТ: BGN 141.60".
_CUR = r"(?:BGN|лв\.?|лева|EUR|€|евро)?"
# Tolerant separator: optional colon/dash and an optional bracketed currency annotation
# (": ", " [EUR]: ", " (лв): ") between the label and the amount.
_SEP = rf"\s*[:\-\[\(]*\s*{_CUR}\s*[:\-\]\)]*\s*"
# A monetary amount: integer part either run-together (30686.57) or grouped with
# thousands separators (space / nbsp / dot / comma: "3,580.85", "2 259,90"), then a
# 2-digit decimal. clean_amount() normalizes the locale afterwards.
_AMT = r"(-?(?:\d{1,3}(?:[   .,]\d{3})+|\d+)[.,]\d{2})(?![./-]\d)"


_CUR_TOKENS = {"EUR": ("eur", "€", "евро"), "BGN": ("лв", "bgn", "лева")}
_OPPOSITE = {"EUR": "BGN", "BGN": "EUR"}


def _line_at(text: str, pos: int) -> str:
    start = text.rfind("\n", 0, pos) + 1
    end = text.find("\n", pos)
    return text[start: end if end != -1 else len(text)]


def _amount_near_currency(line: str, same: tuple[str, ...]) -> Decimal | None:
    """On a line that lists both currencies ("... -269.79 лв. -137.94 €"), return the
    amount immediately followed by one of the wanted currency tokens."""
    for m in re.finditer(_AMT, line):
        if any(t in line[m.end():m.end() + 5].lower() for t in same):
            return clean_amount(m.group(1))
    return None


def _extract_amount(text: str, labels: list[str], currency: str | None = None) -> ExtractedField:
    """Pull a labelled amount, kept in the invoice's currency. Euro-transition invoices
    print both EUR and BGN — sometimes as separate lines ("Общо с ДДС [EUR]: 46.19" /
    "(лв): 90.34"), sometimes on one line ("Данъчна основа: -269.79 лв. -137.94 €"). Pick
    the value carrying the detected currency; never a value off the other currency."""
    same = _CUR_TOKENS.get(currency or "")
    opp = _CUR_TOKENS.get(_OPPOSITE.get(currency or "", ""))

    def from_line(line: str, fallback_amt: str | None) -> Decimal | None:
        low = line.lower()
        has_same = bool(same) and any(t in low for t in same)
        has_opp = bool(opp) and any(t in low for t in opp)
        if has_same and has_opp:                       # both currencies on the line
            return _amount_near_currency(line, same)
        if has_opp and not has_same:                   # other-currency-only line
            return None
        return clean_amount(fallback_amt) if fallback_amt else None

    # strict: amount right after the label
    for label in labels:
        for m in re.finditer(label + _SEP + _AMT, text, re.IGNORECASE):
            amt = from_line(_line_at(text, m.start()), m.group(1))
            if amt is not None:
                return ExtractedField(value=str(amt), confidence=_HIGH)
    # tolerant same-line: a rate clause and wide column whitespace can sit between the label
    # and its amount ("Данъчна основа (за 20%):      94.83", "Данъчна основа 20%: 240.00 €").
    # Layout mode can push the value far to the right, so the gap spans (almost) the whole
    # line. _AMT needs two decimals, so a bare "20%" is never taken for the value; the
    # non-greedy gap picks the first real amount after the label (its own row's value).
    for label in labels:
        for m in re.finditer(label + r"[^\n]{0,120}?" + _AMT, text, re.IGNORECASE):
            amt = from_line(_line_at(text, m.start()), m.group(1))
            if amt is not None:
                return ExtractedField(value=str(amt), confidence=_HIGH)
    # fallback ONLY for a dual-currency line where a rate ("20%") sits between the label
    # and the amount, e.g. "Данъчна основа 20%: -269.79 лв. -137.94 €".
    if same and opp:
        for label in labels:
            for lm in re.finditer(label, text, re.IGNORECASE):
                line = _line_at(text, lm.start())
                low = line.lower()
                if any(t in low for t in same) and any(t in low for t in opp):
                    amt = _amount_near_currency(line, same)
                    if amt is not None:
                        return ExtractedField(value=str(amt), confidence=_HIGH)
    return ExtractedField(value=None, confidence=0.0)


def _recover_vat_total_from_row(text: str, net: Decimal) -> tuple[Decimal, Decimal] | None:
    """Last-resort, self-verifying recovery of (vat, total) anchored on a TRUSTED net.

    Some scans (esp. a broken-font text layer that OCR re-reads) get the net right but drop
    or mangle the VAT/total labels, so the label-based extractor picks the base for the total
    and leaves VAT at 0. The amounts themselves are still on the page — on the summary row
    "<base> <rate> <vat> <total>". We scan each line that contains the net figure and look for
    two more amounts b, c on it with net + b = c (±0.02) and b/net a real BG VAT rate
    (~9% or ~20%). Both the arithmetic and the rate must hold, so this cannot corrupt a
    correctly-read invoice; a genuinely 0-rated invoice (no such pair) is left untouched."""
    rates = (Decimal("0.09"), Decimal("0.20"))
    for line in text.splitlines():
        amts = [clean_amount(m.group(1)) for m in re.finditer(_AMT, line)]
        amts = [a for a in amts if a is not None]
        if net not in amts:
            continue
        for b in amts:
            if b is None or b <= 0 or b == net:
                continue
            for c in amts:
                if c is None or abs(net + b - c) > Decimal("0.02"):
                    continue
                if any(abs(b / net - r) < Decimal("0.006") for r in rates):
                    return b, c
    return None


def extract_net_amount(text: str, currency: str | None = None) -> ExtractedField:
    return _extract_amount(text, [
        r"Данъчна\s+основа",
        r"Сума\s+без\s+ДДС",
        r"Общо\s+без\s+ДДС",
        r"Облагаема\s+стойност",
        r"Стойност\s+без\s+ДДС",
        # telecom/utility wording: "Обща стойност облагаеми 20% ДДС", "Обща стойност на
        # услугите" — the taxable base. Specific so it never matches "Обща стойност с ДДС".
        r"Обща\s+стойност\s+облагаеми",
        r"Обща\s+стойност\s+на\s+услугите",
        r"Нето",
        # a bare "Общо <amount>" subtotal, but NOT a total line ("Общо с ДДС", "Общо (BGN)").
        r"Общо(?!\s*(?:с\s+ДДС|за\s+плащане|стойност|\(|\[))",
    ], currency)


def extract_vat_amount(text: str, currency: str | None = None) -> ExtractedField:
    # only the explicitly labelled VAT amount; a bare "ДДС 20%" would capture the
    # rate, not the amount. when absent, the value is derived from total - net.
    return _extract_amount(text, [
        r"Размер\s+на\s+данъка",
        r"Начислен\s+ДДС",
        r"Сума\s+на\s+ДДС",
        r"Стойност\s+на\s+ДДС",
        r"ДДС\s+ставка\s+и\s+сума",  # telecom wording: "ДДС ставка и сума  20%  12.81"
        r"B00\s*[-—–]?\s*ДДС",  # customs declaration VAT tax line: "B00 - ДДС 6148.18"
    ], currency)


_TOTAL_LABELS = [
    r"Обща\s+стойност\s+за\s+плащане",
    r"Обща\s+стойност\s+на\s+фактурата",
    r"Обща\s+стойност",
    r"Всичко\s+за\s+плащане",
    r"Сума\s+за\s+плащане",
    r"Крайна\s+сума",
    r"Дължима\s+сума",
    r"Общо\s+за\s+плащане",
    r"Обща\s+сума\s+с\s+ДДС",
    r"Общо\s+с\s+ДДС",
    r"Сума\s+с\s+ДДС",
    r"Общо\s*\(\s*BGN\s*\)",
]


def extract_total_amount(text: str, currency: str | None = None) -> ExtractedField:
    f = _extract_amount(text, _TOTAL_LABELS, currency)
    if f.value is not None:
        return f
    # Tolerant fallback: real layouts put a few words between the label and the amount
    # ("Обща стойност на фактурата 131.62", "Общо за плащане в лева: 3,580.85"). Allow a
    # short non-digit gap; skip lines carrying the opposite currency on dual-currency docs.
    opp = _CUR_TOKENS.get(_OPPOSITE.get(currency or "", ""))
    for label in _TOTAL_LABELS:
        for m in re.finditer(label + r"[^\d\n]{0,20}?" + _AMT, text, re.IGNORECASE):
            if opp and any(t in _line_at(text, m.start()).lower() for t in opp):
                continue
            amt = clean_amount(m.group(1))
            if amt is not None:
                return ExtractedField(value=str(amt), confidence=_LOW)
    return f


_DUE_LABELS = r"(?:дата\s+на\s+падеж|дата\s+падеж|падеж|срок\s+за\s+плащане|срок\s+на\s+плащане)"
# Explicit "payment method" labels only. A bare "Плащане" counts too, but never the "…за
# плащане" of a totals line ("Всичко за плащане", "Сума за плащане") — excluded by lookbehind.
_PAY_LABELS = r"(?:начин\s+на\s+плащане|условие\s+на\s+плащане|(?<!за )плащане)"
_DATE_TOK = r"(\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4}|\d{4}[.\-/]\d{1,2}[.\-/]\d{1,2})"


_PAY_KEYWORDS = (
    "брой", "кеш", "cash", "нало", "карт", "card", "банк", "превод",
    "платежно", "по сметка", "bank", "wire", "swift",
)


def _payment_method_code(raw: str, has_iban: bool) -> str:
    """Map the free-text payment field to the frontend enum (bank/card/cash/cod). The
    explicit field wins; an IBAN only decides when no words are present."""
    r = raw.lower()
    if "брой" in r or "кеш" in r or "cash" in r:
        return "cash"
    if "нало" in r:  # наложен платеж
        return "cod"
    if "карт" in r or "card" in r:
        return "card"
    if any(k in r for k in ("банк", "превод", "платежно", "по сметка", "bank", "wire")):
        return "bank"
    return "bank"  # default; safe for the common IBAN-only case


def extract_payment_and_due(text: str) -> dict[str, str]:
    """Read the payment method (verbatim + enum) and the due date into a plain dict for
    Invoice.extra. Free and deterministic; the vision pass fills these when OCR text is poor."""
    out: dict[str, str] = {}
    m = re.search(_DUE_LABELS + r"[^\n\d]{0,15}" + _DATE_TOK, text, re.IGNORECASE)
    if m:
        iso = normalize_date(m.group(1))
        if _valid_iso(iso):
            out["due_date"] = iso
    pm = re.search(_PAY_LABELS + r"\s*[:\-]?\s*([^\n]{1,40})", text, re.IGNORECASE)
    if pm:
        raw = re.split(r"\s{2,}|IBAN|Банка|Дата|BIC", pm.group(1).strip(" .:-"),
                       maxsplit=1, flags=re.IGNORECASE)[0].strip()
        # Store only a value carrying a real payment keyword — mangled OCR ("отъ") is dropped
        # so the vision pass, which reads the field far better on photos, fills it instead.
        if raw and any(k in raw.lower() for k in _PAY_KEYWORDS):
            out["payment_method_raw"] = raw
            out["payment_method"] = _payment_method_code(raw, "IBAN" in text.upper())
    return out


def parse_invoice_fields(
    text: str, low_conf_tokens: set[str] | None = None
) -> dict[str, ExtractedField]:
    """Extract all known fields with confidence. Keys are stable field names.

    Each party's name/VAT/EIK is read from its own block so a single party never mixes
    two companies' data; names fall back to a whole-document scan when a block is empty.
    """
    sup_block, rec_block = _split_party_blocks(text)
    currency = detect_currency_text(text)  # amounts are picked in the invoice's currency

    fields = {
        "number": extract_invoice_number(text),
        "date": extract_date(text),
        "supplier_name": _apply_low_conf(extract_supplier_name(sup_block or text), low_conf_tokens),
        "recipient_name": _apply_low_conf(extract_recipient_name(rec_block or text), low_conf_tokens),
        "net_amount": extract_net_amount(text, currency),
        "vat_amount": extract_vat_amount(text, currency),
        "total_amount": extract_total_amount(text, currency),
    }

    sup_vat, rec_vat = _assign_owners(sup_block, rec_block, _all_vat(text), _all_vat)
    fields["supplier_vat"] = ExtractedField(value=sup_vat, confidence=_LOW if sup_vat else 0.0)
    fields["recipient_vat"] = ExtractedField(value=rec_vat, confidence=_LOW if rec_vat else 0.0)

    sup_eik, rec_eik = _assign_owners(sup_block, rec_block, _eik_list(text), _eik_list)
    fields["supplier_eik"] = ExtractedField(value=sup_eik, confidence=_HIGH if sup_eik else 0.0)
    fields["recipient_eik"] = ExtractedField(value=rec_eik, confidence=_HIGH if rec_eik else 0.0)
    return fields


def resolve_perspective(perspective: str, direction: str) -> str:
    """Resolve the party-of-interest. "auto" follows the VAT direction; an explicit
    "supplier"/"recipient" is kept as-is."""
    if perspective in ("supplier", "recipient"):
        return perspective
    if direction == Direction.SALE.value:
        return "supplier"
    if direction == Direction.PURCHASE.value:
        return "recipient"
    return "supplier"


def swap_parties(invoice: Invoice) -> Invoice:
    """Swap supplier and recipient, including their per-field confidences. Used as the
    manual override when the roles were captured the wrong way round."""
    invoice.supplier, invoice.recipient = invoice.recipient, invoice.supplier
    fc = invoice.field_confidence
    for a, b in (
        ("supplier_name", "recipient_name"),
        ("supplier_vat", "recipient_vat"),
        ("supplier_eik", "recipient_eik"),
    ):
        if a in fc or b in fc:
            fc[a], fc[b] = fc.get(b, 0.0), fc.get(a, 0.0)
    return invoice


def derive_eiks_from_vat(invoice: Invoice) -> Invoice:
    """A Bulgarian company VAT is 'BG' + EIK, so when only the VAT was read, the EIK is the
    VAT's digits (when the checksum holds). This is free and offline — no register call — so
    it runs unconditionally, unlike the network name lookup in recover_parties."""
    for party in (invoice.supplier, invoice.recipient):
        if not party.eik and party.vat_number:
            cand = re.sub(r"\D", "", party.vat_number)
            if validate_eik(cand):
                party.eik = cand
        # And the reverse: a BG company's VAT is 'BG' + its EIK, so fill a missing VAT from
        # a checksum-valid EIK (only for plain 9/13-digit EIKs, i.e. Bulgarian entities).
        if not party.vat_number and party.eik and validate_eik(party.eik):
            party.vat_number = f"BG{party.eik}"
    return invoice


def recover_parties(invoice: Invoice) -> Invoice:
    """Fill or correct counterparty fields from the commercial register, keyed by a
    checksum-valid EIK. A name is overridden only when missing or low-confidence, so a
    clean reading from the document is trusted over the register."""
    derive_eiks_from_vat(invoice)  # free/offline; also useful when the register is disabled
    if not get_settings().company_lookup_enabled:
        return invoice
    for party, name_key, vat_key in (
        (invoice.supplier, "supplier_name", "supplier_vat"),
        (invoice.recipient, "recipient_name", "recipient_vat"),
    ):
        if not validate_eik(party.eik or ""):
            continue
        info = lookup_company(party.eik)
        if info is None:
            continue
        recovered: list[str] = []
        name_conf = invoice.field_confidence.get(name_key, 0.0)
        if info.name and (not party.name or name_conf < _HIGH):
            party.name = info.name
            invoice.field_confidence[name_key] = 0.95
            recovered.append("name")
        if info.vat_number and not party.vat_number:
            party.vat_number = info.vat_number
            invoice.field_confidence[vat_key] = 0.95
            recovered.append("vat_number")
        if info.address_line1 and not party.address:
            party.address = info.address_line1
            recovered.append("address")
        if recovered:
            party.source = "merged"
            party.recovered_fields = recovered
    return invoice


def _apply_field_models(invoice: Invoice, text: str, f: dict[str, ExtractedField]) -> None:
    """Let the trained selectors fill fields the deterministic pass was weak on. Each value
    is a verbatim candidate; amounts are only *selected*, never invented. No-op without
    models. Imported lazily to avoid the candidates<->invoice_extractor import cycle."""
    from . import field_models

    if not field_models.available():
        return

    sel_amt = field_models.select_amounts(text)
    for cls, attr in (("net", "net_amount"), ("vat", "vat_amount"), ("total", "total_amount")):
        if f[attr].confidence < _HIGH and cls in sel_amt:
            try:
                setattr(invoice, attr, Decimal(sel_amt[cls]))
                invoice.field_confidence[attr] = 0.8
            except (InvalidOperation, TypeError):
                pass

    sel_p = field_models.select_parties(text)
    for cls, party in (("supplier", invoice.supplier), ("recipient", invoice.recipient)):
        if f[f"{cls}_name"].confidence < _HIGH and cls in sel_p:
            cand = sel_p[cls]
            party.name = cand.name
            invoice.field_confidence[f"{cls}_name"] = 0.8
            if cand.eik and not party.eik:
                party.eik = cand.eik
            if cand.vat and not party.vat_number:
                party.vat_number = cand.vat

    if f["number"].confidence < _HIGH:
        exclude = _all_eik(text) | {v.removeprefix("BG") for v in _all_vat(text)}
        n = field_models.select_number(text, exclude)
        if n:
            invoice.number = n
            invoice.field_confidence["number"] = 0.8
    if f["date"].confidence < _HIGH:
        d = field_models.select_date(text)
        if d:
            invoice.date = d
            invoice.field_confidence["date"] = 0.8

    if invoice.direction == Direction.UNKNOWN.value:
        dr = field_models.select_direction(text)
        if dr:
            invoice.direction = dr


def extract_invoice_from_text(
    text: str,
    doc_id: str,
    source: str = "ocr",
    *,
    perspective: str = "auto",
    low_conf_tokens: set[str] | None = None,
) -> Invoice:
    """Build a typed Invoice from raw OCR/plain text."""
    f = parse_invoice_fields(text, low_conf_tokens)

    def amt(key: str) -> Decimal | None:
        v = f[key].value
        return Decimal(v) if v else None

    invoice = Invoice(
        id=doc_id,
        source=source,
        currency=detect_currency_text(text) or "BGN",
        number=f["number"].value,
        date=f["date"].value,
        supplier=Party(
            name=f["supplier_name"].value,
            vat_number=f["supplier_vat"].value,
            eik=f["supplier_eik"].value,
        ),
        recipient=Party(
            name=f["recipient_name"].value,
            vat_number=f["recipient_vat"].value,
            eik=f["recipient_eik"].value,
        ),
        net_amount=amt("net_amount"),
        vat_amount=amt("vat_amount"),
        total_amount=amt("total_amount"),
        field_confidence={k: v.confidence for k, v in f.items()},
        doc_type=classify_document_type(text).value,
        direction=detect_direction(text).value,
        reverse_charge=detect_reverse_charge(text),
    )

    # Trained selectors fill any weak fields (verbatim candidates; amounts selection-only).
    _apply_field_models(invoice, text, f)

    # Recover the total from the "Словом:" words line when the numeric one is garbled.
    if invoice.total_amount is None:
        words_total = total_from_words(text)
        if words_total is not None:
            invoice.total_amount = words_total
            invoice.field_confidence["total_amount"] = _LOW

    # Fill in the missing leg when two of net/VAT/total are known.
    net, vat, total = invoice.net_amount, invoice.vat_amount, invoice.total_amount
    if total is None and net is not None and vat is not None:
        invoice.total_amount = net + vat
    elif net is None and total is not None and vat is not None:
        invoice.net_amount = total - vat

    # Suspicious amount reading (VAT 0/absent, or the total equals the base) but the net was
    # read confidently: try to recover a VAT+total pair off the summary row, anchored on that
    # net. Only fires when the arithmetic AND a real VAT rate both check out (see helper), so
    # a truly 0-rated invoice stays as-is. Fixes broken-font scans whose VAT/total labels are
    # unreadable while the figures are intact.
    n = invoice.net_amount
    if (n is not None and n > 0
            and invoice.field_confidence.get("net_amount", 0.0) >= _HIGH
            and (invoice.vat_amount in (None, Decimal("0"))
                 or invoice.total_amount in (None, invoice.net_amount))):
        rec = _recover_vat_total_from_row(text, n)
        if rec is not None:
            rec_vat, rec_total = rec
            if invoice.field_confidence.get("vat_amount", 0.0) < _HIGH:
                invoice.vat_amount = rec_vat
                invoice.field_confidence["vat_amount"] = 0.85
            if invoice.field_confidence.get("total_amount", 0.0) < _HIGH or invoice.total_amount == n:
                invoice.total_amount = rec_total
                invoice.field_confidence["total_amount"] = 0.85

    # Recompute/derive VAT = total - net only when that difference is a plausible BG VAT
    # rate (0 / ~9% / ~20% of the base). This fills a missing or garbled VAT without
    # corrupting invoices that carry non-taxable components (leasing, telecom balances)
    # where total != net + VAT by design. Deterministic arithmetic, so it stays auditable.
    n, v, t = invoice.net_amount, invoice.vat_amount, invoice.total_amount
    # Only fill/fix a weak VAT — never override one we read explicitly (e.g. a customs
    # "B00 - ДДС" line), since the total may itself be a mis-picked figure. n != 0 so
    # credit notes (negative) still reconcile.
    if (n is not None and t is not None and n != 0
            and invoice.field_confidence.get("vat_amount", 0.0) < _HIGH):
        derived = t - n
        rate = abs(derived / n)
        plausible = any(abs(rate - r) < Decimal("0.015") for r in (Decimal("0"), Decimal("0.09"), Decimal("0.20")))
        inconsistent = v is None or abs((n + v) - t) > Decimal("0.02")
        if plausible and inconsistent:
            invoice.vat_amount = derived
            invoice.field_confidence["vat_amount"] = max(invoice.field_confidence.get("vat_amount", 0.0), 0.8)

    # Derive a VAT tax line when we have base + amount.
    if invoice.net_amount and invoice.vat_amount and invoice.net_amount > 0:
        rate = (invoice.vat_amount / invoice.net_amount).quantize(Decimal("0.01"))
        invoice.tax_lines.append(
            TaxLine(rate=rate, base=invoice.net_amount, amount=invoice.vat_amount)
        )

    # Item rows from the column-preserving text layer (digital PDFs). Left empty on
    # photo/image OCR, whose reflowed text has no stable columns — vision fills those.
    # Local import avoids a circular dependency (line_items reuses clean_amount here).
    from .line_items import parse_line_items
    if not invoice.line_items:
        rows = parse_line_items(text)
        # Sanity gate: a correct table's amounts sum to the tax base. When they miss it by a
        # wide margin the header slicing went wrong (e.g. a running-balance column captured
        # as rows), so drop the misparse — an empty list routes the invoice to the vision
        # reader, which is far more reliable on item tables than column heuristics.
        if rows and invoice.net_amount and invoice.net_amount > 0:
            s = sum((li.amount for li in rows if li.amount is not None), Decimal(0))
            if not (invoice.net_amount * Decimal("0.5") <= s <= invoice.net_amount * Decimal("1.5")):
                rows = []
        invoice.line_items = rows

    # Payment method + due date into extra (the frontend reads payment_method_raw/due_date).
    for k, v in extract_payment_and_due(text).items():
        invoice.extra.setdefault(k, v)

    invoice.perspective = resolve_perspective(perspective, invoice.direction)
    recover_parties(invoice)
    return tag_company(invoice)
