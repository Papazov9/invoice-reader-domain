"""Detection of Bulgarian accounting document types and VAT direction."""

from __future__ import annotations

import re
from enum import Enum


class DocumentType(str, Enum):
    INVOICE = "invoice"                      # Фактура
    CREDIT_NOTE = "credit_note"              # Кредитно известие
    DEBIT_NOTE = "debit_note"                # Дебитно известие
    PROFORMA = "proforma"                    # Проформа фактура
    SIMPLIFIED_INVOICE = "simplified_invoice"  # Опростена фактура
    PROTOCOL = "protocol"                    # Протокол (напр. по чл. 117 ЗДДС)
    FISCAL_RECEIPT = "fiscal_receipt"        # Касов / фискален бон
    CUSTOMS_DECLARATION = "customs_declaration"  # Митническа декларация
    BANK_STATEMENT = "bank_statement"        # Банково извлечение
    GOODS_RECEIPT = "goods_receipt"          # Стокова / складова разписка
    EXPENSE_REPORT = "expense_report"        # Авансов отчет
    BALANCE_CONFIRMATION = "balance_confirmation"  # Одиторско потвърждение на салда / разчети
    OTHER = "other"


class Direction(str, Enum):
    SALE = "sale"          # Продажба (издадена) - изходящо ДДС
    PURCHASE = "purchase"  # Покупка (получена) - входящо ДДС
    UNKNOWN = "unknown"


# Decisive phrases for the specific (non-invoice) types, most specific first. A
# customs declaration or credit note also contains "фактура", so these are checked
# before the invoice baseline. Presence of any phrase fixes the type.
_SPECIFIC_PRIMARY: list[tuple[DocumentType, tuple[str, ...]]] = [
    (DocumentType.CREDIT_NOTE, ("кредитно известие", "кредитно изв", "credit note")),
    (DocumentType.DEBIT_NOTE, ("дебитно известие", "дебитно изв", "debit note")),
    (DocumentType.PROFORMA, ("проформа", "pro-forma", "proforma")),
    (DocumentType.SIMPLIFIED_INVOICE, ("опростена фактура",)),
    (DocumentType.PROTOCOL, ("протокол по чл", "протокол №", "протокол no")),
    (DocumentType.FISCAL_RECEIPT,
     ("фискален бон", "касов бон", "касова бележка", "фискална касова", "fiscal receipt")),
    (DocumentType.CUSTOMS_DECLARATION,
     ("митническа декларация", "единен административен документ",
      "single administrative document", "ескд")),
    (DocumentType.BANK_STATEMENT,
     ("банково извлечение", "извлечение по сметка", "движение по сметка", "отчет по сметка",
      "bank statement", "account statement")),
    (DocumentType.GOODS_RECEIPT,
     ("стокова разписка", "складова разписка", "delivery note", "приемо-предавателен")),
    (DocumentType.EXPENSE_REPORT, ("авансов отчет", "отчет за разходи", "expense report")),
    # Audit balance/turnover confirmation request — has a "Фактура №" schedule attached,
    # so it must be caught before the invoice baseline.
    (DocumentType.BALANCE_CONFIRMATION,
     ("потвърдите салдата", "потвърждение на салд", "потвърждаване на салд", "салдата по разчети",
      "одиторско потвърждение", "потвърждение на разчети", "balance confirmation",
      "справка за фактурирани и неизплатени")),
]

# Weaker, weighted signals used when no decisive phrase is found — e.g. an H1/CDS
# customs declaration whose "митническа декларация" header was OCR-garbled, or a
# format variant. These are scored so no single garbled token decides the type.
_SECONDARY_SIGNALS: dict[DocumentType, tuple[tuple[str, int], ...]] = {
    DocumentType.CUSTOMS_DECLARATION: (
        ("митническа стойност", 3), ("допускане за свободно обращение", 4),
        ("свободно обращение", 2), ("ставки на митата", 2),
        ("вносител", 2), ("износител", 2), ("non-union goods", 3), ("тарик", 2),
    ),
    DocumentType.BANK_STATEMENT: (
        ("начално салдо", 3), ("крайно салдо", 3), ("движение по сметка", 3),
    ),
    DocumentType.FISCAL_RECEIPT: (
        ("фискален", 2), ("фу рег", 2), ("артикул", 1),
    ),
}
_SECONDARY_MIN = 4  # minimum score for a secondary match to win over the invoice baseline

# An 18-character Movement Reference Number (year + country + 14 alphanumerics) is a
# strong customs signal on its own.
_MRN_RE = re.compile(r"\b\d{2}[A-Z]{2}[A-Z0-9]{14}\b")

_INVOICE_KEYWORDS = ("фактура", "ф-ра", "invoice", "факт. №")

_SALE_HINTS = ("продажб", "prodajb", "издадена фактура", "издадени фактури", "изх. фактура", "sales")
_PURCHASE_HINTS = (
    "покупк", "pokupk", "получена фактура", "получени фактури", "вх. фактура", "purchase",
    "допускане за свободно обращение", "import of non-union", "вносител",
)
_REVERSE_CHARGE = (
    "чл. 117", "чл.117", "чл 117", "чл. 163", "чл.163",
    "обратно начисляване", "обратно начисление", "reverse charge",
    "вътреобщностно придобиване",
)


def detect_document_type(text: str) -> DocumentType:
    """Classify a document. Decisive phrases win first; otherwise weighted secondary
    signals (incl. an MRN) catch format variants; the invoice baseline is the fallback."""
    t = (text or "").lower()

    # An invoice template often lists BOTH "кредитно известие" and "дебитно известие" as
    # options in its header — that's not a note, it's a (usually) plain фактура. A real
    # credit/debit note names only its own type.
    note_template = "кредитно известие" in t and "дебитно известие" in t

    for doc_type, keywords in _SPECIFIC_PRIMARY:
        if note_template and doc_type in (DocumentType.CREDIT_NOTE, DocumentType.DEBIT_NOTE):
            continue
        if any(k in t for k in keywords):
            return doc_type

    best_type, best_score = DocumentType.OTHER, 0
    for doc_type, signals in _SECONDARY_SIGNALS.items():
        score = sum(weight for kw, weight in signals if kw in t)
        # An MRN only counts as customs when other customs signals are also present —
        # otherwise line-item product codes (e.g. "50IP29C01604005101") false-trigger it.
        if doc_type is DocumentType.CUSTOMS_DECLARATION and score > 0 and _MRN_RE.search(text or ""):
            score += 4
        if score > best_score:
            best_type, best_score = doc_type, score
    if best_score >= _SECONDARY_MIN:
        return best_type

    if any(k in t for k in _INVOICE_KEYWORDS):
        return DocumentType.INVOICE
    return DocumentType.OTHER


def classify_document_type(text: str) -> DocumentType:
    """Document type with an assist from the trained classifier. The keyword detector
    wins whenever it is sure; only when it falls through to OTHER do we consult the model,
    and only if the model is enabled and confident. The model never affects any other
    field — just the type."""
    kw = detect_document_type(text)
    if kw is not DocumentType.OTHER:
        return kw

    from app.core import get_settings

    settings = get_settings()
    if not settings.doctype_classifier_enabled:
        return kw
    from app.tools.ingest import classifier

    pred = classifier.predict(text)
    if pred and pred[1] >= settings.doctype_model_min_proba:
        try:
            return DocumentType(pred[0])
        except ValueError:  # model emitted an unknown label
            return kw
    return kw


def detect_direction(text: str, hint: str = "") -> Direction:
    t = f"{text or ''} {hint or ''}".lower()
    # Strong import/export markers (customs) outrank the generic продажба/покупка words.
    if "допускане за свободно обращение" in t or "import of non-union" in t:
        return Direction.PURCHASE
    if "декларация за износ" in t or "export declaration" in t:
        return Direction.SALE
    if any(k in t for k in _SALE_HINTS):
        return Direction.SALE
    if any(k in t for k in _PURCHASE_HINTS):
        return Direction.PURCHASE
    return Direction.UNKNOWN


def detect_reverse_charge(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in _REVERSE_CHARGE)
