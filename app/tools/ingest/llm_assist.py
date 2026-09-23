"""LLM few-shot assist for weak NON-AMOUNT fields on hard documents.

When the deterministic pass can't read the type or the parties (garbled scan, unusual
layout), an LLM reads the text and fills ONLY the missing/low-confidence non-amount fields
(document type signal, party names/VAT/EIK, number, date). Amounts and VAT are never sent
for override — they stay rule-computed and auditable.

Gated by llm_assist_enabled + a configured model. Few-shot examples are drawn from the
local labeled dataset when present (dev), otherwise it runs zero-shot. Kept independent of
app.rag so importing the ingest layer stays lightweight.
"""

from __future__ import annotations

from app.core import get_settings
from app.domain import ExtractedField, Invoice

from .vision_extract import _parse_json

_HIGH = 0.9
_ASSIST_CONF = 0.85

_NONAMOUNT_FIELDS = (
    "number", "date", "currency",
    "supplier_name", "supplier_vat", "supplier_eik",
    "recipient_name", "recipient_vat", "recipient_eik",
)

_PROMPT = (
    "You read Bulgarian accounting documents. From the text, return ONLY a JSON object with "
    "these string keys (null when absent): number, date (ISO YYYY-MM-DD), currency, "
    "supplier_name, supplier_vat, supplier_eik, recipient_name, recipient_vat, recipient_eik. "
    "Доставчик/Продавач/Изпълнител = supplier; Получател/Купувач/Клиент = importer is "
    "recipient. Do NOT return amounts. No commentary, no code fences."
)

_examples_cache: dict[str, list[tuple[str, dict]]] | None = None


def _resolve_model() -> tuple[str, str, str]:
    """(model, api_base, api_key) — mirrors app.rag.llm.resolve_llm without importing the
    heavy rag package: a hosted model wins, else the local/containerized fallback."""
    s = get_settings()
    if s.llm_model:
        return s.llm_model, s.llm_api_base, s.llm_api_key
    if s.llm_fallback_enabled and s.llm_fallback_model:
        return s.llm_fallback_model, s.llm_fallback_api_base, ""
    return "", "", ""


def is_available() -> bool:
    return get_settings().llm_assist_enabled and bool(_resolve_model()[0])


def should_assist(invoice: Invoice) -> bool:
    """Trigger only on genuinely weak non-amount results."""
    if not is_available():
        return False
    if invoice.doc_type == "other" or not invoice.number:
        return True
    def weak(party) -> bool:
        return not (party.name or party.vat_number or party.eik)
    return weak(invoice.supplier) or weak(invoice.recipient)


def assist_fields(text: str, doc_type: str = "", complete=None) -> dict[str, ExtractedField] | None:
    if not is_available() or not (text or "").strip():
        return None
    complete = complete or _complete
    try:
        raw = complete(_build_messages(text, doc_type))
        data = _parse_json(raw)
    except Exception:  # pragma: no cover - model/transport failure
        return None
    return _to_fields(data) if data else None


def merge_fields(invoice: Invoice, fields: dict[str, ExtractedField]) -> Invoice:
    """Fill only missing/low-confidence NON-AMOUNT fields. Amounts are untouched."""
    fc = invoice.field_confidence

    def take(key: str) -> bool:
        return key in fields and bool(fields[key].value) and fc.get(key, 0.0) < _HIGH

    if take("number"):
        invoice.number = fields["number"].value
        fc["number"] = _ASSIST_CONF
    if take("date"):
        invoice.date = fields["date"].value
        fc["date"] = _ASSIST_CONF
    if take("currency") and not invoice.currency:
        invoice.currency = fields["currency"].value

    for who, party in (("supplier", invoice.supplier), ("recipient", invoice.recipient)):
        touched = False
        for attr, key in (("name", f"{who}_name"), ("vat_number", f"{who}_vat"), ("eik", f"{who}_eik")):
            if take(key):
                setattr(party, attr, fields[key].value)
                fc[key] = _ASSIST_CONF
                touched = True
        if touched and party.source == "extracted":
            party.source = "assist"
    return invoice


def _build_messages(text: str, doc_type: str) -> list[dict]:
    blocks = [_PROMPT]
    for snippet, fields in _few_shot(doc_type):
        blocks.append(f"\nExample document:\n{snippet}\nExample JSON:\n{fields}")
    blocks.append(f"\nDocument:\n{text[:6000]}\nJSON:")
    return [{"role": "user", "content": "\n".join(blocks)}]


def _complete(messages: list[dict]) -> str:
    import litellm

    s = get_settings()
    model, api_base, api_key = _resolve_model()
    kwargs: dict = {"model": model, "messages": messages, "temperature": 0, "timeout": s.llm_timeout}
    if api_base:
        kwargs["api_base"] = api_base
    if api_key:
        kwargs["api_key"] = api_key
    resp = litellm.completion(**kwargs)
    msg = resp["choices"][0]["message"]
    return msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")


def _to_fields(data: dict) -> dict[str, ExtractedField]:
    out: dict[str, ExtractedField] = {}
    for key in _NONAMOUNT_FIELDS:
        value = data.get(key)
        if value in (None, "", "null"):
            continue
        out[key] = ExtractedField(value=str(value).strip(), confidence=_ASSIST_CONF)
    return out


def _few_shot(doc_type: str) -> list[tuple[str, dict]]:
    """A few labeled (text, non-amount-fields) examples of this type from the local
    dataset, using only already-cached text (no runtime OCR). Empty in production."""
    global _examples_cache
    n = get_settings().llm_assist_examples
    if n <= 0:
        return []
    if _examples_cache is None:
        _examples_cache = _load_examples()
    pool = _examples_cache.get(doc_type) or _examples_cache.get("invoice") or []
    return pool[:n]


def _load_examples() -> dict[str, list[tuple[str, dict]]]:
    out: dict[str, list[tuple[str, dict]]] = {}
    try:
        import hashlib
        from pathlib import Path

        from training.dataset import data_root, load_dataset

        root = data_root()
        cache = root / ".cache"
        if not root.exists():
            return out
        for ex in load_dataset(root):
            digest = hashlib.sha1(Path(ex.path).read_bytes()).hexdigest()
            txt = cache / f"{digest}.txt"
            if not txt.exists():
                continue  # only use pre-cached text, never OCR at runtime
            fields = {
                "number": ex.label.number, "date": ex.label.date, "currency": ex.label.currency,
                "supplier_name": ex.label.supplier.name, "supplier_vat": ex.label.supplier.vat_number,
                "supplier_eik": ex.label.supplier.eik, "recipient_name": ex.label.recipient.name,
                "recipient_vat": ex.label.recipient.vat_number, "recipient_eik": ex.label.recipient.eik,
            }
            out.setdefault(ex.label.doc_type, []).append((txt.read_text(encoding="utf-8")[:1200], fields))
    except Exception:  # pragma: no cover - dataset is optional
        return {}
    return out


def reset() -> None:
    global _examples_cache
    _examples_cache = None
