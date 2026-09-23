"""LLM vision fallback for poorly scanned documents.

When OCR confidence is low, the page images are sent to a vision-capable model which
reads the fields directly and returns strict JSON. The result fills only the fields the
regex pass was unsure about, so a clean text reading is never overwritten. Self-disables
when no vision model is configured; any failure returns None so OCR remains the result.
"""

from __future__ import annotations

import base64
import json
import re
from decimal import Decimal, InvalidOperation
from typing import Callable

from app.core import get_settings
from app.domain import ExtractedField, Invoice, LineItem

_HIGH = 0.9
_VISION_CONF = 0.85  # below _HIGH so the register can still canonicalise a legal name
_MAX_PAGES = 2  # a vision model's field accuracy drops sharply with many page images;
#                 Bulgarian invoices are almost always 1-2 pages (extra pages are copies)
# Document types that always carry a tax base + item lines; an incomplete rule read of one
# is worth a vision look. Values match DocumentType (a str-enum), compared as plain strings.
_ITEM_DOC_TYPES = frozenset({
    "invoice", "credit_note", "debit_note", "proforma", "simplified_invoice", "goods_receipt",
})

_PROMPT = (
    "You read Bulgarian tax invoices (фактура). Return ONLY a JSON object, no prose, no "
    "code fences. Keep all names/text in the ORIGINAL Cyrillic exactly as printed — never "
    "translate or transliterate.\n"
    "LAYOUT of a Bulgarian invoice — read it correctly:\n"
    "- Two parties at the top: 'Получател'/'Купувач' = recipient (buyer); "
    "'Изпълнител'/'Доставчик'/'Продавач' = supplier (seller). Do NOT swap them.\n"
    "- 'Място на издаване' is a branch/address, NOT a party. Never use it as a name.\n"
    "- Each party has a name, 'ИН по ЗДДС' (VAT, starts with BG) and 'ЕИК' (digits only). "
    "ЕИК and ИН по ЗДДС are DIFFERENT fields — do not put one in the other.\n"
    "- 'Плащане' section: 'Банка', 'IBAN', 'BIC' are NOT parties and NOT EIK/VAT numbers.\n"
    "- Item table ('Асортименти'/'Стоки') columns: № | Номер/Партида (code) | Наименование "
    "(description) | Мярка (unit, e.g. 'бр.', text not a number) | Количество (quantity, "
    "usually a whole number) | Ед. цена (unit price) | Стойност (line amount). "
    "quantity × unit_price = amount; the amounts sum to net_amount.\n"
    "JSON keys (use null when absent): number, date (ISO YYYY-MM-DD), currency, "
    "supplier_name, supplier_vat, supplier_eik, recipient_name, recipient_vat, "
    "recipient_eik, net_amount, vat_amount, total_amount, "
    "due_date (Дата на падеж, ISO YYYY-MM-DD), payment_method_raw (the exact text of the "
    "'Плащане'/'Начин на плащане' field, e.g. 'Платежно нареждане', 'в брой'), "
    "items (array of {code, description, unit, quantity, unit_price, amount}). "
    "Copy digits exactly, keep cents (e.g. 464.02). "
    "Verify quantity × unit_price = amount for every row before answering."
)

_STRING_FIELDS = (
    "number", "date",
    "supplier_name", "supplier_vat", "supplier_eik",
    "recipient_name", "recipient_vat", "recipient_eik",
    "currency", "due_date", "payment_method_raw",
)
_AMOUNT_FIELDS = ("net_amount", "vat_amount", "total_amount")
_ITEMS_KEY = "__items__"  # reserved fields key carrying the JSON-encoded item rows


def _to_decimal(raw: object) -> Decimal | None:
    """Parse a model-supplied money/quantity string (EU or US formatting) to Decimal."""
    if raw in (None, "", "null"):
        return None
    s = re.sub(r"[^\d,.\-]", "", str(raw))
    if not s or not any(c.isdigit() for c in s):
        return None
    if "," in s and "." in s:  # last separator is the decimal point
        s = s.replace(",", "") if s.rfind(".") > s.rfind(",") else s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return Decimal(s)
    except (InvalidOperation, ValueError):
        return None


def _reconcile_row(qty: Decimal | None, price: Decimal | None, amount: Decimal | None):
    """Make a single item row internally consistent (quantity × unit_price = amount).

    The line amount is the most reliable cell (it also feeds the invoice net), so when a
    vision misread makes qty×price disagree with it, we recompute the odd cell out. This is
    what turns a misread 'quantity 6' into 4 when 4 × 96.67 = 386.68 = amount."""
    tol = Decimal("0.02")
    if price and amount is not None and price != 0:
        calc = amount / price
        rounded = calc.quantize(Decimal("1"))
        if abs(calc - rounded) <= tol and (qty is None or qty != rounded):
            qty = rounded  # trust price+amount over a misread quantity
    if qty and amount is not None and (price is None or price == 0):
        price = (amount / qty)
    if qty and price and amount is None:
        amount = qty * price
    return qty, price, amount


def _reconcile_totals(invoice: Invoice) -> None:
    """Fill the one missing figure among net/vat/total from the other two (net+vat=total),
    and correct a cents-level mismatch (a common OCR misread of the total) in favour of the
    internally consistent net+vat sum."""
    n, v, t = invoice.net_amount, invoice.vat_amount, invoice.total_amount
    if n is not None and v is not None and t is None:
        invoice.total_amount = n + v
    elif n is not None and t is not None and v is None:
        invoice.vat_amount = t - n
    elif v is not None and t is not None and n is None:
        invoice.net_amount = t - v
    elif n is not None and v is not None and t is not None and n + v != t:
        summed = n + v
        if abs(summed - t) <= abs(summed) * Decimal("0.01"):
            invoice.total_amount = summed  # cents-level OCR rounding; safe to normalise
            return
        # A MATERIAL gap between net+vat and the printed total. DIRECTION decides safety: a
        # total BELOW net+vat is impossible on a real invoice, so it's a misread/truncation
        # (e.g. 464.02 read as 64) — prefer the sum. A total ABOVE net+vat is legitimate — a
        # genuine total can carry non-taxable components (deposits, telecom carry-over balances,
        # levies), so overwriting it with net+vat is a real regression (BG131391369: true
        # 6331.20 must not become 5658.28 just because VAT is 20% of the base). Only correct the
        # short case, and only when net+vat is structurally trustworthy (VAT a standard 0/9/20%
        # fraction of net, or the item amounts sum to net). Compare magnitudes so credit notes
        # (net/vat/total all negative) reconcile the same way.
        if abs(t) >= abs(summed):
            return
        rate_ok = n != 0 and any(
            abs(v / n - r) < Decimal("0.015") for r in (Decimal("0"), Decimal("0.09"), Decimal("0.20"))
        )
        items_sum = sum((li.amount for li in invoice.line_items if li.amount is not None), Decimal(0))
        items_ok = items_sum > 0 and abs(items_sum - n) <= abs(n) * Decimal("0.01")
        if rate_ok or items_ok:
            invoice.total_amount = summed


def _vision_line_items(raw_json: str) -> list[LineItem]:
    """Decode + reconcile the vision item rows into domain LineItems."""
    try:
        rows = json.loads(raw_json)
    except (json.JSONDecodeError, TypeError):
        return []
    items: list[LineItem] = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        qty, price, amount = _reconcile_row(
            _to_decimal(row.get("quantity")),
            _to_decimal(row.get("unit_price")),
            _to_decimal(row.get("amount")),
        )
        desc = (row.get("description") or "").strip() or None
        code = (row.get("code") or "").strip() or None
        unit = (row.get("unit") or "").strip() or None
        if not (desc or amount):  # a truly empty row
            continue
        items.append(LineItem(
            description=desc, quantity=qty, unit_price=price, amount=amount, code=code, unit=unit,
        ))
    return items


def vision_model() -> str:
    s = get_settings()
    return s.ocr_vision_model or s.llm_model


def should_use_vision(invoice: Invoice, mean_conf: float) -> bool:
    """Decide whether the vision model is worth consulting for this page."""
    s = get_settings()
    if not s.ocr_vision_fallback or not vision_model():
        return False
    # A VAT invoice always states a tax base and item lines whose amounts sum to it. If the
    # rule pass got no tax base, no items, or items that don't reconcile to the tax base, the
    # reading is incomplete or wrong (typical of a photo, or a table the column heuristics
    # mis-sliced) — worth the vision look regardless of the token confidences.
    if invoice.doc_type in _ITEM_DOC_TYPES and (
        invoice.net_amount is None or not invoice.line_items or not _items_reconcile(invoice)
        or _supplier_unidentified(invoice)
    ):
        return True
    return mean_conf < s.ocr_vision_conf_min or _key_fields_weak(invoice) or _parties_collide(invoice)


def _supplier_unidentified(invoice: Invoice) -> bool:
    """The supplier carries neither EIK nor VAT — so the vendor can't be identified. Common
    when the issuer's identity sits only in a letterhead logo (utility/telecom bills), which
    the text layer misses but the rendered page shows. A purchase invoice must name a vendor."""
    s = invoice.supplier
    return not s.eik and not s.vat_number


def _items_reconcile(invoice: Invoice) -> bool:
    """Whether the item amounts sum to the tax base (±2%). True (don't force vision) when
    there's nothing to check — no items or no net."""
    if not invoice.line_items or invoice.net_amount is None or invoice.net_amount == 0:
        return True
    s = sum((li.amount for li in invoice.line_items if li.amount is not None), Decimal(0))
    return abs(s - invoice.net_amount) <= abs(invoice.net_amount) * Decimal("0.02")


def _key_fields_weak(invoice: Invoice) -> bool:
    fc = invoice.field_confidence
    weak = sum(
        fc.get(k, 0.0) < _HIGH
        for k in ("number", "total_amount", "supplier_name", "recipient_name")
    )
    return weak >= 2


def _parties_collide(invoice: Invoice) -> bool:
    """Supplier == recipient is impossible on a real document — a sure sign the parties
    were mis-assigned (e.g. a two-column header read into one block). Worth a vision look."""
    from .company import normalize_company_name

    s, r = invoice.supplier, invoice.recipient
    if s.eik and s.eik == r.eik:
        return True
    sn, rn = normalize_company_name(s.name or ""), normalize_company_name(r.name or "")
    return bool(sn) and sn == rn


def extract_invoice_via_vision(
    page_images: list[bytes],
    doc_id: str = "",
    *,
    complete: Callable[[str, list[bytes]], str] | None = None,
) -> dict[str, ExtractedField] | None:
    """Ask the vision model to read the page images; return field -> ExtractedField."""
    model = vision_model()
    if not model or not page_images:
        return None
    complete = complete or _vision_complete
    try:
        raw = complete(model, page_images[:_MAX_PAGES])
        data = _parse_json(raw)
    except Exception:  # pragma: no cover - model/transport failure
        return None
    return _to_fields(data) if data else None


def _vision_eik(fields: dict[str, ExtractedField], who: str) -> str:
    """A checksum-valid EIK for a party from vision, taken from its ЕИК or its BG VAT."""
    from .invoice_extractor import validate_eik

    for key in (f"{who}_eik", f"{who}_vat"):
        f = fields.get(key)
        cand = re.sub(r"\D", "", f.value) if f and f.value else ""
        if validate_eik(cand):
            return cand
    return ""


def _ocr_swapped_parties(invoice: Invoice, fields: dict[str, ExtractedField]) -> bool:
    """True when vision's EIKs show the rule pass put the two parties in the wrong roles —
    e.g. OCR labelled the buyer as the supplier. Reflowing a two-column header can do this
    with high token confidence, so the collision test misses it; matching EIKs across the
    two readers catches it. When detected, vision's role assignment is authoritative."""
    v_sup, v_rec = _vision_eik(fields, "supplier"), _vision_eik(fields, "recipient")
    if not (v_sup and v_rec and v_sup != v_rec):
        return False
    o_sup = invoice.supplier.eik or re.sub(r"\D", "", invoice.supplier.vat_number or "")
    o_rec = invoice.recipient.eik or re.sub(r"\D", "", invoice.recipient.vat_number or "")
    return (o_sup and o_sup == v_rec) or (o_rec and o_rec == v_sup)


def _party_id(party) -> str:
    """A party's comparable id: its EIK digits, else its VAT digits."""
    return re.sub(r"\D", "", party.eik or "") or re.sub(r"\D", "", party.vat_number or "")


def _acceptable_party_value(attr: str, value: str) -> bool:
    """Whether a vision-supplied party identifier is plausible enough to write. A vision model
    misreads a digit or drops/adds one, yielding an EIK of the wrong length or a bad checksum
    (e.g. 123711888 read as 1237118889). Writing that over a clean deterministic id is a
    regression, so an EIK must be a checksum-valid 9/13-digit company id and a VAT must look
    like a real EU VAT; a wrong id is dropped in favour of leaving the field empty. Names are
    free text and always accepted."""
    from .invoice_extractor import validate_eik

    if attr == "eik":
        return validate_eik(re.sub(r"\D", "", value or ""))
    if attr == "vat_number":
        return bool(re.match(r"^[A-Za-z]{2}\d{8,12}$", (value or "").replace(" ", "")))
    return True


def _vision_roles_swapped(invoice: Invoice, fields: dict[str, ExtractedField]) -> bool:
    """True when vision named both parties correctly but assigned them the wrong roles,
    relative to a RELIABLE deterministic anchor. This happens on utility/telecom bills:
    the only party in the text layer is the customer (under a 'Получател'/'Име' block), the
    issuer sits in a letterhead logo, so the rule pass collapses both roles onto the
    customer; vision then reads the logo issuer too but often labels it the recipient.

    The tell is an ASYMMETRIC id match: exactly one of the deterministic ids lands in the
    OPPOSITE vision slot. A genuine two-column reflow (or an OCR role-swap vision fixes)
    cross-matches on BOTH sides — there vision's assignment is authoritative and left alone.
    Only the one-sided (xor) case is a vision swap to undo."""
    vs, vr = _vision_eik(fields, "supplier"), _vision_eik(fields, "recipient")
    if not (vs and vr and vs != vr):
        return False
    os_, or_ = _party_id(invoice.supplier), _party_id(invoice.recipient)
    sup_cross = bool(os_ and os_ == vr)  # our supplier's id is in vision's recipient slot
    rec_cross = bool(or_ and or_ == vs)  # our recipient's id is in vision's supplier slot
    return sup_cross != rec_cross


def merge_into_invoice(invoice: Invoice, fields: dict[str, ExtractedField]) -> Invoice:
    """Apply vision fields, overriding only what was missing or low-confidence."""
    fc = invoice.field_confidence
    changed = False

    def take(key: str) -> bool:
        return key in fields and bool(fields[key].value) and fc.get(key, 0.0) < _HIGH

    if take("number"):
        invoice.number = fields["number"].value
        fc["number"] = _VISION_CONF
        changed = True
    if take("date"):
        from .invoice_extractor import _valid_iso, normalize_date

        iso = normalize_date(fields["date"].value)
        if _valid_iso(iso):  # the model may return dd.mm.yyyy; normalise + sanity-check
            invoice.date = iso
            fc["date"] = _VISION_CONF
            changed = True

    # When the deterministic parties collided (supplier == recipient) or were swapped into
    # the wrong roles, they're definitely wrong, so let vision re-assign them even over a
    # high-confidence (register-recovered) but mistaken value.
    swap = _vision_roles_swapped(invoice, fields)  # vision named both but reversed the roles
    force = _parties_collide(invoice) or _ocr_swapped_parties(invoice, fields) or swap
    for who, party in (("supplier", invoice.supplier), ("recipient", invoice.recipient)):
        # Read from the opposite vision slot when vision reversed the roles (utility bills).
        vwho = ("recipient" if who == "supplier" else "supplier") if swap else who
        touched = False
        if force and fields.get(f"{vwho}_name") and fields[f"{vwho}_name"].value:
            party.eik = party.vat_number = None  # drop the mis-assigned ids; re-fill below
        for attr, okey, vkey in (
            ("name", f"{who}_name", f"{vwho}_name"),
            ("vat_number", f"{who}_vat", f"{vwho}_vat"),
            ("eik", f"{who}_eik", f"{vwho}_eik"),
        ):
            vf = fields.get(vkey)  # the vision value (from the opposite slot when swapped)
            if (vf and vf.value and (fc.get(okey, 0.0) < _HIGH or force)
                    and _acceptable_party_value(attr, vf.value)):
                setattr(party, attr, vf.value)
                fc[okey] = _VISION_CONF
                touched = True
        if touched:
            changed = True
            if party.source in ("extracted", "merged"):
                party.source = "vision"

    for key in _AMOUNT_FIELDS:
        # Amounts stay deterministic/auditable: vision only FILLS an amount the rules left
        # empty, it never overrides one already read (a model can misread a blurry figure).
        if getattr(invoice, key) is None and fields.get(key) and fields[key].value:
            try:
                setattr(invoice, key, Decimal(fields[key].value))
                fc[key] = _VISION_CONF
                changed = True
            except (InvalidOperation, TypeError):
                pass

    # Currency: the deterministic detector reads the whole text and handles dual BGN/EUR
    # euro-transition invoices, so trust it; only take (and normalise) vision's currency when
    # the rules found none. Prevents a vision "EUR" from clobbering a correctly detected BGN.
    if invoice.currency is None and "currency" in fields and fields["currency"].value:
        from .currency import normalize_currency

        nc = normalize_currency(fields["currency"].value)
        if nc:
            invoice.currency = nc

    # Payment method + due date into extra. Vision runs only when the OCR-text read was
    # suspect, so a value it returns here beats the rule pass's (which may have parsed garbage
    # from mangled photo OCR) — override when vision has a value, else keep the rule value.
    from .invoice_extractor import _payment_method_code, _valid_iso, normalize_date

    if "due_date" in fields and fields["due_date"].value:
        iso = normalize_date(fields["due_date"].value)
        if _valid_iso(iso):
            invoice.extra["due_date"] = iso
            changed = True
    if "payment_method_raw" in fields and fields["payment_method_raw"].value:
        raw = fields["payment_method_raw"].value.strip()
        invoice.extra["payment_method_raw"] = raw
        invoice.extra["payment_method"] = _payment_method_code(raw, False)
        changed = True

    # Line items: the reflowed OCR text has no stable columns (photos), and the column
    # heuristics miss or mis-slice many digital tables — vision reads item rows far more
    # reliably. Adopt the vision rows when the rule pass found none OR its rows don't
    # reconcile to the tax base; a clean, reconciling digital-PDF table is left untouched.
    if _ITEMS_KEY in fields and (not invoice.line_items or not _items_reconcile(invoice)):
        rows = _vision_line_items(fields[_ITEMS_KEY].value)
        if rows:
            invoice.line_items = rows
            changed = True

    # Cross-field arithmetic: recover the one missing total from the other two, and — when
    # the item rows reconcile to a net the rules didn't read — adopt it. Amounts the rules
    # already read stay untouched; this only fills gaps, keeping the figures auditable.
    _reconcile_totals(invoice)
    if invoice.net_amount is None and invoice.line_items:
        s = sum((li.amount for li in invoice.line_items if li.amount is not None), Decimal(0))
        if s > 0:
            invoice.net_amount = s
            _reconcile_totals(invoice)

    if changed:
        invoice.vision_used = True
    return invoice


def _vision_complete(model: str, images: list[bytes]) -> str:
    s = get_settings()
    api_base = s.ocr_vision_api_base or s.llm_api_base
    # Free path: talk to Ollama directly over its native HTTP API (no litellm dependency,
    # stdlib only). Uses format=json so the model returns a strict object.
    if model.startswith("ollama/") or "11434" in (api_base or ""):
        return _ollama_vision_complete(model, images, api_base)
    return _litellm_vision_complete(model, images)


def _ollama_vision_complete(model: str, images: list[bytes], api_base: str) -> str:
    import urllib.request

    s = get_settings()
    base = (api_base or "http://localhost:11434").rstrip("/")
    name = model.split("/", 1)[1] if model.startswith("ollama/") else model
    payload = {
        "model": name,
        "prompt": _PROMPT,
        "images": [base64.b64encode(img).decode("ascii") for img in images],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0, "num_ctx": 4096},
    }
    req = urllib.request.Request(
        f"{base}/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=s.llm_timeout) as resp:
        return json.loads(resp.read()).get("response", "")


def _litellm_vision_complete(model: str, images: list[bytes]) -> str:
    import litellm

    s = get_settings()
    blocks: list[dict] = [{"type": "text", "text": _PROMPT}]
    for img in images:
        b64 = base64.b64encode(img).decode("ascii")
        blocks.append(
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
        )
    kwargs: dict = {
        "model": model,
        "messages": [{"role": "user", "content": blocks}],
        "temperature": 0,
        "timeout": s.llm_timeout,
    }
    api_base = s.ocr_vision_api_base or s.llm_api_base
    if api_base:
        kwargs["api_base"] = api_base
    if s.llm_api_key and not s.ocr_vision_api_base:  # a self-hosted vision base needs no key
        kwargs["api_key"] = s.llm_api_key
    resp = litellm.completion(**kwargs)
    msg = resp["choices"][0]["message"]
    return msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")


def _parse_json(raw: str | None) -> dict:
    if not raw:
        return {}
    text = raw.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.IGNORECASE).strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def _to_fields(data: dict) -> dict[str, ExtractedField]:
    out: dict[str, ExtractedField] = {}
    for key in (*_STRING_FIELDS, *_AMOUNT_FIELDS):
        value = data.get(key)
        if value in (None, "", "null"):
            continue
        out[key] = ExtractedField(value=str(value).strip(), confidence=_VISION_CONF)
    # Line items ride through as a JSON string under a reserved key so the fields contract
    # (dict[str, ExtractedField]) is unchanged; merge_into_invoice decodes + reconciles them.
    items = data.get("items")
    if isinstance(items, list) and items:
        out[_ITEMS_KEY] = ExtractedField(
            value=json.dumps(items, ensure_ascii=False), confidence=_VISION_CONF
        )
    return out
