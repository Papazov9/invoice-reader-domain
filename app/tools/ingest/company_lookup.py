"""Commercial-register lookup by EIK.

Given a valid EIK this fetches the public company record from web.company.guru and
returns a typed CompanyInfo, used to recover or confirm a counterparty whose name was
mangled by OCR. Network and HTML parsing are best-effort: any failure returns None so
ingestion never breaks on a third-party hiccup. Results are cached per process.

Enabling this sends EIKs to a third party; it is gated by company_lookup_enabled.
The dependencies are optional and import-guarded.
"""

from __future__ import annotations

import re
from collections import OrderedDict

from app.core import get_settings
from app.domain import CompanyInfo

from .eik import validate_eik

try:  # optional deps, shared with the laws RAG
    import requests
    from bs4 import BeautifulSoup

    _LOOKUP_IMPORTED = True
except Exception:  # pragma: no cover - depends on environment
    _LOOKUP_IMPORTED = False


_LEGAL_FORMS = {
    "Еднолично дружество с ограничена отговорност": "ЕООД",
    "Дружество с ограничена отговорност": "ООД",
    "Акционерно дружество": "АД",
    "Едноличен търговец": "ЕТ",
    "Кооперация": "Кооп.",
}

_cache: "OrderedDict[str, CompanyInfo | None]" = OrderedDict()


def lookup_status() -> dict[str, object]:
    """Report whether register lookup is enabled and usable."""
    settings = get_settings()
    if not settings.company_lookup_enabled:
        return {"enabled": False, "available": False, "reason": "disabled"}
    if not _LOOKUP_IMPORTED:
        return {"enabled": True, "available": False, "reason": "requests/bs4 not installed"}
    return {"enabled": True, "available": True, "base_url": settings.company_lookup_base_url}


def lookup_company(eik: str | None) -> CompanyInfo | None:
    """Look up a company by EIK, or None if invalid/disabled/unavailable/not found."""
    if not validate_eik(eik):
        return None  # reject garbles before any network call
    eik = eik.strip()
    settings = get_settings()
    if not settings.company_lookup_enabled or not _LOOKUP_IMPORTED:
        return None
    if eik in _cache:
        _cache.move_to_end(eik)
        return _cache[eik]
    info = _scrape(eik, settings)
    _cache[eik] = info
    _cache.move_to_end(eik)
    while len(_cache) > max(1, settings.company_lookup_cache_size):
        _cache.popitem(last=False)
    return info


def _shorten_legal_form(full: str | None) -> str:
    return _LEGAL_FORMS.get((full or "").strip(), (full or "").strip())


def _field(soup, label: str) -> str | None:
    div = soup.find("div", string=lambda s: s and s.strip() == label)
    if div:
        value = div.find_next_sibling("div", class_="value")
        return value.get_text(strip=True) if value else None
    return None


def _top_field(soup, label: str) -> str | None:
    for div in soup.select(".col-md-4 .text-right"):
        if label in div.text:
            return div.text.split(":", 1)[-1].strip()
    return None


def _normalize_vat(raw: str | None, eik: str) -> str | None:
    """The register shows the VAT number, or a "Да"/"Не" registered flag. Turn a flag
    into the standard BG+EIK number for a 9-digit company."""
    if not raw:
        return None
    v = raw.strip().upper().replace(" ", "")
    if v.startswith("BG") and v[2:].isdigit():
        return v
    if v.isdigit() and len(v) in (9, 10):
        return f"BG{v}"
    if raw.strip().lower() in ("да", "yes") and len(eik) == 9:
        return f"BG{eik}"
    return None


def _city(address: str) -> str | None:
    m = re.search(r"(гр|с)\.\s?([\wА-Яа-я\- ]+)", address)
    return f"{m.group(1)}. {m.group(2).strip()}" if m else None


def _split_address(address: str | None) -> dict[str, str | None]:
    if not address:
        return {"country": None, "city": None, "zip_code": None, "address_line1": None}
    parts = address.split(", ")
    country = parts[0] if parts else None
    zip_match = re.search(r"\b(\d{4})\b", address)
    zip_code = zip_match.group(1) if zip_match else None
    city = _city(address)
    known = [p for p in (country, city, zip_code) if p]
    line = ", ".join(p for p in parts if all(k not in p for k in known))
    return {"country": country, "city": city, "zip_code": zip_code, "address_line1": line}


def _scrape(eik: str, settings) -> CompanyInfo | None:
    url = f"{settings.company_lookup_base_url.rstrip('/')}/{eik}"
    try:
        res = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=settings.company_lookup_timeout,
        )
        if res.status_code != 200:
            return None
        soup = BeautifulSoup(res.text, "html.parser")
        heading = soup.find("h1")
        name = heading.text.strip() if heading else None
        legal_form = _shorten_legal_form(_field(soup, "Правна форма"))
        address = _field(soup, "Адрес")
        parts = _split_address(address)
        return CompanyInfo(
            eik=eik,
            name=f"{name} {legal_form}".strip() if name else None,
            vat_number=_normalize_vat(_top_field(soup, "ДДС"), eik),
            status=_field(soup, "Актуален статус на лицето"),
            country=parts["country"],
            city=parts["city"],
            zip_code=parts["zip_code"],
            address_line1=parts["address_line1"],
            manager=_field(soup, "Управители"),
        )
    except Exception:  # pragma: no cover - third-party reliability
        return None
