"""Schema-agnostic XML to DocCandidate extraction.

Auto-detects record-like elements, cleans namespaces, picks a stable id, and falls
back to regex for malformed XML. Parsing goes through defusedxml so external entities
and billion-laughs payloads can't hit untrusted accounting files.
"""

from __future__ import annotations

import re
from pathlib import Path

from defusedxml.ElementTree import fromstring as safe_fromstring

from app.domain import DocCandidate, TextUnit

# Element names that typically wrap one record/document.
_RECORD_TAGS = {
    "row", "record", "document", "item", "entry",
    "invoice", "invoicerecord", "transaction",
}
_CONTAINER_TAGS = {"data", "documents", "records", "items", "invoices", "transactions"}
# Field names to prefer when deriving a stable document id (most specific first).
_ID_PRIORITY = [
    "documentnumber", "invoicenumber", "docnumber", "docno", "invoice_number", "s1",
]
_XML_DECL = re.compile(r"<\?xml[^>]*\?>")


def _local(tag: str) -> str:
    """Strip namespace (Clark ``{ns}local`` or prefix ``ns:local``) and lowercase."""
    if "}" in tag:
        tag = tag.split("}")[-1]
    if ":" in tag:
        tag = tag.split(":")[-1]
    return tag.lower()


def parse_xml(xml_content: str, label: str = "") -> list[DocCandidate]:
    """Extract one DocCandidate per record-like element in *any* XML document."""
    content = _XML_DECL.sub("", xml_content).strip()
    try:
        root = safe_fromstring(content)
    except Exception:
        return _regex_fallback(content, label)

    records = _find_records(root) or [root]
    docs = [_extract_record(rec, idx, label) for idx, rec in enumerate(records)]
    return [d for d in docs if d is not None]


def parse_xml_file(path: str, label: str = "") -> list[DocCandidate]:
    p = Path(path)
    return parse_xml(p.read_text(encoding="utf-8"), label or p.stem)


def _find_records(root) -> list:
    records = [el for el in root.iter() if _local(el.tag) in _RECORD_TAGS]
    if records:
        return records

    # repeated direct children are the records
    counts: dict[str, int] = {}
    for child in root:
        counts[_local(child.tag)] = counts.get(_local(child.tag), 0) + 1
    for tag, count in counts.items():
        if count > 1:
            return [c for c in root if _local(c.tag) == tag]

    # Children of a known container element.
    for el in root.iter():
        if _local(el.tag) in _CONTAINER_TAGS:
            return list(el)
    return []


def _extract_record(elem, idx: int, label: str) -> DocCandidate | None:
    units: list[TextUnit] = []
    field_values: dict[str, str] = {}

    def add_attrs(node) -> None:
        for name, value in node.attrib.items():
            value = (value or "").strip()
            if value:
                fname = _local(name)
                units.append(TextUnit(kind="fieldName", text=fname))
                units.append(TextUnit(kind="value", text=value))
                field_values.setdefault(fname, value)

    add_attrs(elem)
    for child in elem.iter():
        if child is elem:
            continue
        add_attrs(child)
        text = (child.text or "").strip()
        if text:
            tag = _local(child.tag)
            units.append(TextUnit(kind="fieldName", text=tag))
            units.append(TextUnit(kind="value", text=text))
            field_values.setdefault(tag, text)

    if not units:
        return None

    doc_id = next((field_values[f] for f in _ID_PRIORITY if field_values.get(f)), str(idx))
    full_id = f"{label}-{doc_id}" if label else f"DOC-{doc_id}"
    return DocCandidate(id=full_id, units=units)


def _regex_fallback(xml_content: str, label: str) -> list[DocCandidate]:
    """Best-effort attribute extraction when the XML will not parse."""
    docs: list[DocCandidate] = []
    row_pattern = re.compile(r"<[^>]*\brow\b[^>]*?([^/>]*)/?>|<Document\s+([^>]+)>", re.IGNORECASE)
    attr_pattern = re.compile(r"(\w+)\s*=\s*[\"']([^\"']*)[\"']")

    for idx, match in enumerate(row_pattern.finditer(xml_content)):
        attrs_str = match.group(1) or match.group(2) or ""
        units: list[TextUnit] = []
        field_values: dict[str, str] = {}
        for name, value in attr_pattern.findall(attrs_str):
            value = value.strip()
            if value:
                units.append(TextUnit(kind="fieldName", text=name.lower()))
                units.append(TextUnit(kind="value", text=value))
                field_values.setdefault(name.lower(), value)
        if not units:
            continue
        doc_id = next((field_values[f] for f in _ID_PRIORITY if field_values.get(f)), str(idx))
        full_id = f"{label}-{doc_id}" if label else f"DOC-{doc_id}"
        docs.append(DocCandidate(id=full_id, units=units))
    return docs
