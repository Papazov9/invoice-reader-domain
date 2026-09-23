"""Persistent, tenant-scoped workspace: store invoices, browse per-company sets,
and run vector search. All routes require auth and run under RLS."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app.api.deps import Principal, get_principal, get_tenant_db
from app.domain import Invoice
from app.api.schemas import (
    GroupResponse,
    PersistRequest,
    PersistResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
    WorkspaceInvoicesResponse,
)
from app.services import workspace as ws

router = APIRouter(prefix="/workspace", tags=["workspace"])


@router.post("/invoices", response_model=PersistResponse)
def persist(
    req: PersistRequest,
    principal: Principal = Depends(get_principal),
    db: Session = Depends(get_tenant_db),
) -> PersistResponse:
    """Persist parsed invoices for the current tenant (embeds + upserts)."""
    n = ws.store_invoices(db, principal.tenant_id, req.invoices)
    return PersistResponse(stored=n)


@router.get("/invoices", response_model=WorkspaceInvoicesResponse)
def list_invoices(db: Session = Depends(get_tenant_db)) -> WorkspaceInvoicesResponse:
    return WorkspaceInvoicesResponse(invoices=ws.list_invoices(db))


@router.get("/invoices/by-id/{invoice_id}", response_model=Invoice)
def get_invoice(invoice_id: uuid.UUID, db: Session = Depends(get_tenant_db)) -> Invoice:
    """Resolve a citation's stored-invoice id to the full invoice (for drill-down)."""
    inv = ws.get_invoice_by_id(db, invoice_id)
    if inv is None:
        raise HTTPException(status_code=404, detail="invoice not found")
    return inv


@router.get("/companies", response_model=GroupResponse)
def companies(db: Session = Depends(get_tenant_db)) -> GroupResponse:
    return GroupResponse(groups=ws.group_companies(db))


@router.post("/search", response_model=SearchResponse)
def search(req: SearchRequest, db: Session = Depends(get_tenant_db)) -> SearchResponse:
    hits = ws.search_invoices(db, req.query, req.top_k, req.company_key)
    return SearchResponse(hits=[SearchHit(invoice=inv, score=score) for inv, score in hits])


@router.delete("/invoices/{external_id}")
def delete_invoice(external_id: str, db: Session = Depends(get_tenant_db)) -> dict:
    return {"deleted": ws.delete_invoice(db, external_id)}


_MAX_FILE_BYTES = 15_000_000


@router.post("/documents/{external_id}/file")
async def upload_document_file(
    external_id: str,
    file: UploadFile = File(...),
    principal: Principal = Depends(get_principal),
    db: Session = Depends(get_tenant_db),
) -> dict:
    """Store the original uploaded file for a document, so it can be previewed later."""
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty file")
    if len(data) > _MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="file too large")
    fid = ws.store_document_file(
        db, principal.tenant_id, external_id, file.filename, file.content_type, data
    )
    return {"id": str(fid), "size": len(data)}


@router.get("/documents/files")
def list_document_files(db: Session = Depends(get_tenant_db)) -> dict:
    """external_ids of documents whose original file is stored, so the UI can show an
    'open original' affordance even after a refresh (object URLs don't survive that)."""
    return {"external_ids": ws.list_document_file_ids(db)}


@router.get("/documents/{external_id}/file")
def get_document_file(external_id: str, db: Session = Depends(get_tenant_db)) -> Response:
    """Return the original file for a document (tenant-scoped by RLS)."""
    row = ws.get_document_file(db, external_id)
    if row is None:
        raise HTTPException(status_code=404, detail="file not found")
    return Response(
        content=row.data,
        media_type=row.content_type or "application/pdf",
        headers={"Content-Disposition": f'inline; filename="{row.filename or external_id}"'},
    )


@router.post("/documents/{external_id}/label")
def save_label(
    external_id: str,
    invoice: Invoice,
    principal: Principal = Depends(get_principal),
    db: Session = Depends(get_tenant_db),
) -> dict:
    """Save a user-corrected document as a labeled training example (opt-in). Writes the
    stored original + a label.json into the local gitignored dataset for later
    train/evaluate. Amounts are recorded as labels only — never model-derived."""
    import json
    import re
    from pathlib import Path

    from app.core import get_settings

    settings = get_settings()
    if not settings.training_capture_enabled:
        raise HTTPException(status_code=503, detail="training capture disabled")
    row = ws.get_document_file(db, external_id)
    if row is None:
        raise HTTPException(status_code=404, detail="no stored original to label")

    data_dir = Path(settings.preprocessing_data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^0-9A-Za-zА-Яа-я_.-]+", "_", external_id) or "doc"
    ext = ".pdf" if "pdf" in (row.content_type or "") else (Path(row.filename or "").suffix or ".bin")
    (data_dir / f"{stem}{ext}").write_bytes(row.data)

    def party(p) -> dict:
        return {"name": p.name, "vat_number": p.vat_number, "eik": p.eik}

    def amt(v) -> str | None:
        return None if v is None else str(v)

    label = {
        "doc_type": invoice.doc_type, "number": invoice.number, "date": invoice.date,
        "currency": invoice.currency, "supplier": party(invoice.supplier),
        "recipient": party(invoice.recipient), "net_amount": amt(invoice.net_amount),
        "vat_amount": amt(invoice.vat_amount), "total_amount": amt(invoice.total_amount),
        "extra": invoice.extra,
    }
    (data_dir / f"{stem}.label.json").write_text(
        json.dumps(label, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"saved": stem}
