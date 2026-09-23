"""Persistence service: store/list/group/search invoices for the current tenant.

All queries run on a tenant-scoped session (app.current_tenant is set by the request
dependency), so Row-Level Security guarantees isolation; these functions never filter
by tenant_id themselves on reads.
"""

from __future__ import annotations

import uuid

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.domain import CompanyGroup, Invoice
from app.db.models import DocumentFile, StoredInvoice
from invoice_rag.indexing.dense import embed_invoice, embed_text
from app.tools.ingest import group_by_company


def _to_row(tenant_id: uuid.UUID, inv: Invoice) -> StoredInvoice:
    return StoredInvoice(
        tenant_id=tenant_id,
        external_id=inv.id,
        company_key=inv.company_key,
        company_name=inv.company_name,
        number=inv.number,
        date=inv.date,
        currency=inv.currency,
        supplier_name=inv.supplier.name,
        supplier_vat=inv.supplier.vat_number,
        supplier_eik=inv.supplier.eik,
        recipient_name=inv.recipient.name,
        recipient_vat=inv.recipient.vat_number,
        net_amount=inv.net_amount,
        vat_amount=inv.vat_amount,
        total_amount=inv.total_amount,
        source=inv.source,
        payload=inv.model_dump(mode="json"),
        embedding=embed_invoice(inv),
    )


def store_invoices(db: Session, tenant_id: uuid.UUID, invoices: list[Invoice]) -> int:
    """Upsert invoices by external_id (delete-then-insert within the tenant)."""
    ext_ids = [i.id for i in invoices]
    if ext_ids:
        db.execute(delete(StoredInvoice).where(StoredInvoice.external_id.in_(ext_ids)))
    for inv in invoices:
        db.add(_to_row(tenant_id, inv))
    db.flush()
    try:
        from invoice_rag.indexing.pipeline import build_bm25_for_tenant
        build_bm25_for_tenant(db, tenant_id)
    except Exception:  # indexing is best-effort; dense search still works
        pass
    return len(invoices)


def list_invoices(db: Session) -> list[Invoice]:
    rows = db.execute(select(StoredInvoice).order_by(StoredInvoice.created_at.desc())).scalars().all()
    return [Invoice.model_validate(r.payload) for r in rows]


def get_invoice_by_id(db: Session, invoice_id: uuid.UUID) -> Invoice | None:
    """Resolve a stored_invoices primary key (the id used by RAG citations) to the
    full domain Invoice. RLS scopes the lookup to the current tenant."""
    row = db.get(StoredInvoice, invoice_id)
    return Invoice.model_validate(row.payload) if row else None


def group_companies(db: Session) -> list[CompanyGroup]:
    return group_by_company(list_invoices(db))


def search_invoices(
    db: Session, query: str, top_k: int = 10, company_key: str | None = None
) -> list[tuple[Invoice, float]]:
    qvec = embed_text(query)
    distance = StoredInvoice.embedding.cosine_distance(qvec).label("distance")
    stmt = select(StoredInvoice, distance)
    if company_key:
        stmt = stmt.where(StoredInvoice.company_key == company_key)
    stmt = stmt.order_by(distance).limit(top_k)

    hits: list[tuple[Invoice, float]] = []
    for row, dist in db.execute(stmt).all():
        score = max(0.0, 1.0 - float(dist))  # cosine similarity
        hits.append((Invoice.model_validate(row.payload), round(score, 4)))
    return hits


def delete_invoice(db: Session, external_id: str) -> int:
    result = db.execute(delete(StoredInvoice).where(StoredInvoice.external_id == external_id))
    return result.rowcount or 0


def store_document_file(
    db: Session,
    tenant_id: uuid.UUID,
    external_id: str,
    filename: str | None,
    content_type: str | None,
    data: bytes,
) -> uuid.UUID:
    """Upsert the original file for a document (keyed by external_id, within the tenant)."""
    db.execute(delete(DocumentFile).where(DocumentFile.external_id == external_id))
    row = DocumentFile(
        tenant_id=tenant_id,
        external_id=external_id,
        filename=filename,
        content_type=content_type or "application/pdf",
        size=len(data),
        data=data,
    )
    db.add(row)
    db.flush()
    return row.id


def get_document_file(db: Session, external_id: str) -> DocumentFile | None:
    """Most-recent stored file for a document. RLS scopes the lookup to the tenant."""
    return (
        db.execute(
            select(DocumentFile)
            .where(DocumentFile.external_id == external_id)
            .order_by(DocumentFile.created_at.desc())
        )
        .scalars()
        .first()
    )


def list_document_file_ids(db: Session) -> list[str]:
    """external_ids of documents that have a stored original (RLS-scoped to the tenant),
    so the UI can show an 'original available' affordance after a refresh."""
    rows = db.execute(select(DocumentFile.external_id).distinct()).scalars().all()
    return [r for r in rows if r]
