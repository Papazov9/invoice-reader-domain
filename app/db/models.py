"""SQLAlchemy ORM models for tenants, users and stored invoices."""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKey, Integer, LargeBinary, Numeric, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core import get_settings
from app.db.base import Base

_DIM = get_settings().embedding_dim


def _uuid() -> uuid.UUID:
    return uuid.uuid4()


class Tenant(Base):
    # A firm / account boundary, top of the isolation hierarchy.
    __tablename__ = "tenants"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    users: Mapped[list["User"]] = relationship(back_populates="tenant")


class User(Base):
    # An authenticated user belonging to exactly one tenant.
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id"), nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    role: Mapped[str] = mapped_column(String(40), default="owner")
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    tenant: Mapped[Tenant] = relationship(back_populates="users")


class StoredInvoice(Base):
    # A persisted invoice. Tenant-scoped and protected by RLS; carries an embedding
    # for semantic search (pgvector).
    __tablename__ = "stored_invoices"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id"), nullable=False, index=True)

    external_id: Mapped[str | None] = mapped_column(Text)  # invoice.id from parsing
    company_key: Mapped[str | None] = mapped_column(Text, index=True)
    company_name: Mapped[str | None] = mapped_column(Text)

    number: Mapped[str | None] = mapped_column(Text)
    date: Mapped[str | None] = mapped_column(String(20))
    currency: Mapped[str | None] = mapped_column(String(8))

    supplier_name: Mapped[str | None] = mapped_column(Text)
    supplier_vat: Mapped[str | None] = mapped_column(String(32))
    supplier_eik: Mapped[str | None] = mapped_column(String(32))
    recipient_name: Mapped[str | None] = mapped_column(Text)
    recipient_vat: Mapped[str | None] = mapped_column(String(32))

    net_amount: Mapped[Decimal | None] = mapped_column(Numeric(18, 2))
    vat_amount: Mapped[Decimal | None] = mapped_column(Numeric(18, 2))
    total_amount: Mapped[Decimal | None] = mapped_column(Numeric(18, 2))

    source: Mapped[str | None] = mapped_column(String(20))
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)  # full Invoice JSON
    embedding: Mapped[list[float] | None] = mapped_column(Vector(_DIM))

    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class DocumentFile(Base):
    # The original uploaded file for a document, kept so the user can preview it against
    # the extraction. Tenant-scoped and protected by RLS; keyed by external_id (the
    # domain Invoice.id) so the frontend can address it directly.
    __tablename__ = "document_files"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id"), nullable=False, index=True)
    external_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)

    filename: Mapped[str | None] = mapped_column(Text)
    content_type: Mapped[str] = mapped_column(String(100), default="application/pdf")
    size: Mapped[int] = mapped_column(Integer, default=0)
    data: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)

    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
