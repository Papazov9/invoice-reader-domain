"""Request dependencies: authenticated principal and tenant-scoped DB sessions."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Iterator

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.security import decode_access_token
from app.db.base import SessionLocal

_bearer = HTTPBearer(auto_error=True)


@dataclass
class Principal:
    user_id: uuid.UUID
    tenant_id: uuid.UUID


def get_principal(
    creds: HTTPAuthorizationCredentials = Depends(_bearer),
) -> Principal:
    try:
        payload = decode_access_token(creds.credentials)
        return Principal(
            user_id=uuid.UUID(payload["sub"]),
            tenant_id=uuid.UUID(payload["tenant"]),
        )
    except Exception as exc:  # invalid/expired token
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid or expired token"
        ) from exc


def get_db() -> Iterator[Session]:
    """Plain session with no tenant context, for auth endpoints only."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_tenant_db(principal: Principal = Depends(get_principal)) -> Iterator[Session]:
    """Tenant-scoped session: sets app.current_tenant so RLS isolates this tenant."""
    db = SessionLocal()
    try:
        # set_config(..., is_local=true) is transaction-scoped and accepts bind params
        # (plain SET LOCAL does not). The first statement opens the transaction.
        db.execute(
            text("SELECT set_config('app.current_tenant', :tid, true)"),
            {"tid": str(principal.tenant_id)},
        )
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
