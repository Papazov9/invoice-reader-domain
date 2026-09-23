"""Authentication: register a firm + owner, log in, and identify the current user."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import Principal, get_db, get_principal
from app.api.schemas import LoginRequest, RegisterRequest, TokenResponse, UserOut
from app.core.security import create_access_token, hash_password, verify_password
from app.db.models import Tenant, User

router = APIRouter(prefix="/auth", tags=["auth"])


def _user_out(user: User, tenant: Tenant) -> UserOut:
    return UserOut(
        id=str(user.id),
        email=user.email,
        role=user.role,
        tenant_id=str(tenant.id),
        tenant_name=tenant.name,
    )


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(req: RegisterRequest, db: Session = Depends(get_db)) -> TokenResponse:
    email = req.email.strip().lower()
    if db.execute(select(User).where(User.email == email)).scalar_one_or_none():
        raise HTTPException(status_code=409, detail="email already registered")

    tenant = Tenant(name=req.tenant_name.strip())
    db.add(tenant)
    db.flush()  # assign tenant.id
    user = User(
        tenant_id=tenant.id, email=email, password_hash=hash_password(req.password), role="owner"
    )
    db.add(user)
    db.flush()

    token = create_access_token(user.id, tenant.id)
    return TokenResponse(access_token=token, user=_user_out(user, tenant))


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    email = req.email.strip().lower()
    user = db.execute(select(User).where(User.email == email)).scalar_one_or_none()
    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail="invalid email or password")
    tenant = db.get(Tenant, user.tenant_id)
    token = create_access_token(user.id, user.tenant_id)
    return TokenResponse(access_token=token, user=_user_out(user, tenant))


@router.get("/me", response_model=UserOut)
def me(principal: Principal = Depends(get_principal), db: Session = Depends(get_db)) -> UserOut:
    user = db.get(User, principal.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="user not found")
    return _user_out(user, db.get(Tenant, user.tenant_id))
