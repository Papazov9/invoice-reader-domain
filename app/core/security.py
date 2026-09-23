"""Password hashing (bcrypt) and JWT access tokens."""

from __future__ import annotations

import datetime as dt
import uuid

import bcrypt
import jwt

from app.core import get_settings


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


def create_access_token(user_id: uuid.UUID, tenant_id: uuid.UUID) -> str:
    s = get_settings()
    now = dt.datetime.now(tz=dt.timezone.utc)
    payload = {
        "sub": str(user_id),
        "tenant": str(tenant_id),
        "iat": now,
        "exp": now + dt.timedelta(minutes=s.jwt_expire_minutes),
    }
    return jwt.encode(payload, s.jwt_secret, algorithm=s.jwt_algorithm)


def decode_access_token(token: str) -> dict:
    s = get_settings()
    return jwt.decode(token, s.jwt_secret, algorithms=[s.jwt_algorithm])
