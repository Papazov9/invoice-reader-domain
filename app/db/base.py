"""Database engines and session factory.

Two engines on purpose:
- ``admin_engine`` runs DDL/bootstrap as a superuser.
- ``engine`` (the app role) runs request queries and is NOSUPERUSER, so Row-Level
  Security is enforced for it.
"""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.core import get_settings

_settings = get_settings()

engine = create_engine(_settings.database_url, pool_pre_ping=True, future=True)
admin_engine = create_engine(_settings.database_admin_url, pool_pre_ping=True, future=True)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


class Base(DeclarativeBase):
    pass
