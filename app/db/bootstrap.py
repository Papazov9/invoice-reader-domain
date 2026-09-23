"""Idempotent database bootstrap, run at startup as the admin role.

Creates the pgvector extension, the app role, the tables, grants, and Row-Level
Security (with FORCE) on the tenant-scoped table so the non-superuser app role is
isolated per tenant. Safe to re-run.
"""

from __future__ import annotations

import logging

from sqlalchemy import text

from app.core import get_settings
from app.db.base import Base, admin_engine
from app.db import models  # noqa: F401  (register tables on Base.metadata)

logger = logging.getLogger(__name__)


def init_db() -> None:
    settings = get_settings()
    role = settings.app_db_role
    pwd = settings.app_db_password

    with admin_engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        # App login role (NOSUPERUSER => subject to RLS).
        conn.execute(
            text(
                f"""
                DO $$ BEGIN
                  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '{role}') THEN
                    CREATE ROLE {role} LOGIN PASSWORD '{pwd}' NOSUPERUSER NOBYPASSRLS;
                  END IF;
                END $$;
                """
            )
        )

    # Create tables (as admin / table owner).
    Base.metadata.create_all(admin_engine)

    with admin_engine.begin() as conn:
        # Privileges for the app role.
        conn.execute(text(f"GRANT USAGE ON SCHEMA public TO {role}"))
        conn.execute(
            text(f"GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO {role}")
        )
        conn.execute(text(f"GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO {role}"))
        conn.execute(
            text(
                f"ALTER DEFAULT PRIVILEGES IN SCHEMA public "
                f"GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO {role}"
            )
        )

        # Row-Level Security: tenant isolation on the tenant-scoped tables.
        for table in ("stored_invoices", "document_files"):
            conn.execute(text(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY"))
            conn.execute(text(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY"))
            conn.execute(text(f"DROP POLICY IF EXISTS tenant_isolation ON {table}"))
            conn.execute(
                text(
                    f"""
                    CREATE POLICY tenant_isolation ON {table}
                    USING (tenant_id = current_setting('app.current_tenant', true)::uuid)
                    WITH CHECK (tenant_id = current_setting('app.current_tenant', true)::uuid)
                    """
                )
            )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_docfiles_tenant_external "
                "ON document_files (tenant_id, external_id)"
            )
        )

        # Migrate the embedding column to the configured dimension if needed.
        # pgvector stores the dimension in atttypmod. Resizing clears existing
        # vectors (USING NULL); they are recomputed by rebuild_invoice_index.
        target_dim = get_settings().embedding_dim
        current_dim = conn.execute(
            text(
                "SELECT a.atttypmod FROM pg_attribute a "
                "JOIN pg_class c ON a.attrelid = c.oid "
                "WHERE c.relname = 'stored_invoices' AND a.attname = 'embedding'"
            )
        ).scalar()
        if current_dim is not None and current_dim != target_dim:
            conn.execute(text("DROP INDEX IF EXISTS idx_invoices_embedding"))
            conn.execute(
                text(
                    f"ALTER TABLE stored_invoices "
                    f"ALTER COLUMN embedding TYPE vector({target_dim}) USING NULL"
                )
            )
            logger.info("Resized stored_invoices.embedding %s -> %s", current_dim, target_dim)

        # Indexes: per-company lookups + vector similarity (cosine).
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_invoices_tenant_company "
                "ON stored_invoices (tenant_id, company_key)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_invoices_embedding "
                "ON stored_invoices USING hnsw (embedding vector_cosine_ops)"
            )
        )

    logger.info("Database bootstrap complete (pgvector, tables, RLS, role=%s)", role)
