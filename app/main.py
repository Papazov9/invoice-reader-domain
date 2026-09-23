"""FastAPI application entry point.

Wires the tool layer (ingest/compare/validate) and the persistent multi-tenant
workspace (auth, per-company storage, vector search) into one API.

Run: uvicorn app.main:app --reload --port 8000
Requires Postgres+pgvector, see infra/docker-compose.yml.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.routes import auth, companies, compare, documents, rag, validate, workspace
from app.api.schemas import HealthResponse
from app.core import get_settings
from app.db import init_db
from app.tools.ingest import ocr_status

settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        init_db()
    except Exception as exc:  # DB is optional for the stateless tool endpoints
        logger.warning("Database bootstrap skipped/failed: %s", exc)

    from app.rag import lex_index

    lex_index.start()
    yield


app = FastAPI(
    title=settings.app_name,
    version=__version__,
    description="Accounting AI Assistant — tools + persistent multi-tenant workspace.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Stateless tools
app.include_router(documents.router)
app.include_router(compare.router)
app.include_router(validate.router)
app.include_router(companies.router)
# Auth + persistent workspace
app.include_router(auth.router)
app.include_router(workspace.router)
# Retrieval (RAG) + chat orchestrator
app.include_router(rag.router)


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__, ocr=ocr_status())
