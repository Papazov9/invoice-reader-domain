"""Lean entrypoint for the invoice-reading microservice (no database, no RAG index).

This is the service the gumialex `parse-expense-invoice` edge function calls. It defines
the two extraction endpoints inline (importing only the DB/RAG-free `app.tools.ingest`),
so it starts fast and needs no Postgres, no sqlalchemy and none of the ~4GB laws index
that `app.main` pulls in through `app.api.schemas`.

Run locally:   uvicorn app.ingest_service:app --host 0.0.0.0 --port 8000
Vision model:  set OCR_VISION_MODEL=ollama/qwen2.5vl:7b and OCR_VISION_API_BASE=http://<ollama>:11434
Auth:          set INGEST_SERVICE_TOKEN and send it as the `x-ingest-token` header.
"""

from __future__ import annotations

import logging
import os

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.domain import Invoice
from app.tools.ingest import extract_from_image_bytes, extract_from_pdf_bytes, ocr_status

logger = logging.getLogger(__name__)


def require_ingest_token(x_ingest_token: str | None = Header(default=None)) -> None:
    """Guard the extraction endpoints with a shared secret. When INGEST_SERVICE_TOKEN is
    unset the service is open (convenient for local dev) — always set it in production, and
    keep the service on a private network besides."""
    expected = os.environ.get("INGEST_SERVICE_TOKEN")
    if not expected:
        return
    if x_ingest_token != expected:
        raise HTTPException(status_code=401, detail="invalid or missing x-ingest-token")


app = FastAPI(title="gumialex invoice reader", version=__version__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # requests come server-to-server from the edge function
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_AUTH = [Depends(require_ingest_token)]


def _ensure_ocr() -> None:
    if not ocr_status().get("available"):
        raise HTTPException(status_code=503, detail="OCR not available on this server")


@app.post("/documents/extract-pdf", dependencies=_AUTH, tags=["extract"])
async def extract_pdf(
    file: UploadFile = File(...),
    perspective: str = Form("auto"),
    vision: bool = Form(True),
) -> dict[str, Invoice]:
    """OCR/read a PDF invoice and return the structured Invoice. `vision=false` skips the
    (slow) vision fallback for bulk use."""
    _ensure_ocr()
    content = await file.read()
    doc_id = (file.filename or "invoice").rsplit(".", 1)[0]
    try:
        invoice = extract_from_pdf_bytes(content, doc_id, source="ocr", perspective=perspective, use_vision=vision)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"OCR failed: {exc}") from exc
    return {"invoice": invoice}


@app.post("/documents/extract-image", dependencies=_AUTH, tags=["extract"])
async def extract_image(
    file: UploadFile = File(...),
    perspective: str = Form("auto"),
    vision: bool = Form(True),
) -> dict[str, Invoice]:
    """OCR/read a photographed or scanned image invoice and return the structured Invoice."""
    _ensure_ocr()
    content = await file.read()
    doc_id = (file.filename or "image").rsplit(".", 1)[0]
    try:
        invoice = extract_from_image_bytes(content, doc_id, source="ocr", perspective=perspective, use_vision=vision)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"image OCR failed: {exc}") from exc
    return {"invoice": invoice}


@app.get("/health", tags=["health"])
def health() -> dict:
    if not os.environ.get("INGEST_SERVICE_TOKEN"):
        logger.warning("INGEST_SERVICE_TOKEN is not set — the extraction endpoints are OPEN.")
    return {"status": "ok", "version": __version__, "ocr": ocr_status()}
