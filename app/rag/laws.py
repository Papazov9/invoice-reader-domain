"""Laws RAG, backed by the lex engine (backend/lex).

lex scrapes lex.bg, chunks by член/алинея and does hybrid BM25 + dense (bge-m3)
retrieval with cross-encoder reranking. This adapter loads lex's RetrievalPipeline
once and maps its results into the backend's RetrievedChunk so the /chat orchestrator
can merge laws with the invoices RAG.

lex needs the backend dependencies (requirements.txt) and a built index
(`python lex/run_ingest.py`, which scrapes lex.bg and downloads the bge models). If
the index is missing the retriever logs once and returns nothing, so the app still runs
and /chat answers from the invoices RAG alone.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from app.rag.base import RetrievedChunk

logger = logging.getLogger(__name__)

# backend/app/rag/laws.py -> parents[2] == backend, so the engine lives at backend/lex.
_LEX_DIR = Path(__file__).resolve().parents[2] / "lex"

_pipeline = None
_tried = False


def _get_pipeline():
    """Build lex's RetrievalPipeline once and reuse it (it loads ~2GB of models and
    opens the index). Returns None if lex's deps or index are not available yet."""
    global _pipeline, _tried
    if _pipeline is not None:
        return _pipeline
    if _tried:
        return None
    _tried = True
    if str(_LEX_DIR) not in sys.path:
        sys.path.insert(0, str(_LEX_DIR))
    try:
        # On Apple Silicon the bge models exhaust the MPS GPU pool under memory
        # pressure; force CPU (query-time cost is small). No-op on Linux/CUDA.
        import torch

        torch.backends.mps.is_available = lambda: False
        from rag.retrieval.pipeline import RetrievalPipeline

        _pipeline = RetrievalPipeline()
    except Exception as exc:
        logger.warning(
            "Laws RAG (lex) unavailable: %s. Install requirements.txt and "
            "build the index with `python lex/run_ingest.py`.",
            exc,
        )
        _pipeline = None
    return _pipeline


def reset_pipeline() -> None:
    """Drop the cached pipeline so the next query reopens the (possibly rebuilt) index."""
    global _pipeline, _tried
    _pipeline = None
    _tried = False


class LawsRetriever:
    name = "laws"

    def __init__(self, db=None):
        self.db = db

    def retrieve(self, query: str, top_k: int = 6, **_: object) -> list[RetrievedChunk]:
        pipeline = _get_pipeline()
        if pipeline is None:
            return []
        try:
            result = pipeline.retrieve(query)
        except Exception as exc:
            logger.warning("Laws RAG (lex) query failed: %s", exc)
            return []
        if not result.has_confident_source:
            return []

        chunks: list[RetrievedChunk] = []
        for rc in result.results[:top_k]:
            cit = rc.chunk.citation
            score = rc.rerank_score if rc.rerank_score is not None else rc.fused_score
            chunks.append(
                RetrievedChunk(
                    id=f"law:{rc.chunk.id}",
                    text=rc.chunk.text,
                    source=cit.label(),
                    score=float(score),
                    kind="law",
                    metadata={
                        "url": cit.url,
                        "law_abbr": cit.law_abbr,
                        "law_name": cit.law_name,
                        "article": cit.article,
                        "paragraph": cit.paragraph,
                    },
                )
            )
        return chunks
