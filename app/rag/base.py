"""Shared contracts for the retrieval layer.

Both RAGs (invoices and laws) return the same RetrievedChunk shape so the chat
orchestrator can merge their results and build one cited context for the LLM.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    id: str                       # stable id, e.g. "invoice:sales-2000002487" or "law:zdds-art117"
    text: str                     # the snippet handed to the LLM
    source: str                   # human-readable origin for citation
    score: float = 0.0            # relevance (1.0 = best)
    kind: str = "invoice"         # "invoice" | "law"
    metadata: dict = Field(default_factory=dict)


class Citation(BaseModel):
    id: str
    source: str
    kind: str
    url: str | None = None   # source link (lex.bg article for laws)


class Retriever(Protocol):
    """A retrieval source. Both the invoices RAG and the laws RAG implement this."""

    name: str

    def retrieve(self, query: str, top_k: int = 6, **filters: object) -> list[RetrievedChunk]:
        ...
