"""Comparison endpoints: pairwise compare and duplicate detection."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.schemas import (
    CompareRequest,
    CompareResponse,
    DuplicatesRequest,
    DuplicatesResponse,
)
from app.tools.compare import compare_documents, find_duplicates

router = APIRouter(prefix="/compare", tags=["compare"])


@router.post("", response_model=CompareResponse)
def compare(req: CompareRequest) -> CompareResponse:
    """Score similarity between two documents with interpretable evidence."""
    return CompareResponse(evidence=compare_documents(req.a, req.b))


@router.post("/duplicates", response_model=DuplicatesResponse)
def duplicates(req: DuplicatesRequest) -> DuplicatesResponse:
    """Rank candidates against a query and flag likely duplicates."""
    matches = find_duplicates(req.query, req.candidates, top_k=req.top_k)
    return DuplicatesResponse(matches=matches)
