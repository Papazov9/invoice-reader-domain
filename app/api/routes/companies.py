"""Company grouping endpoint: organise invoices into per-company working sets."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.schemas import GroupRequest, GroupResponse
from app.tools.ingest import group_by_company

router = APIRouter(prefix="/companies", tags=["companies"])


@router.post("/group", response_model=GroupResponse)
def group(req: GroupRequest) -> GroupResponse:
    return GroupResponse(groups=group_by_company(req.invoices))
