"""Validation endpoint: run the deterministic rule suite over an invoice."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.schemas import ValidateResponse
from app.domain import Invoice, Severity
from app.tools.validate import validate_invoice

router = APIRouter(prefix="/validate", tags=["validate"])


@router.post("", response_model=ValidateResponse)
def validate(invoice: Invoice) -> ValidateResponse:
    results = validate_invoice(invoice)
    # "valid" = no ERROR-severity rule failed.
    is_valid = not any(
        (not r.passed) and r.severity == Severity.ERROR for r in results
    )
    return ValidateResponse(invoice_id=invoice.id, results=results, is_valid=is_valid)
