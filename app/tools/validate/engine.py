"""Validation engine: run the deterministic rule suite over an invoice."""

from __future__ import annotations

from app.domain import Invoice, ValidationResult

from .rules import ALL_RULES


def validate_invoice(invoice: Invoice) -> list[ValidationResult]:
    """Run each rule and collect the results of the applicable ones."""
    results: list[ValidationResult] = []
    for rule in ALL_RULES:
        result = rule(invoice)
        if result is not None:
            results.append(result)
    return results
