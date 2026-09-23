"""Individual, deterministic invoice validation rules.

Each rule is a pure function Invoice -> ValidationResult | None (None means not
applicable, e.g. a required input is missing and a different completeness rule owns
that complaint). All arithmetic uses Decimal: no LLM, no floats in the money math.
"""

from __future__ import annotations

from decimal import Decimal

from app.core import get_settings
from app.domain import Invoice, Severity, ValidationResult


def _tol() -> Decimal:
    return Decimal(str(get_settings().validation_amount_tolerance))


def rule_arithmetic_total(inv: Invoice) -> ValidationResult | None:
    """net + vat == total (within tolerance)."""
    if inv.net_amount is None or inv.vat_amount is None or inv.total_amount is None:
        return None
    expected = inv.net_amount + inv.vat_amount
    diff = abs(expected - inv.total_amount)
    passed = diff <= _tol()
    return ValidationResult(
        rule_id="arithmetic.net_plus_vat_equals_total",
        passed=passed,
        severity=Severity.ERROR,
        message=(
            "net + VAT equals total"
            if passed
            else f"net ({inv.net_amount}) + VAT ({inv.vat_amount}) = {expected} "
            f"≠ total ({inv.total_amount})"
        ),
        evidence={"expected_total": str(expected), "stated_total": str(inv.total_amount),
                  "difference": str(diff)},
    )


def rule_line_items_sum(inv: Invoice) -> ValidationResult | None:
    """Sum of line-item amounts == net (within tolerance)."""
    amounts = [li.amount for li in inv.line_items if li.amount is not None]
    if not amounts or inv.net_amount is None:
        return None
    total = sum(amounts, Decimal("0"))
    diff = abs(total - inv.net_amount)
    passed = diff <= _tol()
    return ValidationResult(
        rule_id="arithmetic.line_items_sum_equals_net",
        passed=passed,
        severity=Severity.WARNING,
        message=("line items sum to net" if passed
                 else f"line items sum to {total} ≠ net ({inv.net_amount})"),
        evidence={"line_items_sum": str(total), "net": str(inv.net_amount)},
    )


def rule_vat_rate_plausible(inv: Invoice) -> ValidationResult | None:
    """Each VAT tax line uses a recognised rate, and base*rate == amount."""
    if not inv.tax_lines:
        return None
    valid = [Decimal(str(r)) for r in get_settings().validation_valid_vat_rates]
    tol = _tol()
    problems: list[str] = []
    for i, tl in enumerate(inv.tax_lines):
        if all(abs(tl.rate - v) > Decimal("0.005") for v in valid):
            problems.append(f"line {i}: rate {tl.rate} not in {valid}")
        if tl.base is not None and tl.amount is not None:
            expected = (tl.base * tl.rate)
            if abs(expected - tl.amount) > tol:
                problems.append(
                    f"line {i}: base {tl.base} * rate {tl.rate} = {expected} ≠ {tl.amount}"
                )
    passed = not problems
    return ValidationResult(
        rule_id="vat.rate_plausible_and_consistent",
        passed=passed,
        severity=Severity.ERROR,
        message="VAT rates valid and consistent" if passed else "; ".join(problems),
        evidence={"valid_rates": ", ".join(f"{float(v) * 100:g}%" for v in valid)},
    )


def rule_vat_number_format(inv: Invoice) -> ValidationResult | None:
    """VAT numbers look like Bulgarian (BG + 9/10 digits) or another EU VAT
    (2-letter country code + 8-12 alphanumerics); foreign counterparties are valid."""
    import re

    bg = re.compile(r"^BG\d{9,10}$")
    eu = re.compile(r"^[A-Z]{2}[0-9A-Z]{8,12}$")
    bad = []
    for who, party in (("supplier", inv.supplier), ("recipient", inv.recipient)):
        if party.vat_number:
            v = party.vat_number.replace(" ", "").upper()
            if not (bg.match(v) or eu.match(v)):
                bad.append(f"{who}: {party.vat_number}")
    if not bad and not inv.supplier.vat_number and not inv.recipient.vat_number:
        return None  # nothing to check; completeness rule's concern
    passed = not bad
    return ValidationResult(
        rule_id="format.vat_number",
        passed=passed,
        severity=Severity.WARNING,
        message="VAT number format valid" if passed else f"invalid VAT format: {', '.join(bad)}",
    )


def rule_eik_format(inv: Invoice) -> ValidationResult | None:
    """A Bulgarian EIK is 9 or 13 digits with a modulo-11 check digit. Only all-digit
    values are checked, since a foreign counterparty's tax id may be stored here and is
    not an EIK."""
    from app.tools.ingest.eik import validate_eik

    bad = []
    for who, party in (("supplier", inv.supplier), ("recipient", inv.recipient)):
        eik = party.eik
        if not eik or not eik.isdigit():
            continue
        if len(eik) not in (9, 13):
            bad.append(f"{who}: {eik} (invalid length)")
        elif not validate_eik(eik):
            bad.append(f"{who}: {eik} (checksum failed)")
    if not bad and not inv.supplier.eik and not inv.recipient.eik:
        return None
    passed = not bad
    return ValidationResult(
        rule_id="format.eik",
        passed=passed,
        severity=Severity.WARNING,
        message="EIK format valid" if passed else f"invalid EIK: {', '.join(bad)}",
    )


def rule_completeness(inv: Invoice) -> ValidationResult:
    """Required fields for a usable invoice: number, date, total, and an identifiable
    counterparty (a journal often omits our own firm, so we require at least one party)."""
    required = {
        "number": inv.number,
        "date": inv.date,
        "total_amount": inv.total_amount,
    }
    missing = [name for name, value in required.items() if value in (None, "")]
    has_party = any([
        inv.supplier.name, inv.supplier.vat_number, inv.supplier.eik,
        inv.recipient.name, inv.recipient.vat_number, inv.recipient.eik,
    ])
    if not has_party:
        missing.append("counterparty")
    passed = not missing
    return ValidationResult(
        rule_id="completeness.required_fields",
        passed=passed,
        severity=Severity.ERROR,
        message="all required fields present" if passed else f"missing: {', '.join(missing)}",
        evidence={"missing": ",".join(missing)} if missing else {},
    )


# Order matters only for presentation.
ALL_RULES = [
    rule_completeness,
    rule_arithmetic_total,
    rule_line_items_sum,
    rule_vat_rate_plausible,
    rule_vat_number_format,
    rule_eik_format,
]
