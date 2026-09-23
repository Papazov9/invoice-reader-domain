"""Validation of Bulgarian EIK / BULSTAT identifiers.

The check digit uses a weighted modulo-11 scheme that is specific to BULSTAT and
differs from the EGN scheme, so a Luhn or EGN check would reject valid numbers. A
9-digit identifier carries one check digit; a 13-digit branch identifier carries a
second check digit over the trailing four positions.

Validating before any register lookup lets us reject OCR digit garbles cheaply,
without a network round trip.
"""

from __future__ import annotations


def _weighted_check(digits: list[int], weights1: list[int], weights2: list[int]) -> int:
    """Modulo-11 check digit with the BULSTAT two-weight fallback."""
    r = sum(d * w for d, w in zip(digits, weights1)) % 11
    if r != 10:
        return r
    r = sum(d * w for d, w in zip(digits, weights2)) % 11
    return 0 if r == 10 else r


def eik_check_digit_9(first8: list[int]) -> int:
    return _weighted_check(first8, [1, 2, 3, 4, 5, 6, 7, 8], [3, 4, 5, 6, 7, 8, 9, 10])


def eik_check_digit_13(middle4: list[int]) -> int:
    return _weighted_check(middle4, [2, 7, 3, 5], [4, 9, 5, 7])


def validate_eik(eik: str | None) -> bool:
    """True if eik is a structurally valid 9- or 13-digit BULSTAT identifier."""
    if not eik:
        return False
    s = eik.strip()
    if not s.isdigit() or len(s) not in (9, 13):
        return False
    d = [int(c) for c in s]
    if eik_check_digit_9(d[:8]) != d[8]:
        return False
    if len(s) == 13 and eik_check_digit_13(d[8:12]) != d[12]:
        return False
    return True
