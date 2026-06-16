"""
Field-level comparison between extracted values and ground-truth labels.

compare_field() is the single public entry point — it returns a FieldResult
with a match flag and, on mismatch, a failure_mode classification:

  missed_field   — label has a value, extractor returned empty/null
  hallucination  — extractor returned a value but label is empty
  wrong_format   — both non-empty, both look numeric/date, but values differ
  wrong_value    — both non-empty, neither numeric/date logic applies
"""

from __future__ import annotations

import re
from datetime import date

from .types import FieldResult


# ── Normalisation ─────────────────────────────────────────────────────────────

def _normalize(value: str | None) -> str:
    return (value or "").strip()


def _to_numeric(s: str) -> float | None:
    """Parse a currency/number string to float. Returns None if not parseable."""
    cleaned = s.replace(" ", "")
    for sym in ("SEK", "USD", "EUR", "GBP", "NOK", "DKK", "INR", "$", "€", "£", "₹", "kr"):
        cleaned = cleaned.replace(sym, "")
    cleaned = cleaned.strip()

    if "," in cleaned and "." in cleaned:
        if cleaned.rindex(",") > cleaned.rindex("."):
            # European: "1.234,56" → "1234.56"
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:
            # English: "1,234.56" → "1234.56"
            cleaned = cleaned.replace(",", "")
    elif "," in cleaned:
        parts = cleaned.split(",")
        if len(parts) == 2 and len(parts[1]) <= 2:
            cleaned = cleaned.replace(",", ".")   # "132,30" → "132.30"
        else:
            cleaned = cleaned.replace(",", "")    # "1,234" → "1234"

    try:
        return float(cleaned)
    except ValueError:
        return None


def _to_date(s: str) -> date | None:
    """Parse ISO or common abbreviated date formats."""
    try:
        return date.fromisoformat(s)
    except ValueError:
        pass

    months = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }
    parts = re.split(r"[\s,/-]+", s.strip())
    if len(parts) == 3:
        for i, p in enumerate(parts):
            m = months.get(p[:3].lower())
            if m:
                others = [parts[j] for j in range(3) if j != i]
                try:
                    nums = [int(x) for x in others]
                    day, year = (nums[0], nums[1]) if nums[1] > 31 else (nums[1], nums[0])
                    return date(year, m, day)
                except (ValueError, IndexError):
                    pass
    return None


# ── Value matching ─────────────────────────────────────────────────────────────

def _values_match(field: str, expected: str, actual: str) -> bool:
    """Return True if the two normalised, non-empty strings represent the same value."""
    if expected.lower() == actual.lower():
        return True

    ne, na = _to_numeric(expected), _to_numeric(actual)
    if ne is not None and na is not None:
        return abs(ne - na) < 0.005  # half-cent tolerance

    # Heuristic: treat fields containing "date" as date fields
    if "date" in field.lower():
        de, da = _to_date(expected), _to_date(actual)
        if de is not None and da is not None:
            return de == da

    return False


def _looks_numeric_or_date(field: str, value: str) -> bool:
    return _to_numeric(value) is not None or (
        "date" in field.lower() and _to_date(value) is not None
    )


# ── Failure mode classification ────────────────────────────────────────────────

def _classify_failure(field: str, expected: str, actual: str) -> str:
    """
    Classify a mismatch into one of four modes:
      missed_field   expected has value, actual is empty
      hallucination  actual has value, expected is empty
      wrong_format   both non-empty, structured type (numeric/date) differs
      wrong_value    both non-empty, plain string mismatch
    """
    exp_empty = expected == ""
    act_empty = actual == ""

    if not exp_empty and act_empty:
        return "missed_field"
    if exp_empty and not act_empty:
        return "hallucination"
    # Both non-empty
    if _looks_numeric_or_date(field, expected) or _looks_numeric_or_date(field, actual):
        return "wrong_format"
    return "wrong_value"


# ── Public API ─────────────────────────────────────────────────────────────────

def compare_field(field: str, expected: str | None, actual: str | None) -> FieldResult:
    """
    Compare one extracted field value against its ground-truth label.

    Both values are coerced to strings before comparison; None is treated as "".
    """
    exp = _normalize(str(expected) if expected is not None else "")
    act = _normalize(str(actual) if actual is not None else "")

    # Both empty → correct (field not present in doc, not hallucinated)
    if exp == "" and act == "":
        return FieldResult(field=field, expected=exp, actual=act,
                           match=True, failure_mode=None)

    match = _values_match(field, exp, act)
    failure_mode = None if match else _classify_failure(field, exp, act)

    return FieldResult(field=field, expected=exp, actual=act,
                       match=match, failure_mode=failure_mode)
