"""Tests for doceval.compare — field comparison and failure mode classification."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from doceval.compare import compare_dicts, compare_field, flatten


# ── Exact match ───────────────────────────────────────────────────────────────

def test_exact_match_strings():
    r = compare_field("vendor", "Acme Corp", "Acme Corp")
    assert r.match is True
    assert r.failure_mode is None


def test_case_insensitive_match():
    r = compare_field("vendor", "acme corp", "ACME CORP")
    assert r.match is True


def test_both_empty_is_match():
    r = compare_field("discount", "", None)
    assert r.match is True
    assert r.failure_mode is None


# ── Numeric matching ──────────────────────────────────────────────────────────

def test_numeric_match_with_currency_symbol():
    r = compare_field("total", "748.36", "$748.36")
    assert r.match is True


def test_numeric_match_european_format():
    r = compare_field("total", "1234.56", "1.234,56")
    assert r.match is True


def test_numeric_match_comma_thousands():
    r = compare_field("subtotal", "1,234.56", "1234.56")
    assert r.match is True


def test_numeric_mismatch():
    r = compare_field("total", "100.00", "200.00")
    assert r.match is False
    assert r.failure_mode == "wrong_format"


# ── Date matching ──────────────────────────────────────────────────────────────

def test_date_match_iso_vs_abbreviated():
    r = compare_field("invoice_date", "2012-11-15", "Nov 15 2012")
    assert r.match is True


def test_date_match_iso_vs_iso():
    r = compare_field("date", "2026-01-15", "2026-01-15")
    assert r.match is True


def test_date_mismatch():
    r = compare_field("date", "2026-01-15", "2026-02-15")
    assert r.match is False
    assert r.failure_mode == "wrong_format"


# ── Failure mode classification ────────────────────────────────────────────────

def test_missed_field():
    r = compare_field("shipping", "73.00", "")
    assert r.match is False
    assert r.failure_mode == "missed_field"


def test_hallucination():
    r = compare_field("discount", "", "10.00")
    assert r.match is False
    assert r.failure_mode == "hallucination"


def test_wrong_value_string():
    r = compare_field("vendor", "Acme Corp", "Beta LLC")
    assert r.match is False
    assert r.failure_mode == "wrong_value"


def test_wrong_format_numeric():
    r = compare_field("total", "100.00", "101.00")
    assert r.match is False
    assert r.failure_mode == "wrong_format"


# ── None handling ──────────────────────────────────────────────────────────────

def test_none_actual_vs_expected_value():
    r = compare_field("order_id", "ORD-123", None)
    assert r.match is False
    assert r.failure_mode == "missed_field"


def test_none_both():
    r = compare_field("customer_name", None, None)
    assert r.match is True


# ── flatten ──────────────────────────────────────────────────────────────────

def test_flatten_nested_dict():
    assert flatten({"address": {"city": "NY", "zip": "10001"}}) == {
        "address.city": "NY", "address.zip": "10001",
    }


def test_flatten_skips_meta_key():
    assert flatten({"vendor": "Acme", "__meta__": {"difficulty": "hard"}}) == {
        "vendor": "Acme",
    }


def test_flatten_list_value_is_json_encoded():
    assert flatten({"tags": ["a", "b"]}) == {"tags": '["a", "b"]'}


def test_flatten_none_value_becomes_empty_string():
    assert flatten({"discount": None}) == {"discount": ""}


# ── compare_dicts ──────────────────────────────────────────────────────────────

def test_compare_dicts_all_match():
    results = compare_dicts(
        {"vendor": "Acme Corp", "total": "100.00"},
        {"vendor": "Acme Corp", "total": "100.00"},
    )
    assert len(results) == 2
    assert all(r.match for r in results)


def test_compare_dicts_reports_mismatches():
    results = compare_dicts(
        {"vendor": "Acme Corp", "total": "100.00"},
        {"vendor": "Beta LLC", "total": "100.00"},
    )
    by_field = {r.field: r for r in results}
    assert by_field["vendor"].match is False
    assert by_field["vendor"].failure_mode == "wrong_value"
    assert by_field["total"].match is True


def test_compare_dicts_only_scores_expected_fields():
    results = compare_dicts(
        {"vendor": "Acme Corp"},
        {"vendor": "Acme Corp", "extra_field": "should not be scored"},
    )
    assert len(results) == 1
    assert results[0].field == "vendor"


def test_compare_dicts_handles_nested_input():
    results = compare_dicts(
        {"address": {"city": "NY"}},
        {"address": {"city": "NY"}},
    )
    assert len(results) == 1
    assert results[0].field == "address.city"
    assert results[0].match is True
