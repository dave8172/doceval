"""Tests for doceval.report — Markdown report generation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from doceval.report import generate_report
from doceval.types import DocResult, FieldResult, FieldStats, RunResult


def _run_result(**overrides):
    defaults = dict(
        run_id="2026-01-01T00-00-00",
        extractor_name="test-extractor",
        total_documents=2,
        successful=2,
        failed=0,
        fields_correct=3,
        fields_total=4,
        overall_accuracy=0.75,
        failure_modes={"wrong_value": 1},
        by_field={
            "vendor": FieldStats(correct=2, total=2, failure_modes={}, examples=[]),
            "total": FieldStats(
                correct=1, total=2,
                failure_modes={"wrong_value": 1},
                examples=[{"doc": "doc2.pdf", "expected": "100.00",
                           "actual": "999.00", "failure_mode": "wrong_value"}],
            ),
        },
        total_cost_usd=0.02,
        avg_cost_per_doc_usd=0.01,
        documents=[
            DocResult(filename="doc1.pdf", fields=[
                FieldResult(field="vendor", expected="Acme", actual="Acme", match=True, failure_mode=None),
                FieldResult(field="total", expected="100.00", actual="100.00", match=True, failure_mode=None),
            ], fields_correct=2, fields_total=2, accuracy=1.0, duration_s=1.0, cost_usd=0.01, error=None),
            DocResult(filename="doc2.pdf", fields=[
                FieldResult(field="vendor", expected="Beta", actual="Beta", match=True, failure_mode=None),
                FieldResult(field="total", expected="100.00", actual="999.00", match=False, failure_mode="wrong_value"),
            ], fields_correct=1, fields_total=2, accuracy=0.5, duration_s=1.2, cost_usd=0.01, error=None),
        ],
        errors=[],
    )
    defaults.update(overrides)
    return RunResult(**defaults)


def test_report_includes_header_summary():
    md = generate_report(_run_result())
    assert "test-extractor" in md
    assert "75.0%" in md  # field accuracy
    assert "3/4 fields" in md


def test_report_includes_cost_when_present():
    md = generate_report(_run_result())
    assert "$0.0100" in md


def test_report_omits_cost_line_when_absent():
    md = generate_report(_run_result(total_cost_usd=None, avg_cost_per_doc_usd=None))
    assert "Avg cost/doc" not in md


def test_report_includes_failure_mode_breakdown():
    md = generate_report(_run_result())
    assert "## Failure Modes" in md
    assert "`wrong_value`" in md
    assert "100.0%" in md  # only failure mode present is 100% share


def test_report_omits_failure_modes_section_when_perfect():
    md = generate_report(_run_result(failure_modes={}))
    assert "## Failure Modes" not in md


def test_report_includes_field_level_accuracy_table():
    md = generate_report(_run_result())
    assert "## Field-Level Accuracy" in md
    assert "`vendor`" in md
    assert "`total`" in md


def test_report_includes_mismatch_examples_for_failing_fields():
    md = generate_report(_run_result())
    assert "### Mismatch Examples" in md
    assert "Expected: `100.00`" in md
    assert "Actual:   `999.00`" in md


def test_report_includes_hardest_documents():
    md = generate_report(_run_result())
    assert "## Hardest Documents" in md
    assert "doc2.pdf" in md  # the 50% accuracy doc should be surfaced


def test_report_includes_extraction_errors_section():
    result = _run_result(
        failed=1,
        errors=[{"filename": "doc3.pdf", "error": "API timeout"}],
    )
    md = generate_report(result)
    assert "## Extraction Errors" in md
    assert "doc3.pdf" in md
    assert "API timeout" in md


def test_report_omits_errors_section_when_only_unpaired_warnings():
    result = _run_result(
        errors=[{"filename": "doc3.pdf", "error": "No matching label file found"}],
    )
    md = generate_report(result)
    assert "## Extraction Errors" not in md


def test_report_handles_empty_by_field():
    md = generate_report(_run_result(by_field={}, documents=[]))
    assert "## Field-Level Accuracy" not in md
    assert "## Hardest Documents" not in md
