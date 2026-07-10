"""Tests for doceval.mcp_server — the MCP tool functions.

These call the tool functions directly (they remain plain, callable Python
functions after @mcp.tool() registration) rather than driving the full MCP
stdio protocol — see the manual smoke test in the README for a protocol-level
check.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from doceval.mcp_server import run_eval, score_extraction


# ── score_extraction ────────────────────────────────────────────────────────────

def test_score_extraction_all_match():
    result = score_extraction(
        {"vendor": "Acme Corp", "total": "100.00"},
        {"vendor": "Acme Corp", "total": "100.00"},
    )
    assert result["fields_correct"] == 2
    assert result["fields_total"] == 2
    assert result["accuracy"] == 1.0
    assert all(f["match"] for f in result["fields"])


def test_score_extraction_reports_mismatches():
    result = score_extraction(
        {"vendor": "Acme Corp", "total": "100.00"},
        {"vendor": "WRONG", "total": "100.00"},
    )
    assert result["fields_correct"] == 1
    assert result["fields_total"] == 2
    assert result["accuracy"] == 0.5
    vendor_field = next(f for f in result["fields"] if f["field"] == "vendor")
    assert vendor_field["match"] is False
    assert vendor_field["failure_mode"] == "wrong_value"


def test_score_extraction_handles_nested_dicts():
    result = score_extraction(
        {"address": {"city": "NY"}},
        {"address": {"city": "NY"}},
    )
    assert result["fields"][0]["field"] == "address.city"
    assert result["accuracy"] == 1.0


def test_score_extraction_empty_expected_returns_zero_total():
    result = score_extraction({}, {"vendor": "unexpected"})
    assert result["fields_total"] == 0
    assert result["accuracy"] == 0.0


# ── run_eval ───────────────────────────────────────────────────────────────────

def test_run_eval_over_paired_dataset(paired_dataset, tmp_path):
    docs_dir, labels_dir = paired_dataset
    extractor_module = tmp_path / "stub_extractor.py"
    extractor_module.write_text(
        "def extract(doc_bytes, filepath):\n"
        "    if 'invoice_001' in filepath:\n"
        "        return {'vendor': 'Acme Corp', 'total': '100.00'}\n"
        "    return {'vendor': 'Beta LLC', 'total': '200.00'}\n"
    )
    sys.path.insert(0, str(tmp_path))
    sys.modules.pop("stub_extractor", None)

    output = run_eval(str(docs_dir), str(labels_dir), "stub_extractor:extract")

    assert output["result"]["total_documents"] == 2
    assert output["result"]["overall_accuracy"] == 1.0
    assert "# doceval" in output["report_markdown"]
    assert "Field accuracy" in output["report_markdown"]


def test_run_eval_bad_extractor_spec_raises(paired_dataset):
    docs_dir, labels_dir = paired_dataset
    try:
        run_eval(str(docs_dir), str(labels_dir), "no_colon_here")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "module:function" in str(exc)


def test_run_eval_accepts_csv_manifest(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    docs_dir.joinpath("doc1.pdf").write_bytes(b"x")
    manifest = tmp_path / "labels.csv"
    manifest.write_text("filename,vendor\ndoc1,Acme Corp\n")

    extractor_module = tmp_path / "stub_extractor2.py"
    extractor_module.write_text(
        "def extract(doc_bytes, filepath):\n"
        "    return {'vendor': 'Acme Corp'}\n"
    )
    sys.path.insert(0, str(tmp_path))
    sys.modules.pop("stub_extractor2", None)

    output = run_eval(str(docs_dir), str(manifest), "stub_extractor2:extract")

    assert output["result"]["overall_accuracy"] == 1.0
