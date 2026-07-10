"""Tests for doceval.harness — run_eval orchestration."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from doceval.harness import run_eval


# ── Happy path ──────────────────────────────────────────────────────────────────

def test_happy_path_exact_match(paired_dataset):
    docs_dir, labels_dir = paired_dataset

    def extract(doc_bytes, filepath):
        if "invoice_001" in filepath:
            return {"vendor": "Acme Corp", "total": "100.00"}
        return {"vendor": "Beta LLC", "total": "200.00"}

    result = run_eval(docs_dir, labels_dir, extract)

    assert result.total_documents == 2
    assert result.successful == 2
    assert result.failed == 0
    assert result.fields_correct == 4
    assert result.fields_total == 4
    assert result.overall_accuracy == 1.0
    assert result.errors == []


def test_partial_match_produces_failure_modes(paired_dataset):
    docs_dir, labels_dir = paired_dataset

    def extract(doc_bytes, filepath):
        return {"vendor": "WRONG NAME", "total": "100.00"}

    result = run_eval(docs_dir, labels_dir, extract)

    assert result.overall_accuracy < 1.0
    assert "wrong_value" in result.failure_modes


# ── Extractor failure handling ────────────────────────────────────────────────

def test_extractor_exception_does_not_crash_run(paired_dataset):
    docs_dir, labels_dir = paired_dataset

    def extract(doc_bytes, filepath):
        if "invoice_001" in filepath:
            raise RuntimeError("API timeout")
        return {"vendor": "Beta LLC", "total": "200.00"}

    result = run_eval(docs_dir, labels_dir, extract)

    assert result.total_documents == 2
    assert result.successful == 1
    assert result.failed == 1
    failed_doc = next(d for d in result.documents if d.error)
    assert failed_doc.filename == "invoice_001.pdf"
    assert "API timeout" in failed_doc.error
    assert any(e["filename"] == "invoice_001.pdf" for e in result.errors)
    # a failed doc contributes no fields to the overall tally
    assert result.fields_total == 2


def test_extractor_returns_wrong_type(paired_dataset):
    docs_dir, labels_dir = paired_dataset

    def extract(doc_bytes, filepath):
        return "not a dict"

    result = run_eval(docs_dir, labels_dir, extract)

    assert result.failed == 2
    assert all("must return dict" in d.error for d in result.documents)


# ── Unpaired docs / labels ────────────────────────────────────────────────────

def test_unpaired_document_is_reported_not_crashed(paired_dataset):
    docs_dir, labels_dir = paired_dataset
    docs_dir.joinpath("invoice_003.pdf").write_bytes(b"fake-pdf-3")  # no label

    result = run_eval(docs_dir, labels_dir, lambda b, f: {"vendor": "x", "total": "1"})

    assert result.total_documents == 2  # only paired docs are evaluated
    unpaired = [e for e in result.errors if e["filename"] == "invoice_003.pdf"]
    assert len(unpaired) == 1
    assert "No matching label" in unpaired[0]["error"]


def test_unpaired_label_is_reported_not_crashed(paired_dataset):
    docs_dir, labels_dir = paired_dataset
    labels_dir.joinpath("invoice_999.json").write_text(json.dumps({"vendor": "x"}))  # no doc

    result = run_eval(docs_dir, labels_dir, lambda b, f: {"vendor": "x", "total": "1"})

    assert result.total_documents == 2
    unpaired = [e for e in result.errors if e["filename"] == "invoice_999"]
    assert len(unpaired) == 1
    assert "No matching document" in unpaired[0]["error"]


# ── Cost tracking ──────────────────────────────────────────────────────────────

def test_cost_tuple_is_tracked(paired_dataset):
    docs_dir, labels_dir = paired_dataset

    def extract(doc_bytes, filepath):
        return {"vendor": "Acme Corp", "total": "100.00"}, 0.01

    result = run_eval(docs_dir, labels_dir, extract)

    assert result.total_cost_usd == 0.02
    assert result.avg_cost_per_doc_usd == 0.01
    assert all(d.cost_usd == 0.01 for d in result.documents)


def test_no_cost_reported_leaves_cost_fields_none(paired_dataset):
    docs_dir, labels_dir = paired_dataset

    def extract(doc_bytes, filepath):
        return {"vendor": "Acme Corp", "total": "100.00"}

    result = run_eval(docs_dir, labels_dir, extract)

    assert result.total_cost_usd is None
    assert result.avg_cost_per_doc_usd is None
    assert all(d.cost_usd is None for d in result.documents)


# ── Nested labels and __meta__ ────────────────────────────────────────────────

def test_nested_label_fields_are_flattened(tmp_path):
    docs_dir = tmp_path / "docs"
    labels_dir = tmp_path / "labels"
    docs_dir.mkdir()
    labels_dir.mkdir()
    docs_dir.joinpath("doc1.pdf").write_bytes(b"x")
    labels_dir.joinpath("doc1.json").write_text(json.dumps({
        "address": {"city": "NY", "zip": "10001"},
    }))

    def extract(doc_bytes, filepath):
        return {"address": {"city": "NY", "zip": "10001"}}

    result = run_eval(docs_dir, labels_dir, extract)

    fields = {f.field for f in result.documents[0].fields}
    assert fields == {"address.city", "address.zip"}
    assert result.overall_accuracy == 1.0


def test_meta_key_is_passed_through_not_compared(tmp_path):
    docs_dir = tmp_path / "docs"
    labels_dir = tmp_path / "labels"
    docs_dir.mkdir()
    labels_dir.mkdir()
    docs_dir.joinpath("doc1.pdf").write_bytes(b"x")
    labels_dir.joinpath("doc1.json").write_text(json.dumps({
        "vendor": "Acme Corp",
        "__meta__": {"difficulty": "hard"},
    }))

    def extract(doc_bytes, filepath):
        return {"vendor": "Acme Corp"}

    result = run_eval(docs_dir, labels_dir, extract)

    doc = result.documents[0]
    assert doc.metadata == {"difficulty": "hard"}
    assert doc.fields_total == 1  # __meta__ itself is not a scored field


# ── progress_callback ──────────────────────────────────────────────────────────

def test_progress_callback_called_before_and_after_each_doc(paired_dataset):
    docs_dir, labels_dir = paired_dataset
    calls = []

    def on_progress(idx, total, filename, result):
        calls.append((idx, total, filename, result is None))

    run_eval(docs_dir, labels_dir, lambda b, f: {"vendor": "x", "total": "1"},
              progress_callback=on_progress)

    assert len(calls) == 4  # 2 docs × (before, after)
    assert calls[0] == (1, 2, "invoice_001.pdf", True)
    assert calls[1] == (1, 2, "invoice_001.pdf", False)
    assert calls[2] == (2, 2, "invoice_002.pdf", True)
    assert calls[3] == (2, 2, "invoice_002.pdf", False)


# ── Aggregation across documents ──────────────────────────────────────────────

def test_by_field_aggregation_across_documents(paired_dataset):
    docs_dir, labels_dir = paired_dataset

    def extract(doc_bytes, filepath):
        # vendor always right, total always wrong
        if "invoice_001" in filepath:
            return {"vendor": "Acme Corp", "total": "999.00"}
        return {"vendor": "Beta LLC", "total": "999.00"}

    result = run_eval(docs_dir, labels_dir, extract)

    assert result.by_field["vendor"].correct == 2
    assert result.by_field["vendor"].total == 2
    assert result.by_field["total"].correct == 0
    assert result.by_field["total"].total == 2
    assert result.failure_modes.get("wrong_format", 0) == 2


# ── CSV / JSONL label manifests ───────────────────────────────────────────────

def test_csv_manifest_labels(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    docs_dir.joinpath("invoice_001.pdf").write_bytes(b"x")
    docs_dir.joinpath("invoice_002.pdf").write_bytes(b"y")

    manifest = tmp_path / "labels.csv"
    manifest.write_text(
        "filename,vendor,total\n"
        "invoice_001.pdf,Acme Corp,100.00\n"
        "invoice_002,Beta LLC,200.00\n"  # no extension — still matches by stem
    )

    def extract(doc_bytes, filepath):
        if "invoice_001" in filepath:
            return {"vendor": "Acme Corp", "total": "100.00"}
        return {"vendor": "Beta LLC", "total": "200.00"}

    result = run_eval(docs_dir, manifest, extract)

    assert result.total_documents == 2
    assert result.overall_accuracy == 1.0


def test_csv_manifest_missing_filename_column_raises(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    manifest = tmp_path / "labels.csv"
    manifest.write_text("vendor,total\nAcme Corp,100.00\n")

    try:
        run_eval(docs_dir, manifest, lambda b, f: {})
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "filename" in str(exc)


def test_jsonl_manifest_labels(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    docs_dir.joinpath("invoice_001.pdf").write_bytes(b"x")

    manifest = tmp_path / "labels.jsonl"
    manifest.write_text(
        json.dumps({"filename": "invoice_001", "vendor": "Acme Corp", "total": "100.00"}) + "\n"
    )

    def extract(doc_bytes, filepath):
        return {"vendor": "Acme Corp", "total": "100.00"}

    result = run_eval(docs_dir, manifest, extract)

    assert result.total_documents == 1
    assert result.overall_accuracy == 1.0


def test_jsonl_manifest_missing_filename_key_raises(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    manifest = tmp_path / "labels.jsonl"
    manifest.write_text(json.dumps({"vendor": "Acme Corp"}) + "\n")

    try:
        run_eval(docs_dir, manifest, lambda b, f: {})
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "filename" in str(exc)


def test_unsupported_labels_path_raises(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    bad = tmp_path / "labels.txt"
    bad.write_text("not a manifest")

    try:
        run_eval(docs_dir, bad, lambda b, f: {})
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "Unsupported labels path" in str(exc)
