"""Tests for doceval.cli — the `doceval run` command."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from click.testing import CliRunner

from doceval.cli import cli


def _write_extractor_module(tmp_path, body):
    """Write a my_extractor.py module into tmp_path and put tmp_path on sys.path."""
    module_path = tmp_path / "my_extractor.py"
    module_path.write_text(body)
    if str(tmp_path) not in sys.path:
        sys.path.insert(0, str(tmp_path))
    sys.modules.pop("my_extractor", None)  # avoid stale module across tests


HAPPY_EXTRACTOR = """
def extract(doc_bytes, filepath):
    return {"vendor": "Acme Corp", "total": "100.00"}
"""

FAILING_EXTRACTOR = """
def extract(doc_bytes, filepath):
    raise RuntimeError("boom")
"""


def _dataset(paired_dataset):
    docs_dir, labels_dir = paired_dataset
    # Overwrite with a single doc/label pair so happy-path extractors match exactly.
    for p in docs_dir.iterdir():
        p.unlink()
    for p in labels_dir.iterdir():
        p.unlink()
    docs_dir.joinpath("doc1.pdf").write_bytes(b"x")
    labels_dir.joinpath("doc1.json").write_text(json.dumps({
        "vendor": "Acme Corp", "total": "100.00",
    }))
    return docs_dir, labels_dir


def test_run_happy_path_exits_zero(tmp_path, paired_dataset):
    docs_dir, labels_dir = _dataset(paired_dataset)
    _write_extractor_module(tmp_path, HAPPY_EXTRACTOR)

    runner = CliRunner()
    result = runner.invoke(cli, [
        "run", "--docs", str(docs_dir), "--labels", str(labels_dir),
        "--extractor", "my_extractor:extract",
    ])

    assert result.exit_code == 0, result.output
    assert "Overall: 2/2 fields correct (100.0%)" in result.output


def test_run_exits_nonzero_on_extraction_failure(tmp_path, paired_dataset):
    docs_dir, labels_dir = _dataset(paired_dataset)
    _write_extractor_module(tmp_path, FAILING_EXTRACTOR)

    runner = CliRunner()
    result = runner.invoke(cli, [
        "run", "--docs", str(docs_dir), "--labels", str(labels_dir),
        "--extractor", "my_extractor:extract",
    ])

    assert result.exit_code == 1


def test_run_quiet_suppresses_progress_lines(tmp_path, paired_dataset):
    docs_dir, labels_dir = _dataset(paired_dataset)
    _write_extractor_module(tmp_path, HAPPY_EXTRACTOR)

    runner = CliRunner()
    result = runner.invoke(cli, [
        "run", "--docs", str(docs_dir), "--labels", str(labels_dir),
        "--extractor", "my_extractor:extract", "--quiet",
    ])

    assert result.exit_code == 0
    assert "doc1.pdf" not in result.output
    assert "Overall:" in result.output  # final summary still prints


def test_run_json_out_matches_documented_schema(tmp_path, paired_dataset):
    docs_dir, labels_dir = _dataset(paired_dataset)
    _write_extractor_module(tmp_path, HAPPY_EXTRACTOR)
    json_path = tmp_path / "result.json"

    runner = CliRunner()
    result = runner.invoke(cli, [
        "run", "--docs", str(docs_dir), "--labels", str(labels_dir),
        "--extractor", "my_extractor:extract", "--json-out", str(json_path),
    ])

    assert result.exit_code == 0, result.output
    data = json.loads(json_path.read_text())
    for key in (
        "run_id", "extractor_name", "total_documents", "successful", "failed",
        "fields_correct", "fields_total", "overall_accuracy", "failure_modes",
        "by_field", "total_cost_usd", "avg_cost_per_doc_usd", "documents", "errors",
    ):
        assert key in data
    assert data["overall_accuracy"] == 1.0


def test_run_output_flag_controls_report_path(tmp_path, paired_dataset):
    docs_dir, labels_dir = _dataset(paired_dataset)
    _write_extractor_module(tmp_path, HAPPY_EXTRACTOR)
    report_path = tmp_path / "custom-report.md"

    runner = CliRunner()
    result = runner.invoke(cli, [
        "run", "--docs", str(docs_dir), "--labels", str(labels_dir),
        "--extractor", "my_extractor:extract", "--output", str(report_path),
    ])

    assert result.exit_code == 0
    assert report_path.exists()
    assert "doceval" in report_path.read_text().lower()


def test_run_bad_extractor_spec_missing_colon(tmp_path, paired_dataset):
    docs_dir, labels_dir = _dataset(paired_dataset)

    runner = CliRunner()
    result = runner.invoke(cli, [
        "run", "--docs", str(docs_dir), "--labels", str(labels_dir),
        "--extractor", "no_colon_here",
    ])

    assert result.exit_code != 0
    assert "module:function" in result.output


def test_run_bad_extractor_module_not_found(tmp_path, paired_dataset):
    docs_dir, labels_dir = _dataset(paired_dataset)

    runner = CliRunner()
    result = runner.invoke(cli, [
        "run", "--docs", str(docs_dir), "--labels", str(labels_dir),
        "--extractor", "nonexistent_module_xyz:extract",
    ])

    assert result.exit_code != 0
    assert "Cannot import module" in result.output


def test_run_bad_extractor_function_not_found(tmp_path, paired_dataset):
    docs_dir, labels_dir = _dataset(paired_dataset)
    _write_extractor_module(tmp_path, HAPPY_EXTRACTOR)

    runner = CliRunner()
    result = runner.invoke(cli, [
        "run", "--docs", str(docs_dir), "--labels", str(labels_dir),
        "--extractor", "my_extractor:does_not_exist",
    ])

    assert result.exit_code != 0
    assert "not found in module" in result.output


def test_run_accepts_csv_manifest_for_labels(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    docs_dir.joinpath("doc1.pdf").write_bytes(b"x")
    manifest = tmp_path / "labels.csv"
    manifest.write_text("filename,vendor,total\ndoc1,Acme Corp,100.00\n")
    _write_extractor_module(tmp_path, HAPPY_EXTRACTOR)

    runner = CliRunner()
    result = runner.invoke(cli, [
        "run", "--docs", str(docs_dir), "--labels", str(manifest),
        "--extractor", "my_extractor:extract",
    ])

    assert result.exit_code == 0, result.output
    assert "Overall: 2/2 fields correct (100.0%)" in result.output
