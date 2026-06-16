"""
Eval harness — orchestrates a full evaluation run.

Usage (programmatic):
    from doceval.harness import run_eval

    result = run_eval(
        docs_dir="./dataset/docs",
        labels_dir="./dataset/labels",
        extract_fn=my_extractor,        # def extract(bytes, str) -> dict
                                        # or -> (dict, float) with cost
        extractor_name="my_extractor",
    )

The extractor function signature:
    def extract(doc_bytes: bytes, filepath: str) -> dict
    # OR return a 2-tuple to report per-document cost:
    def extract(doc_bytes: bytes, filepath: str) -> tuple[dict, float]

Label files:
    One JSON file per document, named <doc_stem>.json, containing a flat or
    nested dict of expected field values.  An optional "__meta__" key holds
    arbitrary metadata (e.g. difficulty, doc_type) that is passed through to
    DocResult.metadata but not compared.

Supported document extensions: .pdf, .png, .jpg, .jpeg, .tiff, .tif, .webp
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable

from .compare import compare_field
from .types import DocResult, FieldResult, FieldStats, RunResult

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".webp"}


# ── Dataset loading ────────────────────────────────────────────────────────────

def _find_documents(docs_dir: Path) -> dict[str, Path]:
    """Return {stem: path} for all supported docs in docs_dir."""
    found: dict[str, Path] = {}
    for p in sorted(docs_dir.iterdir()):
        if p.suffix.lower() in SUPPORTED_EXTENSIONS:
            found[p.stem] = p
    return found


def _load_labels(labels_dir: Path) -> dict[str, dict]:
    """Return {stem: label_dict} for all .json files in labels_dir."""
    found: dict[str, dict] = {}
    for p in sorted(labels_dir.glob("*.json")):
        found[p.stem] = json.loads(p.read_text())
    return found


def _flatten(obj: dict, prefix: str = "") -> dict[str, str]:
    """
    Flatten a nested label dict into dot-separated keys.
    e.g. {"address": {"city": "NY"}} → {"address.city": "NY"}
    Skips the reserved "__meta__" key.
    """
    out: dict[str, str] = {}
    for k, v in obj.items():
        if k == "__meta__":
            continue
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, full_key))
        elif isinstance(v, list):
            out[full_key] = json.dumps(v)
        else:
            out[full_key] = "" if v is None else str(v)
    return out


# ── Per-document eval ──────────────────────────────────────────────────────────

def _eval_document(
    doc_path: Path,
    label: dict,
    extract_fn: Callable,
) -> DocResult:
    metadata = label.get("__meta__", {})
    ground_truth = _flatten(label)

    t0 = time.monotonic()
    try:
        doc_bytes = doc_path.read_bytes()
        raw = extract_fn(doc_bytes, str(doc_path))
    except Exception as exc:
        return DocResult(
            filename=doc_path.name,
            fields=[],
            fields_correct=0,
            fields_total=0,
            accuracy=0.0,
            duration_s=round(time.monotonic() - t0, 2),
            cost_usd=None,
            error=str(exc),
            metadata=metadata,
        )

    duration = round(time.monotonic() - t0, 2)

    # Unpack optional cost
    cost_usd: float | None = None
    if isinstance(raw, tuple):
        extraction, cost_usd = raw[0], float(raw[1])
    else:
        extraction = raw

    if not isinstance(extraction, dict):
        return DocResult(
            filename=doc_path.name,
            fields=[],
            fields_correct=0,
            fields_total=0,
            accuracy=0.0,
            duration_s=duration,
            cost_usd=cost_usd,
            error=f"Extractor must return dict or (dict, float); got {type(extraction).__name__}",
            metadata=metadata,
        )

    flat_extraction = _flatten(extraction)

    # Compare only fields present in the ground truth
    field_results: list[FieldResult] = []
    for field, expected_val in ground_truth.items():
        actual_val = flat_extraction.get(field)
        fr = compare_field(field, expected_val, actual_val)
        field_results.append(fr)

    fields_correct = sum(r.match for r in field_results)
    fields_total = len(field_results)
    accuracy = round(fields_correct / fields_total, 4) if fields_total else 0.0

    return DocResult(
        filename=doc_path.name,
        fields=field_results,
        fields_correct=fields_correct,
        fields_total=fields_total,
        accuracy=accuracy,
        duration_s=duration,
        cost_usd=cost_usd,
        error=None,
        metadata=metadata,
    )


# ── Aggregation ────────────────────────────────────────────────────────────────

def _aggregate(docs: list[DocResult]) -> tuple[dict[str, FieldStats], dict[str, int]]:
    """Return (by_field stats, overall failure_modes counts)."""
    field_data: dict[str, dict] = defaultdict(
        lambda: {"correct": 0, "total": 0, "failure_modes": defaultdict(int), "examples": []}
    )
    overall_modes: dict[str, int] = defaultdict(int)

    for doc in docs:
        if doc.error:
            continue
        for fr in doc.fields:
            fd = field_data[fr.field]
            fd["total"] += 1
            if fr.match:
                fd["correct"] += 1
            else:
                fd["failure_modes"][fr.failure_mode] += 1
                overall_modes[fr.failure_mode] += 1
                if len(fd["examples"]) < 3:
                    fd["examples"].append({
                        "doc": doc.filename,
                        "expected": fr.expected,
                        "actual": fr.actual,
                        "failure_mode": fr.failure_mode,
                    })

    by_field: dict[str, FieldStats] = {}
    for field, fd in field_data.items():
        by_field[field] = FieldStats(
            correct=fd["correct"],
            total=fd["total"],
            failure_modes=dict(fd["failure_modes"]),
            examples=fd["examples"],
        )

    return by_field, dict(overall_modes)


# ── Public entry point ─────────────────────────────────────────────────────────

def run_eval(
    docs_dir: str | Path,
    labels_dir: str | Path,
    extract_fn: Callable,
    extractor_name: str = "extractor",
    progress_callback: Callable[[int, int, str, DocResult | None], None] | None = None,
) -> RunResult:
    """
    Run a full evaluation and return a RunResult.

    Args:
        docs_dir:          Directory containing documents.
        labels_dir:        Directory containing label JSON files (one per doc).
        extract_fn:        Callable(bytes, str) -> dict | (dict, float).
        extractor_name:    Display name used in reports.
        progress_callback: Optional fn(index, total, filename, result_or_none).
                           Called before (result=None) and after each document.
    """
    docs_dir = Path(docs_dir)
    labels_dir = Path(labels_dir)

    documents_map = _find_documents(docs_dir)
    labels_map = _load_labels(labels_dir)

    # Only eval docs that have both a document file and a label
    paired = sorted(set(documents_map) & set(labels_map))
    unpaired_docs = sorted(set(documents_map) - set(labels_map))
    unpaired_labels = sorted(set(labels_map) - set(documents_map))

    run_id = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    doc_results: list[DocResult] = []

    for i, stem in enumerate(paired):
        doc_path = documents_map[stem]
        if progress_callback:
            progress_callback(i + 1, len(paired), doc_path.name, None)

        result = _eval_document(doc_path, labels_map[stem], extract_fn)
        doc_results.append(result)

        if progress_callback:
            progress_callback(i + 1, len(paired), doc_path.name, result)

    # Warn about unpaired items (surfaced in RunResult.errors)
    errors: list[dict] = []
    for stem in unpaired_docs:
        errors.append({"filename": documents_map[stem].name,
                       "error": "No matching label file found"})
    for stem in unpaired_labels:
        errors.append({"filename": f"{stem}.json",
                       "error": "No matching document file found"})
    for doc in doc_results:
        if doc.error:
            errors.append({"filename": doc.filename, "error": doc.error})

    successful = [d for d in doc_results if not d.error]
    failed = [d for d in doc_results if d.error]
    fields_correct = sum(d.fields_correct for d in successful)
    fields_total = sum(d.fields_total for d in successful)
    overall_accuracy = round(fields_correct / fields_total, 4) if fields_total else 0.0

    by_field, failure_modes = _aggregate(doc_results)

    cost_docs = [d for d in successful if d.cost_usd is not None]
    total_cost = round(sum(d.cost_usd for d in cost_docs), 6) if cost_docs else None
    avg_cost = round(total_cost / len(cost_docs), 6) if cost_docs else None

    return RunResult(
        run_id=run_id,
        extractor_name=extractor_name,
        total_documents=len(paired),
        successful=len(successful),
        failed=len(failed),
        fields_correct=fields_correct,
        fields_total=fields_total,
        overall_accuracy=overall_accuracy,
        failure_modes=failure_modes,
        by_field=by_field,
        total_cost_usd=total_cost,
        avg_cost_per_doc_usd=avg_cost,
        documents=doc_results,
        errors=errors,
    )
