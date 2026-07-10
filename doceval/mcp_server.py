"""
doceval MCP server — exposes doc-extraction accuracy scoring as agent tools.

Run directly:
    doceval-mcp
    # or
    python -m doceval.mcp_server

Configure in an MCP client (e.g. Claude Code, Claude Desktop) by pointing it at
this command over stdio. See README.md for a config example.

Tools:
    score_extraction  Score one extraction result against expected labels.
                       No filesystem access needed — pass both dicts directly.
    run_eval           Run a full eval over a docs/labels dataset on disk,
                       using a Python extractor function. Returns the run
                       result plus a rendered Markdown report.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from mcp.server.fastmcp import FastMCP

from .compare import compare_dicts
from .harness import run_eval as _run_eval
from .loader import load_callable
from .report import generate_report

mcp = FastMCP(
    "doceval",
    instructions=(
        "Score LLM document-extraction accuracy against ground-truth labels. "
        "Use score_extraction to check a single extraction result you already "
        "have. Use run_eval to score a whole docs/labels dataset on disk "
        "against a Python extractor function."
    ),
)


@mcp.tool()
def score_extraction(expected: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    """
    Score one extraction result against its expected (ground-truth) values.

    Both dicts may be flat or nested (nested keys are dot-flattened, e.g.
    {"address": {"city": "NY"}} becomes field "address.city"). Only fields
    present in `expected` are scored — extra fields in `actual` are ignored.

    Each mismatch is classified as one of:
      missed_field   — expected has a value, actual is empty
      hallucination  — actual has a value, expected is empty
      wrong_format   — both non-empty, numeric/date values differ
      wrong_value    — both non-empty, string values differ
    """
    results = compare_dicts(expected, actual)
    correct = sum(r.match for r in results)
    total = len(results)
    return {
        "fields_correct": correct,
        "fields_total": total,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "fields": [asdict(r) for r in results],
    }


@mcp.tool()
def run_eval(
    docs_dir: str,
    labels_dir: str,
    extractor: str,
    extractor_name: str | None = None,
) -> dict[str, Any]:
    """
    Run a full evaluation over a docs/labels dataset and return the result.

    Args:
        docs_dir:       Directory containing documents to evaluate.
        labels_dir:     Either a directory of one JSON label file per document
                        (named <doc_stem>.json), or a single .csv/.jsonl/.ndjson
                        manifest file with a "filename" column/key.
        extractor:      A Python extractor as 'module:function', importable
                        from the current working directory. Signature:
                        def extract(doc_bytes: bytes, filepath: str) -> dict
                        (or -> tuple[dict, float] to also report cost).
        extractor_name: Display name used in the report. Defaults to `extractor`.

    Returns a dict with the structured run result (accuracy, failure modes,
    per-field stats, per-document results) and a rendered Markdown report.
    """
    extract_fn = load_callable(extractor)
    result = _run_eval(
        docs_dir=docs_dir,
        labels_dir=labels_dir,
        extract_fn=extract_fn,
        extractor_name=extractor_name or extractor,
    )
    return {
        "result": asdict(result),
        "report_markdown": generate_report(result),
    }


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
