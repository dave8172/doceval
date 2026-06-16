"""
Markdown report generator for a RunResult.

Usage:
    from doceval.report import generate_report
    md = generate_report(result)
    Path("report.md").write_text(md)
"""

from __future__ import annotations

from .types import RunResult


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pct(correct: int, total: int) -> str:
    if not total:
        return "n/a"
    return f"{correct / total * 100:.1f}%"


def _bar(correct: int, total: int, width: int = 20) -> str:
    if not total:
        return " " * width
    filled = round(correct / total * width)
    return "█" * filled + "░" * (width - filled)


def _perfect_count(result: RunResult) -> int:
    return sum(
        1 for d in result.documents
        if not d.error and d.fields_total > 0 and d.accuracy == 1.0
    )


# ── Sections ──────────────────────────────────────────────────────────────────

def _section_header(result: RunResult) -> str:
    perfect = _perfect_count(result)
    cost_line = ""
    if result.avg_cost_per_doc_usd is not None:
        cost_line = (
            f"| Avg cost/doc | **${result.avg_cost_per_doc_usd:.4f}** "
            f"(${result.total_cost_usd:.4f} total) |\n"
        )

    return f"""# doceval — Extraction Accuracy Report

**Run:** `{result.run_id}`
**Extractor:** `{result.extractor_name}`
**Documents:** {result.total_documents} ({result.successful} processed, {result.failed} errors)

---

## Summary

| Metric | Value |
|--------|-------|
| Field accuracy | **{_pct(result.fields_correct, result.fields_total)}** ({result.fields_correct}/{result.fields_total} fields) |
| Perfect documents | **{perfect}/{result.total_documents}** ({_pct(perfect, result.total_documents)}) |
| Failed extractions | {result.failed} |
{cost_line}
"""


def _section_failure_modes(result: RunResult) -> str:
    if not result.failure_modes:
        return ""

    total_mismatches = sum(result.failure_modes.values())
    lines = ["## Failure Modes\n"]
    lines.append("How mismatches break down across all fields and documents:\n")
    lines.append("| Failure Mode | Count | Share | Description |")
    lines.append("|--------------|-------|-------|-------------|")

    descriptions = {
        "missed_field":  "Label has a value; extractor returned empty",
        "hallucination": "Extractor returned a value; label is empty",
        "wrong_format":  "Both non-empty; numeric/date values differ",
        "wrong_value":   "Both non-empty; string values differ",
    }
    for mode in ("missed_field", "hallucination", "wrong_format", "wrong_value"):
        n = result.failure_modes.get(mode, 0)
        if n == 0:
            continue
        share = f"{n / total_mismatches * 100:.1f}%" if total_mismatches else "n/a"
        desc = descriptions.get(mode, "")
        lines.append(f"| `{mode}` | {n} | {share} | {desc} |")

    return "\n".join(lines) + "\n\n"


def _section_field_accuracy(result: RunResult) -> str:
    if not result.by_field:
        return ""

    lines = ["## Field-Level Accuracy\n"]
    lines.append("| Field | Accuracy | Correct/Total | Bar |")
    lines.append("|-------|----------|---------------|-----|")

    sorted_fields = sorted(
        result.by_field.items(),
        key=lambda x: -(x[1].correct / max(x[1].total, 1)),
    )
    for field, stats in sorted_fields:
        bar = _bar(stats.correct, stats.total)
        lines.append(
            f"| `{field}` | **{_pct(stats.correct, stats.total)}** "
            f"| {stats.correct}/{stats.total} | `{bar}` |"
        )

    # Mismatch examples
    failing = [(f, s) for f, s in sorted_fields if s.correct < s.total]
    if failing:
        lines.append("\n### Mismatch Examples\n")
        for field, stats in failing:
            lines.append(
                f"\n**`{field}`** — {_pct(stats.correct, stats.total)} "
                f"({stats.correct}/{stats.total})\n"
            )
            for ex in stats.examples:
                mode_tag = f" `[{ex['failure_mode']}]`" if ex.get("failure_mode") else ""
                lines.append(f"- `{ex['doc']}`{mode_tag}")
                lines.append(f"  - Expected: `{ex['expected']}`")
                lines.append(f"  - Actual:   `{ex['actual']}`")

    return "\n".join(lines) + "\n\n"


def _section_hardest_docs(result: RunResult, n: int = 5) -> str:
    successful = [
        d for d in result.documents
        if not d.error and d.fields_total > 0
    ]
    if not successful:
        return ""

    hardest = sorted(successful, key=lambda d: d.accuracy)[:n]
    lines = ["## Hardest Documents\n"]
    lines.append(f"The {min(n, len(hardest))} documents with the lowest field accuracy:\n")

    for doc in hardest:
        meta_parts = [f"{k}: {v}" for k, v in doc.metadata.items()] if doc.metadata else []
        meta_str = f"  \n- {', '.join(meta_parts)}" if meta_parts else ""
        lines.append(f"### `{doc.filename}` — {_pct(doc.fields_correct, doc.fields_total)}")
        lines.append(f"- Fields: {doc.fields_correct}/{doc.fields_total}{meta_str}\n")
        wrong = [f for f in doc.fields if not f.match]
        if wrong:
            lines.append("| Field | Expected | Actual | Failure Mode |")
            lines.append("|-------|----------|--------|--------------|")
            for f in wrong:
                lines.append(
                    f"| `{f.field}` | `{f.expected}` | `{f.actual}` | `{f.failure_mode}` |"
                )
        lines.append("")

    return "\n".join(lines) + "\n"


def _section_errors(result: RunResult) -> str:
    extraction_errors = [e for e in result.errors if "No matching" not in e["error"]]
    if not extraction_errors:
        return ""
    lines = ["## Extraction Errors\n"]
    lines.append("Documents where the extractor raised an exception:\n")
    for err in extraction_errors:
        lines.append(f"- `{err['filename']}`: {err['error']}")
    return "\n".join(lines) + "\n\n"


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_report(result: RunResult) -> str:
    """Return a Markdown report string for a RunResult."""
    sections = [
        _section_header(result),
        _section_failure_modes(result),
        _section_field_accuracy(result),
        _section_hardest_docs(result),
        _section_errors(result),
    ]
    return "".join(sections)
