from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FieldResult:
    field: str
    expected: str
    actual: str
    match: bool
    # None when match=True; one of: missed_field, hallucination, wrong_format, wrong_value
    failure_mode: str | None


@dataclass
class DocResult:
    filename: str
    fields: list[FieldResult]
    fields_correct: int
    fields_total: int
    accuracy: float
    duration_s: float
    # None when extractor does not report cost
    cost_usd: float | None
    error: str | None
    # Anything in the label file under the optional "__meta__" key
    metadata: dict = field(default_factory=dict)


@dataclass
class FieldStats:
    correct: int
    total: int
    failure_modes: dict[str, int]  # mode → count
    # Up to 3 example mismatches for the report
    examples: list[dict]

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


@dataclass
class RunResult:
    run_id: str
    extractor_name: str
    total_documents: int
    successful: int
    failed: int
    fields_correct: int
    fields_total: int
    overall_accuracy: float
    # Aggregated failure mode counts across all mismatches
    failure_modes: dict[str, int]
    # Per-field stats
    by_field: dict[str, FieldStats]
    # Cost totals — None when extractor does not report cost
    total_cost_usd: float | None
    avg_cost_per_doc_usd: float | None
    documents: list[DocResult]
    errors: list[dict]  # [{"filename": ..., "error": ...}]
