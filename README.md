# doceval

**Eval harness for document extraction pipelines.**

Point it at your extractor + a labeled dataset. Get back field-level accuracy, a failure taxonomy, and optional cost tracking — without writing any eval infrastructure yourself.

Works with any extraction function (Claude, GPT, regex, rules) and any document schema.

---

## The problem

You've built an LLM-based document extractor. It seems to work. But:

- How accurate is it, actually?
- Which fields fail most? Why — wrong value, or the field was missed entirely?
- Did accuracy change when you updated the prompt?
- How much does each extraction cost?

Without answers, "seems to work" is all you have. That's not good enough for production.

---

## Try it in 60 seconds

No API key setup, no writing an extractor — this runs a real Claude-based invoice
extractor against 20 bundled sample PDFs and prints a full accuracy report.

```bash
git clone https://github.com/dave8172/doceval
cd doceval
pip install -e ".[examples]"
export ANTHROPIC_API_KEY=sk-ant-...

doceval run \
  --docs    examples/invoices/docs \
  --labels  examples/invoices/labels \
  --extractor examples.invoices.extractor:extract \
  --name "claude-haiku invoice extractor"
```

Output:

```
doceval run
  docs:      examples/invoices/docs
  labels:    examples/invoices/labels
  extractor: claude-haiku invoice extractor
  documents: 20 found

[  1/20] invoice_Shahid Shariari_30140.pdf  ... 9/9 (100%)  1.3s  $0.0009
[  2/20] invoice_Shaun Weien_31134.pdf      ... 10/10 (100%) 1.1s  $0.0008
[  3/20] invoice_Sheri Gordon_1260.pdf      ... 8/9 (89%)   1.4s  $0.0009
...

==================================================
Overall: 172/180 fields correct (95.6%)
Successful: 20  Failed: 0

Failure modes:
  missed_field    : 5
  wrong_format    : 3

Cost: $0.0009/doc avg  ($0.0178 total)

Report → eval-report-2026-07-01T14-22-10.md
```

The markdown report includes a field-level accuracy table, mismatch examples with
failure mode tags, and the hardest documents ranked by accuracy. Open
`examples/invoices/extractor.py` to see exactly what the extractor function looks like —
it's the same shape you'll write for your own pipeline below.

---

## Use it on your own documents

```bash
pip install doceval
```

Write an extractor — a Python function that takes `(doc_bytes, filepath)` and returns a dict:

```python
# my_extractor.py
import anthropic, base64, json

client = anthropic.Anthropic()

def extract(doc_bytes: bytes, filepath: str) -> dict:
    b64 = base64.standard_b64encode(doc_bytes).decode()
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": [
                {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": b64}},
                {"type": "text", "text": "Extract: vendor, date, total, invoice_number. Return JSON."},
            ],
        }],
    )
    return json.loads(response.content[0].text)
```

Add a label file for each document (`labels/invoice_001.json`):

```json
{
  "vendor": "Acme Corp",
  "date": "2026-01-15",
  "total": "1234.56",
  "invoice_number": "INV-001"
}
```

Run the eval:

```bash
doceval run \
  --docs    ./dataset/docs \
  --labels  ./dataset/labels \
  --extractor my_extractor:extract
```

This prints the same kind of report shown above, scored against your own labels.

---

## Optional: report cost per document

Return a `(dict, cost_usd)` tuple from your extractor and doceval tracks cost automatically:

```python
def extract(doc_bytes: bytes, filepath: str) -> tuple[dict, float]:
    response = client.messages.create(...)
    usage = response.usage
    cost = usage.input_tokens / 1e6 * 0.80 + usage.output_tokens / 1e6 * 4.00
    return json.loads(response.content[0].text), cost
```

---

## Label format

One JSON file per document, named `{document_stem}.json`. Values are strings (or numbers — doceval normalizes both).

```json
{
  "vendor": "Acme Corp",
  "date": "2026-01-15",
  "total": "1234.56",
  "invoice_number": "INV-001"
}
```

Optional `__meta__` key passes through to the report (e.g., difficulty, document type) but is not compared:

```json
{
  "vendor": "Acme Corp",
  "total": "1234.56",
  "__meta__": { "difficulty": "hard", "doc_type": "scanned_invoice" }
}
```

Nested dicts are supported and flattened with dot notation: `{"address": {"city": "NY"}}` → field `address.city`.

**Manifest alternative:** for large datasets, hand-authoring one JSON file per document
doesn't scale. Point `--labels` at a single `.csv` or `.jsonl`/`.ndjson` file instead —
each row/line needs a `filename` column/key identifying the document (matched by stem;
the extension is optional):

```csv
filename,vendor,date,total,invoice_number
invoice_001,Acme Corp,2026-01-15,1234.56,INV-001
invoice_002,Beta LLC,2026-01-20,540.00,INV-002
```

```bash
doceval run --docs ./dataset/docs --labels ./labels.csv --extractor my_extractor:extract
```

JSONL supports the same nesting and `__meta__` passthrough as per-file JSON labels; CSV
rows are flat by nature.

---

## Failure mode taxonomy

Every mismatch is classified:

| Mode | Meaning |
|------|---------|
| `missed_field` | Label has a value; extractor returned empty |
| `hallucination` | Extractor returned a value; label is empty |
| `wrong_format` | Both non-empty; numeric or date values differ |
| `wrong_value` | Both non-empty; string values differ |

doceval handles numeric normalization (`$1,234.56` = `1234.56` = `1.234,56`) and date normalization (`Nov 15 2012` = `2012-11-15`) before comparison.

---

## CLI reference

```bash
doceval run --docs DIR --labels DIR --extractor MODULE:FUNC [OPTIONS]
```

| Flag | Required | Default | Meaning |
|------|----------|---------|---------|
| `--docs` | yes | — | Directory of documents to evaluate. Scanned recursively. |
| `--labels` | yes | — | Directory of label JSON files, one per document (see [Label format](#label-format)). |
| `--extractor` | yes | — | Extractor function as `module:function` (importable from cwd). |
| `--name` | no | value of `--extractor` | Display name for the extractor in reports. |
| `--output` | no | `eval-report-<run_id>.md` in cwd | Path for the generated Markdown report. |
| `--json-out` | no | not written | Path to also write the full result as JSON (see [Output](#output) below). |
| `--quiet` | no | off | Suppress per-document progress lines; only the final summary prints. |

Only documents that have a matching label file (same stem) are scored. Unmatched
documents or labels are listed as errors in the report rather than silently skipped.
The process exits non-zero if any extraction raised an error, so `doceval run` is
safe to use as a CI gate.

---

## Output

Every run produces:

1. **Console summary** — overall accuracy, failure mode counts, cost (if reported), always printed unless `--quiet`.
2. **Markdown report** — written to `--output` (or an auto-named `eval-report-<timestamp>.md`). Includes a field-level accuracy table, up to 3 mismatch examples per field tagged with failure mode, and documents ranked by accuracy.
3. **JSON report** *(optional, `--json-out`)* — the full result, useful for tracking accuracy over time or feeding a dashboard. Top-level shape:

```json
{
  "run_id": "2026-07-01T14-22-10",
  "extractor_name": "claude-haiku invoice extractor",
  "total_documents": 20,
  "successful": 20,
  "failed": 0,
  "fields_correct": 172,
  "fields_total": 180,
  "overall_accuracy": 0.9556,
  "failure_modes": { "missed_field": 5, "wrong_format": 3 },
  "by_field": {
    "vendor": { "correct": 20, "total": 20, "failure_modes": {}, "examples": [] },
    "total":  { "correct": 18, "total": 20, "failure_modes": { "wrong_format": 2 }, "examples": [ /* up to 3 */ ] }
  },
  "total_cost_usd": 0.0178,
  "avg_cost_per_doc_usd": 0.0009,
  "documents": [ /* per-document DocResult: filename, fields[], accuracy, duration_s, cost_usd, error */ ],
  "errors": [ /* {"filename": ..., "error": ...} for unmatched docs/labels or extractor exceptions */ ]
}
```

The same structure is what `run_eval()` returns as a `RunResult` object in the
programmatic API below (as a dataclass, not JSON — access fields directly, e.g.
`result.overall_accuracy`).

doceval also logs unpaired docs/labels and extractor failures through the stdlib
`logging` module (logger name `"doceval"`), independent of the console/report output.
Call `logging.basicConfig()` in your own code to see them.

---

## Programmatic API

```python
from doceval import run_eval, generate_report
from pathlib import Path

result = run_eval(
    docs_dir="./dataset/docs",
    labels_dir="./dataset/labels",
    extract_fn=my_extract_fn,
    extractor_name="my extractor v2",
)

print(f"Overall accuracy: {result.overall_accuracy:.1%}")
print(f"Failure modes: {result.failure_modes}")

Path("report.md").write_text(generate_report(result))
```

---

## Supported document types

PDF, PNG, JPG, JPEG, TIFF, WEBP

---

## Why this exists

Shipping an LLM extraction pipeline without eval infrastructure is building on sand. Field-level accuracy by document type, a failure taxonomy, and cost visibility are the minimum bar before calling something production-ready.

This tool exists so you don't have to build that infrastructure from scratch.

---

## License

MIT
