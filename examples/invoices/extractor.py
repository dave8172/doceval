"""
Example extractor for the invoices dataset.

Uses Claude claude-haiku-4-5-20251001 to extract key fields from invoice PDFs and returns
(result_dict, cost_usd) so doceval can report per-document cost.

Requirements:
    pip install anthropic python-dotenv
    export ANTHROPIC_API_KEY=sk-ant-...

Usage with doceval:
    doceval run \\
        --docs    examples/invoices/docs \\
        --labels  examples/invoices/labels \\
        --extractor examples.invoices.extractor:extract \\
        --name "claude-haiku invoice extractor"
"""

from __future__ import annotations

import base64
import json
import os

import anthropic

# Haiku is fast and cheap — good for eval runs on many documents
MODEL = "claude-haiku-4-5-20251001"

# Haiku pricing (USD per million tokens) as of mid-2026
INPUT_COST_PER_M  = 0.80
OUTPUT_COST_PER_M = 4.00

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client


SYSTEM_PROMPT = """You extract structured data from invoice documents.
Return a single JSON object with these fields (use null for missing fields):
  vendor, document_number, date, ship_mode, order_id, currency,
  subtotal, discount, shipping, total

Rules:
- Dates in ISO format: YYYY-MM-DD
- Amounts as strings preserving the original (e.g. "748.36")
- Return only the JSON object, no explanation."""


def extract(doc_bytes: bytes, filepath: str) -> tuple[dict, float]:
    """
    Extract invoice fields from doc_bytes.
    Returns (fields_dict, cost_usd).
    """
    b64 = base64.standard_b64encode(doc_bytes).decode()

    response = _get_client().messages.create(
        model=MODEL,
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": b64,
                    },
                },
                {"type": "text", "text": "Extract the invoice fields."},
            ],
        }],
    )

    # Cost calculation
    usage = response.usage
    cost = (
        usage.input_tokens  / 1_000_000 * INPUT_COST_PER_M
        + usage.output_tokens / 1_000_000 * OUTPUT_COST_PER_M
    )

    raw_text = response.content[0].text.strip()
    # Strip markdown code fences if present
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]

    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError:
        result = {"_parse_error": raw_text}

    return result, round(cost, 6)
