"""
Example extractor for the leads dataset — a non-document input.

Same harness, same scoring, same report; the input is an email rather than a PDF.
It exists to make one thing concrete: doceval measures field-level extraction
correctness, and that never depended on the source being a document.

Uses Claude claude-haiku-4-5-20251001 and returns (result_dict, cost_usd) so
doceval can report per-document cost.

Requirements:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...

Usage with doceval:
    doceval run \\
        --docs    examples/leads/docs \\
        --labels  examples/leads/labels \\
        --extractor examples.leads.extractor:extract \\
        --name "claude-haiku lead extractor"
"""

from __future__ import annotations

import json
import os

import anthropic

MODEL = "claude-haiku-4-5-20251001"

# Haiku pricing (USD per million tokens) as of mid-2026
INPUT_COST_PER_M = 0.80
OUTPUT_COST_PER_M = 4.00

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client


# The "leave it blank rather than guessing" rule is deliberate: several of the
# bundled emails are not live enquiries, and an extractor that invents a
# quantity for them should score worse, not better.
SYSTEM_PROMPT = """You extract structured lead data from inbound sales emails.
Return a single JSON object with these fields (use null for anything the email
does not state):
  company, contact_name, email, phone, product, quantity, target_date, budget

Rules:
- target_date in ISO format: YYYY-MM-DD. Resolve relative dates against the
  email's own Date header.
- quantity as a bare number string, e.g. "250".
- budget as amount + currency code, e.g. "4000 GBP".
- Use the originating sender on a forwarded thread, not the forwarder.
- Translate product descriptions into English.
- Never guess. If the email does not state a field, return null for it.
- Return only the JSON object, no explanation."""


def extract(doc_bytes: bytes, filepath: str) -> tuple[dict, float]:
    """
    Extract lead fields from an email's raw bytes.
    Returns (fields_dict, cost_usd).
    """
    email_text = doc_bytes.decode("utf-8", errors="replace")

    response = _get_client().messages.create(
        model=MODEL,
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Extract the lead fields from this email.\n\n{email_text}",
        }],
    )

    usage = response.usage
    cost = (
        usage.input_tokens / 1_000_000 * INPUT_COST_PER_M
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
