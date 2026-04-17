"""
Analyzer microservice.

Diagnoses a raw prompt on seven dimensions derived from the prompt-engineering
literature:

  1. role              — is a persona or narrator role specified?
  2. audience          — is the intended audience explicit?
  3. task_specificity  — are sub-goals and constraints nailed down?
  4. input_format      — does the prompt say what input to expect?
  5. output_format     — does the prompt say what output to produce?
  6. length_constraint — is there a length / word budget?
  7. tone_constraint   — is the tone / register specified?

The service returns a structured record; it explicitly does NOT rewrite.
That separation mirrors compiler front-end / back-end and lets the
Optimizer be swapped without disturbing the diagnostic contract.
"""

from __future__ import annotations

import hashlib
from typing import Any

from app import inference
from app.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    DimensionDiagnostic,
    DimensionStatus,
)


SEVEN_DIMENSIONS = [
    "role",
    "audience",
    "task_specificity",
    "input_format",
    "output_format",
    "length_constraint",
    "tone_constraint",
]


META_PROMPT = """You are a senior prompt engineer auditing a user's prompt.

Diagnose the following prompt on SEVEN dimensions. For each dimension,
output one of: "present", "partial", or "missing", plus a one-sentence
justification.

DIMENSIONS:
  role, audience, task_specificity, input_format, output_format,
  length_constraint, tone_constraint

USER PROMPT:
\"\"\"
{prompt}
\"\"\"

TASK DESCRIPTION (user's true intent, may be empty):
\"\"\"
{task_description}
\"\"\"

Respond ONLY with a single JSON object, no prose, no code fences, of the form:
{{
  "dimensions": {{
    "role": {{"status": "missing", "justification": "..."}},
    "audience": {{"status": "partial", "justification": "..."}}
  }},
  "summary": "one-sentence summary of the prompt's biggest weakness"
}}
"""


def _normalise_status(value: Any) -> DimensionStatus:
    """Tolerate casing and trailing punctuation in the model's status string."""
    text = str(value).lower().strip().rstrip(".!,;:")
    if text in {"present", "partial", "missing"}:
        return DimensionStatus(text)
    return DimensionStatus.MISSING


def _normalise_dimension_entry(entry: Any) -> DimensionDiagnostic:
    """
    Accepts either:
    1. {"status": "...", "justification": "..."}
    2. "present" / "partial" / "missing"
    3. anything malformed -> safe fallback
    """
    if isinstance(entry, dict):
        status = _normalise_status(entry.get("status", "missing"))
        justification = str(entry.get("justification", "Not assessed.")).strip()
        if not justification:
            justification = "Not assessed."
        return DimensionDiagnostic(
            status=status,
            justification=justification,
        )

    if isinstance(entry, str):
        status = _normalise_status(entry)
        return DimensionDiagnostic(
            status=status,
            justification="Model returned an unstructured status string.",
        )

    return DimensionDiagnostic(
        status=DimensionStatus.MISSING,
        justification="Not assessed.",
    )


async def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    prompt_hash = hashlib.sha256(req.prompt.encode("utf-8")).hexdigest()[:16]

    meta = META_PROMPT.format(
        prompt=req.prompt,
        task_description=req.task_description or "",
    )

    raw = await inference.generate_json(meta, max_new_tokens=512)

    dims_raw = raw.get("dimensions", {}) or {}
    if not isinstance(dims_raw, dict):
        dims_raw = {}

    dims: dict[str, DimensionDiagnostic] = {}

    for name in SEVEN_DIMENSIONS:
        entry = dims_raw.get(name, {})
        dims[name] = _normalise_dimension_entry(entry)

    present = sum(1 for d in dims.values() if d.status == DimensionStatus.PRESENT)
    partial = sum(1 for d in dims.values() if d.status == DimensionStatus.PARTIAL)
    quality = (present + 0.5 * partial) / len(SEVEN_DIMENSIONS)

    summary = raw.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        missing_dims = [
            n for n, d in dims.items()
            if d.status != DimensionStatus.PRESENT
        ]
        summary = "Prompt is under-specified on " + ", ".join(missing_dims) + "."

    return AnalyzeResponse(
        prompt_hash=prompt_hash,
        dimensions=dims,
        overall_quality=round(quality, 3),
        summary=summary.strip(),
    )
