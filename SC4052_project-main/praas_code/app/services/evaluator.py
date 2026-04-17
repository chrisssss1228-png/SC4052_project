"""
Evaluator microservice.

Runs each adapted variant against its target model (or a shared substitute
model when the target API is unavailable) and scores the outputs against
a task-specific rubric using an LLM-as-judge.

Three guards against known judge failure modes:

  1. Position-bias mitigation via randomised ordering: the order in which
     variants are presented to the judge is shuffled per call, and the
     mapping is reversed when we decode the judge's response.
  2. Rubric-based scoring rather than free-form preference: the judge is
     constrained to score each output on completion / compliance / structure.
  3. Tolerant key decoding: real LLM judges often emit key variants such
     as "Output A", "a", or "variant_a" instead of the exact "A" we ask
     for. We accept any of these rather than silently falling back to
     default 3.0s. Winner decoding, by contrast, is intentionally strict
     — we only parse the "winner" field, and only accept a standalone
     A-D letter inside it — to avoid the risk of picking up an unrelated
     letter from a verbose explanation.
"""

from __future__ import annotations

import random
import re
from typing import Any

from app import inference
from app.schemas import (
    EvaluateRequest, EvaluateResponse,
    VariantScore,
)


JUDGE_META = """You are evaluating multiple outputs produced for the same task.

TASK DESCRIPTION:
\"\"\"
{task_description}
\"\"\"

RUBRIC (score each output on each axis from 0 to 5):
  - completion : does it accomplish the stated task?
  - compliance : does it respect stated constraints (length, tone, format)?
  - structure  : is it well-organised and easy to follow?

OUTPUTS:

[A]
\"\"\"
{output_a}
\"\"\"

[B]
\"\"\"
{output_b}
\"\"\"

[C]
\"\"\"
{output_c}
\"\"\"

Respond ONLY with a JSON object of the form:
{{
  "A": {{"completion": 4.0, "compliance": 3.5, "structure": 4.0}},
  "B": {{"completion": 3.5, "compliance": 3.0, "structure": 3.5}},
  "C": {{"completion": 4.5, "compliance": 4.0, "structure": 4.0}},
  "winner": "A",
  "explanation": "one-sentence reason the winner beats the others"
}}
"""


def _find_rubric(raw: dict, slot: str) -> dict:
    """
    Look up the rubric block for a slot letter, accepting the key variants
    real LLM judges tend to produce:
      - "A", "a"
      - "Output A", "output_a"
      - "variant_a", "Variant A"
      - "option_a", "Option A"
    Returns an empty dict if no plausible match is found.
    """
    if not isinstance(raw, dict):
        return {}

    lower = slot.lower()
    candidates = [
        slot,
        lower,
        f"Output {slot}", f"output_{lower}", f"output {lower}",
        f"Variant {slot}", f"variant_{lower}", f"variant {lower}",
        f"Option {slot}", f"option_{lower}", f"option {lower}",
        f"[{slot}]", f"[{lower}]",
    ]
    for key in candidates:
        if key in raw and isinstance(raw[key], dict):
            return raw[key]
    return {}


def _parse_winner(raw: dict, valid_slots: set[str]) -> str | None:
    """
    Extract the winner slot letter from the judge's `winner` field only.
    We require a standalone A-D letter (no adjacent alphabetic characters),
    which correctly handles:
      - "A", "a"
      - "Output A."   → A
      - "The winner is C"   → C
    And correctly refuses:
      - "OutputA" (no boundary) → None, fall back to max score
      - A stray letter inside an explanation field

    Returns one of valid_slots, or None if nothing plausible was found.
    """
    winner_raw = raw.get("winner")
    if not isinstance(winner_raw, str):
        return None

    # First isolated A/B/C/D letter, not flanked by other letters.
    m = re.search(r"(?<![A-Za-z])([A-Da-d])(?![A-Za-z])", winner_raw)
    if not m:
        return None

    letter = m.group(1).upper()
    return letter if letter in valid_slots else None


def _safe_float(value: Any, default: float = 3.0) -> float:
    """Coerce judge numeric fields that may arrive as ints, strs, or junk."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    # Clip to rubric range.
    return max(0.0, min(5.0, f))


async def evaluate(req: EvaluateRequest) -> EvaluateResponse:
    if not req.variants:
        raise ValueError("EvaluateRequest.variants must be non-empty.")

    # 1. Produce one output from each adapted variant.
    outputs: list[str] = []
    for variant in req.variants:
        out = await inference.generate(
            variant.prompt,
            max_new_tokens=400,
            temperature=0.5,
        )
        outputs.append(out.strip())

    # 2. Randomise order before sending outputs to the judge.
    order = list(range(len(req.variants)))
    random.shuffle(order)

    slot_labels = ["A", "B", "C", "D"][:len(order)]
    slot_to_variant_idx = {
        slot: order[i] for i, slot in enumerate(slot_labels)
    }
    variant_idx_to_slot = {
        variant_idx: slot for slot, variant_idx in slot_to_variant_idx.items()
    }

    # Stable template: support up to A/B/C in the current judge prompt.
    a = outputs[slot_to_variant_idx["A"]] if "A" in slot_to_variant_idx else ""
    b = outputs[slot_to_variant_idx["B"]] if "B" in slot_to_variant_idx else ""
    c = outputs[slot_to_variant_idx["C"]] if "C" in slot_to_variant_idx else ""

    meta = JUDGE_META.format(
        task_description=req.task_description or "(none provided)",
        output_a=a,
        output_b=b,
        output_c=c,
    )

    raw = await inference.generate_json(meta, max_new_tokens=400)
    if not isinstance(raw, dict):
        raw = {}

    # 3. Decode judge response and map slot labels back to original variants.
    scores: list[VariantScore] = []
    for i, variant in enumerate(req.variants):
        slot = variant_idx_to_slot.get(i)
        rubric_raw = _find_rubric(raw, slot) if slot else {}

        completion = _safe_float(rubric_raw.get("completion"), default=3.0)
        compliance = _safe_float(rubric_raw.get("compliance"), default=3.0)
        structure  = _safe_float(rubric_raw.get("structure"),  default=3.0)

        breakdown = {
            "completion": completion,
            "compliance": compliance,
            "structure":  structure,
        }
        total = (completion + compliance + structure) / 3.0

        scores.append(
            VariantScore(
                family=variant.family,
                output=outputs[i],
                score=round(total, 2),
                rubric_breakdown=breakdown,
            )
        )

    # 4. Winner: decode the judge's slot choice (strict — only from the
    #    "winner" field, only a standalone A-D letter). Fall back to
    #    max aggregate score if parsing fails.
    winner_slot = _parse_winner(raw, set(slot_to_variant_idx.keys()))
    if winner_slot is not None:
        winner_family = req.variants[slot_to_variant_idx[winner_slot]].family
    else:
        winner_family = max(scores, key=lambda s: s.score).family

    explanation = raw.get(
        "explanation",
        "Winner chosen by highest aggregate rubric score.",
    )
    if not isinstance(explanation, str):
        explanation = str(explanation)

    return EvaluateResponse(
        scores=scores,
        winner=winner_family,
        explanation=explanation,
    )
