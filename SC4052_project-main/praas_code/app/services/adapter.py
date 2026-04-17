"""
Multi-Model Adapter microservice.

Given an optimised prompt, produce three prompt variants aligned with the
documented formatting conventions of different model families:

  - Claude  : XML-structured, constraint tags as first-class citizens
  - GPT     : markdown-sectioned, explicit role assignment at the top
  - Gemini  : enumerated step-by-step instructions

The Adapter does not claim to produce the globally optimal prompt for each
family. Instead, it generates model-family-aware variants that preserve the
original task semantics while changing structure and presentation.

Actual text generation is delegated to app.inference.generate(...), which may
route to a real backend (e.g. Ollama / Hugging Face) or a controlled fallback
backend, depending on runtime configuration.

Design note: variants are produced serially rather than concurrently. A
single local Ollama worker (the default configuration) would serialise
concurrent requests anyway, so apparent parallelism via asyncio.gather
would buy nothing and would obscure per-variant latency during debugging.
If a multi-worker backend is deployed (OLLAMA_NUM_PARALLEL>=3 or a hosted
API), parallelising this loop is a safe local change.
"""

from __future__ import annotations

from app import inference
from app.schemas import (
    AdaptRequest, AdaptResponse, AdaptedVariant, ModelFamily,
)


FAMILY_CONFIG = {
    ModelFamily.CLAUDE: {
        "style": "XML-structured",
        "note": (
            "Claude-class prompts typically benefit from XML-like structure, "
            "with task and constraints clearly separated."
        ),
    },
    ModelFamily.GPT: {
        "style": "markdown-sectioned",
        "note": (
            "GPT-class prompts typically benefit from markdown headings and "
            "an explicit role/task layout."
        ),
    },
    ModelFamily.GEMINI: {
        "style": "enumerated-steps",
        "note": (
            "Gemini-class prompts typically benefit from numbered instructions "
            "and explicit step-by-step structure."
        ),
    },
}


ADAPT_META = """You are adapting a prompt to the formatting conventions of a
specific model family.

TARGET FAMILY: {family}
TARGET STYLE: {style}
FAMILY NOTES: {note}

ORIGINAL PROMPT:
\"\"\"
{prompt}
\"\"\"

Rewrite the prompt in the target style.

Requirements:
- Preserve the original task and all requirements.
- Do not invent new task requirements.
- You may reorganise structure, headings, tags, and formatting.
- Return only the adapted prompt.
- Do not include explanations or commentary.
"""


async def adapt(req: AdaptRequest) -> AdaptResponse:
    variants: list[AdaptedVariant] = []

    for family in req.families:
        cfg = FAMILY_CONFIG[family]

        meta = ADAPT_META.format(
            family=family.value,
            style=cfg["style"],
            note=cfg["note"],
            prompt=req.prompt,
        )

        adapted = (
            await inference.generate(
                meta,
                max_new_tokens=512,
                temperature=0.2,
            )
        ).strip()

        if not adapted or len(adapted) < 10:
            adapted = _fallback_adapt(req.prompt, family)

        variants.append(
            AdaptedVariant(
                family=family,
                prompt=adapted,
                notes=cfg["note"],
            )
        )

    return AdaptResponse(variants=variants)


def _fallback_adapt(prompt: str, family: ModelFamily) -> str:
    """
    Deterministic structural fallback used only if generation fails
    or returns an implausibly short result. Keeps the pipeline
    runnable even when the inference backend misbehaves.
    """
    if family == ModelFamily.CLAUDE:
        return (
            f"<task>\n{prompt}\n</task>\n"
            f"<constraints>\n"
            f"  Preserve all requirements in the original prompt.\n"
            f"</constraints>"
        )

    if family == ModelFamily.GPT:
        return (
            f"## Role\nYou are a helpful assistant.\n\n"
            f"## Task\n{prompt}\n\n"
            f"## Constraints\n"
            f"- Preserve all requirements in the original prompt."
        )

    if family == ModelFamily.GEMINI:
        return (
            f"Follow these steps:\n"
            f"1. Read the request carefully.\n"
            f"2. Complete this task: {prompt}\n"
            f"3. Produce the requested output.\n"
            f"4. Check that all requirements are satisfied."
        )

    return prompt
