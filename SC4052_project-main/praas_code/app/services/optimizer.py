"""
Optimizer microservice.

Implements a simplified TextGrad-style two-call update:

  Step 1 (gradient) — generate a natural-language description of what,
                      specifically, should change in the prompt.
  Step 2 (update)   — apply the gradient to produce a new prompt, with
                      an intent-preservation guard that prepends the
                      original task description to the context.

We cap the default loop at one iteration for latency; the client may
request up to three. When multiple iterations are requested, every
iteration's gradient is preserved in the response (joined by a separator)
so the full reasoning trail is auditable.
"""

from __future__ import annotations

from app import inference
from app.schemas import (
    OptimizeRequest,
    OptimizeResponse,
    AnalyzeRequest,
    AnalyzeResponse,
)
from app.services.analyzer import analyze


GRADIENT_META = """You are improving a user's prompt to a large language model.

Given the prompt below and its diagnostic summary, describe CONCRETELY
how the prompt should be improved. Do not rewrite the prompt. Only list the
specific changes that should be made.

ORIGINAL PROMPT:
\"\"\"
{prompt}
\"\"\"

TASK DESCRIPTION:
\"\"\"
{task_description}
\"\"\"

DIAGNOSTIC SUMMARY:
\"\"\"
{diagnostic_summary}
\"\"\"

Produce the gradient as a short bullet-style list of improvements only.
"""


UPDATE_META = """You are rewriting a user's prompt based on an improvement gradient.

Apply the gradient below to the original prompt. Produce ONLY the improved
prompt. Do not include explanation or commentary. Preserve the user's
original intent as expressed in the task description.

ORIGINAL PROMPT:
\"\"\"
{prompt}
\"\"\"

TASK DESCRIPTION (this is the user's true intent — preserve it):
\"\"\"
{task_description}
\"\"\"

GRADIENT (what to change):
\"\"\"
{gradient}
\"\"\"

Now apply the gradient and rewrite the prompt. Respond with the rewritten
prompt only.
"""


ITERATION_SEPARATOR = "\n\n--- iteration boundary ---\n\n"


async def optimize(req: OptimizeRequest) -> OptimizeResponse:
    # Ensure we have a diagnostic for the gradient step.
    if req.diagnostic is not None:
        diagnostic: AnalyzeResponse = req.diagnostic
    else:
        diagnostic = await analyze(
            AnalyzeRequest(
                prompt=req.prompt,
                task_description=req.task_description,
            )
        )

    current_prompt = req.prompt
    gradients: list[str] = []
    iterations_run = 0

    requested_iterations = max(1, min(req.iterations, 3))

    for _ in range(requested_iterations):
        grad_prompt = GRADIENT_META.format(
            prompt=current_prompt,
            task_description=req.task_description or "",
            diagnostic_summary=diagnostic.summary,
        )

        gradient_text = (
            await inference.generate(
                grad_prompt,
                max_new_tokens=256,
                temperature=0.2,
            )
        ).strip()

        gradients.append(gradient_text)

        upd_prompt = UPDATE_META.format(
            prompt=current_prompt,
            task_description=req.task_description or "",
            gradient=gradient_text,
        )

        updated = (
            await inference.generate(
                upd_prompt,
                max_new_tokens=512,
                temperature=0.3,
            )
        ).strip()

        if updated and len(updated) > 10:
            current_prompt = updated

        iterations_run += 1

    # Preserve the full gradient trail while keeping the schema
    # field single-string compatible: iterations are separated by a
    # clearly labelled divider.
    combined_gradient = (
        ITERATION_SEPARATOR.join(gradients)
        if len(gradients) > 1
        else (gradients[0] if gradients else "")
    )

    return OptimizeResponse(
        original_prompt=req.prompt,
        optimised_prompt=current_prompt,
        gradient=combined_gradient,
        iterations_run=iterations_run,
    )
