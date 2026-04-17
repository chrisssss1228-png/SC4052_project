"""
Inference backend abstraction for PraaS.

Three backends, tried in this order at call time:

  1. Local Ollama   (preferred when OLLAMA_MODEL is reachable; no key required)
  2. Hugging Face   (used when HF_TOKEN is set and Ollama is unavailable)
  3. Mock backend   (deterministic, offline; clearly tagged in its output)

Backend behaviour on error:
  - By default, a failing primary backend logs a warning to stderr and
    falls back to mock so the pipeline stays runnable for demos.
  - Set PRAAS_STRICT=1 to re-raise errors instead — recommended when
    running benchmarks, so that silent mock fallback cannot pollute the
    results.
"""

from __future__ import annotations

import json
import os
import re
import sys
import hashlib
from typing import Optional

import httpx


# ---------------------------------------------------------------------------
# Configuration (read once at module import; surface via /backend/status)
# ---------------------------------------------------------------------------

HF_MODEL   = os.environ.get("HF_MODEL", "HuggingFaceH4/zephyr-7b-beta")
HF_TOKEN   = os.environ.get("HF_TOKEN")
HF_ENDPOINT = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

OLLAMA_MODEL    = os.environ.get("OLLAMA_MODEL", "llama3.2")
OLLAMA_HOST     = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_ENDPOINT = f"{OLLAMA_HOST.rstrip('/')}/api/generate"

# Enable by default if the env var is not set to "0" or "false".
_USE_OLLAMA_ENV = os.environ.get("USE_OLLAMA", "1").lower()
USE_OLLAMA = _USE_OLLAMA_ENV not in {"0", "false", "no"}

# Strict mode — raise on backend errors instead of silently using mock.
# Use this when running benchmarks.
STRICT = os.environ.get("PRAAS_STRICT", "0").lower() in {"1", "true", "yes"}

# Ollama generation timeout. llama3.2 on CPU/M-series can easily spend
# 100+ seconds on a 400-token reply; 120s was a silent-failure trap.
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "300"))
HF_TIMEOUT     = float(os.environ.get("HF_TIMEOUT", "90"))


def _warn(msg: str) -> None:
    """Print a visible warning to stderr (benchmark logs will capture it)."""
    print(f"[praas.inference] WARN: {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def generate(prompt: str, *, max_new_tokens: int = 512,
                   temperature: float = 0.3) -> str:
    """
    Return a single text completion for `prompt`.

    Backend order: Ollama → HF → mock. Falls back only if the primary
    backend raises. In STRICT mode, the original exception is re-raised
    instead of being swallowed.
    """
    if USE_OLLAMA:
        try:
            return await _ollama_generate(prompt, max_new_tokens, temperature)
        except Exception as e:
            _warn(f"Ollama call failed: {type(e).__name__}: {e}")
            if STRICT:
                raise
            # Fall through to HF or mock.

    if HF_TOKEN:
        try:
            return await _hf_generate(prompt, max_new_tokens, temperature)
        except Exception as e:
            _warn(f"HF call failed: {type(e).__name__}: {e}")
            if STRICT:
                raise

    _warn("Falling back to MOCK backend — output will be tagged [mock-output#].")
    return _mock_generate(prompt)


async def generate_json(prompt: str, *, max_new_tokens: int = 512) -> dict:
    """
    Generate a response and parse it as JSON. Retries once with a stricter
    reformat instruction on parse failure.
    """
    raw = await generate(prompt, max_new_tokens=max_new_tokens,
                          temperature=0.1)
    parsed = _extract_json(raw)
    if parsed is not None:
        return parsed

    retry_prompt = (
        prompt
        + "\n\nYour previous response was not valid JSON. "
          "Respond ONLY with a single JSON object, no prose, no markdown fences."
    )
    raw = await generate(retry_prompt, max_new_tokens=max_new_tokens,
                          temperature=0.0)
    parsed = _extract_json(raw)
    if parsed is not None:
        return parsed

    # Last-resort: empty dict so callers don't crash. This is the hot path
    # for "judge produced garbage" — evaluator.py will apply its rubric
    # defaults and record a flaggable all-3.0 row.
    return {}


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

async def _ollama_generate(prompt: str, max_new_tokens: int,
                           temperature: float) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_new_tokens,
            "temperature": temperature,
        },
    }
    async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
        r = await client.post(OLLAMA_ENDPOINT, json=payload)
        r.raise_for_status()
        data = r.json()

    text = data.get("response", "")
    if not isinstance(text, str):
        raise ValueError(
            f"Ollama returned non-string 'response' field: {type(text).__name__}"
        )
    return text


# ---------------------------------------------------------------------------
# Hugging Face
# ---------------------------------------------------------------------------

async def _hf_generate(prompt: str, max_new_tokens: int,
                       temperature: float) -> str:
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "return_full_text": False,
        },
        "options": {"wait_for_model": True},
    }
    async with httpx.AsyncClient(timeout=HF_TIMEOUT) as client:
        r = await client.post(HF_ENDPOINT, json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()

    if isinstance(data, list) and data and "generated_text" in data[0]:
        return data[0]["generated_text"]
    if isinstance(data, dict) and "generated_text" in data:
        return data["generated_text"]
    return str(data)


# ---------------------------------------------------------------------------
# Mock backend — deterministic, offline, clearly tagged
# ---------------------------------------------------------------------------

def _mock_generate(prompt: str) -> str:
    """
    Produce a deterministic mock response. Every branch returns content
    that includes the literal token "[mock-output#...]" or "[mock]", so
    benchmark evidence files can be grepped for accidental mock pollution:

        grep -c 'mock-output' benchmark/evidence.json
        grep -c '\\[mock\\]'  benchmark/evidence.json
    """
    p = prompt.lower()
    seed = int(hashlib.md5(prompt.encode()).hexdigest(), 16) % 1000

    # Analyzer meta-prompt
    if "seven dimensions" in p or "diagnose" in p:
        return json.dumps({
            "dimensions": {
                "role":              {"status": "missing",
                                      "justification": "[mock] No persona specified."},
                "audience":          {"status": "partial",
                                      "justification": "[mock] Audience implied, not stated."},
                "task_specificity":  {"status": "partial",
                                      "justification": "[mock] Task named, sub-goals implicit."},
                "input_format":      {"status": "missing",
                                      "justification": "[mock] No expected input defined."},
                "output_format":     {"status": "missing",
                                      "justification": "[mock] No output structure."},
                "length_constraint": {"status": "missing",
                                      "justification": "[mock] No length given."},
                "tone_constraint":   {"status": "missing",
                                      "justification": "[mock] Tone unspecified."},
            },
            "summary": "[mock] Prompt is under-specified on structure and audience.",
        })

    # Evaluator judge
    if "rubric" in p and "score each output" in p:
        return json.dumps({
            "A": {"completion": 3.0, "compliance": 3.0, "structure": 3.0},
            "B": {"completion": 3.0, "compliance": 3.0, "structure": 3.0},
            "C": {"completion": 3.0, "compliance": 3.0, "structure": 3.0},
            "winner": "A",
            "explanation": "[mock-output#judge] deterministic placeholder.",
        })

    # Generic tagged output
    return (
        f"[mock-output#{seed}] A deterministic placeholder produced by the "
        "mock inference backend. If you see this in benchmark results, the "
        "primary backend (Ollama or HF) failed and fell back to mock."
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> Optional[dict]:
    """
    Extract the first balanced top-level JSON object from `text`.

    Handles real-model failure modes the greedy \\{.*\\} regex misses:
      - JSON preceded or followed by prose
      - Multiple JSON-like fragments (takes the first valid one)
      - Nested braces
      - Markdown code fences (```json ... ```)
      - Braces inside string literals (tracked as non-structural)

    Returns None if no valid JSON object is found.
    """
    if not text:
        return None

    # Strip markdown code fences first.
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = cleaned.replace("```", "")

    # Brace-matching scan. Walk the string, track nesting depth, and
    # collect each balanced {...} region. Try parsing each in order.
    depth = 0
    start = -1
    candidates: list[str] = []
    in_string = False
    escape = False

    for i, ch in enumerate(cleaned):
        # Track whether we're inside a JSON string literal — braces inside
        # strings don't count toward nesting.
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    candidates.append(cleaned[start : i + 1])
                    start = -1

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    return None


# ---------------------------------------------------------------------------
# Startup probe: log configured backend at module import, so a benchmark
# run always writes the resolved config to its logs.
# ---------------------------------------------------------------------------

def probe() -> dict:
    """
    Return current resolved config without actually making a network call.
    Safe to call at any time; used by /backend/status.
    """
    return {
        "use_ollama": USE_OLLAMA,
        "ollama_model": OLLAMA_MODEL,
        "ollama_endpoint": OLLAMA_ENDPOINT,
        "ollama_timeout": OLLAMA_TIMEOUT,
        "hf_model": HF_MODEL,
        "hf_token_configured": bool(HF_TOKEN),
        "hf_timeout": HF_TIMEOUT,
        "strict_mode": STRICT,
    }


print(
    f"[praas.inference] backend: "
    f"ollama={USE_OLLAMA}({OLLAMA_MODEL}) "
    f"hf={bool(HF_TOKEN)} "
    f"strict={STRICT}",
    file=sys.stderr,
    flush=True,
)
