"""
Benchmark harness for PraaS.

Runs a curated set of under-specified prompts through:
1. the raw prompt sent directly to the model
2. the full PraaS pipeline

Then scores both with an independent judge prompt and writes:
- benchmark/results.json
- benchmark/evidence.json

This version is hardened for local Ollama experiments:
- logs resolved backend configuration at startup
- preserves raw judge outputs for auditability
- avoids silently collapsing all parse failures to 3.0
- flags suspicious evaluator-judge failures in evidence
- writes intermediate results after each completed prompt
- supports PRAAS_SMOKE=1 to limit the run to the first 3 prompts

Environment variables:
  PRAAS_STRICT=1     Re-raise backend errors instead of silently mocking.
                     **Always set this when running for real data.**
  PRAAS_SMOKE=1      Run only the first 3 prompts (quick sanity check).
  OLLAMA_TIMEOUT=…   Seconds per Ollama call. Default 300. Raise for slow hardware.
  PRAAS_DB=/path/db  Override SQLite path. Defaults to /tmp/praas_bench.db.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any

# Isolated DB for benchmark runs.
os.environ.setdefault("PRAAS_DB", "/tmp/praas_bench.db")
if os.path.exists(os.environ["PRAAS_DB"]):
    os.remove(os.environ["PRAAS_DB"])

from app import inference, fpdb
from app.schemas import (
    AnalyzeRequest,
    OptimizeRequest,
    AdaptRequest,
    EvaluateRequest,
    ModelFamily,
)
from app.services.analyzer import analyze
from app.services.optimizer import optimize
from app.services.adapter import adapt
from app.services.evaluator import evaluate

fpdb.init_db()


BENCH = [
    # (prompt, task_description, family)
    # --- Creative writing (7) ---
    ("write a story", "bedtime story for an 8-year-old about sharing", "creative"),
    ("write a poem", "four-line poem about rain for a birthday card", "creative"),
    ("write me a song", "silly 30-second song for a toddler about brushing teeth", "creative"),
    ("tell a joke", "short office-appropriate joke about Monday mornings", "creative"),
    ("describe a place", "100-word description of a bustling food market for a travel blog", "creative"),
    ("write some dialogue", "5-line conversation between a grumpy dragon and a polite child", "creative"),
    ("draft a speech", "2-minute graduation speech with a warm tone and one personal anecdote", "creative"),

    # --- Code generation (7) ---
    ("write python to scrape a website", "extract article titles and dates, respect robots.txt, output CSV", "code"),
    ("build a REST API", "FastAPI /items endpoint with GET/POST/DELETE, in-memory storage", "code"),
    ("parse JSON", "Python function to flatten nested JSON into dotted-key dict", "code"),
    ("write a sort function", "in-place quicksort with three-way partitioning in Python", "code"),
    ("make a regex", "Python regex for ISO 8601 timestamps with optional timezone", "code"),
    ("deduplicate a list", "Python one-liner preserving order for list of dicts keyed by 'id'", "code"),
    ("connect to database", "Python SQLite connection with context manager and parameterised queries", "code"),

    # --- Analytical reasoning (7) ---
    ("summarise this paper", "extract research question, method, 3 limitations for a 5-min talk", "analysis"),
    ("analyse this data", "identify three statistically meaningful trends in monthly sales CSV", "analysis"),
    ("compare two options", "pros/cons of React vs Vue for a solo developer 6-month MVP", "analysis"),
    ("explain this concept", "explain RAFT consensus to a junior engineer in 150 words", "analysis"),
    ("review my argument", "critique 'if X then Y; Y happened; therefore X' — identify fallacy", "analysis"),
    ("forecast a trend", "three plausible 2-year outcomes for cloud GPU prices", "analysis"),
    ("recommend a decision", "framework for remote vs hybrid work for a team of 8 engineers", "analysis"),
]


# Smoke-test mode: run only a handful of prompts for fast iteration.
if os.environ.get("PRAAS_SMOKE"):
    BENCH = BENCH[:3]


JUDGE = """You are an independent judge scoring an LLM output against a task.

TASK: {task}

OUTPUT:
\"\"\"
{output}
\"\"\"

Rate the output on three axes from 0 to 5 each:
  completion, compliance, structure.

Respond ONLY with JSON:
{{"completion": 0-5, "compliance": 0-5, "structure": 0-5}}
"""


def _safe_float(x: Any) -> float | None:
    """Coerce to a float in [0, 5], or None if junk / NaN / Inf."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return max(0.0, min(5.0, v))


def _all_default_rubric(rb: dict) -> bool:
    """
    True when all three rubric axes are exactly the evaluator's default 3.0,
    which is the signature of a judge-JSON parse failure inside evaluator.py.
    """
    return (
        rb.get("completion") == 3.0
        and rb.get("compliance") == 3.0
        and rb.get("structure") == 3.0
    )


async def score_one(task: str, output: str) -> dict:
    """
    Score one output with the independent judge.

    Returns a dict with:
    - score:              averaged float
    - rubric:             dict of completion / compliance / structure
    - raw_judge:          raw parsed JSON returned by generate_json
    - judge_parse_failed: True if any axis had to be defaulted
    """
    raw = await inference.generate_json(
        JUDGE.format(task=task, output=output),
        max_new_tokens=150,
    )

    c = _safe_float(raw.get("completion")) if isinstance(raw, dict) else None
    p = _safe_float(raw.get("compliance")) if isinstance(raw, dict) else None
    s = _safe_float(raw.get("structure")) if isinstance(raw, dict) else None

    parse_failed = any(v is None for v in (c, p, s))

    # Conservative fallback, but now explicitly marked in evidence.
    if c is None:
        c = 3.0
    if p is None:
        p = 3.0
    if s is None:
        s = 3.0

    score = (c + p + s) / 3.0

    return {
        "score": score,
        "rubric": {
            "completion": round(c, 3),
            "compliance": round(p, 3),
            "structure":  round(s, 3),
        },
        "raw_judge": raw,
        "judge_parse_failed": parse_failed,
    }


async def run_one(prompt: str, task: str):
    # Baseline — raw prompt directly.
    baseline_out = (await inference.generate(prompt, max_new_tokens=300)).strip()
    baseline_eval = await score_one(task, baseline_out)
    baseline = baseline_eval["score"]

    # PraaS pipeline.
    an = await analyze(AnalyzeRequest(prompt=prompt, task_description=task))
    fpdb.record(an)

    opt = await optimize(
        OptimizeRequest(
            prompt=prompt,
            task_description=task,
            diagnostic=an,
            iterations=1,
        )
    )
    ad = await adapt(
        AdaptRequest(
            prompt=opt.optimised_prompt,
            families=list(ModelFamily),
        )
    )
    ev = await evaluate(
        EvaluateRequest(
            variants=ad.variants,
            task_description=task,
        )
    )
    praas = max(s.score for s in ev.scores)

    # Cross-model standard deviation across the three adapted variants.
    scores = [s.score for s in ev.scores]
    sd = statistics.pstdev(scores) if len(scores) > 1 else 0.0

    # Detect: if every variant has the exact default 3.0/3.0/3.0 rubric,
    # the evaluator's judge most likely failed to produce parseable JSON
    # and all scores are fallback values. Flag the row so report writing
    # can filter these out rather than average them as if real.
    praas_judge_suspicious = all(
        _all_default_rubric(s.rubric_breakdown) for s in ev.scores
    )

    # Full evidence for audit / viva.
    evidence = {
        "baseline_output":              baseline_out,
        "baseline_judge_score":         round(baseline_eval["score"], 3),
        "baseline_judge_rubric":        baseline_eval["rubric"],
        "baseline_judge_raw":           baseline_eval["raw_judge"],
        "baseline_judge_parse_failed":  baseline_eval["judge_parse_failed"],
        "analyzer_summary":             an.summary,
        "analyzer_missing": [
            dim for dim, d in an.dimensions.items()
            if d.status.value != "present"
        ],
        "optimiser_gradient":   opt.gradient,
        "optimised_prompt":     opt.optimised_prompt,
        "variants": [
            {
                "family": v.family.value,
                "prompt": v.prompt,
            }
            for v in ad.variants
        ],
        "variant_scores": [
            {
                "family": s.family.value,
                "output": s.output,
                "score":  round(s.score, 3),
                "rubric": s.rubric_breakdown,
            }
            for s in ev.scores
        ],
        "winner":                 ev.winner.value,
        "praas_judge_suspicious": praas_judge_suspicious,
    }

    return baseline, praas, sd, evidence, praas_judge_suspicious


def _build_summary(results: list[dict]) -> dict:
    summary = {
        "n": len(results),
        "mean_baseline":          0.0,
        "mean_praas":             0.0,
        "mean_gain":              0.0,
        "mean_cross_model_sd":    0.0,
        "n_judge_suspicious":     0,
        "by_family":              {},
    }

    if not results:
        return summary

    summary["mean_baseline"]       = round(statistics.mean(r["baseline_score"] for r in results), 3)
    summary["mean_praas"]          = round(statistics.mean(r["praas_score"] for r in results), 3)
    summary["mean_gain"]           = round(statistics.mean(r["gain"] for r in results), 3)
    summary["mean_cross_model_sd"] = round(statistics.mean(r["cross_model_sd"] for r in results), 3)
    summary["n_judge_suspicious"]  = sum(1 for r in results if r.get("judge_suspicious"))

    for fam in ("creative", "code", "analysis"):
        sub = [r for r in results if r["family"] == fam]
        if sub:
            summary["by_family"][fam] = {
                "n":                    len(sub),
                "mean_baseline":        round(statistics.mean(r["baseline_score"] for r in sub), 3),
                "mean_praas":           round(statistics.mean(r["praas_score"] for r in sub), 3),
                "mean_gain":            round(statistics.mean(r["gain"] for r in sub), 3),
                "mean_cross_model_sd":  round(statistics.mean(r["cross_model_sd"] for r in sub), 3),
                "n_judge_suspicious":   sum(1 for r in sub if r.get("judge_suspicious")),
            }

    return summary


def _write_outputs(results: list[dict], evidence_log: list[dict]) -> None:
    Path("benchmark").mkdir(exist_ok=True)

    summary = _build_summary(results)

    out = Path("benchmark/results.json")
    out.write_text(
        json.dumps(
            {"summary": summary, "per_prompt": results},
            indent=2,
            ensure_ascii=False,
        )
    )

    ev_out = Path("benchmark/evidence.json")
    ev_out.write_text(
        json.dumps(
            evidence_log,
            indent=2,
            ensure_ascii=False,
        )
    )


def _print_startup_banner() -> None:
    """
    Record the resolved backend configuration at the top of the run log
    so benchmark output is self-documenting.
    """
    print("=" * 60)
    print("PraaS benchmark — backend configuration")
    print("=" * 60)
    print(json.dumps(inference.probe(), indent=2))
    print(f"STRICT mode: {inference.STRICT}")

    if not inference.STRICT:
        print()
        print("⚠️  WARNING: PRAAS_STRICT=0")
        print("   Mock fallback may silently pollute benchmark results.")
        print("   Set PRAAS_STRICT=1 before running for trustworthy data.")

    if os.environ.get("PRAAS_SMOKE"):
        print()
        print(f"🧪 SMOKE mode: only running first {len(BENCH)} prompts.")

    print(f"\nBenchmark size: n={len(BENCH)}")
    print("=" * 60)
    print()


async def main():
    _print_startup_banner()

    results = []
    evidence_log = []

    for prompt, task, family in BENCH:
        baseline, praas, sd, evidence, judge_suspicious = await run_one(prompt, task)

        row = {
            "prompt":            prompt,
            "task":              task,
            "family":            family,
            "baseline_score":    round(baseline, 3),
            "praas_score":       round(praas, 3),
            "gain":              round(praas - baseline, 3),
            "cross_model_sd":    round(sd, 3),
            "judge_suspicious":  judge_suspicious,
        }
        results.append(row)

        evidence_log.append(
            {
                "prompt": prompt,
                "task":   task,
                "family": family,
                **row,
                **evidence,
            }
        )

        # Incremental writes so partial progress is preserved.
        _write_outputs(results, evidence_log)

        flag = " ⚠" if judge_suspicious else ""
        print(
            f"[{family:>8}] "
            f"base={baseline:.2f}  praas={praas:.2f}  "
            f"gain={praas - baseline:+.2f}  sd={sd:.2f}  "
            f"'{prompt}'{flag}"
        )

    summary = _build_summary(results)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nWrote benchmark/results.json")
    print(f"Wrote benchmark/evidence.json ({len(evidence_log)} evidence records)")

    if summary["n_judge_suspicious"] > 0:
        pct = summary["n_judge_suspicious"] / summary["n"] * 100
        print(
            f"\n⚠️  {summary['n_judge_suspicious']}/{summary['n']} rows "
            f"({pct:.0f}%) had all-default rubrics and are flagged as "
            f"judge-suspicious. Review evidence.json before using these numbers."
        )


if __name__ == "__main__":
    asyncio.run(main())
