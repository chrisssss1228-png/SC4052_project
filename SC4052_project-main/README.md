# Prompt-as-a-Service (PraaS)

**Engineering prompts as cloud-native software artifacts.**

Cloud Computing course project · Topic 8: X-as-a-Service · Nanyang Technological University · April 2026
Author: **Chen Leyi** (U2410501C)

---

## What is PraaS?

PraaS turns **prompt engineering itself** into a cloud service. Users submit a
vague prompt plus a task description; the platform diagnoses it, rewrites it,
produces three per-model variants, and scores each variant with an
LLM-as-judge. Every call silently contributes an anonymised diagnostic to a
**Failure Pattern Database** — the dual-utility side channel that satisfies
the crowdsourcing principle in the project brief (one workflow, two
objectives: user improvement + research dataset).

Four microservices behind one RESTful gateway:

| Service | Job |
|---|---|
| **Analyzer** | Score the prompt on 7 dimensions (role, audience, task_specificity, input_format, output_format, length, tone) |
| **Optimizer** | Rewrite using a TextGrad-style gradient → update loop |
| **Multi-Model Adapter** | Produce Claude (XML) / GPT (markdown) / Gemini (step-enumerated) variants |
| **Evaluator** | Run each variant, score with LLM-as-judge, declare a winner |

---

## Quick start

### Requirements

- **Python 3.12+**
- **Ollama** installed and running locally, with the `llama3.2` model pulled
  (the default backend — no API key, no paid service needed)
- macOS / Linux (tested on Apple Silicon)

### Install

```bash
cd praas_code
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Start Ollama (in a separate terminal)

```bash
ollama serve                # keep this running
ollama pull llama3.2        # one-time, ~2GB download
```

### Run the web UI

```bash
uvicorn app.main:app --port 8000
```

Then open a browser at:

| URL | What it is |
|---|---|
| `http://localhost:8000/`                  | Interactive demo frontend (4-stage pipeline view) |
| `http://localhost:8000/docs`              | Auto-generated OpenAPI / Swagger docs |
| `http://localhost:8000/backend/status`    | Machine-readable backend configuration |
| `http://localhost:8000/fpdb/stats`        | Aggregated Failure Pattern Database statistics |

---

## Running the benchmark

The benchmark reported in Section 7 of the project report is reproducible
end-to-end. It runs 21 under-specified prompts (7 creative / 7 code / 7
analysis) through the full pipeline on your local Ollama.

```bash
# 1. From praas_code/, clear any stale state
rm -f benchmark/results.json benchmark/evidence.json
rm -f /tmp/praas_bench.db*

# 2. Strict mode — CRITICAL for trustworthy data.
#    Without STRICT, an Ollama timeout would silently return a mock
#    response and pollute the benchmark. STRICT re-raises on failure.
export PRAAS_STRICT=1
export OLLAMA_TIMEOUT=300

# 3. Run (caffeinate prevents macOS from sleeping mid-run)
caffeinate -i env PYTHONPATH=. python benchmark/run_benchmark.py \
  2>&1 | tee benchmark/run.log

# 4. Verify no mock pollution — both must return 0
grep -c 'mock-output' benchmark/evidence.json
grep -c '\[mock\]'    benchmark/evidence.json
```

**Expected runtime**: ~35 minutes on Apple Silicon (no GPU offload).

**Quick sanity check**: `export PRAAS_SMOKE=1` before step 3 to run only the
first 3 prompts (~5 minutes).

### What the benchmark produces

| File | Contents |
|---|---|
| `benchmark/results.json` | Per-prompt scores + summary table by family |
| `benchmark/evidence.json` | Complete audit trail per prompt: raw prompt, diagnostic, gradient, optimised prompt, three adapted variants, every variant's output, every rubric score, declared winner |
| `benchmark/run.log`      | Full stdout/stderr of the run, including backend configuration banner |

### Headline numbers (21 prompts, llama3.2 backend)

| Family | n | Baseline | PraaS | Gain |
|---|---|---|---|---|
| Creative writing | 7 | 2.05 | 4.17 | **+2.12** |
| Code generation | 7 | 3.24 | 4.29 | **+1.05** |
| Analytical reasoning | 7 | 2.91 | 4.17 | **+1.27** |
| **Overall** | **21** | **2.73** | **4.21** | **+1.48** |

---

## Project layout

```
praas_code/
├── app/
│   ├── main.py                FastAPI gateway (7 endpoints)
│   ├── schemas.py             Pydantic request/response models
│   ├── inference.py           Ollama → HF → mock 3-level backend
│   ├── fpdb.py                SQLite-backed Failure Pattern Database
│   ├── services/
│   │   ├── analyzer.py        Seven-dimension diagnostic
│   │   ├── optimizer.py       TextGrad-style gradient → update
│   │   ├── adapter.py         Claude / GPT / Gemini variants
│   │   └── evaluator.py       LLM-as-judge with position-bias mitigation
│   └── static/                HTML frontend (one page, vanilla JS)
├── benchmark/
│   ├── run_benchmark.py       21-prompt benchmark harness
│   ├── results.json           Summary + per-prompt scores (generated)
│   └── evidence.json          Full audit trail (generated)
├── tests/
│   └── test_e2e.py            pytest end-to-end tests
├── requirements.txt
└── README.md
```

---

## Environment variables

All are optional. Defaults are chosen so that a fresh `uvicorn app.main:app`
works out of the box when Ollama is running.

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.2` | Which model Ollama should serve |
| `OLLAMA_HOST` | `http://localhost:11434` | Where Ollama is listening |
| `OLLAMA_TIMEOUT` | `300` | Seconds before an Ollama request is considered failed |
| `USE_OLLAMA` | `1` | Set to `0` to skip the Ollama backend |
| `HF_TOKEN` | *(unset)* | If set and Ollama unavailable, use the Hugging Face Inference API |
| `HF_MODEL` | `HuggingFaceH4/zephyr-7b-beta` | Which HF model to use |
| `PRAAS_STRICT` | `0` | **Set to `1` for benchmarks.** Re-raise backend errors instead of falling back to mock. |
| `PRAAS_SMOKE` | *(unset)* | Set to `1` to limit the benchmark to the first 3 prompts |
| `PRAAS_DB` | `/tmp/praas_bench.db` (benchmark) / `praas.db` (app) | SQLite path for the FPDB |

---

## Design choices worth knowing

Two engineering decisions are less obvious from the code and worth calling
out explicitly.

### 1. Strict mode + mock tagging (prevents silent data corruption)

Every mock response contains the literal string `[mock-output#]` or
`[mock]`. After a benchmark run, you can grep the evidence file to prove
no mock responses leaked in:

```bash
grep -c 'mock-output' benchmark/evidence.json    # must be 0
grep -c '\[mock\]'    benchmark/evidence.json    # must be 0
```

In strict mode (`PRAAS_STRICT=1`) a failing Ollama call re-raises instead
of silently falling back to mock. Without this, a transient Ollama timeout
in the middle of a 35-minute benchmark would pass unnoticed and produce
fake numbers.

### 2. Tolerant judge-key decoding (prevents silent 3.0 defaults)

The Evaluator asks its LLM judge for keys `A`, `B`, `C`. Real models emit
`Output A`, `variant_a`, lowercase `a`, etc. A naive `raw.get("A")` would
silently miss and default every rubric axis to 3.0 — producing data that
looks successful but contains no signal. We accept all plausible variants:

```python
def _find_rubric(raw: dict, slot: str) -> dict:
    candidates = [
        slot, slot.lower(),
        f"Output {slot}",  f"output_{slot.lower()}",
        f"Variant {slot}", f"variant_{slot.lower()}",
        # …
    ]
    for key in candidates:
        if key in raw and isinstance(raw[key], dict):
            return raw[key]
    return {}
```

Winner-field decoding, by contrast, is deliberately strict — it only
accepts a standalone `A`-`D` letter inside the `winner` field — because
the cost of a false positive is higher than the cost of falling back to
max-score.

---

## Running the tests

```bash
cd praas_code
source .venv/bin/activate
pytest -q
```

The tests run against the mock backend (no Ollama required) and cover
the full pipeline end-to-end.

---

## API example

```bash
# Full pipeline
curl -X POST http://localhost:8000/pipeline \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "write a story",
    "task_description": "bedtime story for an 8-year-old about sharing"
  }'

# Just analyze (fast, single call)
curl -X POST http://localhost:8000/analyze \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "write a story", "task_description": "bedtime story"}'

# FPDB aggregate stats
curl http://localhost:8000/fpdb/stats
```

---

## Honest limitations

See Section 7.6–7.8 of the project report for detail. The main ones:

- **Small benchmark** (n=21). A deliberate trade for full per-prompt
  auditability. A 5× scale-up is feasible with no code changes.
- **llama3.2-3B as judge has coarse score resolution.** Scores cluster
  tightly around `{3.33, 3.83, 4.17}` because a 3B model tends to return
  only integer/half-integer values on a 0–5 scale. Relative comparisons
  remain meaningful; absolute numbers should be read with that caveat.
- **Winner vs max-score** occasionally disagrees (11/21 prompts). Both
  signals are preserved in the API response — they answer different
  questions. See report Section 7.7.
- **Prompts are authored by the project team**, not sampled from
  production traffic. This may overstate PraaS's typical gain on the
  prompts real users write.

---

## Licence

MIT. See `LICENSE` if present; otherwise consider the code freely
reusable with attribution.

---

## Repository

Source code: https://github.com/chrisssss1228-png/SC4052_project/edit/main/SC4052_project-main
