"""
PraaS FastAPI gateway.

Exposes the four microservices behind a single REST API:

  POST /analyze         diagnose a prompt
  POST /optimize        rewrite a prompt (TextGrad-style)
  POST /adapt           produce per-model variants
  POST /evaluate        run & judge variants
  POST /pipeline        end-to-end orchestration
  GET  /fpdb/stats      aggregated Failure Pattern Database statistics
  GET  /backend/status  report current inference backend configuration
  GET  /                serve the HTML demo frontend
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import fpdb
from app import inference
from app.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    OptimizeRequest,
    OptimizeResponse,
    AdaptRequest,
    AdaptResponse,
    EvaluateRequest,
    EvaluateResponse,
    PipelineRequest,
    PipelineResponse,
    FPDBStats,
    ModelFamily,
)
from app.services.analyzer import analyze as _analyze
from app.services.optimizer import optimize as _optimize
from app.services.adapter import adapt as _adapt
from app.services.evaluator import evaluate as _evaluate


STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    fpdb.init_db()
    yield


app = FastAPI(
    title="Prompt-as-a-Service (PraaS)",
    description=(
        "Prompt engineering exposed as four independently addressable "
        "cloud microservices: Analyzer, Optimizer, Multi-Model Adapter, "
        "Evaluator. Every call anonymously contributes to a Failure "
        "Pattern Database (dual-utility)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Static demo frontend
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Backend status
# ---------------------------------------------------------------------------

@app.get("/backend/status", summary="Report current inference backend configuration.")
def backend_status() -> dict:
    return {
        "hf_model": inference.HF_MODEL,
        "hf_token_configured": bool(inference.HF_TOKEN),
        "use_ollama": inference.USE_OLLAMA,
        "ollama_model": inference.OLLAMA_MODEL,
        "ollama_endpoint": inference.OLLAMA_ENDPOINT,
        "mock_available": True,
    }


# ---------------------------------------------------------------------------
# Four service endpoints
# ---------------------------------------------------------------------------

@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Diagnose a prompt on seven dimensions.",
)
async def analyze_endpoint(req: AnalyzeRequest) -> AnalyzeResponse:
    result = await _analyze(req)
    fpdb.record(result)
    return result


@app.post(
    "/optimize",
    response_model=OptimizeResponse,
    summary="Rewrite a prompt using a TextGrad-style gradient update.",
)
async def optimize_endpoint(req: OptimizeRequest) -> OptimizeResponse:
    return await _optimize(req)


@app.post(
    "/adapt",
    response_model=AdaptResponse,
    summary="Produce Claude / GPT / Gemini variants of a prompt.",
)
async def adapt_endpoint(req: AdaptRequest) -> AdaptResponse:
    return await _adapt(req)


@app.post(
    "/evaluate",
    response_model=EvaluateResponse,
    summary="Run adapted variants, judge them, return ranked scores.",
)
async def evaluate_endpoint(req: EvaluateRequest) -> EvaluateResponse:
    return await _evaluate(req)


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

@app.post(
    "/pipeline",
    response_model=PipelineResponse,
    summary="Run the full Analyzer → Optimizer → Adapter → Evaluator chain.",
)
async def pipeline_endpoint(req: PipelineRequest) -> PipelineResponse:
    analysis = await _analyze(
        AnalyzeRequest(
            prompt=req.prompt,
            task_description=req.task_description,
        )
    )
    fpdb.record(analysis)

    opt = await _optimize(
        OptimizeRequest(
            prompt=req.prompt,
            task_description=req.task_description,
            diagnostic=analysis,
            iterations=1,
        )
    )

    ad = await _adapt(
        AdaptRequest(
            prompt=opt.optimised_prompt,
            families=list(ModelFamily),
        )
    )

    ev = await _evaluate(
        EvaluateRequest(
            variants=ad.variants,
            task_description=req.task_description,
        )
    )

    return PipelineResponse(
        analyze=analysis,
        optimize=opt,
        adapt=ad,
        evaluate=ev,
    )


# ---------------------------------------------------------------------------
# FPDB stats (dual-utility)
# ---------------------------------------------------------------------------

@app.get(
    "/fpdb/stats",
    response_model=FPDBStats,
    summary="Aggregated Failure Pattern Database statistics.",
)
def fpdb_stats_endpoint() -> FPDBStats:
    return fpdb.stats()
