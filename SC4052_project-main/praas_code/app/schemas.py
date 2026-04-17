"""
Pydantic data models for PraaS service I/O.

Every microservice in PraaS communicates through these schemas, so changes
here ripple through the whole system.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class DimensionStatus(str, Enum):
    PRESENT = "present"
    PARTIAL = "partial"
    MISSING = "missing"


class DimensionDiagnostic(BaseModel):
    """Diagnostic for one of the seven prompt dimensions."""
    status: DimensionStatus
    justification: str = Field(
        ...,
        description="One-line human-readable reason.",
    )


class AnalyzeRequest(BaseModel):
    prompt: str = Field(
        ...,
        min_length=1,
        description="Raw user prompt.",
    )
    task_description: Optional[str] = Field(
        default=None,
        description="Optional clarification of the user's true intent.",
    )


class AnalyzeResponse(BaseModel):
    prompt_hash: str
    dimensions: Dict[str, DimensionDiagnostic]
    overall_quality: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Simple heuristic quality score in [0, 1].",
    )
    summary: str


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

class OptimizeRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    task_description: Optional[str] = None
    diagnostic: Optional[AnalyzeResponse] = None
    iterations: int = Field(1, ge=1, le=3)


class OptimizeResponse(BaseModel):
    original_prompt: str
    optimised_prompt: str
    gradient: str = Field(
        ...,
        description="Textual gradient produced in step 1.",
    )
    iterations_run: int = Field(..., ge=0, le=3)


# ---------------------------------------------------------------------------
# Multi-Model Adapter
# ---------------------------------------------------------------------------

class ModelFamily(str, Enum):
    CLAUDE = "claude"
    GPT = "gpt"
    GEMINI = "gemini"


class AdaptRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    families: List[ModelFamily] = Field(
        default_factory=lambda: list(ModelFamily),
    )


class AdaptedVariant(BaseModel):
    family: ModelFamily
    prompt: str
    notes: str


class AdaptResponse(BaseModel):
    variants: List[AdaptedVariant]


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class EvaluateRequest(BaseModel):
    variants: List[AdaptedVariant] = Field(..., min_length=1)
    task_description: Optional[str] = None


class VariantScore(BaseModel):
    family: ModelFamily
    output: str
    score: float = Field(..., ge=0.0, le=5.0)
    rubric_breakdown: Dict[str, float]


class EvaluateResponse(BaseModel):
    scores: List[VariantScore]
    winner: ModelFamily
    explanation: str


# ---------------------------------------------------------------------------
# Pipeline (end-to-end)
# ---------------------------------------------------------------------------

class PipelineRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    task_description: Optional[str] = None


class PipelineResponse(BaseModel):
    analyze: AnalyzeResponse
    optimize: OptimizeResponse
    adapt: AdaptResponse
    evaluate: EvaluateResponse


# ---------------------------------------------------------------------------
# Failure Pattern Database
# ---------------------------------------------------------------------------

class FPDBStats(BaseModel):
    total_prompts_analysed: int = Field(..., ge=0)
    missing_dimension_counts: Dict[str, int]
    missing_dimension_pct: Dict[str, float]
    top_weaknesses: List[str]
