"""End-to-end tests for the PraaS gateway (mock backend, no network)."""

import os
# Force mock backend: ensure no HF_TOKEN leaks from the env.
os.environ.pop("HF_TOKEN", None)
# Use a throwaway DB per test run.
os.environ["PRAAS_DB"] = "/tmp/praas_test.db"

import pytest
from fastapi.testclient import TestClient

# Clean DB before import.
if os.path.exists("/tmp/praas_test.db"):
    os.remove("/tmp/praas_test.db")

from app.main import app
from app import fpdb

# Initialise DB directly (lifespan only runs under `with TestClient(app)`).
fpdb.init_db()

client = TestClient(app)


def test_index_serves():
    r = client.get("/")
    assert r.status_code == 200
    assert "Prompt-as-a-Service" in r.text


def test_analyze():
    r = client.post("/analyze", json={
        "prompt": "write a story",
        "task_description": "bedtime story for an 8-year-old about sharing",
    })
    assert r.status_code == 200
    body = r.json()
    assert "dimensions" in body
    assert len(body["dimensions"]) == 7
    assert 0.0 <= body["overall_quality"] <= 1.0
    assert body["prompt_hash"]


def test_optimize():
    r = client.post("/optimize", json={
        "prompt": "write a story",
        "task_description": "bedtime story",
        "iterations": 1,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["original_prompt"] == "write a story"
    assert len(body["optimised_prompt"]) > 10
    assert body["iterations_run"] == 1


def test_adapt():
    r = client.post("/adapt", json={
        "prompt": "Write a 150-word bedtime story.",
        "families": ["claude", "gpt", "gemini"],
    })
    assert r.status_code == 200
    body = r.json()
    fams = [v["family"] for v in body["variants"]]
    assert set(fams) == {"claude", "gpt", "gemini"}
    # Each variant should be a different string.
    prompts = [v["prompt"] for v in body["variants"]]
    assert len(set(prompts)) == 3


def test_evaluate():
    # First adapt to get variants.
    ad = client.post("/adapt", json={
        "prompt": "Write a 150-word bedtime story.",
    }).json()
    r = client.post("/evaluate", json={
        "variants": ad["variants"],
        "task_description": "bedtime story",
    })
    assert r.status_code == 200
    body = r.json()
    assert len(body["scores"]) == 3
    assert body["winner"] in {"claude", "gpt", "gemini"}
    for s in body["scores"]:
        assert 0.0 <= s["score"] <= 5.0
        assert set(s["rubric_breakdown"].keys()) == {"completion", "compliance", "structure"}


def test_pipeline_end_to_end():
    r = client.post("/pipeline", json={
        "prompt": "write a story",
        "task_description": "bedtime story for an 8-year-old about sharing",
    })
    assert r.status_code == 200
    body = r.json()
    assert "analyze" in body
    assert "optimize" in body
    assert "adapt" in body
    assert "evaluate" in body
    assert len(body["adapt"]["variants"]) == 3
    assert len(body["evaluate"]["scores"]) == 3


def test_fpdb_records_and_reports():
    # FPDB should have entries after the pipeline test ran.
    r = client.get("/fpdb/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["total_prompts_analysed"] >= 1
    # After at least one call, we should have some missing dims counted.
    assert sum(body["missing_dimension_counts"].values()) >= 1
