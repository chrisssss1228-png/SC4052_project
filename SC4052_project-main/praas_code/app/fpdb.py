"""
Failure Pattern Database (FPDB).

A SQLite-backed aggregate of anonymised diagnostic records. On every
Analyzer call we persist a row; only the prompt hash and the structured
diagnostic are kept — no raw prompt text, no user identifiers.

The FPDB is the research-utility face of the dual-utility XaaS design:
individual users benefit from immediate prompt improvement, while
educators and prompt-injection defence researchers benefit from the
aggregated failure patterns.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from typing import Dict

from app.schemas import AnalyzeResponse, FPDBStats


DB_PATH = os.environ.get("PRAAS_DB", "praas.db")
_lock = threading.Lock()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db() -> None:
    with _lock, _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS diagnostics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_hash TEXT NOT NULL,
                dimension TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_diagnostics_dimension
            ON diagnostics(dimension, status)
            """
        )
        conn.commit()


def record(diagnostic: AnalyzeResponse) -> None:
    """Persist one diagnostic record, dimension by dimension."""
    rows = [
        (diagnostic.prompt_hash, name, d.status.value)
        for name, d in diagnostic.dimensions.items()
    ]

    with _lock, _connect() as conn:
        conn.executemany(
            """
            INSERT INTO diagnostics (prompt_hash, dimension, status)
            VALUES (?, ?, ?)
            """,
            rows,
        )
        conn.commit()


def stats() -> FPDBStats:
    with _lock, _connect() as conn:
        cur = conn.cursor()

        # Total number of distinct prompts ever analysed.
        cur.execute("SELECT COUNT(DISTINCT prompt_hash) FROM diagnostics")
        total = cur.fetchone()[0] or 0

        # Count of missing statuses by dimension.
        cur.execute(
            """
            SELECT dimension, COUNT(*)
            FROM diagnostics
            WHERE status = 'missing'
            GROUP BY dimension
            """
        )
        missing_counts: Dict[str, int] = {
            row[0]: row[1] for row in cur.fetchall()
        }

    pct: Dict[str, float] = {
        dim: (cnt / total if total else 0.0)
        for dim, cnt in missing_counts.items()
    }

    top = sorted(
        missing_counts.items(),
        key=lambda kv: kv[1],
        reverse=True,
    )
    top_weaknesses = [dim for dim, _ in top[:3]]

    return FPDBStats(
        total_prompts_analysed=total,
        missing_dimension_counts=missing_counts,
        missing_dimension_pct={k: round(v, 3) for k, v in pct.items()},
        top_weaknesses=top_weaknesses,
    )
