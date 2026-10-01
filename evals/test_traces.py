"""The eval suite: every check against every trace in one recorded file.

    EVAL_TRACES=evals/traces/baseline.jsonl pytest evals/ -q
    make eval TRACES=evals/traces/baseline.jsonl

One test per (trace, check), so pytest's own pass/fail count is the eval's
pass/fail count, and a failure's message is the check's one-line reason.
Without EVAL_TRACES the newest file in evals/traces/ is used.

This lives outside `tests/` on purpose. `make test` must pass on a fresh
clone with no API keys; this suite needs a recorded file, and its failures
are measurements of the agents, not bugs in the code. The checks themselves
are unit-tested in `tests/test_evals.py`.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.config import Settings
from evals.checks import Row, TruthDB, evaluate_file, load_cases
from evals.trace import read_traces

TRACES_DIR = Path(__file__).resolve().parent / "traces"


def _trace_file() -> Path | None:
    if env := os.environ.get("EVAL_TRACES"):
        return Path(env)
    files = sorted(TRACES_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def _rows() -> list[Row]:
    path = _trace_file()
    if path is None or not path.exists():
        return []
    return evaluate_file(read_traces(path), load_cases(), TruthDB(Settings().clues_db_path))


ROWS = _rows()


@pytest.mark.parametrize("row", ROWS, ids=[f"{r.trace.key}::{r.result.check}" for r in ROWS])
def test_check(row: Row) -> None:
    if not row.result.passed:
        # pytrace=False: the message *is* the report, one line, no traceback.
        pytest.fail(f"[{row.result.category}] {row.result.reason}", pytrace=False)
