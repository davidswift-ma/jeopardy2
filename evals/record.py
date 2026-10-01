#!/usr/bin/env python3
"""Run every eval case through the live multi-agent system; write JSONL traces.

    python -m evals.record --label baseline
    python -m evals.record --label baseline --trials 3
    python -m evals.record --label smoke --cases search-rivers,judge-jordan-river

Costs real money: each case is a router call plus one or more specialist
calls on Gemini, and every clue search is an OpenAI embedding. Checking a
recorded file afterwards costs nothing (`make eval`).

The judge runs over A2A in its own process. If nothing is listening at
JUDGE_AGENT_URL this starts one for the duration of the run and stops it
afterwards, so a recording is one command rather than two terminals.

A daily-quota error stops the run instead of carrying on. Otherwise every
remaining case would be recorded as a dead end and the failure count would
measure the quota, not the agents. What was recorded before the stop is
kept, and `--resume` continues from there.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import subprocess
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from google.adk.agents.run_config import RunConfig  # noqa: E402
from google.adk.runners import Runner  # noqa: E402
from google.adk.sessions import InMemorySessionService  # noqa: E402
from google.genai import types  # noqa: E402

from agents.run_demo import MAX_LLM_CALLS  # noqa: E402
from agents.system import build_router  # noqa: E402
from app.config import Settings  # noqa: E402
from evals.checks import TruthDB, evaluate, load_cases  # noqa: E402
from evals.trace import (  # noqa: E402
    Trace,
    TraceBuilder,
    append_trace,
    instructions_digest,
    read_traces,
)

APP_NAME = "jeopardy_eval"
TRACES_DIR = Path(__file__).resolve().parent / "traces"
REPO_ROOT = Path(__file__).resolve().parent.parent

#: Per-minute limits clear on their own; these are retried. Daily limits don't.
TRANSIENT_CODES = {429, 500, 503}
RETRIES = 3
BACKOFF_S = 20.0


class QuotaExhausted(RuntimeError):
    """A daily limit: retrying today is pointless."""


def _is_daily_quota(exc: BaseException) -> bool:
    return "PerDay" in str(exc)


def system_digest(router) -> str:
    """Fingerprint every instruction, including the remote judge's."""
    texts = [router.instruction or ""]
    texts += [getattr(a, "instruction", "") or "" for a in router.sub_agents]
    from agents.judge_agent import judge_agent  # noqa: PLC0415 - remote side's prompt

    texts.append(judge_agent.instruction or "")
    return instructions_digest(texts)


async def run_case(router, question: str, trace: Trace) -> Trace:
    """One question, fresh session, with retries on transient errors only."""
    for attempt in range(1, RETRIES + 1):
        trace.attempts = attempt
        builder = TraceBuilder(trace)
        service = InMemorySessionService()
        runner = Runner(agent=router, app_name=APP_NAME, session_service=service)
        session = await service.create_session(app_name=APP_NAME, user_id="eval")
        started = time.perf_counter()
        try:
            async for event in runner.run_async(
                user_id="eval",
                session_id=session.id,
                new_message=types.Content(role="user", parts=[types.Part(text=question)]),
                run_config=RunConfig(max_llm_calls=MAX_LLM_CALLS),
            ):
                builder.record(event)
            trace.latency_s = round(time.perf_counter() - started, 2)
            return trace
        except Exception as exc:  # noqa: BLE001 - every failure becomes a trace
            trace.latency_s = round(time.perf_counter() - started, 2)
            if _is_daily_quota(exc):
                raise QuotaExhausted(str(exc)[:300]) from exc
            code = getattr(exc, "code", None)
            if code in TRANSIENT_CODES and attempt < RETRIES:
                print(f"      transient {code}, retrying in {BACKOFF_S * attempt:.0f}s")
                await asyncio.sleep(BACKOFF_S * attempt)
                # A retried attempt starts clean; half a failed run is noise.
                trace = Trace(**{**trace.__dict__, **_empty_run_fields()})
                continue
            builder.fail(exc)
            return trace
    return trace


def _empty_run_fields() -> dict:
    return {
        "route": [],
        "delegated_to": None,
        "tool_calls": [],
        "observations": [],
        "output": "",
        "error": None,
        "llm_calls": 0,
        "tokens": {"prompt": 0, "output": 0, "thoughts": 0},
    }


def _judge_is_up(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/.well-known/agent-card.json", timeout=2):
            return True
    except OSError:
        return False


@contextlib.contextmanager
def judge_server(url: str):
    """Use a running judge, or start one and stop it afterwards."""
    if _judge_is_up(url):
        yield
        return
    port = url.rsplit(":", 1)[-1].strip("/")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "agents.judge_agent:app", "--port", port],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(60):
            if _judge_is_up(url):
                break
            if proc.poll() is not None:
                raise RuntimeError("judge server exited on startup; try `make judge` to see why")
            time.sleep(0.5)
        else:
            raise RuntimeError(f"judge server did not come up at {url}")
        yield
    finally:
        proc.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=5)


async def record(args: argparse.Namespace) -> int:
    settings = Settings()
    if settings.google_api_key is None:
        print("GOOGLE_API_KEY is not set; see agents/run_demo.py.", file=sys.stderr)
        return 1

    cases = load_cases()
    if args.cases:
        wanted = args.cases.split(",")
        missing = [c for c in wanted if c not in cases]
        if missing:
            print(f"unknown case ids: {missing}", file=sys.stderr)
            return 1
        cases = {k: cases[k] for k in wanted}

    out = TRACES_DIR / f"{args.label}.jsonl"
    done: set[str] = set()
    if out.exists():
        if args.resume:
            done = {t.key for t in read_traces(out)}
        elif args.overwrite:
            out.unlink()
        else:
            print(f"{out} exists; pass --resume or --overwrite", file=sys.stderr)
            return 1

    router = build_router(settings)
    digest = system_digest(router)
    truth = TruthDB(settings.clues_db_path)
    print(f"model={settings.gemini_model} instructions={digest} -> {out}\n")

    try:
        with judge_server(settings.judge_agent_url):
            for trial in range(1, args.trials + 1):
                for case in cases.values():
                    trace = Trace(
                        case_id=case["id"],
                        input=case["question"],
                        run_label=args.label,
                        trial=trial,
                        recorded_at=datetime.now(UTC).isoformat(timespec="seconds"),
                        model=settings.gemini_model,
                        instructions_digest=digest,
                    )
                    if trace.key in done:
                        continue
                    trace = await run_case(router, case["question"], trace)
                    append_trace(out, trace)
                    fails = [r for r in evaluate(case, trace, truth) if not r.passed]
                    mark = "ok  " if not fails else "FAIL"
                    detail = "; ".join(f"{r.check}: {r.reason}" for r in fails)
                    print(f"  {mark} {trace.key:34} -> {trace.delegated_to or '-':18} {detail}")
    except QuotaExhausted as exc:
        print(f"\nstopped: daily quota exhausted ({exc})", file=sys.stderr)
        print(f"kept what was recorded; continue with --label {args.label} --resume")
        return 2
    finally:
        await _close_toolsets(router)
    return 0


async def _close_toolsets(router) -> None:
    """Stop the MCP subprocess cleanly rather than on interpreter exit."""
    for agent in router.sub_agents:
        for tool in getattr(agent, "tools", None) or []:
            if hasattr(tool, "close"):
                with contextlib.suppress(Exception):
                    await tool.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--label", required=True, help="Names the output file: traces/<label>.jsonl"
    )
    parser.add_argument("--trials", type=int, default=1, help="Runs per case (default 1).")
    parser.add_argument("--cases", default=None, help="Comma-separated case ids (default: all).")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume", action="store_true", help="Skip cases already in the file.")
    group.add_argument("--overwrite", action="store_true", help="Replace an existing file.")
    return asyncio.run(record(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
