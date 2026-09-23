#!/usr/bin/env python3
"""Run the multi-agent system and show which specialist handled each query.

    # routing only -- no MCP subprocess, no second server needed
    python -m agents.run_demo --no-mcp --no-a2a

    # everything (start the judge first, in another terminal):
    #   uvicorn agents.judge_agent:app --port 8001
    python -m agents.run_demo

    python -m agents.run_demo --ask "what categories came up most in the 90s?"

Needs GOOGLE_API_KEY. ADK runs on Gemini; the OpenAI/Claude harness in `app/`
is a separate system and is not involved here.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from google.adk.agents.run_config import RunConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

load_dotenv()

from agents.system import build_router  # noqa: E402
from agents.trace_log import LoopLogger  # noqa: E402
from app.config import Settings  # noqa: E402

APP_NAME = "jeopardy_multi_agent"

#: Higher than the single agent's limit because a router spends a call
#: deciding before the specialist does any work, and a lookup-then-judge
#: query legitimately crosses two specialists. Still bounded: ADK's default
#: of 500 is not a safety net.
MAX_LLM_CALLS = 20

SCENARIOS = [
    ("SEMANTIC SEARCH", "Find me some clues about rivers or lakes."),
    ("SQL / MCP", "How many clues are in the archive, and what is the date range?"),
    (
        "JUDGE / A2A",
        "The clue was 'River mentioned most often in the Bible'. I said "
        "'Jordan River'. Is that correct?",
    ),
    ("GENERAL", "Explain what a large language model is in two sentences."),
]


async def ask(agent, message: str, *, echo: bool = True) -> tuple[str, list[str], LoopLogger]:
    """Run one query; return the answer, the agent trail, and the T/A/O trace.

    The author trail is what a routing demo needs that a single-agent demo
    does not: without it you cannot tell whether the router delegated or the
    root just answered by itself.
    """
    service = InMemorySessionService()
    runner = Runner(agent=agent, app_name=APP_NAME, session_service=service)
    session = await service.create_session(app_name=APP_NAME, user_id="user1")
    content = types.Content(role="user", parts=[types.Part(text=message)])

    answer = "(no response)"
    trail: list[str] = []
    logger = LoopLogger(echo=echo)
    try:
        async for event in runner.run_async(
            user_id="user1",
            session_id=session.id,
            new_message=content,
            run_config=RunConfig(max_llm_calls=MAX_LLM_CALLS),
        ):
            logger.record(event)
            author = getattr(event, "author", None)
            if author and (not trail or trail[-1] != author):
                trail.append(author)
            if event.is_final_response() and event.content and event.content.parts:
                text = event.content.parts[0].text
                if text:
                    answer = text
    except Exception as exc:  # noqa: BLE001 - surfaced, including the limit error
        if type(exc).__name__ == "LlmCallsLimitExceededError":
            answer = f"(stopped at the {MAX_LLM_CALLS}-call limit: {exc})"
        else:
            raise
    return answer, trail, logger


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ask", default=None, help="Run a single query instead of the scenarios.")
    parser.add_argument("--no-mcp", action="store_true", help="Omit the MCP specialist.")
    parser.add_argument("--no-a2a", action="store_true", help="Omit the remote judge.")
    args = parser.parse_args()

    settings = Settings()
    if settings.google_api_key is None:
        print(
            "GOOGLE_API_KEY is not set. ADK runs on Gemini; get a key at\n"
            "  https://aistudio.google.com/apikey\n"
            "then add GOOGLE_API_KEY=... to .env.",
            file=sys.stderr,
        )
        return 1

    router = build_router(settings, include_mcp=not args.no_mcp, include_a2a=not args.no_a2a)
    print(f"router: {router.name}")
    print(f"specialists: {', '.join(a.name for a in router.sub_agents)}\n")

    queries = [("ASK", args.ask)] if args.ask else SCENARIOS
    for label, query in queries:
        print(f"--- {label} ---")
        print(f"User: {query}\n")
        answer, trail, logger = await ask(router, query)
        print(f"\nRouted through: {' -> '.join(trail) or '(unknown)'}")
        print(f"Loop: {' -> '.join(logger.labels) or '(no events)'}")
        print(f"Agent: {answer}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
