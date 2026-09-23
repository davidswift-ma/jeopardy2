#!/usr/bin/env python3
"""The minimal ADK agent: ONE agent, ONE tool, a step limit, a labelled loop.

    make agent-minimal
    python -m agents.minimal_agent --ask "find clues about Norse mythology"

This is deliberately *not* multi-agent. The job below is one multi-step task
-- look a clue up, then answer from what came back -- and it needs a tool,
not a router. The four-specialist system in `agents/system.py` is a separate,
larger thing; start here.

Patterns copied from the course sample, file by file:

| From | What was copied |
|---|---|
| `demo1_routing.py:60-65` | `Agent(name=, model=, description=, instruction=, tools=)` |
| `demo1_routing.py:20-27` | tools as plain functions returning `dict` |
| `demo1_routing.py:89-97` | the `Runner` + `InMemorySessionService` + `run_async` loop |
| `demo1_routing.py:99-108` | a `main()` that runs labelled scenarios |

What was *not* copied: the hardcoded dictionaries. The tool here hits the
real clue index. And two things the sample does not do are added below --
a step limit, and Think/Act/Observe labelling of the event stream.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.agents.run_config import RunConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

load_dotenv()

from agents.tools import search_clues  # noqa: E402
from agents.trace_log import LoopLogger  # noqa: E402
from app.config import Settings  # noqa: E402

APP_NAME = "jeopardy_minimal_agent"

#: The step limit. ADK's default is 500 LLM calls per run, which is not a
#: safety net -- a tool that keeps returning "no results" can loop until the
#: bill notices. Eight is enough for think -> search -> maybe re-search ->
#: answer, and small enough that a runaway stops in seconds.
MAX_LLM_CALLS = 8

INSTRUCTION = """\
You answer questions about a Jeopardy clue archive.

GOAL
Find real clues from the archive that match what the user asked about, and
answer using only what the archive returned.

HOW
Before you call a tool, say in one short sentence what you are about to look
up and why. Then call search_clues with a short topic phrase drawn from the
user's question. Read the results before answering.

CONSTRAINTS
- Never invent a clue, a response, a category or an air date. If the archive
  did not return it, you do not know it.
- If search_clues returns an `error` key, tell the user exactly what it says.
  Do not retry the same query more than once.
- At most two searches per question. If the second returns nothing useful,
  say so and stop.
- Use plain ASCII punctuation only.

DONE LOOKS LIKE
Either: a short answer quoting at least one real clue with its correct
response, category and year. Or: a plain statement that the archive has
nothing matching, or that it is unavailable and why. Both are complete
answers. Do not ask the user a follow-up question instead of finishing.
"""


def build_minimal_agent(settings: Settings) -> Agent:
    """One agent, one tool."""
    return Agent(
        name="jeopardy_clue_agent",
        model=settings.gemini_model,
        description="Answers questions about the Jeopardy clue archive by searching it.",
        instruction=INSTRUCTION,
        tools=[search_clues],
    )


async def run_once(question: str, settings: Settings, *, echo: bool = True) -> LoopLogger:
    """Run one task and return the labelled Think/Act/Observe trace."""
    service = InMemorySessionService()
    runner = Runner(agent=build_minimal_agent(settings), app_name=APP_NAME, session_service=service)
    session = await service.create_session(app_name=APP_NAME, user_id="user1")
    content = types.Content(role="user", parts=[types.Part(text=question)])

    logger = LoopLogger(echo=echo)
    try:
        async for event in runner.run_async(
            user_id="user1",
            session_id=session.id,
            new_message=content,
            # The step limit. Exceeding it raises rather than spinning.
            run_config=RunConfig(max_llm_calls=MAX_LLM_CALLS),
        ):
            logger.record(event)
    except Exception as exc:  # noqa: BLE001 - surfaced, including the limit error
        if type(exc).__name__ == "LlmCallsLimitExceededError":
            print(f"\n  [LIMIT   ] stopped after {MAX_LLM_CALLS} LLM calls: {exc}")
        else:
            raise
    return logger


#: A question that cannot be answered from the model's own memory: it has to
#: look in *this* archive to know what is in it.
DEFAULT_TASK = "What Jeopardy clues does the archive have about rivers? Quote one exactly."


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ask", default=DEFAULT_TASK)
    args = parser.parse_args()

    settings = Settings()
    if settings.google_api_key is None:
        print(
            "GOOGLE_API_KEY is not set. ADK runs on Gemini; get a key at\n"
            "  https://aistudio.google.com/apikey\n"
            "then add GOOGLE_API_KEY=... and RETRIEVAL_ENABLED=true to .env.",
            file=sys.stderr,
        )
        return 1
    if not settings.retrieval_enabled:
        print(
            "RETRIEVAL_ENABLED is false, so the one tool this agent has will "
            "report itself unavailable. Set RETRIEVAL_ENABLED=true in .env.",
            file=sys.stderr,
        )

    print(
        f"agent: jeopardy_clue_agent  model: {settings.gemini_model}  max_llm_calls: "
        f"{MAX_LLM_CALLS}"
    )
    print(f"task:  {args.ask}\n")

    logger = await run_once(args.ask, settings)
    print(f"\n{logger.summary()}")
    return 0 if logger.proved_the_loop() else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
