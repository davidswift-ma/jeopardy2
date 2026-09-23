#!/usr/bin/env python3
"""The judge: a standalone agent exposed over A2A.

Run it in its own terminal, then the router can reach it:

    uvicorn agents.judge_agent:app --port 8001

Why this one is remote rather than a local sub-agent. Judging "is this
contestant response correct?" is a genuinely separable service: it is
stateless, it needs no clue index and no database, and it is the piece you
would scale or swap independently (a stricter judge, a human-in-the-loop
judge, a cheaper model). Splitting it across a network boundary is therefore
a real architectural choice rather than A2A for its own sake.

The router does not know any of this. It sees a name and a description, and
that is the point of A2A: `RemoteA2aAgent` in the router's `sub_agents` list
is indistinguishable from a local agent at the call site.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.agents import Agent

load_dotenv()

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
PORT = int(os.getenv("JUDGE_AGENT_PORT", "8001"))


def judge_response(clue_text: str, correct_response: str, contestant_response: str) -> dict:
    """Decide whether a contestant's response counts as correct for a clue.

    Jeopardy scoring is lenient about form and strict about substance:
    "the Jordan" and "Jordan River" are both correct, "Jordan" the country is
    not. Use this to compare a contestant's wording against the official
    response and return a verdict with a short reason.
    """
    expected = (correct_response or "").strip().lower()
    given = (contestant_response or "").strip().lower()

    # Cheap exact/containment checks first; the model handles everything else
    # and can override this by reading the returned `match_type`.
    for article in ("the ", "a ", "an "):
        expected = expected.removeprefix(article)
        given = given.removeprefix(article)

    if not given:
        return {"match_type": "empty", "exact": False, "note": "No response given."}
    if expected == given:
        return {"match_type": "exact", "exact": True, "note": "Identical after article removal."}
    if expected in given or given in expected:
        return {
            "match_type": "substring",
            "exact": False,
            "note": "One contains the other; likely correct but confirm it is the same referent.",
        }
    return {
        "match_type": "different",
        "exact": False,
        "note": "No textual overlap. Judge on meaning, not spelling.",
    }


judge_agent = Agent(
    name="judge_agent",
    model=MODEL,
    # This description is what the router reads to decide whether to delegate
    # here. Keep it about *when to use it*, not about how it works.
    description=(
        "Judges whether a contestant's answer to a Jeopardy clue is correct. "
        "Use when the user gives their own answer and wants it marked, or "
        "asks whether a specific response would have been accepted."
    ),
    instruction=(
        "You are a Jeopardy judge. Given a clue, the official correct "
        "response, and what the contestant said, rule whether it is "
        "acceptable.\n"
        "Call judge_response first to get a textual comparison, then apply "
        "judgement: Jeopardy accepts alternate phrasings, common nicknames, "
        "and missing articles, but not a different referent. Ignore whether "
        "they phrased it as a question.\n"
        "Reply with a verdict of ACCEPT or REJECT, then one sentence of "
        "reasoning. Use plain ASCII punctuation only."
    ),
    tools=[judge_response],
)

#: `to_a2a` wraps the agent in an ASGI app that serves the A2A protocol,
#: including the agent card at /.well-known/agent-card.json that
#: RemoteA2aAgent fetches to discover what this agent can do.
app = to_a2a(judge_agent, port=PORT)
