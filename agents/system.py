"""The multi-agent system: three local specialists, one remote, one router.

Mapped to the session 3 requirements:

| Requirement | Here |
|---|---|
| 3+ specialists | clue_search, clue_stats, general, judge |
| Root router that delegates | `build_router` |
| At least one MCP integration | clue_stats -> our own SQLite MCP server |
| At least one A2A integration | judge -> `agents/judge_agent.py` over HTTP |

The shape worth noticing is the one the course demos build up to: by the time
you reach `build_router`, a local-tool agent, an MCP-backed agent and a
remote HTTP agent are all just entries in `sub_agents`. The router's code
does not change as the backing implementation moves from function call to
subprocess to network service.

**Routing is decided by each sub-agent's `description`, not its
`instruction`.** The description says *when to come here*; the instruction is
that specialist's own system prompt. A precise instruction with a vague
description still routes badly.

This package is deliberately separate from `app/`. ADK's `Runner` owns its
own orchestration, so it does not and should not share the retry, backoff and
failover machinery in `app/harness/orchestrator.py` -- two schedulers fighting
over one request is worse than either alone. The only thing shared is
`app/retrieval`.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from google.adk.agents import Agent, BaseAgent

from agents.tools import check_clue_index_status, search_clues
from app.config import Settings

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


def build_clue_search_agent(settings: Settings) -> Agent:
    """Semantic search over the vector index (local tools)."""
    return Agent(
        name="clue_search_agent",
        model=settings.gemini_model,
        description=(
            "Finds Jeopardy clues by topic or similarity, using semantic "
            "search. Use for fuzzy questions like 'clues about Norse "
            "mythology' or 'clues like this one', where the wording will not "
            "appear literally in the clue."
        ),
        instruction=(
            "You search a Jeopardy clue archive by meaning.\n"
            "Call search_clues with the user's topic. If it reports an error "
            "or returns nothing, call check_clue_index_status and tell the "
            "user precisely what is missing rather than saying 'no results'.\n"
            "When you present clues, always give both the clue text and the "
            "correct response, and say which category and year each came "
            "from. Use plain ASCII punctuation only."
        ),
        tools=[search_clues, check_clue_index_status],
    )


def build_clue_stats_agent(settings: Settings) -> Agent:
    """Structured/aggregate questions over SQLite, via MCP.

    The toolset is constructed lazily and the import is local, so importing
    this module does not require the `mcp` package or spawn a subprocess.
    """
    from google.adk.tools import McpToolset  # noqa: PLC0415
    from google.adk.tools.mcp_tool.mcp_session_manager import (  # noqa: PLC0415
        StdioConnectionParams,
    )
    from mcp.client.stdio import StdioServerParameters  # noqa: PLC0415

    clue_mcp = McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command=sys.executable,
                args=["-m", "agents.clue_mcp_server"],
                cwd=str(REPO_ROOT),
            ),
            timeout=30.0,
        ),
    )

    return Agent(
        name="clue_stats_agent",
        model=settings.gemini_model,
        description=(
            "Answers counting, ranking, filtering and date-range questions "
            "about the Jeopardy archive by writing SQL. Use for 'which "
            "categories came up most in the 90s', 'all $2000 opera clues', "
            "'how many clues mention Shakespeare'."
        ),
        instruction=(
            "You answer questions about a Jeopardy archive by writing SQL.\n"
            "Call describe_clues first if you are unsure of the schema, then "
            "query_clues with a single read-only SELECT.\n"
            "The column names matter: clue_text is what was read to "
            "contestants, correct_response is what they had to answer. A "
            "question like 'find clues about Jordan' is ambiguous -- say "
            "which column you searched.\n"
            "If a query returns an error, read it and fix your SQL rather "
            "than giving up. Use plain ASCII punctuation only."
        ),
        tools=[clue_mcp],
    )


def build_general_agent(settings: Settings) -> Agent:
    """General knowledge, no tools.

    Exists so the router has somewhere to send questions that have nothing to
    do with Jeopardy. Without it, 'explain AI to my grandfather' would be
    forced into a clue specialist and answered with five irrelevant clues --
    which is worse than no retrieval at all.
    """
    return Agent(
        name="general_agent",
        model=settings.gemini_model,
        description=(
            "Answers general knowledge and trivia questions from its own "
            "knowledge, with no archive lookup. Use for anything that is not "
            "about the Jeopardy clue archive itself."
        ),
        instruction=(
            "Answer the user's question directly and concisely from your own "
            "knowledge. Do not claim to have searched any archive. If you do "
            "not know, say so plainly rather than inventing detail. Use "
            "plain ASCII punctuation only."
        ),
    )


def build_judge_agent(settings: Settings) -> BaseAgent:
    """The remote judge, reached over A2A.

    Note what is absent: no tools, no model, no instruction. Those live in
    the other process. All this side knows is a URL and a description --
    which is why the return type is `BaseAgent`, not `Agent`: the router
    cannot tell the difference and neither should this signature.
    """
    from google.adk.agents.remote_a2a_agent import RemoteA2aAgent  # noqa: PLC0415

    return RemoteA2aAgent(
        name="judge_agent",
        agent_card=settings.judge_agent_url,
        description=(
            "Judges whether a contestant's answer to a clue is correct. Use "
            "when the user gives their own answer and wants it marked."
        ),
    )


def build_router(
    settings: Settings, *, include_mcp: bool = True, include_a2a: bool = True
) -> Agent:
    """Assemble the root router.

    `include_mcp` / `include_a2a` exist so the system is demonstrable in
    pieces: the MCP agent spawns a subprocess and the A2A agent needs another
    server running, and neither should be a precondition for showing that
    routing works.
    """
    sub_agents: list[BaseAgent] = [
        build_clue_search_agent(settings),
        build_general_agent(settings),
    ]
    if include_mcp:
        sub_agents.insert(1, build_clue_stats_agent(settings))
    else:
        logger.warning("MCP specialist omitted; clue_stats_agent will not be routable")
    if include_a2a:
        sub_agents.append(build_judge_agent(settings))
    else:
        logger.warning("A2A specialist omitted; judge_agent will not be routable")

    return Agent(
        name="jeopardy_router",
        model=settings.gemini_model,
        # No tools and no description: a router is defined by having
        # sub_agents, and nothing routes *to* the root.
        instruction=(
            "You route Jeopardy questions to the right specialist. Never "
            "answer directly.\n"
            "- clue_search_agent: finding clues by topic or similarity\n"
            "- clue_stats_agent: counts, rankings, filters, date ranges, "
            "anything needing SQL over the archive\n"
            "- judge_agent: the user gave their own answer and wants it "
            "marked correct or incorrect\n"
            "- general_agent: anything not about the clue archive\n"
            "If a question needs both a lookup and a judgement, delegate the "
            "lookup first, then the judgement."
        ),
        sub_agents=sub_agents,
    )
