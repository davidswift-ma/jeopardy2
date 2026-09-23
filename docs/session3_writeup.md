# Session 3 — Multi-Agent System Writeup

**Domain:** a Jeopardy clue archive (544,111 clues; a 4,001-row local sample
in these runs).
**Stack:** Google ADK, `gemini-3.6-flash` / `gemini-3.5-flash`.
**Code:** `agents/` in my Session 1/2 capstone. Run logs in `docs/runs/`.

---

## Architecture

```
                        jeopardy_router
                      (no tools, sub_agents only)
                              │
      ┌───────────────┬───────┴────────┬────────────────┐
      ▼               ▼                ▼                ▼
clue_search_agent  clue_stats_agent  general_agent   judge_agent
  local tools         MCP              no tools          A2A
      │               │                                  │
      ▼               ▼                                  ▼
  Chroma index   SQLite via our own              separate process
  (embeddings)   stdio MCP server                on :8001 over HTTP
```

| Agent | Transport | Handles |
|---|---|---|
| `clue_search_agent` | local `FunctionTool` | "clues about Norse mythology" — semantic |
| `clue_stats_agent` | **MCP** | "how many clues, what date range" — SQL |
| `general_agent` | none | anything not about the archive |
| `judge_agent` | **A2A** | "I said Jordan River — correct?" |

Routing is driven by each sub-agent's `description`, not its `instruction`.
The description tells the router *when to come here*; the instruction is that
specialist's own system prompt. A precise instruction behind a vague
description still routes badly.

`general_agent` is not filler. Without it the router forces "explain a large
language model" into a clue specialist and answers with five irrelevant
Jeopardy clues, which is worse than no retrieval at all. It is also how I
answered "when should we retrieve?" — the router decides, not a heuristic.

## MCP vs A2A, and why each

**MCP — `clue_stats_agent`.** I wrote my own stdio MCP server
(`agents/clue_mcp_server.py`) exposing `query_clues` and `describe_clues`
over a SQLite export, rather than using the demo's Supabase server. The clue
dataset asks not to be redistributed in a public-facing product and my repo
is public, so uploading it to a hosted database was not an option. A local
MCP server keeps the data on disk and still demonstrates what MCP is for: the
agent discovered both tools at runtime and wrote its own SQL.

**A2A — `judge_agent`.** Judging "is this contestant response correct?" is
genuinely separable: stateless, needs no index and no database, and it is the
piece you would scale or swap independently (a stricter judge, a
human-in-the-loop judge, a cheaper model). That makes a network boundary a
real architectural choice rather than A2A for its own sake.

The point the three layers make together: by the time you reach the router,
a local-tool agent, an MCP-backed agent and a remote HTTP agent are all just
entries in `sub_agents`. **The router's code does not change** as a
specialist moves from function call to subprocess to network service.

## The challenge I solved: a SQL tool the model cannot abuse

Giving a model a SQL tool means the *model* authors the SQL. `query_clues`
takes a string straight from an LLM and runs it against my database. Nothing
stops it writing `DROP TABLE clues` except what I build.

Three layers, in `agents/clue_mcp_server.py`:

1. **Read-only connection.** Opened `file:...?mode=ro&immutable=1`. Plain
   `mode=ro` still permits journal/WAL writes; `immutable=1` does not.
2. **Statement allowlist.** Only a single `SELECT` or `WITH` is accepted.
   Write keywords are rejected, and so is statement stacking
   (`SELECT 1; DROP TABLE clues`).
3. **Errors as observations, not exceptions.** A bad query returns
   `{"error": "SQL error: no such table: nope"}` — data the model can read
   and correct. Raising would just end the turn, which wastes the main
   advantage of giving a model SQL in the first place.

Ten tests cover it, including one that runs `DROP TABLE clues` and then
asserts the table still has its rows.

**A second thing worth recording: how I proved A2A.** A passing run proves
less than a failing one, because a local fallback would look identical in the
log. So I killed the judge process and re-ran the same query. It still routed
to `judge_agent`, then died at `All connection attempts failed` — no tool
call, no answer. `judge_response` is defined only in `agents/judge_agent.py`
and is never imported by the router's process, so the socket is the only
place it could have run.

## Evidence

Full logs in `docs/runs/`. The MCP run, condensed:

```
[ACT    ] jeopardy_router: transfer_to_agent({"agent_name": "clue_stats_agent"})
[ACT    ] clue_stats_agent: describe_clues({})
[OBSERVE] clue_stats_agent: {"table": "clues", "columns": [...]}
[ACT    ] clue_stats_agent: query_clues({"sql": "SELECT COUNT(*) AS total_clues,
           MIN(air_date) AS earliest_air_date, MAX(air_date) AS latest_air_date
           FROM clues"})
[OBSERVE] clue_stats_agent: {"rows": [{"total_clues": 4001,
           "earliest_air_date": "1984-09-10", "latest_air_date": "2026-07-23"}]}
[ANSWER ] 4,001 clues, 1984-09-10 to 2026-07-23
```

That matches the SQLite ground truth exactly, and no SQL was hand-written
anywhere in the path.

**This is an agent because** the model decides on its own to call a tool, and
the archive's reply — not its training data — is what the answer is built
from. `LoopLogger.proved_the_loop()` makes that checkable: it is true only
when a tool was proposed, returned a real result, and an answer followed *in
that order*. A model answering from memory fails it.

## Two notes for the cohort

- **`gemini-2.5-flash` 404s for new API keys**, which is the model the course
  demos use. It still appears in the `models` list, so listing a model is not
  proof you can call it. The error points at `gemini-3.6-flash`.
- **The free tier allows 20 generate requests per day, per model**
  (`GenerateRequestsPerDayPerProjectPerModel-FreeTier`). It is *per model*, so
  changing `GEMINI_MODEL` gives a fresh 20 — which is how I finished the MCP
  and A2A runs on the same afternoon.

Also: the sample repo's `demo3_full_system.py` imports `langfuse` and
`openinference.instrumentation.google_adk`, neither of which is in its
`pyproject.toml`, and `mcp` 2.x renamed `FastMCP` to `MCPServer`.
