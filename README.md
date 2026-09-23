# Jeopardy2

A FastAPI agent with a resilient **harness**: OpenAI is the primary engine,
Claude is the fallback, every call is retried on a configured schedule, the
response is a validated Pydantic object rather than a string, and the service
never returns a 5xx.

> The model is the engine, the harness is the car.

---

## Quick start

### Docker (the shareable path)

Needs Docker Desktop and nothing else — no Python, no venv.

```bash
cp .env.example .env       # then add your own API keys to .env
docker compose up --build
```

Open <http://localhost:8000>.

Requires Docker Compose v2.24 or newer (Desktop 4.27+), for the
`env_file: required: false` key that lets the app start without a `.env`.
To check the whole deployment rather than just start it:

```bash
make verify-docker         # build, both CPU archs, no-.env case, live answer
```

### Local (for development)

Needs [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
make setup                 # venv + dependencies + creates .env
# add your API keys to .env
make dev                   # http://localhost:8000
```

`make help` lists every target.

### Keys

At least one of `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` is required. With only
one, the other provider is *skipped* cleanly and the app still answers — it
does not crash or hang. With neither, every request returns a `degraded`
response that says exactly which keys are missing.

---

## Using it

| Route | Purpose |
|---|---|
| `GET /` | Browser UI |
| `POST /jeopardy2` | Ask a question, wait for the answer |
| `GET /jeopardy2/stream` | Same, as Server-Sent Events with live progress |
| `GET /jeopardy2/config` | Effective configuration, secrets redacted |
| `GET /health` | Liveness |
| `GET /docs` | Auto-generated OpenAPI docs |

```bash
curl -X POST http://localhost:8000/jeopardy2 \
  -H 'Content-Type: application/json' \
  -d '{"question": "Explain AI in one or two sentences that my grandfather could understand."}'
```

```jsonc
{
  "status": "ok",
  "question": "Explain AI in one or two sentences...",
  "answer": {                        // a validated object, not a string
    "answer": "...",
    "confidence": 0.9,
    "caveats": []
  },
  "trace": {                         // how it got answered
    "attempts": [
      {"provider": "openai", "model": "gpt-5.5", "attempt": 1,
       "outcome": "success", "duration_ms": 1840, "slept_before_ms": 0}
    ],
    "served_by_provider": "openai",
    "served_by_model": "gpt-5.5",
    "used_fallback": false,
    "total_ms": 1843
  }
}
```

---

## How the harness works

```
request
   │
   ├─ openai      attempt 1 ─── fail ──▶ wait 10s
   │              attempt 2 ─── fail ──▶ wait 20s
   │              attempt 3 ─── fail
   │                   │
   │                   ▼  fall over
   ├─ anthropic   attempt 1 ─── fail ──▶ wait 10s
   │              attempt 2 ─── fail ──▶ wait 20s
   │              attempt 3 ─── fail
   │
   ▼
status: "degraded"  (HTTP 200, never 500)
```

Three attempts per provider with 10s then 20s between them — so ~30s worst
case per provider, ~60s across the chain. All of it is configuration
(`MAX_ATTEMPTS`, `BACKOFF_SECONDS`, `PROVIDER_ORDER`); none of it is hardcoded
at the call sites.

**Error classification.** Only *transient* failures are worth waiting on, so
they're the only ones retried:

| Retried (transient) | Fails over immediately (permanent) |
|---|---|
| 429 rate limit | 401 bad or missing API key |
| 5xx / 529 overloaded | 404 unknown model ID |
| Timeout | 400 malformed request |
| Connection error | 403 permission denied |
| | Refusal / unparsable structured output |

Without this, a typo in an API key would cost 30 seconds of sleeping before
the harness discovered that waiting cannot fix a typo. Set
`CLASSIFY_ERRORS=false` to retry everything uniformly instead.

**Structured output.** Both providers are asked for the *same* Pydantic
`Answer` schema, enforced provider-side — OpenAI via `responses.parse`,
Anthropic via `messages.parse`. That shared schema is what makes them
interchangeable: a client cannot tell which engine served a request except by
reading the trace.

**No 5xx.** Provider failures return a 200 with `status: "degraded"` and a
`message` explaining what happened. An unexpected exception anywhere is caught
and reported in the same envelope, so a client only ever parses one shape.
Malformed *input* is still a 422 from FastAPI's validation — that's a client
error with a precise message, and flattening it into a 200 would hide real
bugs.

**Streaming.** The backoff schedule can hold a request open for a minute, so
`/jeopardy2/stream` pushes progress events as they happen (`attempt_failed`,
`waiting`, `falling_back`) and ends with one terminal `result` event carrying
the full response. The UI renders these live. Both routes consume the same
generator, so their retry behaviour cannot drift apart.

---

## Output quality: a measured prompt rule

`SYSTEM_PROMPT` forbids non-ASCII punctuation. That rule is not stylistic —
it's the fix for a real, high-rate defect, and removing it brings the defect
back.

Against `claude-opus-5` with structured output, the model tried to write em
dashes inside its JSON string and mis-escaped them. A single em dash surfaced
as five distinct corruptions across runs:

| What landed in the `answer` field | |
|---|---|
| `—` | the literal characters, not a dash |
| a line break | mid-sentence |
| `\ndash` | newline followed by the word "dash" |
| `"` or `""` | stray quotes mid-sentence |
| `—` | correct (1 run in 8) |

Measured rate: **7 of 8 responses corrupted** without the rule, **0 of 12**
with it. The garbling was pure formatting — no repetition, no lost meaning —
and in one case the model flagged its own broken output in `caveats` and
supplied a clean rewrite, with self-reported confidence dropping to 0.6.

This is a known trait of the Opus 5 family: unusual Unicode escaping inside
structured-output JSON. Keeping the field values ASCII-only removes the
problem at the source.

**It is provider-specific.** Tested against `gpt-5.5` over 8 trials with the
rule removed: zero corruption. OpenAI emits correct curly apostrophes
(`U+2019`) that render fine. So the rule is load-bearing for Claude and
cosmetic for OpenAI, where it only standardizes apostrophes so output reads
the same whichever engine answered. Don't remove it because "OpenAI is fine" —
Claude isn't.

A caution about the measurement itself: the detector flags *any* non-ASCII
character, which means it reports OpenAI's perfectly good curly apostrophes as
artifacts. Genuine corruption (Claude) and harmless style (OpenAI) look
identical in the summary counts. Read the actual text, not just the tally.

Re-measure before touching the prompt. Both prompts that produced those
numbers are named variants in `app/prompts.py`, so the comparison is a flag
rather than a hand-edit:

```bash
make probe-compare              # control vs production, prints both rates
make probe TRIALS=12            # just the current prompt

# the same thing, spelled out
python scripts/probe_answer_quality.py --trials 8 --system-prompt no-ascii-rule
python scripts/probe_answer_quality.py --trials 12 --system-prompt ascii-guard
```

Costs one API call per trial (two per trial with `--compare`).

A cautionary note on the detector in that script: its first version counted
only line breaks and reported 30%, under-reporting the true 87% by ~3x,
because line breaks were just one of five artifact forms. If you extend it,
check the bytes (`repr()`), not how the text looks.

---

## Prompt versions and tracing

The measurement above used to require editing a constant and remembering to
put it back. Prompts are now data (`app/prompts.py`): each variant is named,
carries its own notes, and hashes to a digest that identifies it in a trace.

| Variant | Purpose |
|---|---|
| `ascii-guard` | Production. Includes the measured ASCII-punctuation rule. |
| `no-ascii-rule` | The control. Identical minus that rule, for reproducing the 7/8 result. |

The control differs from production in **exactly one dimension**, and a test
enforces that (`tests/test_prompts.py`). If the control were also reworded, a
change in corruption rate could not be attributed to the rule, which is the
whole claim. `PROMPT_VARIANT` selects one; an unknown name fails at startup
rather than silently falling back, because a silent default would report the
production rate under the control's name.

`GET /jeopardy2/config` reports the active variant and its digest. Compare
digests, not names — a name can be reused after an edit, a hash cannot.

### Langfuse

Optional and off by default.

```bash
make setup-obs                  # adds the langfuse dependency
# then in .env:
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=http://localhost:3000
```

Each request becomes **one trace**, with one span per retry attempt nested
inside it and a single generation for the call that actually answered. A
request that retries three times and then fails over is one trace with four
spans — not six top-level generations, which would double-count the failures
in every downstream metric.

Scores attached per trace: `served`, `used_fallback`, `attempts`,
`self_reported_confidence`, and `answer_clean` — the last computed by the
same detector `scripts/probe_answer_quality.py` uses (`app/quality.py`), so
live traffic and offline eval runs produce comparable numbers.

Three properties worth knowing:

- **Telemetry cannot break a request.** Missing key, unreachable collector,
  renamed SDK method — all degrade to a no-op plus a log line. There's a test
  that drives the whole harness with a tracer that raises on every call and
  asserts the answer still comes back. A monitoring outage becoming a
  user-visible failure is exactly backwards.
- **`app/harness/orchestrator.py` is untouched by any of this.** It already
  yields a full description of what happened, so tracing is a *consumer* of
  that generator (`app/obs/harness.py`), not an edit to it.
- **The SDK lives in one file.** Langfuse changed its tracing surface between
  v2 and v3; `app/obs/langfuse_tracer.py` detects which one is installed and
  normalizes both. Verify that file against the version you actually install
  — it is deliberately the only place that has to change.

Self-hosting is the default host. Traces carry your prompts, so pointing
`LANGFUSE_HOST` at a vendor cloud is a data decision, not just a config one.
Langfuse's own compose stack is several services; it is deliberately *not*
folded into this repo's `docker-compose.yml`, which stays a single service.

---

## Demonstrating the fallback

Prompt *content* cannot reliably make a provider fail — a refusal comes back
as a normal 200 and never reaches the retry path. What does fail reliably is a
bad key, an unknown model, a dead endpoint, or a timeout. So rather than
hunting for magic inputs, use the `force_fail` hook (gated by
`FAULT_INJECTION_ENABLED`):

```bash
# Watch 3 attempts, 10s + 20s of backoff, then the handoff to Claude
curl -N "http://localhost:8000/jeopardy2/stream?question=hi&force_fail=openai"

# Permanent error: immediate failover, no backoff at all
curl -N "http://localhost:8000/jeopardy2/stream?question=hi&force_fail=openai_permanent"

# Both engines down: ends degraded
curl -N "http://localhost:8000/jeopardy2/stream?question=hi&force_fail=both"
```

The UI has a dropdown for the same thing. Values: `openai`,
`openai_permanent`, `anthropic`, `both`.

---

## The Jeopardy dataset (phase 2)

Not wired into the agent yet; the config seam (`DATASET_PATH`) is in place.

**No clue data is committed to this repository.** The source dataset asks that
it not be used in a public-facing site, app, or product, and this repo is
public, so `data/` ships empty (see `data/README.md`). The agent's general
question-answering does not need it — everything in the Quick start above runs
without any download.

For the dataset-backed work, fetch it yourself:

```bash
# 1. Download from https://github.com/jwolle1/jeopardy_clue_dataset/releases
# 2. Either point at the full file (544,111 clues):
DATASET_PATH=/path/to/combined_season1-42.tsv
# 3. ...or generate a small local sample from your download:
make sample      # 4,001 clues, a deterministic every-136th stride across
                 # all 42 seasons; written to data/ and gitignored
```

### The column trap

In this dataset the names are **inverted** from intuition:

| Column | Actually contains |
|---|---|
| `answer` | the **clue read to contestants** |
| `question` | the **correct response** |

So the very first row is `answer = "River mentioned most often in the Bible"`,
`question = "the Jordan"`. A request to "find answers containing *Jordan*"
is therefore ambiguous — it matches the clue text under a literal reading, but
"the Jordan" is a *response*. Any dataset-facing code and prompt here should
use unambiguous names (`clue_text` / `correct_response`) rather than passing
`answer`/`question` through to a model and hoping.

### The vector index

Built, queryable, and **not yet wired into the prompt** — `/jeopardy2/config`
reports `"wired_into_prompt": false`. The agent still answers from general
knowledge.

```bash
make setup-rag          # adds chromadb
make index              # build (no-op if there is no clue data)
make index-status       # report state, change nothing

# exercise the whole pipeline offline, with no key and no cost
EMBEDDING_PROVIDER=hash RETRIEVAL_ENABLED=true make index
```

**Chunking is a no-op on this dataset, and that is the finding.** A clue
averages 28 tokens; the longest in a 4,001-row sample is 93, and zero rows
exceed 512. The row *is* the chunk — splitting would only risk severing a
clue from its response. What the config does expose is the *scheme*:

| `CHUNK_SCHEME` | Embedded text |
|---|---|
| `qa-glued` | category + clue + correct response in one chunk |
| `clue-only` | category + clue; the response stays as metadata |

`qa-glued` puts the answer inside the text you then search with the clue, so
retrieval scores well partly because you indexed the answer. `clue-only` is
the honest control for measuring retrieval quality. Both keep the response
on the chunk as metadata, so correctness can always be checked.

**Three states, and a fresh clone lands in the third.**

| State | Behaviour |
|---|---|
| Index present, fingerprint matches | Load. Near-instant. |
| Index absent, dataset present | `make index` builds it. |
| Index absent, **dataset absent** | Reports the reason; app answers normally. |

The third is the default, because `data/` ships empty — so it is what your
first run looks like. `make index` exits 0 and says "nothing to index"
rather than failing a build nobody asked for.

**The index is never committed and never built inside a request.** Chroma
stores document text next to each vector, so committing `data/chroma/` would
commit the clue text verbatim in a binary blob — it is gitignored, and the
ignore rule says why. Builds happen from `make index` or are reported at
startup; two uvicorn workers lazily building on first request would duplicate
the work and write to Chroma's SQLite concurrently.

**Staleness is detected, because the bad case is silent.** A manifest next to
the index records source size/mtime, row count, embedder ID, dimensions, and
chunk scheme. Changing *dimensions* fails loudly at query time and you fix it
in a minute. Swapping to a different model at the *same* dimensions returns
plausible-looking garbage with no error anywhere — which is why `embedder_id`
is compared, not just `dimensions`:

```
state:     stale -- index inputs changed
  changed: embedder_id: 'hash:64' -> 'openai:text-embedding-3-small:64'
```

**Cost.** Embedding is not the expensive part. The full 544,111 rows is
~15.4M tokens; the 4,001-row sample is ~112k. Generation per query costs far
more. `EMBEDDING_PROVIDER=hash` is free, offline and deterministic — and
produces meaningless geometry, so it is for tests and smoke runs only.

### Design note for later

These clues are **structured** — round, category, dollar value, air date. Many
natural questions ("which categories came up most in the 90s?", "all $2000
opera clues") are SQL, not semantic search; others ("clues similar to this
one") genuinely need embeddings. The plan is a tool-using agent with both a
`query_clues` SQL tool over DuckDB/SQLite and an optional semantic search
tool — not a reflexive RAG pipeline over 544k rows.

Those columns are already on every chunk as filterable metadata, which is the
half of that plan this phase delivers.

---

## The minimal ADK agent

Start here. One agent, one tool, a step limit, and a loop you can read.

```bash
make setup-adk                            # google-adk, a2a-sdk, mcp
# add GOOGLE_API_KEY and RETRIEVAL_ENABLED=true to .env
make index                                # the tool needs something to search
make agent-minimal
```

### Five bullets a beginner can repeat

1. **An agent is a model plus a loop.** It thinks, calls a tool, reads the
   result, and decides whether it is done — instead of answering in one shot.
2. **Tools are plain Python functions.** ADK reads the type hints for the
   parameter schema and the **docstring** for the description the model sees,
   so the docstring is interface, not commentary.
3. **The instruction has to say what "done" looks like**, or the model asks a
   clarifying question instead of finishing. Ours states a goal, constraints,
   and two acceptable endings.
4. **The loop needs a ceiling.** ADK allows 500 LLM calls by default;
   `RunConfig(max_llm_calls=8)` means a tool that keeps returning nothing
   stops in seconds rather than spinning.
5. **It is an agent only if the tool result changes the answer.** If the model
   could have replied from memory, you built a workflow with extra steps.

### Think / Act / Observe

ADK emits a flat event stream, not phases. `agents/trace_log.py` maps it:

| Event carries | Phase |
|---|---|
| Text, not final | **THINK** |
| `get_function_calls()` | **ACT** |
| `get_function_responses()` | **OBSERVE** |
| `is_final_response()` | **ANSWER** |

`LoopLogger.proved_the_loop()` turns "is this really an agent?" into an
assertion: it is true only when a tool was proposed, returned a real result,
and an answer followed *in that order*. A model answering from memory fails
it. So does a tool call that never reaches an answer.

**This is an agent because** the model decides on its own to call
`search_clues`, and the archive's reply — not its training data — is what the
final answer is built from. Ask it what clues *this* archive holds and it
cannot fake the answer.

**Stack, in one word:** ADK.

### Patterns copied from the course sample

| From `adk-multi-agent-systems/` | What was taken |
|---|---|
| `demo1_routing.py:60-65` | `Agent(name=, model=, description=, instruction=, tools=)` |
| `demo1_routing.py:20-27` | tools as plain functions returning `dict` |
| `demo1_routing.py:89-97` | `Runner` + `InMemorySessionService` + `run_async` |
| `demo1_routing.py:81-85` | `sub_agents` routing (used by the larger system below) |
| `shipping_agent.py:39-49` | `to_a2a(agent, port=...)` for the remote agent |
| `demo2_mcp.py:33-47` | `McpToolset` + `StdioConnectionParams` |

Not taken: the hardcoded dictionaries. Every tool here returns real data.
Added beyond the sample: the step limit and the Think/Act/Observe labelling.

---

## Multi-agent system (ADK)

A second, **separate** system living in `agents/`: a Gemini-based router that
delegates to four specialists, one reached over MCP and one over A2A.

```bash
make setup-adk                 # google-adk, a2a-sdk, mcp
# add GOOGLE_API_KEY to .env   # https://aistudio.google.com/apikey
make clues-db                  # export clues to SQLite for the MCP server

make agents-routing            # routing only, nothing else needed
make judge                     # terminal 1: the A2A agent on :8001
make agents-demo               # terminal 2: the full system
```

| Specialist | Backed by | Handles |
|---|---|---|
| `clue_search_agent` | local tools → Chroma | "clues about Norse mythology" |
| `clue_stats_agent` | **MCP** → SQLite | "which categories came up most in the 90s" |
| `general_agent` | nothing | anything not about the archive |
| `judge_agent` | **A2A** → `:8001` | "I said Jordan River. Correct?" |

Three things worth knowing:

**Routing is decided by `description`, not `instruction`.** The description
tells the router *when to come here*; the instruction is that specialist's
own system prompt. A precise instruction behind a vague description still
routes badly.

**The router cannot tell local from remote.** `judge_agent` is a
`RemoteA2aAgent` and the rest are `LlmAgent`s, but all four are just entries
in `sub_agents`. That is the point of the MCP/A2A layering: the router's code
does not change as a specialist moves from function call to subprocess to
network service.

**`general_agent` exists to protect the general path.** Without it, "explain
AI to my grandfather" gets forced into a clue specialist and answered with
five irrelevant Jeopardy clues, which is worse than no retrieval at all. This
is the routed answer to "when should we retrieve?" — the router decides, not
a heuristic.

### Proof, not description

Every claim above was run against live Gemini. Logs in `docs/runs/`.

| Path | Evidence |
|---|---|
| Single agent loop | `THINK -> ACT -> OBSERVE -> ANSWER`, real clue quoted |
| Routing | `transfer_to_agent(clue_search_agent)`, author switches |
| General path protected | "explain a large language model" → `general_agent` |
| **MCP** | agent called `describe_clues`, then wrote **its own SQL** |
| **A2A** | router → HTTP → remote process → its own tool → verdict |

The MCP result is worth reading closely. The agent discovered the schema at
runtime and composed this itself:

```sql
SELECT COUNT(*) AS total_clues, MIN(air_date) AS earliest_air_date,
       MAX(air_date) AS latest_air_date FROM clues
```

It returned 4,001 clues spanning 1984-09-10 to 2026-07-23 — matching the
SQLite ground truth exactly. No SQL was written by hand anywhere.

The A2A proof is a **negative control**, because a passing test proves less
than a failing one here. With the judge process killed, the same query still
routes to `judge_agent` and then dies at `All connection attempts failed` —
no tool call, no answer. `judge_response` is defined only in
`agents/judge_agent.py` and is never imported by the router's process, so
the only place it could have run is the other side of the socket.

### Quotas, and a dead model

Two things the live runs taught, both worth knowing before you burn an
afternoon:

- **The free tier allows 20 generate requests per day, per model.** The
  quota ID is `GenerateRequestsPerDayPerProjectPerModel-FreeTier`. Because
  it is *per model*, switching `GEMINI_MODEL` gives you a fresh 20 — which
  is how the MCP and A2A runs above got finished.
- **`gemini-2.5-flash` 404s for new keys.** It still appears in the models
  list, so listing a model is not proof you can call it. The API's own error
  points at `gemini-3.6-flash`. Same lesson as `OPENAI_MODEL`: a 404 is
  almost always a stale ID, not a bug.

### Security

Full threat model in [`docs/threat_model.md`](docs/threat_model.md). The
short version, because the honest framing matters more than the controls:
**prompt injection has no complete solution**, and two architectural facts
are doing the real work — there is **no egress tool** anywhere in `agents/`,
and the SQL path is `mode=ro&immutable=1`. Everything else is depth.

`agents/` is the first place in this project where untrusted retrieved text
reaches a model holding tools; the FastAPI path still reports
`"wired_into_prompt": false`.

| Surface | Control |
|---|---|
| Archive text in context | fenced with an unforgeable sentinel, scanned, flagged to the model |
| Model-authored SQL | read-only connection; `ATTACH`/`load_extension`/stacking all tested and blocked |
| Output rendered in a UI | `scan_for_exfiltration` catches the markdown-image channel; non-zero exit |
| A2A endpoint | `to_a2a` has **no auth**; binding to `0.0.0.0` raises unless opted in |
| Committed run logs | `redact_secrets` runs inside the trace formatter |
| Runaway loops | `max_llm_calls` 8 / 20 against ADK's default of 500 |

```bash
make injection-probe        # control vs defences, canary-based
```

Measured so far: with `gemini-3.5-flash-lite`, n=1 per arm, the naive
injection failed against **both** arms — so this does not yet show the
defences change the outcome, only that the defended run reported the
attempt. n=1 is not a measurement; it is a harness that works.

### Why this is separate from `app/`

ADK's `Runner` owns its own orchestration: retry, delegation, tool loops.
That is the same job `app/harness/orchestrator.py` does, and two schedulers
fighting over one request is worse than either alone. So `agents/` is a
sibling, not a replacement — nothing in `app/` imports it, the FastAPI
service is untouched, and the only shared code is `app/retrieval`. The
`[adk]` extra keeps the dependency out of the base install.

The tradeoff is real and deliberate: this system does **not** get the no-5xx
guarantee, the backoff schedule, or the provider failover. It runs on Gemini,
a third provider.

### Our own MCP server

`agents/clue_mcp_server.py` speaks MCP over stdio and exposes `query_clues`
and `describe_clues` against the SQLite export. Written rather than borrowed
because the course demo's Supabase server would mean uploading clue data to a
hosted database, which is the thing this repo does not do.

The model writes its own SQL, so the server is **read-only by construction**:
the connection is opened `mode=ro&immutable=1`, only a single `SELECT`/`WITH`
is accepted, and write keywords are rejected. A malformed query comes back as
an error string the model can read and correct, not an exception that ends
the turn. Tests cover `DROP`, `DELETE`, `UPDATE`, `INSERT`, `PRAGMA` and
statement stacking.

### A note on the course materials

The bootcamp demos are written against `google-adk` 1.x. On 2.9.2 the MCP
import path has moved:

```python
# demos (google-adk 1.x)
from google.adk.tools.mcp_tool import McpToolset, StdioConnectionParams

# google-adk 2.9.2
from google.adk.tools import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
```

`demo3_full_system.py` also imports `langfuse` and
`openinference.instrumentation.google_adk`, neither of which is in that
project's `pyproject.toml`, and `mcp` 2.x renamed `FastMCP` to `MCPServer`.

---

## Development

```bash
make test      # 179 tests, no network, no real sleeping
make lint      # ruff check + format --check
make check     # both
```

Tests use a scriptable `FakeEngine` and a millisecond backoff, so the retry
logic under test is the same code production runs — only the numbers differ.
The suite covers retry counts, classification, failover, skipping an
unconfigured provider, SSE framing, and the no-5xx guarantee.

### Layout

```
app/
  main.py               FastAPI routes, SSE framing, the no-5xx handler
  config.py             all tunables (models, retries, backoff, prompt, dataset)
  schemas.py            Pydantic contracts: model-facing vs. telemetry
  prompts.py            named, digest-identified system prompt variants
  quality.py            the corruption detector, shared by probe and live path
  harness/
    orchestrator.py     the provider chain: retry, backoff, failover
  engines/
    base.py             Engine protocol + the EngineError the harness classifies
    openai_engine.py    primary
    claude_engine.py    fallback
    faults.py           force_fail injection (ContextVar, per-request)
  obs/
    base.py             Tracer protocol, NullTracer, the never_raises guard
    langfuse_tracer.py  the only module importing the Langfuse SDK
    harness.py          traces run_agent by consuming it, without modifying it
  retrieval/
    base.py             Embedder/ClueStore protocols, the staleness manifest
    chunks.py           TSV -> chunks; where the inverted columns are renamed
    embedders.py        openai (real) and hash (offline, free, meaningless)
    chroma_store.py     the only module importing chromadb
    index.py            the build/stale/ready state machine
  static/index.html     the browser UI
agents/
  minimal_agent.py      ONE agent, ONE tool, step limit, T/A/O logging
  trace_log.py          maps ADK events to Think / Act / Observe
  system.py             router + specialists; MCP and A2A wiring
  tools.py              local tools (semantic clue search)
  clue_mcp_server.py    our own MCP server: read-only SQL over the clues
  judge_agent.py        the standalone A2A agent (run on :8001)
  run_demo.py           runnable demo, prints which specialist handled what
scripts/
  make_sample.py           generate a local sample from your own download
  build_index.py           build/inspect the vector index
  export_clues_db.py       TSV -> SQLite, for the MCP server
  probe_answer_quality.py  measure the non-ASCII corruption rate (uses API calls)
  verify_docker.sh         the deployment check behind `make verify-docker`
tests/
  conftest.py              FakeEngine, RecordingTracer, env isolation
  test_api.py              routes, SSE framing, the no-5xx guarantee
  test_engines.py          engine adapters and error classification
  test_harness.py          retry counts, backoff, failover
  test_obs.py              trace shape, scoring, telemetry-cannot-break-a-request
  test_prompts.py          prompt registry invariants and the one-dimension rule
  test_retrieval.py        the column trap, chunk schemes, staleness, states
  test_agents.py           router graph, MCP read-only guarantee, judge tool
  test_trace_log.py        the T/A/O mapping and what "proved the loop" means
```

Tests construct `Settings()` with `.env` and the shell environment disabled
(an autouse fixture in `conftest.py`). Without that the suite is
machine-dependent: a test asserting "missing `OPENAI_API_KEY` is reported"
passes on CI and fails for anyone who has actually configured the app.

### Model IDs

`OPENAI_MODEL` and `ANTHROPIC_MODEL` are configuration because providers
rename and retire models. Defaults are `gpt-5.5` and `claude-opus-5`. **A 404
from a provider almost always means a stale model ID in `.env`, not a code
bug.**

### Cost

Measured, not estimated: one request is ~320 input and ~80 output tokens
(163 of the input is the system prompt, 188 the JSON schema).

| Model | Per call | Calls per $1 |
|---|---|---|
| `gpt-5.5` ($5 / $30 per 1M) | ~$0.0039 | ~256 |
| `gpt-5.4-mini` ($0.75 / $4.50 per 1M) | ~$0.0006 | ~1,600 |

`gpt-5.5` used **zero** reasoning tokens on simple questions, so the GPT-5
reasoning-token surcharge did not materialize for this workload. Set
`OPENAI_MODEL=gpt-5.4-mini` to develop at roughly a sixth the cost.

### A note on quota errors

`credit_balance_exhausted` (HTTP 429) is classified **permanent**, so it fails
over immediately rather than spending 30s of backoff rediscovering that an
account has no money. Caveat found in practice: for roughly an hour after a
credit top-up, OpenAI returned that error *intermittently* (~1 call in 3) on
an account that did have credits, and a retry would have succeeded. During
such a window these requests fail over to Claude unnecessarily. The behaviour
is still correct — you get an answer either way — but if quota errors ever look
flaky rather than absolute, this is why.
