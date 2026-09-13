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

Re-measure before touching the prompt:

```bash
python scripts/probe_answer_quality.py --trials 12      # costs one API call per trial
```

A cautionary note on the detector in that script: its first version counted
only line breaks and reported 30%, under-reporting the true 87% by ~3x,
because line breaks were just one of five artifact forms. If you extend it,
check the bytes (`repr()`), not how the text looks.

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

### Design note for later

These clues are **structured** — round, category, dollar value, air date. Many
natural questions ("which categories came up most in the 90s?", "all $2000
opera clues") are SQL, not semantic search; others ("clues similar to this
one") genuinely need embeddings. The plan is a tool-using agent with both a
`query_clues` SQL tool over DuckDB/SQLite and an optional semantic search
tool — not a reflexive RAG pipeline over 544k rows.

---

## Development

```bash
make test      # 36 tests, no network, no real sleeping
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
  config.py             all tunables (models, retries, backoff, dataset path)
  schemas.py            Pydantic contracts: model-facing vs. telemetry
  harness/
    orchestrator.py     the provider chain: retry, backoff, failover
  engines/
    base.py             Engine protocol + the EngineError the harness classifies
    openai_engine.py    primary
    claude_engine.py    fallback
    faults.py           force_fail injection (ContextVar, per-request)
  static/index.html     the browser UI
scripts/make_sample.py  regenerate the committed dataset sample
```

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
