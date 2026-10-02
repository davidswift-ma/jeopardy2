# CLAUDE.md

Rules for any coding agent working in this repository. They live here rather
than in chat because a long session gets summarized, and anything said only in
chat can be lost in the summary. This file is re-read at the start of every
session.

## Hard rules

1. **No secrets in chat, code, commits, or logs.** Keys live only in `.env`
   (gitignored) and, in production, in `fly secrets`. Never print a key, and
   never ask the user to paste one into the conversation; ask them to edit
   `.env` themselves. This repo is **public** on GitHub. Before every commit,
   check that nothing key-shaped or clue-shaped is staged.
2. **No clue data in git.** The Jeopardy dataset's terms forbid redistributing
   it. `data/*.tsv`, `data/*.sqlite3*`, `data/chroma/`, and `evals/traces/` stay
   ignored. The memory database is also personal data and stays ignored.
3. **Ask before committing, pushing, or deploying** (`git commit`, `git push`,
   `make deploy`, `fly ...`). Each one needs its own go-ahead.
4. **Ask before launching GUI applications** (Chrome, Docker Desktop, anything
   in /Applications). On this Mac they trigger permission prompts that are
   attributed to the IDE.
5. **Work in `~/git/jeopardy2`.** The copy at `~/git/projects/Jeopardy2` is
   stale.
6. **`make check` must pass** (ruff and pytest, offline, about 2s) before
   calling work done. Live API calls cost the user money: say what a live run
   will cost before making it.

## Where the requirements are

The grading rubric is the course notes in
`/Users/davidswift/jeopardy/jeopardy_clue_dataset/course_2_notes.txt`, plus
each week's assignment text the user pastes in. Read them before inferring
scope from the code. Standing requirements: the endpoint is `/jeopardy2`; it
returns Pydantic objects, never bare strings; it streams; it **never returns
a 5xx** (failures are a 200 with `status: "degraded"`); and it ships as a
Docker image.

## Invariants the code relies on

- `Answer` (app/schemas.py) fields are **all required, no defaults, no
  min/max**. OpenAI and Anthropic strict modes both depend on this, and
  `test_answer_schema_has_no_unsupported_keywords` enforces it.
- Engines own no retries. Every SDK's built-in retry is switched off; the
  harness owns the retry schedule.
- Memory is scoped by `user_id`. A user can only ever read, write, or delete
  their own facts. Remembered facts go through `app/memory.check_fact` on the
  way in and the way out.
- The public deployment runs Gemini only, on its own capped key, with the
  rate limit and daily cap on and fault injection off (`fly.toml`).

## About the user

David is a Java developer learning Python through this course. Explain the
Python idioms when they are not obvious, and prefer plain English over
jargon.
