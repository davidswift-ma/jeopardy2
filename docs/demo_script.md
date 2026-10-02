# Five-minute demo script

Every requirement from weeks 1 to 5, in the order the course taught them.
Each step says what to **do**, what to **ask**, and what to **say**.

Total live cost: a few cents of API credit.

---

## Before class (15 minutes ahead)

**First, bring the public site back online.** It is kept switched off between
demos so it can't spend anything. In `~/git/jeopardy2`, run:

```bash
make deploy
```

It takes 2 to 3 minutes and ends with *Visit your newly deployed app at
https://jeopardy2-memory.fly.dev/*. The saved memories are still on the
volume, so nothing is lost. Docker doesn't need to be running: Fly builds the
image on its own servers.

Then open four terminal tabs, each in `~/git/jeopardy2`.

| Tab | Run | Leave it showing |
|---|---|---|
| 1 | `make dev` | the local app, at http://localhost:8000 |
| 2 | `make judge` | the A2A judge agent, on port 8001 |
| 3 | `make eval-dashboard` | the evals page, which opens at http://localhost:8501 |
| 4 | *(nothing yet)* | an empty prompt, ready for the agents demo |

Then set up three browser tabs:

1. **http://localhost:8000**: the local app.
2. **http://localhost:8501**: the evals dashboard. In its left sidebar, set
   **Trace file** to `baseline` and **Compare against** to `after-fix`.
3. **https://jeopardy2-memory.fly.dev**: the public site. Load it now, since it
   takes a few seconds to wake up. If the "What I remember about you" panel
   shows any facts, click **Forget everything**, so the demo starts empty.

> **Don't give the class the public URL until your demo is over.** The class
> shares one Wi-Fi address, and the site allows 6 questions a minute per
> address, so a room full of classmates would use up your demo's allowance.

---

## 0:00, start the slow part first

**Do:** in **tab 4**, run:

```bash
make agents-demo
```

**Say:** "This runs the multi-agent system from week 3. It takes about 40
seconds, so I'll show week 1 while it works and come back to it."

---

## 0:10, week 1: the harness

**Do:** go to browser tab 1 (localhost:8000). Click the example button
**capital of Australia**, then **Ask**.

**Ask:** *What is the capital of Australia, and why do people get it wrong?*

**Say:**
- "The *Harness progress* box streams live from the server. That's the
  **streaming** requirement."
- "The answer is a validated **Pydantic** object, not a string. It has a
  confidence score and a list of caveats."
- "The *Engine trace* table shows every provider call: which model served
  it, how long it took, and whether it retried."

**Do:** in **Force failure**, choose the **OpenAI, permanent** option
(immediate failover), then **Ask** again.

**Say:**
- "I've deliberately broken OpenAI. The harness sees the error can't be fixed
  by retrying, so it skips the 10- and 20-second **backoff** and **fails over
  to Claude**. You can see it in the trace: OpenAI fails, Claude succeeds."
- "If every provider fails, the user gets a polite *degraded* answer. The
  service **never returns a 500**."
- "The whole thing ships as a **Docker image**. A verification script checks
  17 things, including that the image contains no API keys."

---

## 1:20, weeks 2 and 3: RAG and multi-agent

**Do:** switch to **tab 4**. The demo should be finished. Scroll up to the
`router:` line.

**Say:**
- "One **router** agent hands each question to a **specialist**. The
  *Routed through* lines show the path each question took."
- **SEMANTIC SEARCH** (rivers or lakes): "This is the week 2 **RAG**. Each
  clue and its answer are stored as one chunk, embedded, and saved in
  **ChromaDB**. The agent finds clues by meaning, not by keyword."
- **SQL / MCP** (how many clues): "This specialist uses our own **MCP
  server**, which runs read-only SQL over the clue database. That's why the
  count is exact: 4,001."
- **JUDGE / A2A** ("Jordan River"): "The judge is a separate agent running
  in another process (tab 2), reached over **A2A**, the agent-to-agent
  protocol. It ruled ACCEPT."
- The *Loop:* lines: "Each line is the **Think / Act / Observe** loop. Every
  run has a 20-call **step limit**, so it can't loop forever."
- "**Security:** archive text is fenced off and treated as data rather than
  instructions. The SQL connection is read-only, and no agent has a tool that
  can send data out."

---

## 2:30, week 4: evals

**Do:** switch to browser tab 2 (localhost:8501), already showing
`baseline` compared with `after-fix`.

**Say:**
- "To prove the agents work despite randomness, I ran **31 fixed questions,
  3 times each**, and recorded every run as a JSONL trace."
- "**10 pass/fail checks**, one per failure category. They're plain code
  rather than a second AI grading the first, because a grader that is itself
  random would add noise."
- "The top real failure was **false matches**. Ask about the Kardashians,
  which aren't in the archive, and the agent offered unrelated clues as if
  they were about them. That passed **14 of 18** times."
- "One prompt fix took it to **18 of 18**, with no regressions anywhere else."

---

## 3:30, week 5: memory and the public deployment

**Do:** switch to browser tab 3 (the public fly.dev site). Click
**tell it about you**, then **Ask**.

**Ask:** *I'm cramming opera for a pub quiz, and I like short answers.*

**Say:** "Look at the *What I remember about you* panel. It just saved two
facts about me."

**Do:** open a **new browser tab** with the same URL. Click **quiz me**, then
**Ask**.

**Ask:** *Quiz me with one question.*

**Say:**
- "I never mentioned opera in this tab, but the quiz question is about opera,
  and it's short. That's **cross-session recall**."
- "The facts are stored in **SQLite on a Fly.io volume**, so they survive a
  restart. I tested that by restarting the server on Fly. A test in the repo
  does the same thing with two separate processes."
- "This is a **public HTTPS URL**. To protect my credit it runs on a separate,
  capped Gemini key, with a rate limit, a daily cap, and a robots.txt."
- "Memory is defended too. Stored facts are checked for injection attempts,
  and each user can only see their own."
- "The project's rules live in **CLAUDE.md**, so the coding agent re-reads
  them every session instead of losing them when a long chat gets
  summarized."

---

## 4:40, wrap up

**Say:** "That's all five weeks: the harness, RAG, multi-agent with MCP and
A2A, measured evals, and durable memory on a public URL. The code is public
at github.com/davidswift-ma/jeopardy2."

Now share the public URL with the class: **https://jeopardy2-memory.fly.dev**

---

## Shut down

| Tab | What to do |
|---|---|
| 1 (`make dev`) | **Ctrl+C** |
| 2 (`make judge`) | **Ctrl+C** |
| 3 (`make eval-dashboard`) | **Ctrl+C** |
| 4 (`make agents-demo`) | nothing; it already exited |

After you've shared the URL and the class has tried it, take the public
site offline again, so nothing can spend your credit until the next time:

```bash
fly scale count 0 --yes      # take it offline (saved memories are kept)
make deploy                  # bring it back, as in "Before class"
```

---

## If something goes wrong

- **Agents demo errors out:** a recorded run of each part is in `docs/runs/`
  (`routing_run.log`, `mcp_run.log`, `a2a_run.log`). Open one and talk through
  it.
- **The public site is slow to answer:** it was asleep. Wait about 5 seconds
  and ask again.
- **"Too many questions from you in the last minute":** the rate limit. Wait
  a minute, or keep talking and come back to it.
