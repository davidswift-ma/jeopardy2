# Session 4 — Evals: proving the agents work, measurably

**System under test:** the ADK multi-agent system from Session 3 (router,
clue search, SQL over MCP, remote A2A judge, general), on `gemini-3.6-flash`.
**Code:** `evals/`. Checks are unit-tested in `tests/test_evals.py`.

```
make eval-record LABEL=baseline TRIALS=3    # live, costs money
make eval TRACES=evals/traces/baseline.jsonl # pytest, free
make eval-report TRACES="evals/traces/baseline.jsonl evals/traces/after-fix.jsonl"
make eval-dashboard                          # Streamlit
```

---

## Method

- **31 fixed questions** (`evals/cases.jsonl`) across all four specialists,
  **3 trials each**, recorded as JSONL traces: input, route, every tool call
  and its *whole* result, output, errors, tokens, and a `human_notes` field.
  Six are traps: topics with zero clues in the 4,001-row sample.
- **10 binary checks**, one per failure category, each pass/fail with a
  one-line reason. All are deterministic string, JSON or SQL comparisons.
  None asks a model to grade, since a grader that is itself random would add
  noise to the very thing being measured.
- Recording and checking are separate, so a fixed or new check can be re-run
  on old traces for free. Every trace records a fingerprint of all agent
  instructions, so each number is tied to the prompts that produced it.

## Failure taxonomy

| Category | Check | Fails when |
|---|---|---|
| dead-end | `completed` | the run ends with no answer |
| error-as-answer | `no_error_leak` | a provider/tool error is presented as the answer |
| misroute | `routed` | the router delegates to a specialist the case doesn't allow |
| fabricated-content | `grounded` | a quoted clue or air year appears in no tool result |
| false-match | `admits_absence` | the topic isn't in the archive and the answer doesn't say so |
| incomplete-clue | `clue_fields` | a clue is shown without its response, category or year |
| wrong-stat | `stats_truth` | a count/date differs from the same SQL run against the archive |
| judge-format | `verdict_format` | the judge doesn't lead with ACCEPT or REJECT |
| wrong-verdict | `verdict_correct` | the judge's verdict is wrong |
| non-ascii | `ascii_only` | non-ASCII punctuation (all agents are told not to) |

`error-as-answer` was not in the original list. The first smoke run found it:
a rate-limited remote judge relayed ADK's raw `429 RESOURCE_EXHAUSTED` text
back as its verdict, and the `completed` check passed it because the answer
wasn't empty.

## Baseline, and the ranking

93 traces, 0 errors. Only one category failed:

| Failure | Frequency | Impact | Rank |
|---|---|---|---|
| **false-match** | **4/18 absent-topic runs (22%)**; Kardashians 3/3 | High: the user is misled with real but irrelevant clues presented as matches | **1: target** |
| error-as-answer | 1/3 in the smoke run on the free tier; 0/93 on paid | High, but a quota side effect | 2 |
| all others | 0 failures | — | — |

A false match is worse than "no results" because every quoted clue is *real*,
so it passes the grounding check and looks trustworthy. Asked about the
Kardashians, the agent offered *Toddlers & Tiaras*, Uday and Qusay Hussein,
and the Dionne quintuplets as "related to your search". The cause is
structural: semantic search always returns its five nearest neighbours, even
when none is about the topic.

## The fix

One rule in `clue_search_agent`'s instruction (`agents/system.py`): check that
each result is actually about the topic, and if none is, begin with *"The
archive has no clues about \<topic\>."* before offering any nearby clues,
labelled as not matching. A fixed opening sentence is easier for the model to
follow, and to check, than "be honest".

## Result: same 31 questions × 3 trials

| check | category | baseline | after-fix |
|---|---|---|---|
| completed | dead-end | 93/93 | 93/93 |
| no_error_leak | error-as-answer | 93/93 | 93/93 |
| routed | misroute | 93/93 | 93/93 |
| grounded | fabricated-content | 42/42 | 42/42 |
| **admits_absence** | **false-match** | **14/18** | **18/18** |
| clue_fields | incomplete-clue | 15/15 | 15/15 |
| stats_truth | wrong-stat | 18/18 | 18/18 |
| verdict_format | judge-format | 21/21 | 21/21 |
| verdict_correct | wrong-verdict | 21/21 | 21/21 |
| ascii_only | non-ascii | 93/93 | 93/93 |

The false-match rate went from **22% to 0%**, with no regressions. A stricter
rule risked the agent denying topics that *are* present, so the full set was
re-run, not just the traps. The fixed answers still show the nearest clues,
now labelled "none of which match the topic", which is more useful than a
bare refusal.

**How strong is this?** 4/18 → 0/18 has a one-sided Fisher exact p ≈ 0.05.
That's suggestive rather than conclusive on its own. The Kardashians question
alone went 3/3 → 0/3, and the after-fix answers were read by hand, not just
counted.

## What the evals caught in themselves

Three of the corrections were to *my* evals, not the agents, and each is
recorded in the code and tests:

1. **A case label was wrong.** "Are there any Bitcoin clues?" went to the SQL
   agent 2/3 times, which found 0 rows by exact search. For an "are there
   any" question that's the more rigorous route, so the case now allows it.
2. **The absence check missed "returned 0 matching clues"**, an honest
   answer. It now accepts the digit form.
3. **The quote extractor paired quotes across two quotations**
   (`for "TikTok" (and variations such as "Tik Tok")`) and called the text in
   between a fabricated quote. It now requires a span to start and end on a
   non-space character.

The first two were found by reading traces, not by the checks. Every failure
got read before it counted, and so did a sample of passes. A check that has
never been seen to fail isn't known to work, so each check has unit tests
that fail it deliberately.

## Cost

About $1.15 on Gemini for 198 recorded runs, including the smoke run. That
counts the router side only, since the remote judge's tokens are spent in its
own process. Thinking tokens are counted, and they run about 1,000 per
question even when the visible answer is short.

## Dashboard

`evals/dashboard.py` runs the real pytest suite (`evals/test_traces.py`) and
reads its JUnit report, so its numbers are pytest's. It compares two
recordings, lists every failure with its reason, shows each trace's route,
tools and raw observations, and saves human notes into the trace file.
Screenshot: `docs/evals/`.

Trace files are gitignored: they contain whole tool results, which means clue
text, and the dataset shouldn't be redistributed from a public repo.
