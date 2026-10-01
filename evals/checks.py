"""The failure taxonomy, as code: one binary check per failure category.

Every check returns pass or fail with a one-line reason on fail, and nothing
in between. A check that doesn't apply to a case (a verdict check on a stats
question) is simply not run, not a third outcome.

| Category | Check | Fails when |
|---|---|---|
| dead-end | `completed` | the run ends with no answer, with or without an error |
| error-as-answer | `no_error_leak` | a provider or tool error is presented as the answer |
| misroute | `routed` | the router delegates somewhere the case doesn't allow |
| fabricated-content | `grounded` | a quoted clue or air year appears in no tool result |
| false-match | `admits_absence` | the topic isn't in the archive and the answer doesn't say so |
| incomplete-clue | `clue_fields` | a clue is shown without its response, category or year |
| wrong-stat | `stats_truth` | a count or date differs from the same SQL run by us |
| judge-format | `verdict_format` | the judge doesn't lead with ACCEPT or REJECT |
| wrong-verdict | `verdict_correct` | the judge's verdict is the wrong one |
| non-ascii | `ascii_only` | non-ASCII punctuation, which every agent is told not to use |

All of these are deterministic string, JSON or SQL comparisons. None asks a
model whether an answer is good; a grader that is itself non-deterministic
would add noise to exactly the measurement this is for.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from app.quality import escape_artifacts
from app.security import _FENCE
from evals.trace import Trace

CASES_PATH = Path(__file__).resolve().parent / "cases.jsonl"


@dataclass(frozen=True)
class CheckResult:
    check: str
    category: str
    passed: bool
    reason: str = ""  # one line, set only on fail


def _pass(check: str, category: str) -> CheckResult:
    return CheckResult(check, category, True)


def _fail(check: str, category: str, reason: str) -> CheckResult:
    return CheckResult(check, category, False, " ".join(reason.split())[:200])


# --------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------
def normalize(text: str) -> str:
    """Lowercase, punctuation to spaces, whitespace collapsed.

    Makes 'the Thames' match "The Thames." and survives markdown bold, but is
    still an exact word-sequence match, not fuzzy similarity.
    """
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def _unfence(text: str) -> str:
    """Strip the untrusted-content fence `search_clues` wraps clue text in."""
    lines = [ln for ln in text.splitlines() if _FENCE not in ln]
    return "\n".join(lines).strip()


def _strings(value: Any) -> Iterator[str]:
    """Every string and number inside a tool result, decoded, not JSON-escaped."""
    if isinstance(value, str):
        # MCP results arrive as JSON *inside* a text field; look inside it.
        stripped = value.strip()
        if stripped[:1] in "{[":
            try:
                yield from _strings(json.loads(stripped))
                return
            except ValueError:
                pass
        yield _unfence(value)
    elif isinstance(value, bool) or value is None:
        return
    elif isinstance(value, int | float):
        yield str(value)
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v)


def observed_text(trace: Trace) -> str:
    return normalize(" ".join(s for o in trace.observations for s in _strings(o.get("response"))))


def search_results(trace: Trace) -> list[dict[str, Any]]:
    """The clue rows `search_clues` returned in this run, fence removed."""
    rows = []
    for o in trace.observations:
        if o.get("name") != "search_clues" or not isinstance(o.get("response"), dict):
            continue
        for r in o["response"].get("results") or []:
            rows.append({**r, "clue_text": _unfence(r.get("clue_text") or "")})
    return rows


#: A double-quoted span of 4+ words, or the rest of a line labelled "Clue:".
#: The span must start and end on a non-space character. Otherwise, in
#: `for "TikTok" (and variations such as "Tik Tok")`, the too-short first
#: quote's *closing* mark pairs with the next opening one, and the text
#: between two quotations reads as a quotation itself.
_QUOTED = re.compile(r'"([^"\s][^"\n]{10,}[^"\s])"')
_LABELLED = re.compile(r"(?im)^[\s*>#-]*(?:\*\*)?clue(?:\s*text)?(?:\*\*)?\s*:\s*(?:\*\*)?(.+)$")


def quoted_spans(output: str) -> list[str]:
    spans = [m.group(1) for m in _QUOTED.finditer(output)]
    for m in _LABELLED.finditer(output):
        line = m.group(1)
        # "Clue: <text> - Correct Response: ..." on one line: keep the clue part.
        line = re.split(r"\s[-|]\s|\bcorrect response\b|\banswer\s*:", line, flags=re.I)[0]
        spans.append(line.strip(" \"'*"))
    return [s for s in spans if len(normalize(s).split()) >= 4]


_YEAR = re.compile(r"\b(19[89]\d|20[0-2]\d)\b")

#: "0" as a digit too: an honest SQL-backed answer said "returned **0 matching
#: clues**" and the first version of this pattern marked it a false match.
_ABSENCE = re.compile(
    r"\b(no|none|not|n't|zero|0|unable|couldn't|could not|doesn't|does not|did not|didn't)\b"
    r"[^.]{0,80}\b(clues?|results?|matches?|anything|find|found|contain|contains|have|has|"
    r"mention|related|about)\b",
    re.IGNORECASE,
)

#: Signatures of an error message standing in for an answer. Observed: a
#: rate-limited A2A judge relayed ADK's whole 429 text back as its verdict.
_ERROR_TEXT = re.compile(
    r"RESOURCE_EXHAUSTED|\b(429|500|503)\b[^\n]{0,40}\b(error|exhausted|unavailable)\b"
    r"|Traceback \(most recent call last\)|\{'error': \{|\"error\": \{"
    r"|On how to mitigate this issue",
    re.IGNORECASE,
)

_VERDICT = re.compile(r"\b(ACCEPT|REJECT)(?:ED)?\b")


# --------------------------------------------------------------------------
# Ground truth for stats cases
# --------------------------------------------------------------------------
class TruthDB:
    """Runs a case's SQL against the same SQLite archive the MCP agent queries."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def available(self) -> bool:
        return self.path.exists()

    def scalar(self, sql: str) -> Any:
        conn = sqlite3.connect(f"file:{self.path}?mode=ro&immutable=1", uri=True)
        try:
            row = conn.execute(sql).fetchone()
        finally:
            conn.close()
        return row[0] if row else None


def value_in(value: Any, output: str) -> bool:
    """Whether a truth value is stated in the answer, in any common rendering."""
    if isinstance(value, int):
        forms = {str(value), f"{value:,}"}
        return any(re.search(rf"(?<![\d,.]){re.escape(f)}(?![\d,])", output) for f in forms)
    text = str(value)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        d = date.fromisoformat(text)
        forms = {text, f"{d:%B} {d.day}, {d.year}", f"{d.day} {d:%B} {d.year}"}
        return any(f.lower() in output.lower() for f in forms)
    return normalize(text) in normalize(output)


# --------------------------------------------------------------------------
# The checks
# --------------------------------------------------------------------------
Check = Callable[[dict[str, Any], Trace, "TruthDB | None"], "CheckResult | None"]


def completed(case, trace, truth):
    name, cat = "completed", "dead-end"
    if trace.output.strip():
        return _pass(name, cat)
    if trace.error:
        return _fail(name, cat, f"no answer: {trace.error['type']}: {trace.error['message']}")
    return _fail(name, cat, "no answer and no error recorded (silent dead end)")


def answered(trace: Trace) -> bool:
    """A real answer exists. Content checks judge nothing else, so a dead end
    or a leaked error is counted once, under its own category, instead of
    also failing every check that would have read the answer."""
    return bool(trace.output.strip()) and not _ERROR_TEXT.search(trace.output)


def no_error_leak(case, trace, truth):
    name, cat = "no_error_leak", "error-as-answer"
    if not trace.output:
        return None
    if m := _ERROR_TEXT.search(trace.output):
        return _fail(name, cat, f"answer is an error message ({m.group(0)!r}), not an answer")
    return _pass(name, cat)


def routed(case, trace, truth):
    name, cat = "routed", "misroute"
    allowed = case["route"]
    if trace.delegated_to in allowed:
        return _pass(name, cat)
    if trace.delegated_to is None:
        if not trace.route:
            return _fail(name, cat, "no events at all, so nothing was routed")
        return _fail(name, cat, f"router never delegated (expected {' or '.join(allowed)})")
    return _fail(name, cat, f"went to {trace.delegated_to}, expected {' or '.join(allowed)}")


def grounded(case, trace, truth):
    """Nothing presented as archive content that the archive didn't return."""
    name, cat = "grounded", "fabricated-content"
    if "clue_search_agent" not in case["route"] or not answered(trace):
        return None
    seen = observed_text(trace)
    asked = normalize(case["question"])
    for span in quoted_spans(trace.output):
        n = normalize(span)
        if n in asked:
            continue  # the user's own words, quoted back
        if n not in seen:
            where = "no tool was called" if not trace.observations else "not in any tool result"
            return _fail(name, cat, f'quoted "{span[:70]}" but it is {where}')
    for year in sorted(set(_YEAR.findall(trace.output))):
        if year not in seen and year not in asked:
            return _fail(name, cat, f"states the year {year}, which no tool result contains")
    return _pass(name, cat)


def admits_absence(case, trace, truth):
    name, cat = "admits_absence", "false-match"
    if not case.get("expect_none") or not answered(trace):
        return None
    if _ABSENCE.search(trace.output):
        return _pass(name, cat)
    return _fail(name, cat, "topic is absent from the archive but the answer never says so")


def clue_fields(case, trace, truth):
    name, cat = "clue_fields", "incomplete-clue"
    if not case.get("expect_clue") or not answered(trace):
        return None
    out = normalize(trace.output)
    shown = [
        r for r in search_results(trace) if r["clue_text"] and normalize(r["clue_text"]) in out
    ]
    if not shown:
        return _fail(name, cat, "no clue from the search results is quoted in full")
    for r in shown:
        label = normalize(r["clue_text"])[:40]
        response = re.sub(r"^(the|a|an) ", "", normalize(r.get("correct_response") or ""))
        if response and response not in out:
            return _fail(name, cat, f'"{label}..." shown without its correct response')
        if r.get("category") and normalize(r["category"]) not in out:
            return _fail(name, cat, f'"{label}..." shown without its category')
        year = (r.get("air_date") or "")[:4]
        if year and year not in trace.output:
            return _fail(name, cat, f'"{label}..." shown without its year')
    return _pass(name, cat)


def stats_truth(case, trace, truth):
    name, cat = "stats_truth", "wrong-stat"
    if not (case.get("truth_all") or case.get("truth_any")) or not answered(trace):
        return None
    if truth is None or not truth.available():
        return None  # can't judge without the archive; not the agent's fault
    if sql_all := case.get("truth_all"):
        for sql in sql_all:
            v = truth.scalar(sql)
            if not value_in(v, trace.output):
                return _fail(name, cat, f"answer does not state {v!r} (from: {sql})")
        return _pass(name, cat)
    values = [truth.scalar(sql) for sql in case["truth_any"]]
    if any(value_in(v, trace.output) for v in values):
        return _pass(name, cat)
    return _fail(name, cat, f"answer states none of the valid readings {values}")


def _verdicts(output: str) -> list[str]:
    return [m.group(1) for m in _VERDICT.finditer(output)]


def verdict_format(case, trace, truth):
    name, cat = "verdict_format", "judge-format"
    if "verdict" not in case or not answered(trace):
        return None
    first_word = re.sub(r"^[\s*#>_`\"'-]+", "", trace.output).split(maxsplit=1)
    lead = first_word[0].strip("*:.,!_`\"'").upper() if first_word else ""
    if lead in {"ACCEPT", "REJECT"}:
        return _pass(name, cat)
    return _fail(name, cat, f"answer starts with {trace.output[:50]!r}, not ACCEPT or REJECT")


def verdict_correct(case, trace, truth):
    name, cat = "verdict_correct", "wrong-verdict"
    if "verdict" not in case or not answered(trace):
        return None
    found = set(_verdicts(trace.output))
    if found == {case["verdict"]}:
        return _pass(name, cat)
    if not found:
        return _fail(name, cat, f"no verdict found, expected {case['verdict']}")
    if len(found) > 1:
        return _fail(name, cat, f"says both ACCEPT and REJECT, expected {case['verdict']}")
    return _fail(name, cat, f"ruled {found.pop()}, expected {case['verdict']}")


def ascii_only(case, trace, truth):
    name, cat = "ascii_only", "non-ascii"
    if not answered(trace):
        return None
    # Only the two artifacts that are violations in a multi-line agent answer;
    # line breaks and quotes are legitimate here, unlike in `Answer.answer`.
    bad = [a for a in escape_artifacts(trace.output) if a.startswith(("non-ascii", "literal-\\u"))]
    if not bad:
        return _pass(name, cat)
    return _fail(name, cat, f"contains {', '.join(bad)}")


CHECKS: tuple[Check, ...] = (
    completed,
    no_error_leak,
    routed,
    grounded,
    admits_absence,
    clue_fields,
    stats_truth,
    verdict_format,
    verdict_correct,
    ascii_only,
)

#: check name -> failure category, for display.
CATEGORIES = {
    "completed": "dead-end",
    "no_error_leak": "error-as-answer",
    "routed": "misroute",
    "grounded": "fabricated-content",
    "admits_absence": "false-match",
    "clue_fields": "incomplete-clue",
    "stats_truth": "wrong-stat",
    "verdict_format": "judge-format",
    "verdict_correct": "wrong-verdict",
    "ascii_only": "non-ascii",
}


# --------------------------------------------------------------------------
# Running them
# --------------------------------------------------------------------------
def load_cases(path: Path = CASES_PATH) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    return {c["id"]: c for c in cases}


def evaluate(case: dict[str, Any], trace: Trace, truth: TruthDB | None) -> list[CheckResult]:
    results = []
    for check in CHECKS:
        if (r := check(case, trace, truth)) is not None:
            results.append(r)
    return results


@dataclass(frozen=True)
class Row:
    """One check applied to one trace: the unit the suite and dashboard count."""

    trace: Trace
    result: CheckResult


def evaluate_file(
    traces: list[Trace], cases: dict[str, dict[str, Any]], truth: TruthDB | None
) -> list[Row]:
    rows = []
    for t in traces:
        case = cases.get(t.case_id)
        if case is None:
            continue  # a case since removed from the set
        rows.extend(Row(t, r) for r in evaluate(case, t, truth))
    return rows


def tally(rows: list[Row]) -> dict[str, dict[str, int]]:
    """check -> {"pass": n, "fail": n}, in taxonomy order."""
    out = {c.__name__: {"pass": 0, "fail": 0} for c in CHECKS}
    for row in rows:
        out[row.result.check]["pass" if row.result.passed else "fail"] += 1
    return {k: v for k, v in out.items() if v["pass"] or v["fail"]}


def main() -> int:
    """Print the tally for one or more trace files, side by side."""
    import argparse  # noqa: PLC0415

    from app.config import Settings  # noqa: PLC0415
    from evals.trace import read_traces  # noqa: PLC0415

    parser = argparse.ArgumentParser(description="Tally eval checks for trace files.")
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--failures", action="store_true", help="List every failure.")
    args = parser.parse_args()

    cases = load_cases()
    truth = TruthDB(Settings().clues_db_path)
    tallies = {}
    for path in args.traces:
        rows = evaluate_file(read_traces(path), cases, truth)
        tallies[path.stem] = tally(rows)
        if args.failures:
            print(f"\n## failures in {path.stem}")
            for row in rows:
                if not row.result.passed:
                    print(f"  {row.trace.key:34} {row.result.check:16} {row.result.reason}")

    names = list(tallies)
    print(f"\n| check | category | {' | '.join(names)} |")
    print(f"|---|---|{'---|' * len(names)}")
    for check in (c.__name__ for c in CHECKS):
        cells = []
        for n in names:
            t = tallies[n].get(check)
            cells.append(f"{t['pass']}/{t['pass'] + t['fail']}" if t else "-")
        print(f"| {check} | {CATEGORIES[check]} | {' | '.join(cells)} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
