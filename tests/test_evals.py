"""The eval checks and trace capture, offline.

Each check is tested failing as well as passing: a check that has never been
seen to fail is not known to be able to. Traces are built by hand, and ADK
events are stand-ins with the same attributes, so nothing here calls a model.
"""

from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace

import pytest

from app.security import wrap_untrusted
from evals import checks
from evals.checks import (
    CATEGORIES,
    CHECKS,
    TruthDB,
    evaluate,
    load_cases,
    quoted_spans,
    tally,
    value_in,
)
from evals.trace import Trace, TraceBuilder, read_traces, write_traces

THAMES = {
    "clue_text": wrap_untrusted(
        "The section of this river near London Bridge is called the Pool", source="clue_archive"
    ),
    "correct_response": "the Thames",
    "category": "RIVERS",
    "air_date": "1997-04-28",
}
JORDAN = {
    "clue_text": wrap_untrusted("River mentioned most often in the Bible", source="clue_archive"),
    "correct_response": "the Jordan",
    "category": "LAKES & RIVERS",
    "air_date": "1984-09-10",
}


def search_trace(output: str, results=(THAMES, JORDAN), **kw) -> Trace:
    return Trace(
        case_id="c",
        input="clues about rivers",
        delegated_to="clue_search_agent",
        route=["jeopardy_router", "clue_search_agent"],
        tool_calls=[{"agent": "clue_search_agent", "name": "search_clues", "args": {}}],
        observations=[
            {
                "agent": "clue_search_agent",
                "name": "search_clues",
                "response": {"results": list(results), "result_count": len(results)},
            }
        ],
        output=output,
        **kw,
    )


def case(**kw) -> dict:
    return {"id": "c", "question": "clues about rivers", "route": ["clue_search_agent"], **kw}


def result(check, c, t, truth=None):
    return check(c, t, truth)


GOOD_THAMES = (
    'From RIVERS (1997): "The section of this river near London Bridge is called the Pool"'
    " -- correct response: the Thames."
)


# --------------------------------------------------------------------------
# completed / routed
# --------------------------------------------------------------------------
def test_completed_fails_on_a_silent_dead_end():
    r = checks.completed(case(), Trace(case_id="c", input="q", route=["r"]), None)
    assert not r.passed and "silent dead end" in r.reason


def test_completed_reports_the_error_when_there_is_one():
    t = Trace(case_id="c", input="q", error={"type": "ClientError", "message": "429 quota"})
    r = checks.completed(case(), t, None)
    assert not r.passed and "ClientError" in r.reason


def test_completed_passes_with_an_answer():
    assert checks.completed(case(), search_trace("hello"), None).passed


def test_no_error_leak_catches_a_relayed_429():
    """What the rate-limited A2A judge actually returned as its verdict."""
    leaked = (
        "On how to mitigate this issue, please refer to:\n\nhttps://google.github.io/...\n\n"
        "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded...'}}"
    )
    r = checks.no_error_leak(case(), search_trace(leaked), None)
    assert not r.passed and r.category == "error-as-answer"


def test_a_leaked_error_is_counted_once():
    t = search_trace("429 RESOURCE_EXHAUSTED. {'error': {'code': 429}}")
    failed = [r.check for r in evaluate(case(verdict="REJECT"), t, None) if not r.passed]
    assert failed == ["no_error_leak"]


def test_no_error_leak_allows_numbers_that_happen_to_match():
    t = search_trace("There are 429 clues worth $500 in the archive.")
    assert checks.no_error_leak(case(), t, None).passed


def test_routed_names_the_wrong_specialist():
    t = search_trace("x")
    t.delegated_to = "general_agent"
    r = checks.routed(case(), t, None)
    assert not r.passed and "general_agent" in r.reason


def test_routed_accepts_any_allowed_route():
    t = search_trace("x")
    assert checks.routed(case(route=["judge_agent", "clue_search_agent"]), t, None).passed


def test_routed_fails_when_the_router_answers_itself():
    t = Trace(case_id="c", input="q", route=["jeopardy_router"], output="an answer")
    assert "never delegated" in checks.routed(case(), t, None).reason


# --------------------------------------------------------------------------
# grounded
# --------------------------------------------------------------------------
def test_grounded_passes_a_real_quote():
    assert checks.grounded(case(), search_trace(GOOD_THAMES), None).passed


def test_grounded_fails_an_invented_quote():
    t = search_trace('Here is one: "This river flows through Paris and Rouen" (the Seine)')
    r = checks.grounded(case(), t, None)
    assert not r.passed and "Paris" in r.reason


def test_grounded_fails_a_quote_when_no_tool_was_called():
    t = search_trace('"This river flows through Paris and Rouen"')
    t.observations = []
    assert "no tool was called" in checks.grounded(case(), t, None).reason


def test_grounded_fails_an_invented_year():
    t = search_trace(GOOD_THAMES.replace("1997", "2003"))
    r = checks.grounded(case(), t, None)
    assert not r.passed and "2003" in r.reason


def test_grounded_allows_the_users_own_quote():
    c = case(question="Find clues like 'This planet is known as the Red Planet'")
    t = search_trace('You asked about "This planet is known as the Red Planet". Try Thames.')
    assert checks.grounded(c, t, None).passed


def test_grounded_reads_labelled_clue_lines():
    """The minimal agent's real output used 'Clue: ...' lines, not quotes."""
    t = search_trace("- Clue: This river flows through Paris and Rouen\n- Correct Response: Seine")
    assert not checks.grounded(case(), t, None).passed


def test_grounded_ignores_markdown_and_case():
    t = search_trace('**"the section of this river near London Bridge is called the Pool."**')
    assert checks.grounded(case(), t, None).passed


def test_grounded_skips_non_search_cases():
    assert checks.grounded(case(route=["general_agent"]), search_trace("x"), None) is None


def test_quoted_spans_ignores_short_quotes():
    assert quoted_spans('He said "yes" and "no way"') == []


def test_quoted_spans_does_not_pair_across_two_quotations():
    """Verbatim from the after-fix run, where this was flagged as a fabricated quote."""
    text = 'columns for "TikTok" (and variations such as "Tik Tok"). No clues were found.'
    assert quoted_spans(text) == []


# --------------------------------------------------------------------------
# admits_absence
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text",
    [
        "The archive has no clues about cryptocurrency.",
        "I couldn't find any clues about Bitcoin.",
        "There are none about the Kardashians in this archive.",
        "The archive does not contain clues mentioning Bitcoin.",
        # Verbatim from the baseline, where the digit form was first missed:
        'A search for "sudoku" returned **0 matching clues** out of the 4,001 clues.',
    ],
)
def test_admits_absence_passes_honest_answers(text):
    assert checks.admits_absence(case(expect_none=True), search_trace(text), None).passed


def test_admits_absence_is_not_fooled_by_zeros_inside_numbers():
    t = search_trace(f"From the 4,001 clues, here are clues about Minecraft: {GOOD_THAMES}")
    assert not checks.admits_absence(case(expect_none=True), t, None).passed


def test_admits_absence_fails_presenting_neighbours_as_matches():
    t = search_trace(f"Here are clues about cryptocurrency: {GOOD_THAMES}")
    r = checks.admits_absence(case(expect_none=True), t, None)
    assert not r.passed and r.category == "false-match"


# --------------------------------------------------------------------------
# clue_fields
# --------------------------------------------------------------------------
def test_clue_fields_passes_a_complete_clue():
    assert checks.clue_fields(case(expect_clue=True), search_trace(GOOD_THAMES), None).passed


@pytest.mark.parametrize(
    "drop,missing",
    [("the Thames", "correct response"), ("RIVERS", "category"), ("1997", "year")],
)
def test_clue_fields_names_the_missing_field(drop, missing):
    t = search_trace(GOOD_THAMES.replace(drop, ""))
    r = checks.clue_fields(case(expect_clue=True), t, None)
    assert not r.passed and missing in r.reason


def test_clue_fields_fails_when_no_clue_is_quoted():
    t = search_trace("The archive has several river clues.")
    assert "no clue" in checks.clue_fields(case(expect_clue=True), t, None).reason


def test_clue_fields_accepts_a_response_without_its_article():
    t = search_trace(GOOD_THAMES.replace("the Thames", "Thames"))
    assert checks.clue_fields(case(expect_clue=True), t, None).passed


# --------------------------------------------------------------------------
# stats_truth
# --------------------------------------------------------------------------
@pytest.fixture
def truth(tmp_path):
    path = tmp_path / "clues.sqlite3"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE clues (n INTEGER, d TEXT)")
    conn.executemany("INSERT INTO clues VALUES (?, ?)", [(i, "1984-09-10") for i in range(4001)])
    conn.commit()
    conn.close()
    return TruthDB(path)


def test_stats_truth_passes_with_a_thousands_separator(truth):
    c = case(truth_all=["SELECT COUNT(*) FROM clues"])
    assert checks.stats_truth(c, search_trace("There are 4,001 clues."), truth).passed


def test_stats_truth_fails_a_wrong_number(truth):
    c = case(truth_all=["SELECT COUNT(*) FROM clues"])
    r = checks.stats_truth(c, search_trace("There are 4,000 clues."), truth)
    assert not r.passed and "4001" in r.reason


def test_stats_truth_is_not_fooled_by_a_longer_number(truth):
    c = case(truth_all=["SELECT COUNT(*) FROM clues"])
    assert not checks.stats_truth(c, search_trace("There are 14001 clues."), truth).passed


def test_stats_truth_any_accepts_one_valid_reading(truth):
    c = case(truth_any=["SELECT 12", "SELECT 2"])
    assert (
        checks.stats_truth(c, search_trace("Two clues name him in the answer."), truth).passed
        is False
    )
    assert checks.stats_truth(c, search_trace("2 clues, searching correct_response"), truth).passed


def test_stats_truth_skips_without_the_archive(tmp_path):
    c = case(truth_all=["SELECT 1"])
    assert checks.stats_truth(c, search_trace("1"), TruthDB(tmp_path / "absent")) is None


@pytest.mark.parametrize(
    "text", ["from 1984-09-10", "from September 10, 1984", "from 10 September 1984"]
)
def test_value_in_reads_common_date_forms(text):
    assert value_in("1984-09-10", text)


# --------------------------------------------------------------------------
# judge
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "output", ["ACCEPT. Same river.", "**ACCEPT** same river", "Accept: same river"]
)
def test_verdict_format_accepts_a_leading_verdict(output):
    assert checks.verdict_format(case(verdict="ACCEPT"), search_trace(output), None).passed


def test_verdict_format_fails_a_buried_verdict():
    t = search_trace("Great question! I would ACCEPT that.")
    assert not checks.verdict_format(case(verdict="ACCEPT"), t, None).passed


def test_verdict_correct_fails_the_wrong_ruling():
    t = search_trace("ACCEPT - Jordan River contains Jordan.")
    r = checks.verdict_correct(case(verdict="REJECT"), t, None)
    assert not r.passed and "ruled ACCEPT, expected REJECT" in r.reason


def test_verdict_correct_fails_a_hedge():
    t = search_trace("ACCEPT or REJECT depending on the judges.")
    assert "both" in checks.verdict_correct(case(verdict="ACCEPT"), t, None).reason


# --------------------------------------------------------------------------
# ascii_only
# --------------------------------------------------------------------------
def test_ascii_only_fails_an_em_dash():
    r = checks.ascii_only(case(), search_trace("the Thames — in London"), None)
    assert not r.passed and "—" in r.reason


def test_ascii_only_allows_line_breaks_and_quotes():
    assert checks.ascii_only(case(), search_trace('Line one\n"Line" two'), None).passed


# --------------------------------------------------------------------------
# Whole-suite plumbing
# --------------------------------------------------------------------------
def test_every_check_has_a_category():
    assert {c.__name__ for c in CHECKS} == set(CATEGORIES)


def test_a_dead_end_runs_only_the_checks_that_can_judge_nothing():
    """No answer: `completed` and `routed` still say something; the rest don't pile on."""
    t = Trace(
        case_id="c",
        input="q",
        route=["jeopardy_router", "clue_search_agent"],
        delegated_to="clue_search_agent",
    )
    names = {r.check for r in evaluate(case(expect_clue=True), t, None)}
    assert names == {"completed", "routed"}  # no_error_leak needs an output too


def test_tally_counts_passes_and_fails():
    rows = checks.evaluate_file(
        [search_trace(GOOD_THAMES), search_trace("")], {"c": case(expect_clue=True)}, None
    )
    t = tally(rows)
    assert t["completed"] == {"pass": 1, "fail": 1}


def test_shipped_cases_are_well_formed():
    cases = load_cases()
    specialists = {"clue_search_agent", "clue_stats_agent", "general_agent", "judge_agent"}
    assert len(cases) >= 20
    for c in cases.values():
        assert c["question"].strip()
        assert set(c["route"]) <= specialists, c["id"]
        assert c.get("verdict") in (None, "ACCEPT", "REJECT"), c["id"]
        for sql in c.get("truth_all", []) + c.get("truth_any", []):
            assert sql.lstrip().upper().startswith("SELECT"), c["id"]


def test_every_route_is_covered_by_the_shipped_cases():
    routes = {r for c in load_cases().values() for r in c["route"]}
    assert routes >= {"clue_search_agent", "clue_stats_agent", "general_agent", "judge_agent"}


# --------------------------------------------------------------------------
# Trace capture from ADK-shaped events
# --------------------------------------------------------------------------
def event(author, *, calls=(), responses=(), text=None, final=False, usage=None, partial=False):
    parts = [SimpleNamespace(text=text, thought=False)] if text else []
    return SimpleNamespace(
        author=author,
        partial=partial,
        get_function_calls=lambda: [SimpleNamespace(name=n, args=a) for n, a in calls],
        get_function_responses=lambda: [SimpleNamespace(name=n, response=r) for n, r in responses],
        is_final_response=lambda: final,
        content=SimpleNamespace(parts=parts),
        usage_metadata=usage,
        error_code=None,
        error_message=None,
    )


def test_builder_captures_route_tools_and_answer():
    t = Trace(case_id="c", input="q")
    b = TraceBuilder(t)
    usage = SimpleNamespace(
        prompt_token_count=100, candidates_token_count=10, thoughts_token_count=50
    )
    b.record(event("user", text="q"))
    b.record(
        event(
            "jeopardy_router",
            calls=[("transfer_to_agent", {"agent_name": "clue_search_agent"})],
            usage=usage,
        )
    )
    b.record(event("clue_search_agent", calls=[("search_clues", {"query": "rivers"})], usage=usage))
    b.record(event("clue_search_agent", responses=[("search_clues", {"results": [THAMES]})]))
    b.record(event("clue_search_agent", text=GOOD_THAMES, final=True, usage=usage))

    assert t.route == ["jeopardy_router", "clue_search_agent"]
    assert t.delegated_to == "clue_search_agent"
    assert t.tools_used() == ["search_clues"]
    assert t.observations[0]["response"]["results"][0]["correct_response"] == "the Thames"
    assert t.output == GOOD_THAMES
    assert t.llm_calls == 3 and t.tokens == {"prompt": 300, "output": 30, "thoughts": 150}


def test_builder_keeps_observations_whole():
    """LoopLogger cuts at 300 chars; grounding needs the whole result."""
    long = {"results": [THAMES] * 20}
    t = Trace(case_id="c", input="q")
    TraceBuilder(t).record(event("a", responses=[("search_clues", long)]))
    assert len(t.observations[0]["response"]["results"]) == 20


def test_builder_ignores_partial_chunks():
    t = Trace(case_id="c", input="q")
    TraceBuilder(t).record(event("a", text="half", final=True, partial=True))
    assert t.output == ""


def test_builder_redacts_secrets_in_tool_args():
    t = Trace(case_id="c", input="q")
    key = "sk-proj-" + "a" * 40
    TraceBuilder(t).record(event("a", calls=[("search_clues", {"query": key})]))
    assert key not in json.dumps(t.tool_calls)


def test_builder_records_an_exception():
    t = Trace(case_id="c", input="q")
    TraceBuilder(t).fail(RuntimeError("boom"))
    assert t.error == {"type": "RuntimeError", "message": "boom"}


def test_jsonl_round_trip(tmp_path):
    path = tmp_path / "t.jsonl"
    original = search_trace(GOOD_THAMES, human_notes="looks right")
    write_traces(path, [original])
    (loaded,) = read_traces(path)
    assert loaded == original
    # One JSON object per line, with the four fields the course asks for.
    record = json.loads(path.read_text().splitlines()[0])
    assert {"input", "observations", "output", "human_notes"} <= set(record)
