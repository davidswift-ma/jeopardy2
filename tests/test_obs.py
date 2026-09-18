"""Observability: trace shape, scoring, and the rule that telemetry cannot break a request.

No network and no Langfuse SDK required -- the tracer is a protocol, and
these exercise the same `traced_run_agent` wrapper the routes use.
"""

from __future__ import annotations

import pytest

from app.obs.base import NullTracer, never_raises
from app.obs.harness import traced_run_agent
from app.quality import diagnose, escape_artifacts, is_degenerate
from app.schemas import AgentResponse, Answer
from tests.conftest import (
    ExplodingTracer,
    FakeEngine,
    RecordingTracer,
    ok_answer,
    permanent,
    transient,
)


async def drain(gen) -> AgentResponse:
    final = None
    async for item in gen:
        if isinstance(item, AgentResponse):
            final = item
    assert final is not None
    return final


# --------------------------------------------------------------------------
# Trace shape
# --------------------------------------------------------------------------
async def test_one_trace_per_request_not_per_attempt(fast_settings):
    """A retrying request must produce one trace, not one per provider call.

    Six top-level generations for one question would double-count failures
    in every quality metric downstream.
    """
    tracer = RecordingTracer()
    engines = [
        FakeEngine("openai", script=[transient("openai")] * 3),
        FakeEngine("anthropic", script=[ok_answer()]),
    ]
    await drain(traced_run_agent("q", fast_settings, tracer, engines=engines))

    assert len(tracer.traces) == 1
    trace = tracer.traces[0]
    # 3 failed openai attempts + 1 successful anthropic attempt
    assert len(trace.spans) == 4
    assert [s["name"] for s in trace.spans] == [
        "attempt:openai#1",
        "attempt:openai#2",
        "attempt:openai#3",
        "attempt:anthropic#1",
    ]
    # Exactly one generation: the call that actually produced the answer.
    assert len(trace.generations) == 1


async def test_trace_records_fallback_and_closes(fast_settings):
    tracer = RecordingTracer()
    engines = [
        FakeEngine("openai", script=[permanent("openai")]),
        FakeEngine("anthropic", script=[ok_answer("hello")]),
    ]
    await drain(traced_run_agent("q", fast_settings, tracer, engines=engines))

    trace = tracer.traces[0]
    assert trace.ended
    assert trace.end_metadata["used_fallback"] is True
    assert trace.end_metadata["served_by_provider"] == "anthropic"
    assert trace.scores["used_fallback"] == 1.0
    assert trace.scores["served"] == 1.0


async def test_degraded_request_scores_zero_and_still_traces(fast_settings):
    tracer = RecordingTracer()
    engines = [
        FakeEngine("openai", script=[permanent("openai")]),
        FakeEngine("anthropic", script=[permanent("anthropic")]),
    ]
    final = await drain(traced_run_agent("q", fast_settings, tracer, engines=engines))

    assert final.status == "degraded"
    trace = tracer.traces[0]
    assert trace.scores["served"] == 0.0
    assert "answer_clean" not in trace.scores  # nothing to score
    assert trace.ended


async def test_prompt_variant_is_in_trace_metadata(fast_settings):
    """Without this a trace cannot tell you which prompt produced the answer."""
    tracer = RecordingTracer()
    engines = [FakeEngine("openai", script=[ok_answer()])]
    await drain(traced_run_agent("q", fast_settings, tracer, engines=engines))

    assert tracer.traces[0].metadata["prompt_variant"] == "ascii-guard"


async def test_abandoned_request_still_closes_its_trace(fast_settings):
    """The SSE route abandons the generator when a client disconnects.

    An unclosed trace would stay open forever and skew duration statistics.
    """
    tracer = RecordingTracer()
    engines = [FakeEngine("openai", script=[ok_answer()])]

    gen = traced_run_agent("q", fast_settings, tracer, engines=engines)
    await gen.__anext__()  # consume only the "started" event
    await gen.aclose()

    trace = tracer.traces[0]
    assert trace.ended
    assert trace.end_metadata["aborted"] is True


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
async def test_clean_answer_scores_one(fast_settings):
    tracer = RecordingTracer()
    engines = [FakeEngine("openai", script=[ok_answer("A plain ASCII answer.")])]
    await drain(traced_run_agent("q", fast_settings, tracer, engines=engines))

    assert tracer.traces[0].scores["answer_clean"] == 1.0


async def test_corrupted_answer_scores_zero_with_the_artifact_named(fast_settings):
    """Live traffic is scored by the same detector the offline probe uses."""
    tracer = RecordingTracer()
    bad = Answer(answer="an em dash — here", confidence=0.9, caveats=[])
    engines = [FakeEngine("openai", script=[bad])]
    await drain(traced_run_agent("q", fast_settings, tracer, engines=engines))

    trace = tracer.traces[0]
    assert trace.scores["answer_clean"] == 0.0
    assert "non-ascii" in (trace.score_comments["answer_clean"] or "")


# --------------------------------------------------------------------------
# Telemetry must never break a request
# --------------------------------------------------------------------------
async def test_exploding_tracer_does_not_break_the_request(fast_settings):
    """A monitoring outage turning into a user-visible failure is backwards."""
    engines = [FakeEngine("openai", script=[ok_answer("still fine")])]
    final = await drain(traced_run_agent("q", fast_settings, ExplodingTracer(), engines=engines))
    assert final.status == "ok"
    assert final.answer is not None
    assert final.answer.answer == "still fine"


def test_never_raises_swallows_and_returns_none():
    @never_raises
    def boom() -> None:
        raise RuntimeError("nope")

    assert boom() is None


async def test_null_tracer_is_a_working_no_op(fast_settings):
    engines = [FakeEngine("openai", script=[ok_answer()])]
    final = await drain(traced_run_agent("q", fast_settings, NullTracer("off"), engines=engines))
    assert final.status == "ok"


def test_null_tracer_reports_why_it_is_unavailable():
    assert NullTracer("LANGFUSE_ENABLED is false").is_available() == "LANGFUSE_ENABLED is false"


# --------------------------------------------------------------------------
# The detector itself
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,expected",
    [
        ("plain ascii text", []),
        ("a curly quote ’ here", ["non-ascii:’"]),
        ("line\nbreak", ["line-break"]),
        ("a literal \\u2014 escape", ["literal-\\u-escape"]),
        ("literal \\n text", ["literal-\\n-text"]),
        ("the word dash with no hyphen", ["the-word-dash"]),
        # Detectors overlap on purpose. A surviving em dash trips both the
        # word-dash rule (no ASCII hyphen present) and the non-ASCII rule;
        # the point is that no corruption form escapes unnoticed, not that
        # each maps to exactly one label.
        ("an em dash — here", ["the-word-dash", "non-ascii:—"]),
    ],
)
def test_escape_artifacts_detects_each_observed_form(text, expected):
    """All five corruption forms, not just line breaks.

    Counting newlines alone reported 30% when the real rate was 87%.
    """
    assert escape_artifacts(text) == expected


def test_detector_cannot_distinguish_openai_style_from_claude_corruption():
    """A documented limitation, pinned so nobody reads the counts naively.

    OpenAI's perfectly good curly apostrophe and a genuine Claude mangling
    both surface as `non-ascii:`. The summary tally cannot tell them apart;
    only reading the text can.
    """
    benign = escape_artifacts("OpenAI’s output")
    assert benign == ["non-ascii:’"]
    answer = Answer(answer="OpenAI’s output", confidence=0.9, caveats=[])
    assert is_degenerate(diagnose(answer))


def test_self_flagged_caveat_counts_as_degenerate():
    """Observed in practice: the model flagged its own garbled output."""
    answer = Answer(
        answer="clean text",
        confidence=0.6,
        caveats=["There was a formatting error in my first attempt."],
    )
    assert is_degenerate(diagnose(answer))
