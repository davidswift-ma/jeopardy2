"""The Think/Act/Observe mapping, and what "proved the loop" means.

Exercised with fake ADK-shaped events rather than a live Gemini run, so the
classification logic is testable with no key and no network. The event shape
mirrors what `Event` actually exposes: `author`, `content.parts[].text`,
`get_function_calls()`, `get_function_responses()`, `is_final_response()`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agents.trace_log import LoopLogger, phases_of


@dataclass
class FakePart:
    text: str | None = None


@dataclass
class FakeContent:
    parts: list[FakePart] = field(default_factory=list)


@dataclass
class FakeCall:
    name: str
    args: dict


@dataclass
class FakeResponse:
    name: str
    response: object


@dataclass
class FakeEvent:
    author: str = "agent"
    text: str | None = None
    calls: list[FakeCall] = field(default_factory=list)
    responses: list[FakeResponse] = field(default_factory=list)
    final: bool = False

    @property
    def content(self):
        return FakeContent([FakePart(self.text)]) if self.text else None

    def get_function_calls(self):
        return self.calls

    def get_function_responses(self):
        return self.responses

    def is_final_response(self):
        return self.final


def test_text_on_a_non_final_event_is_think():
    (phase,) = phases_of(FakeEvent(text="I should search the archive."))
    assert phase.label == "THINK"


def test_a_proposed_tool_call_is_act():
    (phase,) = phases_of(FakeEvent(calls=[FakeCall("search_clues", {"query": "rivers"})]))
    assert phase.label == "ACT"
    assert "search_clues" in phase.detail
    assert "rivers" in phase.detail


def test_a_tool_result_is_observe():
    (phase,) = phases_of(FakeEvent(responses=[FakeResponse("search_clues", {"result_count": 5})]))
    assert phase.label == "OBSERVE"
    assert "result_count" in phase.detail


def test_final_text_is_answer_not_think():
    (phase,) = phases_of(FakeEvent(text="The Jordan.", final=True))
    assert phase.label == "ANSWER"


def test_one_event_can_be_think_then_act():
    """Models routinely narrate and call a tool in the same turn."""
    phases = phases_of(
        FakeEvent(text="Let me look.", calls=[FakeCall("search_clues", {"query": "x"})])
    )
    assert [p.label for p in phases] == ["THINK", "ACT"]


def test_an_event_with_nothing_useful_yields_no_phases():
    assert phases_of(FakeEvent()) == []


# --------------------------------------------------------------------------
# "Prove the loop"
# --------------------------------------------------------------------------
def _full_loop() -> LoopLogger:
    logger = LoopLogger(echo=False)
    for event in (
        FakeEvent(text="I need the archive for this."),
        FakeEvent(calls=[FakeCall("search_clues", {"query": "rivers"})]),
        FakeEvent(responses=[FakeResponse("search_clues", {"result_count": 3})]),
        FakeEvent(text="The archive has a clue about the Jordan.", final=True),
    ):
        logger.record(event)
    return logger


def test_a_full_loop_is_proved():
    logger = _full_loop()
    assert logger.labels == ["THINK", "ACT", "OBSERVE", "ANSWER"]
    assert logger.proved_the_loop()
    assert "LOOP PROVED" in logger.summary()


def test_answering_without_calling_a_tool_is_not_proved():
    """The workflow-vs-agent distinction, made checkable.

    A model that answers straight from memory never proposed a tool call, so
    nothing about the run demonstrates agency.
    """
    logger = LoopLogger(echo=False)
    logger.record(FakeEvent(text="Rivers are bodies of water.", final=True))
    assert not logger.proved_the_loop()


def test_a_tool_call_with_no_answer_after_it_is_not_proved():
    """Calling a tool is not enough; the result has to reach an answer."""
    logger = LoopLogger(echo=False)
    logger.record(FakeEvent(calls=[FakeCall("search_clues", {"query": "x"})]))
    logger.record(FakeEvent(responses=[FakeResponse("search_clues", {"result_count": 0})]))
    assert not logger.proved_the_loop()


def test_order_matters():
    """An answer before the observation does not count."""
    logger = LoopLogger(echo=False)
    logger.record(FakeEvent(text="Done already.", final=True))
    logger.record(FakeEvent(calls=[FakeCall("search_clues", {"query": "x"})]))
    logger.record(FakeEvent(responses=[FakeResponse("search_clues", {})]))
    assert not logger.proved_the_loop()


def test_an_immediately_repeated_phase_is_collapsed():
    """Observed live: an A2A hop emits its final response twice.

    Once from the remote agent, once as the router relays it. Left alone the
    summary reads ANSWER -> ANSWER, which looks like two answers to one
    question.
    """
    logger = LoopLogger(echo=False)
    for _ in range(2):
        logger.record(FakeEvent(author="judge_agent", text="ACCEPT", final=True))
    assert logger.labels == ["ANSWER"]


def test_a_repeat_that_is_not_adjacent_is_kept():
    """Two genuine searches for the same term are two real events."""
    logger = LoopLogger(echo=False)
    call = FakeCall("search_clues", {"query": "rivers"})
    logger.record(FakeEvent(calls=[call]))
    logger.record(FakeEvent(responses=[FakeResponse("search_clues", {})]))
    logger.record(FakeEvent(calls=[call]))
    assert logger.labels == ["ACT", "OBSERVE", "ACT"]


def test_tool_calls_are_recoverable_for_assertions():
    assert "search_clues" in _full_loop().tool_calls()[0]


def test_long_values_are_truncated_not_dropped():
    (phase,) = phases_of(FakeEvent(text="x" * 5000))
    assert phase.detail.endswith("...")
    assert len(phase.detail) < 400


def test_rendering_lines_up():
    rendered = _full_loop().phases[0].render()
    assert rendered.startswith("[THINK   ]")
